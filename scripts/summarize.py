#!/usr/bin/env python3
"""核验一次 ACT 训练；仅关联该次运行指定的评估、配置和图片。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

from plot_training import load_history

ROOT = Path(__file__).resolve().parents[1]
MAE_KEYS = ('first_action_mae', 'chunk_action_mae', 'hold_first_action_mae', 'hold_chunk_action_mae')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def located(path, root):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_checkpoint(checkpoint):
    """只在 CPU 逐张量检查模型、处理器及训练状态，不运行模型。"""
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    files = sorted(checkpoint.parent.rglob('*.safetensors'))
    require((checkpoint / 'model.safetensors') in files, '最终 checkpoint 缺少 model.safetensors')
    require((checkpoint.parent / 'training_state/optimizer_state.safetensors') in files,
            '最终 checkpoint 缺少优化器状态')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        require((checkpoint / name).is_file(), f'最终 checkpoint 缺少处理器配置：{name}')
    result = {'status': 'passed', 'device': 'cpu', 'method': 'one safetensors tensor at a time',
              'scope': str(checkpoint.parent), 'tensor_count': 0, 'element_count': 0, 'files': []}
    for path in files:
        count = elements = 0
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if tensor.is_floating_point() or tensor.is_complex():
                    require(bool(torch.isfinite(tensor).all()), f'非有限 checkpoint 张量：{path.name}:{name}')
                count += 1
                elements += tensor.numel()
                del tensor
        require(count > 0, f'空 safetensors 文件：{path}')
        result['files'].append({'path': str(path), 'sha256': sha256(path), 'tensor_count': count,
                                'element_count': elements, 'size_bytes': path.stat().st_size})
        result['tensor_count'] += count
        result['element_count'] += elements
    return result


def validate_evaluation(path, checkpoint, split, info, split_path, inputs, root):
    require(path.is_file(), f'指定的评估报告不存在：{path}')
    inputs['evaluation'] = path
    report = read_json(path)
    require(report.get('status') == 'passed', '指定的评估没有成功完成')
    require(not report.get('self_test') and not report.get('excluded_from_training_results'), '不能把自测作为正式评估')
    require(located(report['checkpoint'], root) == checkpoint, '评估 checkpoint 路径与本次训练不一致')
    require(report['checkpoint_sha256'] == sha256(checkpoint / 'model.safetensors'), '评估 checkpoint SHA256 不匹配')
    test, episodes = split['test_episode_indices'], report['episode_indices']
    require(report.get('split') == 'test' and len(episodes) == len(set(episodes)) == len(test)
            and set(episodes) == set(test), '评估 episode 与本次测试集不一致')
    if report.get('split_file'):
        require(located(report['split_file'], root) == split_path, '评估的数据划分路径不一致')
    require(report['evaluated_episode_count'] == len(test), '评估没有覆盖全部保留测试 episode')
    require(report['sample_count'] == report['planned_samples'] > 0, '评估样本未完整处理')
    require(report['dataset_split_frames'] == split['test_frames'], '评估数据帧数不一致')
    require(report.get('padding_excluded') is True and report['valid_chunk_action_count'] > 0,
            '评估未确认排除动作窗口末尾 padding')
    require(report.get('normalization_scope') == 'saved checkpoint processors, checked against training-only stats',
            '评估没有确认训练集归一化统计')
    feature = info['features']['action']
    dimension = feature['shape'][0]
    names = feature.get('names') or [f'action_{i}' for i in range(dimension)]
    require(len(names) == dimension and len(set(names)) == dimension, '数据集动作维度名称无效')
    for metric in MAE_KEYS:
        require(set(report[metric]) == set(names)
                and all(finite(value) and value >= 0 for value in report[metric].values()), f'无效的动作误差：{metric}')
    require(set(report['metric_units']) == set(names)
            and all(isinstance(value, str) and value for value in report['metric_units'].values()), '评估缺少逐维单位')
    if 'normalized_masked_l1' in report:
        require(finite(report['normalized_masked_l1']) and report['normalized_masked_l1'] >= 0, '归一化测试误差无效')
    return {key: report.get(key) for key in (
        'status', 'checkpoint', 'checkpoint_sha256', 'sample_count', 'planned_samples', 'evaluated_episode_count',
        'dataset_split_frames', 'sampling', 'valid_chunk_action_count', 'normalized_masked_l1', 'chunk_size',
        'first_action_mae', 'chunk_action_mae', 'hold_first_action_mae', 'hold_chunk_action_mae',
        'metric_units', 'normalization_scope', 'elapsed_evaluation_seconds')}


def validate_plots(monitor_dir, snapshot, history, inputs, root):
    path = monitor_dir / 'plots/plot_summary.json'
    expected = bool(snapshot and snapshot.get('plot', {}).get('enabled'))
    if not path.is_file():
        require(not expected, '配置要求自动绘图，但缺少 plots/plot_summary.json')
        return {'status': 'disabled' if snapshot else 'not_recorded', 'files': []}
    inputs['plot_summary'] = path
    plots = read_json(path)
    require(plots.get('status') == 'completed', '绘图没有成功完成')
    plotted_source = located(plots['input'], root)
    require(plotted_source.parent == monitor_dir and plotted_source.name in ('loss.csv', 'loss.jsonl', 'metrics.jsonl', 'console.log'),
            '图片来源不是本次训练的日志')
    require(plotted_source.is_file() and sha256(plotted_source) == plots['input_sha256'], '绘图后原始损失日志发生变化')
    plotted = load_history(plotted_source)
    require(plotted['steps'] == history['steps'], '图片和训练记录的步数范围不一致')
    require(plotted['values']['loss'] == history['values']['loss'], '图片和训练记录的总 loss 不一致')
    for metric in history['metrics_available']:
        if metric in plotted['metrics_available'] or history['source_precision'] == plotted['source_precision'] == 'recorded_numeric_values':
            require(plotted['values'][metric] == history['values'][metric],
                    f'图片和训练记录的 {metric} 不一致')
    paths = [located(value, path.parent) for value in plots['output_files']]
    require(paths and len(paths) == len(set(paths)), '图片列表为空或重复')
    for image in paths:
        require(image.is_file() and image.stat().st_size > 0, f'缺少训练效果图片：{image}')
    if expected:
        require({image.suffix.lstrip('.') for image in paths} == set(snapshot['plot']['formats']), '图片格式未满足配置')
    return {'status': 'completed', 'source_precision': plots['source_precision'], 'plot_config': plots['plot_config'],
            'files': [{'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size} for path in paths]}


def build_report(training_report, evaluation_report=None, project_root=ROOT, skip_evaluation=False):
    root = Path(project_root).resolve()
    training_report = located(training_report, root)
    inputs = {'training_report': training_report}
    latest = read_json(training_report)
    require(latest.get('status') == 'completed', '本次训练尚未成功完成')
    monitor_dir = located(latest['monitor_report'], root)
    inputs['monitor'] = monitor_dir / 'summary.json'
    monitor = read_json(inputs['monitor'])
    require(monitor.get('status') == 'passed' and monitor.get('returncode') == 0 and not monitor.get('stop_reasons'),
            '训练监督进程未成功完成')
    require(finite(monitor['elapsed_seconds']) and monitor['elapsed_seconds'] >= 0, '训练耗时无效')
    snapshot_path = latest.get('config_snapshot') or monitor.get('config_snapshot')
    snapshot, warnings = None, []
    if snapshot_path:
        import yaml
        inputs['config_snapshot'] = located(snapshot_path, root)
        require(inputs['config_snapshot'].parent == monitor_dir, '配置快照不属于本次训练报告目录')
        snapshot = yaml.safe_load(inputs['config_snapshot'].read_text(encoding='utf-8'))
        require(isinstance(snapshot, dict), '训练配置快照不是有效配置')
    else:
        warnings.append('旧运行没有项目配置快照；使用 checkpoint 配置与保存的数据划分核验。')
    dataset_config = snapshot.get('dataset', {}) if snapshot else {}
    split_path = located(latest.get('split_file') or dataset_config.get('split_file', 'data/split.json'), root)
    inputs['split'] = split_path
    split = read_json(split_path)
    require(split.get('status') == 'prepared' and split.get('normalization_scope') == 'training episodes only',
            '数据划分或训练集归一化尚未准备完成')
    train, test, total = split['train_episode_indices'], split['test_episode_indices'], split['total_episodes']
    require(train and test and len(train) == len(set(train)) and len(test) == len(set(test)), '训练/测试 episode 为空或重复')
    require(not set(train) & set(test) and set(train) | set(test) == set(range(total)), '数据划分有交叉或遗漏')
    require(split['train_episode_count'] == len(train) and split['test_episode_count'] == len(test), '划分 episode 数量不一致')
    dataset = located(split['dataset_root'], split_path.parent)
    if dataset_config.get('root'):
        require(located(dataset_config['root'], root) == dataset, '配置快照的数据路径与划分不一致')
    inputs.update(stats=dataset / 'meta/stats.json', dataset_info=dataset / 'meta/info.json')
    stats_hash = sha256(inputs['stats'])
    require(stats_hash == split['train_stats_sha256'], '训练集统计 SHA256 不匹配')
    if latest.get('train_stats_sha256'):
        require(latest['train_stats_sha256'] == stats_hash, '当前统计与训练启动时记录的 SHA256 不匹配')
    info = read_json(inputs['dataset_info'])
    require(info['total_episodes'] == total and info['total_frames'] == split['total_frames'], '数据集规模与划分不一致')
    require(split['train_frames'] > 0 and split['test_frames'] > 0
            and split['train_frames'] + split['test_frames'] == split['total_frames'], '划分帧数不一致')
    checkpoint, output = located(latest['checkpoint'], root), located(latest['output'], root)
    require(checkpoint.is_relative_to(output), 'checkpoint 不属于本次训练输出目录')
    inputs.update(checkpoint_config=checkpoint / 'train_config.json',
                  checkpoint_step=checkpoint.parent / 'training_state/training_step.json')
    config, metadata = read_json(inputs['checkpoint_config']), read_json(inputs['checkpoint_step'])
    require(metadata['step'] == latest['steps'] == config['steps'], '最终 checkpoint 步数与训练计划不一致')
    require(metadata['batch_size'] == latest['batch_size'] == config['batch_size'], '最终 checkpoint batch 与训练报告不一致')
    require(metadata['step'] > 0 and metadata['batch_size'] > 0, '训练步数和 batch 必须为正')
    require(metadata.get('dp_world_size', 1) == metadata.get('grad_accum_steps', 1) == 1,
            'checkpoint 并行或梯度累积设置不符合本项目的单 GPU 训练')
    if 'accelerator' in config:
        require(metadata.get('mixed_precision') == config['accelerator']['mixed_precision'],
                'checkpoint 精度与训练状态不一致')
    require(located(config['output_dir'], root) == output, 'checkpoint 保存的输出目录不一致')
    selected = config['dataset']['episodes']
    require(len(selected) == len(train) and set(selected) == set(train), 'checkpoint 的训练 episode 与划分不一致')
    require(located(config['dataset']['root'], root) == dataset, 'checkpoint 的数据路径不一致')
    require(float(config['dataset'].get('eval_split', 0)) == 0 and int(config.get('eval_steps', 0)) == 0,
            '本项目要求训练期间不自动评估保留测试集')
    require(config['policy']['type'] == 'act', 'checkpoint 策略不是 ACT')
    if snapshot:
        require(snapshot['training']['steps'] == metadata['step'] and snapshot['training']['batch_size'] == metadata['batch_size'],
                '配置快照与 checkpoint 步数或 batch 不一致')
        training = snapshot['training']
        if 'run_name' in training:
            require(training['run_name'] == latest['run_name'], '配置快照的运行名称不一致')
        if training.get('output_dir'):
            require(located(training['output_dir'], root) == output, '配置快照的输出目录不一致')
        for key in ('seed', 'num_workers', 'prefetch_factor'):
            if key in training:
                require(training[key] == config.get(key), f'配置快照的 {key} 与 checkpoint 不一致')
        if 'mixed_precision' in training:
            require(training['mixed_precision'] == metadata.get('mixed_precision'), '配置快照的精度与 checkpoint 不一致')
        for key, value in snapshot.get('policy', {}).items():
            require(value == config['policy'].get(key), f'配置快照的 policy.{key} 与 checkpoint 不一致')
        for key in ('repo_id', 'video_backend'):
            if key in dataset_config:
                require(dataset_config[key] == config['dataset'].get(key), f'配置快照的 dataset.{key} 与 checkpoint 不一致')
        require(snapshot['logging']['every_steps'] == config['log_freq']
                and snapshot['checkpoint']['every_steps'] == config['save_freq'], '记录/保存间隔与配置快照不一致')
    loss_file = next((monitor_dir / name for name in ('loss.jsonl', 'loss.csv', 'metrics.jsonl')
                      if (monitor_dir / name).is_file()), None)
    require(loss_file is not None, '缺少本次训练损失记录')
    inputs['loss'] = loss_file
    history = load_history(loss_file)
    require(not history['invalid_step_rows'] and not history['flagged_nonfinite_rows']
            and not any(history['nonfinite_values'].values()) and not history['missing_values']['loss'], '损失记录包含无效或非有限值')
    require(history['steps'][-1] <= metadata['step'], '记录的步数超过最终 checkpoint')
    if history['source_precision'] != 'console_rounded':
        require(history['steps'][-1] == metadata['step'], '原始损失记录缺少最终训练步')
    else:
        warnings.append('使用旧控制台解析日志；数值已舍入，未记录的损失分量无法补回。')
    metric_summary = {}
    for name, values in history['values'].items():
        present = [value for value in values if value is not None]
        metric_summary[name] = {'first': present[0], 'last': present[-1], 'records': len(present)} if present else None
    plots = validate_plots(monitor_dir, snapshot, history, inputs, root)
    require(not (skip_evaluation and evaluation_report is not None), '不能同时跳过和指定评估报告')
    evaluation_path = None if skip_evaluation else located(evaluation_report, root) if evaluation_report is not None else \
        located(latest['test_evaluation'], root) if latest.get('test_evaluation') else None
    if evaluation_path is not None and (evaluation_report is not None or evaluation_path.is_file()):
        evaluation = validate_evaluation(evaluation_path, checkpoint, split, info, split_path, inputs, root)
        require(evaluation['chunk_size'] == config['policy']['chunk_size'], '评估动作窗口长度与 checkpoint 不一致')
        evaluation['selection'] = 'explicit' if evaluation_report is not None else 'run_linked'
    else:
        evaluation = {'status': 'not_run', 'selection': 'disabled' if skip_evaluation else 'unavailable',
                      'reason': '本次未关联已完成的评估；没有读取历史全局测试结果。'}
    integrity = verify_checkpoint(checkpoint)
    return {'schema_version': 1, 'status': 'completed', 'run_name': latest['run_name'],
            'monitor_report': str(monitor_dir), 'checkpoint': str(checkpoint),
            'training': {'completed_steps': metadata['step'], 'batch_size': metadata['batch_size'],
                         'mixed_precision': metadata.get('mixed_precision'), 'elapsed_seconds': monitor['elapsed_seconds'],
                         'gpu_peak_used_gib': monitor.get('gpu_peak_used_mib') / 1024 if monitor.get('gpu_peak_used_mib') is not None else None,
                         'gpu_peak_temperature_c': monitor.get('gpu_peak_temperature_c'),
                         'logging_every_steps': config.get('log_freq'), 'checkpoint_every_steps': config.get('save_freq'),
                         'kl_weight': config['policy'].get('kl_weight'), 'first_logged_step': history['steps'][0],
                         'last_logged_step': history['steps'][-1], 'loss_source_precision': history['source_precision'],
                         'metric_records': history['records_plotted'], 'interval_steps_observed': history['logged_interval_steps'],
                         'metrics': metric_summary},
            'dataset': {key: split[key] for key in ('repo_id', 'train_episode_count', 'test_episode_count',
                       'train_frames', 'test_frames', 'normalization_scope', 'train_stats_sha256')},
            'plots': plots, 'evaluation': evaluation, 'checkpoint_integrity': integrity, 'warnings': warnings,
            'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in inputs.items()},
            'generated_at_utc': datetime.now(timezone.utc).isoformat()}


def number(value, digits=4):
    return '未记录' if value is None else f'{value:,.{digits}f}'


def markdown(report, output_dir):
    if report['status'] != 'completed':
        return '# ACT 训练报告\n\n状态：' + report['status'] + '\n\n' + '\n'.join(f'- {issue}' for issue in report['issues']) + '\n'
    train, data, evaluation = report['training'], report['dataset'], report['evaluation']
    lines = ['# ACT 训练报告', '', f'运行：`{report["run_name"]}`。核验状态：**已完成**。', '',
             f'最终 checkpoint 为 **{train["completed_steps"]:,} 步，batch={train["batch_size"]}，'
             f'混合精度={train["mixed_precision"]}**。本次受监控进程耗时 {number(train["elapsed_seconds"] / 60, 2)} 分钟；'
             '恢复训练时该耗时只覆盖本次进程。', '',
             f'训练集：{data["train_episode_count"]} 条轨迹 / {data["train_frames"]:,} 帧；'
             f'保留测试集：{data["test_episode_count"]} 条轨迹 / {data["test_frames"]:,} 帧。两组无交叉，归一化统计仅使用训练集。', '',
             f'配置记录间隔：{train["logging_every_steps"]} 步；checkpoint 间隔：{train["checkpoint_every_steps"]} 步（另保存最终步）。'
             f'共 {train["metric_records"]:,} 条损失记录，步数范围 {train["first_logged_step"]:,}–{train["last_logged_step"]:,}。', '',
             '## 训练损失', '', '| 指标 | 首个已记录值 | 最后已记录值 |', '|---|---:|---:|']
    for name, label in {'loss': '总 loss', 'l1_loss': 'L1', 'kld_loss': 'KL', 'weighted_kl_loss': '加权 KL', 'lr': '学习率'}.items():
        metric = train['metrics'][name]
        values = (metric['first'], metric['last']) if metric else (None, None)
        display = [(f'{value:.6g}' if value is not None else '未记录') if name == 'lr' else number(value, 6) for value in values]
        lines.append(f'| {label} | {display[0]} | {display[1]} |')
    lines += ['', '以上是原日志的每步值或窗口均值；最后一个不足记录间隔的窗口可能较短。'
              '它们是训练目标，不是独立测试分数，也不足以单独证明策略收敛。', '',
              f'日志精度来源：`{train["loss_source_precision"]}`。未记录的分量显示为“未记录”。', '',
              f'整卡显存观测峰值：{number(train["gpu_peak_used_gib"], 3)} GiB；'
              f'最高 GPU 温度：{number(train["gpu_peak_temperature_c"], 1)} °C。显存含其他进程，定时采样可能错过瞬时峰值。', '']
    if report['plots']['status'] == 'completed':
        lines += ['## 训练曲线', '']
        for artifact in report['plots']['files']:
            path = Path(artifact['path'])
            relative = quote(os.path.relpath(path, output_dir).replace(os.sep, '/'), safe='/._-')
            lines += [f'![训练曲线]({relative})' if path.suffix == '.png' else
                      f'[{path.suffix.lstrip(".").upper()} 图片]({relative})', '']
        lines += ['平滑窗口的单位是记录条数，缺失值保留为空档；原始记录保存在 CSV/JSONL 中。', '']
    else:
        lines += [f'训练曲线状态：`{report["plots"]["status"]}`。', '']
    lines += ['## 独立测试', '']
    if evaluation['status'] == 'not_run':
        lines += ['**本次未执行或未关联独立测试。** 本报告没有混入其他 checkpoint 的历史评估。', '']
    else:
        if evaluation.get('selection') == 'run_linked':
            lines += ['关联该次训练已经保存的评估报告；生成本报告不会重新执行推理。', '']
        lines += [f'已核对评估模型的路径及 SHA256。覆盖 {evaluation["evaluated_episode_count"]} 条测试轨迹、'
                  f'{evaluation["sample_count"]:,} 个观测位置，动作窗口长度 {evaluation["chunk_size"]}；'
                  f'排除 padding 后计入 {evaluation["valid_chunk_action_count"]:,} 个动作位置。重叠窗口不代表独立帧。', '',
                  '| 维度 | 单位 | 首步 MAE | 保持基线首步 MAE | 动作块 MAE | 保持基线动作块 MAE |',
                  '|---|---|---:|---:|---:|---:|']
        for name in evaluation['first_action_mae']:
            vals = [number(evaluation[key][name]) for key in ('first_action_mae', 'hold_first_action_mae', 'chunk_action_mae', 'hold_chunk_action_mae')]
            lines.append(f'| {name} | {evaluation["metric_units"][name]} | ' + ' | '.join(vals) + ' |')
        lines += ['', f'归一化测试 L1：{number(evaluation.get("normalized_masked_l1"), 6)}。'
                  '不同单位的动作维度分别解读，离线误差不等于实机任务成功率。', '']
    integrity = report['checkpoint_integrity']
    lines += ['## 文件核验', '', f'最终 checkpoint：`{report["checkpoint"]}`。', '',
              f'已在 CPU 逐张量检查 {len(integrity["files"])} 个 safetensors 文件、{integrity["tensor_count"]:,} 个张量；'
              '所有浮点张量均为有限值。训练步数、batch、episode 划分及训练统计 SHA256 核对通过。', '',
              '输入文件路径与 SHA256、详细指标见同目录 `final_summary.json`。', '']
    if report['warnings']:
        lines += ['记录说明：', '', *[f'- {item}' for item in report['warnings']], '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-report', type=Path, default=ROOT / 'reports/latest_training.json')
    evaluation_args = parser.add_mutually_exclusive_group()
    evaluation_args.add_argument('--evaluation-report', type=Path, help='Optional explicit evaluation for this exact checkpoint')
    evaluation_args.add_argument('--skip-evaluation', action='store_true', help='Do not attach even a run-linked saved evaluation')
    parser.add_argument('--output-dir', type=Path, help='Default: this run\'s monitor_report directory')
    args = parser.parse_args(argv)
    training_report = args.training_report.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve() if args.output_dir else None
    try:
        latest = read_json(training_report)
        if destination is None:
            destination = located(latest['monitor_report'], ROOT)
        report = build_report(training_report, args.evaluation_report, skip_evaluation=args.skip_evaluation)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        if destination is None:
            destination = training_report.parent / 'summary_unavailable'
        report = {'status': 'pending' if not training_report.is_file() else 'failed',
                  'issues': [f'{type(exc).__name__}: {exc}'], 'generated_at_utc': datetime.now(timezone.utc).isoformat()}
    if destination == ROOT / 'reports' and (destination / 'RESULTS.md').exists():
        parser.error('Refusing to overwrite historical reports/RESULTS.md; choose a run-specific output directory')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'final_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (destination / 'RESULTS.md').write_text(markdown(report, destination), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(destination / 'RESULTS.md'),
                      'issues': report.get('issues', [])}, ensure_ascii=False))
    return 0 if report['status'] == 'completed' else 2 if report['status'] == 'pending' else 1


if __name__ == '__main__':
    raise SystemExit(main())
