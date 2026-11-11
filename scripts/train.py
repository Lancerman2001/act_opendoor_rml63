#!/usr/bin/env python3
"""Launch official LeRobot ACT training using a reproducible project YAML config."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from project_config import DEFAULT_CONFIG, ROOT, dump_config, load_config, resolve_path, validate_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_split(config):
    dataset_root = resolve_path(config['dataset']['root'])
    split_file = resolve_path(config['dataset']['split_file'])
    if not split_file.is_file():
        raise FileNotFoundError(f'Missing split: {split_file}. Run ./run.sh prepare --config <your config> first.')
    split = json.loads(split_file.read_text())
    require(split.get('status') == 'prepared', 'Dataset split has not completed successfully')
    train, test = split['train_episode_indices'], split['test_episode_indices']
    total = split['total_episodes']
    require(isinstance(total, int) and total >= 2, 'Split must contain at least two episodes')
    require(train and test, 'Both training and held-out test episodes are required')
    require(all(isinstance(x, int) and not isinstance(x, bool) for x in train + test), 'Episode indices must be integers')
    require(len(set(train)) == len(train) and len(set(test)) == len(test), 'Split contains duplicate episodes')
    require(not set(train) & set(test) and set(train) | set(test) == set(range(total)),
            'Training and test selections must partition all dataset episodes')
    require(len(train) == split['train_episode_count'] and len(test) == split['test_episode_count'],
            'Split episode counts do not match the index selections')
    recorded_root = Path(split['dataset_root'])
    if not recorded_root.is_absolute():
        recorded_root = split_file.parent / recorded_root
    require(recorded_root.resolve() == dataset_root, 'Configured dataset root differs from the prepared split')
    require(config['dataset']['repo_id'] == split['repo_id'], 'Configured repo_id differs from the prepared split')
    require(config['dataset']['seed'] == split['seed'], 'Configured dataset seed differs from the prepared split')
    recorded_ratio = split.get('train_ratio')
    if recorded_ratio is not None:
        require(abs(float(recorded_ratio) - config['dataset']['train_ratio']) < 1e-12,
                'Configured train_ratio differs from the prepared split')
    else:
        require(len(train) == max(1, min(total - 1, int(total * config['dataset']['train_ratio']))),
                'Configured train_ratio does not match the prepared episode counts')
    require(split.get('normalization_scope') == 'training episodes only', 'Expected training-only normalization statistics')
    stats_hash = hashlib.sha256((dataset_root / 'meta/stats.json').read_bytes()).hexdigest()
    require(stats_hash == split['train_stats_sha256'], 'Training normalization statistics have changed')
    info = json.loads((dataset_root / 'meta/info.json').read_text())
    require(info['total_episodes'] == total, 'Dataset and split episode totals differ')
    require(info['total_frames'] == split['total_frames'], 'Dataset and split frame totals differ')
    return split, stats_hash, info


def choose_batch(config, stats_hash, info):
    value = config['training']['batch_size']
    if value != 'auto':
        return value
    report_path = resolve_path(config['benchmark']['report_dir']) / 'summary.json'
    report = json.loads(report_path.read_text())
    require(report.get('status') == 'completed', 'Automatic batch selection requires a completed benchmark')
    selected_batch = report.get('recommended_batch_size')
    require(isinstance(selected_batch, int) and selected_batch > 0, 'Benchmark has no safe batch recommendation')
    selected = next((row for row in report['results'] if row.get('batch_size') == selected_batch), None)
    require(selected and selected.get('status') == 'passed' and selected.get('returncode') == 0,
            'Recommended benchmark candidate did not pass')
    require(selected.get('train_stats_sha256') == stats_hash, 'Benchmark used different normalization statistics')
    require(resolve_path(report['dataset_root']) == resolve_path(config['dataset']['root']),
            'Benchmark dataset root differs from the current configuration')
    require(resolve_path(report['split']) == resolve_path(config['dataset']['split_file']),
            'Benchmark split path differs from the current configuration')
    require(selected.get('estimated_board_headroom_bytes', -1) >= max(
        report['headroom_target_bytes'], config['benchmark']['headroom_gib'] * 1024**3),
        'Recommended batch does not satisfy the configured memory headroom')
    policy = config['policy']
    checks = {'model': policy['type'].upper(), 'vision_backbone': policy['vision_backbone'],
              'pretrained_backbone_weights': policy['pretrained_backbone_weights'],
              'chunk_size': policy['chunk_size'], 'mixed_precision': config['training']['mixed_precision'],
              'learning_rate': policy['optimizer_lr']}
    for key, value in checks.items():
        require(selected.get(key) == value, f'Benchmark {key} does not match this training configuration')
    # Historical project reports used these two fixed values; newer reports record them explicitly.
    require(selected.get('learning_rate_backbone', 1e-5) == policy['optimizer_lr_backbone'],
            'Benchmark backbone learning rate differs')
    require(selected.get('kl_weight', 10.0) == policy['kl_weight'], 'Benchmark KL weight differs')
    cameras = {}
    for key, feature in info['features'].items():
        if feature['dtype'] in {'video', 'image'}:
            shape = feature['shape']
            names = feature.get('names', [])
            cameras[key] = [shape[2], shape[0], shape[1]] if names == ['height', 'width', 'channel'] else shape
    if 'image_shapes' in selected:
        require(selected['image_shapes'] == cameras, 'Benchmark camera shapes differ from the dataset')
    else:
        require(len(cameras) == 1 and selected.get('image_shape') == next(iter(cameras.values())),
                'The historical single-camera benchmark does not match this dataset')
    return selected_batch


def value_argument(value):
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    return str(value)


def plan_training(config, args):
    split, stats_hash, info = load_split(config)
    train, policy = config['training'], config['policy']
    command = [sys.executable, '-u', str(ROOT / 'scripts/train_entry.py')]
    output = resolve_path(train['output_dir']) if train['output_dir'] else resolve_path(train['output_root']) / train['run_name']
    if train['resume_config']:
        checkpoint_config = resolve_path(train['resume_config'])
        saved = json.loads(checkpoint_config.read_text())
        require(set(saved['dataset']['episodes']) == set(split['train_episode_indices'])
                and len(saved['dataset']['episodes']) == len(split['train_episode_indices']),
                'Resume configuration must use exactly the prepared training episodes')
        require(resolve_path(saved['dataset']['root']) == resolve_path(config['dataset']['root']),
                'Resume dataset root differs from this project')
        require(saved['dataset']['repo_id'] == config['dataset']['repo_id'], 'Resume dataset repo_id differs')
        require(float(saved['dataset'].get('eval_split', 0)) == 0 and int(saved.get('eval_steps', 0)) == 0,
                'Resume configuration must keep held-out evaluation disabled during training')
        require(saved['policy']['type'] == 'act' and saved['policy']['device'] in {'cuda', 'cuda:0'},
                'Resume configuration must describe a CUDA ACT policy')
        require(saved['accelerator'].get('gradient_accumulation', {}).get('steps', 1) == 1,
                'This project launcher currently supports one optimizer update per microbatch')
        for cli, field in ((args.batch_size, 'batch_size'), (args.seed, 'seed'), (args.num_workers, 'num_workers')):
            require(cli is None or (field == 'batch_size' and cli == 'auto') or int(cli) == int(saved[field]),
                    f'Resume retains checkpoint {field}={saved[field]}; omit the conflicting CLI override')
        for field in ('batch_size', 'seed', 'num_workers', 'prefetch_factor'):
            train[field] = saved[field]
        train['mixed_precision'] = saved['accelerator']['mixed_precision']
        for field in policy:
            policy[field] = saved['policy'][field]
        config['dataset']['video_backend'] = saved['dataset']['video_backend']
        output = resolve_path(saved['output_dir'])
        train['resume_config'] = str(checkpoint_config)
        step_path = checkpoint_config.parent.parent / 'training_state/training_step.json'
        metadata = json.loads(step_path.read_text())
        require(train['steps'] > metadata['step'], f'Requested total steps must exceed saved step {metadata["step"]}')
        command += [f'--config_path={checkpoint_config}', '--resume=true', f'--steps={train["steps"]}',
                    f'--log_freq={config["logging"]["every_steps"]}',
                    f'--save_freq={config["checkpoint"]["every_steps"]}']
    else:
        train['batch_size'] = choose_batch(config, stats_hash, info)
        require(not output.exists(), f'Output already exists: {output}; choose a new run_name or resume_config')
        command += [f'--dataset.repo_id={config["dataset"]["repo_id"]}',
                    f'--dataset.root={resolve_path(config["dataset"]["root"])}',
                    '--dataset.episodes=' + json.dumps(split['train_episode_indices'], separators=(',', ':')),
                    '--dataset.eval_split=0', f'--dataset.video_backend={config["dataset"]["video_backend"]}']
        command += [f'--policy.{key}={value_argument(value)}' for key, value in policy.items()]
        command += ['--policy.push_to_hub=false', f'--batch_size={train["batch_size"]}',
                    f'--steps={train["steps"]}', f'--num_workers={train["num_workers"]}',
                    f'--prefetch_factor={train["prefetch_factor"]}', '--dataloader_multiprocessing_context=spawn',
                    f'--accelerator.mixed_precision={train["mixed_precision"]}',
                    '--env_eval_freq=0', '--eval_steps=0',
                    f'--save_freq={config["checkpoint"]["every_steps"]}',
                    f'--log_freq={config["logging"]["every_steps"]}',
                    '--trackio.enable=false', f'--seed={train["seed"]}',
                    f'--output_dir={output}', f'--job_name={train["run_name"]}']
    train['output_dir'] = str(output)
    validate_config(config)
    return {'config': config, 'command': command, 'output_dir': str(output),
            'train_episode_count': len(split['train_episode_indices']),
            'test_episode_count': len(split['test_episode_indices']),
            'train_frames': split['train_frames'], 'train_stats_sha256': stats_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size')
    parser.add_argument('--run-name')
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume-config', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Validate prepared data and print the command without CUDA or training')
    parser.add_argument('--print-config', action='store_true', help='Print merged configuration without requiring data or CUDA')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        for flag, field in ((args.steps, 'steps'), (args.run_name, 'run_name'),
                            (args.num_workers, 'num_workers'), (args.seed, 'seed')):
            if flag is not None:
                config['training'][field] = flag
        if args.batch_size is not None:
            config['training']['batch_size'] = 'auto' if args.batch_size == 'auto' else int(args.batch_size)
        if args.resume_config is not None:
            config['training']['resume_config'] = str(args.resume_config)
        if args.log_every is not None:
            config['logging']['every_steps'] = args.log_every
        if args.save_every is not None:
            config['checkpoint']['every_steps'] = args.save_every
        validate_config(config)
        if args.print_config:
            print(json.dumps(config, ensure_ascii=False, indent=2))
            return 0
        plan = plan_training(config, args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        import torch
        from importlib.metadata import version
        require(version('lerobot') == '0.7.0', 'This launcher requires LeRobot 0.7.0')
        require(torch.cuda.is_available(), 'CUDA unavailable; refusing CPU fallback')
        if config['training']['mixed_precision'] == 'bf16':
            require(torch.cuda.is_bf16_supported(), 'This GPU does not support the configured BF16 precision')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    attempt = config['training']['run_name'] + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report_dir = ROOT / 'reports' / attempt
    monitor = config['monitor']
    print(json.dumps({'output': plan['output_dir'], 'report_dir': str(report_dir),
                      'steps': config['training']['steps'], 'batch_size': config['training']['batch_size'],
                      'train_episodes': plan['train_episode_count'], 'held_out_test_episodes': plan['test_episode_count']}), flush=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.launch-', dir=ROOT / 'logs') as temporary:
        snapshot = Path(temporary) / 'resolved_config.yaml'
        dump_config(config, snapshot)
        code = subprocess.call([sys.executable, '-u', str(ROOT / 'scripts/run_monitored.py'),
                                '--name', attempt, '--config-snapshot', str(snapshot),
                                '--interval', str(monitor['interval_seconds']),
                                '--temperature-limit', str(monitor['temperature_limit_c']),
                                '--min-host-free-mib', str(monitor['min_host_free_mib']),
                                '--', *plan['command']], cwd=ROOT)
    if code == 0:
        output = Path(plan['output_dir'])
        latest = {'run_name': config['training']['run_name'], 'output': str(output),
                  'monitor_report': str(report_dir),
                  'checkpoint': str((output / 'checkpoints/last/pretrained_model').resolve()),
                  'steps': config['training']['steps'], 'batch_size': config['training']['batch_size'],
                  'status': 'completed', 'config_snapshot': str(report_dir / 'resolved_config.yaml'),
                  'train_stats_sha256': plan['train_stats_sha256'],
                  'split_file': str(resolve_path(config['dataset']['split_file'])),
                  'loss_csv': str(report_dir / 'loss.csv'), 'loss_jsonl': str(report_dir / 'loss.jsonl'),
                  'plots_dir': str(report_dir / 'plots'),
                  'test_evaluation': str(report_dir / 'test_evaluation.json')}
        result_text = json.dumps(latest, indent=2) + '\n'
        (report_dir / 'training_result.json').write_text(result_text)
        (output / 'training_result.json').write_text(result_text)
        (ROOT / 'reports/latest_training.json').write_text(result_text)
    return code


if __name__ == '__main__':
    sys.exit(main())
