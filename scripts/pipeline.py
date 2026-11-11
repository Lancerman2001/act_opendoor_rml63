#!/usr/bin/env python3
"""Run sequential ACT project stages and reuse only matching completed artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from project_config import DEFAULT_CONFIG, ROOT, load_config, resolve_path
from train import choose_batch, load_split, require


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def run_script(name, *arguments):
    command = [sys.executable, '-u', str(ROOT / 'scripts' / name), *map(str, arguments)]
    print('+ ' + shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def has_flag(arguments, flag):
    return any(value == flag or value.startswith(flag + '=') for value in arguments)


def expected_training(config, stats_hash, info):
    """Mirror the launcher's resume retention rules without launching or importing Torch."""
    result = deepcopy(config)
    training = result['training']
    if training['resume_config']:
        saved = read_json(resolve_path(training['resume_config']))
        for field in ('batch_size', 'seed', 'num_workers', 'prefetch_factor'):
            training[field] = saved[field]
        training['mixed_precision'] = saved['accelerator']['mixed_precision']
        for field in result['policy']:
            result['policy'][field] = saved['policy'][field]
        result['dataset']['video_backend'] = saved['dataset']['video_backend']
        output = resolve_path(saved['output_dir'])
    else:
        training['batch_size'] = choose_batch(config, stats_hash, info)
        output = resolve_path(training['output_dir']) if training['output_dir'] else (
            resolve_path(training['output_root']) / training['run_name'])
    training['output_dir'] = str(output)
    return result, output


def validate_training_result(report, expected, split, stats_hash, output):
    """Fail closed if a historical or incomplete run could be mistaken for this run."""
    require(report.get('status') == 'completed', 'Training result is not completed')
    require(report.get('run_name') == expected['training']['run_name'], 'Training run_name differs')
    require(resolve_path(report['output']) == output, 'Training output directory differs')
    require(report.get('steps') == expected['training']['steps'], 'Training step count differs')
    require(report.get('batch_size') == expected['training']['batch_size'], 'Training batch size differs')
    require(report.get('train_stats_sha256') == stats_hash, 'Training normalization fingerprint differs or is missing')
    require(report.get('split_file') and resolve_path(report['split_file']) == resolve_path(expected['dataset']['split_file']),
            'Training split path differs or is missing')
    snapshot_path = resolve_path(report['config_snapshot'])
    require(snapshot_path.is_file(), 'Resolved training configuration is missing')
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding='utf-8'))
    for field in ('run_name', 'steps', 'batch_size', 'num_workers', 'prefetch_factor', 'mixed_precision', 'seed'):
        require(snapshot['training'][field] == expected['training'][field], f'Training snapshot {field} differs')
    require(resolve_path(snapshot['training']['output_dir']) == output, 'Training snapshot output differs')
    require(snapshot['policy'] == expected['policy'], 'Training policy configuration differs')
    for field in ('root', 'split_file'):
        require(resolve_path(snapshot['dataset'][field]) == resolve_path(expected['dataset'][field]),
                f'Training snapshot dataset.{field} differs')
    for field in ('repo_id', 'train_ratio', 'seed', 'video_backend'):
        require(snapshot['dataset'][field] == expected['dataset'][field], f'Training snapshot dataset.{field} differs')
    require(snapshot['logging'] == expected['logging'], 'Training logging interval differs')
    require(snapshot['checkpoint'] == expected['checkpoint'], 'Training checkpoint interval differs')
    monitor_dir = resolve_path(report['monitor_report'])
    require(snapshot_path.parent == monitor_dir, 'Configuration snapshot is not in this monitor directory')
    monitor = read_json(monitor_dir / 'summary.json')
    require(monitor.get('status') == 'passed' and monitor.get('returncode') == 0 and not monitor.get('stop_reasons'),
            'Training monitor did not report a clean successful exit')
    checkpoint = resolve_path(report['checkpoint'])
    require(checkpoint.is_relative_to(output) and (checkpoint / 'model.safetensors').is_file(),
            'Final checkpoint is missing or outside the requested run')
    metadata = read_json(checkpoint.parent / 'training_state/training_step.json')
    require(metadata.get('step') == expected['training']['steps'], 'Saved checkpoint has a different actual step')
    require(metadata.get('batch_size') == expected['training']['batch_size'], 'Saved checkpoint batch size differs')
    require(metadata.get('mixed_precision') == expected['training']['mixed_precision'], 'Saved checkpoint precision differs')
    require(metadata.get('dp_world_size', 1) == 1 and metadata.get('grad_accum_steps', 1) == 1,
            'Saved checkpoint uses a different parallelism or accumulation topology')
    saved = read_json(checkpoint / 'train_config.json')
    require(resolve_path(saved['output_dir']) == output and saved['steps'] == expected['training']['steps'],
            'Saved official training output/steps differ')
    for field in ('batch_size', 'seed', 'num_workers', 'prefetch_factor'):
        require(saved[field] == expected['training'][field], f'Saved official {field} differs')
    require(saved['accelerator']['mixed_precision'] == expected['training']['mixed_precision'],
            'Saved official precision differs')
    require(saved['log_freq'] == expected['logging']['every_steps']
            and saved['save_freq'] == expected['checkpoint']['every_steps'], 'Saved logging/checkpoint interval differs')
    for field, value in expected['policy'].items():
        require(saved['policy'][field] == value, f'Saved official policy.{field} differs')
    require(resolve_path(saved['dataset']['root']) == resolve_path(expected['dataset']['root']),
            'Saved official dataset root differs')
    require(saved['dataset']['repo_id'] == expected['dataset']['repo_id'], 'Saved official repo_id differs')
    require(saved['dataset']['video_backend'] == expected['dataset']['video_backend'], 'Saved video backend differs')
    require(set(saved['dataset']['episodes']) == set(split['train_episode_indices'])
            and len(saved['dataset']['episodes']) == len(split['train_episode_indices']),
            'Saved training episodes do not match the requested training split')
    require(float(saved['dataset'].get('eval_split', 0)) == 0 and int(saved.get('eval_steps', 0)) == 0,
            'Saved training configuration evaluated held-out data during training')
    return report


def find_training_result(config, split, stats_hash, info, *, required=False):
    expected, output = expected_training(config, stats_hash, info)
    run_pointer = output / 'training_result.json'
    candidates = [run_pointer, ROOT / 'reports/latest_training.json']
    for path in candidates:
        if not path.is_file():
            continue
        report = read_json(path)
        # A global pointer for another experiment is not evidence that this run completed.
        if path != run_pointer and resolve_path(report.get('output', ROOT)) != output:
            continue
        if config['training']['resume_config'] and report.get('steps', 0) < expected['training']['steps']:
            continue
        try:
            validate_training_result(report, expected, split, stats_hash, output)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise ValueError(f'Existing run cannot be reused: {output}. {exc}. '
                             'Choose a new run_name/output_dir, or explicitly resume a suitable checkpoint.') from exc
        return report, path, expected
    if required:
        raise ValueError(f'No matching completed training result for {output}. Run training first or provide an explicit checkpoint/report.')
    if output.exists() and not config['training']['resume_config']:
        raise ValueError(f'Output already exists without a matching completed result: {output}. '
                         'Choose a new run_name/output_dir or explicitly resume; existing files are preserved.')
    return None, None, expected


def validate_evaluation(report, training, config, split):
    checkpoint = resolve_path(training['checkpoint'])
    require(report.get('status') == 'passed', 'Evaluation did not complete successfully')
    require(resolve_path(report['checkpoint']) == checkpoint, 'Evaluation checkpoint differs')
    require(report.get('checkpoint_sha256') == sha256(checkpoint / 'model.safetensors'),
            'Evaluation checkpoint hash differs')
    require(report.get('split') == 'test' and set(report['episode_indices']) == set(split['test_episode_indices'])
            and len(report['episode_indices']) == len(split['test_episode_indices']), 'Evaluation episodes differ')
    require(resolve_path(report['split_file']) == resolve_path(config['dataset']['split_file']),
            'Evaluation split path differs')
    require(report.get('source_split_seed') == split['seed'], 'Evaluation split seed differs')
    require(report.get('dataset_split_frames') == split['test_frames'], 'Evaluation split frame count differs')
    require(report.get('batch_size') == config['evaluation']['batch_size'], 'Evaluation batch size differs')
    require(report.get('chunk_size') == config['policy']['chunk_size'], 'Evaluation action chunk size differs')
    device = 'cuda' if config['policy']['device'] == 'cuda:0' else config['policy']['device']
    require(report.get('device') == device, 'Evaluation device differs')
    precision = 'bf16' if config['training']['mixed_precision'] == 'bf16' else 'fp32'
    require(report.get('precision') == precision, 'Evaluation precision differs')
    require(report.get('video_backend') == config['dataset']['video_backend'], 'Evaluation video backend differs')
    all_frames = config['evaluation']['all_frames']
    sampling = 'all_frames' if all_frames else 'evenly_spaced_within_every_episode_including_ends'
    require(report.get('sampling') == sampling, 'Evaluation sampling strategy differs')
    limit = None if all_frames else config['evaluation']['samples_per_episode']
    require(report.get('samples_per_episode_limit') == limit, 'Evaluation samples-per-episode differs')
    test_ids = set(split['test_episode_indices'])
    expected_count = sum(row['frames'] if all_frames else min(row['frames'], limit)
                         for row in split['episodes'] if row['episode_index'] in test_ids)
    require(report.get('planned_samples') == report.get('sample_count') == expected_count > 0,
            'Evaluation sample count is incomplete or differs')
    require(report.get('evaluated_episode_count') == len(test_ids), 'Evaluation missed test episodes')
    require(report.get('padding_excluded') is True and report.get('valid_chunk_action_count', 0) > 0,
            'Evaluation did not exclude padded actions')
    require(report.get('normalization_scope') == 'saved checkpoint processors, checked against training-only stats',
            'Evaluation did not verify training-only normalization')
    return report


def evaluation_path(training):
    path = resolve_path(training.get('test_evaluation', str(resolve_path(training['monitor_report']) / 'test_evaluation.json')))
    require(path.parent == resolve_path(training['monitor_report']), 'Automatic evaluation must use this run\'s monitor directory')
    return path


def ensure_evaluation(config_path, config, training, split):
    report_path = evaluation_path(training)
    if report_path.exists():
        try:
            validate_evaluation(read_json(report_path), training, config, split)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise ValueError(f'Existing evaluation cannot be reused: {report_path}. {exc}. '
                             'Use evaluate --checkpoint ... --report <new path> for a separate evaluation.') from exc
        print(f'Using matching completed evaluation: {report_path}', flush=True)
        return report_path
    precision = 'bf16' if config['training']['mixed_precision'] == 'bf16' else 'fp32'
    run_script('evaluate.py', '--config', config_path, '--checkpoint', training['checkpoint'], '--report', report_path,
               '--device', config['policy']['device'], '--precision', precision,
               '--video-backend', config['dataset']['video_backend'])
    validate_evaluation(read_json(report_path), training, config, split)
    return report_path


def ensure_benchmark(config_path, config, split, stats_hash, info):
    report_dir = resolve_path(config['benchmark']['report_dir'])
    report_path = report_dir / 'summary.json'
    if report_path.exists():
        report = read_json(report_path)
        require(report.get('candidates') == sorted(set(config['benchmark']['batches'])), 'Existing benchmark candidates differ; use another benchmark.report_dir')
        require(report.get('warmup_steps') == config['benchmark']['warmup']
                and report.get('timed_steps') == config['benchmark']['steps'],
                'Existing benchmark timing configuration differs; use another benchmark.report_dir')
        probe = deepcopy(config)
        probe['training']['batch_size'] = 'auto'
        choose_batch(probe, stats_hash, info)
        require(report.get('test_data_used') is False, 'Existing benchmark did not document training-only data')
        for result in report['results']:
            if result.get('status') == 'passed':
                require(result.get('train_stats_sha256') == stats_hash
                        and set(result['sampled_episode_indices']).issubset(split['train_episode_indices']),
                        'Existing benchmark used a different data split')
        print(f'Using matching completed benchmark: {report_path}', flush=True)
        return
    require(not report_dir.exists() or not any(report_dir.iterdir()),
            f'Benchmark directory has incomplete files: {report_dir}; choose another benchmark.report_dir')
    run_script('benchmark_batch.py', '--config', config_path)
    probe = deepcopy(config)
    probe['training']['batch_size'] = 'auto'
    choose_batch(probe, stats_hash, info)


def plot_arguments(config):
    plot = config['plot']
    return ['--formats', *plot['formats'], '--dpi', str(plot['dpi']),
            '--smoothing-window', str(plot['smoothing_window'])]


def summarize(training_report, evaluation_report=None):
    arguments = ['--training-report', str(training_report)]
    if evaluation_report is not None:
        arguments += ['--evaluation-report', str(evaluation_report)]
    run_script('summarize.py', *arguments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('stage', nargs='?', default='all', choices=(
        'all', 'prepare', 'train', 'benchmark', 'evaluate', 'plot', 'summarize'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args, extra = parser.parse_known_args(argv)
    config_path = resolve_path(args.config)
    try:
        if args.stage == 'all' and any(value in {'--help', '-h'} for value in extra):
            parser.print_help()
            print('Use ./run.sh setup to install dependencies; stage-specific options follow the stage name.')
            return 0
        if args.stage in {'evaluate', 'plot', 'summarize'} and any(value in {'--help', '-h'} for value in extra):
            run_script({'evaluate': 'evaluate.py', 'plot': 'plot_training.py', 'summarize': 'summarize.py'}[args.stage], '--help')
            return 0
        if args.stage in {'prepare', 'train', 'benchmark'}:
            script = {'prepare': 'prepare_data.py', 'train': 'train.py', 'benchmark': 'benchmark_batch.py'}[args.stage]
            run_script(script, '--config', config_path, *extra)
            return 0
        config = load_config(config_path)
        if args.stage == 'evaluate' and has_flag(extra, '--checkpoint'):
            run_script('evaluate.py', '--config', config_path, *extra)
            return 0
        if args.stage == 'plot' and (has_flag(extra, '--input') or has_flag(extra, '--log-dir')):
            run_script('plot_training.py', *plot_arguments(config), *extra)
            return 0
        if args.stage == 'summarize' and has_flag(extra, '--training-report'):
            run_script('summarize.py', *extra)
            return 0
        if args.stage == 'all':
            require(not extra, f'Unknown all-stage options: {extra}; put shared settings in the YAML config')
            precision = config['training']['mixed_precision']
            if config['training']['resume_config']:
                saved = read_json(resolve_path(config['training']['resume_config']))
                precision = saved['accelerator']['mixed_precision']
            run_script('check_environment.py', '--dataset-root', resolve_path(config['dataset']['root']),
                       '--precision', precision)
            run_script('prepare_data.py', '--config', config_path)
        split, stats_hash, info = load_split(config)
        if args.stage == 'all' and (config['benchmark']['enabled'] or config['training']['batch_size'] == 'auto'):
            ensure_benchmark(config_path, config, split, stats_hash, info)
        training, training_report, effective = find_training_result(
            config, split, stats_hash, info, required=args.stage != 'all')
        if args.stage == 'all':
            if training is None:
                run_script('train.py', '--config', config_path)
                training, training_report, effective = find_training_result(config, split, stats_hash, info, required=True)
            else:
                print(f'Using matching completed training: {training_report}', flush=True)
                if config['plot']['enabled']:
                    run_script('plot_training.py', '--log-dir', training['monitor_report'], *plot_arguments(config))
            evaluation_report = ensure_evaluation(config_path, effective, training, split) if config['evaluation']['enabled'] else None
            summarize(training_report, evaluation_report)
            print(f'Pipeline complete. Read {resolve_path(training["monitor_report"]) / "RESULTS.md"}', flush=True)
        elif args.stage == 'evaluate':
            if any(value in {'--help', '-h'} for value in extra):
                run_script('evaluate.py', '--help')
            elif extra:
                require(has_flag(extra, '--report'), 'Evaluation overrides require --report <new path> to preserve the run\'s default report')
                run_script('evaluate.py', '--config', config_path, '--checkpoint', training['checkpoint'], *extra)
            else:
                ensure_evaluation(config_path, effective, training, split)
        elif args.stage == 'plot':
            run_script('plot_training.py', '--log-dir', training['monitor_report'], *plot_arguments(config), *extra)
        elif args.stage == 'summarize':
            evaluation_report = evaluation_path(training) if config['evaluation']['enabled'] else None
            arguments = ['--training-report', str(training_report)]
            if evaluation_report is not None and evaluation_report.exists():
                validate_evaluation(read_json(evaluation_report), training, effective, split)
                arguments += ['--evaluation-report', str(evaluation_report)]
            run_script('summarize.py', *arguments, *extra)
        return 0
    except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as exc:
        print(f'Pipeline stopped: {exc}', file=sys.stderr, flush=True)
        return exc.returncode if isinstance(exc, subprocess.CalledProcessError) else 1


if __name__ == '__main__':
    raise SystemExit(main())
