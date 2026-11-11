#!/usr/bin/env python3
"""Run a child process while recording console, training metrics and hardware telemetry."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
PAIR = re.compile(r'([\w/]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[KMBT]?)')
NONFINITE = re.compile(r'\b(?:loss|grdn|grad_norm)\s*[:=]\s*[-+]?(?:nan|inf)\b', re.I)
FIELDS = ['elapsed_s', 'timestamp_utc', 'gpu_used_mib', 'gpu_total_mib', 'gpu_util_percent',
          'gpu_temp_c', 'gpu_power_w', 'gpu_clock_mhz', 'process_tree_rss_mib', 'host_available_mib']


def number(raw):
    scale = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    return float(raw[:-1]) * scale[raw[-1]] if raw[-1] in scale else float(raw)


def gpu_sample():
    keys = ['gpu_used_mib', 'gpu_total_mib', 'gpu_util_percent', 'gpu_temp_c', 'gpu_power_w', 'gpu_clock_mhz']
    try:
        line = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
            '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL, timeout=5).splitlines()[0]
        values = []
        for item in line.split(','):
            try: values.append(float(item.strip()))
            except ValueError: values.append(None)
        return dict(zip(keys, values))
    except (subprocess.SubprocessError, OSError, IndexError):
        return {key: None for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--temperature-limit', type=float, default=90)
    parser.add_argument('--min-host-free-mib', type=float, default=750)
    parser.add_argument('--config-snapshot', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('Missing command after --')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.name): parser.error('Invalid run name')
    import psutil
    report_dir = ROOT / 'reports' / args.name
    report_dir.mkdir(parents=True, exist_ok=False)
    plot_config = {'enabled': False}
    snapshot = None
    if args.config_snapshot:
        import yaml
        snapshot = report_dir / 'resolved_config.yaml'
        shutil.copyfile(args.config_snapshot, snapshot)
        resolved = yaml.safe_load(snapshot.read_text())
        plot_config = resolved.get('plot', plot_config)
    log_path = report_dir / 'console.log'
    started = time.monotonic()
    stop_reason = []
    metrics = []
    samples = []
    child_env = os.environ.copy()
    child_env['ACT_LOSS_DIR'] = str(report_dir)
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True, env=child_env)
    launch = {'command': command, 'pid': process.pid, 'cwd': str(ROOT),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'environment_overrides': {'ACT_LOSS_DIR': str(report_dir)}}
    if snapshot:
        launch['config_snapshot'] = str(snapshot)
    (report_dir / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')

    def read_output():
        with log_path.open('w', buffering=1) as log, (report_dir / 'metrics.jsonl').open('w', buffering=1) as metric_log:
            for raw in process.stdout:
                log.write(raw)
                print(raw, end='', flush=True)
                line = ANSI.sub('', raw)
                if NONFINITE.search(line): stop_reason.append('Nonfinite training loss/gradient')
                if 'loss:' in line and 'step:' in line:
                    row = {key: number(value) for key, value in PAIR.findall(line)
                           if key in {'step', 'smpl', 'ep', 'epch', 'loss', 'grdn', 'lr', 'data_s',
                                      'prep_s', 'updt_s', 'step_s', 'smp/s', 'mem_gb'}}
                    row['elapsed_s'] = time.monotonic() - started
                    metrics.append(row)
                    metric_log.write(json.dumps(row, allow_nan=False) + '\n')

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    def stop_process_group(first_signal):
        # The process can exit between poll(), a guard decision and killpg().
        # Preserve the final report even when that harmless race occurs.
        try:
            os.killpg(process.pid, first_signal)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    unsafe_streak = 0
    interrupted = False
    with (report_dir / 'telemetry.csv').open('w', newline='', buffering=1) as telemetry:
        writer = csv.DictWriter(telemetry, fieldnames=FIELDS)
        writer.writeheader()
        try:
            while process.poll() is None:
                row = gpu_sample()
                row.update(elapsed_s=round(time.monotonic() - started, 3),
                           timestamp_utc=datetime.now(timezone.utc).isoformat(),
                           host_available_mib=psutil.virtual_memory().available / 2**20)
                try:
                    family = [psutil.Process(process.pid)] + psutil.Process(process.pid).children(recursive=True)
                    row['process_tree_rss_mib'] = sum(p.memory_info().rss for p in family if p.is_running()) / 2**20
                except psutil.Error: row['process_tree_rss_mib'] = None
                writer.writerow(row)
                samples.append(row)
                hot = row['gpu_temp_c'] is not None and row['gpu_temp_c'] >= args.temperature_limit
                low_ram = row['host_available_mib'] < args.min_host_free_mib
                unsafe_streak = unsafe_streak + 1 if hot or low_ram else 0
                if unsafe_streak >= 3:
                    stop_reason.append('Sustained temperature or host-memory guard threshold')
                if stop_reason:
                    stop_process_group(signal.SIGTERM)
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            interrupted = True
            stop_reason.append('Interrupted by operator')
            stop_process_group(signal.SIGINT)
    code = process.wait()
    reader.join(timeout=10)
    def maximum(key):
        values = [row[key] for row in samples if row.get(key) is not None]
        return max(values) if values else None
    summary = {**launch, 'status': 'passed' if code == 0 and not stop_reason else 'failed',
               'returncode': code, 'stop_reasons': list(dict.fromkeys(stop_reason)),
               'elapsed_seconds': round(time.monotonic() - started, 3),
               'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'gpu_peak_used_mib': maximum('gpu_used_mib'), 'gpu_peak_temperature_c': maximum('gpu_temp_c'),
               'peak_process_tree_rss_mib': maximum('process_tree_rss_mib'),
               'host_min_available_mib': min((x['host_available_mib'] for x in samples), default=None),
               'telemetry_samples': len(samples), 'training_metric_samples': len(metrics),
               'first_training_metrics': metrics[0] if metrics else None,
               'last_training_metrics': metrics[-1] if metrics else None,
               'note': 'GPU usage is whole-board usage; process-tree RSS includes shared pages across workers.'}
    plot_result = {'status': 'disabled'}
    if plot_config.get('enabled'):
        command = [sys.executable, str(ROOT / 'scripts/plot_training.py'), '--log-dir', str(report_dir),
                   '--dpi', str(plot_config['dpi']), '--smoothing-window', str(plot_config['smoothing_window']),
                   '--formats', *plot_config['formats']]
        with (report_dir / 'plot.log').open('w') as stream:
            plotted = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        plot_result = {'status': 'passed' if plotted.returncode == 0 else 'failed',
                       'returncode': plotted.returncode, 'output_dir': str(report_dir / 'plots'),
                       'log': str(report_dir / 'plot.log')}
    summary['plot'] = plot_result
    summary['training_status'] = summary['status']
    if code == 0 and plot_result['status'] == 'failed':
        summary['status'] = 'failed'
        summary['stop_reasons'].append('Training finished, but requested plot generation failed; see plot.log')
    (report_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'monitor_report': str(report_dir), 'status': summary['status'], 'returncode': code}), flush=True)
    return 130 if interrupted else (code if code else int(summary['status'] != 'passed'))


if __name__ == '__main__':
    sys.exit(main())
