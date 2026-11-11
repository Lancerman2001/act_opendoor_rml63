#!/usr/bin/env python3
"""Render recorded ACT training losses with Matplotlib (no display or GPU needed).

The plots show training metrics, not evaluation scores or robot success rates.
Smoothing uses a trailing window of log records, not a number of optimizer steps.
Missing/nonfinite values remain gaps; unrecorded loss components are never inferred.
"""
from __future__ import annotations

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


METRICS = ('loss', 'l1_loss', 'kld_loss', 'weighted_kl_loss', 'lr')
ALIASES = {'step': ('step', 'steps'), 'lr': ('lr', 'learning_rate'),
           'kld_loss': ('kld_loss', 'kl_loss')}
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
PAIR = re.compile(
    r'([\w/]+)\s*[:=]\s*([-+]?(?:nan|inf(?:inity)?|'
    r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?[KMBT]?))\b', re.I)


def numeric(value):
    if value is None or isinstance(value, bool) or str(value).strip() == '':
        return None
    text = str(value).strip()
    suffix = text[-1:]
    try:
        result = float(text[:-1]) * {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}[suffix] \
            if suffix in ('K', 'M', 'B', 'T') else float(text)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'Invalid numeric training metric: {value!r}') from None
    return result


def source_file(log_dir: Path | None = None, input_path: Path | None = None) -> Path:
    if (log_dir is None) == (input_path is None):
        raise ValueError('Choose exactly one of --log-dir and --input')
    if input_path is not None:
        path = Path(input_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Training log not found: {path}')
        return path
    directory = Path(log_dir).expanduser().resolve()
    for name in ('loss.csv', 'loss.jsonl', 'metrics.jsonl', 'console.log'):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'No loss.csv, loss.jsonl, metrics.jsonl or console.log in {directory}')


def read_records(path: Path):
    columns, records = set(), []
    abbreviated_steps = False
    kind = 'csv' if path.suffix.lower() == '.csv' else 'jsonl' if path.suffix.lower() in ('.jsonl', '.ndjson') else 'console'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if kind == 'csv':
            reader = csv.DictReader(stream)
            columns.update(reader.fieldnames or ())
            records.extend(reader)
        elif kind == 'jsonl':
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Invalid JSON at {path.name}:{line_number}: {exc.msg}') from exc
                if not isinstance(row, dict):
                    raise ValueError(f'Expected a metric object at {path.name}:{line_number}')
                columns.update(row)
                records.append(row)
        else:
            for line in stream:
                row = dict(PAIR.findall(ANSI.sub('', line)))
                if 'step' not in row or 'loss' not in row:
                    continue
                abbreviated_steps |= row['step'][-1:] in ('K', 'M', 'B', 'T')
                columns.update(row)
                records.append(row)
    if not records:
        raise ValueError(f'No training records in {path}')
    if not {'step', 'steps'} & columns or 'loss' not in columns:
        raise ValueError('Training logs must contain step (or steps) and loss columns')
    return records, columns, kind, abbreviated_steps


def load_history(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    records, columns, kind, abbreviated_steps = read_records(path)
    values = {key: [] for key in METRICS}
    missing = dict.fromkeys(METRICS, 0)
    nonfinite = dict.fromkeys(METRICS, 0)
    steps, intervals = [], []
    skipped_steps = flagged_rows = duplicate_steps = 0

    def get(row, key):
        return next((row[name] for name in ALIASES.get(key, (key,)) if name in row), None)

    for row in records:
        step = numeric(get(row, 'step'))
        if step is None or not math.isfinite(step) or step < 0 or int(step) != step:
            skipped_steps += 1
            continue
        step = int(step)
        if steps and step < steps[-1]:
            raise ValueError('Steps must increase; plot a single training attempt instead of concatenated runs')
        if steps and step == steps[-1]:
            if not abbreviated_steps:
                raise ValueError(f'Duplicate step {step}; exact-step logs must be strictly increasing')
            duplicate_steps += 1
        steps.append(step)
        flagged_rows += str(row.get('finite', '')).lower() in ('false', '0')
        interval = numeric(row.get('interval_steps'))
        if interval is not None and math.isfinite(interval) and interval > 0:
            intervals.append(interval)
        for metric in METRICS:
            value = numeric(get(row, metric))
            if value is None:
                missing[metric] += 1
            elif not math.isfinite(value):
                nonfinite[metric] += 1
                value = None
            values[metric].append(value)
    if not steps or not any(value is not None for value in values['loss']):
        raise ValueError('No finite total-loss values with valid training steps are available')
    precision = 'console_rounded' if kind == 'console' or path.name == 'metrics.jsonl' else 'recorded_numeric_values'
    warnings = []
    if precision == 'console_rounded':
        warnings.append('Legacy console-derived values are rounded; their original precision cannot be recovered.')
    if skipped_steps:
        warnings.append(f'Skipped {skipped_steps} rows with missing, nonfinite or invalid step values.')
    if any(missing.values()) or any(nonfinite.values()):
        warnings.append('Missing/nonfinite metric values remain gaps; unrecorded components are not reconstructed.')
    if flagged_rows:
        warnings.append(f'{flagged_rows} input records have finite=false; inspect the training logs for numerical failures.')
    if abbreviated_steps:
        warnings.append('Console step counters contain K/M/B/T abbreviations; plotted steps use their reported, rounded values.')
    available = [key for key in METRICS if any(value is not None for value in values[key])]
    absent = [key for key in METRICS if not set(ALIASES.get(key, (key,))) & columns]
    return {'path': path, 'format': kind, 'source_precision': precision,
            'step_precision': 'console_rounded' if abbreviated_steps else 'recorded_integer_steps',
            'columns_received': sorted(columns), 'records_received': len(records),
            'records_plotted': len(steps), 'invalid_step_rows': skipped_steps,
            'duplicate_rounded_step_rows': duplicate_steps, 'flagged_nonfinite_rows': flagged_rows,
            'missing_values': missing, 'nonfinite_values': nonfinite,
            'metrics_available': available, 'metrics_not_recorded': absent,
            'metrics_without_finite_values': [key for key in METRICS if key not in available],
            'logged_interval_steps': sorted(set(intervals)), 'steps': steps, 'values': values, 'warnings': warnings}


def moving_average(values, window: int):
    """Trailing arithmetic mean, reset at gaps so smoothing cannot bridge missing data."""
    if window < 1:
        raise ValueError('Smoothing window must be positive')
    recent, output = deque(maxlen=window), []
    for value in values:
        if value is None or not math.isfinite(value):
            recent.clear()
            output.append(float('nan'))
        else:
            recent.append(value)
            output.append(sum(item / len(recent) for item in recent))
    return output


def plot_history(input_path: Path, output_dir: Path | None = None,
                 formats=('png', 'pdf'), dpi=160, smoothing_window=20) -> dict:
    if dpi < 1 or smoothing_window < 1:
        raise ValueError('DPI and smoothing window must be positive')
    formats = list(dict.fromkeys(formats))
    if not formats or any(fmt not in ('png', 'pdf', 'svg') for fmt in formats):
        raise ValueError('Plot formats must be png, pdf or svg')
    history = load_history(input_path)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    output = Path(output_dir).expanduser().resolve() if output_dir else history['path'].parent / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps = history['steps']
    colors = {'loss': '#2563eb', 'l1_loss': '#15803d', 'kld_loss': '#c2410c', 'weighted_kl_loss': '#7c3aed'}

    def loss_curve(axis, metric, label):
        vals = history['values'][metric]
        if metric not in history['metrics_available']:
            return False
        raw = [float('nan') if value is None else value for value in vals]
        color = colors[metric]
        if smoothing_window > 1:
            axis.plot(steps, raw, color=color, alpha=0.22, linewidth=0.8)
            axis.plot(steps, moving_average(vals, smoothing_window), color=color, linewidth=1.7,
                      label=f'{label}: trailing {smoothing_window}-record mean')
        else:
            axis.plot(steps, raw, color=color, linewidth=1.3, label=label)
        return True

    def unavailable(axis, metrics):
        label = 'Not recorded in this log' if all(key in history['metrics_not_recorded'] for key in metrics) \
            else 'No finite values recorded'
        axis.text(0.5, 0.5, label, transform=axis.transAxes,
                  ha='center', va='center', color='#6b7280')

    try:
        loss_curve(axes[0, 0], 'loss', 'Total loss')
        if not loss_curve(axes[0, 1], 'l1_loss', 'L1 loss'):
            unavailable(axes[0, 1], ('l1_loss',))
        kl = loss_curve(axes[1, 0], 'kld_loss', 'KL loss')
        weighted = loss_curve(axes[1, 0], 'weighted_kl_loss', 'Weighted KL loss')
        if not kl and not weighted:
            unavailable(axes[1, 0], ('kld_loss', 'weighted_kl_loss'))
        if 'lr' in history['metrics_available']:
            rates = [float('nan') if value is None else value for value in history['values']['lr']]
            axes[1, 1].step(steps, rates, where='post', color='#334155', label='Learning rate', linewidth=1.5)
        else:
            unavailable(axes[1, 1], ('lr',))
        titles = ('Total training loss', 'L1 reconstruction loss', 'KL training terms', 'Learning rate')
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title)
            axis.set_xlabel('Reported training step' if history['step_precision'] == 'console_rounded' else 'Training step')
            axis.set_ylabel('Learning rate' if title == 'Learning rate' else 'Loss')
            axis.ticklabel_format(axis='x', style='plain', useOffset=False)
            axis.grid(True, linewidth=0.6, alpha=0.25)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        precision = 'legacy rounded values' if history['source_precision'] == 'console_rounded' else 'recorded numeric values'
        figure.suptitle(f'ACT training curves | {precision}\n{history["path"].name}', fontsize=13)
        figure.text(0.02, 0.015, 'Values are as logged (may be window means). Pale curves are raw records; gaps are not filled.\n'
                    'Smoothing counts log records, not optimizer steps. These are training metrics, not test scores.',
                    fontsize=8, color='#475569')
        figure.tight_layout(rect=(0, 0.065, 1, 0.93))
        artifacts = []
        for fmt in formats:
            destination = output / f'training_curves.{fmt}'
            temporary = output / f'.training_curves.{fmt}.tmp'
            try:
                figure.savefig(temporary, format=fmt, dpi=dpi)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            artifacts.append(str(destination))
    finally:
        plt.close(figure)
    summary = {key: value for key, value in history.items() if key not in ('path', 'steps', 'values')}
    summary.update(schema_version=1, status='completed', generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   input=str(history['path']), input_sha256=hashlib.sha256(history['path'].read_bytes()).hexdigest(),
                   first_step=steps[0], last_step=steps[-1], output_files=artifacts,
                   plot_config={'formats': formats, 'dpi': dpi, 'smoothing_window': smoothing_window,
                                'smoothing_unit': 'log_records', 'smoothing': 'trailing_mean_reset_at_missing_values'},
                   matplotlib_version=matplotlib.__version__)
    summary_path = output / 'plot_summary.json'
    temporary = output / '.plot_summary.json.tmp'
    try:
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(summary_path)
    finally:
        temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--log-dir', type=Path, help='Prefer loss.csv, then loss.jsonl, metrics.jsonl, console.log')
    source.add_argument('--input', type=Path, help='Explicit CSV, JSONL or console log')
    parser.add_argument('--output-dir', type=Path, help='Default: input log directory / plots')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg'), default=['png', 'pdf'])
    parser.add_argument('--dpi', type=int, default=160)
    parser.add_argument('--smoothing-window', type=int, default=20, help='Trailing log-record count; 1 disables smoothing')
    args = parser.parse_args(argv)
    try:
        path = source_file(args.log_dir, args.input)
        result = plot_history(path, args.output_dir, args.formats, args.dpi, args.smoothing_window)
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({'status': result['status'], 'input': result['input'], 'files': result['output_files'],
                      'warnings': result['warnings']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
