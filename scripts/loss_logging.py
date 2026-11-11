"""Persist full-precision LeRobot loss windows without changing optimization.

The official trainer logs a MetricsTracker before resetting its averages. Its
values are means over ``interval_steps`` updates (one update with --log-every 1).
``epoch`` uses LeRobot's fixed-batch sample counter, which slightly overcounts a
short final batch. ``weighted_kl_loss`` is derived as total loss minus L1; small
floating-point differences from kl_weight * kld_loss are expected.
"""
from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import os
from pathlib import Path
import threading
import time

from lerobot.utils.logging_utils import MetricsTracker


FIELDS = ('step', 'epoch', 'loss', 'l1_loss', 'kld_loss', 'weighted_kl_loss',
          'grad_norm', 'lr', 'interval_steps', 'elapsed_seconds', 'finite')
_installed: LossLogger | None = None


class LossLogger:
    """One pair of exclusive-create, line-buffered loss files for this attempt."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.paths = [self.directory / name for name in ('loss.csv', 'loss.jsonl')]
        existing = [str(path) for path in self.paths if path.exists() or path.is_symlink()]
        if existing:
            raise FileExistsError('Refusing to overwrite existing loss logs: ' + ', '.join(existing))
        self.closed = False
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._previous_factory = None
        self._factory = None
        self._streams = []
        try:
            for path in self.paths:
                self._streams.append(path.open('x', newline='', buffering=1))
            self._csv = csv.DictWriter(self._streams[0], fieldnames=FIELDS)
            self._csv.writeheader()
        except BaseException:
            # Only remove files this constructor created, never a racing writer's.
            for path, stream in zip(self.paths, self._streams):
                stream.close()
                path.unlink(missing_ok=True)
            raise

    def write(self, tracker: MetricsTracker) -> None:
        with self._lock:
            if self.closed:
                return
            count = tracker.metrics['loss'].count
            if count == 0:
                return
            if not math.isfinite(count) or count < 0 or int(count) != count:
                raise ValueError(f'Invalid loss aggregation window: {count!r}')
            values = tracker.to_dict()
            loss, l1 = values['loss'], values.get('l1_loss')
            row = {
                'step': int(values['steps']), 'epoch': values['epochs'],
                'loss': loss, 'l1_loss': l1, 'kld_loss': values.get('kld_loss'),
                'weighted_kl_loss': loss - l1 if l1 is not None else None,
                'grad_norm': values.get('grad_norm'), 'lr': values.get('lr'),
                'interval_steps': int(count),
                'elapsed_seconds': time.monotonic() - self.started,
            }
            finite = all(value is None or math.isfinite(value) for value in row.values())
            # Keep JSON valid on numerical failure. The unchanged console message
            # still contains nan/inf so run_monitored.py can stop the training job.
            row = {key: value if value is None or math.isfinite(value) else None
                   for key, value in row.items()}
            row['finite'] = finite
            try:
                self._csv.writerow(row)
                self._streams[1].write(json.dumps(row, allow_nan=False) + '\n')
            except (OSError, ValueError) as exc:
                raise RuntimeError(f'Cannot persist training losses in {self.directory}') from exc

    def close(self) -> None:
        global _installed
        with self._lock:
            if self.closed:
                return
            self.closed = True
            if self._factory is not None and logging.getLogRecordFactory() is self._factory:
                logging.setLogRecordFactory(self._previous_factory)
            try:
                for stream in self._streams:
                    stream.close()
            finally:
                if _installed is self:
                    _installed = None


def install_loss_logging(directory: str | Path | None = None) -> LossLogger | None:
    """Capture original numeric tracker records while preserving their display.

    Repeated installation for the same active directory is idempotent; changing
    directories while active is an error. A caller may close the returned writer
    explicitly; otherwise atexit closes it. Missing ACT_LOSS_DIR disables capture.
    """
    global _installed
    directory = directory if directory is not None else os.environ.get('ACT_LOSS_DIR')
    if not directory:
        return None
    target = Path(directory).expanduser().resolve()
    if _installed is not None:
        if _installed.directory != target:
            raise RuntimeError(f'Loss logging already targets {_installed.directory}')
        return _installed
    writer = LossLogger(target)
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        if isinstance(record.msg, MetricsTracker) and 'loss' in record.msg.metrics:
            writer.write(record.msg)
        return record

    writer._previous_factory = previous
    writer._factory = factory
    logging.setLogRecordFactory(factory)
    _installed = writer
    atexit.register(writer.close)
    return writer
