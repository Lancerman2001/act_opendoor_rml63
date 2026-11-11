#!/usr/bin/env python3
"""Run official LeRobot training with exact, machine-readable step counters.

LeRobot 0.7 abbreviates 1,020 and 1,040 as the same ``1K`` in its console.
The metric display formatter and an optional full-precision loss logger are
configured here; optimization and checkpoint handling remain in the unmodified
official training entry point.
"""
import lerobot.utils.logging_utils as logging_utils


def exact_number(value, precision=0):
    return f'{value:.{precision}f}'


if __name__ == '__main__':
    logging_utils.format_big_number = exact_number
    from loss_logging import install_loss_logging
    import lerobot.scripts.lerobot_train as trainer

    loss_logger = install_loss_logging()
    original_update = trainer.update_policy
    last_tracker = None

    def remember_update(*args, **kwargs):
        global last_tracker
        result = original_update(*args, **kwargs)
        last_tracker = result[0]
        return result

    if loss_logger is not None:
        trainer.update_policy = remember_update
    try:
        trainer.main()
        # The official loop logs at exact intervals. Preserve the final partial
        # window too, including short smoke runs below the normal log interval.
        if loss_logger is not None and last_tracker is not None:
            loss_logger.write(last_tracker)  # count=0 after a logged window => no duplicate
    finally:
        trainer.update_policy = original_update
        if loss_logger is not None:
            loss_logger.close()
