"""CPU-only plot/data tests; no training dataset or GPU is used."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plot_training import load_history, moving_average, plot_history, source_file

HAS_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None


class TrainingPlotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def csv(self, name, rows, columns=None):
        path = self.root / name
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_resume_steps_and_full_precision(self):
        path = self.csv('loss.csv', [
            {'step': 20100, 'loss': 1.234567890123, 'l1_loss': 0.2, 'kld_loss': 0.1,
             'weighted_kl_loss': 1.034567890123, 'lr': 1e-5, 'interval_steps': 100},
            {'step': 20200, 'loss': 0.9, 'l1_loss': 0.1, 'kld_loss': 0.08,
             'weighted_kl_loss': 0.8, 'lr': 1e-5, 'interval_steps': 100},
        ])
        history = load_history(path)
        self.assertEqual(history['steps'], [20100, 20200])
        self.assertEqual(history['values']['loss'][0], 1.234567890123)
        self.assertEqual(history['logged_interval_steps'], [100])
        self.assertEqual(history['metrics_not_recorded'], [])

    def test_missing_components_are_not_derived(self):
        path = self.root / 'metrics.jsonl'
        path.write_text(json.dumps({'step': 100, 'loss': 2.123, 'lr': 1e-5}) + '\n')
        history = load_history(path)
        self.assertEqual(history['source_precision'], 'console_rounded')
        self.assertEqual(history['metrics_available'], ['loss', 'lr'])
        self.assertEqual(history['values']['weighted_kl_loss'], [None])

    def test_console_values_and_abbreviated_steps_are_labelled(self):
        path = self.root / 'console.log'
        path.write_text('INFO step:1K loss:1.234 lr:1.0e-05\nINFO step:1K loss:1.123 lr:1.0e-05\n')
        history = load_history(path)
        self.assertEqual(history['steps'], [1000, 1000])
        self.assertEqual(history['duplicate_rounded_step_rows'], 1)
        self.assertEqual(history['step_precision'], 'console_rounded')
        self.assertEqual(history['values']['kld_loss'], [None, None])

    def test_nonfinite_values_are_gaps_not_zero(self):
        path = self.csv('loss.csv', [
            {'step': 100, 'loss': 'nan', 'l1_loss': 'inf', 'finite': False},
            {'step': 200, 'loss': 0.5, 'l1_loss': '', 'finite': True},
            {'step': 'nan', 'loss': 0.4, 'l1_loss': 0.1, 'finite': False},
        ])
        history = load_history(path)
        self.assertEqual(history['steps'], [100, 200])
        self.assertEqual(history['values']['loss'], [None, 0.5])
        self.assertEqual(history['nonfinite_values']['loss'], 1)
        self.assertEqual(history['nonfinite_values']['l1_loss'], 1)
        self.assertEqual(history['flagged_nonfinite_rows'], 1)
        self.assertEqual(history['invalid_step_rows'], 1)
        self.assertNotIn('l1_loss', history['metrics_not_recorded'])
        self.assertIn('l1_loss', history['metrics_without_finite_values'])
        smoothed = moving_average([1.0, 3.0, None, 4.0, 6.0], 2)
        self.assertEqual(smoothed[:2], [1.0, 2.0])
        self.assertTrue(math.isnan(smoothed[2]))
        self.assertEqual(smoothed[3:], [4.0, 5.0])

    def test_empty_missing_required_and_all_nonfinite_rejected(self):
        empty = self.csv('empty.csv', [], ['step', 'loss'])
        missing = self.csv('missing.csv', [{'loss': 0.2}])
        nonfinite = self.csv('nonfinite.csv', [{'step': 1, 'loss': float('inf')}])
        for path in (empty, missing, nonfinite):
            with self.subTest(path=path.name), self.assertRaises(ValueError):
                load_history(path)

    def test_duplicate_exact_steps_and_regressions_rejected(self):
        for steps in ((1, 1), (2, 1)):
            path = self.csv('loss.csv', [{'step': step, 'loss': 0.1} for step in steps])
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                load_history(path)

    def test_source_priority_does_not_hide_empty_native_logs(self):
        self.csv('loss.csv', [], ['step', 'loss'])
        (self.root / 'metrics.jsonl').write_text('{"step":100,"loss":0.2}\n')
        chosen = source_file(log_dir=self.root)
        self.assertEqual(chosen.name, 'loss.csv')
        with self.assertRaises(ValueError):
            load_history(chosen)

    @unittest.skipUnless(HAS_MATPLOTLIB, 'Matplotlib is required for artifact rendering')
    def test_png_pdf_and_summary_from_jsonl(self):
        path = self.root / 'loss.jsonl'
        path.write_text('\n'.join(json.dumps({'step': s, 'loss': v, 'lr': 1e-5, 'interval_steps': 100})
                                  for s, v in ((10100, 2.1), (10200, 1.3), (10300, 1.0))) + '\n')
        result = plot_history(path, dpi=60, smoothing_window=2)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual((result['first_step'], result['last_step']), (10100, 10300))
        self.assertEqual(result['plot_config']['smoothing_unit'], 'log_records')
        for file in result['output_files']:
            file = Path(file)
            self.assertGreater(file.stat().st_size, 1000)
            expected = b'\x89PNG' if file.suffix == '.png' else b'%PDF'
            self.assertTrue(file.read_bytes().startswith(expected))
        saved = json.loads((self.root / 'plots/plot_summary.json').read_text())
        self.assertEqual(saved['metrics_not_recorded'], ['l1_loss', 'kld_loss', 'weighted_kl_loss'])


if __name__ == '__main__':
    unittest.main()
