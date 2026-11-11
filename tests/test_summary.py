"""Run-report validation with tiny CPU tensors and synthetic metadata only."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize import build_report, markdown, sha256

HAS_RUNTIME = all(importlib.util.find_spec(name) is not None for name in ('torch', 'safetensors', 'yaml'))


@unittest.skipUnless(HAS_RUNTIME, 'Runtime dependencies are required for checkpoint checks')
class SummaryTests(unittest.TestCase):
    def setUp(self):
        import torch
        import yaml
        from safetensors.torch import save_file
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'runs/unit'
        self.checkpoint = self.output / 'checkpoints/000003/pretrained_model'
        self.state = self.checkpoint.parent / 'training_state'
        self.monitor = self.root / 'reports/unit_attempt'
        self.dataset = self.root / 'data/full'
        for directory in (self.checkpoint, self.state, self.monitor, self.dataset / 'meta'):
            directory.mkdir(parents=True)
        self.write(self.dataset / 'meta/stats.json', {'action': {'mean': [0, 0], 'std': [1, 1]}})
        self.write(self.dataset / 'meta/info.json', {'total_episodes': 5, 'total_frames': 25, 'fps': 17,
                   'features': {'action': {'shape': [2], 'names': ['linear_x', 'joint_y']}}})
        self.split = {'status': 'prepared', 'repo_id': 'local/unit', 'dataset_root': 'full',
                      'normalization_scope': 'training episodes only', 'seed': 7, 'total_episodes': 5,
                      'total_frames': 25, 'train_episode_indices': [0, 2, 4], 'test_episode_indices': [1, 3],
                      'train_episode_count': 3, 'test_episode_count': 2, 'train_frames': 15, 'test_frames': 10,
                      'train_stats_sha256': sha256(self.dataset / 'meta/stats.json')}
        self.split_path = self.root / 'data/split.json'
        self.write(self.split_path, self.split)
        self.snapshot = {'dataset': {'root': 'data/full', 'split_file': 'data/split.json'},
                         'training': {'steps': 3, 'batch_size': 2}, 'logging': {'every_steps': 2},
                         'checkpoint': {'every_steps': 2}, 'plot': {'enabled': False, 'formats': ['png']}}
        self.snapshot_path = self.monitor / 'resolved_config.yaml'
        self.snapshot_path.write_text(yaml.safe_dump(self.snapshot))
        self.saved_config = {'steps': 3, 'batch_size': 2, 'output_dir': str(self.output),
                             'dataset': {'root': str(self.dataset), 'episodes': [0, 2, 4], 'eval_split': 0},
                             'eval_steps': 0, 'policy': {'type': 'act', 'chunk_size': 5, 'kl_weight': 10},
                             'log_freq': 2, 'save_freq': 2}
        self.write(self.checkpoint / 'train_config.json', self.saved_config)
        self.write(self.state / 'training_step.json', {'step': 3, 'batch_size': 2, 'mixed_precision': 'bf16'})
        self.write(self.checkpoint / 'policy_preprocessor.json', {})
        self.write(self.checkpoint / 'policy_postprocessor.json', {})
        save_file({'weight': torch.tensor([0.1, 0.2])}, str(self.checkpoint / 'model.safetensors'))
        save_file({'mean': torch.tensor([0.0, 0.0])}, str(self.checkpoint / 'processor.safetensors'))
        save_file({'moment': torch.tensor([0.0, 0.0])}, str(self.state / 'optimizer_state.safetensors'))
        self.write(self.monitor / 'summary.json', {'status': 'passed', 'returncode': 0, 'stop_reasons': [],
                   'elapsed_seconds': 1.25, 'gpu_peak_used_mib': 128, 'gpu_peak_temperature_c': 40})
        self.rows = [{'step': 2, 'loss': 0.5, 'l1_loss': 0.3, 'kld_loss': 0.02, 'weighted_kl_loss': 0.2,
                      'lr': 1e-5, 'interval_steps': 2, 'finite': True},
                     {'step': 3, 'loss': 0.4, 'l1_loss': 0.2, 'kld_loss': 0.02, 'weighted_kl_loss': 0.2,
                      'lr': 1e-5, 'interval_steps': 1, 'finite': True}]
        self.loss_path = self.monitor / 'loss.jsonl'
        self.loss_path.write_text('\n'.join(json.dumps(row) for row in self.rows) + '\n')
        self.latest = {'status': 'completed', 'run_name': 'unit', 'output': str(self.output),
                       'checkpoint': str(self.checkpoint), 'steps': 3, 'batch_size': 2,
                       'monitor_report': str(self.monitor), 'config_snapshot': str(self.snapshot_path),
                       'train_stats_sha256': self.split['train_stats_sha256'],
                       'test_evaluation': str(self.monitor / 'test_evaluation.json')}
        self.training_path = self.root / 'reports/latest_training.json'
        self.write(self.training_path, self.latest)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')

    def report(self, evaluation=None):
        return build_report(self.training_path, evaluation, project_root=self.root)

    def evaluation(self):
        result = {'status': 'passed', 'split': 'test', 'checkpoint': str(self.checkpoint),
                  'checkpoint_sha256': sha256(self.checkpoint / 'model.safetensors'),
                  'split_file': str(self.split_path), 'episode_indices': [1, 3],
                  'evaluated_episode_count': 2, 'sample_count': 2, 'planned_samples': 2,
                  'dataset_split_frames': 10, 'padding_excluded': True, 'valid_chunk_action_count': 8,
                  'normalization_scope': 'saved checkpoint processors, checked against training-only stats',
                  'chunk_size': 5, 'normalized_masked_l1': 0.1,
                  'metric_units': {'linear_x': 'meters', 'joint_y': 'radians'}}
        for name in ('first_action_mae', 'chunk_action_mae', 'hold_first_action_mae', 'hold_chunk_action_mae'):
            result[name] = {'linear_x': 0.1, 'joint_y': 0.2}
        return result

    def test_optional_evaluation_is_not_replaced_with_global_history(self):
        self.write(self.root / 'reports/test_evaluation.json', {'status': 'passed', 'checkpoint': '/wrong/historical'})
        report = self.report()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['evaluation']['status'], 'not_run')
        self.assertEqual(report['dataset']['train_episode_count'], 3)
        self.assertEqual(report['checkpoint_integrity']['tensor_count'], 3)
        self.assertEqual(report['training']['interval_steps_observed'], [1, 2])
        self.assertIn('未执行或未关联独立测试', markdown(report, self.monitor))

    def test_wrong_checkpoint_step_and_batch_are_rejected(self):
        for changed in ({'step': 2, 'batch_size': 2}, {'step': 3, 'batch_size': 3}):
            self.write(self.state / 'training_step.json', changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.report()

    def test_snapshot_policy_precision_and_run_identity_are_verified(self):
        import copy
        import yaml
        cases = [
            {'policy': {'kl_weight': 99}},
            {'training': {**self.snapshot['training'], 'mixed_precision': 'fp16'}},
            {'training': {**self.snapshot['training'], 'run_name': 'unrelated-run'}},
        ]
        original = copy.deepcopy(self.snapshot)
        for changes in cases:
            self.snapshot_path.write_text(yaml.safe_dump({**original, **changes}))
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, '配置快照'):
                self.report()

    def test_checkpoint_precision_and_topology_are_verified(self):
        self.saved_config['accelerator'] = {'mixed_precision': 'bf16'}
        self.write(self.checkpoint / 'train_config.json', self.saved_config)
        for change in ({'mixed_precision': 'fp16'}, {'grad_accum_steps': 2}, {'dp_world_size': 2}):
            self.write(self.state / 'training_step.json', {
                'step': 3, 'batch_size': 2, 'mixed_precision': 'bf16', **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.report()

    def test_skip_evaluation_ignores_run_linked_report(self):
        self.write(self.monitor / 'test_evaluation.json', {'status': 'failed', 'checkpoint': '/unrelated'})
        report = build_report(self.training_path, project_root=self.root, skip_evaluation=True)
        self.assertEqual(report['evaluation']['status'], 'not_run')
        self.assertEqual(report['evaluation']['selection'], 'disabled')

    def test_requested_plot_must_exist(self):
        import yaml
        self.snapshot['plot']['enabled'] = True
        self.snapshot_path.write_text(yaml.safe_dump(self.snapshot))
        with self.assertRaisesRegex(ValueError, '绘图'):
            self.report()

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') is not None, 'Matplotlib required')
    def test_plot_source_and_relative_markdown_link(self):
        import yaml
        from plot_training import plot_history
        self.snapshot['plot']['enabled'] = True
        self.snapshot_path.write_text(yaml.safe_dump(self.snapshot))
        plots = plot_history(self.loss_path, formats=['png'], dpi=40, smoothing_window=1)
        report = self.report()
        self.assertEqual(report['plots']['status'], 'completed')
        self.assertIn('](plots/training_curves.png)', markdown(report, self.monitor))
        Path(plots['output_files'][0]).unlink()
        with self.assertRaisesRegex(ValueError, '图片'):
            self.report()

    @unittest.skipUnless(importlib.util.find_spec('matplotlib') is not None, 'Matplotlib required')
    def test_plot_components_must_match_full_precision_log(self):
        import csv
        from plot_training import plot_history
        alternate = self.monitor / 'loss.csv'
        with alternate.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows([{**row, 'l1_loss': row['l1_loss'] + 0.1} for row in self.rows])
        plot_history(alternate, formats=['png'], dpi=40, smoothing_window=1)
        with self.assertRaisesRegex(ValueError, 'l1_loss'):
            self.report()

    def test_evaluation_requires_exact_model_and_test_episode_set(self):
        path = self.monitor / 'test_evaluation.json'
        good = self.evaluation()
        for change in ({'checkpoint_sha256': 'bad'}, {'episode_indices': [0, 1]}, {'checkpoint': '/wrong/model'}):
            self.write(path, {**good, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.report()
        self.write(path, good)
        report = self.report()
        self.assertEqual(report['evaluation']['status'], 'passed')
        self.assertEqual(report['evaluation']['metric_units'], {'linear_x': 'meters', 'joint_y': 'radians'})
        text = markdown(report, self.monitor)
        self.assertIn('linear_x | meters', text)
        self.assertIn('joint_y | radians', text)

    def test_stats_and_episode_partition_are_verified(self):
        self.write(self.dataset / 'meta/stats.json', {'action': {'mean': [1, 0], 'std': [1, 1]}})
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.report()

    def test_nonfinite_checkpoint_tensor_is_rejected(self):
        import torch
        from safetensors.torch import save_file
        save_file({'bad': torch.tensor([float('nan')])}, str(self.checkpoint / 'model.safetensors'))
        with self.assertRaisesRegex(ValueError, '非有限'):
            self.report()

    def test_legacy_rounded_log_is_explicit_and_may_end_before_final_step(self):
        self.loss_path.unlink()
        (self.monitor / 'metrics.jsonl').write_text('{"step":2,"loss":0.500,"lr":1e-5}\n')
        report = self.report()
        self.assertEqual(report['training']['loss_source_precision'], 'console_rounded')
        self.assertIsNone(report['training']['metrics']['kld_loss'])
        self.assertEqual(report['training']['last_logged_step'], 2)
        self.assertTrue(report['warnings'])


if __name__ == '__main__':
    unittest.main()
