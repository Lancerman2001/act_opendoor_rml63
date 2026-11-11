"""Protect against reusing a different experiment or skipping configured stages."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
import pipeline
from project_config import load_config


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(pipeline, 'ROOT', self.root)
        self.root_patch.start()
        self.config = load_config()
        self.output = self.root / 'runs/current'
        self.monitor = self.root / 'reports/current-attempt'
        self.checkpoint = self.output / 'checkpoints/000005/pretrained_model'
        self.config['dataset']['root'] = str(self.root / 'data/full')
        self.config['dataset']['split_file'] = str(self.root / 'data/split.json')
        self.config['training'].update(run_name='current', steps=5, batch_size=2,
                                       output_dir=str(self.output), mixed_precision='fp16')
        self.config['evaluation']['enabled'] = False
        self.split = {'train_episode_indices': [0, 2, 3, 4], 'test_episode_indices': [1],
                      'test_frames': 20, 'seed': 42, 'episodes': [{'episode_index': 1, 'frames': 20}]}
        self.stats_hash = 'a' * 64
        self.info = {}
        self.monitor.mkdir(parents=True)
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / 'model.safetensors').write_bytes(b'checkpoint fingerprint fixture')
        self.snapshot = self.monitor / 'resolved_config.yaml'
        self.snapshot.write_text(yaml.safe_dump(self.config))
        write_json(self.monitor / 'summary.json', {'status': 'passed', 'returncode': 0, 'stop_reasons': []})
        self.metadata_path = self.checkpoint.parent / 'training_state/training_step.json'
        self.metadata = {'step': 5, 'batch_size': 2, 'mixed_precision': 'fp16',
                         'dp_world_size': 1, 'grad_accum_steps': 1}
        write_json(self.metadata_path, self.metadata)
        self.saved = {key: self.config['training'][key] for key in
                      ('steps', 'batch_size', 'seed', 'num_workers', 'prefetch_factor', 'output_dir')}
        self.saved.update(accelerator={'mixed_precision': 'fp16'}, policy=deepcopy(self.config['policy']),
                          dataset={key: self.config['dataset'][key] for key in ('root', 'repo_id', 'video_backend')},
                          log_freq=self.config['logging']['every_steps'],
                          save_freq=self.config['checkpoint']['every_steps'], eval_steps=0)
        self.saved['dataset'].update(episodes=self.split['train_episode_indices'], eval_split=0)
        self.saved_path = self.checkpoint / 'train_config.json'
        write_json(self.saved_path, self.saved)
        self.report = {'status': 'completed', 'run_name': 'current', 'output': str(self.output),
                       'steps': 5, 'batch_size': 2, 'train_stats_sha256': self.stats_hash,
                       'split_file': self.config['dataset']['split_file'],
                       'config_snapshot': str(self.snapshot), 'monitor_report': str(self.monitor),
                       'checkpoint': str(self.checkpoint),
                       'test_evaluation': str(self.monitor / 'test_evaluation.json')}
        self.pointer = self.output / 'training_result.json'
        write_json(self.pointer, self.report)

    def tearDown(self):
        self.root_patch.stop()
        self.temporary.cleanup()

    def find(self, config=None, **kwargs):
        return pipeline.find_training_result(config or self.config, self.split, self.stats_hash, self.info, **kwargs)

    def test_run_result_takes_precedence_over_unrelated_global_pointer(self):
        write_json(self.root / 'reports/latest_training.json', {'output': str(self.root / 'runs/other')})
        report, path, effective = self.find(required=True)
        self.assertEqual(report, self.report)
        self.assertEqual(path, self.pointer)
        self.assertEqual(effective['training']['steps'], 5)
        self.pointer.unlink()
        with self.assertRaisesRegex(ValueError, 'No matching completed training'):
            self.find(required=True)

    def test_previous_short_run_cannot_satisfy_longer_fresh_training(self):
        wanted = deepcopy(self.config)
        wanted['training']['steps'] = 50000
        with self.assertRaisesRegex(ValueError, 'step count differs'):
            self.find(wanted)

    def test_checkpoint_step_and_normalization_are_checked_independently(self):
        write_json(self.metadata_path, {**self.metadata, 'step': 4})
        with self.assertRaisesRegex(ValueError, 'actual step'):
            self.find()
        write_json(self.metadata_path, self.metadata)
        write_json(self.pointer, {**self.report, 'train_stats_sha256': 'different statistics'})
        with self.assertRaisesRegex(ValueError, 'normalization fingerprint'):
            self.find()

    def test_existing_checkpoint_interval_must_match_requested_interval(self):
        requested = deepcopy(self.config)
        requested['checkpoint']['every_steps'] = 2
        with self.assertRaisesRegex(ValueError, 'checkpoint interval differs'):
            self.find(requested)

    def test_explicit_resume_can_extend_run_and_retains_checkpoint_settings(self):
        requested = deepcopy(self.config)
        requested['training'].update(resume_config=str(self.saved_path), steps=10,
                                     run_name='continued', batch_size=48, mixed_precision='bf16')
        requested['policy']['kl_weight'] = 99
        report, path, effective = self.find(requested)
        self.assertIsNone(report)
        self.assertIsNone(path)
        self.assertEqual(effective['training']['batch_size'], 2)
        self.assertEqual(effective['training']['mixed_precision'], 'fp16')
        self.assertEqual(effective['policy']['kl_weight'], self.saved['policy']['kl_weight'])

    def test_all_reuses_matching_training_and_checks_requested_precision(self):
        config_path = self.root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(self.config))
        with patch.object(pipeline, 'load_split', return_value=(self.split, self.stats_hash, self.info)), \
                patch.object(pipeline, 'run_script') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(['--config', str(config_path)]), 0)
        calls = [call.args for call in run.call_args_list]
        self.assertEqual(calls[0], ('check_environment.py', '--dataset-root',
                                   Path(self.config['dataset']['root']), '--precision', 'fp16'))
        self.assertEqual(calls[1], ('prepare_data.py', '--config', config_path))
        self.assertNotIn('train.py', [call[0] for call in calls])
        self.assertEqual(calls[-1], ('summarize.py', '--training-report', str(self.pointer)))

    def test_explicit_stage_overrides_are_forwarded_without_reusing_artifacts(self):
        config_path = self.root / 'config.yaml'
        config_path.write_text(yaml.safe_dump(self.config))
        with patch.object(pipeline, 'run_script') as run, \
                patch.object(pipeline, 'load_split', side_effect=AssertionError('must not read existing split')):
            self.assertEqual(pipeline.main(['train', '--config', str(config_path), '--steps', '25', '--log-every', '1']), 0)
        run.assert_called_once_with('train.py', '--config', config_path, '--steps', '25', '--log-every', '1')

    def test_resume_environment_check_uses_saved_precision(self):
        requested = deepcopy(self.config)
        requested['training'].update(resume_config=str(self.saved_path), mixed_precision='bf16', batch_size=48)
        config_path = self.root / 'resume.yaml'
        config_path.write_text(yaml.safe_dump(requested))
        with patch.object(pipeline, 'load_split', return_value=(self.split, self.stats_hash, self.info)), \
                patch.object(pipeline, 'run_script') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(['all', '--config', str(config_path)]), 0)
        first_call = run.call_args_list[0].args
        self.assertEqual(first_call[-2:], ('--precision', 'fp16'))
        self.assertNotIn('train.py', [call.args[0] for call in run.call_args_list])

    def test_evaluation_accepts_cuda_zero_alias_and_rejects_wrong_weights(self):
        config = deepcopy(self.config)
        config['policy']['device'] = 'cuda:0'
        report = {'status': 'passed', 'checkpoint': str(self.checkpoint),
                  'checkpoint_sha256': pipeline.sha256(self.checkpoint / 'model.safetensors'),
                  'split': 'test', 'episode_indices': [1], 'split_file': config['dataset']['split_file'],
                  'source_split_seed': 42, 'dataset_split_frames': 20,
                  'batch_size': config['evaluation']['batch_size'], 'chunk_size': config['policy']['chunk_size'],
                  'device': 'cuda', 'precision': 'fp32', 'video_backend': config['dataset']['video_backend'],
                  'sampling': 'evenly_spaced_within_every_episode_including_ends',
                  'samples_per_episode_limit': config['evaluation']['samples_per_episode'],
                  'planned_samples': 20, 'sample_count': 20, 'evaluated_episode_count': 1,
                  'padding_excluded': True, 'valid_chunk_action_count': 210,
                  'normalization_scope': 'saved checkpoint processors, checked against training-only stats'}
        self.assertEqual(pipeline.validate_evaluation(report, self.report, config, self.split), report)
        report['checkpoint_sha256'] = 'old checkpoint'
        with self.assertRaisesRegex(ValueError, 'checkpoint hash differs'):
            pipeline.validate_evaluation(report, self.report, config, self.split)


if __name__ == '__main__':
    unittest.main()
