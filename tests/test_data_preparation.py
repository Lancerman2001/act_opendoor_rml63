"""Episode split and train-only normalization on small synthetic v3 Parquet data.

No camera decoding or GPU computation is needed. The fixture deliberately uses
five episodes, 17 Hz, three state dimensions and two action dimensions.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from dataset_utils import validate_split

spec = importlib.util.spec_from_file_location('prepare_data_under_test', PROJECT / 'scripts/prepare_data.py')
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def serialized(stats):
    return {key: {name: np.asarray(value).tolist() for name, value in values.items()}
            for key, values in stats.items()}


def hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob('*') if path.is_file()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def build_v3_fixture(source):
    lengths = [3, 4, 5, 6, 7]
    features = {
        'observation.state': {'dtype': 'float32', 'shape': [3], 'names': ['state_a', 'state_b', 'state_c']},
        'action': {'dtype': 'float32', 'shape': [2], 'names': ['target_a', 'target_b']},
        'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'task_index': {'dtype': 'int64', 'shape': [1], 'names': None},
        'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
    }
    info = {'codebase_version': 'v3.0', 'robot_type': 'synthetic_fixture', 'fps': 17,
            'total_episodes': len(lengths), 'total_frames': sum(lengths), 'total_tasks': 1,
            'total_videos': 0, 'chunks_size': 1000, 'data_files_size_in_mb': 100,
            'video_files_size_in_mb': 500, 'splits': {'train': '0:5'}, 'features': features,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': None}
    rows, episodes, all_stats = [], [], []
    start = 0
    for episode_index, length in enumerate(lengths):
        step = np.arange(length, dtype=np.float32)
        state = np.stack([100 * episode_index + step,
                          200 * episode_index + 2 * step,
                          300 * episode_index - 3 * step], axis=-1)
        action = np.stack([state[:, 0] * .25 + 1, state[:, 1] * -.5], axis=-1)
        stats = compute_episode_stats({'observation.state': state, 'action': action}, features)
        all_stats.append(stats)
        episode = {'episode_index': episode_index, 'length': length,
                   'dataset_from_index': start, 'dataset_to_index': start + length,
                   'data/chunk_index': 0, 'data/file_index': 0,
                   'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
                   'tasks': ['synthetic task']}
        for feature, values in serialized(stats).items():
            episode.update({f'stats/{feature}/{name}': value for name, value in values.items()})
        episodes.append(episode)
        for frame in range(length):
            rows.append({'episode_index': episode_index, 'frame_index': frame, 'index': start + frame,
                         'task_index': 0, 'timestamp': frame / info['fps'],
                         'observation.state': state[frame].tolist(), 'action': action[frame].tolist()})
        start += length
    write_json(source / 'meta/info.json', info)
    write_json(source / 'meta/stats.json', serialized(aggregate_stats(all_stats)))
    metadata = source / 'meta/episodes/chunk-000/file-000.parquet'
    metadata.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(episodes), metadata)
    payload = source / 'data/chunk-000/file-000.parquet'
    payload.parent.mkdir(parents=True)
    schema = pa.schema([
        ('episode_index', pa.int64()), ('frame_index', pa.int64()), ('index', pa.int64()),
        ('task_index', pa.int64()), ('timestamp', pa.float32()),
        ('observation.state', pa.list_(pa.float32(), 3)), ('action', pa.list_(pa.float32(), 2)),
    ])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), payload)
    return rows, lengths


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source, self.output = root / 'source', root / 'prepared'
        self.rows, self.lengths = build_v3_fixture(self.source)
        self.before = hashes(self.source)
        self.args = argparse.Namespace(source=self.source, output=self.output,
                                       dataset_root=None, split_file=None, repo_id='local/synthetic',
                                       train_ratio=.8, seed=42)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, args=None):
        with redirect_stdout(io.StringIO()):
            preparation.prepare(args or self.args)
        return json.loads((self.output / 'split.json').read_text())

    def test_nonstandard_dimensions_fps_and_optional_manifest(self):
        self.assertFalse((self.source / 'source_manifest.json').exists())
        split = self.prepare()
        train, test = validate_split(split)
        expected = list(range(5))
        random.Random(42).shuffle(expected)
        self.assertEqual(train, sorted(expected[:4]))
        self.assertEqual(test, sorted(expected[4:]))
        self.assertEqual((split['train_episode_count'], split['test_episode_count']), (4, 1))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(train) | set(test), set(range(5)))
        self.assertEqual(split['train_frames'], sum(self.lengths[i] for i in train))
        self.assertEqual(split['test_frames'], sum(self.lengths[i] for i in test))
        target = self.output / 'full'
        info = json.loads((target / 'meta/info.json').read_text())
        self.assertEqual(info['fps'], 17)
        self.assertEqual(info['features']['observation.state']['shape'], [3])
        self.assertEqual(info['features']['action']['shape'], [2])
        self.assertTrue(all(row['source_session'] is None and row['source_episode'] is None for row in split['episodes']))
        self.assertEqual(hashes(self.source), self.before)
        self.assertEqual((target / 'meta/stats_full_dataset.json').read_bytes(), (self.source / 'meta/stats.json').read_bytes())
        self.assertEqual((target / 'data/chunk-000/file-000.parquet').read_bytes(),
                         (self.source / 'data/chunk-000/file-000.parquet').read_bytes())

    def test_normalizers_match_only_training_rows(self):
        split = self.prepare()
        stats = json.loads((self.output / 'full/meta/stats.json').read_text())
        all_stats = json.loads((self.source / 'meta/stats.json').read_text())
        selected = [row for row in self.rows if row['episode_index'] in split['train_episode_indices']]
        for feature in ('observation.state', 'action'):
            values = np.asarray([row[feature] for row in selected], dtype=np.float64)
            for operation in ('mean', 'std', 'min', 'max'):
                np.testing.assert_allclose(stats[feature][operation], getattr(np, operation)(values, axis=0),
                                           rtol=1e-12, atol=1e-12)
            self.assertEqual(stats[feature]['count'], [len(selected)])
            self.assertFalse(np.allclose(stats[feature]['mean'], all_stats[feature]['mean']))
        self.assertEqual((self.output / 'train_stats.json').read_bytes(), (self.output / 'full/meta/stats.json').read_bytes())
        self.assertEqual(hashes(self.source), self.before)

    def test_reuse_is_idempotent_and_rejects_changed_payload(self):
        split = self.prepare()
        before = hashes(self.output)
        self.assertEqual(self.prepare(), split)
        self.assertEqual(hashes(self.output), before)
        payload = self.output / 'full/data/chunk-000/file-000.parquet'
        payload.write_bytes(payload.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Prepared payload changed'):
            self.prepare()
        self.assertEqual(hashes(self.source), self.before)

    def test_reuse_rejects_changed_ratio_and_seed(self):
        self.prepare()
        for field, value, message in (('seed', 43, 'another seed'), ('train_ratio', .6, 'another ratio')):
            with self.subTest(field=field):
                args = argparse.Namespace(**vars(self.args))
                setattr(args, field, value)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare(args)

    def test_reuse_rejects_changed_source_payload_with_unchanged_info(self):
        self.prepare()
        info_hash = hashlib.sha256((self.source / 'meta/info.json').read_bytes()).hexdigest()
        payload = self.source / 'data/chunk-000/file-000.parquet'
        payload.write_bytes(payload.read_bytes() + b'changed source')
        self.assertEqual(hashlib.sha256((self.source / 'meta/info.json').read_bytes()).hexdigest(), info_hash)
        with self.assertRaisesRegex(ValueError, 'Source payload changed'):
            self.prepare()

    def test_reuse_works_without_the_original_source(self):
        split = self.prepare()
        output_hashes = hashes(self.output)
        moved = self.source.with_name('source-offline')
        self.source.rename(moved)
        self.assertFalse(self.source.exists())
        self.assertEqual(self.prepare(), split)
        self.assertEqual(hashes(self.output), output_hashes)
        self.assertEqual(hashes(moved), self.before)

    def test_invalid_source_manifest_is_rejected_before_payload_copy(self):
        for manifest, message in (([], 'episode count mismatch'),
                                  ([{'episode_index': i} for i in reversed(range(5))], 'order mismatch')):
            with self.subTest(message=message):
                write_json(self.source / 'source_manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
                self.assertFalse((self.output / 'full').exists())
                self.assertFalse((self.output / '.full.preparing').exists())
                self.assertFalse((self.output / 'split.json').exists())

    def test_split_validator_rejects_overlap_missing_and_wrong_stats_scope(self):
        split = self.prepare()
        mutations = []
        overlap = deepcopy(split)
        overlap['test_episode_indices'] = [overlap['train_episode_indices'][0]]
        mutations.append(overlap)
        omitted = deepcopy(split)
        omitted['train_episode_indices'] = omitted['train_episode_indices'][:-1]
        mutations.append(omitted)
        full_stats = deepcopy(split)
        full_stats['normalization_scope'] = 'all episodes'
        mutations.append(full_stats)
        for invalid in mutations:
            with self.assertRaises(ValueError):
                validate_split(invalid)


if __name__ == '__main__':
    unittest.main()
