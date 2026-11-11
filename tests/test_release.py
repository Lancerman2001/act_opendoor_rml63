"""Source release boundaries and archive integrity; standard-library-only tests."""
from pathlib import Path
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('export_release', PROJECT / 'scripts/export_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'project'
        self.root.mkdir()
        for name in release.REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# portable fixture\n')

    def tearDown(self):
        self.temporary.cleanup()

    def test_runtime_data_and_local_settings_are_excluded(self):
        for name in ('data/private.py', 'runs/model.pt', 'reports/results.json', '.venv/secret.py',
                     '.cache/weights.py', 'vendor/private.py', 'logs/private.txt', '.env',
                     'configs/local.yaml', 'configs/train.local.yaml', 'scripts/__pycache__/secret.py'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('private payload')
        sources = release.collect_sources(self.root)
        self.assertEqual(set(sources), release.REQUIRED_FILES)

    def test_symlink_and_machine_path_are_rejected(self):
        readme = self.root / 'README.md'
        readme.unlink()
        outside = Path(self.temporary.name) / 'private.md'
        outside.write_text('private')
        readme.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            release.collect_sources(self.root)
        readme.unlink()
        readme.write_text('/' + 'home' + '/' + 'private-user' + '/dataset\n')
        with self.assertRaisesRegex(ValueError, 'Machine-specific'):
            release.collect_sources(self.root)

    def test_archives_manifest_hashes_and_executable_permissions(self):
        output = Path(self.temporary.name) / 'release'
        result = release.export_project(self.root, output)
        manifest = json.loads((output / result['manifest']).read_text())
        expected = {'act-training/' + item['path'] for item in manifest['files']} | {'act-training/MANIFEST.json'}
        with zipfile.ZipFile(output / 'act-training-src.zip') as archive:
            self.assertEqual(set(archive.namelist()), expected)
            for item in manifest['files']:
                data = archive.read('act-training/' + item['path'])
                self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])
            self.assertEqual((archive.getinfo('act-training/setup.sh').external_attr >> 16) & 0o777, 0o755)
        with tarfile.open(output / 'act-training-src.tar.gz') as archive:
            self.assertEqual(set(archive.getnames()), expected)
            for member in archive.getmembers():
                self.assertTrue(member.isfile())
                self.assertEqual((member.uid, member.gid, member.mtime), (0, 0, 0))
        for item in result['artifacts']:
            self.assertEqual(hashlib.sha256((output / item['file']).read_bytes()).hexdigest(), item['sha256'])
        with self.assertRaises(FileExistsError):
            release.export_project(self.root, output)

    def test_missing_required_file_and_invalid_release_name(self):
        (self.root / 'requirements.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            release.collect_sources(self.root)
        with self.assertRaises(ValueError):
            release.export_project(self.root, Path(self.temporary.name) / 'out', '../bad')


if __name__ == '__main__':
    unittest.main()
