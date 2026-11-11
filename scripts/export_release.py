#!/usr/bin/env python3
"""Create a source-only GitHub upload bundle with a manifest and SHA-256 sums.

Uses an explicit allowlist. Dataset, model, environment, cache, logs and reports
are never traversed. Selected files must be plain UTF-8 source files; symlinks,
machine-specific home paths and oversized files are rejected for review.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    'README.md', 'requirements.txt', 'requirements.lock.txt', '.gitignore',
    '.gitattributes', '.python-version', 'LICENSE', 'THIRD_PARTY.md', 'NOTICE', 'CITATION.cff',
    'CONTRIBUTING.md', 'CHANGELOG.md', 'setup.sh', 'env.sh', 'train.sh',
    'run.sh', 'benchmark.sh', 'evaluate.sh', 'plot.sh', 'Makefile', 'pyproject.toml',
}
DIRECTORY_SUFFIXES = {
    'scripts': {'.py'},
    'configs': {'.yaml', '.yml'},
    'tests': {'.py', '.json', '.csv', '.yaml', '.yml', '.txt'},
    'docs': {'.md', '.txt'},
    '.github': {'.yml', '.yaml', '.md'},
}
REQUIRED_FILES = {
    'README.md', 'requirements.txt', '.gitignore', '.python-version',
    'setup.sh', 'env.sh', 'train.sh', 'run.sh', 'configs/train.yaml',
    'scripts/train.py', 'scripts/project_config.py', 'scripts/train_entry.py',
    'scripts/run_monitored.py', 'scripts/loss_logging.py', 'scripts/plot_training.py',
}
PRIVATE_PATHS = (
    re.compile(r'/(?:home|Users)/[^\s/`"<>]+'),
    re.compile(r'(?i)\b[A-Z]:[\\/](?:Users|home)[\\/][^\s`"<>]+'),
    re.compile(r'/mnt/[A-Za-z]/[^\s`"<>]+'),
)
MAX_SOURCE_BYTES = 2 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def excluded(relative: Path) -> bool:
    if any(part in {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'} for part in relative.parts):
        return True
    name = relative.name.lower()
    return name in {'local.yaml', 'local.yml'} or '.local.' in name or name.startswith('.env')


def collect_sources(root: Path) -> dict[str, bytes]:
    root = root.resolve()
    paths = [root / name for name in ROOT_FILES if (root / name).exists() or (root / name).is_symlink()]
    for directory, suffixes in DIRECTORY_SUFFIXES.items():
        base = root / directory
        if base.is_symlink():
            raise ValueError(f'Refusing a symlink source directory: {directory}')
        if base.is_dir():
            paths.extend(path for path in base.rglob('*')
                         if (path.is_file() or path.is_symlink()) and path.suffix.lower() in suffixes)
    sources = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if excluded(relative):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Refusing a symlink or external source file: {relative.as_posix()}')
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError(f'Source file exceeds 2 MiB: {relative.as_posix()}')
        data = path.read_bytes()
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError(f'Source must be UTF-8 text: {relative.as_posix()}') from exc
        if '\x00' in text:
            raise ValueError(f'Binary content in source allowlist: {relative.as_posix()}')
        if any(pattern.search(text) for pattern in PRIVATE_PATHS):
            raise ValueError(f'Machine-specific absolute path found; replace it with a relative path: {relative.as_posix()}')
        sources[relative.as_posix()] = data
    missing = sorted(REQUIRED_FILES - sources.keys())
    if missing:
        raise ValueError('Source bundle is incomplete: ' + ', '.join(missing))
    return sources


def create_manifest(sources: dict[str, bytes], name: str) -> dict:
    return {
        'format_version': 1,
        'project': name,
        'scope': 'Source/configuration/documentation only; no datasets, trained weights or runtime artifacts.',
        'files': [{'path': path, 'bytes': len(data), 'sha256': sha256(data)}
                  for path, data in sorted(sources.items())],
    }


def archive_files(sources: dict[str, bytes], name: str, manifest: bytes):
    for path, data in sorted({**sources, 'MANIFEST.json': manifest}.items()):
        yield f'{name}/{path}', data, 0o755 if path.endswith('.sh') else 0o644


def export_project(root: Path, output_dir: Path, name: str = 'act-training',
                   formats: tuple[str, ...] = ('zip', 'tar.gz'), force: bool = False) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Release name must contain only letters, numbers, dot, underscore or dash')
    if not formats or any(kind not in {'zip', 'tar.gz'} for kind in formats):
        raise ValueError('Supported archive formats: zip, tar.gz')
    sources = collect_sources(root)
    manifest = create_manifest(sources, name)
    manifest_data = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [f'{name}-src.{kind}' for kind in dict.fromkeys(formats)]
    targets += [f'{name}-src.manifest.json', f'{name}-src.SHA256SUMS']
    if not force and any((output_dir / item).exists() for item in targets):
        raise FileExistsError('Release output already exists; choose another --name/output directory or use --force')
    artifacts = []
    with tempfile.TemporaryDirectory(prefix='.release-', dir=output_dir) as temporary:
        stage = Path(temporary)
        for kind in dict.fromkeys(formats):
            destination = stage / f'{name}-src.{kind}'
            if kind == 'zip':
                with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                    for path, data, mode in archive_files(sources, name, manifest_data):
                        info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                        info.create_system = 3
                        info.external_attr = (0o100000 | mode) << 16
                        archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            else:
                with destination.open('wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as compressed:
                    with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                        for path, data, mode in archive_files(sources, name, manifest_data):
                            info = tarfile.TarInfo(path)
                            info.size = len(data)
                            info.mode = mode
                            info.mtime = 0
                            info.uid = info.gid = 0
                            info.uname = info.gname = ''
                            archive.addfile(info, io.BytesIO(data))
            artifacts.append({'file': destination.name, 'bytes': destination.stat().st_size,
                              'sha256': sha256(destination.read_bytes())})
        (stage / f'{name}-src.manifest.json').write_bytes(manifest_data)
        (stage / f'{name}-src.SHA256SUMS').write_text(
            ''.join(f'{item["sha256"]}  {item["file"]}\n' for item in artifacts))
        for filename in targets:
            source, target = stage / filename, output_dir / filename
            if force:
                os.replace(source, target)
            else:
                os.link(source, target)  # Atomic publication without replacing another release.
    return {'status': 'completed', 'source_file_count': len(sources),
            'source_bytes': sum(map(len, sources.values())), 'output_dir': str(output_dir.resolve()),
            'artifacts': artifacts, 'manifest': f'{name}-src.manifest.json',
            'checksums': f'{name}-src.SHA256SUMS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--name', default='act-training')
    parser.add_argument('--format', choices=('zip', 'tar.gz', 'both'), default='both')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.project_root.resolve()
    if args.check_only:
        sources = collect_sources(root)
        result = {'status': 'passed', 'source_file_count': len(sources), 'files': sorted(sources)}
    else:
        formats = ('zip', 'tar.gz') if args.format == 'both' else (args.format,)
        result = export_project(root, args.output_dir or root / 'dist', args.name, formats, args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
