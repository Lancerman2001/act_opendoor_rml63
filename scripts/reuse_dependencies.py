#!/usr/bin/env python3
"""在正式安装前，可选通过 uv 的公开 CLI 离线复用旧转换环境的通用依赖。

不修改旧环境、不联网，不安装 GPU 包或 LeRobot。正式安装仍须由 setup.sh
完整解析并校验依赖约束。目标已有 torch 或 lerobot 时跳过，避免改动训练环境。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'lerobot', 'torch', 'torchvision', 'torchcodec', 'torchaudio', 'triton',
            'pytorch-triton', 'nvidia', 'cuda', 'cudnn', 'nccl', 'cutensor'}
GPU_PREFIXES = ('nvidia-', 'cuda-', 'cudnn-', 'nccl-', 'cutensor-', 'pytorch-triton-')


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def installed(venv):
    sites = list((venv / 'lib').glob('python*/site-packages'))
    return {normalized(dist.metadata['Name']): dist.version
            for dist in distributions(path=[str(site) for site in sites]) if dist.metadata.get('Name')}


def reuse(args):
    source = args.source_venv.expanduser().resolve()
    target = args.target_venv.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    report = {'status': 'skipped', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_venv': str(source), 'target_venv': str(target), 'cache_dir': str(cache),
              'offline': True, 'dependencies_resolved': False, 'link_mode': 'hardlink',
              'reused_packages': [],
              'note': 'Optional cache reuse only. Run the normal setup installer afterward to resolve all ACT and CUDA dependency constraints.'}
    uv = shutil.which('uv')
    if uv is None or not (source / 'pyvenv.cfg').is_file() or not cache.is_dir():
        report['reason'] = 'uv、旧环境或旧缓存不存在，跳过可选复用'
        return report
    if source == target or not (target / 'pyvenv.cfg').is_file() or not (target / 'bin/python').exists():
        report['reason'] = '目标必须是已创建且独立的虚拟环境，跳过可选复用'
        return report
    target_packages = installed(target)
    if {'lerobot', 'torch'} & target_packages.keys():
        report['reason'] = '目标已安装 lerobot 或 torch，保留现有训练环境'
        return report
    packages = {name: version for name, version in installed(source).items()
                if name not in EXCLUDED and not name.startswith(GPU_PREFIXES)}
    if not packages:
        report['reason'] = '旧环境没有可复用的通用依赖'
        return report
    pins = [f'{name}=={version}' for name, version in sorted(packages.items())]
    command = [uv, 'pip', 'install', '--python', str(target / 'bin/python'), '--cache-dir', str(cache),
               '--offline', '--no-deps', '--link-mode=hardlink', '--index-url', 'https://pypi.org/simple',
               '--extra-index-url', 'https://download.pytorch.org/whl/cpu',
               '--index-strategy=unsafe-best-match', *pins]
    report.update(command=command, requested_pins=pins)
    # CPU index 只用于定位旧缓存里的通用 wheel；pins 排除 GPU 包，--no-deps 禁止安装其依赖。
    # 两个官方索引都只查本机缓存和精确旧版本，避免默认 first-index 用 CPU 索引旧版本挡住 PyPI。
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['reason'] = f'可选离线复用未完成；继续正式安装：{type(exc).__name__}: {exc}'
        return report
    report.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode != 0:
        report['reason'] = '缓存缺项或离线安装失败，跳过整组复用；继续正式安装'
        return report
    actual = installed(target)
    mismatches = [pin for pin in pins if actual.get(pin.split('==', 1)[0]) != pin.split('==', 1)[1]]
    if mismatches:
        report['reason'] = '离线复用后的精确版本核对未通过；继续正式安装并完整校验依赖'
        report['version_mismatches'] = mismatches
        return report
    report.update(status='reused', reused_packages=pins, reused_package_count=len(pins))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-venv', type=Path, required=True,
                        help='Explicit existing environment to reuse; never modified')
    parser.add_argument('--target-venv', type=Path, default=ROOT / '.venv')
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / '.cache/uv')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--report', type=Path, help='可选 JSON 报告；默认只输出 stdout')
    args = parser.parse_args()
    report = reuse(args)
    output = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output)
    print(output, end='', flush=True)
    return 0  # 可选加速失败不阻止正式依赖安装。


if __name__ == '__main__':
    raise SystemExit(main())
