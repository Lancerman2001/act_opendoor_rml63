#!/usr/bin/env bash
# Install the documented ACT environment without inheriting ROS/Conda imports.
set -euo pipefail
ACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ACT_ROOT/env.sh"
reuse_from=""
use_lock=1
check_gpu=1
write_lock=0
precision=bf16
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse-from)
      [[ $# -ge 2 ]] || { echo '--reuse-from requires an existing venv path' >&2; exit 2; }
      reuse_from="$2"; shift 2 ;;
    --no-lock) use_lock=0; shift ;;
    --skip-gpu-check) check_gpu=0; shift ;;
    --write-lock) write_lock=1; shift ;;
    --precision)
      [[ $# -ge 2 ]] || { echo '--precision requires bf16, fp16 or no' >&2; exit 2; }
      precision="$2"; shift 2 ;;
    -h|--help)
      echo 'Usage: ./setup.sh [--reuse-from PATH] [--precision bf16|fp16|no] [--no-lock] [--skip-gpu-check] [--write-lock]'
      echo 'Default: install requirements.txt constrained by requirements.lock.txt, then validate CUDA.'
      exit 0 ;;
    *) echo "Unknown setup option: $1" >&2; exit 2 ;;
  esac
done
case "$precision" in bf16|fp16|no) ;; *) echo 'Invalid --precision; choose bf16, fp16 or no' >&2; exit 2 ;; esac
mkdir -p "$ACT_ROOT/logs" "$ACT_ROOT/reports" "$ACT_ROOT/tools/ffmpeg"
command -v uv >/dev/null || {
  echo 'uv is required. Install it with: python3 -m pip install --user uv' >&2
  echo 'Official instructions: https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 1
}
if [[ ! -x "$ACT_ROOT/.venv/bin/python" ]]; then
  uv venv --python 3.12 "$ACT_ROOT/.venv"
fi
if [[ -n "$reuse_from" ]]; then
  "$ACT_ROOT/.venv/bin/python" "$ACT_ROOT/scripts/reuse_dependencies.py" \
    --source-venv "$reuse_from" --report "$ACT_ROOT/reports/dependency_reuse.json"
fi
install_args=(--python "$ACT_ROOT/.venv/bin/python" --torch-backend=cu130
  --index-strategy=unsafe-best-match -r "$ACT_ROOT/requirements.txt")
if [[ "$use_lock" == 1 && -f "$ACT_ROOT/requirements.lock.txt" ]]; then
  install_args+=(-c "$ACT_ROOT/requirements.lock.txt")
fi
uv pip install "${install_args[@]}"
uv pip check --python "$ACT_ROOT/.venv/bin/python"
"$ACT_ROOT/.venv/bin/python" "$ACT_ROOT/scripts/fetch_assets.py"

# A local CLI can reuse already-installed system FFmpeg libraries. Extracting
# this package does not install its shared-library dependencies.
if ! command -v ffmpeg >/dev/null; then
  if command -v apt-get >/dev/null && command -v dpkg-deb >/dev/null; then
    (
      cd "$ACT_ROOT/tools/ffmpeg"
      apt-get download ffmpeg
      for package in ./*.deb; do dpkg-deb -x "$package" .; done
    ) || {
      echo 'Install system FFmpeg first. Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg' >&2
      exit 1
    }
  else
    echo 'Install FFmpeg and its shared libraries using your operating-system package manager.' >&2
    exit 1
  fi
fi
if [[ "$check_gpu" == 1 ]]; then
  "$ACT_ROOT/.venv/bin/python" "$ACT_ROOT/scripts/check_environment.py" --precision "$precision"
else
  echo 'GPU validation skipped by explicit request; training still requires CUDA.'
fi
ACT_WRITE_LOCK="$write_lock" "$ACT_ROOT/.venv/bin/python" - <<'PY'
from importlib.metadata import distributions
from pathlib import Path
import os
import re
import sys
import sysconfig

root = Path(os.environ['ACT_PROJECT_ROOT'])
venv = (root / '.venv').resolve()
if Path(sys.prefix).resolve() != venv:
    raise RuntimeError('Refusing to freeze a Python environment outside this project')
sites = {Path(sysconfig.get_paths()[key]).resolve() for key in ('purelib', 'platlib')}
if any(not path.is_relative_to(venv) for path in sites):
    raise RuntimeError('Virtual-environment site-packages resolves outside .venv')
pins = {}
for dist in distributions(path=[str(path) for path in sorted(sites)]):
    name = dist.metadata.get('Name')
    if not name:
        continue
    key = re.sub(r'[-_.]+', '-', name).lower()
    if key in pins and pins[key] != dist.version:
        raise RuntimeError(f'Duplicate installed versions: {key}')
    pins[key] = dist.version
content = '# Generated from the project Python 3.12 virtual environment only.\n'
content += '\n'.join(f'{name}=={version}' for name, version in sorted(pins.items())) + '\n'
report = root / 'reports/installed_requirements.txt'
report.write_text(content)
print(f'Saved {len(pins)} environment-only version pins: {report}')
if os.environ['ACT_WRITE_LOCK'] == '1':
    (root / 'requirements.lock.txt').write_text(content)
    print('Updated requirements.lock.txt (--write-lock)')
PY
echo 'Setup complete. Configure configs/train.yaml, prepare your dataset, then run ./train.sh.'
