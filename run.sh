#!/usr/bin/env bash
# One entry point for installation and the configuration-driven training pipeline.
set -euo pipefail
ACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ACT_ROOT/env.sh"
if [[ $# -eq 0 ]]; then set -- all; fi
stage="$1"
if [[ "$stage" == "--help" || "$stage" == "-h" ]]; then
  echo 'Usage: ./run.sh [all|setup|prepare|train|benchmark|evaluate|plot|summarize] [--config configs/train.yaml] [stage options]'
  echo 'The all stage installs a missing .venv, then runs the configured pipeline. Stage options are forwarded to the selected command.'
  exit 0
fi
if [[ "$stage" == "setup" ]]; then
  shift
  exec "$ACT_ROOT/setup.sh" "$@"
fi
if [[ "$stage" == -* ]]; then stage=all; fi
if [[ ! -x "$ACT_ROOT/.venv/bin/python" ]]; then
  if [[ "$stage" == "all" ]]; then
    # The pipeline checks the precision selected by YAML (or the resume checkpoint).
    "$ACT_ROOT/setup.sh" --skip-gpu-check
  else
    echo 'Project environment is missing. Run ./run.sh setup first.' >&2
    exit 1
  fi
fi
exec "$ACT_ROOT/.venv/bin/python" -u "$ACT_ROOT/scripts/pipeline.py" "$@"
