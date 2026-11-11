#!/usr/bin/env bash
set -euo pipefail
ACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$ACT_ROOT/env.sh"
exec "$ACT_ROOT/.venv/bin/python" -u "$ACT_ROOT/scripts/evaluate.py" "$@"
