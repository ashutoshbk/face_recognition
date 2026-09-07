#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PIP_CACHE_DIR="$ROOT/.cache/pip" TMPDIR="$ROOT/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
PYTHON="${PYTHON_BIN:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert (3,10)<=sys.version_info[:2]<(3,13)' 2>/dev/null; then PYTHON="$candidate"; break; fi
  done
fi
if [[ -z "$PYTHON" ]]; then echo 'Python 3.10-3.12 with venv is required.' >&2; exit 2; fi
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then "$PYTHON" -m venv "$ROOT/.venv"; fi
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT" pytest==8.4.2
"$ROOT/.venv/bin/python" "$ROOT/gpu/setup.py" "$@"
"$ROOT/.venv/bin/python" -m pytest "$ROOT/tests" -q
echo "Ready: bash $ROOT/launch.sh"
