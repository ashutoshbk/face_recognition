#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3.11}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), "Use Python 3.11 for this pinned environment"'
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install numpy==1.26.4 Cython==3.3.0 setuptools
.venv/bin/python -m pip install --no-build-isolation -r requirements.lock.txt
.venv/bin/python -m pip check
printf '\nDependencies ready. Next: python lab.py assets --model-bundle PATH_TO_ZIP --accept-research-terms\n'
