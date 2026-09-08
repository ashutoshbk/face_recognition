#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
backend=docker
python_bin=python3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) backend="${2:?missing backend}"; shift 2 ;;
        --python) python_bin="${2:?missing Python path}"; shift 2 ;;
        -h|--help) printf 'Usage: bash install.sh --backend docker|cpu|cuda [--python python3.11]\n'; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$backend" in docker|cpu|cuda) ;; *) printf 'Invalid backend\n' >&2; exit 2 ;; esac
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
if [[ "$backend" == docker ]]; then
    command -v docker >/dev/null || { printf 'Install Docker and NVIDIA Container Toolkit first, or choose --backend cpu.\n' >&2; exit 1; }
    docker info --format '{{.OSType}}' >/dev/null
    docker build -t camex-faceid:0.1.1 .
    docker run --rm --network none --entrypoint python camex-faceid:0.1.1 -m pytest -q -p no:cacheprovider
else
    "$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10-3.12 (3.10/3.11 tested)"'
    if [[ "$backend" == cuda && "$(uname -s)" != Linux ]]; then
        printf 'Native CUDA setup is supported on Linux. Use CPU on macOS.\n' >&2; exit 1
    fi
    environment=".venv-$backend"
    if [[ ! -x "$environment/bin/python" ]]; then "$python_bin" -m venv "$environment"; fi
    "$environment/bin/python" -m pip install -r "requirements-$backend.txt"
    "$environment/bin/python" -m pip install --no-deps -e .
    "$environment/bin/python" -m pytest -q -p no:cacheprovider
fi
printf '\nSetup finished. Run: python3 scripts/run_demo.py --backend %s --help\n' "$backend"
