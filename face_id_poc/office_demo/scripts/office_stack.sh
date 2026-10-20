#!/usr/bin/env bash
# Only this named Compose project; never stop or remove unrelated services.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OFFICE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export OFFICE_UID="${OFFICE_UID:-$(id -u)}"
export OFFICE_GID="${OFFICE_GID:-$(id -g)}"
: "${OFFICE_RUNTIME_DIR:?Set an absolute OFFICE_RUNTIME_DIR}"
: "${OFFICE_MODELS_DIR:?Set an absolute OFFICE_MODELS_DIR}"
export OFFICE_FIXTURE_DIR="${OFFICE_FIXTURE_DIR:-$OFFICE_RUNTIME_DIR/fixture}"
COMPOSE=(docker compose --project-name camex-office-demo --file "$OFFICE_DIR/docker-compose.yml")

wait_qdrant() {
  "${COMPOSE[@]}" run --rm --no-deps -T --entrypoint python app - <<'PY'
import time
import urllib.request
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen('http://qdrant:6333/readyz', timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit('Office Qdrant did not become ready in 60s; inspect this project logs.')
PY
}

case "${1:-help}" in
  up)
    case "$OFFICE_RUNTIME_DIR:$OFFICE_MODELS_DIR:$OFFICE_FIXTURE_DIR" in
      /*:/*:/*) ;;
      *) echo "Runtime, models and fixture paths must be absolute." >&2; exit 2 ;;
    esac
    test -d "$OFFICE_MODELS_DIR" || { echo "Model directory missing." >&2; exit 2; }
    for model in arcface-r50-w600k-112x112.onnx scrfd-kps-2.5g-640x640.onnx yolox-tiny-crowdhuman-416x416.onnx; do
      test -f "$OFFICE_MODELS_DIR/$model" || { echo "Missing supplied model: $model" >&2; exit 2; }
    done
    mkdir -p -- "$OFFICE_RUNTIME_DIR"
    chmod 700 "$OFFICE_RUNTIME_DIR"
    "${COMPOSE[@]}" up -d qdrant
    wait_qdrant
    "${COMPOSE[@]}" up -d app
    echo "Office app: http://127.0.0.1:9170 (inspect logs for backend/model readiness)"
    ;;
  fixture-up)
    test -f "$OFFICE_FIXTURE_DIR/manifest.json" || { echo "Prepare the fixture first." >&2; exit 2; }
    "${COMPOSE[@]}" --profile fixture up -d mediamtx fixture-publisher
    ;;
  fixture-down) "${COMPOSE[@]}" --profile fixture stop fixture-publisher mediamtx ;;
  down) "${COMPOSE[@]}" --profile fixture down ;;
  status) "${COMPOSE[@]}" --profile fixture ps ;;
  logs) "${COMPOSE[@]}" --profile fixture logs --tail 100 ;;
  config) "${COMPOSE[@]}" --profile fixture config --quiet ;;
  *) echo "Usage: bash scripts/office_stack.sh {up|fixture-up|fixture-down|down|status|logs|config}" ;;
esac
