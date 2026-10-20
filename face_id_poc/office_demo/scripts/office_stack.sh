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
COMPOSE=(docker compose --project-name "${OFFICE_PROJECT_NAME:-camex-office-demo}" --file "$OFFICE_DIR/docker-compose.yml")

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

wait_app() {
  python3 - "${OFFICE_APP_PORT:-9170}" <<'PY'
import json, sys, time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = 'http://127.0.0.1:' + sys.argv[1] + '/api/health'
deadline = time.monotonic() + 120
last = None
while time.monotonic() < deadline:
    try:
        with opener.open(url, timeout=3) as response:
            last = json.load(response)
        if last.get('ready') is True:
            print('Office backend ready:', url)
            break
        if last.get('engine_error'):
            raise SystemExit('Model startup failed: ' + str(last['engine_error']))
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit('Office backend not ready after 120s. Inspect office_stack.sh logs. Last health: '+str(last))
PY
}

case "${1:-help}" in
  up)
    case "$OFFICE_RUNTIME_DIR:$OFFICE_MODELS_DIR:$OFFICE_FIXTURE_DIR" in
      /*:/*:/*) ;;
      *) echo "Runtime, models and fixture paths must be absolute." >&2; exit 2 ;;
    esac
    test -d "$OFFICE_MODELS_DIR" || { echo "Model directory missing." >&2; exit 2; }
    case "${OFFICE_FACE_MODEL:-scrfd_2.5g}" in
      scrfd_2.5g) face_model=scrfd-kps-2.5g-640x640.onnx ;;
      scrfd_10g) face_model=det_10g.onnx ;;
      *) echo "OFFICE_FACE_MODEL must be scrfd_2.5g or scrfd_10g." >&2; exit 2 ;;
    esac
    for model in arcface-r50-w600k-112x112.onnx "$face_model" yolox-tiny-crowdhuman-416x416.onnx; do
      test -f "$OFFICE_MODELS_DIR/$model" || { echo "Missing supplied model: $model" >&2; exit 2; }
    done
    mkdir -p -- "$OFFICE_RUNTIME_DIR"
    chmod 700 "$OFFICE_RUNTIME_DIR"
    "${COMPOSE[@]}" up -d qdrant
    wait_qdrant
    "${COMPOSE[@]}" up -d app
    wait_app
    echo "Office app: http://127.0.0.1:${OFFICE_APP_PORT:-9170}"
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
  doctor) "${COMPOSE[@]}" exec -T app python scripts/check_runtime.py --exercise-models ;;
  *) echo "Usage: bash scripts/office_stack.sh {up|fixture-up|fixture-down|down|status|logs|config|doctor}" ;;
esac
