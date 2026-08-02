#!/usr/bin/env bash
# Dev: FastAPI + Tauri desktop (Vite HMR via beforeDevCommand)
# Aligns with danmo-work / danmo-inbox `make dev-desktop`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"
# shellcheck source=dev_process.sh
source "$SCRIPT_DIR/dev_process.sh"

APP_NAME="${DQ_APP_NAME:-danmo-make}"
BACKEND_PORT="${DQ_BACKEND_PORT}"
FRONTEND_PORT="${DQ_FRONTEND_PORT}"

dq_ensure_out_layout
"$SCRIPT_DIR/stop.sh" 2>/dev/null || true

echo "==> Starting $APP_NAME (dev-desktop) [${DQ_PROJECT}]"
echo "    Backend : http://127.0.0.1:${BACKEND_PORT}"
echo "    Desktop : Tauri webview (Vite HMR on :${FRONTEND_PORT})"
echo "    SKIP_BACKEND=1 to use an already-running API"

PYTHON311="/opt/homebrew/bin/python3.11"
if [[ ! -f "$PYTHON311" ]]; then
  PYTHON311="$(command -v python3.11 || true)"
fi
if [[ -z "$PYTHON311" || ! -f "$PYTHON311" ]]; then
  # Linux / generic: fall back to python3 if 3.11+
  PYTHON311="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON311" ]]; then
  echo "Python 3 not found" >&2
  exit 1
fi

VENV_DIR="$DQ_ROOT/.venv"
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* || "$(uname -s)" == CYGWIN* ]]; then
  VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
  VENV_PIP="$VENV_DIR/Scripts/pip.exe"
else
  VENV_PYTHON="$VENV_DIR/bin/python3"
  VENV_PIP="$VENV_DIR/bin/pip3"
fi

ensure_venv() {
  if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "==> Creating virtual environment..."
    "$PYTHON311" -m venv "$VENV_DIR"
  fi
  if ! "$VENV_PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "==> Installing Python dependencies..."
    "$VENV_PIP" install --upgrade pip -q
    "$VENV_PIP" install -r "$DQ_ROOT/requirements.txt" -q
  fi
  "$VENV_PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${DQ_ROOT}')
from backend.utils.config_paths import resolve_default_config_root
from backend.utils.workspace import prepare_data_directories
root = Path('${DQ_ROOT}').resolve()
default_cfg = resolve_default_config_root(bootstrap_root=root, bundle_root=None)
prepare_data_directories(root, default_config_root=default_cfg)
"
}

cd "$DQ_ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi

if [[ "${SKIP_BACKEND:-0}" == "1" ]]; then
  echo "==> SKIP_BACKEND=1: using external backend on :${BACKEND_PORT}"
  echo ""
else
  ensure_venv
  export DQ_DEV_ENV=$'DANQING_HTTP_PORT='"${BACKEND_PORT}"
  dq_dev_start backend "$DQ_ROOT" \
    "$VENV_PYTHON" -m uvicorn backend.main:app \
    --host 0.0.0.0 --port "$BACKEND_PORT" --reload
  unset DQ_DEV_ENV
  echo "==> Backend PID: $(cat "$DQ_RUN_DIR/backend.pid")"
  echo "    Logs: $DQ_RUN_DIR/backend.log"
  echo ""
fi

cleanup() {
  echo ""
  echo "==> Stopping dev processes..."
  "$SCRIPT_DIR/stop.sh" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting Tauri dev (Vite via beforeDevCommand)..."
echo "    Press Ctrl+C to stop"
echo ""

# Tauri build script requires bundle.resources["danqing-api"] to exist.
"$SCRIPT_DIR/ensure_tauri_resource_stub.sh"

cd "$DQ_ROOT/desktop"
if [[ ! -d node_modules ]]; then
  npm install
fi
npm run tauri dev
