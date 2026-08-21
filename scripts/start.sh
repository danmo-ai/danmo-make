#!/usr/bin/env bash
# Dev: FastAPI (--reload) + Vite HMR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"
# shellcheck source=dev_process.sh
source "$SCRIPT_DIR/dev_process.sh"
# shellcheck source=platform_requirements.sh
source "$SCRIPT_DIR/platform_requirements.sh"

dq_resolve_platform_requirements "$DQ_ROOT"

BACKEND_PORT="${DQ_BACKEND_PORT}"
FRONTEND_PORT="${DQ_FRONTEND_PORT}"

dq_ensure_out_layout
"$SCRIPT_DIR/stop.sh" 2>/dev/null || true

PYTHON311="/opt/homebrew/bin/python3.11"
if [[ ! -f "$PYTHON311" ]]; then
  PYTHON311="$(command -v python3.11 || true)"
fi
if [[ -z "$PYTHON311" || ! -f "$PYTHON311" ]]; then
  echo "Python 3.11 not found (brew install python@3.11)" >&2
  exit 1
fi

VENV_DIR="$DQ_ROOT/.venv"
VENV_PYTHON="$VENV_DIR/bin/python3"
VENV_PIP="$VENV_DIR/bin/pip3"

NEED_CREATE=0
if [[ ! -f "$VENV_PYTHON" || ! -f "$VENV_PIP" ]]; then
  NEED_CREATE=1
else
  VENV_PY_VERSION=$("$VENV_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
  if [[ "$VENV_PY_VERSION" != "3.11" ]]; then
    NEED_CREATE=1
  fi
fi

if [[ "$NEED_CREATE" -eq 1 ]]; then
  echo "Creating virtual environment (Python 3.11)..."
  rm -rf "$VENV_DIR"
  "$PYTHON311" -m venv "$VENV_DIR"
  "$VENV_PYTHON" -m ensurepip --upgrade
  "$VENV_PIP" install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip -q
fi

if ! "$VENV_PYTHON" -c "import fastapi, uvicorn, mlx, pydantic" 2>/dev/null; then
  echo "Installing dependencies from $DQ_PLATFORM_REQS ..."
  "$VENV_PIP" install -i https://pypi.tuna.tsinghua.edu.cn/simple -r "$DQ_PLATFORM_REQS" -q
fi

"$VENV_PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, '${DQ_ROOT}')
from backend.utils.config_paths import resolve_default_config_root
from backend.utils.user_home import ensure_control_plane
from backend.utils.workspace import prepare_data_directories
root = Path('${DQ_ROOT}').resolve()
default_cfg = resolve_default_config_root(bootstrap_root=root, bundle_root=None)
prepare_data_directories(
    root, control_plane=ensure_control_plane(), default_config_root=default_cfg
)
"

echo "==> Starting Danmo Make (dev) [${DQ_PROJECT}]"
echo "    Backend : http://127.0.0.1:${BACKEND_PORT}  (uvicorn --reload)"
echo "    Frontend: http://localhost:${FRONTEND_PORT}/  (Vite HMR)"
echo "    Stop: make stop"

cd "$DQ_ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi

export DQ_DEV_ENV=$'DANQING_HTTP_PORT='"${BACKEND_PORT}"
dq_dev_start backend "$DQ_ROOT" \
  "$VENV_PYTHON" -m uvicorn backend.main:app \
  --host 0.0.0.0 --port "$BACKEND_PORT" --reload
unset DQ_DEV_ENV

export DQ_DEV_ENV=$'DQ_BACKEND_PORT='"${BACKEND_PORT}"$'\nDQ_FRONTEND_PORT='"${FRONTEND_PORT}"
dq_dev_start frontend "$DQ_ROOT/frontend" npm run dev
unset DQ_DEV_ENV

echo "==> PIDs: backend=$(cat "$DQ_RUN_DIR/backend.pid") frontend=$(cat "$DQ_RUN_DIR/frontend.pid")"
echo "    Marker: $(cat "$DQ_RUN_DIR/project.marker")"
echo "    Logs: $DQ_RUN_DIR/backend.log $DQ_RUN_DIR/frontend.log"
