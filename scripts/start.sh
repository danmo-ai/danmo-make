#!/usr/bin/env bash
# Dev: Rust danqing-api + Vite HMR. There is no Python runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"
# shellcheck source=dev_process.sh
source "$SCRIPT_DIR/dev_process.sh"

BACKEND_PORT="${DQ_BACKEND_PORT}"
FRONTEND_PORT="${DQ_FRONTEND_PORT}"

dq_ensure_out_layout
"$SCRIPT_DIR/stop.sh" 2>/dev/null || true

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust: https://rustup.rs/" >&2
  exit 1
fi

echo "==> Starting Danmo Make (dev) [${DQ_PROJECT}]"
echo "    Backend : http://127.0.0.1:${BACKEND_PORT}  (danqing-api, ggml)"
echo "    Frontend: http://localhost:${FRONTEND_PORT}/  (Vite HMR)"
echo "    Stop: make stop"

cd "$DQ_ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi

mkdir -p "$DQ_ROOT/db"
export DQ_DEV_ENV=$'DANQING_HTTP_PORT='"${BACKEND_PORT}"$'\nDANQING_HTTP_HOST=127.0.0.1'$'\nDANQING_DB='"$DQ_ROOT/db/studio.db"$'\nDANQING_REGISTRY='"$DQ_ROOT/default_config/models_registry.json"
dq_dev_start backend "$DQ_ROOT" \
  cargo run --manifest-path "$DQ_ROOT/native/Cargo.toml" -p danmo-server
unset DQ_DEV_ENV

export DQ_DEV_ENV=$'DQ_BACKEND_PORT='"${BACKEND_PORT}"$'\nDQ_FRONTEND_PORT='"${FRONTEND_PORT}"
dq_dev_start frontend "$DQ_ROOT/frontend" npm run dev
unset DQ_DEV_ENV

echo "==> PIDs: backend=$(cat "$DQ_RUN_DIR/backend.pid") frontend=$(cat "$DQ_RUN_DIR/frontend.pid")"
echo "    Logs: $DQ_RUN_DIR/backend.log $DQ_RUN_DIR/frontend.log"
