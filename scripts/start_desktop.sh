#!/usr/bin/env bash
# Dev: ggml danqing-api + Tauri (Vite HMR via beforeDevCommand).
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
echo "    Backend : http://127.0.0.1:${BACKEND_PORT}  (danqing-api, ggml)"
echo "    Desktop : Tauri webview (Vite HMR on :${FRONTEND_PORT})"
echo "    SKIP_BACKEND=1 to use an already-running API"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust: https://rustup.rs/" >&2
  exit 1
fi

cd "$DQ_ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi

if [[ "${SKIP_BACKEND:-0}" == "1" ]]; then
  echo "==> SKIP_BACKEND=1: using external backend on :${BACKEND_PORT}"
  echo ""
else
  mkdir -p "$DQ_ROOT/db"
  export DQ_DEV_ENV=$'DANQING_HTTP_PORT='"${BACKEND_PORT}"$'\nDANQING_HTTP_HOST=127.0.0.1'$'\nDANQING_DB='"$DQ_ROOT/db/studio.db"$'\nDANQING_REGISTRY='"$DQ_ROOT/default_config/models_registry.json"
  dq_dev_start backend "$DQ_ROOT" \
    cargo run --manifest-path "$DQ_ROOT/native/Cargo.toml" -p danmo-server
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
