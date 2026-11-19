#!/usr/bin/env bash
# Dev: MLX danqing-api + Tauri (Vite HMR via beforeDevCommand).
# Aligns with danmo-work / danmo-inbox `make dev-desktop`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"
# shellcheck source=dev_process.sh
source "$SCRIPT_DIR/dev_process.sh"
# shellcheck source=workspace_env.sh
source "$SCRIPT_DIR/workspace_env.sh"

APP_NAME="${DQ_APP_NAME:-danmo-make}"
BACKEND_PORT="${DQ_BACKEND_PORT}"
FRONTEND_PORT="${DQ_FRONTEND_PORT}"

dq_ensure_out_layout
dq_resolve_workspace_env
"$SCRIPT_DIR/stop.sh" 2>/dev/null || true

echo "==> Starting $APP_NAME (dev-desktop) [${DQ_PROJECT}]"
echo "    Backend : http://127.0.0.1:${BACKEND_PORT}  (danqing-api, MLX)"
echo "    Desktop : Tauri webview (Vite HMR on :${FRONTEND_PORT})"
echo "    Workspace: $DANQING_WORKSPACE"
echo "    DB: $DANQING_DB"
echo "    SKIP_BACKEND=1 to use an already-running API"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found. Install Rust: https://rustup.rs/" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found (required to build danmo-mlx-sys). Install: brew install cmake" >&2
  exit 1
fi
if [[ ! -f "$DQ_ROOT/native/third_party/mlx/CMakeLists.txt" ]]; then
  echo "MLX sources missing. Run: git submodule update --init --recursive native/third_party/mlx native/third_party/mlx-c" >&2
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
  CARGO_FEATURES=()
  case "$(uname -s)" in
    Darwin) CARGO_FEATURES=(--features metal,mlx-metal) ;;
  esac
  export DQ_DEV_ENV=$'DANQING_HTTP_PORT='"${BACKEND_PORT}"$'\nDANQING_HTTP_HOST=127.0.0.1'$'\nDANQING_WORKSPACE='"${DANQING_WORKSPACE}"$'\nDANQING_DB='"${DANQING_DB}"$'\nDANQING_OUTPUT='"${DANQING_OUTPUT}"$'\nDANQING_REGISTRY='"${DANQING_REGISTRY}"$'\nDANQING_GGML_BACKEND=metal'$'\nGGML_METAL_FUSION_DISABLE=1'
  dq_dev_start backend "$DQ_ROOT" \
    cargo run --manifest-path "$DQ_ROOT/native/Cargo.toml" -p danmo-server "${CARGO_FEATURES[@]}"
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
