#!/usr/bin/env bash
# Dev: Rust danqing-api + Vite HMR. There is no Python runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"
# shellcheck source=dev_process.sh
source "$SCRIPT_DIR/dev_process.sh"
# shellcheck source=workspace_env.sh
source "$SCRIPT_DIR/workspace_env.sh"

BACKEND_PORT="${DQ_BACKEND_PORT}"
FRONTEND_PORT="${DQ_FRONTEND_PORT}"

dq_ensure_out_layout
dq_resolve_workspace_env
"$SCRIPT_DIR/stop.sh" 2>/dev/null || true

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

echo "==> Starting Danmo Make (dev) [${DQ_PROJECT}]"
echo "    Backend : http://127.0.0.1:${BACKEND_PORT}  (danqing-api, MLX)"
echo "    Frontend: http://localhost:${FRONTEND_PORT}/  (Vite HMR)"
echo "    Workspace: $DANQING_WORKSPACE"
echo "    DB: $DANQING_DB"
echo "    Registry: $DANQING_REGISTRY"
echo "    Stop: make stop"

cd "$DQ_ROOT/frontend"
if [[ ! -d node_modules ]]; then
  npm install
fi

CARGO_FEATURES=()
case "$(uname -s)" in
  Darwin) CARGO_FEATURES=(--features metal,mlx-metal) ;;
esac

export DQ_DEV_ENV=$'DANQING_HTTP_PORT='"${BACKEND_PORT}"$'\nDANQING_HTTP_HOST=127.0.0.1'$'\nDANQING_WORKSPACE='"${DANQING_WORKSPACE}"$'\nDANQING_DB='"${DANQING_DB}"$'\nDANQING_OUTPUT='"${DANQING_OUTPUT}"$'\nDANQING_REGISTRY='"${DANQING_REGISTRY}"$'\nDANQING_GGML_BACKEND=metal'$'\nGGML_METAL_FUSION_DISABLE=1'
dq_dev_start backend "$DQ_ROOT" \
  cargo run --manifest-path "$DQ_ROOT/native/Cargo.toml" -p danmo-server "${CARGO_FEATURES[@]}"
unset DQ_DEV_ENV

export DQ_DEV_ENV=$'DQ_BACKEND_PORT='"${BACKEND_PORT}"$'\nDQ_FRONTEND_PORT='"${FRONTEND_PORT}"
dq_dev_start frontend "$DQ_ROOT/frontend" npm run dev
unset DQ_DEV_ENV

echo "==> PIDs: backend=$(cat "$DQ_RUN_DIR/backend.pid") frontend=$(cat "$DQ_RUN_DIR/frontend.pid")"
echo "    Logs: $DQ_RUN_DIR/backend.log $DQ_RUN_DIR/frontend.log"
