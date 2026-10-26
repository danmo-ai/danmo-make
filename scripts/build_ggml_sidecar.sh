#!/usr/bin/env bash
# Build the ggml sidecar. The desktop launcher still expects `danqing-api`
# under out/sidecar/danqing-api/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURES=()
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"
case "$OS_NAME" in
  MINGW*|MSYS*|CYGWIN*)
    FEATURES+=(vulkan)
    if [[ -n "${CUDA_PATH:-}" ]]; then
      FEATURES+=(cuda)
    fi
    ;;
  Darwin)
    FEATURES+=(metal)
    ;;
  Linux)
    if command -v nvcc >/dev/null 2>&1; then
      FEATURES+=(cuda)
    fi
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists vulkan 2>/dev/null; then
      FEATURES+=(vulkan)
    fi
    ;;
esac

ARGS=()
if [[ ${#FEATURES[@]} -gt 0 ]]; then
  IFS=,
  ARGS+=(--features "${FEATURES[*]}")
  unset IFS
fi

cargo build --release --manifest-path "$ROOT/native/Cargo.toml" -p danmo-server "${ARGS[@]}"
mkdir -p "$ROOT/out/sidecar/danqing-api"
SRC="$ROOT/native/target/release/danqing-api"
if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then
  cp "${SRC}.exe" "$ROOT/out/sidecar/danqing-api/danqing-api.exe"
else
  cp "$SRC" "$ROOT/out/sidecar/danqing-api/danqing-api"
  chmod +x "$ROOT/out/sidecar/danqing-api/danqing-api"
fi
echo "ggml sidecar: $ROOT/out/sidecar/danqing-api/"
