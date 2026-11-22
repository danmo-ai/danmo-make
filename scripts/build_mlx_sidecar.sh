#!/usr/bin/env bash
# Build the danqing-api sidecar for desktop / server packs.
# Desktop launcher expects the binary under out/sidecar/danqing-api/.
#
# Features:
#   macOS  → metal,mlx-metal
#   Linux  → mlx-cuda when nvcc is on PATH (toolkit only; no GPU needed to compile)
#   Windows → vulkan (+ cuda if CUDA_PATH is set)
#
# Override: DANQING_SIDECAR_FEATURES=mlx-cuda,metal,... (comma-separated).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURES=()
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"

if [[ -n "${DANQING_SIDECAR_FEATURES:-}" ]]; then
  IFS=',' read -r -a FEATURES <<< "${DANQING_SIDECAR_FEATURES}"
else
  case "$OS_NAME" in
    MINGW*|MSYS*|CYGWIN*)
      FEATURES+=(vulkan)
      if [[ -n "${CUDA_PATH:-}" ]] || command -v nvcc >/dev/null 2>&1; then
        FEATURES+=(mlx-cuda)
      fi
      ;;
    Darwin)
      FEATURES+=(metal mlx-metal)
      ;;
    Linux)
      if command -v nvcc >/dev/null 2>&1; then
        FEATURES+=(mlx-cuda)
      else
        echo "WARNING: nvcc not found — building danqing-api without mlx-cuda (no MLX GPU backend)" >&2
      fi
      ;;
  esac
fi

ARGS=()
if [[ ${#FEATURES[@]} -gt 0 ]]; then
  IFS=,
  ARGS+=(--features "${FEATURES[*]}")
  unset IFS
  echo "sidecar features: ${FEATURES[*]}"
else
  echo "sidecar features: (default / none)"
fi

if [[ "$OS_NAME" == Linux && "${CI:-}" == "true" ]]; then
  if [[ " ${FEATURES[*]} " != *mlx-cuda* ]]; then
    echo "ERROR: CI Linux release must build danqing-api with mlx-cuda (set DANQING_SIDECAR_FEATURES=mlx-cuda)" >&2
    exit 1
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: mlx-cuda sidecar build requires nvcc on PATH" >&2
    exit 1
  fi
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
echo "sidecar: $ROOT/out/sidecar/danqing-api/ (${FEATURES[*]:-none})"
