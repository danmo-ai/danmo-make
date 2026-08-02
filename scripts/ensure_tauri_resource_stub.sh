#!/usr/bin/env bash
# Tauri build requires bundle.resources paths to exist (even for `tauri dev`).
# Production packs overwrite this with prepare_tauri_resources.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STUB_DIR="$ROOT/desktop/src-tauri/danqing-api"

if [[ -d "$STUB_DIR" ]] && [[ -f "$STUB_DIR/danqing-api" || -f "$STUB_DIR/danqing-api.exe" ]]; then
  # Real sidecar already staged — leave it alone.
  :
else
  mkdir -p "$STUB_DIR"
  cat > "$STUB_DIR/.dev-stub" <<'EOF'
Dev placeholder for Tauri bundle.resources.

Release builds replace this directory via:
  python scripts/prepare_tauri_resources.py
EOF
  echo "==> Ensured Tauri resource stub: desktop/src-tauri/danqing-api/"
fi

# Always ensure ``runtime`` exists: tauri.conf.json lists it, but early-exit above
# used to skip this when a real sidecar was already staged (broke macOS MLX CI).
RUNTIME_DIR="$ROOT/desktop/src-tauri/runtime"
if [[ ! -d "$RUNTIME_DIR/app" ]]; then
  mkdir -p "$RUNTIME_DIR"
  cat > "$RUNTIME_DIR/.dev-stub" <<'EOF'
Dev placeholder for CUDA thin runtime (desktop/src-tauri/runtime).

Windows/Linux release packs replace this via:
  python scripts/stage_cuda_runtime.py --prepare-tauri
EOF
  echo "==> Ensured Tauri runtime stub: desktop/src-tauri/runtime/"
fi
