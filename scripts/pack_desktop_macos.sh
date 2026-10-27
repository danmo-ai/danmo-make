#!/usr/bin/env bash
# macOS Tauri desktop (ggml Metal sidecar).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "pack-macos-desktop must run on macOS" >&2
  exit 1
fi
if [[ "$(uname -m)" != arm64 ]]; then
  echo "Danmo Make macOS desktop requires Apple Silicon (arm64)." >&2
  exit 1
fi

dq_ensure_out_layout
cd "$DQ_ROOT"
export RELEASE_VERSION="${RELEASE_VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"

echo "==> Frontend -> $DQ_FRONTEND_DIST"
(cd "$DQ_ROOT/frontend" && npm install && npm run build)

echo "==> ggml sidecar (Metal)"
bash "$SCRIPT_DIR/build_ggml_sidecar.sh"

echo "==> Tauri shell (macOS)"
bash "$SCRIPT_DIR/tauri_build_macos.sh"

echo "==> Desktop bundle -> $DQ_DESKTOP_BUNDLE"
