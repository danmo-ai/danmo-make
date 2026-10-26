#!/usr/bin/env bash
# Linux Tauri desktop (ggml sidecar) — AppImage + .deb
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

if [[ "$(uname -s)" != Linux ]]; then
  echo "pack-linux-desktop must run on Linux" >&2
  exit 1
fi
if [[ "$(uname -m)" != x86_64 ]]; then
  echo "pack-linux-desktop currently supports x86_64 only (got $(uname -m))" >&2
  exit 1
fi

dq_ensure_out_layout
cd "$DQ_ROOT"
export RELEASE_VERSION="${RELEASE_VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$DQ_DESKTOP_CARGO}"
mkdir -p "$CARGO_TARGET_DIR"

echo "==> Frontend -> $DQ_FRONTEND_DIST"
(cd "$DQ_ROOT/frontend" && npm install && npm run build)

echo "==> ggml sidecar"
bash "$SCRIPT_DIR/build_ggml_sidecar.sh"

if [[ -n "${DANQING_DESKTOP_VERSION:-}" ]]; then
  bash "$SCRIPT_DIR/set_desktop_version.sh" "$DANQING_DESKTOP_VERSION"
elif git -C "$DQ_ROOT" describe --exact-match --tags HEAD >/dev/null 2>&1; then
  bash "$SCRIPT_DIR/set_desktop_version.sh"
fi
bash "$SCRIPT_DIR/prepare_tauri_resources.sh"
rustup target add x86_64-unknown-linux-gnu
(cd "$DQ_ROOT/desktop" && npm install && npm exec tauri build -- --target x86_64-unknown-linux-gnu -b appimage -b deb)
bash "$SCRIPT_DIR/stage_desktop_bundle.sh"

echo "==> Desktop bundle -> $DQ_DESKTOP_BUNDLE"
find "$DQ_DESKTOP_BUNDLE" -type f \( -name '*.AppImage' -o -name '*.deb' \) 2>/dev/null | head -20 || true
