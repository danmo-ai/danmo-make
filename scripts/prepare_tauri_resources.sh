#!/usr/bin/env bash
# Stage the ggml sidecar into desktop/src-tauri/danqing-api for Tauri bundling.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

SRC="${1:-$DQ_SIDECAR_DIR}"
DST="$DQ_ROOT/desktop/src-tauri/danqing-api"

if [[ ! -f "$SRC/danqing-api" && ! -f "$SRC/danqing-api.exe" ]]; then
  echo "Missing sidecar executable under $SRC (expected danqing-api)." >&2
  echo "Run: make pack-ggml-sidecar" >&2
  exit 1
fi

echo "==> Stage sidecar for Tauri: desktop/src-tauri/danqing-api"
rm -rf "$DST"
mkdir -p "$(dirname "$DST")"
cp -a "$SRC" "$DST"
bash "$SCRIPT_DIR/ensure_tauri_resource_stub.sh"
