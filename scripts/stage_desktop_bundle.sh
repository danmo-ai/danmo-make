#!/usr/bin/env bash
# Copy Tauri release/bundle artifacts to out/desktop/bundle/.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

BASE="$DQ_DESKTOP_CARGO"
if [[ ! -d "$BASE" ]]; then
  echo "Missing Cargo target dir: $BASE" >&2
  exit 1
fi

src=""
for candidate in \
  "$BASE/x86_64-unknown-linux-gnu/release/bundle" \
  "$BASE/aarch64-apple-darwin/release/bundle" \
  "$BASE/release/bundle"
do
  if [[ -d "$candidate" ]]; then
    src="$candidate"
    break
  fi
done
if [[ -z "$src" ]]; then
  src="$(find "$BASE" -type d -path '*/release/bundle' -print -quit || true)"
fi
if [[ -z "$src" || ! -d "$src" ]]; then
  echo "No Tauri bundle under $BASE" >&2
  exit 1
fi

dst="$DQ_DESKTOP_BUNDLE"
staging="${dst}.staging"
rm -rf "$staging"
mkdir -p "$(dirname "$dst")"
cp -a "$src" "$staging"
rm -rf "$dst"
mv "$staging" "$dst"
echo "Staged desktop bundle -> out/desktop/bundle/"
ls -1 "$dst"
