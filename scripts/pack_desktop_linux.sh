#!/usr/bin/env bash
# Linux Tauri desktop (MLX sidecar / mlx[cuda]) — AppImage + .deb
# Naming aligned with danmo-work pack_desktop_linux.sh.
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

PYTHON="${PYTHON:-python3}"
if [[ -x "$DQ_ROOT/.venv/bin/python3" ]]; then
  PYTHON="$DQ_ROOT/.venv/bin/python3"
fi

export DANQING_PYINSTALLER_PROFILE="${DANQING_PYINSTALLER_PROFILE:-mlx}"
export RELEASE_VERSION="${RELEASE_VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"

echo "==> Ensure MLX (mlx[cuda]) venv"
if [[ ! -d "$DQ_ROOT/.venv" ]]; then
  python3.11 -m venv "$DQ_ROOT/.venv" 2>/dev/null || python3 -m venv "$DQ_ROOT/.venv"
fi
PYTHON="$DQ_ROOT/.venv/bin/python3"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$DQ_ROOT/requirements-linux.txt" pyinstaller

echo "==> Frontend -> $DQ_FRONTEND_DIST"
(cd "$DQ_ROOT/frontend" && npm install && npm run build)

echo "==> PyInstaller sidecar (MLX)"
DANQING_PYINSTALLER_PROFILE=mlx "$PYTHON" "$SCRIPT_DIR/build_sidecar.py"

echo "==> Tauri shell (Linux AppImage + deb)"
"$PYTHON" "$SCRIPT_DIR/tauri_build.py" --platform linux

echo "==> Desktop bundle -> $DQ_DESKTOP_BUNDLE"
find "$DQ_DESKTOP_BUNDLE" -type f \( -name '*.AppImage' -o -name '*.deb' \) 2>/dev/null | head -20 || true
