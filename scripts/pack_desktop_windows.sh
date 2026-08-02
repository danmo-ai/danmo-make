#!/usr/bin/env bash
# Windows Tauri desktop (CUDA sidecar) — NSIS installer
# Naming aligned with danmo-work / danmo-inbox.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN* | Windows*) ;;
  *)
    echo "pack-windows-desktop must run on Windows" >&2
    exit 1
    ;;
esac

dq_ensure_out_layout
cd "$DQ_ROOT"

if [[ -x "$DQ_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$DQ_ROOT/.venv/Scripts/python.exe"
elif [[ -x "$DQ_ROOT/.venv/bin/python3" ]]; then
  PYTHON="$DQ_ROOT/.venv/bin/python3"
else
  PYTHON="${PYTHON:-python}"
fi

export DANQING_PYINSTALLER_PROFILE="${DANQING_PYINSTALLER_PROFILE:-cuda}"
export RELEASE_VERSION="${RELEASE_VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "==> Ensure CUDA venv"
if [[ ! -d "$DQ_ROOT/.venv" ]]; then
  py -3.11 -m venv "$DQ_ROOT/.venv" 2>/dev/null || python -m venv "$DQ_ROOT/.venv"
fi
if [[ -x "$DQ_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$DQ_ROOT/.venv/Scripts/python.exe"
fi
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install -r "$DQ_ROOT/requirements-cuda.txt" pyinstaller

echo "==> Frontend -> $DQ_FRONTEND_DIST"
(cd "$DQ_ROOT/frontend" && npm install && npm run build)

echo "==> PyInstaller sidecar (CUDA)"
DANQING_PYINSTALLER_PROFILE=cuda "$PYTHON" "$SCRIPT_DIR/build_sidecar.py"

echo "==> Tauri shell (Windows NSIS)"
"$PYTHON" "$SCRIPT_DIR/tauri_build.py" --platform windows

echo "==> Desktop bundle -> $DQ_DESKTOP_BUNDLE"
find "$DQ_DESKTOP_BUNDLE" -type f -name '*.exe' 2>/dev/null | head -20 || true
