#!/usr/bin/env bash
# Windows Tauri desktop (thin CUDA runtime) — portable zip
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

export RELEASE_VERSION="${RELEASE_VERSION:-$(git describe --tags --always --dirty 2>/dev/null || echo dev)}"

echo "==> Ensure build tooling venv (no torch required for thin pack)"
if [[ ! -d "$DQ_ROOT/.venv" ]]; then
  py -3.11 -m venv "$DQ_ROOT/.venv" 2>/dev/null || python -m venv "$DQ_ROOT/.venv"
fi
if [[ -x "$DQ_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON="$DQ_ROOT/.venv/Scripts/python.exe"
fi
"$PYTHON" -m pip install --upgrade pip

echo "==> Frontend -> $DQ_FRONTEND_DIST"
(cd "$DQ_ROOT/frontend" && npm install && npm run build)

echo "==> Stage thin CUDA runtime (portable CPython + app)"
"$PYTHON" "$SCRIPT_DIR/stage_cuda_runtime.py" --platform windows-x86_64 --prepare-tauri

echo "==> Ensure danqing-api stub (Tauri resources still list it)"
bash "$SCRIPT_DIR/ensure_tauri_resource_stub.sh"

echo "==> Tauri shell + thin portable zip"
"$PYTHON" "$SCRIPT_DIR/tauri_build.py" --platform windows --thin-runtime

echo "==> Desktop bundle -> $DQ_DESKTOP_BUNDLE"
find "$DQ_DESKTOP_BUNDLE" -type f -name '*-portable.zip' 2>/dev/null | head -20 || true
