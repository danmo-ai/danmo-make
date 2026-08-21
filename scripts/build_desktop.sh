#!/usr/bin/env bash
# Convenience entry — delegates to platform pack_desktop_*.sh (like danmo-work).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
chmod +x scripts/*.sh

OS="$(uname -s)"
case "$OS" in
  Darwin)
    export DANQING_PYINSTALLER_PROFILE="${DANQING_PYINSTALLER_PROFILE:-mlx}"
    exec ./scripts/pack_desktop_macos.sh
    ;;
  Linux)
    export DANQING_PYINSTALLER_PROFILE="${DANQING_PYINSTALLER_PROFILE:-mlx}"
    exec ./scripts/pack_desktop_linux.sh
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows*)
    echo "Windows temporarily unsupported" >&2
    exit 1
    ;;
  *)
    echo "Unsupported OS for desktop build: $OS" >&2
    exit 1
    ;;
esac
