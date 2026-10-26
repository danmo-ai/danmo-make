#!/usr/bin/env bash
# Convenience entry — delegates to platform pack_desktop_*.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
chmod +x scripts/*.sh

OS="$(uname -s)"
case "$OS" in
  Darwin)
    exec ./scripts/pack_desktop_macos.sh
    ;;
  Linux)
    exec ./scripts/pack_desktop_linux.sh
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows*)
    echo "Use make pack-windows-sidecar for the ggml binary. Tauri Windows packs are not wired yet." >&2
    exit 1
    ;;
  *)
    echo "Unsupported OS for desktop build: $OS" >&2
    exit 1
    ;;
esac
