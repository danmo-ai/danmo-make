#!/usr/bin/env bash
# Windows Tauri desktop is not wired. The ggml sidecar is:
#   make pack-windows-sidecar
set -euo pipefail
echo "Windows desktop pack is not wired. Build the sidecar with: make pack-windows-sidecar" >&2
exit 1
