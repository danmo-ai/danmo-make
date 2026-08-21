#!/usr/bin/env bash
# Resolve platform-specific Python requirements file for Danmo Make.
# Sourced by start.sh / start_desktop.sh. Sets:
#   DQ_PLATFORM_REQS  — path to requirements-macos.txt or requirements-linux.txt
# Exits non-zero on unsupported platforms (Windows, Intel Mac, etc.).

dq_resolve_platform_requirements() {
  local root="${1:-${DQ_ROOT:-.}}"
  local os
  os="$(uname -s 2>/dev/null || echo unknown)"
  case "$os" in
    Darwin)
      local arch
      arch="$(uname -m 2>/dev/null || echo unknown)"
      if [[ "$arch" != "arm64" ]]; then
        echo "Danmo Make requires Apple Silicon (arm64). Intel Mac is not supported (MLX Metal)." >&2
        return 1
      fi
      DQ_PLATFORM_REQS="$root/requirements-macos.txt"
      ;;
    Linux)
      DQ_PLATFORM_REQS="$root/requirements-linux.txt"
      ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
      echo "Windows is temporarily unsupported. Use macOS (MLX) or Linux (mlx[cuda])." >&2
      return 1
      ;;
    *)
      echo "Unsupported platform: $os (need macOS arm64 or Linux x86_64)." >&2
      return 1
      ;;
  esac
  if [[ ! -f "$DQ_PLATFORM_REQS" ]]; then
    echo "Missing requirements file: $DQ_PLATFORM_REQS" >&2
    return 1
  fi
  export DQ_PLATFORM_REQS
  return 0
}
