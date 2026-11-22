#!/usr/bin/env bash
# Sync Tauri desktop version in tauri.conf.json and desktop Cargo.toml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAURI_CONF="$ROOT/desktop/src-tauri/tauri.conf.json"
CARGO_TOML="$ROOT/desktop/src-tauri/Cargo.toml"

version="${1:-${DANQING_DESKTOP_VERSION:-}}"
if [[ -z "$version" ]]; then
  if tag="$(git -C "$ROOT" describe --exact-match --tags HEAD 2>/dev/null)"; then
    version="$tag"
  else
    echo "No desktop version given. Pass VERSION, set DANQING_DESKTOP_VERSION, or run from an exact git tag." >&2
    exit 1
  fi
fi
version="${version#v}"
version="${version#V}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]]; then
  echo "Invalid semver for desktop bundle: $version" >&2
  exit 1
fi

perl -0pi -e 's/^(\s*"version"\s*:\s*")[^"]+(")/${1}'"$version"'$2/m' "$TAURI_CONF"
perl -0pi -e 's/^(version\s*=\s*")[^"]+(")/${1}'"$version"'$2/m' "$CARGO_TOML"
echo "Desktop version -> $version"
echo "  desktop/src-tauri/tauri.conf.json"
echo "  desktop/src-tauri/Cargo.toml"
