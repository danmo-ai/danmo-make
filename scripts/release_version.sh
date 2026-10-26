#!/usr/bin/env bash
# Bump Danmo Make desktop version, optionally commit, tag, push, and open the GitHub release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CARGO_TOML="$ROOT/desktop/src-tauri/Cargo.toml"
CARGO_LOCK="$ROOT/desktop/src-tauri/Cargo.lock"
TAURI_CONF="$ROOT/desktop/src-tauri/tauri.conf.json"

dry_run=0
push=0
no_release=0
requested=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) dry_run=1 ;;
    --push) push=1 ;;
    --no-release) no_release=1 ;;
    -*) echo "unknown flag $arg" >&2; exit 1 ;;
    *) requested="$arg" ;;
  esac
done
if [[ "$dry_run" -eq 1 && "$push" -eq 1 ]]; then
  echo "use either --dry-run or --push, not both" >&2
  exit 1
fi

mapfile -t tags < <(git -C "$ROOT" tag -l 'v*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -t. -k1,1n -k2,2n -k3,3n)
prev=""
if [[ ${#tags[@]} -gt 0 ]]; then
  prev="${tags[-1]}"
fi

if [[ -n "$requested" ]]; then
  version="${requested#v}"
  version="${version#V}"
  if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "invalid version '$requested' (want X.Y.Z)" >&2
    exit 1
  fi
  if git -C "$ROOT" rev-parse -q --verify "refs/tags/v${version}" >/dev/null; then
    echo "tag v${version} already exists" >&2
    exit 1
  fi
elif [[ -z "$prev" ]]; then
  version="0.1.0"
else
  IFS=. read -r major minor patch <<<"${prev#v}"
  version="${major}.${minor}.$((patch + 1))"
fi

range="HEAD"
if [[ -n "$prev" ]]; then
  range="${prev}..HEAD"
fi
notes=""
while IFS= read -r subject; do
  [[ -z "$subject" ]] && continue
  if [[ "$subject" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    continue
  fi
  notes+="- ${subject}"$'\n'
done < <(git -C "$ROOT" log "$range" --pretty=format:%s --no-merges)
if [[ -z "$notes" ]]; then
  body="Release v${version}."
elif [[ -n "$prev" ]]; then
  body="Changes since ${prev}:"$'\n\n'"${notes%$'\n'}"
else
  body="Changes:"$'\n\n'"${notes%$'\n'}"
fi

echo "version: $version"
echo "tag:     v$version"
echo "prev:    ${prev:-(none)}"
echo
echo "--- release notes ---"
echo "$body"
echo "---"
if [[ "$dry_run" -eq 1 ]]; then
  exit 0
fi

perl -0pi -e 's/^(version\s*=\s*")[^"]+(")/${1}'"$version"'$2/m' "$CARGO_TOML"
perl -0pi -e 's/(name = "danqing-desktop"\nversion = ")[^"]+/${1}'"$version"'/m' "$CARGO_LOCK"
perl -0pi -e 's/^(\s*"version"\s*:\s*")[^"]+(")/${1}'"$version"'$2/m' "$TAURI_CONF"
echo "bumped desktop version files"

if [[ "$push" -ne 1 ]]; then
  echo
  echo "Files updated. Commit/tag/push yourself, or re-run with --push."
  exit 0
fi

git -C "$ROOT" add -A
if [[ -z "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "nothing to commit" >&2
  exit 1
fi
git -C "$ROOT" commit -m "v${version}" -m "$body"
git -C "$ROOT" tag -a "v${version}" -m "$body"
git -C "$ROOT" push origin HEAD
git -C "$ROOT" push origin "v${version}"
if [[ "$no_release" -ne 1 ]]; then
  if gh release view "v${version}" >/dev/null 2>&1; then
    gh release edit "v${version}" --notes "$body"
  else
    gh release create "v${version}" --title "v${version}" --notes "$body" --verify-tag
  fi
fi
echo
echo "Done. CI release build should start for v${version}."
