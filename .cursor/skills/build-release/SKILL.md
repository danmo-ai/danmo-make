---
name: build-release
description: >-
  Bump Danmo Make to a new semver, tag vX.Y.Z, push, and trigger the GitHub
  release build. Use when the user asks to build/release a version, bump the
  version, cut a tag, 发版, or says things like "build 0.1.9" / "push; build".
---

# Build Release

Cut a Danmo Make release: bump desktop version files → commit → annotated tag → push → GitHub release notes → CI artifacts.

Aligned with danmo-work (DanQing-Teams) `.cursor/skills/build-release`.

## Script (required)

Always drive version resolution and notes through the repo script:

```bash
# Preview only
bash scripts/release_version.sh --dry-run
bash scripts/release_version.sh 0.1.9 --dry-run

# Bump files only (no git)
bash scripts/release_version.sh
bash scripts/release_version.sh 0.2.0

# Full release: commit + tag + push + gh release notes
bash scripts/release_version.sh --push
bash scripts/release_version.sh 0.1.9 --push
```

### Version rules

- **No version given** → latest `vX.Y.Z` tag, patch +1 (e.g. `v0.1.8` → `0.1.9`)
- **Version given** → use as specified (`0.1.9` or `v0.1.9`)

### Release notes

Script builds notes from `git log` subjects since the previous `vX.Y.Z` tag (skips merge commits and pure `vX.Y.Z` bump commits). That text is the GitHub release body.

## Agent workflow

When the user asks to build/release:

1. **Confirm intent** if the tree has unrelated dirty work you did not just make — otherwise proceed.
2. Run with `--dry-run` first; show the user the resolved version + notes briefly.
3. If they already named a version / said to push/build, run immediately with `--push` (skip dry-run confirmation when the request is explicit, e.g. `build 0.1.9` or `push; build`).
4. After `--push`, report:
   - tag `vX.Y.Z`
   - Actions run URL: `gh run list --workflow=release.yml -L 1`
5. Do **not** invent a second versioning path — do not hand-edit `Cargo.toml` / `tauri.conf.json` for releases (use this script).

## What the script bumps

- `desktop/src-tauri/Cargo.toml` → `version`
- `desktop/src-tauri/Cargo.lock` → `danqing-desktop` package version only
- `desktop/src-tauri/tauri.conf.json` → `version`

CI (`release.yml` on `v*` tags) publishes a macOS `.dmg` and a Linux ggml server `.tar.gz`.

## Notes

- `--push` creates/edits the GitHub Release with notes so CI can attach artifacts to an existing release.
- Use `--no-release` with `--push` only if you want tag-only (no `gh release`).
- Requires `git` and (for `--push` without `--no-release`) `gh` authenticated to `danmo-ai/danmo-make`.
