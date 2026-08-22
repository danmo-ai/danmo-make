#!/usr/bin/env bash
# macOS Apple Silicon (aarch64) Tauri release build. Cargo artifacts -> out/desktop/cargo/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/desktop"

if [[ "$(uname -s)" != Darwin ]]; then
  echo "Danmo Make macOS desktop build must run on Darwin." >&2
  exit 1
fi
if [[ "$(uname -m)" != arm64 ]]; then
  echo "Danmo Make macOS desktop requires Apple Silicon (arm64). MLX is not supported on Intel Mac." >&2
  exit 1
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/out/desktop/cargo}"
mkdir -p "$CARGO_TARGET_DIR"

PYTHON="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python3" ]]; then
  PYTHON="$ROOT/.venv/bin/python3"
fi

if [[ -n "${DANQING_DESKTOP_VERSION:-}" ]]; then
  "$PYTHON" "$ROOT/scripts/set_desktop_version.py" "$DANQING_DESKTOP_VERSION"
elif git -C "$ROOT" describe --exact-match --tags HEAD >/dev/null 2>&1; then
  echo "==> Desktop version from git tag"
  "$PYTHON" "$ROOT/scripts/set_desktop_version.py"
fi

"$PYTHON" "$ROOT/scripts/prepare_tauri_resources.py"
# tauri.conf.json requires bundle.resources["runtime"]; MLX has no thin tree.
bash "$ROOT/scripts/ensure_tauri_resource_stub.sh"

if ! rustup target list --installed | grep -q '^aarch64-apple-darwin$'; then
  echo "==> rustup target add aarch64-apple-darwin"
  rustup target add aarch64-apple-darwin
fi

cleanup_dmg_artifacts() {
  local bundle_root="$CARGO_TARGET_DIR/aarch64-apple-darwin/release/bundle"
  [[ -d "$bundle_root" ]] || return 0
  find "$bundle_root" -maxdepth 3 -name 'rw.*.dmg' -delete 2>/dev/null || true
  while IFS= read -r dmg; do
    [[ -n "$dmg" ]] && hdiutil detach "$dmg" -force 2>/dev/null || true
  done < <(
    hdiutil info 2>/dev/null | awk -v root="$bundle_root" '
      /^image-path/ && index($0, root) {
        sub(/^image-path[[:space:]]*:[[:space:]]*/, "")
        print
      }
    '
  )
}

echo "==> Tauri build (aarch64-apple-darwin)"
echo "    CARGO_TARGET_DIR=$CARGO_TARGET_DIR"
cleanup_dmg_artifacts
npm install
export CI=true
npm exec tauri build -- --target aarch64-apple-darwin
cleanup_dmg_artifacts

"$PYTHON" "$ROOT/scripts/stage_desktop_bundle.py"

if APP=$(find "$ROOT/out/desktop/bundle" -name "*.app" -print -quit); then
  echo "==> codesign .app (ad-hoc)"
  SC_DIR="$APP/Contents/Resources/danqing-api"
  if [[ -d "$SC_DIR" ]]; then
    while IFS= read -r -d '' bin; do
      codesign -s - --force "$bin" 2>/dev/null || true
    done < <(find "$SC_DIR" -type f \( -perm -111 -o -name '*.dylib' -o -name '*.so' \) -print0 2>/dev/null)
  else
    echo "Warning: sidecar not at $SC_DIR (check prepare_tauri_resources.py)" >&2
  fi
  if [[ -d "$APP/Contents/MacOS" ]]; then
    for bin in "$APP/Contents/MacOS"/*; do
      [[ -f "$bin" ]] && codesign -s - --force "$bin" || true
    done
  fi
  codesign -s - --force --deep "$APP"
  xattr -cr "$APP" 2>/dev/null && echo "==> Removed quarantine attribute" || true
  codesign --verify --deep --strict "$APP" 2>/dev/null || \
    echo "Warning: codesign verify reported issues (app may still run after clearing quarantine)" >&2
fi

# Create DMG with installation guide and one-click fix script
if APP=$(find "$ROOT/out/desktop/bundle" -name "*.app" -print -quit); then
  BUNDLE_DIR="$ROOT/out/desktop/bundle"
  DMG_STAGING="$BUNDLE_DIR/_dmg_staging"
  rm -rf "$DMG_STAGING"
  mkdir -p "$DMG_STAGING"

  cp -R "$APP" "$DMG_STAGING/"
  APP_NAME=$(basename "$APP")

  # README
  cat > "$DMG_STAGING/阅读说明.txt" << 'README_EOF'
📦 Danmo Make 安装说明 (macOS)

由于本应用未使用 Apple 开发者签名，macOS 可能会阻止打开。
请按以下步骤操作：

方法一（推荐 — 一键修复）：
  1. 将 Danmo Make.app 拖入「应用程序」文件夹
  2. 双击本 DMG 中的「修复并打开.command」脚本
  3. 如脚本无法执行，请先打开终端运行：
     xattr -cr "/Volumes/Danmo Make"
     然后重新双击脚本

方法二（右键打开）：
  1. 将 Danmo Make.app 拖入「应用程序」文件夹
  2. 在 Finder 中右键点击 app → 选择「打开」
  3. 弹窗中点击「打开」确认

方法三（终端命令）：
  打开终端，执行：
  xattr -cr "/Applications/Danmo Make.app"
  open "/Applications/Danmo Make.app"
README_EOF

  # One-click fix script
  cat > "$DMG_STAGING/修复并打开.command" << 'FIX_EOF'
#!/bin/bash
# 一键移除 macOS 隔离属性并打开 Danmo Make
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PATH="$SCRIPT_DIR/Danmo Make.app"

# Step 1: clear quarantine on the DMG volume itself (needed when downloaded from internet)
DMG_VOL="/Volumes/Danmo Make"
if [ -d "$DMG_VOL" ]; then
  echo "==> 正在移除 DMG 卷隔离属性..."
  xattr -cr "$DMG_VOL" 2>/dev/null
fi

# Step 2: locate app
if [ ! -d "$APP_PATH" ]; then
  APP_PATH="/Applications/Danmo Make.app"
fi

if [ ! -d "$APP_PATH" ]; then
  echo "❌ 未找到 Danmo Make.app"
  echo "请先将 app 拖入应用程序文件夹"
  read -p "按回车退出..."
  exit 1
fi

# Step 3: clear quarantine on app and open
echo "==> 正在移除应用隔离属性..."
xattr -cr "$APP_PATH" 2>/dev/null
echo "==> 正在打开 Danmo Make..."
open "$APP_PATH"
FIX_EOF
  chmod +x "$DMG_STAGING/修复并打开.command"

  ln -s /Applications "$DMG_STAGING/Applications"

  # Create DMG
  DMG_DIR="$BUNDLE_DIR/dmg"
  TAURI_DMG_FALLBACK=""
  if [[ -d "$DMG_DIR" ]]; then
    TAURI_DMG_FALLBACK=$(find "$DMG_DIR" -maxdepth 1 -name '*.dmg' -print -quit)
  fi
  rm -rf "$DMG_DIR"
  mkdir -p "$DMG_DIR"
  APP_VERSION=$(plutil -extract CFBundleShortVersionString raw "$DMG_STAGING/$APP_NAME/Contents/Info.plist" 2>/dev/null || echo "0.0.0")
  ARCH=$(uname -m)
  DMG_NAME="Danmo Make_${APP_VERSION}_${ARCH}.dmg"
  DMG_PATH="$DMG_DIR/$DMG_NAME"
  echo "==> Creating DMG: $DMG_NAME"

  # Detach Tauri/custom DMG mounts (CI often leaves /Volumes/Danmo Make busy).
  if [[ -d "/Volumes/Danmo Make" ]]; then
    hdiutil detach "/Volumes/Danmo Make" -force 2>/dev/null || true
  fi
  while IFS= read -r old_dmg; do
    [[ -n "$old_dmg" ]] && hdiutil detach "$old_dmg" -force 2>/dev/null || true
  done < <(
    hdiutil info 2>/dev/null | awk '/Danmo Make/ { found=1 } found && /^image-path/ { sub(/^image-path[[:space:]]*:[[:space:]]*/, ""); print; found=0 }'
  )
  cleanup_dmg_artifacts

  if hdiutil create -volname "Danmo Make" -srcfolder "$DMG_STAGING" -ov -format UDZO "$DMG_PATH"; then
    echo "==> DMG created: $DMG_PATH"
  elif [[ -n "$TAURI_DMG_FALLBACK" && -f "$TAURI_DMG_FALLBACK" ]]; then
    cp "$TAURI_DMG_FALLBACK" "$DMG_DIR/$(basename "$TAURI_DMG_FALLBACK")"
    echo "WARNING: custom DMG creation failed; restored Tauri DMG: $(basename "$TAURI_DMG_FALLBACK")" >&2
  else
    echo "WARNING: DMG creation failed" >&2
  fi

  # Remove quarantine on the DMG file itself so macOS doesn't flag it as damaged
  xattr -cr "$DMG_PATH" 2>/dev/null && echo "==> Removed quarantine on DMG" || true

  rm -rf "$DMG_STAGING"
fi
