#!/usr/bin/env bash
# macOS Apple Silicon (aarch64) Tauri release build. Cargo artifacts -> out/desktop/cargo/.
#
# 签名/公证（对齐 danmo-novel）：
#   - APPLE_SIGNING_IDENTITY 已设且证书在钥匙串 → Developer ID；CI 先跑 ci_import_apple_cert.sh
#   - 未设 → ad-hoc，DMG 内带右键打开说明
#   - APPLE_ID / APPLE_PASSWORD / APPLE_TEAM_ID 齐全时，对自制 DMG 做 notarytool + staple
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

SIGN_IDENTITY="${APPLE_SIGNING_IDENTITY:-}"
SIGNED=0
NOTARIZED=0
if [[ -n "$SIGN_IDENTITY" ]]; then
  SIGNED=1
  echo "==> Signing identity: $SIGN_IDENTITY"
  if [[ -n "${APPLE_ID:-}" && -n "${APPLE_PASSWORD:-}" && -n "${APPLE_TEAM_ID:-}" ]]; then
    NOTARIZED=1
  else
    echo "==> No APPLE_ID/APPLE_PASSWORD/APPLE_TEAM_ID — signed but NOT notarized"
  fi
else
  echo "==> No APPLE_SIGNING_IDENTITY — ad-hoc signature only (not distributable)"
fi

# sidecar 须在 Tauri 拷进 .app 之前签好；打包后再改 bundle 内二进制会破坏 Developer ID。
sign_mach_o_tree() {
  local dir="$1"
  [[ "$SIGNED" == 1 ]] || return 0
  [[ -d "$dir" ]] || return 0
  echo "==> Signing Mach-O under $(basename "$dir")"
  while IFS= read -r -d '' f; do
    codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$f"
  done < <(find "$dir" -type f \( -perm -111 -o -name '*.dylib' -o -name '*.so' \) -print0 2>/dev/null)
}

notarize() {
  local path="$1"
  if [[ "$NOTARIZED" != 1 ]]; then
    if [[ "$SIGNED" == 1 ]]; then
      echo "WARNING: 有签名身份但缺公证凭据，跳过公证：$(basename "$path")" >&2
    fi
    return 0
  fi
  echo "==> Notarizing $(basename "$path") (may take several minutes)"
  xcrun notarytool submit "$path" \
    --apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID" --wait
  xcrun stapler staple "$path"
  xcrun stapler validate "$path"
}

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/out/desktop/cargo}"
mkdir -p "$CARGO_TARGET_DIR"

if [[ -n "${DANQING_DESKTOP_VERSION:-}" ]]; then
  bash "$ROOT/scripts/set_desktop_version.sh" "$DANQING_DESKTOP_VERSION"
elif git -C "$ROOT" describe --exact-match --tags HEAD >/dev/null 2>&1; then
  echo "==> Desktop version from git tag"
  bash "$ROOT/scripts/set_desktop_version.sh"
fi

bash "$ROOT/scripts/prepare_tauri_resources.sh"
bash "$ROOT/scripts/ensure_tauri_resource_stub.sh"
sign_mach_o_tree "$ROOT/desktop/src-tauri/danqing-api"

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

bash "$ROOT/scripts/stage_desktop_bundle.sh"

if APP=$(find "$ROOT/out/desktop/bundle" -name "*.app" -print -quit); then
  SC_DIR="$APP/Contents/Resources/danqing-api"
  if [[ ! -d "$SC_DIR" ]]; then
    echo "Warning: sidecar not at $SC_DIR (check prepare_tauri_resources.sh)" >&2
  elif [[ ! -f "$SC_DIR/danqing-api" ]]; then
    echo "ERROR: $APP 里没有 danqing-api sidecar" >&2
    ls -la "$SC_DIR" >&2 || true
    exit 1
  fi

  if [[ "$SIGNED" == 1 ]]; then
    echo "==> Verifying code signature..."
    codesign --verify --deep --strict --verbose=2 "$APP"
    sig_report="$(codesign -dv --verbose=2 "$APP" 2>&1 || true)"
    if [[ "$sig_report" != *"Authority=Developer ID Application"* ]]; then
      echo "ERROR: $APP 不是 Developer ID 签名（检查 APPLE_SIGNING_IDENTITY 与 Tauri 签名配置）" >&2
      printf '%s\n' "$sig_report" >&2
      exit 1
    fi
    echo "==> Signature OK"
    if [[ "$NOTARIZED" == 1 ]]; then
      if xcrun stapler validate "$APP" >/dev/null 2>&1; then
        echo "==> App bundle already notarized (ticket stapled)"
      else
        echo "WARNING: .app 没有公证票据，尝试 staple" >&2
        xcrun stapler staple "$APP" || echo "WARNING: staple 失败，app 可能未公证" >&2
      fi
    fi
  else
    echo "==> codesign .app (ad-hoc)"
    if [[ -d "$SC_DIR" ]]; then
      while IFS= read -r -d '' bin; do
        codesign -s - --force "$bin" 2>/dev/null || true
      done < <(find "$SC_DIR" -type f \( -perm -111 -o -name '*.dylib' -o -name '*.so' \) -print0 2>/dev/null)
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
fi

# Create DMG with installation guide and one-click fix script
if APP=$(find "$ROOT/out/desktop/bundle" -name "*.app" -print -quit); then
  BUNDLE_DIR="$ROOT/out/desktop/bundle"
  DMG_STAGING="$BUNDLE_DIR/_dmg_staging"
  rm -rf "$DMG_STAGING"
  mkdir -p "$DMG_STAGING"

  cp -R "$APP" "$DMG_STAGING/"
  APP_NAME=$(basename "$APP")

  if [[ "$NOTARIZED" == 1 ]]; then
    cat > "$DMG_STAGING/阅读说明.txt" << 'README_NOTARIZED_EOF'
📦 Danmo Make 安装说明 (macOS)

  1. 将 Danmo Make.app 拖入「应用程序」文件夹
  2. 双击打开即可

本应用已使用 Developer ID 签名，并通过 Apple 公证。
README_NOTARIZED_EOF
  elif [[ "$SIGNED" == 1 ]]; then
    cat > "$DMG_STAGING/阅读说明.txt" << 'README_SIGNED_EOF'
📦 Danmo Make 安装说明 (macOS)

  1. 将 Danmo Make.app 拖入「应用程序」文件夹
  2. 双击打开即可

本应用已使用 Developer ID 签名，但未经过 Apple 公证。
首次打开若被拦截，请到「系统设置 → 隐私与安全性」点「仍要打开」。
README_SIGNED_EOF
  else
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

    cat > "$DMG_STAGING/修复并打开.command" << 'FIX_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_PATH="$SCRIPT_DIR/Danmo Make.app"
DMG_VOL="/Volumes/Danmo Make"
if [ -d "$DMG_VOL" ]; then
  echo "==> 正在移除 DMG 卷隔离属性..."
  xattr -cr "$DMG_VOL" 2>/dev/null
fi
if [ ! -d "$APP_PATH" ]; then
  APP_PATH="/Applications/Danmo Make.app"
fi
if [ ! -d "$APP_PATH" ]; then
  echo "❌ 未找到 Danmo Make.app"
  read -p "按回车退出..."
  exit 1
fi
xattr -cr "$APP_PATH" 2>/dev/null
open "$APP_PATH"
FIX_EOF
    chmod +x "$DMG_STAGING/修复并打开.command"
  fi

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

  if [[ "$NOTARIZED" == 1 && -f "$DMG_PATH" ]]; then
    notarize "$DMG_PATH"
  elif [[ "$SIGNED" != 1 ]]; then
    xattr -cr "$DMG_PATH" 2>/dev/null && echo "==> Removed quarantine on DMG" || true
  fi

  rm -rf "$DMG_STAGING"
fi
