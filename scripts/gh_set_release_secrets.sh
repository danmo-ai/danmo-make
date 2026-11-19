#!/usr/bin/env bash
# 写入 macOS 发布用的 GitHub repository secrets（Developer ID + 公证）。
#
#   scripts/gh_set_release_secrets.sh                 证书直接从钥匙串导出
#   scripts/gh_set_release_secrets.sh -R owner/repo
#   scripts/gh_set_release_secrets.sh --p12 path.p12  从指定 p12 读（钥匙串里没有时）
#
# 证书、签名身份、Team ID 都能从本机钥匙串里拿到，所以不用准备 p12：
# 脚本用 security export 把 Developer ID 身份导出成临时 p12，转单行 base64 后写入
# APPLE_CERTIFICATE，导出密码随机生成，用户既不用记也不用输。
#
# 值都从 stdin 喂给 gh（不进 ps、不回显），临时文件在退出时删除。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

P12=""
REPO=""
KEY_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -R|--repo) REPO="${2:-}"; shift 2 ;;
    --p12) P12="${2:-}"; shift 2 ;;
    --key) KEY_PATH="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "未知参数：$1" >&2; exit 1 ;;
    *) P12="$1"; shift ;;
  esac
done

command -v gh >/dev/null || { echo "找不到 gh：brew install gh" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "先跑 gh auth login" >&2; exit 1; }

ask() {
  local def="${2:-}" v=""
  if [[ -n "$def" ]]; then printf '%s [%s]: ' "$1" "$def" >&2; else printf '%s: ' "$1" >&2; fi
  read -r v || true
  printf '%s' "${v:-$def}"
}
ask_secret() { printf '%s' "$1" >&2; local v=""; read -rs v || true; printf '\n' >&2; printf '%s' "$v"; }

TMP="$(mktemp -d "${TMPDIR:-/tmp}/ghsecrets.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

put() { # put <名称> <文件>
  printf '  → %s ... ' "$1" >&2
  if gh secret set "$1" --repo "$REPO" < "$2" 2>/dev/null; then echo "ok" >&2; else echo "失败" >&2; fi
}

if [[ -z "$REPO" ]]; then
  REPO="$(git -C "$ROOT" remote get-url origin 2>/dev/null \
    | sed -e 's#^git@[^:]*:##' -e 's#^https\{0,1\}://[^/]*/##' -e 's#\.git$##')"
fi

echo
REPO="$(ask "仓库 OWNER/REPO" "$REPO")"
echo "目标：$REPO"

echo
echo "[1/2] macOS 签名证书"

identities="$(security find-identity -v -p codesigning 2>/dev/null \
  | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p')"

if [[ -z "$identities" && -z "$P12" ]]; then
  echo "  钥匙串里没有 Developer ID Application 证书。" >&2
  echo "  它应该在「钥匙串访问」里（从 Apple Developer 后台下载的 .cer 双击导入）。" >&2
  P12="$(ask "也可以给一个 p12 路径（回车退出）" "")"
  P12="${P12/#\~/$HOME}"
fi

DEVELOPER_ID=""
TEAM_FROM_OU=""
if [[ -n "$P12" ]]; then
  P12="${P12/#\~/$HOME}"
  [[ -f "$P12" ]] || { echo "找不到 ${P12}" >&2; exit 1; }

  p12_pw="$(ask_secret "p12 导出密码（没设密码就直接回车）：")"
  printf '%s\n' "$p12_pw" > "$TMP/pwfile"

  read_subject() {
    local pem
    pem="$(openssl pkcs12 -in "$P12" -passin "file:$TMP/pwfile" -nokeys -clcerts "$@" 2>/dev/null)" || return 1
    [[ -n "$pem" ]] || return 1
    pem="$(printf '%s' "$pem" | openssl x509 -noout -subject -nameopt sep_multiline 2>/dev/null)" || return 1
    DEVELOPER_ID="$(printf '%s\n' "$pem" | sed -n 's/^[[:space:]]*CN[[:space:]]*=[[:space:]]*//p' \
      | sed -n '/^Developer ID Application/p' | head -1)"
    TEAM_FROM_OU="$(printf '%s\n' "$pem" | sed -n 's/^[[:space:]]*OU[[:space:]]*=[[:space:]]*//p' | head -1)"
    [[ -n "$DEVELOPER_ID" ]]
  }
  if ! read_subject; then
    read_subject -legacy || true
  fi

  if [[ -z "$DEVELOPER_ID" ]]; then
    probe="$TMP/probe.keychain-db"
    probe_pw="probe-$$-$RANDOM"
    if security create-keychain -p "$probe_pw" "$probe" >/dev/null 2>&1 \
      && security unlock-keychain -p "$probe_pw" "$probe" >/dev/null 2>&1 \
      && security import "$P12" -k "$probe" -P "$p12_pw" -T /usr/bin/codesign >/dev/null 2>&1; then
      DEVELOPER_ID="$(security find-identity -p codesigning "$probe" 2>/dev/null \
        | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
    fi
    security delete-keychain "$probe" >/dev/null 2>&1 || true
  fi

  [[ -n "$DEVELOPER_ID" ]] || {
    echo "读不出证书里的 Developer ID 身份：密码不对，或导出的不是含私钥的 .p12。" >&2
    exit 1
  }

  base64 < "$P12" | tr -d '\n' > "$TMP/p12"
else
  count="$(printf '%s\n' "$identities" | grep -c . || true)"
  if [[ "$count" -gt 1 ]]; then
    printf '可用签名身份：\n' >&2
    printf '%s\n' "$identities" | nl -w2 -s') ' >&2
    pick="$(ask "用第几个" "1")"
    DEVELOPER_ID="$(printf '%s\n' "$identities" | sed -n "${pick}p")"
  else
    DEVELOPER_ID="$identities"
  fi
  [[ -n "$DEVELOPER_ID" ]] || { echo "没有可用的签名身份" >&2; exit 1; }

  export_pw="$(openssl rand -hex 24)"
  if ! security export -k login.keychain-db -t identities -f pkcs12 -P "$export_pw" -o "$TMP/p12" >/dev/null 2>&1; then
    echo "从钥匙串导出证书失败。可以手动导出 p12 后重跑：--p12 <路径>" >&2
    exit 1
  fi
  base64 < "$TMP/p12" | tr -d '\n' > "$TMP/p12b64"
  mv "$TMP/p12b64" "$TMP/p12"
  printf '%s' "$export_pw" > "$TMP/pwfile"
fi

put APPLE_CERTIFICATE "$TMP/p12"

if [[ -n "${export_pw:-}" ]]; then
  printf '%s' "$export_pw" > "$TMP/p12pw"
else
  printf '%s' "${p12_pw:-}" > "$TMP/p12pw"
fi
put APPLE_CERTIFICATE_PASSWORD "$TMP/p12pw"

printf '%s' "$(ask "签名身份" "$DEVELOPER_ID")" > "$TMP/id"
[[ -s "$TMP/id" ]] || { echo "签名身份不能为空" >&2; exit 1; }
case "$(cat "$TMP/id")" in
  "Developer ID Application: "*) ;;
  *) echo "  ! 这个身份不是「Developer ID Application: …」，公证会失败" >&2 ;;
esac
put APPLE_SIGNING_IDENTITY "$TMP/id"

echo
echo "[2/2] 公证"
printf '%s' "$(ask "Apple ID" "")" > "$TMP/appleid"
put APPLE_ID "$TMP/appleid"

printf '%s' "$(ask_secret "App 专用密码（abcd-efgh-ijkl-mnop）：")" > "$TMP/applepw"
put APPLE_PASSWORD "$TMP/applepw"

TEAM_ID="${TEAM_FROM_OU:-}"
if [[ -z "$TEAM_ID" ]]; then
  team_src="$(cat "$TMP/id")"
  TEAM_ID="$(printf '%s' "$team_src" | sed -n 's/.*(\([A-Z0-9]\{10\}\))[[:space:]]*$/\1/p')"
fi
printf '%s' "$(ask "Team ID" "$TEAM_ID")" > "$TMP/team"
[[ -s "$TMP/team" ]] || { echo "Team ID 不能为空" >&2; exit 1; }
put APPLE_TEAM_ID "$TMP/team"

echo
gh secret list --repo "$REPO" 2>/dev/null | sed 's/^/  /'
echo
echo "触发一次发布：git tag v0.1.0 && git push origin v0.1.0"