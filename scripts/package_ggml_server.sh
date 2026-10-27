#!/usr/bin/env bash
# Headless archive: ggml danqing-api + built frontend. No Python runtime.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

VERSION="${RELEASE_VERSION:-$(git -C "$DQ_ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)}"
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
NAME="danmo-make-${OS_NAME}-ggml-${ARCH}-${VERSION}"
STAGE="$DQ_OUT/server/$NAME"

dq_ensure_out_layout
bash "$SCRIPT_DIR/build_ggml_sidecar.sh"
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/frontend"
cp -a "$DQ_SIDECAR_DIR/." "$STAGE/bin/"
if [[ -d "$DQ_FRONTEND_DIST" ]]; then
  cp -a "$DQ_FRONTEND_DIST/." "$STAGE/frontend/"
fi
cp "$DQ_ROOT/default_config/models_registry.json" "$STAGE/models_registry.json"
cat > "$STAGE/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export DANQING_REGISTRY="${DANQING_REGISTRY:-$ROOT/models_registry.json}"
export DANQING_HTTP_HOST="${DANQING_HTTP_HOST:-127.0.0.1}"
export DANQING_HTTP_PORT="${DANQING_HTTP_PORT:-7800}"
export DANQING_DB="${DANQING_DB:-$ROOT/db/studio.db}"
mkdir -p "$(dirname "$DANQING_DB")"
exec "$ROOT/bin/danqing-api"
EOF
chmod +x "$STAGE/run.sh"
mkdir -p "$DQ_RELEASE_DIST"
tar -C "$DQ_OUT/server" -czf "$DQ_RELEASE_DIST/${NAME}.tar.gz" "$NAME"
echo "packed $DQ_RELEASE_DIST/${NAME}.tar.gz"
