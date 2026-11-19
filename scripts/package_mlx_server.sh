#!/usr/bin/env bash
# Headless archive: MLX danqing-api sidecar + built frontend. No Python runtime.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=out_paths.sh
source "$SCRIPT_DIR/out_paths.sh"

VERSION="${RELEASE_VERSION:-$(git -C "$DQ_ROOT" describe --tags --always --dirty 2>/dev/null || echo dev)}"
OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
# Prefer mlx-cuda label when the sidecar was built with that feature (CI / nvcc hosts).
BACKEND_TAG="${DANQING_SERVER_BACKEND_TAG:-}"
if [[ -z "$BACKEND_TAG" ]]; then
  if command -v nvcc >/dev/null 2>&1 || [[ "${DANQING_SIDECAR_FEATURES:-}" == *mlx-cuda* ]]; then
    BACKEND_TAG=mlx-cuda
  elif [[ "$(uname -s)" == Darwin ]] || [[ "${DANQING_SIDECAR_FEATURES:-}" == *mlx-metal* ]]; then
    BACKEND_TAG=mlx-metal
  else
    echo "ERROR: set DANQING_SERVER_BACKEND_TAG or build sidecar with mlx-cuda / mlx-metal" >&2
    exit 1
  fi
fi
NAME="danmo-make-${OS_NAME}-${BACKEND_TAG}-${ARCH}-${VERSION}"
STAGE="$DQ_OUT/server/$NAME"

dq_ensure_out_layout
bash "$SCRIPT_DIR/build_mlx_sidecar.sh"
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
# Prefer an explicit workspace / control-plane pointer when present.
if [[ -z "${DANQING_WORKSPACE:-}" ]]; then
  _pointer="${DANQING_USER_DATA_DIR:-$HOME/.danmo-make}/workspace.pointer.json"
  if [[ -f "$_pointer" ]]; then
    DANQING_WORKSPACE="$(python3 -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]);
d=json.loads(p.read_text()); print((d.get("custom_workspace_dir") or d.get("workspace_dir") or "").strip())' "$_pointer" 2>/dev/null || true)"
    export DANQING_WORKSPACE
  fi
fi
if [[ -n "${DANQING_WORKSPACE:-}" ]]; then
  export DANQING_DB="${DANQING_DB:-$DANQING_WORKSPACE/db/studio.db}"
  export DANQING_OUTPUT="${DANQING_OUTPUT:-$DANQING_WORKSPACE/outputs}"
else
  export DANQING_DB="${DANQING_DB:-$ROOT/db/studio.db}"
fi
mkdir -p "$(dirname "$DANQING_DB")"
exec "$ROOT/bin/danqing-api"
EOF
chmod +x "$STAGE/run.sh"
mkdir -p "$DQ_RELEASE_DIST"
tar -C "$DQ_OUT/server" -czf "$DQ_RELEASE_DIST/${NAME}.tar.gz" "$NAME"
echo "packed $DQ_RELEASE_DIST/${NAME}.tar.gz"
