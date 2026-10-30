# Resolve media workspace for danqing-api.
# Source after out_paths.sh (needs DQ_ROOT).
#
# Priority:
#   1. DANQING_WORKSPACE (explicit)
#   2. ~/.danmo-make/workspace.pointer.json → custom_workspace_dir
#      (override control plane with DANQING_USER_DATA_DIR)
#   3. DQ_ROOT (repo-local / first-run default)
#
# Also sets DANQING_DB / DANQING_OUTPUT / DANQING_REGISTRY when unset.
# Product catalog stays the factory GGUF registry under default_config/.

if [[ -n "${_DQ_WORKSPACE_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || true
fi
_DQ_WORKSPACE_ENV_LOADED=1

dq_resolve_workspace_env() {
  local user_data pointer custom
  if [[ -z "${DANQING_WORKSPACE:-}" ]]; then
    user_data="${DANQING_USER_DATA_DIR:-$HOME/.danmo-make}"
    pointer="${user_data}/workspace.pointer.json"
    custom=""
    if [[ -f "$pointer" ]]; then
      custom="$(
        python3 - "$pointer" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)
raw = (data.get("custom_workspace_dir") or data.get("workspace_dir") or "").strip()
if raw:
    print(raw)
PY
      )"
    fi
    if [[ -n "$custom" && -d "$custom" ]]; then
      export DANQING_WORKSPACE="$custom"
    else
      export DANQING_WORKSPACE="$DQ_ROOT"
    fi
  fi

  export DANQING_DB="${DANQING_DB:-$DANQING_WORKSPACE/db/studio.db}"
  export DANQING_OUTPUT="${DANQING_OUTPUT:-$DANQING_WORKSPACE/outputs}"
  local factory_registry="$DQ_ROOT/default_config/models_registry.json"
  if [[ -z "${DANQING_REGISTRY:-}" || ! -f "${DANQING_REGISTRY}" ]]; then
    export DANQING_REGISTRY="$factory_registry"
  fi
  if [[ ! -f "$DANQING_REGISTRY" ]]; then
    echo "models registry not found: $DANQING_REGISTRY (DQ_ROOT=$DQ_ROOT)" >&2
    return 1
  fi

  mkdir -p "$(dirname "$DANQING_DB")" "$DANQING_OUTPUT"
}
