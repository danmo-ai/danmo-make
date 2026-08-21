#!/usr/bin/env python3
"""Package thin Linux MLX desktop / server archives (no embedded mlx wheels)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402
import stage_cuda_runtime as stage  # noqa: E402

_RUN_SH = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export DANQING_USER_DATA_DIR="${DANQING_USER_DATA_DIR:-$HOME/.danmo-make}"
mkdir -p "$DANQING_USER_DATA_DIR"/{models,outputs,db,config,logs}
export DANQING_HTTP_HOST="${DANQING_HTTP_HOST:-0.0.0.0}"
export DANQING_HTTP_PORT="${DANQING_HTTP_PORT:-7800}"
export DANQING_APP_ROOT="$ROOT/runtime/app"
export DANQING_PORTABLE_PYTHON="$ROOT/runtime/python"
export PYTHONPATH="$DANQING_APP_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BOOTSTRAP_PY="$DANQING_APP_ROOT/scripts/runtime_bootstrap.py"
PY_BOOT="$DANQING_PORTABLE_PYTHON/bin/python3"
if [[ ! -x "$PY_BOOT" ]]; then
  PY_BOOT="$DANQING_PORTABLE_PYTHON/bin/python"
fi

mode=""
for arg in "$@"; do
  case "$arg" in
    --repair-runtime) mode=repair ;;
    --reinstall-runtime) mode=reinstall ;;
    --status-runtime) mode=status ;;
  esac
done

if [[ "$mode" == "status" ]]; then
  exec "$PY_BOOT" "$BOOTSTRAP_PY" --status --data-dir "$DANQING_USER_DATA_DIR" \\
    --app-root "$DANQING_APP_ROOT" --portable-python "$DANQING_PORTABLE_PYTHON"
fi

need_setup=1
if "$PY_BOOT" "$BOOTSTRAP_PY" --status --data-dir "$DANQING_USER_DATA_DIR" \\
    --app-root "$DANQING_APP_ROOT" --portable-python "$DANQING_PORTABLE_PYTHON" >/dev/null 2>&1; then
  need_setup=0
fi

if [[ "$mode" == "repair" || "$mode" == "reinstall" ]]; then
  need_setup=1
fi

if [[ "$need_setup" -eq 1 ]]; then
  if [[ "${DANQING_RUNTIME_SKIP_AUTO_SETUP:-0}" == "1" && -z "$mode" ]]; then
    echo "Runtime not ready. Run: $0 --repair-runtime" >&2
    echo "Or unset DANQING_RUNTIME_SKIP_AUTO_SETUP to auto-install on start." >&2
    exit 1
  fi
  echo "==> Installing MLX (mlx[cuda]) Python runtime (progress on stderr)…" >&2
  extra=()
  if [[ "$mode" == "repair" ]]; then
    extra+=(--repair)
  elif [[ "$mode" == "reinstall" ]]; then
    extra+=(--reinstall --yes)
  fi
  "$PY_BOOT" "$BOOTSTRAP_PY" --data-dir "$DANQING_USER_DATA_DIR" \\
    --app-root "$DANQING_APP_ROOT" --portable-python "$DANQING_PORTABLE_PYTHON" \\
    "${extra[@]}"
fi

VENV_PY="$DANQING_USER_DATA_DIR/runtime-venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing venv python at $VENV_PY" >&2
  exit 1
fi
cd "$DANQING_APP_ROOT"
exec "$VENV_PY" -m uvicorn backend.main:app --host "$DANQING_HTTP_HOST" --port "$DANQING_HTTP_PORT"
"""

_SERVER_README = """Danmo Make — Linux MLX server (thin bundle)
==========================================

This archive does NOT include mlx wheels. On first start, run.sh downloads and
installs mlx[cuda] + app deps into $DANQING_USER_DATA_DIR/runtime-venv with
console progress.

Contents:
  runtime/python/   Portable CPython
  runtime/app/      Application code + web UI
  run.sh            Start API (auto bootstrap if needed)

Requirements:
  - x86_64 Linux
  - NVIDIA driver (mlx[cuda]; no CPU fallback)
  - Network on first run

Quick start:
  export DANQING_USER_DATA_DIR=$HOME/.danmo-make   # optional
  export DANQING_PIP_MIRROR=tuna                   # optional: official|tuna|aliyun
  ./run.sh

Repair / reinstall:
  ./run.sh --repair-runtime
  ./run.sh --reinstall-runtime
  ./run.sh --status-runtime

Environment:
  DANQING_USER_DATA_DIR           Writable data root
  DANQING_HTTP_HOST / _PORT       Bind (default 0.0.0.0:7800)
  DANQING_PIP_MIRROR              official|tuna|aliyun
  DANQING_RUNTIME_SKIP_AUTO_SETUP Set to 1 to refuse start when not ready
  DANQING_RUNTIME_QUIET           Set to 1 to reduce console progress (still logs)

Open http://127.0.0.1:7800 — API docs at /docs

Windows is temporarily unsupported.
"""


def _release_version(explicit: str | None) -> str:
    if explicit:
        return explicit.strip().lstrip("v")
    env = os.environ.get("RELEASE_VERSION", "").strip()
    if env:
        return env.lstrip("v")
    try:
        out = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=op.PROJECT_ROOT,
            text=True,
        ).strip()
        return out.lstrip("v")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "dev"


def _ensure_runtime(platform: str) -> Path:
    runtime = op.OUT_ROOT / "runtime"
    if not (runtime / "python").is_dir() or not (runtime / "app" / "backend").is_dir():
        stage.stage_runtime(platform=platform, dest=runtime)
    return runtime


def package_linux_server(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    runtime = _ensure_runtime("linux-x86_64")
    dist_root = op.OUT_ROOT / "dist"
    dist_root.mkdir(parents=True, exist_ok=True)
    bundle_name = f"danmo-make-linux-mlx-x86_64-{ver}"
    staging = dist_root / bundle_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(runtime, staging / "runtime")
    run_sh = staging / "run.sh"
    run_sh.write_text(_RUN_SH, encoding="utf-8")
    run_sh.chmod(0o755)
    (staging / "README.txt").write_text(_SERVER_README, encoding="utf-8")
    # Convenience CLI wrapper
    setup = staging / "danqing-runtime-setup"
    setup.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ROOT="$(cd "$(dirname "$0")" && pwd)"\n'
        'export DANQING_USER_DATA_DIR="${DANQING_USER_DATA_DIR:-$HOME/.danmo-make}"\n'
        'exec "$ROOT/runtime/python/bin/python3" "$ROOT/runtime/app/scripts/runtime_bootstrap.py" \\\n'
        '  --data-dir "$DANQING_USER_DATA_DIR" --app-root "$ROOT/runtime/app" \\\n'
        '  --portable-python "$ROOT/runtime/python" "$@"\n',
        encoding="utf-8",
    )
    setup.chmod(0o755)

    archive = dist_root / f"{bundle_name}.tar.gz"
    if archive.exists():
        archive.unlink()
    subprocess.run(
        ["tar", "-C", str(dist_root), "-czf", str(archive), bundle_name],
        check=True,
    )
    print("Release archive:", archive)
    return archive


def package_windows_server(*, version: str | None = None) -> Path:
    raise SystemExit("Windows temporarily unsupported")


def package_windows_desktop(*, version: str | None = None) -> Path:
    raise SystemExit("Windows temporarily unsupported")


def main() -> int:
    p = argparse.ArgumentParser(description="Package thin Linux MLX archives")
    p.add_argument(
        "--product",
        choices=("linux-server", "windows-server", "windows-desktop"),
        required=True,
    )
    p.add_argument("--version", default=None)
    args = p.parse_args()
    if args.product == "linux-server":
        package_linux_server(version=args.version)
    elif args.product in ("windows-server", "windows-desktop"):
        raise SystemExit("Windows temporarily unsupported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
