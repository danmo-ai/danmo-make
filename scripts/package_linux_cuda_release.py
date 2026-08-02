#!/usr/bin/env python3
"""Stage ``out/sidecar/danqing-api`` and produce a Linux CUDA server ``.tar.gz`` (no Tauri)."""

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

_RUN_SH = """#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export DANQING_USER_DATA_DIR="${DANQING_USER_DATA_DIR:-$HOME/danqing-data}"
mkdir -p "$DANQING_USER_DATA_DIR"/{models,outputs,db,config}
export DANQING_HTTP_HOST="${DANQING_HTTP_HOST:-0.0.0.0}"
export DANQING_HTTP_PORT="${DANQING_HTTP_PORT:-7800}"
exec "$ROOT/danqing-api/danqing-api"
"""

_README = """Danmo Make — Linux CUDA server bundle
=====================================

Contents:
  danqing-api/   PyInstaller onedir (FastAPI + web UI)
  run.sh         Start the API server

Requirements on the host:
  - Linux x86_64 (glibc; build on Ubuntu 22.04+ recommended)
  - NVIDIA driver compatible with the bundled PyTorch CUDA runtime

Quick start:
  export DANQING_USER_DATA_DIR=$HOME/danqing-data   # optional
  ./run.sh

Open http://127.0.0.1:7800 (or your host:7800). API docs: /docs

Place model weights under $DANQING_USER_DATA_DIR/models/ per config/models_registry.json.
Copy default_config/models_registry.json into $DANQING_USER_DATA_DIR/config/ on first run if needed.

Environment:
  DANQING_USER_DATA_DIR  Writable data root (models, outputs, db, config)
  DANQING_HTTP_HOST      Bind address (default 0.0.0.0)
  DANQING_HTTP_PORT      Port (default 7800)

Only registry models with backends including \"cuda\" are supported in this bundle.
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


def package(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    dist_root = op.OUT_ROOT / "dist"
    dist_root.mkdir(parents=True, exist_ok=True)

    bundle_name = f"danmo-make-linux-cuda-x86_64-{ver}"
    staging = dist_root / bundle_name
    sidecar = op.SIDECAR_DIR

    if not sidecar.is_dir():
        raise SystemExit(
            f"Missing sidecar at {sidecar}. Build first:\n"
            "  make pack-linux-server-sidecar\n"
        )
    exe = sidecar / "danqing-api"
    if not exe.is_file():
        raise SystemExit(f"Missing executable {exe} (PyInstaller onedir incomplete).")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(sidecar, staging / "danqing-api")

    run_sh = staging / "run.sh"
    run_sh.write_text(_RUN_SH, encoding="utf-8")
    run_sh.chmod(0o755)

    (staging / "README.txt").write_text(_README, encoding="utf-8")

    archive = dist_root / f"{bundle_name}.tar.gz"
    if archive.exists():
        archive.unlink()
    subprocess.run(
        ["tar", "-C", str(dist_root), "-czf", str(archive), bundle_name],
        check=True,
    )
    print("Release archive:", archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Linux CUDA server tar.gz")
    parser.add_argument("--version", help="Release version (default: RELEASE_VERSION or git describe)")
    args = parser.parse_args()
    package(version=args.version)


if __name__ == "__main__":
    main()
