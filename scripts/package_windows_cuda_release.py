#!/usr/bin/env python3
"""Stage ``out/sidecar/danqing-api`` and produce a Windows CUDA server ``.zip`` (no Tauri)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402

_RUN_BAT = r"""@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
if not defined DANQING_USER_DATA_DIR set "DANQING_USER_DATA_DIR=%USERPROFILE%\danqing-data"
if not exist "%DANQING_USER_DATA_DIR%\models" mkdir "%DANQING_USER_DATA_DIR%\models"
if not exist "%DANQING_USER_DATA_DIR%\outputs" mkdir "%DANQING_USER_DATA_DIR%\outputs"
if not exist "%DANQING_USER_DATA_DIR%\db" mkdir "%DANQING_USER_DATA_DIR%\db"
if not exist "%DANQING_USER_DATA_DIR%\config" mkdir "%DANQING_USER_DATA_DIR%\config"
if not defined DANQING_HTTP_HOST set "DANQING_HTTP_HOST=0.0.0.0"
if not defined DANQING_HTTP_PORT set "DANQING_HTTP_PORT=7800"
"%ROOT%danqing-api\danqing-api.exe"
endlocal
"""

_README = """Danmo Make — Windows CUDA server bundle
=======================================

Contents:
  danqing-api\\   PyInstaller onedir (FastAPI + web UI)
  run.bat         Start the API server

Requirements on the host:
  - Windows 10/11 x64
  - NVIDIA driver compatible with the bundled PyTorch CUDA runtime

Quick start:
  set DANQING_USER_DATA_DIR=%USERPROFILE%\\danqing-data
  run.bat

Open http://127.0.0.1:7800 in a browser. API docs: /docs

Place model weights under %DANQING_USER_DATA_DIR%\\models\\ per config/models_registry.json.
Copy default_config\\models_registry.json into %DANQING_USER_DATA_DIR%\\config\\ on first run if needed.

Environment:
  DANQING_USER_DATA_DIR  Writable data root (models, outputs, db, config)
  DANQING_HTTP_HOST      Bind address (default 0.0.0.0)
  DANQING_HTTP_PORT      Port (default 7800)

Only registry models with backends including "cuda" are supported in this bundle.
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


def _sidecar_executable(sidecar: Path) -> Path:
    exe = sidecar / "danqing-api.exe"
    if exe.is_file():
        return exe
    fallback = sidecar / "danqing-api"
    if fallback.is_file():
        return fallback
    raise SystemExit(
        f"Missing executable under {sidecar} (expected danqing-api.exe from PyInstaller onedir)."
    )


def package(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    dist_root = op.OUT_ROOT / "dist"
    dist_root.mkdir(parents=True, exist_ok=True)

    bundle_name = f"danqing-studio-windows-cuda-x86_64-{ver}"
    staging = dist_root / bundle_name
    sidecar = op.SIDECAR_DIR

    if not sidecar.is_dir():
        raise SystemExit(
            f"Missing sidecar at {sidecar}. Build first:\n"
            "  make pack-windows-sidecar\n"
        )
    _sidecar_executable(sidecar)

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(sidecar, staging / "danqing-api")

    (staging / "run.bat").write_text(_RUN_BAT, encoding="utf-8", newline="\r\n")
    (staging / "README.txt").write_text(_README, encoding="utf-8", newline="\r\n")

    archive = dist_root / f"{bundle_name}.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                arcname = f"{bundle_name}/{path.relative_to(staging).as_posix()}"
                zf.write(path, arcname)

    print("Release archive:", archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Windows CUDA server zip")
    parser.add_argument("--version", help="Release version (default: RELEASE_VERSION or git describe)")
    args = parser.parse_args()
    package(version=args.version)


if __name__ == "__main__":
    main()
