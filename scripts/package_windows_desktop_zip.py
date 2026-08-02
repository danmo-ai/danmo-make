#!/usr/bin/env python3
"""Package Windows Tauri desktop + CUDA sidecar as a portable ``.zip``.

NSIS cannot pack CUDA sidecars near/over ~2GB (makensis mmap Internal compiler
error #12345). Ship a portable zip instead; ``resource_dir`` is the folder that
contains the shell exe, with ``danqing-api/`` beside it.
"""

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

_README = """Danmo Make — Windows CUDA desktop (portable)
=============================================

Contents:
  danqing-desktop.exe   Tauri shell
  danqing-api\\          PyInstaller onedir (FastAPI + web UI)

Requirements:
  - Windows 10/11 x64
  - WebView2 Runtime (usually preinstalled on Win11 / recent Win10)
  - NVIDIA driver compatible with the bundled PyTorch CUDA runtime

Quick start:
  1. Unzip this archive to a short path (e.g. C:\\DanmoMake) — avoid deep/Unicode paths.
  2. Run danqing-desktop.exe

Data directory (models, outputs, db, config) defaults under your user profile
unless DANQING_USER_DATA_DIR is set.

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


def _find_shell_exe() -> Path:
    target = op.DESKTOP_CARGO_TARGET / "x86_64-pc-windows-msvc" / "release"
    exe = target / "danqing-desktop.exe"
    if exe.is_file():
        return exe
    fallback = op.DESKTOP_CARGO_TARGET / "release" / "danqing-desktop.exe"
    if fallback.is_file():
        return fallback
    raise SystemExit(
        f"Missing danqing-desktop.exe under {target} (or {fallback.parent}).\n"
        "Run: python scripts/tauri_build.py --platform windows"
    )


def _sidecar_dir() -> Path:
    staged = op.TAURI_STAGED_SIDECAR
    if (staged / "danqing-api.exe").is_file() or (staged / "danqing-api").is_file():
        return staged
    if (op.SIDECAR_DIR / "danqing-api.exe").is_file() or (op.SIDECAR_DIR / "danqing-api").is_file():
        return op.SIDECAR_DIR
    raise SystemExit(
        f"Missing sidecar at {staged} or {op.SIDECAR_DIR}.\n"
        "Run: python scripts/build_sidecar.py (CUDA profile)"
    )


def package(*, version: str | None = None) -> Path:
    ver = _release_version(version)
    exe = _find_shell_exe()
    sidecar = _sidecar_dir()

    zip_dir = op.DESKTOP_BUNDLE_DIR / "zip"
    zip_dir.mkdir(parents=True, exist_ok=True)
    staging = op.DESKTOP_BUNDLE_DIR / "portable-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    shutil.copy2(exe, staging / "danqing-desktop.exe")
    # WebView2 / VC runtime DLLs next to the shell when present.
    for dll in exe.parent.glob("*.dll"):
        shutil.copy2(dll, staging / dll.name)

    shutil.copytree(sidecar, staging / "danqing-api")
    (staging / "README.txt").write_text(_README, encoding="utf-8", newline="\r\n")

    archive = zip_dir / f"DanmoMake_{ver}_x64-portable.zip"
    if archive.exists():
        archive.unlink()
    # Light deflate (level 1) to stay under GitHub's 2 GiB per-asset limit when possible.
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
        allowZip64=True,
    ) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(staging)))

    shutil.rmtree(staging, ignore_errors=True)
    size_gb = archive.stat().st_size / (1024**3)
    print(f"Portable Windows desktop zip -> {archive.relative_to(op.PROJECT_ROOT)} ({size_gb:.2f} GiB)")
    return archive


def main() -> int:
    p = argparse.ArgumentParser(description="Package Windows CUDA desktop portable zip")
    p.add_argument("--version", help="X.Y.Z (default: RELEASE_VERSION or git describe)")
    args = p.parse_args()
    package(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
