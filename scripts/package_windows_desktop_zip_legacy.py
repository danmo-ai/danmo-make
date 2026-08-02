#!/usr/bin/env python3
"""Legacy Windows desktop zip with PyInstaller CUDA sidecar."""

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

_README = """Danmo Make — Windows CUDA desktop (legacy sidecar portable)
==========================================================

Contains danqing-api PyInstaller onedir with torch. Prefer the thin zip.
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
    raise SystemExit(f"Missing danqing-desktop.exe under {target}")


def _sidecar_dir() -> Path:
    staged = op.TAURI_STAGED_SIDECAR
    if (staged / "danqing-api.exe").is_file() or (staged / "danqing-api").is_file():
        return staged
    if (op.SIDECAR_DIR / "danqing-api.exe").is_file() or (op.SIDECAR_DIR / "danqing-api").is_file():
        return op.SIDECAR_DIR
    raise SystemExit(f"Missing sidecar at {staged} or {op.SIDECAR_DIR}")


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
    for dll in exe.parent.glob("*.dll"):
        shutil.copy2(dll, staging / dll.name)

    shutil.copytree(sidecar, staging / "danqing-api")
    (staging / "README.txt").write_text(_README, encoding="utf-8", newline="\r\n")

    archive = zip_dir / f"DanmoMake_{ver}_x64-portable-legacy-sidecar.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as zf:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(staging)))

    shutil.rmtree(staging, ignore_errors=True)
    print(f"Legacy portable zip -> {archive}")
    return archive


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--version")
    args = p.parse_args()
    package(version=args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
