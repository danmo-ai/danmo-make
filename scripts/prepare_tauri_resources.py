#!/usr/bin/env python3
"""Stage PyInstaller sidecar into ``desktop/src-tauri/danqing-api`` for Tauri bundling."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402


def _sidecar_executable(sidecar: Path) -> Path:
    for name in ("danqing-api.exe", "danqing-api"):
        exe = sidecar / name
        if exe.is_file():
            return exe
    raise SystemExit(
        f"Missing sidecar executable under {sidecar} (expected danqing-api or danqing-api.exe).\n"
        "Run: make pack-macos-desktop-sidecar | pack-linux-desktop-sidecar | pack-windows-sidecar"
    )


def prepare(*, src: Path | None = None, dst: Path | None = None) -> Path:
    src_dir = src or op.SIDECAR_DIR
    dst_dir = dst or (op.PROJECT_ROOT / "desktop" / "src-tauri" / "danqing-api")

    if not src_dir.is_dir():
        raise SystemExit(f"Missing sidecar directory: {src_dir}")

    _sidecar_executable(src_dir)

    print(f"==> Stage sidecar for Tauri: {dst_dir.relative_to(op.PROJECT_ROOT)}")
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return dst_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage danqing-api sidecar for Tauri resources")
    parser.add_argument("--src", type=Path, default=None)
    parser.add_argument("--dst", type=Path, default=None)
    args = parser.parse_args()
    prepare(src=args.src, dst=args.dst)


if __name__ == "__main__":
    main()
