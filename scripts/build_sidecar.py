#!/usr/bin/env python3
"""
Build the FastAPI PyInstaller onedir sidecar for the Tauri desktop app.

Output: ``out/sidecar/danqing-api/`` (bundled by Tauri; see ``desktop/src-tauri/tauri.conf.json``)

Usage:
    python scripts/build_sidecar.py
    make pack-macos-desktop-sidecar  # or pack-linux-desktop-sidecar
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import out_paths as op  # noqa: E402
import pyinstaller_common as pc  # noqa: E402


def build(*, clean: bool = True) -> Path:
    if sys.platform == "win32":
        raise SystemExit("Windows is temporarily unsupported")

    try:
        import PyInstaller.__main__
    except ImportError as e:
        print("Error: PyInstaller not installed. pip install pyinstaller")
        raise SystemExit(1) from e

    entry_point = op.PROJECT_ROOT / "backend" / "main.py"
    if not entry_point.exists():
        print(f"Error: missing entry {entry_point}")
        raise SystemExit(1)

    op.ensure_out_layout()
    name = "danqing-api"
    cmd: list[str] = [
        str(entry_point),
        "--name",
        name,
        "--console",
        "--onedir",
    ]
    if clean:
        cmd.append("--clean")
    cmd.append("--noconfirm")

    if sys.platform == "darwin":
        import platform as _platform

        if _platform.machine().lower() == "arm64":
            cmd.extend(["--target-architecture", "arm64"])

    profile = pc.packaging_profile()
    if profile != "mlx":
        raise SystemExit(
            f"Only DANQING_PYINSTALLER_PROFILE=mlx is supported (got {profile!r})."
        )
    if sys.platform not in ("darwin", "linux"):
        raise SystemExit(
            f"Sidecar packaging is only supported on darwin/linux (got {sys.platform})."
        )
    print(f"Packaging profile: {profile}")

    for imp in pc.get_hidden_imports(profile):
        cmd.extend(["--hidden-import", imp])
    for data in pc.get_data_files(profile=profile):
        cmd.extend(["--add-data", data])
    for binary in pc.get_binary_files(op.PROJECT_ROOT, profile=profile):
        cmd.extend(["--add-binary", binary])
    for hook in pc.get_runtime_hooks(op.PROJECT_ROOT):
        cmd.extend(["--runtime-hook", hook])

    hooks_dir = pc.pyinstaller_hooks_dir(op.PROJECT_ROOT)
    if hooks_dir.is_dir():
        cmd.extend(["--additional-hooks-dir", str(hooks_dir)])

    pc.apply_pyinstaller_packaging_filters()
    for mod in pc.get_exclude_modules(profile):
        cmd.extend(["--exclude-module", mod])

    cmd.extend(["--distpath", str(op.SIDECAR_ROOT)])
    cmd.extend(["--workpath", str(op.PYINSTALLER_WORK)])
    cmd.extend(["--specpath", str(op.PYINSTALLER_SPEC)])

    if sys.platform == "darwin":
        cmd.append("--strip")

    print("Building sidecar:", name)
    PyInstaller.__main__.run(cmd)

    out = op.SIDECAR_DIR
    if not out.exists():
        print("Warning: expected output missing:", out)
    else:
        import prune_sidecar as prune  # noqa: E402

        removed, placed = prune.finalize_mlx_sidecar(out)
        if removed:
            print(f"Pruned sidecar ({len(removed)} entries)")
        if placed:
            print(f"MLX layout: {', '.join(placed)} next to danqing-api")
        print("Sidecar bundle:", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="PyInstaller sidecar for Tauri desktop")
    parser.add_argument("--no-clean", action="store_true", help="Skip PyInstaller --clean")
    args = parser.parse_args()
    build(clean=not args.no_clean)


if __name__ == "__main__":
    main()
