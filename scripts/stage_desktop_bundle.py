#!/usr/bin/env python3
"""Copy Tauri ``release/bundle`` artifacts to ``out/desktop/bundle/``."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402


def find_cargo_bundle_dir() -> Path:
    base = op.DESKTOP_CARGO_TARGET
    if not base.is_dir():
        raise SystemExit(f"Missing Cargo target dir: {base}\nRun: make desktop-tauri")

    preferred = [
        base / "x86_64-pc-windows-msvc" / "release" / "bundle",
        base / "x86_64-unknown-linux-gnu" / "release" / "bundle",
        base / "aarch64-apple-darwin" / "release" / "bundle",
        base / "release" / "bundle",
    ]
    candidates = [p for p in preferred if p.is_dir()]
    candidates.extend(
        sorted(
            (p for p in base.glob("*/release/bundle") if p.is_dir() and p not in candidates),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    if not candidates:
        raise SystemExit(
            f"No Tauri bundle under {base}\n"
            "Expected */release/bundle (e.g. aarch64-apple-darwin/release/bundle "
            "or x86_64-pc-windows-msvc/release/bundle with NSIS)"
        )
    return candidates[0]


def _rm_rf(path: Path) -> None:
    """Remove a tree (prefer ``/bin/rm`` on macOS for stubborn .app bundles)."""
    if not path.exists():
        return
    if sys.platform == "darwin":
        subprocess.run(["/bin/rm", "-rf", str(path)], check=True)
    else:
        shutil.rmtree(path)


def stage_desktop_bundle() -> Path:
    src = find_cargo_bundle_dir()
    dst = op.DESKTOP_BUNDLE_DIR
    staging = dst.parent / f"{dst.name}.staging"

    dst.parent.mkdir(parents=True, exist_ok=True)
    _rm_rf(staging)
    shutil.copytree(src, staging)
    _rm_rf(dst)
    staging.rename(dst)
    return dst


def main() -> None:
    dst = stage_desktop_bundle()
    try:
        rel = dst.relative_to(op.PROJECT_ROOT)
    except ValueError:
        rel = dst
    print(f"Staged desktop bundle -> {rel}/")
    for child in sorted(dst.iterdir()):
        print(f"  {child.name}")


if __name__ == "__main__":
    main()
