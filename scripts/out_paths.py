"""
Unified build output paths — all packaging artifacts live under ``out/``.

Layout::

    out/
      frontend/dist/          # Vite production build
      sidecar/danqing-api/  # PyInstaller onedir (Tauri bundles this tree)
      pyinstaller/work/     # PyInstaller cache
      pyinstaller/spec/     # Generated .spec files
      desktop/cargo/        # Cargo target-dir (intermediate Rust build)
      desktop/bundle/       # Staged .app / .dmg (final deliverables)
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"

FRONTEND_DIST = OUT_ROOT / "frontend" / "dist"
SIDECAR_ROOT = OUT_ROOT / "sidecar"
SIDECAR_DIR = SIDECAR_ROOT / "danqing-api"
RELEASE_DIST = OUT_ROOT / "dist"
PYINSTALLER_WORK = OUT_ROOT / "pyinstaller" / "work"
PYINSTALLER_SPEC = OUT_ROOT / "pyinstaller" / "spec"
DESKTOP_CARGO_TARGET = OUT_ROOT / "desktop" / "cargo"
DESKTOP_BUNDLE_DIR = OUT_ROOT / "desktop" / "bundle"
TAURI_STAGED_SIDECAR = PROJECT_ROOT / "desktop" / "src-tauri" / "danqing-api"


def ensure_out_layout() -> None:
    for path in (
        OUT_ROOT,
        FRONTEND_DIST.parent,
        SIDECAR_ROOT,
        RELEASE_DIST,
        PYINSTALLER_WORK,
        PYINSTALLER_SPEC,
        DESKTOP_CARGO_TARGET,
    ):
        path.mkdir(parents=True, exist_ok=True)


TAURI_STAGED_RUNTIME = PROJECT_ROOT / "desktop" / "src-tauri" / "runtime"


def clean_build_artifacts(*, include_frontend: bool = True) -> list[Path]:
    """Delete ``out/`` and staged Tauri resources. Returns paths that were removed."""
    removed: list[Path] = []
    if include_frontend:
        targets: list[Path] = [OUT_ROOT, TAURI_STAGED_SIDECAR, TAURI_STAGED_RUNTIME]
    else:
        targets = [
            OUT_ROOT / "sidecar",
            OUT_ROOT / "runtime",
            OUT_ROOT / "pyinstaller",
            OUT_ROOT / "desktop",
            TAURI_STAGED_SIDECAR,
            TAURI_STAGED_RUNTIME,
        ]

    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed
