#!/usr/bin/env python3
"""Stage thin Linux MLX runtime tree (portable Python + app code, no mlx wheels)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import fetch_portable_python as pbs  # noqa: E402
import out_paths as op  # noqa: E402

_APP_COPY_DIRS = (
    "backend",
    "default_config",
    "bin",
)
_APP_COPY_FILES = (
    "requirements.txt",
    "requirements-linux.txt",
)


def _copytree_filtered(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".mypy_cache",
        "tests",
        "*.egg-info",
    )
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def stage_app_tree(dest_app: Path, *, frontend_dist: Path | None = None) -> Path:
    dest_app.mkdir(parents=True, exist_ok=True)
    root = op.PROJECT_ROOT
    for name in _APP_COPY_DIRS:
        src = root / name
        if not src.is_dir():
            raise SystemExit(f"Missing required directory {src}")
        _copytree_filtered(src, dest_app / name)
    for name in _APP_COPY_FILES:
        src = root / name
        if not src.is_file():
            raise SystemExit(f"Missing required file {src}")
        shutil.copy2(src, dest_app / name)

    # Bootstrap script lives under scripts/ in repo; ship a copy for run.sh / Tauri.
    scripts_dst = dest_app / "scripts"
    scripts_dst.mkdir(exist_ok=True)
    shutil.copy2(root / "scripts" / "runtime_bootstrap.py", scripts_dst / "runtime_bootstrap.py")

    dist = frontend_dist or op.FRONTEND_DIST
    if not dist.is_dir() or not any(dist.iterdir()):
        raise SystemExit(
            f"Missing frontend dist at {dist}. Run: make frontend-build"
        )
    frontend_out = dest_app / "out" / "frontend" / "dist"
    if frontend_out.exists():
        shutil.rmtree(frontend_out)
    frontend_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist, frontend_out)
    return dest_app


def stage_runtime(
    *,
    platform: str | None = None,
    dest: Path | None = None,
    fetch_python: bool = True,
) -> Path:
    out = dest or (op.OUT_ROOT / "runtime")
    out.mkdir(parents=True, exist_ok=True)
    if fetch_python:
        pbs.fetch(platform=platform, dest=out / "python")
    stage_app_tree(out / "app")
    print(f"Staged thin runtime -> {out}")
    return out


def prepare_tauri_resource(src: Path | None = None) -> Path:
    """Copy ``out/runtime`` into ``desktop/src-tauri/runtime`` for bundling."""
    src_dir = src or (op.OUT_ROOT / "runtime")
    dst = op.PROJECT_ROOT / "desktop" / "src-tauri" / "runtime"
    if not src_dir.is_dir():
            raise SystemExit(f"Missing thin runtime at {src_dir}; run stage_cuda_runtime first")
    if not (src_dir / "python").is_dir():
        raise SystemExit(f"Missing portable python under {src_dir / 'python'}")
    if not (src_dir / "app" / "backend").is_dir():
        raise SystemExit(f"Missing app tree under {src_dir / 'app'}")
    if dst.exists():
        shutil.rmtree(dst)
    print(f"==> Stage Tauri runtime resource: {dst.relative_to(op.PROJECT_ROOT)}")
    shutil.copytree(src_dir, dst)
    return dst


def main() -> int:
    p = argparse.ArgumentParser(description="Stage Linux MLX thin runtime (no mlx wheels)")
    p.add_argument("--platform", choices=("linux-x86_64",), default="linux-x86_64")
    p.add_argument("--dest", type=Path, default=None)
    p.add_argument("--skip-fetch-python", action="store_true")
    p.add_argument("--prepare-tauri", action="store_true", help="Also copy into desktop/src-tauri/runtime")
    args = p.parse_args()
    staged = stage_runtime(
        platform=args.platform,
        dest=args.dest,
        fetch_python=not args.skip_fetch_python,
    )
    if args.prepare_tauri:
        prepare_tauri_resource(staged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
