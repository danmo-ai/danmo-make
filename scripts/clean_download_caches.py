#!/usr/bin/env python3
"""Remove Hugging Face and ModelScope download caches (user + workspace models/)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent

_SKIP_DIR_NAMES = {".venv", "node_modules", "out", "dist", ".git"}


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _format_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    units = ("KB", "MB", "GB", "TB")
    value = float(num)
    for unit in units:
        value /= 1024.0
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
    return f"{num} B"


def _read_workspace_root(project_root: Path) -> Path | None:
    pointers = [
        Path.home() / ".danmo-make" / "workspace.pointer.json",
        project_root / "default_config" / "workspace.pointer.json",
    ]
    for pointer in pointers:
        if not pointer.is_file():
            continue
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        raw = (data.get("custom_workspace_dir") or "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    return None


def _user_hf_cache_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HOME

        return Path(HF_HOME).expanduser().resolve()
    except Exception:
        env = os.environ.get("HF_HOME", "").strip()
        if env:
            return Path(env).expanduser().resolve()
        return (Path.home() / ".cache" / "huggingface").resolve()


def _user_modelscope_cache_root() -> Path:
    env = os.environ.get("MODELSCOPE_CACHE", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "modelscope").resolve()


def _scan_roots(project_root: Path) -> list[Path]:
    roots = [project_root.resolve()]
    workspace = _read_workspace_root(project_root)
    if workspace is not None and workspace not in roots:
        roots.append(workspace)
    return roots


def _discover_targets(project_root: Path) -> list[Path]:
    targets: list[Path] = []

    for cache_root in (_user_hf_cache_root(), _user_modelscope_cache_root()):
        if cache_root.exists():
            targets.append(cache_root)

    for root in _scan_roots(project_root):
        models_dir = root / "models"
        if not models_dir.is_dir():
            continue
        for cache_parent in models_dir.rglob(".cache"):
            if not cache_parent.is_dir() or cache_parent.name != ".cache":
                continue
            if any(part in _SKIP_DIR_NAMES for part in cache_parent.parts):
                continue
            for child_name in ("huggingface", "modelscope"):
                child = cache_parent / child_name
                if child.is_dir():
                    targets.append(child.resolve())
        for temp_dir in models_dir.rglob("._____temp"):
            if not temp_dir.is_dir():
                continue
            if any(part in _SKIP_DIR_NAMES for part in temp_dir.parts):
                continue
            targets.append(temp_dir.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in targets:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _remove_empty_cache_parents(project_root: Path) -> list[Path]:
    removed: list[Path] = []
    for root in _scan_roots(project_root):
        models_dir = root / "models"
        if not models_dir.is_dir():
            continue
        for cache_parent in models_dir.rglob(".cache"):
            if not cache_parent.is_dir() or cache_parent.name != ".cache":
                continue
            if any(part in _SKIP_DIR_NAMES for part in cache_parent.parts):
                continue
            try:
                next(cache_parent.iterdir())
            except StopIteration:
                cache_parent.rmdir()
                removed.append(cache_parent.resolve())
            except OSError:
                pass
    return removed


def clean_download_caches(
    project_root: Path,
    *,
    dry_run: bool = False,
) -> list[tuple[Path, int]]:
    removed: list[tuple[Path, int]] = []
    targets = _discover_targets(project_root)
    if not targets:
        return removed

    for path in targets:
        size = _dir_size(path)
        if dry_run:
            print(f"Would remove {_format_bytes(size):>8}  {path}")
            removed.append((path, size))
            continue
        shutil.rmtree(path)
        removed.append((path, size))
        print(f"Removed {_format_bytes(size):>8}  {path}")

    if not dry_run:
        for empty_cache in _remove_empty_cache_parents(project_root):
            print(f"Removed empty  {empty_cache}")

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean Hugging Face and ModelScope download caches",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List cache paths and sizes without deleting",
    )
    args = parser.parse_args()

    removed = clean_download_caches(PROJECT_ROOT, dry_run=args.dry_run)
    if not removed:
        print("Nothing to clean.")
        return

    total = sum(size for _, size in removed)
    label = "Would free" if args.dry_run else "Freed"
    print(f"{label} {_format_bytes(total)} from {len(removed)} path(s).")
    print("Kept: installed model weights under models/, ~/.modelscope/credentials")


if __name__ == "__main__":
    main()
