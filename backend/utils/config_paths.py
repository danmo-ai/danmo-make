"""Shipped defaults (``default_config/``) vs workspace ``config/`` vs control plane.

- Factory templates: install/bundle ``default_config/`` (models_registry, presets, locales).
- Workspace pointer: control plane ``~/.danmo-make/workspace.pointer.json`` (writable).
- First workspace setup: ``seed_workspace_config_from_defaults`` copies registry/presets if missing.
- Later updates: user edits workspace ``config/models_registry.json``; no startup merge.
- Reset: settings ``restore_config_defaults`` overwrites from ``default_config/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DEFAULT_CONFIG_DIRNAME = "default_config"
BOOTSTRAP_POINTER_FILE = "workspace.pointer.json"
WORKSPACE_SETTINGS_FILE = ".app_config.json"
RESTORABLE_CONFIG_FILES = ("models_registry.json", "presets.json")


def resolve_default_config_root(*, bootstrap_root: Path, bundle_root: Path | None) -> Path:
    """Read-only factory defaults: locales, models_registry, presets."""
    if bundle_root is not None:
        bundled = bundle_root / DEFAULT_CONFIG_DIRNAME
        if bundled.is_dir():
            return bundled.resolve()
    candidate = bootstrap_root / DEFAULT_CONFIG_DIRNAME
    if candidate.is_dir():
        return candidate.resolve()
    raise RuntimeError(
        f"Missing {DEFAULT_CONFIG_DIRNAME}/ (expected under install root or PyInstaller bundle)"
    )


def workspace_pointer_path(control_plane: Path) -> Path:
    return control_plane.resolve() / BOOTSTRAP_POINTER_FILE


def _read_pointer_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return (data.get("custom_workspace_dir") or "").strip()
    except Exception:
        return ""
    return ""


def read_workspace_pointer(
    control_plane: Path,
    *,
    legacy_default_config: Path | None = None,
) -> str:
    """Workspace path from control-plane pointer; migrate legacy ``default_config/`` once."""
    path = workspace_pointer_path(control_plane)
    raw = _read_pointer_file(path)
    if raw:
        return raw
    if legacy_default_config is not None:
        legacy = legacy_default_config.resolve() / BOOTSTRAP_POINTER_FILE
        legacy_raw = _read_pointer_file(legacy)
        if legacy_raw:
            write_workspace_pointer(control_plane, legacy_raw)
            return legacy_raw
    return ""


def write_workspace_pointer(control_plane: Path, workspace_dir: str) -> None:
    """Persist custom workspace path under the control plane."""
    control_plane.mkdir(parents=True, exist_ok=True)
    path = workspace_pointer_path(control_plane)
    payload = {"custom_workspace_dir": (workspace_dir or "").strip()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def seed_workspace_config_from_defaults(default_root: Path, workspace_root: Path) -> None:
    """Copy factory registry/presets into workspace ``config/`` if missing."""
    dst_cfg = workspace_root / "config"
    dst_cfg.mkdir(parents=True, exist_ok=True)
    for name in RESTORABLE_CONFIG_FILES:
        src = default_root / name
        dst = dst_cfg / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def restore_workspace_config_from_defaults(
    workspace_root: Path,
    default_root: Path,
    *,
    names: tuple[str, ...] = RESTORABLE_CONFIG_FILES,
) -> list[str]:
    """Overwrite workspace config files from factory defaults."""
    restored: list[str] = []
    dst_cfg = workspace_root / "config"
    dst_cfg.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = default_root / name
        if not src.is_file():
            continue
        shutil.copy2(src, dst_cfg / name)
        restored.append(name)
    return restored
