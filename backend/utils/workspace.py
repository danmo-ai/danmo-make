"""Media workspace root resolution and layout (models / outputs / db / registry config).

Control plane (``~/.danmo-make``): pointer, app settings, logs, runtime-venv — see
``backend.utils.user_home``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from backend.utils.config_paths import (
    read_workspace_pointer,
    resolve_default_config_root,
    seed_workspace_config_from_defaults,
    write_workspace_pointer,
)
from backend.utils.user_home import (
    ensure_control_plane,
    migrate_app_settings_to_control_plane,
    migrate_legacy_data_into_control_plane_once,
    resolve_control_plane_dir,
)

_WORKSPACE_SUBDIRS = (
    "config",
    "db",
    "models",
    "models/Lora",
    "outputs",
    "outputs/assets",
)

_WORKSPACE_TOP_LEVEL = ("config", "db", "models", "outputs", "datasets")

_MEDIA_PRUNE_NAMES = ("db", "models", "outputs", "datasets")

_IGNORE_EMPTY_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def _resolve_default_config(
    bootstrap_root: Path,
    default_config_root: Path | None,
) -> Path:
    if default_config_root is not None:
        return default_config_root.resolve()
    return resolve_default_config_root(bootstrap_root=bootstrap_root.resolve(), bundle_root=None)


def sanitize_workspace_pointer(
    control_plane: Path,
    *,
    media_bootstrap: Path,
    legacy_default_config: Path | None = None,
) -> None:
    """Drop invalid pointers (missing dir)."""
    raw = read_workspace_pointer(
        control_plane, legacy_default_config=legacy_default_config
    )
    if not raw:
        return
    try:
        candidate = normalize_workspace_path(media_bootstrap, raw)
    except ValueError:
        write_workspace_pointer(control_plane, "")
        return
    if not candidate.is_dir():
        write_workspace_pointer(control_plane, "")


def is_workspace_configured(
    control_plane: Path,
    *,
    legacy_default_config: Path | None = None,
) -> bool:
    """True when control-plane pointer names a workspace directory."""
    return bool(
        read_workspace_pointer(
            control_plane, legacy_default_config=legacy_default_config
        ).strip()
    )


def normalize_workspace_path(bootstrap_root: Path, raw: str) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("workspace path is required")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (bootstrap_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def is_empty_directory(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        raise RuntimeError(f"Not a directory: {path}")
    for entry in path.iterdir():
        if entry.name in _IGNORE_EMPTY_NAMES:
            continue
        return False
    return True


def _assert_workspace_paths_safe(old_root: Path, new_root: Path) -> None:
    old_r = old_root.resolve()
    new_r = new_root.resolve()
    if old_r == new_r:
        return
    try:
        if new_r.is_relative_to(old_r) or old_r.is_relative_to(new_r):
            raise RuntimeError(
                f"Workspace path must not be inside the current workspace (or vice versa): {new_r}"
            )
    except ValueError:
        pass


def migrate_workspace_data(old_root: Path, new_root: Path) -> None:
    """Move media workspace directories from old_root into an empty new_root."""
    old_r = old_root.resolve()
    new_r = new_root.resolve()
    if old_r == new_r:
        return
    _assert_workspace_paths_safe(old_r, new_r)
    if not is_empty_directory(new_r):
        raise RuntimeError(f"Target workspace directory is not empty: {new_r}")

    new_r.mkdir(parents=True, exist_ok=True)
    for name in _WORKSPACE_TOP_LEVEL:
        src = old_r / name
        dst = new_r / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        elif name == "models":
            (new_r / "models" / "Lora").mkdir(parents=True, exist_ok=True)
        elif name == "datasets":
            continue
        else:
            (new_r / name).mkdir(parents=True, exist_ok=True)


def apply_workspace_relocation(
    *,
    media_bootstrap: Path,
    control_plane: Path,
    default_config_root: Path,
    old_root: Path,
    new_path_raw: str,
) -> Path:
    """Validate empty target, migrate media data, prepare layout; returns new media root."""
    new_root = normalize_workspace_path(media_bootstrap, new_path_raw)
    if new_root.resolve() == old_root.resolve():
        return new_root
    if not is_empty_directory(new_root):
        raise RuntimeError(f"Target workspace directory is not empty: {new_root}")
    migrate_workspace_data(old_root, new_root)
    ensure_workspace_layout(new_root)
    seed_workspace_config_from_defaults(default_config_root, new_root)
    write_workspace_pointer(control_plane, str(new_root))
    migrate_app_settings_to_control_plane(control_plane, new_root)
    db_path = new_root / "db" / "studio.db"
    if db_path.is_file():
        from backend.persistence.asset_store import repair_asset_paths_in_database

        repair_asset_paths_in_database(
            db_path,
            new_root / "outputs" / "assets",
            former_workspace_roots=[old_root],
        )
    return new_root


def resolve_workspace_root(
    media_bootstrap: Path,
    *,
    control_plane: Path,
    legacy_default_config: Path | None = None,
) -> Path:
    """Effective media root: custom workspace if configured, else media bootstrap."""
    raw = read_workspace_pointer(
        control_plane, legacy_default_config=legacy_default_config
    )
    if not raw:
        return media_bootstrap.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (media_bootstrap / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise RuntimeError(
            f"Configured custom_workspace_dir does not exist or is not a directory: {candidate}"
        )
    return candidate


def _tree_has_user_files(path: Path) -> bool:
    if not path.exists():
        return False
    for entry in path.rglob("*"):
        if entry.is_file() and entry.name not in _IGNORE_EMPTY_NAMES:
            return True
    return False


def prune_obsolete_bootstrap_data_dirs(
    media_bootstrap: Path,
    *,
    control_plane: Path,
    legacy_default_config: Path | None = None,
) -> None:
    """Remove empty legacy media dirs under bootstrap after workspace migration.

    Never deletes control-plane ``config/`` (app settings live there).
    """
    bootstrap = media_bootstrap.resolve()
    control = control_plane.resolve()
    if not is_workspace_configured(
        control, legacy_default_config=legacy_default_config
    ):
        return
    workspace = resolve_workspace_root(
        bootstrap,
        control_plane=control,
        legacy_default_config=legacy_default_config,
    )
    if workspace == bootstrap:
        return
    for name in _MEDIA_PRUNE_NAMES:
        path = bootstrap / name
        if path.exists() and not _tree_has_user_files(path):
            shutil.rmtree(path)
    # Media registry config may remain under bootstrap when it equals control plane —
    # only remove if it no longer holds user files (app settings already migrated).
    legacy_cfg = bootstrap / "config"
    if legacy_cfg.is_dir() and bootstrap != control and not _tree_has_user_files(legacy_cfg):
        shutil.rmtree(legacy_cfg)


def prepare_data_directories(
    media_bootstrap: Path,
    *,
    control_plane: Path | None = None,
    default_config_root: Path | None = None,
) -> Path:
    """Create control-plane + media layout; return effective media workspace root."""
    control = ensure_control_plane(control_plane or resolve_control_plane_dir())
    migrate_legacy_data_into_control_plane_once(control)
    bootstrap = media_bootstrap.resolve()
    default_cfg = _resolve_default_config(bootstrap, default_config_root)
    sanitize_workspace_pointer(
        control,
        media_bootstrap=bootstrap,
        legacy_default_config=default_cfg,
    )
    root = resolve_workspace_root(
        bootstrap,
        control_plane=control,
        legacy_default_config=default_cfg,
    )
    ensure_workspace_layout(root)
    seed_workspace_config_from_defaults(default_cfg, root)
    migrate_app_settings_to_control_plane(control, root)
    prune_obsolete_bootstrap_data_dirs(
        bootstrap,
        control_plane=control,
        legacy_default_config=default_cfg,
    )
    return root


def ensure_workspace_layout(workspace_root: Path) -> None:
    for rel in _WORKSPACE_SUBDIRS:
        (workspace_root / rel).mkdir(parents=True, exist_ok=True)


def workspace_layout_paths(
    workspace_root: Path,
    *,
    control_plane: Path | None = None,
) -> dict[str, str]:
    control = (control_plane or resolve_control_plane_dir()).resolve()
    return {
        "control_plane": str(control),
        "workspace": str(workspace_root),
        "config": str(workspace_root / "config"),
        "db": str(workspace_root / "db"),
        "models": str(workspace_root / "models"),
        "outputs": str(workspace_root / "outputs"),
    }


def pick_directory_native(*, prompt: str) -> str:
    """macOS folder picker via AppleScript; fail loud on other platforms."""
    if sys.platform != "darwin":
        raise RuntimeError("Directory picker is only supported on macOS.")
    safe_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
    script = f'POSIX path of (choose folder with prompt "{safe_prompt}")'
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("Directory picker was cancelled or failed.")
    path = (proc.stdout or "").strip()
    if not path:
        raise RuntimeError("Directory picker returned an empty path.")
    return path
