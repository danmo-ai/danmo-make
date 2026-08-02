"""Danmo Make control plane — ``~/.danmo-make`` (aligned with danmo-work ``~/.danmo-work``).

Control plane holds: workspace pointer, ``.app_config.json``, logs, api.pid, runtime-venv.
Media workspace (models / outputs / db / registry) is separate and may equal the control
plane when no custom workspace pointer is set.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

DANQING_HOME_DIRNAME = ".danmo-make"
CONTROL_SETTINGS_DIRNAME = "config"
CONTROL_SETTINGS_FILE = ".app_config.json"

_LEGACY_MEDIA_TOP = ("config", "db", "models", "outputs", "datasets")


def default_control_plane_dir() -> Path:
    return (Path.home() / DANQING_HOME_DIRNAME).expanduser().resolve()


def resolve_control_plane_dir() -> Path:
    """``DANQING_USER_DATA_DIR`` if set, else ``~/.danmo-make``."""
    raw = os.environ.get("DANQING_USER_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_control_plane_dir()


def ensure_control_plane(control_plane: Path | None = None) -> Path:
    root = (control_plane or resolve_control_plane_dir()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / CONTROL_SETTINGS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return root


def control_settings_path(control_plane: Path) -> Path:
    return control_plane.resolve() / CONTROL_SETTINGS_DIRNAME / CONTROL_SETTINGS_FILE


def default_media_bootstrap(*, install_root: Path, control_plane: Path) -> Path:
    """Default media root when no workspace pointer is set.

    Packaged / env-driven runs use the control plane; ``make dev`` uses the repo root
    so ``./models`` keeps working without a pointer.
    """
    if os.environ.get("DANQING_USER_DATA_DIR", "").strip() or getattr(sys, "frozen", False):
        return control_plane.resolve()
    return install_root.resolve()


def migrate_app_settings_to_control_plane(
    control_plane: Path,
    media_root: Path,
) -> None:
    """Prefer control-plane ``.app_config.json``; copy once from media workspace if needed."""
    dst = control_settings_path(control_plane)
    if dst.is_file():
        return
    src = media_root.resolve() / CONTROL_SETTINGS_DIRNAME / CONTROL_SETTINGS_FILE
    if src.is_file() and src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _legacy_data_candidates() -> list[Path]:
    home = Path.home()
    out: list[Path] = [
        home / "danqing-data",
        home
        / "Library"
        / "Application Support"
        / "com.danqing.studio.desktop"
        / "server-data",
        home / ".local" / "share" / "com.danqing.studio.desktop" / "server-data",
    ]
    return out


def migrate_legacy_data_into_control_plane_once(control_plane: Path) -> None:
    """If ``~/.danmo-make`` has no studio DB yet, adopt the newest legacy data tree."""
    control = control_plane.resolve()
    marker = control / "db" / "studio.db"
    if marker.is_file():
        return
    for src in _legacy_data_candidates():
        if not src.is_dir() or src.resolve() == control:
            continue
        legacy_db = src / "db" / "studio.db"
        if not legacy_db.is_file():
            continue
        for name in _LEGACY_MEDIA_TOP:
            s = src / name
            d = control / name
            if not s.exists() or d.exists():
                continue
            shutil.move(str(s), str(d))
        for name in ("logs", "runtime-venv", "runtime-env.json", "api.pid", "api.port"):
            s = src / name
            d = control / name
            if not s.exists() or d.exists():
                continue
            if s.is_dir():
                shutil.move(str(s), str(d))
            else:
                shutil.move(str(s), str(d))
        break
