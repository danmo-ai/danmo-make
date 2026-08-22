"""Control plane, workspace, and registry path resolution (no FastAPI dependency)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

DANQING_HOME_DIRNAME = ".danmo-make"
CONTROL_SETTINGS_DIRNAME = "config"
CONTROL_SETTINGS_FILE = ".app_config.json"
BOOTSTRAP_POINTER_FILE = "workspace.pointer.json"
LLM_PID_FILE = "llm.pid"
LLM_PORT_FILE = "llm.port"


def resolve_control_plane_dir() -> Path:
    raw = os.environ.get("DANQING_USER_DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / DANQING_HOME_DIRNAME).expanduser().resolve()


def control_plane_dir() -> Path:
    return resolve_control_plane_dir()


def control_settings_path(control_plane: Path | None = None) -> Path:
    root = (control_plane or resolve_control_plane_dir()).resolve()
    return root / CONTROL_SETTINGS_DIRNAME / CONTROL_SETTINGS_FILE


def llm_pid_file(control_plane: Path | None = None) -> Path:
    return (control_plane or resolve_control_plane_dir()).resolve() / LLM_PID_FILE


def llm_port_file(control_plane: Path | None = None) -> Path:
    return (control_plane or resolve_control_plane_dir()).resolve() / LLM_PORT_FILE


def _read_workspace_pointer(control_plane: Path) -> str:
    path = control_plane / BOOTSTRAP_POINTER_FILE
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


def resolve_install_root(*, project_root: Path | None = None) -> Path:
    """Repo root (dev) or PyInstaller bundle parent."""
    if project_root is not None:
        return Path(project_root).resolve()
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        exe_dir = Path(sys.executable).parent.resolve()
        if (
            sys.platform == "darwin"
            and exe_dir.name == "MacOS"
            and (exe_dir.parent / "Resources").exists()
        ):
            return (exe_dir.parent / "Resources").resolve()
        return exe_dir
    # shared/danqing_config/paths.py → repo root
    return Path(__file__).resolve().parents[2]


def default_media_bootstrap(*, install_root: Path, control_plane: Path) -> Path:
    if os.environ.get("DANQING_USER_DATA_DIR", "").strip() or getattr(sys, "frozen", False):
        return control_plane.resolve()
    return install_root.resolve()


def workspace_root(*, install_root: Path | None = None) -> Path:
    """Effective media workspace (models, config/models_registry.json)."""
    control = resolve_control_plane_dir()
    install = install_root or resolve_install_root()
    bootstrap = default_media_bootstrap(install_root=install, control_plane=control)
    pointer = _read_workspace_pointer(control)
    if pointer:
        candidate = Path(pointer).expanduser()
        if not candidate.is_absolute():
            candidate = (bootstrap / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    return bootstrap.resolve()


def models_registry_path(*, install_root: Path | None = None) -> Path:
    return workspace_root(install_root=install_root) / "config" / "models_registry.json"


def resolve_registry_local_path(local_path: str, *, workspace: Path | None = None) -> Path:
    text = (local_path or "").strip()
    if not text:
        raise ValueError("local_path is required")
    root = (workspace or workspace_root()).resolve()
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if text.startswith("models/"):
        return (root / "models" / text[len("models/") :]).resolve()
    return (root / text).resolve()
