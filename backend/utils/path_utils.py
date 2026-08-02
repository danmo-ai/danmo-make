"""
路径解析器和系统工具
"""

import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

from backend.core.interfaces import IPathResolver
from backend.utils.config_paths import (
    RESTORABLE_CONFIG_FILES,
    resolve_default_config_root,
    restore_workspace_config_from_defaults,
)
from backend.utils.user_home import (
    control_settings_path,
    default_media_bootstrap,
    ensure_control_plane,
    resolve_control_plane_dir,
)
from backend.utils.workspace import prepare_data_directories


class PathResolver(IPathResolver):
    """路径解析器实现

    - Control plane (``~/.danmo-make``): pointer, ``.app_config.json``, logs, runtime-venv
    - Media workspace: config/ (registry), db/, models/, outputs/
    """

    def __init__(self, project_root: Optional[Path] = None):
        bundle_root: Path | None = None
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            bundle_root = Path(sys._MEIPASS).resolve()
            exe_dir = Path(sys.executable).parent.resolve()
            if project_root is not None:
                install_root = Path(project_root).resolve()
            elif (
                sys.platform == "darwin"
                and exe_dir.name == "MacOS"
                and (exe_dir.parent / "Resources").exists()
            ):
                install_root = exe_dir.parent / "Resources"
            else:
                install_root = exe_dir
        elif project_root is None:
            # backend/utils/path_utils.py → repo root
            install_root = Path(__file__).resolve().parents[2]
        else:
            install_root = Path(project_root).resolve()

        control = ensure_control_plane(resolve_control_plane_dir())
        media_bootstrap = default_media_bootstrap(
            install_root=install_root, control_plane=control
        )
        # Factory defaults: prefer bundle, else install/repo default_config/
        factory_bootstrap = install_root
        if bundle_root is None and not (install_root / "default_config").is_dir():
            factory_bootstrap = media_bootstrap

        self._bundle_root = bundle_root
        self._install_root = install_root
        self._control_plane = control
        self._bootstrap = media_bootstrap
        self._default_config = resolve_default_config_root(
            bootstrap_root=factory_bootstrap,
            bundle_root=self._bundle_root,
        )
        self._root = prepare_data_directories(
            self._bootstrap,
            control_plane=self._control_plane,
            default_config_root=self._default_config,
        )

    def get_control_plane_dir(self) -> Path:
        return self._control_plane

    def get_bootstrap_root(self) -> Path:
        """Default media root when no custom workspace pointer is set."""
        return self._bootstrap

    def get_default_config_root(self) -> Path:
        return self._default_config

    def get_project_root(self) -> Path:
        """Effective media workspace root."""
        return self._root

    def get_workspace_config_dir(self) -> Path:
        return self._root / "config"

    def get_models_registry_path(self) -> Path:
        return self.get_workspace_config_dir() / "models_registry.json"

    def get_presets_path(self) -> Path:
        return self.get_workspace_config_dir() / "presets.json"

    def get_locales_dir(self) -> Path:
        return self._default_config / "locales"

    def get_models_dir(self) -> Path:
        return self._root / "models"

    def get_loras_dir(self) -> Path:
        return self._root / "models" / "Lora"

    def get_outputs_dir(self) -> Path:
        return self._root / "outputs"

    def get_venv_python(self) -> Path:
        return self._root / ".venv" / "bin" / "python"

    def get_config_path(self) -> Path:
        """App settings live on the control plane (not media workspace)."""
        return control_settings_path(self._control_plane)

    def restore_config_defaults(
        self, *, names: tuple[str, ...] | None = None
    ) -> list[str]:
        """Restore workspace config files from ``default_config/`` (models_registry, presets)."""
        return restore_workspace_config_from_defaults(
            self._root,
            self._default_config,
            names=names if names is not None else RESTORABLE_CONFIG_FILES,
        )

    def resolve_registry_local_path(self, local_path: str) -> Path:
        """Resolve registry paths against the effective workspace (never the repo bootstrap root)."""
        text = (local_path or "").strip()
        if not text:
            raise ValueError("local_path is required")
        candidate = Path(text).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        if text.startswith("models/"):
            return (self.get_models_dir() / text[len("models/") :]).resolve()
        return (self.get_project_root() / text).resolve()

def get_system_info() -> dict:
    """获取系统信息"""
    info = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "memory_gb": get_memory_gb(),
        "dependencies": {},
    }
    used_gb, available_gb = get_memory_usage_gb()
    if used_gb > 0:
        info["memory_used_gb"] = used_gb
    if available_gb > 0:
        info["memory_available_gb"] = available_gb

    deps = {
        "mlx": "mlx.core",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "Pillow": "PIL",
        "huggingface_hub": "huggingface_hub",
    }
    for dep_name, import_name in deps.items():
        try:
            import importlib

            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "unknown")
            if version == "unknown" and import_name == "mlx.core":
                import mlx.core

                version = getattr(mlx.core, "__version__", "unknown")
            info["dependencies"][dep_name] = version
        except Exception:
            info["dependencies"][dep_name] = "not installed"

        if info["dependencies"][dep_name] == "unknown":
            try:
                from importlib.metadata import version

                info["dependencies"][dep_name] = version(dep_name)
            except Exception:
                pass

    return info


def _darwin_memsize_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip()) / (1024**3)
    except Exception:
        return 0.0


def get_memory_gb() -> float:
    """获取系统内存大小（GB）"""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        if platform.system() == "Darwin":
            return _darwin_memsize_gb()
        return 0.0
    except (RuntimeError, OSError):
        # psutil on macOS 15+/26+ can fail: host_statistics64(HOST_VM_INFO64) …
        if platform.system() == "Darwin":
            return _darwin_memsize_gb()
        return 0.0


def get_memory_usage_gb() -> tuple[float, float]:
    """Return (used_gb, available_gb) from psutil; (0, 0) when unavailable."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        used = round(mem.used / (1024**3), 1)
        available = round(getattr(mem, "available", mem.free) / (1024**3), 1)
        return used, available
    except Exception:
        return 0.0, 0.0


def is_apple_silicon() -> bool:
    """检查是否为 Apple Silicon"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"
