"""
Danmo Make - Interface definition layer
Interface-oriented programming; all layers depend only on the interfaces defined in this module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ModelInfo:
    """Model info"""
    name: str
    path: str
    type: str = "flux"  # flux, lora, etc.
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DownloadTask:
    """Download task"""
    id: str
    url: str
    target_path: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    total_size: int = 0
    downloaded_size: int = 0
    error_message: str = ""


@dataclass
class ConversionTask:
    """Model conversion task (generate quantized version)."""
    id: str
    model_name: str
    from_version: str
    to_version: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    stage: str = ""  # loading, quantizing, saving, completed, cancelled, error
    error_message: str = ""
    output_path: str = ""


@dataclass
class AppSettings:
    """App settings"""
    language: str = "zh"
    theme: str = "dark"
    default_model: str = ""
    default_model_image: str = ""
    default_model_video: str = ""
    default_model_audio: str = ""
    default_model_llm: str = "qwen3.5-4b"
    default_model_vlm: str = "qwen3-vl-4b-instruct"
    default_model_llm_think: bool = False
    auto_save_prompts: bool = True
    output_format: str = "png"
    civitai_token: str = ""
    nsfw_enabled: bool = False
    huggingface_token: str = ""
    # Remote access (non-loopback): separate keys for REST vs MCP. Loopback skips auth.
    # Stored as v1:<salt>:<hmac> (plaintext returned once at create). Env overrides:
    # DANQING_HTTP_API_KEY / DANQING_MCP_API_KEY (plaintext).
    http_api_key: str = ""
    http_api_key_hint: str = ""
    mcp_api_key: str = ""
    mcp_api_key_hint: str = ""
    mlx_memory_limit: int = 120
    model_cache_ttl_minutes: int = 30
    queue_image_first: bool = False
    quick_setup_completed: bool = False
    custom_workspace_dir: str = ""


# ===== Task persistence interfaces =====

class IV3TaskStore(ABC):
    """v3 task persistence (``studio.db``: JSON params, dict-row views)."""

    @abstractmethod
    def insert_task(
        self,
        task_id: str,
        kind: str,
        model_id: str,
        params: Dict[str, Any],
        *,
        priority: int = 100,
        status: TaskStatus = TaskStatus.QUEUED,
    ) -> None:
        ...

    @abstractmethod
    def update_status(self, task_id: str, status: TaskStatus) -> None:
        ...

    @abstractmethod
    def update_progress(self, task_id: str, progress: float) -> None:
        ...

    @abstractmethod
    def update_task_priority(self, task_id: str, priority: int) -> bool:
        ...

    @abstractmethod
    def mark_running(self, task_id: str) -> None:
        ...

    @abstractmethod
    def mark_completed(self, task_id: str, result: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def mark_failed(self, task_id: str, message: str) -> None:
        ...

    @abstractmethod
    def mark_cancelled(self, task_id: str) -> None:
        ...

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def list_tasks(
        self,
        limit: int = 200,
        offset: int = 0,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def append_log(self, task_id: str, message: str, level: str = "info") -> None:
        ...

    @abstractmethod
    def get_logs(self, task_id: str, offset: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        ...


# ===== Engine layer interfaces =====
# v3 media engines see backend.core.media_interfaces (IImageEngine / IVideoEngine)

# ===== Persistence layer interfaces =====

class IConfigStore(ABC):
    """Config persistence interface"""

    @abstractmethod
    def load(self) -> AppSettings:
        pass

    @abstractmethod
    def save(self, settings: AppSettings) -> None:
        pass


class IPresetStore(ABC):
    """Preset persistence interface"""

    @abstractmethod
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def save(self, name: str, preset: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass


@dataclass
class DownloadProgress:
    """Download progress info"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    progress: float  # 0.0 - 1.0
    total_size: int = 0
    downloaded_size: int = 0
    speed: str = ""  # e.g. "12.5 MB/s"
    error_message: str = ""
    filename: str = ""


class IDownloadService(ABC):
    """Download service interface - supports HuggingFace, ModelScope, and HTTP multi-source downloads."""

    @abstractmethod
    async def download_model(self, model_name: str,
                            progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                            existing_task_id: Optional[str] = None) -> str:
        """Download base model by registry model name.

        Automatically selects downloader based on the source field in models_registry.json:
        - huggingface: uses huggingface_hub download
        - modelscope: uses modelscope download
        - civitai/http: uses aiohttp download
        """
        pass

    @abstractmethod
    async def download_lora(self, url: str, filename: str,
                           progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
                           existing_task_id: Optional[str] = None) -> str:
        """Download LoRA (generic HTTP)."""
        pass

    @abstractmethod
    def list_downloads(self) -> List[DownloadTask]:
        """List all download tasks."""
        pass

    @abstractmethod
    async def cancel_download(self, task_id: str) -> bool:
        """Cancel a download task."""
        pass

    @abstractmethod
    def delete_download(self, task_id: str) -> bool:
        """Delete a download task."""
        pass

    @abstractmethod
    def get_progress(self, task_id: str) -> Optional[DownloadProgress]:
        """Get progress of a single download task."""
        pass

    @abstractmethod
    async def resume_download(self, task_id: str,
                             progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> str:
        """Resume a download task (after process restart)."""
        pass

    @abstractmethod
    def get_model_download_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get download configuration info for a model."""
        pass

    @abstractmethod
    async def convert_model(self, model_name: str, from_version: str, to_version: str,
                           progress_callback: Optional[Callable[[ConversionTask], None]] = None) -> str:
        """Produce a quantized derived layout (``to_version`` containing ``int4`` or ``int8``).

        Args:
            model_name: model key
            from_version: source version key (e.g. ``fp16``, ``bf16``)
            to_version: target version key (e.g. ``int4``, ``int8``)
            progress_callback: progress callback function

        Returns:
            output directory path
        """
        pass

    @abstractmethod
    def list_conversions(self) -> List[ConversionTask]:
        """List all conversion tasks."""
        pass

    @abstractmethod
    async def cancel_conversion(self, task_id: str) -> bool:
        """Cancel a conversion task."""
        pass

    @abstractmethod
    def get_conversion_progress(self, task_id: str) -> Optional[ConversionTask]:
        """Get progress of a single conversion task."""
        pass

    @abstractmethod
    async def delete_model(self, model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """Delete a model or specified version.

        Args:
            model_name: model name (key in the registry)
            version: optional, the version key to delete; if not provided, delete the entire model

        Returns:
            {"success": bool, "deleted_paths": List[str], "error": Optional[str]}
        """
        pass


@dataclass
class ModelConfig:
    """Single model view from models_registry; actions are v2 verb blocks (create / rewrite / ...)."""

    engine: str
    type: str
    family: str = ""
    name: Dict[str, str] = field(default_factory=dict)
    description: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    recommended: bool = False
    category: str = "base_models"
    dependencies: List[str] = field(default_factory=list)
    source: str = "huggingface"
    download_url: Optional[str] = None
    files: Optional[List[str]] = None
    versions: Optional[Dict[str, Any]] = None
    negative_prompt_support: bool = False
    base_model: Optional[str] = None
    nsfw: bool = False
    commercial_use_allowed: Optional[bool] = None
    media: str = "image"
    actions: Dict[str, Any] = field(default_factory=dict)
    stub_no_download: bool = False


class ISettingsService(ABC):
    """Settings service interface"""

    @abstractmethod
    def get_settings(self) -> AppSettings:
        pass

    @abstractmethod
    def update_settings(self, settings: AppSettings) -> None:
        pass

    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        pass

    @abstractmethod
    def refresh_models(self) -> None:
        pass

    @abstractmethod
    def install_environment(self) -> bool:
        pass

    @abstractmethod
    def check_environment(self) -> bool:
        pass

    @abstractmethod
    def get_model_registry(self) -> Dict[str, ModelConfig]:
        """Get model registry configuration."""
        pass

    @abstractmethod
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a single model."""
        pass


# ===== Utility interfaces =====

class IPathResolver(ABC):
    """Path resolver interface"""

    @abstractmethod
    def get_models_dir(self) -> Path:
        pass

    @abstractmethod
    def get_loras_dir(self) -> Path:
        pass

    @abstractmethod
    def get_outputs_dir(self) -> Path:
        pass

    @abstractmethod
    def get_venv_python(self) -> Path:
        pass

    @abstractmethod
    def get_project_root(self) -> Path:
        """Effective media workspace root (models / outputs / db / registry)."""
        pass

    @abstractmethod
    def get_control_plane_dir(self) -> Path:
        """App home (``~/.danmo-make``): pointer, ``.app_config.json``, logs, runtime-venv."""
        pass

    @abstractmethod
    def get_default_config_root(self) -> Path:
        pass

    @abstractmethod
    def get_workspace_config_dir(self) -> Path:
        pass

    @abstractmethod
    def get_models_registry_path(self) -> Path:
        pass

    @abstractmethod
    def get_locales_dir(self) -> Path:
        pass

    @abstractmethod
    def restore_config_defaults(self, *, names: tuple[str, ...] | None = None) -> list[str]:
        pass

    @abstractmethod
    def resolve_registry_local_path(self, local_path: str) -> Path:
        """Resolve registry ``local_path`` (e.g. ``models/Image/foo``) under the effective workspace."""
        pass
