"""
API routes - settings
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.api.deps import get_model_registry as get_typed_model_registry
from backend.core.container import get_container
from backend.core.interfaces import ISettingsService, AppSettings, ModelConfig, IPresetStore, IPathResolver
from backend.core.i18n import t, resolve_locale
from backend.engine.llm.service_mlx import normalize_app_llm_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class ModelRegistryResponse(BaseModel):
    models: dict
    engines: dict


class SettingsResponse(BaseModel):
    language: str
    theme: str
    default_model: str
    default_model_image: str = ""
    default_model_video: str = ""
    default_model_audio: str = ""
    default_model_llm: str = "qwen3.5-4b"
    default_model_vlm: str = "qwen3-vl-4b-instruct"
    default_model_llm_think: bool = False
    auto_save_prompts: bool
    output_format: str
    mlx_memory_limit: int
    model_cache_ttl_minutes: int = 30
    queue_image_first: bool = False
    quick_setup_completed: bool = False
    civitai_token: str = ""
    huggingface_token: str = ""
    nsfw_enabled: bool = False
    custom_workspace_dir: str = ""


class ApplyWorkspaceRequest(BaseModel):
    path: str


class RestoreConfigDefaultsRequest(BaseModel):
  """Restore workspace copies from ``default_config/`` (factory models_registry / presets)."""

  files: Optional[List[str]] = None


class SettingsUpdateRequest(BaseModel):
    language: Optional[str] = None
    theme: Optional[str] = None
    default_model: Optional[str] = None
    default_model_image: Optional[str] = None
    default_model_video: Optional[str] = None
    default_model_audio: Optional[str] = None
    default_model_llm: Optional[str] = None
    default_model_vlm: Optional[str] = None
    default_model_llm_think: Optional[bool] = None
    auto_save_prompts: Optional[bool] = None
    output_format: Optional[str] = None
    mlx_memory_limit: Optional[int] = None
    model_cache_ttl_minutes: Optional[int] = None
    queue_image_first: Optional[bool] = None
    quick_setup_completed: Optional[bool] = None
    civitai_token: Optional[str] = None
    huggingface_token: Optional[str] = None
    nsfw_enabled: Optional[bool] = None
    custom_workspace_dir: Optional[str] = None


class ModelResponse(BaseModel):
    name: str
    path: str
    type: str
    size: int
    size_human: str


class SystemInfoResponse(BaseModel):
    """GET /system: includes mlx_memory_limit (GB, consistent with AppSettings) for frontend Plan E4 pre-submit soft hint."""

    platform: str
    architecture: str
    processor: str
    python_version: str
    memory_gb: float
    memory_used_gb: Optional[float] = None
    memory_available_gb: Optional[float] = None
    mlx_active_gb: Optional[float] = None
    mlx_peak_gb: Optional[float] = None
    env_ready: bool
    dependencies: Optional[Dict[str, str]] = None
    mlx_memory_limit: int = 120
    controlnet_runtime_available: bool = False


def get_settings_service():
    return get_container().resolve(ISettingsService)


@router.get("", response_model=SettingsResponse)
def get_settings():
    """Get settings"""
    service = get_settings_service()
    settings = service.get_settings()
    registry = get_typed_model_registry()
    if normalize_app_llm_settings(settings, registry):
        service.update_settings(settings)
    return SettingsResponse(**settings.__dict__)


@router.put("")
def update_settings(request: SettingsUpdateRequest, req: Request):
    """Update settings"""
    locale = resolve_locale(req.headers.get("accept-language"))
    service = get_settings_service()
    settings = service.get_settings()
    payload = request.model_dump(exclude_unset=True)

    if "custom_workspace_dir" in payload:
        new_ws = (payload.pop("custom_workspace_dir") or "").strip()
        cur_ws = (settings.custom_workspace_dir or "").strip()
        if new_ws != cur_ws:
            raise HTTPException(
                status_code=400,
                detail=t("error.workspace_use_apply_endpoint", locale),
            )

    if "quick_setup_completed" in payload:
        payload.pop("quick_setup_completed")

    for key, value in payload.items():
        if value is not None:
            setattr(settings, key, value)

    registry = get_typed_model_registry()
    normalize_app_llm_settings(settings, registry)

    service.update_settings(settings)

    memory_keys = {"mlx_memory_limit", "model_cache_ttl_minutes"}
    if memory_keys.intersection(payload.keys()):
        from backend.engine.memory_policy import (
            apply_memory_settings_from_container,
            unload_model_cache_if_present,
        )

        apply_memory_settings_from_container(settings)
        unload_model_cache_if_present()

    llm_keys = {"default_model_llm", "default_model_vlm", "default_model_llm_think"}
    if llm_keys.intersection(payload.keys()):
        from backend.core.container import get_container
        from backend.engine.llm import LLMService

        llm_service = get_container().resolve(LLMService)
        llm_service.apply_model_settings(
            default_model_id=settings.default_model_llm,
            vision_model_id=settings.default_model_vlm,
            llm_think_enabled=settings.default_model_llm_think,
        )

    return {"success": True, "restart_required": False}


@router.get("/workspace-status")
def get_workspace_status():
    """Whether a custom workspace was chosen and the effective data root."""
    path_resolver = get_container().resolve(IPathResolver)
    bootstrap = path_resolver.get_bootstrap_root()
    from backend.utils.workspace import is_workspace_configured

    return {
        "configured": is_workspace_configured(path_resolver.get_default_config_root()),
        "effective_root": str(path_resolver.get_project_root()),
        "bootstrap_root": str(bootstrap),
    }


@router.post("/apply-workspace")
def apply_workspace(request: ApplyWorkspaceRequest, req: Request):
    """Move workspace data to a new empty directory and point settings at it."""
    locale = resolve_locale(req.headers.get("accept-language"))
    path_resolver = get_container().resolve(IPathResolver)
    service = get_settings_service()
    bootstrap = path_resolver.get_bootstrap_root()
    old_root = path_resolver.get_project_root()

    from backend.utils.workspace import apply_workspace_relocation

    raw = (request.path or "").strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail=t("error.workspace_path_required", locale),
        )

    try:
        new_root = apply_workspace_relocation(
            bootstrap_root=bootstrap,
            default_config_root=path_resolver.get_default_config_root(),
            old_root=old_root,
            new_path_raw=raw,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        msg = str(e)
        if "not empty" in msg.lower():
            raise HTTPException(
                status_code=400,
                detail=t("error.workspace_not_empty", locale, path=raw),
            ) from e
        raise HTTPException(status_code=400, detail=msg) from e

    settings = service.get_settings()
    settings.custom_workspace_dir = str(new_root)
    service.update_settings(settings)

    restart_required = new_root.resolve() != old_root.resolve()
    return {
        "success": True,
        "restart_required": restart_required,
        "workspace": str(new_root),
    }


@router.post("/restore-config-defaults")
def restore_config_defaults(
    request: RestoreConfigDefaultsRequest | None = None,
):
    """Overwrite workspace ``config/models_registry.json`` and/or ``presets.json`` from factory defaults."""
    path_resolver = get_container().resolve(IPathResolver)
    names: tuple[str, ...] | None = None
    if request and request.files:
        names = tuple(request.files)
    restored = path_resolver.restore_config_defaults(names=names)
    return {
        "success": True,
        "restored": restored,
        "restart_required": "models_registry.json" in restored,
    }


@router.get("/workspace-paths")
def get_workspace_paths():
    """Resolved data directories under the effective workspace root."""
    path_resolver = get_container().resolve(IPathResolver)
    from backend.utils.workspace import workspace_layout_paths

    return workspace_layout_paths(path_resolver.get_project_root())


@router.post("/pick-workspace-directory")
def pick_workspace_directory(request: Request):
    """Open a native folder picker (macOS)."""
    locale = resolve_locale(request.headers.get("accept-language"))
    from backend.utils.workspace import pick_directory_native

    try:
        path = pick_directory_native(
            prompt=t("settings.pickWorkspacePrompt", locale),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"path": path}


@router.get("/models", response_model=List[ModelResponse])
def list_models():
    """List available models"""
    service = get_settings_service()
    models = service.get_available_models()
    return [
        ModelResponse(
            name=m.name,
            path=m.path,
            type=m.type,
            size=m.size,
            size_human=_human_readable_size(m.size)
        )
        for m in models
    ]


@router.get("/loras", response_model=List[ModelResponse])
def list_loras():
    """List available LoRA models"""
    service = get_settings_service()
    models = service.get_available_models()
    loras = [m for m in models if m.type == "lora"]
    return [
        ModelResponse(
            name=m.name,
            path=m.path,
            type=m.type,
            size=m.size,
            size_human=_human_readable_size(m.size)
        )
        for m in loras
    ]


@router.post("/refresh")
def refresh_models():
    """Refresh model list"""
    service = get_settings_service()
    service.refresh_models()
    return {"success": True}


@router.get("/system", response_model=SystemInfoResponse)
def get_system_info():
    """Get system info"""
    service = get_settings_service()
    info = service.get_system_info()
    info["env_ready"] = service.check_environment()
    settings = service.get_settings()
    info["mlx_memory_limit"] = int(getattr(settings, "mlx_memory_limit", 120) or 120)
    try:
        from backend.engine.platform import PlatformInfo

        mlx = PlatformInfo.get_mlx_memory_stats()
        if mlx.get("active_gb") is not None:
            info["mlx_active_gb"] = mlx["active_gb"]
        if mlx.get("peak_gb") is not None:
            info["mlx_peak_gb"] = mlx["peak_gb"]
    except Exception:
        pass
    from backend.engine.families.flux1.structural import controlnet_runtime_available

    info["controlnet_runtime_available"] = controlnet_runtime_available()
    return SystemInfoResponse(**info)


@router.post("/install")
def install_environment():
    """Install environment"""
    service = get_settings_service()
    success = service.install_environment()
    return {"success": success}


@router.get("/registry")
def get_settings_model_registry():
    """Get model registry"""
    service = get_settings_service()
    registry = service.get_model_registry()
    # Also return readiness status for each model
    status = service.get_models_status()
    return {
        "models": {
            key: {
                "name": val.name,
                "description": val.description,
                "engine": val.engine,
                "type": val.type,
                "category": val.category,
                "media": getattr(val, "media", "image"),
                "actions": getattr(val, "actions", None) or {},
                "parameters": val.parameters,
                "recommended": val.recommended,
                "dependencies": val.dependencies,
                "base_model": val.base_model,
                "nsfw": val.nsfw,
                "commercial_use_allowed": getattr(val, "commercial_use_allowed", None),
                "ready": status.get(key, False),
                "source": val.source,
                "versions": val.versions,
            }
            for key, val in registry.items()
        }
    }


@router.get("/models/status")
def get_models_status():
    """Get readiness status for all models (simplified)"""
    service = get_settings_service()
    status = service.get_models_status()
    return status


@router.get("/models/status/detailed")
def get_models_detailed_status():
    """Get detailed model status (distinguishes not_downloaded/incomplete/ready)"""
    service = get_settings_service()
    return service.get_models_detailed_status()


@router.get("/loras/compatible/{model_name}")
def get_compatible_loras(model_name: str):
    """Get LoRAs compatible with a given model (reads base_model field from registry)"""
    service = get_settings_service()
    rows = service.lora_adapter_picklist(model_name)
    return rows


def _controlnet_matches_scope(actions: object, scope: str | None) -> bool:
    """Filter registry controlnets by intended UI surface (create vs edit drawers)."""
    if not scope:
        return True
    acts = actions if isinstance(actions, dict) else {}
    has_retouch = acts.get("retouch") is not None
    has_extend = acts.get("extend") is not None
    if scope == "create":
        return not (has_retouch or has_extend)
    if scope == "retouch":
        return has_retouch
    if scope == "extend":
        return has_extend
    raise HTTPException(status_code=400, detail=f"invalid controlnet scope {scope!r}")


@router.get("/controlnets/compatible/{model_name}")
def get_compatible_controlnets(model_name: str, scope: str | None = None):
    """Get ControlNets compatible with a given model (reads base_model field from registry).

    ``scope``: ``create`` (structural guide / text-to-image), ``retouch``, or ``extend``.
    """
    service = get_settings_service()
    registry = service.get_model_registry()
    detailed_status = service.get_models_detailed_status()

    results = []
    for key, config in registry.items():
        if config.category != "controlnets":
            continue
        if config.type not in ("controlnet",):
            continue

        net_base = config.base_model or ""
        # Matching logic: FLUX models are compatible with all FLUX ControlNets
        if model_name.startswith("flux"):
            is_compatible = net_base == "" or net_base.startswith("flux")
        elif model_name.startswith("z-image"):
            is_compatible = (
                net_base == ""
                or net_base.startswith("z-image")
                or net_base in ("z-image", "z-image-turbo")
            )
        else:
            is_compatible = net_base == model_name

        if not is_compatible:
            continue

        actions = getattr(config, "actions", None) or {}
        if not _controlnet_matches_scope(actions, scope):
            continue

        # Check readiness status
        status_info = detailed_status.get(key, {})
        ready = status_info.get("ready", False)
        versions_ready = {}
        for vk, vs in status_info.get("versions", {}).items():
            versions_ready[vk] = vs.get("ready", False)

        from backend.engine.families.flux1.structural import (
            CONTROLNET_CUDA_BATCH_PLANNED,
            CONTROLNET_DECLARED_BACKENDS,
            controlnet_runtime_available,
        )

        results.append({
            "name": config.name or key,
            "name_en": getattr(config, "name_en", None) or config.name or key,
            "key": key,
            "base_model": net_base,
            "ready": ready,
            "versions_ready": versions_ready,
            "actions": actions,
            "parameters": config.parameters,
            "runtime_backends": list(CONTROLNET_DECLARED_BACKENDS),
            "runtime_available": controlnet_runtime_available(),
            "cuda_batch_planned": CONTROLNET_CUDA_BATCH_PLANNED,
        })

    return results


@router.get("/disk-space")
def get_disk_space():
    """Get disk space usage"""
    service = get_settings_service()
    space = service.get_disk_space()
    return space


@router.post("/models/{model_name}/parameters")
def update_model_parameters(model_name: str, request: dict, req: Request = None):
    locale = resolve_locale(req.headers.get("Accept-Language")) if req else "zh"
    import json
    import shutil
    from datetime import datetime
    from backend.core.container import get_container
    from backend.core.interfaces import IPathResolver
    
    path_resolver = get_container().resolve(IPathResolver)
    registry_path = path_resolver.get_models_registry_path()
    
    if not registry_path.exists():
        return {"success": False, "error": t("error.registry_not_found", locale)}
    
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        if model_name not in registry.get("models", {}):
            return {"success": False, "error": t("error.model_not_found", locale, name=model_name)}
        
        # Back up original file (with timestamp)
        backup_path = registry_path.parent / f"models_registry.json.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(registry_path, backup_path)
        
        # Update parameters
        model = registry["models"][model_name]
        if "parameters" not in model:
            model["parameters"] = {}
        
        # Plan C3: all writable scalars/enums with default in the registry are allowed to be updated from the settings page (aligned with RegistryParamsForm)
        for key, value in request.items():
            spec = model["parameters"].get(key)
            if not isinstance(spec, dict) or "default" not in spec:
                continue
            if spec.get("type") == "bool" and str(key).endswith("_support"):
                continue
            if isinstance(value, dict) and "default" in value:
                spec["default"] = value["default"]
            else:
                spec["default"] = value

        # Write back to file
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)
        
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/system/monitor")
def get_system_monitor():
    """Get real-time system resource monitoring (CPU, memory, GPU)"""
    import platform
    import subprocess
    
    data = {
        "cpu_percent": 0,
        "memory": {
            "total_gb": 0,
            "used_gb": 0,
            "percent": 0
        },
        "gpu": None
    }
    
    try:
        import psutil
        # CPU usage
        data["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        
        # Memory info
        mem = psutil.virtual_memory()
        data["memory"] = {
            "total_gb": round(mem.total / (1024**3), 1),
            "used_gb": round(mem.used / (1024**3), 1),
            "percent": mem.percent
        }
    except ImportError:
        # psutil not installed, try system commands
        try:
            if platform.system() == "Darwin":
                # CPU usage
                result = subprocess.run(
                    ["top", "-l", "1", "-n", "0"],
                    capture_output=True, text=True, timeout=3
                )
                for line in result.stdout.split("\n"):
                    if "CPU usage" in line:
                        # Extract user+system CPU usage
                        parts = line.split(":")[-1].strip().split(",")
                        user = float(parts[0].strip().replace("% user", ""))
                        sys = float(parts[1].strip().replace("% sys", ""))
                        data["cpu_percent"] = round(user + sys, 1)
                        break
                
                # Memory info
                result = subprocess.run(
                    ["vm_stat"],
                    capture_output=True, text=True, timeout=3
                )
                page_size = 4096
                mem_stats = {}
                for line in result.stdout.split("\n"):
                    if "page size of" in line:
                        # Extract page size from "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                        import re
                        match = re.search(r'page size of (\d+)', line)
                        if match:
                            page_size = int(match.group(1))
                    elif ":" in line and "Pages" in line:
                        key, val = line.split(":")
                        try:
                            mem_stats[key.strip()] = int(val.strip().replace(".", ""))
                        except ValueError:
                            pass
                
                if mem_stats:
                    used_pages = mem_stats.get("Pages active", 0) + mem_stats.get("Pages wired down", 0)
                    free_pages = mem_stats.get("Pages free", 0)
                    inactive_pages = mem_stats.get("Pages inactive", 0)
                    total_pages = free_pages + used_pages + inactive_pages
                    if total_pages > 0:
                        data["memory"]["total_gb"] = round(total_pages * page_size / (1024**3), 1)
                        data["memory"]["used_gb"] = round(used_pages * page_size / (1024**3), 1)
                        data["memory"]["percent"] = round(used_pages / total_pages * 100, 1)
        except Exception:
            pass
    except Exception:
        pass
    
    # GPU info
    try:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Try to get Apple Silicon chip model
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=2
                )
                chip_model = result.stdout.strip()
                # Extract chip model, e.g. "Apple M1 Max" -> "M1 Max"
                if "Apple" in chip_model:
                    chip_model = chip_model.replace("Apple ", "")
            except Exception:
                chip_model = "Apple Silicon"
            
            data["gpu"] = {
                "available": True,
                "model": chip_model,
                "note": "Unified Memory Architecture",
                "memory_gb": data["memory"]["total_gb"],
                "type": "integrated"
            }
    except Exception:
        pass
    
    return data


@router.get("/presets")
def get_presets():
    """Get all prompt presets"""
    preset_store = get_container().resolve(IPresetStore)
    return preset_store.load_all()


class PresetCreateRequest(BaseModel):
    name: str
    preset: dict


_PRESET_MEDIA_SCOPES = frozenset({"image", "video"})


def _validate_preset_plan_v2(preset: dict) -> None:
    """Plan G: presets must include non-empty applies_to and explicit media_scope (no runtime default)."""
    if not isinstance(preset, dict):
        raise HTTPException(status_code=400, detail="preset must be an object")
    app = preset.get("applies_to")
    if not isinstance(app, list) or len(app) == 0:
        raise HTTPException(status_code=400, detail="preset.applies_to must be a non-empty array")
    ms = preset.get("media_scope")
    if ms not in _PRESET_MEDIA_SCOPES:
        raise HTTPException(
            status_code=400,
            detail="preset.media_scope must be image or video",
        )


@router.post("/presets")
def create_preset(request: PresetCreateRequest):
    """Create or update prompt preset"""
    _validate_preset_plan_v2(request.preset)
    preset_store = get_container().resolve(IPresetStore)
    preset_store.save(request.name, request.preset)
    return {"success": True}


@router.delete("/presets/{name}")
def delete_preset(name: str):
    """Delete prompt preset"""
    preset_store = get_container().resolve(IPresetStore)
    preset_store.delete(name)
    return {"success": True}


def _human_readable_size(size_bytes: int) -> str:
    """Convert to human-readable size"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
