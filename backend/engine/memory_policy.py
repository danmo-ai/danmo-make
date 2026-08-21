"""GPU memory budget and shared :class:`ModelCache` wiring (MLX, REST + CLI)."""
from __future__ import annotations

import gc
import os
from typing import Any, Callable, Mapping, TYPE_CHECKING

from backend.core.interfaces import AppSettings

if TYPE_CHECKING:
    from backend.engine.cache import ModelCache


def clamp_mlx_memory_limit_gb(gb: int | float | None, *, default: int = 120) -> int:
    try:
        value = int(float(gb))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = default
    return max(16, min(value, 512))


def release_cached_model(model: Any) -> None:
    """Drop a cached model and release MLX allocator caches."""
    del model
    gc.collect()
    try:
        from backend.engine.memory_policy_mlx import clear_mlx_cache

        clear_mlx_cache()
    except Exception:
        pass


def build_shared_model_cache(
    load_settings: Callable[[], AppSettings],
    *,
    reserve_gb: float = 20.0,
    max_entries: int = 1,
) -> ModelCache:
    from backend.engine.cache import ModelCache

    settings = load_settings()
    ttl = int(getattr(settings, "model_cache_ttl_minutes", 30) or 30)
    return ModelCache(
        get_memory_limit=lambda: float(load_settings().mlx_memory_limit),
        reserve_gb=reserve_gb,
        ttl_minutes=max(1, ttl),
        max_entries=max_entries,
        release_fn=release_cached_model,
    )


def resolve_mlx_memory_limit_gb(settings: AppSettings) -> int:
    """Workspace setting, overridable via ``DANQING_MLX_MEMORY_LIMIT_GB`` (CLI / bench subprocess)."""
    override = os.environ.get("DANQING_MLX_MEMORY_LIMIT_GB", "").strip()
    if override:
        return clamp_mlx_memory_limit_gb(override)
    return clamp_mlx_memory_limit_gb(settings.mlx_memory_limit)


def build_gpu_runtimes(settings: AppSettings) -> dict[str, Any]:
    from backend.engine.platform import PlatformInfo

    platforms = PlatformInfo.detect()
    runtimes: dict[str, Any] = {}
    limit_gb = resolve_mlx_memory_limit_gb(settings)
    if "mlx" in platforms:
        from backend.engine.runtime.mlx import MLXContext

        runtimes["mlx"] = MLXContext(memory_limit_gb=limit_gb)
    return runtimes


def apply_memory_settings(
    settings: AppSettings,
    runtimes: Mapping[str, Any] | None,
    cache: ModelCache | None,
) -> None:
    """MLX: memory limit. Cache TTL always."""
    limit_gb = clamp_mlx_memory_limit_gb(settings.mlx_memory_limit)
    mlx = (runtimes or {}).get("mlx")
    if mlx is not None and hasattr(mlx, "apply_memory_limit_gb"):
        mlx.apply_memory_limit_gb(limit_gb)
    if cache is not None:
        ttl = int(getattr(settings, "model_cache_ttl_minutes", 30) or 30)
        cache.set_ttl_minutes(max(1, ttl))


def apply_memory_settings_from_container(settings: AppSettings) -> None:
    from backend.core.container import get_container

    c = get_container()
    apply_memory_settings(
        settings,
        c.try_resolve_named("gpu_runtimes"),
        c.try_resolve_named("shared_model_cache"),
    )


def unload_model_cache_if_present() -> None:
    from backend.core.container import get_container

    cache = get_container().try_resolve_named("shared_model_cache")
    if cache is not None:
        cache.unload_all()


def resolve_lora_worker_memory_gb(
    settings: AppSettings | None = None,
    *,
    parent_reserve_gb: int = 16,
) -> int:
    """MLX budget for the LoRA child process (API keeps ``parent_reserve_gb`` headroom)."""
    if settings is None:
        try:
            from backend.core.container import get_container

            cfg = get_container().try_resolve_named("config_store")
            settings = cfg.load() if cfg is not None else AppSettings()
        except Exception:
            settings = AppSettings()
    limit = resolve_mlx_memory_limit_gb(settings)
    return max(48, int(limit) - max(8, int(parent_reserve_gb)))


def prepare_host_for_vlm_audit(*, mlx_runtime: Any | None = None) -> int:
    """Unload DiT cache in the API process before spawning an isolated VLM worker."""
    unload_model_cache_if_present()
    if mlx_runtime is not None and hasattr(mlx_runtime, "clear_cache"):
        mlx_runtime.clear_cache()
    else:
        from backend.engine.memory_policy_mlx import clear_mlx_cache

        clear_mlx_cache()
    try:
        from backend.core.container import get_container

        cfg = get_container().try_resolve_named("config_store")
        settings = cfg.load() if cfg is not None else AppSettings()
    except Exception:
        settings = AppSettings()
    limit = resolve_mlx_memory_limit_gb(settings)
    return max(32, int(limit) - 32)


def prepare_host_for_lora_worker(*, mlx_runtime: Any | None = None) -> int:
    """Unload cached models in the API process before spawning the LoRA worker."""
    unload_model_cache_if_present()
    if mlx_runtime is not None and hasattr(mlx_runtime, "clear_cache"):
        mlx_runtime.clear_cache()
    else:
        from backend.engine.memory_policy_mlx import clear_mlx_cache

        clear_mlx_cache()
    return resolve_lora_worker_memory_gb()
