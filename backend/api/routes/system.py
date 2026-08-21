"""Plan §6.2：``GET /api/system/health``、``GET /api/system/metrics``、``GET /api/system/cache``。"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter

from backend.core.container import get_container

router = APIRouter(prefix="/api/system", tags=["system"])


def _gpu_memory_fields() -> dict[str, Optional[float]]:
    try:
        import psutil

        m = psutil.virtual_memory()
        return {
            "memory_total": round(m.total / (1024**3), 2),
            "free": round(getattr(m, "available", m.free) / (1024**3), 2),
        }
    except Exception:
        return {"memory_total": None, "free": None}


@router.get("/health")
def health() -> dict[str, Any]:
    """存活 + 依赖探测 + v4 后端状态。"""
    from backend.engine.platform import PlatformInfo

    out: dict[str, Any] = {
        "status": "ok",
        "gpu": _gpu_memory_fields(),
        "backends": {
            "mlx": "unavailable",
        },
        "engines": {
            "danqing-image": "unavailable",
            "danqing-video": "unavailable",
        },
    }
    # 依赖探测
    try:
        __import__("mlx.core", fromlist=["*"])
        out["backends"]["mlx"] = "ok"
    except Exception:
        out["backends"]["mlx"] = "unavailable"

    for b in PlatformInfo.detect():
        out["backends"][b] = "ok"

    # v4 engines registered?
    try:
        from backend.core.container import get_container
        from backend.core.media_interfaces import IImageEngine, IVideoEngine
        c = get_container()
        img_eng = c.try_resolve(IImageEngine)
        if img_eng is not None and hasattr(img_eng, "engine_id"):
            out["engines"]["danqing-image"] = img_eng.engine_id
        vid_eng = c.try_resolve(IVideoEngine)
        if vid_eng is not None and hasattr(vid_eng, "engine_id"):
            out["engines"]["danqing-video"] = vid_eng.engine_id
    except Exception as e:
        out["engines"]["danqing-image"] = str(e)
        out["engines"]["danqing-video"] = str(e)

    return out


@router.get("/metrics")
def metrics() -> dict[str, Any]:
    """轻量 CPU/内存（与设置页 monitor 互补；不阻塞过久）。"""
    data: dict[str, Any] = {
        "cpu_percent": None,
        "memory": {"total_gb": None, "used_gb": None, "percent": None},
    }
    try:
        import psutil

        data["cpu_percent"] = round(psutil.cpu_percent(interval=0.1), 1)
        m = psutil.virtual_memory()
        data["memory"] = {
            "total_gb": round(m.total / (1024**3), 2),
            "used_gb": round(m.used / (1024**3), 2),
            "percent": m.percent,
        }
    except Exception:
        pass
    return data


@router.get("/cache")
def cache_status() -> dict[str, Any]:
    """Model cache status (ModelCache stats)."""
    from backend.engine.platform import PlatformInfo

    c = get_container()
    cache = c.try_resolve_named("shared_model_cache")
    out: dict[str, Any] = {"cache": None, "mlx": {}}
    if cache is not None:
        out["cache"] = cache.stats
    out["mlx"] = PlatformInfo.get_mlx_memory_stats()
    return out
