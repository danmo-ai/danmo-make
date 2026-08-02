"""Video inference option wiring (TeaCache, etc.) — no pipeline imports."""
from __future__ import annotations

from typing import Any

from backend.core.contracts import VideoEditRequest, VideoGenerationRequest

_VIDEO_INFERENCE_OPTION_ATTRS = ("teacache_mode",)


def apply_video_inference_options(
    ctx: Any,
    request: VideoGenerationRequest | VideoEditRequest,
    extra_cond: dict[str, Any] | None,
) -> dict[str, Any]:
    """Wire MLX-only video inference enum options from API request into ``extra_cond``."""
    extra = dict(extra_cond or {})
    for attr in _VIDEO_INFERENCE_OPTION_ATTRS:
        val = getattr(request, attr, None)
        if val is None:
            continue
        norm = str(val).strip().lower()
        if norm in ("", "none", "off"):
            extra[attr] = "none"
            continue
        from backend.engine.runtime.mlx_guards import require_mlx_if_option_active

        require_mlx_if_option_active(ctx, feature=attr, option=val)
        extra[attr] = norm
    return extra
