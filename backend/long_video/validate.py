"""Long-video request capability validation (API, scheduler, engine)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.core.contracts import VideoLongGenerationRequest


@dataclass(frozen=True)
class LongVideoValidationError(Exception):
    code: str
    message: str
    http_status: int = 400

    def __str__(self) -> str:
        return self.message


class _CapabilityEngine(Protocol):
    def supports(self, model_id: str, action: str) -> bool: ...


def validate_long_video_request(
    request: VideoLongGenerationRequest,
    *,
    video_engine: _CapabilityEngine,
    image_engine: _CapabilityEngine | None = None,
) -> str:
    """Validate model capabilities for a long-video task. Returns keyframe_model id (may be empty)."""
    spec = request.long_video
    segment_model = (request.model or spec.segment_video_model or "").strip()
    if not segment_model:
        raise LongVideoValidationError("invalid", "model is required", 400)

    phase = (request.metadata or {}).get("long_video_phase") or ""
    if spec.strategy == "segmented_i2v":
        if phase != "assemble_only":
            raise LongVideoValidationError(
                "invalid",
                "segmented_i2v requires metadata.long_video_phase=assemble_only; "
                "generate keyframes and segments via image/video edit APIs first",
                400,
            )
        if not spec.shots:
            raise LongVideoValidationError("invalid", "long_video.shots required for assemble", 400)
        missing = [
            s.id for s in spec.shots if not (getattr(s, "segment_asset_id", None) or "").strip()
        ]
        if missing:
            raise LongVideoValidationError(
                "invalid",
                f"every shot must have segment_asset_id before assemble (missing: {', '.join(missing[:5])})",
                400,
            )
        return (spec.keyframe_model or "").strip()

    if spec.strategy == "latent_extend":
        if not video_engine.supports(segment_model, "generate"):
            raise LongVideoValidationError(
                "unsupported", "model does not support latent_extend create", 409
            )
        return (spec.keyframe_model or "").strip()

    raise LongVideoValidationError(
        "invalid", f"unsupported long_video strategy {spec.strategy!r}", 400
    )


def resolve_long_video_image_engine(request: VideoLongGenerationRequest, engines: Any) -> Any:
    """Resolve image engine for VIDEO_LONG_GENERATION after capability validation."""
    spec = request.long_video
    segment_model = (request.model or spec.segment_video_model or "").strip()
    video_eng = engines.get_video(segment_model)

    if spec.strategy == "segmented_i2v":
        validate_long_video_request(request, video_engine=video_eng, image_engine=None)
        kf_model = (spec.keyframe_model or "").strip()
        if kf_model:
            return engines.get_image(kf_model)
        return _fallback_image_engine(engines)

    validate_long_video_request(request, video_engine=video_eng, image_engine=None)
    kf_model = (spec.keyframe_model or "").strip()
    if kf_model:
        return engines.get_image(kf_model)

    return _fallback_image_engine(engines)


def _fallback_image_engine(engines: Any) -> Any:
    from backend.engine.danqing_image_engine import DanQingImageEngine

    image_eng = next(
        (e for e in engines._by_engine_id.values() if isinstance(e, DanQingImageEngine)),
        None,
    )
    if image_eng is None:
        raise LongVideoValidationError(
            "unsupported", "image engine not available for long video", 503
        )
    return image_eng
