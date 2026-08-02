"""Engine → Session dispatch (generation families use FamilyPlugin + Session only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
    VideoEditRequest,
    VideoGenerationRequest,
    VideoAvatarRequest,
    VideoLongGenerationRequest,
    VideoUpscaleRequest,
    parse_model_version,
)
from backend.engine.config.model_configs import FAMILY_CONFIG_MAP
from backend.engine.contracts import require_entry_family
from backend.engine.registry.family_registry import is_family_plugin_registered
from backend.engine.sessions.audio_session import (
    AudioSession,
    routes_to_audio_edit_session,
    routes_to_audio_session,
)
from backend.engine.sessions.image_session import (
    ImageSession,
    routes_to_image_edit_session,
    routes_to_image_session,
)
from backend.engine.sessions.upscale_session import (
    UpscaleSession,
    routes_to_upscale_session,
)
from backend.engine.sessions.video_session import VideoSession, routes_to_video_session
from backend.engine.sessions.session_routing import routes_with_plugin_and_action
from backend.engine.sessions.video_upscale_session import (
    VideoUpscaleSession,
    routes_to_video_upscale_session,
)


def assert_generation_family_has_plugin(
    model_field: str,
    registry: Any,
    *,
    expected_media: str,
) -> None:
    """Fail loud when a generation family lacks a registered plugin."""
    model_key, _ = parse_model_version(model_field)
    entry = registry.get(model_key)
    if entry is None or entry.media != expected_media:
        return
    family = require_entry_family(entry, model_id=model_key)
    if family not in FAMILY_CONFIG_MAP:
        return
    if is_family_plugin_registered(family):
        return
    raise RuntimeError(
        f"Model {model_key!r} (family={family!r}) has no FamilyPlugin; "
        "register in bootstrap_family_plugins()."
    )


def _session_kwargs(
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
) -> dict[str, Any]:
    return {
        "runtime_ctx": runtime,
        "model_registry": registry,
        "asset_store": asset_store,
        "model_cache": model_cache,
        "project_root": project_root or Path.cwd(),
    }


def _require_session_route(
    routes: bool,
    *,
    model: str,
    operation: str,
) -> None:
    if routes:
        return
    raise RuntimeError(
        f"Model {model!r} is not supported for {operation} via engine session; "
        "check registry actions and FamilyPlugin registration."
    )


def dispatch_image_create(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: ImageGenerationRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="image")
    _require_session_route(
        routes_to_image_session(request.model, registry),
        model=request.model,
        operation="image create",
    )
    return ImageSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_image_edit(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: ImageEditRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="image")
    _require_session_route(
        routes_to_image_edit_session(request.model, registry),
        model=request.model,
        operation="image edit",
    )
    return ImageSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run_edit(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_image_upscale(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: ImageUpscaleRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="image")
    _require_session_route(
        routes_to_upscale_session(request.model, registry),
        model=request.model,
        operation="image upscale",
    )
    return UpscaleSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_video_create(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: VideoGenerationRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="video")
    _require_session_route(
        routes_to_video_session(request.model, registry),
        model=request.model,
        operation="video create",
    )
    return VideoSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_video_edit(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: VideoEditRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="video")
    _require_session_route(
        routes_to_video_session(request.model, registry),
        model=request.model,
        operation="video edit",
    )
    from backend.long_video.segment_source import resolve_long_video_edit_chain_source

    request = resolve_long_video_edit_chain_source(
        request,
        asset_store=asset_store,
        work_dir=Path(exec_ctx.work_dir),
        task_id=exec_ctx.task_id,
    )
    return VideoSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run_edit(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_video_avatar(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: VideoAvatarRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="video")
    _require_session_route(
        routes_with_plugin_and_action(
            request.model,
            registry,
            expected_media="video",
            api_action="avatar",
        ),
        model=request.model,
        operation="video avatar",
    )
    return VideoSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run_avatar(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_video_upscale(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: VideoUpscaleRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="video")
    _require_session_route(
        routes_to_video_upscale_session(request.model, registry),
        model=request.model,
        operation="video upscale",
    )
    return VideoUpscaleSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run(
        request, exec_ctx, on_progress=on_progress, on_log=on_log
    )


def dispatch_long_video(
    *,
    image_runtime: Any,
    video_runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: VideoLongGenerationRequest,
    exec_ctx: ExecutionContext,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> Any:
    from backend.engine.inference.long_video_orchestrator import run_long_video_orchestrator

    spec = request.long_video
    phase = (request.metadata or {}).get("long_video_phase") or ""
    if spec.strategy == "segmented_i2v" and phase != "assemble_only":
        kf_model = (spec.keyframe_model or "").strip()
        if not kf_model:
            raise RuntimeError("long_video.keyframe_model is required for segmented_i2v")
        assert_generation_family_has_plugin(kf_model, registry, expected_media="image")
        _require_session_route(
            routes_to_image_session(kf_model, registry),
            model=kf_model,
            operation="image create (long video keyframe)",
        )
    seg_model = (request.model or spec.segment_video_model or "").strip()
    if spec.strategy == "segmented_i2v" and phase != "assemble_only":
        assert_generation_family_has_plugin(seg_model, registry, expected_media="video")
        _require_session_route(
            routes_to_video_session(seg_model, registry),
            model=seg_model,
            operation="video animate (long video segment)",
        )
    elif spec.strategy == "latent_extend":
        assert_generation_family_has_plugin(seg_model, registry, expected_media="video")
        _require_session_route(
            routes_to_video_session(seg_model, registry),
            model=seg_model,
            operation="video create (latent extend)",
        )

    kwargs = {
        "registry": registry,
        "asset_store": asset_store,
        "model_cache": model_cache,
        "project_root": project_root or Path.cwd(),
    }
    return run_long_video_orchestrator(
        request=request,
        exec_ctx=exec_ctx,
        image_dispatch={"runtime": image_runtime, **kwargs},
        video_dispatch={"runtime": video_runtime, **kwargs},
        on_progress=on_progress,
        on_log=on_log,
    )


def dispatch_audio_create(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: AudioGenerationRequest,
    exec_ctx: ExecutionContext,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="audio")
    _require_session_route(
        routes_to_audio_session(request.model, registry),
        model=request.model,
        operation="audio create",
    )
    return AudioSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run(
        request, exec_ctx
    )


def dispatch_audio_edit(
    *,
    runtime: Any,
    registry: Any,
    asset_store: Any,
    model_cache: Any | None,
    project_root: Path | None,
    request: AudioEditRequest,
    exec_ctx: ExecutionContext,
) -> Any:
    assert_generation_family_has_plugin(request.model, registry, expected_media="audio")
    _require_session_route(
        routes_to_audio_edit_session(request.model, registry),
        model=request.model,
        operation="audio edit",
    )
    return AudioSession(**_session_kwargs(runtime, registry, asset_store, model_cache, project_root)).run_edit(
        request, exec_ctx
    )
