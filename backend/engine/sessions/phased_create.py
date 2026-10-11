"""Phased create orchestration (session trace spans + shared create helpers)."""

from __future__ import annotations

import time
from contextlib import nullcontext
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
    VideoUpscaleRequest,
)
from backend.engine.families._image_backbone import plugin_backbone_model_if_ready
from backend.engine.pipelines.audio_create_phases import build_audio_create_run_context
from backend.engine.pipelines.audio_edit_phases import build_audio_edit_run_context
from backend.engine.pipelines.image_create_phases import (
    _NOT_GENERATOR as _IMAGE_NOT_GENERATOR,
    _maybe_run_image_family_generator,
    build_create_run_context,
)
from backend.engine.pipelines.image_run_common import execute_family_image_generator, uses_family_image_generator
from backend.engine.config.model_configs import apply_image_registry_config_overrides, get_config_class
from backend.engine.pipelines.image_edit_phases import (
    build_image_edit_context,
)
from backend.engine.pipelines.image_fill_edit_phases import (
    build_image_fill_edit_context,
)
from backend.engine.pipelines.pipeline_progress import emit_post_progress
from backend.engine.pipelines.audio_persist import raise_if_cancelled as raise_if_audio_cancelled
from backend.engine.pipelines.upscale_create_phases import (
    build_upscale_create_context,
    persist_upscale_create,
)
from backend.engine.pipelines.video_upscale_create_phases import (
    build_video_upscale_create_context,
    persist_video_upscale_create,
)
from backend.engine.pipelines.video_create_phases import (
    _NOT_GENERATOR,
    _maybe_run_video_family_avatar,
    _maybe_run_video_family_generator,
    build_video_create_run_context,
    persist_video_create,
)
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._phases.decode import decode_phase
from backend.engine.sessions._phases.trace import phase_trace_span
from backend.engine.sessions._phases.infer import infer_phase
from backend.engine.sessions._phases.persist import persist_audio_edit_phase, persist_audio_phase, persist_phase, traced_persist
from backend.engine.sessions._phases.schedule import ScheduleState, schedule_phase

_JOB_SCHEDULE = ScheduleState(None, [], None, "job")


def phase_cm_factory(resolved: ResolvedRun):
    trace = getattr(resolved.exec_ctx, "trace", None)
    if trace is None:
        return lambda _name: nullcontext()

    def _factory(name: str):
        return trace.span_ctx(name, kind="phase", family_id=resolved.family_id)

    return _factory


def _run_infer_job_phased(
    resolved: ResolvedRun,
    pipeline: Any,
    *,
    ctx: Any,
    persist_fn: Callable[..., Any],
) -> Any:
    with phase_trace_span(resolved, "infer"):
        out = infer_phase(
            resolved,
            {},
            _JOB_SCHEDULE,
            runtime_ctx=pipeline.ctx,
            run_ctx=ctx,
        )
    return traced_persist(resolved, persist_fn, ctx, out)


def _run_ctx_infer_then(
    resolved: ResolvedRun,
    pipeline: Any,
    ctx: Any,
    finalize: Callable[[ResolvedRun, Any, Any], Any],
) -> Any:
    """Infer via ``run_ctx.session_infer`` then family-specific persist (audio paths)."""
    with phase_trace_span(resolved, "infer"):
        out = infer_phase(
            resolved,
            {},
            _JOB_SCHEDULE,
            runtime_ctx=pipeline.ctx,
            run_ctx=ctx,
        )
    return finalize(resolved, out, ctx)


def _run_image_infer_persist(
    resolved: ResolvedRun,
    pipeline: Any,
    ctx: Any,
    *,
    schedule_name: str,
    persist_fn: Callable[..., Any],
    decode_fn: Callable[..., Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    try:
        with phase_trace_span(resolved, "denoise"):
            latents = infer_phase(
                resolved,
                {},
                ScheduleState(ctx.scheduler, ctx.timesteps, ctx.sigmas, schedule_name),
                runtime_ctx=pipeline.ctx,
                pipeline=pipeline,
                run_ctx=ctx,
            )
        if latents is None:
            return None
        if decode_fn is not None:
            image = decode_fn(resolved, latents, edit_ctx=ctx)
            emit_post_progress(ctx.on_progress, n_steps=len(ctx.timesteps), within_post=0.5)
            return traced_persist(resolved, persist_fn, ctx, image)
        return traced_persist(resolved, persist_fn, ctx, latents)
    finally:
        cleanup = getattr(ctx, "structural_cleanup", None)
        if cleanup is not None:
            cleanup()


def _run_video_denoise_phased(
    resolved: ResolvedRun,
    pipeline: Any,
    ctx: Any,
) -> tuple[str, dict[str, Any]] | None:
    schedule = ScheduleState(ctx.scheduler, ctx.timesteps, ctx.sigmas, "video")
    with phase_trace_span(resolved, "denoise"):
        latents = infer_phase(
            resolved,
            {},
            schedule,
            runtime_ctx=pipeline.ctx,
            pipeline=pipeline,
            run_ctx=ctx,
        )
    if latents is None:
        return None
    return traced_persist(resolved, persist_video_create, ctx, latents)


def run_image_create_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: ImageGenerationRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> list[tuple[str, dict[str, Any]]] | None:
    phase_cm = phase_cm_factory(resolved)
    gen = _maybe_run_image_family_generator(
        pipeline,
        request,
        exec_ctx,
        resolved,
        phase_cm=phase_cm,
        on_progress=on_progress,
        on_log=on_log,
    )
    if gen is not _IMAGE_NOT_GENERATOR:
        if gen is None:
            return None
        if isinstance(gen, list):
            return gen
        return [gen]

    ctx = build_create_run_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        preloaded_model=plugin_backbone_model_if_ready(resolved.plugin, request=request),
    )
    if ctx is None:
        return None

    schedule = schedule_phase(resolved, ctx=ctx)
    conditioning: dict[str, Any] = {
        "txt_embeds": ctx.txt_embeds,
        "neg_embeds": ctx.neg_embeds,
    }

    results: list[tuple[str, dict[str, Any]]] = []
    try:
        for i in range(ctx.n):
            if exec_ctx.cancel_token.is_cancelled():
                return results if results else None
            batch_seed = ctx.base_seed + i
            batch_on_progress = ctx.on_progress
            if ctx.n > 1 and ctx.on_progress is not None:
                from backend.engine.pipelines.pipeline_progress import scale_batch_progress

                batch_on_progress = scale_batch_progress(ctx.on_progress, i, ctx.n)
            if on_log:
                on_log("info", f"batch {i + 1}/{ctx.n} seed={batch_seed}")

            with phase_trace_span(resolved, "denoise"):
                latents = infer_phase(
                    resolved,
                    conditioning,
                    schedule,
                    runtime_ctx=pipeline.ctx,
                    pipeline=pipeline,
                    run_ctx=ctx,
                    batch_seed=batch_seed,
                    batch_idx=i,
                    batch_on_progress=batch_on_progress,
                )
            if latents is None:
                return results if results else None

            image = decode_phase(resolved, latents, pipeline=pipeline, create_ctx=ctx)
            saved = persist_phase(
                resolved,
                image,
                pipeline=pipeline,
                create_ctx=ctx,
                batch_seed=batch_seed,
                batch_idx=i,
            )
            if saved is None:
                return results if results else None
            if isinstance(saved, tuple):
                results.append(saved)
            else:
                results.extend(saved)
        return results
    finally:
        if ctx.structural_cleanup is not None:
            ctx.structural_cleanup()


def run_image_edit_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: ImageEditRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    phase_cm = phase_cm_factory(resolved)
    entry = resolved.registry_entry
    family = resolved.family_id
    config = get_config_class(family)()
    apply_image_registry_config_overrides(entry, config)
    if uses_family_image_generator(config):
        with phase_cm("infer"):
            return execute_family_image_generator(
                pipeline,
                request,
                exec_ctx,
                is_edit=True,
                on_progress=on_progress,
                on_log=on_log,
            )

    preloaded = plugin_backbone_model_if_ready(resolved.plugin, request=request)

    if request.operation in ("retouch", "extend"):
        ctx = build_image_fill_edit_context(
            pipeline,
            request,
            exec_ctx,
            resolved=resolved,
            on_progress=on_progress,
            on_log=on_log,
            phase_cm=phase_cm,
        )
        if ctx is None:
            return None
        return _run_image_infer_persist(
            resolved,
            pipeline,
            ctx,
            schedule_name=ctx.edit_trace_name(),
            persist_fn=ctx.session_persist,
        )

    ctx = build_image_edit_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        preloaded_model=preloaded,
    )
    if ctx is None:
        return None

    decode_fn = None
    edit_decode = getattr(ctx, "edit_decode_fn", None)
    if callable(edit_decode):
        decode_fn = edit_decode()
    return _run_image_infer_persist(
        resolved,
        pipeline,
        ctx,
        schedule_name=ctx.edit_trace_name(),
        persist_fn=ctx.session_persist,
        decode_fn=decode_fn,
    )


def run_video_create_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: VideoGenerationRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    phase_cm = phase_cm_factory(resolved)
    gen = _maybe_run_video_family_generator(
        pipeline,
        request,
        exec_ctx,
        resolved,
        is_edit=False,
        phase_cm=phase_cm,
        on_progress=on_progress,
        on_log=on_log,
    )
    if gen is not _NOT_GENERATOR:
        return gen

    ctx = build_video_create_run_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        is_edit=False,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        plugin=resolved.plugin,
    )
    if ctx is None:
        return None
    return _run_video_denoise_phased(resolved, pipeline, ctx)


def run_video_edit_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: VideoEditRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    phase_cm = phase_cm_factory(resolved)
    gen = _maybe_run_video_family_generator(
        pipeline,
        request,
        exec_ctx,
        resolved,
        is_edit=True,
        phase_cm=phase_cm,
        on_progress=on_progress,
        on_log=on_log,
    )
    if gen is not _NOT_GENERATOR:
        return gen

    ctx = build_video_create_run_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        is_edit=True,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        plugin=resolved.plugin,
    )
    if ctx is None:
        return None
    return _run_video_denoise_phased(resolved, pipeline, ctx)


def run_video_avatar_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: VideoAvatarRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    phase_cm = phase_cm_factory(resolved)
    gen = _maybe_run_video_family_avatar(
        pipeline,
        request,
        exec_ctx,
        resolved,
        phase_cm=phase_cm,
        on_progress=on_progress,
        on_log=on_log,
    )
    if gen is not _NOT_GENERATOR:
        return gen
    raise RuntimeError(
        f"Model {request.model!r} does not use family_avatar pipeline "
        "(check registry family and model_configs video_pipeline_shape)"
    )


def run_upscale_create_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: ImageUpscaleRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    phase_cm = phase_cm_factory(resolved)
    ctx = build_upscale_create_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        plugin=resolved.plugin,
    )
    if ctx is None:
        return None
    return _run_infer_job_phased(
        resolved,
        pipeline,
        ctx=ctx,
        persist_fn=persist_upscale_create,
    )


def run_video_upscale_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: VideoUpscaleRequest,
    exec_ctx: ExecutionContext,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    phase_cm = phase_cm_factory(resolved)
    ctx = build_video_upscale_create_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        plugin=resolved.plugin,
    )
    if ctx is None:
        return None
    return _run_infer_job_phased(
        resolved,
        pipeline,
        ctx=ctx,
        persist_fn=persist_video_upscale_create,
    )


def run_audio_create_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: AudioGenerationRequest,
    exec_ctx: ExecutionContext,
) -> Any:
    raise_if_audio_cancelled(exec_ctx)
    t0 = time.monotonic()
    phase_cm = phase_cm_factory(resolved)

    paradigm = (
        resolved.plugin.select_paradigm() if resolved.plugin is not None else "flow_matching"
    )

    ctx = build_audio_create_run_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        t0=t0,
        phase_cm=phase_cm,
        plugin=resolved.plugin,
        paradigm=paradigm,
    )
    return _run_ctx_infer_then(
        resolved,
        pipeline,
        ctx,
        lambda r, out, c: persist_audio_phase(r, out, create_ctx=c),
    )


def run_audio_edit_phased(
    pipeline: Any,
    resolved: ResolvedRun,
    request: AudioEditRequest,
    exec_ctx: ExecutionContext,
) -> Any:
    raise_if_audio_cancelled(exec_ctx)
    t0 = time.monotonic()
    phase_cm = phase_cm_factory(resolved)
    paradigm = (
        resolved.plugin.select_paradigm() if resolved.plugin is not None else "flow_matching"
    )

    ctx = build_audio_edit_run_context(
        pipeline,
        request,
        exec_ctx,
        resolved=resolved,
        t0=t0,
        phase_cm=phase_cm,
        plugin=resolved.plugin,
        paradigm=paradigm,
    )
    return _run_ctx_infer_then(
        resolved,
        pipeline,
        ctx,
        lambda r, out, c: persist_audio_edit_phase(r, out, edit_ctx=c),
    )
