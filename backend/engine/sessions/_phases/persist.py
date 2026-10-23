"""Phase: write assets + lineage."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import EngineResult
from backend.engine.pipelines.image_create_phases import ImageCreateRunContext, persist_create_image
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._phases.trace import phase_trace_span


def traced_persist(resolved: ResolvedRun, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    with phase_trace_span(resolved, "persist"):
        return fn(*args, **kwargs)


def persist_phase(
    resolved: ResolvedRun,
    pixels: Any,
    *,
    pipeline: Any | None = None,
    create_ctx: ImageCreateRunContext | None = None,
    batch_seed: int | None = None,
    batch_idx: int = 0,
    metadata: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Persist one decoded image to work_dir."""
    _ = pipeline, metadata
    if create_ctx is None:
        raise RuntimeError("persist_phase requires ImageCreateRunContext")
    seed = batch_seed if batch_seed is not None else create_ctx.base_seed
    return traced_persist(
        resolved,
        persist_create_image,
        create_ctx,
        pixels,
        batch_seed=seed,
        batch_idx=batch_idx,
    )


def persist_audio_phase(
    resolved: ResolvedRun,
    batch_result: tuple[list[str], list[float], Any, Any],
    *,
    create_ctx: Any,
) -> EngineResult:
    from backend.engine.pipelines.audio_create_phases import persist_audio_create

    output_paths, output_durations, lyrics_captures, generator = batch_result
    return traced_persist(
        resolved,
        persist_audio_create,
        create_ctx,
        output_paths,
        output_durations,
        lyrics_captures,
        generator,
    )


def persist_audio_edit_phase(
    resolved: ResolvedRun,
    infer_result: tuple[list[str], list[float], Any, Any],
    *,
    edit_ctx: Any,
) -> EngineResult:
    from backend.engine.pipelines.audio_edit_phases import persist_audio_edit

    output_paths, output_durations, batch, quality = infer_result
    return traced_persist(
        resolved,
        persist_audio_edit,
        edit_ctx,
        output_paths,
        output_durations,
        batch,
        quality,
    )
