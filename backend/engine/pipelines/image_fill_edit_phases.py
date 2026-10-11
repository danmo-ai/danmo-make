"""Fill retouch/extend — phased session helpers (factory-dispatched per family)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, ImageEditRequest
from backend.engine._transformer_registry import (
    get_image_fill_edit_builder,
    get_image_fill_edit_execute,
    get_image_fill_edit_persist,
)
from backend.engine.contracts.runtime_contracts import FamilyRuntimeContract
from backend.engine.sessions._context import MediaRunContext, ResolvedRun

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class ImageFillEditRunContext(MediaRunContext):
    """State for Fill retouch/extend (encode → schedule → denoise → persist)."""

    pipeline: Any
    request: ImageEditRequest
    exec_ctx: ExecutionContext
    entry: Any
    config: Any
    runtime_contract: FamilyRuntimeContract
    family: str
    model_key: str
    version_key: str | None
    bundle_root: Path | None
    model: Any
    extra_cond: dict[str, Any]
    txt_embeds: Any
    neg_embeds: Any
    txt_attn_mask: Any
    neg_attn_mask: Any
    pooled_embeds: Any
    neg_pooled_embeds: Any
    encoder_type: str
    scheduler: Any
    timesteps: list[Any]
    sigmas: Any
    sched_ts: Any
    timestep_embed_schedule: list[float] | None
    semantics: Any
    latents: Any
    w: int
    h: int
    lh: int
    lw: int
    seed: int
    steps: int
    guidance: float
    flux_unpack: Callable[..., Any]
    flux_pack: Callable[..., Any]
    preview_mode: str
    preview_interval: int
    preview_max_edge: int
    preview_state: dict[str, Any]
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(self, **_ignored: Any) -> Any | None:
        return execute_image_fill_edit_denoise(self)

    def session_persist(self, latents: Any) -> tuple[str, dict[str, Any]] | None:
        return persist_image_fill_edit(self, latents)

    def edit_trace_name(self) -> str:
        return "image_fill_edit"


def build_image_fill_edit_context(
    pipeline: Any,
    request: ImageEditRequest,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
) -> ImageFillEditRunContext | None:
    """Prepare Fill edit through encode + schedule (registry family builder)."""
    if request.operation not in ("retouch", "extend"):
        raise RuntimeError(
            f"build_image_fill_edit_context requires retouch|extend, got {request.operation!r}"
        )
    family = resolved.family_id
    builder = get_image_fill_edit_builder(family)
    if builder is None:
        raise RuntimeError(
            f"retouch/extend Fill edit is not registered for family={family!r}; "
            "add a row in _IMAGE_FILL_EDIT_BUILD"
        )
    return builder(
        pipeline,
        request,
        ctx_exec,
        resolved=resolved,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
    )


def execute_image_fill_edit_denoise(ctx: ImageFillEditRunContext) -> Any | None:
    """Run Fill denoise via family registry execute."""
    execute = get_image_fill_edit_execute(ctx.family)
    if execute is None:
        raise RuntimeError(
            f"Fill edit execute is not registered for family={ctx.family!r}; "
            "add a row in _IMAGE_FILL_EDIT_EXECUTE"
        )
    return execute(ctx)


def persist_image_fill_edit(
    ctx: ImageFillEditRunContext,
    latents: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Decode Fill latents and write work_dir asset metadata."""
    persist = get_image_fill_edit_persist(ctx.family)
    if persist is None:
        raise RuntimeError(
            f"Fill edit persist is not registered for family={ctx.family!r}; "
            "add a row in _IMAGE_FILL_EDIT_PERSIST"
        )
    return persist(ctx, latents)
