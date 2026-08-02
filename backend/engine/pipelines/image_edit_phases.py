"""Image edit phased helpers (``ImageSession``)."""

from __future__ import annotations

import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    ExecutionContext,
    ImageEditRequest,
    work_title_metadata,
)
from backend.engine._transformer_registry import attach_image_edit_extra_cond, attach_image_conditioning, augment_image_inference_request
from backend.engine.contracts import FamilyRuntimeContract
from backend.engine.inference.image_denoise import run_image_denoise
from backend.engine.pipelines.image_run_common import (
    ResolvedImageModel,
    apply_image_inference_options,
    assert_edit_rewrite_schedule,
    build_image_vae_preview_session,
    image_model_from_resolved_run,
    image_vae_decode,
    encode_edit_reference_latent,
    load_edit_source_rgb,
    load_image_encoded_model,
    log_inference_sigma_schedule,
    packed_edit_latent_dims,
    prepare_edit_rewrite_latents,
    resolve_image_preview,
    resolve_image_steps_guidance,
    schedule_image_run,
    validate_edit_vae_latent_grid,
    warm_image_step_preview_decoders,
)
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._context import MediaRunContext
from backend.engine.pipelines.pipeline_progress import (
    emit_complete,
    emit_post_progress,
    pipeline_graph_step,
)

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class ImageEditRunContext(MediaRunContext):
    """State for rewrite img2img edit (encode → schedule → infer → decode → persist)."""

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
    seed: int
    steps: int
    guidance: float
    init_timestep: int
    fidelity: float
    source_pil: Any
    preview_mode: str
    preview_interval: int
    preview_max_edge: int
    preview_decoder: str
    preview_state: dict[str, Any]
    packed_edit: bool
    flux_unpack_edit: Callable[..., Any] | None
    lh_edit: int
    lw_edit: int
    edit_conditioning_concat: bool
    structural_output_meta: dict[str, Any] | None
    structural_cleanup: Callable[[], None] | None = None
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(self, **_ignored: Any) -> Any | None:
        return execute_image_edit_denoise(self)


def build_image_edit_context(
    pipeline: Any,
    request: ImageEditRequest,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
    preloaded_model: Any | None = None,
) -> Any | None:
    """Rewrite img2img or config-flag edit (Qwen VL) — encode + schedule."""
    if request.rewrite_mode == "instruct":
        raise RuntimeError(
            "Image edit rewrite_mode instruct (flux1-kontext instruction editing) "
            "is not implemented in this pipeline."
        )

    image = image_model_from_resolved_run(pipeline, resolved)

    if bool(getattr(image.config, "edit_use_vl_vision", False)):
        from backend.engine.families.qwen.edit_util import build_qwen_image_edit_context

        return build_qwen_image_edit_context(
            pipeline,
            request,
            ctx_exec,
            image=image,
            on_progress=on_progress,
            on_log=on_log,
            phase_cm=phase_cm,
        )

    return build_image_edit_rewrite_context(
        pipeline,
        request,
        ctx_exec,
        resolved=image,
        on_progress=on_progress,
        on_log=on_log,
        phase_cm=phase_cm,
        preloaded_model=preloaded_model,
    )


def build_image_edit_rewrite_context(
    pipeline: Any,
    request: ImageEditRequest,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedImageModel | None = None,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
    preloaded_model: Any | None = None,
) -> ImageEditRunContext | None:
    """Prepare rewrite img2img state through encode + schedule."""
    phase_cm = phase_cm or (lambda _name: nullcontext())
    if resolved is None:
        raise RuntimeError("build_image_edit_rewrite_context requires ResolvedImageModel")

    acts = getattr(resolved.entry, "actions", frozenset())
    if "edit" not in acts:
        raise RuntimeError(
            f"Model {resolved.model_key!r} is not registered for image edit (actions need rewrite/retouch/extend); "
            "refusing image edit — see config/models_registry.json."
        )

    model_key = resolved.model_key
    version_key = resolved.version_key
    entry = resolved.entry
    family = resolved.family
    config = resolved.config
    runtime_contract = resolved.runtime_contract
    bundle_root = resolved.bundle_root

    if ctx_exec.cancel_token.is_cancelled():
        return None

    request = augment_image_inference_request(request, pipeline.ctx)

    with phase_cm("encode"):
        pil, w, h = load_edit_source_rgb(pipeline, ctx_exec, request.source_asset_id)
        validate_edit_vae_latent_grid(
            family=family,
            config=config,
            bundle_root=bundle_root,
            height=h,
            width=w,
            ctx=pipeline.ctx,
        )
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        reference_latent = encode_edit_reference_latent(
            pipeline,
            pil=pil,
            entry=entry,
            version_key=version_key,
            config=config,
            model_key=model_key,
            family=family,
            width=w,
            height=h,
            on_log=on_log,
        )
        fidelity = max(0.0, min(1.0, float(request.source_fidelity)))
        edit_conditioning_concat = bool(getattr(config, "edit_conditioning_concat", False))

        steps, guidance, _meta_ed = resolve_image_steps_guidance(
            entry, request, runtime_contract
        )
        init_timestep = 0
        if fidelity > 0.0 and not edit_conditioning_concat:
            init_timestep = max(1, int(steps * fidelity))
        preview_mode, preview_interval, preview_max_edge, preview_decoder = resolve_image_preview(entry)
        preview_state: dict[str, Any] = {}

        enc_loaded = load_image_encoded_model(
            pipeline,
            request=request,
            resolved=resolved,
            steps=steps,
            guidance=guidance,
            exec_ctx=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
            preloaded_model=preloaded_model,
        )
        if enc_loaded is None:
            return None
        model = enc_loaded.model
        extra_cond = enc_loaded.extra_cond
        if "original_size" in extra_cond or "target_size" in extra_cond:
            extra_cond = dict(extra_cond)
            extra_cond["original_size"] = (w, h)
            extra_cond["target_size"] = (w, h)
        txt_embeds = enc_loaded.txt_embeds
        neg_embeds = enc_loaded.neg_embeds
        txt_attn_mask = enc_loaded.txt_attn_mask
        neg_attn_mask = enc_loaded.neg_attn_mask
        pooled_embeds = enc_loaded.pooled_embeds
        neg_pooled_embeds = enc_loaded.neg_pooled_embeds
        encoder_type = enc_loaded.encoder_type
        if edit_conditioning_concat:
            extra_cond = attach_image_edit_extra_cond(
                family, extra_cond, reference_latent, height=h, width=w
            )

        structural_cleanup: Callable[[], None] | None = None
        structural_output_meta: dict[str, Any] | None = None
        try:
            extra_cond, structural_cleanup = attach_image_conditioning(
                pipeline,
                request=request,
                family=family,
                model=model,
                entry=entry,
                version_key=version_key,
                extra_cond=extra_cond,
                width=w,
                height=h,
                ctx_exec=ctx_exec,
                on_log=on_log,
            )
        except Exception:
            if structural_cleanup is not None:
                structural_cleanup()
            raise

        extra_cond = apply_image_inference_options(pipeline.ctx, request, extra_cond)

        guide = getattr(request, "structural_guide", None)
        if guide is not None:
            structural_output_meta = {
                "structural_guide_model": (getattr(guide, "model_id", None) or "").strip(),
                "structural_guide_type": getattr(guide, "type", None) or "",
                "structural_guide_weight": float(guide.weight),
                "structural_guide_asset_id": guide.asset_id,
            }

    with phase_cm("schedule"):
        scheduled = schedule_image_run(
            pipeline,
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            metadata=_meta_ed,
            steps=steps,
            width=w,
            height=h,
            init_timestep=init_timestep,
        )
        semantics = scheduled.semantics
        scheduler = scheduled.scheduler
        scheduler_default = scheduled.scheduler_name
        timesteps = scheduled.timesteps
        sigmas = scheduled.sigmas
        sched_ts = scheduled.sched_ts
        timestep_embed_schedule = scheduled.timestep_embed_schedule

        assert_edit_rewrite_schedule(
            scheduler_name=scheduler_default,
            timesteps=timesteps,
            init_timestep=init_timestep,
            steps=steps,
            fidelity=fidelity,
            sigmas=sigmas,
        )
        latents = prepare_edit_rewrite_latents(
            pipeline,
            model=model,
            config=config,
            runtime_contract=runtime_contract,
            encoded=reference_latent,
            seed=seed,
            init_timestep=init_timestep,
            sigmas=sigmas,
        )

        if on_log:
            on_log(
                "info",
                f"edit rewrite model={model_key} family={family} size={w}x{h} seed={seed} "
                f"steps={steps} init_timestep={init_timestep} scheduler={scheduler_default} "
                f"source_fidelity={fidelity}"
                + (" edit_conditioning_concat=1" if edit_conditioning_concat else ""),
            )
            log_inference_sigma_schedule(on_log, sigmas)

        from backend.engine.inference.optimization_plan import attach_resolved_inference_plan

        extra_cond = attach_resolved_inference_plan(
            extra_cond,
            family=family,
            config=config,
            entry=entry,
            ctx=pipeline.ctx,
            num_steps=len(timesteps),
        )
        from backend.engine.inference.optimization_plan import log_inference_plan_from_cond

        log_inference_plan_from_cond(extra_cond, on_log)

        latents, extra_cond = model.before_denoise(
            latents,
            timesteps,
            sigmas,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            **extra_cond,
        )

    packed_edit = getattr(config, "latent_noise_packed", False)
    flux_unpack_edit = model.unpack_latents if packed_edit else None
    lh_edit, lw_edit = packed_edit_latent_dims(latents, packed=packed_edit)

    return ImageEditRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=ctx_exec,
        entry=entry,
        config=config,
        runtime_contract=runtime_contract,
        family=family,
        model_key=model_key,
        version_key=version_key,
        bundle_root=bundle_root,
        model=model,
        extra_cond=extra_cond,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        txt_attn_mask=txt_attn_mask,
        neg_attn_mask=neg_attn_mask,
        pooled_embeds=pooled_embeds,
        neg_pooled_embeds=neg_pooled_embeds,
        encoder_type=encoder_type,
        scheduler=scheduler,
        timesteps=timesteps,
        sigmas=sigmas,
        sched_ts=sched_ts,
        timestep_embed_schedule=timestep_embed_schedule,
        semantics=semantics,
        latents=latents,
        w=w,
        h=h,
        seed=seed,
        steps=steps,
        guidance=guidance,
        init_timestep=init_timestep,
        fidelity=fidelity,
        source_pil=pil,
        preview_mode=preview_mode,
        preview_interval=preview_interval,
        preview_max_edge=preview_max_edge,
        preview_decoder=preview_decoder,
        preview_state=preview_state,
        packed_edit=packed_edit,
        flux_unpack_edit=flux_unpack_edit,
        lh_edit=lh_edit,
        lw_edit=lw_edit,
        edit_conditioning_concat=edit_conditioning_concat,
        structural_output_meta=structural_output_meta,
        structural_cleanup=structural_cleanup,
        on_progress=on_progress,
        on_log=on_log,
    )


def execute_image_edit_denoise(ctx: ImageEditRunContext) -> Any | None:
    """Rewrite img2img denoise via ``DiffusionInference``."""
    pipeline = ctx.pipeline
    preview_state = ctx.preview_state
    preview_state["on_log"] = ctx.on_log
    if ctx.preview_mode == "stream":
        warm_image_step_preview_decoders(
            pipeline,
            ctx.entry, ctx.version_key or None, preview_state,
            config=ctx.config,
            preview_decoder=ctx.preview_decoder,
            on_log=ctx.on_log,
        )
        try:
            preview_state["vae_session"] = build_image_vae_preview_session(
                pipeline,
                ctx.entry, ctx.version_key or None, on_log=ctx.on_log
            )
        except Exception as exc:
            preview_state["vae_session"] = False
            if ctx.on_log:
                ctx.on_log("warning", f"preview VAE warmup skipped: {exc}")

    pipeline_graph_step("denoise", ctx.on_log)
    latents = run_image_denoise(
        pipeline,
        model=ctx.model,
        scheduler=ctx.scheduler,
        timesteps=ctx.timesteps,
        latents=ctx.latents,
        config=ctx.config,
        runtime_contract=ctx.runtime_contract,
        guidance=ctx.guidance,
        txt_embeds=ctx.txt_embeds,
        neg_embeds=ctx.neg_embeds,
        pooled_embeds=ctx.pooled_embeds,
        neg_pooled_embeds=ctx.neg_pooled_embeds,
        txt_attn_mask=ctx.txt_attn_mask,
        neg_attn_mask=ctx.neg_attn_mask,
        encoder_type=ctx.encoder_type,
        width=ctx.w,
        height=ctx.h,
        sched_ts=ctx.sched_ts,
        sigmas=ctx.sigmas,
        timestep_embed_schedule=ctx.timestep_embed_schedule,
        extra_cond=ctx.extra_cond,
        semantics=ctx.semantics,
        ctx_exec=ctx.exec_ctx,
        on_progress=ctx.on_progress,
        on_log=ctx.on_log,
        preview_mode=ctx.preview_mode,
        preview_interval=ctx.preview_interval,
        preview_max_edge=ctx.preview_max_edge,
        preview_state=preview_state,
        entry=ctx.entry,
        version_key=ctx.version_key,
        timestep_offset=ctx.init_timestep,
        packed_denoise=ctx.packed_edit and ctx.latents.ndim == 3,
        flux_pack=None,
        flux_unpack=ctx.flux_unpack_edit,
        latent_h=ctx.lh_edit,
        latent_w=ctx.lw_edit,
    )
    if latents is None:
        return None
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None
    if ctx.packed_edit and latents.ndim == 3 and ctx.flux_unpack_edit is not None:
        latents = ctx.flux_unpack_edit(pipeline.ctx, latents, ctx.lh_edit, ctx.lw_edit)
    return latents


def decode_image_edit_latents(ctx: ImageEditRunContext, latents: Any) -> Any:
    from PIL import Image

    from backend.engine.families.z_image.latent_refine_mlx import apply_latent_refine_if_requested

    latents = apply_latent_refine_if_requested(
        ctx.pipeline,
        latents,
        request=ctx.request,
        entry=ctx.entry,
        version_key=ctx.version_key,
        model=ctx.model,
        timesteps=ctx.timesteps,
        sigmas=ctx.sigmas,
        txt_embeds=ctx.txt_embeds,
        neg_embeds=ctx.neg_embeds,
        guidance=ctx.guidance,
        extra_cond=ctx.extra_cond,
        on_log=ctx.on_log,
        scheduler=ctx.scheduler,
        config=ctx.config,
        runtime_contract=ctx.runtime_contract,
        semantics=ctx.semantics,
        exec_ctx=ctx.exec_ctx,
        encoder_type=ctx.encoder_type,
        sched_ts=ctx.sched_ts,
        timestep_embed_schedule=ctx.timestep_embed_schedule,
        pooled_embeds=ctx.pooled_embeds,
        neg_pooled_embeds=ctx.neg_pooled_embeds,
        txt_attn_mask=ctx.txt_attn_mask,
        neg_attn_mask=ctx.neg_attn_mask,
    )
    pipeline_graph_step("decode_vae", ctx.on_log)
    image = image_vae_decode(
        ctx.pipeline,
        latents, ctx.entry, ctx.version_key or None, on_log=ctx.on_log
    )
    if getattr(ctx.config, "edit_rmbg_composite_output", False):
        matte = image.convert("L")
        composite = ctx.source_pil.copy()
        composite.putalpha(matte.resize(composite.size, Image.LANCZOS))
        return composite
    return image


def persist_image_edit(
    ctx: ImageEditRunContext,
    image: Any,
) -> tuple[str, dict[str, Any]] | None:
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None

    pipeline_graph_step("save_asset", ctx.on_log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx.exec_ctx.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = work / f"{ctx.model_key}_edit_{ctx.seed}_{timestamp}.png"
    if hasattr(image, "save"):
        image.save(str(out_path))
    emit_post_progress(ctx.on_progress, n_steps=len(ctx.timesteps), within_post=1.0)
    emit_complete(ctx.on_progress, len(ctx.timesteps))

    meta: dict[str, Any] = {
        "model": ctx.request.model,
        "seed": ctx.seed,
        "prompt": ctx.request.prompt,
        "steps": ctx.steps,
        "guidance": ctx.guidance,
        "width": ctx.w,
        "height": ctx.h,
        "mime_type": "image/png",
        "operation": ctx.request.operation,
        "source_fidelity": ctx.fidelity,
    }
    if ctx.structural_output_meta:
        meta.update(ctx.structural_output_meta)
    from backend.engine.inference.optimization_plan import pop_inference_run_metadata

    meta.update(pop_inference_run_metadata(ctx.model))
    meta.update(work_title_metadata(ctx.request.title))
    return str(out_path), meta

