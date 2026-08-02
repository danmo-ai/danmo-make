"""Image create run context + phased helpers (``ImageSession``)."""

from __future__ import annotations

import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    ExecutionContext,
    ImageGenerationRequest,
    parse_size,
    work_title_metadata,
)
from backend.engine._transformer_registry import (
    augment_image_generation_request,
    attach_image_conditioning,
)
from backend.engine.contracts import FamilyRuntimeContract
from backend.engine.pipelines.image_run_common import (
    apply_image_inference_options,
    build_image_vae_preview_session,
    execute_family_image_generator,
    image_model_from_resolved_run,
    image_vae_decode,
    load_image_encoded_model,
    log_inference_sigma_schedule,
    resolve_image_preview,
    resolve_image_steps_guidance,
    schedule_image_run,
    uses_family_image_generator,
    warm_image_step_preview_decoders,
)
from backend.engine.config.model_configs import apply_image_registry_config_overrides, get_config_class
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._context import MediaRunContext
from backend.engine.pipelines.pipeline_progress import (
    emit_complete,
    emit_post_progress,
    pipeline_graph_step,
)

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class ImageCreateRunContext(MediaRunContext):
    """Mutable state for one image create run (encode → schedule → batched infer/decode/persist)."""

    pipeline: Any
    request: ImageGenerationRequest
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
    sigmas: Any | None
    sched_ts: Any
    timestep_embed_schedule: list[float] | None
    semantics: Any
    w: int
    h: int
    steps: int
    guidance: float
    base_seed: int
    n: int
    preview_mode: str
    preview_interval: int
    preview_max_edge: int
    preview_decoder: str
    preview_state: dict[str, Any]
    latent_noise_dtype: Any
    noise_sample_dtype: Any
    packed_denoise: bool
    flux_pack: Callable[..., Any] | None
    flux_unpack: Callable[..., Any] | None
    latent_h: int
    latent_w: int
    packed_shape: tuple[int, ...] | None
    structural_output_meta: dict[str, Any] | None
    structural_cleanup: Callable[[], None] | None = None
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(
        self,
        *,
        pipeline: Any | None = None,
        batch_seed: int | None = None,
        batch_idx: int = 0,
        batch_on_progress: Callable | None = None,
        **_ignored: Any,
    ) -> Any | None:
        _ = pipeline
        seed = batch_seed if batch_seed is not None else self.base_seed
        return execute_create_denoise(
            self,
            batch_seed=seed,
            batch_idx=batch_idx,
            batch_on_progress=batch_on_progress,
        )


def _scale_progress(cb: Callable | None, batch_idx: int, total: int) -> Callable | None:
    if cb is None or total <= 1:
        return cb

    def wrapped(p, s, t, msg=None, phase=None):
        overall_p = (batch_idx + float(p)) / total
        prefix = f"[{batch_idx + 1}/{total}]"
        msg_out = f"{prefix} {msg}" if msg else prefix
        cb(overall_p, s, t, msg_out, phase)

    return wrapped


def build_create_run_context(
    pipeline: Any,
    request: ImageGenerationRequest,
    exec_ctx: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
    preloaded_model: Any | None = None,
) -> ImageCreateRunContext | None:
    """Prepare encode + schedule state for v3 image create."""
    phase_cm = phase_cm or (lambda _name: nullcontext())

    request = augment_image_generation_request(request, pipeline.ctx)
    w, h = parse_size(request.size)
    base_seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    image = image_model_from_resolved_run(pipeline, resolved)
    acts = getattr(image.entry, "actions", frozenset())
    if "generate" not in acts:
        raise RuntimeError(
            f"Model {image.model_key!r} is not registered for text-to-image (actions must include create); "
            "refusing image create — see config/models_registry.json."
        )

    if exec_ctx.cancel_token.is_cancelled():
        return None

    steps, guidance, _meta = resolve_image_steps_guidance(
        image.entry, request, image.runtime_contract
    )
    preview_mode, preview_interval, preview_max_edge, preview_decoder = resolve_image_preview(image.entry)
    preview_state: dict[str, Any] = {}

    with phase_cm("encode"):
        encoded = load_image_encoded_model(
            pipeline,
            request=request,
            resolved=image,
            steps=steps,
            guidance=guidance,
            exec_ctx=exec_ctx,
            on_progress=on_progress,
            on_log=on_log,
            preloaded_model=preloaded_model,
        )
    if encoded is None:
        return None
    model = encoded.model
    extra_cond = encoded.extra_cond
    txt_embeds = encoded.txt_embeds
    neg_embeds = encoded.neg_embeds
    txt_attn_mask = encoded.txt_attn_mask
    neg_attn_mask = encoded.neg_attn_mask
    pooled_embeds = encoded.pooled_embeds
    neg_pooled_embeds = encoded.neg_pooled_embeds
    encoder_type = encoded.encoder_type
    entry = image.entry
    config = image.config
    runtime_contract = image.runtime_contract
    family = image.family
    model_key = image.model_key
    version_key = image.version_key
    bundle_root = image.bundle_root

    structural_cleanup: Callable[[], None] | None = None
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
            ctx_exec=exec_ctx,
            on_log=on_log,
        )
    except Exception:
        if structural_cleanup is not None:
            structural_cleanup()
        raise

    extra_cond = apply_image_inference_options(pipeline.ctx, request, extra_cond)

    with phase_cm("schedule"):
        scheduled = schedule_image_run(
            pipeline,
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            metadata=_meta,
            steps=steps,
            width=w,
            height=h,
        )
        semantics = scheduled.semantics
        scheduler = scheduled.scheduler
        scheduler_default = scheduled.scheduler_name
        timesteps = scheduled.timesteps
        sigmas = scheduled.sigmas
        sched_ts = scheduled.sched_ts
        timestep_embed_schedule = scheduled.timestep_embed_schedule
        vae_scale = getattr(config, "vae_scale", 8)
        image_seq_len = int(semantics.set_timesteps_kwargs.get("image_seq_len") or 0)

    if on_log:
        parts = [
            f"infer model={model_key}",
            f"family={family}",
            f"version={version_key or 'default'}",
            f"size={w}x{h}",
            f"base_seed={base_seed}",
            f"steps={steps}",
            f"guidance={guidance}",
            f"scheduler={scheduler_default}",
            f"supports_guidance={getattr(config, 'supports_guidance', False)}",
            f"cfg_on={bool(neg_embeds is not None)}",
            f"image_seq_len={image_seq_len}",
            f"vae_scale={vae_scale}",
        ]
        if semantics.sigma_schedule is not None:
            parts.append(f"sigma_schedule={semantics.sigma_schedule}")
        parts.append(f"use_empirical_mu={semantics.use_empirical_mu}")
        parts.append(f"requires_sigma_shift={semantics.requires_sigma_shift}")
        if semantics.sched_extra.get("mu") is not None:
            parts.append(f"scheduler_mu={semantics.sched_extra['mu']}")
        if timestep_embed_schedule and len(timestep_embed_schedule) >= 2:
            parts.append(
                f"t_embed_ends=[{timestep_embed_schedule[0]:.6g},{timestep_embed_schedule[-1]:.6g}]"
            )
        elif timestep_embed_schedule and len(timestep_embed_schedule) == 1:
            parts.append(f"t_embed=[{timestep_embed_schedule[0]:.6g}]")
        if semantics.cfg_renorm:
            parts.append(f"cfg_renorm=True cfg_renorm_min={semantics.cfg_renorm_min}")
        on_log("info", " ".join(parts))
        log_inference_sigma_schedule(on_log, sigmas)

    _lnd = runtime_contract.denoise_latent_noise_dtype(pipeline.ctx)
    _noise_sample_dtype = runtime_contract.noise_sample_dtype(pipeline.ctx, _lnd)
    packed_denoise = getattr(config, "latent_noise_packed", False)
    flux_pack = model.pack_latents if packed_denoise else None
    flux_unpack = model.unpack_latents if packed_denoise else None
    lh = lw = 0
    packed_shape = None
    if packed_denoise:
        seq_len = (h // 16) * (w // 16)
        lh, lw = h // vae_scale, w // vae_scale
        packed_shape = (1, seq_len, 64)

    n = max(getattr(request, "n", 1), 1)
    structural_output_meta: dict[str, Any] | None = None
    guide = getattr(request, "structural_guide", None)
    if guide is not None:
        structural_output_meta = {
            "structural_guide_model": (getattr(guide, "model_id", None) or "").strip(),
            "structural_guide_type": getattr(guide, "type", None) or "",
            "structural_guide_weight": float(guide.weight),
            "structural_guide_asset_id": guide.asset_id,
        }

    return ImageCreateRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=exec_ctx,
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
        w=w,
        h=h,
        steps=steps,
        guidance=guidance,
        base_seed=base_seed,
        n=n,
        preview_mode=preview_mode,
        preview_interval=preview_interval,
        preview_max_edge=preview_max_edge,
        preview_decoder=preview_decoder,
        preview_state=preview_state,
        latent_noise_dtype=_lnd,
        noise_sample_dtype=_noise_sample_dtype,
        packed_denoise=packed_denoise,
        flux_pack=flux_pack,
        flux_unpack=flux_unpack,
        latent_h=lh,
        latent_w=lw,
        packed_shape=packed_shape,
        structural_output_meta=structural_output_meta,
        structural_cleanup=structural_cleanup,
        on_progress=on_progress,
        on_log=on_log,
    )


def execute_create_denoise(
    ctx: ImageCreateRunContext,
    *,
    batch_seed: int,
    batch_idx: int = 0,
    batch_on_progress: Callable | None = None,
) -> Any | None:
    """Sample latents and run denoise loop; returns latents ready for VAE decode."""
    pipeline = ctx.pipeline
    model = ctx.model
    config = ctx.config
    exec_ctx = ctx.exec_ctx
    on_log = ctx.on_log

    if ctx.packed_denoise:
        if batch_seed is not None:
            latents = pipeline.ctx.seeded_randn(
                ctx.packed_shape, batch_seed, dtype=ctx.noise_sample_dtype
            )
        else:
            latents = pipeline.ctx.randn(ctx.packed_shape, dtype=ctx.noise_sample_dtype)
        if ctx.noise_sample_dtype != ctx.latent_noise_dtype:
            latents = latents.astype(ctx.latent_noise_dtype)
    elif getattr(config, "encoder_step_kwargs", None) == "qwen_image":
        lh, lw = ctx.h // getattr(config, "vae_scale", 8), ctx.w // getattr(config, "vae_scale", 8)
        q_seq = lh * lw
        if batch_seed is not None:
            packed_noise = pipeline.ctx.seeded_randn(
                (1, q_seq, 64), batch_seed, dtype=ctx.noise_sample_dtype
            )
        else:
            packed_noise = pipeline.ctx.randn((1, q_seq, 64), dtype=ctx.noise_sample_dtype)
        if ctx.noise_sample_dtype != ctx.latent_noise_dtype:
            packed_noise = packed_noise.astype(ctx.latent_noise_dtype)
        packed_noise = pipeline.ctx.reshape(packed_noise, (1, lh, lw, 64))
        latents = pipeline.ctx.permute(packed_noise, (0, 3, 1, 2))
    else:
        vae_scale = getattr(config, "vae_scale", 8)
        latent_shape = (1, config.in_channels, ctx.h // vae_scale, ctx.w // vae_scale)
        latents = ctx.runtime_contract.sample_txt2img_noise(
            pipeline.ctx,
            latent_shape=latent_shape,
            seed=batch_seed,
            sample_dtype=ctx.noise_sample_dtype,
            target_dtype=ctx.latent_noise_dtype,
        )

    local_extra_cond = dict(ctx.extra_cond)
    from backend.engine.inference.optimization_plan import attach_resolved_inference_plan

    local_extra_cond = attach_resolved_inference_plan(
        local_extra_cond,
        family=ctx.family,
        config=config,
        entry=ctx.entry,
        ctx=pipeline.ctx,
        num_steps=len(ctx.timesteps),
    )
    from backend.engine.inference.optimization_plan import log_inference_plan_from_cond

    log_inference_plan_from_cond(local_extra_cond, on_log)
    if ctx.packed_denoise:
        latents_nchw = ctx.flux_unpack(pipeline.ctx, latents, ctx.latent_h, ctx.latent_w)
        latents_nchw, local_extra_cond = model.before_denoise(
            latents_nchw,
            ctx.timesteps,
            ctx.sigmas,
            txt_embeds=ctx.txt_embeds,
            neg_embeds=ctx.neg_embeds,
            **local_extra_cond,
        )
        latents = ctx.flux_pack(pipeline.ctx, latents_nchw)
    else:
        latents, local_extra_cond = model.before_denoise(
            latents,
            ctx.timesteps,
            ctx.sigmas,
            txt_embeds=ctx.txt_embeds,
            neg_embeds=ctx.neg_embeds,
            **local_extra_cond,
        )

    batch_preview_state: dict[str, Any] = {"on_log": on_log}
    if ctx.preview_mode == "stream":
        warm_image_step_preview_decoders(
            pipeline,
            ctx.entry,
            ctx.version_key,
            batch_preview_state,
            config=config,
            preview_decoder=ctx.preview_decoder,
            on_log=on_log,
        )
        try:
            batch_preview_state["vae_session"] = build_image_vae_preview_session(
                pipeline,
                ctx.entry, ctx.version_key, on_log=on_log
            )
        except Exception as exc:
            batch_preview_state["vae_session"] = False
            if on_log:
                on_log("warning", f"preview VAE warmup skipped: {exc}")

    pipeline_graph_step("denoise", on_log)
    from backend.engine.inference.image_denoise import run_image_denoise

    latents = run_image_denoise(
        pipeline,
        model=model,
        scheduler=ctx.scheduler,
        timesteps=ctx.timesteps,
        latents=latents,
        config=config,
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
        extra_cond=local_extra_cond,
        semantics=ctx.semantics,
        ctx_exec=exec_ctx,
        on_progress=batch_on_progress,
        on_log=on_log,
        preview_mode=ctx.preview_mode,
        preview_interval=ctx.preview_interval,
        preview_max_edge=ctx.preview_max_edge,
        preview_state=batch_preview_state,
        entry=ctx.entry,
        version_key=ctx.version_key,
        packed_denoise=ctx.packed_denoise,
        flux_pack=ctx.flux_pack,
        flux_unpack=ctx.flux_unpack,
        latent_h=ctx.latent_h,
        latent_w=ctx.latent_w,
    )
    if latents is None:
        return None
    if exec_ctx.cancel_token.is_cancelled():
        return None
    if ctx.packed_denoise:
        latents = ctx.flux_unpack(pipeline.ctx, latents, ctx.latent_h, ctx.latent_w)
    return latents


def decode_create_latents(ctx: ImageCreateRunContext, latents: Any) -> Any:
    """VAE decode only (no filesystem write)."""
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
    return image_vae_decode(
        ctx.pipeline,
        latents, ctx.entry, ctx.version_key, on_log=ctx.on_log
    )


def persist_create_image(
    ctx: ImageCreateRunContext,
    image: Any,
    *,
    batch_seed: int,
    batch_idx: int = 0,
) -> tuple[str, dict[str, Any]] | None:
    """Write PNG + metadata to work_dir."""
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None

    pipeline_graph_step("save_asset", ctx.on_log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx.exec_ctx.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    name_infix = f"_b{batch_idx + 1}" if ctx.n > 1 else ""
    out_path = work / f"{ctx.model_key}{name_infix}_{batch_seed}_{timestamp}.png"
    if hasattr(image, "save"):
        image.save(str(out_path))
    emit_post_progress(ctx.on_progress, n_steps=len(ctx.timesteps), within_post=1.0)
    emit_complete(ctx.on_progress, len(ctx.timesteps))

    meta: dict[str, Any] = {
        "model": ctx.request.model,
        "seed": batch_seed,
        "prompt": ctx.request.prompt,
        "steps": ctx.steps,
        "guidance": ctx.guidance,
        "width": ctx.w,
        "height": ctx.h,
        "mime_type": "image/png",
    }
    if ctx.structural_output_meta:
        meta.update(ctx.structural_output_meta)
    from backend.engine.inference.optimization_plan import pop_inference_run_metadata

    meta.update(pop_inference_run_metadata(ctx.model))
    meta.update(work_title_metadata(ctx.request.title))
    return str(out_path), meta


_NOT_GENERATOR = object()


def _maybe_run_image_family_generator(
    pipeline: Any,
    request: ImageGenerationRequest,
    ctx_exec: ExecutionContext,
    resolved: ResolvedRun,
    *,
    phase_cm: PhaseCmFactory,
    on_progress: Callable | None,
    on_log: Callable | None,
) -> Any:
    entry = resolved.registry_entry
    family = resolved.family_id
    config = get_config_class(family)()
    apply_image_registry_config_overrides(entry, config)
    if not uses_family_image_generator(config):
        return _NOT_GENERATOR
    with phase_cm("infer"):
        return execute_family_image_generator(
            pipeline,
            request,
            ctx_exec,
            is_edit=False,
            on_progress=on_progress,
            on_log=on_log,
        )

