"""Qwen-Image-Edit — VL encode, VAE conditioning, denoise, persist."""

from __future__ import annotations

import math
import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from backend.core.contracts import ExecutionContext, ImageEditRequest
from backend.engine.config.model_configs import assert_image_family_contract
from backend.engine.contracts import FamilyRuntimeContract
from backend.engine.pipelines.image_run_common import ResolvedImageModel
from backend.engine.sessions._context import MediaRunContext
from backend.engine.inference.image_denoise import run_image_denoise
from backend.engine.pipelines.image_run_common import (
    apply_image_lora_adapters,
    finalize_image_from_latents,
    image_vae_encode_tensor,
    load_image_dit_model,
    resolve_image_preview,
    resolve_image_steps_guidance,
    schedule_image_run,
    warm_image_step_preview_decoders,
)
from backend.engine.pipelines.pipeline_progress import (
    emit_phase,
    pipeline_graph_step,
    validate_bundle_graph_step,
)

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


def compute_qwen_edit_dimensions(
    source: Image.Image,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """返回 ``(out_w, out_h, vl_w, vl_h, vae_w, vae_h)``，均为 32 对齐。"""
    image_size = source.size
    ratio = image_size[0] / max(image_size[1], 1)
    multiple_of = 16
    if width is None and height is None:
        use_width = max(multiple_of, int(image_size[0]) // multiple_of * multiple_of)
        use_height = max(multiple_of, int(image_size[1]) // multiple_of * multiple_of)
    else:
        target_area = 1024 * 1024
        calculated_width = math.sqrt(target_area * ratio)
        calculated_height = calculated_width / ratio
        calculated_width = round(calculated_width / 32) * 32
        calculated_height = round(calculated_height / 32) * 32
        use_height = int(height or calculated_height)
        use_width = int(width or calculated_width)
        use_width = max(multiple_of, use_width // multiple_of * multiple_of)
        use_height = max(multiple_of, use_height // multiple_of * multiple_of)

    condition_area = 384 * 384
    condition_ratio = image_size[0] / max(image_size[1], 1)
    vl_width = round(math.sqrt(condition_area * condition_ratio) / 32) * 32
    vl_height = round((vl_width / condition_ratio) / 32) * 32

    vae_area = 1024 * 1024
    vae_ratio = image_size[0] / max(image_size[1], 1)
    vae_width = round(math.sqrt(vae_area * vae_ratio) / 32) * 32
    vae_height = round((vae_width / vae_ratio) / 32) * 32

    return (
        int(use_width),
        int(use_height),
        int(vl_width),
        int(vl_height),
        int(vae_width),
        int(vae_height),
    )


def pack_qwen_latents_to_sequence(ctx: Any, latents_nchw: Any) -> Any:
    """``[B,64,H,W]`` → ``[B, H*W, 64]``。"""
    b = int(latents_nchw.shape[0])
    c = int(latents_nchw.shape[1])
    h = int(latents_nchw.shape[2])
    w = int(latents_nchw.shape[3])
    x = ctx.permute(latents_nchw, (0, 2, 3, 1))
    return ctx.reshape(x, (b, h * w, c))


def unpack_qwen_sequence_to_nchw(ctx: Any, seq_bsc: Any, height_px: int, width_px: int) -> Any:
    """``[B, seq, 64]`` → ``[B,64,H/16,W/16]``。"""
    b = int(seq_bsc.shape[0])
    h_lat = height_px // 16
    w_lat = width_px // 16
    x = ctx.reshape(seq_bsc, (b, h_lat, w_lat, 64))
    return ctx.permute(x, (0, 3, 1, 2))


def create_qwen_edit_conditioning_latents(
    ctx: Any,
    *,
    vae_encode_fn,
    source: Image.Image,
    vae_width: int,
    vae_height: int,
    on_log: Any | None = None,
) -> tuple[Any, tuple[int, int, int]]:
    """VAE 编码参考图并 pack；返回 ``(packed_nchw, cond_image_grid)``。"""
    from backend.engine.vae_codec_registry import qwen_pack_latents_nchw

    src = source.convert("RGB").resize((vae_width, vae_height), Image.BICUBIC)
    import numpy as np

    arr = np.asarray(src, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    img_n11 = ctx.array(arr.transpose(2, 0, 1)[None, ...])

    encoded = vae_encode_fn(
        img_n11,
        height_px=vae_height,
        width_px=vae_width,
    )
    packed = qwen_pack_latents_nchw(ctx, encoded, vae_height, vae_width)
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(packed)
    cond_h = vae_height // 16
    cond_w = vae_width // 16
    if on_log:
        on_log(
            "info",
            f"qwen_edit conditioning vae={vae_width}x{vae_height} grid=1x{cond_h}x{cond_w} "
            f"packed={tuple(packed.shape)}",
        )
    return packed, (1, cond_h, cond_w)


def create_qwen_edit_multi_conditioning_latents(
    ctx: Any,
    *,
    vae_encode_fn,
    sources: list[Image.Image],
    on_log: Any | None = None,
) -> tuple[Any, list[tuple[int, int, int]]]:
    """多参考图 VAE 条件：序列维拼接 packed latents。"""
    if not sources:
        raise RuntimeError("Qwen edit multi-image conditioning requires at least one source image.")

    grids: list[tuple[int, int, int]] = []
    seq_parts: list[Any] = []
    for idx, source in enumerate(sources):
        _w, _h, _vl_w, _vl_h, vae_w, vae_h = compute_qwen_edit_dimensions(source)
        packed, grid = create_qwen_edit_conditioning_latents(
            ctx,
            vae_encode_fn=vae_encode_fn,
            source=source,
            vae_width=vae_w,
            vae_height=vae_h,
            on_log=on_log if idx == 0 else None,
        )
        seq_parts.append(pack_qwen_latents_to_sequence(ctx, packed))
        grids.append(grid)

    combined = seq_parts[0]
    for part in seq_parts[1:]:
        combined = ctx.concat([combined, part], axis=1)
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(combined)
    if on_log:
        on_log(
            "info",
            f"qwen_edit multi conditioning refs={len(sources)} grids={grids} "
            f"seq={tuple(combined.shape)}",
        )
    return combined, grids


def resolve_qwen_edit_reference_images(
    request: ImageEditRequest,
    ctx_exec: ExecutionContext,
    primary: Image.Image,
    *,
    max_images: int,
) -> list[Image.Image]:
    """Primary source first, then ``reference_asset_ids`` (deduped, capped)."""
    images = [primary.convert("RGB")]
    seen = {request.source_asset_id}
    cap = max(1, int(max_images))
    for asset_id in request.reference_asset_ids or []:
        aid = str(asset_id or "").strip()
        if not aid or aid in seen:
            continue
        seen.add(aid)
        ref_path = ctx_exec.asset_store.get_file_path(aid)
        images.append(Image.open(ref_path).convert("RGB"))
        if len(images) >= cap:
            break
    return images


@dataclass
class QwenImageEditRunContext(MediaRunContext):
    """Qwen VL edit — encode through persist (VAE finalize, not rewrite decode)."""

    pipeline: Any
    request: ImageEditRequest
    exec_ctx: ExecutionContext
    entry: Any
    config: Any
    runtime_contract: FamilyRuntimeContract
    family: str
    model_key: str
    version_key: str | None
    bundle_root: Any
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
    preview_mode: str
    preview_interval: int
    preview_max_edge: int
    preview_state: dict[str, Any]
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(self, **_ignored: Any) -> Any | None:
        return execute_qwen_image_edit_denoise(self)


def build_qwen_image_edit_context(
    pipeline: Any,
    request: ImageEditRequest,
    ctx_exec: ExecutionContext,
    *,
    image: ResolvedImageModel,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
) -> QwenImageEditRunContext | None:
    phase_cm = phase_cm or (lambda _name: nullcontext())
    model_key = image.model_key
    version_key = image.version_key
    entry = image.entry
    config = image.config
    family = image.family
    runtime_contract = image.runtime_contract
    vae_scale = int(getattr(config, "vae_scale", 16))

    if ctx_exec.cancel_token.is_cancelled():
        return None

    bundle_root = image.bundle_root
    validate_bundle_graph_step(
        bundle_root, family=family, model_id=model_key, on_log=on_log
    )

    steps, guidance, _meta_ed = resolve_image_steps_guidance(
        entry,
        request,
        runtime_contract,
        steps_default=int(getattr(config, "edit_default_steps", 0) or 20),
        guidance_default=4.0,
    )
    seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
    preview_mode, preview_interval, preview_max_edge, _preview_decoder = resolve_image_preview(entry)
    preview_state: dict[str, Any] = {}
    use_picture_prefix = bool(getattr(config, "edit_use_picture_prefix", False))
    edit_plus_multi = bool(getattr(config, "edit_plus_multi_image", False))
    max_ref_images = int(getattr(config, "edit_max_reference_images", 1) or 1)

    with phase_cm("encode"):
        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        pil = Image.open(src_path).convert("RGB")
        w, h, vl_w, vl_h, vae_w, vae_h = compute_qwen_edit_dimensions(pil)
        ref_images = resolve_qwen_edit_reference_images(
            request,
            ctx_exec,
            pil,
            max_images=max_ref_images if edit_plus_multi else 1,
        )

        emit_phase(on_progress, phase="encoding", progress=0.02, n_steps=steps)
        pipeline_graph_step("encode_prompt", on_log)
        neg_prompt = (request.negative_prompt or "").strip()
        if getattr(pipeline.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"Qwen edit prompt encode requires MLX backend "
                f"(got {getattr(pipeline.ctx, 'backend', None)!r})"
            )
        from backend.engine.families.qwen.edit_encoder_mlx import (
            build_qwen_edit_vl_tokenizer,
            encode_qwen_edit_prompts_mlx,
            load_qwen_edit_vl_encoder,
        )

        tok_root = bundle_root / "tokenizer"
        if not tok_root.is_dir():
            tok_root = bundle_root / "text_encoder"
        vl_encoder = load_qwen_edit_vl_encoder(bundle_root, pipeline.ctx)
        vl_tokenizer = build_qwen_edit_vl_tokenizer(
            tok_root,
            use_picture_prefix=use_picture_prefix,
        )
        txt_embeds, txt_attn_mask, neg_embeds, neg_attn_mask = encode_qwen_edit_prompts_mlx(
            vl_encoder=vl_encoder,
            vl_tokenizer=vl_tokenizer,
            ctx=pipeline.ctx,
            prompt=request.prompt,
            negative_prompt=neg_prompt,
            sources=ref_images,
            vl_width=vl_w,
            vl_height=vl_h,
        )
        pooled_embeds = neg_pooled_embeds = None
        encoder_type = getattr(config, "encoder_type", "qwen_image")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        emit_phase(on_progress, phase="loading_model", progress=0.08, n_steps=steps)
        pipeline_graph_step("load_transformer", on_log)
        allow_cache = not (getattr(request, "adapters", None) or [])
        model = load_image_dit_model(
            pipeline,
            family, config, entry, version_key or None, allow_cache=allow_cache
        )
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
        apply_image_lora_adapters(pipeline, family, model, request, on_log)
        extra_cond = model.prepare_conditioning(
            request, bundle_root=str(bundle_root) if bundle_root else None
        )

        def _vae_enc(img_n11: Any, *, height_px: int, width_px: int) -> Any:
            return image_vae_encode_tensor(
                pipeline,
                img_n11,
                entry,
                version_key or None,
                height_px=height_px,
                width_px=width_px,
                on_log=on_log,
            )

        cond_latents, cond_grid = create_qwen_edit_multi_conditioning_latents(
            pipeline.ctx,
            vae_encode_fn=lambda img, height_px, width_px: _vae_enc(
                img, height_px=height_px, width_px=width_px
            ),
            sources=ref_images,
            on_log=on_log,
        )
        extra_cond = dict(extra_cond)
        extra_cond["edit_conditioning_latents"] = cond_latents
        extra_cond["edit_cond_image_grid"] = cond_grid if len(cond_grid) > 1 else cond_grid[0]

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
        )
        semantics = scheduled.semantics
        scheduler = scheduled.scheduler
        timesteps = scheduled.timesteps
        sigmas = scheduled.sigmas
        sched_ts = scheduled.sched_ts
        timestep_embed_schedule = scheduled.timestep_embed_schedule

        _lnd = runtime_contract.denoise_latent_noise_dtype(pipeline.ctx)
        _noise_sample_dtype = runtime_contract.noise_sample_dtype(pipeline.ctx, _lnd)
        lh, lw = h // vae_scale, w // vae_scale
        q_seq = lh * lw
        packed_noise = pipeline.ctx.seeded_randn((1, q_seq, 64), seed, dtype=_noise_sample_dtype)
        if _noise_sample_dtype != _lnd:
            packed_noise = packed_noise.astype(_lnd)
        latents = pipeline.ctx.reshape(packed_noise, (1, lh, lw, 64))
        latents = pipeline.ctx.permute(latents, (0, 3, 1, 2))
        if getattr(pipeline.ctx, "backend", None) == "mlx":
            pipeline.ctx.eval(latents)

        if on_log:
            on_log(
                "info",
                f"qwen_image_edit model={model_key} out={w}x{h} refs={len(ref_images)} "
                f"vae_cond={vae_w}x{vae_h} vl={vl_w}x{vl_h} steps={steps} guidance={guidance} seed={seed}",
            )

        latents, extra_cond = model.before_denoise(
            latents,
            timesteps,
            sigmas,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            **extra_cond,
        )

        preview_state["on_log"] = on_log
        if preview_mode == "stream":
            warm_image_step_preview_decoders(
                pipeline,
                entry, version_key or None, preview_state, config=config, on_log=on_log
            )

    return QwenImageEditRunContext(
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
        preview_mode=preview_mode,
        preview_interval=preview_interval,
        preview_max_edge=preview_max_edge,
        preview_state=preview_state,
        on_progress=on_progress,
        on_log=on_log,
    )


def execute_qwen_image_edit_denoise(ctx: QwenImageEditRunContext) -> Any | None:
    pipeline_graph_step("denoise", ctx.on_log)
    return run_image_denoise(
        ctx.pipeline,
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
        preview_state=ctx.preview_state,
        entry=ctx.entry,
        version_key=ctx.version_key,
    )


def persist_qwen_image_edit(ctx: QwenImageEditRunContext, latents: Any) -> tuple[str, dict[str, Any]] | None:
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None
    return finalize_image_from_latents(
        ctx.pipeline,
        latents=latents,
        timesteps=ctx.timesteps,
        entry=ctx.entry,
        version_key=ctx.version_key,
        model_key=ctx.model_key,
        seed=ctx.seed,
        request=ctx.request,
        ctx_exec=ctx.exec_ctx,
        steps=ctx.steps,
        guidance=ctx.guidance,
        w=ctx.w,
        h=ctx.h,
        on_progress=ctx.on_progress,
        on_log=ctx.on_log,
        name_infix="_edit",
        extra_meta={"operation": ctx.request.operation, "edit_model": ctx.model_key},
    )

