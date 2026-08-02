"""Shared resolve / encode / schedule helpers for image create + edit."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from backend.core.contracts import (
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    parse_model_version,
    work_title_metadata,
)
from backend.engine._transformer_registry import (
    encode_image_text_conditioning,
    merge_image_lora_adapters as _merge_image_lora_adapters,
)
from backend.engine.codecs import (
    decode_vae_preview,
    get_vae_decode_handler,
    get_vae_encode_handler,
    get_vae_preview_warmup_handler,
    warmup_vae_preview,
)
from backend.engine.config.model_configs import (
    apply_image_registry_config_overrides,
    assert_image_family_contract,
    get_config_class,
)

if TYPE_CHECKING:
    from backend.engine.sessions._context import ResolvedRun
from backend.engine.contracts import (
    FamilyRuntimeContract,
    SchedulerSemanticsResolver,
    local_bundle_root as local_bundle_root_fn,
    local_bundle_root as _local_bundle_root_fn,
    registry_scalar_default,
    registry_scalar_default as _registry_scalar_default_fn,
    require_entry_family,
)
from backend.engine.common.ops.schedulers import get_scheduler
from backend.engine.pipelines.image_model_load import load_image_transformer
from backend.engine.pipelines.pipeline_progress import (
    DENOISE_PROGRESS_SHARE,
    emit_complete,
    emit_phase,
    emit_post_progress,
    pipeline_graph_step,
    timestep_embed_schedule_from_scheduler,
    validate_bundle_graph_step,
)

_IMAGE_SCHEDULER_SEMANTICS = SchedulerSemanticsResolver()

_IMAGE_INFERENCE_OPTION_ATTRS = ("lemica_mode", "teacache_mode")


def apply_image_inference_options(
    ctx: Any,
    request: ImageGenerationRequest | ImageEditRequest,
    extra_cond: dict[str, Any] | None,
) -> dict[str, Any]:
    """Wire MLX-only inference enum options from API request into ``extra_cond``."""
    extra = dict(extra_cond or {})
    for attr in _IMAGE_INFERENCE_OPTION_ATTRS:
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


def resolve_image_preview_settings(entry: Any) -> tuple[str, int, int, str]:
    """Return (preview_mode, interval_steps, max_edge_px, preview_decoder)."""
    mode = _registry_scalar_default_fn(entry, "preview_mode", None)
    if mode is None:
        raw = getattr(entry, "raw", None) or {}
        model_type = str(raw.get("type", "") if isinstance(raw, dict) else "")
        if model_type != "diffusion":
            mode = "none"
        else:
            mode = "stream"
    mode = str(mode).strip().lower()
    if mode not in ("stream", "none"):
        mode = "none"
    interval = int(_registry_scalar_default_fn(entry, "preview_interval_steps", 2) or 2)
    max_edge = int(_registry_scalar_default_fn(entry, "preview_max_edge", 512) or 512)
    decoder = str(_registry_scalar_default_fn(entry, "preview_decoder", "auto") or "auto").strip().lower()
    if decoder not in ("auto", "standard", "taesd"):
        decoder = "auto"
    return mode, max(1, interval), max(64, min(2048, max_edge)), decoder


@dataclass(frozen=True)
class ResolvedImageModel:
    model_key: str
    version_key: str | None
    entry: Any
    family: str
    config: Any
    runtime_contract: FamilyRuntimeContract
    bundle_root: Path | None


def _build_resolved_image_model(
    pipeline: Any,
    *,
    model_key: str,
    version_key: str | None,
    entry: Any,
    bundle_root: Path | None = None,
) -> ResolvedImageModel:
    family = require_entry_family(entry, model_id=model_key)
    config_cls = get_config_class(family)
    config = config_cls()
    apply_image_registry_config_overrides(entry, config)
    assert_image_family_contract(family, config)
    runtime_contract = FamilyRuntimeContract(family=family, config=config)
    if bundle_root is None:
        bundle_root = local_bundle_root_fn(pipeline._project_root, entry, version_key or None)
    return ResolvedImageModel(
        model_key=model_key,
        version_key=version_key,
        entry=entry,
        family=family,
        config=config,
        runtime_contract=runtime_contract,
        bundle_root=bundle_root,
    )


def image_model_from_resolved_run(pipeline: Any, resolved: "ResolvedRun") -> ResolvedImageModel:
    """Map session ``ResolvedRun`` to image encode/schedule fields (no second registry lookup)."""
    return _build_resolved_image_model(
        pipeline,
        model_key=resolved.model_id,
        version_key=resolved.version_key,
        entry=resolved.registry_entry,
        bundle_root=resolved.bundle_root,
    )


def resolve_image_steps_guidance(
    entry: Any,
    request: Any,
    runtime_contract: FamilyRuntimeContract,
    *,
    steps_default: int = 4,
    guidance_default: float = 0.0,
) -> tuple[int, float, dict[str, Any]]:
    metadata = request.metadata or {}
    steps = int(request.steps) if request.steps is not None else int(
        registry_scalar_default(entry, "steps", steps_default)
    )
    steps = max(1, steps)
    if request.guidance is not None:
        guidance = float(request.guidance)
    else:
        guidance = float(registry_scalar_default(entry, "guidance", guidance_default))
    guidance = runtime_contract.resolve_guidance_scalar(guidance)
    return steps, guidance, metadata


def resolve_image_preview(entry: Any) -> tuple[str, int, int, str]:
    return resolve_image_preview_settings(entry)


@dataclass
class ImageEncodedModel:
    model: Any
    extra_cond: dict[str, Any]
    txt_embeds: Any
    neg_embeds: Any
    txt_attn_mask: Any
    neg_attn_mask: Any
    pooled_embeds: Any
    neg_pooled_embeds: Any
    encoder_type: str


def load_image_encoded_model(
    pipeline: Any,
    *,
    request: Any,
    resolved: ResolvedImageModel,
    steps: int,
    guidance: float,
    exec_ctx: Any,
    on_progress: Callable | None,
    on_log: Callable | None,
    preloaded_model: Any | None,
) -> ImageEncodedModel | None:
    validate_bundle_graph_step(
        resolved.bundle_root,
        family=resolved.family,
        model_id=resolved.model_key,
        on_log=on_log,
    )
    enc_loaded = image_encode_load_for_inference(
        pipeline,
        request=request,
        bundle_root=resolved.bundle_root,
        config=resolved.config,
        guidance=guidance,
        runtime_contract=resolved.runtime_contract,
        family=resolved.family,
        entry=resolved.entry,
        version_key=resolved.version_key,
        model_key=resolved.model_key,
        steps=steps,
        ctx_exec=exec_ctx,
        on_progress=on_progress,
        on_log=on_log,
        preloaded_model=preloaded_model,
    )
    if enc_loaded is None:
        return None
    (
        model,
        extra_cond,
        txt_embeds,
        neg_embeds,
        txt_attn_mask,
        neg_attn_mask,
        pooled_embeds,
        neg_pooled_embeds,
        encoder_type,
    ) = enc_loaded
    return ImageEncodedModel(
        model=model,
        extra_cond=extra_cond,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        txt_attn_mask=txt_attn_mask,
        neg_attn_mask=neg_attn_mask,
        pooled_embeds=pooled_embeds,
        neg_pooled_embeds=neg_pooled_embeds,
        encoder_type=encoder_type,
    )


@dataclass
class ImageScheduleBundle:
    semantics: Any
    scheduler: Any
    timesteps: list[Any]
    sigmas: Any | None
    sched_ts: Any
    timestep_embed_schedule: list[float] | None
    scheduler_name: str


def load_edit_source_rgb(
    pipeline: Any,
    ctx_exec: Any,
    source_asset_id: str,
) -> tuple[Any, int, int]:
    """Load edit source asset, align to 16px, center-crop. Returns ``(pil, w, h)``."""
    from PIL import Image

    src_path = ctx_exec.asset_store.get_file_path(source_asset_id)
    pil = Image.open(src_path).convert("RGB")
    w0, h0 = pil.size
    w, h = align_image_hw_multiples(w0, h0, align=16)
    pil = center_crop_image_pil(pil, w, h)
    return pil, w, h


def validate_edit_vae_latent_grid(
    *,
    family: str,
    config: Any,
    bundle_root: Path | None,
    height: int,
    width: int,
    ctx: Any | None = None,
) -> None:
    """Fail loud when rewrite img2img VAE grid does not match transformer latent grid."""
    from backend.engine.codecs import get_vae_encode_handler
    from backend.engine.common.codecs.vae import resolve_vae_bundle_config

    vae_scale = int(getattr(config, "vae_scale", 8))
    enc_spatial_div = 8
    latent_h, latent_w = height // vae_scale, width // vae_scale
    enc_h, enc_w = height // enc_spatial_div, width // enc_spatial_div

    vae_dir_pre = (bundle_root / "vae") if bundle_root else None
    vae_cfg_pre, _, _ = resolve_vae_bundle_config(vae_dir_pre, ctx=ctx)
    vae_cls_pre = str(vae_cfg_pre.get("_class_name") or "")
    if get_vae_encode_handler(vae_cls_pre, entry_family=family) is not None:
        return
    if (latent_h, latent_w) == (enc_h, enc_w):
        return
    raise RuntimeError(
        f"Image edit (rewrite) is not wired for vae_scale={vae_scale} when the VAE encoder "
        f"outputs a {enc_h}×{enc_w} latent grid but the transformer expects "
        f"{latent_h}×{latent_w} (image {width}×{height}). Models with vae_scale≠8 "
        f"(e.g. Flux2) need a dedicated encode / pack bridge before img2img can run."
    )


def encode_edit_reference_latent(
    pipeline: Any,
    *,
    pil: Any,
    entry: Any,
    version_key: str | None,
    config: Any,
    model_key: str,
    family: str,
    width: int,
    height: int,
    on_log: Callable | None,
) -> Any:
    img_f01 = pil_image_to_nchw_float01(pipeline, pil, width, height)
    encoded = image_vae_encode_tensor(
        pipeline,
        img_f01,
        entry,
        version_key or None,
        height_px=height,
        width_px=width,
        on_log=on_log,
    )
    if encoded.shape[1] != config.in_channels:
        raise RuntimeError(
            f"VAE encode produced {encoded.shape[1]} latent channels but model {model_key!r} "
            f"(family={family}) expects in_channels={config.in_channels}. "
            "Check bundle VAE config / model family alignment."
        )
    return encoded


def assert_edit_rewrite_schedule(
    *,
    scheduler_name: str,
    timesteps: list[Any],
    init_timestep: int,
    steps: int,
    fidelity: float,
    sigmas: Any | None,
) -> None:
    if init_timestep > 0 and (not timesteps or int(timesteps[0]) != int(init_timestep)):
        raise RuntimeError(
            f"Image edit (rewrite): scheduler {scheduler_name!r} did not honor "
            f"init_timestep={init_timestep} (got timesteps={timesteps!r}). "
            "reference img2img requires FlowMatchEuler / Linear / flow_match_euler_flux_dynamic "
            "/ flow_match_euler_cogview4."
        )
    if init_timestep >= steps:
        raise RuntimeError(
            f"Image edit (rewrite): source_fidelity={fidelity} implies init_timestep={init_timestep} "
            f">= steps={steps}; no denoising steps remain (reference path would also skip the loop)."
        )
    if init_timestep > 0 and sigmas is None:
        raise RuntimeError(
            "Image edit (rewrite): scheduler produced no sigmas; cannot build reference img2img latents."
        )


def prepare_edit_rewrite_latents(
    pipeline: Any,
    *,
    model: Any,
    config: Any,
    runtime_contract: FamilyRuntimeContract,
    encoded: Any,
    seed: int,
    init_timestep: int,
    sigmas: Any | None,
) -> Any:
    """Blend reference VAE latent with noise for rewrite img2img."""
    ctx = pipeline.ctx
    latent_dtype = runtime_contract.denoise_latent_noise_dtype(ctx)
    noise_dtype = runtime_contract.noise_sample_dtype(ctx, latent_dtype)

    if getattr(config, "encoder_step_kwargs", None) == "qwen_image":
        q_h = int(encoded.shape[2])
        q_w = int(encoded.shape[3])
        q_seq = q_h * q_w
        packed_noise = ctx.seeded_randn((1, q_seq, 64), seed, dtype=noise_dtype)
        if noise_dtype != latent_dtype:
            packed_noise = packed_noise.astype(latent_dtype)
        packed_noise = ctx.reshape(packed_noise, (1, q_h, q_w, 64))
        noise = ctx.permute(packed_noise, (0, 3, 1, 2))
    elif getattr(config, "latent_noise_packed", False):
        _, _, lh, lw = encoded.shape
        seq_len = (lh // 2) * (lw // 2)
        packed = ctx.seeded_randn((1, seq_len, 64), seed, dtype=noise_dtype)
        if noise_dtype != latent_dtype:
            packed = packed.astype(latent_dtype)
        noise = model.unpack_latents(ctx, packed, lh, lw)
    else:
        noise = runtime_contract.sample_edit_noise(
            ctx,
            encoded_shape=tuple(encoded.shape),
            seed=seed,
            sample_dtype=noise_dtype,
            target_dtype=latent_dtype,
        )

    if init_timestep == 0:
        latents = noise
    else:
        sig_blend = sigmas[init_timestep]
        latents = (1.0 - sig_blend) * encoded + sig_blend * noise
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(latents)
    return latents


def packed_edit_latent_dims(
    latents: Any,
    *,
    packed: bool,
) -> tuple[int, int]:
    if not packed:
        return 0, 0
    if latents.ndim == 3:
        _, seq_len, _ = latents.shape
        lh = int(seq_len ** 0.5)
        return lh, seq_len // max(lh, 1)
    if latents.ndim == 4:
        _, _, lh, lw = latents.shape
        return int(lh), int(lw)
    return 0, 0


def schedule_image_run(
    pipeline: Any,
    *,
    entry: Any,
    config: Any,
    request_scheduler: Any,
    metadata: dict[str, Any],
    steps: int,
    width: int,
    height: int,
    init_timestep: int = 0,
) -> ImageScheduleBundle:
    semantics = _IMAGE_SCHEDULER_SEMANTICS.resolve(
        entry=entry,
        config=config,
        request_scheduler=request_scheduler,
        request_metadata=metadata,
        steps=steps,
        width=width,
        height=height,
        init_timestep=init_timestep,
    )
    scheduler_name = semantics.scheduler_name
    scheduler = get_scheduler(scheduler_name, ctx=pipeline.ctx)
    timesteps = scheduler.set_timesteps(**semantics.set_timesteps_kwargs)
    return ImageScheduleBundle(
        semantics=semantics,
        scheduler=scheduler,
        timesteps=timesteps,
        sigmas=getattr(scheduler, "sigmas", None),
        sched_ts=getattr(scheduler, "timesteps", None),
        timestep_embed_schedule=timestep_embed_schedule_from_scheduler(scheduler),
        scheduler_name=scheduler_name,
    )


def log_inference_sigma_schedule(on_log: Callable | None, sigmas: Any) -> None:
    """Log the σ values the model denoises at, to compare against training σ distribution."""
    if on_log is None or sigmas is None:
        return
    try:
        sig = np.asarray(sigmas).reshape(-1).astype(np.float64)
        body = sig[:-1] if sig.size > 1 and float(sig[-1]) == 0.0 else sig
        if body.size == 0:
            return
        frac_lo = float((body < 0.3).mean())
        frac_mid = float(((body >= 0.3) & (body <= 0.7)).mean())
        frac_hi = float((body > 0.7).mean())
        vals = "[" + ", ".join("%.3f" % float(v) for v in body) + "]"
        on_log(
            "info",
            f"[diag] inference σ schedule (n={body.size}): σ={vals} "
            f"frac σ<0.3={frac_lo:.2f} 0.3-0.7={frac_mid:.2f} >0.7={frac_hi:.2f} "
            "(compare with training '[diag] train σ dist'; LoRA must be trained where inference denoises)",
        )
    except Exception:  # noqa: BLE001 — diagnostics must never break generation
        pass


def align_image_hw_multiples(w0: int, h0: int, *, align: int) -> tuple[int, int]:
    """Width/height floored to multiples of ``align`` (at least ``align``)."""
    w = max(align, (w0 // align) * align)
    h = max(align, (h0 // align) * align)
    return w, h

def center_crop_image_pil(pil: Any, w: int, h: int) -> Any:
    from PIL import Image

    w0, h0 = pil.size
    left = max(0, (w0 - w) // 2)
    top = max(0, (h0 - h) // 2)
    box = (left, top, min(left + w, w0), min(top + h, h0))
    cropped = pil.crop(box)
    if cropped.size != (w, h):
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    return cropped

def pil_image_to_nchw_float01(pipeline, pil: Any, w: int, h: int) -> Any:
    """Resize PIL RGB → float01 tensor ``[1,3,H,W]`` (NCHW)."""
    from PIL import Image

    if pil.size != (w, h):
        pil = pil.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr[None, ...]
    t = pipeline.ctx.array(arr)
    return pipeline.ctx.permute(t, (0, 3, 1, 2))

def load_image_vae_dir_cfg_weights(pipeline,
    entry,
    version_key: str | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    from backend.engine.common.codecs.vae import load_vae_weight_dict, resolve_vae_bundle_config

    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
    vae_dir = (bundle_root / "vae") if bundle_root else None
    if vae_dir is None or not vae_dir.exists():
        raise RuntimeError(f"VAE: no vae directory under bundle {bundle_root}")

    vae_weights = load_vae_weight_dict(pipeline.ctx, vae_dir)
    vae_cfg, _, _ = resolve_vae_bundle_config(vae_dir, vae_weights=vae_weights, ctx=pipeline.ctx)
    if not vae_weights:
        raise RuntimeError(f"VAE encode: no weights under {vae_dir}")
    return vae_dir, vae_cfg, vae_weights

def image_vae_encode_tensor(pipeline,
    image_nchw_f01: Any,
    entry,
    version_key: str | None,
    *,
    height_px: int | None = None,
    width_px: int | None = None,
    on_log: Callable | None = None,
) -> Any:
    """Encode ``[1,3,H,W]`` float **linear RGB in [0, 1]** → model latent (shape depends on VAE class).

    Applies **[-1, 1] pixel normalization** before ``conv_in`` (diffusers img2img).
    """
    from backend.engine.common.codecs.vae import VAEEncoder, prepare_vae_encoder_weight_items

    image_n11 = image_nchw_f01 * 2.0 - 1.0

    _, vae_cfg, vae_weights = load_image_vae_dir_cfg_weights(pipeline, entry, version_key)
    vae_cls = str(vae_cfg.get("_class_name") or "")
    entry_family = str(getattr(entry, "family", "") or "")

    encode_handler = get_vae_encode_handler(vae_cls, entry_family=entry_family)
    if encode_handler is not None:
        bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
        return encode_handler(
            ctx=pipeline.ctx,
            image_n11=image_n11,
            bundle_root=bundle_root,
            project_root=pipeline._project_root,
            height_px=height_px,
            width_px=width_px,
            on_log=on_log,
        )

    from backend.engine.common.codecs.vae import infer_latent_channels

    config = get_config_class(str(getattr(entry, "family", "") or ""))()
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
    shift_factor = float(vae_cfg.get("shift_factor", 0.0))

    latent_c = infer_latent_channels(vae_cfg, vae_weights)
    enc = VAEEncoder(
        latent_channels=latent_c,
        ctx=pipeline.ctx,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    enc_items = prepare_vae_encoder_weight_items(vae_weights)
    loaded, skipped = enc.load_weights(enc_items, strict=False)
    if (
        getattr(pipeline.ctx, "backend", None) == "mlx"
        and bool(getattr(config, "vae_encoder_cast_bfloat16", False))
    ):
        enc.cast_floating_params(pipeline.ctx.bfloat16())
    if on_log:
        on_log(
            "info",
            f"vae_encode loaded={len(loaded)} skipped={len(skipped)} latent_channels={latent_c}",
        )
    if not any(k.startswith("conv_in.") for k in loaded):
        raise RuntimeError(
            "VAE encoder failed to load conv_in weights; check bundle encoder.* tensors. "
            f"skipped_sample={skipped[:8]}"
        )

    latent5 = enc.encode(image_n11)
    z = latent5[:, :, 0, :, :] if getattr(latent5, "ndim", 0) == 5 else latent5
    return z

def encode_image_text_for_pipeline(pipeline,
    *,
    prompt: str,
    negative_prompt: str | None,
    bundle_root: Path | None,
    config: Any,
    guidance: float,
    runtime_contract: FamilyRuntimeContract,
    entry: Any | None = None,
    version_key: str | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, str]:
    use_cache = bool(getattr(config, "use_text_embedding_cache", True))
    encoder_id = f"{getattr(config, 'encoder_type', 'unknown')}:{version_key or 'default'}"
    if use_cache:
        from backend.engine.common.ops.text_embedding_cache import global_text_embedding_cache

        cached = global_text_embedding_cache().get(
            encoder_id=encoder_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
        )
        if cached is not None:
            return cached  # type: ignore[return-value]

    result = encode_image_text_conditioning(
        pipeline.ctx,
        prompt=prompt,
        negative_prompt=negative_prompt,
        bundle_root=bundle_root,
        config=config,
        guidance=guidance,
        encode_negative=runtime_contract.should_encode_negative_prompt(guidance),
        registry_entry=entry,
        registry_version_key=version_key,
    )
    if use_cache:
        from backend.engine.common.ops.text_embedding_cache import global_text_embedding_cache

        global_text_embedding_cache().put(
            encoder_id=encoder_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance=guidance,
            value=result,
        )
    return result

def image_encode_load_for_inference(pipeline,
    *,
    request: ImageGenerationRequest | ImageEditRequest,
    bundle_root: Path | None,
    config: Any,
    guidance: float,
    runtime_contract: FamilyRuntimeContract,
    family: str,
    entry: Any,
    version_key: str | None,
    model_key: str,
    steps: int,
    ctx_exec: ExecutionContext,
    on_progress: Callable | None,
    on_log: Callable | None,
    preloaded_model: Any | None = None,
) -> tuple[Any, dict[str, Any], Any, Any, Any, Any, Any, Any, str] | None:
    emit_phase(on_progress, phase="encoding", progress=0.02, n_steps=steps)
    pipeline_graph_step("encode_prompt", on_log)
    (
        txt_embeds,
        neg_embeds,
        txt_attn_mask,
        neg_attn_mask,
        pooled_embeds,
        neg_pooled_embeds,
        encoder_type,
    ) = encode_image_text_for_pipeline(pipeline, 
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        bundle_root=bundle_root,
        config=config,
        guidance=guidance,
        runtime_contract=runtime_contract,
        entry=entry,
        version_key=version_key,
    )
    if ctx_exec.cancel_token.is_cancelled():
        return None

    # TE weights released inside encode_image_text_conditioning; gc + MLX cache flush before DiT.
    import gc

    gc.collect()
    if hasattr(pipeline.ctx, "clear_cache"):
        pipeline.ctx.clear_cache()

    emit_phase(on_progress, phase="loading_model", progress=0.08, n_steps=steps)
    if preloaded_model is not None:
        model = preloaded_model
    else:
        from backend.engine.common.bundle.quant_inference import assert_quantized_dit_lora_compatible

        assert_quantized_dit_lora_compatible(
            entry, version_key or None, getattr(request, "adapters", None)
        )
        pipeline_graph_step("load_transformer", on_log)
        allow_cache = not (getattr(request, "adapters", None) or [])
        model = load_image_dit_model(
            pipeline,
            family,
            config,
            entry,
            version_key or None,
            allow_cache=allow_cache,
            on_log=on_log,
        )
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
    apply_image_lora_adapters(pipeline, family, model, request, on_log)
    extra_cond = model.prepare_conditioning(
        request, bundle_root=str(bundle_root) if bundle_root else None
    )
    return (
        model,
        extra_cond,
        txt_embeds,
        neg_embeds,
        txt_attn_mask,
        neg_attn_mask,
        pooled_embeds,
        neg_pooled_embeds,
        encoder_type,
    )

def finalize_image_from_latents(pipeline,
    *,
    latents: Any,
    timesteps: Any,
    entry: Any,
    version_key: str | None,
    model_key: str,
    seed: int,
    request: ImageGenerationRequest | ImageEditRequest,
    ctx_exec: ExecutionContext,
    steps: int,
    guidance: float,
    w: int,
    h: int,
    on_progress: Callable | None,
    on_log: Callable | None,
    name_infix: str = "",
    post_decode: Callable[[Any], Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    pipeline_graph_step("decode_vae", on_log)
    image = image_vae_decode(pipeline, latents, entry, version_key or None, on_log=on_log)
    if post_decode is not None:
        image = post_decode(image)
    emit_post_progress(on_progress, n_steps=len(timesteps), within_post=0.5)
    if ctx_exec.cancel_token.is_cancelled():
        return None

    pipeline_graph_step("save_asset", on_log)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = work / f"{model_key}{name_infix}_{seed}_{timestamp}.png"
    if hasattr(image, "save"):
        image.save(str(out_path))
    emit_post_progress(on_progress, n_steps=len(timesteps), within_post=1.0)
    emit_complete(on_progress, len(timesteps))

    meta: dict[str, Any] = {
        "model": request.model,
        "seed": seed,
        "prompt": request.prompt,
        "steps": steps,
        "guidance": guidance,
        "width": w,
        "height": h,
        "mime_type": "image/png",
    }
    if extra_meta:
        meta.update(extra_meta)
    meta.update(work_title_metadata(request.title))
    return str(out_path), meta

def build_image_vae_preview_session(pipeline,
    entry: Any,
    version_key: str | None,
    *,
    on_log: Callable | None = None,
) -> dict[str, Any] | None:
    from backend.engine.common.codecs.vae import build_standard_vae_preview_session, read_vae_dir_config

    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
    vae_dir = (bundle_root / "vae") if bundle_root else None
    vae_cfg, _, _ = read_vae_dir_config(vae_dir)
    vae_cls = str(vae_cfg.get("_class_name") or "")
    entry_family = str(getattr(entry, "family", "") or "")
    if get_vae_decode_handler(vae_cls, entry_family=entry_family) is not None:
        return None
    return build_standard_vae_preview_session(pipeline.ctx, vae_dir, on_log=on_log)

def image_vae_decode_with_preview_session(pipeline,
    latents: Any,
    entry: Any,
    version_key: str | None,
    preview_state: dict[str, Any],
    *,
    on_log: Callable | None = None,
) -> Any:
    from backend.engine.common.codecs.vae import (
        apply_flux2_latent_preprocess_if_enabled,
        reshape_packed_latents_to_nchw,
        vae_forward_to_pil,
    )

    session = preview_state.get("vae_session")
    if session is None:
        try:
            session = build_image_vae_preview_session(pipeline, 
                entry, version_key, on_log=on_log
            )
        except Exception as exc:
            preview_state["vae_session"] = False
            if on_log:
                on_log("warning", f"preview VAE session build failed: {exc}")
            return image_vae_decode(pipeline, 
                latents, entry, version_key, on_log=on_log
            )
        preview_state["vae_session"] = session if session else False

    if not session or session is False:
        return image_vae_decode(pipeline, latents, entry, version_key, on_log=on_log)

    from backend.engine.common.codecs.vae import reshape_packed_latents_to_nchw

    z = reshape_packed_latents_to_nchw(latents_for_image_vae_preview(latents))

    if session.get("use_special_preprocess"):
        from backend.engine.common.codecs.vae import apply_flux2_latent_preprocess_if_enabled

        z, _, _ = apply_flux2_latent_preprocess_if_enabled(
            pipeline.ctx,
            z,
            session["vae_cfg"],
            session["vae_weights"],
            session["orig_scaling"],
            session["orig_shift"],
        )

    return vae_forward_to_pil(pipeline.ctx, session["vae"], z)

def latents_for_image_vae_preview(latents: Any) -> Any:
    z = latents
    if getattr(z, "ndim", None) == 5 and int(z.shape[2]) == 1:
        z = z[:, :, 0, :, :]
    return z

def warm_image_step_preview_decoders(pipeline,
    entry: Any,
    version_key: str | None,
    preview_state: dict[str, Any],
    *,
    config: Any = None,
    preview_decoder: str = "auto",
    on_log: Callable | None = None,
) -> None:
    if preview_decoder in ("auto", "taesd") and preview_state.get("taesd_decoder") is None:
        from backend.engine.common.codecs.vae.taesd_mlx import load_taesd_preview_decoder

        family = str(getattr(entry, "family", "") or "")
        variant = "taef2" if family in ("flux2", "ernie_image") else "taef1"
        try:
            preview_state["taesd_decoder"] = load_taesd_preview_decoder(
                pipeline.ctx,
                project_root=Path(pipeline._project_root),
                variant=variant,
                on_log=on_log,
            )
        except Exception as exc:
            preview_state["taesd_decoder"] = False
            if on_log:
                on_log("warning", f"TAESD preview warmup failed: {exc}")
        if preview_decoder == "taesd" and not preview_state.get("taesd_decoder"):
            raise RuntimeError(
                "preview_decoder=taesd but TAESD weights were not found. "
                "Place taef1_decoder.safetensors under models/taesd/ or set DANQING_TAESD_DIR."
            )

    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
    if bool(getattr(config, "vae_preview_warmup", False)) and bundle_root:
        vae_dir = bundle_root / "vae"
        from backend.engine.common.codecs.vae import read_vae_dir_config

        vae_cfg, _, _ = read_vae_dir_config(vae_dir if vae_dir.is_dir() else None)
        vae_cls = str(vae_cfg.get("_class_name") or "")
        entry_family = str(getattr(entry, "family", "") or "")
        if preview_state.get("vae_preview_model") is None and get_vae_preview_warmup_handler(
            vae_cls, entry_family=entry_family
        ) is not None:
            try:
                preview_state["vae_preview_model"] = warmup_vae_preview(
                    pipeline.ctx,
                    bundle_root=bundle_root,
                    vae_class_name=vae_cls,
                    entry_family=entry_family,
                    on_log=on_log,
                )
            except Exception as exc:
                preview_state["vae_preview_model"] = False
                if on_log:
                    on_log("warning", f"VAE preview warmup failed: {exc}")

def decode_latents_for_image_step_preview(pipeline,
    latents: Any,
    entry: Any,
    version_key: str | None,
    preview_state: dict[str, Any],
    *,
    packed_denoise: bool,
    flux_unpack: Callable[..., Any] | None,
    latent_h: int,
    latent_w: int,
    on_log: Callable | None = None,
) -> Any:
    """Same decode semantics as final frame; prefer TAESD then warmed VAE session."""
    decode_latents = latents_for_image_vae_preview(latents)
    if packed_denoise and flux_unpack is not None:
        decode_latents = flux_unpack(pipeline.ctx, latents, latent_h, latent_w)
        decode_latents = latents_for_image_vae_preview(decode_latents)

    taesd = preview_state.get("taesd_decoder")
    if taesd not in (None, False):
        from backend.engine.common.codecs.vae import read_vae_dir_config
        from backend.engine.common.codecs.vae.taesd_mlx import decode_taesd_preview

        bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        vae_cfg, scaling_factor, shift_factor = read_vae_dir_config(vae_dir if vae_dir and vae_dir.is_dir() else None)
        rgb = decode_taesd_preview(
            pipeline.ctx,
            taesd,
            decode_latents,
            vae_scaling_factor=float(scaling_factor),
            vae_shift_factor=float(shift_factor),
        )
        arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        if arr.shape[-1] == 3:
            arr = arr[..., ::-1]
        from PIL import Image

        return Image.fromarray(arr)

    flux2_vae = preview_state.get("vae_preview_model")
    if flux2_vae not in (None, False):
        _, vae_dir_cfg, _ = load_image_vae_dir_cfg_weights(pipeline, entry, version_key)
        vae_cls = str(vae_dir_cfg.get("_class_name") or "")
        entry_family = str(getattr(entry, "family", "") or "")
        return decode_vae_preview(
            pipeline.ctx,
            warmed_model=flux2_vae,
            latents=decode_latents,
            vae_class_name=vae_cls,
            entry_family=entry_family,
            on_log=on_log,
        )

    session = preview_state.get("vae_session")
    if session and session is not False:
        return image_vae_decode_with_preview_session(pipeline, 
            decode_latents,
            entry,
            version_key,
            preview_state,
            on_log=on_log,
        )

    return image_vae_decode(pipeline, decode_latents, entry, version_key, on_log=on_log)

def maybe_emit_image_step_preview(pipeline,
    *,
    step_index_0based: int,
    n_steps: int,
    latents: Any,
    entry: Any,
    version_key: str | None,
    ctx_exec: ExecutionContext,
    on_progress: Callable[..., None] | None,
    preview_interval: int,
    preview_max_edge: int,
    preview_state: dict[str, Any],
    packed_denoise: bool,
    flux_unpack: Callable[..., Any] | None,
    latent_h: int,
    latent_w: int,
) -> None:
    interval = max(1, int(preview_interval))
    step_1 = step_index_0based + 1
    is_last = step_1 >= n_steps
    if step_1 > 1 and step_1 % interval != 0 and not is_last:
        return
    step_log = preview_state.get("on_log")
    try:
        from PIL import Image

        image = decode_latents_for_image_step_preview(pipeline, 
            latents,
            entry,
            version_key,
            preview_state,
            packed_denoise=packed_denoise,
            flux_unpack=flux_unpack,
            latent_h=latent_h,
            latent_w=latent_w,
            on_log=step_log,
        )
        if image is None:
            return
        if not hasattr(image, "save"):
            return
        pil = image
        if max(pil.size) > preview_max_edge:
            pil = pil.copy()
            pil.thumbnail(
                (preview_max_edge, preview_max_edge),
                Image.Resampling.BILINEAR,
            )
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / "preview_latest.png"
        pil.save(str(out_path), format="PNG", optimize=True)
        nbytes = out_path.stat().st_size if out_path.is_file() else 0
        n = max(1, int(n_steps))
        p = DENOISE_PROGRESS_SHARE * (step_1 / n)
        # Step preview image: frontend polls GET /api/tasks/{task_id}/preview (not SSE).
        if on_progress is not None:
            on_progress(p, step_1, n, None, "denoising")
        if preview_state.get("on_log"):
            preview_state["on_log"](
                "info",
                f"preview step {step_1}/{n_steps} saved {nbytes} bytes -> {out_path.name}",
            )
        if getattr(pipeline.ctx, "backend", None) == "mlx":
            pipeline.ctx.eval()
    except Exception as exc:
        shape = getattr(latents, "shape", None)
        if preview_state.get("on_log"):
            preview_state["on_log"](
                "error",
                f"step preview failed at {step_1}/{n_steps}: {exc} latent_shape={shape}",
            )

def apply_image_lora_adapters(pipeline,
    family: str,
    model: Any,
    request: ImageGenerationRequest | ImageEditRequest,
    on_log: Callable[..., None] | None,
) -> None:
    adapters = getattr(request, "adapters", None) or []
    if not adapters:
        return
    base_model_id, _ = parse_model_version(request.model)
    entry = pipeline._registry.get(base_model_id)
    if entry is not None:
        lora_support = _registry_scalar_default_fn(entry, "lora_support", False)
        if not lora_support:
            raise RuntimeError(
                f"Model {base_model_id!r} does not declare LoRA support; "
                "remove adapters from the request or use a LoRA-capable base model."
            )
    from backend.catalog.lora_meta import expand_z_image_scheme4_adapters
    from backend.engine.runtime.mlx import MLXContext

    if not isinstance(pipeline.ctx, MLXContext):
        raise RuntimeError(
            "LoRA merging for Flux.1 / Flux2 / Z-Image / Qwen Image is only implemented on the MLX runtime; "
            f"current runtime is {type(pipeline.ctx).__name__}."
        )
    config_dir = pipeline._project_root / "config"
    expanded = expand_z_image_scheme4_adapters(
        adapters,
        base_model_id=base_model_id,
        registry=pipeline._registry,
        config_dir=config_dir,
        project_root=pipeline._project_root,
    )
    if len(expanded) > len(adapters) and on_log is not None:
        on_log(
            "info",
            "Scheme 4: appended Z-Image-Turbo DistillPatch LoRA for Base-trained adapter "
            "(official acceleration: 8 steps, cfg_scale=1 / guidance=0, linear scheduler).",
        )
    _merge_image_lora_adapters(
        family=family,
        model=model,
        adapters=expanded,
        base_model_id=base_model_id,
        project_root=pipeline._project_root,
        registry=pipeline._registry,
        ctx=pipeline.ctx,
        on_log=on_log,
    )

def load_image_dit_model(
    pipeline,
    family: str,
    config,
    entry,
    version_key: str | None,
    *,
    allow_cache: bool = True,
    on_log: Callable | None = None,
):
    return load_image_transformer(
        ctx=pipeline.ctx,
        family=family,
        config=config,
        entry=entry,
        version_key=version_key,
        project_root=pipeline._project_root,
        model_cache=pipeline._cache,
        allow_cache=allow_cache,
        on_log=on_log,
    )

def image_vae_decode(pipeline,
    latents,
    entry,
    version_key,
    *,
    on_log: Callable | None = None,
):
    """VAE decode latent → PIL Image."""
    ctx = pipeline.ctx
    from backend.engine.common.bundle.quant_inference import resolve_component_inference_weight_mode
    from backend.engine.common.bundle.safetensors_affine_quant import read_bundle_affine_bits_if_quantized
    from backend.engine.common.codecs.vae import (
        apply_flux2_latent_preprocess_if_enabled,
        create_loaded_vae_decoder,
        load_vae_weight_dict,
        read_vae_dir_config,
        release_vae_decoder_memory,
        reshape_packed_latents_to_nchw,
        vae_forward_to_pil,
        vae_output_to_uint8_hwc,
    )
    from PIL import Image

    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
    vae_dir = (bundle_root / "vae") if bundle_root else None
    vae_cfg, scaling_factor, shift_factor = read_vae_dir_config(vae_dir)
    latents = reshape_packed_latents_to_nchw(latents)

    vae_cls = str(vae_cfg.get("_class_name") or "")
    entry_family = str(getattr(entry, "family", "") or "") if entry is not None else ""
    decode_handler = get_vae_decode_handler(vae_cls, entry_family=entry_family)
    if decode_handler is not None:
        return decode_handler(
            ctx=pipeline.ctx,
            latents=latents,
            bundle_root=bundle_root,
            project_root=pipeline._project_root,
            on_log=on_log,
            vae_output_to_uint8_hwc=vae_output_to_uint8_hwc,
            image_cls=Image,
        )

    vae_weights = load_vae_weight_dict(pipeline.ctx, vae_dir, fail_if_config_only=True)
    bundle_affine_bits = read_bundle_affine_bits_if_quantized(vae_weights, vae_dir or Path("."))
    vae_inference_mode = resolve_component_inference_weight_mode(
        entry,
        version_key,
        ctx,
        component="vae",
        weight_keys=frozenset(vae_weights.keys()),
        bundle_affine_bits=bundle_affine_bits,
    )
    latents, scaling_factor, shift_factor = apply_flux2_latent_preprocess_if_enabled(
        ctx, latents, vae_cfg, vae_weights, scaling_factor, shift_factor
    )
    release_after = bool(_registry_scalar_default_fn(entry, "vae_release_after_decode", True))
    vae = None
    try:
        vae, decoder_w, loaded, skipped = create_loaded_vae_decoder(
            ctx,
            latents,
            vae_weights,
            scaling_factor,
            shift_factor,
            vae_cfg=vae_cfg,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=vae_inference_mode,
        )
        if not decoder_w:
            raise RuntimeError(
                f"VAE weights under {vae_dir} produced no decoder tensors after remap; check bundle."
            )
        if on_log:
            on_log(
                "info",
                " ".join(
                    [
                        f"vae_decode latent_shape={tuple(latents.shape)}",
                        f"{vae_inference_mode.log_label()}",
                        f"scaling_factor={scaling_factor}",
                        f"shift_factor={shift_factor}",
                        f"decoder_tensors={len(decoder_w)}",
                        f"loaded_params={len(loaded)}",
                        f"skipped_params={len(skipped)}",
                    ]
                ),
            )
        return vae_forward_to_pil(ctx, vae, latents)
    finally:
        if release_after:
            release_vae_decoder_memory(ctx, vae)
            if on_log:
                on_log("info", "vae_released_after_decode=yes")


def uses_family_image_generator(config: Any) -> bool:
    return str(getattr(config, "image_pipeline_shape", "dit_standard") or "dit_standard") == "family_generator"


def execute_family_image_generator(
    pipeline: Any,
    request: ImageGenerationRequest | ImageEditRequest,
    ctx_exec: ExecutionContext,
    *,
    is_edit: bool,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | list[tuple[str, dict[str, Any]]] | None:
    """Shape C image generation — family-owned stack (T2I / edit / multi-ref)."""
    import random

    from backend.core.contracts import parse_size, work_title_metadata
    from backend.engine._transformer_registry import get_image_generation_factory, validate_image_generation_params
    from backend.engine.pipelines.pipeline_progress import emit_complete, pipeline_graph_step

    model_key, version_key = parse_model_version(request.model)
    entry = pipeline._registry.require(model_key)
    family = require_entry_family(entry, model_id=model_key)
    config_cls = get_config_class(family)
    config = config_cls()
    apply_image_registry_config_overrides(entry, config)

    if not uses_family_image_generator(config):
        raise RuntimeError(
            f"Internal error: execute_family_image_generator called for "
            f"image_pipeline_shape={getattr(config, 'image_pipeline_shape', '')!r} "
            f"(family={family!r})"
        )

    if ctx_exec.cancel_token.is_cancelled():
        return None

    size_raw = getattr(request, "size", None)
    if is_edit and not size_raw:
        src_id = getattr(request, "source_asset_id", None)
        if not src_id:
            raise RuntimeError("Image edit requires source_asset_id")
        src = pipeline._asset_store.get_file_path(src_id)
        if src is None or not src.exists():
            raise RuntimeError(f"Source image asset not found: {src_id!r}")
        from PIL import Image

        with Image.open(src) as im:
            w, h = im.size
    else:
        w, h = parse_size(size_raw or "1024x1024")
    steps = int(getattr(request, "steps", None) or registry_scalar_default(entry, "steps", 28) or 28)
    guidance = float(getattr(request, "guidance", None) or registry_scalar_default(entry, "guidance", 1.0) or 1.0)
    seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
    n = max(1, int(getattr(request, "n", 1) or 1))

    bundle_root = _local_bundle_root_fn(pipeline._project_root, entry, version_key)
    if bundle_root is None or not bundle_root.is_dir():
        raise RuntimeError(f"Image model bundle not found for {request.model!r}")

    ref_paths: list[str] = []
    if is_edit:
        src_id = getattr(request, "source_asset_id", None)
        if not src_id:
            raise RuntimeError("Image edit requires source_asset_id")
        src = pipeline._asset_store.get_file_path(src_id)
        if src is None or not src.exists():
            raise RuntimeError(f"Source image asset not found: {src_id!r}")
        ref_paths = [str(src)]
    else:
        ref_ids = list(getattr(request, "reference_asset_ids", None) or [])
        meta = getattr(request, "metadata", None) or {}
        if not ref_ids and isinstance(meta.get("reference_asset_ids"), list):
            ref_ids = [str(x) for x in meta["reference_asset_ids"]]
        max_refs = int(getattr(config, "hidream_edit_max_refs", 3) or getattr(config, "edit_max_reference_images", 3) or 3)
        for aid in ref_ids[:max_refs]:
            p = pipeline._asset_store.get_file_path(aid)
            if p is None or not p.exists():
                raise RuntimeError(f"Reference image asset not found: {aid!r}")
            ref_paths.append(str(p))

    validate_image_generation_params(
        family=family,
        entry=entry,
        config=config,
        ref_image_paths=ref_paths or None,
    )

    factory = get_image_generation_factory(family)
    gen = factory(
        pipeline.ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
    )

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    f"infer model={model_key}",
                    f"family={family}",
                    f"image_pipeline_shape=family_generator",
                    f"size={w}x{h}",
                    f"steps={steps}",
                    f"seed={seed}",
                    f"edit={is_edit}",
                    f"refs={len(ref_paths)}",
                ]
            ),
        )

    pipeline_graph_step("load_model", on_log)
    gen.load()

    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    from backend.engine._transformer_registry import get_image_output_path_resolver

    resolve_out = get_image_output_path_resolver(family)
    neg = getattr(request, "negative_prompt", None) or ""

    results: list[tuple[str, dict[str, Any]]] = []
    for i in range(n):
        if ctx_exec.cancel_token.is_cancelled():
            return results if results else None
        batch_seed = seed + i
        out_path = resolve_out(work, model_key, batch_seed)

        batch_progress = on_progress
        if n > 1 and on_progress is not None:
            from backend.engine.pipelines.image_create_phases import _scale_progress

            batch_progress = _scale_progress(on_progress, i, n)

        pipeline_graph_step("denoise", on_log)
        snap_resolution = bool(getattr(config, "hidream_snap_resolution", True))
        if os.environ.get("DANQING_BENCH_EVAL") == "1" and hasattr(
            config, "hidream_snap_resolution"
        ):
            snap_resolution = False
        saved = gen.generate_and_save(
            prompt=request.prompt,
            output_path=out_path,
            width=w,
            height=h,
            seed=batch_seed,
            steps=steps,
            guidance=guidance,
            negative_prompt=neg,
            ref_image_paths=ref_paths or None,
            snap_resolution=snap_resolution,
            blend_seams=int(getattr(config, "hidream_blend_seams", 0) or 0),
            on_log=on_log,
            on_progress=batch_progress,
            cancel_token=ctx_exec.cancel_token,
        )

        pipeline_graph_step("save_asset", on_log)
        metadata = {
            "model": request.model,
            "seed": batch_seed,
            "prompt": request.prompt,
            "negative_prompt": neg,
            "steps": steps,
            "guidance": guidance,
            "width": w,
            "height": h,
            "mime_type": "image/png",
        }
        if is_edit and getattr(request, "source_asset_id", None):
            metadata["source_asset_id"] = request.source_asset_id
        metadata.update(work_title_metadata(getattr(request, "title", None)))
        results.append((saved, metadata))

    emit_complete(on_progress, n_steps=steps)
    if n == 1:
        return results[0]
    return results
