"""Video-family runtime semantics — config-driven (no ``family ==`` in helpers)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def video_encoder_type(config: Any) -> str:
    enc = getattr(config, "encoder_type", None)
    if enc:
        return str(enc)
    fallback = getattr(config, "default_encoder_type", None)
    if fallback:
        return str(fallback)
    return "t5"


def video_t5_max_seq_len(config: Any) -> int:
    if hasattr(config, "max_text_seq_length"):
        return int(getattr(config, "max_text_seq_length", 226))
    if hasattr(config, "text_len"):
        return int(getattr(config, "text_len", 512))
    return 512


def _merge_hunyuan_bundle(config: Any, bundle_root: Path | None) -> None:
    from backend.engine.config.model_configs import (
        HunyuanVideoConfig,
        merge_hunyuan_transformer_config_from_bundle,
    )

    if isinstance(config, HunyuanVideoConfig):
        merge_hunyuan_transformer_config_from_bundle(config, bundle_root)


def _merge_wan_bundle(config: Any, bundle_root: Path | None) -> None:
    from backend.engine.config.model_configs import WanConfig, merge_wan_bundle_config

    if isinstance(config, WanConfig):
        merge_wan_bundle_config(config, bundle_root)


_BUNDLE_CONFIG_MERGE: dict[str, Callable[[Any, Path | None], None]] = {
    "hunyuan": _merge_hunyuan_bundle,
    "wan": _merge_wan_bundle,
}


def merge_video_bundle_config(config: Any, bundle_root: Path | None) -> None:
    merger = str(getattr(config, "bundle_config_merger", "") or "")
    if not merger:
        return
    fn = _BUNDLE_CONFIG_MERGE.get(merger)
    if fn is None:
        raise RuntimeError(f"Unknown video bundle_config_merger: {merger!r}")
    fn(config, bundle_root)


def video_scheduler_ctor_kwargs(
    config: Any,
    scheduler_name: str,
    bundle_root: Path | None,
) -> dict[str, Any]:
    extras = str(getattr(config, "scheduler_bundle_extras", "") or "")
    if not extras or scheduler_name != extras:
        return {}
    if extras == "wan_flow_unipc":
        ctor_kwargs: dict[str, Any] = {
            "num_train_timesteps": 1000,
            "use_corrector": getattr(config, "wan_scheduler_use_corrector", False),
            "velocity_scale": getattr(config, "wan_velocity_scale", None),
        }
        if bundle_root is not None:
            sched_cfg = bundle_root / "scheduler" / "scheduler_config.json"
            if sched_cfg.is_file():
                try:
                    data = json.loads(sched_cfg.read_text(encoding="utf-8"))
                    if "flow_shift" in data:
                        ctor_kwargs["shift"] = float(data["flow_shift"])
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"Wan: cannot read scheduler config {sched_cfg}: {e}"
                    ) from e
        return ctor_kwargs
    raise RuntimeError(f"Unknown video scheduler_bundle_extras: {extras!r}")


def video_cfg_negative_prompt(config: Any, negative_prompt: str | None) -> str:
    style = str(getattr(config, "cfg_negative_prompt_style", "default") or "default")
    if style == "wan":
        from backend.engine.families.wan.conditioning import WAN_SAMPLE_NEG_PROMPT

        return (negative_prompt or "").strip() or WAN_SAMPLE_NEG_PROMPT
    return negative_prompt.strip() if negative_prompt else " "


def video_resolve_shift_value(
    config: Any,
    *,
    request_shift: Any | None,
    registry_shift: Any | None,
    scheduler_default_shift: Any | None,
    on_log: Callable | None = None,
) -> float | None:
    if bool(getattr(config, "uses_wan_shift", False)):
        return resolve_wan_shift_value(
            request_shift=request_shift,
            registry_shift=registry_shift,
            scheduler_default_shift=scheduler_default_shift,
            on_log=on_log,
        )
    if request_shift is not None:
        return float(request_shift)
    if registry_shift is not None:
        return float(registry_shift)
    return None


def resolve_wan_shift_value(
    *,
    request_shift: Any | None,
    registry_shift: Any | None,
    scheduler_default_shift: Any | None,
    on_log: Callable | None = None,
) -> float | None:
    req = float(request_shift) if request_shift is not None else None
    reg = float(registry_shift) if registry_shift is not None else None
    sch = float(scheduler_default_shift) if scheduler_default_shift is not None else None
    eps = 1e-6

    if req is not None:
        return req
    if reg is not None:
        return reg
    if sch is not None and abs(sch - 1.0) > eps:
        return sch
    if on_log is not None:
        on_log("warning", "Wan shift unresolved; using scheduler default")
    return sch


def video_rotary_model_kwargs(
    config: Any,
    ctx: Any,
    pixel_h: int,
    pixel_w: int,
    latents: Any,
) -> dict[str, Any]:
    return {}


def video_infer_log_extras(config: Any, scheduler: Any, extra_cond: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    if bool(getattr(config, "uses_wan_shift", False)) and extra_cond.get("wan_expand_timesteps") is not None:
        parts.append(
            f"wan_expand_timesteps={bool(extra_cond.get('wan_expand_timesteps', False))}"
        )
    if bool(getattr(config, "uses_prediction_type", False)) and hasattr(scheduler, "_prediction_type"):
        parts.append(f"sched_prediction={scheduler._prediction_type}")
        parts.append(f"sched_spacing={getattr(scheduler, '_timestep_spacing', '?')}")
    return parts


def video_validate_generate_geometry(config: Any, w: int, h: int, num_frames: int) -> None:
    if w <= 0 or h <= 0:
        raise RuntimeError(f"invalid video size {w}x{h}")
    vae_sf = int(getattr(config, "vae_scale", 8) or 8)
    mode = str(getattr(config, "geometry_check", "generic") or "generic")
    tvs = int(getattr(config, "temporal_vae_scale", 0) or 0)

    if mode == "wan" and tvs > 0:
        if (num_frames - 1) % tvs != 0:
            raise RuntimeError(
                f"Wan requires (num_frames - 1) % {tvs} == 0; got {num_frames} "
                f"(valid examples: 81, 49 for temporal VAE scale {tvs})"
            )
        _, ph, pw = getattr(config, "patch_size", (1, 2, 2))
        step_h = vae_sf * int(ph)
        step_w = vae_sf * int(pw)
        if h % step_h != 0 or w % step_w != 0:
            raise RuntimeError(
                f"Wan requires height % {step_h} == 0 and width % {step_w} == 0 "
                f"(vae_scale={vae_sf}, patch_size={getattr(config, 'patch_size', (1, 2, 2))}); "
                f"got {w}x{h}. Example valid size: 480x704 for 480x720 intent."
            )
    if w % vae_sf != 0 or h % vae_sf != 0:
        raise RuntimeError(
            f"video width and height must be divisible by vae_scale={vae_sf}; got {w}x{h}"
        )
    if tvs > 0 and mode != "wan" and (num_frames - 1) % tvs != 0:
        raise RuntimeError(
            f"video requires (num_frames - 1) % {tvs} == 0; got {num_frames}"
        )


def video_snap_pixel_dims_if_needed(
    config: Any,
    w: int,
    h: int,
    *,
    on_log: Callable | None = None,
) -> tuple[int, int]:
    if not bool(getattr(config, "snap_pixel_dims", False)):
        return w, h
    from backend.engine.families.wan.conditioning import snap_wan_pixel_dims

    vae_scale = int(getattr(config, "vae_scale", 16) or 16)
    patch_size = tuple(getattr(config, "patch_size", (1, 2, 2)))
    ow, oh = w, h
    w, h = snap_wan_pixel_dims(w, h, vae_scale=vae_scale, patch_size=patch_size)
    if (w, h) != (ow, oh) and on_log is not None:
        on_log("info", f"Wan adjusted output size {ow}x{oh} -> {w}x{h} for patch/VAE alignment")
    return w, h


def wan_t5_bundle_paths(bundle_root: Path) -> tuple[str, str]:
    from backend.engine.families.wan.text_encoder_mlx import resolve_wan_umt5_pth

    assets = resolve_wan_umt5_pth(bundle_root)
    if assets is None:
        raise RuntimeError(
            f"Wan T5 assets not found under {bundle_root}. Expected "
            f"``models_t5_umt5-xxl-enc-bf16.pth`` (or ``models_t5*.pth``) and "
            f"``google/umt5-xxl`` tokenizer."
        )
    pth_path, tok_dir = assets
    return str(pth_path), str(tok_dir)


def _apply_hunyuan_i2v(ctx: Any, latents: Any, vae_latent: Any, extra_cond: dict[str, Any]) -> None:
    B, C, T, H, W = latents.shape
    cond_lat = ctx.zeros((B, C, T, H, W), dtype=latents.dtype)
    cond_lat = ctx.concat([vae_latent[:, :, :1, :, :], cond_lat[:, :, 1:, :, :]], axis=2)
    mask = ctx.zeros((B, 1, T, H, W), dtype=latents.dtype)
    mask_slice = ctx.ones((B, 1, 1, H, W), dtype=latents.dtype)
    mask = ctx.concat([mask_slice, mask[:, :, 1:, :, :]], axis=2)
    extra_cond["cond_latents"] = cond_lat
    extra_cond["mask_concat"] = mask
    extra_cond["i2v_mode"] = True


def _apply_wan_i2v(ctx: Any, latents: Any, vae_latent: Any, extra_cond: dict[str, Any]) -> None:
    from backend.engine.families.wan.conditioning import expand_wan_cond_latent, masks_like

    z = ctx.squeeze(vae_latent, 0)
    noise_cthw = ctx.squeeze(latents, 0)
    z = expand_wan_cond_latent(ctx, z, int(noise_cthw.shape[1]))
    _, mask2 = masks_like(ctx, [noise_cthw], zero=True)
    extra_cond["wan_cond_latent"] = z
    extra_cond["wan_i2v_mask"] = mask2[0]
    extra_cond["wan_i2v"] = True


def _apply_ltx_i2v(ctx: Any, latents: Any, vae_latent: Any, extra_cond: dict[str, Any]) -> Any:
    """LTX I2V: pass encoded latent to transformer hooks via extra_cond.

    Unlike concat/hunyuan/wan, LTX uses ``LTXLatentState`` + ``denoise_mask``
    managed by ``LTXTransformer.before_denoise`` / ``reblend_i2v_latents``.
    """
    extra_cond["ltx_i2v_cond_latent"] = vae_latent
    return latents


def video_apply_i2v_conditioning(
    config: Any,
    ctx: Any,
    latents: Any,
    vae_latent: Any,
    extra_cond: dict[str, Any],
) -> Any:
    style = str(getattr(config, "video_i2v_style", "concat") or "concat")
    if style == "hunyuan":
        _apply_hunyuan_i2v(ctx, latents, vae_latent, extra_cond)
        return latents
    if style == "wan":
        _apply_wan_i2v(ctx, latents, vae_latent, extra_cond)
        return latents
    if style == "ltx":
        return _apply_ltx_i2v(ctx, latents, vae_latent, extra_cond)
    if style == "concat":
        return ctx.concat([vae_latent[:, :, :1, :, :], latents[:, :, 1:, :, :]], axis=2)
    raise RuntimeError(f"Unknown video_i2v_style: {style!r}")


def resolve_video_edit_source_image(
    asset_store: Any,
    asset_id: str,
    *,
    mode: str,
    work_dir: Path | None = None,
):
    """Load animate/edit source as PIL RGB — image assets directly, video assets via first frame."""
    from PIL import Image

    record = asset_store.get_asset_record(asset_id) if hasattr(asset_store, "get_asset_record") else None
    src_path = asset_store.get_file_path(asset_id)
    if not src_path or not Path(src_path).exists():
        raise RuntimeError(f"Video edit source asset not found: {asset_id!r}")

    mime = str((record or {}).get("mime_type") or "").lower()
    kind = str((record or {}).get("kind") or "").lower()
    is_video = mime.startswith("video/") or kind == "video"
    if is_video and str(mode or "image_only") == "first_frame":
        from backend.engine.common.codecs.vae.video_stitch import extract_first_frame_image

        if work_dir is None:
            raise RuntimeError("video_edit_source_mode=first_frame requires a work_dir for frame extraction.")
        out = Path(work_dir) / f"edit_source_{asset_id}_frame0.png"
        extract_first_frame_image(Path(src_path), output_path=out)
        return Image.open(out).convert("RGB")
    if is_video and str(mode or "image_only") == "source_video":
        raise RuntimeError(
            f"Video edit source {asset_id!r} is a video asset; use load_video_source_path for full V2V."
        )
    if is_video:
        raise RuntimeError(
            f"Video edit source {asset_id!r} is a video asset; set registry video_edit_source_mode=first_frame "
            "or provide an image source_asset_id."
        )
    return Image.open(str(src_path)).convert("RGB")


def resolve_video_reference_asset_paths(asset_store: Any, asset_ids: list[str]) -> list[str]:
    """Resolve gallery asset ids to local file paths (fail loud on missing)."""
    paths: list[str] = []
    for asset_id in asset_ids:
        aid = str(asset_id or "").strip()
        if not aid:
            continue
        src = asset_store.get_file_path(aid)
        if src is None or not Path(src).exists():
            raise RuntimeError(f"Reference asset not found: {aid!r}")
        paths.append(str(src))
    return paths


def resolve_bernini_source_paths(
    asset_store: Any,
    request: Any,
    config: Any,
    *,
    work_dir: Path | None = None,
) -> tuple[str | None, str | None]:
    """Bernini edit sources: ``(full_video_path, first_frame_image_path)``."""
    mode = str(getattr(config, "video_edit_source_mode", "image_only") or "image_only")
    asset_id = str(getattr(request, "source_asset_id", "") or "").strip()
    if not asset_id:
        return None, None
    src = asset_store.get_file_path(asset_id)
    if src is None or not Path(src).exists():
        raise RuntimeError(f"Video edit source asset not found: {asset_id!r}")

    record = asset_store.get_asset_record(asset_id) if hasattr(asset_store, "get_asset_record") else None
    mime = str((record or {}).get("mime_type") or "").lower()
    kind = str((record or {}).get("kind") or "").lower()
    is_video = mime.startswith("video/") or kind == "video"

    meta = getattr(request, "metadata", None)
    relation = ""
    if isinstance(meta, dict):
        relation = str(meta.get("relation_type") or "").strip().lower()
    # UI ``animate`` (图生视频) must not encode full source clips even when the canvas
    # node is a video asset (quick-animate binds asset:video as the start reference).
    i2v_intent = relation == "animate"

    if is_video and mode == "source_video" and not i2v_intent:
        return str(src), None
    if is_video and (mode == "first_frame" or i2v_intent):
        img = resolve_video_edit_source_image(
            asset_store, asset_id, mode="first_frame", work_dir=work_dir,
        )
        if work_dir is None:
            raise RuntimeError("Bernini first_frame source requires work_dir.")
        out = Path(work_dir) / f"bernini_source_{asset_id}_frame0.png"
        img.save(out)
        return None, str(out)
    if is_video:
        raise RuntimeError(
            f"Video source {asset_id!r} requires video_edit_source_mode=source_video or first_frame."
        )
    return None, str(src)


def video_i2v_encode_failure_message(config: Any) -> str:
    style = str(getattr(config, "video_i2v_style", "concat") or "concat")
    if style == "wan":
        return (
            "Wan image-to-video (animate) failed to VAE-encode the source image. "
            "Ensure the model bundle includes Wan2.1_VAE.pth or Wan2.2_VAE.pth "
            "(or vae/*.safetensors) and text_encoder assets under the bundle root."
        )
    if style == "ltx":
        return (
            "LTX image-to-video (animate) failed to VAE-encode the source image. "
            "Ensure the model bundle includes LTX Video VAE encoder weights "
            "(vae/ or vae_decoder.safetensors) under the bundle root."
        )
    if style == "hunyuan":
        return (
            "HunyuanVideo image-to-video (animate) failed to VAE-encode the source image. "
            "Ensure the model bundle includes vae/ weights and image_encoder/ SigLIP "
            "vision weights under the bundle root."
        )
    return (
        "Image-to-video (animate) requires encoding the first RGB frame into "
        "video latents. DanQing does not yet implement the Lightricks "
        "`AutoencoderKLLTXVideo`-class encoder in MLX; first-frame conditioning "
        "cannot be applied. Use text-to-video (`create`) or add an MLX encoder "
        "port for your bundle VAE."
    )


def video_prepare_i2v_source_image(config: Any, src_img: Any, w: int, h: int) -> Any:
    if str(getattr(config, "video_i2v_style", "concat") or "concat") == "wan":
        from backend.engine.families.wan.conditioning import prepare_wan_reference_image

        return prepare_wan_reference_image(src_img, w, h)
    return src_img.resize((w, h))


def inject_hunyuan_text_encoder_paths(
    entry: Any, config: Any, project_root: Path,
) -> None:
    """Resolve registry-declared native TE roots (ModelScope Qwen + ByT5)."""
    from backend.engine.contracts.pipeline_registry import (
        registry_scalar_default,
        resolve_project_path,
    )

    for param_key in ("text_encoder_qwen_local", "text_encoder_byt5_local"):
        val = registry_scalar_default(entry, param_key, None)
        if val is not None and str(val).strip():
            resolved = resolve_project_path(project_root, str(val).strip())
            setattr(config, param_key, str(resolved))
    rel = registry_scalar_default(entry, "text_encoder_release_after_encode", None)
    if rel is not None:
        setattr(config, "text_encoder_release_after_encode", bool(rel))


def inject_ltx_text_encoder_paths(
    entry: Any, config: Any, project_root: Path,
) -> None:
    """Resolve registry-declared Gemma 3 root under ``models/Text/…``."""
    from backend.engine.contracts.pipeline_registry import (
        registry_scalar_default,
        resolve_project_path,
    )

    val = registry_scalar_default(entry, "text_encoder_gemma_local", None)
    if val is not None and str(val).strip():
        resolved = resolve_project_path(project_root, str(val).strip())
        setattr(config, "text_encoder_gemma_local", str(resolved))


def create_video_t5_encoder(
    ctx: Any,
    bundle_root: Path,
    config: Any,
    max_seq_len: int,
) -> Any:
    """Construct T5 or Wan UMT5 encoder from bundle layout + config flags."""
    from backend.engine.common.bundle.layout import t5_encoder_bundle_paths
    from backend.engine.common.codecs.text_encoders import T5Encoder

    if bool(getattr(config, "uses_wan_t5_bundle", False)):
        pth_path, tok_dir = wan_t5_bundle_paths(bundle_root)
        if getattr(ctx, "backend", "mlx") == "cuda":
            from backend.engine.families.wan.text_encoder_cuda import WanUMT5EncoderCUDA
            return WanUMT5EncoderCUDA(ctx, pth_path, tok_dir, text_len=max_seq_len)
        from backend.engine.families.wan.text_encoder_mlx import WanUMT5EncoderMLX
        return WanUMT5EncoderMLX(ctx, pth_path, tok_dir, text_len=max_seq_len)
    t5_dir, t5_tok_dir = t5_encoder_bundle_paths(bundle_root)
    return T5Encoder(ctx, t5_dir, max_seq_len=max_seq_len, tokenizer_path=t5_tok_dir)


def video_uses_ltx_distilled_timesteps(
    config: Any,
    *,
    step_distill: bool,
    scheduler_default: str,
) -> bool:
    return bool(
        step_distill
        and scheduler_default == "flow_match_euler"
        and str(getattr(config, "video_i2v_style", "")) == "ltx23"
    )


def video_uses_hunyuan_step_distill_timesteps(
    config: Any,
    *,
    step_distill: bool,
    scheduler_default: str,
) -> bool:
    return bool(
        step_distill
        and scheduler_default == "flow_match_euler"
        and str(getattr(config, "video_i2v_style", "")) == "hunyuan"
    )


def video_uses_wan_step_distill_timesteps(
    config: Any,
    *,
    step_distill: bool,
    scheduler_default: str,
) -> bool:
    return bool(
        step_distill
        and scheduler_default == "wan_flow_unipc"
        and str(getattr(config, "video_i2v_style", "")) == "wan"
    )


def video_apply_hunyuan_step_distill_scheduler_timesteps(
    ctx: Any,
    scheduler: Any,
    *,
    steps: int,
    config: Any | None = None,
) -> Any:
    """Apply HunyuanVideo-1.5 LightX2V 4-step distilled timesteps."""
    from backend.engine.families.hunyuan.distill import configure_hunyuan_lightx2v_distill_timesteps

    return configure_hunyuan_lightx2v_distill_timesteps(ctx, scheduler, steps, config=config)


def video_apply_wan_step_distill_scheduler_timesteps(
    ctx: Any,
    scheduler: Any,
    *,
    steps: int,
    config: Any | None = None,
) -> Any:
    """Apply LightX2V Wan 2.2 4-step distilled timesteps."""
    from backend.engine.families.wan.distill import configure_wan_step_distill_timesteps

    return configure_wan_step_distill_timesteps(ctx, scheduler, steps, config=config)


def video_apply_ltx_distilled_scheduler_timesteps(
    ctx: Any,
    scheduler: Any,
    *,
    bundle_root: Path | None,
    steps: int,
    w: int,
    h: int,
    latent_frames: int,
    vae_scale: int,
    on_log: Callable | None,
) -> list[int]:
    """Apply LTX distilled sigma shift; returns denoise step indices."""
    import numpy as np

    from backend.engine.families.ltx.pipeline_math_mlx import (
        ltx_calculate_shift,
        ltx_scheduler_shift_kwargs,
        ltx_stretch_shift_to_terminal,
        ltx_video_sequence_length,
    )

    video_seq_len = ltx_video_sequence_length(
        latent_frames=latent_frames,
        pixel_w=w,
        pixel_h=h,
        vae_scale=vae_scale,
    )
    shift_cfg = ltx_scheduler_shift_kwargs(bundle_root)
    sigmas_arr = np.linspace(1.0, 1.0 / steps, steps, dtype=np.float32)
    mu = ltx_calculate_shift(
        video_seq_len,
        base_image_seq_len=int(shift_cfg["base_image_seq_len"]),
        max_image_seq_len=int(shift_cfg["max_image_seq_len"]),
        base_shift=float(shift_cfg["base_shift"]),
        max_shift=float(shift_cfg["max_shift"]),
    )
    sigmas_t = ctx.array(sigmas_arr, dtype=ctx.float32())
    sigmas_shifted = scheduler._time_shift_exponential_array(mu, 1.0, sigmas_t)
    shift_terminal = shift_cfg.get("shift_terminal")
    if shift_terminal is not None:
        ctx.eval(sigmas_shifted)
        sigmas_shifted_np = np.array(sigmas_shifted, dtype=np.float32)
        sigmas_shifted = ctx.array(
            ltx_stretch_shift_to_terminal(sigmas_shifted_np, float(shift_terminal)),
            dtype=ctx.float32(),
        )
    scheduler._sigmas = ctx.concat(
        [sigmas_shifted, ctx.zeros((1,), dtype=ctx.float32())],
        axis=0,
    )
    scheduler._timesteps = sigmas_shifted * float(scheduler.num_train_timesteps)
    scheduler._step_index = 0
    if on_log is not None:
        on_log(
            "info",
            f"LTX distilled timesteps: video_seq_len={video_seq_len}, mu={mu:.4f}, steps={steps}",
        )
    return list(range(steps))
