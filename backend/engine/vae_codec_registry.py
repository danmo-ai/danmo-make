"""VAE encode/decode dispatch — keyed by diffusers ``_class_name`` (no wrapper subdirs)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def qwen_pack_latents_nchw(ctx: Any, encoded_b16hw: Any, height_px: int, width_px: int) -> Any:
    """[B,16,H_lat,W_lat] → [B,64,H_px/16,W_px/16] (Flux pack_latents, NCHW)."""
    B = int(encoded_b16hw.shape[0])
    Hg = height_px // 16
    Wg = width_px // 16
    x = ctx.reshape(encoded_b16hw, (B, 16, Hg, 2, Wg, 2))
    x = ctx.permute(x, (0, 2, 4, 1, 3, 5))
    x = ctx.reshape(x, (B, Hg * Wg, 64))
    x = ctx.reshape(x, (B, Hg, Wg, 64))
    return ctx.permute(x, (0, 3, 1, 2))


def qwen_unpack_latents_nchw(ctx: Any, packed_b64hw: Any) -> Any:
    """Inverse of ``qwen_pack_latents_nchw`` → [B,16,H_lat,W_lat]."""
    B = int(packed_b64hw.shape[0])
    Hg, Wg = int(packed_b64hw.shape[2]), int(packed_b64hw.shape[3])
    x = ctx.permute(packed_b64hw, (0, 2, 3, 1))
    x = ctx.reshape(x, (B, Hg, Wg, 16, 2, 2))
    x = ctx.permute(x, (0, 3, 1, 4, 2, 5))
    return ctx.reshape(x, (B, 16, Hg * 2, Wg * 2))


def _flux2_crop_even_hw_latent(z: Any, ctx: Any) -> Any:
    if int(z.shape[2]) % 2 != 0:
        z = z[:, :, :-1, :]
    if int(z.shape[3]) % 2 != 0:
        z = z[:, :, :, :-1]
    return z


def _flux2_patchify_mean_latent(z_bchw: Any, ctx: Any) -> Any:
    B, C, H, W = (int(z_bchw.shape[0]), int(z_bchw.shape[1]), int(z_bchw.shape[2]), int(z_bchw.shape[3]))
    x = ctx.reshape(z_bchw, (B, C, H // 2, 2, W // 2, 2))
    x = ctx.permute(x, (0, 1, 3, 5, 2, 4))
    return ctx.reshape(x, (B, C * 4, H // 2, W // 2))


def _flux2_bn_normalize_editing_latent(
    latents: Any, vae_weights: dict[str, Any], bn_eps: float, ctx: Any
) -> Any:
    bn_mean = vae_weights.get("bn.running_mean")
    bn_var = vae_weights.get("bn.running_var")
    if bn_mean is None or bn_var is None:
        raise RuntimeError("Flux2 img2img: bundle missing bn.running_mean / bn.running_var on VAE checkpoint.")
    bm = bn_mean.reshape(1, -1, 1, 1).astype(latents.dtype)
    bv = bn_var.reshape(1, -1, 1, 1).astype(latents.dtype)
    std = ctx.sqrt(bv + float(bn_eps))
    return (latents - bm) / std


def _encode_flux2(
    *,
    ctx: Any,
    image_n11: Any,
    bundle_root: Path | None,
    project_root: Path,
    height_px: int | None,
    width_px: int | None,
    on_log: Callable | None,
) -> Any:
    del project_root, height_px, width_px
    from backend.engine.common.codecs.vae import (
        VAEEncoder,
        infer_latent_channels,
        load_vae_weight_dict,
        prepare_vae_encoder_weight_items,
        resolve_vae_bundle_config,
    )

    if bundle_root is None:
        raise RuntimeError("Flux2 VAE encode: bundle_root missing")
    vae_dir = bundle_root / "vae"
    if not vae_dir.is_dir():
        raise RuntimeError(f"Flux2 VAE encode: no vae directory under {bundle_root}")
    vae_weights = load_vae_weight_dict(ctx, vae_dir)
    if not vae_weights:
        raise RuntimeError(f"Flux2 VAE encode: no weights under {vae_dir}")
    vae_cfg, _, _ = resolve_vae_bundle_config(vae_dir, vae_weights=vae_weights, ctx=ctx)

    scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
    shift_factor = float(vae_cfg.get("shift_factor", 0.0))
    latent_c = infer_latent_channels(vae_cfg, vae_weights)
    enc = VAEEncoder(
        latent_channels=latent_c,
        ctx=ctx,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
    )
    enc_items = prepare_vae_encoder_weight_items(vae_weights)
    loaded, skipped = enc.load_weights(enc_items, strict=False)
    if not any(k.startswith("conv_in.") for k in loaded):
        raise RuntimeError(
            "Flux2 VAE encode failed to load conv_in weights; "
            f"skipped_sample={skipped[:8]}"
        )

    h64 = enc.encode_conv_out_nchw(image_n11)
    qw = vae_weights.get("quant_conv.weight")
    qb = vae_weights.get("quant_conv.bias")
    if qw is None or qb is None:
        raise RuntimeError("Flux2 img2img: VAE checkpoint missing quant_conv.* tensors.")
    t_nhwc = ctx.permute(h64, (0, 2, 3, 1))
    from backend.engine.common.codecs.vae.weight_remap import vae_conv_weight_for_runtime

    qw_kernel = vae_conv_weight_for_runtime(ctx, qw)
    t_q = ctx.conv2d(t_nhwc, qw_kernel, stride=1, padding=0)
    t_q = t_q + qb.reshape(1, 1, 1, -1)
    t_q = ctx.permute(t_q, (0, 3, 1, 2))
    mean = t_q[:, :latent_c]
    z = (mean - shift_factor) * scaling_factor
    z = _flux2_crop_even_hw_latent(z, ctx)
    z = _flux2_patchify_mean_latent(z, ctx)
    bn_eps = float(vae_cfg.get("batch_norm_eps", 1e-4))
    z = _flux2_bn_normalize_editing_latent(z, vae_weights, bn_eps, ctx)
    if on_log:
        on_log("info", f"vae_encode flux2 transformer_latent_shape={tuple(z.shape)}")
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(z)
    return z


def _decode_flux2(
    *,
    ctx: Any,
    latents: Any,
    bundle_root: Path | None,
    project_root: Path,
    on_log: Callable | None,
    vae_output_to_uint8_hwc: Callable[..., Any],
    image_cls: Any,
) -> Any:
    del vae_output_to_uint8_hwc, image_cls, project_root
    from backend.engine.families.flux2.vae_mlx import decode_flux2_packed_latents_to_pil

    if bundle_root is None:
        raise RuntimeError("Flux2 VAE decode: missing bundle_root")
    return decode_flux2_packed_latents_to_pil(
        ctx,
        latents,
        bundle_root,
        on_log=(lambda level, msg: on_log(level, msg)) if on_log else None,
    )


def _decode_qwen(
    *,
    ctx: Any,
    latents: Any,
    bundle_root: Path | None,
    project_root: Path,
    on_log: Callable | None,
    vae_output_to_uint8_hwc: Callable[..., Any],
    image_cls: Any,
) -> Any:
    from backend.engine.families.qwen.vae import QwenVAE, apply_qwen_vae_weights_from_bundle

    if bundle_root is None:
        raise RuntimeError("Qwen VAE decode: missing bundle_root")
    z = qwen_unpack_latents_nchw(ctx, latents)
    vae_q = QwenVAE()
    apply_qwen_vae_weights_from_bundle(vae_q, bundle_root, project_root=project_root)
    decoded = vae_q.decode(z)
    if getattr(decoded, "ndim", 0) == 5 and int(decoded.shape[2]) == 1:
        decoded = decoded[:, :, 0, :, :]
    if on_log:
        on_log("info", f"vae_decode qwen unpacked_z_shape={tuple(z.shape)} decoded_shape={tuple(decoded.shape)}")
    pixels = vae_output_to_uint8_hwc(decoded, ctx)
    return image_cls.fromarray(pixels)


def _encode_qwen(
    *,
    ctx: Any,
    image_n11: Any,
    bundle_root: Path | None,
    project_root: Path,
    height_px: int | None,
    width_px: int | None,
    on_log: Callable | None,
) -> Any:
    from backend.engine.families.qwen.vae import QwenVAE, apply_qwen_vae_weights_from_bundle

    if height_px is None or width_px is None:
        raise RuntimeError("Qwen VAE encode: height_px and width_px are required for latent packing.")
    if bundle_root is None:
        raise RuntimeError("Qwen VAE encode: bundle_root missing")
    vae = QwenVAE()
    apply_qwen_vae_weights_from_bundle(vae, bundle_root, project_root=project_root)
    enc_out = vae.encode(image_n11)
    if getattr(enc_out, "ndim", 0) == 5 and int(enc_out.shape[2]) == 1:
        enc_out = enc_out[:, :, 0, :, :]
    packed = qwen_pack_latents_nchw(ctx, enc_out, height_px, width_px)
    if on_log:
        on_log("info", f"vae_encode qwen packed_shape={tuple(packed.shape)}")
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(packed)
    return packed


def _decode_wan_image(
    *,
    ctx: Any,
    latents: Any,
    bundle_root: Path | None,
    project_root: Path,
    on_log: Callable | None,
    vae_output_to_uint8_hwc: Callable[..., Any],
    image_cls: Any,
) -> Any:
    del vae_output_to_uint8_hwc, project_root
    from backend.engine.families.wan.vae import decode_wan_latents_to_pil_frames

    if bundle_root is None:
        raise RuntimeError("Wan VAE decode: missing bundle_root")
    z = latents
    if z.ndim == 4:
        z = z[:, :, None, :, :]
    frames = decode_wan_latents_to_pil_frames(
        ctx,
        z,
        bundle_root,
        on_log=(lambda msg: on_log("info", msg)) if on_log else None,
    )
    if not frames:
        raise RuntimeError("Wan VAE decode produced no frames")
    if on_log:
        on_log("info", f"vae_decode wan frame_size={frames[0].size}")
    return frames[0]


def _encode_wan_image(
    *,
    ctx: Any,
    image_n11: Any,
    bundle_root: Path | None,
    project_root: Path,
    height_px: int | None,
    width_px: int | None,
    on_log: Callable | None,
) -> Any:
    del project_root, height_px, width_px
    from backend.engine.families.wan.vae import encode_wan_image_to_latent

    if bundle_root is None:
        raise RuntimeError("Wan VAE encode: bundle_root missing")
    if image_n11.ndim != 4 or int(image_n11.shape[1]) != 3:
        raise RuntimeError(f"Wan VAE encode expects [1,3,H,W], got {tuple(image_n11.shape)}")
    latents = encode_wan_image_to_latent(ctx, image_n11[0], bundle_root)
    if int(latents.shape[2]) == 1:
        latents = latents[:, :, 0, :, :]
    if on_log:
        on_log("info", f"vae_encode wan latent_shape={tuple(latents.shape)}")
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(latents)
    return latents


def _decode_fibo_wan(
    *,
    ctx: Any,
    latents: Any,
    bundle_root: Path | None,
    project_root: Path,
    on_log: Callable | None,
    vae_output_to_uint8_hwc: Callable[..., Any],
    image_cls: Any,
) -> Any:
    del vae_output_to_uint8_hwc, project_root
    from backend.engine.families.fibo import vae_mlx

    if bundle_root is None:
        raise RuntimeError("FIBO VAE decode: missing bundle_root")
    pixels = vae_mlx.decode_latents_nchw(ctx, latents, bundle_root)
    arr = np.asarray(pixels)
    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
    if arr.ndim == 4:
        arr = arr[0].transpose(1, 2, 0)
    if on_log:
        on_log("info", f"vae_decode fibo frame_size={(int(arr.shape[1]), int(arr.shape[0]))}")
    return image_cls.fromarray((arr * 255.0).astype(np.uint8))


def _encode_fibo_wan(
    *,
    ctx: Any,
    image_n11: Any,
    bundle_root: Path | None,
    project_root: Path,
    height_px: int | None,
    width_px: int | None,
    on_log: Callable | None,
) -> Any:
    del project_root, height_px, width_px
    from backend.engine.families.fibo import vae_mlx

    if bundle_root is None:
        raise RuntimeError("FIBO VAE encode: model bundle is not installed at local_path.")
    latents = vae_mlx.encode_image_n11(ctx, image_n11, bundle_root)
    if on_log:
        on_log("info", f"vae_encode fibo wan latent_shape={tuple(latents.shape)}")
    return latents


_VAE_DECODE: dict[str, Callable[..., Any]] = {
    "AutoencoderKLQwenImage": _decode_qwen,
    "AutoencoderKLFlux2": _decode_flux2,
    "AutoencoderKLWan": _decode_wan_image,
}

_VAE_ENCODE: dict[str, Callable[..., Any]] = {
    "AutoencoderKLFlux2": _encode_flux2,
    "AutoencoderKLQwenImage": _encode_qwen,
    "AutoencoderKLWan": _encode_wan_image,
}

_VAE_DECODE_FAMILY_PAIR: dict[tuple[str, str], Callable[..., Any]] = {
    ("fibo", "AutoencoderKLWan"): _decode_fibo_wan,
}

_VAE_ENCODE_FAMILY_PAIR: dict[tuple[str, str], Callable[..., Any]] = {
    ("fibo", "AutoencoderKLWan"): _encode_fibo_wan,
}


def get_vae_decode_handler(
    vae_class_name: str,
    *,
    entry_family: str = "",
) -> Callable[..., Any] | None:
    pair = (entry_family, vae_class_name)
    fn = _VAE_DECODE_FAMILY_PAIR.get(pair)
    if fn is None:
        fn = _VAE_DECODE.get(vae_class_name)
    return fn


def get_vae_encode_handler(
    vae_class_name: str,
    *,
    entry_family: str = "",
) -> Callable[..., Any] | None:
    pair = (entry_family, vae_class_name)
    fn = _VAE_ENCODE_FAMILY_PAIR.get(pair)
    if fn is None:
        fn = _VAE_ENCODE.get(vae_class_name)
    return fn


def registered_vae_decode_classes() -> frozenset[str]:
    return frozenset(_VAE_DECODE.keys())


def registered_vae_encode_classes() -> frozenset[str]:
    return frozenset(_VAE_ENCODE.keys())


def _warmup_flux2_vae_preview(
    *,
    ctx: Any,
    bundle_root: Path | None,
    on_log: Callable | None,
) -> Any:
    from backend.engine.families.flux2.vae_mlx import load_flux2_vae_decoder

    if bundle_root is None:
        raise RuntimeError("Flux2 VAE preview warmup: missing bundle_root")
    return load_flux2_vae_decoder(ctx, bundle_root, on_log=on_log)


def _decode_flux2_vae_preview(
    *,
    ctx: Any,
    warmed_model: Any,
    latents: Any,
    on_log: Callable | None,
    **_kw: Any,
) -> Any:
    from backend.engine.families.flux2.vae_mlx import decode_flux2_latents_with_model

    return decode_flux2_latents_with_model(
        ctx, warmed_model, latents, on_log=on_log
    )


_VAE_PREVIEW_WARMUP: dict[str, Callable[..., Any]] = {
    "AutoencoderKLFlux2": _warmup_flux2_vae_preview,
}

_VAE_PREVIEW_DECODE: dict[str, Callable[..., Any]] = {
    "AutoencoderKLFlux2": _decode_flux2_vae_preview,
}


def get_vae_preview_warmup_handler(
    vae_class_name: str,
    *,
    entry_family: str = "",
) -> Callable[..., Any] | None:
    del entry_family
    return _VAE_PREVIEW_WARMUP.get(vae_class_name)


def get_vae_preview_decode_handler(
    vae_class_name: str,
    *,
    entry_family: str = "",
) -> Callable[..., Any] | None:
    del entry_family
    return _VAE_PREVIEW_DECODE.get(vae_class_name)


def warmup_vae_preview(
    ctx: Any,
    *,
    bundle_root: Path | None,
    vae_class_name: str,
    entry_family: str = "",
    on_log: Callable | None = None,
) -> Any | None:
    """Load a reusable VAE for denoise-step preview when a handler exists."""
    fn = get_vae_preview_warmup_handler(vae_class_name, entry_family=entry_family)
    if fn is None:
        return None
    return fn(ctx=ctx, bundle_root=bundle_root, on_log=on_log)


def decode_vae_preview(
    ctx: Any,
    *,
    warmed_model: Any,
    latents: Any,
    vae_class_name: str,
    entry_family: str = "",
    on_log: Callable | None = None,
) -> Any:
    """Decode latents with a preview-warmed VAE model."""
    fn = get_vae_preview_decode_handler(vae_class_name, entry_family=entry_family)
    if fn is None:
        raise RuntimeError(
            f"No VAE preview decode handler for {vae_class_name!r} (family={entry_family!r})"
        )
    return fn(ctx=ctx, warmed_model=warmed_model, latents=latents, on_log=on_log)
