"""LTX-2.5 latent spatial upsampler (x2) — MLX port of upstream ``LatentUpsampler``.

The official checkpoint uses ``dims=3``: 3D convolutions for the main path
(``initial_conv`` / ``res_blocks`` / ``post_upsample_res_blocks`` /
``final_conv``) with a per-frame 2D ``upsampler.0`` (Conv2d + pixel shuffle).
Weights are stored MLX-layout (channel-last ``(O, kD, kH, kW, I)``) after ingest
transposes them from torch layout; tensors flow channel-first here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.ltx25.pipeline_math_mlx import get_upsampler_config
from backend.engine.runtime.mlx_runtime import load_weights_dict

_UPSAMPLER_FILE = "upsampler.safetensors"


class _GroupNorm3dCF(nn.Module):
    """GroupNorm over channels of ``(B, C, F, H, W)`` (channel-first)."""

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.num_groups = num_groups

    def __call__(self, x: mx.array) -> mx.array:
        b, c, f, h, w = x.shape
        x = x.reshape(b, self.num_groups, c // self.num_groups, f, h, w)
        mean = mx.mean(x, axis=(2, 3, 4, 5), keepdims=True)
        var = mx.var(x, axis=(2, 3, 4, 5), keepdims=True)
        x = (x - mean) * mx.rsqrt(var + 1e-6)
        x = x.reshape(b, c, f, h, w)
        return x * self.weight.reshape(1, c, 1, 1, 1) + self.bias.reshape(1, c, 1, 1, 1)


class _Conv3dCF(nn.Module):
    """3D conv with channel-first interface; weight stored MLX layout (O,D,H,W,I)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, kernel_size, kernel_size, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, F, H, W) → channel-last (B, F, H, W, C)
        x = x.transpose(0, 2, 3, 4, 1)
        y = mx.conv3d(x, self.weight, padding=self.weight.shape[1] // 2)
        return (y + self.bias).transpose(0, 4, 1, 2, 3)


class _ResBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _Conv3dCF(channels, channels)
        self.norm1 = _GroupNorm3dCF(32, channels)
        self.conv2 = _Conv3dCF(channels, channels)
        self.norm2 = _GroupNorm3dCF(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = nn.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return nn.silu(x + residual)


def _pixel_shuffle_2d(x: mx.array, factor: int) -> mx.array:
    b, h, w, c_total = x.shape
    c = c_total // (factor * factor)
    x = x.reshape(b, h, w, c, factor, factor)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, h * factor, w * factor, c)


class LTX25LatentUpsampler(nn.Module):
    """Spatial x2 latent upsampler (3D conv backbone + per-frame 2D upsampler)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        in_channels = int(cfg.get("in_channels", 128))
        mid_channels = int(cfg.get("mid_channels", 512))
        num_blocks_per_stage = int(cfg.get("num_blocks_per_stage", 4))
        spatial_upsample = bool(cfg.get("spatial_upsample", True))
        temporal_upsample = bool(cfg.get("temporal_upsample", False))
        rational_resampler = bool(cfg.get("rational_resampler", False))
        if rational_resampler:
            raise RuntimeError("LTX 2.5 upsampler: rational_resampler not supported (fail loud).")
        if temporal_upsample:
            raise RuntimeError("LTX 2.5 upsampler: temporal upsampling not supported (fail loud).")
        if not spatial_upsample:
            raise RuntimeError("LTX 2.5 upsampler config must enable spatial_upsample.")

        self.initial_conv = _Conv3dCF(in_channels, mid_channels)
        self.initial_norm = _GroupNorm3dCF(32, mid_channels)
        self.res_blocks = [_ResBlock3d(mid_channels) for _ in range(num_blocks_per_stage)]
        # Checkpoint key layout: ``upsampler.0.weight`` (Conv2d inside a Sequential).
        self.upsampler = [nn.Conv2d(mid_channels, 4 * mid_channels, 3, padding=1)]
        self.post_upsample_res_blocks = [_ResBlock3d(mid_channels) for _ in range(num_blocks_per_stage)]
        self.final_conv = _Conv3dCF(mid_channels, in_channels)

    def __call__(self, latent: mx.array) -> mx.array:
        """Latent ``(B, C, F, H, W)`` → spatially doubled ``(B, C, F, 2H, 2W)``."""
        output_dtype = latent.dtype
        x = latent.astype(mx.bfloat16)
        x = nn.silu(self.initial_norm(self.initial_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        # Per-frame 2D upsampler: fold frames into batch, conv + pixel shuffle.
        b, c, f, h, w = x.shape
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * f, h, w, c)
        x = self.upsampler[0](x)
        x = _pixel_shuffle_2d(x, 2)
        _, h2, w2, c2 = x.shape
        x = x.reshape(b, f, h2, w2, c2).transpose(0, 4, 1, 2, 3)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x.astype(output_dtype)


def load_ltx25_latent_upsampler(bundle_root: Path, *, load_fn: Any | None = None) -> LTX25LatentUpsampler:
    cfg = get_upsampler_config(bundle_root)
    upsampler = LTX25LatentUpsampler(cfg)
    path = Path(bundle_root) / _UPSAMPLER_FILE
    if not path.is_file():
        raise RuntimeError(f"LTX 2.5 upsampler weights missing: {path}")
    raw = load_weights_dict(load_fn, str(path))
    prefixes = {k.split(".", 1)[0] for k in raw if "." in k}
    if len(prefixes) == 1:
        prefix = f"{next(iter(prefixes))}."
        raw = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in raw.items()}
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(upsampler.parameters()))
    missing = [k for k in params if k not in raw]
    if missing:
        raise RuntimeError(f"LTX 2.5 upsampler weights missing: {missing[:8]}")
    upsampler.load_weights(list(raw.items()), strict=False)
    return upsampler


def upsample_video_latent(
    bundle_root: Path,
    latent: mx.array,
    *,
    load_fn: Any | None = None,
) -> mx.array:
    """Normalize → upsampler → re-normalize (upstream ``upsample_video``)."""
    from backend.engine.families.ltx25.vae_mlx import load_ltx25_video_encoder

    encoder = load_ltx25_video_encoder(bundle_root, load_fn=load_fn)
    upsampler = load_ltx25_latent_upsampler(bundle_root, load_fn=load_fn)
    mean = encoder.per_channel_statistics_mean.reshape(1, -1, 1, 1, 1)
    std = encoder.per_channel_statistics_std.reshape(1, -1, 1, 1, 1)
    latent = latent * std + mean
    latent = upsampler(latent)
    return (latent - mean) / std
