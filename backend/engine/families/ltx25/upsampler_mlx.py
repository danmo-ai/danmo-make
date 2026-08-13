"""LTX-2.5 latent spatial upsampler (x2) — MLX port of upstream ``LatentUpsampler``."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.ltx25.pipeline_math_mlx import get_upsampler_config
from backend.engine.runtime.mlx_runtime import load_weights_dict

_UPSAMPLER_FILE = "upsampler.safetensors"


class _GroupNorm2d(nn.Module):
    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.num_groups = num_groups

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        x = x.reshape(b, h, w, self.num_groups, c // self.num_groups)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) * mx.rsqrt(var + 1e-6)
        x = x.reshape(b, h, w, c)
        return x * self.weight + self.bias


class _ResBlock(nn.Module):
    def __init__(self, channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels if mid_channels is not None else channels
        self.conv1 = nn.Conv2d(channels, mid_channels, 3, padding=1)
        self.norm1 = _GroupNorm2d(32, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, channels, 3, padding=1)
        self.norm2 = _GroupNorm2d(32, channels)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = nn.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return nn.silu(x + residual)


def _pixel_shuffle_2d(x: mx.array, factor: int) -> mx.array:
    """Depth-to-space over H/W for ``(B, H, W, C*f²)`` → ``(B, H*f, W*f, C)``."""
    b, h, w, c_total = x.shape
    c = c_total // (factor * factor)
    x = x.reshape(b, h, w, c, factor, factor)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, h * factor, w * factor, c)


class LTX25LatentUpsampler(nn.Module):
    """Spatial x2 latent upsampler (conv2d chain, per-frame)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        in_channels = int(cfg.get("in_channels", 128))
        mid_channels = int(cfg.get("mid_channels", 512))
        num_blocks_per_stage = int(cfg.get("num_blocks_per_stage", 4))
        spatial_upsample = bool(cfg.get("spatial_upsample", True))
        temporal_upsample = bool(cfg.get("temporal_upsample", False))
        rational_resampler = bool(cfg.get("rational_resampler", False))
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        if rational_resampler:
            raise RuntimeError("LTX 2.5 upsampler: rational_resampler not supported (fail loud).")

        self.initial_conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.initial_norm = _GroupNorm2d(32, mid_channels)
        self.res_blocks = [_ResBlock(mid_channels) for _ in range(num_blocks_per_stage)]
        if spatial_upsample and temporal_upsample:
            raise RuntimeError("LTX 2.5 upsampler: joint spatiotemporal upsampling not supported (fail loud).")
        if spatial_upsample:
            self.upsampler_conv = nn.Conv2d(mid_channels, 4 * mid_channels, 3, padding=1)
            self._spatial_factor = 2
        elif temporal_upsample:
            raise RuntimeError("LTX 2.5 upsampler: temporal-only upsampling not supported (fail loud).")
        else:
            raise RuntimeError("LTX 2.5 upsampler config must enable spatial_upsample.")
        self.post_upsample_res_blocks = [_ResBlock(mid_channels) for _ in range(num_blocks_per_stage)]
        self.final_conv = nn.Conv2d(mid_channels, in_channels, 3, padding=1)

    def __call__(self, latent: mx.array) -> mx.array:
        """Latent ``(B, C, F, H, W)`` → spatially doubled ``(B, C, F, 2H, 2W)``."""
        output_dtype = latent.dtype
        x = latent.astype(mx.bfloat16)
        b, _, f, _, _ = x.shape
        x = x.transpose(0, 2, 3, 4, 1)  # (B, F, H, W, C)
        x = x.reshape(b * f, x.shape[2], x.shape[3], x.shape[4])
        x = nn.silu(self.initial_norm(self.initial_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        x = self.upsampler_conv(x)
        x = _pixel_shuffle_2d(x, self._spatial_factor)
        for block in self.post_upsample_res_blocks:
            x = block(x)
        x = self.final_conv(x)
        x = x.reshape(b, f, x.shape[1], x.shape[2], x.shape[3])
        return x.transpose(0, 4, 1, 2, 3).astype(output_dtype)


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
    """Normalize → upsampler → re-normalize (upstream ``upsample_video``).

    Uses the video VAE encoder's per-channel statistics for the normalize /
    un-normalize brackets.
    """
    from backend.engine.families.ltx25.vae_mlx import load_ltx25_video_encoder

    encoder = load_ltx25_video_encoder(bundle_root, load_fn=load_fn)
    upsampler = load_ltx25_latent_upsampler(bundle_root, load_fn=load_fn)
    mean = encoder.per_channel_statistics_mean.reshape(1, -1, 1, 1, 1)
    std = encoder.per_channel_statistics_std.reshape(1, -1, 1, 1, 1)
    latent = latent * std + mean
    latent = upsampler(latent)
    return (latent - mean) / std
