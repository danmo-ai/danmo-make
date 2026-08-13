"""LTX-2.5 codecs — config-driven conv video VAE + audio VAE + BigVGAN vocoder (MLX).

All shapes/block lists come from ``bundle_config.json`` (extracted from the
checkpoint metadata by ``ingest.py``); nothing is hardcoded per generation.
Reuses the in-repo LTX 2.3 conv primitives (``families.ltx.vae_mlx``) for the
shared CausalVideoAutoencoder block family.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.families.ltx.vae_mlx import (
    Conv3dBlock,
    SpaceToDepthDownsample,
    patchify_spatial,
    pixel_shuffle_3d,
    unpatchify_spatial,
)
from backend.engine.families.ltx25.pipeline_math_mlx import (
    get_audio_config,
    get_vae_config,
    get_vocoder_config,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval

_EVAL_EVERY = int(os.environ.get("LTX25_VAE_EVAL_EVERY", "2"))

_VIDEO_VAE_FILE = "video_vae.safetensors"
_AUDIO_VAE_FILE = "audio_vae.safetensors"


def _materialize(*arrays: mx.array) -> None:
    run_eval(None, *arrays)


def _load_bundle_weights(bundle_root: Path, filename: str, prefix: str, load_fn: Any | None) -> dict[str, mx.array]:
    path = Path(bundle_root) / filename
    if not path.is_file():
        raise RuntimeError(f"LTX 2.5 bundle file missing: {path}")
    raw = load_weights_dict(load_fn, str(path))
    if not prefix:
        return dict(raw)
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in raw.items()}


def _block_config(block_spec: Any) -> dict[str, Any]:
    if isinstance(block_spec, int):
        return {"num_layers": block_spec}
    if isinstance(block_spec, dict):
        return dict(block_spec)
    raise RuntimeError(f"Invalid VAE block spec: {block_spec!r}")


def _pixel_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
    return mx.fast.rms_norm(x, weight=None, eps=eps)


def _sinusoidal_timestep_proj(timesteps: mx.array, embedding_dim: int, flip_sin_to_cos: bool = True) -> mx.array:
    """DDPM sinusoidal embedding (matches upstream ``get_timestep_embedding``)."""
    half = embedding_dim // 2
    exponent = -mx.log(mx.array(10000.0)) * mx.arange(half).astype(mx.float32) / float(half)
    emb = mx.exp(exponent)
    emb = timesteps.astype(mx.float32)[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    return emb


class _TimestepEmbedder(nn.Module):
    """PixArtAlpha-combined timestep embedder: sinusoidal(256) → SiLU → MLP."""

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(256, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        proj = _sinusoidal_timestep_proj(sample, 256, flip_sin_to_cos=True).astype(sample.dtype)
        return self.linear_2(nn.silu(self.linear_1(proj)))


class _GroupNorm3d(nn.Module):
    """GroupNorm over the channel (last) axis of ``(B, D, H, W, C)`` tensors."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        b, d, h, w, c = x.shape
        x = x.reshape(b, d, h, w, self.num_groups, c // self.num_groups)
        mean = mx.mean(x, axis=(1, 2, 3, 5), keepdims=True)
        var = mx.var(x, axis=(1, 2, 3, 5), keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        x = x.reshape(b, d, h, w, c)
        return x * self.weight + self.bias


class _ResnetBlock3D(nn.Module):
    """Pre-activation resnet (pixel/group-norm), optional noise + block timestep."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        groups: int = 32,
        eps: float = 1e-6,
        norm_layer: str = "pixel_norm",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.in_channels = in_channels
        self.inject_noise = inject_noise
        self.timestep_conditioning = timestep_conditioning
        self._norm_kind = norm_layer

        def _make_norm(ch: int) -> nn.Module:
            if norm_layer == "group_norm":
                return _GroupNorm3d(groups, ch, eps=eps)
            return _PixNorm()

        self.norm1 = _make_norm(in_channels)
        self.conv1 = Conv3dBlock(
            in_channels, out_channels, kernel_size=3, padding=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        if inject_noise:
            self.per_channel_scale1 = mx.zeros((1, in_channels, 1, 1))
        self.norm2 = _make_norm(out_channels)
        self.conv2 = Conv3dBlock(
            out_channels, out_channels, kernel_size=3, padding=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        if inject_noise:
            self.per_channel_scale2 = mx.zeros((1, in_channels, 1, 1))
        self.conv_shortcut = (
            Conv3dBlock(in_channels, out_channels, kernel_size=1, padding=0, causal=True,
                        spatial_padding_mode=spatial_padding_mode)
            if in_channels != out_channels
            else None
        )
        self.norm3 = _GroupNorm3d(1, in_channels, eps=eps) if in_channels != out_channels else None
        if timestep_conditioning:
            self.scale_shift_table = mx.zeros((4, in_channels))

    def _feed_spatial_noise(self, hidden_states: mx.array, per_channel_scale: mx.array, seed: int | None) -> mx.array:
        _, _, h, w, _ = hidden_states.shape
        key = mx.random.key(seed if seed is not None else 0)
        noise = mx.random.normal((1, h, w, 1), key=key, dtype=hidden_states.dtype)
        scaled = noise * per_channel_scale.reshape(1, 1, 1, -1)
        return hidden_states + scaled

    def __call__(
        self,
        x: mx.array,
        causal: bool = True,
        timestep: mx.array | None = None,
        generator_seed: int | None = None,
    ) -> mx.array:
        hidden_states = self.norm1(x)
        shift1 = scale1 = shift2 = scale2 = None
        if self.timestep_conditioning:
            if timestep is None:
                raise RuntimeError("timestep required when timestep_conditioning=True")
            # timestep: (B, 4*C) — reshape (B, 4, C) then add static table (4, C)
            b = hidden_states.shape[0]
            ada = timestep.reshape(b, 4, -1) + self.scale_shift_table
            shift1, scale1, shift2, scale2 = ada[:, 0], ada[:, 1], ada[:, 2], ada[:, 3]
            hidden_states = hidden_states * (1.0 + scale1[:, None, None]) + shift1[:, None, None]
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(hidden_states, self.per_channel_scale1, generator_seed)
        hidden_states = self.norm2(hidden_states)
        if self.timestep_conditioning:
            hidden_states = hidden_states * (1.0 + scale2[:, None, None]) + shift2[:, None, None]
        hidden_states = nn.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.inject_noise:
            hidden_states = self._feed_spatial_noise(hidden_states, self.per_channel_scale2, generator_seed)
        shortcut = self.norm3(x) if self.norm3 is not None else x
        if self.conv_shortcut is not None:
            shortcut = self.conv_shortcut(shortcut)
        return shortcut + hidden_states


class _PixNorm(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return _pixel_norm(x)


class _ResStage(nn.Module):
    """``UNetMidBlock3D``-equivalent: N res blocks + optional block timestep embedder."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        *,
        norm_layer: str,
        inject_noise: bool,
        timestep_conditioning: bool,
        spatial_padding_mode: str,
        has_attn: bool = False,
    ):
        super().__init__()
        self.timestep_conditioning = timestep_conditioning
        if timestep_conditioning:
            self.time_embedder = _TimestepEmbedder(channels * 4)
        self.res_blocks = [
            _ResnetBlock3D(
                channels,
                groups=32,
                norm_layer=norm_layer,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )
            for _ in range(num_blocks)
        ]

    def __call__(
        self,
        x: mx.array,
        causal: bool = True,
        timestep: mx.array | None = None,
        generator_seed: int | None = None,
    ) -> mx.array:
        block_emb = None
        if self.timestep_conditioning:
            if timestep is None:
                raise RuntimeError("timestep required when timestep_conditioning=True")
            block_emb = self.time_embedder(timestep)
        for blk in self.res_blocks:
            x = blk(x, causal=causal, timestep=block_emb, generator_seed=generator_seed)
        return x


class _RMSNorm2D(nn.Module):
    """Channel-first RMSNorm with per-channel gain (upstream ``_RMSNorm2D``)."""

    def __init__(self, channels: int):
        super().__init__()
        self.gamma = mx.ones((channels, 1, 1))
        self._scale = channels**0.5

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, D, H, W, C)
        normed = _pixel_norm(x) * self._scale
        return normed * self.gamma.reshape(1, 1, 1, -1)


class _AttnBlock3D(nn.Module):
    """Single-head per-frame spatial self-attention (upstream ``AttnBlock3D``)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = _RMSNorm2D(in_channels)
        self.to_qkv = Conv3dBlock(
            in_channels, in_channels * 3, kernel_size=1, padding=0, causal=False,
            spatial_padding_mode="zeros",
        )
        self.proj = Conv3dBlock(
            in_channels, in_channels, kernel_size=1, padding=0, causal=False,
            spatial_padding_mode="zeros",
        )

    def __call__(self, x: mx.array) -> mx.array:
        b, d, h, w, c = x.shape
        folded = x.reshape(b * d, h, w, c)
        folded = self.norm(folded)
        qkv = self.to_qkv(folded)
        qkv = qkv.reshape(b * d, h * w, 3 * c)
        q, k, v = qkv[..., :c], qkv[..., c:2 * c], qkv[..., 2 * c:]
        scale = c**-0.5
        attn = mx.softmax((q @ k.transpose(0, 2, 1)) * scale, axis=-1)
        out = attn @ v
        out = self.proj(out.reshape(b * d, h, w, c))
        return x + out.reshape(b, d, h, w, c)


class _DepthToSpaceUpsample(nn.Module):
    """Conv + pixel-shuffle upsampling (upstream ``DepthToSpaceUpsample``)."""

    def __init__(self, in_channels: int, out_channels: int, *, causal: bool, spatial_padding_mode: str,
                 spatial_factor: int = 2, temporal_factor: int = 2):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        self.conv = Conv3dBlock(
            in_channels, out_channels, kernel_size=3, padding=1, causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )


def _decoder_bottleneck_channels(base_channels: int, decoder_blocks: list[Any]) -> int:
    channel_multiplier = 1
    for block_name, block_params in decoder_blocks:
        config = _block_config(block_params)
        if block_name in ("compress_time", "compress_space", "compress_all"):
            channel_multiplier *= int(config.get("multiplier", 1))
        elif block_name == "res_x_y":
            channel_multiplier *= int(config.get("multiplier", 2))
    return base_channels * channel_multiplier


class LTX25VideoDecoder(nn.Module):
    """Config-driven conv video VAE decoder (``CausalVideoAutoencoder`` family)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        in_channels = int(cfg.get("latent_channels", 128))
        out_channels = int(cfg.get("out_channels", 3))
        decoder_blocks = list(cfg.get("decoder_blocks", []))
        patch_size = int(cfg.get("patch_size", 4))
        norm_layer = str(cfg.get("norm_layer", "pixel_norm"))
        causal = bool(cfg.get("causal_decoder", False))
        timestep_conditioning = bool(cfg.get("timestep_conditioning", True))
        spatial_padding_mode = str(cfg.get("spatial_padding_mode", "reflect"))
        base_channels = int(cfg.get("decoder_base_channels", 128))

        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self.patch_size = patch_size
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05
        self._norm_kind = norm_layer
        self.per_channel_statistics_mean = mx.zeros((in_channels,))
        self.per_channel_statistics_std = mx.ones((in_channels,))

        feature_channels = _decoder_bottleneck_channels(base_channels, decoder_blocks)
        self.conv_in = Conv3dBlock(
            in_channels, feature_channels, kernel_size=3, padding=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.up_blocks: list[Any] = []
        for block_name, block_params in reversed(decoder_blocks):
            config = _block_config(block_params)
            if block_name in ("res_x", "attn_res_x"):
                num_layers = int(config.get("num_layers", 1))
                self.up_blocks.append(
                    _ResStage(
                        feature_channels,
                        num_layers,
                        norm_layer=norm_layer,
                        inject_noise=bool(config.get("inject_noise", False)),
                        timestep_conditioning=timestep_conditioning,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif block_name == "res_x_y":
                multiplier = int(config.get("multiplier", 2))
                self.up_blocks.append(
                    _ResnetBlock3D(
                        feature_channels,
                        out_channels=feature_channels // multiplier,
                        groups=32,
                        norm_layer=norm_layer,
                        inject_noise=bool(config.get("inject_noise", False)),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
                feature_channels = feature_channels // multiplier
            elif block_name == "compress_time":
                red = int(config.get("multiplier", 1))
                self.up_blocks.append(
                    _DepthToSpaceUpsample(
                        feature_channels, feature_channels // red * 2, causal=causal,
                        spatial_padding_mode=spatial_padding_mode,
                        spatial_factor=1, temporal_factor=2,
                    )
                )
                feature_channels = feature_channels // red
            elif block_name == "compress_space":
                red = int(config.get("multiplier", 1))
                self.up_blocks.append(
                    _DepthToSpaceUpsample(
                        feature_channels, feature_channels // red * 4, causal=causal,
                        spatial_padding_mode=spatial_padding_mode,
                        spatial_factor=2, temporal_factor=1,
                    )
                )
                feature_channels = feature_channels // red
            elif block_name == "compress_all":
                red = int(config.get("multiplier", 1))
                self.up_blocks.append(
                    _DepthToSpaceUpsample(
                        feature_channels, feature_channels // red * 8, causal=causal,
                        spatial_padding_mode=spatial_padding_mode,
                        spatial_factor=2, temporal_factor=2,
                    )
                )
                feature_channels = feature_channels // red
            elif block_name == "attn":
                self.up_blocks.append(_AttnBlock3D(feature_channels))
            else:
                raise RuntimeError(f"LTX 2.5 VAE decoder: unknown block {block_name!r}")

        self.conv_out = Conv3dBlock(
            feature_channels,
            out_channels * patch_size * patch_size,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        if timestep_conditioning:
            self.timestep_scale_multiplier = mx.array(1000.0)
            self.last_time_embedder = _TimestepEmbedder(feature_channels * 2)
            self.last_scale_shift_table = mx.zeros((2, feature_channels))

    def _denormalize_latent(self, latent: mx.array) -> mx.array:
        mean = self.per_channel_statistics_mean.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics_std.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def decode(self, latent: mx.array, *, seed: int | None = None) -> mx.array:
        """Decode ``(B, C, F', H', W')`` latents → ``(B, 3, F, H, W)`` pixels in [-1, 1]."""
        output_dtype = latent.dtype
        latent = latent.astype(mx.bfloat16)
        b = latent.shape[0]
        if self.timestep_conditioning:
            key = mx.random.key(seed if seed is not None else 0)
            noise = mx.random.normal(latent.shape, key=key, dtype=latent.dtype) * self.decode_noise_scale
            latent = noise + (1.0 - self.decode_noise_scale) * latent
        x = latent.transpose(0, 2, 3, 4, 1)
        x = self._denormalize_latent(x)
        x = self.conv_in(x)
        scaled_timestep = None
        if self.timestep_conditioning:
            scaled_timestep = mx.full((b,), self.decode_timestep, dtype=x.dtype) * self.timestep_scale_multiplier

        stage_idx = 0
        for entry in self.up_blocks:
            if isinstance(entry, _ResStage):
                x = entry(x, causal=self.causal, timestep=scaled_timestep, generator_seed=seed)
            elif isinstance(entry, _ResnetBlock3D):
                x = entry(x, causal=self.causal, generator_seed=seed)
            elif isinstance(entry, _DepthToSpaceUpsample):
                x = entry.conv(x)
                x = pixel_shuffle_3d(x, spatial_factor=entry.spatial_factor, temporal_factor=entry.temporal_factor)
                if entry.temporal_factor > 1:
                    x = x[:, 1:, :, :, :]
                stage_idx += 1
                if stage_idx % _EVAL_EVERY == 0:
                    _materialize(x)
            elif isinstance(entry, _AttnBlock3D):
                x = entry(x)

        x = _pixel_norm(x)
        if self.timestep_conditioning:
            embedded = self.last_time_embedder(scaled_timestep)  # (B, 2*C)
            ada = embedded.reshape(b, 2, -1) + self.last_scale_shift_table
            shift, scale = ada[:, 0], ada[:, 1]
            x = x * (1.0 + scale[:, None, None]) + shift[:, None, None]
        x = nn.silu(x)
        x = self.conv_out(x)
        x = unpatchify_spatial(x, patch_size=self.patch_size)
        return x.transpose(0, 4, 1, 2, 3).astype(output_dtype)


class LTX25VideoEncoder(nn.Module):
    """Config-driven conv video VAE encoder (i2v first-frame conditioning)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        in_channels = int(cfg.get("in_channels", 3))
        out_channels = int(cfg.get("latent_channels", 128))
        encoder_blocks = list(cfg.get("encoder_blocks", []))
        patch_size = int(cfg.get("patch_size", 4))
        norm_layer = str(cfg.get("norm_layer", "pixel_norm"))
        spatial_padding_mode = str(cfg.get("spatial_padding_mode", "zeros"))
        latent_log_var = str(cfg.get("latent_log_var", "uniform"))

        self.patch_size = patch_size
        self.latent_log_var = latent_log_var
        self.per_channel_statistics_mean = mx.zeros((out_channels,))
        self.per_channel_statistics_std = mx.ones((out_channels,))

        feature_channels = out_channels
        self.conv_in = Conv3dBlock(
            in_channels * patch_size * patch_size, feature_channels, kernel_size=3, padding=1,
            causal=True, spatial_padding_mode=spatial_padding_mode,
        )
        self.down_blocks: list[Any] = []
        for block_name, block_params in encoder_blocks:
            config = _block_config(block_params)
            if block_name == "res_x":
                num_layers = int(config.get("num_layers", 1))
                self.down_blocks.append(
                    _ResStage(
                        feature_channels,
                        num_layers,
                        norm_layer=norm_layer,
                        inject_noise=False,
                        timestep_conditioning=False,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif block_name == "res_x_y":
                multiplier = int(config.get("multiplier", 2))
                self.down_blocks.append(
                    _ResnetBlock3D(
                        feature_channels, out_channels=feature_channels * multiplier, groups=32,
                        norm_layer=norm_layer, spatial_padding_mode=spatial_padding_mode,
                    )
                )
                feature_channels *= multiplier
            elif block_name in ("compress_time", "compress_space", "compress_all"):
                strides = {
                    "compress_time": (2, 1, 1),
                    "compress_space": (1, 2, 2),
                    "compress_all": (2, 2, 2),
                }[block_name]
                self.down_blocks.append(
                    Conv3dBlock(feature_channels, feature_channels, kernel_size=3, stride=strides, padding=1,
                                causal=True, spatial_padding_mode=spatial_padding_mode)
                )
            elif block_name in ("compress_time_res", "compress_space_res", "compress_all_res"):
                strides = {
                    "compress_time_res": (2, 1, 1),
                    "compress_space_res": (1, 2, 2),
                    "compress_all_res": (2, 2, 2),
                }[block_name]
                multiplier = int(config.get("multiplier", 2))
                self.down_blocks.append(
                    SpaceToDepthDownsample(feature_channels, feature_channels * multiplier, stride=strides)
                )
                feature_channels *= multiplier
            elif block_name == "compress_all_x_y":
                multiplier = int(config.get("multiplier", 2))
                self.down_blocks.append(
                    Conv3dBlock(feature_channels, feature_channels * multiplier, kernel_size=3, stride=(2, 2, 2),
                                padding=1, causal=True, spatial_padding_mode=spatial_padding_mode)
                )
                feature_channels *= multiplier
            elif block_name == "attn":
                self.down_blocks.append(_AttnBlock3D(feature_channels))
            else:
                raise RuntimeError(f"LTX 2.5 VAE encoder: unknown block {block_name!r}")

        conv_out_channels = out_channels
        if latent_log_var == "per_channel":
            conv_out_channels *= 2
        elif latent_log_var in ("uniform", "constant"):
            conv_out_channels += 1
        self.conv_out = Conv3dBlock(
            feature_channels, conv_out_channels, kernel_size=3, padding=1, causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def encode(self, sample: mx.array) -> mx.array:
        """Encode ``(B, C, F, H, W)`` pixels in [-1, 1] → normalized ``(B, 128, F', H', W')``."""
        output_dtype = sample.dtype
        x = sample.astype(mx.bfloat16)
        x = x.transpose(0, 2, 3, 4, 1)  # → BFHWC for MLX convs
        if self.patch_size > 1:
            x = patchify_spatial(x, patch_size=self.patch_size)
        x = self.conv_in(x)
        for entry in self.down_blocks:
            if isinstance(entry, _ResStage):
                x = entry(x, causal=True)
            elif isinstance(entry, _ResnetBlock3D):
                x = entry(x, causal=True)
            else:
                x = entry(x)
        x = nn.silu(_pixel_norm(x))
        x = self.conv_out(x)
        if self.latent_log_var in ("uniform", "constant"):
            x = x[..., :-1]
        x = x.transpose(0, 4, 1, 2, 3)  # → BCFHW
        mean = self.per_channel_statistics_mean.reshape(1, -1, 1, 1, 1)
        std = self.per_channel_statistics_std.reshape(1, -1, 1, 1, 1)
        return ((x - mean) / std).astype(output_dtype)


# ---------------------------------------------------------------------------
# Audio VAE (group-norm resnet autoencoder)
# ---------------------------------------------------------------------------

_LRELU_SLOPE = 0.1


class _AudioGroupNorm(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))
        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, f = x.shape
        x = x.reshape(b, self.num_groups, c // self.num_groups, t, f)
        mean = mx.mean(x, axis=(2, 3, 4), keepdims=True)
        var = mx.var(x, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        x = x.reshape(b, c, t, f)
        return x * self.weight.reshape(1, c, 1, 1) + self.bias.reshape(1, c, 1, 1)


class _AudioPixelNorm(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        mean_sq = mx.mean(x * x, axis=1, keepdims=True)
        return x / mx.sqrt(mean_sq + 1e-6)


class _AudioCausalConv2d(nn.Module):
    """Conv2d with causal asymmetric padding along the time axis (channel-first)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple = 3,
                 stride: int = 1, dilation: int = 1, causality_axis: str = "height"):
        super().__init__()
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.causality_axis = causality_axis
        pad_h = (ks[0] - 1) * dilation
        pad_w = (ks[1] - 1) * dilation
        if causality_axis == "none":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif causality_axis in ("width", "width_compatibility"):
            self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif causality_axis == "height":
            self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise RuntimeError(f"Invalid causality_axis: {causality_axis!r}")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=0, dilation=dilation)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T, F) channel-first; MLX conv2d is channel-last.
        pl, pr, pt, pb = self.padding
        x = mx.pad(x, [(0, 0), (0, 0), (pt, pb), (pl, pr)])
        x_nhwc = x.transpose(0, 2, 3, 1)  # (B, T, F, C)
        y = mx.conv2d(x_nhwc, self.conv.weight, stride=self.conv.stride, dilation=self.conv.dilation)
        y = y + self.conv.bias if getattr(self.conv, "bias", None) is not None else y
        return y.transpose(0, 3, 1, 2)


class _AudioResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None,
                 norm_type: str = "group", causality_axis: str = "height"):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self._norm_type = norm_type
        self.norm1 = self._make_norm(in_channels)
        self.conv1 = _AudioCausalConv2d(in_channels, out_channels, 3, causality_axis=causality_axis)
        self.norm2 = self._make_norm(out_channels)
        self.conv2 = _AudioCausalConv2d(out_channels, out_channels, 3, causality_axis=causality_axis)
        if in_channels != out_channels:
            self.nin_shortcut = _AudioCausalConv2d(in_channels, out_channels, 1, causality_axis=causality_axis)
        else:
            self.nin_shortcut = None

    def _make_norm(self, ch: int) -> nn.Module:
        if self._norm_type == "group":
            return _AudioGroupNorm(32, ch)
        return _AudioPixelNorm()

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.silu(self.norm1(x))
        h = self.conv1(h)
        h = nn.silu(self.norm2(h))
        h = self.conv2(h)
        shortcut = self.nin_shortcut(x) if self.nin_shortcut is not None else x
        return shortcut + h


class _AudioAttnBlock(nn.Module):
    def __init__(self, in_channels: int, norm_type: str = "group"):
        super().__init__()
        self.norm = _AudioGroupNorm(32, in_channels) if norm_type == "group" else _AudioPixelNorm()
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, f = x.shape
        h = self.norm(x)

        def _conv1x1(conv: nn.Conv2d, inp: mx.array) -> mx.array:
            y = mx.conv2d(inp.transpose(0, 2, 3, 1), conv.weight)
            y = y + conv.bias if getattr(conv, "bias", None) is not None else y
            return y.transpose(0, 3, 1, 2)

        q = _conv1x1(self.q, h).reshape(b, c, t * f)
        k = _conv1x1(self.k, h).reshape(b, c, t * f)
        v = _conv1x1(self.v, h).reshape(b, c, t * f)
        w = (q.transpose(0, 2, 1) @ k) * (c**-0.5)
        w = mx.softmax(w, axis=-1)
        out = (v @ w.transpose(0, 2, 1)).reshape(b, c, t, f)
        return x + _conv1x1(self.proj_out, out)


class _AudioUpsample(nn.Module):
    def __init__(self, in_channels: int, causality_axis: str = "height"):
        super().__init__()
        self.causality_axis = causality_axis
        self.conv = _AudioCausalConv2d(in_channels, in_channels, 3, causality_axis=causality_axis)

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, f = x.shape
        x = mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)
        x = self.conv(x)
        if self.causality_axis == "height":
            x = x[:, :, 1:, :]
        elif self.causality_axis == "width":
            x = x[:, :, :, 1:]
        return x


class _AudioStage(nn.Module):
    def __init__(self, block: list[Any], attn: list[Any], upsample: Any | None):
        super().__init__()
        self.block = block
        self.attn = attn
        if upsample is not None:
            self.upsample = upsample


class LTX25AudioDecoder(nn.Module):
    """Config-driven audio VAE decoder (latent → log-mel spectrogram)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        ddconfig = cfg.get("ddconfig", {})
        self.ch = int(ddconfig.get("ch", 128))
        self.out_ch = int(ddconfig.get("out_ch", 2))
        ch_mult = tuple(int(x) for x in ddconfig.get("ch_mult", (1, 2, 4)))
        num_res_blocks = int(ddconfig.get("num_res_blocks", 2))
        attn_resolutions = set(int(x) for x in ddconfig.get("attn_resolutions", [8, 16, 32]))
        resolution = int(ddconfig.get("resolution", 256))
        z_channels = int(ddconfig.get("z_channels", 8))
        norm_type = str(ddconfig.get("norm_type", "pixel"))
        causality_axis = str(ddconfig.get("causality_axis", "height"))
        mid_block_add_attention = bool(ddconfig.get("mid_block_add_attention", True))
        self.causality_axis = causality_axis
        self.mel_bins = int(ddconfig.get("mel_bins") or cfg.get("mel_bins") or 64)

        self.per_channel_statistics_mean = mx.zeros((self.ch,))
        self.per_channel_statistics_std = mx.ones((self.ch,))

        base_block_channels = self.ch * ch_mult[-1]
        base_resolution = resolution // (2 ** (len(ch_mult) - 1))

        self.conv_in = _AudioCausalConv2d(z_channels, base_block_channels, 3, causality_axis=causality_axis)

        self.mid = nn.Module()
        self.mid.block_1 = _AudioResnetBlock(base_block_channels, norm_type=norm_type, causality_axis=causality_axis)
        self.mid.attn_1 = (
            _AudioAttnBlock(base_block_channels, norm_type=norm_type) if mid_block_add_attention else _IdentityAudio()
        )
        self.mid.block_2 = _AudioResnetBlock(base_block_channels, norm_type=norm_type, causality_axis=causality_axis)

        self.up: list[_AudioStage] = []
        block_in = base_block_channels
        curr_res = base_resolution
        for level in reversed(range(len(ch_mult))):
            block_out = self.ch * ch_mult[level]
            stage_blocks: list[Any] = []
            stage_attns: list[Any] = []
            for _ in range(num_res_blocks + 1):
                stage_blocks.append(
                    _AudioResnetBlock(block_in, block_out, norm_type=norm_type, causality_axis=causality_axis)
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    stage_attns.append(_AudioAttnBlock(block_in, norm_type=norm_type))
            upsample = None
            if level != 0:
                upsample = _AudioUpsample(block_in, causality_axis=causality_axis)
                curr_res *= 2
            self.up.insert(0, _AudioStage(stage_blocks, stage_attns, upsample))

        self.norm_out = _AudioGroupNorm(32, block_in) if norm_type == "group" else _AudioPixelNorm()
        self.conv_out = _AudioCausalConv2d(block_in, self.out_ch, 3, causality_axis=causality_axis)

    def decode(self, latent: mx.array) -> mx.array:
        """Latent ``(B, 8, T, F)`` → log-mel spectrogram ``(B, out_ch, T', mel_bins)``."""
        output_dtype = latent.dtype
        x = latent.astype(mx.bfloat16)
        # Per-channel stats live on the 128-wide patched tokens (8 ch × 16 mel bins).
        b, c, t, f = x.shape
        patched = x.transpose(0, 2, 3, 1).reshape(b, t, c * f)
        mean = self.per_channel_statistics_mean.reshape(1, 1, -1)
        std = self.per_channel_statistics_std.reshape(1, 1, -1)
        patched = patched * std + mean
        x = patched.reshape(b, t, f, c).transpose(0, 3, 1, 2)
        x = self.conv_in(x)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))
        for stage in reversed(self.up):
            for idx, blk in enumerate(stage.block):
                x = blk(x)
                if idx < len(stage.attn):
                    x = stage.attn[idx](x)
            if getattr(stage, "upsample", None) is not None:
                x = stage.upsample(x)
        x = self.conv_out(nn.silu(self.norm_out(x)))
        # Crop/pad to the causal target shape (upstream ``_adjust_output_shape``).
        target_time = t * 4 - 3 if self.causality_axis != "none" else t * 4
        current_time, current_freq = x.shape[2], x.shape[3]
        x = x[:, :, :min(current_time, target_time), :min(current_freq, self.mel_bins)]
        pad_t = target_time - x.shape[2]
        pad_f = self.mel_bins - x.shape[3]
        if pad_t > 0 or pad_f > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, pad_t), (0, pad_f)])
        return x.astype(output_dtype)


class _IdentityAudio(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


# ---------------------------------------------------------------------------
# BigVGAN vocoder (+ bandwidth extension)
# ---------------------------------------------------------------------------


class _AMPBlock1(nn.Module):
    """BigVGAN v2 AMP block (SnakeBeta activation, dilated convs)."""

    def __init__(self, channels: int, kernel_size: int, dilations: list[int], activation: str = "snakebeta"):
        super().__init__()
        self.alpha1 = mx.zeros((channels,))
        self.alpha2 = mx.zeros((channels,))
        self.beta1 = mx.zeros((channels,))
        self.beta2 = mx.zeros((channels,))
        self.convs1 = [nn.Conv1d(channels, channels, kernel_size, dilation=d) for d in dilations]
        self.convs2 = [nn.Conv1d(channels, channels, kernel_size, dilation=1) for _ in dilations]

    @staticmethod
    def _act(x: mx.array, alpha: mx.array, beta: mx.array) -> mx.array:
        # x: (B, C, L) channel-first
        a = mx.exp(alpha).reshape(1, -1, 1)
        b = mx.exp(beta).reshape(1, -1, 1)
        return x + (1.0 / (b + 1e-9)) * mx.square(mx.sin(a * x))

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = self._act(x, self.alpha1, self.beta1)
            xt = _conv1d_same(conv1, xt)
            xt = self._act(xt, self.alpha2, self.beta2)
            xt = _conv1d_same(conv2, xt)
            x = x + xt
        return x


class _ResBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: list[int]):
        super().__init__()
        self.convs1 = [nn.Conv1d(channels, channels, kernel_size, dilation=d) for d in dilations]
        self.convs2 = [nn.Conv1d(channels, channels, kernel_size, dilation=1) for _ in dilations]

    def __call__(self, x: mx.array) -> mx.array:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = mx.maximum(x, _LRELU_SLOPE * x)
            xt = _conv1d_same(conv1, xt)
            xt = mx.maximum(xt, _LRELU_SLOPE * xt)
            xt = _conv1d_same(conv2, xt)
            x = x + xt
        return x


def _conv1d_same(conv: nn.Conv1d, x: mx.array) -> mx.array:
    """'same' padding for Conv1d (centered, causal-free); x is (B, C, L) channel-first."""
    k = conv.weight.shape[1]
    dilation = int(getattr(conv, "dilation", 1) or 1)
    k_eff = (k - 1) * dilation + 1
    left = (k_eff - 1) // 2
    right = k_eff - 1 - left
    if left or right:
        x = mx.pad(x, [(0, 0), (0, 0), (left, right)])
    y = mx.conv1d(x.transpose(0, 2, 1), conv.weight, dilation=dilation)
    y = y + conv.bias if getattr(conv, "bias", None) is not None else y
    return y.transpose(0, 2, 1)


class LTX25Vocoder(nn.Module):
    """BigVGAN vocoder (log-mel spectrogram → waveform)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        resblock_kernel_sizes = [int(x) for x in cfg.get("resblock_kernel_sizes", [3, 7, 11])]
        upsample_rates = [int(x) for x in cfg.get("upsample_rates", [6, 5, 2, 2, 2])]
        upsample_kernel_sizes = [int(x) for x in cfg.get("upsample_kernel_sizes", [16, 15, 8, 4, 4])]
        resblock_dilation_sizes = [
            [int(d) for d in dils] for dils in cfg.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        ]
        upsample_initial_channel = int(cfg.get("upsample_initial_channel", 1024))
        resblock = str(cfg.get("resblock", "1"))
        self.output_sampling_rate = int(cfg.get("output_sampling_rate") or 24000)
        self.apply_final_activation = bool(cfg.get("apply_final_activation", True))
        self.use_tanh_at_final = bool(cfg.get("use_tanh_at_final", True))
        use_bias_at_final = bool(cfg.get("use_bias_at_final", True))

        self.is_amp = resblock == "AMP1"
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = nn.Conv1d(128, upsample_initial_channel, 7, padding=3)
        self.ups = [
            nn.ConvTranspose1d(
                upsample_initial_channel // (2**i),
                upsample_initial_channel // (2 ** (i + 1)),
                kernel_size,
                stride,
                padding=(kernel_size - stride) // 2,
            )
            for i, (stride, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        ]
        final_channels = upsample_initial_channel // (2 ** len(upsample_rates))
        self.resblocks: list[Any] = []
        for i in range(len(upsample_rates)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                if self.is_amp:
                    self.resblocks.append(_AMPBlock1(ch, kernel_size, dilations, activation=str(cfg.get("activation", "snakebeta"))))
                else:
                    self.resblocks.append(_ResBlock1(ch, kernel_size, dilations))
        self.act_post = _AMPBlock1._act if self.is_amp else None
        self._amp_post_alpha = mx.zeros((final_channels,))
        self._amp_post_beta = mx.zeros((final_channels,))
        self.conv_post = nn.Conv1d(final_channels, 2, 7, padding=3, bias=use_bias_at_final)

    def __call__(self, x: mx.array) -> mx.array:
        """Mel ``(B, 2, T, mel_bins)`` → waveform ``(B, 2, T_out)``."""
        output_dtype = x.dtype
        x = x.astype(mx.bfloat16)
        x = x.transpose(0, 1, 3, 2)  # (B, 2, mel_bins, T)
        b, s, _, _ = x.shape
        x = x.reshape(b, s * x.shape[2], -1)  # (B, 2*mel_bins, T)
        # MLX conv1d is channel-last: work in (B, L, C) and transpose back.
        y = mx.conv1d(x.transpose(0, 2, 1), self.conv_pre.weight, padding=3)
        y = y + self.conv_pre.bias
        x = y.transpose(0, 2, 1)
        for i in range(self.num_upsamples):
            if not self.is_amp:
                x = mx.maximum(x, _LRELU_SLOPE * x)
            u = self.ups[i]
            y = mx.conv_transpose1d(x.transpose(0, 2, 1), u.weight, stride=u.stride, padding=u.padding)
            y = y + u.bias if getattr(u, "bias", None) is not None else y
            x = y.transpose(0, 2, 1)
            start = i * self.num_kernels
            outs = mx.stack([self.resblocks[idx](x) for idx in range(start, start + self.num_kernels)], axis=0)
            x = outs.mean(axis=0)
        if self.is_amp:
            x = _AMPBlock1._act(x, self._amp_post_alpha, self._amp_post_beta)
        else:
            x = mx.maximum(x, _LRELU_SLOPE * x)
        y = mx.conv1d(x.transpose(0, 2, 1), self.conv_post.weight, padding=3)
        y = y + self.conv_post.bias if getattr(self.conv_post, "bias", None) is not None else y
        x = y.transpose(0, 2, 1)
        if self.apply_final_activation:
            x = mx.tanh(x) if self.use_tanh_at_final else mx.clip(x, -1.0, 1.0)
        return x.astype(output_dtype)


class _STFTFn(nn.Module):
    """Causal STFT via Conv1d with checkpoint-provided DFT×Hann bases."""

    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.forward_basis = mx.zeros((n_freqs * 2, 1, filter_length))
        self.inverse_basis = mx.zeros((n_freqs * 2, 1, filter_length))

    def __call__(self, y: mx.array) -> tuple[mx.array, mx.array]:
        # y: (B, T)
        b = y.shape[0]
        y = y[:, None, :]  # (B, 1, T) → channel-last (B, T, 1)
        left_pad = max(0, self.win_length - self.hop_length)
        y = mx.pad(y, [(0, 0), (0, 0), (left_pad, 0)])
        y_cl = y.transpose(0, 2, 1)
        basis = self.forward_basis.transpose(0, 2, 1).astype(y.dtype)  # (O, k, I)
        spec = mx.conv1d(y_cl, basis, stride=self.hop_length)  # (B, T_f, 2nf)
        spec = spec.transpose(0, 2, 1)  # (B, 2nf, T_f)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = mx.sqrt(real * real + imag * imag)
        phase = mx.arctan2(imag.astype(mx.float32), real.astype(mx.float32)).astype(real.dtype)
        return magnitude, phase


class _MelSTFT(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int, n_mel_channels: int):
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.mel_basis = mx.zeros((n_mel_channels, n_freqs))

    def mel_spectrogram(self, y: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        magnitude, phase = self.stft_fn(y)
        energy = mx.sqrt(mx.sum(magnitude * magnitude, axis=1))
        mel = mx.matmul(self.mel_basis.astype(magnitude.dtype), magnitude)
        log_mel = mx.log(mx.maximum(mel, 1e-5))
        return log_mel, magnitude, phase, energy


class _HannResampler(nn.Module):
    """Hann-windowed sinc resampler (torchaudio-compatible), computed at runtime."""

    def __init__(self, ratio: int):
        super().__init__()
        self.ratio = ratio
        rolloff = 0.99
        lowpass_filter_width = 6
        width = int(math_ceil(lowpass_filter_width / rolloff))
        self.kernel_size = 2 * width * ratio + 1
        self.pad = width
        self.pad_left = 2 * width * ratio
        self.pad_right = self.kernel_size - ratio
        time_axis = (np.arange(self.kernel_size) / ratio - width) * rolloff
        time_clamped = np.clip(time_axis, -lowpass_filter_width, lowpass_filter_width)
        window = np.cos(time_clamped * np.pi / lowpass_filter_width / 2) ** 2
        sinc_filter = (np.sinc(time_axis) * window * rolloff / ratio).reshape(1, 1, -1)
        self.filter = mx.array(sinc_filter.astype(np.float32))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T); MLX conv_transpose1d is channel-last (B, L, C_in) with weight (C_out, k, C_in).
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad, self.pad)])
        x_cl = x.transpose(0, 2, 1)  # (B, T+2p, C)
        base = self.filter.transpose(0, 2, 1).astype(x.dtype)  # (1, k, 1)
        ups = []
        for ch in range(x_cl.shape[-1]):
            channel = x_cl[:, :, ch:ch + 1]  # (B, L, 1)
            ups.append(mx.conv_transpose1d(channel, base, stride=self.ratio) * self.ratio)
        up = mx.concatenate(ups, axis=-1)  # (B, L', C)
        up = up.transpose(0, 2, 1)  # (B, C, L')
        return up[..., self.pad_left:-self.pad_right]


def math_ceil(value: float) -> int:
    return int(-(-value // 1))


class LTX25VocoderWithBWE(nn.Module):
    """Vocoder + bandwidth extension (BigVGAN v2 residual + sinc upsampling)."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        vocoder_cfg = dict(cfg.get("vocoder", {}))
        bwe_cfg = dict(cfg.get("bwe", {}))
        if not bwe_cfg:
            raise RuntimeError("LTX 2.5 vocoder config lacks 'bwe' section.")
        self.input_sampling_rate = int(bwe_cfg.get("input_sampling_rate", 24000))
        self.output_sampling_rate = int(bwe_cfg.get("output_sampling_rate", 48000))
        self.hop_length = int(bwe_cfg.get("hop_length", 300))
        vocoder_cfg = dict(vocoder_cfg)
        vocoder_cfg["output_sampling_rate"] = self.input_sampling_rate
        bwe_gen_cfg = dict(bwe_cfg)
        bwe_gen_cfg["apply_final_activation"] = False
        bwe_gen_cfg["output_sampling_rate"] = self.output_sampling_rate
        self.vocoder = LTX25Vocoder(vocoder_cfg)
        self.bwe_generator = LTX25Vocoder(bwe_gen_cfg)
        self.mel_stft = _MelSTFT(
            filter_length=int(bwe_cfg.get("n_fft", 1024)),
            hop_length=self.hop_length,
            win_length=int(bwe_cfg.get("n_fft", 1024)),
            n_mel_channels=int(bwe_cfg.get("num_mels", 80)),
        )
        self.resampler = _HannResampler(self.output_sampling_rate // self.input_sampling_rate)

    def __call__(self, mel_spec: mx.array) -> mx.array:
        """Mel ``(B, 2, T, mel_bins)`` → waveform ``(B, 2, T_out)`` at output rate."""
        input_dtype = mel_spec.dtype
        mel_spec = mel_spec.astype(mx.float32)
        x = self.vocoder(mel_spec)  # (B, 2, T_low)
        length_low = x.shape[-1]
        output_length = length_low * self.output_sampling_rate // self.input_sampling_rate
        remainder = length_low % self.hop_length
        if remainder != 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, self.hop_length - remainder)])
        b, n_channels, _ = x.shape
        flat = x.reshape(b * n_channels, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        mel = mel.reshape(b, n_channels, mel.shape[1], mel.shape[2])  # (B, C, n_mels, T_f)
        mel_for_bwe = mel.transpose(0, 1, 3, 2)  # (B, C, T_f, n_mels)
        residual = self.bwe_generator(mel_for_bwe.astype(mx.float32))
        skip = self.resampler(x)
        out = mx.clip(residual + skip, -1.0, 1.0)[..., :output_length]
        return out.astype(input_dtype)


# ---------------------------------------------------------------------------
# Bundle-facing API
# ---------------------------------------------------------------------------


def _normalize_vae_keys(weights: dict[str, Any]) -> dict[str, Any]:
    """Map official checkpoint keys to module attribute names."""
    out: dict[str, Any] = {}
    for key, tensor in weights.items():
        nk = key
        nk = nk.replace(".per_channel_statistics.std-of-means", ".per_channel_statistics_std")
        nk = nk.replace(".per_channel_statistics.mean-of-means", ".per_channel_statistics_mean")
        nk = nk.replace(".time_embedder.timestep_embedder.", ".time_embedder.")
        nk = nk.replace(".last_time_embedder.timestep_embedder.", ".last_time_embedder.")
        out[nk] = tensor
    return out


def _strip_component_prefix(weights: dict[str, Any], *, component: str, vae_prefix: str = "vae.") -> dict[str, Any]:
    """Accept both ``vae.<component>.*`` (official) and bare ``<component>.*`` (Comfy-split) keys."""
    out: dict[str, Any] = {}
    stat_prefix = "per_channel_statistics."
    stats: dict[str, Any] = {}
    for key, tensor in weights.items():
        nk = key
        if nk.startswith(f"{vae_prefix}{component}."):
            nk = nk[len(f"{vae_prefix}{component}."):]
        elif nk.startswith(f"{component}."):
            nk = nk[len(f"{component}."):]
        elif nk.startswith(f"{vae_prefix}{stat_prefix}"):
            stats[nk[len(f"{vae_prefix}{stat_prefix}"):]] = tensor
            continue
        elif nk.startswith(stat_prefix):
            stats[nk[len(stat_prefix):]] = tensor
            continue
        else:
            continue
        out[nk] = tensor
    out.update(_normalize_vae_keys(stats))
    return out


def load_ltx25_video_decoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX25VideoDecoder:
    cfg = get_vae_config(bundle_root)
    decoder = LTX25VideoDecoder(cfg)
    weights = _strip_component_prefix(
        _load_bundle_weights(bundle_root, _VIDEO_VAE_FILE, "", load_fn=load_fn), component="decoder"
    )
    params = _flat_param_map(decoder)
    missing = [k for k in params if k not in weights]
    if missing:
        raise RuntimeError(f"LTX 2.5 video VAE decoder weights missing: {missing[:8]}")
    decoder.load_weights(list(weights.items()), strict=False)
    return decoder


def load_ltx25_video_encoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX25VideoEncoder:
    cfg = get_vae_config(bundle_root)
    encoder = LTX25VideoEncoder(cfg)
    weights = _strip_component_prefix(
        _load_bundle_weights(bundle_root, _VIDEO_VAE_FILE, "", load_fn=load_fn), component="encoder"
    )
    params = _flat_param_map(encoder)
    missing = [k for k in params if k not in weights]
    if missing:
        raise RuntimeError(f"LTX 2.5 video VAE encoder weights missing: {missing[:8]}")
    encoder.load_weights(list(weights.items()), strict=False)
    return encoder


def load_ltx25_audio_decoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX25AudioDecoder:
    cfg = get_audio_config(bundle_root)
    decoder = LTX25AudioDecoder(cfg)
    weights = _strip_component_prefix(
        _load_bundle_weights(bundle_root, _AUDIO_VAE_FILE, "", load_fn=load_fn),
        component="decoder",
        vae_prefix="audio_vae.",
    )
    params = _flat_param_map(decoder)
    missing = [k for k in params if k not in weights]
    if missing:
        raise RuntimeError(f"LTX 2.5 audio VAE decoder weights missing: {missing[:8]}")
    decoder.load_weights(list(weights.items()), strict=False)
    return decoder


def load_ltx25_vocoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX25Vocoder | LTX25VocoderWithBWE:
    cfg = get_vocoder_config(bundle_root)
    if "bwe" in cfg:
        module = LTX25VocoderWithBWE(cfg)
    else:
        module = LTX25Vocoder(dict(cfg.get("vocoder", cfg)))
    weights = _normalize_vae_keys(_load_bundle_weights(bundle_root, _AUDIO_VAE_FILE, "vocoder.", load_fn=load_fn))
    # Strip the single extra "vocoder." prefix emitted by BWE keys.
    weights = {k[len("vocoder."):] if k.startswith("vocoder.") else k: v for k, v in weights.items()}
    params = _flat_param_map(module)
    missing = [k for k in params if k not in weights]
    if missing:
        raise RuntimeError(f"LTX 2.5 vocoder weights missing: {missing[:8]}")
    module.load_weights(list(weights.items()), strict=False)
    return module


def _flat_param_map(module: nn.Module) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(module.parameters()))


def decode_audio_latent(
    bundle_root: Path,
    audio_latent: mx.array,
    *,
    load_fn: Any | None = None,
) -> tuple[np.ndarray, int]:
    """Decode audio latent ``(B, 8, T, 16)`` → waveform numpy (2, T) + sample rate."""
    decoder = load_ltx25_audio_decoder(bundle_root, load_fn=load_fn)
    spec = decoder.decode(audio_latent)
    _materialize(spec)
    vocoder = load_ltx25_vocoder(bundle_root, load_fn=load_fn)
    waveform = vocoder(spec)
    waveform_np = waveform[0].astype(mx.float32)
    _materialize(waveform_np)
    waveform_np = np.asarray(waveform_np)
    return waveform_np, vocoder.output_sampling_rate


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH; install with: brew install ffmpeg")
    return path


def mux_video_audio_mp4(
    bundle_root: Path,
    video_latent: mx.array,
    audio_latent: mx.array,
    output_path: str,
    *,
    frame_rate: float = 24.0,
    load_fn: Any | None = None,
    seed: int | None = None,
) -> str:
    """Decode video + audio latents and mux into an mp4 (ffmpeg required)."""
    from PIL import Image

    decoder = load_ltx25_video_decoder(bundle_root, load_fn=load_fn)
    pixels = decoder.decode(video_latent, seed=seed)
    pixels_np = pixels[0].astype(mx.float32)
    _materialize(pixels_np)
    pixels = np.asarray(pixels_np)  # (3, F, H, W)
    frames = []
    for idx in range(pixels.shape[1]):
        frame = pixels[:, idx, :, :].transpose(1, 2, 0)
        frame = np.clip((frame + 1.0) / 2.0, 0.0, 1.0)
        frames.append(Image.fromarray((frame * 255.0).astype(np.uint8)))

    waveform, sample_rate = decode_audio_latent(bundle_root, audio_latent, load_fn=load_fn)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ltx25-mux-") as tmp:
        tmp_path = Path(tmp)
        audio_path = tmp_path / "audio.wav"
        channels = waveform.shape[0] if waveform.ndim > 1 else 1
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((np.clip(waveform, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())
        video_only = tmp_path / "video_only.mp4"
        ffmpeg = _find_ffmpeg()
        cmd_v = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{frames[0].width}x{frames[0].height}",
            "-r", f"{frame_rate:.6f}",
            "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            str(video_only),
        ]
        raw = b"".join(np.asarray(f, dtype=np.uint8).tobytes() for f in frames)
        proc = subprocess.run(cmd_v, input=raw, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"LTX 2.5 mux: video encode failed: {(proc.stderr or b'').decode(errors='ignore')[:1200]}")
        cmd_m = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_only), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_path),
        ]
        proc = subprocess.run(cmd_m, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"LTX 2.5 mux: audio mux failed: {(proc.stderr or b'').decode(errors='ignore')[:1200]}")
    return str(output_path)
