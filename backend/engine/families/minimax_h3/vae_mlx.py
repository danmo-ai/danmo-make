"""MLX ports of MiniMax-H3 video VAE (CNN encoder + ViT decoder) and audio VAE (DAC + BigVGAN).

Decode is the generation hot path; encode supports fl2va keyframe conditioning.
Latents use per-channel ``latents_mean`` / ``latents_std`` (not a scalar scaling_factor).
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval
from backend.utils.video_sr_ffmpeg import require_ffmpeg

# ImageNet pixel convention over [0, 1] (matches packing.MINIMAX_H3_PIXEL_*).
_PIXEL_MEAN = mx.array([0.485, 0.456, 0.406], dtype=mx.float32).reshape(1, 3, 1, 1, 1)
_PIXEL_STD = mx.array([0.229, 0.224, 0.225], dtype=mx.float32).reshape(1, 3, 1, 1, 1)


def _eval(x: Any | None = None) -> None:
    run_eval(x) if x is not None else run_eval()


def _as_tuple(v: Any) -> tuple:
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return (v,)


def _conv3d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    if w.ndim != 5:
        return w
    # Torch (O, I, kT, kH, kW) → MLX (O, kT, kH, kW, I)
    o, d1, d2, d3, d4 = (int(x) for x in w.shape)
    if d2 <= 7 and d3 <= 7 and d4 <= 7 and d1 >= d2:
        return mx.transpose(w, (0, 2, 3, 4, 1))
    return w


def _conv1d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    if w.ndim != 3:
        return w
    # Torch (O, I, K) → MLX (O, K, I)
    o, d1, d2 = (int(x) for x in w.shape)
    if d2 <= 64 and d1 >= 1:
        return mx.transpose(w, (0, 2, 1))
    return w


def _conv_transpose1d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    if w.ndim != 3:
        return w
    # Torch ConvTranspose1d: (I, O, K) → MLX (O, K, I) roughly via transpose
    # MLX ConvTranspose1d weight: (out_channels, kernel, in_channels) — check docs
    return mx.transpose(w, (1, 2, 0))


def _fuse_weight_norm(weight_g: Any, weight_v: Any, *, eps: float = 1e-9) -> np.ndarray:
    g = np.asarray(weight_g, dtype=np.float32)
    v = np.asarray(weight_v, dtype=np.float32)
    norm = np.sqrt(np.sum(v * v, axis=tuple(range(1, v.ndim)), keepdims=True) + eps)
    return (g * v / norm).astype(np.float32)


def _pad_reflect_spatial(x: mx.array, pad: int) -> mx.array:
    """Symmetric reflect pad on H and W of NCTHW (PyTorch F.pad reflect)."""
    if pad <= 0:
        return x
    # W
    left = x[..., 1 : pad + 1][..., ::-1]
    right = x[..., -pad - 1 : -1][..., ::-1]
    x = mx.concatenate([left, x, right], axis=-1)
    # H
    top = x[..., 1 : pad + 1, :][..., ::-1, :]
    bot = x[..., -pad - 1 : -1, :][..., ::-1, :]
    return mx.concatenate([top, x, bot], axis=-2)


def _pad_const_temporal_front(x: mx.array, pad_t: int) -> mx.array:
    if pad_t <= 0:
        return x
    b, c, _t, h, w = x.shape
    z = mx.zeros((b, c, pad_t, h, w), dtype=x.dtype)
    return mx.concatenate([z, x], axis=2)


def _pad_edge_spatial_br(x: mx.array) -> mx.array:
    """Asymmetric bottom/right pad of 1 (downsample path)."""
    x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 0)], mode="edge")
    x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 0), (0, 1)], mode="edge")
    return x


# ---------------------------------------------------------------------------
# Video VAE — encoder CNN
# ---------------------------------------------------------------------------


class MiniMaxH3VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # MLX Conv3d expects N D H W C
        if self.spatial_padding > 0:
            if self.spatial_padding_mode != "reflect":
                raise RuntimeError(
                    f"MiniMax-H3 video VAE expects spatial_padding_mode='reflect', "
                    f"got {self.spatial_padding_mode!r}"
                )
            hidden_states = _pad_reflect_spatial(hidden_states, self.spatial_padding)
        if self.temporal_padding > 0:
            hidden_states = _pad_const_temporal_front(hidden_states, self.temporal_padding)
        x = mx.transpose(hidden_states, (0, 2, 3, 4, 1))  # N C T H W → N T H W C
        x = self.conv(x)
        return mx.transpose(x, (0, 4, 1, 2, 3))


class MiniMaxH3VideoGroupNorm(nn.Module):
    """GroupNorm with T folded into batch (per-frame stats).

    MLX ``nn.GroupNorm`` (including ``pytorch_compatible=True``) applies affine over the
    **last** axis — feed NHWC, not NCHW.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True, pytorch_compatible=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        b, c, t, h, w = hidden_states.shape
        x = mx.transpose(hidden_states, (0, 2, 1, 3, 4)).reshape(b * t, c, h, w)
        x = mx.transpose(x, (0, 2, 3, 1))  # NHWC for MLX GroupNorm
        x = self.norm(x.astype(mx.float32)).astype(hidden_states.dtype)
        x = mx.transpose(x, (0, 3, 1, 2))  # NCHW
        return x.reshape(b, t, c, h, w).transpose(0, 2, 1, 3, 4)


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ):
        super().__init__()
        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels, out_channels, 3, spatial_padding=1, temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels, out_channels, 3, spatial_padding=1, temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, 1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = nn.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = nn.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ):
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels, out_channels, 3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0, temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.spatial_stride == 2:
            if self.spatial_padding_mode == "reflect":
                # bottom/right pad 1 — edge approximates released tiling path for odd sizes
                hidden_states = _pad_edge_spatial_br(hidden_states)
            else:
                hidden_states = _pad_edge_spatial_br(hidden_states)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ):
        super().__init__()
        self.resnets = [
            MiniMaxH3VideoResnetBlock3d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                spatial_padding_mode=spatial_padding_mode,
            )
            for i in range(num_layers)
        ]
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = [
                MiniMaxH3VideoDownsample3d(
                    out_channels, out_channels,
                    temporal_stride=temporal_downsample_factor,
                    spatial_stride=spatial_downsample_factor,
                    spatial_padding_mode=spatial_padding_mode,
                )
            ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for down in self.downsamplers:
                hidden_states = down(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ):
        super().__init__()
        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels, block_out_channels[0], 3,
            spatial_padding=1, temporal_padding=2, spatial_padding_mode=spatial_padding_mode,
        )
        block_in = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = [
            MiniMaxH3VideoDownBlock3d(
                in_channels=block_in[i],
                out_channels=block_out_channels[i],
                num_layers=layers_per_block,
                temporal_downsample_factor=temporal_downsample_factors[i],
                spatial_downsample_factor=spatial_downsample_factors[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                spatial_padding_mode=spatial_padding_mode,
            )
            for i in range(len(block_out_channels))
        ]
        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1], out_channels, 3,
            spatial_padding=1, temporal_padding=2, spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.conv_in(hidden_states)
        for block in self.down_blocks:
            hidden_states = block(hidden_states)
        hidden_states = nn.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


# ---------------------------------------------------------------------------
# Video VAE — ViT decoder
# ---------------------------------------------------------------------------


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3):
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / (theta ** mx.arange(0, 1, 2 * num_axes / dim, dtype=mx.float32))
        self.inv_freq = inv_freq

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (B, S, 3)
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        b, s, _a, f = angles.shape
        angles = angles.reshape(b, s, _a * f)
        angles = mx.concatenate([angles, angles], axis=-1)[:, :, None, :]
        return mx.cos(angles), mx.sin(angles)


class MiniMaxH3VideoAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        inner = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        # elementwise_affine=False in Diffusers — MLX RMSNorm always has weight; zero it after load if needed
        self.to_q = nn.Linear(dim, inner, bias=bias)
        self.to_k = nn.Linear(dim, inner, bias=bias)
        self.to_v = nn.Linear(dim, inner, bias=bias)
        self.to_out = [nn.Linear(inner, dim, bias=bias), nn.Identity()]

    def __call__(self, hidden_states: mx.array, rotary_emb: tuple[mx.array, mx.array] | None = None) -> mx.array:
        b, s, _ = hidden_states.shape
        q = self.to_q(hidden_states).reshape(b, s, self.heads, self.dim_head)
        k = self.to_k(hidden_states).reshape(b, s, self.heads, self.dim_head)
        v = self.to_v(hidden_states).reshape(b, s, self.heads, self.dim_head)
        q = self.norm_q(q.astype(mx.float32)).astype(q.dtype)
        k = self.norm_k(k.astype(mx.float32)).astype(k.dtype)
        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.astype(q.dtype)
            sin = sin.astype(q.dtype)
            rotary_dim = int(cos.shape[-1])
            def _rope(x):
                xr, xp = x[..., :rotary_dim], x[..., rotary_dim:]
                a, b_ = mx.split(xr, 2, axis=-1)
                rot = mx.concatenate([-b_, a], axis=-1)
                return mx.concatenate([xr * cos + rot * sin, xp], axis=-1)
            q, k = _rope(q), _rope(k)
        qh = mx.transpose(q, (0, 2, 1, 3))
        kh = mx.transpose(k, (0, 2, 1, 3))
        vh = mx.transpose(v, (0, 2, 1, 3))
        out = scaled_dot_product_attention_bhsd_mx(mx, qh, kh, vh, scale=self.scale)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, s, -1)
        return self.to_out[0](out)


class _VideoSwiGLU(nn.Module):
    def __init__(self, dim: int, inner: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim, inner * 2, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        value, gate = mx.split(self.proj(x), 2, axis=-1)
        return value * nn.silu(gate)


class MiniMaxH3VideoFeedForward(nn.Module):
    """SwiGLU FFN matching Diffusers ``FeedForward(..., activation_fn='swiglu')``."""

    def __init__(self, dim: int, mult: int = 4, bias: bool = True):
        super().__init__()
        inner = dim * mult
        self.net = [_VideoSwiGLU(dim, inner, bias=bias), nn.Identity(), nn.Linear(inner, dim, bias=bias)]

    def __call__(self, x: mx.array) -> mx.array:
        for m in self.net:
            x = m(x)
        return x


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ffn_mult: int = 4, eps: float = 1e-5, bias: bool = True):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps)
        self.attn = MiniMaxH3VideoAttention(dim, heads, dim_head, eps=eps, bias=bias)
        self.scale1 = mx.zeros((dim,))
        self.norm2 = nn.RMSNorm(dim, eps=eps)
        self.ff = MiniMaxH3VideoFeedForward(dim, mult=ffn_mult, bias=bias)
        self.scale2 = mx.zeros((dim,))

    def __call__(self, hidden_states: mx.array, rotary_emb: tuple[mx.array, mx.array] | None = None) -> mx.array:
        normed = self.norm1(hidden_states.astype(mx.float32)).astype(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(normed, rotary_emb) * self.scale1
        normed = self.norm2(hidden_states.astype(mx.float32)).astype(hidden_states.dtype)
        hidden_states = hidden_states + self.ff(normed) * self.scale2
        return hidden_states


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = mx.zeros((1, num_register_tokens, dim))
        # Upstream ``mask_token`` (CLS-like) — MiniMax packs ship ``decoder.mask_token``.
        self.mask_token = mx.zeros((1, 1, dim))
        self.transformer_blocks = [
            MiniMaxH3VideoTransformerBlock(
                dim=dim, heads=num_attention_heads, dim_head=attention_head_dim,
                ffn_mult=ffn_mult, eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        b, c, t, h, w = hidden_states.shape
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 4, 1)).reshape(b, t * h * w, c)
        hidden_states = self.proj_in(hidden_states)
        num_patches = int(hidden_states.shape[1])
        reg = mx.broadcast_to(self.register_tokens, (b, self.num_register_tokens, hidden_states.shape[-1]))
        cls = mx.broadcast_to(self.mask_token.astype(hidden_states.dtype), (b, 1, hidden_states.shape[-1]))
        hidden_states = mx.concatenate([hidden_states, reg, cls], axis=1)

        def _axis_grid(size: int) -> mx.array:
            return 2.0 * (mx.arange(0.5, size, dtype=mx.float32) / size) - 1.0

        gt, gh, gw = _axis_grid(t), _axis_grid(h), _axis_grid(w)
        # meshgrid ij
        tt = mx.broadcast_to(gt.reshape(t, 1, 1), (t, h, w))
        hh = mx.broadcast_to(gh.reshape(1, h, 1), (t, h, w))
        ww = mx.broadcast_to(gw.reshape(1, 1, w), (t, h, w))
        position_ids = mx.stack([tt, hh, ww], axis=-1).reshape(t * h * w, 3)
        position_ids = mx.broadcast_to(position_ids[None, :, :], (b, t * h * w, 3))
        suffix = mx.zeros((b, self.num_register_tokens + 1, 3), dtype=mx.float32)
        position_ids = mx.concatenate([position_ids, suffix], axis=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)[:, :num_patches, :]
        ps, pst = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.reshape(b, t, h, w, self.out_channels, pst, ps, ps)
        hidden_states = mx.transpose(hidden_states, (0, 4, 1, 5, 2, 6, 3, 7))
        return hidden_states.reshape(b, self.out_channels, t * pst, h * ps, w * ps)


class AutoencoderKLMiniMaxH3MLX(nn.Module):
    """Video VAE: causal CNN encoder + ViT decoder."""

    def __init__(self, **cfg: Any):
        super().__init__()
        in_channels = int(cfg.get("in_channels", 3))
        out_channels = int(cfg.get("out_channels", 3))
        latent_channels = int(cfg.get("latent_channels", 24))
        block_out_channels = tuple(int(x) for x in _as_tuple(cfg.get("block_out_channels", (128, 256, 256, 512, 512, 1024))))
        layers_per_block = int(cfg.get("layers_per_block", 2))
        spatial_ds = tuple(int(x) for x in _as_tuple(cfg.get("spatial_downsample_factors", (2, 2, 2, 2, 1, 1))))
        temporal_ds = tuple(int(x) for x in _as_tuple(cfg.get("temporal_downsample_factors", (1, 2, 2, 1, 1, 1))))
        norm_num_groups = int(cfg.get("norm_num_groups", 32))
        norm_eps = float(cfg.get("norm_eps", 1e-6))
        spatial_padding_mode = str(cfg.get("spatial_padding_mode", "reflect"))
        self.latent_channels = latent_channels
        self.spatial_compression_ratio = int(math.prod(spatial_ds))
        self.temporal_compression_ratio = int(math.prod(temporal_ds))
        self.clip_length = int(cfg.get("clip_length", 17))
        self.token_drop = int(cfg.get("token_drop", 3))
        mean = cfg.get("latents_mean", (0.0,) * latent_channels)
        std = cfg.get("latents_std", (1.0,) * latent_channels)
        self.latents_mean = mx.array([float(x) for x in _as_tuple(mean)], dtype=mx.float32).reshape(
            1, latent_channels, 1, 1, 1
        )
        self.latents_std = mx.array([float(x) for x in _as_tuple(std)], dtype=mx.float32).reshape(
            1, latent_channels, 1, 1, 1
        )

        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=in_channels, out_channels=2 * latent_channels,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_ds, temporal_downsample_factors=temporal_ds,
            norm_num_groups=norm_num_groups, norm_eps=norm_eps, spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, 1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent_channels, out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=int(cfg.get("decoder_num_layers", 36)),
            num_attention_heads=int(cfg.get("decoder_num_attention_heads", 32)),
            attention_head_dim=int(cfg.get("decoder_attention_head_dim", 64)),
            num_register_tokens=int(cfg.get("decoder_num_register_tokens", 4)),
            ffn_mult=int(cfg.get("decoder_ffn_mult", 4)),
            rope_theta=float(cfg.get("decoder_rope_theta", 100.0)),
            rope_dim_ratio=float(cfg.get("decoder_rope_dim_ratio", 0.75)),
            norm_eps=float(cfg.get("decoder_norm_eps", 1e-5)),
        )
        self.frame_pre_padding = (-self.clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(self.clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-self.token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)
        self.use_tiling = bool(cfg.get("use_tiling", True))
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AutoencoderKLMiniMaxH3MLX":
        return cls(**cfg)

    def denormalize_latents(self, z: mx.array) -> mx.array:
        return z.astype(mx.float32) * self.latents_std + self.latents_mean

    def normalize_latents(self, z: mx.array) -> mx.array:
        return (z.astype(mx.float32) - self.latents_mean) / self.latents_std

    def _conv3d_ncthw(self, conv: nn.Conv3d, x: mx.array) -> mx.array:
        y = mx.transpose(x, (0, 2, 3, 4, 1))
        y = conv(y)
        return mx.transpose(y, (0, 4, 1, 2, 3))

    def _blend(self, a: mx.array, b: mx.array, blend_extent: int, dim: int) -> mx.array:
        blend_extent = min(int(a.shape[dim]), int(b.shape[dim]), blend_extent)
        if blend_extent <= 0:
            return b
        positions = mx.arange(blend_extent, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).reshape(shape)
        weight_b = (positions / blend_extent).reshape(shape)
        # slice helpers
        def _sl(t, start, end):
            slices = [slice(None)] * t.ndim
            slices[dim] = slice(start, end)
            return t[tuple(slices)]
        blended = _sl(a, -blend_extent, None) * weight_a + _sl(b, 0, blend_extent) * weight_b
        if blend_extent == int(b.shape[dim]):
            return blended
        return mx.concatenate([blended, _sl(b, blend_extent, None)], axis=dim)

    def _decode_clip(self, z: mx.array) -> mx.array:
        z = self._conv3d_ncthw(self.post_quant_conv, z)
        return self.decoder(z)

    def _decode(self, z: mx.array) -> mx.array:
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio
        num_tokens = int(z.shape[2]) + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = mx.concatenate([z, mx.repeat(z[:, :, -1:, :, :], pad_tokens, axis=2)], axis=2)
        decoded_chunks: list[mx.array] = []
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=2)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)
        dec = mx.concatenate(decoded_chunks, axis=2)
        if pad_tokens > 0:
            intra_tail = self.clip_length % temporal_ratio
            num_tokens_before_pad = int(z.shape[2]) - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            if pad_frames > 0:
                dec = dec[:, :, :-pad_frames]
        return dec

    def decode(self, z: mx.array, *, denormalize: bool = True) -> mx.array:
        """Decode denormalized (or normalized if ``denormalize=True``) latents → ImageNet-space NCTHW."""
        if z.ndim != 5:
            raise ValueError(f"video latents must be NCTHW, got {tuple(z.shape)}")
        if denormalize:
            z = self.denormalize_latents(z)
        return self._decode(z)

    def encode_clip(self, x: mx.array) -> mx.array:
        """Encode one temporal clip (keyframe / short clip) → moments ``[B, 2C, T', H', W']``."""
        if x.ndim != 5:
            raise ValueError(f"pixels must be NCTHW, got {tuple(x.shape)}")
        h = self.encoder(x)
        return self._conv3d_ncthw(self.quant_conv, h)

    def encode_sample(self, x: mx.array, *, normalize: bool = True) -> mx.array:
        """Posterior sample + float16 round-trip (PipeNetwork / diffusers keyframe path)."""
        moments = self.encode_clip(x)
        mean, logvar = mx.split(moments, 2, axis=1)
        logvar = mx.clip(logvar, -30.0, 20.0)
        std = mx.exp(0.5 * logvar)
        latent = mean + std * mx.random.normal(mean.shape)
        latent = latent.astype(mx.float16).astype(mx.float32)
        if normalize:
            latent = self.normalize_latents(latent)
        return latent

    def encode_mode(self, x: mx.array, *, normalize: bool = True) -> mx.array:
        """Encode pixels → latent mode, optionally normalized with ``latents_mean/std``."""
        moments = self.encode_clip(x)
        mean, _logvar = mx.split(moments, 2, axis=1)
        if normalize:
            mean = self.normalize_latents(mean)
        return mean

    def load_weights(self, path: str | Path, *, load_fn: Any | None = None) -> None:
        raw = load_weights_dict(load_fn, str(path))
        remapped = remap_minimax_h3_video_vae_weights(raw)
        params = dict(tree_flatten_params(self))
        # Soft buffers / no-affine RMSNorm leaves — keep init when absent from checkpoint.
        allowed_missing = {
            k
            for k in params
            if k == "decoder.rope.inv_freq"
            or k.endswith(".attn.norm_q.weight")
            or k.endswith(".attn.norm_k.weight")
        }
        missing = [k for k in params if k not in remapped and k not in allowed_missing]
        if missing:
            raise RuntimeError(
                f"MiniMax-H3 video VAE weight load failed: {len(missing)}/{len(params)} keys missing "
                f"(example: {missing[:5]})"
            )
        assigned = 0
        for k, v in remapped.items():
            if k not in params:
                continue
            params[k] = v.astype(params[k].dtype) if hasattr(params[k], "dtype") else v
            assigned += 1
        if assigned == 0:
            raise RuntimeError(
                "MiniMax-H3 video VAE load_weights assigned 0 keys — checkpoint layout mismatch "
                f"(saw {len(remapped)} remapped keys, model has {len(params)} params)"
            )
        self.update(tree_unflatten_params(params))


def _strip_model_prefix(key: str) -> str:
    return key[6:] if key.startswith("model.") else key


def _remap_video_vae_encoder_key(key: str) -> str:
    """Map MiniMax / Comfy compact encoder keys onto nested CausalConv3d + GroupNorm params."""
    if key.startswith("encoder.down."):
        key = "encoder.down_blocks." + key[len("encoder.down.") :]
        key = key.replace(".block.", ".resnets.")
        key = key.replace(".downsample.", ".downsamplers.0.")
        key = key.replace(".nin_shortcut.", ".conv_shortcut.")
    # Wrap nn.Conv3d leaves inside MiniMaxH3VideoCausalConv3d (.conv).
    for leaf in (
        ".conv_in.",
        ".conv_out.",
        ".conv1.",
        ".conv2.",
        ".conv_shortcut.",
    ):
        if leaf in key and f"{leaf}conv." not in key:
            key = key.replace(leaf, f"{leaf}conv.", 1)
    # Downsample: …downsamplers.0.conv.{weight,bias} → …conv.conv.{weight,bias}
    if ".downsamplers." in key and ".conv.conv." not in key and ".conv." in key:
        # after rename: encoder.down_blocks.N.downsamplers.0.conv.weight
        key = key.replace(".downsamplers.0.conv.", ".downsamplers.0.conv.conv.")
    # Wrap GroupNorm leaves inside MiniMaxH3VideoGroupNorm (.norm).
    for leaf in (".norm1.", ".norm2.", ".norm_out."):
        if leaf in key and f"{leaf}norm." not in key:
            key = key.replace(leaf, f"{leaf}norm.", 1)
    return key


def _remap_video_vae_decoder_key(key: str) -> str:
    """Map compact decoder keys (x_embedder / fused qkv / SwiGLU) onto Diffusers-style modules."""
    if key.startswith("decoder.x_embedder."):
        key = "decoder.proj_in." + key[len("decoder.x_embedder.") :]
    if ".attn.to_out." in key and ".attn.to_out.0." not in key:
        key = key.replace(".attn.to_out.", ".attn.to_out.0.")
    if ".ff.w1." in key:
        key = key.replace(".ff.w1.", ".ff.net.0.proj.")
    if ".ff.w2." in key:
        key = key.replace(".ff.w2.", ".ff.net.2.")
    return key


def remap_minimax_h3_video_vae_weights(weights: dict[str, Any]) -> dict[str, mx.array]:
    """Remap MiniMax-H3 ``video_vae`` checkpoint keys onto ``AutoencoderKLMiniMaxH3MLX`` params.

    Upstream packs use compact names (``encoder.down.*.block`` / fused ``to_qkv`` / ``ff.w1``).
    Our MLX modules mirror Diffusers nesting (``down_blocks`` / ``resnets`` / separate ``to_q|k|v``).
    """
    out: dict[str, mx.array] = {}
    for raw_key, raw_val in weights.items():
        key = _strip_model_prefix(raw_key)
        arr = raw_val if isinstance(raw_val, mx.array) else mx.array(raw_val)

        if key in ("latents_mean", "latents_std") and arr.ndim == 1:
            arr = arr.reshape(1, -1, 1, 1, 1)

        # Fused QKV → separate projections (same layout as Diffusers Attention).
        if ".attn.to_qkv.weight" in key or key.endswith(".attn.to_qkv.weight"):
            base = key[: -len(".attn.to_qkv.weight")]
            q, k, v = mx.split(arr, 3, axis=0)
            out[f"{base}.attn.to_q.weight"] = q
            out[f"{base}.attn.to_k.weight"] = k
            out[f"{base}.attn.to_v.weight"] = v
            continue
        if ".attn.to_qkv.bias" in key or key.endswith(".attn.to_qkv.bias"):
            base = key[: -len(".attn.to_qkv.bias")]
            q, k, v = mx.split(arr, 3, axis=0)
            out[f"{base}.attn.to_q.bias"] = q
            out[f"{base}.attn.to_k.bias"] = k
            out[f"{base}.attn.to_v.bias"] = v
            continue

        if key.startswith("encoder."):
            key = _remap_video_vae_encoder_key(key)
        elif key.startswith("decoder."):
            key = _remap_video_vae_decoder_key(key)

        if key.endswith(".weight") and arr.ndim == 5:
            arr = _conv3d_weight_torch_to_mlx(arr)
        out[key] = arr
    return out


# ---------------------------------------------------------------------------
# Audio VAE
# ---------------------------------------------------------------------------


class MiniMaxH3AudioSnake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        return x + (1.0 / (self.alpha + 1e-9)) * (mx.sin(self.alpha * x) ** 2)


class MiniMaxH3AudioSnakeBeta(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha.reshape(1, -1, 1))
        beta = mx.exp(self.beta.reshape(1, -1, 1))
        return x + (1.0 / (beta + 1e-9)) * (mx.sin(alpha * x) ** 2)


def _kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    # Kaiser window via numpy for portability
    n = np.arange(kernel_size, dtype=np.float64)
    window = np.i0(beta * np.sqrt(np.maximum(0, 1 - ((n - half_size) / half_size) ** 2))) / np.i0(beta)
    if kernel_size % 2 == 0:
        time = np.arange(-half_size, half_size) + 0.5
    else:
        time = np.arange(kernel_size) - half_size
    filt = 2 * cutoff * window * np.sinc(2 * cutoff * time)
    filt = filt / filt.sum()
    return mx.array(filt.reshape(1, 1, kernel_size).astype(np.float32))


class MiniMaxH3AudioLowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float, half_width: float, stride: int, kernel_size: int):
        super().__init__()
        even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = _kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad_left, self.pad_right)], mode="edge")
        return _depthwise_conv1d(x, self.filter, stride=self.stride)


def _depthwise_conv1d(x_ncl: mx.array, filt_11k: mx.array, stride: int = 1) -> mx.array:
    """Depthwise conv: ``x`` NCTHW-like ``[N,C,L]``, filter ``[1,1,K]``."""
    n, c, length = (int(x) for x in x_ncl.shape)
    k = int(filt_11k.shape[-1])
    x_nlc = mx.transpose(x_ncl, (0, 2, 1))
    # Expand filter to (C, K, 1) for MLX Conv1d groups — fall back to einsum unfold
    # unfold
    out_len = (length - k) // stride + 1
    # Use as_strided-like via indexing
    idx = stride * mx.arange(out_len)[:, None] + mx.arange(k)[None, :]
    patches = x_ncl[:, :, idx]  # N C out K
    f = filt_11k.reshape(1, 1, 1, k)
    y = mx.sum(patches * f, axis=-1)  # N C out
    return y


class MiniMaxH3AudioUpSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        self.filter = _kaiser_sinc_filter1d(0.5 / ratio, 0.6 / ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (0, 0), (self.pad, self.pad)], mode="edge")
        n, c, length = (int(v) for v in x.shape)
        up_np = np.zeros((n, c, length * self.ratio), dtype=np.float32)
        up_np[:, :, :: self.ratio] = np.array(x.astype(mx.float32))
        up = mx.pad(mx.array(up_np), [(0, 0), (0, 0), (int(self.filter.shape[-1]) // 2, int(self.filter.shape[-1]) // 2)], mode="edge")
        y = self.ratio * _depthwise_conv1d(up, self.filter, stride=1)
        end = None if self.pad_right == 0 else -self.pad_right
        return y[..., self.pad_left : end]


class MiniMaxH3AudioDownSample1d(nn.Module):
    def __init__(self, ratio: int, kernel_size: int):
        super().__init__()
        self.lowpass = MiniMaxH3AudioLowPassFilter1d(0.5 / ratio, 0.6 / ratio, ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class MiniMaxH3AudioActivation1d(nn.Module):
    def __init__(self, activation: nn.Module, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = MiniMaxH3AudioUpSample1d(ratio, kernel_size)
        self.downsample = MiniMaxH3AudioDownSample1d(ratio, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.downsample(self.act(self.upsample(x)))


class _WNConv1d(nn.Module):
    """Plain Conv1d; weight-norm is fused at load into ``weight``."""

    def __init__(self, in_c: int, out_c: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def __call__(self, x_ncl: mx.array) -> mx.array:
        x = mx.transpose(x_ncl, (0, 2, 1))
        y = self.conv(x)
        return mx.transpose(y, (0, 2, 1))


class MiniMaxH3AudioResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = MiniMaxH3AudioSnake1d(dim)
        self.conv1 = _WNConv1d(dim, dim, 7, dilation=dilation, padding=pad)
        self.snake2 = MiniMaxH3AudioSnake1d(dim)
        self.conv2 = _WNConv1d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv2(self.snake2(self.conv1(self.snake1(x))))
        pad = (int(x.shape[-1]) - int(y.shape[-1])) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class MiniMaxH3AudioEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.res1 = MiniMaxH3AudioResidualUnit(dim // 2, 1)
        self.res2 = MiniMaxH3AudioResidualUnit(dim // 2, 3)
        self.res3 = MiniMaxH3AudioResidualUnit(dim // 2, 9)
        self.snake = MiniMaxH3AudioSnake1d(dim // 2)
        self.down = _WNConv1d(dim // 2, dim, 2 * stride, stride=stride, padding=math.ceil(stride / 2))

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(self.snake(self.res3(self.res2(self.res1(x)))))


class MiniMaxH3AudioEncoder(nn.Module):
    """DAC encoder; ``block`` list mirrors Diffusers ``nn.Sequential`` indices for keys."""

    def __init__(self, d_model: int, strides: tuple[int, ...], d_latent: int):
        super().__init__()
        block: list[nn.Module] = [_WNConv1d(1, d_model, 7, padding=3)]
        dm = d_model
        for stride in strides:
            dm *= 2
            block.append(MiniMaxH3AudioEncoderBlock(dm, stride))
        block += [MiniMaxH3AudioSnake1d(dm), _WNConv1d(dm, d_latent, 3, padding=1)]
        self.block = block

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.block:
            x = layer(x)
        return x


class MiniMaxH3AudioGeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        # tanh-approximate GELU
        return self.w2(nn.gelu_approx(self.w0(x)) * self.w1(x))


class MiniMaxH3AudioCausalAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = mx.zeros((in_dim,))
        self.v_bias = mx.zeros((in_dim,))
        self.zero_k_bias = mx.zeros((in_dim,))
        self.proj = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        b, s, _ = x.shape
        bias = mx.concatenate([self.q_bias, self.zero_k_bias, self.v_bias], axis=0)
        qkv = x @ mx.transpose(self.qkv.weight, (1, 0)) + bias
        q, k, v = mx.split(qkv.reshape(b, s, 3, self.num_heads, self.head_dim), 3, axis=2)
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        # causal mask
        # [1, 1, S, S] additive causal mask for SDPA
        mask = nn.MultiHeadAttention.create_additive_causal_mask(s)
        mask = mask.astype(q.dtype).reshape(1, 1, s, s)
        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=self.head_dim**-0.5, mask=mask)
        out = mx.mean(out, axis=1)  # mean over heads → [B, S, head_dim]
        # adaptive avg pool head_dim → out_dim
        if int(out.shape[-1]) != self.out_dim:
            out = _adaptive_avg_pool1d_last(out, self.out_dim)
        return self.proj(out)


def _adaptive_avg_pool1d_last(x: mx.array, out_dim: int) -> mx.array:
    """Pool last dim of ``[B, S, D]`` to ``out_dim``."""
    b, s, d = (int(v) for v in x.shape)
    if d == out_dim:
        return x
    # reshape groups
    x_np = np.array(x.astype(mx.float32))
    # simple linear resample average
    edges = np.linspace(0, d, out_dim + 1).astype(np.int32)
    chunks = [x_np[:, :, edges[i] : edges[i + 1]].mean(axis=-1) for i in range(out_dim)]
    return mx.array(np.stack(chunks, axis=-1))


class MiniMaxH3AudioAttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = MiniMaxH3AudioCausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.mlp = MiniMaxH3AudioGeGluMlp(out_dim, out_dim * mlp_ratio)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MiniMaxH3AudioAMPBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: tuple[int, ...]):
        super().__init__()
        self.convs1 = [
            _WNConv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size * d - d) // 2)
            for d in dilation
        ]
        self.convs2 = [
            _WNConv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)
            for _ in dilation
        ]
        self.activations = [
            MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
            for _ in range(2 * len(dilation))
        ]

    def __call__(self, x: mx.array) -> mx.array:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class MiniMaxH3AudioBigVGANDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        upsample_initial_channel: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = _WNConv1d(in_channels, upsample_initial_channel, 7, padding=3)
        self.ups = []
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # nested list to match ups.<i>.0 checkpoint keys
            self.ups.append([
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                    bias=True,
                )
            ])
        self.resblocks = []
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(MiniMaxH3AudioAMPBlock(channels, kernel, tuple(dilation)))
        self.activation_post = MiniMaxH3AudioActivation1d(MiniMaxH3AudioSnakeBeta(channels))
        self.conv_post = _WNConv1d(channels, 1, 7, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            # ConvTranspose1d: N L C
            xt = mx.transpose(x, (0, 2, 1))
            xt = self.ups[i][0](xt)
            x = mx.transpose(xt, (0, 2, 1))
            residual = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j](x)
                residual = block if residual is None else residual + block
            x = residual / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return mx.clip(x, -1.0, 1.0)


class AutoencoderKLMiniMaxH3AudioMLX(nn.Module):
    """Mono 32 kHz audio VAE (stereo = batch of 2)."""

    def __init__(self, **cfg: Any):
        super().__init__()
        encoder_dim = int(cfg.get("encoder_dim", 64))
        encoder_rates = tuple(int(x) for x in _as_tuple(cfg.get("encoder_rates", (2, 4, 4, 5, 5))))
        latent_dim = int(cfg.get("latent_dim", 2048))
        latent_channels = int(cfg.get("latent_channels", 32))
        num_attention_heads = int(cfg.get("num_attention_heads", 8))
        decoder_dim = int(cfg.get("decoder_dim", 1024))
        decoder_rates = tuple(int(x) for x in _as_tuple(cfg.get("decoder_rates", (5, 5, 2, 2, 2, 2, 2))))
        decoder_kernel_sizes = tuple(int(x) for x in _as_tuple(cfg.get("decoder_kernel_sizes", (9, 9, 4, 4, 4, 4, 4))))
        resblock_kernel_sizes = tuple(int(x) for x in _as_tuple(cfg.get("resblock_kernel_sizes", (3, 7, 11))))
        resblock_dilation_sizes = tuple(
            tuple(int(d) for d in _as_tuple(dil))
            for dil in cfg.get("resblock_dilation_sizes", ((1, 3, 5), (1, 3, 5), (1, 3, 5)))
        )
        self.sampling_rate = int(cfg.get("sampling_rate", 32000))
        self.hop_length = int(math.prod(encoder_rates))
        self.latent_channels = latent_channels
        if math.prod(decoder_rates) != self.hop_length:
            raise ValueError(
                f"decoder_rates product must equal hop_length {self.hop_length}, "
                f"got {math.prod(decoder_rates)}"
            )
        mean = cfg.get("latents_mean") or [0.0] * latent_channels
        std = cfg.get("latents_std") or [1.0] * latent_channels
        self.latents_mean = mx.array([float(x) for x in _as_tuple(mean)], dtype=mx.float32).reshape(
            1, latent_channels, 1
        )
        self.latents_std = mx.array([float(x) for x in _as_tuple(std)], dtype=mx.float32).reshape(
            1, latent_channels, 1
        )
        self.encoder = MiniMaxH3AudioEncoder(encoder_dim, encoder_rates, latent_dim)
        self.pre_block = MiniMaxH3AudioAttnProjection(latent_dim, latent_channels, num_attention_heads)
        self.mean_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.logs_proj = nn.Conv1d(latent_channels, latent_channels, 1)
        self.dec_in_proj = nn.Conv1d(latent_channels, latent_dim, 1)
        self.decoder = MiniMaxH3AudioBigVGANDecoder(
            in_channels=latent_dim,
            upsample_initial_channel=decoder_dim,
            upsample_rates=decoder_rates,
            upsample_kernel_sizes=decoder_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AutoencoderKLMiniMaxH3AudioMLX":
        return cls(**cfg)

    def denormalize_latents(self, z: mx.array) -> mx.array:
        return z.astype(mx.float32) * self.latents_std + self.latents_mean

    def normalize_latents(self, z: mx.array) -> mx.array:
        return (z.astype(mx.float32) - self.latents_mean) / self.latents_std

    def encode(self, sample: mx.array, *, normalize: bool = True) -> mx.array:
        if sample.ndim != 3 or int(sample.shape[1]) != 1:
            raise ValueError(f"`sample` must be [B,1,samples], got {tuple(sample.shape)}")
        right_pad = math.ceil(int(sample.shape[-1]) / self.hop_length) * self.hop_length - int(sample.shape[-1])
        if right_pad > 0:
            sample = mx.pad(sample, [(0, 0), (0, 0), (0, right_pad)])
        h = self.encoder(sample)
        h = self.pre_block(mx.transpose(h, (0, 2, 1)))
        h = mx.transpose(h, (0, 2, 1))
        h_nlc = mx.transpose(h, (0, 2, 1))
        mean = mx.transpose(self.mean_proj(h_nlc), (0, 2, 1))
        if normalize:
            mean = self.normalize_latents(mean)
        return mean

    def decode(self, latents: mx.array, *, denormalize: bool = True) -> mx.array:
        if latents.ndim != 3:
            raise ValueError(f"audio latents must be [B,C,T], got {tuple(latents.shape)}")
        if denormalize:
            latents = self.denormalize_latents(latents)
        h = mx.transpose(self.dec_in_proj(mx.transpose(latents, (0, 2, 1))), (0, 2, 1))
        return self.decoder(h)

    def load_weights(self, path: str | Path, *, load_fn: Any | None = None) -> None:
        raw = load_weights_dict(load_fn, str(path))
        fused: dict[str, mx.array] = {}
        skip = set()
        for k, v in raw.items():
            key = k[6:] if k.startswith("model.") else k
            if key.endswith(".weight_g"):
                base = key[: -len(".weight_g")]
                v_key = None
                for cand in (base + ".weight_v", "model." + base + ".weight_v"):
                    if cand in raw:
                        v_key = cand
                        break
                if v_key is None:
                    raise RuntimeError(f"MiniMax-H3 audio VAE missing weight_v for {key}")
                w = _fuse_weight_norm(raw[key] if key in raw else v, raw[v_key])
                # Map WN conv keys: encoder.block.0.weight_g → need structural remap
                # Store under base + ".weight" for modules that use plain Conv — our _WNConv1d uses .conv.weight
                fused[base + ".weight"] = mx.array(w)
                skip.add(key)
                skip.add(v_key[6:] if v_key.startswith("model.") else v_key)
                continue
            if key.endswith(".weight_v"):
                skip.add(key)
                continue
            fused[key] = v if isinstance(v, mx.array) else mx.array(v)
        # Transpose conv weights
        for k, arr in list(fused.items()):
            if k.endswith(".weight") and arr.ndim == 3:
                if "ups." in k or "ConvTranspose" in k:
                    fused[k] = _conv_transpose1d_weight_torch_to_mlx(arr)
                else:
                    fused[k] = _conv1d_weight_torch_to_mlx(arr)
        # Remap weight_norm fused keys onto _WNConv1d.conv.weight where possible
        remapped = _remap_audio_wn_keys(fused)
        params = dict(tree_flatten_params(self))
        assigned = 0
        for k, v in remapped.items():
            if k in params:
                params[k] = v.astype(params[k].dtype)
                assigned += 1
        if assigned == 0:
            raise RuntimeError(
                "MiniMax-H3 audio VAE load_weights assigned 0 keys — checkpoint layout mismatch "
                f"(saw {len(remapped)} remapped keys, model has {len(params)} params)"
            )
        self.update(tree_unflatten_params(params))


def _remap_audio_wn_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Map Diffusers WN / Sequential keys onto ``_WNConv1d.conv.*`` and nested AMP paths."""
    out: dict[str, mx.array] = {}
    for k, v in weights.items():
        nk = k
        # weight_norm fused as ``….weight`` on a WN Conv1d → our ``….conv.weight``
        if nk.endswith(".weight") and ".conv.weight" not in nk and "proj" not in nk.split(".")[-2:]:
            # Heuristic: leaf Conv1d modules under encoder/decoder/resblocks
            if any(p in nk for p in ("encoder.block.", "decoder.conv_", "resblocks.", "convs1.", "convs2.")):
                if not nk.endswith((".alpha", ".beta")):
                    nk = nk[: -len(".weight")] + ".conv.weight"
        if nk.endswith(".bias") and ".conv.bias" not in nk:
            if any(p in nk for p in ("encoder.block.", "decoder.conv_pre", "convs1.", "convs2.")):
                nk = nk[: -len(".bias")] + ".conv.bias"
        out[nk] = v
        if nk != k:
            out[k] = v  # keep original too
    return out


# ---------------------------------------------------------------------------
# Param tree helpers + loaders + mux
# ---------------------------------------------------------------------------


def tree_flatten_params(module: nn.Module) -> list[tuple[str, mx.array]]:
    from mlx.utils import tree_flatten

    return list(tree_flatten(module.parameters()))


def tree_unflatten_params(flat: dict[str, mx.array]) -> Any:
    from mlx.utils import tree_unflatten

    return tree_unflatten(list(flat.items()))


from backend.engine.families.minimax_h3.bundle_paths_mlx import minimax_h3_aux_root


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing config: {path}")
    return json.loads(path.read_text())


def load_video_vae(bundle_root: Path, *, load_fn: Any | None = None) -> AutoencoderKLMiniMaxH3MLX:
    aux = minimax_h3_aux_root(Path(bundle_root))
    cfg_path = aux / "video_vae" / "config.json"
    weight_candidates = [
        aux / "video_vae" / "source" / "model.safetensors",
        aux / "video_vae" / "model.safetensors",
    ]
    weight_path = next((p for p in weight_candidates if p.is_file()), None)
    if weight_path is None:
        raise RuntimeError(f"MiniMax-H3 video VAE weights not found under {aux / 'video_vae'}")
    model = AutoencoderKLMiniMaxH3MLX.from_config(_load_json(cfg_path) if cfg_path.is_file() else {})
    model.load_weights(weight_path, load_fn=load_fn)
    return model


def load_audio_vae(bundle_root: Path, *, load_fn: Any | None = None) -> AutoencoderKLMiniMaxH3AudioMLX:
    aux = minimax_h3_aux_root(Path(bundle_root))
    cfg_path = aux / "audio_vae" / "config.json"
    weight_candidates = [
        aux / "audio_vae" / "model.safetensors",
    ]
    weight_path = next((p for p in weight_candidates if p.is_file()), None)
    if weight_path is None:
        raise RuntimeError(f"MiniMax-H3 audio VAE weights not found under {aux / 'audio_vae'}")
    model = AutoencoderKLMiniMaxH3AudioMLX.from_config(_load_json(cfg_path) if cfg_path.is_file() else {})
    model.load_weights(weight_path, load_fn=load_fn)
    return model


def decode_video_latents_ncthw(
    ctx: Any,
    latents_bcthw: mx.array,
    bundle_root: Path,
    *,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> mx.array:
    if on_log:
        on_log(f"MiniMax-H3 video VAE decode start {tuple(latents_bcthw.shape)}")
    if on_stage:
        on_stage(0.05)
    vae = load_video_vae(bundle_root, load_fn=getattr(ctx, "load_weights", None))
    sample = vae.decode(latents_bcthw, denormalize=True)
    _eval(sample)
    if on_stage:
        on_stage(1.0)
    if on_log:
        on_log(f"MiniMax-H3 video VAE decode done {tuple(sample.shape)}")
    return sample


def decode_audio_latents(
    ctx: Any,
    audio_latent: mx.array,
    bundle_root: Path,
) -> mx.array:
    """Decode ``[2, 32, T]`` (stereo as batch) → ``[2, 1, samples]`` waveform."""
    vae = load_audio_vae(bundle_root, load_fn=getattr(ctx, "load_weights", None))
    wav = vae.decode(audio_latent, denormalize=True)
    _eval(wav)
    return wav


def _imagenet_to_uint8_frames(sample_ncthw: mx.array) -> np.ndarray:
    """ImageNet-normalized NCTHW → uint8 NHWC frames ``[T, H, W, 3]``."""
    x = sample_ncthw.astype(mx.float32) * _PIXEL_STD + _PIXEL_MEAN
    x = mx.clip(x, 0.0, 1.0)
    x = (x[0].transpose(1, 2, 3, 0) * 255.0).astype(mx.uint8)  # T H W C
    _eval(x)
    return np.array(x)


def _save_stereo_wav(waveform_2_1_t: mx.array, path: str, sample_rate: int = 32000) -> None:
    """``[2, 1, samples]`` → interleaved stereo wav."""
    w = np.array(waveform_2_1_t.astype(mx.float32))
    if w.ndim != 3 or w.shape[0] != 2 or w.shape[1] != 1:
        raise ValueError(f"expected stereo audio [2,1,T], got {w.shape}")
    stereo = np.stack([w[0, 0], w[1, 0]], axis=-1)  # T, 2
    arr = np.clip(stereo, -1.0, 1.0)
    with wave.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((arr * 32767.0).astype(np.int16).tobytes())


def mux_video_audio_mp4(
    ctx: Any,
    video_latent: mx.array,
    audio_latent: mx.array,
    output_path: str,
    bundle_root: Path,
    *,
    frame_rate: float = 24.0,
    video_vae: Any | None = None,
    audio_vae: Any | None = None,
    stream_frames: bool = True,
    upscale_to: tuple[int, int] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> str:
    """Decode video+audio latents and mux with ffmpeg (stereo wav + raw RGB frames)."""
    ffmpeg = require_ffmpeg()
    load_fn = getattr(ctx, "load_weights", None)
    if video_vae is None:
        video_vae = load_video_vae(bundle_root, load_fn=load_fn)
    if audio_vae is None:
        audio_vae = load_audio_vae(bundle_root, load_fn=load_fn)
    pixels = video_vae.decode(video_latent, denormalize=True)
    wav = audio_vae.decode(audio_latent, denormalize=True)
    _eval(pixels)
    _eval(wav)
    frames = _imagenet_to_uint8_frames(pixels)
    if upscale_to is not None:
        target_h, target_w = upscale_to
        from PIL import Image

        upscaled = np.empty((frames.shape[0], target_h, target_w, 3), dtype=np.uint8)
        for i in range(frames.shape[0]):
            upscaled[i] = np.array(
                Image.fromarray(frames[i]).resize((target_w, target_h), Image.Resampling.LANCZOS)
            )
        frames = upscaled
    t, h, w, _c = frames.shape
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    _save_stereo_wav(wav, audio_path, sample_rate=audio_vae.sampling_rate)
    if on_log:
        on_log(f"MiniMax-H3 mux: {t} frames {w}x{h} @ {frame_rate}fps → {output_path}")
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(frame_rate),
        "-i", "-",
        "-i", audio_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-shortest",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        if stream_frames:
            for i in range(t):
                proc.stdin.write(frames[i].tobytes())
        else:
            proc.stdin.write(frames.tobytes())
        proc.stdin.close()
    except BrokenPipeError as exc:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg pipe failed during MiniMax-H3 mux: {err}") from exc
    rc = proc.wait()
    Path(audio_path).unlink(missing_ok=True)
    if rc != 0:
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"ffmpeg mux failed (rc={rc}): {err}")
    return output_path
