"""Wan 2.2 3D causal VAE — MLX port of official ``Wan2_2_VAE`` / ``WanVAE_``.

PyTorch checkpoint tensors use ``[out_c, in_c, kt, kh, kw]`` for Conv3d / ``[out_c, in_c, kh, kw]`` for Conv2d.
MLX Conv3d / Conv2d expect ``[out_c, kt, kh, kw, in_c]`` / ``[out_c, kh, kw, in_c]`` with activations in **NDHWC** / **NHWC**.
Internal activations use **NCTHW** between layers; Conv3d paths transpose as needed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime.mlx_runtime import load_weights_dict
from backend.engine.runtime._base import RuntimeContext

logger = logging.getLogger(__name__)

CACHE_T = 2


def _vae_feat_snapshot(x: mx.array) -> mx.array:
    """Detached copy for causal ``feat_cache`` entries (official ``Tensor.clone()``)."""
    return mx.stop_gradient(x + 0.0)


# Official Wan2.1 VAE latent mean / std (16-dim, ``Wan-Video/Wan2.1``).
_WAN21_VAE_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
_WAN21_VAE_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

# Official Wan2.2 VAE latent mean / std (48-dim).
_WAN22_VAE_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
_WAN22_VAE_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


def _conv3d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    if w.ndim != 5:
        return w
    out, d1, d2, d3, d4 = (int(x) for x in w.shape)
    # PyTorch (O, I, kT, kH, kW): channel dim at index 1, kernel at 2–4.
    if d2 in (1, 3, 5, 7) and d3 in (1, 3, 5, 7) and d4 in (1, 3, 5, 7) and d1 > d4:
        return mx.transpose(w, (0, 2, 3, 4, 1))
    # Already MLX / mlx-community layout (O, kT, kH, kW, I).
    return w


def _conv2d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    if w.ndim != 4:
        return w
    out, d1, d2, d3 = (int(x) for x in w.shape)
    if d2 in (1, 3, 5, 7) and d3 in (1, 3, 5, 7) and d1 > d3:
        return mx.transpose(w, (0, 2, 3, 1))
    return w


def _ncthw_to_ndhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 4, 1))


def _ndhwc_to_ncthw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 4, 1, 2, 3))


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def _pad_ncthw(x: mx.array, *, pad_t: int, pad_h: int, pad_w: int) -> mx.array:
    return mx.pad(
        x,
        [
            (0, 0),
            (0, 0),
            (pad_t, 0),
            (pad_h, pad_h),
            (pad_w, pad_w),
        ],
    )


def _flatten_bt_ncthw(x: mx.array) -> mx.array:
    """``rearrange(x, 'b c t h w -> (b t) c h w')`` — not equivalent to naive ``reshape``."""
    b, c, t, h, w = x.shape
    x = mx.transpose(x, (0, 2, 1, 3, 4))
    return mx.reshape(x, (b * t, c, h, w))


def _unflatten_bt_ncthw(x_bt: mx.array, *, b: int, t: int) -> mx.array:
    """``rearrange(x, '(b t) c h w -> b c t h w', t=t)``."""
    _, c, h, w = x_bt.shape
    x = mx.reshape(x_bt, (b, t, c, h, w))
    return mx.transpose(x, (0, 2, 1, 3, 4))


def _l2_normalize(x: mx.array, axis: int) -> mx.array:
    denom = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True) + 1e-12)
    return x / denom


def _upsample_nearest_2d_nchw(x: mx.array) -> mx.array:
    return mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)


def _zero_pad2d_nchw(x: mx.array, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int) -> mx.array:
    if pad_left <= 0 and pad_right <= 0 and pad_top <= 0 and pad_bottom <= 0:
        return x
    return mx.pad(
        x,
        [
            (0, 0),
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        ],
    )


class CausalConv3d(nn.Module):
    """Official Wan causal 3D convolution (NCTHW activations)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        pt, ph, pw = padding
        self._pad_t = 2 * pt
        self._pad_h = ph
        self._pad_w = pw
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(self, x_ncthw: mx.array, cache_x: mx.array | None = None) -> mx.array:
        pad_t = self._pad_t
        x = x_ncthw
        if cache_x is not None and pad_t > 0:
            x = mx.concatenate([cache_x, x], axis=2)
            pad_t -= int(cache_x.shape[2])
        x = _pad_ncthw(x, pad_t=pad_t, pad_h=self._pad_h, pad_w=self._pad_w)
        out = self.conv(_ncthw_to_ndhwc(x))
        return _ndhwc_to_ncthw(out)


class WanVAERMSNorm(nn.Module):
    def __init__(self, dim: int, *, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        del channel_first
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims)
        self.scale = dim**0.5
        self.gamma = mx.ones(shape)
        self.bias = mx.zeros(shape) if bias else None
        self._images = images

    def __call__(self, x: mx.array) -> mx.array:
        axis = 1 if x.ndim == 5 or (x.ndim == 4 and self._images) else -1
        out = _l2_normalize(x, axis=axis) * self.scale * self.gamma
        if self.bias is not None:
            out = out + self.bias
        return out


class Resample(nn.Module):
    def __init__(self, dim: int, mode: str, *, layout: str = "wan22"):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        assert layout in ("wan21", "wan22")
        self.dim = dim
        self.mode = mode
        self.layout = layout
        if mode == "upsample2d":
            self.resample = ("upsample2d", dim)
        elif mode == "upsample3d":
            self.resample = ("upsample3d", dim)
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = ("downsample2d", dim)
        elif mode == "downsample3d":
            self.resample = ("downsample3d", dim)
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = ("none", dim)

        if mode in ("upsample2d", "upsample3d"):
            out_ch = dim // 2 if layout == "wan21" else dim
            self.resample_conv = nn.Conv2d(dim, out_ch, 3, padding=1)
        elif mode in ("downsample2d", "downsample3d"):
            self.resample_conv = nn.Conv2d(dim, dim, 3, stride=(2, 2))

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> mx.array:
        feat_idx = feat_idx if feat_idx is not None else [0]
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = mx.concatenate(
                            [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                            axis=2,
                        )
                    if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = mx.concatenate([mx.zeros_like(cache_x), cache_x], axis=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = _vae_feat_snapshot(cache_x)
                    feat_idx[0] += 1
                    x = mx.reshape(x, (b, 2, c, t, h, w))
                    x = mx.stack((x[:, 0], x[:, 1]), axis=3)
                    x = mx.reshape(x, (b, c, t * 2, h, w))
            elif first_chunk and t > 1:
                first_frame = x[:, :, :1, :, :]
                rest = x[:, :, 1:, :, :]
                tc_out = self.time_conv(rest)
                tc_out = mx.reshape(tc_out, (b, 2, c, t - 1, h, w))
                stream0 = tc_out[:, 0]
                stream1 = tc_out[:, 1]
                interleaved = mx.stack((stream0, stream1), axis=3)
                interleaved = mx.reshape(interleaved, (b, c, (t - 1) * 2, h, w))
                x = mx.concatenate([first_frame, interleaved], axis=2)
            else:
                tc_out = self.time_conv(x)
                tc_out = mx.reshape(tc_out, (b, 2, c, t, h, w))
                stream0 = tc_out[:, 0]
                stream1 = tc_out[:, 1]
                x = mx.stack((stream0, stream1), axis=3)
                x = mx.reshape(x, (b, c, t * 2, h, w))
        t = int(x.shape[2])
        x_bt = _flatten_bt_ncthw(x)
        mode_tag = self.resample[0]
        if mode_tag in ("upsample2d", "upsample3d"):
            x_bt = _upsample_nearest_2d_nchw(x_bt)
        elif mode_tag in ("downsample2d", "downsample3d"):
            x_bt = _zero_pad2d_nchw(x_bt, 0, 1, 0, 1)
        x_bt = _nhwc_to_nchw(self.resample_conv(_nchw_to_nhwc(x_bt)))
        h2, w2 = int(x_bt.shape[2]), int(x_bt.shape[3])
        x = _unflatten_bt_ncthw(x_bt, b=b, t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = _vae_feat_snapshot(x)
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(mx.concatenate([feat_cache[idx][:, :, -1:, :, :], x], axis=2))
                    feat_cache[idx] = _vae_feat_snapshot(cache_x)
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        del dropout
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None
        self.norm1 = WanVAERMSNorm(in_dim, images=False)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanVAERMSNorm(out_dim, images=False)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        feat_idx = feat_idx if feat_idx is not None else [0]
        h = x if self.shortcut is None else self.shortcut(x)
        for layer in (self.norm1, "silu", self.conv1, self.norm2, "silu", self.conv2):
            if layer == "silu":
                x = nn.silu(x)
            elif isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                    cache_x = mx.concatenate(
                        [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                        axis=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = _vae_feat_snapshot(cache_x)
                feat_idx[0] += 1
            elif isinstance(layer, (WanVAERMSNorm, CausalConv3d)):
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = WanVAERMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        b, c, t, h, w = x.shape
        x_bt = _flatten_bt_ncthw(x)
        x_bt = self.norm(x_bt)
        qkv = self.to_qkv(_nchw_to_nhwc(x_bt))
        qkv = mx.reshape(qkv, (b * t, h * w, 3, c))
        qkv = mx.transpose(qkv, (0, 2, 1, 3))
        q, k, v = mx.split(qkv, 3, axis=1)
        q = mx.reshape(q, (b * t, 1, h * w, c))
        k = mx.reshape(k, (b * t, 1, h * w, c))
        v = mx.reshape(v, (b * t, 1, h * w, c))
        attn = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=(c ** -0.5))
        # Match official: squeeze head, ``(B, HW, C) -> (B, C, H, W)``.
        attn = mx.squeeze(attn, axis=1)
        attn = mx.transpose(attn, (0, 2, 1))
        attn = mx.reshape(attn, (b * t, c, h, w))
        out = self.proj(_nchw_to_nhwc(attn))
        out = _unflatten_bt_ncthw(_nhwc_to_nchw(out), b=b, t=t)
        return out + identity


def patchify(x: mx.array, patch_size: int) -> mx.array:
    if patch_size == 1:
        return x
    if x.ndim == 4:
        b, c, h, w = x.shape
        q = patch_size
        x = mx.reshape(x, (b, c, h // q, q, w // q, q))
        x = mx.transpose(x, (0, 1, 3, 5, 2, 4))
        return mx.reshape(x, (b, c * q * q, h // q, w // q))
    if x.ndim == 5:
        b, c, f, h, w = x.shape
        q = patch_size
        x = mx.reshape(x, (b, c, f, h // q, q, w // q, q))
        # Match diffusers ``AutoencoderKLWan.patchify`` (Wan2.2-VAE training layout).
        x = mx.transpose(x, (0, 1, 6, 4, 2, 3, 5))
        return mx.reshape(x, (b, c * q * q, f, h // q, w // q))
    raise RuntimeError(f"Invalid input shape for patchify: {x.shape}")


def unpatchify(x: mx.array, patch_size: int) -> mx.array:
    if patch_size == 1:
        return x
    if x.ndim == 4:
        b, c, h, w = x.shape
        q = patch_size
        c0 = c // (q * q)
        x = mx.reshape(x, (b, c0, q, q, h, w))
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
        return mx.reshape(x, (b, c0, h * q, w * q))
    if x.ndim == 5:
        b, c, f, h, w = x.shape
        q = patch_size
        c0 = c // (q * q)
        x = mx.reshape(x, (b, c0, q, q, f, h, w))
        # Match diffusers ``AutoencoderKLWan.unpatchify``.
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))
        return mx.reshape(x, (b, c0, f, h * q, w * q))
    raise RuntimeError(f"Invalid input shape for unpatchify: {x.shape}")


class AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if (in_channels * self.factor) % out_channels != 0:
            raise RuntimeError("AvgDown3D channel/group mismatch")
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        pad_t = (self.factor_t - int(x.shape[2]) % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = mx.pad(x, [(0, 0), (0, 0), (pad_t, 0), (0, 0), (0, 0)])
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = mx.reshape(x, (b, c, t // ft, ft, h // fs, fs, w // fs, fs))
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = mx.reshape(x, (b, c * self.factor, t // ft, h // fs, w // fs))
        x = mx.reshape(x, (b, self.out_channels, self.group_size, t // ft, h // fs, w // fs))
        return mx.mean(x, axis=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if (out_channels * self.factor) % in_channels != 0:
            raise RuntimeError("DupUp3D channel/repeat mismatch")
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        x = _repeat_interleave_channels(x, self.repeats)
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = mx.reshape(x, (b, self.out_channels, ft, fs, fs, t, h, w))
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = mx.reshape(x, (b, self.out_channels, t * ft, h * fs, w * fs))
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


def _repeat_interleave_channels(x: mx.array, repeats: int) -> mx.array:
    if repeats <= 1:
        return x
    b, c, t, h, w = x.shape
    x = mx.reshape(x, (b, c, 1, t, h, w))
    x = mx.repeat(x, repeats, axis=2)
    return mx.reshape(x, (b, c * repeats, t, h, w))


class DownResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        *,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        blocks: list[nn.Module] = []
        cur = in_dim
        for _ in range(mult):
            blocks.append(ResidualBlock(cur, out_dim, dropout))
            cur = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            blocks.append(Resample(out_dim, mode))
        self.blocks = blocks

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        x_copy = x
        for module in self.blocks:
            if isinstance(module, (ResidualBlock, Resample)):
                x = module(x, feat_cache, feat_idx)
            else:
                x = module(x)
        return x + self.avg_shortcut(x_copy)


class UpResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        *,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ):
        super().__init__()
        self.avg_shortcut = (
            DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2 if up_flag else 1)
            if up_flag
            else None
        )
        blocks: list[nn.Module] = []
        cur = in_dim
        for _ in range(mult):
            blocks.append(ResidualBlock(cur, out_dim, dropout))
            cur = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            blocks.append(Resample(out_dim, mode))
        self.blocks = blocks

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> mx.array:
        x_main = _vae_feat_snapshot(x)
        for module in self.blocks:
            if isinstance(module, Resample):
                x_main = module(x_main, feat_cache, feat_idx, first_chunk)
            elif isinstance(module, ResidualBlock):
                x_main = module(x_main, feat_cache, feat_idx)
            else:
                x_main = module(x_main)
        if self.avg_shortcut is not None:
            return x_main + self.avg_shortcut(x, first_chunk)
        return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        in_channels: int = 12,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [True, True, False]
        dims = [dim * u for u in [1] + dim_mult]
        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)
        downsamples: list[DownResidualBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                DownResidualBlock(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks,
                    temperal_downsample=t_down,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.downsamples = downsamples
        self.mid_res0 = ResidualBlock(out_dim, out_dim, dropout)
        self.mid_attn = AttentionBlock(out_dim)
        self.mid_res1 = ResidualBlock(out_dim, out_dim, dropout)
        self.head_norm = WanVAERMSNorm(out_dim, images=False)
        self.head_conv = CausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        feat_idx = feat_idx if feat_idx is not None else [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)

        for layer in (self.mid_res0, self.mid_attn, self.mid_res1):
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        x = self.head_norm(x)
        x = nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.head_conv(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.head_conv(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        pixel_out_channels: int = 12,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_upsample = temperal_upsample or [False, True, True]
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_res0 = ResidualBlock(dims[0], dims[0], dropout)
        self.mid_attn = AttentionBlock(dims[0])
        self.mid_res1 = ResidualBlock(dims[0], dims[0], dropout)
        upsamples: list[UpResidualBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                UpResidualBlock(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks + 1,
                    temperal_upsample=t_up,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = upsamples
        self.head_norm = WanVAERMSNorm(out_dim, images=False)
        self.head_conv = CausalConv3d(out_dim, pixel_out_channels, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> mx.array:
        feat_idx = feat_idx if feat_idx is not None else [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in (self.mid_res0, self.mid_attn, self.mid_res1):
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if isinstance(layer, UpResidualBlock):
                x = layer(x, feat_cache, feat_idx, first_chunk)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
            else:
                x = layer(x)

        x = self.head_norm(x)
        x = nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.head_conv(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.head_conv(x)
        return x


class Encoder3dWan21(nn.Module):
    """Flat ``ResidualBlock`` + ``Resample`` stack (official Wan2.1 ``Encoder3d``)."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        in_channels: int = 3,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [False, True, True]
        dims = [dim * u for u in [1] + dim_mult]
        self.conv1 = CausalConv3d(in_channels, dims[0], 3, padding=1)
        downsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            cur = in_dim
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(cur, out_dim, dropout))
                cur = out_dim
            if i != len(dim_mult) - 1:
                t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
                mode = "downsample3d" if t_down else "downsample2d"
                downsamples.append(Resample(out_dim, mode, layout="wan21"))
        self.downsamples = downsamples
        self.mid_res0 = ResidualBlock(out_dim, out_dim, dropout)
        self.mid_attn = AttentionBlock(out_dim)
        self.mid_res1 = ResidualBlock(out_dim, out_dim, dropout)
        self.head_norm = WanVAERMSNorm(out_dim, images=False)
        self.head_conv = CausalConv3d(out_dim, z_dim * 2, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> mx.array:
        feat_idx = feat_idx if feat_idx is not None else [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if isinstance(layer, (ResidualBlock, Resample)):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in (self.mid_res0, self.mid_attn, self.mid_res1):
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        x = self.head_norm(x)
        x = nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.head_conv(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.head_conv(x)
        return x


class Decoder3dWan21(nn.Module):
    """Flat upsample stack (official Wan2.1 ``Decoder3d``)."""

    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        pixel_out_channels: int = 3,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_upsample = temperal_upsample or [False, True, True]
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_res0 = ResidualBlock(dims[0], dims[0], dropout)
        self.mid_attn = AttentionBlock(dims[0])
        self.mid_res1 = ResidualBlock(dims[0], dims[0], dropout)
        upsamples: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            cur = in_dim // 2 if i in (1, 2, 3) else in_dim
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(cur, out_dim, dropout))
                cur = out_dim
            if i != len(dim_mult) - 1:
                t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
                mode = "upsample3d" if t_up else "upsample2d"
                upsamples.append(Resample(out_dim, mode, layout="wan21"))
        self.upsamples = upsamples
        self.head_norm = WanVAERMSNorm(out_dim, images=False)
        self.head_conv = CausalConv3d(out_dim, pixel_out_channels, 3, padding=1)

    def __call__(
        self,
        x: mx.array,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> mx.array:
        del first_chunk
        feat_idx = feat_idx if feat_idx is not None else [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in (self.mid_res0, self.mid_attn, self.mid_res1):
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if isinstance(layer, Resample):
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x, feat_cache, feat_idx)

        x = self.head_norm(x)
        x = nn.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = mx.concatenate(
                    [mx.expand_dims(feat_cache[idx][:, :, -1, :, :], 2), cache_x],
                    axis=2,
                )
            x = self.head_conv(x, feat_cache[idx])
            feat_cache[idx] = _vae_feat_snapshot(cache_x)
            feat_idx[0] += 1
        else:
            x = self.head_conv(x)
        return x


def _count_conv3d(model: nn.Module) -> int:
    count = 0
    for mod in model.modules():
        if isinstance(mod, CausalConv3d):
            count += 1
    return count


class WanVAE(nn.Module):
    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        encoder_in_channels: int = 12,
        pixel_out_channels: int = 12,
        layout: str = "wan22",
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [False, True, True]
        self.z_dim = z_dim
        self.layout = layout
        self.temperal_upsample = temperal_downsample[::-1]
        if layout == "wan21":
            self.patch_size = 1
            self.encoder = Encoder3dWan21(
                dim,
                z_dim,
                dim_mult,
                num_res_blocks,
                temperal_downsample,
                dropout,
                in_channels=encoder_in_channels,
            )
            self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
            self.decoder = Decoder3dWan21(
                dim,
                z_dim,
                dim_mult,
                num_res_blocks,
                self.temperal_upsample,
                dropout,
                pixel_out_channels=pixel_out_channels,
            )
        else:
            self.patch_size = 2 if int(encoder_in_channels) == 12 else 1
            self.encoder = Encoder3d(
                dim,
                z_dim * 2,
                dim_mult,
                num_res_blocks,
                temperal_downsample,
                dropout,
                in_channels=encoder_in_channels,
            )
            self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
            self.conv2 = CausalConv3d(z_dim, z_dim, 1)
            self.decoder = Decoder3d(
                dec_dim,
                z_dim,
                dim_mult,
                num_res_blocks,
                self.temperal_upsample,
                dropout,
                pixel_out_channels=pixel_out_channels,
            )
        self._conv_num = 0
        self._conv_idx = [0]
        self._feat_map: list[Any] = []
        self._enc_conv_num = 0
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[Any] = []

    def clear_cache(self) -> None:
        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def encode(self, x: mx.array, scale: tuple[mx.array, mx.array]) -> mx.array:
        self.clear_cache()
        if self.patch_size > 1:
            x = patchify(x, self.patch_size)
        t = int(x.shape[2])
        iter_ = 1 + (t - 1) // 4
        out_parts: list[mx.array] = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                chunk = x[:, :, :1, :, :]
            else:
                chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            part = self.encoder(chunk, self._enc_feat_map, self._enc_conv_idx)
            out_parts.append(part)
        out = out_parts[0] if len(out_parts) == 1 else mx.concatenate(out_parts, axis=2)
        mu, _log_var = mx.split(self.conv1(out), 2, axis=1)
        mean, inv_std = scale
        mu = (mu - mean) * inv_std
        self.clear_cache()
        return mu

    def decode(
        self,
        z: mx.array,
        scale: tuple[mx.array, mx.array],
        on_stage: Callable[[float], None] | None = None,
    ) -> mx.array:
        """Decode latent volume with per-frame causal cache (official ``WanVAE_.decode``)."""
        self.clear_cache()
        mean, inv_std = scale
        z = z / inv_std + mean
        if on_stage is not None:
            on_stage(0.05)
        x = self.conv2(z)
        iter_t = int(z.shape[2])
        out: mx.array | None = None
        for i in range(iter_t):
            self._conv_idx = [0]
            chunk = x[:, :, i : i + 1, :, :]
            if i == 0:
                part = self.decoder(
                    chunk, self._feat_map, self._conv_idx, first_chunk=True,
                )
            else:
                part = self.decoder(
                    chunk, self._feat_map, self._conv_idx, first_chunk=False,
                )
            out = part if out is None else mx.concatenate([out, part], axis=2)
            if on_stage is not None and iter_t > 0:
                on_stage(0.05 + 0.9 * float(i + 1) / float(iter_t))
        if out is None:
            raise RuntimeError("Wan VAE decode received empty latent volume")
        if self.patch_size > 1:
            out = unpatchify(out, self.patch_size)
        if on_stage is not None:
            on_stage(1.0)
        self.clear_cache()
        return out


@dataclass(frozen=True)
class _WanVaeTileParams:
    tile_sample_min_height: int
    tile_sample_min_width: int
    tile_latent_min_height: int
    tile_latent_min_width: int
    overlap_factor: float


def _wan_vae_tile_params(vae_cfg: dict[str, Any], spatial_scale: int) -> _WanVaeTileParams:
    spatial_up = max(1, int(spatial_scale))
    sample_h = int(vae_cfg.get("tile_sample_min_height", 256))
    sample_w = int(vae_cfg.get("tile_sample_min_width", 256))
    overlap = float(vae_cfg.get("tile_overlap_factor", 0.25))
    return _WanVaeTileParams(
        tile_sample_min_height=sample_h,
        tile_sample_min_width=sample_w,
        tile_latent_min_height=max(1, sample_h // spatial_up),
        tile_latent_min_width=max(1, sample_w // spatial_up),
        overlap_factor=overlap,
    )


def _needs_wan_spatial_tiling(
    latent_h: int,
    latent_w: int,
    params: _WanVaeTileParams,
    *,
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    return (
        latent_h > params.tile_latent_min_height
        or latent_w > params.tile_latent_min_width
    )


def _blend_v_wan(a_ncthw: mx.array, b_ncthw: mx.array, blend_extent: int) -> mx.array:
    be = min(int(a_ncthw.shape[3]), int(b_ncthw.shape[3]), int(blend_extent))
    if be <= 0:
        return b_ncthw
    rows: list[mx.array] = []
    for y in range(int(b_ncthw.shape[3])):
        if y < be:
            alpha = float(y) / float(be)
            merged = (
                a_ncthw[:, :, :, -be + y, :] * (1.0 - alpha)
                + b_ncthw[:, :, :, y, :] * alpha
            )
            rows.append(mx.expand_dims(merged, axis=3))
        else:
            rows.append(mx.expand_dims(b_ncthw[:, :, :, y, :], axis=3))
    return mx.concatenate(rows, axis=3)


def _blend_h_wan(a_ncthw: mx.array, b_ncthw: mx.array, blend_extent: int) -> mx.array:
    be = min(int(a_ncthw.shape[4]), int(b_ncthw.shape[4]), int(blend_extent))
    if be <= 0:
        return b_ncthw
    cols: list[mx.array] = []
    for x in range(int(b_ncthw.shape[4])):
        if x < be:
            alpha = float(x) / float(be)
            merged = (
                a_ncthw[:, :, :, :, -be + x] * (1.0 - alpha)
                + b_ncthw[:, :, :, :, x] * alpha
            )
            cols.append(mx.expand_dims(merged, axis=4))
        else:
            cols.append(mx.expand_dims(b_ncthw[:, :, :, :, x], axis=4))
    return mx.concatenate(cols, axis=4)


def _decode_wan_latent_volume(
    model: WanVAE,
    latents: mx.array,
    scale: tuple[mx.array, mx.array],
    on_stage: Callable[[float], None] | None = None,
) -> mx.array:
    """Full temporal decode for one latent volume ``[B,C,T,H,W]`` (fresh causal cache)."""
    model.clear_cache()
    return model.decode(latents, scale, on_stage=on_stage)


def _tiled_spatial_decode_wan(
    ctx: RuntimeContext,
    model: WanVAE,
    latents: mx.array,
    scale: tuple[mx.array, mx.array],
    params: _WanVaeTileParams,
    *,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> mx.array:
    """Spatial tiling over latent H/W; each tile runs full causal temporal decode."""
    _, _, _, height, width = latents.shape
    overlap_height = max(1, int(params.tile_latent_min_height * (1.0 - params.overlap_factor)))
    overlap_width = max(1, int(params.tile_latent_min_width * (1.0 - params.overlap_factor)))
    blend_height = int(params.tile_sample_min_height * params.overlap_factor)
    blend_width = int(params.tile_sample_min_width * params.overlap_factor)
    row_limit_height = params.tile_sample_min_height - blend_height
    row_limit_width = params.tile_sample_min_width - blend_width

    row_starts = list(range(0, int(height), overlap_height))
    col_starts = list(range(0, int(width), overlap_width))
    total_tiles = len(row_starts) * len(col_starts)
    tile_idx = 0

    rows: list[list[mx.array]] = []
    for i in row_starts:
        row: list[mx.array] = []
        for j in col_starts:
            tile_idx += 1
            msg = (
                f"Wan VAE spatial tile {tile_idx}/{total_tiles} "
                f"(latent y={i}, x={j})"
            )
            logger.info(msg)
            if on_log is not None:
                on_log(msg)
            if on_stage is not None:
                on_stage(0.05 + 0.9 * ((tile_idx - 1) / max(total_tiles, 1)))

            z_tile = latents[
                :,
                :,
                :,
                i : i + params.tile_latent_min_height,
                j : j + params.tile_latent_min_width,
            ]

            def _tile_stage(frac: float) -> None:
                if on_stage is None:
                    return
                base = 0.05 + 0.9 * ((tile_idx - 1) / max(total_tiles, 1))
                span = 0.9 / max(total_tiles, 1)
                on_stage(min(1.0, base + span * float(frac)))

            decoded = _decode_wan_latent_volume(
                model, z_tile, scale, on_stage=_tile_stage,
            )
            ctx.eval(decoded)
            row.append(decoded)
            if hasattr(ctx, "clear_cache"):
                ctx.clear_cache()
        rows.append(row)

    result_rows: list[mx.array] = []
    for ri, row in enumerate(rows):
        result_row: list[mx.array] = []
        for ci, tile in enumerate(row):
            blended = tile
            if ri > 0:
                blended = _blend_v_wan(rows[ri - 1][ci], blended, blend_height)
            if ci > 0:
                blended = _blend_h_wan(row[ci - 1], blended, blend_width)
            out_h = int(blended.shape[3])
            out_w = int(blended.shape[4])
            rh = min(row_limit_height, out_h)
            rw = min(row_limit_width, out_w)
            result_row.append(blended[:, :, :, :rh, :rw])
        result_rows.append(mx.concatenate(result_row, axis=4))
    sample = mx.concatenate(result_rows, axis=3)
    ctx.eval(sample)
    return sample


@dataclass
class Wan22VAE:
    model: WanVAE
    scale_mean: mx.array
    scale_inv_std: mx.array
    z_dim: int = 48


def _set_causal_conv3d(mod: CausalConv3d, prefix: str, weights: dict[str, mx.array]) -> None:
    w = weights[f"{prefix}.weight"]
    b = weights[f"{prefix}.bias"]
    mod.conv.weight = _conv3d_weight_torch_to_mlx(w)
    mod.conv.bias = b


def _set_conv2d(mod: nn.Conv2d, prefix: str, weights: dict[str, mx.array]) -> None:
    mod.weight = _conv2d_weight_torch_to_mlx(weights[f"{prefix}.weight"])
    mod.bias = weights[f"{prefix}.bias"]


def _set_rms_norm(mod: WanVAERMSNorm, prefix: str, weights: dict[str, mx.array]) -> None:
    mod.gamma = weights[f"{prefix}.gamma"]


def _set_residual_block(mod: ResidualBlock, prefix: str, weights: dict[str, mx.array]) -> None:
    _set_rms_norm(mod.norm1, f"{prefix}.residual.0", weights)
    _set_causal_conv3d(mod.conv1, f"{prefix}.residual.2", weights)
    _set_rms_norm(mod.norm2, f"{prefix}.residual.3", weights)
    _set_causal_conv3d(mod.conv2, f"{prefix}.residual.6", weights)
    if mod.shortcut is not None:
        if f"{prefix}.shortcut.weight" in weights:
            _set_causal_conv3d(mod.shortcut, f"{prefix}.shortcut", weights)
        else:
            mod.shortcut = None


def _set_attention_block(mod: AttentionBlock, prefix: str, weights: dict[str, mx.array]) -> None:
    _set_rms_norm(mod.norm, f"{prefix}.norm", weights)
    _set_conv2d(mod.to_qkv, f"{prefix}.to_qkv", weights)
    _set_conv2d(mod.proj, f"{prefix}.proj", weights)


def _set_resample(mod: Resample, prefix: str, weights: dict[str, mx.array]) -> None:
    if mod.mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
        _set_conv2d(mod.resample_conv, f"{prefix}.resample.1", weights)
    if mod.mode in ("upsample3d", "downsample3d"):
        _set_causal_conv3d(mod.time_conv, f"{prefix}.time_conv", weights)


def _set_down_block(
    mod: DownResidualBlock, prefix: str, start_idx: int, weights: dict[str, mx.array]
) -> int:
    idx = start_idx
    for block in mod.blocks:
        if isinstance(block, ResidualBlock):
            _set_residual_block(block, f"{prefix}.{idx}", weights)
            idx += 1
        elif isinstance(block, Resample):
            _set_resample(block, f"{prefix}.{idx}", weights)
            idx += 1
    return idx


def _set_up_block(
    mod: UpResidualBlock, prefix: str, start_idx: int, weights: dict[str, mx.array]
) -> int:
    idx = start_idx
    for block in mod.blocks:
        if isinstance(block, ResidualBlock):
            _set_residual_block(block, f"{prefix}.{idx}", weights)
            idx += 1
        elif isinstance(block, Resample):
            _set_resample(block, f"{prefix}.{idx}", weights)
            idx += 1
    return idx


def _assign_wan21_vae_weights(model: WanVAE, weights: dict[str, mx.array]) -> None:
    _set_causal_conv3d(model.encoder.conv1, "encoder.conv1", weights)
    for i, layer in enumerate(model.encoder.downsamples):
        if isinstance(layer, ResidualBlock):
            _set_residual_block(layer, f"encoder.downsamples.{i}", weights)
        elif isinstance(layer, Resample):
            _set_resample(layer, f"encoder.downsamples.{i}", weights)
    _set_residual_block(model.encoder.mid_res0, "encoder.middle.0", weights)
    _set_attention_block(model.encoder.mid_attn, "encoder.middle.1", weights)
    _set_residual_block(model.encoder.mid_res1, "encoder.middle.2", weights)
    _set_rms_norm(model.encoder.head_norm, "encoder.head.0", weights)
    _set_causal_conv3d(model.encoder.head_conv, "encoder.head.2", weights)

    _set_causal_conv3d(model.conv1, "conv1", weights)
    _set_causal_conv3d(model.conv2, "conv2", weights)

    _set_causal_conv3d(model.decoder.conv1, "decoder.conv1", weights)
    _set_residual_block(model.decoder.mid_res0, "decoder.middle.0", weights)
    _set_attention_block(model.decoder.mid_attn, "decoder.middle.1", weights)
    _set_residual_block(model.decoder.mid_res1, "decoder.middle.2", weights)
    for i, layer in enumerate(model.decoder.upsamples):
        if isinstance(layer, ResidualBlock):
            _set_residual_block(layer, f"decoder.upsamples.{i}", weights)
        elif isinstance(layer, Resample):
            _set_resample(layer, f"decoder.upsamples.{i}", weights)
    _set_rms_norm(model.decoder.head_norm, "decoder.head.0", weights)
    _set_causal_conv3d(model.decoder.head_conv, "decoder.head.2", weights)


def _assign_wan_vae_weights(model: WanVAE, weights: dict[str, mx.array]) -> None:
    if getattr(model, "layout", "wan22") == "wan21":
        _assign_wan21_vae_weights(model, weights)
        return
    _set_causal_conv3d(model.encoder.conv1, "encoder.conv1", weights)
    enc_idx = 0
    for block in model.encoder.downsamples:
        enc_idx = _set_down_block(block, "encoder.downsamples", enc_idx, weights)
    _set_residual_block(model.encoder.mid_res0, "encoder.middle.0", weights)
    _set_attention_block(model.encoder.mid_attn, "encoder.middle.1", weights)
    _set_residual_block(model.encoder.mid_res1, "encoder.middle.2", weights)
    _set_rms_norm(model.encoder.head_norm, "encoder.head.0", weights)
    _set_causal_conv3d(model.encoder.head_conv, "encoder.head.2", weights)

    _set_causal_conv3d(model.conv1, "conv1", weights)
    _set_causal_conv3d(model.conv2, "conv2", weights)

    _set_causal_conv3d(model.decoder.conv1, "decoder.conv1", weights)
    _set_residual_block(model.decoder.mid_res0, "decoder.middle.0", weights)
    _set_attention_block(model.decoder.mid_attn, "decoder.middle.1", weights)
    _set_residual_block(model.decoder.mid_res1, "decoder.middle.2", weights)
    dec_idx = 0
    for block in model.decoder.upsamples:
        dec_idx = _set_up_block(block, "decoder.upsamples", dec_idx, weights)
    _set_rms_norm(model.decoder.head_norm, "decoder.head.0", weights)
    _set_causal_conv3d(model.decoder.head_conv, "decoder.head.2", weights)


def _is_diffusers_vae_keys(keys: set[str]) -> bool:
    return any(k.startswith("encoder.conv_in.") or k.startswith("encoder.down_blocks.") for k in keys)


def _diffusers_to_official_vae_key(key: str) -> str:
    replacements = {
        "encoder.conv_in.": "encoder.conv1.",
        "decoder.conv_in.": "decoder.conv1.",
        "encoder.norm_out.": "encoder.head.0.",
        "encoder.conv_out.": "encoder.head.2.",
        "decoder.norm_out.": "decoder.head.0.",
        "decoder.conv_out.": "decoder.head.2.",
        "quant_conv.": "conv1.",
        "post_quant_conv.": "conv2.",
    }
    for old, new in replacements.items():
        if key.startswith(old):
            return new + key[len(old) :]

    if key.startswith("encoder.down_blocks."):
        key = key.replace("encoder.down_blocks.", "encoder.downsamples.")
        key = re.sub(r"\.resnets\.(\d+)\.norm1\.", r".downsamples.\1.residual.0.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv1\.", r".downsamples.\1.residual.2.", key)
        key = re.sub(r"\.resnets\.(\d+)\.norm2\.", r".downsamples.\1.residual.3.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv2\.", r".downsamples.\1.residual.6.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv_shortcut\.", r".downsamples.\1.shortcut.", key)
        key = key.replace(".downsampler.resample.1.", ".downsamples.2.resample.1.")
        key = key.replace(".downsampler.time_conv.", ".downsamples.2.time_conv.")
    elif key.startswith("decoder.up_blocks."):
        key = key.replace("decoder.up_blocks.", "decoder.upsamples.")
        key = re.sub(r"\.resnets\.(\d+)\.norm1\.", r".upsamples.\1.residual.0.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv1\.", r".upsamples.\1.residual.2.", key)
        key = re.sub(r"\.resnets\.(\d+)\.norm2\.", r".upsamples.\1.residual.3.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv2\.", r".upsamples.\1.residual.6.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv_shortcut\.", r".upsamples.\1.shortcut.", key)
        key = key.replace(".upsampler.resample.1.", ".upsamples.3.resample.1.")
        key = key.replace(".upsampler.time_conv.", ".upsamples.3.time_conv.")

    key = key.replace("encoder.mid_block.resnets.0.", "encoder.middle.0.")
    key = key.replace("encoder.mid_block.resnets.1.", "encoder.middle.2.")
    key = key.replace("encoder.mid_block.attentions.0.", "encoder.middle.1.")
    key = key.replace("decoder.mid_block.resnets.0.", "decoder.middle.0.")
    key = key.replace("decoder.mid_block.resnets.1.", "decoder.middle.2.")
    key = key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.")

    # Mid-block resnets: norm/conv/shortcut (same as down_blocks/up_blocks but fixed indices)
    key = key.replace("encoder.middle.0.norm1.", "encoder.middle.0.residual.0.")
    key = key.replace("encoder.middle.0.conv1.", "encoder.middle.0.residual.2.")
    key = key.replace("encoder.middle.0.norm2.", "encoder.middle.0.residual.3.")
    key = key.replace("encoder.middle.0.conv2.", "encoder.middle.0.residual.6.")
    key = key.replace("encoder.middle.0.conv_shortcut.", "encoder.middle.0.shortcut.")
    key = key.replace("encoder.middle.2.norm1.", "encoder.middle.2.residual.0.")
    key = key.replace("encoder.middle.2.conv1.", "encoder.middle.2.residual.2.")
    key = key.replace("encoder.middle.2.norm2.", "encoder.middle.2.residual.3.")
    key = key.replace("encoder.middle.2.conv2.", "encoder.middle.2.residual.6.")
    key = key.replace("encoder.middle.2.conv_shortcut.", "encoder.middle.2.shortcut.")

    key = key.replace("decoder.middle.0.norm1.", "decoder.middle.0.residual.0.")
    key = key.replace("decoder.middle.0.conv1.", "decoder.middle.0.residual.2.")
    key = key.replace("decoder.middle.0.norm2.", "decoder.middle.0.residual.3.")
    key = key.replace("decoder.middle.0.conv2.", "decoder.middle.0.residual.6.")
    key = key.replace("decoder.middle.0.conv_shortcut.", "decoder.middle.0.shortcut.")
    key = key.replace("decoder.middle.2.norm1.", "decoder.middle.2.residual.0.")
    key = key.replace("decoder.middle.2.conv1.", "decoder.middle.2.residual.2.")
    key = key.replace("decoder.middle.2.norm2.", "decoder.middle.2.residual.3.")
    key = key.replace("decoder.middle.2.conv2.", "decoder.middle.2.residual.6.")
    key = key.replace("decoder.middle.2.conv_shortcut.", "decoder.middle.2.shortcut.")
    return key


def _normalize_vae_state_dict(raw: dict[str, mx.array]) -> dict[str, mx.array]:
    if not _is_diffusers_vae_keys(set(raw)):
        return raw
    out: dict[str, mx.array] = {}
    for key, value in raw.items():
        out[_diffusers_to_official_vae_key(key)] = value
    return out


def _read_wan_vae_config(bundle_root: Path) -> dict[str, Any]:
    cfg_path = bundle_root / "vae" / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        import json
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Wan: cannot read VAE config {cfg_path}: {e}") from e


def _scale_tensors_from_vae_cfg(
    cfg: dict[str, Any],
    z_dim: int,
    *,
    array_fn: Any | None = None,
    layout: str = "wan22",
) -> tuple[mx.array, mx.array]:
    if array_fn is None:
        array_fn = mx.array
    mean = cfg.get("latents_mean")
    std = cfg.get("latents_std")
    if (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == z_dim
        and len(std) == z_dim
    ):
        mean_a = array_fn(mean, dtype=mx.float32).reshape(1, z_dim, 1, 1, 1)
        inv_std = array_fn([1.0 / float(s) for s in std], dtype=mx.float32).reshape(
            1, z_dim, 1, 1, 1
        )
        return mean_a, inv_std
    if layout == "wan21" or z_dim == 16:
        mean_src, std_src = _WAN21_VAE_MEAN, _WAN21_VAE_STD
    else:
        mean_src, std_src = _WAN22_VAE_MEAN, _WAN22_VAE_STD
    if len(mean_src) != z_dim or len(std_src) != z_dim:
        raise RuntimeError(
            f"Wan VAE scale vector length mismatch: z_dim={z_dim}, layout={layout!r}"
        )
    mean_a = array_fn(mean_src, dtype=mx.float32).reshape(1, z_dim, 1, 1, 1)
    inv_std = array_fn([1.0 / s for s in std_src], dtype=mx.float32).reshape(
        1, z_dim, 1, 1, 1
    )
    return mean_a, inv_std


def _load_vae_state_dict(
    bundle_root: Path, *, array_fn: Any | None = None, load_fn: Any | None = None
) -> dict[str, mx.array]:
    if array_fn is None:
        array_fn = mx.array
    pth_candidates = [
        bundle_root / "Wan2.2_VAE.pth",
        bundle_root / "Wan2_2_VAE.pth",
        bundle_root / "Wan2.1_VAE.pth",
    ]
    for pth in pth_candidates:
        if pth.is_file():
            logger.info("Loading Wan VAE weights from %s", pth)
            from backend.engine.common.bundle.pytorch_bin_numpy import state_dict_to_numpy

            sd = state_dict_to_numpy(pth)
            out: dict[str, mx.array] = {}
            for k, v in sd.items():
                arr = np.asarray(v, dtype=np.float32)
                out[k] = array_fn(arr).astype(mx.float32)
            return out

    vae_dir = bundle_root / "vae"
    if vae_dir.is_dir():
        merged: dict[str, mx.array] = {}
        for sf in sorted(vae_dir.glob("*.safetensors")):
            merged.update(load_weights_dict(load_fn, str(sf)))
        if merged:
            logger.info("Loading Wan VAE weights from %s (%d tensors)", vae_dir, len(merged))
            merged = {k: v.astype(mx.float32) for k, v in merged.items()}
            return _normalize_vae_state_dict(merged)

    root_vae = bundle_root / "vae.safetensors"
    if root_vae.is_file():
        merged = load_weights_dict(load_fn, str(root_vae))
        if merged:
            logger.info("Loading Wan VAE weights from %s (%d tensors)", root_vae, len(merged))
            merged = {k: v.astype(mx.float32) for k, v in merged.items()}
            return _normalize_vae_state_dict(merged)

    raise RuntimeError(
        f"Wan VAE weights not found under {bundle_root}: "
        "expected Wan2.2_VAE.pth, vae/*.safetensors, or vae.safetensors"
    )


def _infer_wan_vae_layout(weights: dict[str, Any]) -> str:
    """Return ``wan21`` or ``wan22`` from checkpoint topology."""
    if "encoder.head.2.weight" in weights:
        head_out = int(weights["encoder.head.2.weight"].shape[0])
        if head_out <= 32:
            return "wan21"
    enc_in, _pix = _infer_wan_vae_io_channels(weights)
    if enc_in == 12:
        return "wan22"
    if any(k.startswith("encoder.downsamples.10.") for k in weights):
        return "wan21"
    return "wan22"


def _infer_wan21_vae_dims(weights: dict[str, Any]) -> tuple[int, int]:
    c_dim = int(weights["encoder.conv1.weight"].shape[0])
    z_dim = int(weights["encoder.head.2.weight"].shape[0]) // 2
    return c_dim, z_dim


def _infer_wan_vae_io_channels(weights: dict[str, Any]) -> tuple[int, int]:
    """Return ``(encoder_in_channels, pixel_out_channels)`` from checkpoint keys."""
    if "encoder.conv1.weight" in weights:
        enc_in = int(weights["encoder.conv1.weight"].shape[1])
    elif "encoder.conv_in.weight" in weights:
        enc_in = int(weights["encoder.conv_in.weight"].shape[1])
    else:
        enc_in = 12
    if "decoder.head.2.weight" in weights:
        pix_out = int(weights["decoder.head.2.weight"].shape[0])
    elif "decoder.conv_out.weight" in weights:
        pix_out = int(weights["decoder.conv_out.weight"].shape[0])
    else:
        pix_out = 12 if enc_in == 12 else 3
    return enc_in, pix_out


def build_wan21_vae_mlx(
    *,
    dim: int = 96,
    z_dim: int = 16,
    dim_mult: list[int] | None = None,
    temperal_downsample: list[bool] | None = None,
    encoder_in_channels: int = 3,
    pixel_out_channels: int = 3,
) -> WanVAE:
    return WanVAE(
        dim=dim,
        z_dim=z_dim,
        dim_mult=dim_mult or [1, 2, 4, 4],
        temperal_downsample=temperal_downsample or [False, True, True],
        dropout=0.0,
        encoder_in_channels=encoder_in_channels,
        pixel_out_channels=pixel_out_channels,
        layout="wan21",
    )


def build_wan22_vae_mlx(
    *,
    z_dim: int = 48,
    c_dim: int = 160,
    dim_mult: list[int] | None = None,
    temperal_downsample: list[bool] | None = None,
    encoder_in_channels: int = 12,
    pixel_out_channels: int = 12,
) -> WanVAE:
    return WanVAE(
        dim=c_dim,
        dec_dim=256,
        z_dim=z_dim,
        dim_mult=dim_mult or [1, 2, 4, 4],
        temperal_downsample=temperal_downsample or [False, True, True],
        dropout=0.0,
        encoder_in_channels=encoder_in_channels,
        pixel_out_channels=pixel_out_channels,
        layout="wan22",
    )


_vae_cache: dict[str, Wan22VAE] = {}


def load_wan_vae(ctx: RuntimeContext, bundle_root: Path | str) -> Wan22VAE:
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(f"Wan VAE bundle directory missing: {root}")

    key = str(root.resolve())
    cached = _vae_cache.get(key)
    if cached is not None:
        return cached

    weights = _load_vae_state_dict(
        root,
        array_fn=ctx.array,
        load_fn=getattr(ctx, "load_weights", None),
    )
    if "encoder.conv1.weight" not in weights and "encoder.conv_in.weight" in weights:
        weights = _normalize_vae_state_dict(weights)

    enc_in, pix_out = _infer_wan_vae_io_channels(weights)
    layout = _infer_wan_vae_layout(weights)
    vae_cfg = _read_wan_vae_config(root)
    if layout == "wan21":
        c_dim, z_dim = _infer_wan21_vae_dims(weights)
        model = build_wan21_vae_mlx(
            dim=c_dim,
            z_dim=z_dim,
            dim_mult=list(vae_cfg.get("dim_mult") or [1, 2, 4, 4]),
            temperal_downsample=list(vae_cfg.get("temperal_downsample") or [False, True, True]),
            encoder_in_channels=enc_in,
            pixel_out_channels=pix_out,
        )
    else:
        z_dim = int(vae_cfg.get("z_dim", 48))
        model = build_wan22_vae_mlx(
            z_dim=z_dim,
            c_dim=int(vae_cfg.get("base_dim", 160)),
            dim_mult=list(vae_cfg.get("dim_mult") or [1, 2, 4, 4]),
            temperal_downsample=list(vae_cfg.get("temperal_downsample") or [False, True, True]),
            encoder_in_channels=enc_in,
            pixel_out_channels=pix_out,
        )
    _assign_wan_vae_weights(model, weights)

    mean, inv_std = _scale_tensors_from_vae_cfg(
        vae_cfg, model.z_dim, array_fn=ctx.array, layout=layout,
    )
    wrapper = Wan22VAE(model=model, scale_mean=mean, scale_inv_std=inv_std, z_dim=model.z_dim)
    _vae_cache[key] = wrapper
    logger.info("Wan VAE loaded from %s (layout=%s, z_dim=%d)", root, layout, model.z_dim)
    return wrapper


def decode_wan_vae_latents(
    ctx: RuntimeContext,
    latents_bcthw: mx.array,
    bundle_root: Path | str,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    spatial_tiling: bool = False,
    spatial_scale: int = 16,
) -> mx.array:
    """Decode ``[B,C,T,H,W]`` latents to RGB pixels ``[B,C,T,H,W]`` in ``[-1, 1]``."""
    vae = load_wan_vae(ctx, bundle_root)
    logger.info("Wan VAE decode: latent shape=%s", tuple(latents_bcthw.shape))
    if latents_bcthw.ndim != 5:
        raise RuntimeError(f"Wan VAE decode expects 5D latents [B,C,T,H,W], got {latents_bcthw.shape}")
    if int(latents_bcthw.shape[0]) != 1:
        raise RuntimeError(f"Wan VAE decode batch size must be 1, got {latents_bcthw.shape[0]}")

    def _stage(frac: float) -> None:
        if on_stage is not None:
            on_stage(min(1.0, max(0.0, float(frac))))

    scale = (vae.scale_mean, vae.scale_inv_std)
    vae_cfg = _read_wan_vae_config(Path(bundle_root))
    tile_params = _wan_vae_tile_params(vae_cfg, spatial_scale)
    _, _, _, lh, lw = latents_bcthw.shape
    needs_tiling = _needs_wan_spatial_tiling(int(lh), int(lw), tile_params, enabled=True)
    use_tiling = bool(spatial_tiling) or needs_tiling
    if needs_tiling and not spatial_tiling and on_log is not None:
        on_log(
            f"Wan VAE auto-enabled spatial tiling (latent {lh}x{lw}, "
            f"tile {tile_params.tile_latent_min_height}x{tile_params.tile_latent_min_width})"
        )

    _stage(0.02)
    if use_tiling:
        if spatial_tiling and on_log is not None:
            on_log(
                f"Wan VAE spatial tiling enabled (latent {lh}x{lw}, "
                f"tile {tile_params.tile_latent_min_height}x{tile_params.tile_latent_min_width})"
            )
        sample = _tiled_spatial_decode_wan(
            ctx,
            vae.model,
            latents_bcthw,
            scale,
            tile_params,
            on_stage=_stage,
            on_log=on_log,
        )
    else:
        sample = _decode_wan_latent_volume(
            vae.model, latents_bcthw, scale, on_stage=_stage,
        )
    sample = mx.clip(sample, -1.0, 1.0)
    ctx.eval(sample)
    _stage(1.0)
    logger.info("Wan VAE decode done: pixel shape=%s", tuple(sample.shape))
    return sample


def encode_wan_vae_volume(
    ctx: RuntimeContext,
    pixels_bcthw: mx.array,
    bundle_root: Path | str,
) -> mx.array:
    """Encode RGB video ``[B,C,T,H,W]`` float ``[-1,1]`` to latents ``[B,C,T,h,w]``."""
    vae = load_wan_vae(ctx, bundle_root)
    if pixels_bcthw.ndim != 5:
        raise RuntimeError(f"Wan VAE encode expects [B,C,T,H,W], got {pixels_bcthw.shape}")
    logger.info("Wan VAE encode: pixel shape=%s", tuple(pixels_bcthw.shape))
    latents = vae.model.encode(pixels_bcthw, (vae.scale_mean, vae.scale_inv_std))
    ctx.eval(latents)
    logger.info("Wan VAE encode done: latent shape=%s", tuple(latents.shape))
    return latents


def encode_wan_vae_image(
    ctx: RuntimeContext,
    image_chw: mx.array,
    bundle_root: Path | str,
) -> mx.array:
    """Encode a single RGB frame ``[C,H,W]`` (float ``[-1,1]``) to latents ``[1,C,1,h,w]``."""
    vae = load_wan_vae(ctx, bundle_root)
    if image_chw.ndim != 3:
        raise RuntimeError(f"Wan VAE encode expects image [C,H,W], got {image_chw.shape}")
    pixels = mx.expand_dims(mx.expand_dims(image_chw, 0), 2)
    logger.info("Wan VAE encode: pixel shape=%s", tuple(pixels.shape))
    latents = vae.model.encode(pixels, (vae.scale_mean, vae.scale_inv_std))
    ctx.eval(latents)
    logger.info("Wan VAE encode done: latent shape=%s", tuple(latents.shape))
    return latents
