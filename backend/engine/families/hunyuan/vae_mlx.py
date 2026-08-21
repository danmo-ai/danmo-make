"""HunyuanVideo-1.5 3D causal VAE — MLX port of diffusers ``AutoencoderKLHunyuanVideo15``.

PyTorch checkpoint tensors use ``[out_c, in_c, kt, kh, kw]`` for Conv3d.
MLX Conv3d expects ``[out_c, kt, kh, kw, in_c]`` with activations in **NDHWC**.
Internal activations use **NCTHW** (PyTorch layout) between layers; Conv3d paths transpose as needed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.ops.attention import (
    build_frame_prefix_causal_bias,
    scaled_dot_product_attention_bhsd_mx,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict
from backend.engine.runtime.mlx_dtype import cast_module_parameters
from backend.engine.runtime.mlx_runtime import run_eval
from backend.engine.runtime._base import RuntimeContext

logger = logging.getLogger(__name__)


def _run_eval(*values: Any) -> None:
    run_eval(None, *values)


def _conv3d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    return mx.transpose(w, (0, 2, 3, 4, 1))


def _ncthw_to_ndhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 4, 1))


def _ndhwc_to_ncthw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 4, 1, 2, 3))


def _repeat_interleave_channels(x_ncthw: mx.array, repeats: int) -> mx.array:
    if repeats <= 1:
        return x_ncthw
    b, c, t, h, w = x_ncthw.shape
    x = mx.reshape(x_ncthw, (b, c, 1, t, h, w))
    x = mx.repeat(x, repeats, axis=2)
    return mx.reshape(x, (b, c * repeats, t, h, w))


def _pad_spatial_replicate_ncthw(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    if pad_h <= 0 and pad_w <= 0:
        return x
    if pad_h > 0:
        top = mx.repeat(x[:, :, :, :1, :], pad_h, axis=3)
        bottom = mx.repeat(x[:, :, :, -1:, :], pad_h, axis=3)
        x = mx.concatenate([top, x, bottom], axis=3)
    if pad_w > 0:
        left = mx.repeat(x[:, :, :, :, :1], pad_w, axis=4)
        right = mx.repeat(x[:, :, :, :, -1:], pad_w, axis=4)
        x = mx.concatenate([left, x, right], axis=4)
    return x


def _dcae_upsample_rearrange(tensor_ncthw: mx.array, r1: int = 1, r2: int = 2, r3: int = 2) -> mx.array:
    """Convert ``(b, r1*r2*r3*c, f, h, w)`` -> ``(b, c, r1*f, r2*h, r3*w)``."""
    b, packed_c, f, h, w = tensor_ncthw.shape
    factor = r1 * r2 * r3
    c = packed_c // factor
    x = mx.reshape(tensor_ncthw, (b, r1, r2, r3, c, f, h, w))
    x = mx.transpose(x, (0, 4, 5, 1, 6, 2, 7, 3))
    return mx.reshape(x, (b, c, f * r1, h * r2, w * r3))


def _dcae_downsample_rearrange(tensor_ncthw: mx.array, r1: int = 1, r2: int = 2, r3: int = 2) -> mx.array:
    """Convert ``(b, c, r1*f, r2*h, r3*w)`` -> ``(b, r1*r2*r3*c, f, h, w)``."""
    b, c, packed_f, packed_h, packed_w = tensor_ncthw.shape
    f, h, w = packed_f // r1, packed_h // r2, packed_w // r3
    x = mx.reshape(tensor_ncthw, (b, c, f, r1, h, r2, w, r3))
    x = mx.transpose(x, (0, 3, 5, 7, 1, 2, 4, 6))
    return mx.reshape(x, (b, r1 * r2 * r3 * c, f, h, w))


def _prepare_causal_attention_mask(
    n_frame: int,
    n_hw: int,
    batch_size: int,
) -> mx.array:
    return build_frame_prefix_causal_bias(
        mx,
        n_frame,
        n_hw,
        batch_size,
        dtype=mx.float32,
        neg_value=float("-inf"),
    )


class HunyuanVideo15CausalConv3d(nn.Module):
    """Maps to diffusers ``HunyuanVideo15CausalConv3d`` (replicate causal pad + Conv3d)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.kernel_size = kernel_size
        self.pad_mode = pad_mode
        kt, kh, kw = kernel_size
        self.time_pad = kt - 1
        self.spatial_pad_h = kh // 2
        self.spatial_pad_w = kw // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, hidden_states_ncthw: mx.array) -> mx.array:
        x = hidden_states_ncthw
        if self.time_pad > 0:
            first = x[:, :, :1, :, :]
            if self.pad_mode == "replicate":
                x = mx.concatenate([mx.repeat(first, self.time_pad, axis=2), x], axis=2)
            else:
                zeros = mx.repeat(mx.zeros_like(first), self.time_pad, axis=2)
                x = mx.concatenate([zeros, x], axis=2)
        x = _pad_spatial_replicate_ncthw(x, self.spatial_pad_h, self.spatial_pad_w)
        x_ndhwc = _ncthw_to_ndhwc(x)
        out = self.conv(x_ndhwc)
        return _ndhwc_to_ncthw(out)


class HunyuanVideo15RMSNorm(nn.Module):
    """Maps to diffusers ``HunyuanVideo15RMS_norm`` (channel-first, video)."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        del channel_first
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims)
        self.scale = dim**0.5
        self.gamma = mx.ones(shape)
        self.bias = mx.zeros(shape) if bias else None
        self._use_bias = bias

    def __call__(self, x_ncthw: mx.array) -> mx.array:
        norm = x_ncthw / (mx.sqrt(mx.sum(x_ncthw * x_ncthw, axis=1, keepdims=True) + 1e-6))
        out = norm * self.scale * self.gamma
        if self._use_bias and self.bias is not None:
            out = out + self.bias
        return out


class HunyuanVideo15AttnBlock(nn.Module):
    """Maps to diffusers ``HunyuanVideo15AttnBlock`` with causal temporal attention."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = HunyuanVideo15RMSNorm(in_channels, images=False)
        self.to_q = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_k = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.to_v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1)

    def __call__(self, x_ncthw: mx.array) -> mx.array:
        identity = x_ncthw
        x = self.norm(x_ncthw)

        q = _ndhwc_to_ncthw(self.to_q(_ncthw_to_ndhwc(x)))
        k = _ndhwc_to_ncthw(self.to_k(_ncthw_to_ndhwc(x)))
        v = _ndhwc_to_ncthw(self.to_v(_ncthw_to_ndhwc(x)))

        batch_size, channels, frames, height, width = q.shape
        n_hw = height * width
        seq_len = frames * n_hw

        def _flatten_qkv(t: mx.array) -> mx.array:
            t = mx.reshape(t, (batch_size, channels, seq_len))
            t = mx.transpose(t, (0, 2, 1))
            return mx.expand_dims(t, axis=1)

        query = _flatten_qkv(q)
        key = _flatten_qkv(k)
        value = _flatten_qkv(v)

        attention_mask = _prepare_causal_attention_mask(frames, n_hw, batch_size)
        x_attn = scaled_dot_product_attention_bhsd_mx(
            mx,
            query,
            key,
            value,
            scale=channels**-0.5,
            mask=attention_mask,
            compute_dtype=mx.float32,
            out_dtype=x.dtype,
        )

        x_attn = mx.squeeze(x_attn, axis=1)
        x_attn = mx.transpose(x_attn, (0, 2, 1))
        x_attn = mx.reshape(x_attn, (batch_size, channels, frames, height, width))
        x_attn = _ndhwc_to_ncthw(self.proj_out(_ncthw_to_ndhwc(x_attn)))
        return x_attn + identity


class HunyuanVideo15ResnetBlock(nn.Module):
    """Maps to diffusers ``HunyuanVideo15ResnetBlock``."""

    def __init__(self, in_channels: int, out_channels: int | None = None, non_linearity: str = "swish"):
        super().__init__()
        del non_linearity
        out_channels = out_channels or in_channels
        self.norm1 = HunyuanVideo15RMSNorm(in_channels, images=False)
        self.conv1 = HunyuanVideo15CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = HunyuanVideo15RMSNorm(out_channels, images=False)
        self.conv2 = HunyuanVideo15CausalConv3d(out_channels, out_channels, kernel_size=3)
        self.conv_shortcut: nn.Conv3d | None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        hidden = self.norm1(hidden_states)
        hidden = nn.silu(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = nn.silu(hidden)
        hidden = self.conv2(hidden)
        if self.conv_shortcut is not None:
            residual = _ndhwc_to_ncthw(self.conv_shortcut(_ncthw_to_ndhwc(residual)))
        return hidden + residual


class HunyuanVideo15MidBlock(nn.Module):
    """Maps to diffusers ``HunyuanVideo15MidBlock``."""

    def __init__(self, in_channels: int, num_layers: int = 1, add_attention: bool = True):
        super().__init__()
        self.add_attention = add_attention
        self.resnet_0 = HunyuanVideo15ResnetBlock(in_channels, in_channels)
        self._num_layers = num_layers
        for i in range(num_layers):
            if add_attention:
                setattr(self, f"attn_{i}", HunyuanVideo15AttnBlock(in_channels))
            setattr(self, f"resnet_{i + 1}", HunyuanVideo15ResnetBlock(in_channels, in_channels))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden = self.resnet_0(hidden_states)
        for i in range(self._num_layers):
            if self.add_attention:
                hidden = getattr(self, f"attn_{i}")(hidden)
            hidden = getattr(self, f"resnet_{i + 1}")(hidden)
            _run_eval(hidden)
        return hidden


class HunyuanVideo15Downsample(nn.Module):
    """Maps to diffusers ``HunyuanVideo15Downsample``."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        self.conv = HunyuanVideo15CausalConv3d(in_channels, out_channels // factor, kernel_size=3)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = _dcae_downsample_rearrange(h[:, :, :1, :, :], r1=1, r2=2, r3=2)
            h_first = mx.concatenate([h_first, h_first], axis=1)
            h_next = _dcae_downsample_rearrange(h[:, :, 1:, :, :], r1=r1, r2=2, r3=2)
            h = mx.concatenate([h_first, h_next], axis=2)

            x_first = _dcae_downsample_rearrange(x[:, :, :1, :, :], r1=1, r2=2, r3=2)
            b, _, t, hi, wi = x_first.shape
            x_first = mx.mean(
                mx.reshape(x_first, (b, h.shape[1], self.group_size // 2, t, hi, wi)),
                axis=2,
            )
            x_next = _dcae_downsample_rearrange(x[:, :, 1:, :, :], r1=r1, r2=2, r3=2)
            b, _, t, hi, wi = x_next.shape
            x_next = mx.mean(
                mx.reshape(x_next, (b, h.shape[1], self.group_size, t, hi, wi)),
                axis=2,
            )
            shortcut = mx.concatenate([x_first, x_next], axis=2)
        else:
            h = _dcae_downsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = _dcae_downsample_rearrange(x, r1=r1, r2=2, r3=2)
            b, _, t, hi, wi = shortcut.shape
            shortcut = mx.mean(
                mx.reshape(shortcut, (b, h.shape[1], self.group_size, t, hi, wi)),
                axis=2,
            )
        return h + shortcut


class HunyuanVideo15Upsample(nn.Module):
    """Maps to diffusers ``HunyuanVideo15Upsample``."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = HunyuanVideo15CausalConv3d(in_channels, out_channels * factor, kernel_size=3)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def __call__(self, x: mx.array) -> mx.array:
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = _dcae_upsample_rearrange(h[:, :, :1, :, :], r1=1, r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = _dcae_upsample_rearrange(h[:, :, 1:, :, :], r1=r1, r2=2, r3=2)
            h = mx.concatenate([h_first, h_next], axis=2)

            x_first = _dcae_upsample_rearrange(x[:, :, :1, :, :], r1=1, r2=2, r3=2)
            x_first = _repeat_interleave_channels(x_first, self.repeats // 2)
            x_next = _dcae_upsample_rearrange(x[:, :, 1:, :, :], r1=r1, r2=2, r3=2)
            x_next = _repeat_interleave_channels(x_next, self.repeats)
            shortcut = mx.concatenate([x_first, x_next], axis=2)
        else:
            h = _dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
            shortcut = _repeat_interleave_channels(x, self.repeats)
            shortcut = _dcae_upsample_rearrange(shortcut, r1=r1, r2=2, r3=2)
        return h + shortcut


class HunyuanVideo15DownBlock3D(nn.Module):
    """Maps to diffusers ``HunyuanVideo15DownBlock3D``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample_out_channels: int | None = None,
        add_temporal_downsample: bool = True,
    ):
        super().__init__()
        for i in range(num_layers):
            inch = in_channels if i == 0 else out_channels
            setattr(self, f"resnet_{i}", HunyuanVideo15ResnetBlock(inch, out_channels))
        self._num_layers = num_layers
        if downsample_out_channels is not None:
            self.downsampler_0 = HunyuanVideo15Downsample(
                out_channels,
                downsample_out_channels,
                add_temporal_downsample=add_temporal_downsample,
            )
        else:
            self.downsampler_0 = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = hidden_states
        for i in range(self._num_layers):
            h = getattr(self, f"resnet_{i}")(h)
        if self.downsampler_0 is not None:
            h = self.downsampler_0(h)
        return h


class HunyuanVideo15UpBlock3D(nn.Module):
    """Maps to diffusers ``HunyuanVideo15UpBlock3D``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample_out_channels: int | None = None,
        add_temporal_upsample: bool = True,
    ):
        super().__init__()
        for i in range(num_layers):
            inch = in_channels if i == 0 else out_channels
            setattr(self, f"resnet_{i}", HunyuanVideo15ResnetBlock(inch, out_channels))
        self._num_layers = num_layers
        if upsample_out_channels is not None:
            self.upsampler_0 = HunyuanVideo15Upsample(
                out_channels,
                upsample_out_channels,
                add_temporal_upsample=add_temporal_upsample,
            )
        else:
            self.upsampler_0 = None

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = hidden_states
        for i in range(self._num_layers):
            h = getattr(self, f"resnet_{i}")(h)
            _run_eval(h)
        if self.upsampler_0 is not None:
            h = self.upsampler_0(h)
            _run_eval(h)
        return h


class HunyuanVideo15Encoder3DMlx(nn.Module):
    """MLX clone of diffusers ``HunyuanVideo15Encoder3D``."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 1024, 1024),
        layers_per_block: int = 2,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 16,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.group_size = block_out_channels[-1] // out_channels
        self.conv_in = HunyuanVideo15CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        input_channel = block_out_channels[0]
        self._num_down_blocks = len(block_out_channels)
        for i in range(self._num_down_blocks):
            add_spatial_downsample = i < int(np.log2(spatial_compression_ratio))
            output_channel = block_out_channels[i]
            if not add_spatial_downsample:
                down_block = HunyuanVideo15DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=None,
                    add_temporal_downsample=False,
                )
                input_channel = output_channel
            else:
                add_temporal = i >= int(np.log2(spatial_compression_ratio // temporal_compression_ratio))
                downsample_out = block_out_channels[i + 1] if downsample_match_channel else output_channel
                down_block = HunyuanVideo15DownBlock3D(
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    downsample_out_channels=downsample_out,
                    add_temporal_downsample=add_temporal,
                )
                input_channel = downsample_out
            setattr(self, f"down_block_{i}", down_block)

        self.mid_block = HunyuanVideo15MidBlock(in_channels=block_out_channels[-1])
        self.norm_out = HunyuanVideo15RMSNorm(block_out_channels[-1], images=False)
        self.conv_out = HunyuanVideo15CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden = self.conv_in(hidden_states)
        for i in range(self._num_down_blocks):
            hidden = getattr(self, f"down_block_{i}")(hidden)
            _run_eval(hidden)
        hidden = self.mid_block(hidden)
        _run_eval(hidden)

        batch_size, _, frame, height, width = hidden.shape
        shortcut = mx.mean(
            mx.reshape(hidden, (batch_size, -1, self.group_size, frame, height, width)),
            axis=2,
        )
        hidden = self.norm_out(hidden)
        hidden = nn.silu(hidden)
        hidden = self.conv_out(hidden)
        return hidden + shortcut


class HunyuanVideo15Decoder3DMlx(nn.Module):
    """MLX clone of diffusers ``HunyuanVideo15Decoder3D``."""

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (1024, 1024, 512, 256, 128),
        layers_per_block: int = 2,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.repeat = block_out_channels[0] // in_channels
        self.conv_in = HunyuanVideo15CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)
        self.mid_block = HunyuanVideo15MidBlock(in_channels=block_out_channels[0])

        input_channel = block_out_channels[0]
        self._num_up_blocks = len(block_out_channels)
        for i in range(self._num_up_blocks):
            output_channel = block_out_channels[i]
            add_spatial = i < int(np.log2(spatial_compression_ratio))
            add_temporal = i < int(np.log2(temporal_compression_ratio))
            if add_spatial or add_temporal:
                upsample_out = block_out_channels[i + 1] if upsample_match_channel else output_channel
                up_block = HunyuanVideo15UpBlock3D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=upsample_out,
                    add_temporal_upsample=add_temporal,
                )
                input_channel = upsample_out
            else:
                up_block = HunyuanVideo15UpBlock3D(
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    upsample_out_channels=None,
                    add_temporal_upsample=False,
                )
                input_channel = output_channel
            setattr(self, f"up_block_{i}", up_block)

        self.norm_out = HunyuanVideo15RMSNorm(block_out_channels[-1], images=False)
        self.conv_out = HunyuanVideo15CausalConv3d(block_out_channels[-1], out_channels, kernel_size=3)

    def __call__(
        self,
        hidden_states: mx.array,
        on_stage: Callable[[float], None] | None = None,
    ) -> mx.array:
        def _stage(frac: float) -> None:
            if on_stage is not None:
                on_stage(min(1.0, max(0.0, float(frac))))

        hidden = self.conv_in(hidden_states) + _repeat_interleave_channels(hidden_states, self.repeat)
        _run_eval(hidden)
        _stage(0.08)

        hidden = self.mid_block(hidden)
        _run_eval(hidden)
        _stage(0.22)

        for i in range(self._num_up_blocks):
            hidden = getattr(self, f"up_block_{i}")(hidden)
            _run_eval(hidden)
            _stage(0.22 + 0.68 * ((i + 1) / max(self._num_up_blocks, 1)))

        hidden = self.norm_out(hidden)
        hidden = nn.silu(hidden)
        hidden = self.conv_out(hidden)
        _stage(1.0)
        return hidden


def _set_conv3d(mod: nn.Conv3d, prefix: str, weights: dict[str, mx.array]) -> None:
    mod.weight = _conv3d_weight_torch_to_mlx(weights[f"{prefix}.weight"])
    mod.bias = weights[f"{prefix}.bias"]


def _set_causal_conv(mod: HunyuanVideo15CausalConv3d, prefix: str, weights: dict[str, mx.array]) -> None:
    _set_conv3d(mod.conv, f"{prefix}.conv", weights)


def _set_rms_norm(mod: HunyuanVideo15RMSNorm, prefix: str, weights: dict[str, mx.array]) -> None:
    mod.gamma = weights[f"{prefix}.gamma"]
    if mod._use_bias and f"{prefix}.bias" in weights:
        mod.bias = weights[f"{prefix}.bias"]


def _set_resnet_block(rb: HunyuanVideo15ResnetBlock, prefix: str, weights: dict[str, mx.array]) -> None:
    _set_rms_norm(rb.norm1, f"{prefix}.norm1", weights)
    _set_rms_norm(rb.norm2, f"{prefix}.norm2", weights)
    _set_causal_conv(rb.conv1, f"{prefix}.conv1", weights)
    _set_causal_conv(rb.conv2, f"{prefix}.conv2", weights)
    if rb.conv_shortcut is not None:
        _set_conv3d(rb.conv_shortcut, f"{prefix}.conv_shortcut", weights)


def _set_attn_block(attn: HunyuanVideo15AttnBlock, prefix: str, weights: dict[str, mx.array]) -> None:
    _set_rms_norm(attn.norm, f"{prefix}.norm", weights)
    _set_conv3d(attn.to_q, f"{prefix}.to_q", weights)
    _set_conv3d(attn.to_k, f"{prefix}.to_k", weights)
    _set_conv3d(attn.to_v, f"{prefix}.to_v", weights)
    _set_conv3d(attn.proj_out, f"{prefix}.proj_out", weights)


def _set_mid_block(mid: HunyuanVideo15MidBlock, prefix: str, weights: dict[str, mx.array]) -> None:
    _set_resnet_block(mid.resnet_0, f"{prefix}.resnets.0", weights)
    for i in range(mid._num_layers):
        if mid.add_attention:
            _set_attn_block(getattr(mid, f"attn_{i}"), f"{prefix}.attentions.{i}", weights)
        _set_resnet_block(getattr(mid, f"resnet_{i + 1}"), f"{prefix}.resnets.{i + 1}", weights)


def _set_down_block(db: HunyuanVideo15DownBlock3D, prefix: str, weights: dict[str, mx.array]) -> None:
    for i in range(db._num_layers):
        _set_resnet_block(getattr(db, f"resnet_{i}"), f"{prefix}.resnets.{i}", weights)
    if db.downsampler_0 is not None:
        _set_causal_conv(db.downsampler_0.conv, f"{prefix}.downsamplers.0.conv", weights)


def _set_up_block(ub: HunyuanVideo15UpBlock3D, prefix: str, weights: dict[str, mx.array]) -> None:
    for i in range(ub._num_layers):
        _set_resnet_block(getattr(ub, f"resnet_{i}"), f"{prefix}.resnets.{i}", weights)
    if ub.upsampler_0 is not None:
        _set_causal_conv(ub.upsampler_0.conv, f"{prefix}.upsamplers.0.conv", weights)


def _assign_encoder_weights(enc: HunyuanVideo15Encoder3DMlx, weights: dict[str, mx.array]) -> None:
    _set_causal_conv(enc.conv_in, "conv_in", weights)
    for i in range(enc._num_down_blocks):
        _set_down_block(getattr(enc, f"down_block_{i}"), f"down_blocks.{i}", weights)
    _set_mid_block(enc.mid_block, "mid_block", weights)
    _set_rms_norm(enc.norm_out, "norm_out", weights)
    _set_causal_conv(enc.conv_out, "conv_out", weights)


def _assign_decoder_weights(dec: HunyuanVideo15Decoder3DMlx, weights: dict[str, mx.array]) -> None:
    _set_causal_conv(dec.conv_in, "conv_in", weights)
    _set_mid_block(dec.mid_block, "mid_block", weights)
    for i in range(dec._num_up_blocks):
        _set_up_block(getattr(dec, f"up_block_{i}"), f"up_blocks.{i}", weights)
    _set_rms_norm(dec.norm_out, "norm_out", weights)
    _set_causal_conv(dec.conv_out, "conv_out", weights)


def build_hunyuan_encoder_mlx(vae_cfg: dict[str, Any]) -> HunyuanVideo15Encoder3DMlx:
    block_out = tuple(vae_cfg.get("block_out_channels", (128, 256, 512, 1024, 1024)))
    latent_channels = int(vae_cfg.get("latent_channels", 32))
    return HunyuanVideo15Encoder3DMlx(
        in_channels=int(vae_cfg.get("in_channels", 3)),
        out_channels=latent_channels * 2,
        block_out_channels=block_out,
        layers_per_block=int(vae_cfg.get("layers_per_block", 2)),
        temporal_compression_ratio=int(vae_cfg.get("temporal_compression_ratio", 4)),
        spatial_compression_ratio=int(vae_cfg.get("spatial_compression_ratio", 16)),
        downsample_match_channel=bool(vae_cfg.get("downsample_match_channel", True)),
    )


def build_hunyuan_decoder_mlx(vae_cfg: dict[str, Any]) -> HunyuanVideo15Decoder3DMlx:
    block_out = tuple(vae_cfg.get("block_out_channels", (128, 256, 512, 1024, 1024)))
    return HunyuanVideo15Decoder3DMlx(
        in_channels=int(vae_cfg.get("latent_channels", 32)),
        out_channels=int(vae_cfg.get("out_channels", 3)),
        block_out_channels=tuple(reversed(block_out)),
        layers_per_block=int(vae_cfg.get("layers_per_block", 2)),
        spatial_compression_ratio=int(vae_cfg.get("spatial_compression_ratio", 16)),
        temporal_compression_ratio=int(vae_cfg.get("temporal_compression_ratio", 4)),
        upsample_match_channel=bool(vae_cfg.get("upsample_match_channel", True)),
    )


def _read_vae_config(bundle_root: Path) -> dict[str, Any]:
    vae_dir = bundle_root / "vae"
    if not vae_dir.is_dir():
        raise RuntimeError(f"HunyuanVideo VAE directory missing: {vae_dir}")
    cfg_path = vae_dir / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"HunyuanVideo VAE config missing: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _remap_hunyuan_vae_native_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """ModelScope native VAE keys → diffusers-style keys expected by ``_assign_*_weights``."""
    import re

    if not weights:
        return weights
    sample = next(iter(weights))
    if sample.startswith("down_blocks.") or sample.startswith("up_blocks."):
        return weights

    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        nk = key
        nk = re.sub(r"^down\.(\d+)\.block\.(\d+)\.", r"down_blocks.\1.resnets.\2.", nk)
        nk = re.sub(r"^down\.(\d+)\.downsample\.conv\.", r"down_blocks.\1.downsamplers.0.conv.", nk)
        nk = re.sub(r"^up\.(\d+)\.block\.(\d+)\.", r"up_blocks.\1.resnets.\2.", nk)
        nk = re.sub(r"^up\.(\d+)\.upsample\.conv\.", r"up_blocks.\1.upsamplers.0.conv.", nk)
        nk = re.sub(r"^mid\.block_1\.", "mid_block.resnets.0.", nk)
        nk = re.sub(r"^mid\.block_2\.", "mid_block.resnets.1.", nk)
        nk = re.sub(r"^mid\.attn_1\.q\.", "mid_block.attentions.0.to_q.", nk)
        nk = re.sub(r"^mid\.attn_1\.k\.", "mid_block.attentions.0.to_k.", nk)
        nk = re.sub(r"^mid\.attn_1\.v\.", "mid_block.attentions.0.to_v.", nk)
        nk = re.sub(r"^mid\.attn_1\.norm\.", "mid_block.attentions.0.norm.", nk)
        nk = re.sub(r"^mid\.attn_1\.proj_out\.", "mid_block.attentions.0.proj_out.", nk)
        out[nk] = value
    return out


def load_vae_bundle_weights(
    bundle_root: Path, *, load_fn: Any | None = None
) -> tuple[dict[str, mx.array], dict[str, mx.array], dict[str, Any]]:
    vae_cfg = _read_vae_config(bundle_root)
    vae_dir = bundle_root / "vae"

    merged: dict[str, mx.array] = {}
    for sf in sorted(vae_dir.glob("*.safetensors")):
        merged.update(load_weights_dict(load_fn, str(sf)))

    enc_weights: dict[str, mx.array] = {}
    dec_weights: dict[str, mx.array] = {}
    for k, v in merged.items():
        if k.startswith("encoder."):
            enc_weights[k[len("encoder.") :]] = v
        elif k.startswith("decoder."):
            dec_weights[k[len("decoder.") :]] = v

    if not dec_weights:
        raise RuntimeError(
            "No tensors with prefix `decoder.` found in VAE safetensors — cannot load HunyuanVideo decoder."
        )
    if not enc_weights:
        raise RuntimeError(
            "No tensors with prefix `encoder.` found in VAE safetensors — cannot load HunyuanVideo encoder."
        )

    enc_weights = _remap_hunyuan_vae_native_keys(enc_weights)
    dec_weights = _remap_hunyuan_vae_native_keys(dec_weights)

    return enc_weights, dec_weights, vae_cfg


_decoder_cache: dict[str, HunyuanVideo15Decoder3DMlx] = {}
_encoder_cache: dict[str, HunyuanVideo15Encoder3DMlx] = {}


def _get_hunyuan_decoder(
    bundle_root: Path, *, load_fn: Any | None = None
) -> HunyuanVideo15Decoder3DMlx:
    key = str(bundle_root.resolve())
    cached = _decoder_cache.get(key)
    if cached is not None:
        return cached
    enc_w, dec_w, vae_cfg = load_vae_bundle_weights(bundle_root, load_fn=load_fn)
    del enc_w
    dec = build_hunyuan_decoder_mlx(vae_cfg)
    _assign_decoder_weights(dec, dec_w)
    cast_module_parameters(dec, mx.bfloat16)
    _decoder_cache[key] = dec
    return dec


def _get_hunyuan_encoder(
    bundle_root: Path, *, load_fn: Any | None = None
) -> HunyuanVideo15Encoder3DMlx:
    key = str(bundle_root.resolve())
    cached = _encoder_cache.get(key)
    if cached is not None:
        return cached
    enc_w, dec_w, vae_cfg = load_vae_bundle_weights(bundle_root, load_fn=load_fn)
    del dec_w
    enc = build_hunyuan_encoder_mlx(vae_cfg)
    _assign_encoder_weights(enc, enc_w)
    cast_module_parameters(enc, mx.bfloat16)
    _encoder_cache[key] = enc
    return enc


def _gaussian_mode(moments_ncthw: mx.array) -> mx.array:
    """First half of encoder output channels = Gaussian mean (diffusers ``DiagonalGaussianDistribution.mode``)."""
    c = int(moments_ncthw.shape[1]) // 2
    return moments_ncthw[:, :c]


@dataclass(frozen=True)
class _HunyuanVaeTileParams:
    """Diffusers ``AutoencoderKLHunyuanVideo15`` tiling defaults (``tile_overlap_factor=0.25``)."""

    tile_sample_min_height: int
    tile_sample_min_width: int
    tile_latent_min_height: int
    tile_latent_min_width: int
    overlap_factor: float


def _hunyuan_vae_tile_params(vae_cfg: dict[str, Any]) -> _HunyuanVaeTileParams:
    spatial_up = int(vae_cfg.get("spatial_compression_ratio", 16) or 16)
    sample_h = int(vae_cfg.get("tile_sample_min_height", 256))
    sample_w = int(vae_cfg.get("tile_sample_min_width", 256))
    overlap = float(vae_cfg.get("tile_overlap_factor", 0.25))
    return _HunyuanVaeTileParams(
        tile_sample_min_height=sample_h,
        tile_sample_min_width=sample_w,
        tile_latent_min_height=max(1, sample_h // spatial_up),
        tile_latent_min_width=max(1, sample_w // spatial_up),
        overlap_factor=overlap,
    )


def _needs_hunyuan_spatial_tiling(
    spatial_tiling: bool,
    latent_h: int,
    latent_w: int,
    params: _HunyuanVaeTileParams,
) -> bool:
    if not spatial_tiling:
        return False
    return (
        latent_h > params.tile_latent_min_height
        or latent_w > params.tile_latent_min_width
    )


def _blend_v(a_ncthw: mx.array, b_ncthw: mx.array, blend_extent: int) -> mx.array:
    """Vertical blend for tiled decode (diffusers ``blend_v``)."""
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


def _blend_h(a_ncthw: mx.array, b_ncthw: mx.array, blend_extent: int) -> mx.array:
    """Horizontal blend for tiled decode (diffusers ``blend_h``)."""
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


def _decode_latent_volume_temporal(
    ctx: RuntimeContext,
    dec: HunyuanVideo15Decoder3DMlx,
    latents: mx.array,
    *,
    temporal_chunk_size: int,
    tcr: int,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> mx.array:
    """Decode ``[B,C,T,H,W]`` with optional causal temporal chunking."""
    _, _, t, _, _ = latents.shape
    chunk = int(temporal_chunk_size or 0)
    if chunk > 0 and t > chunk:
        if on_log is not None:
            on_log(f"HunyuanVideo VAE temporal chunked decode (T={t}, chunk={chunk})")
        parts: list[mx.array] = []
        starts = list(range(0, t, chunk))
        for start in starts:
            end = min(start + chunk, t)
            if start > 0:
                z_slice = latents[:, :, start - 1 : end, :, :]
            else:
                z_slice = latents[:, :, start:end, :, :]
            piece = dec(z_slice, on_stage=None)
            ctx.eval(piece)
            if start > 0:
                piece = piece[:, :, tcr:, :, :]
            parts.append(piece)
            if hasattr(ctx, "clear_cache"):
                ctx.clear_cache()
        return mx.concatenate(parts, axis=2)

    sample = dec(latents, on_stage=on_stage)
    ctx.eval(sample)
    return sample


def _tiled_spatial_decode_hunyuan(
    ctx: RuntimeContext,
    dec: HunyuanVideo15Decoder3DMlx,
    latents: mx.array,
    params: _HunyuanVaeTileParams,
    *,
    temporal_chunk_size: int,
    tcr: int,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> mx.array:
    """Spatial tiling + optional temporal chunking (diffusers ``tiled_decode``)."""
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
                f"HunyuanVideo VAE spatial tile {tile_idx}/{total_tiles} "
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
            decoded = _decode_latent_volume_temporal(
                ctx,
                dec,
                z_tile,
                temporal_chunk_size=temporal_chunk_size,
                tcr=tcr,
                on_log=on_log,
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
                blended = _blend_v(rows[ri - 1][ci], blended, blend_height)
            if ci > 0:
                blended = _blend_h(row[ci - 1], blended, blend_width)
            out_h = int(blended.shape[3])
            out_w = int(blended.shape[4])
            rh = min(row_limit_height, out_h)
            rw = min(row_limit_width, out_w)
            result_row.append(blended[:, :, :, :rh, :rw])
        result_rows.append(mx.concatenate(result_row, axis=4))

    if on_stage is not None:
        on_stage(0.98)
    return mx.concatenate(result_rows, axis=3)


def decode_latents_ncthw(
    ctx: RuntimeContext,
    latents_bcthw: mx.array,
    bundle_root: Path,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    temporal_chunk_size: int = 0,
    spatial_tiling: bool = False,
) -> mx.array:
    """Decode ``latents_bcthw`` ``[B,C,T,H,W]`` to pixels ``[B,C,T,H,W]`` float in ``[-1, 1]``.

    When ``spatial_tiling`` is enabled and latent H/W exceed diffusers tile mins (256px → 16 latent
    at ``spatial_compression_ratio=16``), decode in overlapping spatial tiles with blend stitching.

    When ``temporal_chunk_size > 0`` and ``T`` exceeds it, decode in causal temporal chunks
    to reduce peak unified memory (overlap = 1 latent frame for 3×3×3 causal conv).
    """
    vae_cfg = _read_vae_config(bundle_root)
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.03682))
    latents = latents_bcthw / scaling_factor

    logger.info("HunyuanVideo VAE decode: latent shape=%s", tuple(latents.shape))
    if on_log is not None:
        on_log(f"HunyuanVideo VAE decode start (latent shape {tuple(latents.shape)})")

    dec = _get_hunyuan_decoder(
        bundle_root, load_fn=getattr(ctx, "load_weights", None)
    )
    ctx.eval(latents)
    if on_stage is not None:
        on_stage(0.02)

    b, c, t, h, w = latents.shape
    chunk = int(temporal_chunk_size or 0)
    tcr = int(vae_cfg.get("temporal_compression_ratio", 4) or 4)
    tile_params = _hunyuan_vae_tile_params(vae_cfg)
    use_spatial = _needs_hunyuan_spatial_tiling(spatial_tiling, int(h), int(w), tile_params)

    if use_spatial:
        if on_log is not None:
            on_log(
                f"HunyuanVideo VAE spatial tiled decode "
                f"(latent {w}x{h}, tile {tile_params.tile_latent_min_width}x"
                f"{tile_params.tile_latent_min_height}, overlap={tile_params.overlap_factor})"
            )
        sample = _tiled_spatial_decode_hunyuan(
            ctx,
            dec,
            latents,
            tile_params,
            temporal_chunk_size=chunk,
            tcr=tcr,
            on_stage=on_stage,
            on_log=on_log,
        )
    else:
        sample = _decode_latent_volume_temporal(
            ctx,
            dec,
            latents,
            temporal_chunk_size=chunk,
            tcr=tcr,
            on_stage=on_stage,
            on_log=on_log,
        )
        if chunk > 0 and t > chunk and on_stage is not None:
            on_stage(0.98)

    sample = mx.clip(sample, -1.0, 1.0)
    ctx.eval(sample)
    if on_stage is not None:
        on_stage(1.0)
    logger.info("HunyuanVideo VAE decode done: pixel shape=%s", tuple(sample.shape))
    return sample


def encode_video_ncthw(
    ctx: RuntimeContext,
    pixels_bcthw: mx.array,
    bundle_root: Path,
    on_log: Callable[[str], None] | None = None,
) -> mx.array:
    """Encode RGB video ``[B,C,T,H,W]`` float ``[-1,1]`` to latents ``[B,C,T,H,W]`` (Gaussian mode × scaling)."""
    vae_cfg = _read_vae_config(bundle_root)
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.03682))

    logger.info("HunyuanVideo VAE encode: pixel shape=%s", tuple(pixels_bcthw.shape))
    if on_log is not None:
        on_log(f"HunyuanVideo VAE encode start (pixel shape {tuple(pixels_bcthw.shape)})")

    enc = _get_hunyuan_encoder(
        bundle_root, load_fn=getattr(ctx, "load_weights", None)
    )
    ctx.eval(pixels_bcthw)
    moments = enc(pixels_bcthw)
    latents = _gaussian_mode(moments) * scaling_factor
    ctx.eval(latents)
    logger.info("HunyuanVideo VAE encode done: latent shape=%s", tuple(latents.shape))
    return latents
