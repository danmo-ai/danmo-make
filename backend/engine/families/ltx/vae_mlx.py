"""LTX 2.3 video/audio codec — MLX implementation (in-repo)."""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval
from backend.engine.runtime._base import RuntimeContext

logger = logging.getLogger(__name__)


def _eval(*vals: Any) -> None:
    run_eval(None, *vals)


def _find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("ffmpeg not found on PATH; install with: brew install ffmpeg")
    return path


def _load_bundle_weights(bundle_root: Path, filename: str, prefix: str, load_fn: Any | None) -> dict[str, mx.array]:
    path = bundle_root / filename
    if not path.is_file():
        raise RuntimeError(f"LTX 2.3 bundle file missing: {path}")
    raw = load_weights_dict(load_fn, str(path))
    if not prefix:
        return dict(raw)
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in raw.items()}


def _remap_encoder_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v for k, v in weights.items()}


def _remap_audio_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    return {k.replace("._mean_of_means", ".mean_of_means").replace("._std_of_means", ".std_of_means"): v for k, v in weights.items()}


# --- normalization.py ---
import mlx.core as mx


def pixel_norm(x: mx.array, eps: float = 1e-8) -> mx.array:
    """PixelNorm: x / sqrt(mean(x^2, dim=channels) + eps).

    No learnable parameters -- matches the reference VAE's PixelNorm.
    Applied per-pixel across the channel dimension (last dim in BFHWC).
    """
    return mx.fast.rms_norm(x, weight=None, eps=eps)

# --- convolution.py ---
import mlx.core as mx
import mlx.nn as nn


class Conv3dBlock(nn.Module):
    """3D convolution with causal or non-causal temporal padding.

    When causal=True: replicates first frame for temporal padding (front only).
    When causal=False: standard symmetric zero-padding on all dimensions
    (matching reference ``make_conv_nd`` with ``causal=False``).

    MLX Conv3d weight layout: (O, D, H, W, I)
    Produces key: ``conv.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 1,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.spatial_padding = (padding[1], padding[2])
        self.spatial_padding_mode = spatial_padding_mode

        # Reference CausalConv3d: always created the same way, with causal
        # flag controlling runtime padding behavior. We handle padding manually
        # so Conv3d always gets padding=0.
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: x is (B, D, H, W, C) in MLX convention."""
        tk = self.kernel_size[0]
        if self.causal:
            # Causal: replicate first frame (kernel_size-1) times at front only
            if tk > 1:
                first_frame = mx.repeat(x[:, :1, :, :, :], tk - 1, axis=1)
                x = mx.concatenate([first_frame, x], axis=1)
        else:
            # Non-causal: symmetric replicate padding (first AND last frame)
            # Reference: first_frame_pad repeats (kernel_size-1)//2 times at front,
            #            last_frame_pad repeats (kernel_size-1)//2 times at back.
            pad_size = (tk - 1) // 2
            if pad_size > 0:
                first_pad = mx.repeat(x[:, :1, :, :, :], pad_size, axis=1)
                last_pad = mx.repeat(x[:, -1:, :, :, :], pad_size, axis=1)
                x = mx.concatenate([first_pad, x, last_pad], axis=1)

        # Spatial padding (always symmetric)
        sp_h, sp_w = self.spatial_padding
        if sp_h > 0 or sp_w > 0:
            if self.spatial_padding_mode == "reflect":
                # Manual reflect padding for BDHWC layout.
                # For pad size p, reflect takes pixels [1..p] and [-(1+p)..-1].
                if sp_h > 0:
                    x = mx.concatenate(
                        [x[:, :, 1 : 1 + sp_h, :, :], x, x[:, :, -(1 + sp_h) : -1, :, :]],
                        axis=2,
                    )
                if sp_w > 0:
                    x = mx.concatenate(
                        [x[:, :, :, 1 : 1 + sp_w, :], x, x[:, :, :, -(1 + sp_w) : -1, :]],
                        axis=3,
                    )
            else:
                x = mx.pad(x, [(0, 0), (0, 0), (sp_h, sp_h), (sp_w, sp_w), (0, 0)])

        return self.conv(x)

# --- ops.py ---
import mlx.core as mx
import mlx.nn as nn


class PerChannelStatistics(nn.Module):
    """Stores per-channel mean and std for latent (de)normalization.

    Produces keys: ``mean``, ``std``
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mean = mx.zeros((channels,))
        self.std = mx.ones((channels,))


class EncoderPerChannelStatistics(nn.Module):
    """Stores per-channel normalization stats for the encoder.

    Weight file keys use underscore prefix (``_mean_of_means``, ``_std_of_means``)
    but MLX nn.Module ignores underscore-prefixed attributes. We store them
    without the underscore and remap during weight loading.

    Produces keys: ``mean_of_means``, ``std_of_means``
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mean_of_means = mx.zeros((channels,))
        self.std_of_means = mx.ones((channels,))


# Weight key remapping: safetensors key -> module key
_ENCODER_KEY_REMAP: dict[str, str] = {
    "per_channel_statistics._mean_of_means": "per_channel_statistics.mean_of_means",
    "per_channel_statistics._std_of_means": "per_channel_statistics.std_of_means",
}


def remap_encoder_weight_keys(
    weights: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Remap weight keys from safetensors format to module format.

    Handles the underscore-prefixed ``_mean_of_means`` / ``_std_of_means`` keys
    in the encoder's per-channel statistics.

    Args:
        weights: Weight dict with keys from the safetensors file.

    Returns:
        New dict with remapped keys.
    """
    remapped: dict[str, mx.array] = {}
    for k, v in weights.items():
        new_key = _ENCODER_KEY_REMAP.get(k, k)
        remapped[new_key] = v
    return remapped

# --- resnet.py ---


class ResBlock3d(nn.Module):
    """Pre-activation residual block: PixelNorm -> SiLU -> Conv.

    Reference: ltx-core ResnetBlock3D with NormLayerType.PIXEL_NORM.
    PixelNorm is parameterless so no norm weights in safetensors.

    Forward: norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 + skip
    Produces keys: ``conv1.conv.{weight,bias}``, ``conv2.conv.{weight,bias}``
    """

    def __init__(self, channels: int, causal: bool = True, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.conv1 = Conv3dBlock(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = Conv3dBlock(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(nn.silu(pixel_norm(x)))
        x = self.conv2(nn.silu(pixel_norm(x)))
        return x + residual


class ResBlockStage(nn.Module):
    """A stage of N residual blocks at a fixed channel count.

    Produces keys: ``res_blocks.{i}.conv{1,2}.conv.{weight,bias}``
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.res_blocks = [
            ResBlock3d(channels, causal=causal, spatial_padding_mode=spatial_padding_mode) for _ in range(num_blocks)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.res_blocks:
            x = block(x)
        return x

# --- sampling.py ---

# ---------------------------------------------------------------------------
# Pixel-shuffle / space-to-depth helpers
# ---------------------------------------------------------------------------


def pixel_shuffle_3d(
    x: mx.array,
    spatial_factor: int,
    temporal_factor: int,
) -> mx.array:
    """Rearrange channels into spatial/temporal dimensions (depth-to-space).

    Matches: ``"b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)"``
    Channel split order: (c, p1=temporal, p2=height, p3=width) — c outermost.
    In BDHWC layout: C_total = C * tf * sf * sf where C varies slowest.

    Input:  (B, D, H, W, C * sf^2 * tf)
    Output: (B, D*tf, H*sf, W*sf, C)
    """
    B, D, H, W, C_total = x.shape
    C = C_total // (spatial_factor * spatial_factor * temporal_factor)
    x = x.reshape(B, D, H, W, C, temporal_factor, spatial_factor, spatial_factor)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, D * temporal_factor, H * spatial_factor, W * spatial_factor, C)
    return x


def unpatchify_spatial(
    x: mx.array,
    patch_size: int,
) -> mx.array:
    """Reverse spatial patchification: depth-to-space for the final VAE output.

    Matches the reference ``unpatchify`` from ltx-core ops.py:
        ``"b (c p r q) f h w -> b c (f p) (h q) (w r)"``
    with ``p=1, q=patch_size, r=patch_size``.

    Channel split order: (c, p=1, r=width, q=height) — note r (width) comes
    BEFORE q (height). This differs from ``pixel_shuffle_3d`` which uses
    (c, temporal, height, width). Using ``pixel_shuffle_3d`` for unpatchify
    swaps H/W sub-pixels and causes checkerboard artifacts.

    Input:  (B, F, H, W, C * patch_size^2)   (BFHWC, temporal patch=1)
    Output: (B, F, H*patch_size, W*patch_size, C)
    """
    B, F, H, W, C_total = x.shape
    ps = patch_size
    C = C_total // (ps * ps)
    # Split channels as (C, r_width, q_height) matching reference (c, p=1, r, q)
    x = x.reshape(B, F, H, W, C, ps, ps)
    # Indices: B=0, F=1, H=2, W=3, C=4, r_W=5, q_H=6
    # Target:  (B, F, H, q_H, W, r_W, C) -> (B, F, H*ps, W*ps, C)
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    x = x.reshape(B, F, H * ps, W * ps, C)
    return x


def patchify_spatial(x: mx.array, patch_size: int = 4) -> mx.array:
    """Spatial patchification: space-to-depth rearrangement.

    Reference: ltx-core ops.py patchify with patch_size_hw=4, patch_size_t=1.
    einops: ``"b c (f p) (h q) (w r) -> b (c p r q) f h w"`` with p=1, q=4, r=4.

    Channel ordering: (c, p=1, r=patch_W, q=patch_H) -- c outermost.
    In BFHWC layout: splits H and W into patches, packs into channels.

    Args:
        x: (B, F, H, W, C) in BFHWC layout, C=3.
        patch_size: Spatial patch size (default 4).

    Returns:
        (B, F, H/ps, W/ps, C * ps * ps) in BFHWC layout.
    """
    B, F, H, W, C = x.shape
    ps = patch_size
    # Split spatial dims: (B, F, H//ps, ps_h, W//ps, ps_w, C)
    x = x.reshape(B, F, H // ps, ps, W // ps, ps, C)
    # Rearrange to channel order (C, r_W, q_H) matching reference (c, p, r, q)
    # From indices: 0=B, 1=F, 2=H//ps, 3=q_H, 4=W//ps, 5=r_W, 6=C
    # To: (B, F, H//ps, W//ps, C, r_W, q_H)
    x = x.transpose(0, 1, 2, 4, 6, 5, 3)
    return x.reshape(B, F, H // ps, W // ps, C * ps * ps)


def space_to_depth(
    x: mx.array,
    stride: tuple[int, int, int],
) -> mx.array:
    """Space-to-depth rearrangement for downsampling.

    Reference einops: ``"b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w"``
    Channel ordering: (c, p1=temporal, p2=H, p3=W) -- c outermost, p3 innermost.

    In BDHWC layout, produces (B, D, H, W, C * prod(stride)).

    Args:
        x: (B, D_full, H_full, W_full, C) in BDHWC layout.
        stride: (temporal_stride, height_stride, width_stride).

    Returns:
        (B, D, H, W, C * prod(stride)) with reference channel ordering.
    """
    B, D_full, H_full, W_full, C = x.shape
    st, sh, sw = stride
    D = D_full // st
    H = H_full // sh
    W = W_full // sw
    # Split dims: (B, D, st, H, sh, W, sw, C)
    x = x.reshape(B, D, st, H, sh, W, sw, C)
    # Rearrange to (B, D, H, W, C, st, sh, sw) -- C outermost in channel group
    x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(B, D, H, W, C * st * sh * sw)


# ---------------------------------------------------------------------------
# Upsample / downsample modules
# ---------------------------------------------------------------------------


class DepthToSpaceUpsample(nn.Module):
    """Convolution used as an upsample layer (depth-to-space).

    The conv may output more channels than the input (for pixel-shuffle
    spatial/temporal upsampling). The caller handles the rearrangement.

    Produces key: ``conv.conv.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = Conv3dBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class SpaceToDepthDownsample(nn.Module):
    """Downsampling via space-to-depth with group-mean skip connection.

    Reference: ltx-core sampling.py SpaceToDepthDownsample.

    Two-branch architecture:
      - Skip: space-to-depth rearrange -> group channels -> mean -> out_channels
      - Conv: conv3d -> space-to-depth rearrange -> out_channels
      - Output = conv + skip

    Produces key: ``conv.conv.{weight,bias}``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
    ):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * int.__mul__(*stride[1:]) * stride[0] // out_channels
        conv_out_ch = out_channels // (stride[0] * stride[1] * stride[2])
        self.conv = Conv3dBlock(
            in_channels,
            conv_out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,  # Encoder always uses causal convolutions
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Forward: x is (B, D, H, W, C) in MLX convention."""
        # Prepend first frame for temporal downsampling (causal padding)
        if self.stride[0] == 2:
            x = mx.concatenate([x[:, :1, :, :, :], x], axis=1)

        # Skip connection: space-to-depth -> group-mean
        x_in = space_to_depth(x, self.stride)
        if self.group_size > 1:
            B, D, H, W, C_total = x_in.shape
            C_out = C_total // self.group_size
            x_in = x_in.reshape(B, D, H, W, C_out, self.group_size)
            x_in = x_in.mean(axis=-1)

        # Conv branch: conv -> space-to-depth
        x_conv = self.conv(x)
        x_conv = space_to_depth(x_conv, self.stride)

        return x_conv + x_in

# --- video_vae ---

import itertools
import os
from dataclasses import dataclass, replace
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


# --- tiling (VAE decode temporal/spatial blend) ---


def _compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    *,
    left_starts_from_0: bool = False,
) -> mx.array:
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))
    mask = mx.ones(length)
    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = mx.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask = mx.concatenate([mask[:ramp_left] * fade_in, mask[ramp_left:]])
    if ramp_right > 0:
        fade_out = mx.linspace(1.0, 0.0, ramp_right + 2)[1:-1]
        mask = mx.concatenate([mask[:-ramp_right], mask[-ramp_right:] * fade_out])
    return mx.clip(mask, 0.0, 1.0)


def _compute_rectangular_mask_1d(length: int, left_ramp: int, right_ramp: int) -> mx.array:
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    mask = mx.ones(length)
    if left_ramp > 0:
        mask = mx.concatenate([mx.zeros(left_ramp), mask[left_ramp:]])
    if right_ramp > 0:
        mask = mx.concatenate([mask[:-right_ramp], mx.zeros(right_ramp)])
    return mask


@dataclass(frozen=True)
class _SpatialTilingConfig:
    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0


@dataclass(frozen=True)
class _TemporalTilingConfig:
    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0


@dataclass(frozen=True)
class _TilingConfig:
    spatial_config: _SpatialTilingConfig | None = None
    temporal_config: _TemporalTilingConfig | None = None


@dataclass(frozen=True)
class _DimensionIntervals:
    starts: list[int]
    ends: list[int]
    left_ramps: list[int]
    right_ramps: list[int]


class _Tile(NamedTuple):
    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[mx.array | None, ...]

    @property
    def blend_mask(self) -> mx.array:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[mx.array] = []
        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                view_shape[dim_idx] = 1
                per_dimension_masks.append(mx.ones(1).reshape(*view_shape))
                continue
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.reshape(*view_shape))
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask
        return combined_mask


_SplitOperation = Callable[[int], _DimensionIntervals]
_MappingOperation = Callable[[_DimensionIntervals], tuple[list[slice], list[mx.array | None]]]


def _default_split_operation(length: int) -> _DimensionIntervals:
    return _DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


def _default_mapping_operation(
    _intervals: _DimensionIntervals,
) -> tuple[list[slice], list[mx.array | None]]:
    return [slice(0, None)], [None]


def _create_tiles_from_intervals_and_mappers(
    original_shape: tuple[int, ...],
    dimension_intervals: tuple[_DimensionIntervals, ...],
    mappers: list[_MappingOperation],
) -> list[_Tile]:
    full_dim_input_slices: list[list[slice]] = []
    full_dim_output_slices: list[list[slice]] = []
    full_dim_masks_1d: list[list[mx.array | None]] = []
    for axis_index in range(len(original_shape)):
        dim_intervals = dimension_intervals[axis_index]
        input_slices = [slice(s, e) for s, e in zip(dim_intervals.starts, dim_intervals.ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dim_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)
    tiles: list[_Tile] = []
    for in_coord, out_coord, mask_1d in zip(
        itertools.product(*full_dim_input_slices),
        itertools.product(*full_dim_output_slices),
        itertools.product(*full_dim_masks_1d),
        strict=True,
    ):
        tiles.append(_Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d))
    return tiles


def _create_tiles(
    tensor_shape: tuple[int, ...],
    splitters: list[_SplitOperation],
    mappers: list[_MappingOperation],
) -> list[_Tile]:
    intervals = tuple(splitter(length) for splitter, length in zip(splitters, tensor_shape, strict=True))
    return _create_tiles_from_intervals_and_mappers(tensor_shape, intervals, mappers)


def _split_with_symmetric_overlaps(size: int, overlap: int) -> _SplitOperation:
    def split(dimension_size: int) -> _DimensionIntervals:
        if dimension_size <= size:
            return _default_split_operation(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return _DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def _split_temporal_latents(size: int, overlap: int) -> _SplitOperation:
    non_causal_split = _split_with_symmetric_overlaps(size, overlap)

    def split(dimension_size: int) -> _DimensionIntervals:
        if dimension_size <= size:
            return _default_split_operation(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = list(intervals.starts)
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = list(intervals.left_ramps)
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def _make_mapping_operation(
    map_func: Callable[[int, int, int, int, int], tuple[slice, mx.array | None]],
    scale: int,
) -> _MappingOperation:
    def map_op(intervals: _DimensionIntervals) -> tuple[list[slice], list[mx.array | None]]:
        output_slices: list[slice] = []
        masks_1d: list[mx.array | None] = []
        for i in range(len(intervals.starts)):
            output_slice, mask_1d = map_func(
                intervals.starts[i],
                intervals.ends[i],
                intervals.left_ramps[i],
                intervals.right_ramps[i],
                scale,
            )
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def _map_temporal_interval_to_frame(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, mx.array]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp_frames = 0 if left_ramp == 0 else 1 + (left_ramp - 1) * scale
    right_ramp_frames = right_ramp * scale
    mask_1d = _compute_trapezoidal_mask_1d(
        stop - start, left_ramp_frames, right_ramp_frames, left_starts_from_0=True
    )
    return slice(start, stop), mask_1d


def _map_spatial_interval_to_pixel(
    begin: int,
    end: int,
    left_ramp: int,
    right_ramp: int,
    scale: int,
) -> tuple[slice, mx.array]:
    start = begin * scale
    stop = end * scale
    mask_1d = _compute_trapezoidal_mask_1d(stop - start, left_ramp * scale, right_ramp * scale)
    return slice(start, stop), mask_1d


_SCALE_TIME = 8
_SCALE_HEIGHT = 32
_SCALE_WIDTH = 32


def _prepare_tiles_for_decoding(
    latent_shape: tuple[int, ...],
    tiling_config: _TilingConfig | None = None,
) -> list[_Tile]:
    ndim = len(latent_shape)
    splitters: list[_SplitOperation] = [_default_split_operation] * ndim
    mappers: list[_MappingOperation] = [_default_mapping_operation] * ndim
    if tiling_config is not None and tiling_config.spatial_config is not None:
        cfg = tiling_config.spatial_config
        long_side = max(latent_shape[3], latent_shape[4])

        def _enable_spatial_axis(axis_idx: int, factor: int) -> None:
            tile_size = cfg.tile_size_in_pixels // factor
            overlap = cfg.tile_overlap_in_pixels // factor
            axis_length = latent_shape[axis_idx]
            lower_threshold = max(2, overlap + 1)
            adjusted_size = max(lower_threshold, round(tile_size * axis_length / long_side))
            splitters[axis_idx] = _split_with_symmetric_overlaps(adjusted_size, overlap)
            mappers[axis_idx] = _make_mapping_operation(_map_spatial_interval_to_pixel, scale=factor)

        _enable_spatial_axis(3, _SCALE_HEIGHT)
        _enable_spatial_axis(4, _SCALE_WIDTH)
    if tiling_config is not None and tiling_config.temporal_config is not None:
        cfg = tiling_config.temporal_config
        tile_size = cfg.tile_size_in_frames // _SCALE_TIME
        overlap = cfg.tile_overlap_in_frames // _SCALE_TIME
        splitters[2] = _split_temporal_latents(tile_size, overlap)
        mappers[2] = _make_mapping_operation(_map_temporal_interval_to_frame, scale=_SCALE_TIME)
    return _create_tiles(latent_shape, splitters, mappers)


def _compute_decode_tiling(
    latent_shape: tuple[int, ...],
    frame_rate: float = 24.0,
) -> _TilingConfig | None:
    """Return tiling config when full VAE decode exceeds memory budget, else None."""
    peak_budget_gb = float(os.environ.get("LTX2_VAE_DECODE_BUDGET_GB", "8.0"))
    _, _, f_lat, h_lat, w_lat = latent_shape
    budget_bytes = int(peak_budget_gb * 1024**3)
    block3_bytes_per_lat_frame = 512 * 4 * (h_lat * 4) * (w_lat * 4) * 2
    if block3_bytes_per_lat_frame * f_lat <= budget_bytes:
        return None
    max_lat_frames = max(2, budget_bytes // block3_bytes_per_lat_frame)
    tile_frames = max(16, max_lat_frames * 8)
    one_second_frames = max(8, (int(frame_rate) // 8) * 8)
    overlap = min(one_second_frames, (tile_frames // 32) * 8)
    if overlap >= tile_frames:
        raise RuntimeError(f"LTX VAE decode tiling overlap {overlap} >= tile_frames {tile_frames}")
    return _TilingConfig(
        temporal_config=_TemporalTilingConfig(
            tile_size_in_frames=tile_frames,
            tile_overlap_in_frames=overlap,
        )
    )


def _add_at(buffer: mx.array, coords: tuple[slice, ...], values: mx.array) -> mx.array:
    """Add values into buffer at the given slice coordinates.

    MLX arrays are immutable, so we use slice assignment via __setitem__
    on a copy. In practice MLX handles this efficiently.
    """
    # MLX supports in-place-style slice assignment that returns a new array
    buffer[coords] = buffer[coords] + values
    return buffer


def _group_tiles_by_temporal_slice(tiles: list[_Tile]) -> list[list[_Tile]]:
    """Group tiles by their temporal output slice."""
    if not tiles:
        return []

    groups: list[list[_Tile]] = []
    current_slice = tiles[0].out_coords[2]
    current_group: list[_Tile] = []

    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]

    if current_group:
        groups.append(current_group)

    return groups


class LTX23VideoDecoder(nn.Module):
    """Video VAE Decoder with streaming frame output.

    Decodes latent (B, C, F', H', W') to pixels, streaming frames
    to ffmpeg for memory efficiency.

    Architecture matches the weight file exactly:
        conv_in -> up_blocks (alternating ResStage / DepthToSpaceUpsample) -> conv_out

    up_blocks layout:
        0: ResStage  1024, 2 blocks
        1: DepthToSpaceUpsample 1024 -> 4096  (pixel-shuffle 2xspatial + 2xtemporal -> 512ch)
        2: ResStage  512,  2 blocks
        3: DepthToSpaceUpsample 512 -> 4096   (pixel-shuffle 2xspatial + 2xtemporal -> 512ch)
        4: ResStage  512,  4 blocks
        5: DepthToSpaceUpsample 512 -> 512    (pixel-shuffle 2xtemporal -> 256ch)
        6: ResStage  256,  6 blocks
        7: DepthToSpaceUpsample 256 -> 512    (pixel-shuffle 2xspatial -> 128ch)
        8: ResStage  128,  4 blocks

    Args:
        causal: If True, uses causal temporal padding (replicate first frame,
            remove first frame after temporal upsample). If False (LTX-2.3
            default), uses symmetric zero-padding and no frame removal.
    """

    def __init__(self, causal: bool = False, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self._causal = causal

        # LTX-2.3 model was trained with zero padding (per embedded_config.json
        # "spatial_padding_mode": "zeros"). Previously hardcoded "reflect" which
        # caused cumulative temporal divergence in decoder forward (visible as
        # the keyframe hold-cut-decay regression at the latent boundary).
        sp_mode = spatial_padding_mode

        # Input convolution: 128 latent channels -> 1024
        self.conv_in = Conv3dBlock(
            128,
            1024,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=sp_mode,
        )

        # Flat list of up_blocks -- indices must match weight keys exactly.
        self.up_blocks: list[Any] = [
            ResBlockStage(1024, num_blocks=2, causal=causal, spatial_padding_mode=sp_mode),  # 0
            DepthToSpaceUpsample(1024, 4096, causal=causal, spatial_padding_mode=sp_mode),  # 1
            ResBlockStage(512, num_blocks=2, causal=causal, spatial_padding_mode=sp_mode),  # 2
            DepthToSpaceUpsample(512, 4096, causal=causal, spatial_padding_mode=sp_mode),  # 3
            ResBlockStage(512, num_blocks=4, causal=causal, spatial_padding_mode=sp_mode),  # 4
            DepthToSpaceUpsample(512, 512, causal=causal, spatial_padding_mode=sp_mode),  # 5
            ResBlockStage(256, num_blocks=6, causal=causal, spatial_padding_mode=sp_mode),  # 6
            DepthToSpaceUpsample(256, 512, causal=causal, spatial_padding_mode=sp_mode),  # 7
            ResBlockStage(128, num_blocks=4, causal=causal, spatial_padding_mode=sp_mode),  # 8
        ]

        # Output convolution: 128 -> 48 (3 RGB x 16 for spatial pixel shuffle)
        self.conv_out = Conv3dBlock(
            128,
            48,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=sp_mode,
        )

        # Per-channel normalization statistics
        self.per_channel_statistics = PerChannelStatistics(128)

        # Upsample config: (spatial_factor, temporal_factor) per DepthToSpaceUpsample
        # up_blocks indices 1, 3, 5, 7
        self._upsample_config: list[tuple[int, int]] = [
            (2, 2),  # block 1: 4096 / (2*2*2) = 512
            (2, 2),  # block 3: 4096 / (2*2*2) = 512
            (1, 2),  # block 5: 512 / (1*1*2) = 256
            (2, 1),  # block 7: 512 / (2*2*1) = 128
        ]

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        """Reverse per-channel normalization: x * std + mean.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Denormalized latent.
        """
        mean = self.per_channel_statistics.mean.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def decode(self, latent: mx.array, *, _materialize_stages: bool = False) -> mx.array:
        """Decode latent to pixel frames.

        Args:
            latent: (B, C, F, H, W) latent in PyTorch layout.
            _materialize_stages: If True, force-eval after each upsample stage so
                prior activations can be freed before the next (larger) stage begins.
                Only set by :meth:`tiled_decode`; the no-tiling path omits this to
                avoid breaking kernel fusion across upsample stages.

        Returns:
            Pixels (B, 3, F, H, W) in [-1, 1], same dtype as ``latent``.
        """
        # Cast input to weights dtype and remember caller dtype to restore on
        # return. Matches Lightricks/LTX-2 PR #179 commit b604d3f — defensive
        # guard against dtype mismatch between the caller and weights.
        output_dtype = latent.dtype
        flat_params = mlx.utils.tree_flatten(self.parameters())
        weights_dtype = flat_params[0][1].dtype if flat_params else output_dtype
        if latent.dtype != weights_dtype:
            latent = latent.astype(weights_dtype)

        # Convert BCFHW -> BFHWC for MLX convolutions
        x = latent.transpose(0, 2, 3, 4, 1)
        x = self.denormalize_latent(x)

        x = self.conv_in(x)

        upsample_idx = 0
        for i, block in enumerate(self.up_blocks):
            x = block(x)

            # Apply pixel shuffle after each DepthToSpaceUpsample (odd indices)
            if i % 2 == 1:
                sf, tf = self._upsample_config[upsample_idx]
                x = pixel_shuffle_3d(x, spatial_factor=sf, temporal_factor=tf)
                # Reference: ALWAYS remove first frame after temporal upsample
                # (unconditional on causal mode, gated on stride[0]==2 only)
                if tf > 1:
                    x = x[:, 1:, :, :, :]
                upsample_idx += 1
                if _materialize_stages:
                    # Free prior-stage activations before the next, larger stage.
                    mx.eval(x)

        # Pre-activation PixelNorm + SiLU before final conv
        x = self.conv_out(nn.silu(pixel_norm(x)))

        # Final spatial unpatchify: 48 -> 3 channels, 4x spatial expansion.
        # Uses unpatchify_spatial (not pixel_shuffle_3d) because the reference
        # unpatchify has channel order (c, p, r_W, q_H) — width factor before
        # height factor — which differs from DepthToSpaceUpsample's (c, p1, p2_H, p3_W).
        x = unpatchify_spatial(x, patch_size=4)

        # BFHWC -> BCFHW, restored to caller's dtype.
        return x.transpose(0, 4, 1, 2, 3).astype(output_dtype)

    def tiled_decode(
        self,
        latent: mx.array,
        tiling_config: _TilingConfig | None = None,
    ) -> Iterator[mx.array]:
        if tiling_config is None:
            pixels = self.decode(latent)
            _eval(pixels)
            yield pixels
            return

        tiles = _prepare_tiles_for_decoding(latent.shape, tiling_config)
        temporal_groups = _group_tiles_by_temporal_slice(tiles)
        _, _, _f_lat, h_lat, w_lat = latent.shape
        out_h = h_lat * 32
        out_w = w_lat * 32

        previous_chunk: mx.array | None = None
        previous_weights: mx.array | None = None
        previous_temporal_slice: slice | None = None

        for temporal_group_tiles in temporal_groups:
            curr_temporal_slice = temporal_group_tiles[0].out_coords[2]
            temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
            buffer = mx.zeros((latent.shape[0], 3, temporal_len, out_h, out_w))
            weights = mx.zeros_like(buffer)

            for tile in temporal_group_tiles:
                decoded_tile = self.decode(latent[tile.in_coords], _materialize_stages=True)
                _eval(decoded_tile)
                mask = tile.blend_mask
                temporal_offset = tile.out_coords[2].start - curr_temporal_slice.start
                expected_temporal_len = tile.out_coords[2].stop - tile.out_coords[2].start
                decoded_temporal_len = decoded_tile.shape[2]
                actual_temporal_len = min(
                    expected_temporal_len, decoded_temporal_len, buffer.shape[2] - temporal_offset
                )
                chunk_coords = (
                    slice(None),
                    slice(None),
                    slice(temporal_offset, temporal_offset + actual_temporal_len),
                    tile.out_coords[3],
                    tile.out_coords[4],
                )
                decoded_slice = decoded_tile[:, :, :actual_temporal_len, :, :]
                mask_slice = mask[:, :, :actual_temporal_len, :, :] if mask.shape[2] > 1 else mask
                buffer = _add_at(buffer, chunk_coords, decoded_slice * mask_slice)
                weights = _add_at(weights, chunk_coords, mask_slice)
                _eval(buffer, weights)
                del decoded_tile, mask, decoded_slice, mask_slice

            if previous_chunk is not None and previous_temporal_slice is not None:
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    prev_overlap_start = curr_temporal_slice.start - previous_temporal_slice.start
                    prev_overlap = previous_chunk[:, :, prev_overlap_start:, :, :]
                    prev_w_overlap = previous_weights[:, :, prev_overlap_start:, :, :]
                    curr_overlap = buffer[:, :, :overlap_len, :, :]
                    curr_w_overlap = weights[:, :, :overlap_len, :, :]
                    merged = prev_overlap + curr_overlap
                    merged_w = prev_w_overlap + curr_w_overlap
                    previous_chunk = mx.concatenate(
                        [previous_chunk[:, :, :prev_overlap_start, :, :], merged], axis=2
                    )
                    previous_weights = mx.concatenate(
                        [previous_weights[:, :, :prev_overlap_start, :, :], merged_w], axis=2
                    )
                    buffer = mx.concatenate([merged, buffer[:, :, overlap_len:, :, :]], axis=2)
                    weights = mx.concatenate([merged_w, weights[:, :, overlap_len:, :, :]], axis=2)
                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                if yield_len > 0:
                    safe_weights = mx.maximum(previous_weights, 1e-8)
                    chunk = (previous_chunk / safe_weights)[:, :, :yield_len, :, :]
                    _eval(chunk)
                    yield chunk

            previous_chunk = buffer
            previous_weights = weights
            previous_temporal_slice = curr_temporal_slice

        if previous_chunk is not None and previous_weights is not None:
            safe_weights = mx.maximum(previous_weights, 1e-8)
            chunk = previous_chunk / safe_weights
            _eval(chunk)
            yield chunk

    def decode_and_stream(
        self,
        latent: mx.array,
        output_path: str,
        *,
        frame_rate: float,
        audio_path: str | None = None,
    ) -> None:
        """Decode latent and stream frames to ffmpeg.

        Automatically applies temporal tiling when the full-volume decode would
        exceed the memory budget (``LTX2_VAE_DECODE_BUDGET_GB``, default 8 GB).
        Budget is measured against the block-3 bf16 activation
        (512 x 4 x 4H_lat x 4W_lat x 2 bytes per latent frame). At 8 GB:

        - 720p  (H_lat=22, W_lat=40): ~55 MB/frame → tiling at ~47s @25fps
        - 1080p (H_lat=33, W_lat=60): ~124 MB/frame → tiling at ~22s @25fps

        Note: the budget covers the block-3 activation of a single tile decode.
        The tiling accumulation buffer (fp32, B x 3 x T x H x W) and its weights
        twin live on top of this; at 1080p / >100 frames they add several GB.

        Falls through to a single-pass decode with no overhead for shorter clips.

        Args:
            latent: (B, C, F, H, W) latent.
            output_path: Path to output video file.
            frame_rate: Output frames per second.
            audio_path: Optional audio file to mux.
        """
        ffmpeg = _find_ffmpeg()
        tiling = _compute_decode_tiling(latent.shape, frame_rate=frame_rate)
        if tiling is not None and tiling.temporal_config is not None:
            tc = tiling.temporal_config
            logger.info(
                "vae-decode tiled: tile_frames=%d overlap=%d",
                tc.tile_size_in_frames,
                tc.tile_overlap_in_frames,
            )

        # Estimate output dimensions from latent
        _, _, _F_lat, H_lat, W_lat = latent.shape
        out_H = H_lat * 32
        out_W = W_lat * 32

        # Build ffmpeg command
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{out_W}x{out_H}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(frame_rate),
            "-i",
            "-",
        ]
        if audio_path:
            cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])
        cmd.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path])

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None

        frames_written = 0
        pipe_broken = False
        for chunk in self.tiled_decode(latent, tiling):  # (B, 3, T, H, W)
            if pipe_broken:
                break
            num_frames = chunk.shape[2]
            for i in range(num_frames):
                frame = chunk[:, :, i, :, :]
                frame = mx.clip(frame, -1.0, 1.0)
                frame = ((frame + 1.0) * 127.5).astype(mx.uint8)
                frame_hwc = frame[0].transpose(1, 2, 0)  # (H, W, 3)
                mx.eval(frame_hwc)  # required: memoryview races GPU writes without this sync
                try:
                    proc.stdin.write(bytes(memoryview(frame_hwc)))
                except BrokenPipeError:
                    logger.warning(
                        "ffmpeg pipe closed after %d frames (expected %d); output may be truncated",
                        frames_written,
                        latent.shape[2] * 8 - 7,
                    )
                    pipe_broken = True
                    break
                frames_written += 1
                del frame, frame_hwc
                if i % 8 == 0:
                    _eval()
            del chunk
            _eval()
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        _eval()


class LTX23VideoEncoder(nn.Module):
    """Video VAE Encoder.

    Encodes pixel frames (B, 3, F, H, W) to latent (B, C, F', H', W').
    Temporal 8x, spatial 32x compression with 128 latent channels.

    Reference architecture:
        patchify(4x4 spatial) -> conv_in -> down_blocks -> norm+silu -> conv_out

    down_blocks layout (from config encoder_blocks):
        0: ResStage  128,  4 blocks
        1: SpaceToDepthDownsample 128->256, stride=(1,2,2) -- spatial 2x
        2: ResStage  256,  6 blocks
        3: SpaceToDepthDownsample 256->512, stride=(2,1,1) -- temporal 2x
        4: ResStage  512,  4 blocks
        5: SpaceToDepthDownsample 512->1024, stride=(2,2,2) -- all 2x
        6: ResStage  1024, 2 blocks
        7: SpaceToDepthDownsample 1024->1024, stride=(2,2,2) -- all 2x (mult=1)
        8: ResStage  1024, 2 blocks

    Weight loading: use :func:`~ltx_2_mlx.model.video_vae.ops.remap_encoder_weight_keys`
    before calling ``load_weights`` to handle the underscore-prefixed per-channel stats keys.
    """

    def __init__(self):
        super().__init__()

        # Input convolution: 48 channels (3 RGB x 4x4 spatial patchify) -> 128
        self.conv_in = Conv3dBlock(48, 128, kernel_size=3, padding=1, causal=True)

        # Flat list of down_blocks -- indices must match weight keys exactly.
        self.down_blocks: list = [
            ResBlockStage(128, num_blocks=4, causal=True),  # 0
            SpaceToDepthDownsample(128, 256, stride=(1, 2, 2)),  # 1
            ResBlockStage(256, num_blocks=6, causal=True),  # 2
            SpaceToDepthDownsample(256, 512, stride=(2, 1, 1)),  # 3
            ResBlockStage(512, num_blocks=4, causal=True),  # 4
            SpaceToDepthDownsample(512, 1024, stride=(2, 2, 2)),  # 5
            ResBlockStage(1024, num_blocks=2, causal=True),  # 6
            SpaceToDepthDownsample(1024, 1024, stride=(2, 2, 2)),  # 7
            ResBlockStage(1024, num_blocks=2, causal=True),  # 8
        ]

        # Output convolution: 1024 -> 129 channels
        self.conv_out = Conv3dBlock(1024, 129, kernel_size=3, padding=1, causal=True)

        # Per-channel normalization statistics
        self.per_channel_statistics = EncoderPerChannelStatistics(128)

    def normalize_latent(self, latent: mx.array) -> mx.array:
        """Apply per-channel normalization: (x - mean) / std.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Normalized latent.
        """
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, 1, 1, -1)
        return (latent - mean) / std

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        """Reverse per-channel normalization: x * std + mean.

        Used to unwrap encoder normalization before the upsampler (which
        operates in un-normalized space) and re-normalize after.

        Args:
            latent: (B, F, H, W, C) in MLX layout.

        Returns:
            Denormalized latent.
        """
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def encode(self, pixels: mx.array) -> mx.array:
        """Encode pixel frames to latent.

        Args:
            pixels: (B, 3, F, H, W) in [-1, 1], PyTorch layout.

        Returns:
            Latent (B, C, F', H', W') in PyTorch layout.
        """
        # BCFHW -> BFHWC for MLX convolutions
        x = pixels.transpose(0, 2, 3, 4, 1)

        # Spatial patchification: (B, F, H, W, 3) -> (B, F, H/4, W/4, 48)
        # Reference: patchify(sample, patch_size_hw=4, patch_size_t=1)
        x = patchify_spatial(x, patch_size=4)

        x = self.conv_in(x)

        for block in self.down_blocks:
            x = block(x)

        # PixelNorm + SiLU before conv_out (reference: conv_norm_out + conv_act)
        x = self.conv_out(nn.silu(pixel_norm(x)))

        # Take first 128 channels (mean), discard the rest (log_var or dummy)
        x = x[:, :, :, :, :128]

        x = self.normalize_latent(x)

        # BFHWC -> BCFHW
        return x.transpose(0, 4, 1, 2, 3)


# --- upsampler ---
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Pixel shuffle helpers
# ---------------------------------------------------------------------------


def _pixel_shuffle_2d(x: mx.array, factor: int) -> mx.array:
    """2D pixel shuffle in BHWC layout.

    Matches PyTorch: rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)")
    MLX layout: (B, H, W, C*p1*p2) -> (B, H*p1, W*p2, C)
    """
    B, H, W, C_total = x.shape
    C = C_total // (factor * factor)
    # C is outermost (varies slowest), matching PyTorch (c, p1, p2) ordering
    x = x.reshape(B, H, W, C, factor, factor)
    # Interleave: (B, H, p1, W, p2, C)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(B, H * factor, W * factor, C)
    return x


def _pixel_shuffle_3d(
    x: mx.array,
    spatial_factor: int,
    temporal_factor: int,
) -> mx.array:
    """3D pixel shuffle in BDHWC layout.

    Matches PyTorch: rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")
    MLX layout: (B, D, H, W, C*tf*sf*sf) -> (B, D*tf, H*sf, W*sf, C)
    """
    B, D, H, W, C_total = x.shape
    C = C_total // (spatial_factor * spatial_factor * temporal_factor)
    x = x.reshape(B, D, H, W, C, temporal_factor, spatial_factor, spatial_factor)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, D * temporal_factor, H * spatial_factor, W * spatial_factor, C)
    return x


# ---------------------------------------------------------------------------
# BlurDownsample — depthwise Conv2d with binomial kernel
# ---------------------------------------------------------------------------


def _blur_downsample(x: mx.array, kernel: mx.array, stride: int) -> mx.array:
    """Apply depthwise blur-then-downsample on BHWC tensor.

    Args:
        x: (B, H, W, C) input.
        kernel: (1, K, K, 1) binomial kernel in MLX OHWI format.
        stride: Downsampling stride.

    Returns:
        (B, H', W', C) downsampled output.
    """
    if stride == 1:
        return x

    B, H, W, C = x.shape
    K = kernel.shape[1]
    pad = K // 2

    # Depthwise convolution: convolve each channel with the same kernel
    # Reshape to (B*C, H, W, 1) to process channels independently
    x = x.transpose(0, 3, 1, 2)  # (B, C, H, W)
    x = x.reshape(B * C, H, W, 1)  # (B*C, H, W, 1)

    # Pad spatially
    x = mx.pad(x, [(0, 0), (pad, pad), (pad, pad), (0, 0)])

    # Apply conv2d with stride — kernel is (1, K, K, 1) = (O=1, K, K, I=1)
    x = mx.conv2d(x, kernel, stride=(stride, stride))  # (B*C, H', W', 1)

    _, H2, W2, _ = x.shape
    x = x.reshape(B, C, H2, W2)
    x = x.transpose(0, 2, 3, 1)  # (B, H', W', C)
    return x


# ---------------------------------------------------------------------------
# ResBlock — matches reference ltx-core res_block.py
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block: conv1 -> norm1 -> SiLU -> conv2 -> norm2 -> SiLU(x + residual).

    Weight keys: conv1.weight/bias, conv2.weight/bias, norm1.weight/bias, norm2.weight/bias
    """

    def __init__(self, channels: int, dims: int = 3):
        super().__init__()
        if dims == 2:
            conv_cls = nn.Conv2d
        else:
            conv_cls = nn.Conv3d

        self.conv1 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.conv2 = conv_cls(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.silu(x + residual)
        return x


# ---------------------------------------------------------------------------
# Upsampler sub-module for rational resampling (spatial_x1_5)
# ---------------------------------------------------------------------------


class SpatialRationalResampler(nn.Module):
    """Rational spatial resampling: Conv2d -> PixelShuffle2D(num) -> BlurDownsample(den).

    For scale=1.5: num=3, den=2 (upsample 3x then downsample 2x = 1.5x).

    Weight keys:
        conv.weight, conv.bias — the Conv2d
        blur_down.kernel — the binomial kernel (1, K, K, 1)
    """

    def __init__(self, mid_channels: int, scale: float = 1.5):
        super().__init__()
        mapping: dict[float, tuple[int, int]] = {
            0.75: (3, 4),
            1.5: (3, 2),
            2.0: (2, 1),
            4.0: (4, 1),
        }
        if scale not in mapping:
            raise ValueError(f"Unsupported scale {scale}. Choose from {list(mapping.keys())}")
        self.num, self.den = mapping[scale]
        self.conv = nn.Conv2d(mid_channels, (self.num**2) * mid_channels, kernel_size=3, padding=1)

        # BlurDownsample stores just the kernel — loaded from weights
        self.blur_down = BlurDownsampleModule(stride=self.den)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, D, H, W, C) in BDHWC layout — applied per-frame."""
        B, D, H, W, C = x.shape
        x = x.reshape(B * D, H, W, C)
        x = self.conv(x)
        x = _pixel_shuffle_2d(x, self.num)
        x = self.blur_down(x)
        _, H2, W2, C2 = x.shape
        x = x.reshape(B, D, H2, W2, C2)
        return x


class BlurDownsampleModule(nn.Module):
    """Learnable-kernel blur downsample. Kernel loaded from weights.

    Weight key: kernel (shape 1, K, K, 1 in MLX OHWI format).
    """

    def __init__(self, stride: int = 2, kernel_size: int = 5):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        # Compute deterministic binomial kernel (non-learnable buffer).
        # Reference registers this as a buffer; we compute it here so the
        # module works even when weights don't include the kernel entry.
        k = [math.comb(kernel_size - 1, i) for i in range(kernel_size)]
        k_arr = mx.array(k, dtype=mx.float32)
        k2d = k_arr[:, None] * k_arr[None, :]  # outer product
        k2d = k2d / mx.sum(k2d)
        # MLX OHWI format (1, K, K, 1)
        self.kernel = k2d.reshape(1, kernel_size, kernel_size, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return _blur_downsample(x, self.kernel, self.stride)


# ---------------------------------------------------------------------------
# Main LatentUpsampler
# ---------------------------------------------------------------------------


class LTX23LatentUpsampler(nn.Module):
    """Neural latent upsampler supporting spatial and temporal variants.

    Architecture:
        initial_conv -> initial_norm -> SiLU
        -> 4x ResBlock (res_blocks)
        -> variant-specific upsampler
        -> 4x ResBlock (post_upsample_res_blocks)
        -> final_conv

    Weight key structure:
        initial_conv.weight/bias
        initial_norm.weight/bias
        res_blocks.{0-3}.conv1/conv2/norm1/norm2.weight/bias
        upsampler.{variant-specific keys}
        post_upsample_res_blocks.{0-3}.conv1/conv2/norm1/norm2.weight/bias
        final_conv.weight/bias

    Upsampler weight keys by variant:
        spatial_x2:   upsampler.0.weight, upsampler.0.bias
        spatial_x1_5: upsampler.conv.weight/bias, upsampler.blur_down.kernel
        temporal_x2:  upsampler.0.weight, upsampler.0.bias

    Args:
        in_channels: Input/output latent channels (128).
        mid_channels: Hidden channel dimension.
        num_blocks_per_stage: Number of ResBlocks per stage.
        spatial_upsample: Whether to spatially upsample.
        temporal_upsample: Whether to temporally upsample.
        spatial_scale: Spatial scale factor (2.0 or 1.5).
        rational_resampler: Use rational resampler for non-integer scales.
    """

    def __init__(
        self,
        in_channels: int = 128,
        mid_channels: int = 512,
        num_blocks_per_stage: int = 4,
        spatial_upsample: bool = True,
        temporal_upsample: bool = False,
        spatial_scale: float = 2.0,
        rational_resampler: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.spatial_upsample = spatial_upsample
        self.temporal_upsample = temporal_upsample
        self.spatial_scale = float(spatial_scale)
        self.rational_resampler = rational_resampler

        self.initial_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.initial_norm = nn.GroupNorm(32, mid_channels, pytorch_compatible=True)

        self.res_blocks = [ResBlock(mid_channels, dims=3) for _ in range(num_blocks_per_stage)]

        # Variant-specific upsampler
        # For spatial_x2 and temporal_x2, the reference uses nn.Sequential
        # which produces keys like upsampler.0.weight. We use a list to match.
        # For rational_resampler, the reference uses a named sub-module with
        # keys upsampler.conv.weight and upsampler.blur_down.kernel.
        self._upsampler_type: str
        if spatial_upsample and temporal_upsample:
            raise NotImplementedError("Combined spatial+temporal upsample not yet supported")
        elif spatial_upsample:
            if rational_resampler:
                self._upsampler_type = "rational"
                self.upsampler = SpatialRationalResampler(mid_channels, scale=spatial_scale)
            else:
                self._upsampler_type = "spatial_sequential"
                # List produces keys upsampler.0.weight/bias (matching nn.Sequential)
                self.upsampler = [nn.Conv2d(mid_channels, 4 * mid_channels, kernel_size=3, padding=1)]
        elif temporal_upsample:
            self._upsampler_type = "temporal_sequential"
            # List produces keys upsampler.0.weight/bias
            self.upsampler = [nn.Conv3d(mid_channels, 2 * mid_channels, kernel_size=3, padding=1)]
        else:
            raise ValueError("Either spatial_upsample or temporal_upsample must be True")

        self.post_upsample_res_blocks = [ResBlock(mid_channels, dims=3) for _ in range(num_blocks_per_stage)]

        self.final_conv = nn.Conv3d(mid_channels, in_channels, kernel_size=3, padding=1)

    def _apply_upsampler(self, x: mx.array) -> mx.array:
        """Apply variant-specific upsampler.

        Args:
            x: (B, D, H, W, C) in BDHWC layout.

        Returns:
            Upsampled tensor in BDHWC layout.
        """
        if self._upsampler_type == "spatial_sequential":
            # Conv2d per-frame + PixelShuffle2D(2)
            B, D, H, W, C = x.shape
            x = x.reshape(B * D, H, W, C)
            x = self.upsampler[0](x)
            x = _pixel_shuffle_2d(x, 2)
            _, H2, W2, C2 = x.shape
            x = x.reshape(B, D, H2, W2, C2)
            return x

        elif self._upsampler_type == "rational":
            # SpatialRationalResampler handles everything
            return self.upsampler(x)

        elif self._upsampler_type == "temporal_sequential":
            # Conv3d + PixelShuffle3D(temporal=2)
            x = self.upsampler[0](x)
            x = _pixel_shuffle_3d(x, spatial_factor=1, temporal_factor=2)
            return x

        else:
            raise ValueError(f"Unknown upsampler type: {self._upsampler_type}")

    def __call__(self, latent: mx.array) -> mx.array:
        """Upsample latent.

        Args:
            latent: (B, C, F, H, W) in PyTorch channel-first layout.

        Returns:
            Upsampled latent in (B, C, F, H', W') layout.
        """
        # BCFHW -> BFHWC for MLX
        x = latent.transpose(0, 2, 3, 4, 1)

        # Initial conv + norm + activation
        x = self.initial_conv(x)
        x = self.initial_norm(x)
        x = nn.silu(x)

        # Pre-upsample residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Upsampler (variant-specific)
        x = self._apply_upsampler(x)

        # Remove first frame after temporal upsample
        # (first frame encodes one pixel frame in the VAE)
        if self.temporal_upsample:
            x = x[:, 1:, :, :, :]

        # Post-upsample residual blocks
        for block in self.post_upsample_res_blocks:
            x = block(x)

        # Final conv
        x = self.final_conv(x)

        # BFHWC -> BCFHW
        return x.transpose(0, 4, 1, 2, 3)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "LTX23LatentUpsampler":
        """Create the right upsampler variant from an embedded config dict.

        Expected config keys (from embedded_config.json):
            in_channels, mid_channels, num_blocks_per_stage,
            spatial_upsample, temporal_upsample, spatial_scale,
            rational_resampler
        """
        return cls(
            in_channels=config.get("in_channels", 128),
            mid_channels=config.get("mid_channels", 512),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
            spatial_scale=config.get("spatial_scale", 2.0),
            rational_resampler=config.get("rational_resampler", False),
        )

# --- audio_vae ---
import mlx.core as mx
import mlx.nn as nn


def pixel_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """RMS normalization over the channel dimension (PixelNorm)."""
    return mx.fast.rms_norm(x, weight=None, eps=eps)


class WrappedConv2d(nn.Module):
    """Conv2d with optional causal (height-axis) padding.

    Causal mode pads height asymmetrically (all on top, none on bottom)
    matching reference CausalConv2d with causality_axis=HEIGHT.

    Weight keys become: ``<name>.conv.weight`` / ``<name>.conv.bias``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 1,
        causal: bool = False,
    ):
        super().__init__()
        self._causal = causal
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]

        if causal and ks > 1:
            # Causal: manual asymmetric height padding, symmetric width padding
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
            )
            self._pad_h_top = ks - 1  # Full causal pad on top
            self._pad_h_bottom = 0
            pw = padding if isinstance(padding, int) else padding[1]
            self._pad_w = pw
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            self._pad_h_top = 0
            self._pad_h_bottom = 0
            self._pad_w = 0

    def __call__(self, x: mx.array) -> mx.array:
        if self._causal and (self._pad_h_top > 0 or self._pad_w > 0):
            # NHWC layout: axis 0=B, 1=H, 2=W, 3=C
            x = mx.pad(x, [(0, 0), (self._pad_h_top, self._pad_h_bottom), (self._pad_w, self._pad_w), (0, 0)])
        return self.conv(x)


class AudioResBlock(nn.Module):
    """Residual block matching ``up.*.block.*.`` or ``mid.block_*`` weight keys.

    Keys produced:
        conv1.conv.{weight,bias}
        conv2.conv.{weight,bias}
        nin_shortcut.conv.{weight,bias}   (only when in != out channels)
    """

    def __init__(self, in_channels: int, out_channels: int | None = None, causal: bool = False):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = WrappedConv2d(in_channels, out_channels, 3, padding=1, causal=causal)
        self.conv2 = WrappedConv2d(out_channels, out_channels, 3, padding=1, causal=causal)
        if in_channels != out_channels:
            self.nin_shortcut = WrappedConv2d(in_channels, out_channels, 1, padding=0)
        else:
            self.nin_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C) in MLX NHWC layout."""
        residual = x
        x = pixel_norm(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = pixel_norm(x)
        x = nn.silu(x)
        x = self.conv2(x)
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)
        return x + residual


class AudioAttnBlock(nn.Module):
    """Self-attention block for audio VAE.

    Weight keys: norm.{weight,bias}, q.conv.{w,b}, k.conv.{w,b}, v.conv.{w,b}, proj_out.conv.{w,b}
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, pytorch_compatible=True)
        self.q = WrappedConv2d(channels, channels, 1, padding=0)
        self.k = WrappedConv2d(channels, channels, 1, padding=0)
        self.v = WrappedConv2d(channels, channels, 1, padding=0)
        self.proj_out = WrappedConv2d(channels, channels, 1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C)"""
        B, H, W, C = x.shape
        residual = x
        h = self.norm(x)

        q = self.q(h).reshape(B, H * W, C)
        k = self.k(h).reshape(B, H * W, C)
        v = self.v(h).reshape(B, H * W, C)

        scale = C**-0.5
        attn = (q @ k.transpose(0, 2, 1)) * scale
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).reshape(B, H, W, C)
        out = self.proj_out(out)
        return residual + out


class AudioUpsample(nn.Module):
    """2x spatial upsample via nearest interpolation + Conv2d.

    In causal mode, drops first row after conv to maintain temporal alignment.

    Key: ``upsample.conv.conv.{weight,bias}``
    """

    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self.conv = WrappedConv2d(channels, channels, 3, padding=1, causal=causal)
        self._causal = causal

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, H, W, C)"""
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        x = self.conv(x)
        if self._causal:
            x = x[:, 1:, :, :]  # Drop first row for causal alignment
        return x


class AudioUpBlock(nn.Module):
    """One decoder up-stage: N resblocks (with optional per-block attention) + optional upsample.

    Key prefix: ``up.<idx>.``
    Children:
        block.{0,1,...} — AudioResBlock
        attn.{0,1,...}  — AudioAttnBlock (optional)
        upsample        — AudioUpsample (optional)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        add_upsample: bool = False,
        add_attention: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.block = [
            AudioResBlock(in_channels if i == 0 else out_channels, out_channels, causal=causal)
            for i in range(num_blocks)
        ]
        if add_attention:
            self.attn = [AudioAttnBlock(out_channels) for _ in range(num_blocks)]
        else:
            self.attn = None
        self.upsample = AudioUpsample(out_channels, causal=causal) if add_upsample else None

    def __call__(self, x: mx.array) -> mx.array:
        for i, blk in enumerate(self.block):
            x = blk(x)
            if self.attn is not None:
                x = self.attn[i](x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class AudioMidBlock(nn.Module):
    """Mid block: resblock, optional attention, resblock.

    Keys: mid.block_1, mid.attn_1 (optional), mid.block_2.
    """

    def __init__(self, channels: int, causal: bool = False, add_attention: bool = False):
        super().__init__()
        self.block_1 = AudioResBlock(channels, causal=causal)
        self.attn_1 = AudioAttnBlock(channels) if add_attention else None
        self.block_2 = AudioResBlock(channels, causal=causal)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.block_1(x)
        if self.attn_1 is not None:
            x = self.attn_1(x)
        x = self.block_2(x)
        return x


class AudioPerChannelStatistics(nn.Module):
    """Per-channel normalization statistics for audio VAE (loaded from weights).

    Safetensors keys have underscore prefix: ``_mean_of_means``, ``_std_of_means``.
    MLX treats underscore-prefixed attrs as private, so we use public names
    and remap during weight loading.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.mean_of_means = mx.zeros((channels,))
        self.std_of_means = mx.ones((channels,))


class AudioDownsample(nn.Module):
    """2x spatial downsample via stride-2 Conv2d (inverse of AudioUpsample)."""

    def __init__(self, channels: int, causal: bool = False):
        super().__init__()
        self.conv = WrappedConv2d(channels, channels, 3, stride=2, padding=1, causal=causal)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class AudioDownBlock(nn.Module):
    """Encoder down-stage: optional downsample + N resblocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 3,
        add_downsample: bool = False,
        add_attention: bool = False,
        causal: bool = False,
    ):
        super().__init__()
        self.downsample = AudioDownsample(in_channels, causal=causal) if add_downsample else None
        self.block = [
            AudioResBlock(in_channels if i == 0 else out_channels, out_channels, causal=causal)
            for i in range(num_blocks)
        ]
        self.attn = [AudioAttnBlock(out_channels) for _ in range(num_blocks)] if add_attention else None

    def __call__(self, x: mx.array) -> mx.array:
        if self.downsample is not None:
            x = self.downsample(x)
        for i, blk in enumerate(self.block):
            x = blk(x)
            if self.attn is not None:
                x = self.attn[i](x)
        return x


class LTX23AudioEncoder(nn.Module):
    """Audio VAE encoder: mel (B, 2, T, 64) -> latent (B, 8, T', 16)."""

    def __init__(self):
        super().__init__()
        self.conv_in = WrappedConv2d(2, 128, 3, padding=1, causal=True)
        self.down = [
            AudioDownBlock(128, 256, num_blocks=3, add_downsample=True, add_attention=False, causal=True),
            AudioDownBlock(256, 512, num_blocks=3, add_downsample=True, add_attention=False, causal=True),
            AudioDownBlock(512, 512, num_blocks=3, add_downsample=False, add_attention=False, causal=True),
        ]
        self.mid = AudioMidBlock(512, causal=True, add_attention=False)
        self.conv_out = WrappedConv2d(512, 8, 3, padding=1, causal=True)
        self.per_channel_statistics = AudioPerChannelStatistics(128)

    def encode(self, mel: mx.array) -> mx.array:
        """mel: (B, 2, T, 64) -> latent (B, 8, T', 16)."""
        x = mel.transpose(0, 2, 3, 1)  # (B, T, 64, 2) NHWC
        x = self.conv_in(x)
        for blk in self.down:
            x = blk(x)
        x = self.mid(x)
        x = pixel_norm(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        b, h, w, c = x.shape
        x_flat = x.reshape(b, h * w, c)
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, -1)
        x_flat = (x_flat - mean) / (std + 1e-6)
        if h * w * c != 8 * h * 16:
            raise RuntimeError(f"LTX audio encoder output shape mismatch: got {(b, h, w, c)}")
        return x_flat.reshape(b, h, 8, 16).transpose(0, 2, 1, 3)


class LTX23AudioDecoder(nn.Module):
    """Audio VAE decoder: latent (B, 8, T, 16) -> mel (B, 2, T', 64).

    Architecture (reverse order of ``up`` indices — up.2 runs first):
        conv_in  : Conv2d(8, 512, 3)
        mid      : ResBlock(512) + AttnBlock(512) + ResBlock(512)
        up.2     : 3x ResBlock(512) + Upsample  → freq 16→32
        up.1     : 3x ResBlock(512→256) + Upsample → freq 32→64
        up.0     : 3x ResBlock(256→128), no upsample
        conv_out : Conv2d(128, 2, 3)
    """

    def __init__(self):
        super().__init__()
        # conv_in: 8 input channels (latent C1 dim)
        self.conv_in = WrappedConv2d(8, 512, 3, padding=1, causal=True)

        # Mid — config: mid_block_add_attention=False for distilled model
        self.mid = AudioMidBlock(512, causal=True, add_attention=False)

        # Up blocks — stored in list indexed [0, 1, 2] but run in REVERSE order
        # Config: attn_resolutions=[] → no attention in any up block
        # up.0: 256→128, no upsample
        # up.1: 512→256, upsample
        # up.2: 512→512, upsample
        self.up = [
            AudioUpBlock(256, 128, num_blocks=3, add_upsample=False, add_attention=False, causal=True),
            AudioUpBlock(512, 256, num_blocks=3, add_upsample=True, add_attention=False, causal=True),
            AudioUpBlock(512, 512, num_blocks=3, add_upsample=True, add_attention=False, causal=True),
        ]

        # Output
        self.conv_out = WrappedConv2d(128, 2, 3, padding=1, causal=True)

        # Per-channel normalization for latent denormalization
        self.per_channel_statistics = AudioPerChannelStatistics(128)

    def decode(self, latent: mx.array) -> mx.array:
        """Decode audio latent to mel spectrogram.

        Args:
            latent: (B, 8, T, 16) audio latent.

        Returns:
            Mel spectrogram (B, 2, T', 64).
        """
        B, C1, T, C2 = latent.shape  # (B, 8, T, 16)

        # Flatten to (B, T, 128) for denormalization
        x_flat = latent.transpose(0, 2, 1, 3).reshape(B, T, C1 * C2)

        # Denormalize using per-channel statistics
        mean = self.per_channel_statistics.mean_of_means.reshape(1, 1, -1)
        std = self.per_channel_statistics.std_of_means.reshape(1, 1, -1)
        x_flat = x_flat * std + mean

        # Reshape back to 2D spatial: (B, T, 16, 8) — NHWC for Conv2d
        # T = height, 16 = width (frequency), 8 = channels
        x = x_flat.reshape(B, T, C1, C2).transpose(0, 1, 3, 2)  # (B, T, 8, 16) → (B, T, 16, 8) NHWC

        # Encoder/Decoder
        x = self.conv_in(x)  # (B, T, 16, 512)
        x = self.mid(x)

        # Up blocks run in reverse index order: up.2, up.1, up.0
        for i in reversed(range(len(self.up))):
            x = self.up[i](x)

        x = pixel_norm(x)  # norm_out
        x = nn.silu(x)
        x = self.conv_out(x)  # (B, T', 64, 2) in NHWC

        # Convert to (B, 2, T', 64)
        x = x.transpose(0, 3, 1, 2)
        return x

# --- vocoder ---
import math

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# SnakeBeta activation
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """SnakeBeta activation: x + (1/b) * sin^2(a * x).

    Weights are stored in LOG-SCALE. Forward applies exp() to get actual values.
    Weight keys: ``act.alpha``, ``act.beta``.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Initialized to zeros — exp(0) = 1.0, matching reference default
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C)"""
        alpha = mx.exp(self.alpha).reshape(1, 1, -1)
        beta = mx.exp(self.beta).reshape(1, 1, -1)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


# ---------------------------------------------------------------------------
# Anti-aliased Activation1d (wraps SnakeBeta)
# ---------------------------------------------------------------------------


class LowPassKernel(nn.Module):
    """Holds the low-pass filter kernel.

    Weight key: ``filter`` — shape (1, K, 1) in MLX Conv1d format (O, K, I).
    Pre-transposed by mlx-forge from PyTorch (1, 1, K).
    """

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.filter = mx.ones((1, kernel_size, 1))


class DownSample1d(nn.Module):
    """Anti-aliased 2x downsampler with low-pass filter.

    Weight keys:
        lowpass.filter — (1, 1, K)
    """

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.lowpass = LowPassKernel(kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) -> (B, T//2, C)"""
        B, T, C = x.shape
        # Reshape for grouped conv1d: (B*C, T, 1)
        x = x.transpose(0, 2, 1).reshape(B * C, T, 1)

        # Replicate pad — matches reference LowPassFilter1d(padding_mode='replicate')
        K = self.lowpass.filter.shape[1]
        even = 1 if K % 2 == 0 else 0
        pad_left = K // 2 - even
        pad_right = K // 2
        left_edge = mx.repeat(x[:, :1, :], pad_left, axis=1)
        right_edge = mx.repeat(x[:, -1:, :], pad_right, axis=1)
        x = mx.concatenate([left_edge, x, right_edge], axis=1)

        # Apply filter — already in MLX (O=1, K, I=1) format
        x = mx.conv1d(x, self.lowpass.filter, stride=2)

        # Reshape back: (B, C, T') -> (B, T', C)
        T_out = x.shape[1]
        return x.reshape(B, C, T_out).transpose(0, 2, 1)


class UpSample1d(nn.Module):
    """2x upsample with anti-aliasing filter.

    Weight key: ``filter`` — shape (1, K, 1) in MLX Conv1d format.
    Pre-transposed by mlx-forge from PyTorch (1, 1, K).
    """

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.filter = mx.ones((1, kernel_size, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) -> (B, T*2, C)"""
        B, T, C = x.shape
        # Insert zeros between samples: (B, T, C) -> (B, T*2, C)
        x_up = mx.zeros((B, T * 2, C))
        # Slice assign avoids mlx 0.31.2 Metal scatter bug in .at[strided].add().
        x_up[:, ::2, :] = x

        # Reshape for grouped conv1d: (B*C, T*2, 1)
        x_up = x_up.transpose(0, 2, 1).reshape(B * C, T * 2, 1)

        K = self.filter.shape[1]
        pad = K // 2
        left_edge = mx.repeat(x_up[:, :1, :], pad, axis=1)
        right_edge = mx.repeat(x_up[:, -1:, :], pad - 1, axis=1)
        x_up = mx.concatenate([left_edge, x_up, right_edge], axis=1)

        # Filter already in MLX (O=1, K, I=1) format
        x_up = mx.conv1d(x_up, self.filter)

        T_out = x_up.shape[1]
        return x_up.reshape(B, C, T_out).transpose(0, 2, 1) * 2.0


class Activation1d(nn.Module):
    """Anti-aliased activation: upsample -> activation -> downsample.

    Weight keys:
        act.alpha, act.beta         — SnakeBeta params
        upsample.filter             — (1, K, 1) MLX Conv1d format
        downsample.lowpass.filter   — (1, K, 1) MLX Conv1d format
    """

    def __init__(self, channels: int, up_ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(kernel_size)
        self.downsample = DownSample1d(kernel_size)
        self.up_ratio = up_ratio

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C)"""
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# AMPBlock1 (anti-aliased multi-periodicity residual block)
# ---------------------------------------------------------------------------


class AMPBlock1(nn.Module):
    """Anti-aliased multi-periodicity block with Activation1d.

    Weight keys:
        convs1.{0,1,2}.{weight,bias}
        convs2.{0,1,2}.{weight,bias}
        acts1.{0,1,2}.act.{alpha,beta}
        acts1.{0,1,2}.upsample.filter
        acts1.{0,1,2}.downsample.lowpass.filter
        acts2.{0,1,2}.act.{alpha,beta}
        acts2.{0,1,2}.upsample.filter
        acts2.{0,1,2}.downsample.lowpass.filter
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.acts1 = []
        self.acts2 = []

        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.acts1.append(Activation1d(channels))
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=d,
                )
            )
            self.acts2.append(Activation1d(channels))
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            residual = x
            x = act1(x)
            x = conv1(x)
            x = act2(x)
            x = conv2(x)
            x = x + residual
        return x


# ---------------------------------------------------------------------------
# BigVGAN Vocoder
# ---------------------------------------------------------------------------


class BigVGANVocoder(nn.Module):
    """BigVGAN v2 vocoder: mel spectrogram -> waveform.

    Base vocoder: 128-mel -> 16kHz stereo (2-ch output)
        upsample_rates = [5, 2, 2, 2, 2, 2] -> 160x
        upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        channels: 1536 -> 768 -> 384 -> 192 -> 96 -> 48 -> 24

    BWE generator: 128-mel -> 48kHz stereo
        upsample_rates = [6, 5, 2, 2, 2] -> 240x
        upsample_kernel_sizes = [12, 11, 4, 4, 4]  (weight shapes show actual kernel)
        channels: 512 -> 256 -> 128 -> 64 -> 32 -> 16
    """

    def __init__(
        self,
        in_channels: int = 128,
        upsample_initial_channel: int = 1536,
        upsample_rates: tuple[int, ...] = (5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (11, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        out_channels: int = 2,
        apply_final_activation: bool = True,
    ):
        super().__init__()
        self._apply_final_activation = apply_final_activation

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)

        # Upsample layers (directly indexed, no interleaved activations)
        self.ups = []
        channels = upsample_initial_channel

        # Flat resblocks list (3 per upsample stage)
        self.resblocks = []

        for _i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = channels // 2
            padding = (kernel - rate) // 2
            self.ups.append(nn.ConvTranspose1d(channels, out_ch, kernel_size=kernel, stride=rate, padding=padding))

            # 3 resblocks per stage (one per kernel size)
            for _j, (rk, rd) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(out_ch, rk, rd))

            channels = out_ch

        self.act_post = Activation1d(channels)
        self.conv_post = nn.Conv1d(channels, out_channels, kernel_size=7, padding=3, bias=False)

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

    def __call__(self, mel: mx.array) -> mx.array:
        """Convert mel spectrogram to waveform.

        Args:
            mel: (B, T, n_mels) mel spectrogram in MLX layout,
                 or (B, C, T, n_mels) for stereo processing.

        Returns:
            Waveform (B, T_audio) or (B, C, T_audio).
        """
        process_channels = False
        if mel.ndim == 4:
            B, C, T, M = mel.shape
            mel = mel.reshape(B * C, T, M)
            process_channels = True

        x = self.conv_pre(mel)

        for i in range(self.num_upsamples):
            # Activation1d before upsample conv
            x = self.ups[i](x)

            # Average resblocks for this stage
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = self.act_post(x)
        x = self.conv_post(x)  # (B, T_audio, 2)
        if self._apply_final_activation:
            x = mx.tanh(x)

        if process_channels:
            # x is (B*C, T_audio, 2) — but for stereo mel input, output is already 2-ch
            x = x.reshape(B, C, x.shape[1], x.shape[2])

        return x

    @property
    def hop_length(self) -> int:
        """Total upsample ratio = product of all upsample rates."""
        return math.prod([5, 2, 2, 2, 2, 2])  # 160

import mlx.core as mx
import mlx.nn as nn
import numpy as np



# ---------------------------------------------------------------------------
# Hann-sinc resampler (no learned weights)
# ---------------------------------------------------------------------------


class HannSincResampler:
    """3x upsampler using Hann-windowed sinc interpolation.

    Matches reference UpSample1d(ratio=3, window_type="hann").
    Not an nn.Module — no learnable parameters.

    Reference implementation:
        1. Replicate-pads input by ``width`` on each side
        2. Applies conv_transpose1d(stride=ratio) with the sinc kernel
        3. Scales by ratio and slices [pad_left:-pad_right]

    MLX equivalent:
        1. Replicate-pads input by ``width``
        2. Zero-inserts (stride) between samples
        3. Applies forward conv1d with the sinc kernel
        4. Scales by ratio and slices to match reference output
    """

    def __init__(self, upsample_factor: int = 3):
        self.upsample_factor = upsample_factor
        self._rolloff = 0.99
        self._lowpass_filter_width = 6
        self._width = int(np.ceil(self._lowpass_filter_width / self._rolloff))  # 7
        kernel = self._build_kernel(upsample_factor)
        self.kernel = mx.array(kernel[:, None])  # (K, 1) for conv1d
        # Padding/slicing params matching reference UpSample1d (Hann path)
        self._pad = self._width  # replicate-pad on input: 7
        self._kernel_size = 2 * self._width * upsample_factor + 1  # 43
        self._pad_left = 2 * self._width * upsample_factor  # 42
        self._pad_right = self._kernel_size - upsample_factor  # 40

    def _build_kernel(self, ratio: int) -> np.ndarray:
        """Build Hann-windowed sinc filter matching reference exactly.

        Reference formula (UpSample1d, window_type="hann"):
            time_axis = (arange(kernel_size) / ratio - width) * rolloff
            time_clamped = clip(time_axis, -lpfw, lpfw)
            window = cos(time_clamped * pi / lpfw / 2) ** 2
            kernel = sinc(time_axis) * window * rolloff / ratio
        """
        kernel_size = 2 * self._width * ratio + 1  # 43
        idx = np.arange(kernel_size, dtype=np.float64)
        time_axis = (idx / ratio - self._width) * self._rolloff
        time_clamped = np.clip(time_axis, -self._lowpass_filter_width, self._lowpass_filter_width)
        window = np.cos(time_clamped * np.pi / self._lowpass_filter_width / 2) ** 2
        kernel = (np.sinc(time_axis) * window * self._rolloff / ratio).astype(np.float32)
        return kernel

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample: (B, T) -> (B, T * factor).

        Matches reference: replicate-pad -> conv_transpose1d -> scale -> slice.
        Implemented as: replicate-pad -> zero-insert -> full conv1d -> scale -> slice.
        """
        B, T = x.shape
        ratio = self.upsample_factor

        # 1. Replicate-pad input (matches F.pad(x, (pad, pad), mode='replicate'))
        first = mx.repeat(x[:, :1], self._pad, axis=1)  # (B, pad)
        last = mx.repeat(x[:, -1:], self._pad, axis=1)  # (B, pad)
        x_padded = mx.concatenate([first, x, last], axis=1)  # (B, T + 2*pad)
        T_padded = x_padded.shape[1]

        # 2. Zero-insert between samples (conv_transpose1d style):
        #    output length = (T_padded - 1) * ratio + 1
        zi_len = (T_padded - 1) * ratio + 1
        upsampled = mx.zeros((B, zi_len))
        # Slice assign avoids mlx 0.31.2 Metal scatter bug in .at[strided].add().
        upsampled[:, ::ratio] = x_padded

        # 3. Full convolution via zero-pad + valid conv1d
        #    Full conv output = zi_len + K - 1
        upsampled = upsampled[:, :, None]  # (B, zi_len, 1)
        K = self.kernel.shape[0]
        upsampled = mx.pad(upsampled, [(0, 0), (K - 1, K - 1), (0, 0)])
        filt = self.kernel[None, :, :]  # (1, K, 1)
        result = mx.conv1d(upsampled, filt, padding=0)
        result = result.squeeze(-1)  # (B, zi_len + K - 1)

        # 4. Scale by ratio (matching reference: self.ratio * conv_transpose1d(...))
        result = result * ratio

        # 5. Slice to match reference output: [pad_left:-pad_right]
        result = result[:, self._pad_left : -self._pad_right]

        return result[:, : T * ratio]


# ---------------------------------------------------------------------------
# STFT function (loads basis from weights)
# ---------------------------------------------------------------------------


class STFTFunction(nn.Module):
    """STFT using pre-computed basis matrices.

    Weight keys (MLX Conv1d format, pre-transposed by mlx-forge):
        forward_basis  — (n_fft+2, n_fft, 1)  i.e. (O, K, I)
        inverse_basis  — (n_fft+2, n_fft, 1)
    """

    def __init__(self, n_fft: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.forward_basis = mx.zeros((n_fft + 2, n_fft, 1))
        self.inverse_basis = mx.zeros((n_fft + 2, n_fft, 1))


# ---------------------------------------------------------------------------
# MelSTFT (loads mel_basis and stft_fn from weights)
# ---------------------------------------------------------------------------


class MelSTFT(nn.Module):
    """Mel spectrogram transform for BWE input.

    Weight keys:
        mel_basis                   — (64, 257)
        stft_fn.forward_basis       — (514, 1, 512)
        stft_fn.inverse_basis       — (514, 1, 512)

    Note: hop_length=80 matches the BWE config (not 160 from the audio VAE
    preprocessing). This ensures BWE generator output length matches the
    3x-resampled skip connection: mel_frames * 240 == vocoder_output * 3.
    """

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 80,
        n_mels: int = 64,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_basis = mx.zeros((n_mels, n_fft // 2 + 1))
        self.stft_fn = STFTFunction(n_fft)

    def __call__(self, waveform: mx.array) -> mx.array:
        """Compute mel spectrogram.

        Args:
            waveform: (B, T) waveform.

        Returns:
            (B, T_frames, n_mels) mel spectrogram.
        """
        B, T = waveform.shape

        # Use the loaded forward_basis for STFT
        # forward_basis: (n_fft+2, 1, n_fft) — real and imag interleaved
        # Reshape waveform for conv: (B, T, 1)
        x = waveform[:, :, None]

        # Causal padding: left-only, matching reference _STFTFn
        left_pad = max(0, self.n_fft - self.hop_length)  # 512 - 80 = 432
        x = mx.pad(x, [(0, 0), (left_pad, 0), (0, 0)])

        # Apply STFT basis via conv1d
        # forward_basis in MLX Conv1d format: (O, K, I) = (n_fft+2, n_fft, 1)
        # Pre-transposed by mlx-forge
        basis = self.stft_fn.forward_basis  # (514, 512, 1)

        stft_out = mx.conv1d(x, basis, stride=self.hop_length)  # (B, T', 514)

        # Split real and imaginary parts
        n_fft_bins = self.n_fft // 2 + 1  # 257
        real = stft_out[:, :, :n_fft_bins]
        imag = stft_out[:, :, n_fft_bins:]

        # Magnitude
        mag = mx.sqrt(real * real + imag * imag + 1e-9)  # (B, T', 257)

        # Apply mel filterbank
        mel = mag @ self.mel_basis.T  # (B, T', n_mels)

        # Log mel
        return mx.log(mx.maximum(mel, 1e-5))


# ---------------------------------------------------------------------------
# Full vocoder + BWE pipeline
# ---------------------------------------------------------------------------


class LTX23Vocoder(nn.Module):
    """Full audio pipeline: mel -> 16kHz waveform -> 48kHz waveform.

    Combines base vocoder, 3x Kaiser-sinc resampler, and BWE generator.
    BWE output = clamp(resampled_base + bwe_residual, -1, 1).

    The forward pass must run in fp32: bfloat16 accumulation errors compound
    through 108 sequential convolutions (BigVGAN v2) and degrade spectral
    metrics (mel_l1, MRSTFT) by 40-90%. Call ``upcast_weights_to_fp32()`` once
    after ``load_weights()`` to promote all parameters to fp32.

    Weight key hierarchy (all under vocoder.* prefix):
        conv_pre, ups, resblocks, act_post, conv_post — base vocoder
        bwe_generator.*                                — BWE BigVGAN
        mel_stft.*                                     — MelSTFT for BWE
    """

    def __init__(self):
        super().__init__()
        # Base vocoder: mel -> 16kHz (stereo, 2-ch output)
        # This is a PEER module — base vocoder keys are at the same level,
        # NOT nested under a "vocoder" attribute. The base vocoder keys
        # (conv_pre, ups, resblocks, etc.) live directly in this module.

        # We need the base vocoder conv_pre, ups, resblocks, act_post, conv_post
        # to load as direct attributes of this class (not nested).
        # Use composition and expose the relevant attributes.

        # NOTE: The vocoder.safetensors has the base vocoder keys at top level
        # (after stripping "vocoder." prefix). So we DON'T wrap in a sub-module.
        # Instead we construct the BigVGAN components directly.

        # Base vocoder components are defined above in this module.
        in_channels = 128
        upsample_initial_channel = 1536
        upsample_rates = (5, 2, 2, 2, 2, 2)
        upsample_kernel_sizes = (11, 4, 4, 4, 4, 4)
        resblock_kernel_sizes = (3, 7, 11)
        resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)

        self.ups = []
        self.resblocks = []
        channels = upsample_initial_channel

        for _i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = channels // 2
            padding = (kernel - rate) // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    out_ch,
                    kernel_size=kernel,
                    stride=rate,
                    padding=padding,
                )
            )
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(out_ch, rk, rd))
            channels = out_ch

        self.act_post = Activation1d(channels)
        self.conv_post = nn.Conv1d(channels, 2, kernel_size=7, padding=3, bias=False)

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # --- BWE generator ---
        self.bwe_generator = BigVGANVocoder(
            in_channels=128,
            upsample_initial_channel=512,
            upsample_rates=(6, 5, 2, 2, 2),
            upsample_kernel_sizes=(12, 11, 4, 4, 4),
            out_channels=2,
            apply_final_activation=False,
        )

        # --- Mel STFT for BWE ---
        self.mel_stft = MelSTFT()

        # --- Resampler (no weights) ---
        self._resampler = HannSincResampler(upsample_factor=3)

    def upcast_weights_to_fp32(self) -> None:
        """Promote all parameters to fp32 in-place.

        Required after ``load_weights()`` — running the forward in bf16/fp16
        compounds accumulation errors through 108 sequential convolutions and
        degrades spectral metrics (mel_l1, MRSTFT) by 40-90%. MLX has no
        autocast; the weights themselves must be fp32 for the conv kernels to
        run in fp32.
        """
        from mlx.utils import tree_map

        self.update(tree_map(lambda p: p.astype(mx.float32), self.parameters()))

    def _run_base_vocoder(self, mel: mx.array) -> mx.array:
        """Run base vocoder: mel (B, T, 64) -> waveform (B, T_audio, 2)."""
        x = self.conv_pre(mel)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = self.act_post(x)
        x = self.conv_post(x)  # (B, T_audio, 2)
        # embedded_config: use_tanh_at_final=false for base vocoder
        return mx.clip(x, -1.0, 1.0)

    def __call__(self, mel: mx.array) -> mx.array:
        """Full pipeline: mel -> 48kHz stereo waveform.

        Reference: ltx-core VocoderWithBWE.forward

        The stereo mel channels are CONCATENATED (not processed separately):
        (B, 2, T, 64) → rearrange to (B, T, 128) → vocoder → (B, T_audio, 2).

        Args:
            mel: (B, 2, T, n_mels) stereo mel spectrogram.

        Returns:
            (B, 2, T_audio) waveform at 48kHz.
        """
        input_dtype = mel.dtype
        mel = mel.astype(mx.float32)

        B, C, T, M = mel.shape  # (B, 2, T, 64)

        # 1. Run base vocoder: concatenate stereo channels for input
        # (B, 2, T, 64) → transpose mel_bins to front → (B, 2, 64, T) → rearrange → (B, 128, T) → (B, T, 128)
        mel_concat = mel.transpose(0, 1, 3, 2)  # (B, 2, 64, T)
        mel_concat = mel_concat.reshape(B, C * M, T)  # (B, 128, T)
        mel_concat = mel_concat.transpose(0, 2, 1)  # (B, T, 128)

        waveform_16k = self._run_base_vocoder(mel_concat)  # (B, T_audio_16k, 2)
        waveform_16k = waveform_16k.transpose(0, 2, 1)  # (B, 2, T_audio_16k)

        # Compute output length before padding
        length_16k = waveform_16k.shape[-1]
        output_length = length_16k * 3  # 3x upsample (16kHz → 48kHz)

        # Pad to multiple of hop_length for exact mel frame count
        hop = self.mel_stft.hop_length
        remainder = length_16k % hop
        if remainder != 0:
            pad_amount = hop - remainder
            waveform_16k = mx.pad(waveform_16k, [(0, 0), (0, 0), (0, pad_amount)])

        # 2. Compute mel of vocoder output: (B, 2, T) → (B*2, T) → mel → (B, 2, n_mels, T')
        flat_wav = waveform_16k.reshape(B * C, -1)  # (B*2, T)
        bwe_mel = self.mel_stft(flat_wav)  # (B*2, T', n_mels)
        T_frames = bwe_mel.shape[1]
        bwe_mel = bwe_mel.reshape(B, C, T_frames, M)  # (B, 2, T', 64)

        # 3. Run BWE generator on mel: (B, 2, T', 64) → same rearrange as base vocoder
        bwe_mel_concat = bwe_mel.transpose(0, 1, 3, 2)  # (B, 2, 64, T')
        bwe_mel_concat = bwe_mel_concat.reshape(B, C * M, T_frames)  # (B, 128, T')
        bwe_mel_concat = bwe_mel_concat.transpose(0, 2, 1)  # (B, T', 128)

        residual = self.bwe_generator(bwe_mel_concat)  # (B, T_bwe, 2)
        residual = residual.transpose(0, 2, 1)  # (B, 2, T_bwe)

        # 4. Resample base vocoder output to 48kHz
        skip_channels = []
        for c in range(C):
            resampled = self._resampler(waveform_16k[:, c, :])  # (B, T_48k)
            skip_channels.append(resampled)
        skip = mx.stack(skip_channels, axis=1)  # (B, 2, T_48k)

        # 5. Add residual and clip
        min_len = min(skip.shape[-1], residual.shape[-1])
        output = skip[:, :, :min_len] + residual[:, :, :min_len]
        output = mx.clip(output, -1.0, 1.0)[:, :, :output_length]
        return output.astype(input_dtype)


_decoder_cache: dict[str, LTX23VideoDecoder] = {}
_encoder_cache: dict[str, LTX23VideoEncoder] = {}
_upsampler_cache: dict[str, LTX23LatentUpsampler] = {}
_audio_decoder_cache: dict[str, LTX23AudioDecoder] = {}
_audio_encoder_cache: dict[str, LTX23AudioEncoder] = {}
_vocoder_cache: dict[str, LTX23Vocoder] = {}


def _remap_video_decoder_keys(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Normalize per-channel stat keys (dgrauet ``mean``/``std`` vs legacy ``_*`` names)."""
    out: dict[str, mx.array] = {}
    for k, v in weights.items():
        nk = k
        nk = nk.replace("per_channel_statistics._mean_of_means", "per_channel_statistics.mean")
        nk = nk.replace("per_channel_statistics._std_of_means", "per_channel_statistics.std")
        nk = nk.replace("per_channel_statistics.mean_of_means", "per_channel_statistics.mean")
        nk = nk.replace("per_channel_statistics.std_of_means", "per_channel_statistics.std")
        out[nk] = v
    return out


def load_ltx23_video_decoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX23VideoDecoder:
    key = str(bundle_root.resolve())
    if key in _decoder_cache:
        return _decoder_cache[key]
    dec = LTX23VideoDecoder(causal=False, spatial_padding_mode="zeros")
    raw = _remap_video_decoder_keys(
        _load_bundle_weights(bundle_root, "vae_decoder.safetensors", "vae_decoder.", load_fn)
    )
    dec.load_weights(list(raw.items()), strict=False)
    _decoder_cache[key] = dec
    return dec


def load_ltx23_video_encoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX23VideoEncoder:
    key = str(bundle_root.resolve())
    if key in _encoder_cache:
        return _encoder_cache[key]
    enc = LTX23VideoEncoder()
    enc.load_weights(list(_remap_encoder_keys(_load_bundle_weights(bundle_root, "vae_encoder.safetensors", "vae_encoder.", load_fn)).items()))
    _encoder_cache[key] = enc
    return enc


def _strip_shared_weight_prefix(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Drop a single outer key prefix when every tensor shares it (dgrauet upscaler shards)."""
    if not weights:
        return weights
    prefixes = {k.split(".", 1)[0] for k in weights if "." in k}
    if len(prefixes) != 1:
        return weights
    prefix = f"{next(iter(prefixes))}."
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in weights.items()}


def _load_upsampler_config(config_path: Path) -> dict[str, Any]:
    import json

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"LTX 2.3 upsampler config unreadable: {config_path}: {e}") from e
    if isinstance(data, dict) and isinstance(data.get("config"), dict):
        return data["config"]
    if isinstance(data, dict):
        return data
    raise RuntimeError(f"LTX 2.3 upsampler config must be a JSON object: {config_path}")


_UPSAMPLER_WEIGHT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "spatial_x2": (
        "latent_upsampler.safetensors",
        "spatial_upscaler_x2_v1_1.safetensors",
    ),
    "spatial_x1_5": (
        "latent_upsampler_x1_5.safetensors",
        "spatial_upscaler_x1_5_v1_0.safetensors",
    ),
    "temporal_x2": (
        "latent_upsampler_temporal_x2.safetensors",
        "temporal_upscaler_x2_v1_0.safetensors",
    ),
}


def _resolve_upsampler_weight_path(bundle_root: Path, variant: str) -> Path:
    """Resolve latent upsampler shard (diffusers names vs dgrauet / mlx-forge bundles)."""
    candidates = _UPSAMPLER_WEIGHT_CANDIDATES.get(variant, _UPSAMPLER_WEIGHT_CANDIDATES["spatial_x2"])
    for fname in candidates:
        path = bundle_root / fname
        if path.is_file():
            return path
    tried = ", ".join(str(bundle_root / fname) for fname in candidates)
    raise RuntimeError(
        f"LTX 2.3 upsampler weights missing for variant={variant!r}; tried: {tried}"
    )


def load_ltx23_latent_upsampler(bundle_root: Path, *, variant: str = "spatial_x2", load_fn: Any | None = None) -> LTX23LatentUpsampler:
    key = f"{bundle_root.resolve()}:{variant}"
    if key in _upsampler_cache:
        return _upsampler_cache[key]
    path = _resolve_upsampler_weight_path(bundle_root, variant)
    config_path = path.with_name(f"{path.stem}_config.json")
    if config_path.is_file():
        up = LTX23LatentUpsampler.from_config(_load_upsampler_config(config_path))
    else:
        up = LTX23LatentUpsampler()
    raw = _strip_shared_weight_prefix(_load_bundle_weights(bundle_root, path.name, "", load_fn))
    up.load_weights(list(raw.items()), strict=False)
    _upsampler_cache[key] = up
    return up


def load_ltx23_audio_encoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX23AudioEncoder:
    key = str(bundle_root.resolve())
    if key in _audio_encoder_cache:
        return _audio_encoder_cache[key]
    enc = LTX23AudioEncoder()
    raw = _load_bundle_weights(bundle_root, "audio_vae.safetensors", "audio_vae.", load_fn)
    weights = {k[len("encoder."):]: v for k, v in raw.items() if k.startswith("encoder.")}
    for k, v in raw.items():
        if k.startswith("per_channel_statistics."):
            weights[k] = v
    enc.load_weights(list(_remap_audio_keys(weights).items()), strict=False)
    _audio_encoder_cache[key] = enc
    return enc


def load_ltx23_audio_decoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX23AudioDecoder:
    key = str(bundle_root.resolve())
    if key in _audio_decoder_cache:
        return _audio_decoder_cache[key]
    dec = LTX23AudioDecoder()
    raw = _load_bundle_weights(bundle_root, "audio_vae.safetensors", "audio_vae.", load_fn)
    weights = {k[len("decoder."):]: v for k, v in raw.items() if k.startswith("decoder.")}
    for k, v in raw.items():
        if k.startswith("per_channel_statistics."):
            weights[k] = v
    dec.load_weights(list(_remap_audio_keys(weights).items()), strict=False)
    _audio_decoder_cache[key] = dec
    return dec


def load_ltx23_vocoder(bundle_root: Path, *, load_fn: Any | None = None) -> LTX23Vocoder:
    key = str(bundle_root.resolve())
    if key in _vocoder_cache:
        return _vocoder_cache[key]
    voc = LTX23Vocoder()
    voc.load_weights(list(_load_bundle_weights(bundle_root, "vocoder.safetensors", "vocoder.", load_fn).items()))
    voc.upcast_weights_to_fp32()
    _vocoder_cache[key] = voc
    return voc


def decode_audio_latent_to_waveform(ctx: RuntimeContext, audio_latent: mx.array, bundle_root: Path) -> mx.array:
    load_fn = getattr(ctx, "load_weights", None)
    mel = load_ltx23_audio_decoder(bundle_root, load_fn=load_fn).decode(audio_latent)
    wav = load_ltx23_vocoder(bundle_root, load_fn=load_fn)(mel)
    _eval(wav)
    return wav


def encode_waveform_to_mel(
    ctx: RuntimeContext,
    waveform: mx.array,
    bundle_root: Path,
) -> mx.array:
    """Mono/stereo waveform → stereo mel ``(B, 2, T, 64)`` using vocoder MelSTFT."""
    load_fn = getattr(ctx, "load_weights", None)
    voc = load_ltx23_vocoder(bundle_root, load_fn=load_fn)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim == 2 and int(waveform.shape[0]) <= 2 and int(waveform.shape[0]) != int(waveform.shape[-1]):
        # (C, T)
        ch0 = waveform[0:1, :]
        ch1 = waveform[1:2, :] if waveform.shape[0] > 1 else ch0
    else:
        ch0 = waveform.reshape(1, -1)
        ch1 = ch0
    mel0 = voc.mel_stft(ch0)
    mel1 = voc.mel_stft(ch1)
    t = min(int(mel0.shape[1]), int(mel1.shape[1]))
    mel0 = mel0[:, :t, :]
    mel1 = mel1[:, :t, :]
    mel = mx.stack([mel0[0], mel1[0]], axis=0)
    mel = mel[None, :, :, :]
    _eval(mel)
    return mel


def encode_waveform_to_audio_latent(
    ctx: RuntimeContext,
    waveform: mx.array,
    bundle_root: Path,
) -> mx.array:
    """Waveform → audio VAE latent ``(B, 8, T', 16)``."""
    mel = encode_waveform_to_mel(ctx, waveform, bundle_root)
    load_fn = getattr(ctx, "load_weights", None)
    latent = load_ltx23_audio_encoder(bundle_root, load_fn=load_fn).encode(mel)
    _eval(latent)
    return latent


def _save_waveform(waveform: mx.array, path: str, sample_rate: int = 48000) -> None:
    import numpy as np
    wav = waveform[0]
    if wav.ndim == 2:
        nch = wav.shape[0]
        wav = wav.T
    else:
        nch = 1
        wav = wav[:, None]
    arr = np.clip(np.array(wav.astype(mx.float32)), -1.0, 1.0)
    with wave.open(path, "w") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((arr * 32767).astype(np.int16).tobytes())


def mux_video_audio_mp4(ctx: RuntimeContext, video_latent: mx.array, audio_latent: mx.array, output_path: str, bundle_root: Path, *, frame_rate: float = 24.0, on_log: Callable[[str], None] | None = None) -> str:
    load_fn = getattr(ctx, "load_weights", None)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name
    _save_waveform(decode_audio_latent_to_waveform(ctx, audio_latent, bundle_root), audio_path)
    if on_log:
        on_log(f"LTX 2.3 mux: streaming video to {output_path}")
    load_ltx23_video_decoder(bundle_root, load_fn=load_fn).decode_and_stream(video_latent, output_path, frame_rate=frame_rate, audio_path=audio_path)
    Path(audio_path).unlink(missing_ok=True)
    return output_path


def decode_latents_ncthw(ctx: RuntimeContext, latents_bcthw: mx.array, bundle_root: Path, on_stage: Callable[[float], None] | None = None, on_log: Callable[[str], None] | None = None) -> mx.array:
    if on_log:
        on_log(f"LTX 2.3 VAE decode start (latent shape {tuple(latents_bcthw.shape)})")
    if on_stage:
        on_stage(0.05)
    latents = latents_bcthw if latents_bcthw.ndim == 5 else mx.expand_dims(latents_bcthw, axis=2)
    sample = mx.clip(load_ltx23_video_decoder(bundle_root, load_fn=getattr(ctx, "load_weights", None)).decode(latents), -1.0, 1.0)
    _eval(sample)
    if on_stage:
        on_stage(1.0)
    if on_log:
        on_log(f"LTX 2.3 VAE decode done (pixel shape {tuple(sample.shape)})")
    return sample
