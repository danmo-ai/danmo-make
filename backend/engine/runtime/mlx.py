"""
MLX Runtime — Reference implementation.
"""
from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ._base import RuntimeContext

_mlx_dtype_map = {
    16: mx.float16,
    "float16": mx.float16,
    "fp16": mx.float16,
    32: mx.float32,
    "float32": mx.float32,
    "fp32": mx.float32,
    "bf16": mx.bfloat16,
    "bfloat16": mx.bfloat16,
}


class _MLXModuleList(nn.Module):
    """PyTorch-like ModuleList: registers children so ``parameters()`` includes nested modules."""

    def __init__(self, layers: list):
        super().__init__()
        self._layers = list(layers)
        for i, layer in enumerate(self._layers):
            setattr(self, str(i), layer)

    def __getitem__(self, idx: int):
        return self._layers[idx]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)


class MLXContext(RuntimeContext):
    backend = "mlx"

    def __init__(self, memory_limit_gb: int = 120):
        os.environ.setdefault("MLX_METAL_DEVICE_ONLY", "1")
        os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        from backend.engine.memory_policy import clamp_mlx_memory_limit_gb

        self._memory_limit_gb = clamp_mlx_memory_limit_gb(memory_limit_gb)
        self.apply_memory_limit_gb(self._memory_limit_gb)

    @property
    def memory_limit_gb(self) -> int:
        return self._memory_limit_gb

    def apply_memory_limit_gb(self, gb: int) -> int:
        from backend.engine.memory_policy import clamp_mlx_memory_limit_gb

        self._memory_limit_gb = clamp_mlx_memory_limit_gb(gb)
        os.environ["MLX_METAL_MEMORY_LIMIT"] = str(self._memory_limit_gb)
        try:
            mx.set_memory_limit(self._memory_limit_gb * 1024**3)
        except Exception:
            pass
        return self._memory_limit_gb

    # ------------------------------------------------------------------
    # Module factories
    # ------------------------------------------------------------------

    def Linear(self, in_features: int, out_features: int, bias: bool = True) -> Any:
        return nn.Linear(in_features, out_features, bias=bias)

    def LayerNorm(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True) -> Any:
        return nn.LayerNorm(dims, eps=eps, affine=affine, bias=bias)

    def RMSNorm(self, dims: int, eps: float = 1e-6) -> Any:
        return nn.RMSNorm(dims, eps=eps)

    def GroupNorm(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                  pytorch_compatible: bool = False) -> Any:
        return nn.GroupNorm(num_groups, num_channels, eps=eps,
                            pytorch_compatible=pytorch_compatible)

    def SiLU(self) -> Any:
        return nn.SiLU()

    def GELU(self, approximate: str = "tanh") -> Any:
        return nn.GELU(approximate)

    def Embedding(self, num_embeddings: int, dim: int) -> Any:
        return nn.Embedding(num_embeddings, dim)

    def Conv1d(self, in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(stride, int):
            stride = (stride,)
        if isinstance(padding, int):
            padding = (padding,)
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def Conv2d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def Conv3d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        # MLX Conv3d currently limited; fallback via Sequential if needed
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def Sequential(self, *layers) -> Any:
        return nn.Sequential(*layers)

    def ModuleList(self, layers: list) -> Any:
        return _MLXModuleList(layers)

    def Dropout(self, p: float = 0.0) -> Any:
        return nn.Dropout(p)

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple, dtype: Any = None) -> Any:
        return mx.zeros(shape, dtype=dtype or mx.float32)

    def ones(self, shape: tuple, dtype: Any = None) -> Any:
        return mx.ones(shape, dtype=dtype or mx.float32)

    def full(self, shape: tuple, value: float, dtype: Any = None) -> Any:
        return mx.full(shape, value, dtype=dtype or mx.float32)

    def arange(self, start: int, end: int | None = None, step: int = 1, dtype: Any = None) -> Any:
        if end is None:
            return mx.arange(0, start, step, dtype=dtype or mx.int32)
        return mx.arange(start, end, step, dtype=dtype or mx.int32)

    def randn(self, shape: tuple, dtype: Any = None) -> Any:
        return mx.random.normal(shape, dtype=dtype or mx.float32)

    def seeded_randn(self, shape: tuple, seed: int, dtype: Any = None) -> Any:
        return mx.random.normal(shape, dtype=dtype or mx.float32, key=mx.random.key(seed))

    def seed_random(self, seed: int) -> None:
        mx.random.seed(int(seed))

    def conv2d(self, x: Any, weight: Any, stride: int = 1, padding: int = 0) -> Any:
        return mx.conv2d(x, weight, stride=stride, padding=padding)

    def array(self, data: Any, dtype: Any = None) -> Any:
        return mx.array(data, dtype=dtype or mx.float32)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return mx.expand_dims(x, axis=axis)

    def squeeze(self, x: Any, axis: int) -> Any:
        return mx.squeeze(x, axis=axis)

    def zeros_like(self, x: Any) -> Any:
        return mx.zeros_like(x)

    def ones_like(self, x: Any) -> Any:
        return mx.ones_like(x)

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    def concat(self, tensors: list, axis: int = 0) -> Any:
        return mx.concatenate(tensors, axis=axis)

    def stack(self, tensors: list, axis: int = 0) -> Any:
        return mx.stack(tensors, axis=axis)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return mx.where(cond, x, y)

    def reshape(self, x: Any, shape: tuple) -> Any:
        return mx.reshape(x, shape)

    def permute(self, x: Any, dims: tuple) -> Any:
        return mx.transpose(x, dims)

    def flip(self, x: Any, axis: int = 0) -> Any:
        ndim = len(x.shape)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"flip axis {axis} out of range for ndim {ndim}")
        sl = [slice(None)] * ndim
        sl[axis] = slice(None, None, -1)
        return x[tuple(sl)]

    def sin(self, x: Any) -> Any:
        return mx.sin(x)

    def cos(self, x: Any) -> Any:
        return mx.cos(x)

    def exp(self, x: Any) -> Any:
        return mx.exp(x)

    def log(self, x: Any) -> Any:
        return mx.log(x)

    def sqrt(self, x: Any) -> Any:
        return mx.sqrt(x)

    def einsum(self, equation: str, *operands) -> Any:
        return mx.einsum(equation, *operands)

    def matmul(self, a: Any, b: Any) -> Any:
        return a @ b

    def mul(self, a: Any, b: Any) -> Any:
        return a * b

    def div(self, a: Any, b: Any) -> Any:
        return a / b

    def softmax(self, x: Any, axis: int = -1) -> Any:
        return mx.softmax(x, axis=axis)

    def silu(self, x: Any) -> Any:
        return nn.silu(x)

    def tanh(self, x: Any) -> Any:
        return mx.tanh(x)

    def split(self, x: Any, indices: list, axis: int = -1) -> list[Any]:
        return mx.split(x, indices, axis=axis)

    def broadcast_to(self, x: Any, shape: tuple) -> Any:
        return mx.broadcast_to(x, shape)

    def outer(self, a: Any, b: Any) -> Any:
        return mx.outer(a, b)

    def max(self, x: Any) -> Any:
        return mx.max(x)

    def sum(self, x: Any, axis: Any = None) -> Any:
        return mx.sum(x, axis=axis) if axis is not None else mx.sum(x)

    def square(self, x: Any) -> Any:
        return mx.square(x)

    def rsqrt(self, x: Any) -> Any:
        return mx.rsqrt(x)

    def power(self, base: Any, exponent: Any) -> Any:
        return mx.power(base, exponent)

    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return mx.mean(x, axis=axis, keepdims=keepdims)

    def gelu(self, x: Any, approximate: str = "none") -> Any:
        return nn.gelu(x)

    def meshgrid(self, *arrays, indexing: str = "ij") -> Any:
        return mx.meshgrid(*arrays, indexing=indexing)

    def repeat(self, x: Any, repeats: int, axis: int = 0) -> Any:
        return mx.repeat(x, repeats, axis=axis)

    def linspace(self, start: float, end: float, steps: int, dtype: Any = None) -> Any:
        return mx.linspace(start, end, steps, dtype=dtype or mx.float32)

    def dequantize(self, weight: Any, scales: Any, biases: Any, group_size: int, bits: int) -> Any:
        return mx.dequantize(weight, scales, biases, group_size, bits)

    def is_tensor(self, x: Any) -> bool:
        return isinstance(x, mx.array)

    def is_integer_dtype_tensor(self, x: Any) -> bool:
        if not isinstance(x, mx.array):
            return False
        return x.dtype in (mx.int32, mx.int64)

    def cast(self, x: Any, dtype: Any) -> Any:
        if isinstance(x, mx.array):
            return x.astype(dtype)
        return mx.array(x, dtype=dtype)

    def to_numpy(self, x: Any) -> Any:
        return x.tolist() if isinstance(x, mx.array) else x

    # ------------------------------------------------------------------
    # Differentiation / evaluation / compilation
    # ------------------------------------------------------------------

    def eval(self, *arrays) -> None:
        mx.eval(*arrays)

    def compile(self, fn, *args, **kwargs) -> Any:
        return mx.compile(fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    def attention(self, q: Any, k: Any, v: Any, scale: float | None = None,
                  mask: Any | None = None) -> Any:
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask,
        )

    def interpolate(self, x: Any, scale_factor: float | tuple, mode: str = "bilinear") -> Any:
        if mode == "bilinear" and x.ndim == 4:
            if isinstance(scale_factor, (int, float)):
                sf = float(scale_factor)
                sf_tuple = (sf, sf)
            else:
                sf_tuple = (float(scale_factor[0]), float(scale_factor[1]))
            x_hwc = mx.transpose(x, (0, 2, 3, 1))
            y_hwc = nn.Upsample(scale_factor=sf_tuple, mode="linear")(x_hwc)
            return mx.transpose(y_hwc, (0, 3, 1, 2))
        raise NotImplementedError(f"interpolate mode={mode} ndim={x.ndim} not supported for MLX")

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        mx.clear_cache()

    def active_memory_gb(self) -> float:
        try:
            return mx.get_active_memory() / (1024 ** 3)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> dict:
        return dict(mx.load(path))

    def save_weights(self, weights: dict, path: str) -> None:
        mx.save_safetensors(path, weights)

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------

    def float32(self) -> Any:
        return mx.float32

    def float64(self) -> Any:
        return mx.float64

    def float16(self) -> Any:
        return mx.float16

    def bfloat16(self) -> Any:
        return mx.bfloat16

    def int32(self) -> Any:
        return mx.int32

    def int64(self) -> Any:
        return mx.int64

    def bool_(self) -> Any:
        return mx.bool_
