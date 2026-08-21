"""
RuntimeContext — backend-agnostic tensor operation context.

**窄契约（治理）**：此处只承载「模块工厂 + 张量/内存 API + 权重 I/O」等跨后端共性。
新增算子/算法能力**默认不进**本 ABC；平台实现见各组件 ``xxx_mlx.py``
（``docs/engine_architecture.md`` §4），``xxx.py`` 保留公共路径或对外接口。

``backend/engine/runtime/mlx.py`` 绑定 ``mlx``。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class RuntimeContext(ABC):
    """Backend-agnostic tensor operation context."""

    backend: ClassVar[str]  # "mlx" | "cuda"

    # ------------------------------------------------------------------
    # Module factories
    # ------------------------------------------------------------------

    @abstractmethod
    def Linear(self, in_features: int, out_features: int, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def LayerNorm(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def RMSNorm(self, dims: int, eps: float = 1e-6) -> Any:
        ...

    @abstractmethod
    def GroupNorm(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                  pytorch_compatible: bool = False) -> Any:
        ...

    @abstractmethod
    def SiLU(self) -> Any:
        ...

    @abstractmethod
    def GELU(self, approximate: str = "tanh") -> Any:
        ...

    @abstractmethod
    def Embedding(self, num_embeddings: int, dim: int) -> Any:
        ...

    @abstractmethod
    def Conv1d(self, in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def Conv2d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def Conv3d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def Sequential(self, *layers) -> Any:
        ...

    @abstractmethod
    def ModuleList(self, layers: list) -> Any:
        ...

    @abstractmethod
    def Dropout(self, p: float = 0.0) -> Any:
        ...

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    @abstractmethod
    def zeros(self, shape: tuple, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def ones(self, shape: tuple, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def full(self, shape: tuple, value: float, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def randn(self, shape: tuple, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def seeded_randn(self, shape: tuple, seed: int, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def seed_random(self, seed: int) -> None:
        """Seed the global RNG for reproducible random sequences."""
        ...

    @abstractmethod
    def conv2d(self, x: Any, weight: Any, stride: int = 1, padding: int = 0) -> Any:
        ...

    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def expand_dims(self, x: Any, axis: int) -> Any:
        ...

    @abstractmethod
    def squeeze(self, x: Any, axis: int) -> Any:
        ...

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        ...

    @abstractmethod
    def ones_like(self, x: Any) -> Any:
        ...

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    @abstractmethod
    def concat(self, tensors: list, axis: int = 0) -> Any:
        ...

    @abstractmethod
    def stack(self, tensors: list, axis: int = 0) -> Any:
        ...

    @abstractmethod
    def where(self, cond: Any, x: Any, y: Any) -> Any:
        ...

    @abstractmethod
    def reshape(self, x: Any, shape: tuple) -> Any:
        ...

    @abstractmethod
    def permute(self, x: Any, dims: tuple) -> Any:
        ...

    @abstractmethod
    def flip(self, x: Any, axis: int = 0) -> Any:
        """Reverse ``x`` along ``axis`` (1D schedules use axis 0)."""
        ...

    @abstractmethod
    def sin(self, x: Any) -> Any:
        ...

    @abstractmethod
    def cos(self, x: Any) -> Any:
        ...

    @abstractmethod
    def exp(self, x: Any) -> Any:
        ...

    @abstractmethod
    def log(self, x: Any) -> Any:
        ...

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        ...

    @abstractmethod
    def einsum(self, equation: str, *operands) -> Any:
        ...

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def mul(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def div(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def softmax(self, x: Any, axis: int = -1) -> Any:
        ...

    @abstractmethod
    def silu(self, x: Any) -> Any:
        ...

    @abstractmethod
    def tanh(self, x: Any) -> Any:
        ...

    @abstractmethod
    def split(self, x: Any, indices: list, axis: int = -1) -> list[Any]:
        ...

    @abstractmethod
    def broadcast_to(self, x: Any, shape: tuple) -> Any:
        ...

    @abstractmethod
    def outer(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def max(self, x: Any) -> Any:
        ...

    @abstractmethod
    def sum(self, x: Any, axis: Any = None) -> Any:
        ...

    @abstractmethod
    def square(self, x: Any) -> Any:
        ...

    @abstractmethod
    def rsqrt(self, x: Any) -> Any:
        ...

    @abstractmethod
    def power(self, base: Any, exponent: Any) -> Any:
        ...

    @abstractmethod
    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        ...

    @abstractmethod
    def meshgrid(self, *arrays, indexing: str = "ij") -> Any:
        ...

    @abstractmethod
    def repeat(self, x: Any, repeats: int, axis: int = 0) -> Any:
        ...

    @abstractmethod
    def linspace(self, start: float, end: float, steps: int, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def dequantize(self, weight: Any, scales: Any, biases: Any, group_size: int, bits: int) -> Any:
        ...

    @abstractmethod
    def is_tensor(self, x: Any) -> bool:
        ...

    @abstractmethod
    def is_integer_dtype_tensor(self, x: Any) -> bool:
        """True if ``x`` is a backend tensor with an integral dtype (e.g. timestep index)."""
        ...

    @abstractmethod
    def cast(self, x: Any, dtype: Any) -> Any:
        """Cast ``x`` to ``dtype`` (``astype`` on MLX, ``to`` on torch)."""
        ...

    @abstractmethod
    def to_numpy(self, x: Any) -> Any:
        ...

    @abstractmethod
    def gelu(self, x: Any, approximate: str = "none") -> Any:
        ...

    # ------------------------------------------------------------------
    # Differentiation / evaluation / compilation (MLX-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def eval(self, *arrays) -> None:
        ...

    @abstractmethod
    def compile(self, fn, *args, **kwargs) -> Any:
        ...

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    @abstractmethod
    def attention(self, q: Any, k: Any, v: Any, scale: float | None = None,
                  mask: Any | None = None) -> Any:
        ...

    @abstractmethod
    def interpolate(self, x: Any, scale_factor: float | tuple, mode: str = "bilinear") -> Any:
        ...

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    @abstractmethod
    def clear_cache(self) -> None:
        ...

    @abstractmethod
    def active_memory_gb(self) -> float:
        ...

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    @abstractmethod
    def load_weights(self, path: str) -> dict:
        ...

    @abstractmethod
    def save_weights(self, weights: dict, path: str) -> None:
        ...

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------

    @abstractmethod
    def float32(self) -> Any:
        ...

    @abstractmethod
    def float64(self) -> Any:
        ...

    @abstractmethod
    def float16(self) -> Any:
        ...

    @abstractmethod
    def bfloat16(self) -> Any:
        ...

    @abstractmethod
    def int32(self) -> Any:
        ...

    @abstractmethod
    def int64(self) -> Any:
        ...

    @abstractmethod
    def bool_(self) -> Any:
        ...

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def platform(self) -> str:
        return self.backend
