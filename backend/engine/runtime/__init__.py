"""Runtime backends (MLX) and MLX-side helpers behind :class:`RuntimeContext`."""

from __future__ import annotations

from ._base import RuntimeContext

__all__ = [
    "RuntimeContext",
    "MLXContext",
    "cast_floating_mx_tree",
    "cast_module_parameters",
    "load_weights_dict",
    "run_eval",
]


def __getattr__(name: str):
    if name == "MLXContext":
        from .mlx import MLXContext as _MLXContext

        return _MLXContext
    if name in ("run_eval", "load_weights_dict"):
        from . import mlx_runtime as _mr

        return getattr(_mr, name)
    if name in ("cast_floating_mx_tree", "cast_module_parameters"):
        from . import mlx_dtype as _md

        return getattr(_md, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
