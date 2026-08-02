"""Fail-loud guards for MLX-only engine features (no silent ignore on CUDA)."""

from __future__ import annotations

from typing import Any


def require_mlx_backend(ctx: Any, *, feature: str) -> None:
    backend = str(getattr(ctx, "backend", "mlx") or "mlx")
    if backend != "mlx":
        raise RuntimeError(
            f"{feature} is MLX-only today (backend={backend!r}). "
            "Use Apple Silicon with MLX runtime, or disable this option."
        )


def require_mlx_if_option_active(
    ctx: Any,
    *,
    feature: str,
    option: Any,
    inactive_values: frozenset[str] = frozenset({"", "none", "off", "0", "false"}),
) -> None:
    if option is None:
        return
    raw = str(option).strip().lower()
    if raw in inactive_values:
        return
    require_mlx_backend(ctx, feature=feature)
