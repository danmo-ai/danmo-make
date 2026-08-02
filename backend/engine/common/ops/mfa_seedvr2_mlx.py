"""Optional mlx-mfa hooks for SeedVR2 / video VAE conv3d (kernel layer)."""
from __future__ import annotations

from typing import Any, Callable


def mfa_conv3d_available() -> bool:
    try:
        import mlx_mfa._ext as ext  # type: ignore[import-untyped]

        return callable(getattr(ext, "conv3d_nax_forward", None))
    except ImportError:
        return False


def resolve_conv3d_backend(requested: str | None) -> str:
    """``auto`` → mlx-mfa when installed, else native ``mx.conv_general``."""
    req = str(requested or "auto").strip().lower()
    if req == "native":
        return "native"
    if req == "mfa":
        if not mfa_conv3d_available():
            raise RuntimeError(
                "conv3d_backend=mfa requested but mlx-mfa is not installed "
                "(missing mlx_mfa._ext.conv3d_nax_forward)."
            )
        return "mfa"
    return "mfa" if mfa_conv3d_available() else "native"


def log_conv3d_backend(
    backend: str,
    *,
    on_log: Callable[[str, str], None] | None = None,
) -> None:
    if on_log:
        on_log("info", f"SeedVR2 conv3d backend={backend}")


def causal_conv3d_forward(
    x: Any,
    *,
    weight: Any,
    bias: Any,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    backend: str,
) -> Any:
    """Dispatch 5D ``[B,C,T,H,W]`` causal conv to mlx-mfa or ``mx.conv_general``."""
    import mlx.core as mx

    route = str(backend or "native").strip().lower()
    if route == "mfa":
        if not mfa_conv3d_available():
            raise RuntimeError("conv3d_backend=mfa but mlx-mfa is not installed")
        import mlx_mfa._ext as ext  # type: ignore[import-untyped]

        fn = ext.conv3d_nax_forward
        # weight layout: (O, kt, kh, kw, I) — same as SeedVR2 CausalConv3d
        return fn(x, weight, bias, stride=stride, padding=padding)

    x_nax = x.transpose(0, 2, 3, 4, 1)
    out = mx.conv_general(
        x_nax,
        weight,
        stride=stride,
        padding=padding,
    )
    out = out + bias
    return out.transpose(0, 4, 1, 2, 3)
