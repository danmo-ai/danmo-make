"""Optional mlx-mfa attention / conv dispatch — kernel layer, no hard dependency.

Pattern borrowed from mlx-mfa: conservative auto-route with explicit fallback to MLX
primitives. Families and codecs opt in via registry flags; missing mlx-mfa fails loud
only when ``attention_backend='mfa'`` is requested explicitly.
"""
from __future__ import annotations

from typing import Any, Literal

AttentionBackend = Literal["auto", "mlx", "mfa"]

_MFA: Any | None = None
_MFA_TRIED = False


def mfa_available() -> bool:
    global _MFA, _MFA_TRIED
    if _MFA_TRIED:
        return _MFA is not None
    _MFA_TRIED = True
    try:
        import mlx_mfa  # type: ignore[import-untyped]

        _MFA = mlx_mfa
    except ImportError:
        _MFA = None
    return _MFA is not None


def resolve_attention_backend(
    *,
    requested: str | None,
    head_dim: int,
    has_sliding_window: bool = False,
    has_additive_bias: bool = False,
    is_sparse: bool = False,
) -> AttentionBackend:
    """Benchmark-style routing: native MLX unless shape/features need mlx-mfa."""
    req = str(requested or "auto").strip().lower()
    if req == "mlx":
        return "mlx"
    if req == "mfa":
        if not mfa_available():
            raise RuntimeError(
                "attention_backend=mfa requested but mlx-mfa is not installed. "
                "Install mlx-mfa or set attention_backend=auto."
            )
        return "mfa"
    if is_sparse or has_additive_bias or has_sliding_window:
        return "mfa" if mfa_available() else "mlx"
    if head_dim not in (64, 128):
        return "mfa" if mfa_available() else "mlx"
    return "mlx"


def flash_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    backend: str = "auto",
    scale: float | None = None,
    mask: Any | None = None,
    attn_bias: Any | None = None,
) -> Any:
    """Dispatch dense attention to mlx-mfa or ``mx.fast.scaled_dot_product_attention``."""
    import mlx.core as mx

    head_dim = int(q.shape[-1])
    route = resolve_attention_backend(
        requested=backend,
        head_dim=head_dim,
        has_additive_bias=attn_bias is not None,
    )
    if route == "mfa" and _MFA is not None:
        fn = getattr(_MFA, "flash_attention", None)
        if callable(fn):
            kwargs: dict[str, Any] = {}
            if scale is not None:
                kwargs["scale"] = scale
            if mask is not None:
                kwargs["mask"] = mask
            if attn_bias is not None:
                kwargs["attn_bias"] = attn_bias
            return fn(q, k, v, **kwargs)
    if hasattr(mx, "fast") and hasattr(mx.fast, "scaled_dot_product_attention"):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    raise RuntimeError("No attention backend available (mlx.fast.scaled_dot_product_attention missing)")
