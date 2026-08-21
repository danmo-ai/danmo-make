"""DiT stem — construct MLX implementation and proxy attributes (no dual-backend dispatch)."""
from __future__ import annotations

from typing import Any, Type


def require_mlx_ctx(ctx: Any, *, feature: str = "DiT") -> None:
    """Fail loud unless ``ctx.backend`` is mlx (Metal or mlx[cuda])."""
    backend = getattr(ctx, "backend", "mlx")
    if backend != "mlx":
        raise RuntimeError(
            f"{feature} requires MLX runtime (got backend={backend!r}; "
            "use mlx on macOS or mlx[cuda] on Linux)"
        )


class DelegatingDiTStem:
    """Construct-only stem: hold ``_inner`` MLX DiT; attribute access proxies to it.

    Does **not** subclass :class:`TransformerBase` — that avoided dual-backend MRO
    gaps that forced per-hook forwarders. Product hooks live on the MLX impl
    (or on a stem subclass override, e.g. Wan).
    """

    _inner: Any

    def __init__(
        self,
        config: Any,
        ctx: Any,
        *,
        mlx_cls: Type[Any],
        **factory_kwargs: Any,
    ) -> None:
        require_mlx_ctx(ctx, feature=getattr(mlx_cls, "__name__", "DiT"))
        self._inner = mlx_cls(config, ctx, **factory_kwargs)
        self.ctx = self._inner.ctx
        self.config = self._inner.config
        self._param_map = getattr(self._inner, "_param_map", {})

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.forward(*args, **kwargs)

    def parameters(self):
        return self._inner.parameters()

    def load_weights(self, *args: Any, **kwargs: Any) -> Any:
        out = self._inner.load_weights(*args, **kwargs)
        self._param_map = getattr(self._inner, "_param_map", {})
        return out

    def _build_param_map(self) -> None:
        if hasattr(self._inner, "_build_param_map"):
            self._inner._build_param_map()
            self._param_map = getattr(self._inner, "_param_map", {})


# Back-compat alias used by a few imports / tests
def dispatch_dit_implementation(
    config: Any,
    ctx: Any,
    *,
    mlx_cls: Type[Any],
    **factory_kwargs: Any,
) -> Any:
    """Instantiate ``mlx_cls`` after mlx backend check."""
    require_mlx_ctx(ctx, feature=getattr(mlx_cls, "__name__", "DiT"))
    return mlx_cls(config, ctx, **factory_kwargs)
