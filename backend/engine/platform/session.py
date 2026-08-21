"""PlatformSession — device config; kernels delegate to RuntimeContext."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class PlatformSession:
    """v3 device session. Numeric work uses ``kernels`` (narrow RuntimeContext bridge)."""

    backend: Literal["mlx", "fake"]
    device: str
    dtype_name: str = "bfloat16"
    memory_limit_gb: int | None = None
    _kernels: Any = None

    @property
    def is_mlx(self) -> bool:
        return self.backend == "mlx"

    @property
    def kernels(self) -> Any:
        """Lazy kernel context — the active ``RuntimeContext``."""
        if self._kernels is None:
            raise RuntimeError(
                f"PlatformSession({self.backend!r}) has no kernel context; "
                "use platform_from_runtime() or inject kernels in tests."
            )
        return self._kernels

    def bind_kernels(self, kernels: Any) -> PlatformSession:
        self._kernels = kernels
        return self

    def gc(self) -> None:
        k = self._kernels
        if k is not None and hasattr(k, "clear_cache"):
            k.clear_cache()

    def get_memory_limit_bytes(self) -> int | None:
        if self.memory_limit_gb is None:
            return None
        return int(self.memory_limit_gb) * 1024 * 1024 * 1024


def platform_from_runtime(ctx: Any) -> PlatformSession:
    """Wrap ``RuntimeContext`` as ``PlatformSession``."""
    backend = getattr(ctx, "backend", "mlx")
    if backend != "mlx":
        raise RuntimeError(
            f"Unsupported runtime backend {backend!r}; Danmo Make is MLX-only "
            "(macOS Metal or Linux mlx[cuda])."
        )
    device = "metal" if sys.platform == "darwin" else "cuda"
    return PlatformSession(
        backend="mlx",
        device=device,
        dtype_name="bfloat16",
    ).bind_kernels(ctx)
