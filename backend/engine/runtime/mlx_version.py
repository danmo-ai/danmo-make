"""Minimum MLX version for Danmo Make (Metal and mlx[cuda])."""
from __future__ import annotations

MIN_MLX_VERSION: tuple[int, int, int] = (0, 32, 3)


def parse_mlx_version(raw: str) -> tuple[int, int, int]:
    """Parse ``0.32.3`` / ``0.32.3.dev`` / ``0.32`` into a 3-tuple."""
    core = str(raw or "").strip().split("+", 1)[0]
    nums: list[int] = []
    for part in core.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        nums.append(int(digits))
    if not nums:
        raise RuntimeError(f"Cannot parse MLX version {raw!r}.")
    while len(nums) < 3:
        nums.append(0)
    return (nums[0], nums[1], nums[2])


def mlx_version_string() -> str:
    """Return the installed MLX distribution version.

    Linux wheels ship ``mlx`` as a namespace package without ``mlx.__version__``.
    The distribution metadata is the source of truth on both platforms.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("mlx")
    except PackageNotFoundError:
        raw = ""
    if raw:
        return str(raw)
    import mlx

    raw = str(getattr(mlx, "__version__", "") or "")
    if not raw:
        raise RuntimeError(
            "MLX is imported but its version is missing. "
            "Reinstall mlx>=0.32.3 (macOS) or mlx[cuda]>=0.32.3 (Linux)."
        )
    return raw


def installed_mlx_version() -> tuple[int, int, int]:
    return parse_mlx_version(mlx_version_string())


def require_mlx_version(minimum: tuple[int, int, int] = MIN_MLX_VERSION) -> str:
    """Fail loud when the installed MLX build is older than ``minimum``.

    MiniMax-H3 packed sequences exceed 32K rows. MLX 0.32.3 fixes quantized
    matmul past that length and makes compiled sigmoid match eager.
    """
    raw = mlx_version_string()
    got = parse_mlx_version(raw)
    if got < minimum:
        need = ".".join(str(p) for p in minimum)
        raise RuntimeError(
            f"MLX {raw or got} is below the required {need}. "
            "Upgrade with pip install -U 'mlx>=0.32.3' (macOS) or "
            "'mlx[cuda]>=0.32.3' (Linux)."
        )
    return raw or ".".join(str(p) for p in got)
