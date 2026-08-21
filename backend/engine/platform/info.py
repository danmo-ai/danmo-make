"""GPU backend detection (MLX only: Metal on macOS, mlx[cuda] on Linux)."""

from __future__ import annotations

import importlib
import platform
import sys


class PlatformInfo:
    """GPU 后端自动检测（仅 MLX）。"""

    @staticmethod
    def detect() -> list[str]:
        """返回可用的后端列表: ["mlx"] 或 []。"""
        backends: list[str] = []

        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                importlib.import_module("mlx.core")
                backends.append("mlx")
            except ImportError:
                pass
        elif sys.platform.startswith("linux"):
            try:
                importlib.import_module("mlx.core")
                backends.append("mlx")
            except ImportError:
                pass

        return backends

    @staticmethod
    def best_available() -> str:
        """返回最佳可用后端名，无可用的则抛异常。"""
        backends = PlatformInfo.detect()
        if not backends:
            raise RuntimeError(
                "No MLX backend available "
                "(need mlx on Apple Silicon, or mlx[cuda] on Linux with NVIDIA)"
            )
        return backends[0]

    @staticmethod
    def is_apple_silicon() -> bool:
        return sys.platform == "darwin" and platform.machine() == "arm64"

    @staticmethod
    def is_linux() -> bool:
        return sys.platform.startswith("linux")

    @staticmethod
    def get_mlx_memory_stats() -> dict:
        """Return MLX GPU memory stats (active/cache/peak in GB). Empty dict if MLX unavailable."""
        try:
            mx = importlib.import_module("mlx.core")
            return {
                "active_gb": round(mx.get_active_memory() / (1024**3), 2),
                "cache_gb": round(mx.get_cache_memory() / (1024**3), 2),
                "peak_gb": round(mx.get_peak_memory() / (1024**3), 2),
            }
        except Exception:
            return {}
