"""HiDream-O1-Image MLX generation — family_generator entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.config.model_configs import HiDreamO1Config
from backend.engine.families.hidream_o1.generation_mlx import (
    HiDreamO1MlxGenerator,
    resolve_hidream_output_path,
)

__all__ = [
    "HiDreamO1GeneratorProto",
    "create_hidream_o1_generator",
    "resolve_hidream_output_path",
    "validate_image_generation_params",
]


class HiDreamO1GeneratorProto(Protocol):
    def load(self) -> None: ...

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        guidance: float,
        negative_prompt: str = "",
        ref_image_paths: list[str] | None = None,
        snap_resolution: bool = True,
        blend_seams: int = 0,
        on_log: Any | None = None,
        on_progress: Any | None = None,
        cancel_token: Any | None = None,
    ) -> str: ...


def create_hidream_o1_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: HiDreamO1Config | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> HiDreamO1GeneratorProto:
    backend = getattr(ctx, "backend", "mlx")
    if backend != "mlx":
        raise RuntimeError(
            f"HiDream-O1-Image requires MLX runtime (got {backend!r}). "
            "Select an MLX backend model version on Apple Silicon."
        )
    if not bundle_root.is_dir():
        raise RuntimeError(f"HiDream-O1 bundle directory not found: {bundle_root}")
    return HiDreamO1MlxGenerator(
        ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
    )


def validate_image_generation_params(
    *,
    entry: Any,
    config: Any,
    ref_image_paths: list[str] | None,
) -> None:
    """Fail loud when quantized HiDream rows are used for edit/multi-ref."""
    _ = entry
    if not ref_image_paths:
        return
    if bool(getattr(config, "hidream_quantized_no_edit", False)):
        raise RuntimeError(
            "This HiDream-O1 quantized variant does not support edit or multi-reference. "
            "Install the BF16 MLX version for edit workflows."
        )
