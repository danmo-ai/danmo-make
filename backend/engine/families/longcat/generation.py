"""LongCat-Video MLX video generation — family_generator entry."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import LongCatConfig
from backend.engine.families.longcat.generation_mlx import LongCatMlxGenerator


class LongCatGeneratorProto(Protocol):
    def load(self) -> None: ...

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        negative_prompt: str = "",
        on_log: Any | None,
        on_progress: Any | None = None,
    ) -> str: ...


def create_longcat_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: LongCatConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> LongCatGeneratorProto:
    require_mlx_ctx(ctx, feature="LongCat-Video")
    if not bundle_root.is_dir():
        raise RuntimeError(f"LongCat-Video bundle directory not found: {bundle_root}")
    return LongCatMlxGenerator(
        ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
    )


def validate_video_generation_params(
    *,
    entry: Any,
    config: Any,
    step_distill: bool,
) -> None:
    """LongCat: ``step_distill`` toggles cfg_step_lora fast path (8-step)."""
    _ = entry, config, step_distill
