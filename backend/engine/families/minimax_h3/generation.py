"""MiniMax-H3 FL2VA MLX video generation — family_generator entry."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import MinimaxH3Config
from backend.engine.families.minimax_h3.generation_mlx import MinimaxH3MlxGenerator


class MinimaxH3GeneratorProto(Protocol):
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
        last_frame_path: str | None = None,
        negative_prompt: str = "",
        on_log: Any | None,
        on_progress: Any | None = None,
    ) -> str: ...


def create_minimax_h3_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: MinimaxH3Config | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> MinimaxH3GeneratorProto:
    require_mlx_ctx(ctx, feature="MiniMax-H3")
    if not bundle_root.is_dir():
        raise RuntimeError(f"MiniMax-H3 bundle directory not found: {bundle_root}")
    return MinimaxH3MlxGenerator(
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
    """MiniMax-H3 FL2VA: CFG-distilled — guidance disabled; no step_distill path."""
    _ = entry, step_distill
    if bool(getattr(config, "supports_guidance", False)):
        raise RuntimeError(
            "MiniMax-H3 FL2VA checkpoints are CFG-distilled; "
            "registry/runtime must set supports_guidance=false."
        )
