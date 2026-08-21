"""LTX-2.5 (22B) MLX video generation — two-stage distilled T2V/I2V + audio mux.

Pipeline and :class:`VideoPipeline` import from this module for the LTX 2.5
backend (``video_pipeline_shape=family_generator``). MLX only; CUDA fails loud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import LTX25Config
from backend.engine.families.ltx25.generation_mlx import LTX25MlxGenerator


class LTX25GeneratorProto(Protocol):
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


def create_ltx25_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: LTX25Config | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> LTX25GeneratorProto:
    require_mlx_ctx(ctx, feature="LTX 2.5")
    if not bundle_root.is_dir():
        raise RuntimeError(f"LTX 2.5 bundle directory not found: {bundle_root}")
    return LTX25MlxGenerator(
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
    """LTX-2.5: only the distilled checkpoint is supported (fail loud otherwise)."""
    if step_distill:
        return
    model_id = str(getattr(entry, "id", "") or "")
    raise RuntimeError(
        f"LTX 2.5 model {model_id!r} requires step_distill=true: only the distilled "
        "8-step pipeline is implemented (the dev 22B guided pipeline is not supported). "
        "Run `make sync-models-registry` and select the distilled version."
    )
