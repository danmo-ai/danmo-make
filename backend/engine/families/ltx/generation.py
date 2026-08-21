"""LTX 2.3 MLX video generation — in-repo two-stage T2V/I2V orchestration.

Pipeline and :class:`VideoPipeline` import from this module for the LTX 2.3
backend (``video_pipeline_shape=family_generator``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import LTXConfig
from backend.engine.families.ltx.generation_mlx import LTX23MlxGenerator


class LTX23GeneratorProto(Protocol):
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


def create_ltx23_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: LTXConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> LTX23GeneratorProto:
    require_mlx_ctx(ctx, feature="LTX 2.3")
    if not bundle_root.is_dir():
        raise RuntimeError(f"LTX 2.3 bundle directory not found: {bundle_root}")
    return LTX23MlxGenerator(
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
    """LTX: distilled registry rows must declare ``parameters.step_distill=true``."""
    marker = str(getattr(config, "distilled_model_id_marker", "") or "")
    if not bool(getattr(config, "require_registry_step_distill_when_distilled", False)):
        return
    if not marker:
        return
    model_id = str(getattr(entry, "id", "") or "")
    if marker.lower() in model_id.lower() and not step_distill:
        raise RuntimeError(
            f"Model {entry.id!r} requires registry parameters.step_distill=true for LTX "
            "distilled sigma scheduling; run `make sync-models-registry` to refresh workspace config."
        )
