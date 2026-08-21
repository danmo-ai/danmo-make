"""Wan-family Shape C video generators (Bernini-R renderer)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import WanConfig


class BerniniRendererProto(Protocol):
    def load(self) -> None: ...

    def generate_and_save(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        source_video_path: str | None,
        source_image_path: str | None,
        reference_image_paths: list[str],
        is_edit: bool,
        on_log: Any | None,
        on_progress: Any | None = None,
    ) -> str: ...


def create_bernini_renderer_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: WanConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
    project_root: Path | None = None,
) -> BerniniRendererProto:
    require_mlx_ctx(ctx, feature="Bernini-R renderer")
    if not bundle_root.is_dir():
        raise RuntimeError(f"Bernini-R bundle directory not found: {bundle_root}")
    cfg = config or WanConfig()
    if not bool(getattr(cfg, "bernini_renderer", False)):
        raise RuntimeError(
            "Wan family_generator factory requires bernini_renderer=true in registry overrides."
        )
    from backend.engine.families.wan.bernini_renderer_mlx import BerniniRendererMLX

    return BerniniRendererMLX(
        ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
        project_root=project_root,
    )


def validate_video_generation_params(
    *,
    entry: Any,
    config: Any,
    step_distill: bool,
) -> None:
    """Bernini-R does not use Wan Lightning 4-step distill schedules."""
    del entry, config, step_distill
