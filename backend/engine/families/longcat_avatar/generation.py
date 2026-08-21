"""LongCat-Video-Avatar MLX — family_avatar entry."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import LongCatAvatarConfig
from backend.engine.families.longcat_avatar.generation_mlx import LongCatAvatarMlxGenerator


class LongCatAvatarGeneratorProto(Protocol):
    def load(self) -> None: ...

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        reference_image_path: str,
        audio_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        negative_prompt: str = "",
        on_log: Any | None,
        on_progress: Any | None = None,
    ) -> str: ...


def create_longcat_avatar_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: LongCatAvatarConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> LongCatAvatarGeneratorProto:
    require_mlx_ctx(ctx, feature="LongCat-Avatar")
    if not bundle_root.is_dir():
        raise RuntimeError(f"LongCat-Avatar bundle directory not found: {bundle_root}")
    return LongCatAvatarMlxGenerator(
        ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
    )


def validate_video_avatar_params(*, entry: Any, config: Any) -> None:
    _ = entry, config
