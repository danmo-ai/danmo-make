"""Step1X-Edit generation — family_generator entry (MLX)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.common.model.dit_stem import require_mlx_ctx
from backend.engine.config.model_configs import Step1XEditConfig


class Step1XEditGeneratorProto(Protocol):
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
        on_log: Any | None = None,
        on_progress: Any | None = None,
        cancel_token: Any | None = None,
    ) -> str: ...


def create_step1x_edit_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: Step1XEditConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> Step1XEditGeneratorProto:
    if not bundle_root.is_dir():
        raise RuntimeError(f"Step1X-Edit bundle directory not found: {bundle_root}")
    require_mlx_ctx(ctx, feature="Step1X-Edit")
    from backend.engine.families.step1x_edit.generation_mlx import Step1XEditMlxGenerator

    return Step1XEditMlxGenerator(
        ctx,
        bundle_root,
        config=config,
        entry=entry,
        version_key=version_key,
    )


def validate_image_generation_params(*, entry: Any, config: Any, **_: Any) -> None:
    """Reject unsupported Step1X variants (v1.2 ReasonEdit not implemented)."""
    _ = entry
    variant = str(getattr(config, "step1x_variant", "") or "")
    if variant:
        raise RuntimeError(
            f"Step1X-Edit variant {variant!r} is not supported. "
            "Danmo Make ships Step1X-Edit v1.1 only (stepfun-ai/Step1X-Edit bundle)."
        )


def resolve_step1x_output_path(work_dir: Path, model_key: str, seed: int) -> str:
    from backend.engine.families.step1x_edit.generation_mlx import (
        resolve_step1x_output_path as _resolve,
    )

    return _resolve(work_dir, model_key, seed)
