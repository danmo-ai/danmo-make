"""Boogu-Image generation — family_generator entry (MLX + CUDA)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from backend.engine.config.model_configs import BooguImageConfig


class BooguImageGeneratorProto(Protocol):
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


def create_boogu_image_generator(
    ctx: Any,
    bundle_root: Path,
    *,
    config: BooguImageConfig | None = None,
    entry: Any | None = None,
    version_key: str | None = None,
) -> BooguImageGeneratorProto:
    if not bundle_root.is_dir():
        raise RuntimeError(f"Boogu-Image bundle directory not found: {bundle_root}")
    backend = getattr(ctx, "backend", "mlx")
    if backend == "mlx":
        from backend.engine.families.boogu.generation_mlx import BooguImageMlxGenerator

        return BooguImageMlxGenerator(
            ctx,
            bundle_root,
            config=config,
            entry=entry,
            version_key=version_key,
        )
    if backend == "cuda":
        from backend.engine.families.boogu.generation_cuda import BooguImageCudaGenerator

        return BooguImageCudaGenerator(
            ctx,
            bundle_root,
            config=config,
            entry=entry,
            version_key=version_key,
        )
    raise RuntimeError(
        f"Boogu-Image requires mlx or cuda runtime (got {backend!r}). "
        "Select a compatible backend model version."
    )


def validate_image_generation_params(
    *,
    entry: Any,
    config: Any,
    ref_image_paths: list[str] | None = None,
    **_: Any,
) -> None:
    variant = str(getattr(config, "boogu_variant", "turbo") or "turbo").lower()
    is_edit = bool(ref_image_paths)
    if variant == "turbo" and is_edit:
        raise RuntimeError(
            "Boogu-Image-Turbo supports text-to-image only. "
            "Use boogu-image-edit for instruction-based editing."
        )
    if variant == "edit" and not is_edit:
        raise RuntimeError(
            "Boogu-Image-Edit requires a source image (rewrite/edit action)."
        )


def resolve_boogu_output_path(work: Path, model_key: str, seed: int) -> str:
    from backend.engine.families.boogu.generation_mlx import (
        resolve_boogu_output_path as _resolve,
    )

    return _resolve(work, model_key, seed)
