"""Per-base-model training image crop / resize (center cover + VAE grid align)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.training.dataset_store import resize_rgb_image

_MODEL_CROP: dict[str, dict[str, Any]] = {
    "flux1-dev": {
        "vae_scale": 8,
        "preset_edge": {"quick": 512, "standard": 512, "quality": 512},
    },
    "z-image": {
        "vae_scale": 8,
        "preset_edge": {"scheme4": 512, "quick": 512, "standard": 512, "quality": 512},
    },
    "z-image-turbo": {
        "vae_scale": 8,
        "preset_edge": {"quick": 512, "standard": 512, "quality": 512},
    },
    "qwen-image": {
        "vae_scale": 16,
        "preset_edge": {"quick": 512, "standard": 512, "quality": 512},
    },
}


def training_crop_policy(base_model_id: str) -> dict[str, Any]:
    mid = (base_model_id or "").split(":", 1)[0].strip()
    policy = _MODEL_CROP.get(mid)
    if policy is None:
        raise RuntimeError(f"No training crop policy for base model {mid!r}")
    return policy


def align_training_edge(px: int, *, vae_scale: int) -> int:
    edge = max(int(px), int(vae_scale))
    return (edge // int(vae_scale)) * int(vae_scale)


def resolve_training_resolution(
    base_model_id: str,
    cfg: dict[str, Any],
    *,
    preset: str | None = None,
) -> tuple[int, int]:
    """Square training size for ``base_model_id``, snapped to its VAE latent grid."""
    policy = training_crop_policy(base_model_id)
    vae_scale = int(policy["vae_scale"])
    preset_edges: dict[str, int] = policy["preset_edge"]

    res = cfg.get("resolution")
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        w, h = int(res[0]), int(res[1])
    else:
        key = (preset or "standard").strip().lower()
        edge = int(preset_edges.get(key) or preset_edges.get("standard") or 512)
        w = h = edge

    return (
        align_training_edge(w, vae_scale=vae_scale),
        align_training_edge(h, vae_scale=vae_scale),
    )


def prepare_training_rgb_image(
    path: Path,
    base_model_id: str,
    cfg: dict[str, Any],
    *,
    preset: str | None = None,
    augmentation_index: int = 0,
    allow_flip: bool = True,
    crop_window: tuple[int, int, int, int] | None = None,
) -> tuple[Any, tuple[int, int]]:
    """Cover-crop (face-biased for portraits, or an explicit face window) to model training size."""
    resolution = resolve_training_resolution(base_model_id, cfg, preset=preset)
    resize_mode = str(cfg.get("resize_mode") or "cover")
    return (
        resize_rgb_image(
            path,
            resolution,
            augmentation_index=augmentation_index,
            resize_mode=resize_mode,
            allow_flip=allow_flip,
            crop_window=crop_window,
        ),
        resolution,
    )


def training_allows_flip(dataset_meta: dict[str, Any] | None) -> bool:
    """Mirror augmentation only for non-identity datasets (``kind`` other than ``concept``)."""
    kind = str((dataset_meta or {}).get("kind") or "concept").strip().lower()
    return kind != "concept"


def presets_with_training_resolution(
    base_model_id: str,
    presets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Attach resolved ``resolution`` for UI (derived from model crop policy)."""
    out: dict[str, dict[str, Any]] = {}
    for name, body in presets.items():
        merged = dict(body)
        merged["resolution"] = list(
            resolve_training_resolution(base_model_id, merged, preset=name)
        )
        out[name] = merged
    return out
