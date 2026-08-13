"""LTX-2.5 pipeline math — sigma schedules, latent states, bundle config (MLX only).

LTX-2.5 shares the latent/patchifier/conditioning layout with the in-repo LTX 2.3
family; those helpers are reused from ``backend.engine.families.ltx.pipeline_math_mlx``.
This module adds the 2.5-specific distilled schedule (ancestral stage-1) and the
``bundle_config.json`` loader consumed by every 2.5 model builder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.engine.families.ltx.pipeline_math_mlx import (  # noqa: F401
    AUDIO_DOWNSAMPLE_FACTOR,
    AUDIO_HOP_LENGTH,
    AUDIO_LATENTS_PER_SECOND,
    AUDIO_SAMPLE_RATE,
    VIDEO_SPATIAL_SCALE,
    VIDEO_TEMPORAL_SCALE,
    AudioPatchifier,
    LatentState,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    apply_conditioning,
    apply_denoise_mask,
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_latent_shape,
    compute_video_positions,
    create_noised_state,
    pin_latent_by_mask,
)

# LTX-2.5 distilled schedule (identical values to 2.3 upstream; stage-2 is the
# subset used for the 2x spatial refinement pass).
DISTILLED_SIGMAS: list[float] = [
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
]

STAGE_2_SIGMAS: list[float] = [
    0.909375,
    0.725,
    0.421875,
    0.0,
]

# LTX-2.5 (>= 2.5) samples stage 1 with the ancestral (SDE) Euler sampler.
# eta=1 injects the full variance-preserving noise amount at every step.
ANCESTRAL_ETA = 1.0
ANCESTRAL_S_NOISE = 1.0
ANCESTRAL_NOISE_SEED_OFFSET = 10000

BUNDLE_CONFIG_NAME = "bundle_config.json"


def load_bundle_config(bundle_root: Path) -> dict[str, Any]:
    """Load ``bundle_config.json`` produced by ``ingest.py`` (fail loud if absent)."""
    path = Path(bundle_root) / BUNDLE_CONFIG_NAME
    if not path.is_file():
        raise RuntimeError(
            f"LTX 2.5 bundle config missing: {path}. "
            "Run the LTX-2.5 ingest/converter (backend.engine.families.ltx25.ingest_mlx) "
            "to produce a valid MLX bundle."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"LTX 2.5 bundle config unreadable: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"LTX 2.5 bundle config must be a JSON object: {path}")
    return data


def get_transformer_config(bundle_root: Path) -> dict[str, Any]:
    cfg = load_bundle_config(bundle_root).get("transformer", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("LTX 2.5 bundle config lacks 'transformer' section.")
    return cfg


def get_vae_config(bundle_root: Path) -> dict[str, Any]:
    cfg = load_bundle_config(bundle_root).get("vae", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("LTX 2.5 bundle config lacks 'vae' section.")
    return cfg


def get_audio_config(bundle_root: Path) -> dict[str, Any]:
    cfg = load_bundle_config(bundle_root).get("audio_vae", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("LTX 2.5 bundle config lacks 'audio_vae' section.")
    return cfg


def get_vocoder_config(bundle_root: Path) -> dict[str, Any]:
    cfg = load_bundle_config(bundle_root).get("vocoder", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("LTX 2.5 bundle config lacks 'vocoder' section.")
    return cfg


def get_upsampler_config(bundle_root: Path) -> dict[str, Any]:
    cfg = load_bundle_config(bundle_root).get("upsampler", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("LTX 2.5 bundle config lacks 'upsampler' section.")
    return cfg


def get_gemma_config(bundle_root: Path) -> dict[str, Any]:
    cfg = load_bundle_config(bundle_root).get("gemma", {})
    if not isinstance(cfg, dict):
        raise RuntimeError("LTX 2.5 bundle config lacks 'gemma' section.")
    return cfg
