"""
DiffRhythm 2 text-to-music — PyTorch / CUDA path.

Upstream: ASLP-lab/DiffRhythm2 (CFM + MuQ-MuLan + G2P + BigVGAN).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.engine.families.diffrhythm.generation import (
    SAMPLE_RATE,
    assert_diffrhythm2_bundle,
    duration_to_latent_frames,
)

logger = logging.getLogger(__name__)

_NOT_INTEGRATED_MSG = (
    "DiffRhythm 2 CUDA inference is not yet integrated in Danmo Make. "
    "Required components: CFM block flow matching (sample_block_cache), "
    "MuQ-MuLan style encoder, G2P lyric tokenizer, and BigVGAN decoder. "
    "See ASLP-lab/DiffRhythm2 inference.py for the reference pipeline."
)


class DiffRhythmCudaGenerator:
    """CUDA DiffRhythm 2 generator (integration pending)."""

    def __init__(self, ctx: Any, bundle_root: Path):
        self._ctx = ctx
        self._bundle_root = Path(bundle_root)
        self._model_config: Any = None
        self.last_latent_frames: int = 0
        self.last_hum_ratio: float = 0.0
        self.last_mains_acf: float = 0.0
        self.last_latent_cos: float = 0.0
        self.last_latent_diff_mean: float = 0.0
        self.last_decode_mode: str = "bigvgan"
        self.last_quality: Any = None

    @property
    def model_config(self) -> Any:
        if self._model_config is None:
            raise RuntimeError("DiffRhythm 2 model not loaded")
        return self._model_config

    def load(self) -> None:
        import json

        bundle = assert_diffrhythm2_bundle(self._bundle_root)
        cfg_path = bundle / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            self._model_config = json.loads(f.read())

        logger.info("DiffRhythm 2 bundle validated at %s", bundle)
        raise RuntimeError(_NOT_INTEGRATED_MSG)

    def generate_waveform(
        self,
        *,
        prompt: str,
        lyrics: str,
        vocal_language: str = "en",
        duration: float = 30.0,
        steps: int = 16,
        guidance: float = 2.0,
        seed: int = 0,
        bpm: Optional[int] = None,
        key_scale: str = "",
        time_signature: str = "",
        shift: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        del (
            prompt,
            lyrics,
            vocal_language,
            steps,
            guidance,
            seed,
            bpm,
            key_scale,
            time_signature,
            shift,
            kwargs,
        )
        if self._model_config is None:
            raise RuntimeError("DiffRhythm 2 CUDA generator not loaded; call load() first")

        block_size = int(self._model_config.get("block_size", 10))
        registry_max = 210.0
        duration = max(5.0, min(registry_max, float(duration)))
        self.last_latent_frames = duration_to_latent_frames(
            duration, block_size=block_size
        )
        raise RuntimeError(_NOT_INTEGRATED_MSG)
