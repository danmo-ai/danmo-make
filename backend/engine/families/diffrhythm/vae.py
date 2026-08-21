"""
DiffRhythm 2 decoder — public MLX entry (``vae_mlx``).

DiffRhythm 2 uses a Music VAE latent (5 Hz, mel_dim=64) decoded by BigVGAN to 48 kHz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx


class DiffRhythmVAE:
    """DiffRhythm VAE — MLX-only."""

    def __init__(self, ctx: Any, vae_dir: str):
        require_mlx_ctx(ctx, feature="DiffRhythm VAE")
        self._ctx = ctx
        self._vae_dir = Path(vae_dir)
        from .vae_mlx import DiffRhythmVAEMLX

        self._vae = DiffRhythmVAEMLX(ctx, vae_dir=str(vae_dir))
        self._backend = "mlx"

    def encode(self, audio: Any) -> Any:
        """Encode audio [B, T, C] to latents [B, L, latent_dim]."""
        return self._vae.encode(audio)

    def encode_mean(self, audio: Any) -> Any:
        """Encode audio to latents (mean, no sampling)."""
        return self._vae.encode_mean(audio)

    def decode(self, latents: Any) -> Any:
        """Decode latents [B, L, latent_dim] to audio [B, T, C]."""
        return self._vae.decode(latents)
