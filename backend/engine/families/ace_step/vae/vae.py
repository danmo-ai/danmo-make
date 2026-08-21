"""
ACE-Step VAE — MLX audio VAE wrapper.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.engine.common.model.dit_stem import require_mlx_ctx


def _vae_kwargs_from_bundle(vae_dir: str) -> dict[str, Any]:
    """Read ``vae/config.json`` so MLX layout matches the Oobleck checkpoint."""
    cfg_path = Path(vae_dir) / "config.json"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[str, Any] = {}
    if "encoder_hidden_size" in raw:
        out["encoder_hidden_size"] = raw["encoder_hidden_size"]
    if "downsampling_ratios" in raw:
        out["downsampling_ratios"] = raw["downsampling_ratios"]
    if "channel_multiples" in raw:
        out["channel_multiples"] = raw["channel_multiples"]
    if "decoder_channels" in raw:
        out["decoder_channels"] = raw["decoder_channels"]
    if "decoder_input_channels" in raw:
        out["decoder_input_channels"] = raw["decoder_input_channels"]
    if "audio_channels" in raw:
        out["audio_channels"] = raw["audio_channels"]
    return out


class AceStepVAE:
    """Audio VAE wrapper (MLX-only)."""

    def __init__(self, ctx: Any, **kwargs):
        require_mlx_ctx(ctx, feature="ACE-Step VAE")
        vae_dir = kwargs.get("vae_dir", "")
        self._backend = "mlx"

        from .vae_mlx import AceStepVAEMLX, load_vae_weights_from_bundle

        mlx_kwargs = _vae_kwargs_from_bundle(vae_dir) if vae_dir else {}
        for k in (
            "encoder_hidden_size",
            "downsampling_ratios",
            "channel_multiples",
            "decoder_channels",
            "decoder_input_channels",
            "audio_channels",
        ):
            if k in kwargs:
                mlx_kwargs[k] = kwargs[k]
        self._vae = AceStepVAEMLX(**mlx_kwargs)
        if vae_dir:
            eval_fn = getattr(ctx, "eval", None)
            array_fn = getattr(ctx, "array", None)
            load_vae_weights_from_bundle(
                vae_dir, self._vae, eval_fn=eval_fn, array_fn=array_fn
            )

    def encode(self, audio) -> Any:
        return self._vae.encode(audio)

    def decode(self, latents) -> Any:
        return self._vae.decode(latents)
