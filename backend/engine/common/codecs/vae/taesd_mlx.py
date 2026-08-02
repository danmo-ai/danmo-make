"""TAESD / TAEF1 — tiny VAE decoder for fast denoise-step previews (MLX)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.runtime._base import RuntimeContext

LATENT_MAGNITUDE = 3.0
LATENT_SHIFT = 0.5

_TAESD_VARIANTS: dict[str, dict[str, Any]] = {
    "taesd": {"latent_channels": 4, "use_midblock_gn": False},
    "taef1": {"latent_channels": 16, "use_midblock_gn": False},
    "taef2": {"latent_channels": 32, "use_midblock_gn": True},
}


def _clamp_latents(x: mx.array) -> mx.array:
    return mx.tanh(x / 3.0) * 3.0


class _TaesdBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, *, use_midblock_gn: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in, n_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_out, n_out, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else None
        self.pool = None
        if use_midblock_gn:
            self.pool = nn.Sequential(
                nn.Conv2d(n_in, n_in * 4, 1, bias=False),
                nn.GroupNorm(4, n_in * 4),
                nn.ReLU(),
                nn.Conv2d(n_in * 4, n_in, 1, bias=False),
            )

    def __call__(self, x: mx.array) -> mx.array:
        if self.pool is not None:
            x = x + self.pool(x)
        skip = x if self.skip is None else self.skip(x)
        return nn.relu(self.conv(x) + skip)


class TaesdDecoder(nn.Module):
    """Decoder-only TAESD stack (preview path)."""

    def __init__(self, *, latent_channels: int = 16, use_midblock_gn: bool = False):
        super().__init__()
        mb = use_midblock_gn
        self.clamp = _clamp_latents
        self.stem = nn.Sequential(nn.Conv2d(latent_channels, 64, 3, padding=1), nn.ReLU())
        self.blocks = [
            _TaesdBlock(64, 64, use_midblock_gn=mb),
            _TaesdBlock(64, 64, use_midblock_gn=mb),
            _TaesdBlock(64, 64, use_midblock_gn=mb),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _TaesdBlock(64, 64),
            _TaesdBlock(64, 64),
            _TaesdBlock(64, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _TaesdBlock(64, 64),
            _TaesdBlock(64, 64),
            _TaesdBlock(64, 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            _TaesdBlock(64, 64),
            nn.Conv2d(64, 3, 3, padding=1),
        ]

    def __call__(self, latents: mx.array) -> mx.array:
        x = self.clamp(latents)
        x = self.stem[0](x)
        x = self.stem[1](x)
        for layer in self.blocks:
            if isinstance(layer, _TaesdBlock):
                x = layer(x)
            elif isinstance(layer, nn.Upsample):
                x = layer(x)
            else:
                x = layer(x)
        return mx.clip(x, 0.0, 1.0)


def resolve_taesd_weights_path(project_root: Path, variant: str = "taef1") -> Path | None:
    env = os.environ.get("DANQING_TAESD_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend([
        project_root / "models" / "taesd",
        project_root / "models" / "TAESD",
    ])
    names = [
        f"{variant}_decoder.safetensors",
        f"{variant}_decoder.pth",
        "diffusion_pytorch_model.safetensors",
        "taef1_decoder.safetensors",
        "taesd_decoder.safetensors",
    ]
    for base in candidates:
        for name in names:
            p = base / name
            if p.is_file():
                return p
    return None


def _load_decoder_weights(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        out: dict[str, Any] = {}
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                out[key] = f.get_tensor(key)
        return out
    from backend.engine.common.bundle.pytorch_bin_numpy import load_pytorch_bin_state_dict

    state = load_pytorch_bin_state_dict(path)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return {k: np.asarray(v) for k, v in state.items()}


def _remap_taesd_decoder_keys(raw: dict[str, Any]) -> dict[str, mx.array]:
    mapped: dict[str, mx.array] = {}
    for key, val in raw.items():
        if not (key.startswith("decoder.") or key[0].isdigit() or key.startswith("blocks.")):
            continue
        nk = key
        if nk.startswith("decoder."):
            nk = nk[len("decoder.") :]
        mapped[nk.replace(".conv.", ".") if ".conv." in nk else nk] = mx.array(val)
    return mapped


def load_taesd_preview_decoder(
    ctx: RuntimeContext,
    *,
    project_root: Path,
    variant: str = "taef1",
    on_log: Callable[[str, str], None] | None = None,
) -> TaesdDecoder | None:
    path = resolve_taesd_weights_path(project_root, variant=variant)
    if path is None:
        return None
    spec = _TAESD_VARIANTS.get(variant, _TAESD_VARIANTS["taef1"])
    model = TaesdDecoder(
        latent_channels=int(spec["latent_channels"]),
        use_midblock_gn=bool(spec["use_midblock_gn"]),
    )
    try:
        raw = _load_decoder_weights(path)
        weights = _remap_taesd_decoder_keys(raw)
        model.update({k: v for k, v in weights.items() if not k.startswith("_")})
    except Exception as exc:
        if on_log:
            on_log("warning", f"TAESD preview weights load failed ({path.name}): {exc}")
        return None
    if on_log:
        on_log("info", f"TAESD preview decoder ready variant={variant} path={path.name}")
    return model


def decode_taesd_preview(
    ctx: RuntimeContext,
    decoder: TaesdDecoder,
    latents: Any,
    *,
    vae_scaling_factor: float = 1.0,
    vae_shift_factor: float = 0.0,
) -> mx.array:
    z = latents
    if hasattr(z, "ndim") and int(z.ndim) == 4 and int(z.shape[0]) == 1:
        z = z[0]
    if vae_scaling_factor not in (0.0, 1.0):
        z = (z / float(vae_scaling_factor)) + float(vae_shift_factor)
    scaled = (z / (2.0 * LATENT_MAGNITUDE)) + LATENT_SHIFT
    scaled = mx.clip(scaled, 0.0, 1.0)
    out = decoder(scaled[None, ...] if scaled.ndim == 3 else scaled)
    ctx.eval(out)
    return out
