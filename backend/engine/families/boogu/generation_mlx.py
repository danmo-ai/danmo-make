"""Boogu-Image MLX generation — Turbo T2I + Edit."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.common.codecs.vae import infer_latent_channels, load_vae_decoder_from_weights
from backend.engine.common.codecs.vae.decoder import (
    VAEDecoder,
    load_vae_weight_dict,
    read_vae_dir_config,
    vae_output_to_uint8_hwc,
)
from backend.engine.common.codecs.vae.encoder import VAEEncoder
from backend.engine.common.codecs.vae.weight_remap import prepare_vae_encoder_weight_items
from backend.engine.families.boogu.conditioner_mlx import BooguQwen3VLEncoderMLX
from backend.engine.families.boogu.scheduler_mlx import FlowMatchEulerDiscreteScheduler
from backend.engine.families.boogu.transformer_mlx import BooguImageDiTMLX
from backend.engine.families.boogu.weights_mlx import load_boogu_dit_mlx, resolve_boogu_bundle_dirs


def resolve_boogu_output_path(work: Path, model_key: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(work / f"{model_key}_{seed}_{ts}.png")


def _snap_dim(value: int, *, multiple: int = 16, minimum: int = 384) -> int:
    v = max(minimum, int(value))
    return max(minimum, v // multiple * multiple)


class BooguImageMlxGenerator:
    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        config: Any | None = None,
        entry: Any | None = None,
        version_key: str | None = None,
    ) -> None:
        self._ctx = ctx
        self._bundle_root = bundle_root
        self._config = config
        self._entry = entry
        self._version_key = version_key
        self._variant = str(getattr(config, "boogu_variant", "turbo") or "turbo").lower()
        self._dit: BooguImageDiTMLX | None = None
        self._vae_dec: VAEDecoder | None = None
        self._vae_enc: VAEEncoder | None = None
        self._scheduler: FlowMatchEulerDiscreteScheduler | None = None
        self._cond: BooguQwen3VLEncoderMLX | None = None
        self._vae_cfg: dict[str, Any] = {}
        self._scaling = 0.3611
        self._shift = 0.1159

    def load(self) -> None:
        dirs = resolve_boogu_bundle_dirs(self._bundle_root)
        tcfg = json.loads((dirs["transformer"] / "config.json").read_text())
        scfg = json.loads((dirs["scheduler"] / "scheduler_config.json").read_text())
        self._vae_cfg, self._scaling, self._shift = read_vae_dir_config(dirs["vae"])

        dit = BooguImageDiTMLX.from_config(tcfg)
        load_boogu_dit_mlx(dit, dirs["transformer"], dtype=mx.bfloat16)
        self._dit = dit

        vae_weights = load_vae_weight_dict(self._ctx, dirs["vae"])
        latent_c = infer_latent_channels(self._vae_cfg, vae_weights)
        # FLUX AutoencoderKL decoder uses layers_per_block+1 resnets per up stage (diffusers convention).
        decoder_layers = int(self._vae_cfg.get("layers_per_block", 2)) + 1
        dec = VAEDecoder(
            latent_channels=latent_c,
            ctx=self._ctx,
            scaling_factor=self._scaling,
            shift_factor=self._shift,
            vae_cfg=self._vae_cfg,
            layers_per_block=decoder_layers,
        )
        _decoder_w, loaded, skipped = load_vae_decoder_from_weights(dec, vae_weights)
        if not loaded:
            raise RuntimeError(
                f"Boogu-Image VAE decoder failed to load from {dirs['vae']} "
                f"(skipped_sample={skipped[:8]})"
            )
        if skipped:
            raise RuntimeError(
                f"Boogu-Image FLUX VAE decoder incomplete load from {dirs['vae']}: "
                f"skipped={len(skipped)} sample={skipped[:8]}. "
                "Expected layers_per_block+1 decoder resnets per up block."
            )
        self._vae_dec = dec

        enc = VAEEncoder(
            latent_channels=latent_c,
            ctx=self._ctx,
            scaling_factor=self._scaling,
            shift_factor=self._shift,
        )
        enc_items = prepare_vae_encoder_weight_items(vae_weights)
        enc.load_weights(enc_items, strict=False)
        self._vae_enc = enc

        self._scheduler = FlowMatchEulerDiscreteScheduler.from_config(scfg)
        self._cond = BooguQwen3VLEncoderMLX(
            mllm_dir=str(dirs["mllm"]),
            processor_dir=str(dirs["processor"]),
            dtype=mx.bfloat16,
        )
        if hasattr(self._ctx, "eval"):
            self._ctx.eval(self._dit.parameters())

    def _vae_encode_ref(self, image: Image.Image, width: int, height: int) -> mx.array:
        assert self._vae_enc is not None
        img = image.convert("RGB").resize((width, height), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        x = mx.array(arr.transpose(2, 0, 1)[None, ...])
        z5 = self._vae_enc.encode(x)
        z = z5[:, :, 0, :, :] if z5.ndim == 5 else z5
        return z.astype(mx.bfloat16)

    def _decode_latent(self, lat: mx.array) -> np.ndarray:
        assert self._vae_dec is not None
        z = lat.astype(mx.float32)
        img = self._vae_dec.forward(z)
        return vae_output_to_uint8_hwc(img, self._ctx)

    def _denoise(
        self,
        *,
        lat: mx.array,
        pos: mx.array,
        neg: mx.array | None,
        ref: mx.array | None,
        steps: int,
        guidance: float,
        height: int,
        width: int,
        on_progress: Callable | None,
        cancel_token: Any | None,
    ) -> mx.array:
        assert self._dit is not None and self._scheduler is not None
        hl, wl = height // 8, width // 8
        self._scheduler.set_timesteps(steps, num_tokens=hl * wl)
        for i in range(steps):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise RuntimeError("Cancelled")
            t = mx.array([float(self._scheduler.timesteps[i])], dtype=pos.dtype)
            pred = self._dit(lat, t, pos, ref_latent=ref)
            if neg is not None and guidance > 1.0:
                pu = self._dit(lat, t, neg, ref_latent=ref)
                pred = pu + guidance * (pred - pu)
            lat = self._scheduler.step(pred, i, lat)
            mx.eval(lat)
            mx.clear_cache()
            if on_progress is not None:
                on_progress((i + 1) / steps, i + 1, steps, f"denoise {i + 1}/{steps}", "denoise")
        return lat

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
        on_log: Callable[[str, str], None] | None = None,
        on_progress: Callable | None = None,
        cancel_token: Any | None = None,
        **_ignored: Any,
    ) -> str:
        if self._dit is None or self._cond is None:
            raise RuntimeError("BooguImageMlxGenerator.load() must be called before generate")

        ref_path = (ref_image_paths or [None])[0]
        is_edit = ref_path is not None or self._variant == "edit"

        if is_edit:
            if not ref_path:
                raise RuntimeError("Boogu-Image-Edit requires a source image.")
            ref_pil = Image.open(ref_path).convert("RGB")
            width = _snap_dim(width or ref_pil.width)
            height = _snap_dim(height or ref_pil.height)
            ref_pil = ref_pil.resize((width, height), Image.BICUBIC)
            pos = self._cond.encode_ti2i(ref_pil, prompt)
            neg = self._cond.encode_ti2i(ref_pil, negative_prompt or "")
            ref_lat = self._vae_encode_ref(ref_pil, width, height)
        else:
            width = _snap_dim(width)
            height = _snap_dim(height)
            pos = self._cond.encode_t2i(prompt)
            neg = self._cond.encode_t2i(negative_prompt) if guidance > 1.0 and negative_prompt else None
            ref_lat = None

        mx.random.seed(int(seed))
        hl, wl = height // 8, width // 8
        lat = mx.random.normal((1, 16, hl, wl)).astype(mx.bfloat16)

        if on_log:
            on_log(
                "info",
                f"boogu infer variant={self._variant} edit={is_edit} size={width}x{height} steps={steps} guidance={guidance}",
            )

        lat = self._denoise(
            lat=lat,
            pos=pos,
            neg=neg,
            ref=ref_lat,
            steps=steps,
            guidance=guidance,
            height=height,
            width=width,
            on_progress=on_progress,
            cancel_token=cancel_token,
        )
        rgb = self._decode_latent(lat)
        Image.fromarray(rgb).save(output_path)
        return output_path
