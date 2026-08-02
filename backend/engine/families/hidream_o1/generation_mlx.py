"""HiDream-O1 MLX image generation — T2I, edit, multi-reference."""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from backend.engine.families.hidream_o1.flow_match_mlx import (
    DEFAULT_TIMESTEPS,
    FlashFlowMatchScheduler,
)
from backend.engine.families.hidream_o1.hidream_model_mlx import (
    HiDreamConfig,
    build_model,
    forward_generation,
    precompute_text_embeds_with_vision,
)
from backend.engine.families.hidream_o1.pipeline_helpers import (
    NOISE_SCALE_DEFAULT,
    PATCH_SIZE,
    T_EPS,
    build_attention_mask,
    build_edit_text_sample,
    build_t2i_text_sample,
    find_closest_resolution,
    patchify,
    unpatchify,
)


def _blend_patch_seams(rgb: np.ndarray, patch: int, radius: int) -> np.ndarray:
    if radius <= 0:
        return rgb
    out = rgb.astype(np.float32).copy()
    h, w, _ = rgb.shape
    weights = np.array(
        [radius - abs(i - radius) + 1 for i in range(2 * radius + 1)],
        dtype=np.float32,
    )
    weights = weights / weights.sum()
    for y in range(patch, h, patch):
        for offset in (-1, 0):
            yy = y + offset
            if 0 <= yy < h:
                lo = max(0, yy - radius)
                hi = min(h, yy + radius + 1)
                w_slice = weights[radius - (yy - lo) : radius + (hi - yy)]
                w_slice = w_slice / w_slice.sum()
                band = out[lo:hi]
                out[yy] = (band * w_slice[:, None, None]).sum(axis=0)
    for x in range(patch, w, patch):
        for offset in (-1, 0):
            xx = x + offset
            if 0 <= xx < w:
                lo = max(0, xx - radius)
                hi = min(w, xx + radius + 1)
                w_slice = weights[radius - (xx - lo) : radius + (hi - xx)]
                w_slice = w_slice / w_slice.sum()
                band = out[:, lo:hi]
                out[:, xx] = (band * w_slice[None, :, None]).sum(axis=1)
    return np.clip(out, 0, 255).astype(np.uint8)


class HiDreamO1MlxGenerator:
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
        self._model: Any | None = None
        self._backbone: Any | None = None
        self._processor: Any | None = None
        self._hidream_cfg = HiDreamConfig()

    def load(self) -> None:
        backend = getattr(self._ctx, "backend", "mlx")
        if backend != "mlx":
            raise RuntimeError(
                f"HiDream-O1-Image requires MLX runtime (got {backend!r}). "
                "Select an MLX model version on Apple Silicon."
            )
        try:
            from mlx_vlm import load as mlx_vlm_load
        except ImportError as exc:
            raise RuntimeError(
                "HiDream-O1-Image requires mlx-vlm. Install with: pip install 'mlx-vlm>=0.5.0'"
            ) from exc
        import mlx.core as mx

        if not self._bundle_root.is_dir():
            raise RuntimeError(f"HiDream-O1 bundle not found: {self._bundle_root}")

        custom_path = self._bundle_root / "extras" / "custom_heads.safetensors"
        if not custom_path.is_file():
            raise RuntimeError(
                f"HiDream-O1 bundle missing custom diffusion heads: {custom_path}. "
                "Use mlx-community HiDream-O1 weights (extras/custom_heads.safetensors)."
            )

        self._backbone, self._processor = mlx_vlm_load(str(self._bundle_root))
        model = build_model(self._hidream_cfg, self._backbone)
        custom_weights = mx.load(str(custom_path))
        model.load_weights(list(custom_weights.items()), strict=False)
        self._model = model

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
        snap_resolution: bool = True,
        blend_seams: int = 0,
        on_log: Callable[[str, str], None] | None = None,
        on_progress: Callable | None = None,
        cancel_token: Any | None = None,
    ) -> str:
        _ = guidance, negative_prompt
        import mlx.core as mx

        if self._model is None or self._backbone is None or self._processor is None:
            raise RuntimeError("HiDreamO1MlxGenerator.load() must be called before generate")

        if cancel_token is not None and cancel_token.is_cancelled():
            raise RuntimeError("Cancelled")

        model = self._model
        backbone = self._backbone
        processor = self._processor
        cfg = self._hidream_cfg

        if snap_resolution:
            sw, sh = find_closest_resolution(width, height)
            if (sw, sh) != (width, height) and on_log:
                on_log("info", f"HiDream-O1 resolution snap {width}x{height} -> {sw}x{sh}")
            width, height = sw, sh
        width = (width // PATCH_SIZE) * PATCH_SIZE
        height = (height // PATCH_SIZE) * PATCH_SIZE
        h_patches = height // PATCH_SIZE
        w_patches = width // PATCH_SIZE

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        for n in ("boi", "bor", "eor", "bot", "tms"):
            if not hasattr(tokenizer, f"{n}_token"):
                setattr(tokenizer, f"{n}_token", f"<|{n}_token|>")

        refs = [p for p in (ref_image_paths or []) if p]
        if refs and not bool(getattr(self._config, "hidream_edit_bf16_only", True)):
            pass
        if refs and getattr(self._config, "hidream_quantized_no_edit", False):
            raise RuntimeError(
                "HiDream-O1 Q6/Q8 quantized weights do not support edit or multi-reference. "
                "Use the BF16 MLX variant for edit workflows."
            )

        if refs:
            if on_log:
                on_log("info", f"HiDream-O1 edit mode: {len(refs)} reference image(s)")
            sample = build_edit_text_sample(
                prompt, refs, height, width, tokenizer, processor, backbone.config,
            )
        else:
            sample = build_t2i_text_sample(
                prompt, height, width, tokenizer, processor, backbone.config,
            )

        input_ids = mx.array(sample["input_ids"])
        position_ids = mx.array(sample["position_ids"])
        token_types = mx.array(sample["token_types"])
        vinput_mask = sample["vinput_mask"]

        pixel_values_mx = (
            mx.array(sample["pixel_values"]).astype(mx.bfloat16) if refs else None
        )
        image_grid_thw_mx = mx.array(sample["image_grid_thw"]) if refs else None
        ref_patches_mx = mx.array(sample["ref_patches"]).astype(mx.bfloat16) if refs else None

        dtype_min = -1e4
        mask4d = mx.array(build_attention_mask(sample["token_types"], dtype_min)).astype(mx.bfloat16)

        noise_scale = float(
            getattr(self._config, "hidream_noise_scale", NOISE_SCALE_DEFAULT) or NOISE_SCALE_DEFAULT
        )
        noise_clip_std = float(getattr(self._config, "hidream_noise_clip_std", 2.5) or 2.5)

        rng_key = mx.random.key(seed + 1)
        noise = noise_scale * mx.random.normal((1, 3, height, width), key=rng_key)
        z = mx.array(patchify(np.asarray(noise)[0])[None]).astype(mx.bfloat16)

        sched = FlashFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
        custom_ts = getattr(self._config, "hidream_custom_timesteps", None)
        sched.set_timesteps(steps, custom_timesteps=custom_ts or DEFAULT_TIMESTEPS)
        noise_scale_schedule = np.linspace(noise_scale, noise_scale, len(sched.timesteps_np))

        vinput_idx = mx.array(np.where(vinput_mask[0])[0].astype(np.int32))
        if refs:
            tgt_mask = sample["vinput_mask_tgt_only"]
            tgt_idx = mx.array(np.where(tgt_mask[0])[0].astype(np.int32))
        else:
            tgt_idx = vinput_idx

        inputs_embeds_pre = precompute_text_embeds_with_vision(
            model,
            cfg,
            input_ids,
            pixel_values=pixel_values_mx,
            image_grid_thw=image_grid_thw_mx,
        )
        mx.eval(inputs_embeds_pre)

        n_steps = len(sched.timesteps_np)
        t_start = time.time()
        for step_idx, step_t in enumerate(sched.timesteps_np):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise RuntimeError("Cancelled")

            t_pixeldit = mx.full([1], 1.0 - float(step_t) / 1000.0, dtype=mx.float32)
            sigma = max(float(step_t) / 1000.0, T_EPS)

            if refs:
                vinputs = mx.concatenate([z, ref_patches_mx], axis=1)
            else:
                vinputs = z

            x_pred = forward_generation(
                model,
                cfg,
                inputs_embeds_with_vision=inputs_embeds_pre,
                position_ids=position_ids,
                vinputs=vinputs,
                timestep=t_pixeldit,
                input_ids=input_ids,
                token_types=token_types,
                attention_mask_4d=mask4d,
            )
            gen_patches_mx = mx.take(x_pred, tgt_idx, axis=1).astype(mx.float32)
            v = (gen_patches_mx - z.astype(mx.float32)) / sigma
            z = sched.step(
                -v,
                float(step_t),
                z,
                s_noise=float(noise_scale_schedule[step_idx]),
                noise_clip_std=noise_clip_std,
                seed=seed,
            )
            mx.eval(z)

            if on_progress is not None:
                on_progress(
                    (step_idx + 1) / max(n_steps, 1),
                    step_idx + 1,
                    n_steps,
                    f"denoise {step_idx + 1}/{n_steps}",
                    "denoise",
                )

        elapsed = time.time() - t_start
        if on_log:
            on_log(
                "info",
                f"HiDream-O1 generation {elapsed:.1f}s ({elapsed / max(n_steps, 1):.2f}s/step)",
            )

        img = (z + 1) / 2
        img_np = np.asarray(img[0].astype(mx.float32))
        rgb = unpatchify(img_np, h_patches, w_patches)
        arr = np.clip(rgb.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        if blend_seams > 0:
            arr = _blend_patch_seams(arr, PATCH_SIZE, radius=blend_seams)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out)
        return str(out)


def resolve_hidream_output_path(work_dir: Path, model_key: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(work_dir / f"{model_key}_{seed}_{ts}.png")
