"""Step1X-Edit MLX image generation — edit + T2I (native inference stack)."""

from __future__ import annotations

import itertools
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.families.step1x_edit import sampling_mlx
from backend.engine.families.step1x_edit.conditioner_mlx import Step1XQwen25VLEmbedderMLX
from backend.engine.families.step1x_edit.transformer_mlx import Step1XEditMLX
from backend.engine.families.step1x_edit.vae_mlx import Step1XAutoEncoderMLX


def _resolve_step1x_paths(bundle_root: Path, *, version: str, dit_filename: str | None) -> tuple[Path, Path, Path]:
    ae = bundle_root / "vae.safetensors"
    if not ae.is_file():
        raise RuntimeError(f"Step1X-Edit bundle missing VAE: {ae}")
    if dit_filename:
        dit = bundle_root / dit_filename
    elif version == "v1.0":
        dit = bundle_root / "step1x-edit-i1258.safetensors"
    else:
        dit = bundle_root / "step1x-edit-v1p1-official.safetensors"
    if not dit.is_file():
        raise RuntimeError(f"Step1X-Edit bundle missing DiT weights: {dit}")
    qwen = bundle_root / "Qwen2.5-VL-7B-Instruct"
    if not qwen.is_dir():
        qwen = bundle_root / "qwen2.5-vl-7b-instruct"
    if not qwen.is_dir():
        raise RuntimeError(f"Step1X-Edit bundle missing Qwen2.5-VL text encoder directory under {bundle_root}")
    return ae, dit, qwen


def _pil_to_chw01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _chw01_to_pil(x: np.ndarray) -> Image.Image:
    x = np.clip(x, 0.0, 1.0)
    rgb = (x.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def _pack_latents(x: mx.array) -> mx.array:
    b, c, h, w = x.shape
    x = mx.reshape(x, (b, c, h // 2, 2, w // 2, 2))
    x = mx.transpose(x, (0, 2, 4, 1, 3, 5))
    return mx.reshape(x, (b, (h // 2) * (w // 2), c * 4))


def _unpack_latents(x: mx.array, height: int, width: int) -> mx.array:
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    b, _, c4 = x.shape
    c = c4 // 4
    x = mx.reshape(x, (b, h, w, c, 2, 2))
    x = mx.transpose(x, (0, 3, 1, 4, 2, 5))
    return mx.reshape(x, (b, c, h * 2, w * 2))


class Step1XEditMlxGenerator:
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
        self._ae: Step1XAutoEncoderMLX | None = None
        self._dit: Step1XEditMLX | None = None
        self._llm: Step1XQwen25VLEmbedderMLX | None = None
        self._version = str(getattr(config, "step1x_version", "v1.1") or "v1.1")
        self._max_length = int(getattr(config, "step1x_max_length", 640) or 640)
        self._size_level = int(getattr(config, "step1x_size_level", 512) or 512)

    def load(self) -> None:
        dit_name = getattr(self._config, "step1x_dit_filename", None)
        ae_path, dit_path, qwen_path = _resolve_step1x_paths(
            self._bundle_root,
            version=self._version,
            dit_filename=str(dit_name) if dit_name else None,
        )
        self._llm = Step1XQwen25VLEmbedderMLX(
            qwen_path,
            max_length=self._max_length,
            ctx=self._ctx,
        )
        ae = Step1XAutoEncoderMLX(
            ch_mult=[1, 2, 4, 4],
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )
        ae_weights = mx.load(str(ae_path))
        ae.load_weights(list(ae_weights.items()), strict=False)
        self._ae = ae
        dit = Step1XEditMLX(
            self._ctx,
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            version=self._version,
        )
        dit_weights = mx.load(str(dit_path))
        dit.load_weights(list(dit_weights.items()), strict=False)
        self._dit = dit
        if hasattr(self._ctx, "eval"):
            self._ctx.eval(self._ae.parameters(), self._dit.parameters())

    @staticmethod
    def _process_diff_norm(diff_norm: mx.array, k: float) -> mx.array:
        pow_result = mx.power(diff_norm, k)
        return mx.where(
            diff_norm > 1.0,
            pow_result,
            mx.where(diff_norm < 1.0, mx.ones_like(diff_norm), diff_norm),
        )

    @staticmethod
    def _input_process_image(img: Image.Image, img_size: int = 512) -> tuple[Image.Image, tuple[int, int]]:
        w, h = img.size
        r = w / h
        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = h_new // 16 * 16
        w_new = w_new // 16 * 16
        return img.resize((w_new, h_new), Image.LANCZOS), img.size

    def _encode_ref(self, ref_chw: np.ndarray) -> mx.array:
        assert self._ae is not None
        x = mx.array(ref_chw[None].astype(np.float32)) * 2.0 - 1.0
        z = self._ae.encode(x.astype(mx.float32))
        if hasattr(self._ctx, "eval"):
            self._ctx.eval(z)
        return z.astype(mx.bfloat16)

    def _make_img_ids(self, h: int, w: int, batch: int, *, ref_fill: float = 0.0) -> mx.array:
        ids = np.zeros((h // 2, w // 2, 3), dtype=np.float32)
        ids[..., 0] = ref_fill
        ids[..., 1] = np.arange(h // 2, dtype=np.float32)[:, None]
        ids[..., 2] = np.arange(w // 2, dtype=np.float32)[None, :]
        flat = np.repeat(ids.reshape(1, -1, 3), batch, axis=0)
        return mx.array(flat)

    def _prepare_edit(
        self,
        prompt: str,
        negative: str,
        img: mx.array,
        ref_image: mx.array,
        ref_image_raw: np.ndarray,
    ) -> dict[str, Any]:
        _, _, h, w = img.shape
        _, _, ref_h, ref_w = ref_image.shape
        if h != ref_h or w != ref_w:
            raise RuntimeError("Step1X-Edit internal shape mismatch between latent and reference.")
        prompt_list = [prompt, negative]
        img_packed = _pack_latents(img)
        ref_packed = _pack_latents(ref_image)
        img_packed = mx.concatenate([img_packed, img_packed], axis=0)
        ref_packed = mx.concatenate([ref_packed, ref_packed], axis=0)
        img_cat = mx.concatenate([img_packed, ref_packed], axis=1)

        ref_fill = 0.0 if self._version == "v1.0" else 1.0
        img_ids = self._make_img_ids(h, w, 2, ref_fill=0.0)
        ref_img_ids = self._make_img_ids(ref_h, ref_w, 2, ref_fill=ref_fill)
        img_ids = mx.concatenate([img_ids, ref_img_ids], axis=1)
        txt, mask = self._llm(prompt_list, ref_image_raw)
        txt_ids = mx.zeros((2, txt.shape[1], 3), dtype=mx.float32)
        return {
            "img": img_cat,
            "mask": mask,
            "img_ids": img_ids,
            "llm_embedding": txt,
            "txt_ids": txt_ids,
        }

    def _prepare_t2i(
        self,
        prompt: str,
        negative: str,
        img: mx.array,
        ref_image_raw: np.ndarray,
    ) -> dict[str, Any]:
        _, _, h, w = img.shape
        prompt_list = [prompt, negative]
        img_packed = _pack_latents(img)
        img_packed = mx.concatenate([img_packed, img_packed], axis=0)
        img_ids = self._make_img_ids(h, w, 2, ref_fill=0.0)
        txt, mask = self._llm(prompt_list, ref_image_raw)
        txt_ids = mx.zeros((2, txt.shape[1], 3), dtype=mx.float32)
        return {
            "img": img_packed,
            "mask": mask,
            "img_ids": img_ids,
            "llm_embedding": txt,
            "txt_ids": txt_ids,
        }

    def _denoise_edit(
        self,
        *,
        img: mx.array,
        img_ids: mx.array,
        llm_embedding: mx.array,
        txt_ids: mx.array,
        timesteps: list[float],
        cfg_guidance: float,
        mask: mx.array | None,
        on_progress: Callable | None,
        cancel_token: Any | None,
    ) -> mx.array:
        assert self._dit is not None
        half = img.shape[1] // 2
        ref_img_tensor = img[0:1, half:].copy()
        n_steps = max(len(timesteps) - 1, 1)
        for idx, (t_curr, t_prev) in enumerate(itertools.pairwise(timesteps)):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise RuntimeError("Cancelled")
            t_vec = mx.full((img.shape[0],), t_curr, dtype=img.dtype)
            pred = self._dit(img, img_ids, txt_ids, t_vec, llm_embedding, t_vec, mask)
            pred = pred[:, : pred.shape[1] // 2]
            cond, uncond = pred[0:1], pred[1:2]
            if t_curr > 0.93:
                diff = cond - uncond
                diff_norm = mx.linalg.norm(diff, axis=-1, keepdims=True)
                pred = uncond + cfg_guidance * (cond - uncond) / self._process_diff_norm(diff_norm, k=0.4)
            else:
                pred = uncond + cfg_guidance * (cond - uncond)
            tem_img = img[0:1, :half] + (t_prev - t_curr) * pred
            img = mx.concatenate([tem_img, ref_img_tensor], axis=1)
            if on_progress is not None:
                on_progress((idx + 1) / n_steps, idx + 1, n_steps, f"denoise {idx + 1}/{n_steps}", "denoise")
        return img[:, :half]

    def _denoise_t2i(
        self,
        *,
        img: mx.array,
        img_ids: mx.array,
        llm_embedding: mx.array,
        txt_ids: mx.array,
        timesteps: list[float],
        cfg_guidance: float,
        mask: mx.array | None,
        on_progress: Callable | None,
        cancel_token: Any | None,
    ) -> mx.array:
        assert self._dit is not None
        n_steps = max(len(timesteps) - 1, 1)
        for idx, (t_curr, t_prev) in enumerate(itertools.pairwise(timesteps)):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise RuntimeError("Cancelled")
            t_vec = mx.full((img.shape[0],), t_curr, dtype=img.dtype)
            pred = self._dit(img, img_ids, txt_ids, t_vec, llm_embedding, t_vec, mask)
            cond, uncond = pred[0:1], pred[1:2]
            if t_curr > 0.93:
                diff = cond - uncond
                diff_norm = mx.linalg.norm(diff, axis=-1, keepdims=True)
                pred = uncond + cfg_guidance * (cond - uncond) / self._process_diff_norm(diff_norm, k=0.4)
            else:
                pred = uncond + cfg_guidance * (cond - uncond)
            img = img[0:1] + (t_prev - t_curr) * pred
            if on_progress is not None:
                on_progress((idx + 1) / n_steps, idx + 1, n_steps, f"denoise {idx + 1}/{n_steps}", "denoise")
        return img

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
    ) -> str:
        if self._ae is None or self._dit is None or self._llm is None:
            raise RuntimeError("Step1XEditMlxGenerator.load() must be called before generate")

        neg = negative_prompt or (
            "worst quality, wrong limbs, unreasonable limbs, normal quality, low quality, low res, "
            "blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, "
            "username, error, sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
        )
        ref_path = (ref_image_paths or [None])[0]
        is_edit = ref_path is not None
        if is_edit:
            ref_pil = Image.open(ref_path).convert("RGB")
            ref_pil_proc, orig_size = self._input_process_image(ref_pil, img_size=self._size_level)
            width, height = ref_pil_proc.width, ref_pil_proc.height
            ref_raw = _pil_to_chw01(ref_pil_proc)
            ref_latent = self._encode_ref(ref_raw)
        else:
            orig_size = (width, height)
            ref_pil_proc = Image.new("RGB", (width, height))
            ref_raw = _pil_to_chw01(ref_pil_proc)
            ref_latent = None

        rng = np.random.default_rng(int(seed))
        x = mx.array(rng.standard_normal((1, 16, height // 8, width // 8), dtype=np.float32), dtype=mx.bfloat16)
        timesteps = sampling_mlx.get_schedule(steps, (height // 8) * (width // 8) // 4, shift=True)
        x = mx.concatenate([x, x], axis=0)
        ref_batch = np.stack([ref_raw, ref_raw], axis=0)

        if is_edit:
            assert ref_latent is not None
            ref_latent = mx.concatenate([ref_latent, ref_latent], axis=0)
            inputs = self._prepare_edit(prompt, neg, x, ref_latent, ref_batch)
            x = self._denoise_edit(
                **inputs,
                timesteps=timesteps,
                cfg_guidance=guidance,
                on_progress=on_progress,
                cancel_token=cancel_token,
            )
        else:
            inputs = self._prepare_t2i(prompt, neg, x, ref_batch)
            x = self._denoise_t2i(
                **inputs,
                timesteps=timesteps,
                cfg_guidance=guidance,
                on_progress=on_progress,
                cancel_token=cancel_token,
            )

        x = _unpack_latents(x.astype(mx.float32), height, width)
        decoded = self._ae.decode(x.astype(mx.float32))
        if hasattr(self._ctx, "eval"):
            self._ctx.eval(decoded)
        decoded = mx.clip(decoded, -1.0, 1.0)
        decoded = decoded * 0.5 + 0.5
        out_img = _chw01_to_pil(np.array(decoded[0], dtype=np.float32))
        if is_edit:
            out_img = out_img.resize(orig_size, Image.LANCZOS)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out, lossless=True)
        if on_log:
            on_log("info", f"Step1X-Edit saved -> {out}")
        return str(out)


def resolve_step1x_output_path(work_dir: Path, model_key: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(work_dir / f"{model_key}_{seed}_{ts}.png")
