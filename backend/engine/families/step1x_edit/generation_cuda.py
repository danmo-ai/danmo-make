"""Step1X-Edit CUDA image generation — edit + T2I (native inference stack)."""

from __future__ import annotations

import itertools
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import functional as F

from backend.engine.families.step1x_edit import sampling_cuda
from backend.engine.families.step1x_edit.modules.autoencoder_cuda import AutoEncoder
from backend.engine.families.step1x_edit.modules.conditioner_cuda import Qwen25VL_7b_Embedder as Qwen2VLEmbedder
from backend.engine.families.step1x_edit.modules.model_edit_cuda import Step1XEdit, Step1XParams


def _load_state_dict(model, ckpt_path: str, *, strict: bool = False) -> torch.nn.Module:
    path = Path(ckpt_path)
    if path.suffix == ".safetensors":
        state_dict = load_file(str(path), "cpu")
    else:
        state_dict = torch.load(str(path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=strict, assign=True)
    if missing or unexpected:
        if missing:
            raise RuntimeError(f"Step1X-Edit missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            raise RuntimeError(f"Step1X-Edit unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    return model


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
        raise RuntimeError(
            f"Step1X-Edit bundle missing Qwen2.5-VL text encoder directory under {bundle_root}"
        )
    return ae, dit, qwen


class Step1XEditCudaGenerator:
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
        self._device = torch.device("cuda")
        self._ae: AutoEncoder | None = None
        self._dit: Step1XEdit | None = None
        self._llm: Qwen2VLEmbedder | None = None
        self._version = str(getattr(config, "step1x_version", "v1.1") or "v1.1")
        self._mode = "torch"
        self._max_length = int(getattr(config, "step1x_max_length", 640) or 640)
        self._size_level = int(getattr(config, "step1x_size_level", 512) or 512)

    def load(self) -> None:
        backend = getattr(self._ctx, "backend", "cuda")
        if backend != "cuda":
            raise RuntimeError(
                f"Step1X-Edit requires CUDA runtime (got {backend!r}). "
                "Select a CUDA model version on NVIDIA hardware."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("Step1X-Edit requires torch.cuda but CUDA is unavailable.")

        dit_name = getattr(self._config, "step1x_dit_filename", None)
        ae_path, dit_path, qwen_path = _resolve_step1x_paths(
            self._bundle_root,
            version=self._version,
            dit_filename=str(dit_name) if dit_name else None,
        )

        self._llm = Qwen2VLEmbedder(
            str(qwen_path),
            device=str(self._device),
            max_length=self._max_length,
            dtype=torch.bfloat16,
        )

        with torch.device("meta"):
            ae = AutoEncoder(
                resolution=256,
                in_channels=3,
                ch=128,
                out_ch=3,
                ch_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                z_channels=16,
                scale_factor=0.3611,
                shift_factor=0.1159,
            )
            params = Step1XParams(
                in_channels=64,
                out_channels=64,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=3072,
                mlp_ratio=4.0,
                num_heads=24,
                depth=19,
                depth_single_blocks=38,
                axes_dim=[16, 56, 56],
                theta=10_000,
                qkv_bias=True,
                mode=self._mode,
                version=self._version,
            )
            dit = Step1XEdit(params)

        ae = _load_state_dict(ae, str(ae_path))
        dit = _load_state_dict(dit, str(dit_path))
        ae = ae.to(dtype=torch.float32)
        self._ae = ae.to(self._device)
        self._dit = dit.to(device=self._device, dtype=torch.bfloat16)

    @staticmethod
    def _process_diff_norm(diff_norm: torch.Tensor, k: float) -> torch.Tensor:
        pow_result = torch.pow(diff_norm, k)
        return torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
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

    def _prepare_edit(
        self,
        prompt: str,
        negative: str,
        img: torch.Tensor,
        ref_image: torch.Tensor,
        ref_image_raw: torch.Tensor,
    ) -> dict[str, Any]:
        bs, _, h, w = img.shape
        bs, _, ref_h, ref_w = ref_image.shape
        if h != ref_h or w != ref_w:
            raise RuntimeError("Step1X-Edit internal shape mismatch between latent and reference.")

        prompt_list = [prompt, negative]
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        ref_img = rearrange(ref_image, "b c (ref_h ph) (ref_w pw) -> b (ref_h ref_w) (c ph pw)", ph=2, pw=2)
        img = repeat(img, "1 ... -> bs ...", bs=2)
        ref_img = repeat(ref_img, "1 ... -> bs ...", bs=2)

        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=2)

        if self._version == "v1.0":
            ref_img_ids = torch.zeros(ref_h // 2, ref_w // 2, 3)
        else:
            ref_img_ids = torch.ones(ref_h // 2, ref_w // 2, 3)
        ref_img_ids[..., 1] = ref_img_ids[..., 1] + torch.arange(ref_h // 2)[:, None]
        ref_img_ids[..., 2] = ref_img_ids[..., 2] + torch.arange(ref_w // 2)[None, :]
        ref_img_ids = repeat(ref_img_ids, "ref_h ref_w c -> b (ref_h ref_w) c", b=2)

        txt, mask = self._llm(prompt_list, ref_image_raw)
        txt_ids = torch.zeros(2, txt.shape[1], 3)
        img = torch.cat([img, ref_img.to(device=img.device, dtype=img.dtype)], dim=-2)
        img_ids = torch.cat([img_ids, ref_img_ids], dim=-2)
        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    def _prepare_t2i(
        self,
        prompt: str,
        negative: str,
        img: torch.Tensor,
        ref_image_raw: torch.Tensor,
    ) -> dict[str, Any]:
        bs, _, h, w = img.shape
        prompt_list = [prompt, negative]
        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img = repeat(img, "1 ... -> bs ...", bs=2)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=2)
        txt, mask = self._llm(prompt_list, ref_image_raw)
        txt_ids = torch.zeros(2, txt.shape[1], 3)
        return {
            "img": img,
            "mask": mask,
            "img_ids": img_ids.to(img.device),
            "llm_embedding": txt.to(img.device),
            "txt_ids": txt_ids.to(img.device),
        }

    def _denoise_edit(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float,
        mask: torch.Tensor | None,
        on_progress: Callable | None,
        cancel_token: Any | None,
    ) -> torch.Tensor:
        ref_img_tensor = img[0, img.shape[1] // 2 :].clone()
        n_steps = max(len(timesteps) - 1, 1)
        for idx, (t_curr, t_prev) in enumerate(itertools.pairwise(timesteps)):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise RuntimeError("Cancelled")
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = self._dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )
            pred = pred[:, : pred.shape[1] // 2]
            cond, uncond = pred[0 : pred.shape[0] // 2, :], pred[pred.shape[0] // 2 :, :]
            if t_curr > 0.93:
                diff = cond - uncond
                diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                pred = uncond + cfg_guidance * (cond - uncond) / self._process_diff_norm(diff_norm, k=0.4)
            else:
                pred = uncond + cfg_guidance * (cond - uncond)
            tem_img = img[0 : img.shape[0] // 2, : img.shape[1] // 2] + (t_prev - t_curr) * pred
            img = torch.cat([tem_img, ref_img_tensor.unsqueeze(0)], dim=1)
            if on_progress is not None:
                on_progress((idx + 1) / n_steps, idx + 1, n_steps, f"denoise {idx + 1}/{n_steps}", "denoise")
        return img[:, : img.shape[1] // 2]

    def _denoise_t2i(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        llm_embedding: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: list[float],
        cfg_guidance: float,
        mask: torch.Tensor | None,
        on_progress: Callable | None,
        cancel_token: Any | None,
    ) -> torch.Tensor:
        n_steps = max(len(timesteps) - 1, 1)
        for idx, (t_curr, t_prev) in enumerate(itertools.pairwise(timesteps)):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise RuntimeError("Cancelled")
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            pred = self._dit(
                img=img,
                img_ids=img_ids,
                txt_ids=txt_ids,
                timesteps=t_vec,
                llm_embedding=llm_embedding,
                t_vec=t_vec,
                mask=mask,
            )
            cond, uncond = pred[0 : pred.shape[0] // 2, :], pred[pred.shape[0] // 2 :, :]
            if t_curr > 0.93:
                diff = cond - uncond
                diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                pred = uncond + cfg_guidance * (cond - uncond) / self._process_diff_norm(diff_norm, k=0.4)
            else:
                pred = uncond + cfg_guidance * (cond - uncond)
            img = img[0 : img.shape[0] // 2] + (t_prev - t_curr) * pred
            if on_progress is not None:
                on_progress((idx + 1) / n_steps, idx + 1, n_steps, f"denoise {idx + 1}/{n_steps}", "denoise")
        return img

    @staticmethod
    def _unpack(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @torch.inference_mode()
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
            raise RuntimeError("Step1XEditCudaGenerator.load() must be called before generate")

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
            ref_raw = F.to_tensor(ref_pil_proc).unsqueeze(0).to(self._device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ref_latent = self._ae.encode(ref_raw * 2 - 1)
        else:
            orig_size = (width, height)
            ref_pil_proc = Image.new("RGB", (width, height))
            ref_raw = F.to_tensor(ref_pil_proc).unsqueeze(0).to(self._device)
            ref_latent = None

        gen = torch.Generator(device=self._device).manual_seed(int(seed))
        x = torch.randn(
            1,
            16,
            height // 8,
            width // 8,
            device=self._device,
            dtype=torch.bfloat16,
            generator=gen,
        )
        timesteps = sampling.get_schedule(steps, x.shape[-1] * x.shape[-2] // 4, shift=True)
        x = torch.cat([x, x], dim=0)
        ref_raw = torch.cat([ref_raw, ref_raw], dim=0)

        if is_edit:
            ref_latent = torch.cat([ref_latent, ref_latent], dim=0)
            inputs = self._prepare_edit(prompt, neg, x, ref_latent, ref_raw)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self._denoise_edit(
                    **inputs,
                    timesteps=timesteps,
                    cfg_guidance=guidance,
                    on_progress=on_progress,
                    cancel_token=cancel_token,
                )
        else:
            inputs = self._prepare_t2i(prompt, neg, x, ref_raw)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self._denoise_t2i(
                    **inputs,
                    timesteps=timesteps,
                    cfg_guidance=guidance,
                    on_progress=on_progress,
                    cancel_token=cancel_token,
                )

        x = self._unpack(x.float(), height, width)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self._ae.decode(x)
        x = x.clamp(-1, 1).mul(0.5).add(0.5)
        out_img = F.to_pil_image(x[0].float().cpu())
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
