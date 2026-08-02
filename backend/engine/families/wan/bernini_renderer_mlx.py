"""Bernini-R renderer — SA-3D RoPE + chained CFG (T2V / V2V / R2V / RV2V)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from PIL import Image

from backend.core.contracts import parse_size
from backend.engine.config.model_configs import WanConfig
from backend.engine.families.wan.conditioning import (
    prepare_wan_reference_image,
    WAN_SAMPLE_NEG_PROMPT,
)
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_complete
from backend.engine.pipelines.video_model_load import (
    latent_frame_count_for_video,
    load_video_transformer,
    prepare_video_config,
)

GuidanceMode = Literal["t2v", "r2v", "v2v", "v2v_chain", "rv2v"]


def _resolve_inner(model: Any, step_idx: int) -> Any:
    if hasattr(model, "_ensure_expert"):
        expert = model._ensure_expert(step_idx)
        return getattr(expert, "_inner", expert)
    stem = model
    return getattr(stem, "_inner", stem)


def _concat_rope(ctx: Any, parts: list[tuple[Any, Any]]) -> tuple[Any, Any]:
    if not parts:
        raise RuntimeError("concat_rope requires at least one segment")
    cos = ctx.concat([p[0] for p in parts], axis=0)
    sin = ctx.concat([p[1] for p in parts], axis=0)
    return cos, sin


def _align_latent_volume(
    ctx: Any,
    latent: Any,
    *,
    channels: int,
    frames: int,
    height: int,
    width: int,
) -> Any:
    """Crop or edge-pad ``[C,T,H,W]`` to match the denoise noise tensor."""
    if latent.ndim == 5:
        latent = ctx.squeeze(latent, 0)
    if latent.ndim != 4:
        raise RuntimeError(
            f"Bernini latent align expects [C,T,H,W], got {getattr(latent, 'shape', ())}"
        )
    c, t, h, w = (int(latent.shape[i]) for i in range(4))
    if c != channels:
        raise RuntimeError(
            f"Bernini V2V latent channel mismatch: encoded {c}, expected {channels}"
        )
    if t > frames:
        latent = latent[:, :frames, :, :]
    elif t < frames:
        if t < 1:
            raise RuntimeError("Bernini V2V source latent has zero temporal frames")
        pad = ctx.repeat(latent[:, -1:, :, :], frames - t, axis=1)
        latent = ctx.concat([latent, pad], axis=1)
    if h > height:
        off = (h - height) // 2
        latent = latent[:, :, off : off + height, :]
    elif h < height:
        pad = ctx.zeros((channels, frames, height - h, w), dtype=latent.dtype)
        latent = ctx.concat([latent, pad], axis=2)
    _, _, h2, w2 = (int(latent.shape[i]) for i in range(4))
    if w2 > width:
        off = (w2 - width) // 2
        latent = latent[:, :, :, off : off + width]
    elif w2 < width:
        pad = ctx.zeros((channels, frames, h2, width - w2), dtype=latent.dtype)
        latent = ctx.concat([latent, pad], axis=3)
    out = (int(latent.shape[0]), int(latent.shape[1]), int(latent.shape[2]), int(latent.shape[3]))
    expected = (channels, frames, height, width)
    if out != expected:
        raise RuntimeError(f"Bernini V2V latent align failed: got {out}, expected {expected}")
    return latent


def _make_source_ids(n: int, *, interpolate: bool, max_trained: int) -> list[float]:
    if n <= 0:
        return []
    if interpolate and n > max_trained:
        return [float(x) for x in np.linspace(1.0, float(max_trained), n)]
    return [float(i) for i in range(1, n + 1)]


def _resolve_guidance_mode(has_video: bool, has_refs: bool) -> GuidanceMode:
    if has_video and has_refs:
        return "rv2v"
    if has_video:
        return "v2v_chain"
    if has_refs:
        return "r2v"
    return "t2v"


def _bernini_user_task_label(
    *,
    source_video_path: str | None,
    source_image_path: str | None,
    ref_latents: list[Any],
) -> str:
    """User-facing task kind (distinct from internal CFG mode names)."""
    if source_video_path:
        return "v2v"
    if source_image_path:
        return "rv2v" if ref_latents else "i2v"
    if ref_latents:
        return "r2v"
    return "t2v"


def _load_video_rgb(path: Path, *, width: int, height: int, num_frames: int, fps: float) -> np.ndarray:
    from backend.engine.common.codecs.vae.video_stitch import _extract_video_rgb

    raw = _extract_video_rgb(path, fps=fps)
    if raw.shape[0] < 1:
        raise RuntimeError(f"Bernini source video {path} has no frames")
    if raw.shape[0] > num_frames:
        raw = raw[:num_frames]
    elif raw.shape[0] < num_frames:
        pad = np.repeat(raw[-1:, ...], num_frames - raw.shape[0], axis=0)
        raw = np.concatenate([raw, pad], axis=0)
    frames = []
    for i in range(num_frames):
        img = Image.fromarray(
            np.clip((raw[i] + 1.0) * 127.5, 0, 255).astype(np.uint8), mode="RGB",
        )
        frames.append(np.array(prepare_wan_reference_image(img, width, height)))
    return np.stack(frames, axis=0).astype(np.float32) / 127.5 - 1.0


class BerniniRendererMLX:
    """In-repo Bernini-R sampler (mlx-community / ByteDance weights)."""

    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        config: WanConfig | None = None,
        entry: Any | None = None,
        version_key: str | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or WanConfig()
        self.entry = entry
        self.version_key = version_key
        self._project_root = Path(project_root) if project_root is not None else self.bundle_root
        self._model: Any | None = None

    def load(self) -> None:
        if self.entry is not None:
            self.config = prepare_video_config(
                self.entry,
                "wan",
                self.bundle_root,
                project_root=self._project_root,
            )
        num_frames = 81
        self._model = load_video_transformer(
            ctx=self.ctx,
            family="wan",
            config=self.config,
            entry=self.entry,
            version_key=self.version_key,
            project_root=self._project_root,
            bundle_root=self.bundle_root,
            num_frames=num_frames,
            model_cache=None,
            on_log=None,
        )
        if self._model is None:
            raise RuntimeError(
                f"Failed to load Bernini-R transformer from {self.bundle_root} "
                "(no DiT shards found under transformer/ or model.safetensors at bundle root)."
            )

    def _encode_text(
        self,
        prompt: str,
        negative_prompt: str,
        *,
        on_log: Callable[[str, str], None] | None,
    ) -> tuple[Any, Any]:
        from backend.engine._transformer_registry import encode_video_prompt

        pos = encode_video_prompt(
            self.ctx,
            prompt,
            encoder_type=str(self.config.encoder_type),
            bundle_root=self.bundle_root,
            config=self.config,
        )[0]
        neg_txt = (negative_prompt or "").strip() or WAN_SAMPLE_NEG_PROMPT
        neg = encode_video_prompt(
            self.ctx,
            neg_txt,
            encoder_type=str(self.config.encoder_type),
            bundle_root=self.bundle_root,
            config=self.config,
        )[0]
        if on_log is not None:
            self.ctx.eval(pos, neg)
            peak = float(self.ctx.sqrt(self.ctx.max(self.ctx.square(pos))))
            on_log("info", f"Bernini-R UMT5 embeddings ready (peak={peak:.3f})")
        return pos, neg

    def _encode_reference_images(
        self,
        paths: list[str],
        width: int,
        height: int,
    ) -> list[Any]:
        from backend.engine.families.wan.vae_mlx import encode_wan_vae_image

        latents: list[Any] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            img = prepare_wan_reference_image(img, width, height)
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            chw = self.ctx.array(np.transpose(arr, (2, 0, 1)))
            enc = encode_wan_vae_image(self.ctx, chw, self.bundle_root)
            latents.append(self.ctx.squeeze(enc, 0))
        return latents

    def _encode_source_video(
        self,
        path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
    ) -> Any:
        from backend.engine.families.wan.vae_mlx import encode_wan_vae_volume

        rgb = _load_video_rgb(Path(path), width=width, height=height, num_frames=num_frames, fps=fps)
        bcthw = self.ctx.array(np.transpose(rgb, (3, 0, 1, 2)))
        bcthw = self.ctx.expand_dims(bcthw, 0)
        return self.ctx.squeeze(encode_wan_vae_volume(self.ctx, bcthw, self.bundle_root), 0)

    def _patch_segment(
        self,
        inner: Any,
        latent_cthw: Any,
        source_id: float,
    ) -> tuple[Any, tuple[Any, Any], tuple[int, int, int], int]:
        return inner.patch_latent_volume(
            latent_cthw,
            source_id,
            apply_src_id_rope=bool(getattr(self.config, "use_src_id_rotary_emb", False)),
        )

    def _assemble_inputs(
        self,
        inner: Any,
        noisy_bcthw: Any,
        video_latents: list[Any],
        ref_latents: list[Any],
        *,
        vi_sids: list[float],
        i_sids: list[float],
    ) -> dict[str, Any]:
        ctx = self.ctx
        v_tokens: list[Any] = []
        v_ropes: list[tuple[Any, Any]] = []
        i_tokens: list[Any] = []
        i_ropes: list[tuple[Any, Any]] = []
        vi_tokens: list[Any] = []
        vi_ropes: list[tuple[Any, Any]] = []
        vi_ptr = 0
        i_ptr = 0

        for idx, vol in enumerate(video_latents):
            tok, rope, grid, sl = self._patch_segment(inner, vol, vi_sids[vi_ptr])
            vi_ptr += 1
            vi_tokens.append(tok)
            vi_ropes.append(rope)
            if idx == 0:
                v_tokens.append(tok)
                v_ropes.append(rope)

        for vol in ref_latents:
            tok_vi, rope_vi, _, _ = self._patch_segment(inner, vol, vi_sids[vi_ptr])
            vi_ptr += 1
            vi_tokens.append(tok_vi)
            vi_ropes.append(rope_vi)
            tok_i, rope_i, _, _ = self._patch_segment(inner, vol, i_sids[i_ptr])
            i_ptr += 1
            i_tokens.append(tok_i)
            i_ropes.append(rope_i)

        n_tok, n_rope, grid, n_len = self._patch_segment(inner, noisy_bcthw, 0.0)
        noisy_len = n_len

        def _pack(cond_tokens: list[Any], cond_ropes: list[tuple[Any, Any]], cond_len: int) -> dict[str, Any]:
            if cond_tokens:
                tokens = ctx.concat([*(t for t in cond_tokens), n_tok], axis=1)
                rope = _concat_rope(ctx, cond_ropes + [n_rope])
                total = cond_len + noisy_len
            else:
                tokens = n_tok
                rope = n_rope
                total = noisy_len
            return {
                "tokens": tokens,
                "rope": rope,
                "grid": grid,
                "seq_len": total,
                "cond_len": cond_len,
                "noisy_len": noisy_len,
            }

        v_len = sum(int(t.shape[1]) for t in v_tokens)
        i_len = sum(int(t.shape[1]) for t in i_tokens)
        vi_len = sum(int(t.shape[1]) for t in vi_tokens)

        return {
            "none": _pack([], [], 0),
            "v": _pack(v_tokens, v_ropes, v_len),
            "i": _pack(i_tokens, i_ropes, i_len),
            "vi": _pack(vi_tokens, vi_ropes, vi_len),
            "grid": grid,
            "noisy_len": noisy_len,
        }

    def _forward_combo(
        self,
        model: Any,
        inner: Any,
        combo: dict[str, Any],
        timestep: Any,
        text_emb: Any,
        step_idx: int,
    ) -> Any:
        del model, step_idx
        tokens = combo["tokens"]
        tok_len = int(tokens.shape[1])
        rope_len = int(combo["rope"][0].shape[0])
        if tok_len != rope_len:
            raise RuntimeError(
                f"Bernini token/RoPE length mismatch: tokens={tok_len}, rope={rope_len}, "
                f"cond_len={combo['cond_len']}, noisy_len={combo['noisy_len']}"
            )
        pred_tokens = inner.forward_token_sequence(
            tokens,
            timestep,
            text_emb,
            rope_cos_sin=combo["rope"],
            grid=combo["grid"],
            seq_len=tok_len,
        )
        cond_len = int(combo["cond_len"])
        noisy_pred = pred_tokens[:, cond_len:]
        return inner.unpatchify_token_grid(noisy_pred, combo["grid"])

    def _apply_guidance(
        self,
        mode: GuidanceMode,
        eps: dict[str, Any],
        *,
        omega_vid: float,
        omega_img: float,
        omega_txt: float,
    ) -> Any:
        if mode == "rv2v":
            return (
                eps["none"]
                + omega_vid * (eps["v"] - eps["none"])
                + omega_img * (eps["vi"] - eps["v"])
                + omega_txt * (eps["vti"] - eps["vi"])
            )
        if mode == "v2v":
            return eps["vi_u"] + omega_txt * (eps["vti"] - eps["vi_u"])
        if mode == "v2v_chain":
            return (
                eps["none"]
                + omega_vid * (eps["v"] - eps["none"])
                + omega_txt * (eps["vti"] - eps["v"])
            )
        if mode == "r2v":
            return (
                eps["none"]
                + omega_img * (eps["i"] - eps["none"])
                + omega_txt * (eps["it"] - eps["i"])
            )
        return eps["none"] + omega_txt * (eps["t"] - eps["none"])

    def generate_and_save(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        source_video_path: str | None = None,
        source_image_path: str | None = None,
        reference_image_paths: list[str] | None = None,
        is_edit: bool = False,
        on_log: Any | None,
        on_progress: Any | None = None,
    ) -> str:
        del step_distill, is_edit
        if self._model is None:
            self.load()
        model = self._model
        ctx = self.ctx
        config = self.config

        self.ctx.seed_random(int(seed))
        latent_frames = latent_frame_count_for_video(config, num_frames)
        vae_scale = int(getattr(config, "vae_scale", 16))
        latent_h, latent_w = height // vae_scale, width // vae_scale
        latent_c = int(getattr(config, "vae_z_dim", None) or config.dim_in)

        pos_emb, neg_emb = self._encode_text(prompt, negative_prompt, on_log=on_log)

        video_latents: list[Any] = []
        ref_latents: list[Any] = []
        if source_video_path:
            encoded = self._encode_source_video(source_video_path, width, height, num_frames, fps)
            video_latents.append(
                _align_latent_volume(
                    ctx,
                    encoded,
                    channels=latent_c,
                    frames=latent_frames,
                    height=latent_h,
                    width=latent_w,
                )
            )
        if source_image_path:
            video_latents.append(
                self._encode_reference_images([source_image_path], width, height)[0]
            )
        if reference_image_paths:
            ref_latents.extend(
                self._encode_reference_images(reference_image_paths, width, height)
            )

        mode = _resolve_guidance_mode(bool(video_latents), bool(ref_latents))
        user_task = _bernini_user_task_label(
            source_video_path=source_video_path,
            source_image_path=source_image_path,
            ref_latents=ref_latents,
        )
        if source_video_path:
            source_kind = "video"
        elif source_image_path:
            source_kind = "image"
        else:
            source_kind = "none"
        if on_log is not None:
            on_log(
                "info",
                f"Bernini-R task={user_task} cfg={mode} source={source_kind} "
                f"video_sources={len(video_latents)} ref_images={len(ref_latents)} "
                f"size={width}x{height} frames={num_frames}",
            )

        interp = bool(getattr(config, "interpolate_src_id", True))
        max_trained = int(getattr(config, "max_trained_src_id", 5))
        num_v = len(video_latents)
        num_i = len(ref_latents)
        vi_sids = _make_source_ids(num_v + num_i, interpolate=interp, max_trained=max_trained)
        i_sids = _make_source_ids(num_i, interpolate=interp, max_trained=max_trained)

        noisy = ctx.randn((1, latent_c, latent_frames, latent_h, latent_w), dtype=ctx.float32())

        from backend.engine.pipelines.video_run_common import (
            create_video_scheduler,
            save_video,
            vae_decode_video,
        )

        scheduler = create_video_scheduler(
            type("P", (), {"ctx": ctx})(),
            config=config,
            scheduler_name=str(getattr(config, "default_scheduler", "wan_flow_unipc")),
            bundle_root=self.bundle_root,
        )
        sched_kwargs: dict[str, Any] = {}
        shift = float(getattr(config, "uses_wan_shift", False) and 3.0)
        if getattr(config, "uses_wan_shift", False):
            sched_kwargs["shift"] = shift
        timesteps = scheduler.set_timesteps(int(steps), **sched_kwargs)
        sigmas = getattr(scheduler, "sigmas", None)

        omega_txt = float(guidance)
        omega_vid = omega_txt
        omega_img = omega_txt
        omega_scale = 0.75
        boundary = int(getattr(config, "moe_boundary_step_index", 2))
        switched = False

        n_steps = len(timesteps)
        for step_idx, t in enumerate(timesteps):
            if step_idx >= boundary and not switched and hasattr(model, "_ensure_expert"):
                omega_vid *= omega_scale
                omega_img *= omega_scale
                omega_txt *= omega_scale
                switched = True

            inner = _resolve_inner(model, step_idx)
            combos = self._assemble_inputs(
                inner,
                noisy,
                video_latents,
                ref_latents,
                vi_sids=vi_sids,
                i_sids=i_sids,
            )

            t_scalar = ctx.array(float(t), dtype=ctx.float32())

            def _eps(combo_key: str, emb: Any) -> Any:
                return self._forward_combo(model, inner, combos[combo_key], t_scalar, emb, step_idx)

            eps_none = _eps("none", neg_emb)
            eps_v = _eps("v", neg_emb) if combos["v"]["cond_len"] > 0 else eps_none
            eps_i = _eps("i", neg_emb) if combos["i"]["cond_len"] > 0 else eps_none
            eps_vi_u = _eps("vi", neg_emb) if combos["vi"]["cond_len"] > 0 else eps_none
            eps_vi_t = _eps("vi", pos_emb) if combos["vi"]["cond_len"] > 0 else eps_none
            eps_t = _eps("none", pos_emb)
            eps_it = _eps("i", pos_emb) if combos["i"]["cond_len"] > 0 else eps_t

            eps = {
                "none": eps_none,
                "v": eps_v,
                "i": eps_i,
                "vi": eps_vi_u,
                "vi_u": eps_vi_u,
                "vti": eps_vi_t,
                "t": eps_t,
                "it": eps_it,
            }
            noise_pred = self._apply_guidance(
                mode,
                eps,
                omega_vid=omega_vid,
                omega_img=omega_img,
                omega_txt=omega_txt,
            )

            noisy = scheduler.step(noise_pred, t, noisy, return_dict=False)
            if isinstance(noisy, tuple):
                noisy = noisy[0]

            # MLX lazy graph: materialize latents each step (standard video denoise uses MemoryGuard).
            ctx.eval(noisy)
            if (step_idx + 1) % 4 == 0:
                ctx.clear_cache()

            if on_progress is not None:
                emit_denoise_progress(on_progress, step_idx + 1, n_steps)
            if on_log is not None:
                on_log("info", f"Bernini-R step {step_idx + 1}/{n_steps}")

        if on_progress is not None:
            emit_complete(on_progress, n_steps)

        pipeline_stub = type("P", (), {"ctx": ctx, "_project_root": self._project_root})()
        if on_log is not None:
            on_log("info", "Decoding Bernini-R latents (VAE)...")
        frames = vae_decode_video(
            pipeline_stub,
            noisy,
            self.entry,
            self.version_key,
            config,
            on_post_log=lambda m: on_log("info", m) if on_log else None,
        )
        if on_log is not None:
            on_log("info", f"Saving video ({len(frames)} frames)...")
        save_video(pipeline_stub, frames, output_path, fps=int(fps))
        return output_path
