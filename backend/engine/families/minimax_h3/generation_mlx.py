"""MiniMax-H3 FL2VA MLX generation orchestration (t2va / first-last-frame)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np
from PIL import Image

from backend.engine.config.model_configs import MinimaxH3Config
from backend.engine.families.minimax_h3 import packing as P
from backend.engine.families.minimax_h3.bundle_load_mlx import load_minimax_h3_components
from backend.engine.families.minimax_h3.scheduler_mlx import MiniMaxH3Scheduler
from backend.engine.families.minimax_h3.vae_mlx import mux_video_audio_mp4
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_post_progress
from backend.engine.runtime.mlx_runtime import run_eval


_PROMPT_EMBED_CACHE: dict[str, tuple[Any, np.ndarray]] = {}


class MinimaxH3MlxGenerator:
    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        config: MinimaxH3Config | None = None,
        entry: Any | None = None,
        version_key: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or MinimaxH3Config()
        self.entry = entry
        self.version_key = version_key
        self._video_vae = None
        self._audio_vae = None
        self._text_encoder = None
        self._dit = None
        self._bundle_cfg: dict[str, Any] | None = None
        self._project_root: Path | None = None
        self._registry: Any | None = None
        self._adapters: list[Any] | None = None

    @staticmethod
    def _log(on_log: Callable[[str, str], None] | None, level: str, msg: str) -> None:
        if on_log:
            on_log(level, msg)

    def load(self) -> None:
        (
            self._video_vae,
            self._audio_vae,
            self._text_encoder,
            self._dit,
            self._bundle_cfg,
        ) = load_minimax_h3_components(self.bundle_root, ctx=self.ctx)

    def _ensure_loaded(self, on_log: Callable[[str, str], None] | None) -> None:
        if self._dit is None:
            self._log(on_log, "info", "MiniMax-H3 loading FL2VA components…")
            self.load()
            self._log(on_log, "info", "MiniMax-H3 components ready")

    def _license_notice(self, on_log: Callable[[str, str], None] | None) -> None:
        if not bool(getattr(self.config, "license_territorial_notice", True)):
            return
        self._log(
            on_log,
            "warning",
            "MiniMax H3 Community License: Applicable Territory excludes the EU, UK, "
            "Republic of Korea, and United States. Confirm jurisdiction before use/distribution.",
        )

    def _resolve_canvas(self, width: int, height: int) -> tuple[int, int]:
        if (
            width % P.MINIMAX_H3_CANVAS_MULTIPLE == 0
            and height % P.MINIMAX_H3_CANVAS_MULTIPLE == 0
            and min(width, height) == P.MINIMAX_H3_SHORT_EDGE
            and width * height <= P.MINIMAX_H3_MAX_PIXELS + P.MINIMAX_H3_CANVAS_MULTIPLE**2
        ):
            return height, width
        return P.resolve_canvas_size(float(width), float(height))

    def _internal_canvas(self, canvas_h: int, canvas_w: int) -> tuple[int, int]:
        preset = str(getattr(self.config, "h3_internal_canvas", "off") or "off").strip().lower()
        if preset in ("", "off", "none"):
            return canvas_h, canvas_w
        try:
            edge = int(preset)
        except ValueError:
            return canvas_h, canvas_w
        if edge <= 0:
            return canvas_h, canvas_w
        ratio = canvas_w / canvas_h
        if ratio >= 1.0:
            return edge, max(P.MINIMAX_H3_CANVAS_MULTIPLE, round(edge * ratio / P.MINIMAX_H3_CANVAS_MULTIPLE) * P.MINIMAX_H3_CANVAS_MULTIPLE)
        return max(P.MINIMAX_H3_CANVAS_MULTIPLE, round(edge / ratio / P.MINIMAX_H3_CANVAS_MULTIPLE) * P.MINIMAX_H3_CANVAS_MULTIPLE), edge

    def _prepare_keyframe(self, path: str, height: int, width: int, *, stretch: bool) -> Image.Image:
        img = Image.open(path).convert("RGB")
        try:
            from PIL import ImageOps

            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.size == (width, height):
            return img
        if stretch:
            return img.resize((width, height), Image.Resampling.LANCZOS)
        scale = max(width / img.size[0], height / img.size[1])
        resized_size = (max(width, round(img.size[0] * scale)), max(height, round(img.size[1] * scale)))
        left = max(0, (resized_size[0] - width) // 2)
        top = max(0, (resized_size[1] - height) // 2)
        resized = img.resize(resized_size, Image.Resampling.LANCZOS)
        return resized.crop((left, top, left + width, top + height))

    def _encode_keyframe_latent(self, image: Image.Image) -> mx.array:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array(P.MINIMAX_H3_PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(P.MINIMAX_H3_PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        chw = np.transpose(arr, (2, 0, 1))[None, ...]
        chw = (chw - mean) / std
        x = mx.array(chw[:, :, None, :, :])
        assert self._video_vae is not None
        if hasattr(self._video_vae, "encode_sample"):
            z = self._video_vae.encode_sample(x, normalize=True)
        elif hasattr(self._video_vae, "encode_mode"):
            z = self._video_vae.encode_mode(x, normalize=True)
        else:
            raise RuntimeError("MiniMax-H3 video VAE missing encode_sample/encode_mode for keyframes")
        run_eval(getattr(self.ctx, "eval", None), z)
        return z

    def _encode_keyframe_rows(self, keyframes: list[Image.Image], patch: tuple[int, int, int]) -> np.ndarray:
        mx.random.seed(P.MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        rows: list[np.ndarray] = []
        for kf in keyframes:
            z = self._encode_keyframe_latent(kf)
            rows.append(P.patchify_video_latents(np.array(z.astype(mx.float32)), patch)[0])
        return np.concatenate(rows, axis=0)

    def _release_dit_after_denoise(self) -> None:
        if not bool(getattr(self.config, "h3_low_memory", True)):
            return
        self._dit = None
        self._text_encoder = None
        clear = getattr(self.ctx, "clear_cache", None)
        if callable(clear):
            clear()
        eval_fn = getattr(self.ctx, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def _validate_inference_plan(
        self, steps: int, on_log: Callable[[str, str], None] | None
    ) -> None:
        from backend.engine.inference.optimization_plan import resolve_video_inference_plan

        plan = resolve_video_inference_plan(
            family="minimax_h3",
            config=self.config,
            ctx=self.ctx,
            num_steps=max(2, int(steps)),
        )
        if bool(getattr(self.config, "h3_turbo", False)) and plan.step_cache_enabled:
            raise RuntimeError(
                "MiniMax-H3 h3_turbo is incompatible with TeaCache/step-cache; "
                "set teacache_mode=none."
            )
        if plan.step_cache_enabled:
            raise RuntimeError(
                "MiniMax-H3 TeaCache is not implemented on the family_generator path; "
                "set teacache_mode=none."
            )
        if plan.use_mlx_compile:
            self._log(
                on_log,
                "warning",
                "mlx.compile is not applied to MiniMax-H3 DiT in this path (use h3_denoiser_reuse / turbo instead).",
            )

    def _apply_turbo_lora(self, on_log: Callable[[str, str], None] | None) -> None:
        from backend.engine.families.minimax_h3.lora_mlx import apply_minimax_h3_turbo_lora

        assert self._dit is not None
        if self._project_root is None or self._registry is None:
            if bool(getattr(self.config, "h3_turbo", False)):
                raise RuntimeError(
                    "MiniMax-H3 turbo LoRA requires project registry context; "
                    "use h3_turbo via the video pipeline or install "
                    "minimax-h3-turbo-lora and select it as an adapter."
                )
            return
        apply_minimax_h3_turbo_lora(
            self._dit,
            bundle_root=self.bundle_root,
            config=self.config,
            adapters=self._adapters,
            project_root=self._project_root,
            registry=self._registry,
            ctx=self.ctx,
            on_log=on_log,
        )

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        guidance: float,
        step_distill: bool,
        image_path: str | None,
        last_frame_path: str | None = None,
        negative_prompt: str = "",
        on_log: Any | None = None,
        on_progress: Any | None = None,
    ) -> str:
        _ = guidance, step_distill
        if (negative_prompt or "").strip():
            raise RuntimeError(
                "MiniMax-H3 FL2VA is CFG-distilled and does not accept a negative prompt."
            )
        if float(fps) != float(P.MINIMAX_H3_FPS):
            raise RuntimeError(f"MiniMax-H3 requires fps={P.MINIMAX_H3_FPS}, got {fps}")
        if bool(getattr(self.config, "h3_block_streaming", False)):
            raise RuntimeError(
                "h3_block_streaming is not implemented yet; disable the flag to run."
            )
        if bool(getattr(self.config, "h3_token_reduction", False)):
            raise RuntimeError(
                "h3_token_reduction is experimental and not implemented; disable the flag to run."
            )
        self._validate_inference_plan(steps, on_log)

        self._license_notice(on_log)
        self._ensure_loaded(on_log)
        assert self._dit is not None and self._text_encoder is not None
        assert self._video_vae is not None and self._audio_vae is not None

        self._apply_turbo_lora(on_log)

        canvas_h, canvas_w = self._resolve_canvas(width, height)
        out_h, out_w = canvas_h, canvas_w
        canvas_h, canvas_w = self._internal_canvas(canvas_h, canvas_w)

        aligned_frames = P.align_num_frames(int(num_frames))
        if aligned_frames != num_frames:
            self._log(
                on_log,
                "info",
                f"MiniMax-H3 snapped num_frames {num_frames} → {aligned_frames} (17n+5)",
            )
        P.validate_duration(aligned_frames)

        if bool(getattr(self.config, "h3_turbo", False)):
            steps = min(int(steps), 8)

        keyframes: list[Image.Image] = []
        anchors: list[str] = []
        if image_path:
            keyframes.append(self._prepare_keyframe(image_path, canvas_h, canvas_w, stretch=True))
            anchors.append("first")
        if last_frame_path:
            keyframes.append(
                self._prepare_keyframe(
                    last_frame_path,
                    canvas_h,
                    canvas_w,
                    stretch=not bool(image_path),
                )
            )
            anchors.append("last")

        self._log(
            on_log,
            "info",
            f"MiniMax-H3 encoding prompt"
            + (f" + {len(keyframes)} keyframe(s) via Qwen3-VL vision…" if keyframes else "…"),
        )
        cache_key = f"{hash(prompt)}:{len(keyframes)}:{canvas_h}x{canvas_w}"
        cached = _PROMPT_EMBED_CACHE.get(cache_key)
        if cached is not None:
            prompt_embeds, tags_np = cached
            prompt_embeds = mx.array(prompt_embeds) if not isinstance(prompt_embeds, mx.array) else prompt_embeds
        else:
            prompt_embeds, text_token_tags = self._text_encoder.encode_prompt(
                prompt,
                images=keyframes or None,
            )
            tags_np = np.array(text_token_tags, dtype=np.int64)
            if isinstance(text_token_tags, mx.array):
                tags_np = np.array(text_token_tags)
            _PROMPT_EMBED_CACHE[cache_key] = (prompt_embeds, tags_np)
        run_eval(getattr(self.ctx, "eval", None), prompt_embeds)

        num_latent_frames = P.video_latent_num_frames(aligned_frames)
        latent_h = canvas_h // P.VAE_SPATIAL_SCALE
        latent_w = canvas_w // P.VAE_SPATIAL_SCALE
        num_audio = P.audio_latent_num_frames(aligned_frames)
        patch = tuple(getattr(self.config, "patch_size", P.PATCH_SIZE))

        tags_np = np.array(tags_np, dtype=np.int64)
        layout = P.build_packed_sequence(
            tags_np,
            num_latent_frames=num_latent_frames,
            latent_height=latent_h,
            latent_width=latent_w,
            num_audio_latents=num_audio,
            patch_size=patch,
            keyframe_anchors=tuple(anchors),
        )

        video_channels = int(getattr(self.config, "latent_channels", 24))
        audio_channels = int(getattr(self.config, "audio_latent_channels", 32))
        pt, ph, pw = patch
        patch_dim = video_channels * pt * ph * pw
        rows_per_frame = (latent_h // ph) * (latent_w // pw)
        num_target_video_rows = num_latent_frames * rows_per_frame
        num_cond_rows = layout.num_condition_video_rows
        num_audio_rows = num_audio * P.MINIMAX_H3_AUDIO_CHANNELS

        cond_video = np.zeros((0, patch_dim), dtype=np.float32)
        if keyframes:
            cond_video = self._encode_keyframe_rows(keyframes, patch)

        mx.random.seed(int(seed))
        if cond_video.size:
            cond_noise = np.array(
                mx.random.normal(cond_video.shape).astype(mx.float32),
                dtype=np.float32,
            )
            video_sched_tmp = MiniMaxH3Scheduler(
                shift=float(getattr(self.config, "scheduler_shift", 12.0))
            )
            cond_video = np.array(
                video_sched_tmp.scale_noise(
                    mx.array(cond_video),
                    P.MINIMAX_H3_KEYFRAME_NOISE_AUG,
                    mx.array(cond_noise),
                ).astype(mx.float32)
            )

        target_video = np.array(
            mx.random.normal((num_target_video_rows, patch_dim)).astype(mx.float32),
            dtype=np.float32,
        )
        video_rows = np.concatenate([cond_video, target_video], axis=0)
        audio_rows = np.array(
            mx.random.normal((num_audio_rows, audio_channels)).astype(mx.float32),
            dtype=np.float32,
        )

        video_shift = float(
            (self._bundle_cfg or {}).get("sigma_shift_scales", {}).get("video")
            or getattr(self.config, "scheduler_shift", 12.0)
        )
        audio_shift = float(
            (self._bundle_cfg or {}).get("sigma_shift_scales", {}).get("audio")
            or getattr(self.config, "audio_scheduler_shift", 3.0)
        )
        n_steps = max(2, int(steps))
        video_sched = MiniMaxH3Scheduler(shift=video_shift)
        audio_sched = MiniMaxH3Scheduler(shift=audio_shift)
        video_sched.set_timesteps(n_steps)
        audio_sched.set_timesteps(n_steps)
        num_evals = int(video_sched.num_inference_steps or 0)

        reuse = max(1, int(getattr(self.config, "h3_denoiser_reuse", 1) or 1))
        active_layers = int(getattr(self.config, "h3_active_layers", 50) or 50)
        eval_indices = set(range(num_evals))
        if reuse > 1 and num_evals > 2:
            interval = reuse
            eval_indices = {0, num_evals - 1}
            eval_indices.update(range(0, num_evals, interval))

        position_ids = mx.array(layout.position_ids.astype(np.float32))
        token_tags = mx.array(layout.token_tags.astype(np.int32))
        video_indices = mx.array(layout.video_indices.astype(np.int32))
        audio_indices = mx.array(layout.audio_indices.astype(np.int32))
        text_indices = mx.array(layout.text_indices.astype(np.int32))

        latents = mx.array(video_rows)
        audio_latents = mx.array(audio_rows)
        last_v_pred: mx.array | None = None
        last_a_pred: mx.array | None = None

        self._log(
            on_log,
            "info",
            f"MiniMax-H3 denoise {num_evals} steps @ {canvas_w}x{canvas_h}"
            + (f" (internal; output {out_w}x{out_h})" if (out_w, out_h) != (canvas_w, canvas_h) else "")
            + f", {aligned_frames} frames, seed={seed}",
        )

        for i in range(num_evals):
            v_t = float(video_sched.timesteps[i].item())
            a_t = float(audio_sched.timesteps[i].item())
            cond_v_t = max(v_t, P.MINIMAX_H3_KEYFRAME_NOISE_AUG)
            unique_t, t_idx = P.build_row_timesteps(
                layout,
                video_timestep=v_t,
                audio_timestep=a_t,
                condition_video_timestep=cond_v_t,
                condition_audio_timestep=1.0,
            )
            if i in eval_indices or last_v_pred is None:
                dit_kwargs: dict[str, Any] = dict(
                    hidden_states=latents[None],
                    audio_hidden_states=audio_latents[None],
                    encoder_hidden_states=prompt_embeds,
                    timestep=mx.array(unique_t),
                    timestep_indices=mx.array(t_idx),
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    return_dict=False,
                )
                if active_layers < 50 and hasattr(self._dit, "set_active_layers"):
                    self._dit.set_active_layers(active_layers)
                out = self._dit(**dit_kwargs)
                noise_pred, audio_noise_pred = out
                run_eval(getattr(self.ctx, "eval", None), noise_pred, audio_noise_pred)
                last_v_pred = noise_pred[0]
                last_a_pred = audio_noise_pred[0]
            else:
                assert last_v_pred is not None and last_a_pred is not None
                noise_pred = last_v_pred[None]
                audio_noise_pred = last_a_pred[None]

            if num_cond_rows:
                gen_v = latents[num_cond_rows:]
                stepped_v = video_sched.step(
                    noise_pred[0, num_cond_rows:].astype(mx.float32), v_t, gen_v
                )
                latents = mx.concatenate([latents[:num_cond_rows], stepped_v], axis=0)
            else:
                latents = video_sched.step(noise_pred[0].astype(mx.float32), v_t, latents)
            audio_latents = audio_sched.step(
                audio_noise_pred[0].astype(mx.float32), a_t, audio_latents
            )
            run_eval(getattr(self.ctx, "eval", None), latents, audio_latents)
            emit_denoise_progress(on_progress, i + 1, num_evals)

        target_tokens = latents[num_cond_rows:]
        video_latent = mx.array(
            P.unpatchify_video_tokens(
                np.array(target_tokens.astype(mx.float32)),
                num_latent_frames=num_latent_frames,
                latent_height=latent_h,
                latent_width=latent_w,
                latent_channels=video_channels,
                patch_size=patch,
            )
        )
        audio_np = np.array(audio_latents.astype(mx.float32))
        left = audio_np[:num_audio].T
        right = audio_np[num_audio:].T
        audio_latent = mx.array(np.stack([left, right], axis=0))

        self._release_dit_after_denoise()
        emit_post_progress(on_progress, n_steps=num_evals, within_post=0.2)
        self._log(on_log, "info", f"MiniMax-H3 decode+mux → {output_path}")
        stream_decode = bool(getattr(self.config, "h3_stream_decode", True))
        result = mux_video_audio_mp4(
            self.ctx,
            video_latent,
            audio_latent,
            output_path,
            self.bundle_root,
            frame_rate=float(P.MINIMAX_H3_FPS),
            video_vae=self._video_vae,
            audio_vae=self._audio_vae,
            stream_frames=stream_decode,
            upscale_to=(out_h, out_w) if (out_h, out_w) != (canvas_h, canvas_w) else None,
            on_log=(lambda m: self._log(on_log, "info", m)) if on_log else None,
        )
        emit_post_progress(on_progress, n_steps=num_evals, within_post=1.0)
        return result
