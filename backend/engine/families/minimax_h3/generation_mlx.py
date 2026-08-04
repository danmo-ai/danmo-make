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
from backend.engine.families.minimax_h3.vae_mlx import mux_video_audio_mp4
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_post_progress
from backend.engine.runtime.mlx_runtime import run_eval


def _euler_flow_step(x: mx.array, velocity: mx.array, sigma: float, sigma_next: float) -> mx.array:
    """Flow-matching Euler: ``x' = x + (σ' - σ) * v`` with descending σ ∈ (0, 1]."""
    dt = float(sigma_next) - float(sigma)
    return x + velocity * dt


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

    @staticmethod
    def _log(on_log: Callable[[str, str], None] | None, level: str, msg: str) -> None:
        if on_log:
            on_log(level, msg)

    def load(self) -> None:
        if getattr(self.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"MiniMax-H3 requires MLX runtime (got {getattr(self.ctx, 'backend', None)!r})"
            )
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
        # Prefer registry/request size when already on the H3 canvas grid; else resolve from ratio.
        if (
            width % P.MINIMAX_H3_CANVAS_MULTIPLE == 0
            and height % P.MINIMAX_H3_CANVAS_MULTIPLE == 0
            and min(width, height) == P.MINIMAX_H3_SHORT_EDGE
            and width * height <= P.MINIMAX_H3_MAX_PIXELS + P.MINIMAX_H3_CANVAS_MULTIPLE**2
        ):
            return height, width
        return P.resolve_canvas_size(float(width), float(height))

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
        """Encode one RGB keyframe → ``(1, 24, 1, H/16, W/16)`` normalized latent."""
        arr = np.asarray(image, dtype=np.float32) / 255.0
        mean = np.array(P.MINIMAX_H3_PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(P.MINIMAX_H3_PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        chw = np.transpose(arr, (2, 0, 1))[None, ...]  # 1,3,H,W
        chw = (chw - mean) / std
        # Video VAE expects NCTHW; single frame → T=1 then pad? Diffusers uses clip encode.
        x = mx.array(chw[:, :, None, :, :])  # 1,3,1,H,W
        assert self._video_vae is not None
        # Prefer encode_mode / encode_clip if present.
        if hasattr(self._video_vae, "encode_mode"):
            z = self._video_vae.encode_mode(x, normalize=True)
        elif hasattr(self._video_vae, "encode_clip"):
            z = self._video_vae.encode_clip(x)
        else:
            raise RuntimeError("MiniMax-H3 video VAE missing encode_mode/encode_clip for keyframes")
        run_eval(getattr(self.ctx, "eval", None), z)
        return z

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
            raise RuntimeError(
                f"MiniMax-H3 requires fps={P.MINIMAX_H3_FPS}, got {fps}"
            )

        self._license_notice(on_log)
        self._ensure_loaded(on_log)
        assert self._dit is not None and self._text_encoder is not None
        assert self._video_vae is not None and self._audio_vae is not None

        canvas_h, canvas_w = self._resolve_canvas(width, height)
        aligned_frames = P.align_num_frames(int(num_frames))
        if aligned_frames != num_frames:
            self._log(
                on_log,
                "info",
                f"MiniMax-H3 snapped num_frames {num_frames} → {aligned_frames} (17n+5)",
            )
        P.validate_duration(aligned_frames)

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
        prompt_embeds, text_token_tags = self._text_encoder.encode_prompt(
            prompt,
            images=keyframes or None,
        )
        run_eval(getattr(self.ctx, "eval", None), prompt_embeds)

        num_latent_frames = P.video_latent_num_frames(aligned_frames)
        latent_h = canvas_h // P.VAE_SPATIAL_SCALE
        latent_w = canvas_w // P.VAE_SPATIAL_SCALE
        num_audio = P.audio_latent_num_frames(aligned_frames)
        patch = tuple(getattr(self.config, "patch_size", P.PATCH_SIZE))

        tags_np = np.array(text_token_tags, dtype=np.int64)
        if isinstance(text_token_tags, mx.array):
            tags_np = np.array(text_token_tags)
        layout = P.build_packed_sequence(
            tags_np,
            num_latent_frames=num_latent_frames,
            latent_height=latent_h,
            latent_width=latent_w,
            num_audio_latents=num_audio,
            patch_size=patch,
            keyframe_anchors=tuple(anchors),
        )

        rng = np.random.default_rng(int(seed))
        video_channels = int(getattr(self.config, "latent_channels", 24))
        audio_channels = int(getattr(self.config, "audio_latent_channels", 32))
        pt, ph, pw = patch
        patch_dim = video_channels * pt * ph * pw
        rows_per_frame = (latent_h // ph) * (latent_w // pw)
        num_target_video_rows = num_latent_frames * rows_per_frame
        num_cond_rows = layout.num_condition_video_rows
        num_audio_rows = num_audio * P.MINIMAX_H3_AUDIO_CHANNELS

        # Conditioning keyframe latents (noised to KEYFRAME_NOISE_AUG).
        cond_rows_list: list[np.ndarray] = []
        if keyframes:
            for kf in keyframes:
                z = self._encode_keyframe_latent(kf)  # 1,C,1,h,w
                z_np = np.array(z.astype(mx.float32))
                noise = rng.standard_normal(z_np.shape).astype(np.float32)
                t_aug = P.MINIMAX_H3_KEYFRAME_NOISE_AUG
                z_noisy = (1.0 - t_aug) * z_np + t_aug * noise
                cond_rows_list.append(P.patchify_video_latents(z_noisy, patch)[0])
            cond_video = np.concatenate(cond_rows_list, axis=0)
        else:
            cond_video = np.zeros((0, patch_dim), dtype=np.float32)

        target_video = rng.standard_normal((num_target_video_rows, patch_dim)).astype(np.float32)
        video_rows = np.concatenate([cond_video, target_video], axis=0)
        audio_rows = rng.standard_normal((num_audio_rows, audio_channels)).astype(np.float32)

        video_shift = float(
            (self._bundle_cfg or {}).get("sigma_shift_scales", {}).get("video")
            or getattr(self.config, "scheduler_shift", 12.0)
        )
        audio_shift = float(
            (self._bundle_cfg or {}).get("sigma_shift_scales", {}).get("audio")
            or getattr(self.config, "audio_scheduler_shift", 3.0)
        )
        n_steps = max(2, int(steps))
        video_sigmas = P.flow_match_sigmas(n_steps, shift=video_shift)
        audio_sigmas = P.flow_match_sigmas(n_steps, shift=audio_shift)
        # Model evals = len(sigmas) - 1
        num_evals = len(video_sigmas) - 1

        position_ids = mx.array(layout.position_ids.astype(np.float32))
        token_tags = mx.array(layout.token_tags.astype(np.int32))
        video_indices = mx.array(layout.video_indices.astype(np.int32))
        audio_indices = mx.array(layout.audio_indices.astype(np.int32))
        text_indices = mx.array(layout.text_indices.astype(np.int32))

        latents = mx.array(video_rows)
        audio_latents = mx.array(audio_rows)

        self._log(
            on_log,
            "info",
            f"MiniMax-H3 denoise {num_evals} steps @ {canvas_w}x{canvas_h}, "
            f"{aligned_frames} frames, seed={seed}",
        )

        for i in range(num_evals):
            v_t = float(video_sigmas[i])
            a_t = float(audio_sigmas[i])
            v_next = float(video_sigmas[i + 1])
            a_next = float(audio_sigmas[i + 1])
            unique_t, t_idx = P.build_row_timesteps(
                layout,
                video_timestep=v_t,
                audio_timestep=a_t,
                condition_video_timestep=P.MINIMAX_H3_KEYFRAME_NOISE_AUG,
                condition_audio_timestep=P.MINIMAX_H3_KEYFRAME_NOISE_AUG,
            )
            out = self._dit(
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
            noise_pred, audio_noise_pred = out
            run_eval(getattr(self.ctx, "eval", None), noise_pred, audio_noise_pred)

            # Step only generated rows; conditioning rows stay pinned.
            if num_cond_rows:
                gen_v = latents[num_cond_rows:]
                gen_v = _euler_flow_step(gen_v, noise_pred[0, num_cond_rows:], v_t, v_next)
                latents = mx.concatenate([latents[:num_cond_rows], gen_v], axis=0)
            else:
                latents = _euler_flow_step(latents, noise_pred[0], v_t, v_next)
            audio_latents = _euler_flow_step(audio_latents, audio_noise_pred[0], a_t, a_next)
            run_eval(getattr(self.ctx, "eval", None), latents, audio_latents)
            emit_denoise_progress(on_progress, i + 1, num_evals)

        # Unpack target video / audio latents for decode.
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
        # Audio: channel-major rows → [2, 32, T]
        audio_np = np.array(audio_latents.astype(mx.float32))
        left = audio_np[:num_audio].T  # 32, T
        right = audio_np[num_audio:].T
        audio_latent = mx.array(np.stack([left, right], axis=0))  # 2, 32, T

        emit_post_progress(on_progress, n_steps=num_evals, within_post=0.2)
        self._log(on_log, "info", f"MiniMax-H3 decode+mux → {output_path}")
        result = mux_video_audio_mp4(
            self.ctx,
            video_latent,
            audio_latent,
            output_path,
            self.bundle_root,
            frame_rate=float(P.MINIMAX_H3_FPS),
            on_log=(lambda m: self._log(on_log, "info", m)) if on_log else None,
        )
        emit_post_progress(on_progress, n_steps=num_evals, within_post=1.0)
        return result
