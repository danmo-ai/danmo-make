"""LongCat-Video-Avatar 1.5 MLX generation (ATI2V / AT2V)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler
from backend.engine.config.model_configs import LongCatAvatarConfig
from backend.engine.families.longcat_avatar import bundle_load_mlx
from backend.engine.families.longcat_avatar.audio_mlx import load_audio_mel
from backend.engine.families.longcat_avatar.pipeline_mlx import LongCatAvatarPipeline, PipelineConfig
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_post_progress
from backend.engine.runtime.mlx_runtime import run_eval


class LongCatAvatarMlxGenerator:
    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        config: LongCatAvatarConfig | None = None,
        entry: Any | None = None,
        version_key: str | None = None,
    ) -> None:
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LongCatAvatarConfig()
        self.entry = entry
        self.version_key = version_key
        self._pipeline: LongCatAvatarPipeline | None = None
        self._variant_dir: Path | None = None

    @staticmethod
    def _log(on_log: Callable[[str, str], None] | None, level: str, msg: str) -> None:
        if on_log:
            on_log(level, msg)

    def load(self) -> None:
        if getattr(self.ctx, "backend", None) != "mlx":
            raise RuntimeError(
                f"LongCat-Avatar requires MLX runtime (got {getattr(self.ctx, 'backend', None)!r})"
            )
        vae, umt5, whisper, dit, variant_dir = bundle_load.load_longcat_avatar_components(self.bundle_root)
        sched = FlowMatchEulerScheduler(
            num_train_timesteps=int(getattr(self.config, "num_train_timesteps", 1000)),
            shift=float(getattr(self.config, "scheduler_shift", 7.0)),
            ctx=self.ctx,
        )
        pipe_cfg = PipelineConfig(
            num_sampling_steps=int(getattr(self.config, "default_infer_steps", 8)),
            num_train_timesteps=int(getattr(self.config, "num_train_timesteps", 1000)),
            scheduler_shift=float(getattr(self.config, "scheduler_shift", 7.0)),
            text_guidance_scale=float(getattr(self.config, "default_text_guidance", 4.0)),
            audio_guidance_scale=float(getattr(self.config, "default_audio_guidance", 4.0)),
            num_frames=int(getattr(self.config, "default_num_frames", 93)),
            target_fps=int(getattr(self.config, "default_fps", 25)),
        )
        self._pipeline = LongCatAvatarPipeline(
            vae=vae,
            text_encoder=umt5,
            audio_encoder=whisper,
            dit=dit,
            config=pipe_cfg,
            scheduler=sched,
        )
        self._variant_dir = variant_dir

    def generate_and_save(
        self,
        *,
        prompt: str,
        output_path: str,
        reference_image_path: str,
        audio_path: str,
        width: int,
        height: int,
        num_frames: int,
        fps: float,
        seed: int,
        steps: int,
        negative_prompt: str = "",
        on_log: Callable[[str, str], None] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> str:
        if self._pipeline is None or self._variant_dir is None:
            raise RuntimeError("LongCatAvatarMlxGenerator.load() must be called before generate_and_save")

        pipe = self._pipeline
        pipe.config.num_sampling_steps = max(1, int(steps))
        pipe.config.target_fps = int(fps) if fps else pipe.config.target_fps
        n_steps = pipe.config.num_sampling_steps

        self._log(
            on_log,
            "info",
            " ".join(
                [
                    "LongCat-Avatar MLX ATI2V",
                    f"size={width}x{height}",
                    f"frames={num_frames}",
                    f"fps={fps}",
                    f"seed={seed}",
                    f"steps={n_steps}",
                    f"bundle={self.bundle_root.name}",
                ]
            ),
        )

        image = bundle_load.load_reference_image(Path(reference_image_path), height, width)
        audio_mel = load_audio_mel(audio_path)
        text_embeds, text_mask, uncond_embeds, uncond_mask = bundle_load.encode_prompts(
            pipe.text_encoder,
            prompt or " ",
            negative_prompt,
            self._variant_dir,
        )

        emit_denoise_progress(on_progress, step_idx=0, n_steps=n_steps)
        video = pipe(
            image=image,
            audio_mel=audio_mel,
            text_embeds=text_embeds,
            text_mask=text_mask,
            uncond_embeds=uncond_embeds,
            uncond_mask=uncond_mask,
            num_frames=int(num_frames),
            height=int(height),
            width=int(width),
            seed=int(seed),
        )
        run_eval(getattr(self.ctx, "eval", None), video)
        emit_denoise_progress(on_progress, step_idx=n_steps, n_steps=n_steps)
        emit_post_progress(on_progress, n_steps=n_steps, within_post=0.5)

        frames = bundle_load.video_tensor_to_uint8(video)
        out_fps = float(fps) if fps else float(pipe.config.target_fps)
        self._log(on_log, "info", f"LongCat-Avatar encode mp4 → {output_path}")
        return bundle_load.save_mp4(frames, output_path, fps=out_fps)
