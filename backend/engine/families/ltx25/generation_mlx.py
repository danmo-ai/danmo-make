"""LTX-2.5 distilled T2V/I2V generation — in-repo MLX orchestration.

Two-stage flow (mirrors upstream ``DistilledPipeline``, model_version >= 2.5):

* Stage 1 at half resolution with the 8-step distilled sigma schedule and the
  ancestral (SDE) Euler sampler (``eta=1``).
* Spatial latent upsampler x2.
* Stage 2 at full resolution with the 3-step refined schedule, deterministic
  Euler, starting from the re-noised stage-1 latents (``noise_scale=σ0``).
* Audio is generated jointly by the A/V DiT in both stages; final decode muxes
  video + audio into an mp4 via ffmpeg.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from backend.engine.config.model_configs import LTX25Config
from backend.engine.families.ltx.generation_mlx import _preprocess_i2v_image
from backend.engine.families.ltx.pipeline_math_mlx import (
    DEFAULT_LTX_IMAGE_CRF,
    AudioPatchifier,
    LatentState,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    apply_conditioning,
    apply_denoise_mask,
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_latent_shape,
    compute_video_positions,
    create_noised_state,
    pin_latent_by_mask,
)
from backend.engine.families.ltx25.pipeline_math_mlx import (
    ANCESTRAL_ETA,
    ANCESTRAL_NOISE_SEED_OFFSET,
    ANCESTRAL_S_NOISE,
    DISTILLED_SIGMAS,
    STAGE_2_SIGMAS,
)
from backend.engine.families.ltx25.text_encoder_mlx import LTX25PromptEncoder
from backend.engine.families.ltx25.transformer_mlx import LTX25X0Model, load_ltx25_x0_model
from backend.engine.families.ltx25.upsampler_mlx import upsample_video_latent
from backend.engine.families.ltx25.vae_mlx import load_ltx25_video_encoder, mux_video_audio_mp4
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress, emit_post_progress
from backend.engine.runtime._base import RuntimeContext
from backend.engine.runtime.mlx_runtime import run_eval

_AUDIO_CHANNELS = 8
_AUDIO_MEL_BINS = 16
# Checkpoint RoPE time axis (positional_embedding_max_pos[0]). 24fps × 20s = 481 frames (8k+1).
_MAX_POSITION_SECONDS = 20.0

_SPOKEN_QUOTE = re.compile(r'"([^"]+)"|“([^”]+)”|「([^」]+)」')
# (path, face index). The lower-third is a heavy squared gothic, same weight as
# the "AI" badge — Heiti SC Medium, not Hiragino Sans GB W3.
_CJK_FONT_CANDIDATES = (
    ("/System/Library/Fonts/STHeiti Medium.ttc", 1),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 2),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
)
_CAPTION_PIECE = re.compile(r"[A-Za-z0-9]+|.")
# Navy sampled from the badge letters on the caption pill.
_CAPTION_INK = (2, 25, 73)


def spoken_caption_lines(prompt: str) -> list[str]:
    """Quoted lines that should be burned as on-screen captions.

    Diffusion paints Chinese glyphs as unrelated characters. Only CJK quotes
    are rewritten; an English ``says "hello"`` caption is left alone.
    """
    lines: list[str] = []
    for match in _SPOKEN_QUOTE.finditer(prompt or ""):
        line = next(group for group in match.groups() if group)
        line = " ".join(line.split())
        if line:
            lines.append(line)
    if not any("\u4e00" <= ch <= "\u9fff" for line in lines for ch in line):
        return []
    return lines


def caption_line_at(lines: list[str], frame_index: int, frame_count: int) -> str:
    """Pick the spoken line for this frame, weighted by character count."""
    if not lines:
        raise ValueError("caption_line_at requires at least one line")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    weights = [max(1, len(line)) for line in lines]
    total = sum(weights)
    # Last frame stays on the last line (pos would otherwise land on the boundary).
    pos = min(frame_index, frame_count - 1) / frame_count * total
    acc = 0
    for line, weight in zip(lines, weights):
        acc += weight
        if pos < acc:
            return line
    return lines[-1]


def _cjk_font(size: int):
    from PIL import ImageFont

    for path, index in _CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size, index=index)
    raise RuntimeError(
        "Cannot burn Chinese captions: no CJK font found "
        "(expected Heiti SC Medium, Hiragino Sans GB W6, or Noto Sans CJK)."
    )


def _text_width(draw, text: str, font) -> float:
    if hasattr(draw, "textlength"):
        return float(draw.textlength(text, font=font))
    box = draw.textbbox((0, 0), text, font=font)
    return float(box[2] - box[0])


def _wrap_caption(draw, text: str, font, max_width: int) -> list[str]:
    """Wrap on CJK characters, but keep runs like ``PRO6000`` and ``5090`` intact."""
    lines: list[str] = []
    current = ""
    for piece in _CAPTION_PIECE.findall(text):
        trial = current + piece
        if current and _text_width(draw, trial, font) > max_width:
            lines.append(current)
            current = piece
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [text]


def _fixed_caption_box(image) -> tuple[int, int, int, int]:
    """One lower-third rectangle for the whole clip.

    Geometry comes from the frame size only. Tracking bright clothing or a
    wobbling painted bar makes the rectangle jump every frame.
    """
    width, height = image.size
    margin_x = max(24, int(width * 0.05))
    bar_h = max(96, int(height * 0.11))
    bottom = max(20, int(height * 0.04))
    y2 = height - bottom
    y1 = max(0, y2 - bar_h)
    return margin_x, y1, width - margin_x, y2


def _caption_text_bounds(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1 + 28, y1 + 14, x2 - 28, y2 - 14


def _fit_caption_font(draw, lines: list[str], max_w: int, max_h: int):
    """One font size for every spoken line, so the bar does not resize mid-clip."""
    for size in range(42, 13, -2):
        font = _cjk_font(size)
        wrapped = [_wrap_caption(draw, line, font, max_w) for line in lines]
        if any(len(rows) > 2 for rows in wrapped):
            continue
        if any(_text_width(draw, row, font) > max_w + 1 for rows in wrapped for row in rows):
            continue
        if (size + 4) * max(len(rows) for rows in wrapped) <= max_h:
            return font
    return _cjk_font(14)


def _paint_caption_line(image, box: tuple[int, int, int, int], line: str, font) -> None:
    from PIL import ImageDraw

    x1, y1, x2, y2 = box
    text_left, text_top, text_right, text_bottom = _caption_text_bounds(box)
    draw = ImageDraw.Draw(image)
    radius = max(12, (y2 - y1) // 2)
    draw.rounded_rectangle((x1, y1, x2, y2), radius=radius, fill=(255, 255, 255))
    if text_right - text_left < 40 or text_bottom - text_top < 20:
        return
    max_w = text_right - text_left
    max_h = text_bottom - text_top
    wrapped = _wrap_caption(draw, line, font, max_w)[:2]
    line_h = font.size + 4
    block_h = line_h * len(wrapped)
    y = text_top + max(0, (max_h - block_h) // 2)
    for row in wrapped:
        width = _text_width(draw, row, font)
        x = text_left + max(0, int((max_w - width) / 2))
        draw.text((x, y), row, font=font, fill=_CAPTION_INK)
        y += line_h


def rewrite_spoken_captions(frames: list, lines: list[str]) -> int:
    """Burn the quoted spoken lines onto one fixed lower-third bar."""
    if not frames or not lines:
        return 0
    from PIL import ImageDraw

    box = _fixed_caption_box(frames[0])
    text_left, text_top, text_right, text_bottom = _caption_text_bounds(box)
    font = _fit_caption_font(
        ImageDraw.Draw(frames[0]),
        lines,
        text_right - text_left,
        text_bottom - text_top,
    )
    count = len(frames)
    for index, frame in enumerate(frames):
        _paint_caption_line(frame, box, caption_line_at(lines, index, count), font)
    return count


def _materialize(ctx: RuntimeContext, *arrays: mx.array) -> None:
    run_eval(getattr(ctx, "eval", None), *arrays)


def _is_uniform_mask(mask: mx.array) -> bool:
    return bool(mx.all(mask == 1.0).item())


def _per_token_timesteps(ctx: RuntimeContext, sigma: float, denoise_mask: mx.array) -> mx.array:
    return (denoise_mask * sigma).squeeze(-1)


def _snap_frames(frames: int, temporal_scale: int = 8) -> int:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    return ((frames - 1) // temporal_scale) * temporal_scale + 1


def _assert_resolution(height: int, width: int) -> None:
    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(
            f"LTX 2.5 two-stage resolution ({height}x{width}) must be divisible by 64 "
            "(stage 1 runs at half resolution, which must be divisible by 32)."
        )


def _euler_step(x: mx.array, x0: mx.array, sigma: float, sigma_next: float) -> mx.array:
    if sigma == 0:
        return x0
    d = (x - x0) / sigma
    return x + (sigma_next - sigma) * d


def _ancestral_euler_step(
    x: mx.array,
    x0: mx.array,
    sigma: float,
    sigma_next: float,
    noise: mx.array,
    *,
    eta: float = ANCESTRAL_ETA,
    s_noise: float = ANCESTRAL_S_NOISE,
) -> mx.array:
    """Ancestral (SDE) Euler step for rectified-flow models (upstream)."""
    x32 = x.astype(mx.float32)
    d32 = x0.astype(mx.float32)
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    ratio = sigma_down / sigma
    x_next = ratio * x32 + (1.0 - ratio) * d32
    alpha_next = 1.0 - sigma_next
    alpha_down = 1.0 - sigma_down
    renoise_coeff = math.sqrt(max(sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2, 0.0))
    x_next = (alpha_next / alpha_down) * x_next + noise.astype(mx.float32) * s_noise * renoise_coeff
    return x_next.astype(x.dtype)


def _denoise_loop(
    ctx: RuntimeContext,
    model: LTX25X0Model,
    video_state: LatentState,
    audio_state: LatentState,
    video_text_embeds: mx.array,
    audio_text_embeds: mx.array,
    sigmas: list[float],
    *,
    ancestral: bool,
    noise_seed: int,
    on_progress: Callable[..., None] | None = None,
    progress_step_offset: int = 0,
    progress_total_steps: int = 1,
    on_log: Callable[[str, str], None] | None = None,
    progress_label: str = "denoise",
) -> tuple[mx.array, mx.array]:
    video_x = video_state.latent
    audio_x = audio_state.latent
    video_uniform = _is_uniform_mask(video_state.denoise_mask)
    audio_uniform = _is_uniform_mask(audio_state.denoise_mask)
    n_steps = max(1, len(sigmas) - 1)
    rng_key = mx.random.key(noise_seed)

    for step_idx, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
        sigma_arr = ctx.array([sigma], dtype=ctx.bfloat16())
        b = int(video_x.shape[0])
        call_kwargs: dict[str, Any] = dict(
            video_latent=video_x,
            audio_latent=audio_x,
            sigma=ctx.broadcast_to(sigma_arr, (b,)),
            video_text_embeds=video_text_embeds,
            audio_text_embeds=audio_text_embeds,
            video_positions=video_state.positions,
            audio_positions=audio_state.positions,
        )
        if not video_uniform:
            call_kwargs["video_timesteps"] = _per_token_timesteps(ctx, sigma, video_state.denoise_mask)
        if not audio_uniform:
            call_kwargs["audio_timesteps"] = _per_token_timesteps(ctx, sigma, audio_state.denoise_mask)

        video_x0, audio_x0 = model(**call_kwargs)
        video_x0 = apply_denoise_mask(ctx, video_x0, video_state.clean_latent, video_state.denoise_mask)
        audio_x0 = apply_denoise_mask(ctx, audio_x0, audio_state.clean_latent, audio_state.denoise_mask)

        if sigma_next == 0:
            video_x, audio_x = video_x0, audio_x0
        else:
            if ancestral:
                rng_key, vk = mx.random.split(rng_key)
                video_noise = mx.random.normal(video_x.shape, key=vk, dtype=video_x.dtype)
                rng_key, ak = mx.random.split(rng_key)
                audio_noise = mx.random.normal(audio_x.shape, key=ak, dtype=audio_x.dtype)
                video_x = _ancestral_euler_step(video_x, video_x0, sigma, sigma_next, video_noise)
                audio_x = _ancestral_euler_step(audio_x, audio_x0, sigma, sigma_next, audio_noise)
            else:
                video_x = _euler_step(video_x, video_x0, sigma, sigma_next)
                audio_x = _euler_step(audio_x, audio_x0, sigma, sigma_next)
            video_x = pin_latent_by_mask(ctx, video_x, video_state.clean_latent, video_state.denoise_mask)
            audio_x = pin_latent_by_mask(ctx, audio_x, audio_state.clean_latent, audio_state.denoise_mask)
        _materialize(ctx, video_x, audio_x)

        step_1based = progress_step_offset + step_idx + 1
        emit_denoise_progress(on_progress, step_1based, progress_total_steps)
        if on_log:
            on_log(
                "info",
                f"{progress_label} step {step_1based}/{progress_total_steps} "
                f"(σ {sigma:.4f} → {sigma_next:.4f})",
            )

    return video_x, audio_x


def _load_i2v_image_tensor(
    image_path: str,
    enc_h: int,
    enc_w: int,
    *,
    crf: int = DEFAULT_LTX_IMAGE_CRF,
) -> np.ndarray:
    """Load + H.264 CRF round-trip + resize (center-crop) → ``(1, C, H, W)`` in [-1, 1].

    Official LTX I2V (and our working LTX 2.3 path) run a single-frame H.264
    encode/decode at ``DEFAULT_LTX_IMAGE_CRF`` before VAE encode so first-frame
    conditioning matches the compressed-video domain the DiT was trained on.
    Skipping CRF leaves a "clean photo" pin that later frames cannot follow,
    which shows up as chroma smear / grid collapse after ~1s.
    """
    from PIL import Image

    arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    arr = _preprocess_i2v_image(arr, float(crf))
    img = Image.fromarray(arr, mode="RGB")
    w, h = img.size
    scale = max(enc_w / w, enc_h / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - enc_w) // 2
    top = (nh - enc_h) // 2
    img = img.crop((left, top, left + enc_w, top + enc_h))
    out = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return out.transpose(2, 0, 1)[None]  # (1, C, H, W)


def _i2v_conditionings(
    ctx: RuntimeContext,
    image_path: str,
    *,
    enc_h: int,
    enc_w: int,
    video_encoder: Any,
    strength: float = 1.0,
) -> list[VideoConditionByLatentIndex]:
    pixels = _load_i2v_image_tensor(image_path, enc_h, enc_w)
    pixels = ctx.array(pixels.astype(np.float32))
    pixels = ctx.expand_dims(pixels, axis=2)  # (1, C, 1, H, W) single-frame clip
    latent = video_encoder.encode(pixels)
    patchifier = VideoLatentPatchifier()
    tokens, _ = patchifier.patchify(latent, ctx)
    strength = float(max(0.0, min(1.0, strength)))
    return [VideoConditionByLatentIndex(latent=tokens, frame_idx=0, strength=strength)]


class LTX25MlxGenerator:
    """In-repo two-stage LTX 2.5 distilled T2V/I2V generator (MLX)."""

    def __init__(
        self,
        ctx: RuntimeContext,
        bundle_root: Path,
        config: LTX25Config | None = None,
        *,
        entry: Any | None = None,
        version_key: str | None = None,
    ):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self.config = config or LTX25Config()
        self._registry_entry = entry
        self._version_key = version_key
        self._encoder: LTX25PromptEncoder | None = None
        self._dit: LTX25X0Model | None = None

    def load(self) -> None:
        """No-op — components load lazily in ``generate_and_save``."""

    def _log(self, on_log: Callable[[str, str], None] | None, level: str, message: str) -> None:
        if on_log:
            on_log(level, message)

    def _load_dit(self, on_log: Callable[[str, str], None] | None = None) -> LTX25X0Model:
        if self._dit is None:
            self._log(on_log, "info", "Loading LTX 2.5 distilled transformer")
            self._dit = load_ltx25_x0_model(
                self.ctx,
                self.bundle_root,
                self.config,
                entry=self._registry_entry,
                version_key=self._version_key,
                load_fn=getattr(self.ctx, "load_weights", None),
            )
        return self._dit

    def _encode_prompts(
        self,
        prompt: str,
        on_log: Callable[[str, str], None] | None = None,
    ) -> tuple[mx.array, mx.array]:
        if self._encoder is None:
            self._encoder = LTX25PromptEncoder(self.ctx, self.bundle_root, self.config)
        self._log(on_log, "info", "Encoding prompt (Gemma 4 + connectors)")
        video_embeds, audio_embeds = self._encoder.encode(prompt, on_log=on_log)
        return video_embeds, audio_embeds

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
        negative_prompt: str = "",
        on_log: Callable[[str, str], None] | None = None,
        on_progress: Callable[..., None] | None = None,
    ) -> str:
        ctx = self.ctx
        if not step_distill:
            raise RuntimeError(
                "LTX 2.5 supports the distilled 8-step pipeline only (step_distill=true required)."
            )
        if guidance not in (1.0, 0.0):
            raise RuntimeError(
                f"LTX 2.5 distilled sampling fixes guidance=1.0 (got {guidance})."
            )
        if negative_prompt:
            raise RuntimeError("LTX 2.5 distilled sampling does not use a negative prompt.")
        _assert_resolution(height, width)
        num_frames = _snap_frames(max(1, int(num_frames)))
        if num_frames < 9:
            num_frames = 9
        fps = max(1.0, float(fps))
        duration_sec = (num_frames - 1) / fps
        if duration_sec > _MAX_POSITION_SECONDS + 1e-6:
            raise RuntimeError(
                f"LTX 2.5 positional time axis stops at {_MAX_POSITION_SECONDS:.0f}s "
                f"({num_frames} frames at {fps:.0f}fps is {duration_sec:.2f}s). "
                "Use 481 frames at 24fps for the longest single shot."
            )

        stage1_steps = max(1, len(DISTILLED_SIGMAS) - 1)
        stage2_steps = max(1, len(STAGE_2_SIGMAS) - 1)
        if int(steps) != stage1_steps:
            self._log(
                on_log,
                "warning",
                f"LTX 2.5 distilled stage-1 uses a fixed {stage1_steps}-step schedule "
                f"(requested {steps}); ignoring.",
            )
        total_steps = stage1_steps + stage2_steps

        video_embeds, audio_embeds = self._encode_prompts(prompt, on_log=on_log)
        _materialize(ctx, video_embeds, audio_embeds)
        dit = self._load_dit(on_log=on_log)
        load_fn = getattr(ctx, "load_weights", None)

        # --- Stage 1 (half resolution, ancestral Euler) ---
        stage1_w, stage1_h = width // 2, height // 2
        f_lat, h_lat, w_lat = compute_video_latent_shape(num_frames, stage1_h, stage1_w)
        audio_tokens = compute_audio_token_count(num_frames, fps)

        video_positions = compute_video_positions(ctx, f_lat, h_lat, w_lat, fps)
        audio_positions = compute_audio_positions(ctx, audio_tokens)

        conditionings: list[VideoConditionByLatentIndex] = []
        if image_path:
            video_encoder = load_ltx25_video_encoder(self.bundle_root, load_fn=load_fn)
            conditionings = _i2v_conditionings(
                ctx, image_path, enc_h=stage1_h, enc_w=stage1_w, video_encoder=video_encoder,
            )

        video_state = create_noised_state(
            ctx,
            (1, f_lat * h_lat * w_lat, 128),
            conditionings=conditionings,
            spatial_dims=(f_lat, h_lat, w_lat),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
        )
        audio_state = create_noised_state(
            ctx,
            (1, audio_tokens, _AUDIO_CHANNELS * _AUDIO_MEL_BINS),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
        )
        _materialize(ctx, video_state.latent, audio_state.latent)

        self._log(on_log, "info", f"Stage 1: {stage1_w}x{stage1_h} {num_frames}f @ {fps}fps ({stage1_steps} steps)")
        video_x, audio_x = _denoise_loop(
            ctx,
            dit,
            video_state,
            audio_state,
            video_embeds,
            audio_embeds,
            DISTILLED_SIGMAS,
            ancestral=True,
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            on_progress=on_progress,
            progress_step_offset=0,
            progress_total_steps=total_steps,
            on_log=on_log,
            progress_label="stage1",
        )

        # --- Spatial latent upsampler x2 ---
        self._log(on_log, "info", "Upsampling stage-1 latent x2")
        video_patchifier = VideoLatentPatchifier()
        video_latent_5d = video_patchifier.unpatchify(video_x[:1], (f_lat, h_lat, w_lat), ctx)
        video_latent_5d = upsample_video_latent(self.bundle_root, video_latent_5d, load_fn=load_fn)
        _materialize(ctx, video_latent_5d)
        video_tokens_init, _ = video_patchifier.patchify(video_latent_5d, ctx)

        # --- Stage 2 (full resolution, deterministic Euler) ---
        f_lat2, h_lat2, w_lat2 = compute_video_latent_shape(num_frames, height, width)
        video_positions2 = compute_video_positions(ctx, f_lat2, h_lat2, w_lat2, fps)
        audio_positions2 = compute_audio_positions(ctx, audio_tokens)

        conditionings2: list[VideoConditionByLatentIndex] = []
        if image_path:
            video_encoder = load_ltx25_video_encoder(self.bundle_root, load_fn=load_fn)
            conditionings2 = _i2v_conditionings(
                ctx, image_path, enc_h=height, enc_w=width, video_encoder=video_encoder,
            )

        video_state2 = create_noised_state(
            ctx,
            (1, f_lat2 * h_lat2 * w_lat2, 128),
            conditionings=conditionings2,
            spatial_dims=(f_lat2, h_lat2, w_lat2),
            positions=video_positions2,
            seed=seed,
            sigma=STAGE_2_SIGMAS[0],
            initial_latent=video_tokens_init,
        )
        audio_state2 = create_noised_state(
            ctx,
            (1, audio_tokens, _AUDIO_CHANNELS * _AUDIO_MEL_BINS),
            positions=audio_positions2,
            seed=seed + 1,
            sigma=STAGE_2_SIGMAS[0],
            initial_latent=audio_x,
        )
        _materialize(ctx, video_state2.latent, audio_state2.latent)

        self._log(on_log, "info", f"Stage 2: {width}x{height} refine ({stage2_steps} steps)")
        video_x, audio_x = _denoise_loop(
            ctx,
            dit,
            video_state2,
            audio_state2,
            video_embeds,
            audio_embeds,
            STAGE_2_SIGMAS,
            ancestral=False,
            noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            on_progress=on_progress,
            progress_step_offset=stage1_steps,
            progress_total_steps=total_steps,
            on_log=on_log,
            progress_label="stage2",
        )

        # --- Decode + mux ---
        self._log(on_log, "info", "Decoding video + audio latents and muxing mp4")
        video_latent = video_patchifier.unpatchify(video_x[:1], (f_lat2, h_lat2, w_lat2), ctx)
        audio_patchifier = AudioPatchifier()
        audio_latent = audio_patchifier.unpatchify(audio_x[:1])
        _materialize(ctx, video_latent, audio_latent)

        # Release the DiT / text-encoder weights before VAE decode: the conv
        # decoder materializes multi-GB activations per stage at high
        # resolutions, and the 22B transformer (~21GB dequantized) plus Gemma 4
        # (~12GB) no longer need to be resident. Reloaded lazily on next use.
        self._dit = None
        self._encoder = None
        import gc

        gc.collect()
        if hasattr(ctx, "clear_cache"):
            ctx.clear_cache()

        emit_post_progress(on_progress, n_steps=total_steps, within_post=1.0)
        caption_lines = spoken_caption_lines(prompt)

        def _rewrite_captions(frames: list) -> None:
            if not caption_lines:
                return
            painted = rewrite_spoken_captions(frames, caption_lines)
            if painted:
                self._log(
                    on_log,
                    "info",
                    f"Burned on-screen captions with {len(caption_lines)} spoken lines "
                    f"({painted} frames)",
                )

        return mux_video_audio_mp4(
            self.bundle_root,
            video_latent,
            audio_latent,
            output_path,
            frame_rate=fps,
            load_fn=load_fn,
            seed=seed,
            on_frames=_rewrite_captions if caption_lines else None,
        )
