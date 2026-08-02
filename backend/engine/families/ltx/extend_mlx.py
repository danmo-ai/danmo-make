"""LTX 2.3 latent-space extend pass (single window reference + extend)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from backend.engine.families.ltx.generation_mlx import LTX23MlxGenerator, _resolve_distilled_stage1_steps
from backend.engine.families.ltx.ingest import append_video_with_crossfade, extract_video_tail, trim_video_leading
from backend.engine.families.ltx.long_video_plan import num_frames_for_duration_sec
from backend.engine.families.ltx.pipeline_math_mlx import (
    LatentState,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    apply_conditioning,
    build_ltx_av_extend_masks,
    compute_audio_positions,
    compute_audio_token_count,
    compute_video_latent_shape,
    compute_video_positions,
    merge_extend_latent_states,
    reference_latent_frame_count,
    VIDEO_SPATIAL_SCALE,
)
from backend.engine.families.ltx.vae import decode_ltx23_av_to_mp4
from backend.engine.families.ltx.vae_mlx import (
    encode_waveform_to_audio_latent,
    load_ltx23_video_encoder,
)
from backend.engine.pipelines.pipeline_progress import emit_post_progress
from backend.engine.runtime.mlx_runtime import run_eval


def _materialize(ctx, *arrays: mx.array) -> None:
    run_eval(getattr(ctx, "eval", None), *arrays)


def validate_extend_window_frames(
    *,
    reference_sec: float,
    extend_sec: float,
    fps: float,
    max_frames: int = 257,
) -> int:
    """Return pass window frame count or raise if over ``max_frames``."""
    total_sec = float(reference_sec) + float(extend_sec)
    num_frames = num_frames_for_duration_sec(total_sec, fps)
    if num_frames > int(max_frames):
        raise RuntimeError(
            f"LTX extend window {num_frames} frames exceeds max {max_frames} "
            f"({reference_sec}s reference + {extend_sec}s extend @ {fps} fps). "
            "Reduce segment_extend_sec, reference_duration_sec, or fps."
        )
    return num_frames


def _build_extend_stage1_states(
    generator: LTX23MlxGenerator,
    *,
    source_mp4: Path,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    reference_sec: float,
    seed: int,
) -> tuple[tuple[LatentState, LatentState], tuple[int, int, int], int, int]:
    ctx = generator.ctx
    load_fn = getattr(ctx, "load_weights", None)
    half_h, half_w = height // 2, width // 2
    enc_h, enc_w = half_h * VIDEO_SPATIAL_SCALE, half_w * VIDEO_SPATIAL_SCALE

    ref_pixel_frames = num_frames_for_duration_sec(reference_sec, fps)
    rgb, audio_np = extract_video_tail(
        source_mp4,
        duration_sec=reference_sec,
        fps=fps,
        width=width,
        height=height,
    )
    if rgb.shape[0] < ref_pixel_frames:
        raise RuntimeError(
            f"LTX extend: reference tail has {rgb.shape[0]} frames, need {ref_pixel_frames}"
        )
    rgb = rgb[:ref_pixel_frames]

    if generator._video_encoder is None:
        generator._video_encoder = load_ltx23_video_encoder(generator.bundle_root, load_fn=load_fn)
    rgb_chw = rgb.transpose(3, 0, 1, 2).astype(np.float32)
    pixels = ctx.array(rgb_chw[None, ...])
    ref_latent = generator._video_encoder.encode(pixels)

    f_lat, h_half, w_half = compute_video_latent_shape(num_frames, half_h, half_w)
    ref_lat_f = min(int(ref_latent.shape[2]), reference_latent_frame_count(ref_pixel_frames))

    full_latent = ctx.zeros((1, int(ref_latent.shape[1]), f_lat, h_half, w_half), dtype=ref_latent.dtype)
    full_latent[:, :, :ref_lat_f, :, :] = ref_latent[:, :, :ref_lat_f, :, :]

    patchifier = VideoLatentPatchifier()
    video_tokens, spatial = patchifier.patchify(full_latent, ctx)
    tokens_per_frame = h_half * w_half

    audio_t = compute_audio_token_count(num_frames, frame_rate=fps)
    ref_audio_t = compute_audio_token_count(ref_pixel_frames, frame_rate=fps)
    wav = ctx.array(audio_np.astype(np.float32)[None, :])
    ref_audio_latent = encode_waveform_to_audio_latent(ctx, wav, generator.bundle_root)
    ref_a_tok = ref_audio_latent.reshape(1, ref_audio_latent.shape[2], 128)
    full_audio = ctx.zeros((1, audio_t, 128), dtype=ref_a_tok.dtype)
    copy_a = min(int(ref_a_tok.shape[1]), ref_audio_t, audio_t)
    full_audio[:, :copy_a, :] = ref_a_tok[:, :copy_a, :]

    video_mask, audio_mask = build_ltx_av_extend_masks(
        ctx,
        num_video_latent_frames=f_lat,
        tokens_per_frame=tokens_per_frame,
        num_audio_tokens=audio_t,
        reference_latent_frames=ref_lat_f,
        reference_audio_tokens=ref_audio_t,
    )
    video_positions = compute_video_positions(ctx, f_lat, h_half, w_half, frame_rate=fps)
    audio_positions = compute_audio_positions(ctx, audio_t)

    v_state, a_state = merge_extend_latent_states(
        ctx,
        video_tokens=video_tokens,
        audio_tokens=full_audio,
        video_clean=video_tokens,
        audio_clean=full_audio,
        video_mask=video_mask,
        audio_mask=audio_mask,
        video_positions=video_positions,
        audio_positions=audio_positions,
        seed=seed,
        sigma=1.0,
        spatial_dims=spatial,
    )
    cond = [VideoConditionByLatentIndex(latent=ref_latent, frame_idx=0, strength=1.0)]
    v_state = apply_conditioning(ctx, v_state, cond, spatial)
    return (v_state, a_state), spatial, ref_lat_f, ref_audio_t


def extend_pass(
    generator: LTX23MlxGenerator,
    *,
    source_mp4: str | Path,
    output_segment_path: str,
    prompt: str,
    width: int,
    height: int,
    reference_sec: float,
    extend_sec: float,
    fps: float,
    seed: int,
    steps: int,
    guidance: float,
    step_distill: bool,
    overlap_blend_frames: int = 4,
    max_frames: int = 257,
    on_log: Callable[[str, str], None] | None = None,
    on_progress: Callable[..., None] | None = None,
) -> str:
    """Run one extend window; write extend-only segment MP4 (after trim)."""
    num_frames = validate_extend_window_frames(
        reference_sec=reference_sec,
        extend_sec=extend_sec,
        fps=fps,
        max_frames=max_frames,
    )
    stage1_states, _spatial, _ref_f, _ref_a = _build_extend_stage1_states(
        generator,
        source_mp4=Path(source_mp4),
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        reference_sec=reference_sec,
        seed=seed,
    )

    stage2_steps = int(getattr(generator.config, "ltx_stage2_steps", 3) or 3)
    stage1_steps = (
        _resolve_distilled_stage1_steps(steps, on_log=on_log)
        if step_distill
        else max(1, int(steps))
    )

    video_latent, audio_latent = generator._generate_two_stage(
        prompt=prompt,
        width=int(width),
        height=int(height),
        num_frames=int(num_frames),
        fps=float(fps),
        seed=int(seed),
        stage1_steps=stage1_steps,
        stage2_steps=stage2_steps,
        guidance=float(guidance),
        step_distill=bool(step_distill),
        image_path=None,
        on_log=on_log,
        on_progress=on_progress,
        stage1_states=stage1_states,
    )
    _materialize(generator.ctx, video_latent, audio_latent)

    work = Path(output_segment_path).parent
    work.mkdir(parents=True, exist_ok=True)
    full_pass_mp4 = str(work / "_extend_full_pass.mp4")

    def _decode_log(msg: str) -> None:
        if on_log:
            on_log("info", msg)

    decode_ltx23_av_to_mp4(
        generator.ctx,
        video_latent,
        audio_latent,
        full_pass_mp4,
        generator.bundle_root,
        frame_rate=float(fps),
        on_log=_decode_log,
    )
    trim_video_leading(
        Path(full_pass_mp4),
        skip_sec=reference_sec,
        output_path=Path(output_segment_path),
        fps=fps,
    )
    Path(full_pass_mp4).unlink(missing_ok=True)
    emit_post_progress(on_progress, n_steps=max(1, stage1_steps + stage2_steps), within_post=1.0)
    return output_segment_path


def extend_and_append(
    generator: LTX23MlxGenerator,
    *,
    accumulator_mp4: str | Path,
    work_dir: Path,
    prompt: str,
    width: int,
    height: int,
    reference_sec: float,
    extend_sec: float,
    fps: float,
    seed: int,
    steps: int,
    guidance: float,
    step_distill: bool,
    overlap_blend_frames: int = 4,
    max_frames: int = 257,
    on_log: Callable[[str, str], None] | None = None,
    on_progress: Callable[..., None] | None = None,
) -> Path:
    """Extend pass + crossfade append into accumulator path."""
    acc = Path(accumulator_mp4)
    seg = work_dir / f"_extend_seg_{seed}.mp4"
    merged = work_dir / f"_extend_merged_{seed}.mp4"
    extend_pass(
        generator,
        source_mp4=acc,
        output_segment_path=str(seg),
        prompt=prompt,
        width=width,
        height=height,
        reference_sec=reference_sec,
        extend_sec=extend_sec,
        fps=fps,
        seed=seed,
        steps=steps,
        guidance=guidance,
        step_distill=step_distill,
        overlap_blend_frames=overlap_blend_frames,
        max_frames=max_frames,
        on_log=on_log,
        on_progress=on_progress,
    )
    append_video_with_crossfade(
        accumulator=acc,
        new_segment=seg,
        output=merged,
        overlap_frames=overlap_blend_frames,
        fps=fps,
    )
    seg.unlink(missing_ok=True)
    merged.replace(acc)
    return acc
