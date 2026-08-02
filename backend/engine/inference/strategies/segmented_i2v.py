"""Segmented I2V long video: stitch pre-generated segment assets (per-shot gen uses standard REST)."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import LongVideoShotSpec, VideoLongGenerationRequest
from backend.long_video.plan import build_shot_plan
from backend.engine.common.codecs.vae.video_stitch import stitch_segment_paths


def _resolve_shots(request: VideoLongGenerationRequest) -> list[LongVideoShotSpec]:
    spec = request.long_video
    if spec.shots:
        shots = [s.model_copy(deep=True) for s in spec.shots]
        for i, shot in enumerate(shots):
            if not shot.id:
                shot.id = f"shot_{i:02d}"
            shot.order = i
        return shots

    plan = build_shot_plan(
        target_duration_sec=spec.target_duration_sec,
        segment_duration_sec=spec.segment_duration_sec,
    )
    motion_prompts = list(spec.segment_prompts or [])
    brief = (request.prompt or "").strip()
    shots: list[LongVideoShotSpec] = []
    for i in range(plan.shot_count):
        visual = motion_prompts[i] if i < len(motion_prompts) else brief
        motion = motion_prompts[i] if i < len(motion_prompts) else brief
        if i == 0 and spec.opening_prompt:
            visual = spec.opening_prompt
        shots.append(
            LongVideoShotSpec(
                id=f"shot_{i:02d}",
                order=i,
                visual_prompt=visual,
                motion_prompt=motion,
                duration_sec=plan.segment_durations_sec[i],
                status="draft",
            )
        )
    return shots


def run_segmented_i2v_strategy(
    *,
    request: VideoLongGenerationRequest,
    ctx_exec: Any,
    on_progress: Callable | None,
    on_log: Callable | None,
    span_factory: Callable[[str], Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Stitch existing segment videos. Keyframes/segments are created via /api/images/* and /api/videos/edits."""
    spec = request.long_video
    phase = (request.metadata or {}).get("long_video_phase") or "assemble_only"
    if phase != "assemble_only":
        raise RuntimeError(
            "segmented_i2v batch keyframe/segment generation was removed; "
            "use POST /api/images/generations|edits and POST /api/videos/edits per shot, "
            "then POST /api/videos/long-generations with metadata.long_video_phase=assemble_only"
        )

    fps = max(1, int(request.fps or 16))
    work = Path(ctx_exec.work_dir)
    shots = _resolve_shots(request)

    def _span(name: str):
        if span_factory is None:
            return nullcontext()
        return span_factory(name)

    def _log(level: str, msg: str) -> None:
        if on_log:
            on_log(level, msg)

    def _check_cancel() -> None:
        if ctx_exec.cancel_token.is_cancelled():
            raise RuntimeError("long video cancelled")

    segment_paths: list[Path] = []
    for idx, shot in enumerate(shots):
        if not shot.segment_asset_id:
            raise RuntimeError(f"long_video assemble: shot {idx} missing segment_asset_id")
        segment_paths.append(ctx_exec.asset_store.get_file_path(shot.segment_asset_id))

    with _span("stitch"):
        _check_cancel()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = work / f"long_video_{timestamp}.mp4"
        _log("info", f"long_video: stitching {len(segment_paths)} segments")
        stitch_segment_paths(
            segment_paths,
            output=final_path,
            overlap_frames=int(spec.overlap_frames),
            fps=float(fps),
        )
        if on_progress:
            on_progress(phase="assembly", progress=1.0, n_steps=1)

    metadata = {
        "strategy": "segmented_i2v",
        "long_video": spec.model_dump(),
        "shots": [s.model_dump() for s in shots],
        "phase": "assemble_only",
    }
    return str(final_path), metadata
