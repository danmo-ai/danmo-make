"""Resolve segmented I2V source image from per-shot chain mode."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from backend.core.contracts import LongVideoShotSpec, VideoEditRequest
from backend.engine.common.codecs.vae.video_stitch import extract_last_frame_image
from backend.engine.group_utils import resolve_asset_group_id, resolve_group_id_from_asset

LongVideoChainMode = Literal["keyframe_only", "last_frame", "reference_r2v"]


def effective_shot_chain_mode(
    shot: LongVideoShotSpec,
    default: LongVideoChainMode = "keyframe_only",
) -> LongVideoChainMode:
    mode = getattr(shot, "chain_mode", None)
    if mode == "first_last":
        return "keyframe_only"
    if mode in ("keyframe_only", "last_frame", "reference_r2v"):
        return mode
    return default


def resolve_segment_i2v_source_asset_id(
    *,
    shot: LongVideoShotSpec,
    shot_index: int,
    default_chain_mode: LongVideoChainMode,
    prev_segment_asset_id: str | None,
    asset_store: Any,
    work_dir: Path,
    task_id: str,
) -> str:
    """Return asset id for I2V ``source_asset_id`` (keyframe or extracted last frame)."""
    source_id = (shot.keyframe_asset_id or "").strip()
    mode = effective_shot_chain_mode(shot, default_chain_mode)
    if mode != "last_frame" or shot_index <= 0 or not prev_segment_asset_id:
        if not source_id:
            raise RuntimeError(f"long_video shot {shot_index}: missing keyframe_asset_id")
        return source_id

    seg_video = asset_store.get_file_path(prev_segment_asset_id)
    if seg_video is None or not Path(seg_video).is_file():
        raise RuntimeError(
            f"long_video shot {shot_index}: last_frame chain requires previous segment video "
            f"({prev_segment_asset_id!r})"
        )

    shot_work = Path(work_dir) / "chain" / f"{shot_index:02d}"
    shot_work.mkdir(parents=True, exist_ok=True)
    last_frame = shot_work / "last_frame.png"
    extract_last_frame_image(seg_video, output_path=last_frame)
    group_id = resolve_group_id_from_asset(asset_store, prev_segment_asset_id)
    return asset_store.create_from_file(
        last_frame,
        kind="image",
        mime_type="image/png",
        source_task_id=task_id,
        metadata={"long_video_shot_id": shot.id, "chain": "last_frame"},
        source_action="create",
        parent_asset_id=prev_segment_asset_id,
        relation_type="frame_extract",
        group_id=group_id,
    )


def resolve_long_video_edit_chain_source(
    request: VideoEditRequest,
    *,
    asset_store: Any,
    work_dir: Path,
    task_id: str,
) -> VideoEditRequest:
    """Apply chain modes from request metadata before a standalone segment edit."""
    meta = request.metadata or {}
    mode = meta.get("long_video_chain_mode")
    prev_id = (meta.get("long_video_prev_segment_asset_id") or "").strip()

    if mode == "first_last":
        raise RuntimeError(
            "long_video segment edit: first_last chain is deprecated; use keyframe_only or last_frame"
        )

    if mode != "last_frame" or not prev_id:
        return request

    seg_video = asset_store.get_file_path(prev_id)
    if seg_video is None or not Path(seg_video).is_file():
        raise RuntimeError(
            f"long_video segment edit: last_frame chain requires previous segment video ({prev_id!r})"
        )

    work = Path(work_dir) / "chain" / "edit"
    work.mkdir(parents=True, exist_ok=True)
    last_frame = work / "last_frame.png"
    extract_last_frame_image(seg_video, output_path=last_frame)
    shot_id = meta.get("long_video_shot_id") or ""
    group_id = resolve_asset_group_id(meta, asset_store) or resolve_group_id_from_asset(
        asset_store, prev_id
    )
    source_id = asset_store.create_from_file(
        last_frame,
        kind="image",
        mime_type="image/png",
        source_task_id=task_id,
        metadata={"long_video_shot_id": shot_id, "chain": "last_frame"},
        source_action="create",
        parent_asset_id=prev_id,
        relation_type="frame_extract",
        group_id=group_id,
    )
    return request.model_copy(update={"source_asset_id": source_id})
