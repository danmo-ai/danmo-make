"""Rule-based shot contract repairs (one pass before validation retry)."""
from __future__ import annotations

import re
from typing import Any

from backend.long_video.beat_budget import split_duration_parts
from backend.long_video.constants import MAX_CLIP_SEC, MIN_CLIP_SEC
from backend.long_video.shot_contract_validator import _protagonist_names
from backend.long_video.visibility import clamp_vis_progression, vis_label, vis_rank


def repair_shot_contracts(
    shots: list[dict[str, Any]],
    *,
    character_anchor: str = "",
    max_clip_sec: float = MAX_CLIP_SEC,
    min_clip_sec: float = MIN_CLIP_SEC,
    beat_budgets: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply deterministic fixes for common validator failures."""
    del beat_budgets  # duration guides are soft; never trim shots for total runtime
    if not shots:
        return []
    cap = max(min_clip_sec, float(max_clip_sec))
    lo = max(0.5, float(min_clip_sec))
    out: list[dict[str, Any]] = []
    primary = (_protagonist_names(character_anchor) or [""])[0]

    for i, shot in enumerate(shots):
        row = dict(shot)
        if row.get("duration_sec") is not None:
            row["duration_sec"] = round(max(lo, min(cap, float(row["duration_sec"]))), 1)
        if row.get("flf_mode") == "first_last":
            row["flf_mode"] = "none"
            row["chain_mode"] = "keyframe_only"
        row.pop("end_visual_prompt", None)
        if row.get("start_frame_mode") == "anchor_link":
            row.setdefault(
                "first_frame_requirement",
                "Inherits face-anchor composition; no separate keyframe required.",
            )
        out.append(row)

    if primary:
        first = out[0]
        on_screen = list(first.get("characters_on_screen") or [])
        if primary not in on_screen:
            on_screen.append(primary)
            first["characters_on_screen"] = on_screen
        vis = str(first.get("first_frame_visibility") or "invisible")
        if first.get("start_frame_mode", "keyframe") == "keyframe" and vis_rank(vis) < vis_rank("silhouette"):
            first["first_frame_visibility"] = "silhouette"
            req = str(first.get("first_frame_requirement") or "").strip()
            if primary not in req:
                first["first_frame_requirement"] = f"{primary} visible as silhouette at frame 0. {req}".strip()
            start = str(first.get("start_visual_prompt") or "")
            if primary not in start and not re.search(r"silhouette|侧影|轮廓|剪影", start, re.I):
                first["start_visual_prompt"] = f"{start}，{primary}以侧影出现在画面中".strip("，")

    for i in range(len(out)):
        cur = out[i]
        if cur.get("start_frame_mode") == "prev_segment_tail" and i > 0:
            prev = out[i - 1]
            end_vis = str(prev.get("end_visibility") or prev.get("first_frame_visibility") or "invisible")
            cur["first_frame_visibility"] = vis_label(end_vis)

    for i in range(1, len(out)):
        prev = out[i - 1]
        cur = out[i]
        if cur.get("start_frame_mode") == "prev_segment_tail":
            continue
        prev_chars = set(prev.get("characters_on_screen") or [])
        cur_chars = set(cur.get("characters_on_screen") or [])
        for _name in prev_chars & cur_chars:
            p_end = str(prev.get("end_visibility") or prev.get("first_frame_visibility") or "invisible")
            c_start = str(cur.get("first_frame_visibility") or "invisible")
            if vis_rank(c_start) > vis_rank(p_end) + 1:
                cur["first_frame_visibility"] = clamp_vis_progression(p_end, c_start)

    return out


def _clone_subsegment_plan(sub: Any, **overrides: Any) -> Any:
    from backend.long_video.segment_plan_types import SubsegmentPlan

    data = {
        "role": sub.role,
        "duration_sec": sub.duration_sec,
        "shot_size": sub.shot_size,
        "flf_mode": sub.flf_mode,
        "start_visibility": sub.start_visibility,
        "end_visibility": sub.end_visibility,
        "characters_on_screen": sub.characters_on_screen,
        "first_frame_requirement": sub.first_frame_requirement,
    }
    data.update(overrides)
    return SubsegmentPlan(**data)


def normalize_subsegment_plans(
    subsegs: list[Any],
    *,
    max_clip_sec: float = MAX_CLIP_SEC,
    min_sec: float = MIN_CLIP_SEC,
) -> list[Any]:
    """Clamp/split per subsegment for I2V clip cap; preserve story structure (no budget trim)."""
    if not subsegs:
        return subsegs
    cap = max(min_sec, float(max_clip_sec))
    lo = max(0.5, float(min_sec))
    out: list[Any] = []
    for sub in subsegs:
        dur = max(lo, float(sub.duration_sec))
        if dur <= cap + 0.01:
            out.append(_clone_subsegment_plan(sub, duration_sec=round(dur, 1)))
            continue
        for part_i, part_dur in enumerate(split_duration_parts(dur, max_clip_sec=cap)):
            if part_i == 0:
                out.append(_clone_subsegment_plan(sub, duration_sec=round(part_dur, 1)))
            else:
                out.append(
                    _clone_subsegment_plan(
                        sub,
                        role="tail_continuation",
                        duration_sec=round(part_dur, 1),
                        flf_mode="continuation",
                    )
                )
    return out


def merge_over_budget_beats(
    subsegs: list[Any],
    beat_dur: float,
    *,
    min_sec: float = MIN_CLIP_SEC,
    max_clip_sec: float = MAX_CLIP_SEC,
) -> list[Any]:
    """Legacy name — only enforces per-clip cap; ``beat_dur`` is ignored (soft guide)."""
    del beat_dur
    return normalize_subsegment_plans(subsegs, max_clip_sec=max_clip_sec, min_sec=min_sec)
