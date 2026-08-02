"""Rule-based validation for parsed storyboard shot contracts."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from backend.long_video.beat_budget import validate_beat_duration_sums
from backend.long_video.constants import MAX_CLIP_SEC, MIN_CLIP_SEC
from backend.long_video.visibility import vis_rank

CharacterVisibility = Literal["invisible", "silhouette", "partial", "full_face"]

_VISIBILITY_RANK: dict[str, int] = {
    "invisible": 0,
    "silhouette": 1,
    "partial": 2,
    "full_face": 3,
}

_VISIBILITY_PROGRESSION = (
    ("invisible", "silhouette"),
    ("silhouette", "partial"),
    ("partial", "full_face"),
    ("invisible", "partial"),  # rare: wide partial entry
    ("silhouette", "full_face"),  # allowed only with same-segment motion, not cross-segment
)


@dataclass
class ShotContractIssue:
    code: str
    message: str
    shot_index: int | None = None
    severity: Literal["critical", "warning"] = "critical"


@dataclass
class ShotContractValidationResult:
    ok: bool
    issues: list[ShotContractIssue] = field(default_factory=list)
    warnings: list[ShotContractIssue] = field(default_factory=list)


def _vis_rank(value: str | None) -> int:
    return vis_rank(value)


def _protagonist_names(character_anchor: str) -> list[str]:
    from backend.long_video.parse_quality import protagonist_names_from_anchor

    names = protagonist_names_from_anchor(character_anchor)
    if names:
        return names
    legacy: list[str] = []
    for line in (character_anchor or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^[-*]\s*([^（(：:]+)", line)
        if m:
            legacy.append(m.group(1).strip())
        elif "：" in line[:40] or ":" in line[:40]:
            head = re.split(r"[：:]", line, 1)[0].strip()
            if head and len(head) <= 24:
                legacy.append(head)
    return legacy


def _name_in_text(name: str, text: str) -> bool:
    if not name or not text:
        return False
    return name in text


def validate_shot_contracts(
    shots: list[dict[str, Any]],
    *,
    character_anchor: str = "",
    max_clip_sec: float = MAX_CLIP_SEC,
    min_clip_sec: float = MIN_CLIP_SEC,
    target_duration_sec: float | None = None,
    beat_budgets: dict[int, float] | None = None,
) -> ShotContractValidationResult:
    issues: list[ShotContractIssue] = []
    warnings: list[ShotContractIssue] = []
    cap = max(min_clip_sec, float(max_clip_sec))
    lo = max(0.5, float(min_clip_sec))

    if not shots:
        issues.append(ShotContractIssue("no_shots", "segment pipeline produced no shots"))
        return ShotContractValidationResult(ok=False, issues=issues, warnings=warnings)

    protagonists = _protagonist_names(character_anchor)
    primary = protagonists[0] if protagonists else ""

    for i, shot in enumerate(shots):
        dur = shot.get("duration_sec")
        if dur is None:
            issues.append(
                ShotContractIssue(
                    "missing_duration",
                    f"shot {i}: duration_sec is required",
                    shot_index=i,
                )
            )
            continue
        d = float(dur)
        if d < lo or d > cap + 0.01:
            issues.append(
                ShotContractIssue(
                    "duration_out_of_range",
                    f"shot {i}: duration_sec {d} not in [{lo}, {cap}]",
                    shot_index=i,
                )
            )
        if shot.get("flf_mode") == "first_last":
            issues.append(
                ShotContractIssue(
                    "flf_deprecated",
                    f"shot {i}: flf_mode=first_last is no longer supported",
                    shot_index=i,
                )
            )
        if (shot.get("end_visual_prompt") or "").strip() and shot.get("segment_role") == "pre_anchor":
            warnings.append(
                ShotContractIssue(
                    "end_visual_deprecated",
                    f"shot {i}: end_visual_prompt is deprecated",
                    shot_index=i,
                    severity="warning",
                )
            )

    # Opening shot: protagonist visibility
    if shots and primary:
        first = shots[0]
        on_screen = first.get("characters_on_screen") or []
        vis = str(first.get("first_frame_visibility") or "invisible")
        start_mode = first.get("start_frame_mode") or "keyframe"
        needs_protagonist = primary in on_screen or _name_in_text(
            primary,
            (first.get("start_visual_prompt") or "") + (first.get("video_prompt") or ""),
        )
        if needs_protagonist and start_mode == "keyframe" and _vis_rank(vis) < _vis_rank("silhouette"):
            issues.append(
                ShotContractIssue(
                    "opening_no_protagonist",
                    f"shot 0: first frame must show protagonist ({primary}) at least as silhouette",
                    shot_index=0,
                )
            )

    # Visibility progression across adjacent segments (same character)
    for i in range(1, len(shots)):
        prev = shots[i - 1]
        cur = shots[i]
        if cur.get("start_frame_mode") == "prev_segment_tail":
            continue
        prev_chars = set(prev.get("characters_on_screen") or [])
        cur_chars = set(cur.get("characters_on_screen") or [])
        for name in prev_chars & cur_chars:
            p_end = str(prev.get("end_visibility") or prev.get("first_frame_visibility") or "invisible")
            c_start = str(cur.get("first_frame_visibility") or "invisible")
            pr, cr = _vis_rank(p_end), _vis_rank(c_start)
            if cr > pr + 1:
                issues.append(
                    ShotContractIssue(
                        "visibility_jump",
                        f"shot {i}: {name} visibility jump {p_end} -> {c_start}",
                        shot_index=i,
                    )
                )

    if beat_budgets:
        for msg in validate_beat_duration_sums(shots, beat_budgets):
            warnings.append(
                ShotContractIssue("beat_duration_drift", msg, severity="warning"),
            )

    # Opening hook (soft check): motion should add action beyond static scene line
    if shots:
        from backend.long_video.prompt_overlap import prompt_token_set

        first = shots[0]
        scene = str(first.get("scene_prompt") or first.get("start_visual_prompt") or "")
        motion = str(first.get("video_prompt") or first.get("motion_prompt") or "")
        motion_tokens = prompt_token_set(motion)
        scene_tokens = prompt_token_set(scene)
        if motion_tokens and scene_tokens:
            novel = motion_tokens - scene_tokens
            if len(novel) < max(2, int(len(motion_tokens) * 0.15)):
                warnings.append(
                    ShotContractIssue(
                        "opening_hook_weak",
                        "shot 0: motion prompt adds little beyond scene/first-frame text",
                        shot_index=0,
                        severity="warning",
                    )
                )

    if target_duration_sec is not None and target_duration_sec > 0:
        total = sum(float(s.get("duration_sec") or 0) for s in shots)
        target = float(target_duration_sec)
        if abs(total - target) / target > 0.25:
            warnings.append(
                ShotContractIssue(
                    "duration_budget_drift",
                    f"sum(duration_sec)={total:.1f}s drifts from soft target {target:.1f}s "
                    f"(story/flow priority — informational only)",
                    severity="warning",
                )
            )

    return ShotContractValidationResult(
        ok=not issues,
        issues=issues,
        warnings=warnings,
    )


def clamp_shot_durations(
    shots: list[dict[str, Any]],
    *,
    max_clip_sec: float = MAX_CLIP_SEC,
    min_clip_sec: float = MIN_CLIP_SEC,
) -> list[dict[str, Any]]:
    """Clamp duration_sec in-place copy."""
    cap = max(min_clip_sec, float(max_clip_sec))
    lo = max(0.5, float(min_clip_sec))
    out: list[dict[str, Any]] = []
    for shot in shots:
        row = dict(shot)
        if row.get("duration_sec") is not None:
            row["duration_sec"] = round(max(lo, min(cap, float(row["duration_sec"]))), 1)
        out.append(row)
    return out
