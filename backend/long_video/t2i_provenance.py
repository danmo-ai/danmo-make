"""Lightweight T2I assembly provenance (parse activity / agent diagnostics).

Uses token + bigram overlap only — no story-specific synonym tables.
Aligned with frontend ``longVideoProject.ts`` merge policy.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from backend.long_video.keyframe_prompt_policy import should_merge_scene_prompt_into_t2i
from backend.long_video.prompt_overlap import prompt_token_coverage

NARRATIVE_MERGE_COVERAGE_THRESHOLD = 0.38
_TEXT_ALREADY_COVERED_THRESHOLD = 0.72
_SCENE_NARRATIVE_MERGE_MAX_CHARS = 120
_REQUIREMENT_CLAUSE_SPLIT = re.compile(r"[；;。\n]+")
_CLOSE_UP_SHOT_SIZE_RE = re.compile(
    r"^(?:特写|大特写|近景|close[\s-]?up|cu\b|medium[\s-]?close|extreme[\s-]?close)",
    re.I,
)

NarrativeSkipReason = Literal[
    "face_anchor",
    "close_up",
    "token_coverage_sufficient",
    "narrative_already_covered",
    "empty_narrative",
]
LocationMerge = Literal["none", "prepended", "scene_only"]


def _shot_field(shot: Any, key: str, default: str = "") -> str:
    if isinstance(shot, dict):
        return str(shot.get(key) or default).strip()
    return str(getattr(shot, key, None) or default).strip()


def shot_keyframe_text(shot: Any) -> str:
    role = _shot_field(shot, "segment_role")
    if role == "face_anchor":
        return (
            _shot_field(shot, "anchor_visual_prompt")
            or _shot_field(shot, "start_visual_prompt")
            or _shot_field(shot, "visual_prompt")
        )
    return _shot_field(shot, "start_visual_prompt") or _shot_field(shot, "visual_prompt")


def _normalize_location_key(text: str) -> str:
    return re.sub(r"[\s·/\\|｜\-—–]", "", (text or "").strip().lower())


def _cjk_bigram_set(text: str) -> set[str]:
    t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")
    out: set[str] = set()
    if len(t) == 1:
        out.add(t)
        return out
    for i in range(len(t) - 1):
        out.add(t[i : i + 2])
    return out


def cjk_bigram_overlap_score(a: str, b: str) -> float:
    ba = _cjk_bigram_set(a)
    bb = _cjk_bigram_set(b)
    if not ba or not bb:
        return 0.0
    hit = sum(1 for g in ba if g in bb)
    union = ba.__len__() + bb.__len__() - hit
    return hit / union if union else 0.0


def text_already_covered(haystack: str, needle: str) -> bool:
    h = (haystack or "").strip()
    n = (needle or "").strip()
    if not h or not n:
        return False
    if n in h or h in n:
        return True
    return prompt_token_coverage(h, n) >= _TEXT_ALREADY_COVERED_THRESHOLD


def locations_similar(a: str, b: str) -> bool:
    ka = _normalize_location_key(a)
    kb = _normalize_location_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if len(short) >= 4 and short in long:
        return True
    if prompt_token_coverage(a, b) >= 0.45 or prompt_token_coverage(b, a) >= 0.45:
        return True
    return cjk_bigram_overlap_score(a, b) >= 0.34


def is_close_up_shot_size(shot_size: str) -> bool:
    s = (shot_size or "").strip()
    return bool(s and _CLOSE_UP_SHOT_SIZE_RE.match(s))


def should_skip_beat_narrative_merge(
    *,
    segment_role: str,
    shot_size: str = "",
    start_visual: str = "",
    visibility: str = "",
    is_intentional_empty: bool = False,
) -> bool:
    del shot_size  # legacy callers only; visibility contract replaces shot_size heuristics
    if (start_visual or "").strip():
        return True
    if is_intentional_empty or segment_role == "face_anchor":
        return True
    vis = visibility or "full_face"
    return vis in ("full_face", "partial", "silhouette")


def location_referenced_in_text(loc: str, text: str) -> bool:
    if not loc.strip() or not text.strip():
        return False
    if locations_similar(loc, text) or text_already_covered(text, loc):
        return True
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    loc_cjk = re.sub(r"[^\u4e00-\u9fff]", "", loc)
    if not cjk or not loc_cjk:
        return prompt_token_coverage(text, loc) >= 0.45
    max_len = min(max(len(loc_cjk) + 2, 4), 12)
    for start in range(len(cjk)):
        for length in range(2, max_len + 1):
            if start + length > len(cjk):
                break
            if locations_similar(loc_cjk, cjk[start : start + length]):
                return True
    return prompt_token_coverage(text, loc) >= 0.5


def merge_beat_narrative_fields(
    *,
    location: str,
    scene_prompt: str,
    visual_hint: str = "",
) -> tuple[str, LocationMerge]:
    scene = scene_prompt.strip()
    loc = location.strip()
    if not loc:
        return scene, "none"
    if not scene:
        return loc, "prepended"
    visual = visual_hint.strip()
    redundant = (
        location_referenced_in_text(loc, scene)
        or (bool(visual) and location_referenced_in_text(loc, visual))
        or prompt_token_coverage(scene, loc) >= 0.5
    )
    if redundant:
        return scene, "scene_only"
    return f"{loc}，{scene}", "prepended"


def merge_uncovered_requirement_clauses(
    visual_scene: str,
    requirement: str,
    merged_parts: list[str] | None = None,
) -> tuple[str, int, int]:
    req = requirement.strip()
    if not req:
        return "", 0, 0
    parts = [visual_scene] if visual_scene else []
    if merged_parts:
        parts.extend(merged_parts)
    haystack = "；".join(parts)
    clauses = [c.strip() for c in _REQUIREMENT_CLAUSE_SPLIT.split(req) if c.strip()]
    uncovered = [c for c in clauses if not text_already_covered(haystack, c)]
    return "；".join(uncovered), len(clauses), len(uncovered)


def build_shot_t2i_provenance_summary(shot: Any) -> dict[str, Any]:
    """Agent-facing lightweight T2I assembly flags for one parsed shot."""
    visual = shot_keyframe_text(shot)
    beat_location = _shot_field(shot, "location")
    beat_scene = _shot_field(shot, "scene_prompt")
    segment_role = _shot_field(shot, "segment_role")
    shot_size = _shot_field(shot, "shot_size")
    visibility = _shot_field(shot, "first_frame_visibility") or "full_face"
    is_empty = bool(
        shot.get("is_establishing_empty") if isinstance(shot, dict) else getattr(shot, "is_establishing_empty", False)
    )
    requirement = _shot_field(shot, "first_frame_requirement")

    scene_narrative, location_merge = merge_beat_narrative_fields(
        location=beat_location,
        scene_prompt=beat_scene,
        visual_hint=visual,
    )

    narrative_merged = False
    narrative_skip_reason: NarrativeSkipReason | None = None
    narrative_token_coverage: float | None = None

    if not scene_narrative:
        narrative_skip_reason = "empty_narrative"
    elif should_skip_beat_narrative_merge(
        segment_role=segment_role,
        shot_size=shot_size,
        start_visual=visual,
        visibility=visibility,
        is_intentional_empty=is_empty,
    ):
        narrative_skip_reason = "face_anchor" if segment_role == "face_anchor" else "close_up"
    elif not should_merge_scene_prompt_into_t2i(start_visual=visual, scene_prompt=beat_scene):
        narrative_skip_reason = "token_coverage_sufficient"
    elif text_already_covered(visual, scene_narrative):
        narrative_skip_reason = "narrative_already_covered"
    else:
        narrative_token_coverage = round(prompt_token_coverage(visual, scene_narrative), 3)
        if narrative_token_coverage >= NARRATIVE_MERGE_COVERAGE_THRESHOLD:
            narrative_skip_reason = "token_coverage_sufficient"
        else:
            narrative_merged = True

    ffr_skip_reason = "empty_ffr" if not requirement.strip() else "inspector_only"

    return {
        "id": _shot_field(shot, "id"),
        "order": getattr(shot, "order", None) if not isinstance(shot, dict) else shot.get("order", 0),
        "segment_role": segment_role,
        "location": beat_location,
        "shot_size": shot_size,
        "narrative_merged": narrative_merged,
        "narrative_skip_reason": narrative_skip_reason,
        "narrative_token_coverage": narrative_token_coverage,
        "location_merge": location_merge,
        "first_frame_requirement_merged": False,
        "ffr_skip_reason": ffr_skip_reason,
        "composed_scene_preview": _preview_text(
            _compose_scene_preview(
                visual=visual,
                scene_narrative=scene_narrative,
                segment_role=segment_role,
                visibility=visibility,
                is_intentional_empty=is_empty,
            )
        ),
    }


def _preview_text(text: str, max_len: int = 120) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return f"{t[:max_len].strip()}…"


def _compose_scene_preview(
    *,
    visual: str,
    scene_narrative: str,
    segment_role: str,
    shot_size: str = "",
    visibility: str = "",
    is_intentional_empty: bool = False,
) -> str:
    del shot_size
    if visual.strip():
        return visual.strip()
    parts: list[str] = []
    if (
        scene_narrative
        and not should_skip_beat_narrative_merge(
            segment_role=segment_role,
            start_visual=visual,
            visibility=visibility,
            is_intentional_empty=is_intentional_empty,
        )
        and not text_already_covered(visual, scene_narrative)
        and prompt_token_coverage(visual, scene_narrative) < NARRATIVE_MERGE_COVERAGE_THRESHOLD
    ):
        hint = scene_narrative
        if len(hint) > _SCENE_NARRATIVE_MERGE_MAX_CHARS:
            hint = f"{hint[:_SCENE_NARRATIVE_MERGE_MAX_CHARS].strip()}…"
        parts.append(hint)
    if visual:
        parts.append(visual)
    return "；".join(parts)


def build_shots_summary_with_provenance(shots: list[Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = [build_shot_t2i_provenance_summary(s) for s in shots]
    stats = {
        "total": len(rows),
        "narrative_merged_count": sum(1 for r in rows if r.get("narrative_merged")),
        "face_anchor_skip_count": sum(1 for r in rows if r.get("narrative_skip_reason") == "face_anchor"),
        "close_up_skip_count": sum(1 for r in rows if r.get("narrative_skip_reason") == "close_up"),
        "ffr_merged_count": 0,
    }
    return rows, stats
