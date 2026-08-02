"""T2I keyframe prompt policy driven by parse artifact fields (visibility + role).

Consumers must use these helpers instead of inferring framing from free-text shot_size.
"""
from __future__ import annotations

import re
from typing import Any, Literal

CastReferenceScope = Literal["none", "wardrobe", "face"]
Visibility = Literal["invisible", "silhouette", "partial", "full_face"]
SegmentRole = Literal[
    "establishing",
    "pre_anchor",
    "face_anchor",
    "post_anchor",
    "keyframe",
    "tail_continuation",
]


def cast_reference_scope(
    *,
    visibility: str,
    segment_role: str,
    is_intentional_empty: bool = False,
) -> CastReferenceScope:
    if is_intentional_empty or visibility == "invisible":
        return "none"
    if segment_role == "face_anchor":
        return "face"
    if visibility == "full_face":
        return "face"
    return "wardrobe"


def should_merge_scene_prompt_into_t2i(
    *,
    start_visual: str,
    scene_prompt: str = "",
) -> bool:
    """five_aspect.scene / scene_prompt is for I2V + scene-look binding, not T2I merge."""
    if (start_visual or "").strip():
        return False
    return bool((scene_prompt or "").strip())


def should_merge_beat_narrative_into_t2i(
    *,
    start_visual: str,
    segment_role: str,
    visibility: str,
    is_intentional_empty: bool = False,
) -> bool:
    if (start_visual or "").strip():
        return False
    if is_intentional_empty or segment_role == "face_anchor":
        return False
    if visibility in ("full_face", "partial", "silhouette"):
        return False
    return True


def validate_visibility_role_contract(
    *,
    segment_role: str,
    start_visibility: str,
    beat_index: int | None = None,
    segment_label: str = "segment",
) -> list[str]:
    """Beat plan / shot spec contract: full_face start is face_anchor-only."""
    issues: list[str] = []
    prefix = f"beat {beat_index} {segment_label}" if beat_index is not None else segment_label
    if segment_role == "face_anchor" and start_visibility != "full_face":
        issues.append(f"{prefix}: face_anchor requires start_visibility full_face, got {start_visibility!r}")
    if segment_role != "face_anchor" and start_visibility == "full_face":
        issues.append(
            f"{prefix}: start_visibility full_face is reserved for face_anchor (role={segment_role!r})"
        )
    return issues


_FACE_DEMAND_RE = re.compile(
    r"面部(?:特写|大特写|近景|清晰(?!\s*不可)|可读)"
    r"|五官(?:清晰|特写)"
    r"|(?:^|[^不])脸(?:部)?特写"
    r"|face\s*(?:close[-\s]?up|readable|visible)"
    r"|full\s*face",
    re.IGNORECASE,
)

_FACE_HIDDEN_RE = re.compile(
    r"(?:无|不含|没有|未见)(?:清晰)?面部"
    r"|面部(?:完全)?不可见|不可见(?:的)?面部|不露脸|无(?:清晰)?五官"
    r"|(?:仅|只)?(?:露出)?(?:面部)?轮廓(?:或不可见)?"
    r"|face\s*(?:not|in)visible|no\s*(?:readable\s*)?face",
    re.IGNORECASE,
)


_PARTIAL_BODY_PARTS_RE = re.compile(
    r"手部|手指|碎片|肢体|背影|轮廓|镜面|道具|屏幕",
)


def camera_zone_conflicts_with_partial_visibility(visible_area: str) -> bool:
    """True when a camera zone demands readable face framing but segment is partial."""
    area = (visible_area or "").strip()
    if not area:
        return False
    if _FACE_HIDDEN_RE.search(area):
        return False
    if _PARTIAL_BODY_PARTS_RE.search(area) and "面部" in area and not _FACE_DEMAND_RE.search(area):
        return False
    if _FACE_DEMAND_RE.search(area):
        return True
    if "面部" in area and not re.search(
        r"(?:无|不含|没有|未见)(?:清晰)?面部|面部.{0,10}(?:不可见|轮廓)",
        area,
    ):
        return True
    return False


def sanitize_shot_spec_prompts(
    *,
    role: str,
    start_visual: str,
    anchor_visual: str,
) -> tuple[str, str]:
    """Non face_anchor segments must not carry anchor_visual (LLM often mis-fills)."""
    sv = (start_visual or "").strip()
    av = (anchor_visual or "").strip()
    if (role or "").strip() == "face_anchor":
        return sv, av
    return sv, ""


_FRAMING_MARKERS_RE = re.compile(
    r"特写|近景|中景|大特写|极特写|MCU|CU|ECU|MS|close[\s-]?up|medium[\s-]?close|extreme[\s-]?close",
    re.IGNORECASE,
)

_SINGLE_NAME_ONLY_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z·\s]{2,12}$")


def anchor_visual_is_inadequate(anchor_visual: str) -> bool:
    """True when face_anchor anchor_visual is a stub (name-only or missing framing)."""
    av = (anchor_visual or "").strip()
    if not av:
        return True
    if len(av) < 10:
        return True
    if _SINGLE_NAME_ONLY_RE.fullmatch(av) and not _FRAMING_MARKERS_RE.search(av):
        return True
    if not _FRAMING_MARKERS_RE.search(av) and len(av) < 20:
        return True
    return False


def _richer_face_anchor_candidate(*candidates: str) -> str:
    best = ""
    best_score = -1
    for raw in candidates:
        c = (raw or "").strip()
        if not c:
            continue
        score = len(c)
        if _FRAMING_MARKERS_RE.search(c):
            score += 40
        if _FACE_DEMAND_RE.search(c):
            score += 20
        if score > best_score:
            best_score = score
            best = c
    return best


def coalesce_face_anchor_visual(
    *,
    anchor_visual: str,
    start_visual: str,
    five_aspect_subject: str = "",
    first_frame_requirement: str = "",
    shot_size: str = "",
    primary_name: str = "",
) -> tuple[str, str]:
    """Backfill stub anchor_visual from start_visual / subject / ffr (protocol, not story-specific)."""
    av = (anchor_visual or "").strip()
    sv = (start_visual or "").strip()
    if anchor_visual_is_inadequate(av):
        pool = [c for c in (sv, first_frame_requirement, five_aspect_subject) if (c or "").strip()]
        rich = [c for c in pool if _FRAMING_MARKERS_RE.search(c) or len(c) >= 20]
        candidates = rich or pool
        if primary_name:
            named = [c for c in candidates if primary_name in c]
            if named:
                candidates = named
        pick = _richer_face_anchor_candidate(*candidates)
        if pick:
            av = pick
            if primary_name and primary_name not in av:
                av = f"{primary_name}，{av}"
        elif shot_size.strip():
            base = primary_name or av or (sv.split("，")[0][:8] if sv else "")
            av = f"{shot_size.strip()}，{base}".strip("，")
    if not sv and av:
        sv = av
    return sv, av


def partial_framing_conflicts_with_visibility(text: str) -> bool:
    """True when partial/silhouette framing text still demands readable facial detail."""
    return camera_zone_conflicts_with_partial_visibility(text)


def validate_beat_plan_row_contract(
    *,
    beat_index: int,
    segments: list[Any],
) -> list[str]:
    """Beat-level segment structure contract (visibility + face_anchor cardinality)."""
    issues: list[str] = []
    face_anchor_indices: list[int] = []
    for seg_i, seg in enumerate(segments):
        role = str(getattr(seg, "role", "") or "")
        vis = str(getattr(seg, "start_visibility", "") or "")
        issues.extend(
            validate_visibility_role_contract(
                segment_role=role,
                start_visibility=vis,
                beat_index=beat_index,
                segment_label=f"segment[{seg_i}]",
            )
        )
        if role == "face_anchor":
            face_anchor_indices.append(seg_i)
            on_screen = list(getattr(seg, "characters_on_screen", None) or [])
            if len(on_screen) > 1:
                issues.append(
                    f"beat {beat_index} segment[{seg_i}]: face_anchor allows one character on screen, "
                    f"got {on_screen!r}"
                )
        if vis in ("partial", "silhouette"):
            ffr = str(getattr(seg, "first_frame_requirement", "") or "")
            if ffr and partial_framing_conflicts_with_visibility(ffr):
                issues.append(
                    f"beat {beat_index} segment[{seg_i}]: {vis} first_frame_requirement "
                    f"conflicts with readable-face framing: {ffr!r}"
                )
            spatial = getattr(seg, "spatial", None)
            if spatial and vis == "partial":
                for zone in getattr(spatial, "camera_zones", None) or []:
                    area = str(getattr(zone, "visible_area", "") or "").strip()
                    if partial_framing_conflicts_with_visibility(area):
                        issues.append(
                            f"beat {beat_index} segment[{seg_i}]: partial start_visibility "
                            f"conflicts with camera_zone visible_area {area!r}"
                        )
    if len(face_anchor_indices) > 1:
        issues.append(
            f"beat {beat_index}: at most one face_anchor segment, got indices {face_anchor_indices}"
        )
    return issues


def validate_shot_spec_partial_framing(
    *,
    start_visibility: str,
    start_visual: str,
    segment_index: int,
) -> list[str]:
    if start_visibility not in ("partial", "silhouette"):
        return []
    if start_visual and partial_framing_conflicts_with_visibility(start_visual):
        return [
            f"segment {segment_index}: {start_visibility} start_visual "
            f"conflicts with readable-face wording: {start_visual!r}"
        ]
    return []


def normalize_face_anchor_characters_on_screen(
    role: str,
    characters_on_screen: list[str],
) -> list[str]:
    """face_anchor: single on-screen identity for T2I face reference."""
    names = [str(n).strip() for n in characters_on_screen if str(n).strip()]
    if (role or "").strip() == "face_anchor" and len(names) > 1:
        return names[:1]
    return names
