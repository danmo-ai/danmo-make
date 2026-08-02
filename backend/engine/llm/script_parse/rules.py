"""Deterministic segment expansion and metadata assignment."""
from __future__ import annotations

import hashlib
import re
from typing import Any

from backend.long_video.beat_budget import split_duration_parts
from backend.long_video.constants import MAX_CLIP_SEC, MIN_CLIP_SEC
from backend.long_video.keyframe_prompt_policy import normalize_face_anchor_characters_on_screen
from backend.engine.llm.script_parse.schemas import (
    BeatPlanArtifact,
    BeatPlanRowArtifact,
    BeatPlanRowLLM,
    PlannedSegmentArtifact,
    ScriptArtifact,
    ScriptBeatArtifact,
    sanitize_narrative_role,
    sanitize_reachability,
    sanitize_segment_role,
    sanitize_visibility,
)

_LOCATION_TIME_SUFFIXES = (
    "深夜",
    "拂晓前",
    "清晨",
    "白天",
    "傍晚",
    "子时",
    "凌晨",
    "雨夜",
)
_LOCATION_PLACE_SUFFIXES = ("内部", "外部", "深处", "门口", "区域", "通道", "走廊")
_LOCATION_STOP_TOKENS = frozenset({"内部", "区域", "通道", "场景", "画面", "地点"})


def normalize_location_key(text: str) -> str:
    s = (text or "").strip()
    for sep in ("·", "|", "—", "–", "-"):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
    for suffix in _LOCATION_TIME_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    for suffix in _LOCATION_PLACE_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


def _expand_location_token(token: str) -> set[str]:
    out = {token}
    if len(token) >= 3:
        for width in range(2, min(len(token), 5) + 1):
            for i in range(len(token) - width + 1):
                out.add(token[i : i + width])
    return out


def _location_tokens(text: str) -> set[str]:
    norm = normalize_location_key(text)
    parts = re.split(r"[的之与和]", norm)
    tokens: set[str] = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for match in re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", part):
            tokens.update(_expand_location_token(match))
        if len(part) >= 2:
            tokens.update(_expand_location_token(part))
    return {p for p in tokens if p not in _LOCATION_STOP_TOKENS and len(p) >= 2}


def location_matches(beat_loc: str, scene_name: str) -> bool:
    na = normalize_location_key(beat_loc)
    nb = normalize_location_key(scene_name)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = _location_tokens(na), _location_tokens(nb)
    if not ta or not tb:
        return False
    overlap = ta & tb
    if not overlap:
        return False
    return len(overlap) >= 1 and len(overlap) / min(len(ta), len(tb)) >= 0.34


def resolve_canonical_scene_name(beat_loc: str, scene_names: list[str]) -> str:
    loc = (beat_loc or "").strip()
    if not loc:
        return loc
    for sn in scene_names:
        if location_matches(loc, sn):
            return sn
    return loc


def canonicalize_script_locations(artifact: ScriptArtifact) -> ScriptArtifact:
    scene_names = [s.name.strip() for s in artifact.scenes if (s.name or "").strip()]
    if not scene_names:
        return artifact
    beats: list[ScriptBeatArtifact] = []
    for beat in artifact.beats:
        canon = resolve_canonical_scene_name(beat.location, scene_names)
        beats.append(beat.model_copy(update={"location": canon or beat.location.strip()}))
    return artifact.model_copy(update={"beats": beats})


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def assign_character_scene_ids(artifact: ScriptArtifact) -> ScriptArtifact:
    """Ensure stable ids on characters, scenes, looks."""
    chars = []
    for ch in artifact.characters:
        cid = ch.id or _stable_id("char", ch.name)
        looks = []
        for i, lk in enumerate(ch.looks):
            lid = lk.id or _stable_id("look", f"{ch.name}|{lk.label}|{i}")
            looks.append(lk.model_copy(update={"id": lid}))
        default = ch.default_look_id or (looks[0].id if looks else "")
        chars.append(ch.model_copy(update={"id": cid, "looks": looks, "default_look_id": default}))

    scenes = []
    for sc in artifact.scenes:
        sid = sc.id or _stable_id("scene", sc.name)
        looks = []
        for i, lk in enumerate(sc.looks):
            lid = lk.id or _stable_id("scene_look", f"{sc.name}|{lk.label}|{i}")
            looks.append(lk.model_copy(update={"id": lid}))
        default = sc.default_look_id or (looks[0].id if looks else "")
        scenes.append(sc.model_copy(update={"id": sid, "looks": looks, "default_look_id": default}))

    beats = []
    for i, b in enumerate(artifact.beats):
        beats.append(b.model_copy(update={"index": i}))

    return artifact.model_copy(update={"characters": chars, "scenes": scenes, "beats": beats})


def _start_frame_mode_for_role(role: str, prev_role: str | None) -> str:
    if role == "tail_continuation":
        return "prev_segment_tail"
    if role == "post_anchor" and prev_role == "face_anchor":
        return "anchor_link"
    return "keyframe"


def expand_beat_plan_rows(
    script: ScriptArtifact,
    llm_rows: list[BeatPlanRowLLM],
    *,
    max_clip_sec: float = MAX_CLIP_SEC,
) -> BeatPlanArtifact:
    """Assign global segment indices and group metadata from LLM beat rows."""
    cap = max(MIN_CLIP_SEC, float(max_clip_sec))
    lo = max(0.5, float(MIN_CLIP_SEC))
    beat_map = {b.index: b for b in script.beats}
    out_rows: list[BeatPlanRowArtifact] = []
    seg_idx = 0
    prev_role: str | None = None

    for row in sorted(llm_rows, key=lambda r: r.beat_index):
        beat = beat_map.get(row.beat_index)
        gid = f"beat_{row.beat_index}"
        segments: list[PlannedSegmentArtifact] = []
        group_i = 0

        for seg_llm in row.segments:
            dur = max(lo, min(cap, float(seg_llm.duration_sec)))
            parts = split_duration_parts(dur, max_clip_sec=cap) if dur > cap + 0.01 else [dur]
            for part_i, part_dur in enumerate(parts):
                role = sanitize_segment_role(seg_llm.role)
                if part_i > 0:
                    role = "tail_continuation"
                mode = _start_frame_mode_for_role(role, prev_role)
                face_id = ""
                if role == "face_anchor":
                    face_id = _stable_id("anchor", f"{gid}|{seg_idx}")
                on_screen = normalize_face_anchor_characters_on_screen(
                    role,
                    list(seg_llm.characters_on_screen),
                )
                segments.append(
                    PlannedSegmentArtifact(
                        segment_index=seg_idx,
                        beat_index=row.beat_index,
                        role=role,
                        duration_sec=round(part_dur, 1),
                        shot_size=seg_llm.shot_size or (beat.suggested_shot_size if beat else ""),
                        characters_on_screen=on_screen,
                        start_visibility=sanitize_visibility(seg_llm.start_visibility),
                        end_visibility=sanitize_visibility(seg_llm.end_visibility, default="full_face"),
                        first_frame_requirement=seg_llm.first_frame_requirement,
                        reachability=sanitize_reachability(seg_llm.reachability),
                        is_intentional_empty=seg_llm.is_intentional_empty,
                        spatial=seg_llm.spatial,
                        start_frame_mode=mode,  # type: ignore[arg-type]
                        segment_group_id=gid,
                        segment_group_index=group_i,
                        face_anchor_shot_id=face_id,
                    )
                )
                prev_role = role
                seg_idx += 1
                group_i += 1

        out_rows.append(
            BeatPlanRowArtifact(
                beat_index=row.beat_index,
                shot_intent=row.shot_intent,
                narrative_role=sanitize_narrative_role(row.narrative_role),
                segments=segments,
            )
        )

    return BeatPlanArtifact(beats=out_rows)


def merge_spatial_into_scenes(
    script: ScriptArtifact,
    beat_plan: BeatPlanArtifact,
) -> list[dict[str, Any]]:
    """Write spatial snippets from beat plan onto matching scene entities."""
    scene_by_name = {sc.name.strip(): sc for sc in script.scenes}
    layouts: dict[str, dict] = {}

    for brow in beat_plan.beats:
        beat = next((b for b in script.beats if b.index == brow.beat_index), None)
        loc = (beat.location if beat else "").strip()
        for seg in brow.segments:
            sp = seg.spatial
            if sp and sp.location:
                loc = sp.location.strip()
            if not loc:
                continue
            key = loc
            if sp:
                layouts[key] = {
                    "location": sp.location or loc,
                    "dimensions": sp.dimensions,
                    "objects": sp.objects,
                    "camera_zones": [z.model_dump() for z in sp.camera_zones],
                }

    out: list[dict[str, Any]] = []
    for sc in script.scenes:
        row = sc.model_dump()
        for loc, layout in layouts.items():
            if loc in sc.name or sc.name in loc:
                row["spatial_layout_json"] = layout
                break
        out.append(row)
    return out


def flatten_planned_segments(beat_plan: BeatPlanArtifact) -> list[PlannedSegmentArtifact]:
    segs: list[PlannedSegmentArtifact] = []
    for brow in beat_plan.beats:
        segs.extend(brow.segments)
    return sorted(segs, key=lambda s: s.segment_index)
