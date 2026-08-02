"""Input-driven parse quality checks (no script-specific hardcoding)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Literal

from backend.long_video.beat_budget import group_shots_by_beat
from backend.long_video.visibility import vis_rank

_NAME_LOOK_RE = re.compile(r"([\u4e00-\u9fff]{2,8})（[^）\s]{1,24}）")
_NAME_FALSE_POSITIVE = re.compile(r"镜头|相机|画面|推近|拉远|特写|远景|中景|固定|手持|缓慢")
_DURATION_TAIL_RE = re.compile(
    r"(?:持续\s*[\d.]+\s*秒|[\d.]+\s*秒内|within\s*[\d.]+\s*s|for\s*[\d.]+\s*seconds?)",
    re.I,
)


@dataclass
class ParseQualityIssue:
    code: str
    message: str
    severity: Literal["warning", "critical"] = "warning"
    shot_index: int | None = None
    beat_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "shot_index": self.shot_index,
            "beat_index": self.beat_index,
        }


@dataclass
class ParseQualityResult:
    issues: list[ParseQualityIssue] = field(default_factory=list)

    @property
    def warning_messages(self) -> list[str]:
        return [i.message for i in self.issues]

    @property
    def critical_issues(self) -> list[ParseQualityIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    def issue_dicts(self) -> list[dict[str, Any]]:
        return [i.as_dict() for i in self.issues]


def protagonist_names_from_anchor(character_anchor: str) -> list[str]:
    """Resolve protagonist order from structured anchor blocks."""
    from backend.engine.llm.storyboard_cast import is_lead_character_role, parse_character_roster

    roster, _ = parse_character_roster(character_anchor or "")
    if not roster:
        return _legacy_protagonist_names(character_anchor)

    lead: list[str] = []
    rest: list[str] = []
    for ch in roster:
        is_lead = any(is_lead_character_role(lk.role, body=lk.body) for lk in ch.looks)
        (lead if is_lead else rest).append(ch.name)
    ordered = lead + rest
    return ordered or [ch.name for ch in roster]


def protagonist_names_from_dtos(character_dtos: list[dict]) -> list[str]:
    from backend.engine.llm.storyboard_cast import dtos_to_roster, is_lead_character_role

    roster = dtos_to_roster(character_dtos)
    if not roster:
        return []
    lead: list[str] = []
    rest: list[str] = []
    for ch in roster:
        is_lead = any(is_lead_character_role(lk.role, body=lk.body) for lk in ch.looks)
        (lead if is_lead else rest).append(ch.name)
    return lead + rest


def _legacy_protagonist_names(character_anchor: str) -> list[str]:
    names: list[str] = []
    for line in (character_anchor or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^[-*]\s*([^（(：:]+)", line)
        if m:
            names.append(m.group(1).strip())
            continue
        if "：" in line[:40] or ":" in line[:40]:
            head = re.split(r"[：:]", line, 1)[0].strip()
            if head and len(head) <= 24:
                names.append(head)
    return names


def _style_fragments(style_anchor: str, character_anchor: str) -> list[str]:
    chunks: list[str] = []
    if (style_anchor or "").strip():
        chunks.append(style_anchor.strip())
    for m in re.finditer(r"【画风】\s*(.+)", character_anchor or ""):
        chunks.append(m.group(1).strip())
    for m in re.finditer(r"\[Style\]\s*(.+)", character_anchor or "", re.I):
        chunks.append(m.group(1).strip())
    frags: list[str] = []
    for chunk in chunks:
        for part in re.split(r"[，,；;|]", chunk):
            part = part.strip()
            if len(part) >= 4:
                frags.append(part)
    return frags


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def strip_style_from_motion_prompt(
    text: str,
    *,
    style_anchor: str = "",
    character_anchor: str = "",
) -> str:
    """Remove project style boilerplate from I2V motion lines (style belongs in T2I/cast, not every clip)."""
    out = (text or "").strip()
    if not out:
        return ""

    style_full = (style_anchor or "").strip()
    if style_full:
        if out.startswith(style_full):
            out = out[len(style_full) :].lstrip("，,；;:. \t")
        out = out.replace(style_full, " ")

    frags = _style_fragments(style_anchor, character_anchor)
    for frag in sorted(frags, key=len, reverse=True):
        if frag:
            out = out.replace(frag, " ")

    for sep in ("；", ";"):
        if sep not in out:
            continue
        head, _, tail = out.partition(sep)
        head, tail = head.strip(), tail.strip()
        if not tail:
            continue
        head_residual = re.sub(r"[，,、\s]", "", head)
        if not head_residual or len(head_residual) <= 6:
            out = tail
            break
        if frags and head:
            covered = sum(1 for f in frags if f and f in head)
            if covered and covered >= max(1, len(head) // max(len(frags[0]), 1)):
                out = tail
                break

    out = _DURATION_TAIL_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"^[，,；;:\s]+", "", out)
    out = re.sub(r"[，,；;:\s]+$", "", out)
    out = re.sub(r"[，,]{2,}", "，", out)
    return out.strip()


def _motion_core(text: str, *, style_anchor: str = "", character_anchor: str = "") -> str:
    return _normalize_prompt(
        strip_style_from_motion_prompt(
            text,
            style_anchor=style_anchor,
            character_anchor=character_anchor,
        )
    )


def _prompt_similarity(a: str, b: str, *, style_anchor: str = "", character_anchor: str = "") -> float:
    na, nb = _normalize_prompt(a), _normalize_prompt(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    raw = SequenceMatcher(None, na, nb).ratio()
    ca = _motion_core(a, style_anchor=style_anchor, character_anchor=character_anchor)
    cb = _motion_core(b, style_anchor=style_anchor, character_anchor=character_anchor)
    if ca and cb and ca == cb:
        return 1.0
    if ca and cb:
        core = SequenceMatcher(None, ca, cb).ratio()
        return max(raw, core)
    return raw


def _prompt_fragments(text: str) -> list[str]:
    parts = re.split(r"[，,；;。\.]+", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def _beat_index_from_shot(shot: dict[str, Any], fallback: int) -> int:
    if shot.get("narrative_beat_index") is not None:
        try:
            return int(shot["narrative_beat_index"])
        except (TypeError, ValueError):
            pass
    gid = str(shot.get("segment_group_id") or "")
    if gid.startswith("beat_"):
        try:
            return int(gid.split("_", 1)[1])
        except ValueError:
            pass
    return fallback


def _narrative_from_beat_line(beat_raw: str) -> str:
    from backend.engine.llm.chapter_analyze import parse_structured_beat

    try:
        _title, beat = parse_structured_beat(beat_raw)
        return beat
    except ValueError:
        return beat_raw


def _narrative_visual_for_coverage(beat_raw: str) -> str:
    """Beat narrative body only (exclude title/shot/location metadata from coverage needle)."""
    raw = (beat_raw or "").strip()
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) >= 4:
            return "|".join(parts[3:]).strip()
    return _narrative_from_beat_line(beat_raw)


def _beat_group_prompt_text(shots: list[dict[str, Any]], indices: list[int]) -> str:
    parts: list[str] = []
    for i in indices:
        shot = shots[i]
        for fld in (
            "video_prompt",
            "motion_prompt",
            "scene_prompt",
            "start_visual_prompt",
            "visual_prompt",
            "anchor_visual_prompt",
            "clip_end_state",
        ):
            text = str(shot.get(fld) or "").strip()
            if text:
                parts.append(text)
        for name in shot.get("characters_on_screen") or []:
            n = str(name).strip()
            if n:
                parts.append(n)
    return " ".join(parts)


from backend.long_video.prompt_overlap import prompt_token_set as _token_set


def _roster_name_set(character_anchor: str, character_dtos: list[dict[str, Any]] | None) -> set[str]:
    from backend.engine.llm.storyboard_cast import parse_character_roster

    names: set[str] = set()
    for row in character_dtos or []:
        n = str(row.get("name") or "").strip()
        if n:
            names.add(n)
    roster, _ = parse_character_roster(character_anchor or "")
    for ch in roster:
        if ch.name:
            names.add(ch.name)
    return names


def _video_prompt(shot: dict[str, Any]) -> str:
    return str(shot.get("video_prompt") or shot.get("motion_prompt") or "")


def validate_parse_quality(
    shots: list[dict[str, Any]],
    *,
    beat_sheet: list[str],
    character_anchor: str = "",
    character_dtos: list[dict[str, Any]] | None = None,
    style_anchor: str = "",
    motion_similarity_threshold: float = 0.88,
    motion_core_similarity_threshold: float = 0.82,
    style_fragment_min_hits: int = 3,
) -> ParseQualityResult:
    """Cross-artifact quality checks; warnings only (does not fail parse)."""
    issues: list[ParseQualityIssue] = []
    if not shots:
        return ParseQualityResult(issues=issues)

    roster_names = _roster_name_set(character_anchor, character_dtos)
    beat_blob = "\n".join(beat_sheet)
    groups = group_shots_by_beat(shots)
    sim_kw = {"style_anchor": style_anchor, "character_anchor": character_anchor}

    def _similar(a: str, b: str) -> float:
        return _prompt_similarity(a, b, **sim_kw)

    # --- Motion duplication within beat group ---
    for gid, indices in groups.items():
        if len(indices) < 2:
            continue
        prompts = [_video_prompt(shots[i]) for i in indices]
        nonempty = [p for p in prompts if p.strip()]
        if len(nonempty) < 2:
            continue
        base = nonempty[0]
        dup_raw = all(_similar(base, p) >= motion_similarity_threshold for p in nonempty[1:])
        dup_core = all(
            SequenceMatcher(
                None,
                _motion_core(base, **sim_kw),
                _motion_core(p, **sim_kw),
            ).ratio()
            >= motion_core_similarity_threshold
            for p in nonempty[1:]
        )
        if dup_raw or dup_core:
            beat_i = _beat_index_from_shot(shots[indices[0]], 0)
            issues.append(
                ParseQualityIssue(
                    code="motion_duplicate_in_group",
                    message=(
                        f"{gid}: {len(indices)} segments share near-identical video_prompt "
                        f"(raw≥{motion_similarity_threshold:.2f} or core≥{motion_core_similarity_threshold:.2f}) "
                        f"— pre/face/post should differ"
                    ),
                    beat_index=beat_i,
                    shot_index=indices[0],
                )
            )
            continue
        roles = [str(shots[i].get("segment_role") or "") for i in indices]
        if "face_anchor" in roles:
            by_role = {str(shots[i].get("segment_role") or ""): _video_prompt(shots[i]) for i in indices}
            face = by_role.get("face_anchor", "").strip()
            for role in ("pre_anchor", "post_anchor"):
                other = by_role.get(role, "").strip()
                if not face or not other:
                    continue
                if (
                    _similar(face, other) >= motion_similarity_threshold
                    or SequenceMatcher(
                        None,
                        _motion_core(face, **sim_kw),
                        _motion_core(other, **sim_kw),
                    ).ratio()
                    >= motion_core_similarity_threshold
                ):
                    issues.append(
                        ParseQualityIssue(
                            code="motion_role_undifferentiated",
                            message=(
                                f"{gid}: {role} video_prompt too similar to face_anchor "
                                f"(raw≥{motion_similarity_threshold:.2f} or core≥{motion_core_similarity_threshold:.2f})"
                            ),
                            beat_index=_beat_index_from_shot(shots[indices[0]], 0),
                            shot_index=indices[0],
                        )
                    )

    # --- Style phrase spam (repeated fragments across prompts) ---
    frag_counter: Counter[str] = Counter()
    per_shot_frags: list[set[str]] = []
    for shot in shots:
        frags = set(_prompt_fragments(_video_prompt(shot)))
        per_shot_frags.append(frags)
        for frag in frags:
            frag_counter[frag] += 1
    min_hits = max(style_fragment_min_hits, (len(shots) + 1) // 2)
    for frag, count in frag_counter.most_common(8):
        if count < min_hits or len(frag) < 10:
            continue
        if count >= min_hits:
            style_frags = _style_fragments(style_anchor, character_anchor)
            is_style = any(
                sf and (frag in sf or sf in frag)
                for sf in style_frags
            )
            issues.append(
                ParseQualityIssue(
                    code="style_phrase_spam" if is_style else "prompt_fragment_spam",
                    message=(
                        f"{'style' if is_style else 'motion'} fragment {frag[:48]!r} repeated in "
                        f"{count}/{len(shots)} video_prompt lines — differentiate per segment"
                    ),
                )
            )
            break

    # --- Roster ↔ shots closure ---
    on_screen: set[str] = set()
    prompt_blob_parts: list[str] = []
    for shot in shots:
        on_screen.update(str(n).strip() for n in (shot.get("characters_on_screen") or []) if str(n).strip())
        prompt_blob_parts.append(str(shot.get("start_visual_prompt") or ""))
        prompt_blob_parts.append(_video_prompt(shot))
    prompt_blob = "\n".join(prompt_blob_parts)

    for name in sorted(on_screen):
        if roster_names and name not in roster_names:
            issues.append(
                ParseQualityIssue(
                    code="roster_shot_unknown_character",
                    message=f"characters_on_screen includes {name!r} not present in roster/anchor",
                    severity="critical",
                )
            )

    for i, shot in enumerate(shots):
        for m in _NAME_LOOK_RE.finditer(_video_prompt(shot)):
            ref_name = m.group(1).strip()
            if ref_name.startswith(("与", "和", "跟", "及", "同", "向", "对")):
                continue
            if _NAME_FALSE_POSITIVE.search(ref_name):
                continue
            if roster_names and ref_name not in roster_names:
                issues.append(
                    ParseQualityIssue(
                        code="prompt_unknown_character_ref",
                        message=f"shot {i} video_prompt references {ref_name!r} not in roster",
                        shot_index=i,
                    )
                )

    if roster_names:
        for name in sorted(roster_names):
            if name not in beat_blob and name not in prompt_blob and name not in on_screen:
                continue
            if name in beat_blob and name not in prompt_blob and name not in on_screen:
                issues.append(
                    ParseQualityIssue(
                        code="roster_beat_not_in_shots",
                        message=(
                            f"roster character {name!r} appears in beats but not in any shot "
                            f"prompts or characters_on_screen"
                        ),
                    )
                )

    # --- Cast lock bindings ---
    for i, shot in enumerate(shots):
        on = [str(n).strip() for n in (shot.get("characters_on_screen") or []) if str(n).strip()]
        looks = shot.get("cast_looks") or []
        if on and not looks:
            issues.append(
                ParseQualityIssue(
                    code="cast_look_missing",
                    message=f"shot {i} has characters_on_screen but empty cast_looks after cast_lock",
                    shot_index=i,
                )
            )
        for lk in looks:
            if not isinstance(lk, dict):
                continue
            if not str(lk.get("character_id") or "").strip() or not str(lk.get("look_id") or "").strip():
                issues.append(
                    ParseQualityIssue(
                        code="cast_look_incomplete",
                        message=f"shot {i} cast_looks entry missing character_id or look_id",
                        shot_index=i,
                    )
                )

    # --- Beat narrative coverage per group ---
    for beat_i, beat_raw in enumerate(beat_sheet):
        gid = f"beat_{beat_i}"
        indices = groups.get(gid, [])
        if not indices:
            issues.append(
                ParseQualityIssue(
                    code="beat_no_shots",
                    message=f"beat {beat_i} has no mapped shots",
                    beat_index=beat_i,
                    severity="critical",
                )
            )
            continue
        narrative = _narrative_visual_for_coverage(beat_raw)
        if not narrative.strip():
            continue
        group_prompt = _beat_group_prompt_text(shots, indices)
        from backend.long_video.prompt_overlap import prompt_narrative_coverage

        coverage = prompt_narrative_coverage(group_prompt, narrative)
        if coverage < 0.22:
            issues.append(
                ParseQualityIssue(
                    code="beat_narrative_undercovered",
                    message=(
                        f"beat {beat_i}: shot group motion has weak overlap with beat narrative "
                        f"(token coverage={coverage:.0%})"
                    ),
                    beat_index=beat_i,
                    shot_index=indices[0],
                )
            )

    # --- Visual prompt: inline look/wardrobe tags (names-only policy) ---
    from backend.engine.llm.storyboard_cast import find_name_look_tags

    for i, shot in enumerate(shots):
        for fld in ("scene_prompt", "start_visual_prompt", "visual_prompt", "anchor_visual_prompt"):
            text = str(shot.get(fld) or "")
            tags = find_name_look_tags(text)
            if tags:
                sample = tags[0]
                code = "scene_inline_look_tag" if fld == "scene_prompt" else "visual_inline_look_tag"
                issues.append(
                    ParseQualityIssue(
                        code=code,
                        message=(
                            f"shot {i} {fld} uses Name（…）tag {sample[0]!r}（{sample[1]!r}）"
                            f" — use bare names only; pick outfit in cast panel"
                        ),
                        shot_index=i,
                    )
                )

    # --- Instruction leak ---
    from backend.engine.llm.storyboard import find_instruction_leaks

    for i, shot in enumerate(shots):
        for fld in ("start_visual_prompt", "video_prompt", "motion_prompt"):
            text = str(shot.get(fld) or "")
            leaks = find_instruction_leaks(text)
            if leaks:
                issues.append(
                    ParseQualityIssue(
                        code="instruction_leak",
                        message=f"shot {i} {fld} echoes prompt instructions: {leaks[0]!r}",
                        shot_index=i,
                    )
                )
                break

    # --- Intra-beat visibility jumps ---
    for gid, indices in groups.items():
        for j in range(1, len(indices)):
            prev_i, cur_i = indices[j - 1], indices[j]
            prev, cur = shots[prev_i], shots[cur_i]
            if cur.get("start_frame_mode") in ("prev_segment_tail", "anchor_link"):
                continue
            prev_chars = set(prev.get("characters_on_screen") or [])
            cur_chars = set(cur.get("characters_on_screen") or [])
            for name in prev_chars & cur_chars:
                p_end = str(prev.get("end_visibility") or prev.get("first_frame_visibility") or "invisible")
                c_start = str(cur.get("first_frame_visibility") or "invisible")
                if vis_rank(c_start) > vis_rank(p_end) + 1:
                    issues.append(
                        ParseQualityIssue(
                            code="intra_beat_visibility_jump",
                            message=(
                                f"{gid}: shot {cur_i} {name} visibility jump {p_end} -> {c_start} "
                                f"within same beat group"
                            ),
                            shot_index=cur_i,
                            beat_index=_beat_index_from_shot(cur, 0),
                        )
                    )

    return ParseQualityResult(issues=issues)
