"""Long-video storyboard planning, parsing, and quality checks."""
from __future__ import annotations

import re
from dataclasses import asdict

from backend.long_video.plan import ShotPlan, build_shot_plan
from backend.engine.families.ltx.long_video_plan import LongVideoPlan, build_long_video_plan

EXPAND_BATCH_SIZE = 4

_TAG_BOUNDARY = r"(?=\[Visual|\[Motion|\[Beat|\[Anchor|\[Segment|\[Opening|\Z)"
_ANCHOR_RE = re.compile(r"\[Anchor\]\s*(.+?)(?=\[Beat|\[Segment|\[Visual|\[Motion|\Z)", re.S | re.I)
_BEAT_RE = re.compile(r"\[Beat\s*(\d+)?\]\s*(.+?)(?=\[Beat|\[Segment|\[Visual|\[Motion|\Z)", re.S | re.I)
_OPENING_RE = re.compile(r"\[Opening\]\s*(.+?)(?=\[Segment|\Z)", re.S | re.I)
_SEGMENT_RE = re.compile(r"\[Segment\s*(\d+)?\]\s*(.+?)(?=\[Segment|\[Opening|\Z)", re.S | re.I)
_VISUAL_RE = re.compile(rf"\[Visual\s*(\d+)?\]\s*(.+?){_TAG_BOUNDARY}", re.S | re.I)
_MOTION_RE = re.compile(rf"\[Motion\s*(\d+)?\]\s*(.+?){_TAG_BOUNDARY}", re.S | re.I)


def _extract_loose_prompt_lines(raw: str) -> list[str]:
    """Best-effort lines when LLM ignores [Visual]/[Segment] tags."""
    lines: list[str] = []
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(
            r"^\[(?:Visual|Motion|Segment|Shot|Beat|Opening|Scene)\s*\d*\]\s*",
            "",
            s,
            flags=re.I,
        )
        s = re.sub(r"^\d+[\.)]\s*", "", s)
        s = s.lstrip("-•* ").strip()
        if s and not re.match(r"^\[Anchor\]", s, re.I):
            lines.append(s)
    if not lines:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", (raw or "").strip()) if b.strip()]
        for block in blocks:
            one = " ".join(block.split())
            if one and not re.match(r"^\[Anchor\]", one, re.I):
                lines.append(one)
    return [ln for ln in lines if len(ln) >= 6]


def _pad_strings(
    items: list[str],
    count: int,
    *,
    label: str,
    fallback: list[str] | None = None,
) -> list[str]:
    """Pad a parsed list to *count*; prefer *fallback* entries before repeating the last."""
    if not items and fallback:
        items = [s for s in fallback if s and s.strip()]
    if count <= 0:
        return items
    if not items:
        raise ValueError(f"{label}: no items parsed")
    out = list(items)
    while len(out) < count:
        if fallback and len(fallback) > len(out):
            out.append(fallback[len(out)])
        else:
            out.append(out[-1])
    return out[:count]


def _split_beat_marked_lines(text: str) -> list[str]:
    """Split text that incorrectly merged multiple ``[Beat N]`` blocks."""
    raw = (text or "").strip()
    if not raw:
        return []
    tagged = [m.group(2).strip() for m in _BEAT_RE.finditer(raw)]
    if tagged:
        return tagged
    if not re.search(r"\[Beat\s*\d+\]", raw, re.I):
        return [raw]
    parts = re.split(r"(?=\[Beat\s*\d+\])", raw, flags=re.I)
    out: list[str] = []
    for part in parts:
        line = re.sub(r"^\[Beat\s*\d+\]\s*", "", part.strip(), flags=re.I).strip()
        if line and not re.match(r"^\[Anchor\]", line, re.I):
            out.append(line)
    return out


def _contains_beat_markers(text: str) -> bool:
    return bool(re.search(r"\[Beat\s*\d+\]", text or "", re.I))


def _strip_beat_markers(text: str) -> str:
    s = re.sub(r"\[Beat\s*\d+\]\s*", "", (text or "").strip(), flags=re.I)
    return re.sub(r"\[Anchor\]\s*", "", s, flags=re.I).strip()


def _looks_like_motion_only(text: str) -> bool:
    """Heuristic: camera/movement line mis-assigned as a keyframe visual."""
    s = (text or "").strip()
    if not s:
        return True
    camera_leads = (
        "镜头", "跟拍", "推近", "拉远", "固定机位", "手持", "快切", "环绕",
        "handheld", "dolly", "pan ", "tracking shot", "camera ",
    )
    lower = s.lower()
    if any(s.startswith(cue) or lower.startswith(cue) for cue in camera_leads):
        return True
    if len(s) <= 24 and any(cue in s or cue in lower for cue in camera_leads):
        scene_cues = ("中景", "近景", "远景", "特写", "广角", "背景", "站立", "坐在", "穿", "持", "。")
        return not any(cue in s for cue in scene_cues)
    return False


def _strip_structural_tags(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^\[(?:Anchor|Visual|Motion|Beat|Segment|Opening)\s*\d*\]\s*", "", s, flags=re.I)
    return s.strip()


def _pick_shot_line(
    text: str,
    index: int,
    *,
    character_anchor: str = "",
    beat_sheet: list[str] | None = None,
) -> str:
    """Resolve one shot's prompt from expand output or beat sheet."""
    text = _strip_structural_tags(text)
    if _contains_beat_markers(text):
        parts = _split_beat_marked_lines(text)
        if parts and index < len(parts):
            return parts[index].strip()
        if beat_sheet and index < len(beat_sheet):
            return beat_sheet[index].strip()
    parts = _split_beat_marked_lines(text)
    if len(parts) > 1:
        anchor = (character_anchor or "").strip()
        filtered: list[str] = []
        for part in parts:
            if anchor and (part == anchor or part.startswith(anchor[: min(24, len(anchor))])):
                continue
            filtered.append(part)
        if filtered and index < len(filtered):
            return filtered[index].strip()
        if beat_sheet and index < len(beat_sheet):
            return beat_sheet[index].strip()
    if len(parts) == 1:
        text = parts[0]
    if _looks_like_motion_only(text) and beat_sheet and index < len(beat_sheet):
        return beat_sheet[index].strip()
    anchor = (character_anchor or "").strip()
    if anchor and text.startswith(anchor[: min(len(anchor), 40)]):
        text = text[len(anchor) :].lstrip("。. \n")
    return _strip_beat_markers(text).strip()


def _sanitize_shot_field(text: str, index: int, *, character_anchor: str = "") -> str:
    return _pick_shot_line(text, index, character_anchor=character_anchor)


def _pairs_usable(
    pairs: list[tuple[str, str]],
    shot_count: int,
    *,
    beat_sheet: list[str] | None = None,
) -> bool:
    if len(pairs) < shot_count or shot_count <= 0:
        return False
    beats = beat_sheet or []
    slice_pairs = pairs[:shot_count]
    visuals = [
        resolve_shot_scene_for_index(
            index=i,
            raw_visual=v,
            beat_sheet=beats,
            character_anchor="",
        )
        for i, (v, _) in enumerate(slice_pairs)
    ]
    if any(not is_valid_shot_scene_text(v) for v in visuals):
        return False
    if any(len(v) < 6 for v in visuals):
        return False
    if any(_contains_beat_markers(v) for v in visuals):
        return False
    if any(_looks_like_motion_only(v) for v in visuals):
        return False
    unique = {v[:80] for v in visuals}
    if shot_count >= 3 and len(unique) < min(3, shot_count):
        return False
    for _i, (v, _m) in enumerate(slice_pairs):
        if len(_split_beat_marked_lines(v)) > 1:
            return False
    return True


def _pad_pairs(
    pairs: list[tuple[str, str]],
    count: int,
    *,
    fallback: list[str] | None = None,
) -> list[tuple[str, str]]:
    if count <= 0:
        return pairs
    out = list(pairs)
    beats = [b.strip() for b in (fallback or []) if b and b.strip()]
    while len(out) < count:
        i = len(out)
        if beats and i < len(beats):
            nxt = beats[i + 1] if i + 1 < len(beats) else beats[i]
            out.append((beats[i], nxt))
        elif out:
            out.append(out[-1])
        else:
            break
    return out[:count]


def coalesce_dual_pairs(
    pairs: list[tuple[str, str]],
    beat_sheet: list[str],
    shot_count: int,
    *,
    character_anchor: str = "",
) -> list[tuple[str, str]]:
    """Normalize LLM pairs; fall back to per-beat prompts when expand output is duplicated or bloated."""
    if any(_contains_beat_markers(v) or len(_split_beat_marked_lines(v)) > 1 for v, _ in pairs[:shot_count]):
        return dual_pairs_from_beats(beat_sheet, shot_count, character_anchor=character_anchor)
    normalized: list[tuple[str, str]] = []
    for i, (visual, motion) in enumerate(pairs[:shot_count]):
        v = resolve_shot_scene_for_index(
            index=i,
            raw_visual=visual,
            beat_sheet=beat_sheet,
            character_anchor=character_anchor,
        )
        m = _pick_shot_line(motion, i, character_anchor=character_anchor, beat_sheet=beat_sheet)
        if not is_valid_shot_scene_text(m) or m.strip() == v.strip():
            from backend.engine.llm.motion_prompt import motion_prompt_from_beat

            nxt = beat_sheet[i + 1] if i + 1 < len(beat_sheet) else v
            m = motion_prompt_from_beat(v, beat=v, next_visual=nxt)
        if not m:
            m = v
        normalized.append((v, m))
    padded = _pad_pairs(normalized, shot_count, fallback=beat_sheet)
    visual_keys = {(v or "")[:80] for v, _ in padded}
    motion_keys = {(m or "")[:60] for _, m in padded}
    if padded and beat_sheet and (
        len(visual_keys) < min(3, shot_count) or len(motion_keys) == 1
    ):
        return dual_pairs_from_beats(beat_sheet, shot_count, character_anchor=character_anchor)
    if _pairs_usable(padded, shot_count, beat_sheet=beat_sheet):
        return padded
    return dual_pairs_from_beats(beat_sheet, shot_count, character_anchor=character_anchor)


def plan_to_dto(plan: LongVideoPlan) -> dict:
    d = asdict(plan)
    d["segment_durations_sec"] = list(plan.segment_durations_sec)
    return d


def shot_plan_to_dto(plan: ShotPlan) -> dict:
    d = asdict(plan)
    d["segment_durations_sec"] = list(plan.segment_durations_sec)
    return d


def parse_plan_script(text: str, *, expected_beats: int) -> tuple[str, list[str]]:
    raw = (text or "").strip()
    anchor_m = _ANCHOR_RE.search(raw)
    character_anchor = anchor_m.group(1).strip() if anchor_m else ""
    beats: list[str] = []
    for m in _BEAT_RE.finditer(raw):
        line = m.group(2).strip()
        if line:
            beats.append(line)
    if not beats:
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.lower().startswith("[anchor"):
                continue
            cleaned = re.sub(r"^\[Beat\s*\d+\]\s*", "", s, flags=re.I).lstrip("-•* ").strip()
            if cleaned:
                beats.append(cleaned)
    if expected_beats > 0:
        beats = _pad_strings(beats, expected_beats, label="plan parse")
    return character_anchor, beats[:expected_beats] if expected_beats else beats


def parse_expand_script(
    text: str,
    *,
    expected_segments: int,
    fallback: list[str] | None = None,
) -> tuple[str, list[str]]:
    raw = (text or "").strip()
    opening_m = _OPENING_RE.search(raw)
    opening = opening_m.group(1).strip() if opening_m else ""
    segments: list[str] = []
    for m in _SEGMENT_RE.finditer(raw):
        seg = m.group(2).strip()
        if seg:
            segments.append(seg)
    if not segments:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
        if opening and blocks and blocks[0].startswith("[Opening]"):
            blocks = blocks[1:]
        segments = blocks[1:] if opening and len(blocks) > 1 else blocks
    if not segments:
        segments = _extract_loose_prompt_lines(raw)
    if expected_segments:
        segments = _pad_strings(
            segments,
            expected_segments,
            label="expand parse",
            fallback=fallback,
        )
    return opening, segments[:expected_segments] if expected_segments else segments


def parse_dual_shot_script(
    text: str,
    *,
    expected_shots: int,
    fallback: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Parse [Visual N] / [Motion N] pairs; falls back to [Segment N] for both fields."""
    raw = (text or "").strip()
    visuals: dict[int, str] = {}
    motions: dict[int, str] = {}
    for m in _VISUAL_RE.finditer(raw):
        idx = int(m.group(1) or len(visuals) + 1)
        visuals[idx] = _strip_leaked_instructions(m.group(2).strip())
    for m in _MOTION_RE.finditer(raw):
        idx = int(m.group(1) or len(motions) + 1)
        motions[idx] = _strip_leaked_instructions(m.group(2).strip())
    if visuals or motions:
        beats_from_raw = [m.group(2).strip() for m in _BEAT_RE.finditer(raw)]
        only_visual_one = max(visuals.keys(), default=0) <= 1 and bool(visuals.get(1))
        count = max(expected_shots, max(visuals.keys(), default=0), max(motions.keys(), default=0))
        pairs: list[tuple[str, str]] = []
        for i in range(1, count + 1):
            bi = i - 1
            visual = visuals.get(i, "")
            motion = motions.get(i, "")
            if beats_from_raw and bi < len(beats_from_raw):
                if only_visual_one or not visual or _contains_beat_markers(visual):
                    visual = beats_from_raw[bi]
            if not visual and fallback and bi < len(fallback):
                visual = fallback[bi]
            if not motion and fallback and bi + 1 < len(fallback):
                motion = fallback[bi + 1]
            elif not motion and visual:
                motion = visual
            if visual or motion:
                pairs.append((visual, motion or visual))
        if not pairs:
            _opening, segments = parse_expand_script(
                raw, expected_segments=expected_shots, fallback=fallback,
            )
            return [(seg, seg) for seg in segments]
        if expected_shots:
            pairs = _pad_pairs(pairs, expected_shots, fallback=fallback)
        return pairs

    _opening, segments = parse_expand_script(
        raw, expected_segments=expected_shots, fallback=fallback,
    )
    return [(seg, seg) for seg in segments]


def dual_pairs_from_beats(
    beat_sheet: list[str],
    shot_count: int,
    *,
    character_anchor: str = "",
    locale: str | None = None,
) -> list[tuple[str, str]]:
    """Build visual/motion pairs from plan beats when Expand LLM output is unusable."""
    from backend.engine.llm.motion_prompt import motion_prompt_from_beat

    if shot_count <= 0:
        return []
    beats = [b.strip() for b in beat_sheet if b and b.strip()]
    if not beats and character_anchor.strip():
        beats = [character_anchor.strip()]
    beats = _pad_strings(beats, shot_count, label="beat fallback", fallback=beats or None)
    pairs: list[tuple[str, str]] = []
    for i, visual in enumerate(beats[:shot_count]):
        nxt = beats[i + 1] if i + 1 < len(beats) else visual
        motion = motion_prompt_from_beat(
            visual,
            beat=visual,
            next_visual=nxt,
            locale=locale,
        )
        pairs.append((visual, motion))
    return pairs


def merge_expand_batches(
    opening_parts: list[str],
    segment_batches: list[list[str]],
) -> tuple[str, list[str]]:
    opening = next((o.strip() for o in opening_parts if o and o.strip()), "")
    merged: list[str] = []
    for batch in segment_batches:
        merged.extend(batch)
    return opening, merged


def storyboard_shot_pairs_ok(
    pairs: list[tuple[str, str]],
    *,
    shot_count: int,
    beat_sheet: list[str],
    character_anchor: str = "",
    characters: list[dict] | None = None,
    style_anchor: str = "",
    locale: str | None = None,
) -> bool:
    if not _pairs_usable(pairs, shot_count, beat_sheet=beat_sheet):
        return False
    if len(beat_sheet) < shot_count:
        return False
    if not storyboard_prompts_self_contained(pairs, shot_count):
        return False
    if not storyboard_visuals_include_appearance(
        pairs,
        character_anchor,
        shot_count,
        characters=characters,
        style_anchor=style_anchor,
        beat_sheet=beat_sheet,
        locale=locale,
    ):
        return False
    for beat in beat_sheet[:shot_count]:
        if prompt_leads_with_standalone_pronoun(beat):
            return False
    return True


def storyboard_quality_ok(
    *,
    character_anchor: str,
    opening_prompt: str,
    segment_prompts: list[str],
    beat_sheet: list[str],
    plan: LongVideoPlan,
    min_segment_prompts: int = 0,
) -> bool:
    if len(character_anchor.strip()) < 12:
        return False
    if len(opening_prompt.strip()) < 20:
        return False
    seg_need = min_segment_prompts if min_segment_prompts > 0 else plan.extend_pass_count
    if len(segment_prompts) < seg_need:
        return False
    if len(beat_sheet) < plan.total_segments:
        return False
    seen = set()
    for p in segment_prompts[:seg_need]:
        key = p.strip()[:40]
        if key in seen:
            return False
        seen.add(key)
        if len(p.strip()) < 16:
            return False
    return True


_INSTRUCTION_MARKERS = (
    "必须点名",
    "输出格式",
    "cast roster",
    "角色阵容",
    "Do not invent",
    "预建的 beat",
    "Narrative budget",
    "honor shot-size",
    "Downstream code",
    "Every on-screen character MUST",
    "CRITICAL:",
    "然后是角色",
    "每个块是",
    "budget:",
    "Output constraints",
    "Output language",
    "然后是 【画风】",
    "关键帧时刻",
    "compact=快速",
)


def find_instruction_leaks(text: str) -> list[str]:
    """Return instruction/locale markers present in model output (parse quality)."""
    blob = text or ""
    return [marker for marker in _INSTRUCTION_MARKERS if marker in blob]


def _strip_leaked_instructions(text: str) -> str:
    """Remove locale-block / format-rule echo from LLM expand output."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if KEYFRAME_REF_DIVIDER in raw:
        head = raw.split(KEYFRAME_REF_DIVIDER, 1)[0].strip()
        if head:
            raw = head
    lines: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("- ") and any(m in s for m in _INSTRUCTION_MARKERS):
            continue
        if any(m in s for m in _INSTRUCTION_MARKERS):
            continue
        lines.append(s)
    cleaned = " ".join(lines).strip()
    return cleaned or raw.splitlines()[0].strip() if raw.splitlines() else raw


def is_valid_shot_scene_text(text: str) -> bool:
    """Reject LLM instruction leakage and non-scene boilerplate."""
    raw = (text or "").strip()
    if len(raw) < 8 or len(raw) > 480:
        return False
    lower = raw.lower()
    for marker in _INSTRUCTION_MARKERS:
        if marker.lower() in lower or marker in raw:
            return False
    bullet_lines = [ln for ln in raw.splitlines() if ln.strip().startswith(("-", "•", "*"))]
    if len(bullet_lines) >= 2:
        return False
    if raw.count("---") >= 2:
        return False
    if re.match(r"^【本帧】\s*(必须|不要|如果用户|预算)", raw):
        return False
    return True


def normalize_shot_scene_text(text: str, *, fallback: str = "") -> str:
    """Pick a clean per-shot scene line; prefer structured beats over expand noise."""
    for candidate in (
        extract_keyframe_shot_scene(text),
        (text or "").strip(),
        (fallback or "").strip(),
    ):
        c = (candidate or "").strip()
        if c and is_valid_shot_scene_text(c):
            return c
    fb = (fallback or "").strip()
    if fb:
        return fb
    raw = extract_keyframe_shot_scene(text) or (text or "").strip()
    return raw[:240].strip()


def chapter_beats_ready_for_shots(beats: list[str]) -> bool:
    cleaned = [b.strip() for b in beats if b and b.strip()]
    if len(cleaned) < 2:
        return False
    return all(is_valid_shot_scene_text(b) for b in cleaned)


def resolve_shot_scene_for_index(
    *,
    index: int,
    raw_visual: str,
    beat_sheet: list[str],
    character_anchor: str = "",
) -> str:
    beat = beat_sheet[index].strip() if index < len(beat_sheet) else ""
    picked = _pick_shot_line(
        raw_visual,
        index,
        character_anchor=character_anchor,
        beat_sheet=beat_sheet,
    )
    return normalize_shot_scene_text(picked, fallback=beat)


KEYFRAME_REF_DIVIDER = "---"

_CHARACTER_BLOCK_ZH = re.compile(r"【角色·([^】]+)】\s*(.+)", re.S)
_LABELED_BLOCK_ZH = re.compile(r"【([^】]+)】\s*(.+)", re.S)
_CHARACTER_BLOCK_EN = re.compile(r"\[Character:\s*([^\]]+)\]\s*(.+)", re.I | re.S)
_STYLE_BLOCK_EN = re.compile(r"\[(Style|Look)\]\s*(.+)", re.I | re.S)
_SHOT_SCENE_ZH = re.compile(r"【本帧】\s*(.+)", re.S)
_SHOT_SCENE_EN = re.compile(r"\[Shot\]\s*(.+)", re.I | re.S)
_SHOT_SCENE_HEAD_ZH = re.compile(r"^【本帧】\s*(.+)", re.S)
_SHOT_SCENE_HEAD_EN = re.compile(r"^\[Shot\]\s*(.+)", re.I | re.S)
_STYLE_LABELS = frozenset({"画风", "风格", "style", "look", "palette", "film"})

_LEGACY_CHARACTER_CLAUSE_ZH = re.compile(
    r"^([\u4e00-\u9fffA-Za-z·]{2,10})[，,]\s*(.+)$",
    re.S,
)
_LEGACY_CHARACTER_CLAUSE_EN = re.compile(
    r"^([A-Za-z][A-Za-z\s'.-]{1,24})\s*[,，]\s*(.+)$",
    re.S,
)


def _parse_legacy_character_clause(raw: str) -> tuple[str, str] | None:
    """Legacy anchor line: ``赵今麦，白T恤，黑色短发`` → (name, body)."""
    text = (raw or "").strip()
    if not text or text.startswith("【") or text.startswith("["):
        return None
    m = _LEGACY_CHARACTER_CLAUSE_ZH.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _LEGACY_CHARACTER_CLAUSE_EN.match(text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def storyboard_anchor_format_rule(locale: str) -> str:
    loc = normalize_storyboard_locale(locale)
    if loc == "zh":
        return (
            "CRITICAL [Anchor] format (blocks separated by a line containing only ---):\n"
            "【角色·<姓名>·<装扮名>】<发型、服饰、体型>\n"
            "---\n"
            "【角色·<姓名>·<另一装扮>】<…>  (multiple looks per person allowed)\n"
            "---\n"
            "【画风】<全片镜头/色调>\n"
            "[Visual N] = scene-only; at T2I time code appends selected looks AFTER the scene (scene --- reference)."
        )
    return (
        "CRITICAL [Anchor] format (blocks separated by ---):\n"
        "[Character: <Name> | <Look>] <hair, wardrobe, build>\n"
        "---\n"
        "[Style] <shared palette/lens>\n"
        "[Visual N] = scene-only; selected looks are appended before T2I."
    )


def _split_anchor_raw_blocks(anchor: str) -> list[str]:
    text = (anchor or "").strip()
    if not text:
        return []
    if KEYFRAME_REF_DIVIDER in text:
        parts = [p.strip() for p in re.split(r"\n\s*---\s*\n", text) if p.strip()]
        if parts:
            return parts
    lines: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.strip() == KEYFRAME_REF_DIVIDER:
            if buf:
                lines.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        lines.append("\n".join(buf).strip())
    if len(lines) > 1:
        return lines
    parts = [p.strip() for p in re.split(r"[。\n；;]+", text) if p.strip()]
    return parts or [text]


def parse_anchor_blocks(character_anchor: str) -> list[tuple[str, str, str]]:
    """Parse anchor into (kind, name, body) tuples: kind = character | style | other."""
    blocks: list[tuple[str, str, str]] = []
    for raw in _split_anchor_raw_blocks(character_anchor):
        m = _CHARACTER_BLOCK_ZH.match(raw)
        if m:
            blocks.append(("character", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _CHARACTER_BLOCK_EN.match(raw)
        if m:
            blocks.append(("character", m.group(1).strip(), m.group(2).strip()))
            continue
        m = _LABELED_BLOCK_ZH.match(raw)
        if m:
            label = m.group(1).strip()
            body = m.group(2).strip()
            if label.startswith("角色·"):
                blocks.append(("character", label[3:].strip(), body))
            elif label.lower() in _STYLE_LABELS or label in ("画风", "风格"):
                blocks.append(("style", label, body))
            else:
                blocks.append(("character", label, body))
            continue
        m = _STYLE_BLOCK_EN.match(raw)
        if m:
            blocks.append(("style", m.group(1).strip(), m.group(2).strip()))
            continue
        legacy = _parse_legacy_character_clause(raw)
        if legacy:
            name, body = legacy
            if name and body:
                blocks.append(("character", name, body))
            continue
        blocks.append(("other", "", raw))
    return blocks


def normalize_character_anchor(character_anchor: str, *, locale: str | None = None) -> str:
    """Normalize Plan [Anchor] to --- separated character/style blocks for T2I reference."""
    anchor = (character_anchor or "").strip()
    if not anchor:
        return ""
    if KEYFRAME_REF_DIVIDER in anchor and parse_anchor_blocks(anchor):
        return anchor
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(anchor) == "zh" else "en"
    )
    blocks = parse_anchor_blocks(anchor)
    if not blocks:
        return anchor
    return _format_reference_blocks(blocks, loc)


def _format_reference_blocks(blocks: list[tuple[str, str, str]], locale: str) -> str:
    lines: list[str] = []
    loc = normalize_storyboard_locale(locale)
    for kind, name, body in blocks:
        if kind == "style":
            if loc == "zh":
                lines.append(f"【画风】{body}")
            else:
                lines.append(f"[Style] {body}")
        elif kind == "character":
            if loc == "zh":
                lines.append(f"【角色·{name}】{body}")
            else:
                lines.append(f"[Character: {name}] {body}")
        elif body:
            lines.append(body)
    return f"\n{KEYFRAME_REF_DIVIDER}\n".join(lines)


def anchor_blocks_for_visual(visual: str, character_anchor: str) -> list[tuple[str, str, str]]:
    """Anchor reference blocks for on-screen cast + shared style."""
    scene = extract_keyframe_shot_scene(visual)
    blocks = parse_anchor_blocks(character_anchor)
    if not blocks:
        return []
    if not scene.strip():
        return blocks
    matched: list[tuple[str, str, str]] = []
    style_blocks: list[tuple[str, str, str]] = []
    other: list[tuple[str, str, str]] = []
    for kind, name, body in blocks:
        if kind == "style":
            style_blocks.append((kind, name, body))
        elif kind == "character" and name and name in scene:
            matched.append((kind, name, body))
        elif kind == "other":
            other.append((kind, name, body))
    if matched:
        return matched + style_blocks + other
    characters = [b for b in blocks if b[0] == "character"]
    if characters:
        return characters + style_blocks + other
    return blocks


def extract_keyframe_shot_scene(visual: str) -> str:
    """Return the per-shot scene portion (without reference blocks)."""
    v = (visual or "").strip()
    if not v:
        return ""
    head = _SHOT_SCENE_HEAD_ZH.match(v) or _SHOT_SCENE_HEAD_EN.match(v)
    if head:
        part = head.group(1).strip()
        if KEYFRAME_REF_DIVIDER in part:
            part = part.split(KEYFRAME_REF_DIVIDER, 1)[0].strip()
        return part
    if KEYFRAME_REF_DIVIDER in v:
        tail = v.split(KEYFRAME_REF_DIVIDER)[-1].strip()
        m = _SHOT_SCENE_ZH.search(tail) or _SHOT_SCENE_EN.search(tail)
        if m:
            return m.group(1).strip()
        if not re.search(r"【角色·|\[Character:", tail, re.I):
            return tail
    return v


def join_keyframe_prompt(scene: str, ref: str, *, locale: str) -> str:
    """Scene-first T2I prompt; cast/style reference blocks appended after ---."""
    scene = (scene or "").strip()
    ref = (ref or "").strip()
    if not scene:
        return ref
    if not ref:
        return scene
    loc = normalize_storyboard_locale(locale)
    shot_label = "【本帧】" if loc == "zh" else "[Shot] "
    if scene.startswith("【本帧】") or re.match(r"^\[Shot\]", scene, re.I):
        body = scene
    else:
        body = f"{shot_label}{scene}"
    return f"{body}\n{KEYFRAME_REF_DIVIDER}\n{ref}"


def is_structured_keyframe_visual(visual: str) -> bool:
    v = (visual or "").strip()
    return KEYFRAME_REF_DIVIDER in v and bool(_SHOT_SCENE_ZH.search(v) or _SHOT_SCENE_EN.search(v))


def compose_keyframe_visual_prompt(
    visual: str,
    character_anchor: str,
    *,
    locale: str | None = None,
) -> str:
    """Build T2I prompt: 【本帧】 scene first; cast/style reference appended after ---."""
    scene = extract_keyframe_shot_scene(visual).strip()
    anchor = (character_anchor or "").strip()
    if not scene and not anchor:
        return ""
    if not anchor:
        return scene or visual.strip()
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(scene or anchor) == "zh" else "en"
    )
    normalized_anchor = normalize_character_anchor(anchor, locale=loc)
    blocks = anchor_blocks_for_visual(scene or visual, normalized_anchor)
    if not blocks:
        blocks = parse_anchor_blocks(normalized_anchor)
    if not scene:
        return _format_reference_blocks(blocks, loc)
    if is_structured_keyframe_visual(visual) and visual_includes_anchor_appearance(visual, normalized_anchor):
        return visual.strip()
    ref = _format_reference_blocks(blocks, loc)
    return join_keyframe_prompt(scene, ref, locale=loc)


_APPEARANCE_STOPWORDS = frozenset(
    {
        "近景",
        "远景",
        "广角",
        "中景",
        "特写",
        "镜头",
        "固定",
        "推近",
        "跟拍",
        "手持",
        "缓慢",
        "侧移",
        "环绕",
        "快切",
        "低角度",
        "背景",
        "环境",
        "氛围",
        "画面",
        "场景",
        "写实",
        "电影感",
        "浅景深",
        "background",
        "lighting",
        "medium",
        "close",
        "wide",
        "dolly",
        "camera",
        "shot",
        "frame",
        "motion",
        "slow",
    }
)

_STYLE_LEAD_WORDS = frozenset(
    {"现代", "写实", "赛博", "古装", "硬科幻", "全局", "风格", "palette", "style", "film", "look", "noir"}
)


def _appearance_keywords(text: str) -> set[str]:
    """Tokens likely describing wardrobe/hair/body (not camera/scene)."""
    tokens: set[str] = set()
    for m in re.finditer(r"[\u4e00-\u9fff]{2,6}", text or ""):
        t = m.group(0)
        if t not in _APPEARANCE_STOPWORDS:
            tokens.add(t)
    for m in re.finditer(r"[a-zA-Z]{4,}", text or ""):
        w = m.group(0).lower()
        if w not in _APPEARANCE_STOPWORDS:
            tokens.add(w)
    return tokens


def _anchor_clauses(character_anchor: str) -> list[str]:
    anchor = (character_anchor or "").strip()
    if not anchor:
        return []
    parts = [p.strip() for p in re.split(r"[。\n；;]+", anchor) if p.strip()]
    return parts or [anchor]


def anchor_clauses_for_visual(visual: str, character_anchor: str) -> str:
    """Anchor lines for characters named in *visual*, plus shared style lines."""
    visual = (visual or "").strip()
    anchor = (character_anchor or "").strip()
    if not anchor:
        return ""
    clauses = _anchor_clauses(anchor)
    if not visual:
        return anchor
    matched: list[str] = []
    global_parts: list[str] = []
    for part in clauses:
        name_m = re.match(r"^([^，,]{1,16})[，,]", part)
        if not name_m:
            global_parts.append(part)
            continue
        name = name_m.group(1).strip()
        if name in _STYLE_LEAD_WORDS or len(name) > 10:
            global_parts.append(part)
            continue
        if name in visual:
            matched.append(part)
    if matched:
        merged = matched + global_parts
        return "，".join(merged)
    return anchor


def visual_includes_anchor_appearance(
    visual: str,
    character_anchor: str,
    *,
    min_ratio: float = 0.35,
) -> bool:
    """True when *visual* already carries enough wardrobe/hair tokens from anchor."""
    v = (visual or "").strip()
    a = (character_anchor or "").strip()
    if not v or not a:
        return bool(v)
    normalized = normalize_character_anchor(a)
    if is_structured_keyframe_visual(v):
        ref = v.rsplit(KEYFRAME_REF_DIVIDER, 1)[0]
        anchor_kw = _appearance_keywords(normalized)
        if not anchor_kw:
            return True
        ref_kw = _appearance_keywords(ref)
        if len(anchor_kw & ref_kw) / len(anchor_kw) >= min_ratio:
            return True
    scene = extract_keyframe_shot_scene(v) or v
    blocks = anchor_blocks_for_visual(scene, normalized)
    bodies = " ".join(body for _k, _n, body in blocks)
    anchor_kw = _appearance_keywords(bodies or normalized)
    if not anchor_kw:
        return True
    overlap = anchor_kw & _appearance_keywords(v)
    return len(overlap) / len(anchor_kw) >= min_ratio


def merge_visual_with_character_anchor(
    visual: str,
    character_anchor: str,
    *,
    locale: str | None = None,
) -> str:
    """Append --- separated Anchor reference + 【本帧】 scene for T2I."""
    return compose_keyframe_visual_prompt(visual, character_anchor, locale=locale)


def apply_storyboard_appearance_lock(
    shots: list[dict[str, str | int]],
    *,
    character_anchor: str,
) -> list[dict[str, str | int]]:
    """Ensure every keyframe visual carries shared wardrobe/hair from [Anchor]."""
    anchor = (character_anchor or "").strip()
    if not anchor:
        return shots
    locked: list[dict[str, str | int]] = []
    for shot in shots:
        visual = str(shot.get("visual_prompt", "")).strip()
        locked.append(
            {
                **shot,
                "visual_prompt": merge_visual_with_character_anchor(visual, anchor),
            }
        )
    return locked


def storyboard_visuals_include_appearance(
    pairs: list[tuple[str, str]],
    character_anchor: str,
    shot_count: int,
    *,
    characters: list[dict] | None = None,
    style_anchor: str = "",
    beat_sheet: list[str] | None = None,
    locale: str | None = None,
) -> bool:
    anchor = (character_anchor or "").strip()
    if not anchor or shot_count <= 0:
        return True
    from backend.engine.llm.storyboard_cast import compose_keyframe_with_cast, dtos_to_cast_looks, dtos_to_roster, infer_shot_cast_looks

    roster = dtos_to_roster(characters) if characters else []
    beats = beat_sheet or []
    for i, (visual, _) in enumerate(pairs[:shot_count]):
        scene = extract_keyframe_shot_scene(visual) or visual
        if roster:
            beat = beats[i] if i < len(beats) else scene
            cast = infer_shot_cast_looks(scene=scene, beat=beat, characters=roster)
            composed = compose_keyframe_with_cast(
                scene,
                characters=roster,
                cast=cast,
                style_anchor=style_anchor,
                locale=locale,
                character_anchor=anchor,
            )
        else:
            composed = compose_keyframe_visual_prompt(visual, anchor, locale=locale)
        if not is_structured_keyframe_visual(composed):
            return False
        if not visual_includes_anchor_appearance(composed, anchor):
            return False
    return True


def build_structured_shots(
    *,
    character_anchor: str,
    opening_prompt: str,
    segment_prompts: list[str],
    beat_sheet: list[str],
    target_duration_sec: float,
    segment_duration_sec: float,
    dual_pairs: list[tuple[str, str]] | None = None,
    characters: list[dict] | None = None,
    scenes: list[dict] | None = None,
    style_anchor: str = "",
    locale: str | None = None,
    shot_plan: ShotPlan | None = None,
) -> list[dict[str, str | int | float]]:
    plan = shot_plan or build_shot_plan(
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        beat_texts=beat_sheet[: max(len(beat_sheet), 1)],
    )
    durations = plan.segment_durations_sec
    shots: list[dict[str, str | int | float]] = []
    pairs = list(dual_pairs or [])
    if pairs:
        pairs = coalesce_dual_pairs(
            pairs,
            beat_sheet,
            plan.shot_count,
            character_anchor=character_anchor,
        )
    for i in range(plan.shot_count):
        visual = ""
        motion = ""
        if pairs and i < len(pairs):
            visual, motion = pairs[i]
            visual = resolve_shot_scene_for_index(
                index=i,
                raw_visual=visual,
                beat_sheet=beat_sheet,
                character_anchor=character_anchor,
            )
            motion = _pick_shot_line(
                motion, i, character_anchor=character_anchor, beat_sheet=beat_sheet
            )
            if not is_valid_shot_scene_text(motion) or motion.strip() == visual.strip():
                from backend.engine.llm.motion_prompt import motion_prompt_from_beat

                nxt = beat_sheet[i + 1] if i + 1 < len(beat_sheet) else visual
                motion = motion_prompt_from_beat(visual, beat=beat_sheet[i] if i < len(beat_sheet) else visual, next_visual=nxt, locale=locale)
        elif i < len(beat_sheet):
            visual = normalize_shot_scene_text(beat_sheet[i], fallback=beat_sheet[i])
            from backend.engine.llm.motion_prompt import motion_prompt_from_beat

            nxt = beat_sheet[i + 1] if i + 1 < len(beat_sheet) else beat_sheet[i]
            motion = motion_prompt_from_beat(
                visual,
                beat=beat_sheet[i],
                next_visual=nxt,
                locale=locale,
            )
        elif i < len(segment_prompts):
            motion = segment_prompts[i]
            visual = motion
        visual = _strip_beat_markers(visual)
        if _contains_beat_markers(visual) or len(_split_beat_marked_lines(visual)) > 1:
            if i < len(beat_sheet):
                visual = normalize_shot_scene_text(beat_sheet[i], fallback=beat_sheet[i])
        if _looks_like_motion_only(visual) and i < len(beat_sheet):
            visual = normalize_shot_scene_text(beat_sheet[i], fallback=beat_sheet[i])
        if not is_valid_shot_scene_text(visual) and i < len(beat_sheet):
            visual = normalize_shot_scene_text(beat_sheet[i], fallback=beat_sheet[i])
        shots.append(
            {
                "id": f"shot_{i:02d}",
                "order": i,
                "visual_prompt": visual.strip(),
                "motion_prompt": (motion or visual).strip(),
                "scene_prompt": visual.strip(),
                "duration_sec": float(durations[i]) if i < len(durations) else float(segment_duration_sec),
            }
        )
    return _apply_cast_and_appearance_lock(
        shots,
        character_anchor=character_anchor,
        beat_sheet=beat_sheet,
        characters=characters,
        scenes=scenes,
        style_anchor=style_anchor,
        locale=locale,
    )


def _apply_cast_and_appearance_lock(
    shots: list[dict[str, str | int]],
    *,
    character_anchor: str,
    beat_sheet: list[str],
    characters: list[dict] | None = None,
    scenes: list[dict] | None = None,
    style_anchor: str = "",
    locale: str | None = None,
) -> list[dict[str, str | int | list | dict | None]]:
    from backend.engine.llm.storyboard_cast import (
        cast_looks_to_dtos,
        compose_keyframe_with_cast,
        dtos_to_roster,
        infer_shot_cast_looks,
        parse_character_roster,
    )
    from backend.engine.llm.storyboard_scenes import (
        dtos_to_roster as dtos_to_scene_roster,
        infer_shot_scene_look,
        scene_look_to_dtos,
    )

    roster = dtos_to_roster(characters) if characters else parse_character_roster(character_anchor, locale=locale)[0]
    scene_roster = dtos_to_scene_roster(scenes) if scenes else []
    style = (style_anchor or "").strip()
    if not style and character_anchor.strip():
        style = parse_character_roster(character_anchor, locale=locale)[1]
    if not roster and not scene_roster:
        return apply_storyboard_appearance_lock(shots, character_anchor=character_anchor)

    prev_cast = None
    prev_scene = None
    locked: list[dict[str, str | int | list | dict | None]] = []
    for i, shot in enumerate(shots):
        raw_visual = str(shot.get("visual_prompt", "")).strip()
        beat = beat_sheet[i].strip() if i < len(beat_sheet) else ""
        scene = normalize_shot_scene_text(raw_visual, fallback=beat)
        if not scene and beat:
            scene = beat
        cast = infer_shot_cast_looks(scene=scene, beat=beat or scene, characters=roster, prev=prev_cast)
        prev_cast = cast
        scene_binding = infer_shot_scene_look(beat=beat or scene, scenes=scene_roster, prev=prev_scene)
        prev_scene = scene_binding
        locked.append(
            {
                **shot,
                "scene_prompt": scene,
                "cast_looks": cast_looks_to_dtos(cast),
                "scene_look": scene_look_to_dtos(scene_binding),
                "visual_prompt": scene,
                "motion_prompt": str(shot.get("motion_prompt", "")).strip() or scene,
            }
        )
    return locked


def normalize_storyboard_locale(locale: str | None) -> str:
    loc = (locale or "").strip().lower().split("-")[0]
    return loc if loc in ("zh", "en") else "zh"


def prompt_locale(text: str) -> str:
    """Rough zh/en classifier for storyboard prompt lines."""
    raw = (text or "").strip()
    if not raw:
        return "mixed"
    cn = len(re.findall(r"[\u4e00-\u9fff]", raw))
    en = len(re.findall(r"[a-zA-Z]{3,}", raw))
    if cn >= 4 and cn >= en:
        return "zh"
    if en >= 2 and en > cn:
        return "en"
    if en >= 1 and cn == 0:
        return "en"
    return "mixed"


def storyboard_language_rule(locale: str) -> str:
    """Deprecated for system prompts — use ``storyboard_user_locale_block`` in user messages."""
    loc = normalize_storyboard_locale(locale)
    pronoun_rule = (
        "CRITICAL: Each [Visual]/[Motion]/[Beat] is sent to image/video models alone — "
        "name every character explicitly using proper names from the brief. "
        "Never start with or rely on standalone pronouns (她/他/they/she/he)."
    )
    anchor_fmt = storyboard_anchor_format_rule(loc)
    if loc == "zh":
        return (
            "CRITICAL: All [Anchor], [Beat], [Visual], and [Motion] text MUST be Simplified Chinese (简体中文) only. "
            "Do not write English sentences; keep character names in Chinese script as in the brief (do not romanize).\n"
            + pronoun_rule
            + "\n"
            + anchor_fmt
        )
    return "CRITICAL: All output text MUST be in English only.\n" + pronoun_rule + "\n" + anchor_fmt


def storyboard_language_user_suffix(locale: str) -> str:
    loc = normalize_storyboard_locale(locale)
    if loc == "zh":
        return (
            "\n\nOutput language: Simplified Chinese (简体中文) ONLY for every [Anchor], [Beat], [Visual], and [Motion] line."
        )
    return "\n\nOutput language: English ONLY for every labeled block."


def apply_storyboard_output_locale(
    shots: list[dict[str, str | int]],
    *,
    beat_sheet: list[str],
    locale: str,
) -> list[dict[str, str | int]]:
    """When Expand mixes languages, prefer Plan beats that match the configured locale."""
    loc = normalize_storyboard_locale(locale)
    fixed: list[dict[str, str | int]] = []
    for i, shot in enumerate(shots):
        visual = str(shot.get("visual_prompt", "")).strip()
        motion = str(shot.get("motion_prompt", "")).strip()
        beat = beat_sheet[i].strip() if i < len(beat_sheet) else ""
        nxt = beat_sheet[i + 1].strip() if i + 1 < len(beat_sheet) else ""
        if loc == "zh":
            if prompt_locale(visual) == "en" and prompt_locale(beat) == "zh":
                visual = beat
            if prompt_locale(motion) == "en" and prompt_locale(nxt) == "zh":
                motion = nxt
            elif prompt_locale(motion) == "en" and prompt_locale(beat) == "zh" and not nxt:
                motion = beat
        elif loc == "en":
            if prompt_locale(visual) == "zh" and prompt_locale(beat) == "en":
                visual = beat
            if prompt_locale(motion) == "zh" and prompt_locale(nxt) == "en":
                motion = nxt
        fixed.append({**shot, "visual_prompt": visual, "motion_prompt": motion or visual})
    return fixed


def apply_storyboard_anchor_locale(character_anchor: str, *, beat_sheet: list[str], locale: str) -> str:
    anchor = (character_anchor or "").strip()
    if not anchor or not beat_sheet:
        return anchor
    loc = normalize_storyboard_locale(locale)
    if loc == "zh" and prompt_locale(anchor) == "en":
        for beat in beat_sheet:
            if prompt_locale(beat) == "zh" and len(beat) >= 8:
                return beat
    if loc == "en" and prompt_locale(anchor) == "zh":
        for beat in beat_sheet:
            if prompt_locale(beat) == "en" and len(beat) >= 8:
                return beat
    return anchor


_STANDALONE_PRONOUN_LEAD = re.compile(r"^(她|他|他们|她们|它)", re.I)
_STANDALONE_PRONOUN_LEAD_EN = re.compile(r"^(She|He|They|It)\b", re.I)


def prompt_leads_with_standalone_pronoun(text: str) -> bool:
    """True when a prompt starts with a pronoun — unsafe for isolated T2I/I2V."""
    s = (text or "").strip()
    if not s:
        return False
    return bool(_STANDALONE_PRONOUN_LEAD.match(s) or _STANDALONE_PRONOUN_LEAD_EN.match(s))


def storyboard_prompts_self_contained(pairs: list[tuple[str, str]], shot_count: int) -> bool:
    for visual, motion in pairs[:shot_count]:
        if prompt_leads_with_standalone_pronoun(visual) or prompt_leads_with_standalone_pronoun(motion):
            return False
    return True


def expand_batches_for_shot_count(shot_count: int) -> list[tuple[int, int]]:
    """Batch indices for segmented I2V shot prompts."""
    n = max(0, int(shot_count))
    if n <= 0:
        return []
    batches: list[tuple[int, int]] = []
    i = 0
    while i < n:
        take = min(EXPAND_BATCH_SIZE, n - i)
        batches.append((i, take))
        i += take
    return batches


def expand_batches_for_plan(plan: LongVideoPlan) -> list[tuple[int, int]]:
    """Return (start_idx, count) batches for extend segments (exclude pass0)."""
    n = plan.extend_pass_count
    if n <= 0:
        return []
    batches: list[tuple[int, int]] = []
    i = 0
    while i < n:
        take = min(EXPAND_BATCH_SIZE, n - i)
        batches.append((i, take))
        i += take
    return batches
