"""Pass 2 — beat plan (visibility + segments + spatial in one LLM pass)."""
from __future__ import annotations

from typing import Any, Callable

from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
from backend.engine.llm.script_parse.prompts import BEAT_PLAN_SYSTEM
from backend.engine.llm.script_parse.review import invoke_pass_with_review, validate_pydantic_json
from backend.long_video.keyframe_prompt_policy import validate_beat_plan_row_contract
from backend.engine.llm.script_parse.rules import expand_beat_plan_rows
from backend.engine.llm.script_parse.schemas import (
    BeatPlanArtifact,
    BeatPlanLLMSchema,
    ScriptArtifact,
)
from backend.engine.llm.storyboard import normalize_storyboard_locale
from backend.engine.llm.storyboard_cast import dtos_to_roster, format_character_roster

ProgressFn = Callable[[str, str], None]


def _sanitize_beat_plan_payload(payload: BeatPlanLLMSchema, script: ScriptArtifact) -> BeatPlanLLMSchema:
    """Deterministic LLM output normalization before rule validation."""
    from backend.engine.llm.script_parse.schemas import BeatPlanRowLLM, BeatPlanSegmentLLM, sanitize_segment_role

    protagonist = next((c.name for c in script.characters if c.role == "protagonist"), "")
    beat_rows: list[BeatPlanRowLLM] = []
    for row in payload.beats:
        segments: list[BeatPlanSegmentLLM] = []
        face_seen = False
        for seg in row.segments:
            role = sanitize_segment_role(seg.role)
            if role == "face_anchor":
                if face_seen:
                    role = "post_anchor"
                else:
                    face_seen = True
                    names = [str(n).strip() for n in seg.characters_on_screen if str(n).strip()]
                    if len(names) > 1:
                        pick = protagonist if protagonist in names else names[0]
                        names = [pick]
                    seg = seg.model_copy(update={"characters_on_screen": names})
            if role != seg.role:
                seg = seg.model_copy(update={"role": role})
            segments.append(seg)
        beat_rows.append(row.model_copy(update={"segments": segments}))
    return payload.model_copy(update={"beats": beat_rows})


def _beat_plan_user_message(
    script: ScriptArtifact,
    *,
    target_duration_sec: float,
    segment_duration_sec: float,
    locale_block: str,
    beat_indices: list[int] | None = None,
) -> str:
    rows: list[str] = []
    cast_names = [c.name for c in script.characters]
    allowed = {int(i) for i in beat_indices} if beat_indices else None
    beats = [b for b in script.beats if allowed is None or b.index in allowed]
    for beat in beats:
        rows.append(
            f"[{beat.index}] title={beat.title}\n"
            f"location={beat.location}\n"
            f"shot_size={beat.suggested_shot_size}\n"
            f"estimated_duration_sec={beat.estimated_duration_sec}\n"
            f"narrative={beat.narrative}\n"
            f"enhancement_cues={'; '.join(beat.enhancement_cues)}"
        )
    roster = dtos_to_roster([c.model_dump() for c in script.characters])
    anchor = format_character_roster(roster, script.style_anchor, locale="zh")
    return (
        f"Synopsis:\n{script.synopsis}\n\n"
        f"Mood: {script.mood}\n"
        f"Target total duration: {target_duration_sec}s\n"
        f"Default segment duration hint: {segment_duration_sec}s\n"
        f"Cast: {', '.join(cast_names)}\n\n"
        f"Character anchor:\n{anchor}\n\n"
        f"Beats:\n" + "\n\n".join(rows) + "\n\n"
        f"Scenes:\n"
        + "\n".join(f"- {s.name}: {s.looks[0].body if s.looks else ''}" for s in script.scenes)
        + f"\n{locale_block}"
    )


def _validate_beat_plan(
    payload: BeatPlanLLMSchema,
    script: ScriptArtifact,
    *,
    beat_indices: list[int] | None = None,
) -> tuple[bool, str]:
    issues: list[str] = []
    if beat_indices:
        expected = {int(i) for i in beat_indices}
    else:
        expected = {b.index for b in script.beats}
    got = {r.beat_index for r in payload.beats}
    missing = expected - got
    if missing:
        issues.append(f"missing beat_index rows: {sorted(missing)}")
    for row in payload.beats:
        if not row.segments:
            issues.append(f"beat {row.beat_index}: no segments")
            continue
        if row.beat_index == 0:
            for seg in row.segments:
                if seg.characters_on_screen and seg.start_visibility == "invisible" and not seg.is_intentional_empty:
                    issues.append(f"beat 0 segment: characters on screen but start_visibility invisible")
        reach = any(s.reachability == "identity_critical" for s in row.segments)
        has_face = any(s.role == "face_anchor" for s in row.segments)
        if reach and not has_face:
            issues.append(f"beat {row.beat_index}: identity_critical requires face_anchor segment")
        issues.extend(
            validate_beat_plan_row_contract(beat_index=row.beat_index, segments=row.segments)
        )
    if issues:
        return False, "\n".join(issues[:12])
    return True, ""


def run_beat_plan(
    *,
    script: ScriptArtifact,
    target_duration_sec: float,
    segment_duration_sec: float,
    max_clip_sec: float,
    locale: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
    beat_indices: list[int] | None = None,
) -> tuple[BeatPlanArtifact, int]:
    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)
    user = _beat_plan_user_message(
        script,
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        locale_block=locale_block,
        beat_indices=beat_indices,
    )

    def validate(resp: str) -> tuple[bool, str]:
        ok, fb, payload = validate_pydantic_json(resp, BeatPlanLLMSchema)
        if not ok or payload is None:
            return ok, fb
        payload = _sanitize_beat_plan_payload(payload, script)
        return _validate_beat_plan(payload, script, beat_indices=beat_indices)

    if on_progress:
        on_progress("beat_plan", "beat_plan")

    resp, llm_calls = invoke_pass_with_review(
        chat_fn,
        system=BEAT_PLAN_SYSTEM,
        user=user,
        max_tokens=token_budget(4000),
        think_apply=think_apply,
        validate=validate,
        max_attempts=2,
        pass_name="beat_plan",
        on_progress=on_progress,
    )
    ok, fb, payload = validate_pydantic_json(resp, BeatPlanLLMSchema)
    if not ok or payload is None:
        raise ValueError(f"beat_plan validation failed: {fb}")
    payload = _sanitize_beat_plan_payload(payload, script)
    ok2, fb2 = _validate_beat_plan(payload, script, beat_indices=beat_indices)
    if not ok2:
        raise ValueError(f"beat_plan rule validation failed: {fb2}")

    rows = payload.beats
    if beat_indices:
        allowed = {int(i) for i in beat_indices}
        rows = [r for r in rows if r.beat_index in allowed]
    plan = expand_beat_plan_rows(script, rows, max_clip_sec=max_clip_sec)
    return plan, llm_calls
