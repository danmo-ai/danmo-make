"""Pass 3 — unified shot spec (5-Aspect + video + start_visual)."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.llm_retry import invoke_text_chat_with_feedback
from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
from backend.engine.llm.script_parse.prompts import SHOT_SPEC_SYSTEM
from backend.engine.llm.script_parse.review import invoke_pass_with_review, validate_pydantic_json
from backend.engine.llm.script_parse.rules import flatten_planned_segments
from backend.engine.llm.script_parse.schemas import (
    BeatPlanArtifact,
    PlannedSegmentArtifact,
    ScriptArtifact,
    ShotSpecArtifact,
    ShotSpecBatchLLMSchema,
    ShotSpecRowLLM,
)
from backend.engine.llm.storyboard import normalize_storyboard_locale
from backend.engine.llm.storyboard_cast import strip_name_look_tags
from backend.long_video.keyframe_prompt_policy import (
    coalesce_face_anchor_visual,
    sanitize_shot_spec_prompts,
    validate_shot_spec_partial_framing,
)

ProgressFn = Callable[[str, str], None]

SHOT_SPEC_BATCH = 6


def _segment_context(seg: PlannedSegmentArtifact, script: ScriptArtifact) -> str:
    beat = next((b for b in script.beats if b.index == seg.beat_index), None)
    title = beat.title if beat else ""
    loc = beat.location if beat else ""
    return (
        f"[{seg.segment_index}] beat={seg.beat_index} role={seg.role} duration={seg.duration_sec}s\n"
        f"title={title}\nlocation={loc or seg.spatial.location if seg.spatial else ''}\n"
        f"shot_size={seg.shot_size}\n"
        f"characters_on_screen={list(seg.characters_on_screen)}\n"
        f"start_visibility={seg.start_visibility} end_visibility={seg.end_visibility}\n"
        f"is_intentional_empty={seg.is_intentional_empty}\n"
        f"first_frame_requirement={seg.first_frame_requirement or '(none)'}\n"
        f"reachability={seg.reachability}"
    )


def _build_user_batch(
    segments: list[PlannedSegmentArtifact],
    script: ScriptArtifact,
    locale_block: str,
) -> str:
    header = (
        f"Synopsis: {script.synopsis}\n"
        f"Mood: {script.mood}\n"
        f"Style anchor (do NOT paste into video_prompt): {script.style_anchor}\n\n"
        f"Derive shot specs for these segment_index values ONLY:\n\n"
    )
    body = "\n\n".join(_segment_context(s, script) for s in segments)
    indices = ", ".join(str(s.segment_index) for s in segments)
    return header + body + f"\n\nRequired segment_index values: {indices}\n{locale_block}"


def _rows_to_specs(
    rows: list[ShotSpecRowLLM],
    seg_by_index: dict[int, PlannedSegmentArtifact],
) -> list[ShotSpecArtifact]:
    out: list[ShotSpecArtifact] = []
    for row in rows:
        seg = seg_by_index.get(row.segment_index)
        if seg is None:
            raise ValueError(f"shot_spec unknown segment_index {row.segment_index}")
        start_visual, anchor_visual = sanitize_shot_spec_prompts(
            role=str(seg.role),
            start_visual=strip_name_look_tags(row.start_visual),
            anchor_visual=strip_name_look_tags(row.anchor_visual),
        )
        if str(seg.role) == "face_anchor":
            primary = (seg.characters_on_screen[0] if seg.characters_on_screen else "").strip()
            shot_size = (row.shot_language.shot_size or seg.shot_size or "").strip()
            start_visual, anchor_visual = coalesce_face_anchor_visual(
                anchor_visual=anchor_visual,
                start_visual=start_visual,
                five_aspect_subject=strip_name_look_tags(row.five_aspect.subject),
                first_frame_requirement=(seg.first_frame_requirement or "").strip(),
                shot_size=shot_size,
                primary_name=primary,
            )
        partial_issues = validate_shot_spec_partial_framing(
            start_visibility=str(seg.start_visibility),
            start_visual=start_visual,
            segment_index=row.segment_index,
        )
        if partial_issues:
            raise ValueError(partial_issues[0])
        spec = ShotSpecArtifact(
            segment_index=row.segment_index,
            beat_index=seg.beat_index,
            role=seg.role,
            five_aspect=row.five_aspect,
            shot_language=row.shot_language,
            shot_intent=row.shot_intent,
            narrative_role="",
            video_prompt=strip_name_look_tags(row.video_prompt),
            start_visual=start_visual,
            anchor_visual=anchor_visual,
            characters_on_screen=list(seg.characters_on_screen),
            is_intentional_empty=seg.is_intentional_empty,
            start_visibility=seg.start_visibility,
            end_visibility=seg.end_visibility,
            duration_sec=seg.duration_sec,
            first_frame_requirement=seg.first_frame_requirement,
            location=seg.spatial.location if seg.spatial else "",
            start_frame_mode=seg.start_frame_mode,
            segment_group_id=seg.segment_group_id,
            segment_group_index=seg.segment_group_index,
            face_anchor_shot_id=seg.face_anchor_shot_id,
            camera_zone_id=seg.spatial.camera_zones[0].id if seg.spatial and seg.spatial.camera_zones else "",
        )
        out.append(spec)
    return out


def _validate_batch(
    payload: ShotSpecBatchLLMSchema,
    expected_indices: set[int],
    seg_by_index: dict[int, PlannedSegmentArtifact],
) -> tuple[bool, str]:
    got = {r.segment_index for r in payload.shots}
    missing = expected_indices - got
    extra = got - expected_indices
    issues: list[str] = []
    if missing:
        issues.append(f"missing segment_index: {sorted(missing)}")
    if extra:
        issues.append(f"unexpected segment_index: {sorted(extra)}")
    try:
        _rows_to_specs(payload.shots, seg_by_index)
    except (ValidationError, ValueError) as exc:
        issues.append(str(exc))
    if issues:
        return False, "\n".join(issues)
    return True, ""


def _split_valid_specs(
    payload: ShotSpecBatchLLMSchema | None,
    expected: set[int],
    seg_by_index: dict[int, PlannedSegmentArtifact],
) -> tuple[list[ShotSpecArtifact], set[int]]:
    """Return specs that passed row validation and indices still failing."""
    if payload is None:
        return [], set(expected)
    row_by_idx = {r.segment_index: r for r in payload.shots}
    ok_specs: list[ShotSpecArtifact] = []
    failed = set(expected)
    for idx in sorted(expected):
        row = row_by_idx.get(idx)
        if row is None:
            continue
        try:
            ok_specs.extend(_rows_to_specs([row], seg_by_index))
            failed.discard(idx)
        except (ValidationError, ValueError):
            continue
    return ok_specs, failed


def _invoke_llm_once(
    chat_fn: Callable[..., Any],
    *,
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
) -> tuple[str, int]:
    def _noop(_resp: str) -> tuple[bool, str]:
        return True, ""

    return invoke_text_chat_with_feedback(
        chat_fn,
        system=system,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
        validate=_noop,
        max_attempts=1,
    )


def _retry_failed_indices(
    failed: set[int],
    *,
    seg_by_index: dict[int, PlannedSegmentArtifact],
    script: ScriptArtifact,
    locale_block: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None,
) -> tuple[list[ShotSpecArtifact], int]:
    specs: list[ShotSpecArtifact] = []
    calls = 0
    for idx in sorted(failed):
        seg = seg_by_index.get(idx)
        if seg is None:
            continue
        user = _build_user_batch([seg], script, locale_block)

        def validate(resp: str) -> tuple[bool, str]:
            ok, fb, payload = validate_pydantic_json(resp, ShotSpecBatchLLMSchema)
            if not ok or payload is None:
                return ok, fb
            return _validate_batch(payload, {idx}, seg_by_index)

        resp, n = invoke_pass_with_review(
            chat_fn,
            system=SHOT_SPEC_SYSTEM,
            user=user,
            max_tokens=token_budget(1600),
            think_apply=think_apply,
            validate=validate,
            max_attempts=2,
            pass_name="shot_spec",
            on_progress=on_progress,
        )
        calls += n
        ok, _, payload = validate_pydantic_json(resp, ShotSpecBatchLLMSchema)
        if not ok or payload is None:
            continue
        sub_specs, still_failed = _split_valid_specs(payload, {idx}, seg_by_index)
        if still_failed:
            continue
        specs.extend(sub_specs)
    return specs, calls


def _invoke_batch(
    batch: list[PlannedSegmentArtifact],
    *,
    script: ScriptArtifact,
    locale_block: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None,
) -> tuple[list[ShotSpecArtifact], int]:
    seg_by_index = {s.segment_index: s for s in batch}
    expected = set(seg_by_index.keys())
    user = _build_user_batch(batch, script, locale_block)

    resp, calls = _invoke_llm_once(
        chat_fn,
        system=SHOT_SPEC_SYSTEM,
        user=user,
        max_tokens=token_budget(3200),
        think_apply=think_apply,
    )
    ok, fb, payload = validate_pydantic_json(resp, ShotSpecBatchLLMSchema)
    specs: list[ShotSpecArtifact] = []
    failed = set(expected)

    if ok and payload is not None:
        batch_ok, batch_fb = _validate_batch(payload, expected, seg_by_index)
        if batch_ok:
            specs = _rows_to_specs(payload.shots, seg_by_index)
            failed = set()
        else:
            specs, failed = _split_valid_specs(payload, expected, seg_by_index)
            if failed and specs:
                fb = batch_fb
    elif not ok:
        fb = fb or "invalid JSON"

    if failed == expected and not specs:
        def validate(resp: str) -> tuple[bool, str]:
            ok2, fb2, payload2 = validate_pydantic_json(resp, ShotSpecBatchLLMSchema)
            if not ok2 or payload2 is None:
                return ok2, fb2
            return _validate_batch(payload2, expected, seg_by_index)

        resp, n = invoke_pass_with_review(
            chat_fn,
            system=SHOT_SPEC_SYSTEM,
            user=user,
            max_tokens=token_budget(3200),
            think_apply=think_apply,
            validate=validate,
            max_attempts=2,
            pass_name="shot_spec",
            on_progress=on_progress,
        )
        calls += n
        ok, fb, payload = validate_pydantic_json(resp, ShotSpecBatchLLMSchema)
        if ok and payload is not None:
            batch_ok, batch_fb = _validate_batch(payload, expected, seg_by_index)
            if batch_ok:
                specs = _rows_to_specs(payload.shots, seg_by_index)
                failed = set()
            else:
                specs, failed = _split_valid_specs(payload, expected, seg_by_index)
                fb = batch_fb

    if failed:
        retry_specs, n = _retry_failed_indices(
            failed,
            seg_by_index=seg_by_index,
            script=script,
            locale_block=locale_block,
            chat_fn=chat_fn,
            think_apply=think_apply,
            token_budget=token_budget,
            on_progress=on_progress,
        )
        calls += n
        specs.extend(retry_specs)
        got = {s.segment_index for s in specs}
        still = expected - got
        if still:
            raise ValueError(f"shot_spec failed segment_index after retry: {sorted(still)}; last: {fb}")

    if not specs:
        raise ValueError(f"shot_spec batch validation failed: {fb}")

    specs.sort(key=lambda s: s.segment_index)
    return specs, calls


def run_shot_spec(
    *,
    script: ScriptArtifact,
    beat_plan: BeatPlanArtifact,
    locale: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
    beat_indices: list[int] | None = None,
) -> tuple[list[ShotSpecArtifact], int]:
    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)
    segments = flatten_planned_segments(beat_plan)
    if beat_indices:
        allowed = {int(i) for i in beat_indices}
        segments = [s for s in segments if s.beat_index in allowed]
    if not segments:
        raise ValueError("beat_plan produced no segments")

    if on_progress:
        on_progress("shot_spec", "shot_spec")

    all_specs: list[ShotSpecArtifact] = []
    llm_calls = 0
    for i in range(0, len(segments), SHOT_SPEC_BATCH):
        batch = segments[i : i + SHOT_SPEC_BATCH]
        specs, n = _invoke_batch(
            batch,
            script=script,
            locale_block=locale_block,
            chat_fn=chat_fn,
            think_apply=think_apply,
            token_budget=token_budget,
            on_progress=on_progress,
        )
        all_specs.extend(specs)
        llm_calls += n

    all_specs.sort(key=lambda s: s.segment_index)
    return all_specs, llm_calls
