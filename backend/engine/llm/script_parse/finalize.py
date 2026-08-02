"""Pass 4 — cast lock, validation, targeted repair, quality."""
from __future__ import annotations

from typing import Any, Callable

from backend.long_video.parse_quality import validate_parse_quality
from backend.engine.llm.script_parse.errors import ScriptParseQualityError
from backend.long_video.shot_contract_validator import (
    clamp_shot_durations,
    validate_shot_contracts,
)
from backend.long_video.shot_repair import repair_shot_contracts
from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
from backend.engine.llm.script_parse.rules import merge_spatial_into_scenes
from backend.engine.llm.script_parse.schemas import (
    BeatPlanArtifact,
    ExpandResult,
    ScriptArtifact,
    ShotSpecArtifact,
)
from backend.engine.llm.storyboard import normalize_storyboard_locale
from backend.engine.llm.storyboard_cast import (
    cast_looks_to_dtos,
    dtos_to_roster,
    ensure_cast_covers_on_screen,
    format_character_roster,
    infer_shot_cast_looks,
    strip_name_look_tags,
    supplement_roster_from_shots,
)
from backend.engine.llm.storyboard_scenes import dtos_to_roster as scene_dtos_to_roster
from backend.engine.llm.storyboard_scenes import infer_shot_scene_look, scene_look_to_dtos

ProgressFn = Callable[[str, str], None]

_PROMPT_FIELDS = (
    "start_visual_prompt",
    "visual_prompt",
    "video_prompt",
    "motion_prompt",
    "anchor_visual_prompt",
    "scene_prompt",
)


def _spec_to_shot_dict(spec: ShotSpecArtifact, order: int) -> dict[str, Any]:
    return {
        "id": f"shot_{spec.segment_index}",
        "order": order,
        "visual_prompt": spec.start_visual,
        "start_visual_prompt": spec.start_visual,
        "video_prompt": spec.video_prompt,
        "motion_prompt": spec.video_prompt,
        "anchor_visual_prompt": spec.anchor_visual,
        "scene_prompt": spec.five_aspect.scene,
        "segment_role": spec.role,
        "start_frame_mode": spec.start_frame_mode,
        "segment_group_id": spec.segment_group_id,
        "segment_group_index": spec.segment_group_index,
        "face_anchor_shot_id": spec.face_anchor_shot_id,
        "flf_mode": "continuation" if spec.role == "tail_continuation" else "none",
        "duration_sec": spec.duration_sec,
        "first_frame_visibility": spec.start_visibility,
        "end_visibility": spec.end_visibility,
        "characters_on_screen": list(spec.characters_on_screen),
        "first_frame_requirement": spec.first_frame_requirement,
        "camera_zone_id": spec.camera_zone_id,
        "location": spec.location,
        "shot_size": spec.shot_language.shot_size or "",
        "narrative_beat_index": spec.beat_index,
        "shot_intent": spec.shot_intent,
        "narrative_role": spec.narrative_role or "",
        "camera_movement": spec.shot_language.camera_movement,
        "lighting_key": spec.shot_language.lighting_key,
        "is_establishing_empty": spec.is_intentional_empty,
        "clip_start_state": spec.start_visual,
        "clip_end_state": spec.five_aspect.subject_motion,
    }


def _sanitize_shots(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for shot in shots:
        row = dict(shot)
        for fld in _PROMPT_FIELDS:
            if row.get(fld):
                row[fld] = strip_name_look_tags(str(row[fld]))
        out.append(row)
    return out


def _apply_cast_lock(
    shots: list[dict[str, Any]],
    *,
    character_dtos: list[dict[str, Any]],
    scene_dtos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roster = dtos_to_roster(character_dtos)
    scenes = scene_dtos_to_roster(scene_dtos) if scene_dtos else []
    out: list[dict[str, Any]] = []
    prev_cast = None
    prev_scene = None
    for shot in shots:
        row = dict(shot)
        visual = str(row.get("start_visual_prompt") or row.get("visual_prompt") or "").strip()
        location = str(row.get("location") or "").strip()
        scene_text = str(row.get("scene_prompt") or visual)
        beat = str(row.get("video_prompt") or visual)
        on_screen = [str(n).strip() for n in (row.get("characters_on_screen") or []) if str(n).strip()]
        roster_for_shot = [ch for ch in roster if ch.name in on_screen] if on_screen else roster
        binding = infer_shot_scene_look(beat=scene_text, scenes=scenes, prev=prev_scene) if scenes else None
        scene_changed = bool(prev_scene and binding and binding.scene_id != prev_scene.scene_id)
        cast = infer_shot_cast_looks(
            scene="\n".join(p for p in (location, visual) if p),
            beat="\n".join(p for p in (scene_text, beat) if p),
            characters=roster_for_shot,
            prev=None if scene_changed else prev_cast,
            on_screen_names=on_screen or None,
        )
        cast = ensure_cast_covers_on_screen(
            cast,
            on_screen_names=on_screen,
            characters=roster,
        )
        prev_cast = cast
        row["cast_looks"] = cast_looks_to_dtos(cast)
        if binding:
            dto = scene_look_to_dtos(binding)
            if dto:
                row["scene_look"] = dto
                prev_scene = binding
        out.append(row)
    return out


def _beat_sheet_from_script(script: ScriptArtifact) -> list[str]:
    lines: list[str] = []
    for b in script.beats:
        lines.append(f"{b.title} | {b.suggested_shot_size} | {b.location} | {b.narrative}")
    return lines


def apply_cast_lock(
    shots: list[dict[str, Any]],
    *,
    character_dtos: list[dict[str, Any]],
    scene_dtos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _apply_cast_lock(shots, character_dtos=character_dtos, scene_dtos=scene_dtos)


def run_finalize(
    *,
    script: ScriptArtifact,
    beat_plan: BeatPlanArtifact,
    specs: list[ShotSpecArtifact],
    target_duration_sec: float,
    max_clip_sec: float,
    locale: str,
    chat_fn: Callable[..., Any] | None = None,
    think_apply: Callable[[str], str] | None = None,
    token_budget: Callable[[int], int] | None = None,
    on_progress: ProgressFn | None = None,
) -> ExpandResult:
    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)

    character_dtos = [c.model_dump() for c in script.characters]
    scene_dtos = merge_spatial_into_scenes(script, beat_plan)

    shots = [_spec_to_shot_dict(s, i) for i, s in enumerate(sorted(specs, key=lambda x: x.segment_index))]
    shots = _sanitize_shots(shots)
    shots = _apply_cast_lock(shots, character_dtos=character_dtos, scene_dtos=scene_dtos)

    character_dtos = supplement_roster_from_shots(list(character_dtos), shots, locale=loc)
    roster = dtos_to_roster(character_dtos)
    character_anchor = format_character_roster(roster, script.style_anchor, locale=loc) if roster else ""
    style_anchor = script.style_anchor

    beat_budgets: dict[int, float] = {}
    for brow in beat_plan.beats:
        beat_budgets[brow.beat_index] = sum(s.duration_sec for s in brow.segments)

    if on_progress:
        on_progress("finalize", "finalize")

    shots = clamp_shot_durations(shots, max_clip_sec=max_clip_sec)
    validation = validate_shot_contracts(
        shots,
        character_anchor=character_anchor,
        max_clip_sec=max_clip_sec,
        target_duration_sec=target_duration_sec,
        beat_budgets=beat_budgets,
    )

    llm_calls = 0
    if not validation.ok:
        shots = repair_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=max_clip_sec,
            beat_budgets=beat_budgets,
        )
        validation = validate_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=max_clip_sec,
            target_duration_sec=target_duration_sec,
            beat_budgets=beat_budgets,
        )

    if not validation.ok and chat_fn is not None and think_apply is not None and token_budget is not None:
        from backend.engine.llm.shot_repair_llm import repair_shots_with_llm

        if on_progress:
            on_progress("finalize", "shot_repair_llm")
        shots, n = repair_shots_with_llm(
            shots,
            issues=validation.issues,
            character_anchor=character_anchor,
            chat_fn=chat_fn,
            think_apply=think_apply,
            max_tokens=token_budget(2400),
            locale_block=locale_block,
        )
        llm_calls += n
        shots = repair_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=max_clip_sec,
            beat_budgets=beat_budgets,
        )
        validation = validate_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=max_clip_sec,
            target_duration_sec=target_duration_sec,
            beat_budgets=beat_budgets,
        )

    warnings = [w.message for w in validation.warnings]
    if not validation.ok:
        codes = "; ".join(i.message for i in validation.issues[:5])
        raise RuntimeError(f"shot contract validation failed: {codes}")

    beat_sheet = _beat_sheet_from_script(script)
    quality = validate_parse_quality(
        shots,
        beat_sheet=beat_sheet,
        character_anchor=character_anchor,
        character_dtos=character_dtos,
        style_anchor=style_anchor,
    )
    warnings.extend(quality.warning_messages)

    critical = [i for i in quality.issues if i.severity == "critical"]
    if critical:
        raise ScriptParseQualityError(
            "; ".join(i.message for i in critical[:3]),
            quality_issues=quality.issue_dicts(),
        )

    return ExpandResult(
        shots=shots,
        characters=character_dtos,
        scenes=scene_dtos,
        character_anchor=character_anchor,
        style_anchor=style_anchor,
        llm_calls=llm_calls,
        phases=["finalize"],
        validation_warnings=warnings,
        quality_issues=quality.issue_dicts(),
    )
