"""Artifact and LLM I/O schemas for script_parse pipeline."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.engine.llm.schemas.long_video import _reject_placeholder

Visibility = Literal["invisible", "silhouette", "partial", "full_face"]
SegmentRole = Literal[
    "establishing",
    "pre_anchor",
    "face_anchor",
    "post_anchor",
    "keyframe",
    "tail_continuation",
]
Reachability = Literal["identity_critical", "establishing", "action_wide", "empty"]
CharacterRole = Literal["protagonist", "supporting", "extra"]
NarrativeRole = Literal[
    "establish_context",
    "introduce_subject",
    "build_tension",
    "deliver_payload",
    "transition",
    "emotional_beat",
    "evidence",
    "comparison",
    "resolution",
    "call_to_action",
    "",
]

_VALID_NARRATIVE_ROLES = frozenset(
    {
        "establish_context",
        "introduce_subject",
        "build_tension",
        "deliver_payload",
        "transition",
        "emotional_beat",
        "evidence",
        "comparison",
        "resolution",
        "call_to_action",
    }
)

# LLM often copies reachability enum into beat-level narrative_role — remap, do not fail.
_REACHABILITY_AS_NARRATIVE_ROLE: dict[str, NarrativeRole] = {
    "identity_critical": "introduce_subject",
    "establishing": "establish_context",
    "action_wide": "build_tension",
    "empty": "establish_context",
}


def sanitize_narrative_role(raw: str | None) -> NarrativeRole:
    text = (raw or "").strip()
    if not text:
        return ""
    if text in _VALID_NARRATIVE_ROLES:
        return text  # type: ignore[return-value]
    mapped = _REACHABILITY_AS_NARRATIVE_ROLE.get(text)
    if mapped is not None:
        return mapped
    return ""


_VALID_SEGMENT_ROLES = frozenset(
    {
        "establishing",
        "pre_anchor",
        "face_anchor",
        "post_anchor",
        "keyframe",
        "tail_continuation",
    }
)

_SEGMENT_ROLE_ALIASES: dict[str, SegmentRole] = {
    "wide": "keyframe",
    "action_wide": "keyframe",
    "action": "keyframe",
    "anchor": "face_anchor",
    "face": "face_anchor",
    "establish": "establishing",
    "empty": "establishing",
    "continuation": "tail_continuation",
}

_VALID_VISIBILITY = frozenset({"invisible", "silhouette", "partial", "full_face"})
_VISIBILITY_ALIASES: dict[str, Visibility] = {
    "visible": "full_face",
    "full": "full_face",
    "face": "full_face",
    "hidden": "invisible",
    "none": "invisible",
}

_VALID_REACHABILITY = frozenset({"identity_critical", "establishing", "action_wide", "empty"})
_REACHABILITY_ALIASES: dict[str, Reachability] = {
    "identity": "identity_critical",
    "critical": "identity_critical",
    "wide": "action_wide",
    "action": "action_wide",
    "establish": "establishing",
}


def sanitize_segment_role(raw: str | None) -> SegmentRole:
    text = (raw or "").strip()
    if text in _VALID_SEGMENT_ROLES:
        return text  # type: ignore[return-value]
    mapped = _SEGMENT_ROLE_ALIASES.get(text)
    if mapped is not None:
        return mapped
    return "keyframe"


def sanitize_visibility(raw: str | None, *, default: Visibility = "partial") -> Visibility:
    text = (raw or "").strip()
    if text in _VALID_VISIBILITY:
        return text  # type: ignore[return-value]
    mapped = _VISIBILITY_ALIASES.get(text)
    if mapped is not None:
        return mapped
    return default


def sanitize_reachability(raw: str | None, *, default: Reachability = "establishing") -> Reachability:
    text = (raw or "").strip()
    if text in _VALID_REACHABILITY:
        return text  # type: ignore[return-value]
    mapped = _REACHABILITY_ALIASES.get(text)
    if mapped is not None:
        return mapped
    return default

_MOTION_CAMERA_RE = re.compile(
    r"\b(pan|dolly|truck|zoom|crane|orbit|whip|tracking|tilt)\b",
    re.I,
)


class CharacterLookArtifact(BaseModel):
    id: str = ""
    label: str = "默认"
    body: str = ""
    portrait_prompt_hint: str = ""


class SceneLookArtifact(BaseModel):
    id: str = ""
    label: str = "默认"
    body: str = ""
    environment_prompt_hint: str = ""


class ScriptBeatArtifact(BaseModel):
    index: int = Field(ge=0)
    title: str = ""
    location: str = ""
    narrative: str = ""
    enhancement_cues: list[str] = Field(default_factory=list)
    suggested_shot_size: str = ""
    estimated_duration_sec: float = Field(default=5.0, ge=1.0, le=120.0)

    @field_validator("location", "narrative")
    @classmethod
    def _required(cls, value: str, info) -> str:
        if info.field_name == "narrative":
            return _reject_placeholder(value, "beats[].narrative")
        text = (value or "").strip()
        if not text:
            raise ValueError("beats[].location is empty")
        return text


class ScriptCharacterArtifact(BaseModel):
    id: str = ""
    name: str
    role: CharacterRole = "supporting"
    looks: list[CharacterLookArtifact] = Field(min_length=1)
    default_look_id: str = ""

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _reject_placeholder(value, "characters[].name")


class ScriptSceneArtifact(BaseModel):
    id: str = ""
    name: str
    looks: list[SceneLookArtifact] = Field(min_length=1)
    default_look_id: str = ""
    spatial_layout_json: dict = Field(default_factory=dict)
    grounding_panorama_asset_id: str = ""
    grounding_depth_asset_id: str = ""


class ScriptArtifact(BaseModel):
    version: Literal["2.0"] = "2.0"
    title: str = ""
    synopsis: str = ""
    mood: str = ""
    style_anchor: str = ""
    beats: list[ScriptBeatArtifact] = Field(min_length=2, max_length=24)
    characters: list[ScriptCharacterArtifact] = Field(min_length=1)
    scenes: list[ScriptSceneArtifact] = Field(min_length=1)

    @field_validator("synopsis")
    @classmethod
    def _synopsis(cls, value: str) -> str:
        text = _reject_placeholder(value, "synopsis")
        if len(text) < 8:
            raise ValueError("synopsis is too short")
        return text


class CameraZoneSnippet(BaseModel):
    id: str = ""
    description: str = ""
    visible_area: str = ""


class SpatialSnippet(BaseModel):
    location: str = ""
    dimensions: str = ""
    objects: list[str] = Field(default_factory=list)
    camera_zones: list[CameraZoneSnippet] = Field(default_factory=list)


class PlannedSegmentArtifact(BaseModel):
    segment_index: int = Field(ge=0)
    beat_index: int = Field(ge=0)
    role: SegmentRole = "keyframe"
    duration_sec: float = Field(ge=1.0, le=10.0)
    shot_size: str = ""
    characters_on_screen: list[str] = Field(default_factory=list)
    start_visibility: Visibility = "partial"
    end_visibility: Visibility = "full_face"
    first_frame_requirement: str = ""
    reachability: Reachability = "establishing"
    is_intentional_empty: bool = False
    spatial: SpatialSnippet | None = None
    start_frame_mode: Literal["keyframe", "prev_segment_tail", "anchor_link"] = "keyframe"
    segment_group_id: str = ""
    segment_group_index: int = 0
    face_anchor_shot_id: str = ""


class BeatPlanRowArtifact(BaseModel):
    beat_index: int = Field(ge=0)
    shot_intent: str = ""
    narrative_role: NarrativeRole = ""
    segments: list[PlannedSegmentArtifact] = Field(min_length=1)


class BeatPlanArtifact(BaseModel):
    version: Literal["2.0"] = "2.0"
    beats: list[BeatPlanRowArtifact] = Field(min_length=1)


class ShotLanguageArtifact(BaseModel):
    shot_size: str = ""
    camera_movement: str = "static"
    lighting_key: str = "natural"
    depth_of_field: Literal["shallow", "medium", "deep"] = "medium"
    color_temperature: Literal["cool", "neutral", "warm", "mixed"] = "neutral"


class FiveAspectArtifact(BaseModel):
    subject: str = ""
    subject_motion: str = ""
    scene: str = ""
    spatial_framing: str = ""
    camera: str = ""


class ShotSpecArtifact(BaseModel):
    segment_index: int = Field(ge=0)
    beat_index: int = Field(ge=0)
    role: SegmentRole = "keyframe"
    five_aspect: FiveAspectArtifact
    shot_language: ShotLanguageArtifact
    shot_intent: str = ""
    narrative_role: NarrativeRole = ""
    video_prompt: str = ""
    start_visual: str = ""
    anchor_visual: str = ""
    characters_on_screen: list[str] = Field(default_factory=list)
    is_intentional_empty: bool = False
    start_visibility: Visibility = "partial"
    end_visibility: Visibility = "full_face"
    duration_sec: float = 5.0
    first_frame_requirement: str = ""
    location: str = ""
    start_frame_mode: Literal["keyframe", "prev_segment_tail", "anchor_link"] = "keyframe"
    segment_group_id: str = ""
    segment_group_index: int = 0
    face_anchor_shot_id: str = ""
    camera_zone_id: str = ""

    @model_validator(mode="after")
    def _subject_closure(self) -> ShotSpecArtifact:
        if self.is_intentional_empty or not self.characters_on_screen:
            return self
        blob = f"{self.five_aspect.subject} {self.start_visual} {self.anchor_visual}"
        missing = [n for n in self.characters_on_screen if n and n not in blob]
        if missing:
            raise ValueError(
                f"segment {self.segment_index}: characters_on_screen missing from subject/start_visual: {missing}"
            )
        if self.role != "face_anchor" and (self.anchor_visual or "").strip():
            raise ValueError(
                f"segment {self.segment_index}: anchor_visual only allowed for face_anchor role"
            )
        if self.role == "face_anchor" and self.characters_on_screen:
            primary = self.characters_on_screen[0]
            if primary and primary not in (self.anchor_visual or self.start_visual):
                raise ValueError(
                    f"segment {self.segment_index}: face_anchor requires {primary!r} in anchor_visual"
                )
        from backend.long_video.keyframe_prompt_policy import validate_visibility_role_contract

        for msg in validate_visibility_role_contract(
            segment_role=self.role,
            start_visibility=self.start_visibility,
            beat_index=self.beat_index,
            segment_label=f"segment_index={self.segment_index}",
        ):
            raise ValueError(msg)
        cam = (self.shot_language.camera_movement or "").strip().lower()
        if cam == "static" and _MOTION_CAMERA_RE.search(self.video_prompt or ""):
            raise ValueError(
                f"segment {self.segment_index}: static camera_movement conflicts with video_prompt motion"
            )
        return self


# --- LLM batch I/O schemas ---

class DecomposeLLMSchema(BaseModel):
    title: str = ""
    synopsis: str
    mood: str = ""
    style_anchor: str = ""
    beats: list[ScriptBeatArtifact]
    characters: list[ScriptCharacterArtifact]
    scenes: list[ScriptSceneArtifact]


class BeatPlanSegmentLLM(BaseModel):
    role: str = "keyframe"
    duration_sec: float = Field(ge=1.0, le=10.0)
    shot_size: str = ""
    characters_on_screen: list[str] = Field(default_factory=list)
    start_visibility: str = "partial"
    end_visibility: str = "full_face"
    first_frame_requirement: str = ""
    reachability: str = "establishing"
    is_intentional_empty: bool = False
    spatial: SpatialSnippet | None = None


class BeatPlanRowLLM(BaseModel):
    beat_index: int = Field(ge=0)
    shot_intent: str = ""
    narrative_role: str = ""
    segments: list[BeatPlanSegmentLLM] = Field(min_length=1)


class BeatPlanLLMSchema(BaseModel):
    beats: list[BeatPlanRowLLM] = Field(min_length=1)


class ShotSpecRowLLM(BaseModel):
    segment_index: int = Field(ge=0)
    five_aspect: FiveAspectArtifact
    shot_language: ShotLanguageArtifact
    shot_intent: str = ""
    video_prompt: str = ""
    start_visual: str = ""
    anchor_visual: str = ""


class ShotSpecBatchLLMSchema(BaseModel):
    shots: list[ShotSpecRowLLM] = Field(min_length=1)


class DecomposeResult(BaseModel):
    artifact: ScriptArtifact
    llm_calls: int = 0
    phases: list[str] = Field(default_factory=list)


class ExpandResult(BaseModel):
    shots: list[dict]
    characters: list[dict]
    scenes: list[dict]
    character_anchor: str = ""
    style_anchor: str = ""
    llm_calls: int = 0
    phases: list[str] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)
    quality_issues: list[dict] = Field(default_factory=list)
