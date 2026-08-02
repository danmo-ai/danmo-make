"""Shared segment planning types (legacy repair helpers)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SegmentRole = Literal[
    "establishing",
    "pre_anchor",
    "face_anchor",
    "post_anchor",
    "keyframe",
    "tail_continuation",
]
FlfMode = Literal["none", "first_last", "continuation"]


@dataclass(frozen=True)
class SubsegmentPlan:
    role: SegmentRole
    duration_sec: float
    shot_size: str
    flf_mode: FlfMode
    start_visibility: str = "full_face"
    end_visibility: str = "full_face"
    characters_on_screen: tuple[str, ...] = ()
    first_frame_requirement: str = ""
