"""Long-video timing plans — segmented shot plan + LTX extend plan."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from backend.long_video.constants import MAX_CLIP_SEC, MIN_CLIP_SEC

# LTX extend plan lives in family module; re-export for orchestrator + storyboard.
from backend.engine.families.ltx.long_video_plan import (  # noqa: F401
    LongVideoPlan,
    build_long_video_plan,
    compute_extend_pass_count,
    narrative_budget_for_target,
)


def duration_sec_from_num_frames(num_frames: int, fps: float) -> float:
    rate = max(1.0, float(fps))
    nf = max(1, int(num_frames))
    return max(0.0, (nf - 1) / rate)


def num_frames_for_duration_sec(duration_sec: float, fps: float) -> int:
    rate = max(1.0, float(fps))
    sec = max(0.0, float(duration_sec))
    return max(1, int(round(sec * rate)) + 1)


@dataclass(frozen=True)
class ShotPlan:
    target_duration_sec: float
    segment_duration_sec: float
    shot_count: int
    segment_durations_sec: tuple[float, ...]
    narrative_budget: str


_ACTION_KEYWORDS = re.compile(
    r"追逐|大战|决斗|战斗|打斗|交锋|对决|厮杀|爆炸|"
    r"chase|fight|battle|duel|combat|explosion",
    re.I,
)
_ESTABLISHING_KEYWORDS = re.compile(
    r"远景|全景|建立|航拍|俯瞰|空镜|"
    r"establishing|wide shot|aerial",
    re.I,
)


def _beat_duration_weight(beat: str) -> float:
    text = (beat or "").strip()
    weight = max(1.0, len(text) ** 0.45)
    if _ACTION_KEYWORDS.search(text):
        weight *= 1.35
    if _ESTABLISHING_KEYWORDS.search(text):
        weight *= 0.85
    return weight


def allocate_shot_durations(
    *,
    scene_count: int,
    target_duration_sec: float,
    default_segment_sec: float = 5.0,
    beat_texts: list[str] | None = None,
    min_sec: float = MIN_CLIP_SEC,
    max_sec: float = MAX_CLIP_SEC,
) -> tuple[float, ...]:
    """Per-shot I2V durations. ``target_duration_sec`` is a soft budget — sum may differ."""
    count = max(1, int(scene_count))
    lo = max(0.5, float(min_sec))
    hi = max(lo, float(max_sec))
    default = max(lo, min(hi, float(default_segment_sec)))
    target = max(default, float(target_duration_sec))

    if beat_texts and len(beat_texts) >= count:
        weights = [_beat_duration_weight(beat_texts[i]) for i in range(count)]
    else:
        weights = [1.0] * count

    total_weight = sum(weights) or float(count)
    raw = [target * (w / total_weight) for w in weights]
    durations: list[float] = []
    for value in raw:
        rounded = max(lo, min(hi, round(float(value))))
        durations.append(float(int(rounded) if rounded >= 1 else round(rounded * 2) / 2))
    return tuple(durations)


def total_shot_duration_sec(durations: tuple[float, ...] | list[float]) -> float:
    return sum(float(d) for d in durations if float(d) > 0)


def build_shot_plan(
    *,
    target_duration_sec: float = 60.0,
    segment_duration_sec: float = 5.0,
    beat_texts: list[str] | None = None,
) -> ShotPlan:
    target = max(0.0, float(target_duration_sec))
    seg = max(0.5, float(segment_duration_sec))
    count = max(1, int(math.ceil(target / seg)))
    durations = allocate_shot_durations(
        scene_count=count,
        target_duration_sec=target,
        default_segment_sec=seg,
        beat_texts=beat_texts,
    )
    actual_total = total_shot_duration_sec(durations)
    return ShotPlan(
        target_duration_sec=actual_total,
        segment_duration_sec=seg,
        shot_count=count,
        segment_durations_sec=durations,
        narrative_budget=narrative_budget_for_target(target),
    )


def build_shot_plan_for_scenes(
    *,
    scene_count: int,
    segment_duration_sec: float = 5.0,
    target_duration_sec: float | None = None,
    beat_texts: list[str] | None = None,
) -> ShotPlan:
    """Shot count from chapter scene analysis; per-shot durations allocated on demand."""
    seg = max(0.5, float(segment_duration_sec))
    count = max(2, min(24, int(scene_count)))
    target = float(target_duration_sec) if target_duration_sec is not None else count * seg
    durations = allocate_shot_durations(
        scene_count=count,
        target_duration_sec=target,
        default_segment_sec=seg,
        beat_texts=beat_texts,
    )
    actual_total = total_shot_duration_sec(durations)
    return ShotPlan(
        target_duration_sec=actual_total,
        segment_duration_sec=seg,
        shot_count=count,
        segment_durations_sec=durations,
        narrative_budget=narrative_budget_for_target(target),
    )
