"""Validation helpers for segment_video LLM batches (role-differentiated motion)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.long_video.parse_quality import (
    ParseQualityIssue,
    _motion_core,
    _prompt_similarity,
)


class SegmentPlanLike(Protocol):
    segment_index: int
    segment_group_id: str
    segment_role: str


@dataclass
class SegmentVideoValidationResult:
    ok: bool
    issues: list[ParseQualityIssue]

    @property
    def feedback(self) -> str:
        return "\n".join(f"- {i.message}" for i in self.issues)


def _shots_from_segment_video(
    segments: list[Any],
    video_by_index: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seg in segments:
        idx = int(seg.segment_index)
        rows.append(
            {
                "segment_group_id": str(seg.segment_group_id or ""),
                "segment_role": str(seg.segment_role or ""),
                "video_prompt": str(video_by_index.get(idx) or "").strip(),
            }
        )
    return rows


def validate_segment_video_prompts(
    segments: list[Any],
    video_by_index: dict[int, str],
    *,
    style_anchor: str = "",
    character_anchor: str = "",
    motion_similarity_threshold: float = 0.88,
    motion_core_similarity_threshold: float = 0.82,
) -> SegmentVideoValidationResult:
    """Check motion differentiation within beat groups for a video prompt map."""
    from backend.long_video.parse_quality import validate_parse_quality

    shots = _shots_from_segment_video(segments, video_by_index)
    if not shots:
        return SegmentVideoValidationResult(ok=True, issues=[])

    motion_codes = {
        "motion_duplicate_in_group",
        "motion_role_undifferentiated",
        "style_phrase_spam",
        "prompt_fragment_spam",
    }
    result = validate_parse_quality(
        shots,
        beat_sheet=[],
        character_anchor=character_anchor,
        style_anchor=style_anchor,
        motion_similarity_threshold=motion_similarity_threshold,
        motion_core_similarity_threshold=motion_core_similarity_threshold,
    )
    issues = [i for i in result.issues if i.code in motion_codes]
    return SegmentVideoValidationResult(ok=not issues, issues=issues)


def affected_segment_groups(issues: list[ParseQualityIssue]) -> list[str]:
    groups: list[str] = []
    seen: set[str] = set()
    for issue in issues:
        msg = issue.message or ""
        for token in msg.split():
            if token.startswith("beat_") and token not in seen:
                seen.add(token)
                groups.append(token)
    return groups


def group_ids_for_segment_indices(
    segments: list[Any],
    indices: set[int],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for seg in segments:
        if int(seg.segment_index) not in indices:
            continue
        gid = str(seg.segment_group_id or "")
        if gid and gid not in seen:
            seen.add(gid)
            out.append(gid)
    return out


def validate_segment_video_batch_payload(
    segments: list[Any],
    payload_rows: list[Any],
    *,
    style_anchor: str = "",
    character_anchor: str = "",
) -> tuple[bool, str]:
    """Validate parsed segment_video JSON rows before accepting a batch."""
    from difflib import SequenceMatcher

    index_to_seg = {int(s.segment_index): s for s in segments}
    video_by_index: dict[int, str] = {}
    for row in payload_rows:
        idx = int(row.index)
        text = str(row.video_prompt or "").strip()
        if not text:
            return False, f"index {idx}: video_prompt is empty"
        video_by_index[idx] = text

    expected = {int(s.segment_index) for s in segments}
    missing = expected - set(video_by_index.keys())
    if missing:
        return False, f"missing indices: {sorted(missing)}"

    sim_kw = {"style_anchor": style_anchor, "character_anchor": character_anchor}
    by_group: dict[str, list[int]] = {}
    for idx, seg in index_to_seg.items():
        gid = str(seg.segment_group_id or f"idx_{idx}")
        by_group.setdefault(gid, []).append(idx)

    issues: list[str] = []
    for gid, indices in by_group.items():
        if len(indices) < 2:
            continue
        prompts = [video_by_index[i] for i in indices]
        base = prompts[0]
        dup = all(
            _prompt_similarity(base, p, **sim_kw) >= 0.88
            or SequenceMatcher(
                None,
                _motion_core(base, **sim_kw),
                _motion_core(p, **sim_kw),
            ).ratio()
            >= 0.82
            for p in prompts[1:]
        )
        if dup:
            roles = [str(index_to_seg[i].segment_role or "") for i in indices]
            issues.append(
                f"{gid}: indices {indices} roles {roles} have near-identical video_prompt — "
                f"write distinct motion per role (pre=approach, face=hold/breath, post=continue)"
            )
            continue
        by_role = {str(index_to_seg[i].segment_role or ""): video_by_index[i] for i in indices}
        face = by_role.get("face_anchor", "").strip()
        for role in ("pre_anchor", "post_anchor"):
            other = by_role.get(role, "").strip()
            if not face or not other:
                continue
            if (
                _prompt_similarity(face, other, **sim_kw) >= 0.88
                or SequenceMatcher(
                    None,
                    _motion_core(face, **sim_kw),
                    _motion_core(other, **sim_kw),
                ).ratio()
                >= 0.82
            ):
                issues.append(
                    f"{gid}: {role} too similar to face_anchor at indices "
                    f"{[i for i in indices if index_to_seg[i].segment_role == role]} "
                    f"vs {[i for i in indices if index_to_seg[i].segment_role == 'face_anchor']}"
                )

    if issues:
        return False, "\n".join(issues)
    return True, ""
