"""Long-video project activity helpers (parse runs + generation task linkage)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import backend.core.task_kinds as TK
from backend.long_video.t2i_provenance import build_shots_summary_with_provenance

if TYPE_CHECKING:
    from backend.persistence.long_video_activity_store import LongVideoActivityStore


def extract_long_video_context(params: dict[str, Any]) -> dict[str, str]:
    """Read long-video linkage fields from task/generation params."""
    meta = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
    project_id = str(meta.get("long_video_project_id") or meta.get("group_id") or "").strip()
    return {
        "project_id": project_id,
        "phase": str(meta.get("long_video_phase") or "").strip(),
        "shot_id": str(meta.get("long_video_shot_id") or "").strip(),
        "cast_character_id": str(meta.get("cast_character_id") or "").strip(),
        "cast_look_id": str(meta.get("cast_look_id") or "").strip(),
        "scene_id": str(meta.get("scene_id") or "").strip(),
        "scene_look_id": str(meta.get("scene_look_id") or "").strip(),
        "project_title": str(meta.get("long_video_project_title") or "").strip(),
    }


def category_for_context(*, phase: str, task_kind: str) -> str:
    if phase in ("cast_portrait", "scene_ref", "keyframe"):
        return "image_generation"
    if phase in ("segment", "assemble_only"):
        return "video_generation"
    if TK.is_image_kind(task_kind):
        return "image_generation"
    if TK.is_video_kind(task_kind):
        return "video_generation"
    if TK.is_audio_kind(task_kind):
        return "audio_generation"
    return "generation"


def phase_label(phase: str) -> str:
    labels = {
        "cast_portrait": "cast portrait",
        "scene_ref": "scene reference",
        "keyframe": "storyboard keyframe",
        "segment": "segment video",
        "assemble_only": "long video assemble",
        "plan": "script parse plan",
        "roster": "script parse roster",
        "scenes": "script parse scenes",
        "shots": "script parse shots",
    }
    return labels.get(phase, phase or "generation")


def task_event_summary(
    *,
    event_type: str,
    phase: str,
    task_kind: str,
    model_id: str,
    shot_id: str = "",
) -> str:
    label = phase_label(phase) if phase else task_kind
    parts = [label]
    if shot_id:
        parts.append(f"shot={shot_id}")
    if model_id:
        parts.append(f"model={model_id}")
    parts.append(event_type.replace("_", " "))
    return " · ".join(parts)


class LongVideoActivityRecorder:
    """Records script-parse lifecycle events for one project run."""

    def __init__(self, store: LongVideoActivityStore, project_id: str) -> None:
        from backend.core.contracts import new_parse_run_id

        self._store = store
        self.project_id = project_id.strip()
        self.parse_run_id = new_parse_run_id() if self.project_id else ""

    @property
    def active(self) -> bool:
        return bool(self.project_id and self.parse_run_id)

    def record_started(self, *, chapter_title: str = "", target_duration_sec: float = 0.0) -> None:
        if not self.active:
            return
        self._store.append_event(
            project_id=self.project_id,
            category="script_parse",
            event_type="parse_started",
            phase="script_parse",
            parse_run_id=self.parse_run_id,
            summary=f"Script parse started{f': {chapter_title}' if chapter_title else ''}",
            detail={
                "chapter_title": chapter_title,
                "target_duration_sec": target_duration_sec,
            },
        )

    def record_phase(self, phase: str, message: str = "") -> None:
        if not self.active:
            return
        self._store.append_event(
            project_id=self.project_id,
            category="script_parse",
            event_type="parse_phase",
            phase=phase,
            parse_run_id=self.parse_run_id,
            summary=f"Parse phase: {phase_label(phase)}",
            detail={"message": message},
        )

    def record_completed(self, response: Any) -> None:
        if not self.active:
            return
        shots = getattr(response, "shots", None) or []
        shot_rows, t2i_stats = build_shots_summary_with_provenance(shots)
        quality_issues = getattr(response, "quality_issues", None) or []
        issue_rows = []
        for issue in quality_issues:
            if hasattr(issue, "model_dump"):
                issue_rows.append(issue.model_dump())
            elif isinstance(issue, dict):
                issue_rows.append(issue)
        detail = {
            "chapter_title": getattr(response, "chapter_title", ""),
            "scene_count": getattr(response, "scene_count", 0),
            "shot_count": len(shots),
            "character_count": len(getattr(response, "characters", None) or []),
            "llm_calls": getattr(response, "llm_calls", 0),
            "quality_warning_count": len(getattr(response, "quality_warnings", None) or []),
            "quality_issues": issue_rows[:50],
            "parse_phases": [
                p.model_dump() if hasattr(p, "model_dump") else p
                for p in (getattr(response, "parse_phases", None) or [])
            ],
            "shots_summary": shot_rows[:100],
            "t2i_provenance_stats": t2i_stats,
        }
        self._store.append_event(
            project_id=self.project_id,
            category="script_parse",
            event_type="parse_completed",
            phase="script_parse",
            status="completed",
            parse_run_id=self.parse_run_id,
            summary=f"Script parse completed · {len(shots)} shot(s)",
            detail=detail,
        )

    def record_failed(self, error: str) -> None:
        if not self.active:
            return
        self._store.append_event(
            project_id=self.project_id,
            category="script_parse",
            event_type="parse_failed",
            phase="script_parse",
            status="failed",
            parse_run_id=self.parse_run_id,
            summary="Script parse failed",
            detail={"error": error[:2000]},
        )


def record_task_activity(
    store: LongVideoActivityStore | None,
    *,
    event_type: str,
    task_id: str,
    task_kind: str,
    model_id: str,
    params: dict[str, Any],
    status: str = "",
    result: dict[str, Any] | None = None,
    error_message: str = "",
) -> None:
    if store is None:
        return
    ctx = extract_long_video_context(params)
    project_id = ctx["project_id"]
    if not project_id:
        return
    phase = ctx["phase"]
    category = category_for_context(phase=phase, task_kind=task_kind)
    detail: dict[str, Any] = {
        "task_kind": task_kind,
        "model_id": model_id,
        "cast_character_id": ctx["cast_character_id"],
        "cast_look_id": ctx["cast_look_id"],
        "scene_id": ctx["scene_id"],
        "scene_look_id": ctx["scene_look_id"],
        "project_title": ctx["project_title"],
        "task_link": f"/api/tasks/{task_id}",
    }
    if result:
        detail["primary_asset_id"] = result.get("primary_asset_id")
        detail["asset_ids"] = result.get("asset_ids") or []
    if error_message:
        detail["error"] = error_message[:2000]
    store.append_event(
        project_id=project_id,
        category=category,
        event_type=event_type,
        phase=phase,
        status=status,
        task_id=task_id,
        shot_id=ctx["shot_id"],
        summary=task_event_summary(
            event_type=event_type,
            phase=phase,
            task_kind=task_kind,
            model_id=model_id,
            shot_id=ctx["shot_id"],
        ),
        detail=detail,
    )
