"""Shared REST submit helper for media generation routes."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import HTTPException

from backend.core.contracts import BaseModel
from backend.core.media_roles import reject_request_media
from backend.engine.engine_registry import EngineRegistry
from backend.scheduler.task_scheduler import TaskScheduler


async def submit_media_task(
    *,
    body: BaseModel,
    media: Literal["image", "video", "audio"],
    api_action: str,
    task_kind: str,
    sched: TaskScheduler,
    engines: EngineRegistry,
) -> dict[str, Any]:
    model_id = getattr(body, "model", None)
    if not model_id:
        raise HTTPException(400, detail={"code": "invalid", "message": "model is required"})
    if media == "image":
        eng = engines.get_image(model_id)
    elif media == "video":
        eng = engines.get_video(model_id)
    else:
        eng = engines.get_audio(model_id)
    if not eng.supports(model_id, api_action):
        raise HTTPException(
            409,
            detail={"code": "unsupported", "message": f"model does not support {api_action}"},
        )
    entry = engines.model_registry.get(model_id)
    if entry is not None:
        rejected = reject_request_media(entry, body, media=media, api_action=api_action)
        if rejected is not None:
            status = 409 if rejected.code == "unsupported" else 400
            raise HTTPException(
                status,
                detail={"code": rejected.code, "message": rejected.message},
            )
    priority = getattr(body, "priority", "normal")
    r = await sched.submit(
        kind=task_kind,
        model_id=model_id,
        params=body.model_dump(),
        priority=priority,
    )
    return {"task": r}
