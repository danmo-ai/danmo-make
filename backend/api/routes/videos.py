"""POST /api/videos/* — plan videos.py"""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_engine_registry, get_task_scheduler
from backend.api.routes.submit_helpers import submit_media_task
from backend.core import task_kinds as TK
from backend.core.contracts import (
    VideoAvatarRequest,
    VideoAvatarScriptRequest,
    VideoEditRequest,
    VideoGenerationRequest,
    VideoLongGenerationRequest,
    VideoUpscaleRequest,
)
from backend.long_video.validate import (
    LongVideoValidationError,
    validate_long_video_request,
)
from backend.engine.engine_registry import EngineRegistry
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api/videos", tags=["videos"])


async def _submit_long_video(
    body: VideoLongGenerationRequest,
    sched: TaskScheduler,
    engines: EngineRegistry,
) -> dict:
    spec = body.long_video
    video_engine = engines.get_video(body.model)
    phase = (body.metadata or {}).get("long_video_phase") or ""
    image_engine = None
    if spec.strategy == "segmented_i2v" and phase != "assemble_only":
        image_engine = engines.get_image((spec.keyframe_model or "").strip())
    try:
        validate_long_video_request(
            body, video_engine=video_engine, image_engine=image_engine
        )
    except LongVideoValidationError as exc:
        raise HTTPException(
            exc.http_status, detail={"code": exc.code, "message": exc.message}
        ) from exc

    priority = getattr(body, "priority", "normal")
    r = await sched.submit(
        kind=TK.VIDEO_LONG_GENERATION,
        model_id=body.model,
        params=body.model_dump(),
        priority=priority,
    )
    return {"task": r}


@router.post("/long-generations", status_code=202)
async def post_video_long_generation(
    body: VideoLongGenerationRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await _submit_long_video(body, sched, engines)


@router.post("/generations", status_code=202)
async def post_video_generation(
    body: VideoGenerationRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="video",
        api_action="generate",
        task_kind=TK.VIDEO_GENERATION,
        sched=sched,
        engines=engines,
    )


@router.post("/edits", status_code=202)
async def post_video_edit(
    body: VideoEditRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="video",
        api_action="edit",
        task_kind=TK.VIDEO_EDIT,
        sched=sched,
        engines=engines,
    )


@router.post("/avatars", status_code=202)
async def post_video_avatar(
    body: VideoAvatarRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="video",
        api_action="avatar",
        task_kind=TK.VIDEO_AVATAR,
        sched=sched,
        engines=engines,
    )


@router.post("/avatars/script", status_code=202)
async def post_video_avatar_script(
    body: VideoAvatarScriptRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="video",
        api_action="avatar_script",
        task_kind=TK.VIDEO_AVATAR_SCRIPT,
        sched=sched,
        engines=engines,
    )


@router.post("/upscales", status_code=202)
async def post_video_upscale(
    body: VideoUpscaleRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="video",
        api_action="upscale",
        task_kind=TK.VIDEO_UPSCALE,
        sched=sched,
        engines=engines,
    )
