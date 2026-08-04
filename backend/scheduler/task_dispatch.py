"""Task kind → engine dispatch table for TaskScheduler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
    LoraTrainingRequest,
    VideoAvatarRequest,
    VideoAvatarScriptRequest,
    VideoEditRequest,
    VideoGenerationRequest,
    VideoUpscaleRequest,
    ZImageMergeRequest,
)
import backend.core.task_kinds as TK


@dataclass(frozen=True)
class TaskDispatchSpec:
    request_cls: type
    engine_getter: str  # get_image | get_video | get_audio | get_lora_train
    method_name: str


TASK_DISPATCH: dict[str, TaskDispatchSpec] = {
    TK.IMAGE_GENERATION: TaskDispatchSpec(ImageGenerationRequest, "get_image", "generate"),
    TK.IMAGE_EDIT: TaskDispatchSpec(ImageEditRequest, "get_image", "edit"),
    TK.IMAGE_UPSCALE: TaskDispatchSpec(ImageUpscaleRequest, "get_image", "upscale"),
    TK.VIDEO_GENERATION: TaskDispatchSpec(VideoGenerationRequest, "get_video", "generate"),
    TK.VIDEO_EDIT: TaskDispatchSpec(VideoEditRequest, "get_video", "edit"),
    TK.VIDEO_AVATAR: TaskDispatchSpec(VideoAvatarRequest, "get_video", "avatar"),
    TK.VIDEO_AVATAR_SCRIPT: TaskDispatchSpec(VideoAvatarScriptRequest, "get_video", "avatar_script"),
    TK.VIDEO_UPSCALE: TaskDispatchSpec(VideoUpscaleRequest, "get_video", "upscale"),
    TK.AUDIO_GENERATION: TaskDispatchSpec(AudioGenerationRequest, "get_audio", "generate"),
    TK.AUDIO_EDIT: TaskDispatchSpec(AudioEditRequest, "get_audio", "edit"),
    TK.LORA_TRAINING: TaskDispatchSpec(LoraTrainingRequest, "get_lora_train", "train"),
    TK.TOOLS_Z_IMAGE_MERGE: TaskDispatchSpec(ZImageMergeRequest, "get_tools", "merge_z_image"),
}


async def dispatch_task(
    *,
    kind: str,
    model_id: str,
    params: dict[str, Any],
    engines: Any,
    ctx: Any,
) -> Any:
    spec = TASK_DISPATCH.get(kind)
    if spec is None:
        raise RuntimeError(f"unknown task kind {kind!r}")
    req = spec.request_cls.model_validate(params)
    getter = getattr(engines, spec.engine_getter)
    engine = getter(model_id)
    method: Callable[..., Awaitable[Any]] = getattr(engine, spec.method_name)
    return await method(req, ctx)
