"""
LLM / VLM routes — OpenAI-compatible chat completions.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.deps import get_asset_store, get_llm_service
from backend.api.routes.settings import get_settings_service
from backend.core.contracts import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EnhanceRequest,
    EnhanceResponse,
)
from backend.engine.llm import LLMService
from backend.engine.llm.chat_vision import run_vision_chat_completion
from backend.engine.llm.message_content import (
    flatten_messages_for_text_llm,
    messages_have_images,
)
from backend.core.i18n import resolve_locale
from backend.persistence.asset_store import SQLiteAssetStore

router = APIRouter()


def _resolve_locale(http_request: Request, body_locale: str | None) -> str:
    loc = (body_locale or "").strip().lower().split("-")[0]
    if loc in ("zh", "en"):
        return loc
    try:
        settings = get_settings_service().get_settings()
        lang = getattr(settings, "language", None)
        if lang in ("zh", "en"):
            return str(lang)
    except Exception:
        pass
    return resolve_locale(http_request.headers.get("Accept-Language"))


def _http_error_from_task(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, RuntimeError):
        status = 503 if "not installed" in str(exc).lower() or "not available" in str(exc).lower() else 502
        return HTTPException(status_code=status, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


@router.post("/api/chat/enhance-prompt", response_model=EnhanceResponse)
async def enhance_prompt(
    request: EnhanceRequest,
    service: LLMService = Depends(get_llm_service),
):
    """Polish a creative brief for image, video, or audio generation."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )
    try:
        return await asyncio.to_thread(service.enhance_prompt, request)
    except Exception as exc:
        raise _http_error_from_task(exc) from exc


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """OpenAI-compatible chat completions (text and multimodal vision via ``messages``)."""
    if messages_have_images(request.messages):
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming is not supported for vision (image_url) messages",
            )
        if not service.is_vision_available():
            raise HTTPException(
                status_code=503,
                detail="Vision model not available. Install a VLM from Models page.",
            )
        try:
            result = await asyncio.to_thread(run_vision_chat_completion, service, store, request)
        except Exception as exc:
            raise _http_error_from_task(exc) from exc
        if result.object != "chat.completion":
            result = result.model_copy(update={"object": "chat.completion"})
        return result

    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )

    text_request = request.model_copy(
        update={"messages": flatten_messages_for_text_llm(request.messages)},
    )

    if request.stream:
        return StreamingResponse(
            service.chat_completion_stream(text_request),
            media_type="text/event-stream",
        )

    result = await asyncio.to_thread(service.chat_completion, text_request)
    if result.object != "chat.completion":
        result = result.model_copy(update={"object": "chat.completion"})
    return result


@router.get("/v1/llm/model")
def get_llm_model_info(service: LLMService = Depends(get_llm_service)):
    """Current default text LLM model."""
    return service.get_model_info()


@router.get("/v1/vision/model")
def get_vision_model_info(service: LLMService = Depends(get_llm_service)):
    """Current default VLM model."""
    return service.get_vision_model_info()


@router.post("/api/chat/long-video-storyboard")
async def long_video_storyboard(http_request: Request, service: LLMService = Depends(get_llm_service)):
    """Moved to Danmo Film."""
    del http_request, service
    raise HTTPException(
        status_code=410,
        detail="Long-video storyboard moved to Danmo Film (POST /api/script-parse/* on Film :7803).",
    )


@router.post("/api/chat/long-video-chapter-analyze")
async def long_video_chapter_analyze(http_request: Request, service: LLMService = Depends(get_llm_service)):
    """Moved to Danmo Film."""
    del http_request, service
    raise HTTPException(
        status_code=410,
        detail="Long-video chapter analyze moved to Danmo Film (script-parse on Film :7803).",
    )


@router.post("/api/chat/long-video-chapter-analyze/stream")
async def long_video_chapter_analyze_stream(http_request: Request, service: LLMService = Depends(get_llm_service)):
    """Moved to Danmo Film."""
    del http_request, service
    raise HTTPException(
        status_code=410,
        detail="Long-video chapter analyze stream moved to Danmo Film (script-parse SSE on Film :7803).",
    )
