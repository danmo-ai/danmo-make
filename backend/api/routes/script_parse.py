"""Script parse REST + SSE routes (4-pass pipeline)."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.deps import get_llm_service, get_long_video_activity_store
from backend.api.routes.llm import _http_error_from_task, _resolve_locale
from backend.core.contracts import ScriptParseDecomposeRequest, ScriptParseExpandBeatRequest, ScriptParseExpandRequest
from backend.long_video.activity import LongVideoActivityRecorder
from backend.engine.llm import LLMService
from backend.persistence.long_video_activity_store import LongVideoActivityStore

router = APIRouter()


def _activity_recorder(
    project_id: str,
    store: LongVideoActivityStore,
) -> LongVideoActivityRecorder | None:
    pid = (project_id or "").strip()
    if not pid:
        return None
    return LongVideoActivityRecorder(store, pid)


@router.post("/api/script-parse/decompose")
async def script_parse_decompose(
    request: ScriptParseDecomposeRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not service.is_available():
        raise HTTPException(status_code=503, detail="LLM model not installed. Install via Models page.")
    locale = _resolve_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    recorder = _activity_recorder(req.long_video_project_id, activity_store)
    try:
        return await asyncio.to_thread(
            service.script_parse_decompose,
            req,
            activity_recorder=recorder,
        )
    except (RuntimeError, ValueError) as exc:
        if recorder is not None and recorder.active:
            recorder.record_failed(str(exc))
        raise _http_error_from_task(exc) from exc


@router.post("/api/script-parse/decompose/stream")
async def script_parse_decompose_stream(
    request: ScriptParseDecomposeRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not service.is_available():
        raise HTTPException(status_code=503, detail="LLM model not installed. Install via Models page.")
    locale = _resolve_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    recorder = _activity_recorder(req.long_video_project_id, activity_store)
    return StreamingResponse(
        service.script_parse_decompose_stream(req, activity_recorder=recorder),
        media_type="text/event-stream",
    )


@router.post("/api/script-parse/expand")
async def script_parse_expand(
    request: ScriptParseExpandRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not service.is_available():
        raise HTTPException(status_code=503, detail="LLM model not installed. Install via Models page.")
    locale = _resolve_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    recorder = _activity_recorder(req.long_video_project_id, activity_store)
    try:
        return await asyncio.to_thread(
            service.script_parse_expand,
            req,
            activity_recorder=recorder,
        )
    except (RuntimeError, ValueError) as exc:
        if recorder is not None and recorder.active:
            recorder.record_failed(str(exc))
        raise _http_error_from_task(exc) from exc


@router.post("/api/script-parse/expand/stream")
async def script_parse_expand_stream(
    request: ScriptParseExpandRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not service.is_available():
        raise HTTPException(status_code=503, detail="LLM model not installed. Install via Models page.")
    locale = _resolve_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    recorder = _activity_recorder(req.long_video_project_id, activity_store)
    return StreamingResponse(
        service.script_parse_expand_stream(req, activity_recorder=recorder),
        media_type="text/event-stream",
    )


@router.post("/api/script-parse/expand/beat")
async def script_parse_expand_beat(
    request: ScriptParseExpandBeatRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not service.is_available():
        raise HTTPException(status_code=503, detail="LLM model not installed. Install via Models page.")
    locale = _resolve_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    recorder = _activity_recorder(req.long_video_project_id, activity_store)
    try:
        return await asyncio.to_thread(
            service.script_parse_expand_beat,
            req,
            activity_recorder=recorder,
        )
    except (RuntimeError, ValueError) as exc:
        if recorder is not None and recorder.active:
            recorder.record_failed(str(exc))
        raise _http_error_from_task(exc) from exc


@router.post("/api/script-parse/expand/beat/stream")
async def script_parse_expand_beat_stream(
    request: ScriptParseExpandBeatRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
    activity_store: LongVideoActivityStore = Depends(get_long_video_activity_store),
):
    if not service.is_available():
        raise HTTPException(status_code=503, detail="LLM model not installed. Install via Models page.")
    locale = _resolve_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    recorder = _activity_recorder(req.long_video_project_id, activity_store)
    return StreamingResponse(
        service.script_parse_expand_beat_stream(req, activity_recorder=recorder),
        media_type="text/event-stream",
    )
