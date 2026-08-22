"""OpenAI-style multimodal chat → backend_llm sidecar."""

from __future__ import annotations

import logging

from backend.core.contracts import ChatCompletionRequest, ChatCompletionResponse
from backend.engine.llm.asset_messages import cleanup_temp_paths, prepare_messages_for_sidecar
from backend.engine.llm.message_content import messages_have_images
from backend.engine.llm.service import LLMService
from backend.persistence.asset_store import SQLiteAssetStore

logger = logging.getLogger(__name__)


def run_vision_chat_completion(
    service: LLMService,
    store: SQLiteAssetStore,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if not service.is_vision_available():
        raise RuntimeError(
            "Vision model not available. Install a VLM from Models page and set Settings → Default VLM Model."
        )

    prepared, temp_paths = prepare_messages_for_sidecar(store, request.messages)
    try:
        patched = request.model_copy(update={"messages": prepared})
        return service.chat_completion(patched)
    except Exception as exc:
        from backend.engine.llm.message_content import resolve_message_images

        try:
            resolved = resolve_message_images(store, request.messages)
            primary_row = next((row for _path, row, _temp in resolved if row), None)
        except Exception:
            primary_row = None
        if primary_row is not None:
            logger.warning("Vision chat failed, trying text metadata fallback: %s", exc)
            meta = primary_row.get("metadata") or {}
            hint = service._metadata_hint_lines(primary_row, meta)
            note = service._describe_node_from_metadata(hint)
            model_id = str(request.model or service._vision_model_id)
            return service._format_response(note, model_id)
        raise
    finally:
        cleanup_temp_paths(temp_paths)


def run_chat_completion(
    service: LLMService,
    store: SQLiteAssetStore,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if messages_have_images(request.messages):
        return run_vision_chat_completion(service, store, request)
    return service.chat_completion(request)
