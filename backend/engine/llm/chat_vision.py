"""OpenAI-style multimodal chat → local MLX-VLM inference."""

from __future__ import annotations

import logging
from pathlib import Path

from backend.core.contracts import ChatCompletionRequest, ChatCompletionResponse
from backend.engine.llm.message_content import (
    extract_vision_instruction,
    resolve_message_images,
)
from backend.engine.llm.service_mlx import LLMService
from backend.engine.llm.vision_mlx import analyze_image_file, analyze_images_multi
from backend.persistence.asset_store import SQLiteAssetStore

logger = logging.getLogger(__name__)


def _metadata_hint(service: LLMService, row: dict | None) -> str:
    if not row:
        return ""
    meta = row.get("metadata") or {}
    return service._metadata_hint_lines(row, meta)


def _cleanup_temp_paths(resolved: list[tuple[Path, dict | None, bool]]) -> None:
    for path, _row, is_temp in resolved:
        if not is_temp:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def run_vision_chat_completion(
    service: LLMService,
    store: SQLiteAssetStore,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    if not service.is_vision_available():
        raise RuntimeError(
            "Vision model not available. Install a VLM from Models page and set Settings → Default VLM Model."
        )

    instruction = extract_vision_instruction(request.messages)
    resolved = resolve_message_images(store, request.messages)
    paths = [item[0] for item in resolved]
    primary_row = next((row for _path, row, _temp in resolved if row), None)
    metadata_hint = _metadata_hint(service, primary_row)
    model_dir = service._resolve_vision_model_path()
    max_tokens = min(int(request.max_tokens or 384), 8192)

    try:
        with service._generation_lock:
            if len(paths) == 1:
                text = analyze_image_file(
                    paths[0],
                    model_dir,
                    instruction=instruction,
                    metadata_hint=metadata_hint,
                    max_tokens=max_tokens,
                    temperature=float(request.temperature),
                )
            else:
                text = analyze_images_multi(
                    paths,
                    model_dir,
                    instruction=instruction,
                    metadata_hint=metadata_hint,
                    max_tokens=max_tokens,
                    temperature=float(request.temperature),
                )
    except Exception as exc:
        if primary_row is not None:
            logger.warning("Vision chat failed, trying text metadata fallback: %s", exc)
            note = service._describe_node_from_metadata(_metadata_hint(service, primary_row))
            return service._format_response(note, request.model or service._vision_model_id)
        raise
    finally:
        _cleanup_temp_paths(resolved)

    cleaned = service._clean_vlm_prompt(text)
    model_id = str(request.model or service._vision_model_id)
    return service._format_response(cleaned, model_id)
