"""Resolve asset: URLs in chat messages for sidecar forwarding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import ChatMessage
from backend.engine.llm.message_content import (
    _content_parts,
    iter_image_urls,
    resolve_image_url,
)
from backend.persistence.asset_store import SQLiteAssetStore


def prepare_messages_for_sidecar(
    store: SQLiteAssetStore,
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], list[Path]]:
    """Rewrite ``image_url`` parts to local filesystem paths; track temp files."""
    temp_paths: list[Path] = []
    out: list[ChatMessage] = []

    for msg in messages:
        parts = _content_parts(msg.content)
        if isinstance(msg.content, str) or not any(p.get("type") == "image_url" for p in parts):
            out.append(msg)
            continue

        new_parts: list[dict[str, Any]] = []
        for part in parts:
            if part.get("type") != "image_url":
                new_parts.append(part)
                continue
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url") or "").strip()
            path, _row, is_temp = resolve_image_url(store, url)
            if is_temp:
                temp_paths.append(path)
            new_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(path.resolve())},
                }
            )
        out.append(ChatMessage(role=msg.role, content=new_parts))

    if iter_image_urls(messages) and not any(
        p.get("type") == "image_url" for m in out for p in _content_parts(m.content)
    ):
        raise ValueError("Vision request requires at least one image_url in messages")
    return out, temp_paths


def cleanup_temp_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
