"""HTTP OpenAI client for builtin sidecar and cloud providers."""

from __future__ import annotations

import json
import logging
import secrets
import time
from typing import Any, AsyncIterator, Iterator

import httpx

from backend.core.contracts import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatDeltaChoice,
    ChatMessage,
    DeltaMessage,
)
from backend.core.interfaces import AppSettings
from shared.danqing_config.inference import LlmInferenceConfig, load_llm_inference_config

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=30.0)


class LlmOpenAIClient:
    """Sync/async client for ``/v1/chat/completions``."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._config = load_llm_inference_config(self._settings_snapshot())

    def _settings_snapshot(self):
        from shared.danqing_config.settings import AppSettingsSnapshot

        s = self._settings
        return AppSettingsSnapshot(
            default_model_llm=s.default_model_llm,
            default_model_vlm=s.default_model_vlm,
            default_model_llm_think=s.default_model_llm_think,
            llm_unload_each_request=s.llm_unload_each_request,
            model_cache_ttl_minutes=s.model_cache_ttl_minutes,
            llm_cache_ttl_minutes=s.llm_cache_ttl_minutes,
            llm_inference_provider=(
                "openai_compatible"
                if s.llm_inference_provider == "openai_compatible"
                else "builtin"
            ),
            llm_inference_base_url=s.llm_inference_base_url,
            llm_inference_api_key=s.llm_inference_api_key,
            llm_inference_api_key_hint=s.llm_inference_api_key_hint,
            llm_inference_cloud_model=s.llm_inference_cloud_model,
            llm_builtin_host=s.llm_builtin_host,
            llm_builtin_port=s.llm_builtin_port,
            llm_quantize_activations=s.llm_quantize_activations,
        )

    def refresh_config(self) -> None:
        self._config = load_llm_inference_config(self._settings_snapshot())

    @property
    def config(self) -> LlmInferenceConfig:
        return self._config

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = (self._config.api_key or "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    @staticmethod
    def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            body: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.reasoning:
                body["reasoning"] = msg.reasoning
            out.append(body)
        return out

    def _payload(
        self,
        request: ChatCompletionRequest,
        *,
        model: str,
        stream: bool,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._serialize_messages(request.messages),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        return payload

    def chat_completion(
        self,
        request: ChatCompletionRequest,
        *,
        model: str,
        enable_thinking: bool | None = None,
    ) -> ChatCompletionResponse:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(
            request,
            model=model,
            stream=False,
            enable_thinking=enable_thinking,
        )
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM inference failed ({resp.status_code}): {resp.text[:500]}"
                )
            data = resp.json()
        return self._parse_completion(data, fallback_model=model)

    def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        *,
        model: str,
        enable_thinking: bool | None = None,
    ) -> Iterator[str]:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(
            request,
            model=model,
            stream=True,
            enable_thinking=enable_thinking,
        )
        with httpx.Client(timeout=_DEFAULT_TIMEOUT) as client:
            with client.stream(
                "POST",
                url,
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"LLM inference stream failed ({resp.status_code}): {body[:500]}"
                    )
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        yield f"{line}\n\n"
                    elif line.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"

    async def chat_completion_stream_async(
        self,
        request: ChatCompletionRequest,
        *,
        model: str,
        enable_thinking: bool | None = None,
    ) -> AsyncIterator[str]:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        payload = self._payload(
            request,
            model=model,
            stream=True,
            enable_thinking=enable_thinking,
        )
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as client:
            async with client.stream(
                "POST",
                url,
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"LLM inference stream failed ({resp.status_code}): {body[:500]}"
                    )
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        yield f"{line}\n\n"
                    elif line.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"

    def health_ok(self) -> bool:
        base = self._config.base_url.rstrip("/")
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        url = f"{root}/health"
        try:
            with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
                resp = client.get(url)
                if resp.status_code != 200:
                    return False
                data = resp.json()
                return str(data.get("status", "")).lower() == "healthy"
        except Exception:
            return False

    def unload_model(self) -> None:
        if self._config.provider != "builtin":
            return
        base = self._config.base_url.rstrip("/")
        root = base[: -len("/v1")] if base.endswith("/v1") else base
        url = f"{root}/unload"
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
                client.post(url)
        except Exception as exc:
            logger.warning("Sidecar unload failed: %s", exc)

    @staticmethod
    def _parse_completion(data: dict[str, Any], *, fallback_model: str) -> ChatCompletionResponse:
        choices_raw = data.get("choices") or []
        choices: list[ChatChoice] = []
        for idx, item in enumerate(choices_raw):
            msg_raw = item.get("message") or {}
            message = ChatMessage(
                role=msg_raw.get("role") or "assistant",
                content=msg_raw.get("content") or "",
                reasoning=msg_raw.get("reasoning"),
            )
            choices.append(
                ChatChoice(
                    index=int(item.get("index", idx)),
                    message=message,
                    finish_reason=item.get("finish_reason") or "stop",
                )
            )
        if not choices:
            choices.append(
                ChatChoice(
                    message=ChatMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            )
        return ChatCompletionResponse(
            id=str(data.get("id") or f"chatcmpl-{secrets.token_hex(12)}"),
            object=str(data.get("object") or "chat.completion"),
            created=int(data.get("created") or time.time()),
            model=str(data.get("model") or fallback_model),
            choices=choices,
            usage=data.get("usage") or {},
        )

    @staticmethod
    def parse_stream_chunk(line: str, *, model: str) -> ChatCompletionChunk | None:
        text = line.strip()
        if not text.startswith("data:"):
            return None
        payload = text[5:].strip()
        if payload == "[DONE]":
            return None
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        choices_raw = data.get("choices") or []
        delta_choices: list[ChatDeltaChoice] = []
        for idx, item in enumerate(choices_raw):
            delta_raw = item.get("delta") or {}
            delta_choices.append(
                ChatDeltaChoice(
                    index=int(item.get("index", idx)),
                    delta=DeltaMessage(
                        role=delta_raw.get("role"),
                        content=delta_raw.get("content"),
                        reasoning=delta_raw.get("reasoning"),
                    ),
                    finish_reason=item.get("finish_reason"),
                )
            )
        return ChatCompletionChunk(
            id=str(data.get("id") or f"chatcmpl-{secrets.token_hex(8)}"),
            created=int(data.get("created") or time.time()),
            model=str(data.get("model") or model),
            choices=delta_choices,
        )
