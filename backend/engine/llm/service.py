"""
LLMService — HTTP client to backend_llm sidecar or cloud OpenAI-compatible API.

Product orchestration (enhance, diagnose, lyrics) stays here; protocol + MLX live in backend_llm.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable

from backend.core.bundle_manifest import missing_safetensor_shards
from backend.core.contracts import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    EnhanceRequest,
    EnhanceResponse,
)
from backend.core.i18n import resolve_locale
from backend.core.interfaces import AppSettings
from backend.core.model_registry import ModelRegistry
from backend.engine.llm.llm_settings import (
    DEFAULT_ASSISTANT_MODEL_ID,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_VLM_MODEL_ID,
    require_multimodal_assistant_model,
)
from backend.engine.llm.lyrics_sanitize import lyric_line_has_annotations, sanitize_lyrics_output
from backend.engine.llm.openai_client import LlmOpenAIClient
from backend.engine.llm.prompt_sanitize import (
    prompt_enhance_fidelity_ok,
    prompt_enhance_quality_ok,
    sanitize_enhanced_prompt,
)
from backend.engine.llm.prompts.locale import enhance_user_locale_hint
from backend.engine.llm.prompts.system import (
    DESCRIBE_NODE_SYSTEM_PROMPT,
    ENHANCE_AUDIO_BRIEF_SYSTEM_PROMPT,
    ENHANCE_IMAGE_SYSTEM_PROMPT,
    ENHANCE_VIDEO_SYSTEM_PROMPT,
    IMAGE_TO_PROMPT_INSTRUCTION,
    LONG_VIDEO_OPENING_SYSTEM_PROMPT,
    LYRICS_SYSTEM_PROMPT,
    VIDEO_FRAME_TO_PROMPT_INSTRUCTION,
)
from backend.engine.llm.sidecar_manager import LlmSidecarManager
from backend.engine.llm.think_parse import extract_final_llm_content
from backend.engine.llm.vlm_http import analyze_image_file, vision_inference_available
from backend.utils.path_utils import PathResolver
from shared.danqing_config.inference import cloud_inference_ready, load_llm_inference_config
from shared.danqing_config.llm import is_thinking_model
from shared.danqing_config.settings import AppSettingsSnapshot

logger = logging.getLogger(__name__)

_ASSISTANT_IDLE_CHECK_S = 30.0


class LLMService:
    """Assistant LLM/VLM via OpenAI HTTP (builtin sidecar or cloud)."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        path_resolver: PathResolver,
        default_model_id: str = DEFAULT_LLM_MODEL_ID,
        vision_model_id: str = DEFAULT_VLM_MODEL_ID,
        llm_think_enabled: bool = False,
        *,
        llm_cache_ttl_minutes: int | None = None,
        unload_each_request: bool | None = None,
        settings_provider: Callable[[], AppSettings] | None = None,
        sidecar_manager: LlmSidecarManager | None = None,
    ):
        self._registry = model_registry
        self._path_resolver = path_resolver
        self._model_id = default_model_id
        self._vision_model_id = vision_model_id
        self._llm_think_enabled = bool(llm_think_enabled)
        self._generation_lock = threading.Lock()
        self._settings_provider = settings_provider or (lambda: AppSettings())
        self._sidecar = sidecar_manager or LlmSidecarManager()
        self._llm_cache_ttl_minutes = max(1, int(llm_cache_ttl_minutes or 30))
        self._unload_each_request = bool(unload_each_request or False)
        self._last_assistant_activity: float | None = None
        self._cleanup_stop = threading.Event()
        self._cleanup_thread: threading.Thread | None = None

    def _settings(self) -> AppSettings:
        return self._settings_provider()

    def _snapshot(self, settings: AppSettings | None = None) -> AppSettingsSnapshot:
        s = settings or self._settings()
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

    def _client(self, *, skip_idle_check: bool = False) -> LlmOpenAIClient:
        if not skip_idle_check:
            self._maybe_purge_idle_assistant()
        settings = self._settings()
        if self._sidecar.should_manage_sidecar(settings):
            self._sidecar.ensure_running(
                host=(settings.llm_builtin_host or "127.0.0.1").strip(),
                port=max(1, int(settings.llm_builtin_port or 7801)),
            )
        client = LlmOpenAIClient(settings)
        client.refresh_config()
        return client

    def start_idle_cleanup(self) -> None:
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        self._cleanup_stop.clear()
        self._cleanup_thread = threading.Thread(
            target=self._idle_cleanup_loop,
            name="llm-assistant-idle-purge",
            daemon=True,
        )
        self._cleanup_thread.start()

    def stop_idle_cleanup(self) -> None:
        self._cleanup_stop.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2.0)
            self._cleanup_thread = None

    def _idle_cleanup_loop(self) -> None:
        while not self._cleanup_stop.wait(_ASSISTANT_IDLE_CHECK_S):
            try:
                self._maybe_purge_idle_assistant()
            except Exception:
                logger.exception("Assistant idle purge failed")

    def set_memory_policy(
        self,
        *,
        llm_cache_ttl_minutes: int | None = None,
        unload_each_request: bool | None = None,
    ) -> None:
        if llm_cache_ttl_minutes is not None:
            self._llm_cache_ttl_minutes = max(1, int(llm_cache_ttl_minutes))
        if unload_each_request is not None:
            self._unload_each_request = bool(unload_each_request)

    def _touch_assistant_activity(self) -> None:
        self._last_assistant_activity = time.monotonic()

    def _assistant_idle_minutes(self) -> float | None:
        if self._last_assistant_activity is None:
            return None
        return (time.monotonic() - self._last_assistant_activity) / 60.0

    def _maybe_purge_idle_assistant(self) -> None:
        if self._unload_each_request:
            return
        settings = self._settings()
        if settings.llm_inference_provider != "builtin":
            return
        idle = self._assistant_idle_minutes()
        if idle is None or idle < self._llm_cache_ttl_minutes:
            return
        logger.info(
            "Unloading assistant model after %.1f min idle (ttl=%s min)",
            idle,
            self._llm_cache_ttl_minutes,
        )
        self.unload_assistant_model()
        self._last_assistant_activity = None

    def unload_assistant_model(self) -> None:
        settings = self._settings()
        if settings.llm_inference_provider != "builtin":
            return
        if not self._sidecar.health_ok():
            return
        try:
            client = LlmOpenAIClient(settings)
            client.refresh_config()
            client.unload_model()
        except Exception as exc:
            logger.warning("Sidecar unload failed: %s", exc)

    def _finish_assistant_use(self) -> None:
        self._touch_assistant_activity()
        if self._unload_each_request:
            self.unload_assistant_model()

    def unload_text_model(self) -> None:
        """Backward-compatible alias."""
        self.unload_assistant_model()

    def _assistant_model_id(self) -> str:
        return self._model_id

    def _require_builtin_multimodal(self) -> None:
        settings = self._settings()
        if settings.llm_inference_provider == "openai_compatible":
            return
        require_multimodal_assistant_model(self._model_id, self._registry)

    def builtin_assistant_block_reason(self) -> str | None:
        """Human-readable reason when builtin assistant cannot run, or None if OK."""
        settings = self._settings()
        if settings.llm_inference_provider == "openai_compatible":
            if not cloud_inference_ready(self._snapshot(settings)):
                return "Cloud LLM inference is not configured (base URL and API key required)."
            return None
        try:
            require_multimodal_assistant_model(self._model_id, self._registry)
            path = self._resolve_model_path()
            if not self._llm_weights_ready(path):
                return (
                    f"Assistant model {self._model_id!r} is not installed. "
                    "Install it from the Models page."
                )
        except RuntimeError as exc:
            return str(exc)
        except Exception as exc:
            return str(exc)
        return None

    def apply_model_settings(
        self,
        *,
        default_model_id: str | None = None,
        vision_model_id: str | None = None,
        llm_think_enabled: bool | None = None,
    ) -> None:
        chosen = default_model_id if default_model_id is not None else vision_model_id
        if chosen is not None:
            from backend.engine.llm.llm_settings import _resolve_assistant_model_id

            coerced = _resolve_assistant_model_id(chosen, self._registry)
            self._registry.require(coerced)
            self._model_id = coerced
            self._vision_model_id = coerced
        if llm_think_enabled is not None:
            self._llm_think_enabled = bool(llm_think_enabled)
        if not is_thinking_model(self._model_id):
            self._llm_think_enabled = False

    def get_model_info(self) -> dict[str, Any]:
        entry = self._registry.get(self._model_id)
        return {
            "model_id": self._model_id,
            "name": self._registry_display_name(entry, self._model_id),
            "available": self.is_available(),
            "provider": self._settings().llm_inference_provider,
            "unload_each_request": self._unload_each_request,
            "llm_cache_ttl_minutes": self._llm_cache_ttl_minutes,
            "think_supported": is_thinking_model(self._model_id),
            "think_enabled": self._llm_think_enabled,
        }

    def get_vision_model_info(self) -> dict[str, Any]:
        entry = self._registry.get(self._vision_model_id)
        return {
            "model_id": self._vision_model_id,
            "name": self._registry_display_name(entry, self._vision_model_id),
            "available": self.is_vision_available(),
            "provider": self._settings().llm_inference_provider,
        }

    def is_vision_available(self) -> bool:
        settings = self._settings()
        if settings.llm_inference_provider == "openai_compatible":
            return cloud_inference_ready(self._snapshot(settings))
        try:
            self._require_builtin_multimodal()
            path = self._resolve_vision_model_path()
            return vision_inference_available(settings, path)
        except Exception:
            return False

    def is_available(self) -> bool:
        settings = self._settings()
        if settings.llm_inference_provider == "openai_compatible":
            return cloud_inference_ready(self._snapshot(settings))
        try:
            self._require_builtin_multimodal()
            path = self._resolve_model_path()
            return self._llm_weights_ready(path)
        except Exception:
            return False

    @staticmethod
    def _llm_weights_ready(path: Path) -> bool:
        if not path.is_dir():
            return False
        if missing_safetensor_shards(path):
            return False
        return (
            (path / "model.safetensors").is_file()
            or any(f.suffix == ".safetensors" for f in path.rglob("*") if f.is_file())
            or any(f.suffix == ".bin" for f in path.rglob("*") if f.is_file())
        )

    def _resolve_openai_model(self, registry_id: str, *, vision: bool = False) -> str:
        settings = self._settings()
        if settings.llm_inference_provider == "openai_compatible":
            cloud = (settings.llm_inference_cloud_model or "").strip()
            if cloud:
                return cloud
            return registry_id
        path = self._resolve_model_path(registry_id)
        return str(path.resolve())

    def _resolve_request_registry_id(
        self,
        request: ChatCompletionRequest,
        *,
        has_images: bool,
    ) -> str:
        preferred = (request.model or "").strip()
        if preferred:
            entry = self._registry.get(preferred.split(":", 1)[0])
            if entry is not None:
                return preferred.split(":", 1)[0]
        return self._model_id

    def chat_completion(
        self,
        request: ChatCompletionRequest,
        *,
        enable_thinking: bool | None = None,
    ) -> ChatCompletionResponse:
        from backend.engine.llm.message_content import messages_have_images

        has_images = messages_have_images(request.messages)
        self._require_builtin_multimodal()
        registry_id = self._resolve_request_registry_id(request, has_images=has_images)
        thinking = self._resolve_enable_thinking_for(registry_id, enable_thinking)
        think_active = self._think_is_active_for(registry_id, thinking)
        patched = ChatCompletionRequest(
            model=registry_id,
            messages=self._apply_think_mode_to_messages_for(registry_id, request.messages),
            temperature=request.temperature,
            max_tokens=self._token_budget(request.max_tokens or 512, think_active),
            stream=False,
            top_p=request.top_p,
        )
        openai_model = self._resolve_openai_model(registry_id, vision=has_images)
        self._touch_assistant_activity()
        with self._generation_lock:
            result = self._client().chat_completion(
                patched,
                model=openai_model,
                enable_thinking=thinking,
            )
        if think_active and result.choices:
            content = extract_final_llm_content(
                result.choices[0].message.content,
                think_enabled=True,
            )
            msg = result.choices[0].message
            result.choices[0] = ChatChoice(
                index=result.choices[0].index,
                message=ChatMessage(
                    role=msg.role,
                    content=content,
                    reasoning=msg.reasoning,
                ),
                finish_reason=result.choices[0].finish_reason,
            )
        self._finish_assistant_use()
        return result.model_copy(update={"model": registry_id})

    async def chat_completion_stream(self, request: ChatCompletionRequest):
        from backend.engine.llm.message_content import messages_have_images

        has_images = messages_have_images(request.messages)
        self._require_builtin_multimodal()
        registry_id = self._resolve_request_registry_id(
            request,
            has_images=has_images,
        )
        thinking = self._resolve_enable_thinking_for(registry_id, None)
        think_active = self._think_is_active_for(registry_id, thinking)
        patched = ChatCompletionRequest(
            model=registry_id,
            messages=self._apply_think_mode_to_messages_for(registry_id, request.messages),
            temperature=request.temperature,
            max_tokens=self._token_budget(request.max_tokens or 512, think_active),
            stream=True,
            top_p=request.top_p,
        )
        openai_model = self._resolve_openai_model(registry_id, vision=has_images)
        self._touch_assistant_activity()
        client = self._client()
        async for chunk in client.chat_completion_stream_async(
            patched,
            model=openai_model,
            enable_thinking=thinking,
        ):
            yield chunk
        self._finish_assistant_use()

    def enhance_prompt(self, request: EnhanceRequest) -> EnhanceResponse:
        self._require_builtin_multimodal()
        system_prompt = self._enhance_system_prompt(request.target_action)
        action = (request.target_action or "image_create").strip().lower()
        raw_prompt = (request.prompt or "").strip()
        user_content = raw_prompt
        if action in ("image_create", "create", "image"):
            user_content += enhance_user_locale_hint(raw_prompt)
        style = (request.style_positive or "").strip()
        if style:
            user_content += f"\n\nStyle cues to weave in: {style}"
        user_content = self._apply_think_mode_to_text(user_content)

        think_active = self._think_is_active(self._resolve_enable_thinking(None))
        attempts = (
            (0.65, self._token_budget(200, think_active)),
            (0.45, self._token_budget(160, think_active)),
            (0.35, self._token_budget(140, think_active)),
        )
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        last_clean = ""
        thinking = self._resolve_enable_thinking(None)
        for temperature, max_tokens in attempts:
            internal = ChatCompletionRequest(
                model=self._model_id,
                messages=messages,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens,
                stream=False,
            )
            result = self.chat_completion(internal, enable_thinking=thinking)
            cleaned = sanitize_enhanced_prompt(
                result.choices[0].message.content,
                think_enabled=think_active,
            )
            if prompt_enhance_quality_ok(cleaned, original=raw_prompt):
                return EnhanceResponse(enhanced_prompt=cleaned)
            last_clean = cleaned

        fallback = sanitize_enhanced_prompt(raw_prompt, think_enabled=think_active)
        if prompt_enhance_quality_ok(last_clean, original=raw_prompt):
            return EnhanceResponse(enhanced_prompt=last_clean)
        if prompt_enhance_fidelity_ok(raw_prompt, last_clean):
            return EnhanceResponse(enhanced_prompt=last_clean)
        return EnhanceResponse(enhanced_prompt=fallback or raw_prompt)

    def generate_lyrics(self, prompt: str, style: str | None = None) -> str:
        user_msg = self._with_no_think_suffix(self._build_lyrics_user_message(prompt, style))
        attempts = ((0.65, 420), (0.5, 360), (0.4, 512))
        last_raw = ""
        for temp, max_tokens in attempts:
            result = self.chat_completion(
                self._lyrics_chat_request(user_msg, temperature=temp, max_tokens=max_tokens),
                enable_thinking=False,
            )
            last_raw = result.choices[0].message.content.strip()
            cleaned = sanitize_lyrics_output(last_raw, think_enabled=False)
            if self._lyrics_quality_ok(cleaned):
                return cleaned
        fallback = sanitize_lyrics_output(last_raw, think_enabled=False)
        if fallback and self._lyrics_quality_ok(fallback):
            return fallback
        raise RuntimeError(
            "LLM returned no usable lyrics. Check Settings → Default LLM Model or try again."
        )

    def _format_response(self, content: str, model_id: str) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{secrets.token_hex(12)}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
        )

    def _describe_node_from_metadata(self, metadata_hint: str) -> str:
        user_msg = f"Asset metadata:\n{metadata_hint}\n\nWrite a canvas node note:"
        internal = ChatCompletionRequest(
            model=self._model_id,
            messages=[
                ChatMessage(role="system", content=DESCRIBE_NODE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ],
            temperature=0.6,
            max_tokens=256,
            stream=False,
        )
        result = self.chat_completion(internal)
        return result.choices[0].message.content.strip()

    @staticmethod
    def _clean_vlm_prompt(text: str) -> str:
        t = text.strip()
        if t.lower().startswith("prompt:"):
            t = t.split(":", 1)[1].strip()
        return t

    @staticmethod
    def _metadata_hint_lines(asset_context: dict[str, Any], meta: dict[str, Any]) -> str:
        lines = [
            f"Kind: {asset_context.get('kind', 'image')}",
            f"Title: {meta.get('title') or ''}",
            f"Prompt: {meta.get('prompt') or ''}",
            f"Model: {meta.get('model') or ''}",
            f"Size: {asset_context.get('width') or meta.get('width')}x"
            f"{asset_context.get('height') or meta.get('height')}",
            f"Source action: {asset_context.get('source_action') or ''}",
            f"Relation: {asset_context.get('relation_type') or ''}",
        ]
        if asset_context.get("duration_seconds"):
            lines.append(f"Duration (s): {asset_context.get('duration_seconds')}")
        if asset_context.get("parent_asset_id"):
            lines.append(f"Parent asset: {asset_context.get('parent_asset_id')}")
        return "\n".join(lines)

    def _resolve_vision_model_path(self) -> Path:
        return self._resolve_model_path(self._vision_model_id)

    def _resolve_model_path(self, model_id: str | None = None) -> Path:
        mid = (model_id or self._model_id).strip() or self._model_id
        entry = self._registry.require(mid)
        versions = entry.raw.get("versions") or {}
        default_ver = next(
            (v for v in versions.values() if v.get("default")),
            next(iter(versions.values()), None),
        )
        if default_ver is None:
            raise RuntimeError(f"No versions defined for model {mid!r} in registry")
        local_path = default_ver.get("local_path")
        if not local_path:
            raise RuntimeError(f"No local_path for default version of {mid!r}")
        return self._path_resolver.resolve_registry_local_path(local_path)

    @staticmethod
    def _registry_display_name(entry: Any, fallback: str) -> Any:
        if entry is None:
            return fallback
        raw_name = entry.raw.get("name")
        return raw_name if raw_name is not None else fallback

    @staticmethod
    def _enhance_system_prompt(target_action: str) -> str:
        action = (target_action or "image_create").strip().lower()
        if action in ("video", "video_create", "animate", "video_generation"):
            return ENHANCE_VIDEO_SYSTEM_PROMPT
        if action in ("long_video_opening",):
            return LONG_VIDEO_OPENING_SYSTEM_PROMPT
        if action in ("audio", "audio_create", "music", "audio_generation"):
            return ENHANCE_AUDIO_BRIEF_SYSTEM_PROMPT
        return ENHANCE_IMAGE_SYSTEM_PROMPT

    def _lyrics_chat_request(
        self,
        user_msg: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=self._model_id,
            messages=[
                ChatMessage(role="system", content=LYRICS_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_msg),
            ],
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            stream=False,
        )

    @staticmethod
    def _lyrics_quality_ok(text: str) -> bool:
        cleaned = (text or "").strip()
        if not cleaned:
            return False
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        if not lines:
            return False
        if len(lines) == 1 and lines[0].lower().strip("[]") == "instrumental":
            return True
        section_tags = [ln for ln in lines if ln.startswith("[") and ln.endswith("]")]
        if not section_tags:
            return False
        lyric_lines = [ln for ln in lines if not (ln.startswith("[") and ln.endswith("]"))]
        if not lyric_lines:
            return False
        total_chars = sum(len(ln) for ln in lyric_lines)
        if total_chars < 8:
            return False
        for ln in lyric_lines:
            if lyric_line_has_annotations(ln):
                return False
        return True

    @staticmethod
    def _build_lyrics_user_message(prompt: str, style: str | None = None) -> str:
        parts = ["## Music description", (prompt or "").strip()]
        if (style or "").strip():
            parts.extend(["", "## Style", style.strip()])
        return "\n".join(parts)

    @staticmethod
    def _token_budget(base: int, think_active: bool) -> int:
        if not think_active:
            return base
        return min(max(base + 768, base * 3), 8192)

    def _resolve_enable_thinking_for(self, model_id: str, override: bool | None) -> bool | None:
        if not is_thinking_model(model_id):
            return None
        if override is not None:
            return override
        return self._llm_think_enabled

    def _think_is_active_for(self, model_id: str, thinking: bool | None) -> bool:
        return bool(thinking) if is_thinking_model(model_id) else False

    def _resolve_enable_thinking(self, override: bool | None) -> bool | None:
        return self._resolve_enable_thinking_for(self._model_id, override)

    def _think_is_active(self, thinking: bool | None) -> bool:
        return self._think_is_active_for(self._model_id, thinking)

    @staticmethod
    def _with_think_suffix(text: str) -> str:
        t = (text or "").rstrip()
        if t.endswith("/think"):
            return t
        return f"{t} /think"

    @staticmethod
    def _with_no_think_suffix(text: str) -> str:
        t = (text or "").rstrip()
        if t.endswith("/no_think"):
            return t
        return f"{t} /no_think"

    def _apply_think_mode_to_text(self, text: str) -> str:
        if not is_thinking_model(self._model_id):
            return text
        if self._llm_think_enabled:
            return self._with_think_suffix(text)
        return self._with_no_think_suffix(text)

    def _apply_think_mode_to_messages_for(
        self,
        model_id: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        if not is_thinking_model(model_id):
            return messages
        last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
        if last_user < 0:
            return messages
        msg = messages[last_user]
        if isinstance(msg.content, list):
            return messages
        new_content = (
            self._with_think_suffix(str(msg.content))
            if self._llm_think_enabled
            else self._with_no_think_suffix(str(msg.content))
        )
        if new_content == msg.content:
            return messages
        patched = list(messages)
        patched[last_user] = ChatMessage(role=msg.role, content=new_content)
        return patched
