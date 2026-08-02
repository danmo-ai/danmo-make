"""
LLMService — load/run/unload LLM models via mlx-lm.
Does NOT use shared ModelCache (avoids conflict with DiT models).
Each request: load → infer → unload (load-on-demand, release-immediately).
"""

from __future__ import annotations

import gc
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx_lm
from mlx_lm.sample_utils import make_sampler

from backend.core.contracts import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDeltaChoice,
    ChatMessage,
    DeltaMessage,
    EnhanceRequest,
    EnhanceResponse,
)
from backend.core.interfaces import AppSettings
from backend.core.i18n import resolve_locale
from backend.core.model_registry import ModelEntry, ModelRegistry
from backend.engine.llm.lyrics_sanitize import lyric_line_has_annotations, sanitize_lyrics_output
from backend.engine.llm.prompt_sanitize import (
    prompt_enhance_fidelity_ok,
    prompt_enhance_quality_ok,
    sanitize_enhanced_prompt,
)
from backend.engine.llm.prompts.locale import enhance_user_locale_hint, storyboard_user_locale_block
from backend.engine.llm.prompts.system import (
    DESCRIBE_NODE_SYSTEM_PROMPT,
    ENHANCE_AUDIO_BRIEF_SYSTEM_PROMPT,
    ENHANCE_IMAGE_SYSTEM_PROMPT,
    ENHANCE_VIDEO_SYSTEM_PROMPT,
    IMAGE_TO_PROMPT_INSTRUCTION,
    LONG_VIDEO_CONTINUITY_SHOT_SYSTEM_PROMPT,
    LONG_VIDEO_CONTINUITY_SYSTEM_PROMPT,
    LONG_VIDEO_EXPAND_SHOT_SYSTEM_PROMPT,
    LONG_VIDEO_EXPAND_SYSTEM_PROMPT,
    LONG_VIDEO_OPENING_SYSTEM_PROMPT,
    LONG_VIDEO_PLAN_SHOT_SYSTEM_PROMPT,
    LONG_VIDEO_PLAN_SYSTEM_PROMPT,
    LYRICS_SYSTEM_PROMPT,
    VIDEO_FRAME_TO_PROMPT_INSTRUCTION,
)
from backend.engine.llm.think_parse import extract_final_llm_content
from backend.engine.llm.vision_mlx import (
    analyze_image_file,
    describe_image_file,
    mlx_vlm_importable,
    vision_weights_ready,
)
from backend.core.bundle_manifest import missing_safetensor_shards
from backend.utils.path_utils import PathResolver

logger = logging.getLogger(__name__)

_SETTINGS_DEFAULTS = AppSettings()
DEFAULT_LLM_MODEL_ID = _SETTINGS_DEFAULTS.default_model_llm
DEFAULT_VLM_MODEL_ID = _SETTINGS_DEFAULTS.default_model_vlm


def _is_valid_llm_entry(entry: ModelEntry | None) -> bool:
    if entry is None or entry.media != "llm":
        return False
    return bool(entry.actions & {"chat", "enhance"})


def _is_valid_vlm_entry(entry: ModelEntry | None) -> bool:
    if entry is None or entry.media != "llm":
        return False
    return "describe" in entry.actions


def _pick_first_llm(registry: ModelRegistry) -> str | None:
    for mid in sorted(registry.all()):
        if _is_valid_llm_entry(registry.get(mid)):
            return mid
    return None


def _pick_first_vlm(registry: ModelRegistry) -> str | None:
    for mid in sorted(registry.all()):
        if _is_valid_vlm_entry(registry.get(mid)):
            return mid
    return None


def _coerce_llm_model_id(preferred: str, registry: ModelRegistry) -> str:
    candidate = (preferred or "").strip()
    if candidate and _is_valid_llm_entry(registry.get(candidate)):
        return candidate
    fallback = DEFAULT_LLM_MODEL_ID
    if _is_valid_llm_entry(registry.get(fallback)):
        return fallback
    picked = _pick_first_llm(registry)
    return picked or fallback


def _coerce_vlm_model_id(preferred: str, registry: ModelRegistry) -> str:
    candidate = (preferred or "").strip()
    if candidate and _is_valid_vlm_entry(registry.get(candidate)):
        return candidate
    fallback = DEFAULT_VLM_MODEL_ID
    if _is_valid_vlm_entry(registry.get(fallback)):
        return fallback
    picked = _pick_first_vlm(registry)
    return picked or fallback


def normalize_app_llm_settings(settings: AppSettings, registry: ModelRegistry) -> bool:
    """Align saved LLM/VLM defaults with registry; return True if settings changed."""
    changed = False
    coerced_llm = _coerce_llm_model_id(settings.default_model_llm, registry)
    if settings.default_model_llm != coerced_llm:
        if (settings.default_model_llm or "").strip():
            logger.warning(
                "default_model_llm %r not in registry; using %r",
                settings.default_model_llm,
                coerced_llm,
            )
        settings.default_model_llm = coerced_llm
        changed = True

    coerced_vlm = _coerce_vlm_model_id(settings.default_model_vlm, registry)
    if settings.default_model_vlm != coerced_vlm:
        if (settings.default_model_vlm or "").strip():
            logger.warning(
                "default_model_vlm %r not in registry; using %r",
                settings.default_model_vlm,
                coerced_vlm,
            )
        settings.default_model_vlm = coerced_vlm
        changed = True
    return changed


def resolve_llm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return _coerce_llm_model_id(settings.default_model_llm, registry)


def resolve_vlm_model_id(settings: AppSettings, registry: ModelRegistry) -> str:
    return _coerce_vlm_model_id(settings.default_model_vlm, registry)


class LLMService:
    """Load → infer → unload MLX LLM models on demand.

    Does NOT participate in the shared ModelCache (max_entries=1) used by
    image/video/audio engines.  Each request loads the model, runs inference,
    and immediately releases GPU memory to avoid OOM conflicts.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        path_resolver: PathResolver,
        default_model_id: str = DEFAULT_LLM_MODEL_ID,
        vision_model_id: str = DEFAULT_VLM_MODEL_ID,
        llm_think_enabled: bool = False,
    ):
        self._registry = model_registry
        self._path_resolver = path_resolver
        self._model_id = default_model_id
        self._vision_model_id = vision_model_id
        self._llm_think_enabled = bool(llm_think_enabled)
        self._generation_lock = threading.Lock()

    def apply_model_settings(
        self,
        *,
        default_model_id: str | None = None,
        vision_model_id: str | None = None,
        llm_think_enabled: bool | None = None,
    ) -> None:
        if default_model_id is not None:
            coerced = _coerce_llm_model_id(default_model_id, self._registry)
            self._registry.require(coerced)
            self._model_id = coerced
        if vision_model_id is not None:
            coerced = _coerce_vlm_model_id(vision_model_id, self._registry)
            self._registry.require(coerced)
            self._vision_model_id = coerced
        if llm_think_enabled is not None:
            self._llm_think_enabled = bool(llm_think_enabled)
        if not self._is_thinking_model(self._model_id):
            self._llm_think_enabled = False

    def _resolve_request_llm_model(self, preferred: str | None) -> str:
        candidate = (preferred or "").strip()
        if not candidate:
            return self._model_id
        return _coerce_llm_model_id(candidate, self._registry)

    # ------------------------------------------------------------------
    # Public status
    # ------------------------------------------------------------------

    def get_model_info(self) -> dict[str, Any]:
        """Return current LLM model id, availability, and resolved path."""
        entry = self._registry.get(self._model_id)
        return {
            "model_id": self._model_id,
            "name": self._registry_display_name(entry, self._model_id),
            "available": self.is_available(),
            "think_supported": self._is_thinking_model(self._model_id),
            "think_enabled": self._llm_think_enabled,
        }

    def get_vision_model_info(self) -> dict[str, Any]:
        entry = self._registry.get(self._vision_model_id)
        return {
            "model_id": self._vision_model_id,
            "name": self._registry_display_name(entry, self._vision_model_id),
            "available": self.is_vision_available(),
            "mlx_vlm_installed": mlx_vlm_importable(),
        }

    def is_vision_available(self) -> bool:
        if not mlx_vlm_importable():
            return False
        try:
            path = self._resolve_vision_model_path()
            return vision_weights_ready(path)
        except Exception:
            return False

    def is_available(self) -> bool:
        """True when the model directory exists and contains complete weight files."""
        try:
            path = self._resolve_model_path()
            return self._llm_weights_ready(path)
        except Exception:
            return False

    @staticmethod
    def _llm_weights_ready(path: Path) -> bool:
        if not path.is_dir():
            return False
        missing_shards = missing_safetensor_shards(path)
        if missing_shards:
            return False
        return (
            (path / "model.safetensors").is_file()
            or any(f.suffix == ".safetensors" for f in path.rglob("*") if f.is_file())
            or any(f.suffix == ".bin" for f in path.rglob("*") if f.is_file())
        )

    @staticmethod
    def _assert_llm_weights_ready(path: Path, *, model_id: str) -> None:
        missing_shards = missing_safetensor_shards(path)
        if missing_shards:
            preview = ", ".join(missing_shards[:3])
            suffix = f" (+{len(missing_shards) - 3} more)" if len(missing_shards) > 3 else ""
            raise RuntimeError(
                f"LLM model {model_id!r} at {path} is missing weight shard(s): "
                f"{preview}{suffix}. Re-download the model from Settings → Models."
            )
        if not LLMService._llm_weights_ready(path):
            raise RuntimeError(
                f"LLM model {model_id!r} weights not found under {path}. "
                "Install the model from Settings → Models."
            )

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    def chat_completion(
        self,
        request: ChatCompletionRequest,
        *,
        enable_thinking: bool | None = None,
    ) -> ChatCompletionResponse:
        """Run a single-turn non-streaming chat completion (sync, for asyncio.to_thread)."""
        effective_id = self._resolve_request_llm_model(request.model)
        thinking = self._resolve_enable_thinking_for(effective_id, enable_thinking)
        think_active = self._think_is_active_for(effective_id, thinking)
        with self._generation_lock:
            model, tokenizer = self._load_model(effective_id)
            try:
                messages = self._apply_think_mode_to_messages_for(effective_id, request.messages)
                prompt = self._build_chat_prompt(
                    tokenizer,
                    messages,
                    enable_thinking=thinking,
                )
                result = mlx_lm.generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    verbose=False,
                    **self._generation_kwargs(request, think_active=think_active),
                )
                content = extract_final_llm_content(result, think_enabled=think_active)
                return self._format_response(content, effective_id)
            finally:
                self._unload_model(model, tokenizer)

    # ------------------------------------------------------------------
    # Chat completion (SSE streaming)
    # ------------------------------------------------------------------

    async def chat_completion_stream(self, request: ChatCompletionRequest):
        """Yield SSE lines in OpenAI format: data: {json}\n\n ... data: [DONE]\n\n"""
        import asyncio

        effective_id = self._resolve_request_llm_model(request.model)
        thinking = self._resolve_enable_thinking_for(effective_id, None)
        think_active = self._think_is_active_for(effective_id, thinking)

        with self._generation_lock:
            model, tokenizer = self._load_model(effective_id)
            try:
                messages = self._apply_think_mode_to_messages_for(effective_id, request.messages)
                prompt_text = self._build_chat_prompt(
                    tokenizer,
                    messages,
                    enable_thinking=thinking,
                )
            except Exception:
                self._unload_model(model, tokenizer)
                raise

        # Release the lock while streaming so other requests can queue.
        # The model stays loaded until we explicitly unload.
        try:
            chunk_id = f"chatcmpl-{secrets.token_hex(12)}"
            created = int(time.time())

            def _generate():
                return list(
                    mlx_lm.stream_generate(
                        model,
                        tokenizer,
                        prompt=prompt_text,
                        **self._generation_kwargs(request, think_active=think_active),
                    )
                )

            responses = await asyncio.to_thread(_generate)

            for resp in responses:
                chunk = ChatCompletionChunk(
                    id=chunk_id,
                    created=created,
                    model=effective_id,
                    choices=[
                        ChatDeltaChoice(
                            index=0,
                            delta=DeltaMessage(content=resp.text),
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

            # Final chunk with finish_reason
            last_resp = responses[-1] if responses else None
            finish_reason = last_resp.finish_reason if last_resp else "stop"
            final_chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=effective_id,
                choices=[
                    ChatDeltaChoice(
                        index=0,
                        delta=DeltaMessage(),
                        finish_reason=finish_reason,
                    )
                ],
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            self._unload_model(model, tokenizer)

    # ------------------------------------------------------------------
    # Prompt enhancement
    # ------------------------------------------------------------------

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

    def enhance_prompt(self, request: EnhanceRequest) -> EnhanceResponse:
        """Enhance a creative brief for image, video, or audio generation."""
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
        with self._generation_lock:
            model, tokenizer = self._load_model()
            try:
                for temperature, max_tokens in attempts:
                    internal = ChatCompletionRequest(
                        model=self._model_id,
                        messages=messages,
                        temperature=temperature,
                        top_p=0.9,
                        max_tokens=max_tokens,
                        stream=False,
                    )
                    prompt = self._build_chat_prompt(
                        tokenizer,
                        messages,
                        enable_thinking=thinking,
                    )
                    result = mlx_lm.generate(
                        model,
                        tokenizer,
                        prompt=prompt,
                        verbose=False,
                        **self._generation_kwargs(internal, think_active=think_active),
                    )
                    cleaned = sanitize_enhanced_prompt(
                        extract_final_llm_content(result, think_enabled=think_active),
                        think_enabled=think_active,
                    )
                    if prompt_enhance_quality_ok(cleaned, original=raw_prompt):
                        return EnhanceResponse(enhanced_prompt=cleaned)
                    last_clean = cleaned
            finally:
                self._unload_model(model, tokenizer)

        fallback = sanitize_enhanced_prompt(raw_prompt, think_enabled=think_active)
        if prompt_enhance_quality_ok(last_clean, original=raw_prompt):
            return EnhanceResponse(enhanced_prompt=last_clean)
        if prompt_enhance_fidelity_ok(raw_prompt, last_clean):
            return EnhanceResponse(enhanced_prompt=last_clean)
        return EnhanceResponse(enhanced_prompt=fallback or raw_prompt)

    def _script_parse_runtime(
        self,
        model_id: str | None,
    ) -> tuple[str, bool, Callable[[str], str], Callable[[int], int], Callable[..., str]]:
        from backend.engine.llm.prompt_sanitize import sanitize_structured_llm_response

        analyze_model_id = self._resolve_request_llm_model(model_id)
        thinking = self._resolve_enable_thinking_for(analyze_model_id, None)
        think_active = self._think_is_active_for(analyze_model_id, thinking)
        think_apply = lambda text: self._apply_think_mode_to_text_for(analyze_model_id, text)

        def chat_fn(*, messages: list[ChatMessage], max_tokens: int) -> str:
            token_cap = self._token_budget(max_tokens, think_active)
            if self._is_thinking_model(analyze_model_id):
                token_cap = max(token_cap, 8192)
            resp = self.chat_completion(
                ChatCompletionRequest(
                    model=analyze_model_id,
                    messages=messages,
                    temperature=0.35,
                    top_p=0.9,
                    max_tokens=token_cap,
                    stream=False,
                ),
                enable_thinking=thinking,
            )
            return sanitize_structured_llm_response(
                resp.choices[0].message.content,
                think_enabled=think_active,
            )

        token_budget = lambda b: self._token_budget(b, think_active)
        return analyze_model_id, think_active, think_apply, token_budget, chat_fn

    def script_parse_decompose(
        self,
        request: "ScriptParseDecomposeRequest",
        *,
        on_progress: Callable[[str, str], None] | None = None,
        activity_recorder: Any | None = None,
    ) -> "ScriptParseDecomposeResponse":
        from backend.core.contracts import (
            LongVideoChapterParsePhaseDTO,
            ScriptParseDecomposeRequest,
            ScriptParseDecomposeResponse,
        )
        from backend.engine.llm.script_parse.pipeline import run_decompose
        from backend.engine.llm.script_parse.response import decompose_to_response
        from backend.engine.llm.storyboard import normalize_storyboard_locale

        locale = normalize_storyboard_locale(getattr(request, "locale", None))
        parse_phases: list[LongVideoChapterParsePhaseDTO] = []
        project_id = str(getattr(request, "long_video_project_id", "") or "").strip()

        if activity_recorder is not None and activity_recorder.active:
            activity_recorder.record_started(
                chapter_title=request.title,
                target_duration_sec=0.0,
            )

        def report(phase: str, message: str = "") -> None:
            parse_phases.append(LongVideoChapterParsePhaseDTO(phase=phase, message=message))
            if activity_recorder is not None and activity_recorder.active:
                activity_recorder.record_phase(phase, message)
            if on_progress:
                on_progress(phase, message)

        _, _, think_apply, token_budget, chat_fn = self._script_parse_runtime(getattr(request, "model", None))

        try:
            result = run_decompose(
                script_text=request.script_text,
                title=request.title,
                locale=locale,
                chat_fn=chat_fn,
                think_apply=think_apply,
                token_budget=token_budget,
                on_progress=report,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        response = decompose_to_response(
            result,
            parse_phases=parse_phases,
            parse_run_id=activity_recorder.parse_run_id if activity_recorder and activity_recorder.active else "",
            project_id=project_id,
            locale=locale,
        )
        if activity_recorder is not None and activity_recorder.active:
            activity_recorder.record_completed(response)
        return response

    def script_parse_expand(
        self,
        request: "ScriptParseExpandRequest",
        *,
        on_progress: Callable[[str, str], None] | None = None,
        activity_recorder: Any | None = None,
    ) -> "ScriptParseExpandResponse":
        from backend.core.contracts import (
            LongVideoChapterParsePhaseDTO,
            ScriptParseExpandRequest,
            ScriptParseExpandResponse,
        )
        from backend.engine.llm.script_parse.pipeline import run_expand
        from backend.engine.llm.script_parse.response import expand_to_response, script_dict_to_artifact
        from backend.engine.llm.storyboard import normalize_storyboard_locale

        locale = normalize_storyboard_locale(getattr(request, "locale", None))
        parse_phases: list[LongVideoChapterParsePhaseDTO] = []
        project_id = str(getattr(request, "long_video_project_id", "") or "").strip()

        if activity_recorder is not None and activity_recorder.active:
            activity_recorder.record_started(
                chapter_title="",
                target_duration_sec=float(getattr(request, "target_duration_sec", 60.0) or 60.0),
            )

        def report(phase: str, message: str = "") -> None:
            parse_phases.append(LongVideoChapterParsePhaseDTO(phase=phase, message=message))
            if activity_recorder is not None and activity_recorder.active:
                activity_recorder.record_phase(phase, message)
            if on_progress:
                on_progress(phase, message)

        _, _, think_apply, token_budget, chat_fn = self._script_parse_runtime(getattr(request, "model", None))

        try:
            artifact = script_dict_to_artifact(request.script_artifact)
        except Exception as exc:
            raise RuntimeError(f"invalid script_artifact: {exc}") from exc

        try:
            result = run_expand(
                script=artifact,
                target_duration_sec=float(getattr(request, "target_duration_sec", 60.0) or 60.0),
                segment_duration_sec=float(getattr(request, "segment_duration_sec", 5.0) or 5.0),
                max_clip_sec=float(getattr(request, "max_clip_sec", 10.0) or 10.0),
                locale=locale,
                chat_fn=chat_fn,
                think_apply=think_apply,
                token_budget=token_budget,
                on_progress=report,
                beat_indices=list(request.beat_indices or []),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        report("done", "done")
        response = expand_to_response(
            result,
            artifact=artifact,
            parse_phases=parse_phases,
            parse_run_id=activity_recorder.parse_run_id if activity_recorder and activity_recorder.active else "",
            project_id=project_id,
        )
        if activity_recorder is not None and activity_recorder.active:
            activity_recorder.record_completed(response)
        return response

    def _script_parse_stream(
        self,
        worker: Callable[[Callable[[str, str], None]], Any],
    ):
        import json
        import queue
        import threading

        event_queue: queue.Queue = queue.Queue()
        holder: dict[str, object] = {}

        def on_progress(phase: str, message: str) -> None:
            if phase == "review_retry":
                event_queue.put(
                    {
                        "event": "review_retry",
                        "pass": message,
                        "phase": phase,
                        "message": message,
                    }
                )
                return
            event_queue.put({"event": "progress", "phase": phase, "message": message})

        def run_worker() -> None:
            try:
                holder["result"] = worker(on_progress)
            except Exception as exc:
                holder["error"] = exc
            finally:
                event_queue.put(None)

        threading.Thread(target=run_worker, daemon=True).start()

        while True:
            item = event_queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        err = holder.get("error")
        if err is not None:
            payload: dict[str, object] = {"event": "error", "detail": str(err)}
            issues = getattr(err, "quality_issues", None)
            if issues:
                payload["quality_issues"] = issues
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return

        result = holder.get("result")
        if result is None:
            payload = {"event": "error", "detail": "script parse returned no result"}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            return

        payload = {"event": "result", "data": result.model_dump(mode="json")}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def script_parse_decompose_stream(
        self,
        request: "ScriptParseDecomposeRequest",
        *,
        activity_recorder: Any | None = None,
    ):
        def worker(on_progress: Callable[[str, str], None]):
            try:
                return self.script_parse_decompose(
                    request,
                    on_progress=on_progress,
                    activity_recorder=activity_recorder,
                )
            except Exception as exc:
                if activity_recorder is not None and activity_recorder.active:
                    activity_recorder.record_failed(str(exc))
                raise

        yield from self._script_parse_stream(worker)

    def script_parse_expand_stream(
        self,
        request: "ScriptParseExpandRequest",
        *,
        activity_recorder: Any | None = None,
    ):
        def worker(on_progress: Callable[[str, str], None]):
            try:
                return self.script_parse_expand(
                    request,
                    on_progress=on_progress,
                    activity_recorder=activity_recorder,
                )
            except Exception as exc:
                if activity_recorder is not None and activity_recorder.active:
                    activity_recorder.record_failed(str(exc))
                raise

        yield from self._script_parse_stream(worker)

    def script_parse_expand_beat(
        self,
        request: "ScriptParseExpandBeatRequest",
        *,
        on_progress: Callable[[str, str], None] | None = None,
        activity_recorder: Any | None = None,
    ) -> "ScriptParseExpandResponse":
        from backend.core.contracts import (
            LongVideoChapterParsePhaseDTO,
            ScriptParseExpandBeatRequest,
            ScriptParseExpandResponse,
        )
        from backend.engine.llm.script_parse.pipeline import run_expand_beat
        from backend.engine.llm.script_parse.response import expand_to_response, script_dict_to_artifact
        from backend.engine.llm.storyboard import normalize_storyboard_locale

        locale = normalize_storyboard_locale(getattr(request, "locale", None))
        parse_phases: list[LongVideoChapterParsePhaseDTO] = []
        project_id = str(getattr(request, "long_video_project_id", "") or "").strip()

        if activity_recorder is not None and activity_recorder.active:
            activity_recorder.record_started(
                chapter_title="",
                target_duration_sec=float(getattr(request, "target_duration_sec", 60.0) or 60.0),
            )

        def report(phase: str, message: str = "") -> None:
            parse_phases.append(LongVideoChapterParsePhaseDTO(phase=phase, message=message))
            if activity_recorder is not None and activity_recorder.active:
                activity_recorder.record_phase(phase, message)
            if on_progress:
                on_progress(phase, message)

        _, _, think_apply, token_budget, chat_fn = self._script_parse_runtime(getattr(request, "model", None))

        try:
            artifact = script_dict_to_artifact(request.script_artifact)
        except Exception as exc:
            raise RuntimeError(f"invalid script_artifact: {exc}") from exc

        try:
            result = run_expand_beat(
                script=artifact,
                beat_index=int(request.beat_index),
                target_duration_sec=float(getattr(request, "target_duration_sec", 60.0) or 60.0),
                segment_duration_sec=float(getattr(request, "segment_duration_sec", 5.0) or 5.0),
                max_clip_sec=float(getattr(request, "max_clip_sec", 10.0) or 10.0),
                locale=locale,
                chat_fn=chat_fn,
                think_apply=think_apply,
                token_budget=token_budget,
                on_progress=report,
                existing_shots=list(request.existing_shots or []),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        report("done", "done")
        response = expand_to_response(
            result,
            artifact=artifact,
            parse_phases=parse_phases,
            parse_run_id=activity_recorder.parse_run_id if activity_recorder and activity_recorder.active else "",
            project_id=project_id,
        )
        if activity_recorder is not None and activity_recorder.active:
            activity_recorder.record_completed(response)
        return response

    def script_parse_expand_beat_stream(
        self,
        request: "ScriptParseExpandBeatRequest",
        *,
        activity_recorder: Any | None = None,
    ):
        def worker(on_progress: Callable[[str, str], None]):
            try:
                return self.script_parse_expand_beat(
                    request,
                    on_progress=on_progress,
                    activity_recorder=activity_recorder,
                )
            except Exception as exc:
                if activity_recorder is not None and activity_recorder.active:
                    activity_recorder.record_failed(str(exc))
                raise

        yield from self._script_parse_stream(worker)

    # ------------------------------------------------------------------
    # Lyrics generation
    # ------------------------------------------------------------------

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
        if len(lyric_lines) < 2 and total_chars < 16:
            return False

        for ln in lyric_lines:
            if lyric_line_has_annotations(ln):
                return False
            words = ln.split()
            if len(words) >= 6 and len(set(w.lower() for w in words)) / len(words) < 0.35:
                return False
            if len(ln) > 120:
                return False
        return True

    @staticmethod
    def _build_lyrics_user_message(prompt: str, style: str | None = None) -> str:
        parts = ["## Music description", (prompt or "").strip()]
        if (style or "").strip():
            parts.extend(["", "## Style", style.strip()])
        return "\n".join(parts)

    def generate_lyrics(self, prompt: str, style: str | None = None) -> str:
        """Generate ACE-Step formatted lyrics from a music description."""
        user_msg = self._with_no_think_suffix(self._build_lyrics_user_message(prompt, style))

        attempts = (
            (0.65, 420),
            (0.5, 360),
            (0.4, 512),
        )
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

    # ------------------------------------------------------------------
    # Vision: image/video → generation prompt & reference analysis
    # ------------------------------------------------------------------

    def image_to_prompt(
        self,
        asset_context: dict[str, Any],
        *,
        image_path: Path,
        media: str = "image",
    ) -> tuple[str, bool]:
        """Reverse-engineer a generation prompt from a reference image or video keyframe."""
        if not self.is_vision_available():
            raise RuntimeError(
                "Vision model not available. Install a VLM from Models page and set Settings → Default VLM Model."
            )
        meta = asset_context.get("metadata") or {}
        metadata_hint = self._metadata_hint_lines(asset_context, meta)
        instruction = (
            VIDEO_FRAME_TO_PROMPT_INSTRUCTION
            if media == "video"
            else IMAGE_TO_PROMPT_INSTRUCTION
        )
        with self._generation_lock:
            prompt = analyze_image_file(
                image_path,
                self._resolve_vision_model_path(),
                instruction=instruction,
                metadata_hint=metadata_hint,
                max_tokens=384,
            )
        return self._clean_vlm_prompt(prompt), True

    def analyze_reference(
        self,
        asset_context: dict[str, Any],
        *,
        image_path: Path,
        question: str,
        locale: str | None = None,
    ) -> tuple[str, bool]:
        """Answer a creative question about a reference image (style, palette, subject, etc.)."""
        if not self.is_vision_available():
            raise RuntimeError(
                "Vision model not available. Install a VLM from Models page and set Settings → Default VLM Model."
            )
        meta = asset_context.get("metadata") or {}
        metadata_hint = self._metadata_hint_lines(asset_context, meta)
        lang = resolve_locale(locale)
        output_rule = (
            "Respond ONLY in concise Simplified Chinese (简体中文)."
            if lang == "zh"
            else "Respond ONLY in concise English."
        )
        instruction = (
            "You are a creative director analyzing a reference image for an artist.\n"
            f"Task: {question.strip()}\n"
            f"{output_rule} "
            "Be specific and actionable for the next generation. Output ONLY the analysis text."
        )
        with self._generation_lock:
            answer = analyze_image_file(
                image_path,
                self._resolve_vision_model_path(),
                instruction=instruction,
                metadata_hint=metadata_hint,
                max_tokens=320,
            )
        return answer.strip(), True

    # ------------------------------------------------------------------
    # Canvas node description (metadata-based; text LLM)
    # ------------------------------------------------------------------

    def describe_node(
        self,
        asset_context: dict[str, Any],
        *,
        image_path: Path | None = None,
        prefer_vision: bool = True,
    ) -> tuple[str, bool]:
        """Generate a canvas node note. Uses VLM for image/video when available."""
        meta = asset_context.get("metadata") or {}
        kind = str(asset_context.get("kind") or "image")
        metadata_hint = self._metadata_hint_lines(asset_context, meta)

        if (
            prefer_vision
            and image_path is not None
            and kind in ("image", "video")
            and self.is_vision_available()
        ):
            with self._generation_lock:
                try:
                    note = describe_image_file(
                        image_path,
                        self._resolve_vision_model_path(),
                        metadata_hint=metadata_hint,
                    )
                    return note.strip(), True
                except Exception as exc:
                    logger.warning("Vision describe failed, using text LLM: %s", exc)

        note = self._describe_node_from_metadata(metadata_hint)
        return note, False

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
        if asset_context.get("relation_type"):
            lines.append(f"Lineage relation: {asset_context.get('relation_type')}")
        return "\n".join(lines)

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

    # ------------------------------------------------------------------
    # Model lifecycle (load-on-demand, release-immediately)
    # ------------------------------------------------------------------

    def _resolve_vision_model_path(self) -> Path:
        entry = self._registry.require(self._vision_model_id)
        versions = entry.raw.get("versions") or {}
        default_ver = next(
            (v for v in versions.values() if v.get("default")),
            next(iter(versions.values()), None),
        )
        if default_ver is None:
            raise RuntimeError(
                f"No versions defined for vision model {self._vision_model_id!r} in registry"
            )
        local_path = default_ver.get("local_path")
        if not local_path:
            raise RuntimeError(
                f"No local_path for default version of {self._vision_model_id!r}"
            )
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _resolve_model_path(self, model_id: str | None = None) -> Path:
        mid = (model_id or self._model_id).strip() or self._model_id
        entry = self._registry.require(mid)
        versions = entry.raw.get("versions") or {}
        default_ver = next(
            (v for v in versions.values() if v.get("default")),
            next(iter(versions.values()), None),
        )
        if default_ver is None:
            raise RuntimeError(
                f"No versions defined for LLM model {mid!r} in registry"
            )
        local_path = default_ver.get("local_path")
        if not local_path:
            raise RuntimeError(
                f"No local_path for default version of {mid!r}"
            )
        return self._path_resolver.resolve_registry_local_path(local_path)

    def _load_model(self, model_id: str | None = None) -> tuple[Any, Any]:
        """Load the LLM model + tokenizer into GPU memory."""
        mid = (model_id or self._model_id).strip() or self._model_id
        model_path = self._resolve_model_path(mid)
        self._assert_llm_weights_ready(model_path, model_id=mid)
        logger.info("Loading LLM model %s from %s", mid, model_path)
        model, tokenizer = mlx_lm.load(str(model_path))
        logger.info("LLM model loaded successfully")
        return model, tokenizer

    def _unload_model(self, model: Any, tokenizer: Any) -> None:
        """Release model from GPU memory."""
        del model
        del tokenizer
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass
        logger.info("LLM model unloaded")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _registry_display_name(entry: Any, fallback: str) -> Any:
        """Registry ``name`` may be a bilingual dict or plain string."""
        if entry is None:
            return fallback
        raw_name = entry.raw.get("name")
        return raw_name if raw_name is not None else fallback

    def _generation_kwargs(
        self,
        request: ChatCompletionRequest,
        *,
        think_active: bool = False,
    ) -> dict[str, Any]:
        """Sampling kwargs for mlx-lm >= 0.31 (``sampler`` instead of ``temp``)."""
        max_tokens = request.max_tokens or 512
        if think_active:
            max_tokens = self._token_budget(max_tokens, True)
        return {
            "max_tokens": max_tokens,
            "sampler": make_sampler(
                temp=request.temperature,
                top_p=request.top_p,
            ),
        }

    @staticmethod
    def _token_budget(base: int, think_active: bool) -> int:
        if not think_active:
            return base
        return min(max(base + 768, base * 3), 8192)

    def _resolve_enable_thinking_for(self, model_id: str, override: bool | None) -> bool | None:
        if not self._is_thinking_model(model_id):
            return None
        if override is not None:
            return override
        return self._llm_think_enabled

    def _think_is_active_for(self, model_id: str, thinking: bool | None) -> bool:
        return bool(thinking) if self._is_thinking_model(model_id) else False

    def _apply_think_mode_to_text_for(self, model_id: str, text: str) -> str:
        if not self._is_thinking_model(model_id):
            return text
        if self._llm_think_enabled:
            return self._with_think_suffix(text)
        return self._with_no_think_suffix(text)

    def _apply_think_mode_to_messages_for(
        self,
        model_id: str,
        messages: list[ChatMessage],
    ) -> list[ChatMessage]:
        if not self._is_thinking_model(model_id):
            return messages
        last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)
        if last_user < 0:
            return messages
        msg = messages[last_user]
        if isinstance(msg.content, list):
            return messages
        new_content = (
            self._with_think_suffix(msg.content)
            if self._llm_think_enabled
            else self._with_no_think_suffix(msg.content)
        )
        if new_content == msg.content:
            return messages
        patched = list(messages)
        patched[last_user] = ChatMessage(role=msg.role, content=new_content)
        return patched

    def _resolve_enable_thinking(self, override: bool | None) -> bool | None:
        return self._resolve_enable_thinking_for(self._model_id, override)

    def _think_is_active(self, thinking: bool | None) -> bool:
        return self._think_is_active_for(self._model_id, thinking)

    def _apply_think_mode_to_text(self, text: str) -> str:
        return self._apply_think_mode_to_text_for(self._model_id, text)

    def _apply_think_mode_to_messages(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        return self._apply_think_mode_to_messages_for(self._model_id, messages)

    @staticmethod
    def _is_thinking_model(model_id: str) -> bool:
        """Models that honor /think and /no_think suffixes on the last user turn."""
        mid = (model_id or "").lower()
        if "thinking" in mid:
            return True
        # Qwen3.5 / Qwen3.6 instruct models emit plain-text reasoning unless /no_think.
        for prefix in ("qwen3.5", "qwen3-5", "qwen3.6", "qwen3-6"):
            if mid.startswith(prefix):
                return True
        return False

    @staticmethod
    def _with_no_think_suffix(text: str) -> str:
        body = (text or "").rstrip()
        if not body or "/no_think" in body or "/think" in body:
            return body
        return f"{body} /no_think"

    @staticmethod
    def _with_think_suffix(text: str) -> str:
        body = (text or "").rstrip()
        if not body or "/think" in body or "/no_think" in body:
            return body
        return f"{body} /think"

    @staticmethod
    def _build_chat_prompt(
        tokenizer,
        messages: list[ChatMessage],
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        msg_dicts = [
            {"role": m.role, "content": LLMService._message_content_for_template(m)}
            for m in messages
        ]
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(msg_dicts, **template_kwargs)
        except TypeError:
            if enable_thinking is not None:
                try:
                    return tokenizer.apply_chat_template(
                        msg_dicts,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: simple concatenation for tokenizers without chat template
        parts: list[str] = []
        for m in messages:
            body = LLMService._message_content_for_template(m)
            if m.role == "system":
                parts.append(f"<|system|>\n{body}</s>")
            elif m.role == "user":
                parts.append(f"<|user|>\n{body}</s>")
            elif m.role == "assistant":
                parts.append(f"<|assistant|>\n{body}</s>")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _message_content_for_template(message: ChatMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        chunks: list[str] = []
        for part in content:
            if getattr(part, "type", None) == "text":
                text = str(getattr(part, "text", "") or "").strip()
                if text:
                    chunks.append(text)
        return "\n".join(chunks)

    @staticmethod
    def _format_response(text: str, model_name: str) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{secrets.token_hex(12)}",
            object="chat.completion",
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage={},
        )
