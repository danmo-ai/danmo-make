"""Vision inference via backend_llm sidecar HTTP (no in-process mlx_vlm)."""

from __future__ import annotations

from pathlib import Path

from backend.core.contracts import ChatCompletionRequest, ChatMessage
from backend.core.interfaces import AppSettings
from backend.engine.llm.chat_invoke import build_text_messages
from backend.engine.llm.message_content import extract_vision_instruction
from backend.engine.llm.openai_client import LlmOpenAIClient
from shared.danqing_config.inference import cloud_inference_ready
from shared.danqing_config.llm import llm_weights_ready


def _messages_with_image(
    instruction: str,
    image_path: Path,
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": instruction},
                {
                    "type": "image_url",
                    "image_url": {"url": str(image_path.resolve())},
                },
            ],
        )
    ]


def _resolve_sidecar_model(settings: AppSettings, model_dir: Path, registry_model_id: str) -> str:
    if settings.llm_inference_provider == "openai_compatible":
        cloud = (settings.llm_inference_cloud_model or "").strip()
        if cloud:
            return cloud
        return registry_model_id
    return str(model_dir.resolve())


def analyze_image_file(
    image_path: Path,
    model_dir: Path,
    *,
    instruction: str,
    metadata_hint: str = "",
    max_tokens: int = 384,
    temperature: float = 0.2,
    settings: AppSettings,
    registry_model_id: str,
) -> str:
    prompt = instruction.strip()
    if metadata_hint.strip():
        prompt = f"{prompt}\n\nContext:\n{metadata_hint.strip()}"
    client = LlmOpenAIClient(settings)
    request = ChatCompletionRequest(
        model=registry_model_id,
        messages=_messages_with_image(prompt, image_path),
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    model = _resolve_sidecar_model(settings, model_dir, registry_model_id)
    result = client.chat_completion(request, model=model)
    return (result.choices[0].message.content or "").strip()


def analyze_images_multi(
    image_paths: list[Path],
    model_dir: Path,
    *,
    instruction: str,
    metadata_hint: str = "",
    max_tokens: int = 384,
    temperature: float = 0.2,
    settings: AppSettings,
    registry_model_id: str,
) -> str:
    parts: list[dict] = [{"type": "text", "text": instruction.strip()}]
    if metadata_hint.strip():
        parts[0]["text"] = f"{parts[0]['text']}\n\nContext:\n{metadata_hint.strip()}"
    for path in image_paths:
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": str(path.resolve())},
            }
        )
    client = LlmOpenAIClient(settings)
    request = ChatCompletionRequest(
        model=registry_model_id,
        messages=[ChatMessage(role="user", content=parts)],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    model = _resolve_sidecar_model(settings, model_dir, registry_model_id)
    result = client.chat_completion(request, model=model)
    return (result.choices[0].message.content or "").strip()


def analyze_image_files_batch(
    image_paths: list[Path],
    model_dir: Path,
    *,
    instruction: str,
    max_tokens: int = 200,
    temperature: float = 0.2,
    settings: AppSettings,
    registry_model_id: str,
) -> list[str]:
    texts: list[str] = []
    for path in image_paths:
        texts.append(
            analyze_image_file(
                path,
                model_dir,
                instruction=instruction,
                max_tokens=max_tokens,
                temperature=temperature,
                settings=settings,
                registry_model_id=registry_model_id,
            )
        )
    return texts


def analyze_image_files_batch_messages(
    image_paths: list[Path],
    model_dir: Path,
    *,
    messages: list[ChatMessage],
    max_tokens: int = 128,
    temperature: float = 0.2,
    settings: AppSettings,
    registry_model_id: str,
) -> list[str]:
    instruction = extract_vision_instruction(messages)
    return analyze_image_files_batch(
        image_paths,
        model_dir,
        instruction=instruction,
        max_tokens=max_tokens,
        temperature=temperature,
        settings=settings,
        registry_model_id=registry_model_id,
    )


def vision_inference_available(
    settings: AppSettings,
    model_dir: Path,
) -> bool:
    if settings.llm_inference_provider == "openai_compatible":
        return cloud_inference_ready(
            type(
                "Snap",
                (),
                {
                    "llm_inference_provider": settings.llm_inference_provider,
                    "llm_inference_base_url": settings.llm_inference_base_url,
                    "llm_inference_api_key": settings.llm_inference_api_key,
                },
            )()
        )
    return llm_weights_ready(model_dir)


def build_text_only_messages(system: str, user: str) -> list[ChatMessage]:
    return build_text_messages(system=system, user=user)
