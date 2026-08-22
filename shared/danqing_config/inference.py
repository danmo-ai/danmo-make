"""LLM inference backend configuration (builtin sidecar vs cloud OpenAI-compatible API)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from shared.danqing_config.paths import llm_port_file, resolve_control_plane_dir
from shared.danqing_config.settings import AppSettingsSnapshot

LlmInferenceProvider = Literal["builtin", "openai_compatible"]
DEFAULT_BUILTIN_HOST = "127.0.0.1"
DEFAULT_BUILTIN_PORT = 7801


@dataclass(frozen=True)
class LlmInferenceConfig:
    provider: LlmInferenceProvider
    base_url: str
    api_key: str
    cloud_model: str
    builtin_host: str
    builtin_port: int
    quantize_activations: bool


def _read_port_file() -> int | None:
    path = llm_port_file()
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        return int(text) if text else None
    except (OSError, ValueError):
        return None


def resolve_builtin_base_url(
    *,
    host: str | None = None,
    port: int | None = None,
    control_plane=None,
) -> str:
    env = os.environ.get("DANQING_LLM_BASE_URL", "").strip().rstrip("/")
    if env:
        return env if env.endswith("/v1") else f"{env}/v1"
    effective_port = port
    if effective_port is None:
        env_port = os.environ.get("DANQING_LLM_HTTP_PORT", "").strip()
        if env_port.isdigit():
            effective_port = int(env_port)
    if effective_port is None:
        effective_port = _read_port_file()
    if effective_port is None:
        effective_port = DEFAULT_BUILTIN_PORT
    effective_host = (host or os.environ.get("DANQING_LLM_HTTP_HOST") or DEFAULT_BUILTIN_HOST).strip()
    return f"http://{effective_host}:{effective_port}/v1"


def load_llm_inference_config(
    settings: AppSettingsSnapshot | None = None,
    *,
    control_plane=None,
) -> LlmInferenceConfig:
    snap = settings or AppSettingsSnapshot()
    provider: LlmInferenceProvider = (
        "openai_compatible" if snap.llm_inference_provider == "openai_compatible" else "builtin"
    )
    builtin_host = (snap.llm_builtin_host or DEFAULT_BUILTIN_HOST).strip() or DEFAULT_BUILTIN_HOST
    builtin_port = max(1, int(snap.llm_builtin_port or DEFAULT_BUILTIN_PORT))
    quantize = bool(snap.llm_quantize_activations)

    if provider == "openai_compatible":
        base = (snap.llm_inference_base_url or "").strip().rstrip("/")
        if not base:
            base = "https://api.openai.com/v1"
        elif not base.endswith("/v1"):
            base = f"{base}/v1" if "/v1" not in base else base
        api_key = _resolve_cloud_api_key(snap)
        cloud_model = (snap.llm_inference_cloud_model or "").strip()
        return LlmInferenceConfig(
            provider=provider,
            base_url=base,
            api_key=api_key,
            cloud_model=cloud_model,
            builtin_host=builtin_host,
            builtin_port=builtin_port,
            quantize_activations=quantize,
        )

    return LlmInferenceConfig(
        provider="builtin",
        base_url=resolve_builtin_base_url(host=builtin_host, port=builtin_port),
        api_key="",
        cloud_model="",
        builtin_host=builtin_host,
        builtin_port=builtin_port,
        quantize_activations=quantize,
    )


def _resolve_cloud_api_key(snap: AppSettingsSnapshot) -> str:
    env = os.environ.get("DANQING_LLM_INFERENCE_API_KEY", "").strip()
    if env:
        return env
    stored = (snap.llm_inference_api_key or "").strip()
    if stored.startswith("v1:"):
        return ""
    return stored


def inference_api_key_configured(snap: AppSettingsSnapshot) -> bool:
    if os.environ.get("DANQING_LLM_INFERENCE_API_KEY", "").strip():
        return True
    key = (snap.llm_inference_api_key or "").strip()
    return bool(key)


def cloud_inference_ready(snap: AppSettingsSnapshot) -> bool:
    if snap.llm_inference_provider != "openai_compatible":
        return False
    if not (snap.llm_inference_base_url or "").strip():
        return False
    if inference_api_key_configured(snap):
        return True
    return bool((snap.llm_inference_api_key or "").strip() and not snap.llm_inference_api_key.startswith("v1:"))
