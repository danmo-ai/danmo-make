"""Read app settings JSON from the control plane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shared.danqing_config.paths import control_settings_path

LlmInferenceProviderSetting = Literal["builtin", "openai_compatible"]


@dataclass
class AppSettingsSnapshot:
    default_model_llm: str = "qwen3-vl-4b-instruct"
    default_model_vlm: str = "qwen3-vl-4b-instruct"
    default_model_llm_think: bool = False
    llm_unload_each_request: bool = False
    model_cache_ttl_minutes: int = 30
    llm_cache_ttl_minutes: int = 30
    llm_inference_provider: LlmInferenceProviderSetting = "builtin"
    llm_inference_base_url: str = ""
    llm_inference_api_key: str = ""
    llm_inference_api_key_hint: str = ""
    llm_inference_cloud_model: str = ""
    llm_builtin_host: str = "127.0.0.1"
    llm_builtin_port: int = 7801
    llm_quantize_activations: bool = False


def _nested_llm_inference(data: dict[str, Any]) -> dict[str, Any]:
    block = data.get("llm_inference")
    if isinstance(block, dict):
        return block
    return {}


def load_app_settings(*, control_plane: Path | None = None) -> AppSettingsSnapshot:
    path = control_settings_path(control_plane)
    if not path.is_file():
        return AppSettingsSnapshot()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return AppSettingsSnapshot()
    if not isinstance(data, dict):
        return AppSettingsSnapshot()

    nested = _nested_llm_inference(data)
    provider_raw = str(
        data.get("llm_inference_provider") or nested.get("provider") or "builtin"
    ).strip()
    provider: LlmInferenceProviderSetting = (
        "openai_compatible" if provider_raw == "openai_compatible" else "builtin"
    )

    return AppSettingsSnapshot(
        default_model_llm=str(data.get("default_model_llm") or "qwen3-vl-4b-instruct").strip()
        or "qwen3-vl-4b-instruct",
        default_model_vlm=str(data.get("default_model_vlm") or "qwen3-vl-4b-instruct").strip()
        or "qwen3-vl-4b-instruct",
        default_model_llm_think=bool(data.get("default_model_llm_think")),
        llm_unload_each_request=bool(data.get("llm_unload_each_request")),
        model_cache_ttl_minutes=max(1, int(data.get("model_cache_ttl_minutes") or 30)),
        llm_cache_ttl_minutes=max(1, int(data.get("llm_cache_ttl_minutes") or 30)),
        llm_inference_provider=provider,
        llm_inference_base_url=str(
            data.get("llm_inference_base_url") or nested.get("base_url") or ""
        ).strip(),
        llm_inference_api_key=str(
            data.get("llm_inference_api_key") or nested.get("api_key") or ""
        ).strip(),
        llm_inference_api_key_hint=str(
            data.get("llm_inference_api_key_hint") or nested.get("api_key_hint") or ""
        ).strip(),
        llm_inference_cloud_model=str(
            data.get("llm_inference_cloud_model") or nested.get("model") or ""
        ).strip(),
        llm_builtin_host=str(
            data.get("llm_builtin_host") or nested.get("host") or "127.0.0.1"
        ).strip()
        or "127.0.0.1",
        llm_builtin_port=max(
            1,
            int(data.get("llm_builtin_port") or nested.get("port") or 7801),
        ),
        llm_quantize_activations=bool(
            data.get("llm_quantize_activations") or nested.get("quantize_activations")
        ),
    )


def settings_dict(*, control_plane: Path | None = None) -> dict[str, Any]:
    snap = load_app_settings(control_plane=control_plane)
    return {
        "default_model_llm": snap.default_model_llm,
        "default_model_vlm": snap.default_model_vlm,
        "default_model_llm_think": snap.default_model_llm_think,
        "llm_unload_each_request": snap.llm_unload_each_request,
        "model_cache_ttl_minutes": snap.model_cache_ttl_minutes,
        "llm_cache_ttl_minutes": snap.llm_cache_ttl_minutes,
        "llm_inference_provider": snap.llm_inference_provider,
        "llm_inference_base_url": snap.llm_inference_base_url,
        "llm_inference_cloud_model": snap.llm_inference_cloud_model,
        "llm_builtin_host": snap.llm_builtin_host,
        "llm_builtin_port": snap.llm_builtin_port,
        "llm_quantize_activations": snap.llm_quantize_activations,
    }
