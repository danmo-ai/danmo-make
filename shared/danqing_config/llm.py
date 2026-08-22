"""LLM/VLM model id coercion and absolute weight paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from shared.danqing_config.paths import (
    models_registry_path,
    resolve_install_root,
    resolve_registry_local_path,
    workspace_root,
)
from shared.danqing_config.registry import ModelRecord, default_version_local_path, load_registry
from shared.danqing_config.settings import AppSettingsSnapshot, load_app_settings

DEFAULT_ASSISTANT_MODEL_ID = "qwen3-vl-4b-instruct"
DEFAULT_LLM_MODEL_ID = DEFAULT_ASSISTANT_MODEL_ID
DEFAULT_VLM_MODEL_ID = DEFAULT_ASSISTANT_MODEL_ID


def _is_valid_llm(entry: ModelRecord | None) -> bool:
    return entry is not None and entry.media == "llm" and bool(entry.actions & {"chat", "enhance"})


def _is_valid_vlm(entry: ModelRecord | None) -> bool:
    return entry is not None and entry.media == "llm" and "describe" in entry.actions


def is_multimodal_assistant_model(entry: ModelRecord | None) -> bool:
    """Builtin sidecar loads one model; it must support vision (registry ``describe``)."""
    return _is_valid_vlm(entry)


def coerce_assistant_model_id(preferred: str, models: dict[str, ModelRecord]) -> str:
    """Resolve registry id; unknown ids fall back to the default multimodal model."""
    candidate = (preferred or "").strip()
    if candidate and candidate.split(":", 1)[0] in models:
        return candidate.split(":", 1)[0]
    if DEFAULT_ASSISTANT_MODEL_ID in models and is_multimodal_assistant_model(
        models.get(DEFAULT_ASSISTANT_MODEL_ID)
    ):
        return DEFAULT_ASSISTANT_MODEL_ID
    picked = _pick_first_vlm(models)
    return picked or DEFAULT_ASSISTANT_MODEL_ID


def assistant_model_not_multimodal_message(model_id: str) -> str:
    mid = resolve_sidecar_model_alias(model_id)
    return (
        f"Assistant model {mid!r} is not multimodal. "
        "The LLM sidecar loads one model at a time; choose a VLM with vision support "
        f"(default: {DEFAULT_ASSISTANT_MODEL_ID!r})."
    )


def require_multimodal_assistant_model(
    model_id: str,
    models: dict[str, ModelRecord],
) -> None:
    mid = resolve_sidecar_model_alias(model_id)
    if not is_multimodal_assistant_model(models.get(mid)):
        raise RuntimeError(assistant_model_not_multimodal_message(mid))


def _pick_first_llm(models: dict[str, ModelRecord]) -> str | None:
    for mid in sorted(models):
        if _is_valid_llm(models.get(mid)):
            return mid
    return None


def _pick_first_vlm(models: dict[str, ModelRecord]) -> str | None:
    for mid in sorted(models):
        if _is_valid_vlm(models.get(mid)):
            return mid
    return None


def coerce_llm_model_id(preferred: str, models: dict[str, ModelRecord]) -> str:
    candidate = (preferred or "").strip()
    if candidate and _is_valid_llm(models.get(candidate)):
        return candidate
    if _is_valid_llm(models.get(DEFAULT_LLM_MODEL_ID)):
        return DEFAULT_LLM_MODEL_ID
    picked = _pick_first_llm(models)
    return picked or DEFAULT_LLM_MODEL_ID


def coerce_vlm_model_id(preferred: str, models: dict[str, ModelRecord]) -> str:
    candidate = (preferred or "").strip()
    if candidate and _is_valid_vlm(models.get(candidate)):
        return candidate
    if _is_valid_vlm(models.get(DEFAULT_VLM_MODEL_ID)):
        return DEFAULT_VLM_MODEL_ID
    picked = _pick_first_vlm(models)
    return picked or DEFAULT_VLM_MODEL_ID


def is_thinking_model(model_id: str) -> bool:
    mid = (model_id or "").lower()
    if "thinking" in mid:
        return True
    for prefix in ("qwen3.5", "qwen3-5", "qwen3.6", "qwen3-6"):
        if mid.startswith(prefix):
            return True
    return False


def llm_weights_ready(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if (model_dir / "model.safetensors").is_file():
        return True
    shards = list(model_dir.glob("model-*.safetensors"))
    if shards:
        return True
    return any(f.suffix == ".safetensors" for f in model_dir.rglob("*") if f.is_file())


@dataclass(frozen=True)
class LlmBootstrapConfig:
    model_id: str
    model_path: Path
    think_enabled: bool
    chat_template_enable_thinking: bool | None


def resolve_llm_model_path(
    model_id: str,
    *,
    workspace: Path | None = None,
    models: dict[str, ModelRecord] | None = None,
) -> Path:
    ws = workspace or workspace_root()
    registry = models or load_registry(models_registry_path(install_root=resolve_install_root()))
    entry = registry.get(model_id.split(":", 1)[0])
    if entry is None:
        raise RuntimeError(f"unknown LLM model: {model_id!r}")
    local_path = default_version_local_path(entry)
    path = resolve_registry_local_path(local_path, workspace=ws)
    if not llm_weights_ready(path):
        raise RuntimeError(
            f"LLM weights incomplete or missing for {model_id!r} at {path}. "
            "Install the model from the Models page."
        )
    return path


def resolve_vlm_model_path(
    model_id: str,
    *,
    workspace: Path | None = None,
    models: dict[str, ModelRecord] | None = None,
) -> Path:
    ws = workspace or workspace_root()
    registry = models or load_registry(models_registry_path(install_root=resolve_install_root()))
    entry = registry.get(model_id.split(":", 1)[0])
    if entry is None:
        raise RuntimeError(f"unknown VLM model: {model_id!r}")
    local_path = default_version_local_path(entry)
    return resolve_registry_local_path(local_path, workspace=ws)


def resolve_sidecar_model_alias(model_id: str) -> str:
    """OpenAI ``model`` field for builtin sidecar — registry id."""
    return (model_id or "").strip().split(":", 1)[0]


def resolve_model_path_for_id(
    model_id: str,
    *,
    workspace: Path | None = None,
    models: dict[str, ModelRecord] | None = None,
) -> Path:
    """Resolve registry model id to on-disk weights (LLM or VLM)."""
    mid = resolve_sidecar_model_alias(model_id)
    entry = (models or load_registry(models_registry_path())).get(mid)
    if entry is None:
        raise RuntimeError(f"unknown model: {model_id!r}")
    if _is_valid_vlm(entry):
        return resolve_vlm_model_path(mid, workspace=workspace, models=models)
    return resolve_llm_model_path(mid, workspace=workspace, models=models)


def build_sidecar_registry_paths(
    settings: AppSettingsSnapshot,
    models: dict[str, ModelRecord],
    *,
    workspace: Path | None = None,
) -> dict[str, Path]:
    """Single assistant model id → absolute weight directory (sidecar loads one model)."""
    ws = workspace or workspace_root()
    mid = coerce_assistant_model_id(settings.default_model_llm, models)
    try:
        require_multimodal_assistant_model(mid, models)
        return {mid: resolve_model_path_for_id(mid, workspace=ws, models=models)}
    except RuntimeError:
        return {}


def build_sidecar_argv(
    *,
    host: str,
    port: int,
    settings: AppSettingsSnapshot,
    preload_model_path: str | None = None,
) -> list[str]:
    """Arguments for ``python -m mlx_vlm.server`` (after ``-m mlx_vlm.server``)."""
    argv = [
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "INFO",
    ]
    if settings.default_model_llm_think:
        argv.append("--enable-thinking")
    if preload_model_path:
        argv.extend(["--model", preload_model_path])
    return argv


def load_llm_bootstrap_config(
    *,
    install_root: Path | None = None,
    settings: AppSettingsSnapshot | None = None,
) -> LlmBootstrapConfig:
    install = install_root or resolve_install_root()
    ws = workspace_root(install_root=install)
    registry_path = models_registry_path(install_root=install)
    if not registry_path.is_file():
        raise RuntimeError(
            f"models_registry.json not found at {registry_path}. "
            "Run make sync-models-registry or open Settings."
        )
    models = load_registry(registry_path)
    snap = settings or load_app_settings()
    model_id = coerce_assistant_model_id(snap.default_model_llm, models)
    require_multimodal_assistant_model(model_id, models)
    model_path = resolve_vlm_model_path(model_id, workspace=ws, models=models)
    think_enabled = bool(snap.default_model_llm_think) and is_thinking_model(model_id)
    enable_thinking: bool | None = None
    if is_thinking_model(model_id):
        enable_thinking = think_enabled
    return LlmBootstrapConfig(
        model_id=model_id,
        model_path=model_path,
        think_enabled=think_enabled,
        chat_template_enable_thinking=enable_thinking,
    )
