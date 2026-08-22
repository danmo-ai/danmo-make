"""Shared workspace / registry / LLM settings resolution for danqing-api and backend_llm sidecar."""

from shared.danqing_config.inference import (
    LlmInferenceConfig,
    load_llm_inference_config,
    resolve_builtin_base_url,
)
from shared.danqing_config.llm import (
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_VLM_MODEL_ID,
    LlmBootstrapConfig,
    build_sidecar_argv,
    build_sidecar_registry_paths,
    coerce_llm_model_id,
    coerce_vlm_model_id,
    is_thinking_model,
    load_llm_bootstrap_config,
    resolve_llm_model_path,
    resolve_model_path_for_id,
    resolve_sidecar_model_alias,
    resolve_vlm_model_path,
)
from shared.danqing_config.paths import (
    control_plane_dir,
    control_settings_path,
    llm_port_file,
    llm_pid_file,
    models_registry_path,
    resolve_install_root,
    resolve_registry_local_path,
    workspace_root,
)

__all__ = [
    "DEFAULT_LLM_MODEL_ID",
    "DEFAULT_VLM_MODEL_ID",
    "LlmBootstrapConfig",
    "LlmInferenceConfig",
    "build_sidecar_argv",
    "build_sidecar_registry_paths",
    "coerce_llm_model_id",
    "coerce_vlm_model_id",
    "control_plane_dir",
    "control_settings_path",
    "is_thinking_model",
    "llm_port_file",
    "llm_pid_file",
    "load_llm_bootstrap_config",
    "load_llm_inference_config",
    "models_registry_path",
    "resolve_builtin_base_url",
    "resolve_install_root",
    "resolve_registry_local_path",
    "resolve_llm_model_path",
    "resolve_model_path_for_id",
    "resolve_sidecar_model_alias",
    "resolve_vlm_model_path",
    "workspace_root",
]
