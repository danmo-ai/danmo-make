"""Bootstrap and launch mlx_vlm.server."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from shared.danqing_config.llm import build_sidecar_argv
from shared.danqing_config.settings import load_app_settings

logger = logging.getLogger(__name__)


def _maybe_set_cuda_activation_quantization(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("MLX_VLM_QUANTIZE_ACTIVATIONS", "1")


def launch_server(*, host: str, port: int, preload_model: str | None = None) -> None:
    """Replace ``sys.argv`` and run ``mlx_vlm.server`` (blocks until exit)."""
    settings = load_app_settings()
    _maybe_set_cuda_activation_quantization(settings.llm_quantize_activations)

    argv = [
        "mlx_vlm.server",
        *build_sidecar_argv(
            host=host,
            port=port,
            settings=settings,
            preload_model_path=preload_model,
        ),
    ]
    logger.info("Starting mlx_vlm.server: %s", " ".join(argv[1:]))
    sys.argv = argv
    from mlx_vlm.server import main as server_main

    server_main()


def resolve_preload_model() -> str | None:
    """Optional default VLM path to warm at startup."""
    preload_env = os.environ.get("DANQING_LLM_PRELOAD_MODEL", "").strip()
    if preload_env:
        path = Path(preload_env).expanduser()
        if path.is_dir():
            return str(path.resolve())
        return preload_env
    try:
        from backend_llm.registry_map import assistant_model_map, default_assistant_id

        model_map = assistant_model_map()
        assistant_id = default_assistant_id()
        path = model_map.get(assistant_id)
        if path is not None and path.is_dir():
            return str(path.resolve())
    except Exception as exc:
        logger.warning("Could not resolve preload VLM path: %s", exc)
    return None
