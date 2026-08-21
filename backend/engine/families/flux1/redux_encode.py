"""FLUX.1 Redux — MLX native encode."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def encode_redux_context_tokens(
    pil: Image.Image,
    *,
    redux_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return ``[1, seq, 4096]`` float32 tokens to concat after T5 ``txt_embeds``."""
    from backend.engine.families.flux1.redux_encode_mlx import encode_redux_context_tokens_mlx

    return encode_redux_context_tokens_mlx(
        pil, redux_bundle_root=redux_bundle_root, on_log=on_log
    )


def resolve_redux_bundle_root(registry: Any, project_root: Path, controlnet_model_id: str) -> Path:
    from backend.engine.contracts.pipeline_registry import local_bundle_root as bundle_root_fn
    from backend.engine.contracts.pipeline_registry import resolve_version_block as version_block_fn

    entry = registry.require(controlnet_model_id)
    version_key = version_block_fn(entry, None)
    root = bundle_root_fn(project_root, entry, version_key)
    if root is None:
        raise RuntimeError(f"controlnet bundle path missing for {controlnet_model_id!r}")
    return root
