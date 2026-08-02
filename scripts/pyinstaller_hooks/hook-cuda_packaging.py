"""Drop MLX when ``DANQING_PYINSTALLER_PROFILE=cuda`` (Linux/Windows CUDA bundles)."""

from __future__ import annotations

import os
import sys

excludedimports: list[str] = []

_profile = os.environ.get("DANQING_PYINSTALLER_PROFILE", "").strip().lower()
if _profile in ("cuda", "full") or (
    sys.platform != "darwin" and _profile != "mlx"
):
    excludedimports = [
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_lm",
        "backend.engine.runtime.mlx",
        "backend.engine.families.seedvr2.stem_mlx",
        "backend.engine.families.seedvr2.stem",
        "backend.engine.families.wan.transformer_mlx",
        "backend.engine.families.wan.vae_mlx",
        "backend.engine.families.wan.text_encoder_mlx",
        "backend.engine.families.ltx.generation_mlx",
        "backend.engine.families.longcat.generation_mlx",
        "backend.engine.families.longcat.bundle_load_mlx",
        "backend.engine.families.longcat.transformer_mlx",
        "backend.engine.families.longcat_avatar.generation_mlx",
        "backend.engine.families.longcat_avatar.bundle_load_mlx",
        "backend.engine.families.qwen.text_encoder_mlx",
        "backend.engine.families.flux2.text_encoder_mlx",
        "backend.engine.families.hunyuan.image_encoder_mlx",
    ]
