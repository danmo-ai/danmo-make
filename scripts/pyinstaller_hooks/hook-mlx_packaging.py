"""Drop CUDA / PyTorch when ``DANQING_PYINSTALLER_PROFILE=mlx`` (default on macOS)."""

from __future__ import annotations

import os
import sys

excludedimports: list[str] = []

if os.environ.get("DANQING_PYINSTALLER_PROFILE", "").lower() == "mlx" or (
    sys.platform == "darwin"
    and os.environ.get("DANQING_PYINSTALLER_PROFILE", "").lower() not in ("cuda", "full")
):
    excludedimports = [
        "torch",
        "torchvision",
        "torchaudio",
        "cv2",
        "pyarrow",
        "datasets",
        "pandas",
        "matplotlib",
        "scipy",
        "backend.engine.runtime.cuda",
        "backend.engine.common.codecs.text_encoders.t5_cuda",
        "backend.engine.common.codecs.text_encoders.clip_cuda",
        "backend.engine.common.codecs.text_encoders.qwen25vl_cuda",
        "backend.engine.families.cogview4.text_encoder_cuda",
        "backend.engine.families.z_image.text_encoder_cuda",
        "backend.engine.families.diffrhythm.mulan_cuda",
        "backend.engine.families.flux1.redux_encode_cuda",
        "backend.engine.families.flux1.depth_encode_cuda",
        "backend.engine.families.ace_step.quality.resource_policy_cuda",
        "backend.engine.families.ace_step.transformer_cuda",
        "backend.engine.families.ace_step.vae.vae_cuda",
        "backend.engine.families.ace_step.generation_cuda",
        "hf_xet",
    ]
