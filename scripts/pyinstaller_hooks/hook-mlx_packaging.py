"""Drop torch / heavy optional deps when ``DANQING_PYINSTALLER_PROFILE=mlx``.

Applies on any OS for the mlx packaging profile (macOS Metal + Linux mlx[cuda]).
"""

from __future__ import annotations

import os

excludedimports: list[str] = []

_profile = os.environ.get("DANQING_PYINSTALLER_PROFILE", "").strip().lower()
# Default product profile is mlx when unset (darwin/linux packaging).
if _profile in ("", "mlx"):
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
        "hf_xet",
    ]
