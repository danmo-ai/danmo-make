import os
import sys
from pathlib import Path

# PyInstaller: writable dirs + MLX metallib next to bundled dylibs.
if getattr(sys, "frozen", False):
    raw = os.environ.get("DANQING_USER_DATA_DIR")
    if raw:
        app_dir = Path(raw).expanduser().resolve()
    else:
        app_dir = (Path.home() / ".danmo-make").expanduser().resolve()
    for dir_name in ("models", "outputs", "db", "config", "logs"):
        (app_dir / dir_name).mkdir(parents=True, exist_ok=True)

    # MLX metallib is copied next to the executable by scripts/prune_sidecar.layout_mlx_runtime.
