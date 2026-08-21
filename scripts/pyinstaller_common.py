"""
Shared PyInstaller metadata for Danmo Make desktop sidecar.

Single product profile:
  mlx — macOS (Metal) and Linux (mlx[cuda]): MLX only, no torch.
  cuda / full — unsupported (fail loud).

Windows packaging is temporarily unsupported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from out_paths import FRONTEND_DIST, PROJECT_ROOT

_SHARED_HIDDEN_IMPORTS: tuple[str, ...] = (
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.loops.auto",
    "uvicorn.logging",
    "fastapi.middleware.cors",
    "fastapi.staticfiles",
    "backend.api.routes.adapters",
    "backend.api.routes.assets",
    "backend.api.routes.audios",
    "backend.api.routes.download",
    "backend.api.routes.gallery",
    "backend.api.routes.images",
    "backend.api.routes.models",
    "backend.api.routes.presets",
    "backend.api.routes.queue",
    "backend.api.routes.registry",
    "backend.api.routes.settings",
    "backend.api.routes.system",
    "backend.api.routes.tasks",
    "backend.api.routes.videos",
    "backend.mcp",
    "backend.mcp.server",
    "backend.mcp.bridge",
    "backend.mcp.wait",
    "backend.mcp.base_url",
    "backend.mcp.model_guide",
    "mcp",
    "mcp.server",
    "mcp.server.fastmcp",
    "mcp.server.streamable_http_manager",
    "sse_starlette",
    "backend.core.container",
    "backend.core.i18n",
    "backend.core.interfaces",
    "backend.core.contracts",
    "backend.core.asset_interfaces",
    "backend.core.media_interfaces",
    "backend.core.model_registry",
    "backend.core.registry_format",
    "backend.core.task_kinds",
    "backend.core.install_hooks",
    "backend.engine.engine_registry",
    "backend.engine.danqing_image_engine",
    "backend.engine.danqing_video_engine",
    "backend.engine.danqing_audio_engine",
    "backend.engine.pipelines",
    "backend.engine.pipelines.image_pipeline",
    "backend.engine.pipelines.pipeline_progress",
    "backend.engine.pipelines.image_upscale_pipeline",
    "backend.engine.pipelines.video_pipeline",
    "backend.engine.pipelines.video_upscale_pipeline",
    "backend.engine._transformer_registry",
    "backend.engine.families",
    "backend.engine.families.fibo",
    "backend.engine.families.flux1",
    "backend.engine.families.flux2",
    "backend.engine.families.qwen",
    "backend.engine.families.z_image",
    "backend.engine.families.seedvr2",
    "backend.engine.families.seedvr2.video_upscale",
    "backend.engine.families.hunyuan.video_upscale",
    "backend.engine.video_upscale_registry",
    "backend.engine.families.ltx",
    "backend.engine.families.ltx.generation",
    "backend.engine.families.longcat",
    "backend.engine.families.longcat.generation",
    "backend.engine.families.longcat_avatar",
    "backend.engine.families.longcat_avatar.generation",
    "backend.engine.families.minimax_h3",
    "backend.engine.families.minimax_h3.generation",
    "backend.engine.families.wan.vae",
    "backend.engine.families.wan.conditioning",
    "backend.engine.common.bundle.safetensors_affine_quant",
    "backend.services.services",
    "backend.services.download_service",
    "backend.persistence.stores",
    "backend.persistence.asset_store",
    "backend.persistence.v3_task_store",
    "backend.scheduler.task_scheduler",
    "backend.utils.path_utils",
    "backend.utils.video_sr_ffmpeg",
    "PIL",
    "PIL._imagingtk",
    "PIL._tkinter_finder",
    "psutil",
    "aiohttp",
    "python_multipart",
    "pydantic",
    "huggingface_hub",
    "transformers.models.auto",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.configuration_auto",
    "safetensors",
    "tqdm",
    "requests",
    # Audio WAV persist (ACE-Step / AudioSession) — required at import of audio_persist
    "soundfile",
    "_soundfile",
    "_soundfile_data",
    "cffi",
)

_MLX_ONLY_HIDDEN_IMPORTS: tuple[str, ...] = (
    "backend.engine.runtime.mlx",
    "backend.engine.pipelines.music_pipeline",
    "backend.engine.families.ace_step",
    "backend.engine.families.seedvr2.stem_mlx",
    "backend.engine.families.seedvr2.stem",
    "backend.engine.families.wan.transformer_mlx",
    "backend.engine.families.wan.vae_mlx",
    "backend.engine.families.wan.text_encoder",
    "backend.engine.families.qwen.text_encoder_mlx",
    "backend.engine.common.model.dit_stem",
    "backend.engine.families.flux2.text_encoder_mlx",
    "backend.engine.families.wan.text_encoder_mlx",
    "backend.engine.families.flux1.flux1_dual_mlx",
    "backend.engine.families.fibo.text_encoder_mlx",
    "backend.engine.families.hunyuan.image_encoder_mlx",
    "backend.engine.families.ltx.generation_mlx",
    "backend.engine.families.longcat.generation_mlx",
    "backend.engine.families.longcat.bundle_load_mlx",
    "backend.engine.families.longcat.transformer_mlx",
    "backend.engine.families.longcat.dit_blocks_mlx",
    "backend.engine.families.longcat.dit_attention_mlx",
    "backend.engine.families.longcat.dit_rope_mlx",
    "backend.engine.families.longcat.vae_mlx",
    "backend.engine.families.longcat.text_encoder_mlx",
    "backend.engine.families.longcat.conditioning_mlx",
    "backend.engine.families.longcat.lora_mlx",
    "backend.engine.families.longcat_avatar.generation_mlx",
    "backend.engine.families.longcat_avatar.bundle_load_mlx",
    "backend.engine.families.longcat_avatar.pipeline_mlx",
    "backend.engine.families.longcat_avatar.transformer_mlx",
    "backend.engine.families.longcat_avatar.whisper_mlx",
    "backend.engine.families.longcat_avatar.audio_mlx",
    "backend.engine.families.minimax_h3.generation_mlx",
    "backend.engine.families.minimax_h3.bundle_load_mlx",
    "backend.engine.families.minimax_h3.transformer_mlx",
    "backend.engine.families.minimax_h3.vae_mlx",
    "backend.engine.families.minimax_h3.text_encoder_mlx",
    "backend.engine.families.minimax_h3.packing",
    "mlx_vlm",
    "mlx_vlm.models.qwen3_vl",
    "mlx_vlm.models.qwen3_vl.qwen3_vl",
    "mlx_vlm.models.qwen3_vl.vision",
    "mlx_vlm.models.qwen3_vl.language",
    "mlx_vlm.models.qwen3_vl.config",
    "mlx",
    "mlx.core",
    "mlx._reprlib_fix",
    "mlx_lm",
)

# Always exclude torch (engine is MLX-only; no torch in product bundles).
_MLX_EXCLUDED_MODULES: tuple[str, ...] = (
    "torch",
    "torchvision",
    "torchaudio",
    "torchgen",
    "functorch",
    "triton",
    "cv2",
    "opencv_python",
    "pyarrow",
    "datasets",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "accelerate",
    "bitsandbytes",
    "tensorboard",
    "tensorboard_data_server",
    "torch.utils.tensorboard",
    "hf_xet",
)


def packaging_profile() -> str:
    """Return ``mlx`` (only supported packaging profile)."""
    if sys.platform == "win32":
        raise SystemExit("Windows is temporarily unsupported")

    raw = os.environ.get("DANQING_PYINSTALLER_PROFILE", "").strip().lower()
    if not raw:
        return "mlx"
    if raw in ("cuda", "full"):
        raise SystemExit(
            f"DANQING_PYINSTALLER_PROFILE={raw!r} is unsupported. "
            "Use mlx (macOS Metal / Linux mlx[cuda])."
        )
    if raw != "mlx":
        raise SystemExit(
            f"Unknown DANQING_PYINSTALLER_PROFILE={raw!r}. Only 'mlx' is supported."
        )
    return "mlx"


def get_hidden_imports(profile: str | None = None) -> list[str]:
    profile = profile or packaging_profile()
    if profile != "mlx":
        raise SystemExit(f"Unsupported packaging profile: {profile!r} (only mlx)")
    return list(_SHARED_HIDDEN_IMPORTS) + list(_MLX_ONLY_HIDDEN_IMPORTS)


def get_exclude_modules(profile: str | None = None) -> list[str]:
    profile = profile or packaging_profile()
    if profile != "mlx":
        raise SystemExit(f"Unsupported packaging profile: {profile!r} (only mlx)")
    return list(_MLX_EXCLUDED_MODULES)


def _site_packages_dirs(project_root: Path) -> list[Path]:
    venv_lib = project_root / ".venv" / "lib"
    if not venv_lib.exists():
        return []
    return sorted(venv_lib.glob("python3.*/site-packages"))


def get_data_files(project_root: Path | None = None, *, profile: str | None = None) -> list[str]:
    _ = profile or packaging_profile()
    root = project_root or PROJECT_ROOT
    data: list[str] = []
    separator = ";" if sys.platform == "win32" else ":"

    frontend_dist = FRONTEND_DIST
    if not frontend_dist.is_dir() or not any(frontend_dist.iterdir()):
        raise SystemExit(
            "out/frontend/dist is missing or empty. Build the UI first:\n"
            "  make frontend-build   # or: cd frontend && npm run build"
        )
    data.append(f"{frontend_dist}{separator}frontend/dist")

    default_cfg = root / "default_config"
    if default_cfg.is_dir():
        data.append(f"{default_cfg}{separator}default_config")

    # soundfile loads libsndfile from the ``_soundfile_data`` package (COPYING + __init__).
    for site in _site_packages_dirs(root):
        snd_data = site / "_soundfile_data"
        if snd_data.is_dir():
            for path in snd_data.iterdir():
                if path.is_file() and path.suffix.lower() not in (".dylib", ".so", ".dll"):
                    data.append(f"{path}{separator}_soundfile_data")
            break

    return data


def get_binary_files(project_root: Path, *, profile: str | None = None) -> list[str]:
    """Collect soundfile + MLX native libs (darwin/linux; best-effort if layout differs)."""
    profile = profile or packaging_profile()
    binaries: list[str] = []
    separator = ";" if sys.platform == "win32" else ":"

    for site in _site_packages_dirs(project_root):
        snd_data = site / "_soundfile_data"
        if snd_data.is_dir():
            for pattern in ("*.dylib", "*.so", "*.dll"):
                for lib_file in snd_data.glob(pattern):
                    binaries.append(f"{lib_file}{separator}_soundfile_data")
            break

    if profile != "mlx":
        return binaries

    # Best-effort: macOS uses .dylib/.metallib; Linux mlx[cuda] may ship .so under mlx/lib.
    for site in _site_packages_dirs(project_root):
        mlx_lib = site / "mlx" / "lib"
        if not mlx_lib.is_dir():
            continue
        try:
            for pattern in ("*.dylib", "*.metallib", "*.so"):
                for lib_file in mlx_lib.glob(pattern):
                    binaries.append(f"{lib_file}{separator}mlx/lib")
        except OSError:
            pass
        break

    return binaries


def ensure_runtime_hook_file(project_root: Path) -> Path:
    """Write PyInstaller runtime hook; returns path to hook file."""
    hook_file = project_root / "scripts" / "pyinstaller_runtime_hook.py"
    hook_content = r"""
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
"""
    hook_file.parent.mkdir(parents=True, exist_ok=True)
    hook_file.write_text(hook_content.strip() + "\n", encoding="utf-8")
    return hook_file


def get_runtime_hooks(project_root: Path) -> list[str]:
    return [str(ensure_runtime_hook_file(project_root))]


def apply_pyinstaller_packaging_filters() -> None:
    """Only affects the PyInstaller parent process (not the frozen app)."""
    import logging
    import warnings

    os.environ.setdefault("DANQING_PYINSTALLER_PROFILE", packaging_profile())

    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"PyInstaller\.utils\.hooks",
    )
    for name in (
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def pyinstaller_hooks_dir(project_root: Path) -> Path:
    return project_root / "scripts" / "pyinstaller_hooks"
