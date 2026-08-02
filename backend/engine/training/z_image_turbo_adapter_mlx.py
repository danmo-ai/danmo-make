"""Z-Image-Turbo training adapter v2 (Ostris de-distill LoRA) for MLX DreamBooth."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.bundle.lora_mlx import read_lora_config
from backend.engine.common.bundle.weights import load_safetensors
from backend.engine.families.z_image.weights import remap_zimage_lora_keys
from backend.engine.runtime._base import RuntimeContext
from backend.engine.training.lora_layers_mlx import (
    FrozenAssistantLinear,
    LoRALinear,
    _dit_inner,
    _replace_module_at_path,
    _walk_all_linear_paths,
    set_training_assistant_enabled,
)

TRAINING_ADAPTER_REPO = "ostris/zimage_turbo_training_adapter"
TRAINING_ADAPTER_MS_REPO = "ostris/zimage_turbo_training_adapter"
TRAINING_ADAPTER_FILE = "zimage_turbo_training_adapter_v2.safetensors"
LOCAL_ADAPTER_REL = Path("models/Lora/_training_adapters") / TRAINING_ADAPTER_FILE


class TurboTrainingAssistantHandle:
    """Frozen Ostris v2 assistant layers toggled off for turbo-style preview."""

    def __init__(self, layers: list[Any]):
        self._layers = layers

    @property
    def count(self) -> int:
        return len(self._layers)

    def set_enabled(self, enabled: bool, ctx: RuntimeContext | None = None) -> None:
        set_training_assistant_enabled(self._layers, enabled=enabled)


def _find_adapter_file_in_dir(root: Path) -> Path | None:
    direct = root / TRAINING_ADAPTER_FILE
    if direct.is_file():
        return direct
    matches = sorted(root.rglob(TRAINING_ADAPTER_FILE))
    if matches:
        return matches[0]
    return None


def _download_training_adapter_via_huggingface() -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        downloaded = hf_hub_download(
            repo_id=TRAINING_ADAPTER_REPO,
            filename=TRAINING_ADAPTER_FILE,
        )
    except Exception:
        return None
    path = Path(downloaded)
    return path if path.is_file() else None


def _download_training_adapter_via_modelscope() -> Path | None:
    try:
        from modelscope import snapshot_download
    except ImportError:
        return None
    try:
        root = snapshot_download(
            TRAINING_ADAPTER_MS_REPO,
            allow_file_pattern=[TRAINING_ADAPTER_FILE],
        )
    except Exception:
        return None
    return _find_adapter_file_in_dir(Path(root))


def _cache_training_adapter(project_root: Path, source: Path) -> Path:
    dest = project_root / LOCAL_ADAPTER_REL
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != source.resolve():
        dest.write_bytes(source.read_bytes())
    return dest


def resolve_zimage_turbo_training_adapter_path(project_root: Path) -> Path:
    """Return a local path to the Ostris Turbo training adapter v2, downloading if needed."""
    local = project_root / LOCAL_ADAPTER_REL
    if local.is_file():
        return local

    for downloader in (_download_training_adapter_via_modelscope, _download_training_adapter_via_huggingface):
        fetched = downloader()
        if fetched is not None:
            return _cache_training_adapter(project_root, fetched)

    raise RuntimeError(
        "Z-Image-Turbo LoRA training requires the Ostris training adapter v2 "
        f"({TRAINING_ADAPTER_FILE}, ~340MB). "
        f"Download failed from Hugging Face and ModelScope. "
        f"Place the file manually at: {local} "
        f"(HF: huggingface.co/{TRAINING_ADAPTER_REPO} · "
        f"ModelScope: modelscope.cn/models/{TRAINING_ADAPTER_MS_REPO})"
    )


def _load_remapped_adapter_weights(
    adapter_path: Path,
    model: Any,
    *,
    repair_indexed_weights: Any | None = None,
) -> tuple[dict[str, tuple[mx.array, mx.array, float]], dict[str, mx.array]]:
    path = Path(adapter_path)
    if not path.is_file():
        raise RuntimeError(f"Z-Image-Turbo training adapter not found: {path}")

    weights = dict(load_safetensors(str(path)))
    lora_config = read_lora_config(path.parent if path.parent.name != "_training_adapters" else path.parent)
    if repair_indexed_weights is not None:
        from backend.engine.common.bundle.lora_mlx import _weights_use_indexed_lora_keys

        if _weights_use_indexed_lora_keys(weights):
            weights = repair_indexed_weights(weights, model, lora_config)

    patch_size = int(getattr(getattr(model, "config", None), "patch_size", 2) or 2)
    groups = remap_zimage_lora_keys(weights, patch_size=patch_size)
    dense_deltas = {
        k[: -len(".delta.weight")]: v
        for k, v in weights.items()
        if k.endswith(".delta.weight")
    }
    if not groups and not dense_deltas:
        raise RuntimeError(
            f"Z-Image-Turbo training adapter {path}: no remappable LoRA pairs after key normalization"
        )

    config_alpha = lora_config.get("lora_alpha", lora_config.get("alpha"))
    if config_alpha is not None and not any(".alpha" in key.lower() for key in weights):
        alpha_val = float(config_alpha)
        groups = {key: (down, up, alpha_val) for key, (down, up, _) in groups.items()}

    return groups, dense_deltas


def install_zimage_turbo_training_assistant(
    model: Any,
    adapter_path: Path | str,
    ctx: RuntimeContext,
    *,
    strength: float = 1.0,
    repair_indexed_weights: Any | None = None,
) -> TurboTrainingAssistantHandle:
    """Attach Ostris v2 assistant as frozen overlays (mflux-style; base weights unchanged)."""
    groups, dense_deltas = _load_remapped_adapter_weights(
        Path(adapter_path),
        model,
        repair_indexed_weights=repair_indexed_weights,
    )

    dit = _dit_inner(model)
    path_map = {path: layer for layer, path in _walk_all_linear_paths(dit)}
    assistant_layers: list[Any] = []
    applied = 0

    for module_name, delta in dense_deltas.items():
        layer = path_map.get(module_name)
        if layer is None:
            continue
        if isinstance(layer, LoRALinear):
            layer.attach_frozen_assistant_dense(delta, strength=strength)
        elif isinstance(layer, nn.Linear):
            wrapped = FrozenAssistantLinear.wrap_linear_dense(layer, delta, strength=strength)
            _replace_module_at_path(dit, module_name, wrapped)
            layer = wrapped
        elif isinstance(layer, FrozenAssistantLinear):
            layer = FrozenAssistantLinear.wrap_linear_dense(layer.linear, delta, strength=strength)
            _replace_module_at_path(dit, module_name, layer)
        else:
            continue
        assistant_layers.append(layer)
        applied += 1

    for module_name, (down, up, alpha) in groups.items():
        if module_name in dense_deltas:
            continue
        layer = path_map.get(module_name)
        if layer is None:
            continue
        if isinstance(layer, LoRALinear):
            layer.attach_frozen_assistant(down, up, alpha, strength=strength)
        elif isinstance(layer, nn.Linear):
            wrapped = FrozenAssistantLinear.wrap_linear(layer, down, up, alpha, strength=strength)
            _replace_module_at_path(dit, module_name, wrapped)
            layer = wrapped
        elif isinstance(layer, FrozenAssistantLinear):
            layer = FrozenAssistantLinear.wrap_linear(layer.linear, down, up, alpha, strength=strength)
            _replace_module_at_path(dit, module_name, layer)
        else:
            continue
        assistant_layers.append(layer)
        applied += 1

    if applied == 0:
        raise RuntimeError(
            f"Z-Image-Turbo training adapter {adapter_path}: remap produced "
            f"{len(groups)} LoRA groups and {len(dense_deltas)} dense deltas "
            "but none matched this Z-Image DiT."
        )

    if hasattr(dit, "_build_param_map"):
        dit._build_param_map()
    if hasattr(model, "_inner") and hasattr(dit, "_param_map"):
        model._param_map = dit._param_map
    ctx.eval(*[t for _, t in model.parameters()])
    return TurboTrainingAssistantHandle(assistant_layers)
