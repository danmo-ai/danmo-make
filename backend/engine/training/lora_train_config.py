"""Load LoRA training overrides from JSON/YAML config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_lora_train_config_file(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise RuntimeError(f"LoRA training config not found: {p}")
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "YAML training config requires PyYAML (`pip install pyyaml`) or use a .json config"
            ) from e
        data = yaml.safe_load(text)
    else:
        raise RuntimeError(f"Unsupported training config format {suffix!r}; use .json or .yaml")
    if not isinstance(data, dict):
        raise RuntimeError(f"Training config root must be an object (got {type(data).__name__})")
    return data


def resolve_training_config(params: dict[str, Any]) -> dict[str, Any]:
    """Resolve the same preset/overrides used by all three training runners."""
    from types import SimpleNamespace

    from backend.engine.training.presets import merge_training_request_config, resolve_preset

    base_model = str(params.get("base_model") or "").split(":", 1)[0]
    request = SimpleNamespace(
        model_dump=lambda **kwargs: {
            key: value for key, value in params.items()
            if key not in kwargs.get("exclude", set()) and value is not None
        }
    )
    preset = resolve_preset(str(params.get("preset") or "standard"), base_model=base_model)
    return merge_training_request_config(request, preset)


def training_inference_config(params: dict[str, Any]) -> dict[str, Any] | None:
    """Portable, explicit inference recipe stored with every face adapter."""
    from backend.engine.training.presets import Z_IMAGE_TURBO_INFERENCE, Z_IMAGE_TURBO_SFT_INFERENCE

    cfg = resolve_training_config(params)
    base_model = str(params.get("base_model") or "").split(":", 1)[0]
    preset = str(params.get("preset") or "standard").strip().lower()
    recipe: dict[str, Any] | None = None
    if base_model == "z-image-turbo" and preset == "sft":
        recipe = dict(Z_IMAGE_TURBO_SFT_INFERENCE)
    elif base_model == "z-image-turbo":
        recipe = {
            **Z_IMAGE_TURBO_INFERENCE,
            "steps": int(cfg.get("turbo_infer_steps") or 9),
        }
    elif base_model in {"z-image", "qwen-image"}:
        recipe = {
            "scheme": "base",
            "model": base_model,
            "steps": int(cfg.get("progress_steps") or (28 if base_model == "z-image" else 30)),
            "guidance": float(cfg.get("guidance") or 0),
            "lora_weight": 1.0,
        }
    metadata = params.get("metadata") or {}
    requested = metadata.get("inference") if isinstance(metadata, dict) else None
    if requested is not None:
        if not isinstance(requested, dict):
            raise ValueError("metadata.inference must be an object")
        allowed = {
            "scheme",
            "model",
            "steps",
            "guidance",
            "lora_weight",
            "lora_weight_range",
            "scheduler",
            "extra_adapters",
        }
        unknown = sorted(set(requested) - allowed)
        if unknown:
            raise ValueError(f"metadata.inference contains unsupported keys: {unknown}")
        recipe = {**(recipe or {}), **requested}
    if recipe is None:
        return None
    model = str(recipe.get("model") or base_model).split(":", 1)[0]
    if model != base_model:
        raise ValueError(
            f"Inference model {model!r} must equal training base {base_model!r}; "
            "LoRA training and inference cannot cross base models."
        )
    recipe["model"] = model
    recipe["steps"] = int(recipe.get("steps") or 30)
    recipe["guidance"] = float(recipe.get("guidance") or 0.0)
    recipe["lora_weight"] = float(recipe.get("lora_weight") or 1.0)
    extra_adapters = recipe.get("extra_adapters", [])
    if not isinstance(extra_adapters, list) or not all(isinstance(item, str) and item.strip() for item in extra_adapters):
        raise ValueError("inference.extra_adapters must be a list of adapter ids")
    recipe["extra_adapters"] = [item.strip() for item in extra_adapters]
    if not 1 <= recipe["steps"] <= 150:
        raise ValueError("inference.steps must be in [1, 150]")
    if not 0 <= recipe["guidance"] <= 30:
        raise ValueError("inference.guidance must be in [0, 30]")
    if not 0 <= recipe["lora_weight"] <= 2:
        raise ValueError("inference.lora_weight must be in [0, 2]")
    scheme = recipe.get("scheme")
    if scheme is not None and scheme not in {"sft", "turbo", "base"}:
        raise ValueError(f"inference.scheme {scheme!r} is not a known face-LoRA recipe")
    weight_range = recipe.get("lora_weight_range")
    if weight_range is not None and not (
        isinstance(weight_range, list)
        and len(weight_range) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in weight_range)
    ):
        raise ValueError("inference.lora_weight_range must be [low, high]")
    return recipe


_TRAINING_FAMILY_BY_MODEL = {
    "flux1-dev": "flux1",
    "z-image": "z_image",
    "z-image-turbo": "z_image",
    "qwen-image": "qwen_image",
}


def adapter_export_config(
    *,
    params: dict[str, Any],
    checkpoint_meta: dict[str, Any] | None = None,
    dataset_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive every scale/module field from the actual resolved training configuration."""
    cfg = resolve_training_config(params)
    meta = checkpoint_meta or {}
    dataset = dataset_meta or {}
    if checkpoint_meta is not None:
        required = ("base_model", "lora_rank", "lora_alpha", "lora_blocks", "train_type", "iteration")
        missing = [key for key in required if meta.get(key) is None]
        if missing:
            raise ValueError(
                f"Checkpoint metadata is incomplete ({', '.join(missing)}); "
                "refusing to guess the adapter contract during registration."
            )
    base_model = str(params.get("base_model") or meta.get("base_model") or "").split(":", 1)[0]
    if meta.get("base_model") and str(meta["base_model"]).split(":", 1)[0] != base_model:
        raise ValueError("Checkpoint base model differs from the task configuration; refusing an incorrect export.")
    train_type = str(cfg.get("train_type") or "lora")
    if meta.get("train_type") is not None and str(meta["train_type"]) != train_type:
        raise ValueError("Checkpoint adapter type differs from the task's resolved train_type; refusing an incorrect export.")
    rank = int(cfg.get("lora_rank") or 8)
    alpha = int(cfg.get("lora_alpha")) if cfg.get("lora_alpha") is not None else (
        int(round(float(cfg["lora_scale"]) * rank)) if cfg.get("lora_scale") is not None else rank
    )
    if alpha <= 0:
        raise ValueError("Resolved lora_alpha must be positive.")
    blocks = int(cfg.get("lora_blocks") if cfg.get("lora_blocks") is not None else -1)
    for key, value in (("lora_rank", rank), ("lora_alpha", alpha), ("lora_blocks", blocks)):
        if meta.get(key) is not None and int(meta[key]) != value:
            raise ValueError(f"Checkpoint {key} differs from the task configuration; refusing an incorrect export.")
    if "lora_module_keys" in meta:
        if list(meta.get("lora_module_keys") or []) != list(cfg.get("lora_module_keys") or []):
            raise ValueError("Checkpoint LoRA module keys differ from the task configuration; refusing an incorrect export.")
    out: dict[str, Any] = {
        "format": "dora" if train_type == "dora" else "lora",
        "train_type": train_type,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "alpha": alpha,
        "lora_blocks": blocks,
        "base_model": base_model,
        "family": _TRAINING_FAMILY_BY_MODEL.get(base_model, ""),
        "lora_module_keys": list(cfg.get("lora_module_keys") or []),
        "trigger_word": str(dataset.get("trigger_word") or "").strip(),
        "training_caption": str(params.get("progress_prompt") or ""),
        "training_config": {
            **({"preset": params.get("preset")} if params.get("preset") else {}),
            **{
                key: cfg.get(key) for key in (  # noqa: C408
                "preset", "iterations", "seed", "learning_rate", "grad_accumulate", "lora_rank",
                "lora_alpha", "lora_blocks", "lora_module_keys", "sigma_bias",
                "train_sigma_shift", "caption_dropout", "caption_mode", "resolution",
                "face_crop", "train_type", "min_snr_gamma", "guidance", "progress_steps",
                "turbo_infer_steps", "timestep_low",
                "timestep_high", "timestep_bias", "num_augmentations", "lora_dropout",
                "optimizer", "val_split", "val_every", "qlora_bits", "grad_checkpoint",
                ) if cfg.get(key) is not None
            },
        },
    }
    inference = training_inference_config(params)
    if inference is not None:
        out["inference"] = inference
    return out


def publish_trained_adapter(
    *,
    final_path: Path,
    output_name: str,
    dest_dir: Path,
    config_dir: Path,
    params: dict[str, Any],
    dataset_meta: dict[str, Any],
    checkpoint_meta: dict[str, Any] | None = None,
    auto_register: bool,
    task_id: str,
) -> dict[str, str]:
    """Write one identical adapter/config contract for auto and manual registration."""
    import shutil

    from backend.engine.training.user_lora_registry import register_user_lora

    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in output_name.strip())[:64]
    slug = slug.strip("-") or f"face-lora-{task_id[-8:]}"
    # dest_dir is the user-LoRA root (models/Lora). The registry path is models/Lora/{slug}.
    dest_dir = dest_dir / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "adapter.safetensors"
    if dest_file.exists() and dest_file.read_bytes() != final_path.read_bytes():
        raise RuntimeError(
            f"LoRA output name is already in use: {dest_dir}. Choose a unique name; "
            "refusing to overwrite a different trained adapter."
        )
    shutil.copy2(final_path, dest_file)
    checkpoint_meta = dict(checkpoint_meta or {})
    sidecar_path = final_path.with_suffix(".json")
    if not checkpoint_meta and sidecar_path.is_file():
        checkpoint_meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
    config = adapter_export_config(params=params, checkpoint_meta=checkpoint_meta, dataset_meta=dataset_meta)
    config["task_id"] = task_id
    (dest_dir / "lora_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    user_lora_id = ""
    if auto_register:
        entry = register_user_lora(
            config_dir,
            name=output_name.strip() or slug,
            base_model=config["base_model"],
            local_path=f"models/Lora/{slug}",
            trigger_word=config["trigger_word"],
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            family=str(config.get("family") or ""),
            lora_module_keys=config.get("lora_module_keys"),
            train_type=str(config.get("train_type") or "lora"),
            inference=config.get("inference"),
            task_id=task_id,
        )
        user_lora_id = entry["id"]
    return {
        "adapter_path": str(dest_file),
        "user_lora_id": user_lora_id,
        "output_name": slug,
    }


def finalize_training_adapters(
    *,
    work_dir: Path,
    adapter_dir: Path,
    final_path: Path,
    output_name: str,
    params: dict[str, Any],
    dataset_meta: dict[str, Any],
    loss_history: list[dict[str, Any]],
    training_caption: str,
    path_resolver: Any,
    task_id: str,
    auto_register: bool,
) -> dict[str, Any]:
    """Publish immediately (CLI) or keep checkpoints for the candidate matrix (Studio)."""
    if auto_register:
        published = publish_trained_adapter(
            final_path=final_path,
            checkpoint_meta=json.loads(final_path.with_suffix(".json").read_text(encoding="utf-8"))
            if final_path.with_suffix(".json").is_file()
            else {},
            output_name=output_name,
            dest_dir=path_resolver.get_loras_dir(),
            config_dir=path_resolver.get_workspace_config_dir(),
            params=params,
            dataset_meta=dataset_meta,
            auto_register=True,
            task_id=task_id,
        )
        return {
            **published,
            "loss_history": loss_history,
            "training_caption": training_caption,
            "needs_candidate_evaluation": False,
            "checkpoints": [],
        }
    from backend.engine.training.lora_checkpoint_evaluation import finish_training_without_register

    return finish_training_without_register(
        work_dir=work_dir,
        adapter_dir=adapter_dir,
        final_path=final_path,
        output_name=output_name,
        params=params,
        dataset_meta=dataset_meta,
        loss_history=loss_history,
        training_caption=training_caption,
        task_id=task_id,
    )


def merge_config_into_request_dict(base: dict[str, Any], config_path: Path | str) -> dict[str, Any]:
    """CLI/API helper: file values fill gaps; explicit base keys win."""
    file_cfg = load_lora_train_config_file(config_path)
    return {**file_cfg, **base}
