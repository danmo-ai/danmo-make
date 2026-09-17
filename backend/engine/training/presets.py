"""Training preset profiles (mlx-examples/flux dreambooth defaults)."""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "iterations": 300,
        "lora_rank": 16,
        "lora_blocks": 16,
        "grad_accumulate": 4,
        "warmup_steps": 25,
        "progress_every": 150,
        "checkpoint_every": 150,
        "optimizer": "adamw",
        "min_snr_gamma": 5.0,
    },
    "standard": {
        "iterations": 600,
        "lora_rank": 16,
        "grad_accumulate": 4,
        "progress_every": 300,
        "checkpoint_every": 300,
        "optimizer": "adamw",
        "grad_checkpoint": True,
        "min_snr_gamma": 5.0,
        "val_split": 0.1,
        "val_every": 100,
    },
    "quality": {
        "iterations": 1200,
        "lora_rank": 16,
        "grad_accumulate": 8,
        "progress_every": 600,
        "checkpoint_every": 600,
    },
}

FLUX1_TRAIN_MIN_MEMORY_GB = 50.0
Z_IMAGE_TRAIN_MIN_MEMORY_GB = 48.0
QWEN_IMAGE_TRAIN_MIN_MEMORY_GB = 52.0
TRAINABLE_BASE_MODELS: frozenset[str] = frozenset(
    {"flux1-dev", "z-image", "z-image-turbo", "qwen-image"}
)

# Official Scheme 4 (DiffSynth / HF blog): standard SFT on Z-Image Base, infer on Turbo
# with Z-Image-Turbo-DistillPatch (8 steps, cfg_scale=1 → guidance=0 here).
# https://huggingface.co/blog/kelseye/training-strategies-of-z-image-turbo
Z_IMAGE_SCHEME4_INFERENCE: dict[str, Any] = {
    "scheme": "scheme4",
    "model": "z-image-turbo",
    "extra_adapters": ["z-image-turbo-distillpatch-lora:bf16"],
    "steps": 8,
    "guidance": 0,
    "scheduler": "linear",
    "lora_weight": 0.8,
}

# LoRAs trained directly on Z-Image-Turbo (Ostris assistant on, σ over the 9-step band). They
# are exported without the assistant, so a slightly higher weight than the 0.8 Scheme 4 default
# is needed for the subject to read; below 0.8 identity fades, above 1.0 textures over-bake.
Z_IMAGE_TURBO_INFERENCE: dict[str, Any] = {
    "scheme": "turbo",
    "model": "z-image-turbo",
    "steps": 9,
    "guidance": 0,
    "scheduler": "linear",
    "lora_weight": 0.9,
    "lora_weight_range": [0.8, 1.0],
}

# Base training σ: sample uniform u, apply the SD3-style static shift ``train_sigma_shift``.
# Inference uses shift 6, but training with 6 (let alone 6 + a high bias) puts >60% of samples at
# σ>0.9 where the input is nearly pure noise and only ~7% in the 0.3–0.7 band that decides facial
# structure — LoRAs then never bind identity. Shift 3 keeps the inference-like high-σ emphasis
# while leaving ~30% of supervision in the identity band.
Z_IMAGE_BASE_TRAIN_SIGMA_SHIFT = 3.0

# Iteration counts below are with batch 1; ``iterations // grad_accumulate`` optimizer updates.
# Reference Z-Image LoRA baselines (AI-Toolkit / DiffSynth) use 2000–3000 updates at batch 1,
# so presets keep grad_accumulate small instead of dividing a short run into 4–8× fewer updates.
_Z_IMAGE_BASE_COMMON: dict[str, Any] = {
    "learning_rate": 1e-4,
    "guidance": 5.0,
    "progress_steps": 28,
    "sigma_bias": "uniform",
    "train_sigma_shift": Z_IMAGE_BASE_TRAIN_SIGMA_SHIFT,
    "optimizer": "adamw",
    # Plain flow-match (no min-SNR ε weighting); keeps high-σ identity + low-σ detail bands.
    "min_snr_gamma": 0.0,
    "prior_loss_weight": 0.0,
    # Concept / face datasets are tiny; holding out an image costs identity coverage and a
    # 1-image val loss is too noisy to pick checkpoints. Opt in explicitly via val_split.
    "val_split": 0.0,
    "val_every": 100,
}

# Portrait/concept tuning on top of official SFT defaults (rank 32, all blocks, ~3k steps).
# scheme4_turbo_band_mix: fraction of steps that sample Turbo's 8-step σ band on Base DiT so
# identity survives Base→Turbo inference (DistillPatch alone only restores acceleration).
Z_IMAGE_SCHEME4_CORE: dict[str, Any] = {
    **_Z_IMAGE_BASE_COMMON,
    "iterations": 2500,
    "lora_rank": 32,
    "lora_blocks": -1,
    # MLX attribute is ``to_out`` (sanitize drops diffusers' ``to_out.0``); the matcher also
    # accepts the ``.0`` spelling, but keep the canonical name so all 7 linears get LoRA.
    "lora_module_keys": ["to_q", "to_k", "to_v", "to_out", "w1", "w2", "w3"],
    "grad_accumulate": 1,
    "progress_every": 500,
    "checkpoint_every": 500,
    "scheme4_turbo_band_mix": 0.45,
    "turbo_infer_steps": 8,
    "timestep_low": 1,
    "timestep_high": 8,
    "timestep_bias": "uniform",
    "grad_checkpoint": True,
}

Z_IMAGE_PRESETS: dict[str, dict[str, Any]] = {
    "scheme4": {
        **Z_IMAGE_SCHEME4_CORE,
    },
    "quick": {
        **_Z_IMAGE_BASE_COMMON,
        "iterations": 1000,
        "lora_rank": 16,
        "lora_blocks": 16,
        "grad_accumulate": 1,
        "progress_every": 250,
        "checkpoint_every": 250,
    },
    "standard": {
        **_Z_IMAGE_BASE_COMMON,
        "iterations": 2000,
        "lora_rank": 16,
        "lora_blocks": 24,
        "grad_accumulate": 1,
        "progress_every": 500,
        "checkpoint_every": 500,
        "grad_checkpoint": True,
    },
    "quality": {
        **_Z_IMAGE_BASE_COMMON,
        "iterations": 3000,
        "lora_rank": 16,
        "lora_blocks": -1,
        "grad_accumulate": 2,
        "progress_every": 500,
        "checkpoint_every": 500,
        "learning_rate": 5e-5,
        "grad_checkpoint": True,
    },
}

QWEN_IMAGE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "iterations": 400,
        "lora_rank": 16,
        "lora_blocks": 12,
        "grad_accumulate": 4,
        "progress_every": 200,
        "checkpoint_every": 200,
        "learning_rate": 1e-4,
        "grad_checkpoint": True,
        "optimizer": "adamw",
        "min_snr_gamma": 5.0,
    },
    "standard": {
        "iterations": 800,
        "lora_rank": 16,
        "lora_blocks": 16,
        "grad_accumulate": 4,
        "progress_every": 400,
        "checkpoint_every": 400,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "grad_checkpoint": True,
        "min_snr_gamma": 5.0,
        "val_split": 0.1,
        "val_every": 100,
    },
    "quality": {
        "iterations": 1500,
        "lora_rank": 16,
        "lora_blocks": 24,
        "grad_accumulate": 8,
        "progress_every": 500,
        "checkpoint_every": 500,
        "learning_rate": 5e-5,
    },
}

Z_IMAGE_TURBO_MFLUX_CORE: dict[str, Any] = {
    "lora_rank": 16,
    "lora_blocks": 16,
    "learning_rate": 1e-4,
    "grad_checkpoint": True,
    "guidance": 0.0,
    # Match inference default steps (registry z-image-turbo steps=9) so train σ band aligns with denoise.
    "progress_steps": 9,
    "turbo_infer_steps": 9,
    # Cover the whole 9-step band continuously. Restricting training to steps 4–9 (σ ≤ 0.79)
    # with a low bias never showed the LoRA the first three steps (σ ≈ 1.0 / 0.94 / 0.87) where
    # global layout and facial identity are decided, so faces were not memorized.
    "timestep_low": 1,
    "timestep_high": 9,
    "timestep_bias": "uniform",
    # The Ostris de-distill assistant stays ON for every training step (AI-Toolkit merges it
    # into the base weights). Training part-time on the raw distilled model makes the LoRA
    # spend capacity re-distilling instead of learning the subject.
    "turbo_assistant_off_prob": 0.0,
    "optimizer": "adamw",
    # min-SNR-γ uses the ε-prediction weighting, which disproportionately suppresses the low-σ
    # (high-SNR) end — exactly where skin texture / high-frequency detail is learned — producing
    # over-smoothed ("磨皮") skin. Keep the plain flow-match objective (matches mflux/AI-Toolkit).
    "min_snr_gamma": 0.0,
    "prior_loss_weight": 0.0,
    # See _Z_IMAGE_BASE_COMMON: no held-out face image / noisy 1-image val by default.
    "val_split": 0.0,
    "val_every": 100,
}

Z_IMAGE_TURBO_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        **Z_IMAGE_TURBO_MFLUX_CORE,
        "iterations": 1000,
        "grad_accumulate": 1,
        "progress_every": 250,
        "checkpoint_every": 250,
    },
    "standard": {
        **Z_IMAGE_TURBO_MFLUX_CORE,
        "iterations": 2000,
        "lora_blocks": 24,
        "grad_accumulate": 1,
        "progress_every": 500,
        "checkpoint_every": 500,
    },
    "quality": {
        **Z_IMAGE_TURBO_MFLUX_CORE,
        "iterations": 3000,
        "lora_blocks": -1,
        "grad_accumulate": 2,
        "progress_every": 500,
        "checkpoint_every": 500,
    },
}

Z_IMAGE_TURBO_PRESET_ALIASES: dict[str, str] = {
    "mflux": "standard",
}


def train_min_memory_gb(base_model_id: str) -> float:
    mid = (base_model_id or "").split(":", 1)[0].strip()
    if mid in ("z-image", "z-image-turbo"):
        return Z_IMAGE_TRAIN_MIN_MEMORY_GB
    if mid == "qwen-image":
        return QWEN_IMAGE_TRAIN_MIN_MEMORY_GB
    return FLUX1_TRAIN_MIN_MEMORY_GB


def resolve_preset(name: str | None, *, base_model: str = "flux1-dev") -> dict[str, Any]:
    key = (name or "standard").strip().lower()
    if key == "custom":
        return {}
    mid = (base_model or "").split(":", 1)[0].strip()
    if mid == "z-image-turbo":
        table = Z_IMAGE_TURBO_PRESETS
        key = Z_IMAGE_TURBO_PRESET_ALIASES.get(key, key)
    elif mid == "z-image":
        table = Z_IMAGE_PRESETS
    elif mid == "qwen-image":
        table = QWEN_IMAGE_PRESETS
    else:
        table = PRESETS
    if key == "scheme4" and mid != "z-image":
        raise ValueError(
            "Training preset 'scheme4' is only for z-image (Base); "
            "use quick|standard|quality for z-image-turbo."
        )
    if key not in table:
        raise ValueError(
            f"Unknown training preset {name!r}; choose "
            f"{'scheme4|' if mid == 'z-image' else ''}quick|standard|quality|custom"
        )
    return dict(table[key])


_TRAINING_REQUEST_EXCLUDE = frozenset(
    {
        "base_model",
        "dataset_id",
        "preset",
        "output_name",
        "auto_register",
        "priority",
        "metadata",
        "caption_mode",
    }
)


def merge_training_request_config(request: Any, preset: dict[str, Any]) -> dict[str, Any]:
    """Merge preset with optional request overrides; explicit nulls do not wipe preset."""
    overrides = request.model_dump(exclude=set(_TRAINING_REQUEST_EXCLUDE), exclude_none=True)
    return {**preset, **overrides}
