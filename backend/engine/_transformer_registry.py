"""Transformer / Text Encoder 注册表 — 新增模型只需在此添加条目。

Weight remapping has been internalized into each model's ``sanitize()`` method
on ``TransformerBase`` subclasses (see ``backend/engine/common/model/base.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import importlib


def _clear_text_encoder_mlx_cache(ctx: Any) -> None:
    clear_cache_fn = getattr(ctx, "clear_cache", None)
    if clear_cache_fn is not None:
        clear_cache_fn()
    else:
        importlib.import_module("mlx.core").clear_cache()


def _release_image_text_encoder_if_configured(enc: Any, ctx: Any, config: Any) -> None:
    """Drop TE weights after encode when ``release_text_encoder_after_encode`` (default on)."""
    if enc is None or not bool(getattr(config, "release_text_encoder_after_encode", True)):
        return
    release_fn = getattr(enc, "release_weights", None)
    if release_fn is not None:
        release_fn()
    else:
        _clear_text_encoder_mlx_cache(ctx)


# (模块路径, 类名)
_TRANSFORMER = {
    "z_image":  ("backend.engine.families.z_image.transformer",  "ZImageTransformer"),
    "flux2":    ("backend.engine.families.flux2.transformer",     "Flux2Transformer"),
    "fibo":     ("backend.engine.families.fibo.transformer",      "FIBOTransformer"),
    "flux1":    ("backend.engine.families.flux1.transformer",     "Flux1Transformer"),
    "qwen_image": ("backend.engine.families.qwen.transformer",    "QwenImageTransformer"),
    "ernie_image": ("backend.engine.families.ernie_image.transformer", "ErnieImageTransformer"),
    "cogview4": ("backend.engine.families.cogview4.transformer", "CogView4Transformer"),
    "hidream_o1": ("backend.engine.families.hidream_o1.transformer", "HiDreamO1Transformer"),
    "step1x_edit": ("backend.engine.families.step1x_edit.transformer", "Step1XEditTransformer"),
    "boogu_image": ("backend.engine.families.boogu.transformer", "BooguImageTransformer"),
}

# encoder_type → (模块路径, 类名)
_TEXT_ENCODER = {
    "flux1":    ("backend.engine.families.flux1.text_encoder",     "Flux1TextEncoder"),
    "flux2":    ("backend.engine.families.flux2.text_encoder",     "Flux2TextEncoder"),
    "z_image":  ("backend.engine.families.z_image.text_encoder",   "ZImageTextEncoder"),
    "qwen_image": ("backend.engine.families.qwen.text_encoder", "QwenImageTextEncoder"),
    "fibo":     ("backend.engine.families.fibo.text_encoder", "FiboTextEncoder"),
    "ernie_image": ("backend.engine.families.ernie_image.text_encoder", "ErnieImageTextEncoder"),
    "cogview4": ("backend.engine.families.cogview4.text_encoder", "CogView4TextEncoder"),
}

_IMAGE_LORA_MERGE = {
    "flux1": ("backend.engine.families.flux1.lora_mlx", "merge_flux1_lora_adapters"),
    "flux2": ("backend.engine.families.flux2.lora_mlx", "merge_flux2_lora_adapters"),
    "z_image": ("backend.engine.families.z_image.lora_mlx", "merge_z_image_lora_adapters"),
    "qwen_image": ("backend.engine.families.qwen.lora_mlx", "merge_qwen_image_lora_adapters"),
}

_AUDIO_LORA_MERGE = {
    "ace_step": ("backend.engine.families.ace_step.lora_mlx", "merge_ace_step_lora_adapters"),
}

_VIDEO_LORA_MERGE = {
    "wan": ("backend.engine.families.wan.lora_mlx", "merge_wan_lora_adapters"),
}

_IMAGE_EDIT_EXTRA_COND = {
    "fibo": ("backend.engine.families.fibo.vae_mlx", "attach_edit_conditioning_extra"),
}


def get_transformer_class(family: str):
    import importlib
    entry = _TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown image model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_text_encoder(encoder_type: str):
    import importlib
    entry = _TEXT_ENCODER.get(encoder_type)
    if entry is None:
        raise RuntimeError(f"Unknown encoder type: {encoder_type}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_image_lora_merge(family: str):
    import importlib

    entry = _IMAGE_LORA_MERGE.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_audio_lora_merge(family: str):
    import importlib

    entry = _AUDIO_LORA_MERGE.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_video_lora_merge(family: str):
    import importlib

    entry = _VIDEO_LORA_MERGE.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def merge_image_lora_adapters(
    *,
    family: str,
    model: Any,
    adapters: list[Any],
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: Any,
    on_log: Any | None = None,
) -> None:
    merge_fn = get_image_lora_merge(family)
    if merge_fn is None:
        supported = ", ".join(sorted(_IMAGE_LORA_MERGE.keys()))
        raise RuntimeError(
            "LoRA adapters require in-engine merging; supported image families are "
            f"{supported} (this model is family={family!r}). Remove adapters or switch model."
        )
    kwargs = dict(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        on_log=on_log,
    )
    merge_fn(**kwargs)


def merge_audio_lora_adapters(
    *,
    family: str,
    model: Any,
    adapters: list[Any],
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: Any,
    on_log: Any | None = None,
) -> None:
    merge_fn = get_audio_lora_merge(family)
    if merge_fn is None:
        supported = ", ".join(sorted(_AUDIO_LORA_MERGE.keys()))
        raise RuntimeError(
            "LoRA adapters require in-engine merging; supported audio families are "
            f"{supported} (this model is family={family!r}). Remove adapters or switch model."
        )
    merge_fn(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        on_log=on_log,
    )


def merge_video_lora_adapters(
    *,
    family: str,
    model: Any,
    adapters: list[Any],
    base_model_id: str,
    project_root: Path,
    registry: Any,
    ctx: Any,
    on_log: Any | None = None,
) -> None:
    merge_fn = get_video_lora_merge(family)
    if merge_fn is None:
        supported = ", ".join(sorted(_VIDEO_LORA_MERGE.keys()))
        raise RuntimeError(
            "LoRA adapters require in-engine merging; supported video families are "
            f"{supported} (this model is family={family!r}). Remove adapters or switch model."
        )
    merge_fn(
        model=model,
        adapters=adapters,
        base_model_id=base_model_id,
        project_root=project_root,
        registry=registry,
        ctx=ctx,
        on_log=on_log,
    )


def attach_image_edit_extra_cond(
    family: str,
    extra_cond: dict[str, Any],
    encoded: Any,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Family hook for img2img edit paths that need extra conditioning tensors."""
    import importlib

    entry = _IMAGE_EDIT_EXTRA_COND.get(family)
    if entry is None:
        return extra_cond
    fn = getattr(importlib.import_module(entry[0]), entry[1])
    return fn(extra_cond, encoded, height=height, width=width)


def encode_prompt_with_image_text_encoder(
    ctx: Any,
    text: str,
    *,
    encoder_type: str,
    bundle_root: Path,
    config: Any,
    registry_entry: Any | None = None,
    registry_version_key: str | None = None,
) -> tuple[Any, Any | None, Any | None]:
    """Instantiate registry text encoder for ``encoder_type`` and encode one prompt.

    Centralizes bundle path resolution and ``config`` → encoder kwargs mapping so
    ``ImagePipeline`` does not branch on ``encoder_type`` for construction.

    Returns:
        ``(embeddings, attention_mask, pooled_embeds)``.
        * mask is set for ``qwen_image`` and ``ernie_image`` (``text_lens``);
        * ``pooled_embeds`` for ``flux1`` (CLIP pooled, not a mask).
    """
    enc_cls = get_text_encoder(encoder_type)
    enc_kwargs: dict[str, Any] = {}
    out_layers = getattr(config, "text_encoder_out_layers", None)
    if out_layers is not None:
        enc_kwargs["hidden_state_layers"] = tuple(out_layers)
    enc_kwargs["enable_thinking"] = getattr(config, "enable_thinking", False)

    enc = _instantiate_image_text_encoder(
        ctx,
        enc_cls,
        encoder_type=encoder_type,
        bundle_root=bundle_root,
        config=config,
        enc_kwargs=enc_kwargs,
        registry_entry=registry_entry,
        registry_version_key=registry_version_key,
    )

    out = enc.encode([text])
    if isinstance(out, tuple):
        if len(out) == 2 and encoder_type == "flux1":
            result = out[0], None, out[1]
        elif len(out) == 2 and encoder_type == "fibo":
            result = out[0], out[1], None
        elif len(out) != 2:
            raise RuntimeError(
                f"Text encoder {encoder_type!r} returned a tuple of len {len(out)}; expected 2."
            )
        else:
            result = out[0], out[1], None
    else:
        result = out, None, None
    _release_image_text_encoder_if_configured(enc, ctx, config)
    return result


def _instantiate_image_text_encoder(
    ctx: Any,
    enc_cls: Any,
    *,
    encoder_type: str,
    bundle_root: Path,
    config: Any,
    enc_kwargs: dict[str, Any] | None = None,
    registry_entry: Any | None = None,
    registry_version_key: str | None = None,
) -> Any:
    """Build a registry text encoder instance (bundle paths + config kwargs)."""
    kw = dict(enc_kwargs or {})
    if registry_entry is not None:
        kw["registry_entry"] = registry_entry
    if registry_version_key is not None:
        kw["registry_version_key"] = registry_version_key
    if encoder_type == "flux1":
        return enc_cls(
            ctx,
            bundle_root,
            max_seq_len=getattr(config, "max_seq_len", 512),
            text_dim=getattr(config, "text_dim", 4096),
            pooled_dim=getattr(config, "pooled_dim", 768),
        )
    enc_dir = bundle_root / "text_encoder"
    tok_dir = bundle_root / "tokenizer"
    if not tok_dir.exists():
        tok_dir = enc_dir
    if not enc_dir.exists():
        enc_dir = bundle_root
        tok_dir = bundle_root
    return enc_cls(ctx, str(enc_dir), tokenizer_path=str(tok_dir), **kw)


def encode_image_text_conditioning(
    ctx: Any,
    *,
    prompt: str,
    negative_prompt: str | None,
    bundle_root: Path | None,
    config: Any,
    guidance: float,
    encode_negative: bool,
    registry_entry: Any | None = None,
    registry_version_key: str | None = None,
) -> tuple[Any, Any, Any, Any, Any, Any, str]:
    """Encode prompt (+ optional negative) for image DiT conditioning.

    Returns ``(txt, neg, txt_mask, neg_mask, pooled, neg_pooled, encoder_type)``.
    Fused CFG encoders (``encode_prompt_cfg`` + ``use_mlx_cfg_fusion``) batch uncond/cond
    into ``txt_embeds`` and skip separate negative encoding.
    """
    from backend.engine.common.bundle.layout import t5_encoder_bundle_paths
    from backend.engine.common.codecs.text_encoders import T5Encoder

    txt_embeds = neg_embeds = txt_attn_mask = neg_attn_mask = None
    pooled_embeds = neg_pooled_embeds = None
    encoder_type = getattr(config, "encoder_type", "t5")

    if not prompt:
        return (
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        )

    if encoder_type == "t5":
        if config.text_dim <= 0:
            return (
                txt_embeds,
                neg_embeds,
                txt_attn_mask,
                neg_attn_mask,
                pooled_embeds,
                neg_pooled_embeds,
                encoder_type,
            )
        if bundle_root is None:
            raise RuntimeError("Cannot load T5 text encoder: model bundle is not installed.")
        t5_dir, t5_tok = t5_encoder_bundle_paths(bundle_root)
        enc = T5Encoder(ctx, t5_dir, tokenizer_path=t5_tok)
        txt_embeds = enc.encode([prompt])
        _release_image_text_encoder_if_configured(enc, ctx, config)
        return (
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        )

    if bundle_root is None:
        raise RuntimeError("Cannot load text encoder: model bundle is not installed at local_path.")

    use_fused_cfg = (
        bool(getattr(config, "use_mlx_cfg_fusion", False))
        and float(guidance) > 1.0
    )
    if use_fused_cfg:
        enc_cls = get_text_encoder(encoder_type)
        enc = _instantiate_image_text_encoder(
            ctx,
            enc_cls,
            encoder_type=encoder_type,
            bundle_root=bundle_root,
            config=config,
            registry_entry=registry_entry,
            registry_version_key=registry_version_key,
        )
        if hasattr(enc, "encode_prompt_cfg"):
            txt_embeds, txt_attn_mask = enc.encode_prompt_cfg(
                prompt,
                negative_prompt,
                guidance=float(guidance),
            )
            _release_image_text_encoder_if_configured(enc, ctx, config)
            return (
                txt_embeds,
                neg_embeds,
                txt_attn_mask,
                neg_attn_mask,
                pooled_embeds,
                neg_pooled_embeds,
                encoder_type,
            )

    txt_embeds, txt_attn_mask, pooled_embeds = encode_prompt_with_image_text_encoder(
        ctx,
        prompt,
        encoder_type=encoder_type,
        bundle_root=bundle_root,
        config=config,
        registry_entry=registry_entry,
        registry_version_key=registry_version_key,
    )
    if encode_negative:
        neg_txt = (negative_prompt or "").strip() or " "
        neg_embeds, neg_attn_mask, neg_pooled_embeds = encode_prompt_with_image_text_encoder(
            ctx,
            neg_txt,
            encoder_type=encoder_type,
            bundle_root=bundle_root,
            config=config,
            registry_entry=registry_entry,
            registry_version_key=registry_version_key,
        )
    return (
        txt_embeds,
        neg_embeds,
        txt_attn_mask,
        neg_attn_mask,
        pooled_embeds,
        neg_pooled_embeds,
        encoder_type,
    )


# request field → (module, augment_fn) — run when field is present on request
_IMAGE_REQUEST_AUGMENTS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("structural_guide", (
        "backend.engine.pipelines.structural_guide_dispatch",
        "augment_request_for_structural_guide",
    )),
)

# request field → (module, attach_fn) — conditioning before denoise
_IMAGE_CONDITIONING: tuple[tuple[str, tuple[str, str]], ...] = (
    ("structural_guide", (
        "backend.engine.pipelines.structural_guide_dispatch",
        "attach_structural_conditioning",
    )),
)


def augment_image_generation_request(request: Any, ctx: Any) -> Any:
    """Apply registry request augment hooks (companion LoRA, etc.)."""
    import importlib

    out = request
    for field, entry in _IMAGE_REQUEST_AUGMENTS:
        if getattr(out, field, None) is None:
            continue
        fn = getattr(importlib.import_module(entry[0]), entry[1])
        out = fn(out, ctx)
    return out


# Shared by create + rewrite img2img (structural companion LoRA, etc.)
augment_image_inference_request = augment_image_generation_request


def attach_image_conditioning(
    pipeline: Any,
    *,
    request: Any,
    family: str,
    model: Any,
    entry: Any,
    version_key: str | None,
    extra_cond: dict[str, Any],
    width: int,
    height: int,
    ctx_exec: Any,
    on_log: Any | None,
) -> tuple[dict[str, Any], Any | None]:
    """Attach optional image conditioning (structural guide, …) via registry."""
    import importlib

    for field, mod_entry in _IMAGE_CONDITIONING:
        if getattr(request, field, None) is None:
            continue
        fn = getattr(importlib.import_module(mod_entry[0]), mod_entry[1])
        return fn(
            pipeline,
            request=request,
            family=family,
            model=model,
            entry=entry,
            version_key=version_key,
            extra_cond=extra_cond,
            width=width,
            height=height,
            ctx_exec=ctx_exec,
            on_log=on_log,
        )
    return extra_cond, None


# =========================================================================
# 视频模型注册表
# =========================================================================

_VIDEO_TRANSFORMER = {
    "wan":       ("backend.engine.families.wan.transformer",       "WanTransformer"),
    "ltx":       ("backend.engine.families.ltx.transformer",       "LTXTransformer"),
    "longcat":   ("backend.engine.families.longcat.transformer",   "LongCatTransformer"),
    "longcat_avatar": ("backend.engine.families.longcat_avatar.transformer", "LongCatAvatarTransformer"),
    "hunyuan":   ("backend.engine.families.hunyuan.transformer",   "HunyuanVideoTransformer"),
}

# family → (module, factory_fn) for VideoPipeline Shape C (in-repo family generator)
_VIDEO_GENERATION_FACTORY = {
    "ltx": ("backend.engine.families.ltx.generation", "create_ltx23_generator"),
    "longcat": ("backend.engine.families.longcat.generation", "create_longcat_generator"),
    "wan": ("backend.engine.families.wan.generation", "create_bernini_renderer_generator"),
}

_IMAGE_GENERATION_FACTORY = {
    "hidream_o1": ("backend.engine.families.hidream_o1.generation", "create_hidream_o1_generator"),
    "step1x_edit": ("backend.engine.families.step1x_edit.generation", "create_step1x_edit_generator"),
    "boogu_image": ("backend.engine.families.boogu.generation", "create_boogu_image_generator"),
}

# family → (module, resolve_output_path_fn) for Shape-C image generators
_IMAGE_OUTPUT_PATH_RESOLVER = {
    "hidream_o1": ("backend.engine.families.hidream_o1.generation", "resolve_hidream_output_path"),
    "boogu_image": ("backend.engine.families.boogu.generation", "resolve_boogu_output_path"),
    "step1x_edit": ("backend.engine.families.step1x_edit.generation", "resolve_step1x_output_path"),
}

_STRUCTURAL_GUIDE_ATTACH = {
    "flux1": ("backend.engine.families.flux1.structural", "attach_structural_conditioning"),
    "z_image": ("backend.engine.families.z_image.structural", "attach_structural_conditioning"),
}

_IMAGE_GENERATION_VALIDATE = {
    "hidream_o1": ("backend.engine.families.hidream_o1.generation", "validate_image_generation_params"),
    "step1x_edit": ("backend.engine.families.step1x_edit.generation", "validate_image_generation_params"),
    "boogu_image": ("backend.engine.families.boogu.generation", "validate_image_generation_params"),
}

_VIDEO_AVATAR_FACTORY = {
    "longcat_avatar": (
        "backend.engine.families.longcat_avatar.generation",
        "create_longcat_avatar_generator",
    ),
}

_VIDEO_GENERATION_VALIDATE = {
    "ltx": ("backend.engine.families.ltx.generation", "validate_video_generation_params"),
    "longcat": ("backend.engine.families.longcat.generation", "validate_video_generation_params"),
    "wan": ("backend.engine.families.wan.generation", "validate_video_generation_params"),
}

_VIDEO_AVATAR_VALIDATE = {
    "longcat_avatar": (
        "backend.engine.families.longcat_avatar.generation",
        "validate_video_avatar_params",
    ),
}

_VIDEO_TRANSFORMER_WEIGHT_PREPARE = {
    "ltx": ("backend.engine.families.ltx.weights_mlx", "prepare_ltx_video_transformer_weights"),
}


def prepare_video_transformer_weights(
    family: str,
    config: Any,
    weights: dict[str, Any],
) -> dict[str, Any]:
    """Family-specific pre-``load_weights`` normalize (registry-driven; no pipeline ``family ==``)."""
    import importlib

    mod_entry = _VIDEO_TRANSFORMER_WEIGHT_PREPARE.get(family)
    if mod_entry is None:
        return weights
    fn = getattr(importlib.import_module(mod_entry[0]), mod_entry[1])
    return fn(config, weights)


def validate_video_generation_params(
    family: str,
    *,
    entry: Any,
    config: Any,
    step_distill: bool,
) -> None:
    """Family-specific video generation param guards (registry-driven)."""
    import importlib

    mod_entry = _VIDEO_GENERATION_VALIDATE.get(family)
    if mod_entry is None:
        return
    fn = getattr(importlib.import_module(mod_entry[0]), mod_entry[1])
    fn(entry=entry, config=config, step_distill=step_distill)


_VIDEO_TEXT_ENCODER = {
    "t5":              ("backend.engine.common.codecs.text_encoders", "T5Encoder"),
    "hunyuan_video_dual": ("backend.engine.families.hunyuan.text_encoder", "HunyuanVideoTextEncoder"),
    "wan_umt5":        ("backend.engine.families.wan.text_encoder", "WanUMT5Encoder"),
}


def get_video_text_encoder_class(encoder_type: str):
    import importlib
    entry = _VIDEO_TEXT_ENCODER.get(encoder_type)
    if entry is None:
        raise RuntimeError(f"Unknown video text encoder type: {encoder_type}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def encode_video_prompt(
    ctx: Any,
    text: str,
    *,
    encoder_type: str,
    bundle_root: Path,
    config: Any,
) -> tuple[Any, Any | None, Any | None, Any | None, Any | None, Any | None]:
    """Encode one video prompt. Returns ``(e1, mask1, e2, mask2, pooled, extra)``.

    T5: ``(embeds, None, None, None, None, None)``.
    Hunyuan dual: ``(qwen_embeds, qwen_mask, byt5_embeds, byt5_mask, None, None)``.
    Wan UMT5: ``(embeds, None, None, None, None, None)`` from .pth bundle.
    """
    enc_cls = get_video_text_encoder_class(encoder_type)
    if encoder_type == "t5":
        from backend.engine.common.bundle.layout import t5_encoder_bundle_paths

        t5_dir, t5_tok_dir = t5_encoder_bundle_paths(bundle_root)
        max_seq = int(getattr(config, "max_text_seq_length", 512))
        enc = enc_cls(ctx, t5_dir, max_seq_len=max_seq, tokenizer_path=t5_tok_dir)
        return enc.encode([text]), None, None, None, None, None

    if encoder_type == "hunyuan_video_dual":
        from backend.engine.families.hunyuan.text_encoder import get_hunyuan_text_encoder

        enc = get_hunyuan_text_encoder(ctx, bundle_root, config)
        e1, m1, e2, m2 = enc.encode([text])
        return e1, m1, e2, m2, None, None

    if encoder_type == "wan_umt5":
        from backend.engine.families.wan.text_encoder_mlx import resolve_wan_umt5_pth

        paths = resolve_wan_umt5_pth(bundle_root)
        if paths is None:
            raise RuntimeError(
                f"Wan UMT5 encoder requires checkpoint (models_t5*.pth, text_encoder/*.safetensors, "
                f"or t5_encoder.safetensors) and UMT5 tokenizer under {bundle_root} "
                f"(or a sibling video bundle with tokenizer/)"
            )
        checkpoint_pth, tokenizer_dir = paths
        text_len = int(getattr(config, "text_len", 512))
        enc = enc_cls(ctx, checkpoint_pth, tokenizer_dir, text_len=text_len)
        embeds = enc.encode([text])
        if bool(getattr(config, "release_t5_after_encode", False)):
            release = getattr(enc, "release_weights", None)
            if callable(release):
                release()
        return embeds, None, None, None, None, None

    raise RuntimeError(f"Unsupported video encoder_type: {encoder_type!r}")


def encode_video_hunyuan_dual_cfg_batch(
    ctx: Any,
    *,
    prompt: str,
    negative_prompt: str | None,
    bundle_root: Path,
    config: Any,
    guidance: float,
    encoder_type: str,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any] | None:
    """Batch positive + negative Hunyuan dual-encoder forward when CFG applies."""
    if encoder_type != "hunyuan_video_dual":
        return None
    use_cfg = bool(getattr(config, "supports_guidance", True) and guidance > 1.0)
    if not use_cfg or not prompt or config.text_dim <= 0:
        return None
    from backend.engine.families.hunyuan.text_encoder import get_hunyuan_text_encoder

    neg_txt = negative_prompt.strip() if negative_prompt else " "
    enc = get_hunyuan_text_encoder(ctx, bundle_root, config)
    e1, m1, e2, m2 = enc.encode([prompt, neg_txt])
    return (
        e1[0:1], m1[0:1], e2[0:1], m2[0:1],
        e1[1:2], m1[1:2], e2[1:2], m2[1:2],
    )


def get_video_transformer_class(family: str):
    import importlib
    entry = _VIDEO_TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown video model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_video_generation_factory(family: str):
    import importlib
    entry = _VIDEO_GENERATION_FACTORY.get(family)
    if entry is None:
        supported = ", ".join(sorted(_VIDEO_GENERATION_FACTORY.keys()))
        raise RuntimeError(
            f"VideoPipeline: no generation factory for family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_image_generation_factory(family: str):
    import importlib

    entry = _IMAGE_GENERATION_FACTORY.get(family)
    if entry is None:
        supported = ", ".join(sorted(_IMAGE_GENERATION_FACTORY.keys()))
        raise RuntimeError(
            f"ImagePipeline: no generation factory for family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_image_output_path_resolver(family: str):
    import importlib

    entry = _IMAGE_OUTPUT_PATH_RESOLVER.get(family)
    if entry is None:
        supported = ", ".join(sorted(_IMAGE_OUTPUT_PATH_RESOLVER.keys()))
        raise RuntimeError(
            f"ImagePipeline: no output path resolver for family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_structural_guide_attach(family: str):
    import importlib

    entry = _STRUCTURAL_GUIDE_ATTACH.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


def validate_image_generation_params(
    *,
    family: str,
    entry: Any,
    config: Any,
    ref_image_paths: list[str] | None = None,
    **kwargs: Any,
) -> None:
    import importlib

    mod_entry = _IMAGE_GENERATION_VALIDATE.get(family)
    if mod_entry is None:
        return
    fn = getattr(importlib.import_module(mod_entry[0]), mod_entry[1])
    fn(entry=entry, config=config, ref_image_paths=ref_image_paths, **kwargs)


def get_video_avatar_factory(family: str):
    import importlib
    entry = _VIDEO_AVATAR_FACTORY.get(family)
    if entry is None:
        supported = ", ".join(sorted(_VIDEO_AVATAR_FACTORY.keys()))
        raise RuntimeError(
            f"VideoPipeline: no avatar factory for family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def validate_video_avatar_params(
    family: str,
    *,
    entry: Any,
    config: Any,
) -> None:
    import importlib
    mod_entry = _VIDEO_AVATAR_VALIDATE.get(family)
    if mod_entry is None:
        return
    fn = getattr(importlib.import_module(mod_entry[0]), mod_entry[1])
    fn(entry=entry, config=config)


# =========================================================================
# 音频模型注册表
# =========================================================================

_AUDIO_TRANSFORMER = {
    "diffrhythm": ("backend.engine.families.diffrhythm.transformer", "DiffRhythmTransformer"),
    "ace_step": ("backend.engine.families.ace_step.transformer", "AceStepTransformer"),
}

# family → (module, run_method_name) for MusicPipeline Shape C
_AUDIO_GENERATION_FACTORY = {
    "diffrhythm": ("backend.engine.families.diffrhythm.generation", "create_diffrhythm_generator"),
    "ace_step": ("backend.engine.families.ace_step.generation", "create_ace_step_generator"),
}

# family → (module, prepare_fn) — request normalization before generation
_AUDIO_PREPARE_REQUEST = {
    "ace_step": ("backend.engine.families.ace_step.generation", "prepare_music_request"),
    "diffrhythm": ("backend.engine.families.diffrhythm.generation", "prepare_music_request"),
}

# (family, operation) → (module, handler_fn) for MusicPipeline.run_edit
_AUDIO_EDIT_HANDLERS = {
    ("ace_step", "cover"): ("backend.engine.families.ace_step.generation", "run_cover_edit"),
}


def get_audio_transformer_class(family: str):
    import importlib
    entry = _AUDIO_TRANSFORMER.get(family)
    if entry is None:
        raise RuntimeError(f"Unknown audio model family: {family}")
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_audio_generation_factory(family: str):
    import importlib
    entry = _AUDIO_GENERATION_FACTORY.get(family)
    if entry is None:
        supported = ", ".join(sorted(_AUDIO_GENERATION_FACTORY.keys()))
        raise RuntimeError(
            f"MusicPipeline: unsupported audio family {family!r}; supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_audio_prepare_request(family: str):
    """Resolve family-specific ``prepare_music_request`` (lazy import)."""
    import importlib
    entry = _AUDIO_PREPARE_REQUEST.get(family)
    if entry is None:
        supported = ", ".join(sorted(_AUDIO_PREPARE_REQUEST.keys()))
        raise RuntimeError(
            f"MusicPipeline: no prepare_music_request for family {family!r}; "
            f"supported: {supported}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


def get_audio_edit_handler(family: str, operation: str):
    """Resolve family-specific audio edit handler (lazy import)."""
    import importlib

    key = (family, operation)
    entry = _AUDIO_EDIT_HANDLERS.get(key)
    if entry is None:
        supported = ", ".join(f"{f}/{op}" for f, op in sorted(_AUDIO_EDIT_HANDLERS))
        raise RuntimeError(
            f"MusicPipeline: no audio edit handler for {family!r}/{operation!r}; "
            f"supported: {supported or '(none)'}"
        )
    return getattr(importlib.import_module(entry[0]), entry[1])


# family → post-save hook (sidecars, extra metadata) after each WAV
_AUDIO_POST_GENERATION = {
    "ace_step": ("backend.engine.families.ace_step.generation", "post_generation_artifacts"),
}


# family → (module, fn) for lyrics metadata on assets / task result
_AUDIO_LYRICS_METADATA = {
    "ace_step": ("backend.engine.families.ace_step.generation", "lyrics_capture_metadata"),
}


def audio_lyrics_metadata(family: str, capture: Any, *, duration_sec: float | None = None) -> dict[str, Any]:
    """Optional family hook — serialize lyrics capture for asset/task metadata."""
    import importlib

    entry = _AUDIO_LYRICS_METADATA.get(family)
    if entry is None or capture is None:
        return {}
    fn = getattr(importlib.import_module(entry[0]), entry[1])
    return fn(capture, duration_sec=duration_sec)


def get_audio_post_generation(family: str):
    """Optional family hook after each generated audio file is written."""
    import importlib

    entry = _AUDIO_POST_GENERATION.get(family)
    if entry is None:
        return None
    return getattr(importlib.import_module(entry[0]), entry[1])


# Shared MLX encoder stacks reused across families (registry dispatch, no cross-family imports).
_MLX_ENCODER_LOADERS: dict[str, tuple[str, str]] = {
    "qwen25vl": (
        "backend.engine.families.qwen.text_encoder_mlx",
        "load_qwen25vl_mlx_encoder",
    ),
}


def load_mlx_encoder_stack(kind: str, /, *args: Any, **kwargs: Any) -> Any:
    """Load a reusable MLX encoder stack by registry key (e.g. ``qwen25vl`` for Hunyuan)."""
    import importlib

    entry = _MLX_ENCODER_LOADERS.get(kind)
    if entry is None:
        known = ", ".join(sorted(_MLX_ENCODER_LOADERS))
        raise RuntimeError(f"Unknown MLX encoder stack {kind!r}; known: {known}")
    fn = getattr(importlib.import_module(entry[0]), entry[1])
    return fn(*args, **kwargs)


def check_audio_bundle_ready(family: str, bundle_path: Any) -> bool | None:
    """Return ``True``/``False`` when a family hook exists; ``None`` if no extra check."""
    return None
