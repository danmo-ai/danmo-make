"""
Model configuration dataclasses — architecture hyperparameters per model family.

Reference implementations of ModelConfig and config system.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# =========================================================================
# Image models
# =========================================================================


@dataclass
class Flux1Config:
    """Flux.1 series (schnell / dev / fill / depth / kontext / redux / controlnet)
    
    Architecture: MM-DiT (dual-stream / joint attention) + T5 + CLIP
    """
    in_channels: int = 16            # VAE latent channels（DiT 内 2×2 pack 后为 64 维 token）
    out_channels: int = 64           # ``proj_out`` 每 token 维度，unpack 回 16ch 空间格
    patch_token_dim: int = 64        # packed token width into ``x_embedder`` (Fill=384, structural=128)
    hidden_dim: int = 3072           # hidden dimension
    num_heads: int = 24
    num_joint_layers: int = 19       # diffusers ``num_layers`` (FluxTransformerBlock)
    num_single_layers: int = 38      # diffusers ``num_single_layers`` (FluxSingleTransformerBlock)
    text_dim: int = 4096             # T5 output dim
    clip_dim: int = 0                # diffusers DiT 无 CLIP token 支路（pooled 见 pooled_dim）
    pooled_dim: int = 768            # CLIP pooled → time_text_embed.text_embedder
    encoder_type: str = "flux1"      # T5 + CLIP pooled（见 families/flux1/text_encoder.py）
    latent_noise_packed: bool = True  # 初始噪声在 packed [B, (H//16)*(W//16), 64] 上采样（packed latent 布局）
    max_seq_len: int = 512           # max text token count
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    # Variant flags
    supports_guidance: bool = False   # schnell=False; dev/krea 由 registry 置 True
    supports_img2img: bool = True
    supports_mask: bool = False       # Fill / Depth need this
    supports_controlnet: bool = False
    supports_structural_guide: bool = True
    vae_scale: int = 8               # VAE latent downsampling factor
    passes_guidance_in_kwargs: bool = True
    preserve_guidance_when_disabled: bool = True
    cfg_negative_eligible: bool = False
    vae_encoder_cast_bfloat16: bool = True  # MLX VAEEncoder weights → bfloat16 after load
    release_text_encoder_after_encode: bool = True
    use_batched_cfg: bool = True
    use_mlx_compile: bool = False
    use_mlx_compile_auto: bool = True
    teacache_mode: str = "auto"
    use_mm_dit_text_cache: bool = True

    def __post_init__(self):
        pass


@dataclass
class Flux2Config:
    """Flux.2 Klein series (4B / 9B / base)
    
    Architecture: MM-DiT dual-stream + AdaLayerNormContinuous + Qwen3 text encoder
    """
    in_channels: int = 128           # Flux2 uses 128 (not 64 like Flux1)
    out_channels: int = 128
    inner_dim: int = 4096            # 9B: 4096; 4B: 3072
    num_heads: int = 32              # 9B: 32; 4B: 24
    attn_head_dim: int = 128
    num_layers: int = 8              # joint blocks (9B: 8)
    num_single_layers: int = 24      # single blocks (9B: 24)
    joint_attention_dim: int = 12288 # Qwen3 output dim
    max_seq_len: int = 512
    mlp_ratio: float = 3.0
    qk_norm: bool = True
    supports_guidance: bool = True
    requires_sigma_shift: bool = True
    supports_img2img: bool = True
    supports_edit: bool = True
    encoder_type: str = "flux2"
    text_encoder_out_layers: tuple = (9, 18, 27)  # Flux2 Qwen3 takes 3 layers concatenated
    enable_thinking: bool = False     # Reference: Flux2KleinWeightDefinition explicitly disables
    vae_scale: int = 16              # Flux2 uses 16x tile, not 8x
    latent_noise_dtype: str = "bfloat16"
    noise_sample_fp32: bool = True
    vae_preview_warmup: bool = True
    release_text_encoder_after_encode: bool = True


@dataclass
class QwenImageConfig:
    """Qwen-Image series (txt2img / edit)
    
    Architecture: DiT + Qwen VL vision encoder.
    Transformer ``proj_out`` uses ``patch_size**2 * out_channels``; packed latent token dim is 64 (= 16 VAE latent ch × 4 patch),
    so ``out_channels`` must be **16** (matches diffusers ``QwenImageTransformer2DModel`` default), not 64.
    """
    in_channels: int = 64
    out_channels: int = 16
    hidden_dim: int = 3072
    num_heads: int = 24
    num_layers: int = 60
    text_dim: int = 3584             # Qwen text dim
    vl_dim: int = 3584               # Qwen VL encoder dim
    max_seq_len: int = 2048
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    vae_scale: int = 16
    encoder_type: str = "qwen_image"
    encoder_step_kwargs: str = "qwen_image"
    # Qwen-Image-Edit（独立权重）：VL 图文编码 + VAE 参考 latent 序列拼接
    edit_use_vl_vision: bool = False
    edit_conditioning_latent_concat: bool = False
    # FireRed / QwenImageEditPlus：Picture N 前缀 + 多参考图
    edit_use_picture_prefix: bool = False
    edit_plus_multi_image: bool = False
    edit_max_reference_images: int = 1
    release_text_encoder_after_encode: bool = True
    use_batched_cfg: bool = True
    use_mlx_compile: bool = False
    use_mlx_compile_auto: bool = True
    teacache_mode: str = "auto"


@dataclass
class FIBOConfig:
    """FIBO series (FIBO / FIBO-Lite / FIBO-Edit)

    Architecture: Bria4Transformer2DModel — Joint MM-DiT + Single DiT.
    """
    in_channels: int = 48
    out_channels: int = 48
    hidden_dim: int = 3072
    num_heads: int = 24
    head_dim: int = 128
    num_joint_layers: int = 8
    num_single_layers: int = 38
    text_dim: int = 4096
    text_encoder_dim: int = 2048
    max_seq_len: int = 2048
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    supports_guidance: bool = True
    supports_img2img: bool = True
    structured_prompt: bool = True
    encoder_type: str = "fibo"
    skip_negative_when_structured_prompt: bool = True
    use_mlx_cfg_fusion: bool = True
    vae_scale: int = 16
    text_encoder_mask_key: str = "text_encoder_layers"
    # FIBO-Edit: Reference concat VAE-packed source latents on the sequence axis (not img2img blend).
    edit_conditioning_concat: bool = False
    edit_rmbg_composite_output: bool = False
    release_text_encoder_after_encode: bool = True


@dataclass
class ZImageConfig:
    """Z-Image series (Z-Image / Z-Image-Turbo)
    
    Architecture: ZImageTransformer + Qwen3 text encoder (cap_feat_dim=2560).
    Reference: models/z_image/model/z_image_transformer/transformer.py
    """
    in_channels: int = 16             # VAE latent channels (16, not 64 like Flux)
    out_channels: int = 16
    dim: int = 3840                   # hidden dimension (much larger than Flux1's 3072)
    hidden_dim: int = 3840            # alias for compat
    num_heads: int = 30               # n_heads=30
    num_layers: int = 30              # n_layers=30
    num_refiner_layers: int = 2       # noise_refiner + context_refiner layer count
    text_dim: int = 2560              # Qwen3 cap_feat_dim (not T5)
    cap_feat_dim: int = 2560          # caption feature dimension
    clip_dim: int = 0                 # No CLIP encoder
    max_seq_len: int = 512
    rope_dim: int = 32                # one of axis_dims (32+48+48 3D RoPE)
    rope_theta: float = 256.0
    patch_size: int = 2               # Z-Image uses 2x2 patch (Flux1 uses 1x1)
    t_scale: float = 1000.0           # timestep scaling
    norm_eps: float = 1e-5
    qk_norm: bool = True
    supports_guidance: bool = True    # Z-Image=True, Z-Image-Turbo=False
    requires_sigma_shift: bool = True
    supports_img2img: bool = False
    supports_structural_guide: bool = True
    supports_latent_refine: bool = True
    lemica_mode: str = "none"
    encoder_type: str = "z_image"       # ZImageTextEncoder
    text_encoder_out_layers: Optional[tuple] = None  # flux2=(9,18,27), z_image=None
    enable_thinking: bool = True       # z_image uses True, flux2 uses False
    vae_scale: int = 8
    # Match ``ZImageLatentCreator.create_noise`` (``ModelConfig.precision`` = bf16).
    latent_noise_dtype: str = "bfloat16"
    noise_sample_fp32: bool = True
    z_image_noise_layout: bool = True
    use_mlx_compile: bool = False        # keep numerical path close to reference; prioritize parity
    use_mlx_compile_auto: bool = True
    use_mlx_cfg_fusion: bool = False     # disable fused CFG fast-path; use explicit cond/uncond forwards
    teacache_mode: str = "auto"
    release_text_encoder_after_encode: bool = True


@dataclass
class ErnieImageConfig:
    """ERNIE-Image series (Turbo / SFT)

    Architecture: 8B single-stream DiT + Ministral-3 text encoder + FLUX.2 VAE.
    """
    in_channels: int = 128
    out_channels: int = 128
    hidden_size: int = 4096
    num_heads: int = 32
    num_layers: int = 36
    ffn_hidden_size: int = 12288
    patch_size: int = 1
    text_in_dim: int = 3072
    text_dim: int = 3072
    rope_axes_dim: tuple = (32, 48, 48)
    rope_theta: float = 256.0
    eps: float = 1e-6
    qk_norm: bool = True
    encoder_type: str = "ernie_image"
    max_seq_len: int = 2048
    text_encoder_mask_key: str = "text_lens"
    supports_guidance: bool = False
    requires_sigma_shift: bool = False
    vae_scale: int = 16
    latent_noise_dtype: str = "bfloat16"
    noise_sample_fp32: bool = True
    vae_preview_warmup: bool = True
    supports_img2img: bool = False
    bundle_config_merger: str = "ernie_image"
    release_text_encoder_after_encode: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


def merge_ernie_image_config_from_bundle(config: ErnieImageConfig, bundle_root: Path | None) -> None:
    """Override ``ErnieImageConfig`` from ``transformer/config.json`` (diffusers layout)."""
    if bundle_root is None:
        return
    cfg_path = bundle_root / "transformer" / "config.json"
    if not cfg_path.is_file():
        return
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"ERNIE-Image: cannot read config {cfg_path}: {e}") from e

    key_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_layers": "num_layers",
        "ffn_hidden_size": "ffn_hidden_size",
        "in_channels": "in_channels",
        "out_channels": "out_channels",
        "patch_size": "patch_size",
        "text_in_dim": "text_in_dim",
        "rope_theta": "rope_theta",
        "eps": "eps",
    }
    for src, dst in key_map.items():
        if src in data:
            val = data[src]
            if isinstance(val, bool):
                setattr(config, dst, bool(val))
            elif isinstance(val, float):
                setattr(config, dst, float(val))
            else:
                setattr(config, dst, int(val))
    if "qk_layernorm" in data:
        config.qk_norm = bool(data["qk_layernorm"])
    if "rope_axes_dim" in data:
        object.__setattr__(
            config,
            "rope_axes_dim",
            tuple(int(x) for x in data["rope_axes_dim"]),
        )


@dataclass
class CogView4Config:
    """CogView4-6B — joint-attention DiT + GLM-4-9B + AutoencoderKL."""

    patch_size: int = 2
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 30
    attention_head_dim: int = 40
    num_attention_heads: int = 64
    text_embed_dim: int = 4096
    time_embed_dim: int = 512
    condition_dim: int = 256
    pos_embed_max_size: int = 128
    sample_size: int = 128
    rope_axes_dim: tuple[int, int] = (256, 256)
    encoder_type: str = "cogview4"
    max_seq_len: int = 1024
    supports_guidance: bool = True
    requires_sigma_shift: bool = True
    vae_scale: int = 8
    # diffusers ``pipeline_cogview4``: seq len = latent pixels // patch_size**2
    scheduler_image_seq_len_divisor: int = 4
    latent_noise_dtype: str = "bfloat16"
    noise_sample_fp32: bool = True
    supports_img2img: bool = True
    bundle_config_merger: str = "cogview4"
    release_text_encoder_after_encode: bool = True

    @property
    def text_dim(self) -> int:
        return self.text_embed_dim


def merge_cogview4_config_from_bundle(config: CogView4Config, bundle_root: Path | None) -> None:
    """Override ``CogView4Config`` from ``transformer/config.json`` (diffusers layout)."""
    if bundle_root is None:
        return
    cfg_path = bundle_root / "transformer" / "config.json"
    if not cfg_path.is_file():
        return
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"CogView4: cannot read config {cfg_path}: {e}") from e
    key_map = {
        "patch_size": "patch_size",
        "in_channels": "in_channels",
        "out_channels": "out_channels",
        "num_layers": "num_layers",
        "attention_head_dim": "attention_head_dim",
        "num_attention_heads": "num_attention_heads",
        "text_embed_dim": "text_embed_dim",
        "time_embed_dim": "time_embed_dim",
        "condition_dim": "condition_dim",
        "pos_embed_max_size": "pos_embed_max_size",
        "sample_size": "sample_size",
    }
    for src, dst in key_map.items():
        if src in data:
            setattr(config, dst, int(data[src]))
    if "rope_axes_dim" in data:
        object.__setattr__(
            config,
            "rope_axes_dim",
            tuple(int(x) for x in data["rope_axes_dim"]),
        )
    patch_size = int(getattr(config, "patch_size", 2) or 2)
    object.__setattr__(config, "scheduler_image_seq_len_divisor", patch_size * patch_size)


@dataclass
class SeedVR2Config:
    """SeedVR2 super-resolution model (3B / 7B)
    
    Architecture: unconditional DiT + low-res image condition injection
    """
    in_channels: int = 64           # low-res latent
    out_channels: int = 64          # high-res latent
    hidden_dim: int = 1024          # 3B; 7B = 1536
    num_heads: int = 16             # 3B; 7B = 24
    num_layers: int = 24            # 3B; 7B = 32
    text_dim: int = 0               # unconditional, no text encoder
    rope_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    supports_guidance: bool = False  # unconditional model
    supports_img2img: bool = False
    # Super-resolution specific
    vae_scale: int = 8
    scale_factor: int = 2            # upscale factor
    tile_size: int = 1024            # tiled super-resolution
    denoise_strength: float = 0.3    # default denoising strength
    vae_stream_cache: bool = True
    conv3d_backend: str = "auto"


@dataclass
class EsrganConfig:
    """Real-ESRGAN RRDBNet pixel upscaler (MLX)."""
    netscale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    supports_guidance: bool = False
    supports_img2img: bool = False
    vae_scale: int = 1
    default_tile: int = 256


@dataclass
class HiDreamO1Config:
    """HiDream-O1-Image — Qwen3-VL pixel-patch UiT (MLX, no VAE).

    Shape C family generator: mlx-vlm backbone + custom diffusion heads.
    Dev variant uses 28-step distilled schedule; Full uses 50 steps.
    """
    image_pipeline_shape: str = "family_generator"
    encoder_type: str = "qwen3_vl"
    supports_guidance: bool = False
    supports_img2img: bool = True
    vae_scale: int = 32
    patch_size: int = 32
    default_scheduler: str = "flash_flow_match"
    hidream_noise_scale: float = 7.5
    hidream_noise_clip_std: float = 2.5
    hidream_custom_timesteps: tuple[int, ...] | None = None
    hidream_quantized_no_edit: bool = False
    hidream_edit_max_refs: int = 3
    edit_use_vl_vision: bool = True
    hidream_snap_resolution: bool = True


@dataclass
class Step1XEditConfig:
    """Step1X-Edit v1.1 — Qwen2.5-VL + DiT image edit (MLX / CUDA native inference)."""
    image_pipeline_shape: str = "family_generator"
    encoder_type: str = "qwen2_vl"
    supports_guidance: bool = True
    supports_img2img: bool = True
    vae_scale: int = 8
    step1x_version: str = "v1.1"
    step1x_variant: str = ""
    step1x_dit_filename: str = ""
    step1x_max_length: int = 640
    step1x_size_level: int = 512
    edit_max_reference_images: int = 1


@dataclass
class BooguImageConfig:
    """Boogu-Image-0.1 — Qwen3-VL + Lumina2-lineage DiT + FLUX VAE (MLX family_generator)."""

    image_pipeline_shape: str = "family_generator"
    encoder_type: str = "qwen3_vl"
    supports_guidance: bool = True
    supports_img2img: bool = True
    vae_scale: int = 8
    boogu_variant: str = "turbo"
    edit_max_reference_images: int = 1
    release_text_encoder_after_encode: bool = True



@dataclass
class DiffRhythmConfig:
    """DiffRhythm 2 — block flow matching + Music VAE + BigVGAN decoder.

    Defaults match ``ASLP-lab/DiffRhythm2`` ``config.json``; bundle config is
    authoritative at load time.
    """
    # DiT (Llama-NAR backbone inside CFM)
    dim: int = 2048
    depth: int = 16
    heads: int = 16
    ff_mult: int = 4
    text_dim: int = 512
    mel_dim: int = 64
    text_num_embeds: int = 1000
    block_size: int = 10
    use_flex_attn: bool = False
    repa_depth: int = 6
    repa_dims: tuple[int, ...] = (1024, 768)

    # Audio / latent (5 Hz Music VAE latents → 48 kHz BigVGAN decode)
    latent_frame_rate: float = 5.0
    style_encode_sample_rate: int = 24_000
    sample_rate: int = 48_000
    fake_stereo: bool = True

    # Pipeline
    supports_guidance: bool = True
    default_infer_steps: int = 16
    default_guidance: float = 2.0  # cfg_strength in upstream inference
    max_duration_seconds: int = 210
    mulan_repo_id: str = "OpenMuQ/MuQ-MuLan-large"


@dataclass
class AceStepConfig:
    """ACE-Step series — music generation via DiT + VAE.

    Architecture: Qwen3-based decoder-only DiT (sliding-window self-attn +
    cross-attn) + AdaLN modulation + Oobleck VAE.

    Defaults target XL SFT (4B); bundle config.json is authoritative at load time.
    """
    # DiT decoder (XL SFT defaults; 2B base uses 2048 / 24 / 16)
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    in_channels: int = 192                  # DiT input channels
    audio_acoustic_hidden_dim: int = 64     # output channels (= VAE latent dim)
    patch_size: int = 2
    sliding_window: int = 128
    layer_types: tuple | None = None        # e.g. ("sliding_attention", "full_attention", ...)
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    is_turbo: bool = False                  # turbo: 8-step, no CFG
    turbo_infer_steps: int = 8
    turbo_shift: float = 1.0

    # VAE
    vae_encoder_hidden_size: int = 128
    vae_downsampling_ratios: tuple = (2, 4, 4, 8, 8)
    vae_channel_multiples: tuple = (1, 2, 4, 8, 16)
    vae_decoder_channels: int = 128
    vae_decoder_input_channels: int = 64     # latent dim into VAE decoder
    audio_channels: int = 2                  # stereo output

    # Text encoder
    encoder_type: str = "qwen3_embedding"    # Qwen3-Embedding-0.6B

    # Pipeline
    supports_guidance: bool = True           # CFG / APG
    supports_img2img: bool = False
    default_infer_steps: int = 50            # base/sft; turbo = 8
    default_shift: float = 3.0

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = tuple(
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(self.num_hidden_layers)
            )


# =========================================================================
# Video models
# =========================================================================


@dataclass
class LTXConfig:
    """LTX 2.3 video (Lightricks LTX-2.3, dgrauet MLX bundles).

    Shape C family generator: 48-layer A/V DiT + Gemma 3 + two-stage upscale.
    Registry ``step_distill`` selects distilled vs dev pipelines.
    """
    video_pipeline_shape: str = "family_generator"
    encoder_type: str = "gemma"
    dim: int = 2048
    depth: int = 48
    num_heads: int = 32
    head_dim: int = 64
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    dim_in: int = 128
    dim_out: int = 128
    text_dim: int = 0
    max_seq_len: int = 512
    max_text_seq_length: int = 128
    t5_attention_mask: bool = False
    time_dim: int = 256
    patch_size: int = 1
    temporal_patch_size: int = 1
    supports_guidance: bool = True
    step_distill: bool = False
    distilled_model_id_marker: str = "distilled"
    require_registry_step_distill_when_distilled: bool = True
    supports_img2img: bool = True
    vae_scale: int = 32
    temporal_vae_scale: int = 8
    default_scheduler: str = "flow_match_euler"
    uses_mlx_forge_weight_restore: bool = False
    validate_ltx_block_depth: bool = False
    geometry_check: str = "generic"
    post_denoise_clear_cache: bool = False
    uses_ltx_flat_vae_decoder: bool = False
    video_vae_backend: str = ""
    video_i2v_style: str = "ltx23"
    bundle_config_merger: str = ""
    release_t5_after_encode: bool = False
    cfg_negative_prompt_style: str = "default"
    scheduler_bundle_extras: str = ""
    gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit"
    text_encoder_gemma_local: str = ""
    inject_text_encoder_paths: bool = True
    ltx_low_memory: bool = True
    low_ram_streaming: bool = False
    ltx_stage2_steps: int = 3
    # I2V frame-0 anchor strength (1.0 = fully frozen; reference default). Scene
    # coherence is enforced by pin_latent_by_mask in the denoise loop, not by <1.0.
    ltx_i2v_anchor_strength: float = 1.0
    supports_long_video: bool = False
    ltx_long_video_max_frames: int = 257


@dataclass
class LTX25Config:
    """LTX-2.5 (22B) video — Lightricks LTX-2.5 distilled, MLX only.

    Shape C family generator: 22B A/V DiT + Gemma 4 text encoder + V2 feature
    extractor + 1D connectors. Only the distilled 8+3-step two-stage pipeline
    is implemented (``step_distill`` must be true).
    """

    video_pipeline_shape: str = "family_generator"
    encoder_type: str = "ltx25_gemma4"
    dim: int = 0
    depth: int = 0
    num_heads: int = 0
    head_dim: int = 0
    dim_in: int = 128
    dim_out: int = 128
    text_dim: int = 0
    max_seq_len: int = 0
    time_dim: int = 256
    patch_size: int = 1
    temporal_patch_size: int = 1
    supports_guidance: bool = False
    step_distill: bool = True
    supports_img2img: bool = True
    vae_scale: int = 32
    temporal_vae_scale: int = 8
    default_scheduler: str = "flow_match_euler"
    post_denoise_clear_cache: bool = True
    video_vae_backend: str = ""
    video_i2v_style: str = "ltx25"
    release_t5_after_encode: bool = False
    ltx_low_memory: bool = True
    supports_long_video: bool = False


@dataclass
class LongCatConfig:
    """LongCat-Video (Meituan 13.6B, mlx-community MLX port).

    Shape C family generator: Wan 2.1 VAE + umT5-XXL + LongCat DiT.
    ``step_distill`` / registry ``cfg_step_lora`` enables 8-step cfg_step_lora path.
    """
    video_pipeline_shape: str = "family_generator"
    encoder_type: str = "wan_umt5"
    supports_guidance: bool = True
    supports_img2img: bool = True
    vae_scale: int = 8
    temporal_vae_scale: int = 4
    video_vae_backend: str = "wan"
    default_scheduler: str = "flow_match_euler"
    scheduler_shift: float = 12.0
    default_infer_steps: int = 8
    default_fps: int = 15
    cfg_step_lora_default: bool = True
    cfg_step_lora_steps: int = 8
    supports_long_video: bool = False
    geometry_check: str = "generic"


@dataclass
class MinimaxH3Config:
    """MiniMax-H3 Base FL2VA (omni A/V DiT + Qwen3-VL conditioner).

    Shape C family generator: joint video+audio denoising on a packed sequence.
    Official checkpoints are CFG-distilled (no negative / guidance pass).
    Spatial VAE compresses 16x; DiT patch (1,2,2) yields effective token scale 32.
    """
    video_pipeline_shape: str = "family_generator"
    encoder_type: str = "minimax_h3_qwen3vl"
    supports_guidance: bool = False
    supports_img2img: bool = True
    vae_scale: int = 32
    temporal_vae_scale: int = 4
    video_vae_backend: str = "minimax_h3"
    default_scheduler: str = "flow_match_euler"
    scheduler_shift: float = 12.0
    audio_scheduler_shift: float = 3.0
    default_infer_steps: int = 30
    default_fps: int = 24
    latent_channels: int = 24
    audio_latent_channels: int = 32
    hidden_size: int = 5376
    num_layers: int = 50
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_dim: int = 14336
    text_dim: int = 5120
    text_encoder_layer: int = 50
    patch_size: tuple = (1, 2, 2)
    supports_long_video: bool = False
    geometry_check: str = "generic"
    license_territorial_notice: bool = True
    h3_low_memory: bool = True
    h3_stream_decode: bool = True
    h3_denoiser_reuse: int = 1
    h3_active_layers: int = 50
    h3_internal_canvas: str = "off"
    h3_turbo: bool = False
    h3_block_streaming: bool = False
    h3_token_reduction: bool = False
    teacache_mode: str = "none"


@dataclass
class LongCatAvatarConfig:
    """LongCat-Video-Avatar 1.5 (audio-driven digital human, ATI2V / AT2V).

    Shape C family_avatar generator: Wan VAE + umT5 + Whisper + Avatar DiT + DMD 8-step.
    """
    video_pipeline_shape: str = "family_avatar"
    encoder_type: str = "wan_umt5"
    supports_guidance: bool = False
    supports_img2img: bool = False
    vae_scale: int = 8
    temporal_vae_scale: int = 4
    video_vae_backend: str = "wan"
    default_scheduler: str = "flow_match_euler"
    scheduler_shift: float = 7.0
    num_train_timesteps: int = 1000
    default_infer_steps: int = 8
    default_fps: int = 25
    default_num_frames: int = 93
    default_text_guidance: float = 4.0
    default_audio_guidance: float = 4.0
    supports_long_video: bool = False
    geometry_check: str = "generic"


@dataclass
class WanConfig:
    """Wan Video series (Wan2.1 / Wan2.2).

    Defaults target **Wan2.2-TI2V-5B**; larger variants override via ``merge_wan_config_from_bundle``.
    """
    dim: int = 3072
    depth: int = 30
    num_heads: int = 24
    ffn_dim: int = 14336
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    freq_dim: int = 256
    # Input dimensions
    dim_in: int = 48
    dim_out: int = 48
    text_dim: int = 4096
    max_seq_len: int = 512
    text_len: int = 512
    # Text encoder — UMT5-XXL from .pth bundle (not HuggingFace T5)
    encoder_type: str = "wan_umt5"
    # Spatiotemporal parameters
    patch_size: tuple = (1, 2, 2)
    window_size: tuple = (-1, -1)
    rope_dim: int = 64
    temporal_rope_dim: int = 32
    temporal_attn_every: int = 2
    # Dual model (14B high/low noise); TI2V 5B is single-model
    dual_model: bool = False
    moe_boundary_step_index: int = 2  # LightX2V Wan2.2 distill: steps [0, boundary) → high noise
    step_distill: bool = False
    wan_distill_timesteps: tuple[float, ...] = ()
    expand_timesteps: bool = True  # I2V only: per-token timesteps via wan_expand_timesteps in before_denoise
    # Pipeline / VAE
    vae_scale: int = 16
    temporal_vae_scale: int = 4
    vae_z_dim: int = 48
    supports_guidance: bool = True
    supports_img2img: bool = True
    supports_lora: bool = True
    default_scheduler: str = "wan_flow_unipc"
    num_train_timesteps: int = 1000
    use_mlx_compile: bool = False  # compile partial forward + text KV cache caused denoise drift
    use_mlx_compile_step_distill: bool = True  # safe for 4-step Lightning / distill schedules
    wan_distill_shift: float | None = None  # None → LightX2V default (5.0) in distill scheduler
    vae_spatial_tiling: bool = False  # 默认整幅 decode；分块拼接在 TI2V 5B 分辨率下会出 seam
    uses_wan_t5_bundle: bool = True
    uses_wan_shift: bool = True
    snap_pixel_dims: bool = True
    validate_umt5_embeddings: bool = True
    post_denoise_clear_cache: bool = True
    wan_moe_lazy_experts: bool = True
    attention_backend: str = "auto"
    vae_stream_cache: bool = False
    conv3d_backend: str = "auto"
    use_text_embedding_cache: bool = True
    teacache_mode: str = "auto"
    geometry_check: str = "wan"
    uses_wan_vae_bundle: bool = True
    video_vae_backend: str = "wan"
    video_i2v_style: str = "wan"
    bundle_config_merger: str = "wan"
    release_t5_after_encode: bool = True
    cfg_negative_prompt_style: str = "wan"
    scheduler_bundle_extras: str = "wan_flow_unipc"
    # Disable UniPC corrector for stability (velocity scale mismatch causes instability)
    wan_scheduler_use_corrector: bool = False
    # Velocity scale calibration - None means no scaling (model output matches scheduler assumption)
    wan_velocity_scale: float | None = None
    # ``first_frame`` allows animate/edit with a video asset (extract frame 0 via ffmpeg).
    # ``source_video`` encodes the full source clip for Bernini V2V / RV2V.
    video_edit_source_mode: str = "image_only"
    # Shape C family-owned stack (Bernini-R renderer).
    video_pipeline_shape: str = "dit_standard"
    bernini_renderer: bool = False
    use_src_id_rotary_emb: bool = False
    interpolate_src_id: bool = True
    max_trained_src_id: int = 5


def merge_wan_config_from_bundle(config: WanConfig, bundle_root: Path | None) -> None:
    """Override ``WanConfig`` from ``config.json`` at bundle root or ``transformer/config.json``."""
    if bundle_root is None:
        return
    from backend.engine.pipelines.video_bundle_layout import wan_is_moe_bundle

    is_moe = wan_is_moe_bundle(bundle_root)
    if is_moe:
        config.dual_model = True

    candidates = [bundle_root / "config.json", bundle_root / "transformer" / "config.json"]
    if is_moe:
        # ModelScope / Wan-AI 14B MoE releases ship per-expert config under high/low dirs.
        candidates.extend(
            [
                bundle_root / "high_noise_model" / "config.json",
                bundle_root / "low_noise_model" / "config.json",
                bundle_root / "transformer" / "config.json",
                bundle_root / "transformer_2" / "config.json",
            ]
        )
    cfg_path = next((p for p in candidates if p.is_file()), None)
    if cfg_path is None:
        return
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Wan: cannot read config {cfg_path}: {e}") from e

    key_map = {
        "dim": "dim",
        "hidden_size": "dim",
        "num_layers": "depth",
        "num_heads": "num_heads",
        "num_attention_heads": "num_heads",
        "ffn_dim": "ffn_dim",
        "in_dim": "dim_in",
        "in_channels": "dim_in",
        "out_dim": "dim_out",
        "out_channels": "dim_out",
        "text_dim": "text_dim",
        "text_len": "text_len",
        "freq_dim": "freq_dim",
        "qk_norm": "qk_norm",
        "cross_attn_norm": "cross_attn_norm",
        "eps": "eps",
        "dual_model": "dual_model",
        "expand_timesteps": "expand_timesteps",
        "vae_scale": "vae_scale",
        "temporal_vae_scale": "temporal_vae_scale",
        "vae_z_dim": "vae_z_dim",
        "use_src_id_rotary_emb": "use_src_id_rotary_emb",
        "interpolate_src_id": "interpolate_src_id",
        "max_trained_src_id": "max_trained_src_id",
    }
    for src, dst in key_map.items():
        if src in data:
            val = data[src]
            if isinstance(val, bool):
                setattr(config, dst, bool(val))
            elif isinstance(val, float):
                setattr(config, dst, float(val))
            else:
                setattr(config, dst, int(val))
    if "patch_size" in data:
        ps = data["patch_size"]
        if isinstance(ps, (list, tuple)) and len(ps) == 3:
            object.__setattr__(config, "patch_size", tuple(int(x) for x in ps))
    if "vae_stride" in data:
        vs = data["vae_stride"]
        if isinstance(vs, (list, tuple)) and len(vs) >= 2:
            config.temporal_vae_scale = int(vs[0])
            config.vae_scale = int(vs[1])
    if "model_type" in data and str(data["model_type"]).lower() in ("ti2v", "t2v", "i2v"):
        object.__setattr__(config, "model_type", str(data["model_type"]).lower())


def merge_wan_vae_config_from_bundle(config: WanConfig, bundle_root: Path | None) -> None:
    """Apply ``vae/config.json`` (diffusers layout) → ``WanConfig`` pipeline scalars."""
    if bundle_root is None:
        return
    vae_cfg_path = bundle_root / "vae" / "config.json"
    if vae_cfg_path.is_file():
        try:
            data: dict[str, Any] = json.loads(vae_cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Wan: cannot read VAE config {vae_cfg_path}: {e}") from e

        if "scale_factor_spatial" in data:
            config.vae_scale = int(data["scale_factor_spatial"])
        if "scale_factor_temporal" in data:
            config.temporal_vae_scale = int(data["scale_factor_temporal"])
        z_dim = data.get("z_dim")
        if z_dim is not None:
            z = int(z_dim)
            config.vae_z_dim = z
            config.dim_in = z
            config.dim_out = z
        return

    # Official Wan2.1 ``Wan2.1_VAE.pth`` bundles (I2V/T2V 14B, distill) — no ``vae/config.json``.
    wan21_pth = bundle_root / "Wan2.1_VAE.pth"
    if wan21_pth.is_file():
        config.vae_z_dim = 16
        config.vae_scale = 8
        config.temporal_vae_scale = 4
        return

    # Bernini-R / mlx-community flat bundles: root ``vae.safetensors`` + ``config.json`` ``vae_stride``.
    root_vae = bundle_root / "vae.safetensors"
    root_cfg = bundle_root / "config.json"
    if root_vae.is_file() and root_cfg.is_file():
        try:
            data = json.loads(root_cfg.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Wan: cannot read bundle config {root_cfg}: {e}") from e
        vs = data.get("vae_stride")
        if isinstance(vs, (list, tuple)) and len(vs) >= 2:
            config.temporal_vae_scale = int(vs[0])
            config.vae_scale = int(vs[1])
        if "vae_z_dim" in data:
            z = int(data["vae_z_dim"])
            config.vae_z_dim = z
            config.dim_in = z
            config.dim_out = z


def merge_wan_bundle_config(config: WanConfig, bundle_root: Path | None) -> None:
    """Merge transformer + VAE JSON from a Wan model bundle."""
    merge_wan_config_from_bundle(config, bundle_root)
    merge_wan_vae_config_from_bundle(config, bundle_root)


@dataclass
class HunyuanVideoConfig:
    """HunyuanVideo-1.5 — diffusers ``HunyuanVideo15Transformer3DModel`` / 480p T2V defaults."""

    inner_dim: int = 2048
    num_attention_heads: int = 16
    attention_head_dim: int = 128
    in_channels: int = 65
    out_channels: int = 32
    dim_in: int = 32
    num_layers: int = 54
    num_refiner_layers: int = 2
    mlp_ratio: float = 4.0
    patch_size: int = 1
    patch_size_t: int = 1
    qk_norm: str = "rms_norm"
    text_embed_dim: int = 3584
    text_embed_2_dim: int = 1472
    image_embed_dim: int = 1152
    rope_theta: float = 256.0
    rope_axes_dim: tuple[int, ...] = (16, 56, 56)
    target_size: int = 640
    task_type: str = "t2v"
    use_meanflow: bool = False

    encoder_type: str = "hunyuan_video_dual"
    text_dim: int = 3584
    mllm_max_length: int = 1000
    byt5_max_length: int = 256
    prompt_template_crop_start: int = 108
    vision_num_semantic_tokens: int = 256

    supports_guidance: bool = True
    supports_img2img: bool = True
    default_scheduler: str = "flow_match_euler"
    temporal_vae_scale: int = 4
    vae_scale: int = 16
    min_unified_memory_gb: int = 32
    step_distill: bool = False
    hunyuan_distill_timesteps: tuple[float, ...] = ()
    hunyuan_distill_shift: float = 9.0
    text_encoder_device: str = "auto"
    text_encoder_qwen_local: str = ""
    text_encoder_byt5_local: str = ""
    text_encoder_release_after_encode: bool = True
    vae_temporal_chunk_size: int = 8
    vae_spatial_tiling: bool = False
    inject_text_encoder_paths: bool = True
    post_denoise_clear_cache: bool = True
    geometry_check: str = "generic"
    uses_hunyuan_vae_bundle: bool = True
    video_vae_backend: str = "hunyuan"
    video_i2v_style: str = "hunyuan"
    bundle_config_merger: str = "hunyuan"
    release_t5_after_encode: bool = False
    cfg_negative_prompt_style: str = "default"
    scheduler_bundle_extras: str = ""
    default_encoder_type: str = "hunyuan_video_dual"
    video_edit_source_mode: str = "image_only"
    teacache_mode: str = "auto"
    use_text_embedding_cache: bool = True


def merge_hunyuan_transformer_config_from_bundle(config: HunyuanVideoConfig, bundle_root: Path | None) -> None:
    """Override ``HunyuanVideoConfig`` from ``<bundle>/transformer/config.json``."""
    if bundle_root is None:
        return
    cfg_path = bundle_root / "transformer" / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(
            f"HunyuanVideo: missing transformer config {cfg_path}. "
            "Install the full diffusers bundle (transformer/config.json + weight shards)."
        )
    try:
        data: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"HunyuanVideo: cannot read transformer config {cfg_path}: {e}") from e

    if "heads_num" in data or str(data.get("_class_name", "")).startswith("HunyuanVideo_1_5"):
        _merge_hunyuan_modelscope_transformer_config(config, data, bundle_root)
        return

    _INT_KEYS = (
        "num_attention_heads", "attention_head_dim", "in_channels", "out_channels",
        "num_layers", "num_refiner_layers", "patch_size", "patch_size_t",
        "text_embed_dim", "text_embed_2_dim", "image_embed_dim", "target_size",
    )
    _FLOAT_KEYS = ("mlp_ratio", "rope_theta")
    _STR_KEYS = ("qk_norm", "task_type")
    _BOOL_KEYS = ("use_meanflow",)
    for k in _INT_KEYS:
        if k in data:
            setattr(config, k, int(data[k]))
    for k in _FLOAT_KEYS:
        if k in data:
            setattr(config, k, float(data[k]))
    for k in _STR_KEYS:
        if k in data:
            setattr(config, k, str(data[k]))
    for k in _BOOL_KEYS:
        if k in data:
            setattr(config, k, bool(data[k]))
    if "rope_axes_dim" in data:
        object.__setattr__(config, "rope_axes_dim", tuple(int(x) for x in data["rope_axes_dim"]))

    na = int(getattr(config, "num_attention_heads", 16))
    hd = int(getattr(config, "attention_head_dim", 128))
    object.__setattr__(config, "inner_dim", na * hd)
    object.__setattr__(config, "dim_in", int(getattr(config, "out_channels", 32)))
    object.__setattr__(config, "text_dim", int(getattr(config, "text_embed_dim", 3584)))


def _infer_hunyuan_patch_in_channels(bundle_root: Path, data: dict[str, Any]) -> int | None:
    """Read patch-embed input channels from checkpoint (ModelScope ``img_in`` or diffusers ``x_embedder``)."""
    weight_path = bundle_root / "transformer" / "diffusion_pytorch_model.safetensors"
    if not weight_path.is_file():
        return None
    try:
        from safetensors import safe_open

        with safe_open(str(weight_path), framework="numpy") as f:
            for key in ("img_in.proj.weight", "x_embedder.proj.weight"):
                if key not in f.keys():
                    continue
                tensor = f.get_tensor(key)
                if getattr(tensor, "ndim", 0) != 5:
                    continue
                # PyTorch Conv3d [O, I, T, H, W]
                return int(tensor.shape[1])
    except Exception:
        return None
    return None


def _merge_hunyuan_modelscope_transformer_config(
    config: HunyuanVideoConfig,
    data: dict[str, Any],
    bundle_root: Path,
) -> None:
    """Map Tencent ModelScope ``transformer/<variant>/config.json`` into ``HunyuanVideoConfig``."""
    heads = int(data.get("heads_num", getattr(config, "num_attention_heads", 16)))
    hidden = int(data.get("hidden_size", getattr(config, "inner_dim", 2048)))
    head_dim = hidden // heads if heads else int(getattr(config, "attention_head_dim", 128))

    object.__setattr__(config, "num_attention_heads", heads)
    object.__setattr__(config, "attention_head_dim", head_dim)
    object.__setattr__(config, "inner_dim", hidden)

    if "in_channels" in data:
        inferred = _infer_hunyuan_patch_in_channels(bundle_root, data)
        object.__setattr__(config, "in_channels", inferred if inferred is not None else int(data["in_channels"]))
    if "out_channels" in data:
        object.__setattr__(config, "out_channels", int(data["out_channels"]))

    if "mm_double_blocks_depth" in data:
        object.__setattr__(config, "num_layers", int(data["mm_double_blocks_depth"]))
    # ``mm_single_blocks_depth`` is upstream single-stream depth, not Qwen token-refiner depth.

    patch_raw = data.get("patch_size")
    if isinstance(patch_raw, list) and patch_raw:
        pts = [int(x) for x in patch_raw]
        object.__setattr__(config, "patch_size_t", pts[0])
        object.__setattr__(config, "patch_size", pts[1] if len(pts) > 1 else pts[0])
    elif patch_raw is not None:
        object.__setattr__(config, "patch_size", int(patch_raw))

    if "text_states_dim" in data and data["text_states_dim"] is not None:
        object.__setattr__(config, "text_embed_dim", int(data["text_states_dim"]))
    if "text_states_dim_2" in data and data["text_states_dim_2"] is not None:
        object.__setattr__(config, "text_embed_2_dim", int(data["text_states_dim_2"]))
    if "vision_states_dim" in data and data["vision_states_dim"] is not None:
        object.__setattr__(config, "image_embed_dim", int(data["vision_states_dim"]))

    if "mlp_width_ratio" in data:
        object.__setattr__(config, "mlp_ratio", float(data["mlp_width_ratio"]))
    if "rope_theta" in data:
        object.__setattr__(config, "rope_theta", float(data["rope_theta"]))
    if "rope_dim_list" in data:
        object.__setattr__(
            config,
            "rope_axes_dim",
            tuple(int(x) for x in data["rope_dim_list"]),
        )
    if "ideal_task" in data and data["ideal_task"]:
        object.__setattr__(config, "task_type", str(data["ideal_task"]))
    if "use_meanflow" in data:
        object.__setattr__(config, "use_meanflow", bool(data["use_meanflow"]))

    qk_norm_type = str(data.get("qk_norm_type") or "").strip().lower()
    if qk_norm_type:
        object.__setattr__(
            config,
            "qk_norm",
            "rms_norm" if qk_norm_type in {"rms", "rms_norm"} else qk_norm_type,
        )

    object.__setattr__(config, "dim_in", int(getattr(config, "out_channels", 32)))
    object.__setattr__(config, "text_dim", int(getattr(config, "text_embed_dim", 3584)))


# =========================================================================
# Config registry: family → config class
# =========================================================================

FAMILY_CONFIG_MAP: dict[str, type] = {
    # Image
    "flux1": Flux1Config,
    "flux2": Flux2Config,
    "qwen_image": QwenImageConfig,
    "fibo": FIBOConfig,
    "z_image": ZImageConfig,
    "ernie_image": ErnieImageConfig,
    "cogview4": CogView4Config,
    "seedvr2": SeedVR2Config,
    "esrgan": EsrganConfig,
    "hidream_o1": HiDreamO1Config,
    "step1x_edit": Step1XEditConfig,
    "boogu_image": BooguImageConfig,
    # Audio
    "diffrhythm": DiffRhythmConfig,
    "ace_step": AceStepConfig,
    # Video
    "ltx": LTXConfig,
    "ltx25": LTX25Config,
    "longcat": LongCatConfig,
    "longcat_avatar": LongCatAvatarConfig,
    "minimax_h3": MinimaxH3Config,
    "wan": WanConfig,
    "hunyuan": HunyuanVideoConfig,
}

IMAGE_FAMILY_REUSE_CONTRACT = frozenset({
    "flux1", "flux2", "z_image", "qwen_image", "ernie_image", "fibo", "cogview4", "seedvr2", "esrgan",
    "hidream_o1", "step1x_edit", "boogu_image",
})


def get_config_class(family: str) -> type:
    """Get config dataclass by family name."""
    cls = FAMILY_CONFIG_MAP.get(family)
    if cls is None:
        raise KeyError(f"unknown model family: {family}")
    return cls


_IMAGE_BUNDLE_CONFIG_MERGERS: dict[str, Any] = {
    "ernie_image": merge_ernie_image_config_from_bundle,
    "cogview4": merge_cogview4_config_from_bundle,
}


def apply_image_registry_config_overrides(entry: Any, config: Any) -> None:
    """Apply registry ``ui.parameters`` scalars onto a family config dataclass."""
    from backend.engine.contracts.pipeline_registry import registry_scalar_default

    for param_key in (
        "text_encoder_out_layers",
        "vae_scale",
        "image_pipeline_shape",
        "enable_thinking",
        "latent_noise_dtype",
        "max_seq_len",
        "inner_dim",
        "num_heads",
        "attn_head_dim",
        "num_layers",
        "num_single_layers",
        "joint_attention_dim",
        "edit_conditioning_concat",
        "edit_rmbg_composite_output",
        "edit_use_vl_vision",
        "edit_conditioning_latent_concat",
        "edit_use_picture_prefix",
        "edit_plus_multi_image",
        "edit_max_reference_images",
        "patch_token_dim",
        "release_text_encoder_after_encode",
        "use_batched_cfg",
        "use_mlx_compile",
        "use_mlx_compile_auto",
        "teacache_mode",
        "use_mm_dit_text_cache",
        "hidream_quantized_no_edit",
        "hidream_noise_scale",
        "hidream_noise_clip_std",
        "hidream_edit_max_refs",
        "hidream_snap_resolution",
        "step1x_version",
        "step1x_variant",
        "step1x_dit_filename",
        "step1x_max_length",
        "step1x_size_level",
        "boogu_variant",
    ):
        val = registry_scalar_default(entry, param_key, None)
        if val is not None and hasattr(config, param_key):
            if param_key == "text_encoder_out_layers" and isinstance(val, list):
                setattr(config, param_key, tuple(int(x) for x in val))
            else:
                setattr(config, param_key, val)
    sg = registry_scalar_default(entry, "supports_guidance", None)
    if sg is not None:
        config.supports_guidance = bool(sg)


def apply_image_bundle_config_merger(config: Any, bundle_root: Path | None) -> None:
    """Registry-driven bundle ``config.json`` merge for image families."""
    merger = str(getattr(config, "bundle_config_merger", "") or "")
    if not merger:
        return
    fn = _IMAGE_BUNDLE_CONFIG_MERGERS.get(merger)
    if fn is None:
        raise RuntimeError(
            f"Unknown image bundle_config_merger={merger!r}. "
            "Register a merge function in model_configs._IMAGE_BUNDLE_CONFIG_MERGERS."
        )
    fn(config, bundle_root)


def assert_image_family_contract(family: str, config: Any) -> None:
    """Engine-side governance for image families entering ``ImagePipeline``."""
    if family not in IMAGE_FAMILY_REUSE_CONTRACT:
        raise RuntimeError(
            f"Image family {family!r} is not declared in IMAGE_FAMILY_REUSE_CONTRACT. "
            "Register family/config/transformer instead of adding pipeline special-case branches."
        )
    encoder_type = str(getattr(config, "encoder_type", "") or "")
    if not encoder_type:
        raise RuntimeError(
            f"Image family {family!r} config must declare encoder_type for registry-driven text encoding."
        )
