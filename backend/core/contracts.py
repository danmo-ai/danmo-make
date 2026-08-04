"""
v3 API contracts: request/response DTOs and execution context.
1:1 aligned with REST and engine method signatures.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from backend.core.asset_interfaces import IAssetStore


# ----- Task kinds (consistent with routes / scheduler) -----

TaskKind = Literal[
    "image.generation",
    "image.edit",
    "image.upscale",
    "video.generation",
    "video.long_generation",
    "video.edit",
    "video.upscale",
    "audio.generation",
    "audio.edit",
    "lora.training",
    "tools.z_image_merge",
]


# ----- Image -----


class AdapterRef(BaseModel):
    """LoRA / adapter — ``id`` is the registry LoRA model id (optional ``:version``, e.g. ``realism-lora:fp16``)."""

    id: str
    weight: float = Field(1.0, ge=0.0, le=2.0)


class StructuralGuide(BaseModel):
    asset_id: str
    model_id: str = ""
    type: Literal[
        "canny", "depth", "pose", "hed", "mlsd", "scribble", "gray", "auto", "redux",
    ] = "canny"
    weight: float = Field(0.8, ge=0.0, le=2.0)
    inpaint_source_asset_id: Optional[str] = None
    inpaint_mask_asset_id: Optional[str] = None

    @model_validator(mode="after")
    def _inpaint_pair(self) -> "StructuralGuide":
        src = (self.inpaint_source_asset_id or "").strip()
        msk = (self.inpaint_mask_asset_id or "").strip()
        if bool(src) ^ bool(msk):
            raise ValueError(
                "structural_guide.inpaint_source_asset_id and inpaint_mask_asset_id must both be set for inpaint mode"
            )
        return self


class LatentRefineSpec(BaseModel):
    scale: float = Field(1.0, ge=1.0, le=4.0)
    denoise_strength: float = Field(0.35, ge=0.0, le=1.0)
    hires_steps: int = Field(0, ge=0, le=20)
    interpolation: Literal["nearest", "linear", "cubic"] = "linear"


class StyleGuide(BaseModel):
    asset_id: str
    weight: float = 1.0


def work_title_metadata(title: str) -> dict[str, str]:
    """Optional user-facing work title stored in asset metadata JSON."""
    t = (title or "").strip()
    return {"title": t} if t else {}


class ImageGenerationRequest(BaseModel):
    model: str  # "z-image-turbo:fp16" or "z-image-turbo" (no version uses registry default)
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    size: str = "1024x1024"
    n: int = Field(1, ge=1, le=8)
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    structural_guide: Optional[StructuralGuide] = None
    style_guide: Optional[StyleGuide] = None
    lemica_mode: Optional[str] = None
    teacache_mode: Optional[str] = None
    latent_refine: Optional[LatentRefineSpec] = None
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtendSpec(BaseModel):
    directions: list[Literal["top", "bottom", "left", "right"]]
    pixels: int = Field(256, ge=64, le=2048)


class ImageEditRequest(BaseModel):
    model: str
    operation: Literal["rewrite", "retouch", "extend"]
    source_asset_id: str
    reference_asset_ids: list[str] = Field(default_factory=list)
    title: str = ""
    prompt: str
    source_fidelity: float = Field(0.6, ge=0.0, le=1.0)
    mask_asset_id: Optional[str] = None
    extend: Optional[ExtendSpec] = None
    negative_prompt: str = ""
    n: int = Field(1, ge=1, le=8)
    steps: Optional[int] = None
    seed: Optional[int] = None
    # None → 使用注册表 ``parameters.guidance.default``（与参考 CLI 对齐时应对 rewrite 显式传 0）
    guidance: Optional[float] = None
    scheduler: Optional[str] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    structural_guide: Optional[StructuralGuide] = None
    lemica_mode: Optional[str] = None
    teacache_mode: Optional[str] = None
    latent_refine: Optional[LatentRefineSpec] = None
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    # operation=rewrite: reference=full-image img2img; instruct=instruction-based edit (currently only flux1-kontext / text_editing). None=use legacy auto rules.
    rewrite_mode: Optional[Literal["reference", "instruct"]] = None

    @model_validator(mode="after")
    def _rewrite_mode_consistency(self) -> "ImageEditRequest":
        if self.rewrite_mode is not None and self.operation != "rewrite":
            raise ValueError("rewrite_mode is only valid when operation is rewrite")
        return self


class ImageUpscaleRequest(BaseModel):
    model: str
    source_asset_id: str
    scale: Literal[2, 4] = 2
    denoise: float = Field(0.0, ge=0.0, le=1.0)
    tile_size: int = Field(1024, ge=256, le=4096)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----- Video -----


class VideoGenerationRequest(BaseModel):
    model: str
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    size: str = "832x480"
    num_frames: int = 81
    fps: int = 16
    steps: Optional[int] = None
    guidance: Optional[float] = None
    shift: Optional[float] = None
    seed: Optional[int] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    reference_asset_ids: list[str] = Field(default_factory=list)
    teacache_mode: Optional[str] = None



class VideoAvatarRequest(BaseModel):
    """Audio-driven digital human (LongCat-Avatar ATI2V / AT2V)."""

    model: str
    title: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    reference_asset_id: str
    audio_asset_id: str
    size: str = "512x512"
    num_frames: int = 93
    fps: int = 25
    steps: Optional[int] = None
    seed: Optional[int] = None
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoAvatarScriptRequest(BaseModel):
    """Text-script-driven digital human: portrait + script -> TTS audio -> lip-sync video.

    Phase 1: fully stubbed; the engine raises RuntimeError to fail loud.
    Phase 2: TTS family will synthesize audio before delegating to avatar().
    """

    model: str
    title: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    reference_asset_id: str
    script_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Spoken script to be synthesized into audio.",
    )
    tts_model: str = Field(
        default="",
        description="Optional TTS model id; reserved for phase 2.",
    )
    voice_id: str = Field(
        default="",
        description="Optional voice id; reserved for phase 2.",
    )
    size: str = "512x512"
    num_frames: int = 93
    fps: int = 25
    steps: Optional[int] = None
    seed: Optional[int] = None
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoEditRequest(BaseModel):
    model: str
    operation: Literal["animate"] = "animate"
    source_asset_id: str
    tail_asset_id: Optional[str] = None
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    size: str = "832x480"
    num_frames: int = 81
    fps: int = 16
    steps: Optional[int] = None
    guidance: Optional[float] = None
    shift: Optional[float] = None
    seed: Optional[int] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    reference_asset_ids: list[str] = Field(default_factory=list)
    teacache_mode: Optional[str] = None


class VideoUpscaleRequest(BaseModel):
    model: str
    source_asset_id: str
    scale: Literal[2, 4] = 2
    denoise: float = Field(0.3, ge=0.0, le=1.0)
    tile_size: int = Field(1024, ge=256, le=4096)
    temporal_window: int = Field(5, ge=1, le=16)
    max_frames: int = Field(300, ge=1, le=4000)
    fps: int = 16
    steps: Optional[int] = None
    prompt: str = ""
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----- Audio -----


class AudioGenerationRequest(BaseModel):
    model: str  # e.g. "audio-stub" or "audio-stub:stub"
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    duration: Optional[int] = None  # seconds (10-600)
    instrumental: bool = False
    lyrics: str = ""
    vocal_language: str = ""
    vocal_type: str = ""  # male | female | chorus | duet | auto (caption template, not a DiT knob)
    bpm: Optional[int] = None  # auto-detect if None
    key_scale: str = ""  # e.g. "C Major", empty=auto-detect
    time_signature: str = ""  # "2","3","4","6", empty=auto-detect
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    n: int = Field(2, ge=1, le=8)
    lm_expansion: Optional[str] = None  # auto | format | off
    audio_format: str = "mp3"
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioEditRequest(BaseModel):
    model: str
    operation: Literal["cover"] = "cover"
    source_asset_id: str
    prompt: str = ""
    source_fidelity: float = Field(1.0, ge=0.0, le=1.0)
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    n: int = Field(1, ge=1, le=8)
    audio_format: str = "mp3"
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----- LLM / Chat -----


class ChatContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatImageUrl(BaseModel):
    url: str


class ChatContentPartImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ChatImageUrl


ChatMessageContent = str | list[ChatContentPartText | ChatContentPartImageUrl]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: ChatMessageContent


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(512, ge=1, le=8192)
    stream: bool = False
    top_p: float = Field(1.0, ge=0.0, le=1.0)


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: dict[str, Any] = Field(default_factory=dict)


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatDeltaChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    created: int
    model: str
    choices: list[ChatDeltaChoice]


class EnhanceRequest(BaseModel):
    prompt: str
    style_positive: str = ""
    style_negative: str = ""
    target_action: str = ""
    model_id: str = ""


class EnhanceResponse(BaseModel):
    enhanced_prompt: str



# ----- Tools (offline model ops) -----


class ZImageMergeRequest(BaseModel):
    model_a: str
    model_b: str
    model_c: Optional[str] = None
    method: Literal["weighted_sum", "add_difference"] = "weighted_sum"
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    output_name: str
    auto_register: bool = True
    priority: Literal["normal", "high"] = "normal"


# ----- LoRA training -----


class LoraTrainingRequest(BaseModel):
    base_model: str  # e.g. flux1-dev or flux1-dev:fp16
    dataset_id: str
    progress_prompt: str
    preset: Literal["scheme4", "quick", "standard", "quality", "custom"] = "standard"
    output_name: str = ""
    auto_register: bool = True
    priority: Literal["normal", "high"] = "normal"
    # Advanced (optional overrides; merged after preset)
    iterations: Optional[int] = None
    batch_size: Optional[int] = Field(None, ge=1, le=8)
    lora_rank: Optional[int] = Field(None, ge=1, le=128)
    lora_blocks: Optional[int] = None
    learning_rate: Optional[float] = Field(None, gt=0)
    grad_accumulate: Optional[int] = Field(None, ge=1, le=64)
    warmup_steps: Optional[int] = Field(None, ge=0)
    resolution: Optional[list[int]] = None  # [width, height]
    num_augmentations: Optional[int] = Field(None, ge=1, le=20)
    progress_every: Optional[int] = Field(None, ge=10)
    progress_steps: Optional[int] = Field(None, ge=4, le=100)
    checkpoint_every: Optional[int] = Field(None, ge=10)
    guidance: Optional[float] = Field(None, ge=0)
    qlora_bits: Optional[int] = Field(None, description="4 or 8 for QLoRA base weights")
    grad_checkpoint: Optional[bool] = None
    lora_scale: Optional[float] = Field(None, gt=0, description="Legacy: prefer lora_alpha")
    lora_alpha: Optional[int] = Field(None, gt=0, description="LoRA alpha (alpha/rank = effective scale)")
    lora_dropout: Optional[float] = Field(None, ge=0, le=0.5)
    lora_module_keys: Optional[list[str]] = None
    optimizer: Optional[Literal["adam", "adamw"]] = None
    weight_decay: Optional[float] = Field(None, ge=0)
    val_split: Optional[float] = Field(None, ge=0, le=0.5)
    val_every: Optional[int] = Field(None, ge=10)
    compile_step: Optional[bool] = None
    resume_from: Optional[str] = None
    resume_task_id: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    train_type: Optional[Literal["lora", "dora"]] = None
    min_snr_gamma: Optional[float] = Field(None, ge=0)
    class_prompt: Optional[str] = None
    prior_loss_weight: Optional[float] = Field(None, ge=0)
    early_stop_patience: Optional[int] = Field(None, ge=0)
    fuse_adapters: Optional[bool] = None
    caption_mode: Optional[Literal["unified", "per_image"]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoraTrainingResult(BaseModel):
    adapter_path: str
    user_lora_id: str = ""
    output_name: str = ""
    loss_history: list[dict[str, Any]] = Field(default_factory=list)


class DatasetCreateRequest(BaseModel):
    name: str
    kind: Literal["concept", "style"] = "concept"
    trigger_word: str = ""
    default_prompt: str = ""
    nsfw: bool = False


class DatasetCaptionUpdate(BaseModel):
    captions: list[dict[str, str]]


class DatasetImportAssetsRequest(BaseModel):
    asset_ids: list[str] = Field(..., min_length=1)
    default_prompt: str = ""
    captions: dict[str, str] = Field(default_factory=dict)


class DatasetAutoCaptionRequest(BaseModel):
    files: list[str] = Field(default_factory=list)


class DatasetHealthVlmRequest(BaseModel):
    max_samples: int = Field(0, ge=0, le=64)
    audit_kind: Optional[Literal["concept", "style"]] = None


class LoraRegisterRequest(BaseModel):
    checkpoint: str = "final_adapters.safetensors"
    name: str = ""
    auto_register: bool = True



class CancelToken:
    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()


@dataclass
class ProgressEvent:
    progress: float
    step: Optional[int] = None
    total: Optional[int] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None
    # UX phases: encoding | loading_model | denoising | decoding | saving
    phase: Optional[str] = None


@dataclass
class LogEvent:
    level: Literal["info", "warning", "error", "success"]
    message: str


@dataclass
class ExecutionContext:
    task_id: str
    cancel_token: CancelToken
    on_progress: Callable[[ProgressEvent], None]
    on_log: Callable[[LogEvent], None]
    work_dir: Path
    asset_store: IAssetStore
    trace: Any = None  # backend.observability.trace.RunTrace | None


@dataclass
class EngineResult:
    primary_asset_id: str
    asset_ids: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def new_asset_id() -> str:
    return "ast_" + secrets.token_hex(12)


def new_task_id() -> str:
    return "tsk_" + secrets.token_hex(12)



def parse_model_version(model_field: str) -> tuple[str, str]:
    """'flux1-dev:fp16' -> ('flux1-dev', 'fp16'); no colon means version is empty"""
    if ":" in model_field:
        a, b = model_field.split(":", 1)
        return a.strip(), b.strip()
    return model_field.strip(), ""


def parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        return 1024, 1024
    return int(parts[0]), int(parts[1])
