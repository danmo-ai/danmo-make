//! Request DTOs for the product API / engine boundary.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Default, Clone)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    #[serde(rename = "image.generation")]
    ImageGeneration,
    #[serde(rename = "image.edit")]
    ImageEdit,
    #[serde(rename = "image.upscale")]
    ImageUpscale,
    #[serde(rename = "video.generation")]
    VideoGeneration,
    #[serde(rename = "video.edit")]
    VideoEdit,
    #[serde(rename = "video.upscale")]
    VideoUpscale,
    #[serde(rename = "audio.generation")]
    AudioGeneration,
    #[serde(rename = "audio.edit")]
    AudioEdit,
    /// ACE-Step 5Hz LM inspire / format / generate (phase-1 text; codes later).
    #[serde(rename = "audio.lm")]
    AudioLm,
    #[serde(rename = "lora.training")]
    LoraTraining,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ImageGeneration => "image.generation",
            Self::ImageEdit => "image.edit",
            Self::ImageUpscale => "image.upscale",
            Self::VideoGeneration => "video.generation",
            Self::VideoEdit => "video.edit",
            Self::VideoUpscale => "video.upscale",
            Self::AudioGeneration => "audio.generation",
            Self::AudioEdit => "audio.edit",
            Self::AudioLm => "audio.lm",
            Self::LoraTraining => "lora.training",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    #[serde(default)]
    pub title: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(default = "one")]
    pub n: u32,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub seed: Option<i64>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,
    #[serde(default)]
    pub use_empirical_mu: bool,
    #[serde(default)]
    pub adapters: Vec<ImageAdapter>,
    #[serde(default)]
    pub structural_guide: Option<StructuralGuide>,
    /// Mid-denoise step thumbnails. Omitted stays on.
    #[serde(default = "default_true")]
    pub preview: bool,
    #[serde(default = "default_priority")]
    pub priority: String,
}

/// ControlNet structural guide on an image create, aligned with
/// Structural guide conditioning. `asset_id` is the preprocessed
/// control image. `model_id` is the registry controlnet entry. The inpaint
/// pair is both-or-neither.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralGuide {
    pub asset_id: String,
    #[serde(default)]
    pub model_id: String,
    #[serde(rename = "type", default = "default_guide_type")]
    pub guide_type: String,
    #[serde(default = "default_guide_weight")]
    pub weight: f32,
    #[serde(default)]
    pub inpaint_source_asset_id: Option<String>,
    #[serde(default)]
    pub inpaint_mask_asset_id: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_guide_type() -> String {
    "canny".into()
}

fn default_guide_weight() -> f32 {
    0.8
}

/// One LoRA on an image create. `path` is filled when the request is accepted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAdapter {
    pub id: String,
    #[serde(default = "default_adapter_weight")]
    pub weight: f32,
    #[serde(default)]
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEditRequest {
    pub model: String,
    pub operation: String,
    pub source_asset_id: String,
    #[serde(default)]
    pub reference_asset_ids: Vec<String>,
    #[serde(default)]
    pub title: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    /// Rewrite img2img strength. None is the reference default 0.6: the
    /// denoise starts at `max(1, int(steps * fidelity))`. Zero is full
    /// text-to-image at the source size. One leaves no denoising step and
    /// fails. Only the rewrite img2img families read it.
    pub source_fidelity: Option<f32>,
    /// None keeps the source-derived size (Qwen edit).
    pub size: Option<String>,
    #[serde(default = "one")]
    pub n: u32,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub seed: Option<i64>,
    pub scheduler: Option<String>,
    #[serde(default)]
    pub adapters: Vec<ImageAdapter>,
    #[serde(default)]
    pub metadata: Value,
    /// Mid-denoise step thumbnails. Omitted stays on.
    #[serde(default = "default_true")]
    pub preview: bool,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUpscaleRequest {
    pub model: String,
    pub source_asset_id: String,
    /// Output scale over the source short edge. None is the reference default
    /// 2. Only 2 and 4 are registered.
    pub scale: Option<u32>,
    /// Softness of the bicubic pre-downscale (0 sharp .. 1 strong). None is
    /// the reference default 0.
    pub denoise: Option<f32>,
    pub seed: Option<i64>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUpscaleRequest {
    pub model: String,
    #[serde(default)]
    pub title: String,
    pub source_asset_id: String,
    /// Output scale over the source short edge. None is the reference default
    /// 2. Only 2 and 4 are registered.
    pub scale: Option<u32>,
    /// Softness of the bicubic pre-downscale (0 sharp .. 1 strong). None is
    /// the reference default 0.3 for video.
    pub denoise: Option<f32>,
    /// Frames per spatiotemporal restoration chunk. None is the reference
    /// default 5; values below 4 are clamped to 4.
    pub temporal_window: Option<u32>,
    /// Maximum decoded frames from the source. None is the reference default
    /// 300.
    pub max_frames: Option<u32>,
    pub seed: Option<i64>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    /// "WxH"; None is the reference default 832x480. Wan snaps each side down
    /// to a multiple of vae_scale * patch (32).
    pub size: Option<String>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    /// Flow-match sigma shift; None is the reference default (wan ti2v: 5).
    pub shift: Option<f32>,
    pub seed: Option<i64>,
    /// None is the reference default 81. Wan requires (num_frames - 1) % 4 == 0.
    pub num_frames: Option<u32>,
    /// None is the reference default 16.
    pub fps: Option<u32>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_priority")]
    pub priority: String,
}

/// Video edit (image-to-video "animate"). The source asset is the first
/// frame: image assets load directly, video assets contribute frame 0.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoEditRequest {
    pub model: String,
    pub source_asset_id: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    /// "WxH"; None is the family default (hunyuan i2v: 720x1280).
    pub size: Option<String>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    /// Flow-match sigma shift; None is the family default (hunyuan: 7).
    pub shift: Option<f32>,
    pub seed: Option<i64>,
    /// None is the family default (hunyuan: 81). Hunyuan requires
    /// (num_frames - 1) % 4 == 0.
    pub num_frames: Option<u32>,
    /// None is the family default (hunyuan: 24).
    pub fps: Option<u32>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub instrumental: bool,
    #[serde(default)]
    pub vocal_type: String,
    #[serde(default = "default_audio_batch")]
    pub n: u32,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_audio_format")]
    pub audio_format: String,
    #[serde(default = "default_denoise")]
    pub denoise: String,
    #[serde(default)]
    pub lyrics: String,
    pub duration: Option<f32>,
    pub seed: Option<i64>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub vocal_language: String,
    pub guidance: Option<f32>,
    pub steps: Option<u32>,
    /// BPM hint for DiT / 5Hz LM (optional).
    pub bpm: Option<i32>,
    /// Key/scale hint (`C major`, …). Accepts `keyscale` alias.
    #[serde(default, alias = "keyscale")]
    pub key_scale: String,
    /// Time signature hint (`4`, `4/4`, …). Accepts `timesignature` alias.
    #[serde(default, alias = "timesignature")]
    pub time_signature: String,
    /// `auto` (default) | `format` | `off` — 5Hz LM format_sample before DiT.
    #[serde(default = "default_lm_expansion")]
    pub lm_expansion: String,
    /// Optional 5Hz LM / planner FSQ codebook indices for DiT context hints.
    #[serde(default)]
    pub audio_code_indices: Option<Vec<i32>>,
    /// Optional time-major `[T*64]` latents for the timbre encoder.
    #[serde(default)]
    pub timbre_latents: Option<Vec<f32>>,
    #[serde(default)]
    pub dcw_enabled: Option<bool>,
    #[serde(default)]
    pub dcw_mode: Option<String>,
    #[serde(default)]
    pub dcw_wavelet: Option<String>,
    #[serde(default)]
    pub dcw_scaler: Option<f32>,
    #[serde(default)]
    pub dcw_high_scaler: Option<f32>,
    #[serde(default)]
    pub adapters: Vec<ImageAdapter>,
}

fn default_lm_expansion() -> String {
    "auto".into()
}

fn default_audio_batch() -> u32 {
    2
}

fn default_audio_format() -> String {
    "wav".into()
}

fn default_denoise() -> String {
    "off".into()
}

/// ACE-Step cover / cover-nofsq / repaint edit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEditRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub instrumental: bool,
    #[serde(default)]
    pub vocal_type: String,
    #[serde(default = "one")]
    pub n: u32,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_audio_format")]
    pub audio_format: String,
    #[serde(default = "default_denoise")]
    pub denoise: String,
    pub duration: Option<f32>,
    pub bpm: Option<i32>,
    #[serde(default, alias = "keyscale")]
    pub key_scale: String,
    #[serde(default, alias = "timesignature")]
    pub time_signature: String,
    #[serde(default)]
    pub lyrics: String,
    pub seed: Option<i64>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub vocal_language: String,
    pub guidance: Option<f32>,
    pub steps: Option<u32>,
    /// `cover-nofsq` (default), `cover`, `repaint`, `lego`, `extract`, or `complete`.
    #[serde(default = "default_cover_nofsq")]
    pub task: String,
    /// Gallery asset id. Resolved to PCM before the edit session.
    #[serde(default)]
    pub source_asset_id: Option<String>,
    /// Composer cover slider. Used as `audio_cover_strength` when that field is absent.
    #[serde(default)]
    pub source_fidelity: Option<f32>,
    /// Absolute or workspace-relative path to a PCM16 WAV source.
    pub source_audio: Option<String>,
    /// Optional base64-encoded PCM16 WAV (used when `source_audio` is absent).
    pub source_audio_base64: Option<String>,
    /// Optional pre-encoded time-major latents `[T*64]` (skips VAE encode).
    pub source_latents: Option<Vec<f32>>,
    /// Fraction of DiT steps that keep cover context (default 1.0).
    pub audio_cover_strength: Option<f32>,
    /// 0 = pure noise; 1 = start near source (default 0.0).
    pub cover_noise_strength: Option<f32>,
    /// Repaint region start seconds (required for `task=repaint`).
    pub region_start_sec: Option<f32>,
    /// Repaint region end seconds (required for `task=repaint`).
    pub region_end_sec: Option<f32>,
    /// Fraction of steps that inject preserved regions (default 1.0).
    pub repaint_injection_ratio: Option<f32>,
    /// Latent frames of boundary crossfade (default 8).
    pub repaint_crossfade_frames: Option<i64>,
    /// Sample 750-frame timbre reference from source (default false → silence).
    #[serde(default)]
    pub timbre_from_source: Option<bool>,
    /// Track name(s) for lego / extract / complete (`"drums | bass"` for complete).
    #[serde(default)]
    pub track: Option<String>,
    #[serde(default)]
    pub dcw_enabled: Option<bool>,
    #[serde(default)]
    pub dcw_mode: Option<String>,
    #[serde(default)]
    pub dcw_wavelet: Option<String>,
    #[serde(default)]
    pub dcw_scaler: Option<f32>,
    #[serde(default)]
    pub dcw_high_scaler: Option<f32>,
    #[serde(default)]
    pub adapters: Vec<ImageAdapter>,
}

/// ACE-Step 5Hz LM (inspire / format / generate phase-1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioLmRequest {
    pub model: String,
    /// `inspire` | `format` | `generate` (default inspire).
    #[serde(default = "default_lm_inspire")]
    pub mode: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub lyrics: String,
    pub duration: Option<f32>,
    pub bpm: Option<i32>,
    #[serde(default)]
    pub keyscale: String,
    #[serde(default)]
    pub timesignature: String,
    #[serde(default)]
    pub vocal_language: String,
    /// Max newly sampled tokens (phase-1).
    pub max_new_tokens: Option<u32>,
    /// Sampling temperature (`0` = greedy).
    pub temperature: Option<f32>,
    /// Nucleus top-p in (0, 1].
    pub top_p: Option<f32>,
    /// Top-k (`0` = disabled).
    pub top_k: Option<i32>,
    /// RNG seed for non-greedy sampling.
    pub seed: Option<u64>,
    /// Apply MetadataFSM numeric early-block (needs tokenizer; default false).
    #[serde(default)]
    pub use_fsm: bool,
    /// Phase-2 CFG scale (`<= 1` disables).
    pub cfg_scale: Option<f32>,
    /// Negative caption for uncond CFG.
    #[serde(default)]
    pub negative_prompt: String,
    #[serde(default = "default_priority")]
    pub priority: String,
}

fn default_lm_inspire() -> String {
    "inspire".into()
}

fn default_cover_nofsq() -> String {
    "cover-nofsq".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskAccepted {
    pub id: String,
    pub status: String,
    pub kind: String,
}

fn default_size() -> String {
    "1024x1024".into()
}
fn one() -> u32 {
    1
}
fn default_priority() -> String {
    "normal".into()
}

fn default_adapter_weight() -> f32 {
    1.0
}

fn default_train_preset() -> String {
    "turbo-sft".into()
}

/// One companion adapter declared by a trained LoRA's inference record.
///
/// A JSON string is the same adapter at weight 1. An object carries its own
/// weight. Inference appends these records; it does not special-case a name.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StackedAdapter {
    pub id: String,
    pub weight: f32,
}

impl StackedAdapter {
    pub fn new(id: impl Into<String>, weight: f32) -> Self {
        Self {
            id: id.into(),
            weight,
        }
    }
}

impl<'de> Deserialize<'de> for StackedAdapter {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(id) => stacked_adapter(&id, 1.0).map_err(serde::de::Error::custom),
            Value::Object(map) => {
                let id = map.get("id").and_then(|item| item.as_str()).unwrap_or("");
                let weight = map
                    .get("weight")
                    .and_then(|item| item.as_f64())
                    .unwrap_or(1.0) as f32;
                stacked_adapter(id, weight).map_err(serde::de::Error::custom)
            }
            _ => Err(serde::de::Error::custom(
                "extra adapter must be an id or an {id, weight} object",
            )),
        }
    }
}

fn stacked_adapter(id: &str, weight: f32) -> Result<StackedAdapter, String> {
    let id = id.trim();
    if id.is_empty() {
        return Err("extra adapter id is empty".into());
    }
    if !weight.is_finite() || !(0.0..=2.0).contains(&weight) {
        return Err(format!(
            "extra adapter {id} weight {weight} is not between 0 and 2"
        ));
    }
    Ok(StackedAdapter::new(id, weight))
}

/// Read `inference.extra_adapters`. Missing or null means none.
pub fn stacked_adapters_from_value(value: Option<&Value>) -> Result<Vec<StackedAdapter>, String> {
    let Some(value) = value.filter(|item| !item.is_null()) else {
        return Ok(Vec::new());
    };
    serde_json::from_value(value.clone()).map_err(|err| err.to_string())
}

/// Portable inference recipe stored with a trained adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraInferenceRecipe {
    pub scheme: String,
    pub model: String,
    #[serde(default)]
    pub steps: u32,
    #[serde(default)]
    pub guidance: f32,
    #[serde(default = "default_adapter_weight")]
    pub lora_weight: f32,
    #[serde(default)]
    pub scheduler: String,
    #[serde(default)]
    pub extra_adapters: Vec<StackedAdapter>,
    #[serde(default)]
    pub lora_weight_range: Option<[f32; 2]>,
}

/// DiffSynth-aligned LoRA training submit body (v1: Z-Image / Turbo).
///
/// Knobs come from the named `preset` only. Unified caption is the dataset
/// `default_prompt` (loaded server-side); clients must not send overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainingRequest {
    pub base_model: String,
    pub dataset_id: String,
    #[serde(default = "default_train_preset")]
    pub preset: String,
    #[serde(default)]
    pub output_name: String,
    #[serde(default)]
    pub auto_register: bool,
    #[serde(default = "default_priority")]
    pub priority: String,
    pub seed: Option<i64>,
    /// `auto` / `on` for concept; `off` allowed for style datasets.
    #[serde(default)]
    pub face_crop: Option<String>,
    #[serde(default)]
    pub resume_task_id: String,
    #[serde(default)]
    pub resume_checkpoint: String,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainingResult {
    pub adapter_path: String,
    #[serde(default)]
    pub user_lora_id: String,
    #[serde(default)]
    pub output_name: String,
    #[serde(default)]
    pub loss_history: Vec<Value>,
    #[serde(default)]
    pub needs_candidate_evaluation: bool,
    #[serde(default)]
    pub checkpoints: Vec<String>,
    #[serde(default)]
    pub inference: Option<LoraInferenceRecipe>,
}

/// Dataset `meta.json` fields. `id` and `face_crop` are required on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraDatasetMeta {
    pub id: String,
    #[serde(default)]
    pub name: String,
    /// `concept` (unified caption) or `style` (per-row prompts).
    #[serde(default = "default_dataset_kind")]
    pub kind: String,
    #[serde(default)]
    pub trigger_word: String,
    #[serde(default)]
    pub default_prompt: String,
    #[serde(default)]
    pub nsfw: bool,
    pub face_crop: String,
    #[serde(default)]
    pub image_count: u32,
}

fn default_dataset_kind() -> String {
    "concept".into()
}

/// One `train.jsonl` row (DiffSynth `image,prompt` columns).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraDatasetRow {
    pub image: String,
    #[serde(default)]
    pub prompt: String,
}

/// Resolved training preset (after merge with request overrides).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraTrainPreset {
    pub id: String,
    pub iterations: u32,
    pub lora_rank: u32,
    pub lora_blocks: i32,
    pub learning_rate: f32,
    pub grad_accumulate: u32,
    pub resolution: [u32; 2],
    pub progress_every: u32,
    pub checkpoint_every: u32,
    pub lora_module_keys: Vec<String>,
    pub caption_dropout: f32,
    pub train_sigma_shift: f32,
    pub sigma_bias: String,
    pub inference: LoraInferenceRecipe,
}

pub fn z_image_lora_module_keys() -> Vec<String> {
    ["to_q", "to_k", "to_v", "to_out", "w1", "w2", "w3"]
        .into_iter()
        .map(str::to_string)
        .collect()
}

/// Built-in presets for Z-Image / Turbo (DiffSynth Scheme 4 default).
pub fn resolve_lora_train_preset(preset: &str, base_model: &str) -> Result<LoraTrainPreset, String> {
    let mid = base_model.split(':').next().unwrap_or(base_model).trim();
    let key = preset.trim().to_ascii_lowercase();
    let modules = z_image_lora_module_keys();
    match (mid, key.as_str()) {
        ("z-image-turbo", "turbo-sft" | "sft" | "standard" | "") => Ok(LoraTrainPreset {
            id: "turbo-sft".into(),
            iterations: 2500,
            lora_rank: 32,
            lora_blocks: -1,
            learning_rate: 1e-4,
            grad_accumulate: 1,
            resolution: [1024, 1024],
            progress_every: 500,
            checkpoint_every: 500,
            lora_module_keys: modules,
            caption_dropout: 0.05,
            train_sigma_shift: 3.0,
            sigma_bias: "discrete".into(),
            inference: LoraInferenceRecipe {
                scheme: "sft".into(),
                model: "z-image-turbo".into(),
                steps: 8,
                guidance: 0.0,
                lora_weight: 0.8,
                scheduler: "linear".into(),
                extra_adapters: vec![StackedAdapter::new(
                    "z-image-turbo-distillpatch-lora:bf16",
                    1.0,
                )],
                lora_weight_range: Some([0.7, 0.9]),
            },
        }),
        ("z-image-turbo", "turbo-diff" | "turbo") => Ok(LoraTrainPreset {
            id: "turbo-diff".into(),
            iterations: 2500,
            lora_rank: 32,
            lora_blocks: -1,
            learning_rate: 1e-4,
            grad_accumulate: 1,
            resolution: [1024, 1024],
            progress_every: 500,
            checkpoint_every: 500,
            lora_module_keys: modules,
            caption_dropout: 0.05,
            train_sigma_shift: 1.0,
            sigma_bias: "sigmoid".into(),
            inference: LoraInferenceRecipe {
                scheme: "turbo".into(),
                model: "z-image-turbo".into(),
                steps: 9,
                guidance: 0.0,
                lora_weight: 0.9,
                scheduler: "linear".into(),
                extra_adapters: Vec::new(),
                lora_weight_range: Some([0.8, 1.0]),
            },
        }),
        ("z-image", "base-face" | "standard" | "quality" | "") => Ok(LoraTrainPreset {
            id: "base-face".into(),
            iterations: 2500,
            lora_rank: 32,
            lora_blocks: -1,
            learning_rate: 1e-4,
            grad_accumulate: 1,
            resolution: [1024, 1024],
            progress_every: 500,
            checkpoint_every: 500,
            lora_module_keys: modules,
            caption_dropout: 0.05,
            train_sigma_shift: 1.0,
            sigma_bias: "sigmoid".into(),
            inference: LoraInferenceRecipe {
                scheme: "base".into(),
                model: "z-image".into(),
                steps: 28,
                guidance: 5.0,
                lora_weight: 1.0,
                scheduler: "flow_match".into(),
                extra_adapters: Vec::new(),
                lora_weight_range: None,
            },
        }),
        _ => Err(format!(
            "unknown training preset {preset:?} for base model {mid:?}; \
             choose turbo-sft|turbo-diff|base-face"
        )),
    }
}

/// Validate that the request base model matches the resolved preset recipe.
/// Named presets are closed; request bodies no longer carry knob overrides.
pub fn merge_lora_train_request(
    request: &LoraTrainingRequest,
    preset: LoraTrainPreset,
) -> Result<LoraTrainPreset, String> {
    let mid = request
        .base_model
        .split(':')
        .next()
        .unwrap_or(&request.base_model)
        .trim();
    if preset.inference.model != mid {
        return Err(format!(
            "inference model {:?} must equal training base {mid:?}",
            preset.inference.model
        ));
    }
    Ok(preset)
}

pub fn parse_size(size: &str) -> Result<(u32, u32), String> {
    let (w, h) = size
        .split_once('x')
        .ok_or_else(|| format!("size must be WxH, got {size}"))?;
    let w: u32 = w.parse().map_err(|_| format!("bad width in {size}"))?;
    let h: u32 = h.parse().map_err(|_| format!("bad height in {size}"))?;
    if w == 0 || h == 0 {
        return Err(format!("size must be non-zero, got {size}"));
    }
    Ok((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_preview_defaults_on_and_can_be_disabled() {
        let on: ImageGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "z-image-turbo",
            "prompt": "red"
        }))
        .unwrap();
        assert!(on.preview);
        let off: ImageEditRequest = serde_json::from_value(serde_json::json!({
            "model": "z-image-turbo",
            "operation": "rewrite",
            "source_asset_id": "ast",
            "prompt": "red",
            "preview": false
        }))
        .unwrap();
        assert!(!off.preview);
    }

    #[test]
    fn audio_generation_defaults_lm_expansion_auto() {
        let req: AudioGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "ace-step-xl-sft",
            "prompt": "lofi"
        }))
        .unwrap();
        assert_eq!(req.lm_expansion, "auto");
        assert_eq!(req.n, 2);
        assert_eq!(req.audio_format, "wav");
        assert_eq!(req.denoise, "off");
        assert!(req.key_scale.is_empty());
        assert!(req.time_signature.is_empty());
        assert!(req.bpm.is_none());
    }

    #[test]
    fn audio_generation_accepts_keyscale_aliases_and_lm_off() {
        let req: AudioGenerationRequest = serde_json::from_value(serde_json::json!({
            "model": "ace-step-xl-sft",
            "prompt": "lofi",
            "bpm": 96,
            "keyscale": "A minor",
            "timesignature": "4",
            "lm_expansion": "off"
        }))
        .unwrap();
        assert_eq!(req.bpm, Some(96));
        assert_eq!(req.key_scale, "A minor");
        assert_eq!(req.time_signature, "4");
        assert_eq!(req.lm_expansion, "off");
    }

    #[test]
    fn turbo_sft_preset_is_default_scheme4() {
        let preset = resolve_lora_train_preset("turbo-sft", "z-image-turbo").unwrap();
        assert_eq!(preset.id, "turbo-sft");
        assert_eq!(preset.inference.steps, 8);
        assert_eq!(
            preset.inference.extra_adapters,
            vec![StackedAdapter::new(
                "z-image-turbo-distillpatch-lora:bf16",
                1.0
            )]
        );
        let from_string: LoraInferenceRecipe = serde_json::from_value(serde_json::json!({
            "scheme": "sft",
            "model": "z-image-turbo",
            "extra_adapters": ["companion:bf16"]
        }))
        .unwrap();
        assert_eq!(from_string.extra_adapters[0].weight, 1.0);
        let from_object: LoraInferenceRecipe = serde_json::from_value(serde_json::json!({
            "scheme": "sft",
            "model": "z-image-turbo",
            "extra_adapters": [{ "id": "companion:bf16", "weight": 0.5 }]
        }))
        .unwrap();
        assert_eq!(from_object.extra_adapters[0].id, "companion:bf16");
        assert_eq!(from_object.extra_adapters[0].weight, 0.5);
        let req = LoraTrainingRequest {
            base_model: "z-image-turbo".into(),
            dataset_id: "ds".into(),
            preset: "turbo-sft".into(),
            output_name: String::new(),
            auto_register: false,
            priority: "normal".into(),
            seed: None,
            face_crop: None,
            resume_task_id: String::new(),
            resume_checkpoint: String::new(),
            metadata: Value::Null,
        };
        let merged = merge_lora_train_request(&req, preset).unwrap();
        assert_eq!(merged.iterations, 2500);
        assert_eq!(merged.id, "turbo-sft");
    }
}
