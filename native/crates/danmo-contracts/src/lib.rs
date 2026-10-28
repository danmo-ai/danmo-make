//! Request DTOs aligned with `backend/core/contracts.py`.

use serde::{Deserialize, Serialize};

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
    #[serde(default = "default_priority")]
    pub priority: String,
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
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub seed: Option<i64>,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUpscaleRequest {
    pub model: String,
    pub source_asset_id: String,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub seed: Option<i64>,
    pub num_frames: Option<u32>,
    #[serde(default = "default_priority")]
    pub priority: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioGenerationRequest {
    pub model: String,
    pub prompt: String,
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
    /// Optional 5Hz LM / planner FSQ codebook indices for DiT context hints.
    #[serde(default)]
    pub audio_code_indices: Option<Vec<i32>>,
    /// Optional time-major `[T*64]` latents for the timbre encoder.
    #[serde(default)]
    pub timbre_latents: Option<Vec<f32>>,
}

/// ACE-Step cover / cover-nofsq / repaint edit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioEditRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub lyrics: String,
    pub seed: Option<i64>,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub vocal_language: String,
    pub guidance: Option<f32>,
    pub steps: Option<u32>,
    /// `cover-nofsq` (default), `cover` (FSQ), or `repaint` (region regenerate).
    #[serde(default = "default_cover_nofsq")]
    pub task: String,
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
