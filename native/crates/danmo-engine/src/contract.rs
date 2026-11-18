use crate::error::EngineError;
use crate::runtime::BackendKind;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub use danmo_contracts::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Media {
    Image,
    Video,
    Audio,
    Assistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    DitPipeline,
    JobPipeline,
    GenerationFacade,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    ImageCreate,
    ImageEdit,
    ImageUpscale,
    VideoCreate,
    VideoEdit,
    VideoUpscale,
    VideoAvatar,
    AudioCreate,
    AudioEdit,
    AudioLm,
    Assistant,
}

impl Action {
    pub fn media(self) -> Media {
        match self {
            Self::ImageCreate | Self::ImageEdit | Self::ImageUpscale => Media::Image,
            Self::VideoCreate | Self::VideoEdit | Self::VideoUpscale | Self::VideoAvatar => {
                Media::Video
            }
            Self::AudioCreate | Self::AudioEdit | Self::AudioLm => Media::Audio,
            Self::Assistant => Media::Assistant,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::ImageCreate => "image.create",
            Self::ImageEdit => "image.edit",
            Self::ImageUpscale => "image.upscale",
            Self::VideoCreate => "video.create",
            Self::VideoEdit => "video.edit",
            Self::VideoUpscale => "video.upscale",
            Self::VideoAvatar => "video.avatar",
            Self::AudioCreate => "audio.create",
            Self::AudioEdit => "audio.edit",
            Self::AudioLm => "audio.lm",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedRun {
    pub model: String,
    pub version: Option<String>,
    pub family: String,
    pub action: Action,
    pub bundle: PathBuf,
    pub backend: BackendKind,
}

pub trait RunObserver: Send + Sync {
    fn progress(&self, phase: &str, step: u32, total: u32) -> Result<(), EngineError>;
    fn log(&self, level: &str, message: &str) -> Result<(), EngineError>;
    fn phase(&self, phase: &str) -> Result<(), EngineError>;

    /// Denoise steps between living-canvas previews. `None` writes nothing.
    fn preview_interval_steps(&self) -> Option<u32> {
        None
    }

    /// Longest edge of a step preview. Ignored when previews are off.
    fn preview_max_edge(&self) -> u32 {
        512
    }

    fn write_step_preview(&self, png: &[u8]) -> Result<(), EngineError> {
        let _ = png;
        Ok(())
    }
}

#[derive(Clone)]
pub struct ExecutionContext {
    pub task_id: String,
    pub cancellation: CancellationToken,
    pub observer: Arc<dyn RunObserver>,
    pub started: Instant,
}

impl ExecutionContext {
    pub fn new(task_id: impl Into<String>, observer: Arc<dyn RunObserver>) -> Self {
        Self {
            task_id: task_id.into(),
            cancellation: CancellationToken::default(),
            observer,
            started: Instant::now(),
        }
    }

    pub fn with_cancellation(mut self, cancellation: CancellationToken) -> Self {
        self.cancellation = cancellation;
        self
    }

    pub fn check_cancelled(&self) -> Result<(), EngineError> {
        if self.cancellation.is_cancelled() {
            Err(EngineError::Cancelled)
        } else {
            Ok(())
        }
    }

    pub fn phase(&self, phase: &str) -> Result<(), EngineError> {
        self.check_cancelled()?;
        self.observer.phase(phase)
    }

    pub fn progress(&self, phase: &str, step: u32, total: u32) -> Result<(), EngineError> {
        self.check_cancelled()?;
        self.observer.progress(phase, step, total)
    }

    pub fn log(&self, level: &str, message: &str) -> Result<(), EngineError> {
        self.observer.log(level, message)
    }
}

pub struct NoopObserver;

impl RunObserver for NoopObserver {
    fn progress(&self, _phase: &str, _step: u32, _total: u32) -> Result<(), EngineError> {
        Ok(())
    }

    fn log(&self, _level: &str, _message: &str) -> Result<(), EngineError> {
        Ok(())
    }

    fn phase(&self, _phase: &str) -> Result<(), EngineError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AudioCreateRequest {
    pub model: String,
    pub prompt: String,
    pub instrumental: bool,
    pub vocal_type: String,
    pub n: u32,
    pub metadata: Value,
    pub lyrics: String,
    pub duration_sec: f32,
    pub seed: Option<i64>,
    pub guidance: f32,
    pub steps: Option<u32>,
    pub vocal_language: String,
    pub adapters: Vec<danmo_contracts::ImageAdapter>,
    pub family_options: Value,
}

#[derive(Debug, Clone)]
pub struct AudioEditRequest {
    pub model: String,
    pub prompt: String,
    pub instrumental: bool,
    pub vocal_type: String,
    pub n: u32,
    pub metadata: Value,
    pub duration_sec: Option<f32>,
    pub bpm: Option<i32>,
    pub keyscale: String,
    pub timesignature: String,
    pub lyrics: String,
    pub seed: Option<i64>,
    pub guidance: f32,
    pub steps: Option<u32>,
    pub vocal_language: String,
    pub source_audio: Option<String>,
    pub source_audio_base64: Option<String>,
    pub adapters: Vec<danmo_contracts::ImageAdapter>,
    pub family_options: Value,
}

#[derive(Debug, Clone)]
pub struct AudioLmRequest {
    pub model: String,
    pub mode: String,
    pub prompt: String,
    pub lyrics: String,
    pub family_options: Value,
}

#[derive(Debug, Clone)]
pub enum EngineRequest {
    ImageCreate(danmo_contracts::ImageGenerationRequest),
    ImageEdit(danmo_contracts::ImageEditRequest),
    ImageUpscale(danmo_contracts::ImageUpscaleRequest),
    VideoCreate(danmo_contracts::VideoGenerationRequest),
    VideoEdit(danmo_contracts::VideoEditRequest),
    VideoUpscale(danmo_contracts::VideoUpscaleRequest),
    AudioCreate(AudioCreateRequest),
    AudioEdit(AudioEditRequest),
    AudioLm(AudioLmRequest),
    Unsupported { model: String, action: Action },
}

impl EngineRequest {
    pub fn model(&self) -> &str {
        match self {
            Self::ImageCreate(request) => &request.model,
            Self::ImageEdit(request) => &request.model,
            Self::ImageUpscale(request) => &request.model,
            Self::VideoCreate(request) => &request.model,
            Self::VideoEdit(request) => &request.model,
            Self::VideoUpscale(request) => &request.model,
            Self::AudioCreate(request) => &request.model,
            Self::AudioEdit(request) => &request.model,
            Self::AudioLm(request) => &request.model,
            Self::Unsupported { model, .. } => model,
        }
    }

    pub fn action(&self) -> Action {
        match self {
            Self::ImageCreate(_) => Action::ImageCreate,
            Self::ImageEdit(_) => Action::ImageEdit,
            Self::ImageUpscale(_) => Action::ImageUpscale,
            Self::VideoCreate(_) => Action::VideoCreate,
            Self::VideoEdit(_) => Action::VideoEdit,
            Self::VideoUpscale(_) => Action::VideoUpscale,
            Self::AudioCreate(_) => Action::AudioCreate,
            Self::AudioEdit(_) => Action::AudioEdit,
            Self::AudioLm(_) => Action::AudioLm,
            Self::Unsupported { action, .. } => *action,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sidecar {
    pub suffix: String,
    pub mime: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Artifact {
    pub extension: String,
    pub mime: String,
    pub kind: String,
    pub bytes: Vec<u8>,
    pub metadata: Value,
    pub source_action: String,
    pub parent_asset_id: Option<String>,
    pub relation_type: Option<String>,
    pub duration_seconds: Option<f64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub sidecars: Vec<Sidecar>,
}

impl Artifact {
    pub fn png(bytes: Vec<u8>, width: i64, height: i64, metadata: Value) -> Self {
        Self {
            extension: "png".into(),
            mime: "image/png".into(),
            kind: "image".into(),
            bytes,
            metadata,
            source_action: "create".into(),
            parent_asset_id: None,
            relation_type: None,
            duration_seconds: None,
            width: Some(width),
            height: Some(height),
            sidecars: Vec::new(),
        }
    }

    pub fn wav(bytes: Vec<u8>, sample_rate: u32, frames: i64, metadata: Value) -> Self {
        Self {
            extension: "wav".into(),
            mime: "audio/wav".into(),
            kind: "audio".into(),
            bytes,
            metadata,
            source_action: "create".into(),
            parent_asset_id: None,
            relation_type: None,
            duration_seconds: Some(frames as f64 / 25.0),
            width: None,
            height: None,
            sidecars: Vec::new(),
        }
        .with_sample_rate(sample_rate)
    }

    fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.metadata["sample_rate"] = json!(sample_rate);
        self
    }

    pub fn mp4(
        bytes: Vec<u8>,
        width: i64,
        height: i64,
        duration_seconds: f64,
        metadata: Value,
    ) -> Self {
        Self {
            extension: "mp4".into(),
            mime: "video/mp4".into(),
            kind: "video".into(),
            bytes,
            metadata,
            source_action: "create".into(),
            parent_asset_id: None,
            relation_type: None,
            duration_seconds: Some(duration_seconds),
            width: Some(width),
            height: Some(height),
            sidecars: Vec::new(),
        }
    }
}

#[derive(Debug, Default)]
pub struct EngineOutput {
    pub family: String,
    pub action: String,
    pub steps: u32,
    pub graph_computes: u32,
    pub latent_len: usize,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub sample_rate: Option<u32>,
    pub frames: Option<i64>,
    pub metadata: Value,
    pub timings: Vec<(String, f64)>,
    pub artifacts: Vec<Artifact>,
}

#[derive(Debug)]
pub struct EngineResult {
    pub output: EngineOutput,
    pub asset_ids: Vec<String>,
    pub primary_asset_id: Option<String>,
    pub output_paths: Vec<PathBuf>,
}

impl EngineResult {
    pub fn task_json(&self) -> Value {
        let mut value = json!({
            "family": self.output.family,
            "action": self.output.action,
            "steps": self.output.steps,
            "graph_computes": self.output.graph_computes,
            "latent_len": self.output.latent_len,
            "width": self.output.width,
            "height": self.output.height,
            "sample_rate": self.output.sample_rate,
            "frames": self.output.frames,
            "asset_ids": self.asset_ids,
            "primary_asset_id": self.primary_asset_id,
            "output_paths": self.output_paths,
            "timings": self.output.timings,
        });
        if let Some(object) = value.as_object_mut() {
            if let Some(metadata) = self.output.metadata.as_object() {
                for (key, item) in metadata {
                    object.insert(key.clone(), item.clone());
                }
            }
        }
        value
    }
}
