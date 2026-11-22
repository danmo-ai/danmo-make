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

    /// Attach structured key/value tags to the open Trace (root + later phases).
    /// Default is a no-op so unit tests and lightweight hosts stay unchanged.
    fn annotate(&self, _attributes: &[(&str, String)]) -> Result<(), EngineError> {
        Ok(())
    }

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

    pub fn annotate(&self, attributes: &[(&str, String)]) -> Result<(), EngineError> {
        self.observer.annotate(attributes)
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
    pub title: String,
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
    pub title: String,
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
    VideoAvatar(danmo_contracts::VideoAvatarRequest),
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
            Self::VideoAvatar(request) => &request.model,
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
            Self::VideoAvatar(_) => Action::VideoAvatar,
            Self::AudioCreate(_) => Action::AudioCreate,
            Self::AudioEdit(_) => Action::AudioEdit,
            Self::AudioLm(_) => Action::AudioLm,
            Self::Unsupported { action, .. } => *action,
        }
    }

    /// Family-neutral Trace tags derived from the typed request (seed, size,
    /// hung LoRAs, scheduler, …). Prompt text is truncated for span tags.
    pub fn trace_attributes(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        match self {
            Self::ImageCreate(request) => {
                push_prompt(&mut out, &request.prompt);
                push_optional_text(
                    &mut out,
                    "danmo.generation.negative_prompt",
                    &request.negative_prompt,
                );
                push_size(&mut out, &request.size);
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                push_opt_f32(&mut out, "danmo.generation.guidance", request.guidance);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                push_optional_text(
                    &mut out,
                    "danmo.generation.scheduler",
                    request.scheduler.as_deref().unwrap_or(""),
                );
                push_opt_f32(
                    &mut out,
                    "danmo.generation.scheduler_shift",
                    request.scheduler_shift,
                );
                out.push(("danmo.generation.n", request.n.to_string()));
                push_adapters(&mut out, &request.adapters);
                push_structural_guide(&mut out, request.structural_guide.as_ref());
            }
            Self::ImageEdit(request) => {
                push_prompt(&mut out, &request.prompt);
                push_optional_text(
                    &mut out,
                    "danmo.generation.negative_prompt",
                    &request.negative_prompt,
                );
                if let Some(size) = request.size.as_deref() {
                    push_size(&mut out, size);
                }
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                push_opt_f32(&mut out, "danmo.generation.guidance", request.guidance);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                push_optional_text(
                    &mut out,
                    "danmo.generation.scheduler",
                    request.scheduler.as_deref().unwrap_or(""),
                );
                push_opt_f32(
                    &mut out,
                    "danmo.generation.source_fidelity",
                    request.source_fidelity,
                );
                out.push(("danmo.generation.n", request.n.to_string()));
                push_adapters(&mut out, &request.adapters);
            }
            Self::VideoCreate(request) => {
                push_prompt(&mut out, &request.prompt);
                push_optional_text(
                    &mut out,
                    "danmo.generation.negative_prompt",
                    &request.negative_prompt,
                );
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                push_opt_f32(&mut out, "danmo.generation.guidance", request.guidance);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                push_opt_f32(&mut out, "danmo.generation.shift", request.shift);
                if let Some(frames) = request.num_frames {
                    out.push(("danmo.generation.frames", frames.to_string()));
                }
                if let Some(fps) = request.fps {
                    out.push(("danmo.generation.fps", fps.to_string()));
                }
                if let Some(size) = request.size.as_deref() {
                    push_size(&mut out, size);
                }
                if !request.media_references.is_empty() {
                    out.push((
                        "danmo.generation.media_references",
                        request.media_references.len().to_string(),
                    ));
                }
                push_adapters(&mut out, &request.adapters);
            }
            Self::VideoEdit(request) => {
                push_prompt(&mut out, &request.prompt);
                push_optional_text(
                    &mut out,
                    "danmo.generation.negative_prompt",
                    &request.negative_prompt,
                );
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                push_opt_f32(&mut out, "danmo.generation.guidance", request.guidance);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                push_opt_f32(&mut out, "danmo.generation.shift", request.shift);
                if let Some(frames) = request.num_frames {
                    out.push(("danmo.generation.frames", frames.to_string()));
                }
                if let Some(fps) = request.fps {
                    out.push(("danmo.generation.fps", fps.to_string()));
                }
                if let Some(size) = request.size.as_deref() {
                    push_size(&mut out, size);
                }
                push_optional_text(
                    &mut out,
                    "danmo.generation.source_asset_id",
                    &request.source_asset_id,
                );
            }
            Self::AudioCreate(request) => {
                push_prompt(&mut out, &request.prompt);
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                out.push(("danmo.generation.guidance", request.guidance.to_string()));
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                out.push((
                    "danmo.generation.duration_sec",
                    request.duration_sec.to_string(),
                ));
                out.push(("danmo.generation.n", request.n.to_string()));
                push_optional_text(
                    &mut out,
                    "danmo.generation.vocal_language",
                    &request.vocal_language,
                );
                out.push((
                    "danmo.generation.instrumental",
                    request.instrumental.to_string(),
                ));
                push_adapters(&mut out, &request.adapters);
            }
            Self::AudioEdit(request) => {
                push_prompt(&mut out, &request.prompt);
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                out.push(("danmo.generation.guidance", request.guidance.to_string()));
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                if let Some(duration) = request.duration_sec {
                    out.push(("danmo.generation.duration_sec", duration.to_string()));
                }
                out.push(("danmo.generation.n", request.n.to_string()));
                push_adapters(&mut out, &request.adapters);
            }
            Self::ImageUpscale(request) => {
                push_optional_text(
                    &mut out,
                    "danmo.generation.source_asset_id",
                    &request.source_asset_id,
                );
                push_opt_u32(&mut out, "danmo.generation.scale", request.scale);
                push_opt_u32(&mut out, "danmo.generation.tile_size", request.tile_size);
                push_opt_f32(&mut out, "danmo.generation.denoise", request.denoise);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
            }
            Self::VideoUpscale(request) => {
                push_optional_text(
                    &mut out,
                    "danmo.generation.source_asset_id",
                    &request.source_asset_id,
                );
                push_opt_u32(&mut out, "danmo.generation.scale", request.scale);
                push_opt_f32(&mut out, "danmo.generation.denoise", request.denoise);
                push_opt_u32(
                    &mut out,
                    "danmo.generation.temporal_window",
                    request.temporal_window,
                );
                push_opt_u32(&mut out, "danmo.generation.max_frames", request.max_frames);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
            }
            Self::VideoAvatar(request) => {
                push_prompt(&mut out, &request.prompt);
                push_optional_text(
                    &mut out,
                    "danmo.generation.reference_asset_id",
                    &request.reference_asset_id,
                );
                push_opt_u32(&mut out, "danmo.generation.steps", request.steps);
                push_opt_i64(&mut out, "gen_ai.request.seed", request.seed);
                if let Some(frames) = request.num_frames {
                    out.push(("danmo.generation.frames", frames.to_string()));
                }
                if let Some(size) = request.size.as_deref() {
                    push_size(&mut out, size);
                }
            }
            Self::AudioLm(request) => {
                push_prompt(&mut out, &request.prompt);
                push_optional_text(&mut out, "danmo.generation.mode", &request.mode);
            }
            Self::Unsupported { .. } => {}
        }
        out
    }
}

fn push_size(out: &mut Vec<(&'static str, String)>, size: &str) {
    let trimmed = size.trim();
    if trimmed.is_empty() {
        return;
    }
    out.push(("danmo.generation.size", trimmed.to_string()));
    if let Some((width, height)) = trimmed.split_once(['x', '×', '*']) {
        if let (Ok(w), Ok(h)) = (width.trim().parse::<i64>(), height.trim().parse::<i64>()) {
            out.push(("danmo.generation.width", w.to_string()));
            out.push(("danmo.generation.height", h.to_string()));
        }
    }
}

fn push_opt_u32(out: &mut Vec<(&'static str, String)>, key: &'static str, value: Option<u32>) {
    if let Some(value) = value {
        out.push((key, value.to_string()));
    }
}

fn push_opt_f32(out: &mut Vec<(&'static str, String)>, key: &'static str, value: Option<f32>) {
    if let Some(value) = value {
        out.push((key, value.to_string()));
    }
}

fn push_opt_i64(out: &mut Vec<(&'static str, String)>, key: &'static str, value: Option<i64>) {
    if let Some(value) = value {
        out.push((key, value.to_string()));
    }
}

const TRACE_TEXT_LIMIT: usize = 240;

fn push_prompt(out: &mut Vec<(&'static str, String)>, prompt: &str) {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return;
    }
    out.push((
        "danmo.generation.prompt_chars",
        trimmed.chars().count().to_string(),
    ));
    out.push(("gen_ai.prompt", truncate_chars(trimmed, TRACE_TEXT_LIMIT)));
}

fn push_optional_text(out: &mut Vec<(&'static str, String)>, key: &'static str, value: &str) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return;
    }
    out.push((key, truncate_chars(trimmed, TRACE_TEXT_LIMIT)));
}

fn push_adapters(
    out: &mut Vec<(&'static str, String)>,
    adapters: &[danmo_contracts::ImageAdapter],
) {
    out.push(("danmo.lora.count", adapters.len().to_string()));
    if adapters.is_empty() {
        return;
    }
    let ids = adapters
        .iter()
        .map(|adapter| adapter.id.trim())
        .filter(|id| !id.is_empty())
        .collect::<Vec<_>>()
        .join(",");
    let weights = adapters
        .iter()
        .map(|adapter| format_adapter_weight(adapter.weight))
        .collect::<Vec<_>>()
        .join(",");
    let stacked = adapters
        .iter()
        .filter(|adapter| !adapter.id.trim().is_empty())
        .map(|adapter| {
            format!(
                "{}@{}",
                adapter.id.trim(),
                format_adapter_weight(adapter.weight)
            )
        })
        .collect::<Vec<_>>()
        .join(";");
    if !ids.is_empty() {
        out.push(("danmo.lora.ids", ids));
    }
    if !weights.is_empty() {
        out.push(("danmo.lora.weights", weights));
    }
    if !stacked.is_empty() {
        out.push(("danmo.lora", stacked));
    }
}

fn push_structural_guide(
    out: &mut Vec<(&'static str, String)>,
    guide: Option<&danmo_contracts::StructuralGuide>,
) {
    let Some(guide) = guide else {
        return;
    };
    out.push(("danmo.structural_guide", "true".into()));
    push_optional_text(out, "danmo.structural_guide.model_id", &guide.model_id);
    push_optional_text(out, "danmo.structural_guide.type", &guide.guide_type);
    out.push(("danmo.structural_guide.weight", guide.weight.to_string()));
    if !guide.asset_id.trim().is_empty() {
        out.push((
            "danmo.structural_guide.asset_id",
            guide.asset_id.trim().to_string(),
        ));
    }
}

fn format_adapter_weight(weight: f32) -> String {
    if (weight - weight.round()).abs() < 1e-4 {
        format!("{}", weight.round() as i32)
    } else {
        format!("{weight:.3}")
    }
}

fn truncate_chars(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let truncated: String = text.chars().take(limit).collect();
    format!("{truncated}…(+{})", count - limit)
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
