//! Fixed phase order: resolve → load → encode → schedule → infer → decode → persist.

pub(crate) mod denoise;

use crate::families::{self, Media, Shape};
use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{self, LoraFile, WeightError};
use danmo_contracts::{parse_size, ImageGenerationRequest};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error(transparent)]
    Weights(#[from] WeightError),
    #[error("family {0} is not registered")]
    UnknownFamily(String),
    #[error("{family} has no ggml graph yet (shape {shape})")]
    GraphNotBuilt { family: String, shape: String },
    #[error("{0}")]
    BadRequest(String),
    #[error(transparent)]
    Runtime(#[from] crate::runtime::RuntimeError),
}

pub struct RunOutput {
    pub family: String,
    pub shape: Shape,
    pub latents: Vec<f32>,
    pub steps: u32,
    pub graph_computes: u32,
    pub width: u32,
    pub height: u32,
    pub preview_png: Vec<u8>,
}

fn request_loras(request: &ImageGenerationRequest) -> Result<Vec<LoraFile>, EngineError> {
    let mut files = Vec::new();
    for adapter in &request.adapters {
        if adapter.id.trim().is_empty() {
            return Err(EngineError::BadRequest("lora id is required".into()));
        }
        let path = std::path::PathBuf::from(&adapter.path);
        if adapter.path.is_empty() || !path.is_file() {
            return Err(EngineError::BadRequest(format!(
                "lora {} is not installed",
                adapter.id
            )));
        }
        if !adapter.weight.is_finite() {
            return Err(EngineError::BadRequest(format!(
                "lora {} weight is not a finite number",
                adapter.id
            )));
        }
        files.push(LoraFile {
            id: adapter.id.clone(),
            path,
            weight: adapter.weight,
        });
    }
    Ok(files)
}

fn shape_name(shape: Shape) -> &'static str {
    match shape {
        Shape::DitPipeline => "dit_pipeline",
        Shape::JobPipeline => "job_pipeline",
        Shape::GenerationFacade => "generation_facade",
    }
}

/// Image create. A family with a ggml graph runs that denoise graph. Every
/// other registered family still walks the same phases and then fails loud.
pub fn image_create(
    request: &ImageGenerationRequest,
    family: &str,
    bundle: &Path,
    backend: BackendKind,
    on_progress: impl FnMut(u32, u32),
) -> Result<RunOutput, EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != Media::Image {
        return Err(EngineError::BadRequest(format!(
            "{family} is not an image family"
        )));
    }
    let (width, height) = parse_size(&request.size).map_err(EngineError::BadRequest)?;
    let loras = request_loras(request)?;
    if !loras.is_empty() && !spec.lora_inference {
        return Err(EngineError::BadRequest(
            "lora inference graph is not built".into(),
        ));
    }
    weights::assert_gguf_bundle(bundle)?;
    let steps = request.steps.unwrap_or(4);
    match spec.shape {
        Shape::DitPipeline => {
            let ctx = GgmlContext::new(backend)?;
            let Some(done) = families::run_dit_image(
                family,
                &ctx,
                bundle,
                &request.prompt,
                steps,
                width,
                height,
                request.guidance.unwrap_or(1.0),
                request.seed,
                request.scheduler.as_deref(),
                request.scheduler_shift,
                request.use_empirical_mu,
                &request.negative_prompt,
                &loras,
                on_progress,
            )?
            else {
                return Err(EngineError::GraphNotBuilt {
                    family: family.into(),
                    shape: shape_name(spec.shape).into(),
                });
            };
            Ok(RunOutput {
                family: family.into(),
                shape: spec.shape,
                latents: done.latents,
                steps,
                graph_computes: done.graph_computes,
                width,
                height,
                preview_png: done.preview_png,
            })
        }
        shape => Err(EngineError::GraphNotBuilt {
            family: family.into(),
            shape: shape_name(shape).into(),
        }),
    }
}

pub fn video_create(family: &str, bundle: &Path) -> Result<(), EngineError> {
    route_media(family, bundle, Media::Video)
}

pub struct AudioCreateRequest<'a> {
    pub prompt: &'a str,
    pub lyrics: &'a str,
    pub duration_sec: f32,
    pub seed: Option<i64>,
    pub guidance: f32,
    pub steps: Option<u32>,
    pub vocal_language: &'a str,
    /// Optional 5Hz LM FSQ codebook indices for DiT context.
    pub audio_code_indices: Option<&'a [i32]>,
    /// Optional time-major `[T*64]` latents for the timbre encoder.
    pub timbre_latents: Option<&'a [f32]>,
}

pub struct AudioEditRequest<'a> {
    pub prompt: &'a str,
    pub lyrics: &'a str,
    pub seed: Option<i64>,
    pub guidance: f32,
    pub steps: Option<u32>,
    pub vocal_language: &'a str,
    /// `cover-nofsq` | `cover` | `repaint` | `lego` | `extract` | `complete`.
    pub task: &'a str,
    pub source_audio: Option<&'a [f32]>,
    pub source_latents: Option<&'a [f32]>,
    pub audio_cover_strength: f32,
    pub cover_noise_strength: f32,
    pub region_start_sec: Option<f32>,
    pub region_end_sec: Option<f32>,
    pub repaint_injection_ratio: f32,
    pub repaint_crossfade_frames: i64,
    /// Sample 750-frame timbre reference from source latents.
    pub timbre_from_source: bool,
    /// Track name(s) for lego / extract / complete.
    pub track: &'a str,
}

/// ACE-Step 5Hz LM inspire / format / generate (phase-1).
pub struct AudioLmRequest<'a> {
    pub mode: &'a str,
    pub prompt: &'a str,
    pub lyrics: &'a str,
    pub duration: f32,
    pub bpm: i32,
    pub keyscale: &'a str,
    pub timesignature: &'a str,
    pub vocal_language: &'a str,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u64,
    pub use_fsm: bool,
}

#[derive(Debug)]
pub struct AudioLmOutput {
    pub family: String,
    pub mode: String,
    pub prompt_tokens: usize,
    pub generated_ids: Vec<i32>,
    pub text: String,
    pub caption: String,
    pub lyrics: String,
    pub bpm: Option<i32>,
    pub duration: Option<i32>,
    pub keyscale: String,
    pub timesignature: String,
    pub language: String,
    pub audio_code_indices: Vec<i32>,
}

#[derive(Debug)]
pub struct AudioRunOutput {
    pub family: String,
    pub wav: Vec<u8>,
    pub sample_rate: u32,
    pub frames: i64,
    pub steps: u32,
    pub graph_computes: u32,
    pub lyrics_effective: String,
    pub lyrics_alignment_json: String,
    pub lyrics_txt: Option<String>,
    pub lyrics_lrc: Option<String>,
}

fn audio_out_from_create(
    family: &str,
    out: families::ace_step::CreateOutput,
) -> AudioRunOutput {
    let lyrics_alignment_json =
        serde_json::to_string(&out.lyrics_alignment.as_metadata()).unwrap_or_else(|_| {
            "{\"lyrics_alignment\":\"unknown\"}".into()
        });
    AudioRunOutput {
        family: family.into(),
        wav: out.wav,
        sample_rate: out.sample_rate,
        frames: out.frames,
        steps: out.steps,
        graph_computes: out.graph_computes,
        lyrics_effective: out.lyrics_effective,
        lyrics_alignment_json,
        lyrics_txt: out.lyrics_txt,
        lyrics_lrc: out.lyrics_lrc,
    }
}

pub fn audio_create(
    family: &str,
    bundle: &Path,
    request: &AudioCreateRequest<'_>,
) -> Result<AudioRunOutput, EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != Media::Audio {
        return Err(EngineError::BadRequest(format!(
            "{family} does not match the requested media"
        )));
    }
    let create = families::ace_step::CreateRequest {
        prompt: request.prompt,
        lyrics: request.lyrics,
        duration_sec: request.duration_sec,
        seed: request.seed,
        guidance: request.guidance,
        steps: request.steps,
        vocal_language: request.vocal_language,
        bpm: None,
        keyscale: "",
        timesignature: "",
        audio_code_indices: request.audio_code_indices,
        timbre_latents: request.timbre_latents,
    };
    if let Some(out) = families::run_audio_facade(family, bundle, &create)? {
        return Ok(audio_out_from_create(family, out));
    }
    weights::assert_gguf_bundle(bundle)?;
    Err(EngineError::GraphNotBuilt {
        family: family.into(),
        shape: shape_name(spec.shape).into(),
    })
}

/// Decode a PCM16 WAV to interleaved stereo f32 (ACE-Step cover source).
pub fn read_audio_wav_pcm16(bytes: &[u8]) -> Result<(Vec<f32>, u32), EngineError> {
    families::ace_step_read_wav(bytes).map_err(EngineError::from)
}

pub fn audio_edit(
    family: &str,
    bundle: &Path,
    request: &AudioEditRequest<'_>,
) -> Result<AudioRunOutput, EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != Media::Audio {
        return Err(EngineError::BadRequest(format!(
            "{family} does not match the requested media"
        )));
    }
    let mode = families::ace_step::CoverMode::parse(request.task)?;
    let cover = families::ace_step::CoverRequest {
        prompt: request.prompt,
        lyrics: request.lyrics,
        seed: request.seed,
        guidance: request.guidance,
        steps: request.steps,
        vocal_language: request.vocal_language,
        bpm: None,
        keyscale: "",
        timesignature: "",
        mode,
        source_audio: request.source_audio,
        source_latents: request.source_latents,
        audio_cover_strength: request.audio_cover_strength,
        cover_noise_strength: request.cover_noise_strength,
        region_start_sec: request.region_start_sec,
        region_end_sec: request.region_end_sec,
        repaint_injection_ratio: request.repaint_injection_ratio,
        repaint_crossfade_frames: request.repaint_crossfade_frames,
        timbre_from_source: request.timbre_from_source,
        track: request.track,
    };
    if let Some(out) = families::run_audio_edit_facade(family, bundle, &cover)? {
        return Ok(audio_out_from_create(family, out));
    }
    weights::assert_gguf_bundle(bundle)?;
    Err(EngineError::GraphNotBuilt {
        family: family.into(),
        shape: shape_name(spec.shape).into(),
    })
}

/// ACE-Step 5Hz LM (inspire / format / generate phase-1).
pub fn audio_lm(
    family: &str,
    bundle: &Path,
    request: &AudioLmRequest<'_>,
) -> Result<AudioLmOutput, EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != Media::Audio {
        return Err(EngineError::BadRequest(format!(
            "{family} does not match the requested media"
        )));
    }
    let mode = families::ace_step::LmMode::parse(request.mode)?;
    let lm = families::ace_step::LmRequest {
        mode,
        caption: request.prompt,
        lyrics: request.lyrics,
        duration: request.duration,
        bpm: request.bpm,
        keyscale: request.keyscale,
        timesignature: request.timesignature,
        vocal_language: request.vocal_language,
        max_new_tokens: request.max_new_tokens.max(1),
        tokenizer_json: None,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        seed: request.seed,
        use_fsm: request.use_fsm,
    };
    if let Some(out) = families::run_audio_lm_facade(family, bundle, &lm)? {
        return Ok(AudioLmOutput {
            family: family.into(),
            mode: out.mode,
            prompt_tokens: out.prompt_tokens,
            generated_ids: out.generated_ids,
            text: out.text,
            caption: out.caption,
            lyrics: out.lyrics,
            bpm: out.bpm,
            duration: out.duration,
            keyscale: out.keyscale,
            timesignature: out.timesignature,
            language: out.language,
            audio_code_indices: out.audio_code_indices,
        });
    }
    Err(EngineError::BadRequest(format!(
        "{family} has no 5Hz LM path"
    )))
}

pub fn upscale(family: &str, bundle: &Path) -> Result<(), EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.shape != Shape::JobPipeline {
        return Err(EngineError::BadRequest(format!(
            "{family} is not an upscale job"
        )));
    }
    weights::assert_gguf_bundle(bundle)?;
    Err(EngineError::GraphNotBuilt {
        family: family.into(),
        shape: shape_name(spec.shape).into(),
    })
}

fn route_media(family: &str, bundle: &Path, media: Media) -> Result<(), EngineError> {
    let spec = families::spec(family).ok_or_else(|| EngineError::UnknownFamily(family.into()))?;
    if spec.media != media {
        return Err(EngineError::BadRequest(format!(
            "{family} does not match the requested media"
        )));
    }
    weights::assert_gguf_bundle(bundle)?;
    Err(EngineError::GraphNotBuilt {
        family: family.into(),
        shape: shape_name(spec.shape).into(),
    })
}
