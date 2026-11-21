//! LTX 2.5 text-to-video application session.
//!
//! The facade owns component lifetime, the fixed two-stage orchestration and
//! final A/V mux. Numerical sampling stays in `denoise`, model topology stays
//! in `transformer`/codec modules, and neutral geometry/encoding stays common.

use super::config::{BundleConfig, Quantization};
use crate::contract::{Artifact, EngineOutput, EngineRequest, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::families::{FacadeFamily, FacadeSession};
use crate::ports::ResolvePort;
use crate::runtime::GgmlContext;
use crate::weights::WeightError;
use danmo_contracts::{VideoEditRequest, VideoGenerationRequest};
use serde_json::json;

const DEFAULT_SIZE: &str = "1024x768";
const DEFAULT_FRAMES: u32 = 121;
const DEFAULT_FPS: u32 = 24;

const MAX_REFERENCE_IMAGES: usize = 5;
const MAX_REFERENCE_AUDIOS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunMode {
    Create,
    Animate,
}

#[derive(Clone)]
struct Prepared {
    prompt: String,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    seed: i64,
    mode: RunMode,
    /// Animate ImageReplace still (frame 0). Mutually exclusive with Ref2VA.
    first_frame: Option<ReferenceStill>,
    /// Ref2VA stills in submit order (pic1..picN, optional background last).
    reference_images: Vec<ReferenceStill>,
    /// Paired audio clips (slot_id 1..=2 ↔ pic1/pic2).
    reference_audios: Vec<ReferenceAudio>,
    /// Resolved MSR / style LoRA files (paths filled by the API host).
    loras: Vec<crate::weights::LoraFile>,
}

#[derive(Clone)]
struct ReferenceStill {
    rgb: Vec<u8>,
    width: u32,
    height: u32,
}

#[derive(Clone)]
struct ReferenceAudio {
    /// Interleaved stereo PCM.
    pcm: Vec<f32>,
    sample_rate: u32,
    /// 1-based subject slot (pairs with picN).
    slot_id: usize,
}

pub struct Ltx25VideoCreateFamily;

struct Ltx25VideoCreateSession {
    run: ResolvedRun,
    prepared: Prepared,
    config: BundleConfig,
    quantization: Option<Quantization>,
}

fn prepare_params(request: &VideoGenerationRequest) -> Result<Prepared, EngineError> {
    let mut image_refs = 0usize;
    let mut audio_refs = 0usize;
    for reference in &request.media_references {
        let kind = reference.kind.trim().to_ascii_lowercase();
        match kind.as_str() {
            "image" => image_refs += 1,
            "audio" => audio_refs += 1,
            other => {
                return Err(EngineError::bad_request(format!(
                    "LTX 2.5 Ref2VA accepts image/audio references only (got kind={other})"
                )));
            }
        }
        if reference.asset_id.trim().is_empty() {
            return Err(EngineError::bad_request(
                "LTX 2.5 media_references entries need a non-empty asset_id",
            ));
        }
    }
    if image_refs > MAX_REFERENCE_IMAGES {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 Ref2VA accepts at most {MAX_REFERENCE_IMAGES} reference images (got {image_refs})"
        )));
    }
    if audio_refs > MAX_REFERENCE_AUDIOS {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 Ref2VA accepts at most {MAX_REFERENCE_AUDIOS} reference audios (got {audio_refs})"
        )));
    }
    let subject_slots = subject_slot_count(image_refs);
    if audio_refs > 0 && image_refs == 0 {
        return Err(EngineError::bad_request(
            "LTX 2.5 Ref2VA audio references require at least one subject image",
        ));
    }
    if audio_refs > subject_slots {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 Ref2VA audio references need matching subject images (audios={audio_refs}, subjects={subject_slots})"
        )));
    }
    if (image_refs > 0 || audio_refs > 0) && request.adapters.is_empty() {
        return Err(EngineError::bad_request(
            "LTX 2.5 Ref2VA requires the multi-subject reference LoRA (ltx25-msr-lora)",
        ));
    }
    let loras = crate::pipeline::lora_files(&request.adapters)?;
    if !request.negative_prompt.trim().is_empty() {
        return Err(EngineError::bad_request(
            "LTX 2.5 distilled does not use a negative prompt",
        ));
    }
    if let Some(guidance) = request.guidance {
        if guidance != 0.0 && guidance != 1.0 {
            return Err(EngineError::bad_request(format!(
                "LTX 2.5 distilled fixes guidance at 1.0 (got {guidance})"
            )));
        }
    }
    let size = request.size.as_deref().unwrap_or(DEFAULT_SIZE);
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    if width < 64 || height < 64 || width % 64 != 0 || height % 64 != 0 {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 requires width and height to be positive multiples of 64; got {width}x{height}"
        )));
    }
    let fps = request.fps.unwrap_or(DEFAULT_FPS);
    let frames = super::prepare_distilled(super::DistilledRequest {
        width,
        height,
        num_frames: request.num_frames.unwrap_or(DEFAULT_FRAMES),
        fps: fps as f32,
        steps: request.steps.unwrap_or(8),
        guidance: request.guidance.unwrap_or(1.0),
        step_distill: true,
        negative_prompt: &request.negative_prompt,
    })?
    .num_frames;
    if (frames - 1) as f32 / fps as f32 > 20.0 + 1e-6 {
        return Err(EngineError::bad_request(
            "LTX 2.5 positional time axis is limited to 20 seconds",
        ));
    }
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(EngineError::bad_request("LTX 2.5 prompt must not be empty"));
    }
    Ok(Prepared {
        prompt: prompt.to_string(),
        width,
        height,
        frames,
        fps,
        seed: request.seed.unwrap_or(42),
        mode: RunMode::Create,
        first_frame: None,
        reference_images: Vec::new(),
        reference_audios: Vec::new(),
        loras,
    })
}

fn prepare_animate_params(request: &VideoEditRequest) -> Result<Prepared, EngineError> {
    if request.source_asset_id.trim().is_empty() {
        return Err(EngineError::bad_request(
            "LTX 2.5 animate requires source_asset_id (first frame)",
        ));
    }
    if !request.negative_prompt.trim().is_empty() {
        return Err(EngineError::bad_request(
            "LTX 2.5 distilled does not use a negative prompt",
        ));
    }
    if let Some(guidance) = request.guidance {
        if guidance != 0.0 && guidance != 1.0 {
            return Err(EngineError::bad_request(format!(
                "LTX 2.5 distilled fixes guidance at 1.0 (got {guidance})"
            )));
        }
    }
    let size = request.size.as_deref().unwrap_or(DEFAULT_SIZE);
    let (width, height) = danmo_contracts::parse_size(size).map_err(EngineError::InvalidRequest)?;
    if width < 64 || height < 64 || width % 64 != 0 || height % 64 != 0 {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 requires width and height to be positive multiples of 64; got {width}x{height}"
        )));
    }
    let fps = request.fps.unwrap_or(DEFAULT_FPS);
    let frames = super::prepare_distilled(super::DistilledRequest {
        width,
        height,
        num_frames: request.num_frames.unwrap_or(DEFAULT_FRAMES),
        fps: fps as f32,
        steps: request.steps.unwrap_or(8),
        guidance: request.guidance.unwrap_or(1.0),
        step_distill: true,
        negative_prompt: &request.negative_prompt,
    })?
    .num_frames;
    if (frames - 1) as f32 / fps as f32 > 20.0 + 1e-6 {
        return Err(EngineError::bad_request(
            "LTX 2.5 positional time axis is limited to 20 seconds",
        ));
    }
    let prompt = request.prompt.trim();
    if prompt.is_empty() {
        return Err(EngineError::bad_request("LTX 2.5 prompt must not be empty"));
    }
    Ok(Prepared {
        prompt: prompt.to_string(),
        width,
        height,
        frames,
        fps,
        seed: request.seed.unwrap_or(42),
        mode: RunMode::Animate,
        first_frame: None,
        reference_images: Vec::new(),
        reference_audios: Vec::new(),
        loras: Vec::new(),
    })
}

/// Subject slots available for paired audio (last still is background at max images).
fn subject_slot_count(image_count: usize) -> usize {
    if image_count == 0 {
        return 0;
    }
    if image_count >= MAX_REFERENCE_IMAGES {
        return MAX_REFERENCE_IMAGES - 1;
    }
    image_count
}

#[cfg(test)]
mod prepare_tests {
    use super::{prepare_animate_params, prepare_params, RunMode};
    use danmo_contracts::{ImageAdapter, MediaReference, VideoEditRequest, VideoGenerationRequest};
    use serde_json::json;

    fn base_request() -> VideoGenerationRequest {
        VideoGenerationRequest {
            model: "ltx-2.5-distilled".into(),
            prompt: "a red cube".into(),
            negative_prompt: String::new(),
            size: Some("640x640".into()),
            steps: Some(8),
            guidance: Some(1.0),
            shift: None,
            seed: Some(1),
            num_frames: Some(25),
            fps: Some(24),
            media_references: Vec::new(),
            adapters: Vec::new(),
            metadata: json!({}),
            priority: "normal".into(),
        }
    }

    #[test]
    fn pure_t2v_prepares() {
        let prepared = prepare_params(&base_request()).unwrap();
        assert_eq!(
            (prepared.width, prepared.height, prepared.frames),
            (640, 640, 25)
        );
        assert_eq!(prepared.mode, RunMode::Create);
        assert!(prepared.first_frame.is_none());
    }

    #[test]
    fn animate_prepares_with_source_asset() {
        let request = VideoEditRequest {
            model: "ltx-2.5-distilled".into(),
            source_asset_id: "ast_src".into(),
            prompt: "the cube rotates slowly".into(),
            negative_prompt: String::new(),
            size: Some("640x640".into()),
            steps: Some(8),
            guidance: Some(1.0),
            shift: None,
            seed: Some(7),
            num_frames: Some(25),
            fps: Some(24),
            metadata: json!({}),
            priority: "normal".into(),
        };
        let prepared = prepare_animate_params(&request).unwrap();
        assert_eq!(prepared.mode, RunMode::Animate);
        assert_eq!(
            (prepared.width, prepared.height, prepared.frames),
            (640, 640, 25)
        );
        assert!(prepared.loras.is_empty());
    }

    #[test]
    fn animate_without_source_fails_loud() {
        let request = VideoEditRequest {
            model: "ltx-2.5-distilled".into(),
            source_asset_id: "  ".into(),
            prompt: "move".into(),
            negative_prompt: String::new(),
            size: Some("640x640".into()),
            steps: None,
            guidance: None,
            shift: None,
            seed: None,
            num_frames: None,
            fps: None,
            metadata: json!({}),
            priority: "normal".into(),
        };
        let error = prepare_animate_params(&request).err().unwrap();
        assert!(error.to_string().contains("source_asset_id"), "{error}");
    }

    #[test]
    fn ref2va_without_msr_lora_fails_loud() {
        let mut request = base_request();
        request.media_references.push(MediaReference {
            kind: "image".into(),
            asset_id: "ast_ref".into(),
        });
        let error = prepare_params(&request).err().unwrap();
        assert!(error.to_string().contains("ltx25-msr-lora"), "{error}");
    }

    #[test]
    fn audio_without_image_fails_loud() {
        let mut request = base_request();
        request.media_references.push(MediaReference {
            kind: "audio".into(),
            asset_id: "ast_a".into(),
        });
        request.adapters.push(ImageAdapter {
            id: "ltx25-msr-lora".into(),
            weight: 1.0,
            path: "/tmp/missing-msr.safetensors".into(),
        });
        assert!(prepare_params(&request)
            .err()
            .unwrap()
            .to_string()
            .contains("require at least one subject image"));
    }

    #[test]
    fn video_refs_fail_loud() {
        let mut request = base_request();
        request.media_references.push(MediaReference {
            kind: "video".into(),
            asset_id: "ast_v".into(),
        });
        assert!(prepare_params(&request)
            .err()
            .unwrap()
            .to_string()
            .contains("image/audio"));
    }

    #[test]
    fn adapters_without_installed_path_fail_loud() {
        let mut request = base_request();
        request.adapters.push(ImageAdapter {
            id: "ltx25-msr-lora".into(),
            weight: 1.0,
            path: String::new(),
        });
        let error = prepare_params(&request).err().unwrap();
        assert!(error.to_string().contains("not installed"), "{error}");
    }
}

fn quantization(version: Option<&str>) -> Result<Option<Quantization>, EngineError> {
    match version.unwrap_or("mlx-q4") {
        "mlx-q4" => Ok(Some(Quantization {
            bits: 4,
            group_size: 64,
        })),
        "mlx-q8" => Ok(Some(Quantization {
            bits: 8,
            group_size: 64,
        })),
        "bf16" | "mlx-bf16" => Ok(None),
        other => Err(EngineError::bad_request(format!(
            "unsupported LTX 2.5 MLX version {other}"
        ))),
    }
}

impl FacadeFamily for Ltx25VideoCreateFamily {
    fn prepare(
        &self,
        _ctx: &GgmlContext,
        run: &ResolvedRun,
        request: &EngineRequest,
        resolver: &dyn ResolvePort,
        exec: &ExecutionContext,
    ) -> Result<Box<dyn FacadeSession>, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = (run, request, resolver, exec);
            return Err(WeightError::NotBuilt(
                "LTX 2.5 requires an MLX-enabled danqing-api build".into(),
            )
            .into());
        }
        #[cfg(feature = "mlx")]
        {
            let (mut prepared, steps_override) = match request {
                EngineRequest::VideoCreate(request) => (prepare_params(request)?, request.steps),
                EngineRequest::VideoEdit(request) => {
                    (prepare_animate_params(request)?, request.steps)
                }
                _ => {
                    return Err(EngineError::bad_request(
                        "LTX 2.5 video family accepts video.create or video.edit only",
                    ));
                }
            };
            match prepared.mode {
                RunMode::Create => {
                    let EngineRequest::VideoCreate(request) = request else {
                        return Err(EngineError::bad_request("LTX 2.5 create prepare mismatch"));
                    };
                    let mut next_audio_slot = 1usize;
                    for reference in &request.media_references {
                        let kind = reference.kind.trim().to_ascii_lowercase();
                        match kind.as_str() {
                            "image" => {
                                let (rgb, width, height) =
                                    resolver.read_source_image(run, &reference.asset_id)?;
                                prepared.reference_images.push(ReferenceStill {
                                    rgb,
                                    width,
                                    height,
                                });
                            }
                            "audio" => {
                                let source = format!("asset:{}", reference.asset_id.trim());
                                let bytes = resolver.read_source_audio(run, &source)?;
                                let (pcm, sample_rate) =
                                    crate::common::wav::read_pcm16_interleaved(&bytes)
                                        .map_err(EngineError::bad_request)?;
                                prepared.reference_audios.push(ReferenceAudio {
                                    pcm,
                                    sample_rate,
                                    slot_id: next_audio_slot,
                                });
                                next_audio_slot += 1;
                            }
                            _ => {
                                return Err(EngineError::bad_request(format!(
                                    "LTX 2.5 Ref2VA accepts image/audio references only (got kind={})",
                                    reference.kind
                                )));
                            }
                        }
                    }
                }
                RunMode::Animate => {
                    let EngineRequest::VideoEdit(request) = request else {
                        return Err(EngineError::bad_request("LTX 2.5 animate prepare mismatch"));
                    };
                    let (rgb, width, height) =
                        resolver.read_source_image(run, &request.source_asset_id)?;
                    prepared.first_frame = Some(ReferenceStill { rgb, width, height });
                }
            }
            let config = BundleConfig::load(&run.bundle)?;
            for file in [
                "transformer.safetensors",
                "connector.safetensors",
                "video_vae.safetensors",
                "audio_vae.safetensors",
                "upsampler.safetensors",
                "text_encoder/model.safetensors",
                "text_encoder/config.json",
                "text_encoder/tokenizer.json",
            ] {
                let path = run.bundle.join(file);
                if !path.is_file() {
                    return Err(WeightError::Bundle {
                        path: path.display().to_string(),
                        detail: "required LTX 2.5 component is missing".into(),
                    }
                    .into());
                }
            }
            if steps_override.is_some_and(|steps| steps != super::STAGE_1_STEPS) {
                exec.log(
                    "warning",
                    "LTX 2.5 ignores the requested step count: stage 1 is fixed at 8 steps and stage 2 at 3",
                )?;
            }
            let mode = match prepared.mode {
                RunMode::Animate => "i2v",
                RunMode::Create
                    if prepared.reference_images.is_empty()
                        && prepared.reference_audios.is_empty() =>
                {
                    "t2v"
                }
                RunMode::Create => "ref2va",
            };
            exec.log(
                "info",
                &format!(
                    "ltx25 {mode} {}x{} frames={} fps={} seed={} image_refs={} audio_refs={} first_frame={} stages=8+3",
                    prepared.width,
                    prepared.height,
                    prepared.frames,
                    prepared.fps,
                    prepared.seed,
                    prepared.reference_images.len(),
                    prepared.reference_audios.len(),
                    prepared.first_frame.is_some(),
                ),
            )?;
            Ok(Box::new(Ltx25VideoCreateSession {
                run: run.clone(),
                prepared,
                config,
                quantization: quantization(run.version.as_deref())?,
            }))
        }
    }
}

impl FacadeSession for Ltx25VideoCreateSession {
    fn infer(
        &mut self,
        _ctx: &GgmlContext,
        exec: &ExecutionContext,
    ) -> Result<EngineOutput, EngineError> {
        #[cfg(not(feature = "mlx"))]
        {
            let _ = exec;
            Err(WeightError::NotBuilt("LTX 2.5 requires MLX".into()).into())
        }
        #[cfg(feature = "mlx")]
        {
            infer_mlx(
                &self.run,
                &self.prepared,
                &self.config,
                self.quantization,
                exec,
            )
        }
    }
}

#[cfg(feature = "mlx")]
fn mlx_error(error: danmo_mlx::Error) -> EngineError {
    EngineError::Resolve(error.to_string())
}

#[cfg(feature = "mlx")]
fn io_error(context: &str, error: std::io::Error) -> EngineError {
    EngineError::Persist(format!("{context}: {error}"))
}

#[cfg(feature = "mlx")]
struct StageRope {
    video: crate::common::av_rope::RopeTables,
    audio: crate::common::av_rope::RopeTables,
    video_cross: crate::common::av_rope::RopeTables,
    audio_cross: crate::common::av_rope::RopeTables,
}

/// Build RoPE tables for a denoise stage.
///
/// `reference_time_offsets` has one entry per image reference slot. Each slot
/// contributes a full copy of the target spatial grid with its time axis
/// shifted (MSR negative offsets). `audio_reference_positions` are absolute
/// midpoints prepended before the target audio timeline.
#[cfg(feature = "mlx")]
fn stage_rope(
    config: &super::transformer::TransformerConfig,
    shape: crate::common::av_latent::VideoLatentShape,
    audio_tokens: usize,
    fps: u32,
    reference_time_offsets: &[f32],
    audio_reference_positions: &[f32],
    stream: &danmo_mlx::Stream,
) -> Result<StageRope, EngineError> {
    use crate::common::{av_latent, av_rope};
    let target_positions = av_latent::video_positions(shape, fps as f32, super::VIDEO_LATENT_SPEC)
        .map_err(EngineError::Resolve)?
        .into_iter()
        .map(|position| position.to_vec())
        .collect::<Vec<_>>();
    let mut video_positions =
        Vec::with_capacity(target_positions.len() * (1 + reference_time_offsets.len()));
    for &offset in reference_time_offsets {
        for position in &target_positions {
            video_positions.push(vec![position[0] + offset, position[1], position[2]]);
        }
    }
    video_positions.extend(target_positions);
    let mut audio_positions = Vec::with_capacity(audio_reference_positions.len() + audio_tokens);
    for &position in audio_reference_positions {
        audio_positions.push(vec![position]);
    }
    audio_positions.extend(
        av_latent::audio_positions(audio_tokens, super::AUDIO_LATENT_SPEC)
            .map_err(EngineError::Resolve)?
            .into_iter()
            .map(|position| vec![position]),
    );
    let video_cross_positions = video_positions
        .iter()
        .map(|position| vec![position[0]])
        .collect::<Vec<_>>();
    Ok(StageRope {
        video: av_rope::split_rope_tables(
            &video_positions,
            config.dim,
            config.heads,
            config.rope_theta,
            &config.video_rope_max_pos,
            stream,
        )
        .map_err(mlx_error)?,
        audio: av_rope::split_rope_tables(
            &audio_positions,
            config.audio_dim,
            config.audio_heads,
            config.rope_theta,
            &config.audio_rope_max_pos,
            stream,
        )
        .map_err(mlx_error)?,
        video_cross: av_rope::split_rope_tables(
            &video_cross_positions,
            config.audio_dim,
            config.audio_heads,
            config.rope_theta,
            &[20.0],
            stream,
        )
        .map_err(mlx_error)?,
        audio_cross: av_rope::split_rope_tables(
            &audio_positions,
            config.audio_dim,
            config.audio_heads,
            config.rope_theta,
            &[20.0],
            stream,
        )
        .map_err(mlx_error)?,
    })
}

#[cfg(feature = "mlx")]
#[allow(clippy::too_many_arguments)]
fn denoise_stage(
    model: &super::transformer::Transformer,
    mut video: danmo_mlx::Array,
    mut audio: danmo_mlx::Array,
    prompt: &super::text_encoder::PromptEmbeddings,
    rope: &StageRope,
    sigmas: &[f32],
    ancestral: bool,
    seed: i64,
    progress_offset: u32,
    video_reference: Option<&danmo_mlx::Array>,
    audio_reference: Option<&danmo_mlx::Array>,
    target_first_frame_tokens: usize,
    first_frame: Option<(&danmo_mlx::Array, f32)>,
    exec: &ExecutionContext,
    stream: &danmo_mlx::Stream,
) -> Result<(danmo_mlx::Array, danmo_mlx::Array), EngineError> {
    use danmo_mlx::{Array, Dtype, RandomKey};
    if video_reference.is_some() && first_frame.is_some() {
        return Err(EngineError::bad_request(
            "LTX 2.5 cannot combine Ref2VA prepend references with animate ImageReplace",
        ));
    }
    let target_video_tokens = video.shape()[1] as usize;
    let target_audio_tokens = audio.shape()[1] as usize;
    let reference_tokens = video_reference
        .map(|tokens| tokens.shape()[1] as usize)
        .unwrap_or(0);
    let audio_reference_tokens = audio_reference
        .map(|tokens| tokens.shape()[1] as usize)
        .unwrap_or(0);
    let first_frame_tokens = first_frame
        .map(|(tokens, _)| tokens.shape()[1] as usize)
        .unwrap_or(0);
    let mut key = RandomKey::new(seed as u64).map_err(mlx_error)?;
    for (index, (&sigma, &next)) in sigmas.iter().zip(&sigmas[1..]).enumerate() {
        exec.check_cancelled()?;
        let (video_input, video_sigmas) = if let Some(reference) = video_reference {
            let combined = super::conditioning::prepend_reference_tokens(reference, &video, stream)
                .map_err(mlx_error)?;
            let sigmas = super::conditioning::reference_then_target_sigmas(
                reference_tokens,
                target_video_tokens,
                sigma,
            );
            (combined, Some(sigmas))
        } else if let Some((_, strength)) = first_frame {
            let sigmas = super::conditioning::replace_first_frame_sigmas(
                target_video_tokens,
                first_frame_tokens,
                sigma,
                strength,
            )
            .map_err(mlx_error)?;
            (video.clone(), Some(sigmas))
        } else {
            (video.clone(), None)
        };
        let (audio_input, audio_sigmas) = if let Some(reference) = audio_reference {
            let combined = super::conditioning::prepend_reference_tokens(reference, &audio, stream)
                .map_err(mlx_error)?;
            let sigmas = super::conditioning::reference_then_target_sigmas(
                audio_reference_tokens,
                target_audio_tokens,
                sigma,
            );
            (combined, Some(sigmas))
        } else {
            (audio.clone(), None)
        };
        let (video_velocity, audio_velocity) = model
            .forward_tokens_timed(
                &video_input,
                &audio_input,
                &rope.video,
                &rope.audio,
                &rope.video_cross,
                &rope.audio_cross,
                prompt,
                super::transformer::KeyframeSpan {
                    start: reference_tokens,
                    tokens: target_first_frame_tokens,
                },
                video_sigmas.as_deref(),
                audio_sigmas.as_deref(),
                sigma,
                stream,
            )
            .map_err(mlx_error)?;
        let video_velocity = if reference_tokens > 0 {
            super::conditioning::take_target_suffix(&video_velocity, target_video_tokens, stream)
                .map_err(mlx_error)?
        } else {
            video_velocity
        };
        let audio_velocity = if audio_reference_tokens > 0 {
            super::conditioning::take_target_suffix(&audio_velocity, target_audio_tokens, stream)
                .map_err(mlx_error)?
        } else {
            audio_velocity
        };
        let mut video_x0 = Array::subtract(
            &video,
            &Array::multiply(
                &video_velocity,
                &crate::common::mlx_ops::scalar(sigma).map_err(mlx_error)?,
                stream,
            )
            .map_err(mlx_error)?,
            stream,
        )
        .map_err(mlx_error)?;
        let audio_x0 = Array::subtract(
            &audio,
            &Array::multiply(
                &audio_velocity,
                &crate::common::mlx_ops::scalar(sigma).map_err(mlx_error)?,
                stream,
            )
            .map_err(mlx_error)?,
            stream,
        )
        .map_err(mlx_error)?;
        if let Some((clean, strength)) = first_frame {
            video_x0 =
                super::conditioning::blend_replace_first_frame(&video_x0, clean, strength, stream)
                    .map_err(mlx_error)?;
        }
        if next == 0.0 {
            video = video_x0;
            audio = audio_x0;
        } else if ancestral {
            let (next_key, video_key) = key.split(stream).map_err(mlx_error)?;
            let (next_key, audio_key) = next_key.split(stream).map_err(mlx_error)?;
            key = next_key;
            let video_noise = video_key
                .normal(&video.shape(), Dtype::Float32, stream)
                .map_err(mlx_error)?;
            let audio_noise = audio_key
                .normal(&audio.shape(), Dtype::Float32, stream)
                .map_err(mlx_error)?;
            video = super::denoise::ancestral_euler_x0(
                &video,
                &video_x0,
                sigma,
                next,
                &video_noise,
                stream,
            )
            .map_err(mlx_error)?;
            audio = super::denoise::ancestral_euler_x0(
                &audio,
                &audio_x0,
                sigma,
                next,
                &audio_noise,
                stream,
            )
            .map_err(mlx_error)?;
            if let Some((clean, strength)) = first_frame {
                video =
                    super::conditioning::blend_replace_first_frame(&video, clean, strength, stream)
                        .map_err(mlx_error)?;
            }
        } else {
            video = super::denoise::euler_x0(&video, &video_x0, sigma, next, stream)
                .map_err(mlx_error)?;
            audio = super::denoise::euler_x0(&audio, &audio_x0, sigma, next, stream)
                .map_err(mlx_error)?;
        }
        Array::eval_all(&[&video, &audio]).map_err(mlx_error)?;
        let step = progress_offset + index as u32 + 1;
        exec.progress("denoise", step, super::TOTAL_STEPS)?;
        exec.log(
            "info",
            &format!(
                "ltx25 denoise step {step}/{} sigma {sigma:.6} -> {next:.6}",
                super::TOTAL_STEPS
            ),
        )?;
    }
    Ok((video, audio))
}

#[cfg(feature = "mlx")]
fn encode_first_frame_tokens(
    encoder: &super::video_vae::VideoEncoder,
    still: &ReferenceStill,
    width: u32,
    height: u32,
    stream: &danmo_mlx::Stream,
) -> Result<danmo_mlx::Array, EngineError> {
    use crate::common::image::{resize_lanczos, rgb_to_n11_planar};
    use danmo_mlx::{Array, Dtype};

    let rgb = if still.width == width && still.height == height {
        still.rgb.clone()
    } else {
        resize_lanczos(&still.rgb, still.width, still.height, width, height)
            .map_err(|error| EngineError::Resolve(error.to_string()))?
    };
    let planar = rgb_to_n11_planar(&rgb, width, height)
        .map_err(|error| EngineError::Resolve(error.to_string()))?;
    let pixels = Array::from_f32_slice(&planar, &[1, 3, 1, height as i32, width as i32])
        .map_err(mlx_error)?
        .astype(Dtype::BFloat16, stream)
        .map_err(mlx_error)?;
    let latent = encoder.encode(&pixels, stream).map_err(mlx_error)?;
    let shape = latent.shape();
    if shape.len() != 5 || shape[2] != 1 {
        return Err(EngineError::Resolve(format!(
            "LTX 2.5 first-frame encode expected 1 latent frame, got {shape:?}"
        )));
    }
    super::transformer::patchify_video(&latent, stream).map_err(mlx_error)
}

#[cfg(feature = "mlx")]
fn encode_reference_tokens(
    encoder: &super::video_vae::VideoEncoder,
    still: &ReferenceStill,
    width: u32,
    height: u32,
    frames: u32,
    slot: Option<(&super::msr::SlotEmbedding, usize)>,
    stream: &danmo_mlx::Stream,
) -> Result<danmo_mlx::Array, EngineError> {
    use crate::common::image::{resize_lanczos, rgb_to_n11_planar};
    use danmo_mlx::{Array, Dtype};

    let rgb = if still.width == width && still.height == height {
        still.rgb.clone()
    } else {
        resize_lanczos(&still.rgb, still.width, still.height, width, height)
            .map_err(|error| EngineError::Resolve(error.to_string()))?
    };
    let planar = rgb_to_n11_planar(&rgb, width, height)
        .map_err(|error| EngineError::Resolve(error.to_string()))?;
    let frame = Array::from_f32_slice(&planar, &[1, 3, 1, height as i32, width as i32])
        .map_err(mlx_error)?
        .astype(Dtype::BFloat16, stream)
        .map_err(mlx_error)?;
    let mut frames_list = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        frames_list.push(frame.clone());
    }
    let pixels = Array::concatenate_axis(&frames_list, 2, stream).map_err(mlx_error)?;
    let latent = encoder.encode(&pixels, stream).map_err(mlx_error)?;
    let mut tokens = super::transformer::patchify_video(&latent, stream).map_err(mlx_error)?;
    if let Some((slots, slot_id)) = slot {
        tokens = slots
            .apply_to_tokens(&tokens, slot_id, stream)
            .map_err(mlx_error)?;
    }
    Ok(tokens)
}

#[cfg(feature = "mlx")]
fn encode_all_references(
    encoder: &super::video_vae::VideoEncoder,
    prepared: &Prepared,
    width: u32,
    height: u32,
    slots: Option<&super::msr::SlotEmbedding>,
    stream: &danmo_mlx::Stream,
) -> Result<Option<danmo_mlx::Array>, EngineError> {
    if prepared.reference_images.is_empty() {
        return Ok(None);
    }
    let mut parts = Vec::with_capacity(prepared.reference_images.len());
    for (index, still) in prepared.reference_images.iter().enumerate() {
        let slot = slots.map(|embedding| (embedding, index + 1));
        parts.push(encode_reference_tokens(
            encoder,
            still,
            width,
            height,
            prepared.frames,
            slot,
            stream,
        )?);
    }
    let refs = danmo_mlx::Array::concatenate_axis(&parts, 1, stream).map_err(mlx_error)?;
    Ok(Some(refs))
}

#[cfg(feature = "mlx")]
fn encode_all_audio_references(
    encoder: &super::audio_vae::AudioEncoder,
    prepared: &Prepared,
    slots: Option<&super::msr::SlotEmbedding>,
    stream: &danmo_mlx::Stream,
) -> Result<(Option<danmo_mlx::Array>, Vec<f32>), EngineError> {
    if prepared.reference_audios.is_empty() {
        return Ok((None, Vec::new()));
    }
    let image_slot_count = prepared.reference_images.len().max(1);
    let max_tokens = max_audio_slot_tokens();
    let mut parts = Vec::with_capacity(prepared.reference_audios.len());
    let mut positions = Vec::new();
    for clip in &prepared.reference_audios {
        let resampled = super::mel::resample_stereo_interleaved(&clip.pcm, clip.sample_rate)
            .map_err(mlx_error)?;
        let (planar, time) = super::mel::stereo_log_mel(&resampled).map_err(mlx_error)?;
        let mel = danmo_mlx::Array::from_f32_slice(&planar, &[1, 2, time as i32, 64])
            .map_err(mlx_error)?
            .astype(danmo_mlx::Dtype::BFloat16, stream)
            .map_err(mlx_error)?;
        let mut tokens = encoder.encode_tokens(&mel, stream).map_err(mlx_error)?;
        let token_count = tokens.shape()[1] as usize;
        if token_count > max_tokens {
            // Overflow mode=truncate: keep the leading slot window.
            let shape = tokens.shape();
            tokens = tokens
                .slice(
                    &[0, 0, 0],
                    &[shape[0], max_tokens as i32, shape[2]],
                    &[1, 1, 1],
                    stream,
                )
                .map_err(mlx_error)?;
        }
        let kept = tokens.shape()[1] as usize;
        if let Some(slots) = slots {
            tokens = slots
                .apply_to_tokens(&tokens, clip.slot_id, stream)
                .map_err(mlx_error)?;
        }
        let slot_positions = super::conditioning::msr_audio_slot_positions(
            kept,
            clip.slot_id,
            image_slot_count,
            super::msr::AUDIO_SLOT_DURATION_SECONDS,
            super::msr::AUDIO_SLOT_END_MARGIN_SECONDS,
            super::AUDIO_LATENT_SPEC,
        )
        .map_err(mlx_error)?;
        positions.extend(slot_positions);
        parts.push(tokens);
    }
    let refs = danmo_mlx::Array::concatenate_axis(&parts, 1, stream).map_err(mlx_error)?;
    Ok((Some(refs), positions))
}

#[cfg(feature = "mlx")]
fn max_audio_slot_tokens() -> usize {
    let token_seconds = super::AUDIO_LATENT_SPEC.hop_length as f32
        * super::AUDIO_LATENT_SPEC.downsample_factor as f32
        / super::AUDIO_LATENT_SPEC.sample_rate as f32;
    (super::msr::AUDIO_SLOT_DURATION_SECONDS / token_seconds).floor() as usize
}

#[cfg(feature = "mlx")]
fn infer_mlx(
    run: &ResolvedRun,
    prepared: &Prepared,
    bundle_config: &BundleConfig,
    quantization: Option<Quantization>,
    exec: &ExecutionContext,
) -> Result<EngineOutput, EngineError> {
    // Keep Arrays on the process MLX executor thread (no private Stream::gpu).
    let run = run.clone();
    let prepared = prepared.clone();
    let bundle_config = bundle_config.clone();
    let exec = exec.clone();
    crate::runtime::mlx_executor()
        .map_err(mlx_error)?
        .execute("ltx25 infer", move |_, stream, _cache, _events| {
            Ok(infer_mlx_on_stream(
                &run,
                &prepared,
                &bundle_config,
                quantization,
                &exec,
                stream,
            ))
        })
        .map_err(mlx_error)?
}

#[cfg(feature = "mlx")]
fn infer_mlx_on_stream(
    run: &ResolvedRun,
    prepared: &Prepared,
    bundle_config: &BundleConfig,
    quantization: Option<Quantization>,
    exec: &ExecutionContext,
    stream: &danmo_mlx::Stream,
) -> Result<EngineOutput, EngineError> {
    use crate::common::{av_latent, ffmpeg, png, wav};
    use danmo_mlx::{Array, Dtype, RandomKey};

    exec.check_cancelled()?;
    exec.phase("conditioning")?;
    let prompt =
        super::text_encoder::encode_prompt(&run.bundle, bundle_config, &prepared.prompt, &stream)?;
    Array::eval_all(&[&prompt.video, &prompt.audio]).map_err(mlx_error)?;
    exec.log("info", "ltx25 Gemma 4 prompt conditioning complete")?;
    exec.check_cancelled()?;

    exec.phase("stage1")?;
    let mut model = super::transformer::Transformer::load(
        &run.bundle,
        &bundle_config.transformer,
        quantization,
    )?;
    exec.check_cancelled()?;
    let msr_path = if prepared.reference_images.is_empty() && prepared.reference_audios.is_empty() {
        None
    } else {
        Some(
            prepared
                .loras
                .iter()
                .find(|lora| lora.id.contains("msr"))
                .or_else(|| prepared.loras.first())
                .ok_or_else(|| {
                    EngineError::bad_request(
                        "LTX 2.5 Ref2VA requires the multi-subject reference LoRA (ltx25-msr-lora)",
                    )
                })?
                .path
                .clone(),
        )
    };
    let slot_embedding = match msr_path.as_ref() {
        Some(path) if !prepared.reference_images.is_empty() => {
            exec.log(
                "info",
                &format!(
                    "ltx25 loading MSR image slot embeddings from {}",
                    path.display()
                ),
            )?;
            Some(super::msr::SlotEmbedding::load_image_slots(path, &stream)?)
        }
        _ => None,
    };
    let audio_slot_embedding = match msr_path.as_ref() {
        Some(path) if !prepared.reference_audios.is_empty() => {
            exec.log(
                "info",
                &format!(
                    "ltx25 loading MSR audio slot embeddings from {}",
                    path.display()
                ),
            )?;
            Some(super::msr::SlotEmbedding::load_audio_slots(path, &stream)?)
        }
        _ => None,
    };
    if !prepared.loras.is_empty() {
        exec.log(
            "info",
            &format!("ltx25 attaching {} LoRA adapter(s)", prepared.loras.len()),
        )?;
        model = model.attach_lora(&prepared.loras, &stream)?;
    }
    let stage1_shape = av_latent::video_latent_shape(
        prepared.frames,
        prepared.height / 2,
        prepared.width / 2,
        super::VIDEO_LATENT_SPEC,
    )
    .map_err(EngineError::Resolve)?;
    let audio_tokens = av_latent::audio_token_count(
        prepared.frames,
        prepared.fps as f32,
        super::AUDIO_LATENT_SPEC,
    )
    .map_err(EngineError::Resolve)?;
    let reference_offsets = if prepared.reference_images.is_empty() {
        Vec::new()
    } else {
        super::conditioning::msr_temporal_offsets(
            prepared.reference_images.len(),
            prepared.fps as f32,
        )
        .map_err(mlx_error)?
    };
    let needs_video_encoder =
        !prepared.reference_images.is_empty() || prepared.first_frame.is_some();
    let video_encoder = if needs_video_encoder {
        Some(super::video_vae::VideoEncoder::load(
            &run.bundle,
            &bundle_config.vae,
        )?)
    } else {
        None
    };
    let stage1_reference = if prepared.reference_images.is_empty() {
        None
    } else {
        let encoder = video_encoder
            .as_ref()
            .ok_or_else(|| EngineError::Resolve("LTX 2.5 Ref2VA missing video encoder".into()))?;
        exec.log(
            "info",
            &format!(
                "ltx25 encoding {} Ref2VA still(s) for stage 1",
                prepared.reference_images.len()
            ),
        )?;
        encode_all_references(
            encoder,
            prepared,
            prepared.width / 2,
            prepared.height / 2,
            slot_embedding.as_ref(),
            &stream,
        )?
    };
    let stage1_first_frame = match (prepared.first_frame.as_ref(), video_encoder.as_ref()) {
        (Some(still), Some(encoder)) => {
            exec.log(
                "info",
                &format!(
                    "ltx25 encoding animate first frame for stage 1 (strength={})",
                    super::conditioning::STAGE_1_IMAGE_STRENGTH
                ),
            )?;
            Some(encode_first_frame_tokens(
                encoder,
                still,
                prepared.width / 2,
                prepared.height / 2,
                &stream,
            )?)
        }
        (Some(_), None) => {
            return Err(EngineError::Resolve(
                "LTX 2.5 animate missing video encoder".into(),
            ));
        }
        _ => None,
    };
    let (audio_reference, audio_reference_positions) = if prepared.reference_audios.is_empty() {
        (None, Vec::new())
    } else {
        exec.log(
            "info",
            &format!(
                "ltx25 encoding {} Ref2VA audio clip(s)",
                prepared.reference_audios.len()
            ),
        )?;
        let encoder = super::audio_vae::AudioEncoder::load(&run.bundle, &bundle_config.audio_vae)?;
        encode_all_audio_references(&encoder, prepared, audio_slot_embedding.as_ref(), &stream)?
    };
    drop(audio_slot_embedding);
    let stage1_rope = stage_rope(
        model.config(),
        stage1_shape,
        audio_tokens,
        prepared.fps,
        &reference_offsets,
        &audio_reference_positions,
        &stream,
    )?;
    let mut video = RandomKey::new(prepared.seed as u64)
        .map_err(mlx_error)?
        .normal(
            &[
                1,
                stage1_shape.token_count().map_err(EngineError::Resolve)? as i32,
                super::VIDEO_TOKEN_CHANNELS as i32,
            ],
            Dtype::BFloat16,
            &stream,
        )
        .map_err(mlx_error)?;
    if let Some(clean) = stage1_first_frame.as_ref() {
        video = super::conditioning::init_replace_first_frame(
            &video,
            clean,
            super::conditioning::STAGE_1_IMAGE_STRENGTH,
            &stream,
        )
        .map_err(mlx_error)?;
    }
    let audio = RandomKey::new((prepared.seed + 1) as u64)
        .map_err(mlx_error)?
        .normal(
            &[1, audio_tokens as i32, super::AUDIO_TOKEN_CHANNELS as i32],
            Dtype::BFloat16,
            &stream,
        )
        .map_err(mlx_error)?;
    exec.check_cancelled()?;
    let (video, audio) = denoise_stage(
        &model,
        video,
        audio,
        &prompt,
        &stage1_rope,
        &super::DISTILLED_SIGMAS,
        true,
        prepared.seed + super::ANCESTRAL_NOISE_SEED_OFFSET,
        0,
        stage1_reference.as_ref(),
        audio_reference.as_ref(),
        (stage1_shape.height * stage1_shape.width) as usize,
        stage1_first_frame
            .as_ref()
            .map(|tokens| (tokens, super::conditioning::STAGE_1_IMAGE_STRENGTH)),
        exec,
        &stream,
    )?;
    drop(stage1_reference);
    drop(stage1_first_frame);

    exec.check_cancelled()?;
    exec.phase("latent_upscale")?;
    let latent = super::transformer::unpatchify_video(
        &video,
        stage1_shape.frames as i32,
        stage1_shape.height as i32,
        stage1_shape.width as i32,
        &stream,
    )
    .map_err(mlx_error)?;
    let upsampler = super::upsampler::LatentUpsampler::load(&run.bundle, &bundle_config.upsampler)?;
    let latent = upsampler.upsample(&latent, &stream).map_err(mlx_error)?;
    drop(upsampler);
    Array::eval(&latent).map_err(mlx_error)?;
    let video = super::transformer::patchify_video(&latent, &stream).map_err(mlx_error)?;
    drop(latent);

    exec.phase("stage2")?;
    let stage2_shape = av_latent::video_latent_shape(
        prepared.frames,
        prepared.height,
        prepared.width,
        super::VIDEO_LATENT_SPEC,
    )
    .map_err(EngineError::Resolve)?;
    let stage2_reference = if prepared.reference_images.is_empty() {
        None
    } else {
        let encoder = video_encoder.as_ref().ok_or_else(|| {
            EngineError::Resolve("LTX 2.5 Ref2VA missing video encoder for stage 2".into())
        })?;
        exec.log(
            "info",
            &format!(
                "ltx25 encoding {} Ref2VA still(s) for stage 2",
                prepared.reference_images.len()
            ),
        )?;
        encode_all_references(
            encoder,
            prepared,
            prepared.width,
            prepared.height,
            slot_embedding.as_ref(),
            &stream,
        )?
    };
    let stage2_first_frame = match (prepared.first_frame.as_ref(), video_encoder.as_ref()) {
        (Some(still), Some(encoder)) => {
            exec.log(
                "info",
                &format!(
                    "ltx25 encoding animate first frame for stage 2 (strength={})",
                    super::conditioning::STAGE_2_IMAGE_STRENGTH
                ),
            )?;
            Some(encode_first_frame_tokens(
                encoder,
                still,
                prepared.width,
                prepared.height,
                &stream,
            )?)
        }
        (Some(_), None) => {
            return Err(EngineError::Resolve(
                "LTX 2.5 animate missing video encoder for stage 2".into(),
            ));
        }
        _ => None,
    };
    drop(video_encoder);
    drop(slot_embedding);
    let stage2_rope = stage_rope(
        model.config(),
        stage2_shape,
        audio_tokens,
        prepared.fps,
        &reference_offsets,
        &audio_reference_positions,
        &stream,
    )?;
    drop(stage1_rope);
    let sigma = super::STAGE_2_SIGMAS[0];
    let mut video = renoise(video, prepared.seed + 2, sigma, &stream)?;
    if let Some(clean) = stage2_first_frame.as_ref() {
        video = super::conditioning::init_replace_first_frame(
            &video,
            clean,
            super::conditioning::STAGE_2_IMAGE_STRENGTH,
            &stream,
        )
        .map_err(mlx_error)?;
    }
    let audio = renoise(audio, prepared.seed + 2, sigma, &stream)?;
    exec.check_cancelled()?;
    let (video, audio) = denoise_stage(
        &model,
        video,
        audio,
        &prompt,
        &stage2_rope,
        &super::STAGE_2_SIGMAS,
        true,
        prepared.seed + super::ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET,
        super::STAGE_1_STEPS,
        stage2_reference.as_ref(),
        audio_reference.as_ref(),
        (stage2_shape.height * stage2_shape.width) as usize,
        stage2_first_frame
            .as_ref()
            .map(|tokens| (tokens, super::conditioning::STAGE_2_IMAGE_STRENGTH)),
        exec,
        &stream,
    )?;
    drop(stage2_reference);
    drop(stage2_first_frame);
    drop(audio_reference);
    drop(model);
    drop(prompt);
    drop(stage2_rope);

    exec.check_cancelled()?;
    exec.phase("decode")?;
    let video_latent = super::transformer::unpatchify_video(
        &video,
        stage2_shape.frames as i32,
        stage2_shape.height as i32,
        stage2_shape.width as i32,
        &stream,
    )
    .map_err(mlx_error)?;
    let audio_latent =
        super::transformer::unpatchify_audio(&audio, 8, &stream).map_err(mlx_error)?;
    drop(video);
    drop(audio);
    Array::eval_all(&[&video_latent, &audio_latent]).map_err(mlx_error)?;

    let video_decoder = super::video_vae::VideoDecoder::load(&run.bundle, &bundle_config.vae)?;
    let pixels = video_decoder
        .decode(&video_latent, prepared.seed as u64, &stream)
        .map_err(mlx_error)?;
    drop(video_decoder);
    drop(video_latent);
    let audio_decoder =
        super::audio_vae::AudioDecoder::load(&run.bundle, &bundle_config.audio_vae)?;
    let mel = audio_decoder
        .decode(&audio_latent, &stream)
        .map_err(mlx_error)?;
    drop(audio_decoder);
    drop(audio_latent);
    let vocoder = super::vocoder::Vocoder::load(&run.bundle, &bundle_config.vocoder)?;
    let waveform = vocoder.decode(&mel, &stream).map_err(mlx_error)?;
    drop(vocoder);
    drop(mel);
    Array::eval_all(&[&pixels, &waveform]).map_err(mlx_error)?;

    let work = scratch_dir(&exec.task_id);
    std::fs::create_dir_all(&work)
        .map_err(|error| io_error("create LTX 2.5 work directory", error))?;
    let result = write_av_artifact(
        &pixels,
        &waveform,
        prepared,
        &work,
        &stream,
        &png::encode_rgb,
        &ffmpeg::mux_png_sequence_wav_to_mp4,
        &wav::write_pcm16,
    );
    let _ = std::fs::remove_dir_all(&work);
    let (bytes, duration) = result?;
    let action = match prepared.mode {
        RunMode::Animate => "animate",
        RunMode::Create => "create",
    };
    let metadata = json!({
        "model": run.model,
        "family": "ltx25",
        "action": action,
        "width": prepared.width,
        "height": prepared.height,
        "frames": prepared.frames,
        "fps": prepared.fps,
        "seed": prepared.seed,
        "steps": super::TOTAL_STEPS,
        "stage1_steps": super::STAGE_1_STEPS,
        "stage2_steps": super::STAGE_2_STEPS,
        "sample_rate": 48_000,
        "audio_channels": 2,
        "first_frame": prepared.first_frame.is_some(),
        "reference_images": prepared.reference_images.len(),
        "reference_audios": prepared.reference_audios.len(),
        "adapters": prepared
            .loras
            .iter()
            .map(|lora| json!({ "id": lora.id, "weight": lora.weight }))
            .collect::<Vec<_>>(),
    });
    let artifact = Artifact::mp4(
        bytes,
        prepared.width as i64,
        prepared.height as i64,
        duration,
        metadata.clone(),
    );
    Ok(EngineOutput {
        family: "ltx25".into(),
        action: action.into(),
        steps: super::TOTAL_STEPS,
        graph_computes: super::TOTAL_STEPS,
        latent_len: stage2_shape.token_count().map_err(EngineError::Resolve)?
            * super::VIDEO_TOKEN_CHANNELS
            + audio_tokens * super::AUDIO_TOKEN_CHANNELS,
        width: Some(prepared.width),
        height: Some(prepared.height),
        sample_rate: Some(48_000),
        frames: Some(prepared.frames as i64),
        metadata,
        timings: Vec::new(),
        artifacts: vec![artifact],
    })
}

#[cfg(feature = "mlx")]
fn renoise(
    initial: danmo_mlx::Array,
    seed: i64,
    sigma: f32,
    stream: &danmo_mlx::Stream,
) -> Result<danmo_mlx::Array, EngineError> {
    use danmo_mlx::{Array, RandomKey};
    let noise = RandomKey::new(seed as u64)
        .map_err(mlx_error)?
        .normal(&initial.shape(), initial.dtype(), stream)
        .map_err(mlx_error)?;
    let a = Array::multiply(
        &initial,
        &crate::common::mlx_ops::scalar(1.0 - sigma).map_err(mlx_error)?,
        stream,
    )
    .map_err(mlx_error)?;
    let b = Array::multiply(
        &noise,
        &crate::common::mlx_ops::scalar(sigma).map_err(mlx_error)?,
        stream,
    )
    .map_err(mlx_error)?;
    Array::add(&a, &b, stream).map_err(mlx_error)
}

#[cfg(feature = "mlx")]
fn scratch_dir(task_id: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    std::env::temp_dir().join(format!(
        "danmo-ltx25-{task_id}-{}-{nanos}",
        std::process::id()
    ))
}

#[cfg(feature = "mlx")]
fn planar_stereo_to_interleaved(planar: &[f32]) -> Result<Vec<f32>, EngineError> {
    if planar.is_empty() || planar.len() % 2 != 0 || planar.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Resolve(
            "LTX 2.5 decoded audio is not finite planar stereo".into(),
        ));
    }
    let frames = planar.len() / 2;
    let mut interleaved = Vec::with_capacity(planar.len());
    for frame in 0..frames {
        interleaved.push(planar[frame]);
        interleaved.push(planar[frames + frame]);
    }
    Ok(interleaved)
}

#[cfg(feature = "mlx")]
fn write_av_artifact(
    pixels: &danmo_mlx::Array,
    waveform: &danmo_mlx::Array,
    prepared: &Prepared,
    work: &std::path::Path,
    stream: &danmo_mlx::Stream,
    encode_png: &dyn Fn(u32, u32, &[u8]) -> Vec<u8>,
    mux: &dyn Fn(
        &std::path::Path,
        &std::path::Path,
        &std::path::Path,
        f64,
    ) -> Result<(), EngineError>,
    write_wav: &dyn Fn(&std::path::Path, &[f32], u16, u32) -> std::io::Result<()>,
) -> Result<(Vec<u8>, f64), EngineError> {
    let shape = pixels.shape();
    if shape
        != [
            1,
            3,
            prepared.frames as i32,
            prepared.height as i32,
            prepared.width as i32,
        ]
    {
        return Err(EngineError::Resolve(format!(
            "LTX 2.5 decoded video shape {shape:?} does not match request {}x{}x{}",
            prepared.width, prepared.height, prepared.frames
        )));
    }
    let values = pixels.to_vec_f32(stream).map_err(mlx_error)?;
    let (f, h, w) = (
        prepared.frames as usize,
        prepared.height as usize,
        prepared.width as usize,
    );
    let plane = f * h * w;
    for frame in 0..f {
        let mut rgb = vec![0u8; h * w * 3];
        for y in 0..h {
            for x in 0..w {
                let offset = frame * h * w + y * w + x;
                for channel in 0..3 {
                    let value = values[channel * plane + offset];
                    rgb[(y * w + x) * 3 + channel] =
                        ((value.clamp(-1.0, 1.0) + 1.0) * 127.5).round() as u8;
                }
            }
        }
        let path = work.join(format!("frame_{:06}.png", frame + 1));
        std::fs::write(&path, encode_png(prepared.width, prepared.height, &rgb))
            .map_err(|error| io_error("write decoded PNG frame", error))?;
    }
    let shape = waveform.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 2 {
        return Err(EngineError::Resolve(format!(
            "LTX 2.5 decoded audio shape {shape:?} does not match [1,2,frames]"
        )));
    }
    let planar = waveform.to_vec_f32(stream).map_err(mlx_error)?;
    let audio = planar_stereo_to_interleaved(&planar)?;
    let wav_path = work.join("audio.wav");
    write_wav(&wav_path, &audio, 2, 48_000)
        .map_err(|error| io_error("write decoded stereo WAV", error))?;
    let output = work.join("output.mp4");
    mux(
        &work.join("frame_%06d.png"),
        &wav_path,
        &output,
        prepared.fps as f64,
    )?;
    let bytes =
        std::fs::read(&output).map_err(|error| io_error("read muxed LTX 2.5 MP4", error))?;
    if bytes.is_empty() {
        return Err(EngineError::Resolve(
            "LTX 2.5 mux produced an empty MP4".into(),
        ));
    }
    Ok((bytes, f as f64 / prepared.fps as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "mlx")]
    #[test]
    fn planar_stereo_is_interleaved_for_wav() {
        assert_eq!(
            planar_stereo_to_interleaved(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap(),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
        );
    }

    fn request() -> VideoGenerationRequest {
        VideoGenerationRequest {
            model: "ltx-2.5-distilled".into(),
            prompt: "a cat".into(),
            negative_prompt: String::new(),
            size: None,
            steps: None,
            guidance: None,
            shift: None,
            seed: Some(7),
            num_frames: Some(49),
            fps: Some(24),
            media_references: Vec::new(),
            adapters: Vec::new(),
            metadata: serde_json::Value::Null,
            priority: "normal".into(),
        }
    }

    #[test]
    fn prepare_keeps_reference_images_empty_until_resolver_loads() {
        let dir =
            std::env::temp_dir().join(format!("danmo-ltx25-prepare-ref-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("msr.safetensors");
        std::fs::write(&path, b"placeholder").unwrap();
        let mut req = request();
        req.media_references.push(danmo_contracts::MediaReference {
            kind: "image".into(),
            asset_id: "ast_x".into(),
        });
        req.adapters.push(danmo_contracts::ImageAdapter {
            id: "ltx25-msr-lora".into(),
            weight: 1.0,
            path: path.display().to_string(),
        });
        let prepared = prepare_params(&req).unwrap();
        assert!(prepared.reference_images.is_empty());
        assert_eq!(prepared.loras.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_snaps_frames_and_checks_alignment() {
        let prepared = prepare_params(&request()).unwrap();
        assert_eq!(prepared.frames, 49);
        assert_eq!(prepared.seed, 7);
        let mut invalid = request();
        invalid.size = Some("1280x720".into());
        assert!(prepare_params(&invalid).is_err());
        invalid = request();
        // 497 snaps to 497; at 24fps that is > 20s on the positional axis.
        invalid.num_frames = Some(497);
        invalid.fps = Some(24);
        assert!(prepare_params(&invalid).is_err());
    }

    #[test]
    fn version_declares_quantization() {
        assert_eq!(
            quantization(Some("mlx-q4")).unwrap().unwrap().group_size,
            64
        );
        assert_eq!(quantization(Some("mlx-q8")).unwrap().unwrap().bits, 8);
        assert!(quantization(Some("q4-random")).is_err());
    }
}
