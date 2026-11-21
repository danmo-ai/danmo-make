//! LTX 2.5 distilled audio/video generation.
//!
//! The family owns its checkpoint configuration, fixed distilled schedule,
//! two-stage generation policy, and x0/ancestral sampler formulas. Neutral
//! geometry and position tables come from `common::av_latent`. The execution
//! shape is a generation facade: no image DiT pipeline or another family is
//! reused for the joint audio/video orchestration.

#[cfg(feature = "mlx")]
pub mod audio_vae;
#[cfg(feature = "mlx")]
pub mod bundle;
#[cfg(feature = "mlx")]
pub mod conditioning;
pub mod config;
#[cfg(feature = "mlx")]
pub mod conv;
#[cfg(feature = "mlx")]
pub mod denoise;
#[cfg(feature = "mlx")]
pub mod mel;
#[cfg(feature = "mlx")]
pub mod msr;
mod session;
#[cfg(feature = "mlx")]
pub mod text_encoder;
pub mod tokenizer;
#[cfg(feature = "mlx")]
pub mod transformer;
#[cfg(feature = "mlx")]
pub mod upsampler;
#[cfg(feature = "mlx")]
pub mod video_vae;
#[cfg(feature = "mlx")]
pub mod vocoder;
#[cfg(feature = "mlx")]
pub mod weights;

pub use session::Ltx25VideoCreateFamily;

use crate::common::av_latent::{
    audio_token_count, video_latent_shape, AudioLatentSpec, VideoLatentShape, VideoLatentSpec,
};
use crate::error::EngineError;

pub const VIDEO_LATENT_SPEC: VideoLatentSpec = VideoLatentSpec {
    temporal_compression: 8,
    spatial_compression: 32,
};
pub const AUDIO_LATENT_SPEC: AudioLatentSpec = AudioLatentSpec {
    downsample_factor: 4,
    hop_length: 160,
    sample_rate: 16_000,
};
pub const VIDEO_TOKEN_CHANNELS: usize = 128;
pub const AUDIO_TOKEN_CHANNELS: usize = 8 * 16;
pub const MAX_POSITION_SECONDS: f32 = 20.0;
pub const ANCESTRAL_NOISE_SEED_OFFSET: i64 = 10_000;
pub const ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET: i64 = 20_000;
pub const STAGE_1_STEPS: u32 = 8;
pub const STAGE_2_STEPS: u32 = 3;
pub const TOTAL_STEPS: u32 = STAGE_1_STEPS + STAGE_2_STEPS;

pub const DISTILLED_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];
pub const STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

#[derive(Debug, Clone, Copy)]
pub struct DistilledRequest<'a> {
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub fps: f32,
    pub steps: u32,
    pub guidance: f32,
    pub step_distill: bool,
    pub negative_prompt: &'a str,
}

impl Default for DistilledRequest<'static> {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 768,
            num_frames: 121,
            fps: 24.0,
            steps: 8,
            guidance: 1.0,
            step_distill: true,
            negative_prompt: "",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct StagePlan {
    pub width: u32,
    pub height: u32,
    pub video: VideoLatentShape,
    pub video_tokens: usize,
    pub sigmas: &'static [f32],
    pub ancestral: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TwoStagePlan {
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub fps: f32,
    pub audio_tokens: usize,
    pub stage1: StagePlan,
    pub stage2: StagePlan,
    /// The fixed schedule ignores a non-8 request; the session emits the warning.
    pub ignored_step_request: Option<u32>,
}

impl TwoStagePlan {
    pub fn total_steps(&self) -> usize {
        self.stage1.sigmas.len() + self.stage2.sigmas.len() - 2
    }
}

pub fn prepare_distilled(request: DistilledRequest<'_>) -> Result<TwoStagePlan, EngineError> {
    if !request.step_distill {
        return Err(EngineError::bad_request(
            "LTX 2.5 supports the distilled 8-step pipeline only (step_distill=true required)",
        ));
    }
    if request.guidance != 0.0 && request.guidance != 1.0 {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 distilled sampling fixes guidance=1.0 (got {})",
            request.guidance
        )));
    }
    if !request.negative_prompt.is_empty() {
        return Err(EngineError::bad_request(
            "LTX 2.5 distilled sampling does not use a negative prompt",
        ));
    }
    if request.width == 0
        || request.height == 0
        || request.width % 64 != 0
        || request.height % 64 != 0
    {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 two-stage resolution ({}x{}) must be divisible by 64; stage 1 runs at half resolution",
            request.height, request.width
        )));
    }
    if !request.fps.is_finite() {
        return Err(EngineError::bad_request("LTX 2.5 fps must be finite"));
    }
    let fps = request.fps.max(1.0);
    let snapped = ((request.num_frames.max(1) - 1) / 8) * 8 + 1;
    let num_frames = snapped.max(9);
    let duration = (num_frames - 1) as f32 / fps;
    if duration > MAX_POSITION_SECONDS + 1e-6 {
        return Err(EngineError::bad_request(format!(
            "LTX 2.5 positional time axis stops at {MAX_POSITION_SECONDS:.0}s ({num_frames} frames at {fps:.0}fps is {duration:.2}s); use 481 frames at 24fps for the longest single shot"
        )));
    }
    let audio_tokens =
        audio_token_count(num_frames, fps, AUDIO_LATENT_SPEC).map_err(EngineError::bad_request)?;
    if audio_tokens == 0 {
        return Err(EngineError::bad_request(
            "LTX 2.5 needs at least one audio latent token",
        ));
    }
    let stage1 = stage_plan(
        request.width / 2,
        request.height / 2,
        num_frames,
        &DISTILLED_SIGMAS,
        true,
    )?;
    let stage2 = stage_plan(
        request.width,
        request.height,
        num_frames,
        &STAGE_2_SIGMAS,
        true,
    )?;
    Ok(TwoStagePlan {
        width: request.width,
        height: request.height,
        num_frames,
        fps,
        audio_tokens,
        stage1,
        stage2,
        ignored_step_request: (request.steps != 8).then_some(request.steps),
    })
}

fn stage_plan(
    width: u32,
    height: u32,
    num_frames: u32,
    sigmas: &'static [f32],
    ancestral: bool,
) -> Result<StagePlan, EngineError> {
    let video = video_latent_shape(num_frames, height, width, VIDEO_LATENT_SPEC)
        .map_err(EngineError::bad_request)?;
    let video_tokens = video.token_count().map_err(EngineError::bad_request)?;
    Ok(StagePlan {
        width,
        height,
        video,
        video_tokens,
        sigmas,
        ancestral,
    })
}

#[cfg(test)]
mod tests {
    use super::{prepare_distilled, DistilledRequest, DISTILLED_SIGMAS, STAGE_2_SIGMAS};

    #[test]
    fn published_defaults_make_the_eight_plus_three_step_plan() {
        let plan = prepare_distilled(DistilledRequest::default()).unwrap();
        assert_eq!(plan.num_frames, 121);
        assert_eq!(plan.audio_tokens, 126);
        assert_eq!(plan.total_steps(), 11);
        assert_eq!(plan.ignored_step_request, None);
        assert!(plan.stage1.ancestral);
        assert!(plan.stage2.ancestral);
        assert_eq!(plan.stage1.sigmas, DISTILLED_SIGMAS);
        assert_eq!(plan.stage2.sigmas, STAGE_2_SIGMAS);
        assert_eq!((plan.stage1.width, plan.stage1.height), (512, 384));
        assert_eq!(plan.stage1.video.frames, 16);
        assert_eq!(plan.stage1.video_tokens, 3_072);
        assert_eq!(plan.stage2.video_tokens, 12_288);
    }

    #[test]
    fn frames_snap_down_then_clamp_to_the_shortest_clip() {
        let plan = prepare_distilled(DistilledRequest {
            num_frames: 122,
            ..DistilledRequest::default()
        })
        .unwrap();
        assert_eq!(plan.num_frames, 121);
        let plan = prepare_distilled(DistilledRequest {
            num_frames: 1,
            ..DistilledRequest::default()
        })
        .unwrap();
        assert_eq!(plan.num_frames, 9);
        let plan = prepare_distilled(DistilledRequest {
            steps: 28,
            ..DistilledRequest::default()
        })
        .unwrap();
        assert_eq!(plan.ignored_step_request, Some(28));
        assert_eq!(plan.total_steps(), 11);
    }

    #[test]
    fn checkpoint_position_limit_is_enforced_after_snapping() {
        let plan = prepare_distilled(DistilledRequest {
            num_frames: 488,
            ..DistilledRequest::default()
        })
        .unwrap();
        assert_eq!(plan.num_frames, 481);
        let error = prepare_distilled(DistilledRequest {
            num_frames: 497,
            ..DistilledRequest::default()
        })
        .err()
        .unwrap();
        assert!(
            error.to_string().contains("positional time axis"),
            "{error}"
        );
    }

    #[test]
    fn unsupported_sampling_and_misaligned_resolution_fail() {
        for request in [
            DistilledRequest {
                step_distill: false,
                ..DistilledRequest::default()
            },
            DistilledRequest {
                guidance: 3.0,
                ..DistilledRequest::default()
            },
            DistilledRequest {
                negative_prompt: "blurry",
                ..DistilledRequest::default()
            },
            DistilledRequest {
                width: 1280,
                height: 720,
                ..DistilledRequest::default()
            },
            DistilledRequest {
                fps: f32::NAN,
                ..DistilledRequest::default()
            },
        ] {
            assert!(prepare_distilled(request).is_err());
        }
        // Main accepts guidance zero as the equivalent single-forward mode.
        assert!(prepare_distilled(DistilledRequest {
            guidance: 0.0,
            ..DistilledRequest::default()
        })
        .is_ok());
    }
}
