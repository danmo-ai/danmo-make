//! ACE-Step text2music create orchestration (acestep.cpp `pipeline-synth`).
//!
//! Phases: text encode → cond pack → silence context → DiT denoise (APG CFG) →
//! optional CA+DTW lyric timeline → VAE → WAV.

use super::dcw::DcwSettings;
use super::lyrics::LyricsAlignment;
use super::quality::QualityAssessment;
use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{LoraFile, WeightError};
use std::path::Path;

/// Latent hop: 48 kHz / 1920 upsample = 25 Hz.
pub const LATENT_FPS: f32 = 25.0;
pub const SILENCE_MAX_FRAMES: i64 = 15_000;

pub struct CreateRequest<'a> {
    pub prompt: &'a str,
    pub lyrics: &'a str,
    pub duration_sec: f32,
    pub seed: Option<i64>,
    pub guidance: f32,
    pub steps: Option<u32>,
    pub vocal_language: &'a str,
    pub bpm: Option<i32>,
    pub keyscale: &'a str,
    pub timesignature: &'a str,
    /// Optional 5Hz LM / planner audio codes (FSQ indices). When set, DiT
    /// context uses FSQ-detokenized latents instead of pure silence.
    pub audio_code_indices: Option<&'a [i32]>,
    /// Optional time-major `[T, 64]` latents for the timbre encoder. When
    /// absent, a single silence frame is used (acestep.cpp `ops_encode_timbre`).
    pub timbre_latents: Option<&'a [f32]>,
    pub loras: &'a [LoraFile],
    pub dcw: DcwSettings,
    pub lm_expanded: bool,
    pub pmi_bonus: f32,
}

#[derive(Debug)]
pub struct CreateOutput {
    pub wav: Vec<u8>,
    pub sample_rate: u32,
    pub frames: i64,
    pub steps: u32,
    pub enc_seq: i64,
    pub graph_computes: u32,
    pub dcw: DcwSettings,
    pub dcw_active: bool,
    pub quality: QualityAssessment,
    /// Wall-clock seconds per phase (`text`, `dit_load`, `denoise`, `vae`).
    pub timings: Vec<(&'static str, f64)>,
    /// Effective lyrics fed to DiT (after normalize).
    pub lyrics_effective: String,
    /// Main-compatible structure_estimate timeline (LRC when non-empty).
    pub lyrics_alignment: LyricsAlignment,
    /// `<stem>_lyrics.txt` body, if any.
    pub lyrics_txt: Option<String>,
    /// `<stem>.lrc` body, if any.
    pub lyrics_lrc: Option<String>,
}

pub fn duration_to_latent_frames(duration_sec: f32) -> Result<i64, WeightError> {
    if !duration_sec.is_finite() || duration_sec <= 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step duration {duration_sec} must be a positive finite number"
        )));
    }
    let frames = (duration_sec * LATENT_FPS).round() as i64;
    let frames = frames.max(128);
    if frames > SILENCE_MAX_FRAMES {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step latent frames {frames} exceed silence_latent max {SILENCE_MAX_FRAMES}"
        )));
    }
    Ok(frames)
}

/// Prefer even latent length (odd turbo lengths can collapse in upstream).
pub fn snap_latent_frames(frames: i64) -> i64 {
    if frames % 2 == 0 {
        frames
    } else {
        frames + 1
    }
}

#[cfg(test)]
pub fn run_full_create(
    bundle: &Path,
    request: &CreateRequest<'_>,
    backend: BackendKind,
    on_step: impl FnMut(u32, u32),
) -> Result<CreateOutput, WeightError> {
    let ctx = GgmlContext::new(backend).map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    run_full_create_with_context(bundle, request, &ctx, on_step)
}

pub fn run_full_create_with_context(
    bundle: &Path,
    request: &CreateRequest<'_>,
    ctx: &GgmlContext,
    mut on_step: impl FnMut(u32, u32),
) -> Result<CreateOutput, WeightError> {
    let _ = ctx;
    #[cfg(feature = "mlx")]
    {
        return super::mlx::session::create(bundle, request, on_step);
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (bundle, request, on_step);
        return Err(WeightError::NotBuilt(
            "ACE-Step create runs on MLX. Rebuild danqing-api with the mlx feature.".into(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::{duration_to_latent_frames, snap_latent_frames};
    use crate::families::ace_step::lm_format::validate_audio_code_indices;

    #[test]
    fn duration_maps_to_25hz_latents() {
        assert_eq!(duration_to_latent_frames(5.0).unwrap(), 128);
        assert_eq!(duration_to_latent_frames(30.0).unwrap(), 750);
        assert_eq!(snap_latent_frames(751), 752);
    }

    #[test]
    fn audio_code_indices_reject_out_of_range() {
        assert!(validate_audio_code_indices(&[0, 100]).is_ok());
        assert!(validate_audio_code_indices(&[]).is_err());
        assert!(validate_audio_code_indices(&[64_000]).is_err());
    }
}
