//! ACE-Step cover request types and host context packing.
//!
//! cover, cover-nofsq, repaint, lego, extract, and complete run on the MLX
//! session. `cover` quantizes the source latents through the FSQ codec.
//! Region masks and source-context packing stay here.

use super::create::CreateOutput;
use super::dcw::DcwSettings;
use super::repaint;
use super::text::{
    dit_instr_complete, dit_instr_extract, dit_instr_lego, DIT_INSTR_COVER, DIT_INSTR_REPAINT,
};
use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{LoraFile, WeightError};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverMode {
    /// Clean 25Hz VAE latents → DiT (no FSQ).
    CoverNofsq,
    /// FSQ tokenize/detokenize before DiT.
    Cover,
    /// Region regenerate; silence in zone + mask/splice (no FSQ).
    Repaint,
    /// Add a stem over source; cover everywhere + region mask/splice (no FSQ; base only).
    Lego,
    /// Isolate a stem from source (no FSQ; base only).
    Extract,
    /// Complete partial source with named stems (no FSQ; base only).
    Complete,
}

impl CoverMode {
    pub fn parse(raw: &str) -> Result<Self, WeightError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "cover-nofsq" | "nofsq" => Ok(Self::CoverNofsq),
            "cover" => Ok(Self::Cover),
            "repaint" => Ok(Self::Repaint),
            "lego" => Ok(Self::Lego),
            "extract" => Ok(Self::Extract),
            "complete" => Ok(Self::Complete),
            other => Err(WeightError::NotBuilt(format!(
                "ACE-Step audio.edit task '{other}' is unknown (use cover-nofsq, cover, repaint, lego, extract, or complete)"
            ))),
        }
    }

    pub(crate) fn dit_instruction(self, track: &str) -> Result<String, WeightError> {
        match self {
            Self::Cover | Self::CoverNofsq => Ok(DIT_INSTR_COVER.into()),
            Self::Repaint => Ok(DIT_INSTR_REPAINT.into()),
            Self::Lego => dit_instr_lego(track),
            Self::Extract => dit_instr_extract(track),
            Self::Complete => dit_instr_complete(track),
        }
    }

    pub(crate) fn is_region_task(self) -> bool {
        matches!(self, Self::Repaint | Self::Lego)
    }

    pub(crate) fn is_base_only(self) -> bool {
        matches!(self, Self::Lego | Self::Extract | Self::Complete)
    }

    fn task_name(self) -> &'static str {
        match self {
            Self::CoverNofsq => "cover-nofsq",
            Self::Cover => "cover",
            Self::Repaint => "repaint",
            Self::Lego => "lego",
            Self::Extract => "extract",
            Self::Complete => "complete",
        }
    }
}

pub struct CoverRequest<'a> {
    pub prompt: &'a str,
    pub lyrics: &'a str,
    pub duration_sec: Option<f32>,
    pub seed: Option<i64>,
    pub guidance: f32,
    pub steps: Option<u32>,
    pub vocal_language: &'a str,
    pub bpm: Option<i32>,
    pub keyscale: &'a str,
    pub timesignature: &'a str,
    pub mode: CoverMode,
    /// Interleaved stereo PCM f32 @ 48 kHz. Ignored when `source_latents` is set.
    pub source_audio: Option<&'a [f32]>,
    /// Time-major `[T, 64]` VAE latents (skips encode).
    pub source_latents: Option<&'a [f32]>,
    /// Fraction of DiT steps that keep cover context (1.0 = always). Ignored for repaint.
    pub audio_cover_strength: f32,
    /// 0 = pure noise init; 1 = start from source latents (schedule truncated).
    pub cover_noise_strength: f32,
    /// Repaint region start (seconds). Required when `mode == Repaint`.
    pub region_start_sec: Option<f32>,
    /// Repaint region end (seconds). Required when `mode == Repaint`.
    pub region_end_sec: Option<f32>,
    /// Fraction of steps that inject preserved regions (default 1.0).
    pub repaint_injection_ratio: f32,
    /// Latent frames of cosine crossfade at zone edges (default 8).
    pub repaint_crossfade_frames: i64,
    /// When true, sample a 750-frame timbre reference from source latents
    /// (MLX `cover_timbre_from_source` / `sample_timbre_reference_latents`).
    /// Default false → single silence frame (acestep.cpp).
    pub timbre_from_source: bool,
    /// Track name(s) for lego / extract / complete (`"drums | bass"` for complete).
    pub track: &'a str,
    pub loras: &'a [LoraFile],
    pub dcw: DcwSettings,
}

#[cfg(test)]
pub fn run_full_cover(
    bundle: &Path,
    request: &CoverRequest<'_>,
    backend: BackendKind,
) -> Result<CreateOutput, WeightError> {
    let ctx = GgmlContext::new(backend).map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    run_full_cover_with_context(bundle, request, &ctx, |_, _| {})
}

pub fn run_full_cover_with_context(
    bundle: &Path,
    request: &CoverRequest<'_>,
    ctx: &GgmlContext,
    mut on_step: impl FnMut(u32, u32),
) -> Result<CreateOutput, WeightError> {
    let _ = ctx;
    #[cfg(feature = "mlx")]
    {
        return super::mlx::session::cover(bundle, request, on_step);
    }
    #[cfg(not(feature = "mlx"))]
    {
        let _ = (bundle, request, on_step);
        Err(WeightError::NotBuilt(
            "ACE-Step cover runs on MLX. Rebuild danqing-api with the mlx feature.".into(),
        ))
    }
}

pub(crate) fn resolve_region_frames(
    request: &CoverRequest<'_>,
    duration_sec: f32,
    frames: i64,
    require_explicit: bool,
) -> Result<(i64, i64), WeightError> {
    let start = match request.region_start_sec {
        Some(v) => v,
        None if require_explicit => {
            return Err(WeightError::NotBuilt(
                "ACE-Step repaint needs region_start_sec".into(),
            ));
        }
        None => 0.0,
    };
    let end = match request.region_end_sec {
        Some(v) if v < 0.0 => duration_sec, // acestep.cpp / docs: -1 = end of audio
        Some(v) => v,
        None if require_explicit => {
            return Err(WeightError::NotBuilt(
                "ACE-Step repaint needs region_end_sec".into(),
            ));
        }
        None => duration_sec,
    };
    repaint::region_to_frames(start, end, frames)
}

/// Cover context: src_latents[64] + chunk_mask[64] (mask = 1.0 training distribution).
pub(crate) fn cover_context(
    cover_latents: &[f32],
    frames: i64,
    out_channels: i64,
    in_channels: i64,
) -> Result<Vec<f32>, WeightError> {
    build_source_context(
        cover_latents,
        None,
        frames,
        out_channels,
        in_channels,
        None,
        false,
    )
}

/// Repaint context: silence + mask 1 inside zone; source + mask 0 outside.
pub(crate) fn repaint_context(
    cover_latents: &[f32],
    silence: &[f32],
    frames: i64,
    out_channels: i64,
    in_channels: i64,
    t0: i64,
    t1: i64,
) -> Result<Vec<f32>, WeightError> {
    build_source_context(
        cover_latents,
        Some(silence),
        frames,
        out_channels,
        in_channels,
        Some((t0, t1)),
        true,
    )
}

/// Lego context: keep cover latents everywhere; mask 1 inside zone, 0 outside
/// (acestep.cpp `is_lego_region` — DiT hears the full backing track).
pub(crate) fn lego_context(
    cover_latents: &[f32],
    frames: i64,
    out_channels: i64,
    in_channels: i64,
    t0: i64,
    t1: i64,
) -> Result<Vec<f32>, WeightError> {
    build_source_context(
        cover_latents,
        None,
        frames,
        out_channels,
        in_channels,
        Some((t0, t1)),
        false,
    )
}

fn build_source_context(
    cover_latents: &[f32],
    silence: Option<&[f32]>,
    frames: i64,
    out_channels: i64,
    in_channels: i64,
    region: Option<(i64, i64)>,
    silence_in_region: bool,
) -> Result<Vec<f32>, WeightError> {
    let ctx_channels = in_channels - out_channels;
    let mask_ch = ctx_channels - out_channels;
    if mask_ch != out_channels {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step context expects mask channels == out_channels ({out_channels}), got {mask_ch}"
        )));
    }
    if cover_latents.len() as i64 != frames * out_channels {
        return Err(WeightError::Shape {
            name: "cover_latents".into(),
            expected: vec![frames, out_channels],
            got: vec![cover_latents.len() as i64],
        });
    }
    let mut context = vec![0f32; (frames * ctx_channels) as usize];
    let t_cover = frames as usize;
    for t in 0..frames as usize {
        let in_region = region
            .map(|(t0, t1)| t >= t0 as usize && t < t1 as usize)
            .unwrap_or(false);
        let src = if in_region && silence_in_region {
            let silence = silence.ok_or_else(|| {
                WeightError::NotBuilt("ACE-Step repaint context needs silence_latent".into())
            })?;
            let off = (t * out_channels as usize)
                .min(silence.len().saturating_sub(out_channels as usize));
            &silence[off..off + out_channels as usize]
        } else {
            let t_src = t.min(t_cover.saturating_sub(1));
            &cover_latents[t_src * out_channels as usize..(t_src + 1) * out_channels as usize]
        };
        let mask_val = if region.is_some() {
            if in_region {
                1.0
            } else {
                0.0
            }
        } else {
            1.0
        };
        let dst = &mut context[t * ctx_channels as usize..(t + 1) * ctx_channels as usize];
        dst[..out_channels as usize].copy_from_slice(src);
        for c in 0..mask_ch as usize {
            dst[out_channels as usize + c] = mask_val;
        }
    }
    Ok(context)
}

fn pad_enc(
    enc: &[f32],
    enc_seq: i64,
    target_seq: i64,
    condition_dim: i64,
) -> Result<Vec<f32>, WeightError> {
    if enc.len() as i64 != enc_seq * condition_dim {
        return Err(WeightError::Shape {
            name: "enc_hidden".into(),
            expected: vec![enc_seq, condition_dim],
            got: vec![enc.len() as i64],
        });
    }
    if enc_seq == target_seq {
        return Ok(enc.to_vec());
    }
    if enc_seq > target_seq {
        return Err(WeightError::NotBuilt(
            "ACE-Step enc_seq exceeds graph target".into(),
        ));
    }
    let mut out = vec![0f32; (target_seq * condition_dim) as usize];
    out[..enc.len()].copy_from_slice(enc);
    Ok(out)
}

fn apply_cover_noise_blend(
    xt: &mut [f32],
    cover_latents: &[f32],
    frames: i64,
    out_channels: i64,
    schedule: &mut Vec<f32>,
    cover_noise_strength: f32,
) -> Result<(), WeightError> {
    if schedule.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover noise blend needs a non-empty schedule".into(),
        ));
    }
    let effective_noise = 1.0 - cover_noise_strength;
    let mut start_idx = 0usize;
    let mut best = (schedule[0] - effective_noise).abs();
    for (i, &t) in schedule.iter().enumerate().skip(1) {
        let dist = (t - effective_noise).abs();
        if dist < best {
            best = dist;
            start_idx = i;
        }
    }
    let nearest_t = schedule[start_idx];
    let t_cover = frames as usize;
    for t in 0..frames as usize {
        let t_src = t.min(t_cover.saturating_sub(1));
        for c in 0..out_channels as usize {
            let idx = t * out_channels as usize + c;
            let src = cover_latents[t_src * out_channels as usize + c];
            xt[idx] = nearest_t * xt[idx] + (1.0 - nearest_t) * src;
        }
    }
    schedule.drain(0..start_idx);
    if schedule.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover noise blend emptied the schedule".into(),
        ));
    }
    Ok(())
}

mod tests {
    use super::{cover_context, lego_context, repaint_context, CoverMode};
    const LATENT_CHANNELS: i64 = 64;

    #[test]
    fn cover_mode_parse() {
        assert_eq!(CoverMode::parse("").unwrap(), CoverMode::CoverNofsq);
        assert_eq!(
            CoverMode::parse("cover-nofsq").unwrap(),
            CoverMode::CoverNofsq
        );
        assert_eq!(CoverMode::parse("cover").unwrap(), CoverMode::Cover);
        assert_eq!(CoverMode::parse("repaint").unwrap(), CoverMode::Repaint);
        assert_eq!(CoverMode::parse("lego").unwrap(), CoverMode::Lego);
        assert_eq!(CoverMode::parse("extract").unwrap(), CoverMode::Extract);
        assert_eq!(CoverMode::parse("complete").unwrap(), CoverMode::Complete);
        assert!(CoverMode::parse("flow-edit")
            .unwrap_err()
            .to_string()
            .contains("unknown"));
    }

    #[test]
    fn repaint_context_silences_zone_and_masks() {
        let frames = 4i64;
        let oc = LATENT_CHANNELS;
        let lat = vec![0.5f32; (frames * oc) as usize];
        let silence = vec![0.0f32; (frames * oc) as usize];
        let ctx = repaint_context(&lat, &silence, frames, oc, oc * 3, 1, 3).unwrap();
        // t=0 outside: source + mask 0
        assert_eq!(ctx[0], 0.5);
        assert_eq!(ctx[oc as usize], 0.0);
        // t=1 inside: silence + mask 1
        assert_eq!(ctx[(oc * 2) as usize], 0.0);
        assert_eq!(ctx[(oc * 3) as usize], 1.0);
    }

    #[test]
    fn lego_context_keeps_cover_inside_zone() {
        let frames = 4i64;
        let oc = LATENT_CHANNELS;
        let lat = vec![0.5f32; (frames * oc) as usize];
        let ctx = lego_context(&lat, frames, oc, oc * 3, 1, 3).unwrap();
        // t=0 outside: source + mask 0
        assert_eq!(ctx[0], 0.5);
        assert_eq!(ctx[oc as usize], 0.0);
        // t=1 inside: still source (not silence) + mask 1
        assert_eq!(ctx[(oc * 2) as usize], 0.5);
        assert_eq!(ctx[(oc * 3) as usize], 1.0);
    }

    #[test]
    fn cover_context_packs_src_and_ones_mask() {
        let frames = 2i64;
        let oc = LATENT_CHANNELS;
        let mut lat = vec![0f32; (frames * oc) as usize];
        lat[0] = 0.5;
        lat[oc as usize + 1] = -0.25;
        let ctx = cover_context(&lat, frames, oc, oc * 3).unwrap();
        assert_eq!(ctx.len(), (frames * oc * 2) as usize);
        assert_eq!(ctx[0], 0.5);
        assert_eq!(ctx[oc as usize], 1.0);
        assert_eq!(ctx[(oc * 2) as usize + 1], -0.25);
        assert_eq!(ctx[(oc * 3) as usize], 1.0);
    }
}
