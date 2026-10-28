//! ACE-Step text2music create orchestration (acestep.cpp `pipeline-synth`).
//!
//! Phases: text encode → cond pack → silence context → DiT denoise (APG CFG) →
//! optional CA+DTW lyric timeline → VAE → WAV.

use super::apg::broadcast_null_enc;
use super::ca_capture;
use super::cond;
use super::lyrics::{self, LyricsAlignment};
use super::lyrics_dtw;
use super::schedule::{shift_schedule, TURBO_SHIFT, TURBO_TIMESTEPS};
use super::text::{
    build_prompt_strings, embed_lyrics_with_ids, encode_caption, load_tokenizer_json,
    plain_token_ids, DIT_INSTR_TEXT2MUSIC,
};
use super::token_decode::{self, Decoder, LYRIC_HEADER_TEMPLATE};
use super::transformer::{load_weights, AceStepDit};
use super::wav::{self, SAMPLE_RATE};
use super::weights::{check_layout, config_from_weights};
use crate::common::ops::normal_latent;
use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{self, GgufWeights, WeightError};
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
}

#[derive(Debug)]
pub struct CreateOutput {
    pub wav: Vec<u8>,
    pub sample_rate: u32,
    pub frames: i64,
    pub steps: u32,
    pub enc_seq: i64,
    pub graph_computes: u32,
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

pub fn run_full_create(
    bundle: &Path,
    request: &CreateRequest<'_>,
) -> Result<CreateOutput, WeightError> {
    if !request.guidance.is_finite() || request.guidance < 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step guidance {} must be >= 1",
            request.guidance
        )));
    }
    let gguf = weights::assert_gguf_bundle(bundle)?;
    let dit_path = gguf.transformer.first().ok_or_else(|| {
        WeightError::NotBuilt("ACE-Step DiT GGUF is missing from the bundle".into())
    })?;
    let dit = GgufWeights::open(dit_path)?;
    let config = config_from_weights(&dit)?;
    check_layout(&dit, config)?;
    let is_turbo = config.is_turbo
        || dit_path
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.to_ascii_lowercase().contains("turbo"))
            .unwrap_or(false);

    let frames = snap_latent_frames(duration_to_latent_frames(request.duration_sec)?);
    let steps = request.steps.unwrap_or(if is_turbo { 8 } else { 50 });
    let schedule = if is_turbo && steps == 8 {
        TURBO_TIMESTEPS.to_vec()
    } else {
        let shift = if is_turbo { TURBO_SHIFT } else { 1.0 };
        shift_schedule(steps, shift)?
    };

    let duration_for_align = request.duration_sec.max(0.0);
    let (lyrics_effective, lyrics_alignment, lyrics_txt, lyrics_lrc) =
        lyrics::prepare_create_lyrics(request.lyrics, duration_for_align)
            .map_err(WeightError::NotBuilt)?;

    let tokenizer = load_tokenizer_json(&gguf.root)?;
    let pair = build_prompt_strings(
        request.prompt,
        lyrics_effective.as_ref(),
        DIT_INSTR_TEXT2MUSIC,
        request.duration_sec,
        request.bpm,
        request.keyscale,
        request.timesignature,
        request.vocal_language,
    );

    let ctx = GgmlContext::new(BackendKind::Cpu)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    let (text_hidden, s_text) =
        encode_caption(&ctx, &gguf.text_encoder, &tokenizer, &pair.text)?;
    let (lyric_embed, lyric_ids) =
        embed_lyrics_with_ids(&gguf.text_encoder, &tokenizer, &pair.lyric)?;
    let s_lyric = lyric_ids.len() as i64;
    let silence_full = load_silence_latent(&dit, config.out_channels)?;
    let timbre_owned;
    let timbre_feats: &[f32] = if let Some(lat) = request.timbre_latents.filter(|v| !v.is_empty()) {
        if lat.len() as i64 % cond::TIMBRE_IN != 0 {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step timbre_latents length {} is not a multiple of {}",
                lat.len(),
                cond::TIMBRE_IN
            )));
        }
        lat
    } else {
        // acestep.cpp: single silence frame disables distinctive timbre conditioning.
        timbre_owned = silence_full[..cond::TIMBRE_IN as usize].to_vec();
        &timbre_owned
    };
    let (enc_hidden, enc_seq) = cond::encode(
        &ctx,
        &dit,
        &text_hidden,
        s_text,
        &lyric_embed,
        s_lyric,
        Some(timbre_feats),
    )?;

    let context = if let Some(codes) = request.audio_code_indices.filter(|c| !c.is_empty()) {
        codes_context(
            &ctx,
            &dit,
            codes,
            frames,
            config.out_channels,
            config.in_channels,
        )?
    } else {
        silence_context(&dit, frames, config.out_channels, config.in_channels)?
    };
    let mut types = std::collections::HashMap::new();
    for (name, _, _) in super::graph::graph_weight_layout(config) {
        if let Ok(t) = dit.ggml_type_of(&name) {
            types.insert(name, t);
        }
    }
    let mut model = AceStepDit::build_typed(&ctx, config, frames, enc_seq, &types)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    load_weights(&mut model, &dit, config)?;

    let null_enc = if request.guidance > 1.0 + f32::EPSILON {
        let emb = load_null_condition(&dit, config.condition_dim)?;
        Some(
            broadcast_null_enc(&emb, enc_seq)
                .map_err(WeightError::NotBuilt)?,
        )
    } else {
        None
    };

    let seed = request.seed.unwrap_or(0);
    let xt = normal_latent(seed, model.xt_len());
    let latents = model
        .denoise(
            &xt,
            &context,
            &enc_hidden,
            &schedule,
            request.guidance,
            null_enc.as_deref(),
            None,
            None,
            |_, _| (),
        )
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;

    let (lyrics_alignment, lyrics_lrc) = maybe_upgrade_dit_alignment(
        &ctx,
        &dit,
        config,
        &gguf.text_encoder,
        &tokenizer,
        request.vocal_language,
        &lyric_ids,
        &latents,
        &context,
        &enc_hidden,
        frames,
        enc_seq,
        steps,
        seed,
        duration_for_align,
        lyrics_alignment,
        lyrics_lrc,
    )?;

    let vae_path = gguf
        .vae
        .first()
        .ok_or_else(|| WeightError::NotBuilt("ACE-Step VAE GGUF is missing".into()))?;
    let vae = GgufWeights::open(vae_path)?;
    let audio = super::vae::decode(&ctx, &vae, &latents, frames)?;
    let wav = wav::write_planar_stereo_pcm16(&audio, SAMPLE_RATE)?;

    Ok(CreateOutput {
        wav,
        sample_rate: SAMPLE_RATE,
        frames,
        steps,
        enc_seq,
        graph_computes: model.graph_computes(),
        lyrics_effective: lyrics_effective.into_owned(),
        lyrics_alignment,
        lyrics_txt,
        lyrics_lrc,
    })
}

/// Prefer DiT CA+DTW when layers_config is known; otherwise keep structure_estimate.
fn maybe_upgrade_dit_alignment(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    config: super::weights::AceStepConfig,
    text_paths: &[std::path::PathBuf],
    tokenizer: &str,
    vocal_language: &str,
    lyric_ids: &[i32],
    latents: &[f32],
    context: &[f32],
    enc_hidden: &[f32],
    frames: i64,
    enc_seq: i64,
    steps: u32,
    seed: i64,
    duration_sec: f32,
    fallback: LyricsAlignment,
    fallback_lrc: Option<String>,
) -> Result<(LyricsAlignment, Option<String>), WeightError> {
    if fallback.mode == "instrumental"
        || fallback.mode == "empty"
        || fallback.mode == "conditioning_only"
    {
        return Ok((fallback, fallback_lrc));
    }
    let Some(layers_config) = ca_capture::default_layers_config(config) else {
        return Ok((fallback, fallback_lrc));
    };
    let lang = if vocal_language.is_empty() {
        "unknown"
    } else {
        vocal_language
    };
    let header = LYRIC_HEADER_TEMPLATE.replace("{language}", lang);
    let header_ids = plain_token_ids(text_paths, tokenizer, &header)?;
    let Some((start, end)) = token_decode::pure_lyric_range(lyric_ids, header_ids.len()) else {
        return Ok((fallback, fallback_lrc));
    };
    let pure_ids = &lyric_ids[start..end];
    // Lyric tokens occupy enc[0..s_lyric); start/end are within that span.
    if end as i64 > enc_seq {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step lyric token end {end} exceeds enc_seq {enc_seq}"
        )));
    }
    let (attn, n_heads) = ca_capture::capture_selected_attention(
        ctx,
        dit,
        config,
        latents,
        context,
        enc_hidden,
        frames,
        enc_seq,
        steps,
        seed.wrapping_add(1),
        &layers_config,
    )?;
    let seq = frames / config.patch;
    // attn is [n_heads, enc_seq, seq] — slice lyric token span.
    let enc = enc_seq as usize;
    let seq_u = seq as usize;
    let mut lyric_attn = Vec::with_capacity(n_heads * pure_ids.len() * seq_u);
    for h in 0..n_heads {
        let base = h * enc * seq_u;
        for t in start..end {
            let off = base + t * seq_u;
            lyric_attn.extend_from_slice(&attn[off..off + seq_u]);
        }
    }
    let decoder = Decoder::from_tokenizer_json(tokenizer)?;
    let decode = |ids: &[i32]| decoder.decode(ids);
    let aligned = lyrics_dtw::align_lyrics_from_attention(
        &lyric_attn,
        n_heads,
        pure_ids,
        seq_u,
        duration_sec,
        &decode,
        2.0,
    )?;
    if aligned.segments.is_empty() {
        return Ok((fallback, fallback_lrc));
    }
    let lrc = lyrics::format_lrc(&aligned);
    Ok((aligned, lrc))
}

pub(crate) fn load_null_condition(
    dit: &GgufWeights,
    condition_dim: i64,
) -> Result<Vec<f32>, WeightError> {
    if !dit.contains("null_condition_emb") {
        return Err(WeightError::MissingTensor {
            path: "transformer".into(),
            name: "null_condition_emb".into(),
        });
    }
    // Packs store [2048], [2048,1], or [2048,1,1].
    for ne in [
        vec![condition_dim],
        vec![condition_dim, 1],
        vec![condition_dim, 1, 1],
    ] {
        if let Ok(values) = dit.f32_widened("null_condition_emb", &ne) {
            if values.len() as i64 == condition_dim {
                return Ok(values);
            }
        }
    }
    Err(WeightError::NotBuilt(format!(
        "null_condition_emb does not match condition_dim {condition_dim}"
    )))
}

/// Build DiT context from LM audio codes: FSQ detok → pad/truncate with silence,
/// mask = 1.0 (acestep.cpp `ops_build_context` codes path).
pub(crate) fn codes_context(
    ctx: &GgmlContext,
    dit: &GgufWeights,
    codes: &[i32],
    frames: i64,
    out_channels: i64,
    in_channels: i64,
) -> Result<Vec<f32>, WeightError> {
    super::lm_format::validate_audio_code_indices(codes)?;
    let decoded = super::fsq::detokenize(ctx, dit, codes)?;
    let silence = load_silence_latent(dit, out_channels)?;
    let ctx_channels = in_channels - out_channels;
    let mask_ch = ctx_channels - out_channels;
    if mask_ch != out_channels {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step context expects mask channels == out_channels ({out_channels}), got {mask_ch}"
        )));
    }
    let decoded_t = (decoded.len() as i64) / out_channels;
    let mut context = vec![0f32; (frames * ctx_channels) as usize];
    for t in 0..frames as usize {
        let src = if (t as i64) < decoded_t {
            let off = t * out_channels as usize;
            &decoded[off..off + out_channels as usize]
        } else {
            let silence_t = t - decoded_t as usize;
            let off = (silence_t * out_channels as usize)
                .min(silence.len().saturating_sub(out_channels as usize));
            &silence[off..off + out_channels as usize]
        };
        let dst = &mut context[t * ctx_channels as usize..(t + 1) * ctx_channels as usize];
        dst[..out_channels as usize].copy_from_slice(src);
        for c in 0..mask_ch as usize {
            dst[out_channels as usize + c] = 1.0;
        }
    }
    Ok(context)
}

pub(crate) fn silence_context(
    dit: &GgufWeights,
    frames: i64,
    out_channels: i64,
    in_channels: i64,
) -> Result<Vec<f32>, WeightError> {
    let ctx_channels = in_channels - out_channels;
    let mask_ch = ctx_channels - out_channels;
    if mask_ch != out_channels {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step context expects mask channels == out_channels ({out_channels}), got {mask_ch}"
        )));
    }
    let silence = load_silence_latent(dit, out_channels)?;
    let mut context = vec![0f32; (frames * ctx_channels) as usize];
    for t in 0..frames as usize {
        let src = &silence[t * out_channels as usize..(t + 1) * out_channels as usize];
        let dst = &mut context[t * ctx_channels as usize..(t + 1) * ctx_channels as usize];
        dst[..out_channels as usize].copy_from_slice(src);
        for c in 0..mask_ch as usize {
            dst[out_channels as usize + c] = 1.0;
        }
    }
    Ok(context)
}

pub(crate) fn load_silence_latent(
    dit: &GgufWeights,
    out_channels: i64,
) -> Result<Vec<f32>, WeightError> {
    if !dit.contains("silence_latent") {
        return Err(WeightError::MissingTensor {
            path: "transformer".into(),
            name: "silence_latent".into(),
        });
    }
    match dit.f32_widened("silence_latent", &[out_channels, SILENCE_MAX_FRAMES]) {
        Ok(values) if values.len() as i64 == out_channels * SILENCE_MAX_FRAMES => Ok(values),
        _ => {
            let alt = dit.f32_widened("silence_latent", &[SILENCE_MAX_FRAMES, out_channels])?;
            if alt.len() as i64 != out_channels * SILENCE_MAX_FRAMES {
                return Err(WeightError::NotBuilt(
                    "silence_latent does not match 15000×64".into(),
                ));
            }
            Ok(alt)
        }
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
