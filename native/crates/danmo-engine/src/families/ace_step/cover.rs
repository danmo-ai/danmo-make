//! ACE-Step cover-nofsq (acestep.cpp cover without FSQ roundtrip).
//!
//! Encode source audio (or accept latents) → cover context (src[64]+mask[64]) →
//! DiT Euler (optional cover_strength switch + noise blend) → VAE decode → WAV.
//! Full `cover` (FSQ tokenize/detokenize) fails loud until FSQ is ported.

use super::apg::broadcast_null_enc;
use super::cond;
use super::create::{
    load_null_condition, silence_context, CreateOutput, SILENCE_MAX_FRAMES,
};
use super::lyrics;
use super::schedule::{shift_schedule, TURBO_SHIFT, TURBO_TIMESTEPS};
use super::text::{
    build_prompt_strings, embed_lyrics, encode_caption, load_tokenizer_json, DIT_INSTR_COVER,
    DIT_INSTR_TEXT2MUSIC,
};
use super::transformer::{load_weights, AceStepDit, DenoiseContextSwitch};
use super::vae::{AUDIO_UPSAMPLE, LATENT_CHANNELS};
use super::wav::{self, SAMPLE_RATE};
use super::weights::{check_layout, config_from_weights};
use crate::common::ops::normal_latent;
use crate::runtime::{BackendKind, GgmlContext};
use crate::weights::{self, GgufWeights, WeightError};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverMode {
    /// Clean 25Hz VAE latents → DiT (no FSQ).
    CoverNofsq,
    /// Needs FSQ roundtrip — not built yet.
    Cover,
}

impl CoverMode {
    pub fn parse(raw: &str) -> Result<Self, WeightError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "cover-nofsq" | "nofsq" => Ok(Self::CoverNofsq),
            "cover" => Ok(Self::Cover),
            other => Err(WeightError::NotBuilt(format!(
                "ACE-Step audio.edit task '{other}' is unknown (use cover-nofsq or cover)"
            ))),
        }
    }
}

pub struct CoverRequest<'a> {
    pub prompt: &'a str,
    pub lyrics: &'a str,
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
    /// Fraction of DiT steps that keep cover context (1.0 = always).
    pub audio_cover_strength: f32,
    /// 0 = pure noise init; 1 = start from source latents (schedule truncated).
    pub cover_noise_strength: f32,
}

pub fn run_full_cover(
    bundle: &Path,
    request: &CoverRequest<'_>,
) -> Result<CreateOutput, WeightError> {
    if request.mode == CoverMode::Cover {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover needs FSQ roundtrip (acestep.cpp ops_fsq_roundtrip); use task=cover-nofsq"
                .into(),
        ));
    }
    if !request.guidance.is_finite() || request.guidance < 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step guidance {} must be >= 1",
            request.guidance
        )));
    }
    if !request.audio_cover_strength.is_finite()
        || !(0.0..=1.0).contains(&request.audio_cover_strength)
    {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step audio_cover_strength {} must be in [0, 1]",
            request.audio_cover_strength
        )));
    }
    if !request.cover_noise_strength.is_finite()
        || !(0.0..=1.0).contains(&request.cover_noise_strength)
    {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cover_noise_strength {} must be in [0, 1]",
            request.cover_noise_strength
        )));
    }

    let cover_latents = resolve_cover_latents(bundle, request)?;
    let frames = cover_latents.len() as i64 / LATENT_CHANNELS;
    if frames < 1 || cover_latents.len() as i64 != frames * LATENT_CHANNELS {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover latents must be a non-empty multiple of 64".into(),
        ));
    }
    if frames > SILENCE_MAX_FRAMES {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cover frames {frames} exceed silence_latent max {SILENCE_MAX_FRAMES}"
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

    let mut steps = request.steps.unwrap_or(if is_turbo { 8 } else { 50 });
    let mut schedule = if is_turbo && steps == 8 {
        TURBO_TIMESTEPS.to_vec()
    } else {
        let shift = if is_turbo { TURBO_SHIFT } else { 1.0 };
        shift_schedule(steps, shift)?
    };

    let duration_sec = frames as f32 / 25.0;
    let (lyrics_effective, lyrics_alignment, lyrics_txt, lyrics_lrc) =
        lyrics::prepare_create_lyrics(request.lyrics, duration_sec).map_err(WeightError::NotBuilt)?;

    let tokenizer = load_tokenizer_json(&gguf.root)?;
    let pair_cover = build_prompt_strings(
        request.prompt,
        lyrics_effective.as_ref(),
        DIT_INSTR_COVER,
        duration_sec,
        request.bpm,
        request.keyscale,
        request.timesignature,
        request.vocal_language,
    );

    let ctx = GgmlContext::new(BackendKind::Cpu)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    let (text_hidden, s_text) =
        encode_caption(&ctx, &gguf.text_encoder, &tokenizer, &pair_cover.text)?;
    let (lyric_embed, s_lyric) = embed_lyrics(&gguf.text_encoder, &tokenizer, &pair_cover.lyric)?;
    let (enc_hidden, enc_seq) =
        cond::encode(&ctx, &dit, &text_hidden, s_text, &lyric_embed, s_lyric)?;

    let need_switch = request.audio_cover_strength < 1.0 - f32::EPSILON;
    let (enc_text2music, enc_seq_t2m) = if need_switch {
        let pair_t2m = build_prompt_strings(
            request.prompt,
            lyrics_effective.as_ref(),
            DIT_INSTR_TEXT2MUSIC,
            duration_sec,
            request.bpm,
            request.keyscale,
            request.timesignature,
            request.vocal_language,
        );
        let (th, st) = encode_caption(&ctx, &gguf.text_encoder, &tokenizer, &pair_t2m.text)?;
        let (le, sl) = embed_lyrics(&gguf.text_encoder, &tokenizer, &pair_t2m.lyric)?;
        let (enc, seq) = cond::encode(&ctx, &dit, &th, st, &le, sl)?;
        (Some(enc), seq)
    } else {
        (None, enc_seq)
    };
    let enc_seq_graph = enc_seq.max(enc_seq_t2m);

    let context = cover_context(&cover_latents, frames, config.out_channels, config.in_channels)?;
    let context_silence = if need_switch {
        Some(silence_context(
            &dit,
            frames,
            config.out_channels,
            config.in_channels,
        )?)
    } else {
        None
    };

    let mut types = std::collections::HashMap::new();
    for (name, _, _) in super::graph::graph_weight_layout(config) {
        if let Ok(t) = dit.ggml_type_of(&name) {
            types.insert(name, t);
        }
    }
    let mut model = AceStepDit::build_typed(&ctx, config, frames, enc_seq_graph, &types)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    load_weights(&mut model, &dit, config)?;

    let enc_hidden = pad_enc(&enc_hidden, enc_seq, enc_seq_graph, config.condition_dim)?;
    let enc_text2music = enc_text2music
        .map(|e| pad_enc(&e, enc_seq_t2m, enc_seq_graph, config.condition_dim))
        .transpose()?;

    let null_enc = if request.guidance > 1.0 + f32::EPSILON {
        let emb = load_null_condition(&dit, config.condition_dim)?;
        Some(
            broadcast_null_enc(&emb, enc_seq_graph).map_err(WeightError::NotBuilt)?,
        )
    } else {
        None
    };

    let seed = request.seed.unwrap_or(0);
    let mut xt = normal_latent(seed, model.xt_len());
    if request.cover_noise_strength > 0.0 {
        apply_cover_noise_blend(
            &mut xt,
            &cover_latents,
            frames,
            config.out_channels,
            &mut schedule,
            request.cover_noise_strength,
        )?;
        steps = schedule.len() as u32;
    }

    let cover_steps = if need_switch {
        ((schedule.len() as f32) * request.audio_cover_strength) as u32
    } else {
        u32::MAX
    };
    let switch = match (&context_silence, &enc_text2music) {
        (Some(silence), Some(enc_t2m)) if need_switch => Some(DenoiseContextSwitch {
            context_silence: silence.as_slice(),
            enc_hidden_text2music: enc_t2m.as_slice(),
            cover_steps,
        }),
        _ => None,
    };

    let latents = model
        .denoise(
            &xt,
            &context,
            &enc_hidden,
            &schedule,
            request.guidance,
            null_enc.as_deref(),
            switch,
            |_, _| (),
        )
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;

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
        enc_seq: enc_seq_graph,
        graph_computes: model.graph_computes(),
        lyrics_effective: lyrics_effective.into_owned(),
        lyrics_alignment,
        lyrics_txt,
        lyrics_lrc,
    })
}

fn resolve_cover_latents(
    bundle: &Path,
    request: &CoverRequest<'_>,
) -> Result<Vec<f32>, WeightError> {
    if let Some(latents) = request.source_latents {
        if latents.is_empty() || latents.len() as i64 % LATENT_CHANNELS != 0 {
            return Err(WeightError::NotBuilt(
                "ACE-Step source_latents must be a non-empty multiple of 64".into(),
            ));
        }
        return Ok(latents.to_vec());
    }
    let Some(audio) = request.source_audio else {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover-nofsq needs source_audio or source_latents".into(),
        ));
    };
    if audio.len() < 2 || audio.len() % 2 != 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step source_audio must be interleaved stereo".into(),
        ));
    }
    let t_audio = (audio.len() / 2) as i64;
    let gguf = weights::assert_gguf_bundle(bundle)?;
    let vae_path = gguf
        .vae
        .first()
        .ok_or_else(|| WeightError::NotBuilt("ACE-Step VAE GGUF is missing".into()))?;
    let vae = GgufWeights::open(vae_path)?;
    let ctx = GgmlContext::new(BackendKind::Cpu)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    // Short clips encode directly; longer use tiled path (chunk=256, overlap=64).
    if t_audio <= 256 * AUDIO_UPSAMPLE {
        super::vae_enc::encode(&ctx, &vae, audio, t_audio)
    } else {
        super::vae_enc::encode_tiled(&ctx, &vae, audio, t_audio, 256, 64)
    }
}

/// Cover context: src_latents[64] + chunk_mask[64] (mask = 1.0 training distribution).
pub(crate) fn cover_context(
    cover_latents: &[f32],
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
    if cover_latents.len() as i64 != frames * out_channels {
        return Err(WeightError::Shape {
            name: "cover_latents".into(),
            expected: vec![frames, out_channels],
            got: vec![cover_latents.len() as i64],
        });
    }
    let mut context = vec![0f32; (frames * ctx_channels) as usize];
    for t in 0..frames as usize {
        let src = &cover_latents[t * out_channels as usize..(t + 1) * out_channels as usize];
        let dst = &mut context[t * ctx_channels as usize..(t + 1) * ctx_channels as usize];
        dst[..out_channels as usize].copy_from_slice(src);
        for c in 0..mask_ch as usize {
            dst[out_channels as usize + c] = 1.0;
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

/// Encode helper used by tests / callers that already hold interleaved PCM.
#[allow(dead_code)]
pub fn encode_source_audio(
    bundle: &Path,
    interleaved: &[f32],
) -> Result<Vec<f32>, WeightError> {
    resolve_cover_latents(
        bundle,
        &CoverRequest {
            prompt: "",
            lyrics: "[inst]",
            seed: None,
            guidance: 1.0,
            steps: None,
            vocal_language: "en",
            bpm: None,
            keyscale: "",
            timesignature: "",
            mode: CoverMode::CoverNofsq,
            source_audio: Some(interleaved),
            source_latents: None,
            audio_cover_strength: 1.0,
            cover_noise_strength: 0.0,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::{cover_context, CoverMode};
    use crate::families::ace_step::vae::LATENT_CHANNELS;

    #[test]
    fn cover_mode_parse() {
        assert_eq!(CoverMode::parse("").unwrap(), CoverMode::CoverNofsq);
        assert_eq!(
            CoverMode::parse("cover-nofsq").unwrap(),
            CoverMode::CoverNofsq
        );
        assert_eq!(CoverMode::parse("cover").unwrap(), CoverMode::Cover);
        assert!(CoverMode::parse("lego").unwrap_err().to_string().contains("unknown"));
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
