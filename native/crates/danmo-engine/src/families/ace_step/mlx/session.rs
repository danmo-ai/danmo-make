//! Safetensors ACE-Step create. The DiT, Qwen3 embedding, and Oobleck VAE
//! stay on the MLX executor. Host code keeps lyrics, the flow sampler, DCW,
//! and the WAV writer.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use super::super::cover::{self, CoverMode, CoverRequest};
use super::super::create::{
    duration_to_latent_frames, snap_latent_frames, CreateOutput, CreateRequest,
};
use super::super::lyrics;
use super::super::lyrics_dtw;
use super::super::quality;
use super::super::text::{self, DIT_INSTR_TEXT2MUSIC};
use super::super::token_decode::{Decoder, LYRIC_HEADER_TEMPLATE, QWEN_ENDOFTEXT_ID};
use super::super::wav::{self, SAMPLE_RATE};
use super::model::{self, AceConfig};
use super::vae::{self, VaeSpec};
use super::ModelVariant;
use crate::common::ops::normal_latent;
use crate::common::qwen3::{self, Qwen3Config};
use crate::weights::{LoraFile, MlxWeights, WeightDtype, WeightError};
use danmo_mlx::{Array, CacheKey, CacheRevision, Dtype, ModelCache, Stream};

const SILENCE_FRAMES: usize = 15_000;
const SILENCE_CHANNELS: usize = 64;

struct Encoded {
    encoder: Vec<f32>,
    enc_len: usize,
    enc_dim: usize,
    context: Vec<f32>,
    silence: Vec<f32>,
    load_secs: f64,
    encode_secs: f64,
}

struct CoverEncoded {
    encoder: Vec<f32>,
    enc_len: usize,
    enc_dim: usize,
    non_cover: Option<Vec<f32>>,
    load_secs: f64,
    encode_secs: f64,
}

pub fn create(
    bundle: &Path,
    request: &CreateRequest<'_>,
    mut on_step: impl FnMut(u32, u32),
) -> Result<CreateOutput, WeightError> {
    if !request.guidance.is_finite() || request.guidance < 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step guidance {} must be >= 1",
            request.guidance
        )));
    }
    let loras = request.loras.to_vec();
    let codes = request.audio_code_indices.unwrap_or(&[]).to_vec();
    if !codes.is_empty() {
        super::super::lm_format::validate_audio_code_indices(&codes)?;
    }
    let layout = locate(bundle)?;
    let config =
        AceConfig::from_json(&std::fs::read_to_string(&layout.config).map_err(|error| {
            WeightError::Open {
                path: layout.config.display().to_string(),
                detail: error.to_string(),
            }
        })?)
        .map_err(WeightError::NotBuilt)?;
    if config.hidden != 2560 || config.layers != 32 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step MLX create accepts the XL checkpoint (hidden 2560, 32 layers), got hidden {} and {} layers",
            config.hidden, config.layers
        )));
    }
    let vae_spec = VaeSpec::from_json(&std::fs::read_to_string(&layout.vae_config).map_err(
        |error| WeightError::Open {
            path: layout.vae_config.display().to_string(),
            detail: error.to_string(),
        },
    )?)
    .map_err(WeightError::NotBuilt)?;
    let qwen = qwen_config(&layout.text)?;
    if qwen.hidden as i32 != config.text_hidden {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step text encoder hidden {} does not match text_hidden_dim {}",
            qwen.hidden, config.text_hidden
        )));
    }
    let frames = snap_latent_frames(duration_to_latent_frames(request.duration_sec)?) as usize;
    let (lyrics_effective, lyrics_alignment, lyrics_txt, _lyrics_lrc) =
        lyrics::prepare_create_lyrics(request.lyrics, request.duration_sec.max(0.0))
            .map_err(WeightError::NotBuilt)?;
    let prompts = text::build_prompt_strings(
        request.prompt,
        lyrics_effective.as_ref(),
        DIT_INSTR_TEXT2MUSIC,
        request.duration_sec,
        request.bpm,
        request.keyscale,
        request.timesignature,
        request.vocal_language,
    );
    let tokenizer =
        std::fs::read_to_string(layout.text.join("tokenizer.json")).map_err(|error| {
            WeightError::Open {
                path: layout.text.join("tokenizer.json").display().to_string(),
                detail: error.to_string(),
            }
        })?;
    let text_ids = qwen3::tokenize_plain(&tokenizer, &prompts.text, qwen.vocab)?;
    let lyric_ids = qwen3::tokenize_plain(&tokenizer, &prompts.lyric, qwen.vocab)?;
    let silence = layout.silence.clone();
    let seed = request.seed.unwrap_or_else(draw_seed);
    let steps = request
        .steps
        .unwrap_or(if config.is_turbo { 8 } else { 30 });
    let dit_key = cache_key(&layout.dit)?;
    let text_key = cache_key(&layout.text)?;
    let vae_key = cache_key(&layout.vae)?;
    let dcw = request.dcw;
    let guidance = request.guidance;
    let is_turbo = config.is_turbo;
    let lm_expanded = request.lm_expanded;
    let pmi_bonus = request.pmi_bonus;
    let timbre = request.timbre_latents.map(|values| values.to_vec());

    let started = std::time::Instant::now();
    let encoded = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step encode", {
            let dit_dir = layout.dit.clone();
            let text_dir = layout.text.clone();
            let config = config.clone();
            let silence = silence.clone();
            let text_ids = text_ids.clone();
            let lyric_ids = lyric_ids.clone();
            let dit_key = dit_key.clone();
            let text_key = text_key.clone();
            let codes = codes.clone();
            let loras = loras.clone();
            move |_, stream, cache, _| {
                let dit = resident_dit(cache, dit_key, &dit_dir, &loras, stream)?;
                let text = resident(cache, text_key, &text_dir, true, stream)?;
                let load_secs = started.elapsed().as_secs_f64();
                let text_hidden = encode_text(&text, qwen, &text_ids, stream)?;
                let lyric_hidden = embed_tokens(&text, &lyric_ids, qwen.hidden as i32, stream)?;
                let timbre_hidden = timbre_frames(&silence, timbre.as_deref(), stream)?;
                let encoder = model::encode_condition(
                    &dit,
                    &config,
                    &text_hidden,
                    &lyric_hidden,
                    &timbre_hidden,
                    stream,
                )?;
                encoder.eval()?;
                let context = if codes.is_empty() {
                    silence_context(&silence, frames, stream)?
                } else {
                    let hints = model::detokenize_audio_codes(&dit, &config, &codes, stream)?;
                    hints.eval()?;
                    let packed = code_context(&hints.to_vec_f32(stream)?, &silence, frames)?;
                    Array::from_f32_slice(
                        &packed,
                        &[1, frames as i32, (SILENCE_CHANNELS * 2) as i32],
                    )?
                };
                context.eval()?;
                let encode_secs = started.elapsed().as_secs_f64() - load_secs;
                Ok(Encoded {
                    encoder: encoder.to_vec_f32(stream)?,
                    enc_len: encoder.shape()[1] as usize,
                    enc_dim: encoder.shape()[2] as usize,
                    context: context.to_vec_f32(stream)?,
                    silence: silence_reference(&silence, frames),
                    load_secs,
                    encode_secs,
                })
            }
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let max_attempts = if lm_expanded { 6 } else { 3 };
    let mut attempt = 0u32;
    let denoise_started = std::time::Instant::now();
    let (latents, attempts) = loop {
        let run_seed = quality::diffusion_retry_seed(seed, attempt);
        let noise = normal_latent(run_seed, frames * config.out_channels as usize);
        let mut sample = super::SampleRequest::new(
            if is_turbo {
                ModelVariant::Turbo
            } else {
                ModelVariant::Base
            },
            1,
            frames,
            config.out_channels as usize,
            &noise,
        );
        sample.guidance = guidance;
        sample.cfg_distilled = is_turbo;
        sample.infer_steps = Some(steps);
        sample.shift = if is_turbo { 3.0 } else { 1.0 };
        sample.dcw = Some(dcw);
        let dit_key = dit_key.clone();
        let dit_dir = layout.dit.clone();
        let config_step = config.clone();
        let context_step = encoded.context.clone();
        let encoder_step = encoded.encoder.clone();
        let enc_len = encoded.enc_len;
        let enc_dim = encoded.enc_dim;
        let latents = super::sample_with(
            &sample,
            || {
                Err(WeightError::NotBuilt(
                    "ACE-Step SDE noise is not drawn on this path".into(),
                ))
            },
            |step| {
                let conditional = step.conditional;
                let timestep = step.timestep;
                let xt = step.latents.to_vec();
                let dit_key = dit_key.clone();
                let dit_dir = dit_dir.clone();
                let config_step = config_step.clone();
                let context_step = context_step.clone();
                let encoder_step = encoder_step.clone();
                let enc_len = enc_len as i32;
                let enc_dim = enc_dim as i32;
                let loras = loras.clone();
                let values = crate::runtime::mlx_executor()
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?
                    .execute("ace_step velocity", move |_, stream, cache, _| {
                        let dit = resident_dit(cache, dit_key, &dit_dir, &loras, stream)?;
                        let xt = Array::from_f32_slice(
                            &xt,
                            &[1, frames as i32, config_step.out_channels],
                        )?;
                        let context = Array::from_f32_slice(
                            &context_step,
                            &[
                                1,
                                frames as i32,
                                config_step.in_channels - config_step.out_channels,
                            ],
                        )?;
                        let encoded = if conditional {
                            Array::from_f32_slice(&encoder_step, &[1, enc_len, enc_dim])?
                        } else {
                            null_encoder(&dit, enc_len, enc_dim, stream)?
                        };
                        let velocity = model::velocity(
                            &dit,
                            &config_step,
                            &xt,
                            &context,
                            &encoded,
                            timestep,
                            stream,
                        )?;
                        velocity.eval()?;
                        let values = velocity.to_vec_f32(stream)?;
                        let _ = danmo_mlx::clear_cache();
                        Ok(values)
                    })
                    .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(WeightError::NotBuilt(
                        "ACE-Step velocity is not finite".into(),
                    ));
                }
                Ok(values)
            },
            |done, total| on_step(done, total),
        )?;
        let (near, _, _) = quality::latents_collapsed_to_silence(&latents, &encoded.silence)?;
        attempt += 1;
        if !near || attempt >= max_attempts {
            break (latents, attempt);
        }
    };
    let denoise_secs = denoise_started.elapsed().as_secs_f64();
    let (lyrics_alignment, lyrics_lrc) = align_sung_lyrics(
        &layout.dit,
        dit_key.clone(),
        &config,
        &tokenizer,
        &lyric_ids,
        qwen.vocab,
        request.vocal_language,
        &latents,
        &encoded.context,
        &encoded.encoder,
        encoded.enc_len,
        encoded.enc_dim,
        frames,
        config.out_channels,
        config.in_channels - config.out_channels,
        steps,
        lyrics_alignment,
        &loras,
    )?;
    let vae_started = std::time::Instant::now();
    let audio = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step vae", {
            let vae_key = vae_key.clone();
            let vae_dir = layout.vae.clone();
            let latents = latents.clone();
            let spec = vae_spec.clone();
            let channels = config.out_channels;
            move |_, stream, cache, _| {
                let vae_weights = resident(cache, vae_key, &vae_dir, false, stream)?;
                let latent = Array::from_f32_slice(&latents, &[1, frames as i32, channels])?;
                vae::decode_planar(&vae_weights, &spec, &latent, stream)
            }
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let vae_secs = vae_started.elapsed().as_secs_f64();
    let load_secs = encoded.load_secs;
    let encode_secs = encoded.encode_secs;
    let samples = audio.len() / 2;
    let expected = frames * vae_spec.hop() as usize;
    if samples != expected {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step VAE wrote {samples} samples, expected {expected} ({frames} latent frames × hop {})",
            vae_spec.hop()
        )));
    }
    let audio = quality::normalize_planar_stereo(&audio, samples as i64)?;
    let assessed = quality::assess_output(
        &latents,
        &encoded.silence,
        &audio,
        SAMPLE_RATE,
        lm_expanded,
        pmi_bonus,
        attempts,
    )?;
    let wav = wav::write_planar_stereo_pcm16(&audio, SAMPLE_RATE)?;
    Ok(CreateOutput {
        wav,
        sample_rate: SAMPLE_RATE,
        frames: frames as i64,
        steps,
        enc_seq: encoded.enc_len as i64,
        graph_computes: steps,
        dcw,
        dcw_active: dcw.active(config.is_turbo),
        quality: assessed,
        timings: vec![
            ("dit_load", load_secs),
            ("text", encode_secs),
            ("denoise", denoise_secs),
            ("vae", vae_secs),
        ],
        lyrics_effective: lyrics_effective.into_owned(),
        lyrics_alignment,
        lyrics_txt,
        lyrics_lrc,
    })
}

pub fn cover(
    bundle: &Path,
    request: &CoverRequest<'_>,
    mut on_step: impl FnMut(u32, u32),
) -> Result<CreateOutput, WeightError> {
    let loras = request.loras.to_vec();
    if !request.guidance.is_finite() || request.guidance < 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step guidance {} must be >= 1",
            request.guidance
        )));
    }
    if !request.audio_cover_strength.is_finite() {
        return Err(WeightError::NotBuilt(
            "ACE-Step audio_cover_strength must be finite".into(),
        ));
    }
    let strength = request.audio_cover_strength.clamp(0.0, 1.0);
    let layout = locate(bundle)?;
    let config = load_ace_config(&layout)?;
    if config.hidden != 2560 || config.layers != 32 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step MLX cover accepts the XL checkpoint (hidden 2560, 32 layers), got hidden {} and {} layers",
            config.hidden, config.layers
        )));
    }
    if request.mode.is_base_only() && config.is_turbo {
        return Err(WeightError::NotBuilt(
            "ACE-Step lego, extract, and complete require the XL base or SFT checkpoint".into(),
        ));
    }
    let vae_spec = load_vae_spec(&layout)?;
    let qwen = qwen_config(&layout.text)?;
    if qwen.hidden as i32 != config.text_hidden {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step text encoder hidden {} does not match text_hidden_dim {}",
            qwen.hidden, config.text_hidden
        )));
    }
    let silence = layout.silence.clone();
    let source = source_latents(request, &vae_spec, &layout.vae, &silence)?;
    let src_frames = source.len() / SILENCE_CHANNELS;
    let requested = match request.duration_sec {
        Some(seconds) => snap_latent_frames(duration_to_latent_frames(seconds)?) as usize,
        None => snap_latent_frames(src_frames as i64) as usize,
    };
    let frames = requested.max(src_frames);
    if frames > SILENCE_FRAMES {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cover latent frames {frames} exceed silence_latent max {SILENCE_FRAMES}"
        )));
    }
    let fitted = fit_latents(&source, &silence, frames);
    let acoustic = if request.mode == CoverMode::Cover {
        fsq_cover_acoustic(&layout.dit, &loras, &config, &fitted, &silence, frames)?
    } else {
        fitted.clone()
    };
    let (region, mask) = if request.mode.is_region_task() {
        let (start, end) =
            cover::resolve_region_frames(request, frames as f32 / 25.0, frames as i64, true)?;
        let mut mask = vec![false; frames];
        for frame in start..end {
            mask[frame as usize] = true;
        }
        (Some((start, end)), Some(mask))
    } else {
        (None, None)
    };
    let context = match request.mode {
        CoverMode::Repaint => {
            let (start, end) = region.expect("repaint region");
            cover::repaint_context(
                &acoustic,
                &silence,
                frames as i64,
                config.out_channels as i64,
                config.in_channels as i64,
                start,
                end,
            )?
        }
        CoverMode::Lego => {
            let (start, end) = region.expect("lego region");
            cover::lego_context(
                &acoustic,
                frames as i64,
                config.out_channels as i64,
                config.in_channels as i64,
                start,
                end,
            )?
        }
        _ => cover::cover_context(
            &acoustic,
            frames as i64,
            config.out_channels as i64,
            config.in_channels as i64,
        )?,
    };
    let duration_sec = frames as f32 / 25.0;
    let (lyrics_effective, lyrics_alignment, lyrics_txt, _lyrics_lrc) =
        lyrics::prepare_create_lyrics(request.lyrics, duration_sec)
            .map_err(WeightError::NotBuilt)?;
    let instruction = request.mode.dit_instruction(request.track)?;
    let prompts = text::build_prompt_strings(
        request.prompt,
        lyrics_effective.as_ref(),
        &instruction,
        duration_sec,
        request.bpm,
        request.keyscale,
        request.timesignature,
        request.vocal_language,
    );
    let non_cover_prompt = if strength < 1.0 && !request.mode.is_region_task() {
        Some(text::build_prompt_strings(
            request.prompt,
            lyrics_effective.as_ref(),
            DIT_INSTR_TEXT2MUSIC,
            duration_sec,
            request.bpm,
            request.keyscale,
            request.timesignature,
            request.vocal_language,
        ))
    } else {
        None
    };
    let tokenizer =
        std::fs::read_to_string(layout.text.join("tokenizer.json")).map_err(|error| {
            WeightError::Open {
                path: layout.text.join("tokenizer.json").display().to_string(),
                detail: error.to_string(),
            }
        })?;
    let text_ids = qwen3::tokenize_plain(&tokenizer, &prompts.text, qwen.vocab)?;
    let lyric_ids = qwen3::tokenize_plain(&tokenizer, &prompts.lyric, qwen.vocab)?;
    let non_cover_ids = non_cover_prompt
        .as_ref()
        .map(|prompts| qwen3::tokenize_plain(&tokenizer, &prompts.text, qwen.vocab))
        .transpose()?;
    let seed = request.seed.unwrap_or_else(draw_seed);
    let timbre = request
        .timbre_from_source
        .then(|| sample_timbre_reference_latents(&fitted, seed));
    let steps = request
        .steps
        .unwrap_or(if config.is_turbo { 8 } else { 30 });
    let dit_key = cache_key(&layout.dit)?;
    let text_key = cache_key(&layout.text)?;
    let started = std::time::Instant::now();
    let encoded = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step cover encode", {
            let dit_dir = layout.dit.clone();
            let text_dir = layout.text.clone();
            let config = config.clone();
            let silence = silence.clone();
            let text_ids = text_ids.clone();
            let lyric_ids = lyric_ids.clone();
            let non_cover_ids = non_cover_ids.clone();
            let timbre = timbre.clone();
            let dit_key = dit_key.clone();
            let text_key = text_key.clone();
            let loras = loras.clone();
            move |_, stream, cache, _| {
                let dit = resident_dit(cache, dit_key, &dit_dir, &loras, stream)?;
                let text = resident(cache, text_key, &text_dir, true, stream)?;
                let load_secs = started.elapsed().as_secs_f64();
                let text_hidden = encode_text(&text, qwen, &text_ids, stream)?;
                let lyric_hidden = embed_tokens(&text, &lyric_ids, qwen.hidden as i32, stream)?;
                let timbre_hidden = timbre_frames(&silence, timbre.as_deref(), stream)?;
                let encoder = model::encode_condition(
                    &dit,
                    &config,
                    &text_hidden,
                    &lyric_hidden,
                    &timbre_hidden,
                    stream,
                )?;
                encoder.eval()?;
                let non_cover = if let Some(ids) = non_cover_ids {
                    let hidden = encode_text(&text, qwen, &ids, stream)?;
                    let encoded = model::encode_condition(
                        &dit,
                        &config,
                        &hidden,
                        &lyric_hidden,
                        &timbre_hidden,
                        stream,
                    )?;
                    encoded.eval()?;
                    Some(encoded.to_vec_f32(stream)?)
                } else {
                    None
                };
                let encode_secs = started.elapsed().as_secs_f64() - load_secs;
                Ok(CoverEncoded {
                    encoder: encoder.to_vec_f32(stream)?,
                    enc_len: encoder.shape()[1] as usize,
                    enc_dim: encoder.shape()[2] as usize,
                    non_cover,
                    load_secs,
                    encode_secs,
                })
            }
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let silence_context = cover::cover_context(
        &silence[..frames * SILENCE_CHANNELS],
        frames as i64,
        config.out_channels as i64,
        config.in_channels as i64,
    )?;
    let non_cover_context = encoded.non_cover.as_ref().map(|_| silence_context);
    let denoise_started = std::time::Instant::now();
    let noise = normal_latent(seed, frames * config.out_channels as usize);
    let mut sample = super::SampleRequest::new(
        if config.is_turbo {
            ModelVariant::Turbo
        } else {
            ModelVariant::Base
        },
        1,
        frames,
        config.out_channels as usize,
        &noise,
    );
    sample.guidance = request.guidance;
    sample.cfg_distilled = config.is_turbo;
    sample.infer_steps = Some(steps);
    sample.shift = if config.is_turbo { 3.0 } else { 1.0 };
    sample.dcw = Some(request.dcw);
    sample.cover_strength = if request.mode.is_region_task() {
        1.0
    } else {
        strength
    };
    sample.cover_noise_strength = request.cover_noise_strength;
    sample.source = (request.cover_noise_strength > 0.0).then_some(fitted.as_slice());
    sample.repaint_mask = mask.as_deref();
    sample.clean_source = mask.is_some().then_some(fitted.as_slice());
    sample.repaint_crossfade_frames = request.repaint_crossfade_frames as i32;
    sample.repaint_injection_ratio = request.repaint_injection_ratio;
    let latents = super::sample_with(
        &sample,
        || {
            Err(WeightError::NotBuilt(
                "ACE-Step SDE noise is not drawn on this path".into(),
            ))
        },
        |step| {
            let encoder = if step.text2music {
                encoded.non_cover.as_deref().unwrap_or(&encoded.encoder)
            } else {
                encoded.encoder.as_slice()
            };
            let context = if step.text2music {
                non_cover_context.as_deref().unwrap_or(&context)
            } else {
                context.as_slice()
            };
            step_velocity(
                &dit_key,
                &layout.dit,
                &loras,
                &config,
                frames,
                step.latents,
                context,
                encoder,
                encoded.enc_len as i32,
                encoded.enc_dim as i32,
                step.conditional,
                step.timestep,
            )
        },
        |done, total| on_step(done, total),
    )?;
    let denoise_secs = denoise_started.elapsed().as_secs_f64();
    let (lyrics_alignment, lyrics_lrc) = align_sung_lyrics(
        &layout.dit,
        dit_key,
        &config,
        &tokenizer,
        &lyric_ids,
        qwen.vocab,
        request.vocal_language,
        &latents,
        &context,
        &encoded.encoder,
        encoded.enc_len,
        encoded.enc_dim,
        frames,
        config.out_channels,
        config.in_channels - config.out_channels,
        steps,
        lyrics_alignment,
        &loras,
    )?;
    let vae_started = std::time::Instant::now();
    let audio = decode_latents(&layout, &vae_spec, &latents, frames, config.out_channels)?;
    let vae_secs = vae_started.elapsed().as_secs_f64();
    let samples = audio.len() / 2;
    let expected = frames * vae_spec.hop() as usize;
    if samples != expected {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step VAE wrote {samples} samples, expected {expected} ({frames} latent frames × hop {})",
            vae_spec.hop()
        )));
    }
    let audio = quality::normalize_planar_stereo(&audio, samples as i64)?;
    let assessed = quality::assess_output(&latents, &fitted, &audio, SAMPLE_RATE, false, 0.0, 1)?;
    let wav = wav::write_planar_stereo_pcm16(&audio, SAMPLE_RATE)?;
    Ok(CreateOutput {
        wav,
        sample_rate: SAMPLE_RATE,
        frames: frames as i64,
        steps,
        enc_seq: encoded.enc_len as i64,
        graph_computes: steps,
        dcw: request.dcw,
        dcw_active: request.dcw.active(config.is_turbo),
        quality: assessed,
        timings: vec![
            ("dit_load", encoded.load_secs),
            ("text", encoded.encode_secs),
            ("denoise", denoise_secs),
            ("vae", vae_secs),
        ],
        lyrics_effective: lyrics_effective.into_owned(),
        lyrics_alignment,
        lyrics_txt,
        lyrics_lrc,
    })
}

struct Layout {
    dit: PathBuf,
    text: PathBuf,
    vae: PathBuf,
    config: PathBuf,
    vae_config: PathBuf,
    silence: Vec<f32>,
}

fn locate(bundle: &Path) -> Result<Layout, WeightError> {
    let dit = find_dit(bundle)?;
    let text = find_child(bundle, &dit, "Qwen3-Embedding-0.6B")?;
    let vae = find_child(bundle, &dit, "vae")?;
    let config = dit.join("config.json");
    let vae_config = vae.join("config.json");
    for path in [&config, &vae_config] {
        if !path.is_file() {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step bundle is missing {}",
                path.display()
            )));
        }
    }
    let silence = silence_tensor(bundle, &dit)?;
    Ok(Layout {
        dit,
        text,
        vae,
        config,
        vae_config,
        silence,
    })
}

fn find_dit(bundle: &Path) -> Result<PathBuf, WeightError> {
    let mut stack = vec![bundle.to_path_buf()];
    if let Ok(entries) = std::fs::read_dir(bundle) {
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                stack.push(entry.path());
            }
        }
    }
    let mut found = None;
    for dir in stack {
        let config = dir.join("config.json");
        let index = dir.join("model.safetensors.index.json");
        if !config.is_file() || !index.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&config).map_err(|error| WeightError::Open {
            path: config.display().to_string(),
            detail: error.to_string(),
        })?;
        let parsed = AceConfig::from_json(&text).map_err(WeightError::NotBuilt)?;
        if parsed.hidden == 2560 {
            found = Some(dir);
            break;
        }
    }
    found.ok_or_else(|| {
        WeightError::NotBuilt(format!(
            "ACE-Step XL safetensors (hidden 2560) were not found under {}",
            bundle.display()
        ))
    })
}

fn silence_tensor(bundle: &Path, dit: &Path) -> Result<Vec<f32>, WeightError> {
    let Some(path) = find_published_silence(bundle, dit) else {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step bundle is missing {}",
            dit.join("silence_latent.pt").display()
        )));
    };
    load_silence(&path)
}

fn find_published_silence(bundle: &Path, dit: &Path) -> Option<PathBuf> {
    let mut local = vec![
        dit.join("silence_latent.pt"),
        bundle.join("silence_latent.pt"),
    ];
    if let Ok(entries) = std::fs::read_dir(bundle) {
        for entry in entries.flatten() {
            local.push(entry.path().join("silence_latent.pt"));
        }
    }
    local.into_iter().find(|path| path.is_file())
}

fn find_child(bundle: &Path, dit: &Path, name: &str) -> Result<PathBuf, WeightError> {
    for root in [bundle, dit.parent().unwrap_or(bundle)] {
        let path = root.join(name);
        if path.is_dir() {
            return Ok(path);
        }
    }
    Err(WeightError::NotBuilt(format!(
        "ACE-Step bundle is missing {name} next to {}",
        bundle.display()
    )))
}

fn qwen_config(directory: &Path) -> Result<Qwen3Config, WeightError> {
    let path = directory.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| WeightError::Open {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        WeightError::NotBuilt(format!("Qwen3 config.json is not json: {error}"))
    })?;
    let number = |key: &str| {
        value
            .get(key)
            .and_then(|item| item.as_i64())
            .map(|item| item)
            .ok_or_else(|| WeightError::NotBuilt(format!("Qwen3 config.json is missing {key}")))
    };
    let hidden = number("hidden_size")?;
    let heads = number("num_attention_heads")?;
    Ok(Qwen3Config {
        vocab: number("vocab_size")?,
        hidden,
        layers: number("num_hidden_layers")?,
        heads,
        kv_heads: number("num_key_value_heads")?,
        head_dim: value
            .get("head_dim")
            .and_then(|item| item.as_i64())
            .unwrap_or(hidden / heads),
        intermediate: number("intermediate_size")?,
        rope_theta: value
            .get("rope_theta")
            .and_then(|item| item.as_f64())
            .unwrap_or(1_000_000.0) as f32,
    })
}

fn encode_text(
    weights: &MlxWeights,
    config: Qwen3Config,
    ids: &[i32],
    stream: &Stream,
) -> Result<Array, danmo_mlx::Error> {
    let valid = vec![true; ids.len()];
    let hidden = qwen3::mlx::encode_ids(
        weights,
        config,
        "model.",
        ids,
        &valid,
        &[config.layers],
        stream,
    )?;
    let norm = weights.tensor("model.norm.weight")?;
    hidden.rms_norm(Some(norm), 1e-6, stream)
}

fn embed_tokens(
    weights: &MlxWeights,
    ids: &[i32],
    hidden: i32,
    stream: &Stream,
) -> Result<Array, danmo_mlx::Error> {
    if ids.is_empty() {
        return Err(danmo_mlx::Error::new(
            "ACE-Step lyrics",
            "lyric prompt produced no tokens",
        ));
    }
    let table = weights.embedding("model.embed_tokens.weight", stream)?;
    let indices = Array::from_i32_slice(ids, &[ids.len() as i32])?;
    table
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, ids.len() as i32, hidden], stream)
}

fn timbre_frames(
    silence: &[f32],
    override_frames: Option<&[f32]>,
    stream: &Stream,
) -> Result<Array, danmo_mlx::Error> {
    let frames = if let Some(values) = override_frames {
        if values.len() % SILENCE_CHANNELS != 0 {
            return Err(danmo_mlx::Error::new(
                "ACE-Step timbre",
                format!(
                    "timbre length {} is not a multiple of {SILENCE_CHANNELS}",
                    values.len()
                ),
            ));
        }
        values.to_vec()
    } else {
        silence[..SILENCE_CHANNELS].to_vec()
    };
    let time = (frames.len() / SILENCE_CHANNELS) as i32;
    Array::from_f32_slice(&frames, &[1, time, SILENCE_CHANNELS as i32])
        .and_then(|array| array.astype(Dtype::Float32, stream))
}

fn silence_context(
    silence: &[f32],
    frames: usize,
    stream: &Stream,
) -> Result<Array, danmo_mlx::Error> {
    if silence.len() < frames * SILENCE_CHANNELS {
        return Err(danmo_mlx::Error::new(
            "ACE-Step silence",
            format!("silence latent cannot cover {frames} frames"),
        ));
    }
    let mut context = vec![0.0f32; frames * SILENCE_CHANNELS * 2];
    for frame in 0..frames {
        let src = &silence[frame * SILENCE_CHANNELS..(frame + 1) * SILENCE_CHANNELS];
        let dst = &mut context[frame * SILENCE_CHANNELS * 2..(frame + 1) * SILENCE_CHANNELS * 2];
        dst[..SILENCE_CHANNELS].copy_from_slice(src);
        for value in &mut dst[SILENCE_CHANNELS..] {
            *value = 1.0;
        }
    }
    Array::from_f32_slice(&context, &[1, frames as i32, (SILENCE_CHANNELS * 2) as i32])
        .and_then(|array| array.astype(Dtype::Float32, stream))
}

/// `[frames, 128]` context: detokenized codes in front, silence padding after, ones mask.
fn code_context(
    hints: &[f32],
    silence: &[f32],
    frames: usize,
) -> Result<Vec<f32>, danmo_mlx::Error> {
    if hints.len() % SILENCE_CHANNELS != 0 {
        return Err(danmo_mlx::Error::new(
            "ACE-Step audio codes",
            format!(
                "detokenized length {} is not a multiple of {SILENCE_CHANNELS}",
                hints.len()
            ),
        ));
    }
    if silence.len() < frames * SILENCE_CHANNELS {
        return Err(danmo_mlx::Error::new(
            "ACE-Step audio codes",
            format!("silence latent cannot pad {frames} frames"),
        ));
    }
    let hint_frames = hints.len() / SILENCE_CHANNELS;
    let mut context = vec![0.0f32; frames * SILENCE_CHANNELS * 2];
    for frame in 0..frames {
        let dst = frame * SILENCE_CHANNELS * 2;
        let acoustic = if frame < hint_frames {
            &hints[frame * SILENCE_CHANNELS..(frame + 1) * SILENCE_CHANNELS]
        } else {
            &silence[frame * SILENCE_CHANNELS..(frame + 1) * SILENCE_CHANNELS]
        };
        context[dst..dst + SILENCE_CHANNELS].copy_from_slice(acoustic);
        for value in &mut context[dst + SILENCE_CHANNELS..dst + SILENCE_CHANNELS * 2] {
            *value = 1.0;
        }
    }
    Ok(context)
}

fn silence_reference(silence: &[f32], frames: usize) -> Vec<f32> {
    silence[..frames * SILENCE_CHANNELS].to_vec()
}

fn sample_timbre_reference_latents(source: &[f32], seed: i64) -> Vec<f32> {
    const MAX_FRAMES: usize = 750;
    let frames = source.len() / SILENCE_CHANNELS;
    let count = frames.min(MAX_FRAMES);
    let start = if frames > count {
        seed.unsigned_abs() as usize % (frames - count + 1)
    } else {
        0
    };
    source[start * SILENCE_CHANNELS..(start + count) * SILENCE_CHANNELS].to_vec()
}

fn null_encoder(
    weights: &MlxWeights,
    length: i32,
    hidden: i32,
    stream: &Stream,
) -> Result<Array, danmo_mlx::Error> {
    weights
        .tensor("null_condition_emb")?
        .broadcast_to(&[1, length, hidden], stream)
}

fn load_silence(path: &Path) -> Result<Vec<f32>, WeightError> {
    let bytes = read_zip_stored(path, "silence_latent/data/0")?;
    if bytes.len() != SILENCE_FRAMES * SILENCE_CHANNELS * 4 {
        return Err(WeightError::NotBuilt(format!(
            "silence_latent.pt storage is {} bytes, expected {}",
            bytes.len(),
            SILENCE_FRAMES * SILENCE_CHANNELS * 4
        )));
    }
    let order = read_zip_stored(path, "silence_latent/byteorder")?;
    if order.as_slice() != b"little" {
        return Err(WeightError::NotBuilt(
            "silence_latent.pt is not little-endian".into(),
        ));
    }
    let mut channel_major = vec![0.0f32; SILENCE_FRAMES * SILENCE_CHANNELS];
    for (index, chunk) in bytes.chunks_exact(4).enumerate() {
        channel_major[index] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    // Checkpoint layout is [1, 64, 15000]. The sampler consumes [15000, 64].
    let mut time_major = vec![0.0f32; channel_major.len()];
    for channel in 0..SILENCE_CHANNELS {
        for frame in 0..SILENCE_FRAMES {
            time_major[frame * SILENCE_CHANNELS + channel] =
                channel_major[channel * SILENCE_FRAMES + frame];
        }
    }
    Ok(time_major)
}

fn read_zip_stored(path: &Path, name: &str) -> Result<Vec<u8>, WeightError> {
    let bytes = std::fs::read(path).map_err(|error| WeightError::Open {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let eocd = bytes
        .windows(4)
        .rposition(|window| window == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| WeightError::NotBuilt(format!("{} is not a zip archive", path.display())))?;
    if eocd + 22 > bytes.len() {
        return Err(WeightError::NotBuilt(format!(
            "{} zip directory is truncated",
            path.display()
        )));
    }
    let count = u16::from_le_bytes([bytes[eocd + 10], bytes[eocd + 11]]) as usize;
    let directory = u32::from_le_bytes(bytes[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
    if directory == 0xffff_ffff {
        return Err(WeightError::NotBuilt(format!(
            "{} uses Zip64, which this loader does not read",
            path.display()
        )));
    }
    let mut cursor = directory;
    for _ in 0..count {
        if bytes.get(cursor..cursor + 4) != Some(&[0x50, 0x4b, 0x01, 0x02])
            || cursor + 46 > bytes.len()
        {
            break;
        }
        let method = u16::from_le_bytes([bytes[cursor + 10], bytes[cursor + 11]]);
        let stored =
            u32::from_le_bytes(bytes[cursor + 20..cursor + 24].try_into().unwrap()) as usize;
        let name_len = u16::from_le_bytes([bytes[cursor + 28], bytes[cursor + 29]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[cursor + 30], bytes[cursor + 31]]) as usize;
        let comment_len = u16::from_le_bytes([bytes[cursor + 32], bytes[cursor + 33]]) as usize;
        let local =
            u32::from_le_bytes(bytes[cursor + 42..cursor + 46].try_into().unwrap()) as usize;
        let name_at = cursor + 46;
        let entry_name = bytes.get(name_at..name_at + name_len).unwrap_or_default();
        let next = name_at + name_len + extra_len + comment_len;
        if entry_name == name.as_bytes() {
            if method != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "{} entry {name} is compressed (method {method})",
                    path.display()
                )));
            }
            if local + 30 > bytes.len()
                || bytes.get(local..local + 4) != Some(&[0x50, 0x4b, 0x03, 0x04])
            {
                return Err(WeightError::NotBuilt(format!(
                    "{} entry {name} has no local header",
                    path.display()
                )));
            }
            let local_name = u16::from_le_bytes([bytes[local + 26], bytes[local + 27]]) as usize;
            let local_extra = u16::from_le_bytes([bytes[local + 28], bytes[local + 29]]) as usize;
            let start = local + 30 + local_name + local_extra;
            let end = start + stored;
            if end > bytes.len() {
                return Err(WeightError::NotBuilt(format!(
                    "{} entry {name} extends past the archive",
                    path.display()
                )));
            }
            return Ok(bytes[start..end].to_vec());
        }
        cursor = next;
    }
    Err(WeightError::NotBuilt(format!(
        "{} has no stored entry {name}",
        path.display()
    )))
}

fn load_ace_config(layout: &Layout) -> Result<AceConfig, WeightError> {
    AceConfig::from_json(&std::fs::read_to_string(&layout.config).map_err(|error| {
        WeightError::Open {
            path: layout.config.display().to_string(),
            detail: error.to_string(),
        }
    })?)
    .map_err(WeightError::NotBuilt)
}

fn load_vae_spec(layout: &Layout) -> Result<VaeSpec, WeightError> {
    VaeSpec::from_json(
        &std::fs::read_to_string(&layout.vae_config).map_err(|error| WeightError::Open {
            path: layout.vae_config.display().to_string(),
            detail: error.to_string(),
        })?,
    )
    .map_err(WeightError::NotBuilt)
}

fn fsq_cover_acoustic(
    dit_dir: &Path,
    loras: &[LoraFile],
    config: &AceConfig,
    fitted: &[f32],
    silence: &[f32],
    frames: usize,
) -> Result<Vec<f32>, WeightError> {
    let pool = config.pool_window as usize;
    if pool == 0 || frames == 0 || fitted.len() != frames * SILENCE_CHANNELS {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step FSQ cover expected {frames} frames of {SILENCE_CHANNELS} channels at pool {pool}"
        )));
    }
    let pad = (pool - frames % pool) % pool;
    if silence.len() < pad * SILENCE_CHANNELS {
        return Err(WeightError::NotBuilt(
            "ACE-Step silence_latent is shorter than the FSQ pool pad".into(),
        ));
    }
    let mut padded = fitted.to_vec();
    if pad > 0 {
        padded.extend_from_slice(&silence[..pad * SILENCE_CHANNELS]);
    }
    let padded_frames = frames + pad;
    let dit_key = cache_key(dit_dir)?;
    let dit_dir = dit_dir.to_path_buf();
    let loras = loras.to_vec();
    let config = config.clone();
    let values = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step fsq cover", move |_, stream, cache, _| {
            let dit = resident_dit(cache, dit_key, &dit_dir, &loras, stream)?;
            let latent = Array::from_f32_slice(
                &padded,
                &[1, padded_frames as i32, SILENCE_CHANNELS as i32],
            )?;
            let hints = model::retokenize_cover_latents(&dit, &config, &latent, stream)?;
            hints.eval()?;
            let values = hints.to_vec_f32(stream)?;
            let _ = danmo_mlx::clear_cache();
            Ok(values)
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let need = frames * SILENCE_CHANNELS;
    if values.len() < need || values.iter().any(|value| !value.is_finite()) {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step FSQ cover wrote {} values for {need} acoustic samples",
            values.len()
        )));
    }
    Ok(values[..need].to_vec())
}

fn source_latents(
    request: &CoverRequest<'_>,
    spec: &VaeSpec,
    vae_dir: &Path,
    _silence: &[f32],
) -> Result<Vec<f32>, WeightError> {
    if let Some(latents) = request.source_latents {
        if latents.is_empty() || latents.len() % SILENCE_CHANNELS != 0 {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step source latents length {} is not a positive multiple of {SILENCE_CHANNELS}",
                latents.len()
            )));
        }
        return Ok(latents.to_vec());
    }
    let audio = request.source_audio.ok_or_else(|| {
        WeightError::NotBuilt("ACE-Step cover needs source audio or source latents".into())
    })?;
    let hop = spec.hop() as usize;
    let (nlc, samples) = interleaved_to_nlc(audio, hop)?;
    let vae_key = cache_key(vae_dir)?;
    let vae_dir = vae_dir.to_path_buf();
    let spec = spec.clone();
    crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step vae encode", move |_, stream, cache, _| {
            let weights = resident(cache, vae_key, &vae_dir, false, stream)?;
            let audio = Array::from_f32_slice(&nlc, &[1, samples as i32, spec.audio])?;
            vae::encode_mean(&weights, &spec, &audio, stream)
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))
}

fn interleaved_to_nlc(audio: &[f32], hop: usize) -> Result<(Vec<f32>, usize), WeightError> {
    if hop == 0 || audio.len() % 2 != 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover audio must be interleaved stereo".into(),
        ));
    }
    let samples = audio.len() / 2;
    if samples == 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step cover audio is empty".into(),
        ));
    }
    let padded = samples.div_ceil(hop) * hop;
    let mut nlc = vec![0.0f32; padded * 2];
    for index in 0..samples {
        nlc[index * 2] = audio[index * 2];
        nlc[index * 2 + 1] = audio[index * 2 + 1];
    }
    Ok((nlc, padded))
}

fn fit_latents(source: &[f32], silence: &[f32], frames: usize) -> Vec<f32> {
    let mut fitted = vec![0.0f32; frames * SILENCE_CHANNELS];
    let src_frames = source.len() / SILENCE_CHANNELS;
    let copy = src_frames.min(frames);
    fitted[..copy * SILENCE_CHANNELS].copy_from_slice(&source[..copy * SILENCE_CHANNELS]);
    if frames > copy {
        let extra = frames - copy;
        fitted[copy * SILENCE_CHANNELS..].copy_from_slice(&silence[..extra * SILENCE_CHANNELS]);
    }
    fitted
}

fn step_velocity(
    dit_key: &CacheKey,
    dit_dir: &Path,
    loras: &[LoraFile],
    config: &AceConfig,
    frames: usize,
    xt: &[f32],
    context: &[f32],
    encoder: &[f32],
    enc_len: i32,
    enc_dim: i32,
    conditional: bool,
    timestep: f32,
) -> Result<Vec<f32>, WeightError> {
    let dit_key = dit_key.clone();
    let dit_dir = dit_dir.to_path_buf();
    let loras = loras.to_vec();
    let config = config.clone();
    let xt = xt.to_vec();
    let context = context.to_vec();
    let encoder = encoder.to_vec();
    let values = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step velocity", move |_, stream, cache, _| {
            let dit = resident_dit(cache, dit_key, &dit_dir, &loras, stream)?;
            let xt = Array::from_f32_slice(&xt, &[1, frames as i32, config.out_channels])?;
            let context = Array::from_f32_slice(
                &context,
                &[1, frames as i32, config.in_channels - config.out_channels],
            )?;
            let encoded = if conditional {
                Array::from_f32_slice(&encoder, &[1, enc_len, enc_dim])?
            } else {
                null_encoder(&dit, enc_len, enc_dim, stream)?
            };
            let velocity =
                model::velocity(&dit, &config, &xt, &context, &encoded, timestep, stream)?;
            velocity.eval()?;
            let values = velocity.to_vec_f32(stream)?;
            let _ = danmo_mlx::clear_cache();
            Ok(values)
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(WeightError::NotBuilt(
            "ACE-Step velocity is not finite".into(),
        ));
    }
    Ok(values)
}

fn decode_latents(
    layout: &Layout,
    spec: &VaeSpec,
    latents: &[f32],
    frames: usize,
    channels: i32,
) -> Result<Vec<f32>, WeightError> {
    let vae_key = cache_key(&layout.vae)?;
    let vae_dir = layout.vae.clone();
    let spec = spec.clone();
    let latents = latents.to_vec();
    crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step vae", move |_, stream, cache, _| {
            let weights = resident(cache, vae_key, &vae_dir, false, stream)?;
            let latent = Array::from_f32_slice(&latents, &[1, frames as i32, channels])?;
            vae::decode_planar(&weights, &spec, &latent, stream)
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))
}

fn resident_dit(
    cache: &mut ModelCache,
    key: CacheKey,
    directory: &Path,
    loras: &[LoraFile],
    stream: &Stream,
) -> Result<MlxWeights, danmo_mlx::Error> {
    let weights = resident(cache, key, directory, false, stream)?;
    if loras.is_empty() {
        return Ok(weights);
    }
    weights
        .attach_lora(loras, stream)
        .map_err(|error| danmo_mlx::Error::new("ACE-Step LoRA", error.to_string()))
}

fn resident(
    cache: &mut ModelCache,
    key: CacheKey,
    directory: &Path,
    text_encoder: bool,
    stream: &Stream,
) -> Result<MlxWeights, danmo_mlx::Error> {
    if cache.get::<MlxWeights>(&key)?.is_some() {
        return Ok(cache
            .get::<MlxWeights>(&key)?
            .cloned()
            .expect("resident weights"));
    }
    let weights = if text_encoder {
        qwen3::mlx::load_directory(directory, WeightDtype::Native)?
            .materialize_embedding("model.embed_tokens.weight", stream)?
    } else {
        crate::weights::load_mlx_directory(directory, WeightDtype::Native, None)?
    };
    let bytes = weights.nbytes();
    cache.insert(key.clone(), weights, bytes)?;
    Ok(cache
        .get::<MlxWeights>(&key)?
        .cloned()
        .expect("inserted weights"))
}

fn cache_key(directory: &Path) -> Result<CacheKey, WeightError> {
    let id = std::fs::canonicalize(directory)
        .map_err(|error| WeightError::NotBuilt(format!("{}: {error}", directory.display())))?
        .display()
        .to_string();
    let mut length = 0u64;
    let mut modified_ns = 0u64;
    for entry in
        std::fs::read_dir(directory).map_err(|error| WeightError::NotBuilt(error.to_string()))?
    {
        let path = entry
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?
            .path();
        if path.extension().and_then(|value| value.to_str()) != Some("safetensors") {
            continue;
        }
        let metadata = std::fs::metadata(&path)
            .map_err(|error| WeightError::NotBuilt(format!("{}: {error}", path.display())))?;
        length += metadata.len();
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        modified_ns = modified_ns.max(modified);
    }
    Ok(CacheKey {
        id,
        revision: CacheRevision {
            length,
            modified_ns,
            mix: 0,
        },
    })
}

fn align_sung_lyrics(
    dit: &Path,
    dit_key: CacheKey,
    config: &AceConfig,
    tokenizer_json: &str,
    lyric_ids: &[i32],
    vocab: i64,
    vocal_language: &str,
    latents: &[f32],
    context: &[f32],
    encoder: &[f32],
    enc_len: usize,
    enc_dim: usize,
    frames: usize,
    out_channels: i32,
    context_channels: i32,
    steps: u32,
    current: lyrics::LyricsAlignment,
    loras: &[LoraFile],
) -> Result<(lyrics::LyricsAlignment, Option<String>), WeightError> {
    if matches!(
        current.mode.as_str(),
        "instrumental" | "empty" | "conditioning_only"
    ) {
        return Ok((current.clone(), lyrics::format_lrc(&current)));
    }
    if steps == 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step lyric alignment needs a positive step count".into(),
        ));
    }
    let language = if vocal_language.is_empty() {
        "unknown"
    } else {
        vocal_language
    };
    let header = LYRIC_HEADER_TEMPLATE.replace("{language}", language);
    let header_ids = qwen3::tokenize_plain(tokenizer_json, &header, vocab)?;
    let start = header_ids.len();
    let end = lyric_ids
        .iter()
        .position(|id| *id == QWEN_ENDOFTEXT_ID)
        .unwrap_or(lyric_ids.len());
    if start >= end || end > lyric_ids.len() {
        return Err(WeightError::NotBuilt(
            "ACE-Step lyric alignment found no lyric tokens after the language header".into(),
        ));
    }
    if end > enc_len {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step lyric tokens end at {end}, past the encoder length {enc_len}"
        )));
    }
    let timestep = 1.0 / steps as f32;
    let noise = normal_latent(42, latents.len());
    let xt: Vec<f32> = latents
        .iter()
        .zip(noise)
        .map(|(value, noise)| timestep * noise + (1.0 - timestep) * value)
        .collect();
    let dit_dir = dit.to_path_buf();
    let loras = loras.to_vec();
    let config = config.clone();
    let context = context.to_vec();
    let encoder = encoder.to_vec();
    let (heads, attn_frames, attn_encoder, weights) = crate::runtime::mlx_executor()
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?
        .execute("ace_step lyric alignment", move |_, stream, cache, _| {
            let dit = resident_dit(cache, dit_key, &dit_dir, &loras, stream)?;
            let latents = Array::from_f32_slice(&xt, &[1, frames as i32, out_channels])?;
            let context = Array::from_f32_slice(&context, &[1, frames as i32, context_channels])?;
            let encoder = Array::from_f32_slice(&encoder, &[1, enc_len as i32, enc_dim as i32])?;
            model::capture_lyric_attention(
                &dit, &config, &latents, &context, &encoder, timestep, stream,
            )
        })
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let tokens = end - start;
    if weights.len() != heads * attn_frames * attn_encoder {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step lyric attention length {} does not match {heads}x{attn_frames}x{attn_encoder}",
            weights.len()
        )));
    }
    let mut lyric = vec![0.0f32; heads * tokens * attn_frames];
    for head in 0..heads {
        for frame in 0..attn_frames {
            for token in 0..tokens {
                let source = ((head * attn_frames + frame) * attn_encoder) + start + token;
                lyric[(head * tokens + token) * attn_frames + frame] = weights[source];
            }
        }
    }
    let decoder = Decoder::from_tokenizer_json(tokenizer_json)?;
    let pure = &lyric_ids[start..end];
    let duration = frames as f32 / 25.0;
    let aligned = lyrics_dtw::align_lyrics_from_attention(
        &lyric,
        heads,
        pure,
        attn_frames,
        duration,
        &|ids| decoder.decode(ids),
        2.0,
    )?;
    if aligned.segments.is_empty() {
        return Err(WeightError::NotBuilt(
            "ACE-Step lyric alignment produced no timed lines".into(),
        ));
    }
    let lrc = lyrics::format_lrc(&aligned);
    Ok((aligned, lrc))
}

fn draw_seed() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as i64)
        .unwrap_or(1)
        & 0x7fff_ffff
}

#[cfg(test)]
mod tests {
    use super::{
        cache_key, code_context, fsq_cover_acoustic, locate, resident,
        sample_timbre_reference_latents, silence_tensor, SILENCE_CHANNELS,
    };
    use crate::families::ace_step::mlx::model::{self, AceConfig};
    use std::path::PathBuf;

    #[test]
    fn timbre_reference_is_seeded_bounded_and_uses_short_sources() {
        let source = (0..(1000 * SILENCE_CHANNELS))
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let first = sample_timbre_reference_latents(&source, 7);
        assert_eq!(first.len(), 750 * SILENCE_CHANNELS);
        assert_eq!(first, sample_timbre_reference_latents(&source, 7));
        assert_ne!(first, sample_timbre_reference_latents(&source, 8));
        let short = vec![1.0; 10 * SILENCE_CHANNELS];
        assert_eq!(sample_timbre_reference_latents(&short, 7), short);
    }

    #[test]
    fn code_context_keeps_hints_and_pads_with_silence() {
        let hints = vec![2.0f32; 64];
        let silence = vec![0.5f32; 64 * 3];
        let packed = code_context(&hints, &silence, 3).unwrap();
        assert_eq!(packed.len(), 3 * 128);
        assert!(packed[..64].iter().all(|value| *value == 2.0));
        assert!(packed[64..128].iter().all(|value| *value == 1.0));
        assert!(packed[128..192].iter().all(|value| *value == 0.5));
        assert!(packed[256..320].iter().all(|value| *value == 0.5));
        assert!(packed[320..384].iter().all(|value| *value == 1.0));
    }

    #[test]
    #[ignore = "loads the XL DiT and detokenizes four FSQ codes"]
    fn detokenize_audio_codes_are_finite() {
        let bundle = PathBuf::from(
            "/Users/nil.luo/Workspace/studio-workspace/models/Audio/acestep-v15-xl-turbo",
        );
        if !bundle.join("acestep-v15-xl-turbo/config.json").is_file() {
            eprintln!("skip: turbo DiT config missing");
            return;
        }
        let layout = locate(&bundle).unwrap();
        let config =
            AceConfig::from_json(&std::fs::read_to_string(&layout.config).unwrap()).unwrap();
        let key = cache_key(&layout.dit).unwrap();
        let dit_dir = layout.dit.clone();
        let values = crate::runtime::mlx_executor()
            .unwrap()
            .execute("ace_step detokenize probe", move |_, stream, cache, _| {
                let dit = resident(cache, key, &dit_dir, false, stream)?;
                let hints =
                    model::detokenize_audio_codes(&dit, &config, &[0, 100, 1_000, 63_999], stream)?;
                hints.eval()?;
                hints.to_vec_f32(stream)
            })
            .unwrap();
        assert_eq!(values.len(), 4 * 5 * 64);
        assert!(values.iter().all(|value| value.is_finite()));
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let var = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / values.len() as f32;
        assert!(var > 1e-8, "detokenized codes collapsed, var {var}");
        println!("detokenize mean={mean:.4} var={var:.4}");
    }

    #[test]
    #[ignore = "loads the XL DiT and retokenizes silence through the FSQ cover codec"]
    fn fsq_cover_retokenizes_silence() {
        let bundle = PathBuf::from(
            "/Users/nil.luo/Workspace/studio-workspace/models/Audio/acestep-v15-xl-turbo",
        );
        if !bundle.join("acestep-v15-xl-turbo/config.json").is_file() {
            eprintln!("skip: turbo DiT config missing");
            return;
        }
        let layout = locate(&bundle).unwrap();
        let config =
            AceConfig::from_json(&std::fs::read_to_string(&layout.config).unwrap()).unwrap();
        let silence = silence_tensor(&bundle, &layout.dit).unwrap();
        let frames = 8usize;
        let fitted = silence[..frames * 64].to_vec();
        let started = std::time::Instant::now();
        let hints =
            fsq_cover_acoustic(&layout.dit, &[], &config, &fitted, &silence, frames).unwrap();
        assert_eq!(hints.len(), frames * 64);
        let max = fitted
            .iter()
            .zip(&hints)
            .map(|(source, hint)| (source - hint).abs())
            .fold(0.0f32, f32::max);
        let mean = hints.iter().sum::<f32>() / hints.len() as f32;
        println!(
            "fsq cover max_abs={max:.4} mean={mean:.4} secs={:.2}",
            started.elapsed().as_secs_f64()
        );
        assert!(max > 1e-3, "FSQ cover left the silence latent unchanged");
    }
}
