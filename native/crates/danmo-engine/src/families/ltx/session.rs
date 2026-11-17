//! LTX 2.3 t2v session (Facade). Port of main's two-stage
//! `_generate_two_stage` (`generation_mlx.py`):
//!
//! 1. Gemma-3 encodes the prompt (dev also encodes the negative prompt;
//!    empty maps to main's `DEFAULT_NEGATIVE_PROMPT`) and the connectors
//!    produce the `(video, audio)` text tokens.
//! 2. Stage 1 denoises at half resolution. Dev runs `steps` steps of the
//!    token-adaptive `ltx2_schedule` under the multimodal guidance: four DiT
//!    forwards per step (cond / negative / STG-perturbed at block 28 /
//!    modality-perturbed), combined as
//!    `cond + (cfg-1)(cond-neg) + stg(cond-ptb) + (mod-1)(cond-mod)` with
//!    the 0.7 variance rescale; video cfg is the request guidance, audio
//!    cfg is the fixed 7.0. Distilled runs `DISTILLED_SIGMAS[..steps+1]`
//!    (8 steps by default) with a single forward per step — the distilled
//!    checkpoint takes no guidance.
//! 3. The stage-1 video tokens are unpatchified, denormalized with the
//!    video VAE per-channel statistics, run through the spatial x2 latent
//!    upsampler, renormalized, and repatchified at full resolution.
//! 4. Stage 2 re-noises video and audio to sigma 0.909375 (seed + 2) and
//!    refines for the fixed 3 steps of `STAGE_2_SIGMAS`, plain Euler, no
//!    guidance. Dev loads the distilled `transformer_stage2`; distilled
//!    reuses its own transformer (main `_weight_stem`: both distilled
//!    stages load "transformer-distilled").
//! 5. The video VAE decodes pixels, the audio VAE + vocoder decode the 48
//!    kHz stereo track, and ffmpeg muxes the PNG sequence and the WAV into
//!    the MP4 artifact.
//!
//! The DiT predicts velocity; main wraps it as `x0 = x_t - sigma * v` and
//! Euler-steps `x + (sigma_next - sigma) * (x - x0) / sigma`.

use super::audio::{LtxAudioDecoder, LtxVocoder};
use super::connector::ConnectorEncoder;
use super::dit::{DitPerturb, DitStep, LtxDit};
use super::gemma::GemmaEncoder;
use super::upscaler::LtxLatentUpscaler;
use super::vae::LtxVideoDecoder;
use super::Prepared;
use crate::common::ffmpeg;
use crate::common::ops::{non_finite, normal_latent, sinusoidal_timestep};
use crate::contract::{Artifact, EngineOutput, ExecutionContext, ResolvedRun};
use crate::error::EngineError;
use crate::runtime::GgmlContext;
use crate::weights::{assert_ltx_bundle, GgufWeights, WeightError};
use serde_json::json;
use std::path::{Path, PathBuf};

/// Stage 2 always runs the fixed three-step refine schedule.
pub const STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];
/// Distilled stage-1 schedule (`pipe_math.DISTILLED_SIGMAS`); main slices
/// `DISTILLED_SIGMAS[..steps+1]`.
pub const DISTILLED_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];
/// main `pipe_math._DEFAULT_STG_BLOCKS`.
const STG_BLOCKS: [i64; 1] = [28];
/// Dev guider constants (`ltx_dev_video_guider_params` /
/// `ltx_dev_audio_guider_params`): STG 1.0, modality 3.0, rescale 0.7, and
/// the fixed audio CFG 7.0.
const STG_SCALE: f32 = 1.0;
const MODALITY_SCALE: f32 = 3.0;
const RESCALE: f32 = 0.7;
const AUDIO_CFG: f32 = 7.0;
/// 16000 Hz / 160 hop / 4 downsample: audio latents per second of video.
const AUDIO_LATENTS_PER_SECOND: f64 = 25.0;
const AUDIO_DOWNSAMPLE: f32 = 4.0;
const AUDIO_HOP: f32 = 160.0;
const AUDIO_RATE: f32 = 16000.0;

/// main `pipe_math.DEFAULT_NEGATIVE_PROMPT`.
const DEFAULT_NEGATIVE_PROMPT: &str = "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

// ---------------------------------------------------------------------------
// host math (pipe_math ports)
// ---------------------------------------------------------------------------

/// `ltx2_schedule`: token-adaptive flow-matching sigmas, stretched so the
/// last nonzero sigma lands on the 0.1 terminal.
pub(crate) fn ltx2_schedule(steps: u32, num_tokens: i64) -> Vec<f32> {
    let steps = steps.max(1) as usize;
    let mut sigmas: Vec<f32> = (0..=steps).map(|i| 1.0 - i as f32 / steps as f32).collect();
    let slope = (2.05f32 - 0.95) / (4096.0 - 1024.0);
    let intercept = 0.95 - slope * 1024.0;
    let shift = num_tokens as f32 * slope + intercept;
    for sigma in sigmas.iter_mut() {
        if *sigma != 0.0 {
            *sigma = shift.exp() / (shift.exp() + (1.0 / *sigma - 1.0));
        }
    }
    // Stretch so the smallest nonzero sigma is exactly `terminal` (0.1).
    let terminal = 0.1f32;
    let last_nz = sigmas
        .iter()
        .rev()
        .find(|s| **s != 0.0)
        .copied()
        .unwrap_or(0.0);
    if last_nz != 0.0 {
        let scale = (1.0 - last_nz) / (1.0 - terminal);
        if scale != 0.0 {
            for sigma in sigmas.iter_mut() {
                if *sigma != 0.0 {
                    *sigma = 1.0 - (1.0 - *sigma) / scale;
                }
            }
        }
    }
    sigmas
}

/// `[f, h, w]` per video token (seconds, pixel units), token-major with the
/// frame axis outermost (`compute_video_positions`).
pub(crate) fn video_positions(f: i64, h: i64, w: i64, fps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity((f * h * w * 3) as usize);
    for fi in 0..f {
        let start = (fi * 8 - 7).max(0) as f32;
        let end = ((fi + 1) * 8 - 7).max(0) as f32;
        let f_mid = (start + end) / 2.0 / fps;
        for hi in 0..h {
            let h_mid = hi as f32 * 32.0 + 16.0;
            for wi in 0..w {
                let w_mid = wi as f32 * 32.0 + 16.0;
                out.extend_from_slice(&[f_mid, h_mid, w_mid]);
            }
        }
    }
    out
}

/// Seconds per audio token (`compute_audio_positions`).
pub(crate) fn audio_positions(t: i64) -> Vec<f32> {
    (0..t)
        .map(|i| {
            let start = ((i as f32 * AUDIO_DOWNSAMPLE) + 1.0 - AUDIO_DOWNSAMPLE).max(0.0)
                * AUDIO_HOP
                / AUDIO_RATE;
            let end = (((i + 1) as f32 * AUDIO_DOWNSAMPLE) + 1.0 - AUDIO_DOWNSAMPLE).max(0.0)
                * AUDIO_HOP
                / AUDIO_RATE;
            (start + end) / 2.0
        })
        .collect()
}

/// Tokens `[c + C * token]` -> `[w, h, f, c]` ggml flat (channel slowest);
/// the token index runs frame-outer, w-inner (`VideoLatentPatchifier`).
pub(crate) fn video_unpatchify(tokens: &[f32], c: i64, f: i64, h: i64, w: i64) -> Vec<f32> {
    let (c, f, h, w) = (c as usize, f as usize, h as usize, w as usize);
    let mut out = vec![0f32; w * h * f * c];
    for z in 0..f {
        for y in 0..h {
            for x in 0..w {
                let token = z * h * w + y * w + x;
                for ch in 0..c {
                    out[x + w * (y + h * (z + f * ch))] = tokens[ch + c * token];
                }
            }
        }
    }
    out
}

/// `[w, h, f, c]` ggml flat -> tokens `[c + C * token]` (inverse of
/// [`video_unpatchify`]).
pub(crate) fn video_patchify(latent: &[f32], c: i64, f: i64, h: i64, w: i64) -> Vec<f32> {
    let (c, f, h, w) = (c as usize, f as usize, h as usize, w as usize);
    let mut out = vec![0f32; w * h * f * c];
    for z in 0..f {
        for y in 0..h {
            for x in 0..w {
                let token = z * h * w + y * w + x;
                for ch in 0..c {
                    out[ch + c * token] = latent[x + w * (y + h * (z + f * ch))];
                }
            }
        }
    }
    out
}

/// Audio tokens `[k + 128 * t]` -> `[freq=16, t, ch=8]` ggml flat; the
/// token channel runs `k = c * 16 + f` (`AudioPatchifier`).
pub(crate) fn audio_unpatchify(tokens: &[f32], t: i64, channels: i64) -> Vec<f32> {
    let (t, channels) = (t as usize, channels as usize);
    let mut out = vec![0f32; 16 * t * channels];
    for c in 0..channels {
        for ti in 0..t {
            for f in 0..16 {
                out[f + 16 * (ti + t * c)] = tokens[c * 16 + f + 16 * channels * ti];
            }
        }
    }
    out
}

/// Per-channel `x * std + mean` / `(x - mean) / std` on a channel-slowest
/// `[w, h, f, c]` flat.
fn apply_stats(x: &mut [f32], mean: &[f32], std: &[f32], denormalize: bool) {
    let c = mean.len();
    let plane = x.len() / c;
    for (i, v) in x.iter_mut().enumerate() {
        let ch = i / plane;
        *v = if denormalize {
            *v * std[ch] + mean[ch]
        } else {
            (*v - mean[ch]) / std[ch]
        };
    }
}

/// `x0 = x_t - sigma * velocity` (main `LTX23X0Model`).
fn x0_of(latent: &[f32], velocity: &[f32], sigma: f32) -> Vec<f32> {
    latent
        .iter()
        .zip(velocity)
        .map(|(x, v)| x - sigma * v)
        .collect()
}

/// The dev multimodal guider (`MultiModalGuider.calculate`) with the 0.7
/// variance rescale against the conditional branch.
fn guider(cond: &[f32], neg: &[f32], ptb: &[f32], modality: &[f32], cfg: f32) -> Vec<f32> {
    let mut pred: Vec<f32> = cond
        .iter()
        .zip(neg)
        .zip(ptb)
        .zip(modality)
        .map(|(((c, n), p), m)| {
            c + (cfg - 1.0) * (c - n) + STG_SCALE * (c - p) + (MODALITY_SCALE - 1.0) * (c - m)
        })
        .collect();
    let var = |values: &[f32]| {
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / values.len() as f32
    };
    let factor = var(cond).sqrt() / (var(&pred).sqrt() + 1e-8);
    let factor = RESCALE * factor + (1.0 - RESCALE);
    for v in pred.iter_mut() {
        *v *= factor;
    }
    pred
}

/// main `_euler_step`: `x + (sigma_next - sigma) * (x - x0) / sigma`.
fn euler(x: &[f32], x0: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    if sigma == 0.0 {
        return x0.to_vec();
    }
    x.iter()
        .zip(x0)
        .map(|(x, x0)| x + (sigma_next - sigma) * (x - x0) / sigma)
        .collect()
}

/// Stereo 48 kHz 16-bit WAV from the interleaved `(time, channel)` samples.
fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let frames = samples.len() / 2;
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&2u16.to_le_bytes()); // stereo
    out.extend_from_slice(&48000u32.to_le_bytes());
    out.extend_from_slice(&192_000u32.to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    let _ = frames;
    for sample in samples {
        let v = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn tokenizer_json(root: &Path) -> Result<String, WeightError> {
    for candidate in [
        root.join("tokenizer.json"),
        root.join("tokenizer/tokenizer.json"),
    ] {
        if candidate.is_file() {
            return std::fs::read_to_string(&candidate).map_err(|err| WeightError::Open {
                path: candidate.display().to_string(),
                detail: err.to_string(),
            });
        }
    }
    Err(WeightError::NotBuilt(format!(
        "{} holds no tokenizer.json",
        root.display()
    )))
}

struct WorkDir(PathBuf);

impl Drop for WorkDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------
// session
// ---------------------------------------------------------------------------

pub(crate) fn infer(
    run: &ResolvedRun,
    prepared: &Prepared,
    ctx: &GgmlContext,
    exec: &ExecutionContext,
) -> Result<EngineOutput, EngineError> {
    let p = prepared;
    let bundle = assert_ltx_bundle(&run.bundle)?;
    // Dev refines with the distilled stage-2 transformer; distilled reuses
    // its own transformer for both stages (main `_weight_stem`).
    let stage2_paths: &[PathBuf] = if p.distilled {
        &bundle.base.transformer
    } else {
        if bundle.stage2_transformer.is_empty() {
            return Err(WeightError::Bundle {
                path: run.bundle.display().to_string(),
                detail: "dev LTX needs the stage-2 distilled transformer component".into(),
            }
            .into());
        }
        &bundle.stage2_transformer
    };
    if bundle.upscaler.is_empty() {
        return Err(WeightError::Bundle {
            path: run.bundle.display().to_string(),
            detail: "LTX two-stage t2v needs the latent upscaler component".into(),
        }
        .into());
    }
    let tokenizer = tokenizer_json(&bundle.base.root)?;

    // ---- text conditioning ----
    let gemma = GemmaEncoder::open(&bundle.base.text_encoder)?;
    let pos = gemma.encode(ctx, &tokenizer, &p.request.prompt)?;
    // The distilled checkpoint takes no guidance; main encodes no negative.
    let neg = if p.distilled {
        None
    } else {
        let negative = if p.request.negative_prompt.trim().is_empty() {
            DEFAULT_NEGATIVE_PROMPT
        } else {
            p.request.negative_prompt.as_str()
        };
        Some(gemma.encode(ctx, &tokenizer, negative)?)
    };
    let connectors = ConnectorEncoder::open(&bundle.connectors, &bundle.base.transformer)?;
    let (v_txt, a_txt) = connectors.encode(ctx, &pos)?;
    let (v_neg, a_neg) = match &neg {
        Some(cond) => {
            let (v, a) = connectors.encode(ctx, cond)?;
            (Some(v), Some(a))
        }
        None => (None, None),
    };

    // ---- geometry ----
    let f = p.latent_frames as i64;
    let h_half = (p.height / 2 / 32) as i64;
    let w_half = (p.width / 2 / 32) as i64;
    if h_half < 1 || w_half < 1 {
        return Err(EngineError::bad_request(format!(
            "LTX size {}x{} is too small for the half-resolution stage 1",
            p.width, p.height
        )));
    }
    let (h_full, w_full) = (h_half * 2, w_half * 2);
    let v_seq1 = f * h_half * w_half;
    let v_seq2 = f * h_full * w_full;
    let a_seq = ((p.num_frames as f64 / p.fps as f64) * AUDIO_LATENTS_PER_SECOND).round() as i64;
    let a_seq = a_seq.max(1);
    let t_seq = pos.seq;

    let dit1 = LtxDit::open(&bundle.base.transformer)?;
    let config = dit1.config();
    let latent = config.latent;
    let v_pos1 = video_positions(f, h_half, w_half, p.fps as f32);
    let v_pos2 = video_positions(f, h_full, w_full, p.fps as f32);
    let a_pos = audio_positions(a_seq);
    let sigmas1: Vec<f32> = if p.distilled {
        // main slices DISTILLED_SIGMAS[..steps+1]; steps above 8 saturates.
        let count = (p.steps as usize + 1).min(DISTILLED_SIGMAS.len());
        DISTILLED_SIGMAS[..count].to_vec()
    } else {
        ltx2_schedule(p.steps, v_seq1)
    };
    let stage1_steps = sigmas1.len() - 1;
    let total_steps = (stage1_steps + STAGE_2_SIGMAS.len() - 1) as u32;

    // ---- stage 1: guided denoise at half resolution ----
    let mut graph1 = dit1.build_graph(ctx, v_seq1, a_seq, t_seq)?;
    let seed = p.request.seed.unwrap_or(0);
    let mut video = normal_latent(seed, (latent * v_seq1) as usize);
    let mut audio = normal_latent(seed + 1, (latent * a_seq) as usize);
    for i in 0..stage1_steps {
        exec.check_cancelled()?;
        let (sigma, sigma_next) = (sigmas1[i], sigmas1[i + 1]);
        let t_emb = sinusoidal_timestep(sigma * 1000.0, config.t_dim as usize);
        let mut step = DitStep {
            video_latent: video.clone(),
            audio_latent: audio.clone(),
            t_emb: t_emb.clone(),
            video_text: v_txt.clone(),
            audio_text: a_txt.clone(),
            video_positions: v_pos1.clone(),
            audio_positions: a_pos.clone(),
            perturb: DitPerturb::default(),
        };
        let (cv, ca) = graph1.forward(&step)?;
        let cond_v = x0_of(&video, &cv, sigma);
        let cond_a = x0_of(&audio, &ca, sigma);

        if p.distilled {
            // No guidance on the distilled checkpoint: one forward per step.
            video = euler(&video, &cond_v, sigma, sigma_next);
            audio = euler(&audio, &cond_a, sigma, sigma_next);
        } else {
            let (Some(v_neg), Some(a_neg)) = (&v_neg, &a_neg) else {
                return Err(WeightError::NotBuilt(
                    "dev session lost the negative conditioning".into(),
                )
                .into());
            };
            step.video_text = v_neg.clone();
            step.audio_text = a_neg.clone();
            let (nv, na) = graph1.forward(&step)?;
            let neg_v = x0_of(&video, &nv, sigma);
            let neg_a = x0_of(&audio, &na, sigma);

            step.video_text = v_txt.clone();
            step.audio_text = a_txt.clone();
            step.perturb = DitPerturb {
                skip_video_self: STG_BLOCKS.to_vec(),
                skip_audio_self: STG_BLOCKS.to_vec(),
                skip_av_cross: false,
            };
            let (pv, pa) = graph1.forward(&step)?;
            let ptb_v = x0_of(&video, &pv, sigma);
            let ptb_a = x0_of(&audio, &pa, sigma);

            step.perturb = DitPerturb {
                skip_video_self: Vec::new(),
                skip_audio_self: Vec::new(),
                skip_av_cross: true,
            };
            let (mv, ma) = graph1.forward(&step)?;
            let mod_v = x0_of(&video, &mv, sigma);
            let mod_a = x0_of(&audio, &ma, sigma);

            let pred_v = guider(&cond_v, &neg_v, &ptb_v, &mod_v, p.guidance);
            let pred_a = guider(&cond_a, &neg_a, &ptb_a, &mod_a, AUDIO_CFG);
            video = euler(&video, &pred_v, sigma, sigma_next);
            audio = euler(&audio, &pred_a, sigma, sigma_next);
        }
        if let Some(detail) = non_finite("ltx stage 1 video", &video)
            .or_else(|| non_finite("ltx stage 1 audio", &audio))
        {
            return Err(WeightError::NotBuilt(detail).into());
        }
        exec.progress("denoise", (i + 1) as u32, total_steps)?;
    }

    // ---- inter-stage: denormalize, spatial x2 latent upscale, renormalize ----
    let vae_weights = GgufWeights::open_parts(&bundle.base.vae)?;
    let mean = vae_weights.f32_widened(super::vae::MEAN, &[latent])?;
    let std = vae_weights.f32_widened(super::vae::STD, &[latent])?;
    let mut v1 = video_unpatchify(&video, latent, f, h_half, w_half);
    apply_stats(&mut v1, &mean, &std, true);
    let upscaler = LtxLatentUpscaler::open(&bundle.upscaler)?;
    let mut v2 = upscaler.upscale(ctx, &v1, w_half, h_half, f)?;
    apply_stats(&mut v2, &mean, &std, false);
    let tokens2 = video_patchify(&v2, latent, f, h_full, w_full);

    // ---- stage 2: renoise + 3 plain Euler steps with the distilled DiT ----
    let dit2 = LtxDit::open(stage2_paths)?;
    let config2 = dit2.config();
    if config2.latent != latent {
        return Err(WeightError::NotBuilt(format!(
            "stage-2 transformer latent {} != dev latent {latent}",
            config2.latent
        ))
        .into());
    }
    let mut graph2 = dit2.build_graph(ctx, v_seq2, a_seq, t_seq)?;
    let s0 = STAGE_2_SIGMAS[0];
    let noise_v = normal_latent(seed + 2, tokens2.len());
    let noise_a = normal_latent(seed + 2, audio.len());
    let mut video: Vec<f32> = tokens2
        .iter()
        .zip(&noise_v)
        .map(|(t, n)| n * s0 + t * (1.0 - s0))
        .collect();
    let mut audio: Vec<f32> = audio
        .iter()
        .zip(&noise_a)
        .map(|(t, n)| n * s0 + t * (1.0 - s0))
        .collect();
    for i in 0..STAGE_2_SIGMAS.len() - 1 {
        exec.check_cancelled()?;
        let (sigma, sigma_next) = (STAGE_2_SIGMAS[i], STAGE_2_SIGMAS[i + 1]);
        let step = DitStep {
            video_latent: video.clone(),
            audio_latent: audio.clone(),
            t_emb: sinusoidal_timestep(sigma * 1000.0, config2.t_dim as usize),
            video_text: v_txt.clone(),
            audio_text: a_txt.clone(),
            video_positions: v_pos2.clone(),
            audio_positions: a_pos.clone(),
            perturb: DitPerturb::default(),
        };
        let (vv, va) = graph2.forward(&step)?;
        let x0_v = x0_of(&video, &vv, sigma);
        let x0_a = x0_of(&audio, &va, sigma);
        video = euler(&video, &x0_v, sigma, sigma_next);
        audio = euler(&audio, &x0_a, sigma, sigma_next);
        if let Some(detail) = non_finite("ltx stage 2 video", &video)
            .or_else(|| non_finite("ltx stage 2 audio", &audio))
        {
            return Err(WeightError::NotBuilt(detail).into());
        }
        exec.progress("denoise", stage1_steps as u32 + (i + 1) as u32, total_steps)?;
    }

    // ---- decode video + audio ----
    let v_final = video_unpatchify(&video, latent, f, h_full, w_full);
    let decoder = LtxVideoDecoder::open(&bundle.base.vae)?;
    let (out_w, out_h, out_f) = decoder.output_dims(w_full, h_full, f);
    if out_f != p.num_frames as i64 {
        return Err(WeightError::NotBuilt(format!(
            "ltx video vae decoded {out_f} frames for {} requested",
            p.num_frames
        ))
        .into());
    }
    let decoded = decoder.decode(ctx, &v_final, w_full, h_full, f)?;

    let a_latent = audio_unpatchify(&audio, a_seq, 8);
    let audio_decoder = LtxAudioDecoder::open(&bundle.audio_vae)?;
    let mel_t = audio_decoder.output_len(a_seq);
    let mel = audio_decoder.decode(ctx, &a_latent, a_seq)?;
    let vocoder = LtxVocoder::open(&bundle.audio_vae)?;
    let waveform = vocoder.decode(ctx, &mel, mel_t)?;

    // ---- persist: PNG sequence + WAV -> MP4 ----
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let work = std::env::temp_dir().join(format!("danmo-ltx-t2v-{}-{nanos}", std::process::id()));
    let _guard = WorkDir(work.clone());
    let frames_dir = work.join("frames");
    std::fs::create_dir_all(&frames_dir)
        .map_err(|err| EngineError::Resolve(format!("create frames dir: {err}")))?;
    let (pw, ph, pf) = (out_w as usize, out_h as usize, out_f as usize);
    let plane = pw * ph;
    for f in 0..pf {
        let mut rgb = vec![0u8; plane * 3];
        for y in 0..ph {
            for x in 0..pw {
                for c in 0..3 {
                    let v = decoded[x + pw * (y + ph * (f + pf * c))];
                    rgb[(y * pw + x) * 3 + c] =
                        ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        let png = crate::common::png::encode_rgb(out_w as u32, out_h as u32, &rgb);
        std::fs::write(frames_dir.join(format!("frame_{:06}.png", f + 1)), png)
            .map_err(|err| EngineError::Resolve(format!("write frame: {err}")))?;
    }
    let wav_path = work.join("audio.wav");
    std::fs::write(&wav_path, wav_bytes(&waveform))
        .map_err(|err| EngineError::Resolve(format!("write wav: {err}")))?;
    let out_path = work.join("video.mp4");
    ffmpeg::mux_png_sequence_wav_to_mp4(
        &frames_dir.join("frame_%06d.png"),
        &wav_path,
        &out_path,
        p.fps as f64,
    )?;
    let bytes = std::fs::read(&out_path)
        .map_err(|err| EngineError::Resolve(format!("read muxed mp4: {err}")))?;
    let duration = pf as f64 / (p.fps.max(1) as f64);
    let metadata = json!({
        "model": p.request.model,
        "mime_type": "video/mp4",
        "frame_count": p.num_frames,
        "fps": p.fps,
        "steps": total_steps,
        "stage2_steps": STAGE_2_SIGMAS.len() - 1,
        "guidance": p.guidance,
        "seed": p.request.seed,
        "audio_channels": 2,
        "audio_sample_rate": 48000,
    });
    let artifact = Artifact::mp4(bytes, out_w, out_h, duration, metadata.clone());
    let mut output = EngineOutput {
        steps: total_steps,
        width: Some(out_w as u32),
        height: Some(out_h as u32),
        frames: Some(out_f),
        ..EngineOutput::default()
    };
    let mut full = metadata;
    full["request_metadata"] = p.request.metadata.clone();
    output.metadata = full;
    output.artifacts.push(artifact);
    Ok(output)
}

// ---------------------------------------------------------------------------
// smoke bundle (session e2e test)
// ---------------------------------------------------------------------------

/// Write a deterministic smoke LTX bundle: a tiny merged GGUF transformer
/// (DiT + connector towers), the same GGUF again as the stage-2
/// transformer, the smoke Gemma (48 blocks so the connector stack has 49
/// states), the smoke connectors safetensors, the smoke video VAE, the
/// smoke audio VAE + vocoder pair, the smoke upscaler, and the toy
/// tokenizer.
pub fn write_smoke_bundle(dir: &Path) -> Result<(), WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let tower = super::dit::TowerConfig {
        dim: 8,
        heads: 2,
        head_dim: 4,
    };
    let dit_config = super::dit::DitConfig {
        video: tower,
        audio: tower,
        cross_heads: 2,
        cross_head_dim: 4,
        layers: 2,
        latent: 128,
        t_dim: 16,
    };
    let conn_config = super::connector::ConnectorConfig::smoke();
    let dit_store: std::collections::HashMap<String, Vec<f32>> =
        super::dit::smoke_store(&dit_config).into_iter().collect();
    let conn_store: std::collections::HashMap<String, Vec<f32>> =
        super::connector::smoke_store(&conn_config)
            .into_iter()
            .collect();
    let dit_entries = super::dit::layout(&dit_config);
    let conn_entries: Vec<_> = super::connector::layout(&conn_config)
        .into_iter()
        .filter(|entry| matches!(entry.source, super::connector::Source::Dit))
        .collect();
    let mut rows: Vec<(&str, i64, i64, &Vec<f32>)> = Vec::new();
    for entry in &dit_entries {
        rows.push((
            entry.name.as_str(),
            entry.ne0,
            entry.ne1,
            &dit_store[entry.name.as_str()],
        ));
    }
    for entry in &conn_entries {
        rows.push((
            entry.name.as_str(),
            entry.ne0,
            entry.ne1,
            &conn_store[entry.name.as_str()],
        ));
    }
    let bytes: Vec<Vec<u8>> = rows
        .iter()
        .map(|(_, _, _, values)| values.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let tensors: Vec<crate::weights::TypedTensor> = rows
        .iter()
        .zip(&bytes)
        .map(|((name, ne0, ne1, _), data)| crate::weights::TypedTensor {
            name,
            ne0: *ne0,
            ne1: *ne1,
            ggml_type: crate::runtime::TYPE_F32,
            data,
        })
        .collect();
    let dit_path = dir.join("ltx_dit_smoke.gguf");
    crate::weights::write_typed_tensors(&dit_path, &tensors)?;
    let stage2 = dir.join("transformer_stage2");
    std::fs::create_dir_all(&stage2).map_err(|err| WeightError::Bundle {
        path: stage2.display().to_string(),
        detail: format!("{err}"),
    })?;
    std::fs::copy(&dit_path, stage2.join("ltx_dit_smoke.gguf")).map_err(|err| {
        WeightError::Bundle {
            path: stage2.display().to_string(),
            detail: format!("{err}"),
        }
    })?;

    let text_dir = dir.join("text_encoder");
    let gemma_config = super::gemma::GemmaConfig {
        dim: 8,
        depth: 48,
        heads: 2,
        kv_heads: 1,
        head_dim: 4,
        ffn: 16,
        vocab: 64,
    };
    super::gemma::write_smoke_with(&text_dir, &gemma_config)?;
    super::connector::write_smoke(&dir.join("connectors"))?;
    let vae_config = super::vae::DecoderConfig {
        latent: 128,
        base: 8,
        mid: 8,
        low: 8,
        tail: 8,
        head: 48,
    };
    super::vae::write_smoke_with(&dir.join("vae"), &vae_config)?;
    let audio_dir = dir.join("audio_vae");
    super::audio::write_smoke(&audio_dir)?;
    super::audio::write_vocoder_smoke(&audio_dir)?;
    let upscaler_config = super::upscaler::UpscalerConfig {
        latent: 128,
        mid: 32,
        blocks: 1,
        groups: 32,
    };
    super::upscaler::write_smoke_with(&dir.join("upscaler"), &upscaler_config)?;
    std::fs::write(
        dir.join("tokenizer.json"),
        super::gemma::smoke_tokenizer_json(),
    )
    .map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("tokenizer.json: {err}"),
    })?;
    std::fs::write(
        dir.join("bundle.manifest.json"),
        r#"{"format":"gguf","quant":"bf16","files":{"transformer":"ltx_dit_smoke.gguf","transformer_stage2":"transformer_stage2/ltx_dit_smoke.gguf","text_encoder":"text_encoder/gemma_smoke.gguf","vae":"vae/ltx_video_vae.safetensors","audio_vae":["audio_vae/ltx_audio_vae.safetensors","audio_vae/ltx_vocoder.safetensors"],"connectors":"connectors/ltx_embeddings_connectors.safetensors","upscaler":"upscaler/upscaler_smoke.safetensors"}}"#,
    )
    .map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("bundle.manifest.json: {err}"),
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_reference_values() {
        // main ltx2_schedule(4, 1024): shift 0.95, stretch terminal 0.1.
        let sigmas = ltx2_schedule(4, 1024);
        assert_eq!(sigmas.len(), 5);
        assert_eq!(sigmas[0], 1.0);
        assert_eq!(sigmas[4], 0.0);
        for i in 1..4 {
            let base = 1.0 - i as f32 / 4.0;
            let shifted = 0.95f32.exp() / (0.95f32.exp() + (1.0 / base - 1.0));
            let scale = (1.0 - (0.95f32.exp() / (0.95f32.exp() + (1.0 / 0.25 - 1.0)))) / 0.9;
            let expect = 1.0 - (1.0 - shifted) / scale;
            assert!(
                (sigmas[i] - expect).abs() < 1e-6,
                "{} vs {expect}",
                sigmas[i]
            );
        }
        // The smallest nonzero sigma is exactly the 0.1 terminal.
        assert!((sigmas[3] - 0.1).abs() < 1e-6, "{}", sigmas[3]);
        // Token count shifts the schedule.
        let small = ltx2_schedule(28, 512);
        let large = ltx2_schedule(28, 4096);
        assert!(large[14] > small[14]);
    }

    #[test]
    fn positions_match_the_reference_formula() {
        let pos = video_positions(2, 2, 2, 24.0);
        assert_eq!(pos.len(), 2 * 2 * 2 * 3);
        // frame 0: mids (max(0-7,0)+max(8-7,0))/2/24 = 0.5/24
        assert!((pos[0] - 0.5 / 24.0).abs() < 1e-7, "{}", pos[0]);
        // frame 1: (1 + 9)/2/24 = 5/24
        let f1 = pos[3 * 4];
        assert!((f1 - 5.0 / 24.0).abs() < 1e-6, "{f1}");
        assert_eq!(pos[1], 16.0);
        assert_eq!(pos[2], 16.0);
        assert_eq!(pos[5], 48.0);
        let a = audio_positions(3);
        assert_eq!(a.len(), 3);
        // i=0: (0 + 1)*160/16000/2 = 0.005
        assert!((a[0] - 0.005).abs() < 1e-9, "{}", a[0]);
        // i=1: (1 + 5)*160/16000/2 = 0.03
        assert!((a[1] - 0.03).abs() < 1e-6, "{}", a[1]);
    }

    #[test]
    fn patchify_roundtrip_and_audio_mapping() {
        let (c, f, h, w) = (3i64, 2, 2, 2);
        let tokens: Vec<f32> = (0..(c * f * h * w) as usize).map(|i| i as f32).collect();
        let lat = video_unpatchify(&tokens, c, f, h, w);
        // token 0 = (f0,h0,w0): channels land at plane starts.
        assert_eq!(lat[0], 0.0);
        assert_eq!(lat[(f * h * w) as usize], 1.0);
        // token 1 = (f0,h0,w1): w offset 1.
        assert_eq!(lat[1], c as f32);
        let back = video_patchify(&lat, c, f, h, w);
        assert_eq!(back, tokens);

        let t = 3i64;
        let atokens: Vec<f32> = (0..(128 * t) as usize).map(|i| i as f32).collect();
        let alat = audio_unpatchify(&atokens, t, 8);
        // k = c*16+f: value k at token t lands at f + 16*(t + T*c).
        assert_eq!(alat[0], 0.0);
        assert_eq!(alat[1], 1.0); // f=1, c=0, t=0
        assert_eq!(alat[16], 128.0); // t=1, c=0, f=0
        assert_eq!(alat[16 * t as usize], 16.0); // c=1, t=0, f=0
    }

    #[test]
    fn guider_and_euler_match_the_reference() {
        let cond = vec![1.0, 2.0, 3.0, 4.0];
        let neg = vec![0.5, 1.5, 2.5, 3.5];
        let ptb = vec![0.8, 1.8, 2.8, 3.8];
        let moda = vec![1.2, 2.2, 3.2, 4.2];
        let pred = guider(&cond, &neg, &ptb, &moda, 3.0);
        // un-rescaled: c + 2(c-n) + 1(c-p) + 2(c-m)
        let raw: Vec<f32> = cond
            .iter()
            .zip(&neg)
            .zip(&ptb)
            .zip(&moda)
            .map(|(((c, n), p), m)| c + 2.0 * (c - n) + (c - p) + 2.0 * (c - m))
            .collect();
        let var = |v: &[f32]| {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / v.len() as f32
        };
        let factor = 0.7 * (var(&cond).sqrt() / (var(&raw).sqrt() + 1e-8)) + 0.3;
        for (p, r) in pred.iter().zip(&raw) {
            assert!((p - r * factor).abs() < 1e-6, "{p} vs {}", r * factor);
        }
        // euler: sigma 0 -> x0; otherwise linear blend.
        let x = vec![1.0, 1.0];
        let x0 = vec![0.0, 2.0];
        assert_eq!(euler(&x, &x0, 0.0, 0.0), x0);
        let stepped = euler(&x, &x0, 0.5, 0.25);
        assert_eq!(stepped, vec![0.5, 1.5]);
        // x0 unwrap: x - sigma * v.
        assert_eq!(x0_of(&[1.0, 2.0], &[0.5, 1.0], 0.5), vec![0.75, 1.5]);
    }

    #[test]
    fn wav_header_is_canonical() {
        let bytes = wav_bytes(&[0.0, 0.5, -0.5, 1.0, -1.0, 2.0]);
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(u16::from_le_bytes([bytes[22], bytes[23]]), 2);
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            48000
        );
        assert_eq!(bytes.len(), 44 + 12);
        // clip: 2.0 -> 32767
        assert_eq!(i16::from_le_bytes([bytes[54], bytes[55]]), 32767);
    }

    fn run_smoke_e2e(model: &str, steps: u32, guidance: f32) -> EngineOutput {
        use crate::contract::{Action, EngineRequest, ExecutionContext, NoopObserver};
        use crate::families::FacadeFamily;
        use crate::ports::ResolvePort;
        use crate::runtime::BackendKind;
        use danmo_contracts::VideoGenerationRequest;

        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_err()
        {
            panic!("ffmpeg not installed");
        }
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-t2v-e2e-{}-{}-{}",
            model,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_smoke_bundle(&dir).unwrap();
        let run = ResolvedRun {
            model: model.into(),
            version: Some("bf16".into()),
            family: "ltx".into(),
            action: Action::VideoCreate,
            bundle: dir.clone(),
            backend: BackendKind::Cpu,
        };
        let req = VideoGenerationRequest {
            model: model.into(),
            prompt: "a cat".into(),
            // The smoke tokenizer only covers this tiny alphabet; the real
            // bundle ships the full Gemma tokenizer.
            negative_prompt: "b".into(),
            size: Some("64x64".into()),
            steps: Some(steps),
            guidance: Some(guidance),
            shift: None,
            seed: Some(3),
            num_frames: Some(9),
            fps: Some(8),
            metadata: serde_json::Value::Null,
            priority: "normal".into(),
        };
        struct Resolver;
        impl ResolvePort for Resolver {
            fn resolve(&self, _model: &str, _action: Action) -> Result<ResolvedRun, EngineError> {
                unreachable!()
            }
            fn resolve_component(
                &self,
                _run: &ResolvedRun,
                _role: &str,
                _model: &str,
            ) -> Result<PathBuf, EngineError> {
                unreachable!()
            }
            fn read_source_audio(
                &self,
                _run: &ResolvedRun,
                _source: &str,
            ) -> Result<Vec<u8>, EngineError> {
                unreachable!()
            }
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let exec = ExecutionContext::new("task", std::sync::Arc::new(NoopObserver));
        let mut session = super::super::LtxVideoCreateFamily
            .prepare(
                &ctx,
                &run,
                &EngineRequest::VideoCreate(req),
                &Resolver,
                &exec,
            )
            .unwrap();
        let out = session.infer(&ctx, &exec).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        out
    }

    fn assert_smoke_output(out: &EngineOutput, steps: u32) {
        assert_eq!(out.steps, steps);
        assert_eq!(out.width, Some(64));
        assert_eq!(out.height, Some(64));
        assert_eq!(out.frames, Some(9));
        assert_eq!(out.artifacts.len(), 1);
        let artifact = &out.artifacts[0];
        assert_eq!(artifact.mime, "video/mp4");
        assert_eq!(artifact.width, Some(64));
        assert_eq!(artifact.height, Some(64));
        assert!(artifact.bytes.len() > 64, "mp4 is too small");
        assert_eq!(&artifact.bytes[4..8], b"ftyp", "not an mp4");
    }

    #[test]
    fn t2v_runs_end_to_end_on_the_smoke_bundle() {
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_err()
        {
            eprintln!("ffmpeg not installed; skipping ltx t2v e2e");
            return;
        }
        let out = run_smoke_e2e("ltx-2.3-dev", 2, 2.0);
        assert_smoke_output(&out, 2 + 3);
    }

    #[test]
    fn distilled_t2v_runs_end_to_end_on_the_smoke_bundle() {
        if std::process::Command::new("ffmpeg")
            .arg("-version")
            .output()
            .is_err()
        {
            eprintln!("ffmpeg not installed; skipping ltx distilled t2v e2e");
            return;
        }
        // steps=2 slices DISTILLED_SIGMAS[..3]; the fixed 3 stage-2 refine
        // steps follow. Guidance is forced to 1 for the distilled model.
        let out = run_smoke_e2e("ltx-2.3-distilled", 2, 5.0);
        assert_smoke_output(&out, 2 + 3);
        assert_eq!(out.metadata["guidance"], 1.0);
    }
}
