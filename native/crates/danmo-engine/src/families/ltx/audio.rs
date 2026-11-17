//! LTX 2.3 audio stack (`audio_vae.decoder.*`, `vocoder.vocoder.*`,
//! `vocoder.bwe_generator.*`, `vocoder.mel_stft.*` of the published audio VAE
//! safetensors, BF16). Port of main's `LTX23AudioDecoder` and `LTX23Vocoder`
//! (`vae_mlx.py`):
//! - the audio VAE decoder maps a latent `[freq=16, time=T, ch=8]` to a mel
//!   `[mel=64, time=4T-3, ch=2]`. Every conv is a causal 3x3 `WrappedConv2d`:
//!   the time axis (ggml `ne[1]`) is zero-padded two rows at the top only, the
//!   frequency axis (`ne[0]`) is zero-padded one column per side. Residual
//!   blocks are pre-activation with a weightless per-pixel RMS norm (eps 1e-6,
//!   the audio section of main's file, not the video 1e-8). The up blocks run
//!   in reverse index order (`up.2`, `up.1`, `up.0`); `up.2`/`up.1` upsample
//!   both axes x2 nearest, conv, then drop the first time row, so each takes
//!   `T -> 2T - 1` and `F -> 2F`. The latent is denormalized per channel with
//!   `audio_vae.per_channel_statistics.{mean,std}-of-means` first.
//! - the vocoder is BigVGAN v2: `conv_pre` (7 taps), upsample stages of
//!   transposed 1D convs (ggml's `conv_transpose_1d` requires padding 0, so the
//!   symmetric `(k - r) / 2` crop is a view) followed by the average of three
//!   anti-aliased multi-periodicity residual blocks (dilations 1/3/5, kernels
//!   3/7/11), then `act_post` and a bias-free `conv_post`. The base vocoder
//!   clips to [-1, 1]; the BWE generator does not. SnakeBeta weights are stored
//!   in log scale; `exp(alpha)` and `1 / (exp(beta) + 1e-9)` are folded at load
//!   time. The anti-aliasing filters are depthwise 12-tap convs run as a direct
//!   2D conv with the channels in the batch axis (exact F32, no im2col F16).
//! - the full pipeline resamples the base vocoder's 16 kHz output x3 with a
//!   Hann-windowed sinc kernel (host-built, 43 taps, rolloff 0.99), computes a
//!   log-mel of that output (causal STFT conv, n_fft 512, hop 80, 64 mels),
//!   runs the BWE generator on it, and emits
//!   `clamp(skip + bwe_residual, -1, 1)` at 48 kHz. The 160-sample base hop is
//!   always a multiple of the STFT hop 80, so main's pad-to-multiple branch
//!   never fires and all shapes are static for a given mel length.
//!
//! PyTorch conv bytes double as the ggml kernels: `[O, I, KH, KW]` is
//! `[KW, KH, IC, OC]`, `[O, I, K]` is `[K, IC, OC]`, and the transposed conv's
//! `[I, O, K]` is `[K, OC, IC]`; BF16 is widened to F32 because ggml's direct
//! convs only take F16/F32 kernels.

use crate::runtime::{shape, GgmlContext, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;

const DECODER_GRAPH_MEM: usize = 128 << 20;
const VOCODER_GRAPH_MEM: usize = 512 << 20;
/// Audio VAE pixel norm eps (main's audio section uses 1e-6).
const PIXEL_NORM_EPS: f32 = 1e-6;
/// Latent frequency bins (the 16 in the `[16, T, 8]` latent).
const FREQ: i64 = 16;
/// Mel bins per audio channel.
const MEL: i64 = 64;
/// SnakeBeta / anti-alias filter taps.
const FILTER_TAPS: i64 = 12;
/// AMPBlock1 conv kernels and dilations (main's resblock config).
const RES_KERNELS: [i64; 3] = [3, 7, 11];
const DILATIONS: [i64; 3] = [1, 3, 5];
/// STFT constants of the BWE mel front end.
const N_FFT: i64 = 512;
const HOP: i64 = 80;
const N_BINS: i64 = N_FFT / 2 + 1;
const STFT_LEFT_PAD: i64 = N_FFT - HOP;
/// Hann-sinc resampler constants (main's `HannSincResampler`).
const RESAMPLE_FACTOR: i64 = 3;
const RESAMPLE_ROLLOFF: f64 = 0.99;
const RESAMPLE_LPFW: f64 = 6.0;
const RESAMPLE_WIDTH: i64 = 7; // ceil(6 / 0.99)
const RESAMPLE_TAPS: i64 = 2 * RESAMPLE_WIDTH * RESAMPLE_FACTOR + 1; // 43

const AUDIO_MEAN: &str = "audio_vae.per_channel_statistics.mean-of-means";
const AUDIO_STD: &str = "audio_vae.per_channel_statistics.std-of-means";
const MEL_BASIS: &str = "vocoder.mel_stft.mel_basis";
const STFT_BASIS: &str = "vocoder.mel_stft.stft_fn.forward_basis";
const RESAMPLER_KERNEL: &str = "vocoder.resampler.kernel";

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("ltx audio graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let actual = shape(tensor);
    if actual != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {actual:?}, the ltx audio stack expects {ne:?}"
        )));
    }
    Ok(tensor)
}

/// Exact zeros shaped `[ne0, ne1, ne2]`: `x - x` on a scratch tensor.
fn zeros(b: &GraphBuilder, ne0: i64, ne1: i64, ne2: i64) -> Result<Tensor, RuntimeError> {
    let scratch = nn(b.blank_f32_3d(ne0, ne1, ne2))?;
    nn(b.sub(scratch, scratch))
}

/// Replicate-pad rows (`ne[0]`): `left` copies of row 0 in front, `right`
/// copies of the last row at the back. `x` is `[t, c]` contiguous.
fn replicate_pad(
    b: &GraphBuilder,
    x: Tensor,
    t: i64,
    c: i64,
    left: i64,
    right: i64,
) -> Result<Tensor, RuntimeError> {
    let nb1 = (t * 4) as usize;
    let first = nn(b.cont(b.view_2d(x, 1, c, nb1, 0)))?;
    let first = nn(b.cont(b.repeat_4d(first, left, c, 1, 1)))?;
    let last = nn(b.cont(b.view_2d(x, 1, c, nb1, ((t - 1) * 4) as usize)))?;
    let last = nn(b.cont(b.repeat_4d(last, right, c, 1, 1)))?;
    let y = nn(b.cont(nn(b.concat(first, x, 0))?))?;
    nn(b.cont(nn(b.concat(y, last, 0))?))
}

/// Insert a zero after every row: `[t, c]` -> `[2t, c]` with `out[2i] = x[i]`.
fn interleave_zeros(b: &GraphBuilder, x: Tensor, t: i64, c: i64) -> Result<Tensor, RuntimeError> {
    let x3 = nn(b.reshape_3d(x, 1, t, c))?;
    let z = zeros(b, 1, t, c)?;
    let y = nn(b.cont(nn(b.concat(x3, z, 0))?))?;
    nn(b.reshape_2d(y, 2 * t, c))
}

/// Bias add on a `[t, c]` tensor.
fn add_bias(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    name: &str,
    t: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let bias = nn(b.cont(b.reshape_2d(wt(weights, name)?, 1, c)))?;
    nn(b.add(x, nn(b.repeat(bias, x))?))
}

// ---------------------------------------------------------------------------
// audio VAE decoder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioDecoderConfig {
    /// Latent channels (8).
    pub latent: i64,
    /// conv_in / mid / up.2 channels (512).
    pub base: i64,
    /// up.1 output channels (256).
    pub mid_up: i64,
    /// up.0 output channels (128).
    pub tail: i64,
    /// conv_out output channels (2).
    pub mel: i64,
}

impl AudioDecoderConfig {
    pub fn smoke() -> Self {
        Self {
            latent: 8,
            base: 16,
            mid_up: 8,
            tail: 8,
            mel: 2,
        }
    }

    /// Up blocks in run order: `(index, in, out, upsample)`.
    fn up_blocks(&self) -> [(i64, i64, i64, bool); 3] {
        [
            (2, self.base, self.base, true),
            (1, self.base, self.mid_up, true),
            (0, self.mid_up, self.tail, false),
        ]
    }

    /// Recover the architecture from the weight shapes. The block structure
    /// (res counts, upsample placement, reverse run order) is the published
    /// fixed one; anything else fails loud.
    pub fn infer(w: &GgufWeights) -> Result<Self, WeightError> {
        let fail = |detail: String| WeightError::NotBuilt(format!("ltx audio vae: {detail}"));
        let stats = w.shape(AUDIO_MEAN)?[0];
        if w.shape(AUDIO_STD)?[0] != stats || stats % FREQ != 0 || stats < FREQ {
            return Err(fail(format!(
                "per-channel statistics disagree or are not 16 * latent: mean {stats} std {}",
                w.shape(AUDIO_STD)?[0]
            )));
        }
        let latent = stats / FREQ;
        let conv_in = w.shape("audio_vae.decoder.conv_in.conv.weight")?;
        if conv_in != [3, 3, latent, conv_in[3]] || conv_in[3] % latent != 0 {
            return Err(fail(format!(
                "audio_vae.decoder.conv_in.conv.weight shape {conv_in:?} is not [3, 3, {latent}, base]"
            )));
        }
        let base = conv_in[3];
        for block in ["block_1", "block_2"] {
            for conv in ["conv1", "conv2"] {
                let name = format!("audio_vae.decoder.mid.{block}.{conv}.conv.weight");
                let s = w.shape(&name)?;
                if s != [3, 3, base, base] {
                    return Err(fail(format!(
                        "{name} shape {s:?} is not [3, 3, {base}, {base}]"
                    )));
                }
            }
        }
        // up.2 runs at base channels; up.1 block 0 pins mid_up; up.0 block 0
        // pins tail. Everything else must agree.
        let probe = AudioDecoderConfig {
            latent,
            base,
            mid_up: base,
            tail: base,
            mel: 0,
        };
        let mut channels = [base, 0, 0];
        for (run, (index, _, _, upsample)) in probe.up_blocks().iter().enumerate() {
            let expected_in = if run == 0 { base } else { channels[run - 1] };
            let first = w.shape(&format!(
                "audio_vae.decoder.up.{index}.block.0.conv1.conv.weight"
            ))?;
            if first[0..2] != [3, 3] {
                return Err(fail(format!(
                    "audio_vae.decoder.up.{index}.block.0.conv1.conv.weight shape {first:?} is not 3x3"
                )));
            }
            if first[2] != expected_in {
                return Err(fail(format!(
                    "audio_vae.decoder.up.{index}.block.0.conv1.conv.weight shape {first:?} does not take {expected_in} input channels"
                )));
            }
            channels[run] = first[3];
            let oc = channels[run];
            for block in 0..3 {
                let (want_ic, want_oc) = if block == 0 {
                    (expected_in, oc)
                } else {
                    (oc, oc)
                };
                let c1 = w.shape(&format!(
                    "audio_vae.decoder.up.{index}.block.{block}.conv1.conv.weight"
                ))?;
                if c1 != [3, 3, want_ic, want_oc] {
                    return Err(fail(format!(
                        "audio_vae.decoder.up.{index}.block.{block}.conv1.conv.weight shape {c1:?} is not [3, 3, {want_ic}, {want_oc}]"
                    )));
                }
                let c2 = w.shape(&format!(
                    "audio_vae.decoder.up.{index}.block.{block}.conv2.conv.weight"
                ))?;
                if c2 != [3, 3, want_oc, want_oc] {
                    return Err(fail(format!(
                        "audio_vae.decoder.up.{index}.block.{block}.conv2.conv.weight shape {c2:?} is not [3, 3, {want_oc}, {want_oc}]"
                    )));
                }
                let nin =
                    format!("audio_vae.decoder.up.{index}.block.{block}.nin_shortcut.conv.weight");
                if want_ic != want_oc {
                    let s = w.shape(&nin)?;
                    if s != [1, 1, want_ic, want_oc] {
                        return Err(fail(format!(
                            "{nin} shape {s:?} is not [1, 1, {want_ic}, {want_oc}]"
                        )));
                    }
                }
            }
            if *upsample {
                let s = w.shape(&format!(
                    "audio_vae.decoder.up.{index}.upsample.conv.conv.weight"
                ))?;
                if s != [3, 3, oc, oc] {
                    return Err(fail(format!(
                        "audio_vae.decoder.up.{index}.upsample.conv.conv.weight shape {s:?} is not [3, 3, {oc}, {oc}]"
                    )));
                }
            }
        }
        let conv_out = w.shape("audio_vae.decoder.conv_out.conv.weight")?;
        if conv_out[0..2] != [3, 3] || conv_out[2] != channels[2] {
            return Err(fail(format!(
                "audio_vae.decoder.conv_out.conv.weight shape {conv_out:?} does not take {} input channels",
                channels[2]
            )));
        }
        Ok(Self {
            latent,
            base,
            mid_up: channels[1],
            tail: channels[2],
            mel: conv_out[3],
        })
    }
}

/// One decoder conv: `{name}.conv.{weight,bias}`, kernel 3x3 unless `one_by_one`.
struct AudioConv {
    name: String,
    ic: i64,
    oc: i64,
    one_by_one: bool,
}

fn decoder_convs(config: &AudioDecoderConfig) -> Vec<AudioConv> {
    let mut out = Vec::new();
    let mut push = |name: String, ic: i64, oc: i64, one_by_one: bool| {
        out.push(AudioConv {
            name,
            ic,
            oc,
            one_by_one,
        })
    };
    push(
        "audio_vae.decoder.conv_in".into(),
        config.latent,
        config.base,
        false,
    );
    for block in ["block_1", "block_2"] {
        for conv in ["conv1", "conv2"] {
            push(
                format!("audio_vae.decoder.mid.{block}.{conv}"),
                config.base,
                config.base,
                false,
            );
        }
    }
    for (index, ic, oc, upsample) in config.up_blocks() {
        for block in 0..3 {
            let (bic, boc) = if block == 0 { (ic, oc) } else { (oc, oc) };
            for conv in ["conv1", "conv2"] {
                push(
                    format!("audio_vae.decoder.up.{index}.block.{block}.{conv}"),
                    if conv == "conv1" { bic } else { boc },
                    boc,
                    false,
                );
            }
            if bic != boc {
                push(
                    format!("audio_vae.decoder.up.{index}.block.{block}.nin_shortcut"),
                    bic,
                    boc,
                    true,
                );
            }
        }
        if upsample {
            push(
                format!("audio_vae.decoder.up.{index}.upsample.conv"),
                oc,
                oc,
                false,
            );
        }
    }
    push(
        "audio_vae.decoder.conv_out".into(),
        config.tail,
        config.mel,
        false,
    );
    out
}

/// Causal 3x3 conv on a `[w, h, ic, 1]` tensor: two zero rows on top of the
/// time axis (`ne[1]`), one zero column per side on the frequency axis.
fn conv2d_causal(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    _ic: i64,
    oc: i64,
) -> Result<Tensor, RuntimeError> {
    let x = nn(b.pad_ext(x, 0, 0, 2, 0, 0, 0, 0, 0))?;
    let kernel = wt(weights, &format!("{prefix}.conv.weight"))?;
    let y = nn(b.conv2d_direct(kernel, x, 1, 1, 1, 0, 1, 1))?;
    let y = shaped(
        nn(b.cont(y))?,
        [w, h, oc, 1],
        &format!("{prefix}.conv.weight"),
    )?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.conv.bias"))?, 1, 1, oc, 1)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    shaped(y, [w, h, oc, 1], &format!("{prefix}.conv.bias"))
}

/// 1x1 shortcut conv (no padding).
fn conv2d_1x1(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    _ic: i64,
    oc: i64,
) -> Result<Tensor, RuntimeError> {
    let kernel = wt(weights, &format!("{prefix}.conv.weight"))?;
    let y = nn(b.conv2d_direct(kernel, x, 1, 1, 0, 0, 1, 1))?;
    let y = shaped(
        nn(b.cont(y))?,
        [w, h, oc, 1],
        &format!("{prefix}.conv.weight"),
    )?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.conv.bias"))?, 1, 1, oc, 1)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    shaped(y, [w, h, oc, 1], &format!("{prefix}.conv.bias"))
}

/// Weightless per-pixel RMS norm across channels (`ne[2]`), eps 1e-6.
/// `ggml_permute` uses destination semantics: `ne[axis_i] = old ne[i]`.
fn audio_pixel_norm(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.cont(b.permute(x, 1, 2, 0, 3)))?; // [c, w, h, 1]
    let y = nn(b.rms_norm(y, PIXEL_NORM_EPS))?;
    let y = nn(b.cont(b.permute(y, 2, 0, 1, 3)))?; // [w, h, c, 1]
    shaped(y, [w, h, c, 1], "ltx audio vae pixel_norm")
}

fn audio_res_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    ic: i64,
    oc: i64,
) -> Result<Tensor, RuntimeError> {
    let y = audio_pixel_norm(b, x, w, h, ic)?;
    let y = nn(b.silu(y))?;
    let y = conv2d_causal(b, weights, y, &format!("{prefix}.conv1"), w, h, ic, oc)?;
    let y = audio_pixel_norm(b, y, w, h, oc)?;
    let y = nn(b.silu(y))?;
    let y = conv2d_causal(b, weights, y, &format!("{prefix}.conv2"), w, h, oc, oc)?;
    let residual = if ic != oc {
        conv2d_1x1(
            b,
            weights,
            x,
            &format!("{prefix}.nin_shortcut"),
            w,
            h,
            ic,
            oc,
        )?
    } else {
        x
    };
    nn(b.add(y, residual))
}

/// x2 nearest on both axes, causal conv, drop the first time row:
/// `[w, h, c, 1]` -> `[2w, 2h - 1, c, 1]`.
fn audio_upsample(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.upscale_nearest(x))?;
    let y = shaped(
        nn(b.cont(y))?,
        [2 * w, 2 * h, c, 1],
        "ltx audio vae upsample",
    )?;
    let y = conv2d_causal(b, weights, y, prefix, 2 * w, 2 * h, c, c)?;
    let view = nn(b.view_4d(
        y,
        2 * w,
        2 * h - 1,
        c,
        1,
        (2 * w * 4) as usize,
        (2 * w * 2 * h * 4) as usize,
        (2 * w * 2 * h * c * 4) as usize,
        (2 * w * 4) as usize,
    ))?;
    shaped(
        nn(b.cont(view))?,
        [2 * w, 2 * h - 1, c, 1],
        "ltx audio vae upsample row drop",
    )
}

pub struct LtxAudioDecoder {
    config: AudioDecoderConfig,
    loaded: Vec<(String, Vec<f32>)>,
}

impl LtxAudioDecoder {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let config = AudioDecoderConfig::infer(&weights)?;
        let mut loaded = Vec::new();
        for conv in decoder_convs(&config) {
            let k = if conv.one_by_one { 1 } else { 3 };
            let weight = format!("{}.conv.weight", conv.name);
            let data = weights.f32_widened(&weight, &[k, k, conv.ic, conv.oc])?;
            loaded.push((weight, data));
            let bias = format!("{}.conv.bias", conv.name);
            let data = weights.f32_widened(&bias, &[conv.oc])?;
            loaded.push((bias, data));
        }
        // The file stores the statistics as a flat `c * 16 + f` vector
        // (`(channel, freq)` order); the ggml latent is `[freq, time, channel]`,
        // so the constant is reordered to `f + 16 * c`.
        for name in [AUDIO_MEAN, AUDIO_STD] {
            let raw = weights.f32_widened(name, &[FREQ * config.latent])?;
            let mut reordered = vec![0f32; raw.len()];
            for c in 0..config.latent as usize {
                for f in 0..FREQ as usize {
                    reordered[f + FREQ as usize * c] = raw[c * FREQ as usize + f];
                }
            }
            loaded.push((name.to_string(), reordered));
        }
        Ok(Self { config, loaded })
    }

    pub fn config(&self) -> AudioDecoderConfig {
        self.config
    }

    /// Mel length for a `t`-frame latent: two `T -> 2T - 1` upsamples.
    pub fn output_len(&self, t: i64) -> i64 {
        4 * t - 3
    }

    /// Decode a `[freq=16, time=t, ch=latent]` latent (ggml order) into a
    /// `[mel=64, time=4t-3, ch=2]` mel in ggml order.
    pub fn decode(
        &self,
        ctx: &GgmlContext,
        latent: &[f32],
        t: i64,
    ) -> Result<Vec<f32>, WeightError> {
        let config = &self.config;
        if t < 1 {
            return Err(WeightError::Shape {
                name: "ltx audio vae decode dims".into(),
                expected: vec![1],
                got: vec![t],
            });
        }
        if latent.len() as i64 != FREQ * t * config.latent {
            return Err(WeightError::Shape {
                name: "ltx audio vae latent".into(),
                expected: vec![FREQ, t, config.latent],
                got: vec![latent.len() as i64],
            });
        }
        let mut b = ctx.builder(DECODER_GRAPH_MEM)?;
        let mut wmap = HashMap::new();
        for conv in decoder_convs(config) {
            let k = if conv.one_by_one { 1 } else { 3 };
            let weight = format!("{}.conv.weight", conv.name);
            wmap.insert(
                weight.clone(),
                nn(b.f32_4d(&weight, k, k, conv.ic, conv.oc, true))?,
            );
            let bias = format!("{}.conv.bias", conv.name);
            wmap.insert(bias.clone(), nn(b.f32_1d(&bias, conv.oc, true))?);
        }
        for name in [AUDIO_MEAN, AUDIO_STD] {
            wmap.insert(
                name.to_string(),
                nn(b.f32_2d(name, FREQ, config.latent, true))?,
            );
        }
        let z = nn(b.f32_3d("ltx_audio_latent", FREQ, t, config.latent, true))?;
        let z4 = nn(b.reshape_4d(z, FREQ, t, config.latent, 1))?;
        // Denormalize: x * std + mean per channel.
        let std = nn(b.cont(b.reshape_4d(wt(&wmap, AUDIO_STD)?, FREQ, 1, config.latent, 1)))?;
        let mean = nn(b.cont(b.reshape_4d(wt(&wmap, AUDIO_MEAN)?, FREQ, 1, config.latent, 1)))?;
        let z4 = nn(b.mul(z4, nn(b.repeat(std, z4))?))?;
        let mut x = nn(b.add(z4, nn(b.repeat(mean, z4))?))?;

        let (mut cw, mut ct) = (FREQ, t);
        x = conv2d_causal(
            &b,
            &wmap,
            x,
            "audio_vae.decoder.conv_in",
            cw,
            ct,
            config.latent,
            config.base,
        )?;
        for block in ["block_1", "block_2"] {
            x = audio_res_block(
                &b,
                &wmap,
                x,
                &format!("audio_vae.decoder.mid.{block}"),
                cw,
                ct,
                config.base,
                config.base,
            )?;
        }
        for (index, ic, oc, upsample) in config.up_blocks() {
            for block in 0..3 {
                let (bic, boc) = if block == 0 { (ic, oc) } else { (oc, oc) };
                x = audio_res_block(
                    &b,
                    &wmap,
                    x,
                    &format!("audio_vae.decoder.up.{index}.block.{block}"),
                    cw,
                    ct,
                    bic,
                    boc,
                )?;
            }
            if upsample {
                x = audio_upsample(
                    &b,
                    &wmap,
                    x,
                    &format!("audio_vae.decoder.up.{index}.upsample.conv"),
                    cw,
                    ct,
                    oc,
                )?;
                cw *= 2;
                ct = 2 * ct - 1;
            }
        }
        x = audio_pixel_norm(&b, x, cw, ct, config.tail)?;
        x = nn(b.silu(x))?;
        x = conv2d_causal(
            &b,
            &wmap,
            x,
            "audio_vae.decoder.conv_out",
            cw,
            ct,
            config.tail,
            config.mel,
        )?;
        let x = shaped(
            nn(b.reshape_3d(x, cw, ct, config.mel))?,
            [MEL, self.output_len(t), config.mel, 1],
            "ltx audio vae output",
        )?;
        let mut graph = b.finish(x)?;
        for (name, data) in &self.loaded {
            graph.set_f32(name, data)?;
        }
        graph.set_f32("ltx_audio_latent", latent)?;
        let out = graph.compute()?;
        if let Some(detail) = crate::common::ops::non_finite("ltx audio vae", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// BigVGAN vocoder + BWE
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct VocoderConfig {
    /// Weight prefix: `vocoder.vocoder` (base) or `vocoder.bwe_generator`.
    pub prefix: String,
    /// conv_pre output channels (1536 / 512).
    pub initial: i64,
    /// Transposed conv strides per stage; the rates and kernel sizes are the
    /// published structural constants of the prefix, not read from shapes.
    pub rates: Vec<i64>,
    pub kernels: Vec<i64>,
    /// The base vocoder clamps its output to [-1, 1]; the BWE residual does not.
    pub clip: bool,
}

impl VocoderConfig {
    /// Structural constants of a known prefix: `(rates, kernels, clip)`.
    fn structural(prefix: &str) -> Result<(Vec<i64>, Vec<i64>, bool), WeightError> {
        match prefix {
            "vocoder.vocoder" => Ok((vec![5, 2, 2, 2, 2, 2], vec![11, 4, 4, 4, 4, 4], true)),
            "vocoder.bwe_generator" => Ok((vec![6, 5, 2, 2, 2], vec![12, 11, 4, 4, 4], false)),
            other => Err(WeightError::NotBuilt(format!(
                "ltx vocoder: unknown prefix {other}"
            ))),
        }
    }

    pub fn smoke(prefix: &str) -> Result<Self, WeightError> {
        let (rates, kernels, clip) = Self::structural(prefix)?;
        let initial = if prefix == "vocoder.vocoder" { 64 } else { 32 };
        Ok(Self {
            prefix: prefix.to_string(),
            initial,
            rates,
            kernels,
            clip,
        })
    }

    /// Validate the published structure against the weight shapes; only the
    /// conv_pre width is taken from the file. Anything else fails loud.
    pub fn infer(w: &GgufWeights, prefix: &str) -> Result<Self, WeightError> {
        let fail = |detail: String| WeightError::NotBuilt(format!("ltx vocoder: {detail}"));
        let (rates, kernels, clip) = Self::structural(prefix)?;
        let conv_pre = w.shape(&format!("{prefix}.conv_pre.weight"))?;
        if conv_pre[0] != 7 || conv_pre[1] != 2 * MEL {
            return Err(fail(format!(
                "{prefix}.conv_pre.weight shape {conv_pre:?} is not [7, {}, initial]",
                2 * MEL
            )));
        }
        let config = Self {
            prefix: prefix.to_string(),
            initial: conv_pre[2],
            rates,
            kernels,
            clip,
        };
        let ch = config.channels();
        for (stage, (&rate, &k)) in config.rates.iter().zip(&config.kernels).enumerate() {
            if (k - rate) % 2 != 0 || k < rate {
                return Err(fail(format!(
                    "{prefix} stage {stage} kernel {k} and rate {rate} do not admit symmetric cropping"
                )));
            }
            let s = w.shape(&format!("{prefix}.ups.{stage}.weight"))?;
            if s[..3] != [k, ch[stage + 1], ch[stage]] {
                return Err(fail(format!(
                    "{prefix}.ups.{stage}.weight shape {s:?} is not [{k}, {}, {}]",
                    ch[stage + 1],
                    ch[stage]
                )));
            }
            for j in 0..RES_KERNELS.len() {
                let rb = stage * RES_KERNELS.len() + j;
                let c = ch[stage + 1];
                for conv in ["convs1", "convs2"] {
                    let name = format!("{prefix}.resblocks.{rb}.{conv}.{j}.weight");
                    let s = w.shape(&name)?;
                    if s[..3] != [RES_KERNELS[j], c, c] {
                        return Err(fail(format!(
                            "{name} shape {s:?} is not [{}, {c}, {c}]",
                            RES_KERNELS[j]
                        )));
                    }
                }
                for act in ["acts1", "acts2"] {
                    for tap in ["upsample.filter", "downsample.lowpass.filter"] {
                        let name = format!("{prefix}.resblocks.{rb}.{act}.{j}.{tap}");
                        let s = w.shape(&name)?;
                        if s[0] != FILTER_TAPS {
                            return Err(fail(format!(
                                "{name} shape {s:?} is not [{FILTER_TAPS}, 1, 1]"
                            )));
                        }
                    }
                }
            }
        }
        let last = *ch.last().expect("one stage or more");
        let conv_post = w.shape(&format!("{prefix}.conv_post.weight"))?;
        if conv_post[..3] != [7, last, 2] {
            return Err(fail(format!(
                "{prefix}.conv_post.weight shape {conv_post:?} is not [7, {last}, 2]"
            )));
        }
        Ok(config)
    }

    /// Channels per stage boundary: `initial, initial/2, ...`.
    fn channels(&self) -> Vec<i64> {
        let mut out = vec![self.initial];
        for _ in &self.rates {
            let next = out.last().expect("initial channel") / 2;
            out.push(next);
        }
        out
    }

    /// Total upsample ratio of the vocoder.
    pub fn hop(&self) -> i64 {
        self.rates.iter().product()
    }
}

/// Folded SnakeBeta weights: `exp(alpha)` and `1 / (exp(beta) + 1e-9)`.
fn fold_snake(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|v| v.exp()).collect()
}

fn fold_snake_inv(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|v| 1.0 / (v.exp() + 1e-9)).collect()
}

/// The host-built Hann-windowed sinc kernel of the x3 resampler (43 taps).
pub fn resampler_kernel() -> Vec<f32> {
    (0..RESAMPLE_TAPS)
        .map(|i| {
            let time =
                (i as f64 / RESAMPLE_FACTOR as f64 - RESAMPLE_WIDTH as f64) * RESAMPLE_ROLLOFF;
            let clamped = time.clamp(-RESAMPLE_LPFW, RESAMPLE_LPFW);
            let window = (clamped * std::f64::consts::PI / RESAMPLE_LPFW / 2.0)
                .cos()
                .powi(2);
            let sinc = if time == 0.0 {
                1.0
            } else {
                (time * std::f64::consts::PI).sin() / (time * std::f64::consts::PI)
            };
            (sinc * window * RESAMPLE_ROLLOFF / RESAMPLE_FACTOR as f64) as f32
        })
        .collect()
}

/// One vocoder weight tensor: the graph shape is what the ops consume
/// (conv1d kernels carry the `KH = 1` axis), the file shape is the reversed
/// PyTorch order the safetensors reader reports.
struct VocTensor {
    name: String,
    graph: [i64; 4],
    file: Vec<i64>,
}

fn voc_tensor(name: String, graph: [i64; 4], file: Vec<i64>) -> VocTensor {
    VocTensor { name, graph, file }
}

/// Weight tensors of one vocoder.
fn vocoder_weight_tensors(config: &VocoderConfig) -> Vec<VocTensor> {
    let ch = config.channels();
    let p = &config.prefix;
    let mut out = vec![
        voc_tensor(
            format!("{p}.conv_pre.weight"),
            [7, 1, 2 * MEL, ch[0]],
            vec![7, 2 * MEL, ch[0]],
        ),
        voc_tensor(format!("{p}.conv_pre.bias"), [ch[0], 1, 1, 1], vec![ch[0]]),
    ];
    for (stage, (&_, &k)) in config.rates.iter().zip(&config.kernels).enumerate() {
        out.push(voc_tensor(
            format!("{p}.ups.{stage}.weight"),
            [k, ch[stage + 1], ch[stage], 1],
            vec![k, ch[stage + 1], ch[stage]],
        ));
        out.push(voc_tensor(
            format!("{p}.ups.{stage}.bias"),
            [ch[stage + 1], 1, 1, 1],
            vec![ch[stage + 1]],
        ));
    }
    for rb in 0..config.rates.len() * RES_KERNELS.len() {
        let c = ch[rb / RES_KERNELS.len() + 1];
        for j in 0..RES_KERNELS.len() {
            for conv in ["convs1", "convs2"] {
                out.push(voc_tensor(
                    format!("{p}.resblocks.{rb}.{conv}.{j}.weight"),
                    [RES_KERNELS[j], 1, c, c],
                    vec![RES_KERNELS[j], c, c],
                ));
                out.push(voc_tensor(
                    format!("{p}.resblocks.{rb}.{conv}.{j}.bias"),
                    [c, 1, 1, 1],
                    vec![c],
                ));
            }
            for act in ["acts1", "acts2"] {
                out.push(voc_tensor(
                    format!("{p}.resblocks.{rb}.{act}.{j}.act.alpha"),
                    [c, 1, 1, 1],
                    vec![c],
                ));
                out.push(voc_tensor(
                    format!("{p}.resblocks.{rb}.{act}.{j}.act.beta"),
                    [c, 1, 1, 1],
                    vec![c],
                ));
                out.push(voc_tensor(
                    format!("{p}.resblocks.{rb}.{act}.{j}.upsample.filter"),
                    [FILTER_TAPS, 1, 1, 1],
                    vec![FILTER_TAPS],
                ));
                out.push(voc_tensor(
                    format!("{p}.resblocks.{rb}.{act}.{j}.downsample.lowpass.filter"),
                    [FILTER_TAPS, 1, 1, 1],
                    vec![FILTER_TAPS],
                ));
            }
        }
    }
    let last = *ch.last().expect("one stage or more");
    out.push(voc_tensor(
        format!("{p}.act_post.act.alpha"),
        [last, 1, 1, 1],
        vec![last],
    ));
    out.push(voc_tensor(
        format!("{p}.act_post.act.beta"),
        [last, 1, 1, 1],
        vec![last],
    ));
    out.push(voc_tensor(
        format!("{p}.act_post.upsample.filter"),
        [FILTER_TAPS, 1, 1, 1],
        vec![FILTER_TAPS],
    ));
    out.push(voc_tensor(
        format!("{p}.act_post.downsample.lowpass.filter"),
        [FILTER_TAPS, 1, 1, 1],
        vec![FILTER_TAPS],
    ));
    out.push(voc_tensor(
        format!("{p}.conv_post.weight"),
        [7, 1, last, 2],
        vec![7, last, 2],
    ));
    out
}

fn load_vocoder(
    w: &GgufWeights,
    config: &VocoderConfig,
    loaded: &mut Vec<(String, Vec<f32>)>,
) -> Result<(), WeightError> {
    for tensor in vocoder_weight_tensors(config) {
        let raw = w.f32_widened(&tensor.name, &tensor.file)?;
        if tensor.name.ends_with(".act.alpha") {
            loaded.push((tensor.name, fold_snake(&raw)));
        } else if tensor.name.ends_with(".act.beta") {
            loaded.push((tensor.name, fold_snake_inv(&raw)));
        } else {
            loaded.push((tensor.name, raw));
        }
    }
    Ok(())
}

/// 1D conv as a direct 2D conv with `H = 1`: kernel `[k, 1, ic, oc]`, data
/// `[t, 1, ic, 1]`, symmetric zero padding, optional dilation.
#[allow(clippy::too_many_arguments)]
fn conv1d(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    t: i64,
    ic: i64,
    oc: i64,
    k: i64,
    stride: i64,
    pad: i64,
    dil: i64,
    with_bias: bool,
) -> Result<Tensor, RuntimeError> {
    let x4 = nn(b.reshape_4d(x, t, 1, ic, 1))?;
    let kernel = wt(weights, &format!("{prefix}.weight"))?;
    let y = nn(b.conv2d_direct(kernel, x4, stride as i32, 1, pad as i32, 0, dil as i32, 1))?;
    let out_t = (t + 2 * pad - dil * (k - 1) - 1) / stride + 1;
    let y = shaped(
        nn(b.cont(y))?,
        [out_t, 1, oc, 1],
        &format!("{prefix}.weight"),
    )?;
    let y = nn(b.reshape_2d(y, out_t, oc))?;
    if with_bias {
        add_bias(b, weights, y, &format!("{prefix}.bias"), out_t, oc)
    } else {
        Ok(y)
    }
}

/// Transposed 1D conv: ggml computes the padding-0 output `(t - 1) * rate + k`
/// and the symmetric `(k - rate) / 2` crop is a view, giving exactly `t * rate`.
#[allow(clippy::too_many_arguments)]
fn ups_conv(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    t: i64,
    _ic: i64,
    oc: i64,
    k: i64,
    rate: i64,
) -> Result<Tensor, RuntimeError> {
    let kernel = wt(weights, &format!("{prefix}.weight"))?;
    let full = nn(b.conv_transpose_1d(kernel, x, rate as i32))?;
    let full_t = (t - 1) * rate + k;
    let full = shaped(
        nn(b.cont(full))?,
        [full_t, oc, 1, 1],
        &format!("{prefix}.weight"),
    )?;
    let p = (k - rate) / 2;
    let y = nn(b.cont(b.view_2d(full, t * rate, oc, (full_t * 4) as usize, (p * 4) as usize)))?;
    add_bias(b, weights, y, &format!("{prefix}.bias"), t * rate, oc)
}

/// Depthwise 12-tap filter: the channels ride the batch axis of a direct 2D
/// conv with a `[12, 1, 1, 1]` kernel, so every channel gets the same taps.
fn dw_filter(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    name: &str,
    l: i64,
    c: i64,
    stride: i64,
) -> Result<Tensor, RuntimeError> {
    let x4 = nn(b.reshape_4d(x, l, 1, 1, c))?;
    let kernel = wt(weights, name)?;
    let y = nn(b.conv2d_direct(kernel, x4, stride as i32, 1, 0, 0, 1, 1))?;
    let out_l = (l - FILTER_TAPS) / stride + 1;
    let y = shaped(nn(b.cont(y))?, [out_l, 1, 1, c], name)?;
    nn(b.reshape_2d(y, out_l, c))
}

/// Anti-aliased activation: x2 zero-insert, replicate-padded filter, x2 gain,
/// SnakeBeta, replicate-padded stride-2 lowpass.
fn activation1d(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    t: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let up_t = 2 * t;
    let pad_t = up_t + FILTER_TAPS - 1;
    let xu = interleave_zeros(b, x, t, c)?;
    let xp = replicate_pad(b, xu, up_t, c, FILTER_TAPS / 2, FILTER_TAPS / 2 - 1)?;
    let xf = dw_filter(
        b,
        weights,
        xp,
        &format!("{prefix}.upsample.filter"),
        pad_t,
        c,
        1,
    )?;
    let xf = nn(b.scale(xf, 2.0))?;
    let alpha = nn(b.cont(b.reshape_2d(wt(weights, &format!("{prefix}.act.alpha"))?, 1, c)))?;
    let s = nn(b.mul(xf, nn(b.repeat(alpha, xf))?))?;
    let s = nn(b.sin(s))?;
    let s = nn(b.sqr(s))?;
    let beta = nn(b.cont(b.reshape_2d(wt(weights, &format!("{prefix}.act.beta"))?, 1, c)))?;
    let xs = nn(b.add(xf, nn(b.mul(s, nn(b.repeat(beta, s))?))?))?;
    let xp = replicate_pad(b, xs, up_t, c, FILTER_TAPS / 2 - 1, FILTER_TAPS / 2)?;
    dw_filter(
        b,
        weights,
        xp,
        &format!("{prefix}.downsample.lowpass.filter"),
        pad_t,
        c,
        2,
    )
}

/// One AMPBlock1: three `(act, dilated conv, act, conv)` rounds with residuals.
fn amp_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    t: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let mut x = x;
    for j in 0..RES_KERNELS.len() {
        let residual = x;
        x = activation1d(b, weights, x, &format!("{prefix}.acts1.{j}"), t, c)?;
        let k = RES_KERNELS[j];
        let d = DILATIONS[j];
        x = conv1d(
            b,
            weights,
            x,
            &format!("{prefix}.convs1.{j}"),
            t,
            c,
            c,
            k,
            1,
            (k * d - d) / 2,
            d,
            true,
        )?;
        x = activation1d(b, weights, x, &format!("{prefix}.acts2.{j}"), t, c)?;
        x = conv1d(
            b,
            weights,
            x,
            &format!("{prefix}.convs2.{j}"),
            t,
            c,
            c,
            k,
            1,
            k / 2,
            1,
            true,
        )?;
        x = nn(b.add(x, residual))?;
    }
    Ok(x)
}

/// BigVGAN v2 forward: `[t, 128]` mel -> `[t * hop, 2]` waveform.
fn bigvgan(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: &VocoderConfig,
    mel: Tensor,
    t: i64,
) -> Result<Tensor, RuntimeError> {
    let ch = config.channels();
    let p = &config.prefix;
    let mut x = conv1d(
        b,
        weights,
        mel,
        &format!("{p}.conv_pre"),
        t,
        2 * MEL,
        ch[0],
        7,
        1,
        3,
        1,
        true,
    )?;
    let mut ct = t;
    for (stage, (&rate, &k)) in config.rates.iter().zip(&config.kernels).enumerate() {
        x = ups_conv(
            b,
            weights,
            x,
            &format!("{p}.ups.{stage}"),
            ct,
            ch[stage],
            ch[stage + 1],
            k,
            rate,
        )?;
        ct *= rate;
        let mut acc: Option<Tensor> = None;
        for j in 0..RES_KERNELS.len() {
            let r = amp_block(
                b,
                weights,
                x,
                &format!("{p}.resblocks.{}", stage * RES_KERNELS.len() + j),
                ct,
                ch[stage + 1],
            )?;
            acc = Some(match acc {
                None => r,
                Some(a) => nn(b.add(a, r))?,
            });
        }
        x = nn(b.scale(
            acc.expect("three resblocks per stage"),
            1.0 / RES_KERNELS.len() as f32,
        ))?;
    }
    let last = *ch.last().expect("one stage or more");
    x = activation1d(b, weights, x, &format!("{p}.act_post"), ct, last)?;
    x = conv1d(
        b,
        weights,
        x,
        &format!("{p}.conv_post"),
        ct,
        last,
        2,
        7,
        1,
        3,
        1,
        false,
    )?;
    if config.clip {
        x = nn(b.clamp(x, -1.0, 1.0))?;
    }
    shaped(x, [ct, 2, 1, 1], &format!("{p} output"))
}

/// Log-mel of one waveform channel: causal STFT conv, magnitude, mel matmul,
/// `log(max(mel, 1e-5))`. `wav` is `[l, 1]`; the result is `[64, t']`.
fn mel_stft(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    wav: Tensor,
    l: i64,
) -> Result<Tensor, RuntimeError> {
    let x = nn(b.pad_ext(wav, STFT_LEFT_PAD as i32, 0, 0, 0, 0, 0, 0, 0))?;
    let x4 = nn(b.reshape_4d(x, l + STFT_LEFT_PAD, 1, 1, 1))?;
    let basis = wt(weights, STFT_BASIS)?;
    let y = nn(b.conv2d_direct(basis, x4, HOP as i32, 1, 0, 0, 1, 1))?;
    let tp = (l + STFT_LEFT_PAD - N_FFT) / HOP + 1;
    let y = shaped(nn(b.cont(y))?, [tp, 1, 2 * N_BINS, 1], STFT_BASIS)?;
    let y = nn(b.reshape_2d(y, tp, 2 * N_BINS))?;
    let nb1 = (tp * 4) as usize;
    let real = nn(b.cont(b.view_2d(y, tp, N_BINS, nb1, 0)))?;
    let imag = nn(b.cont(b.view_2d(y, tp, N_BINS, nb1, (N_BINS * tp * 4) as usize)))?;
    // main computes sqrt(real^2 + imag^2 + 1e-9); the 1e-9 only matters at
    // exact zeros and sits far below the 1e-5 mel clamp, so it is dropped.
    let mag = nn(b.sqrt(nn(b.add(nn(b.sqr(real))?, nn(b.sqr(imag))?))?))?;
    let mag_t = nn(b.cont(b.permute(mag, 1, 0, 2, 3)))?; // [bins, t']
    let mel = nn(b.mul_mat(wt(weights, MEL_BASIS)?, mag_t))?;
    let mel = shaped(nn(b.cont(mel))?, [MEL, tp, 1, 1], MEL_BASIS)?;
    let mel = nn(b.clamp(mel, 1e-5, 1e30))?;
    nn(b.log(mel))
}

/// x3 Hann-windowed sinc resampler: replicate pad 7, zero-insert, full conv
/// with the 43-tap kernel, x3 gain, slice `[42:-40]`. `wav` is `[l, 1]`.
fn resample3(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    wav: Tensor,
    l: i64,
) -> Result<Tensor, RuntimeError> {
    let xp = replicate_pad(b, wav, l, 1, RESAMPLE_WIDTH, RESAMPLE_WIDTH)?;
    let lp = l + 2 * RESAMPLE_WIDTH;
    let x3 = nn(b.reshape_3d(xp, 1, lp, 1))?;
    let z = zeros(b, 1, lp, 1)?;
    let y = nn(b.cont(nn(b.concat(x3, z, 0))?))?;
    let y = nn(b.cont(nn(b.concat(y, z, 0))?))?; // [3, lp, 1]
    let y = nn(b.reshape_2d(y, RESAMPLE_FACTOR * lp, 1))?;
    let zi = RESAMPLE_FACTOR * (lp - 1) + 1;
    let y = nn(b.cont(b.view_2d(y, zi, 1, (RESAMPLE_FACTOR * lp * 4) as usize, 0)))?;
    let pad = RESAMPLE_TAPS - 1;
    let yp = nn(b.pad_ext(y, pad as i32, pad as i32, 0, 0, 0, 0, 0, 0))?;
    let y4 = nn(b.reshape_4d(yp, zi + 2 * pad, 1, 1, 1))?;
    let kernel = wt(weights, RESAMPLER_KERNEL)?;
    let out = nn(b.conv2d_direct(kernel, y4, 1, 1, 0, 0, 1, 1))?;
    let out_l = zi + pad;
    let out = shaped(nn(b.cont(out))?, [out_l, 1, 1, 1], RESAMPLER_KERNEL)?;
    let out = nn(b.reshape_2d(out, out_l, 1))?;
    let out = nn(b.scale(out, RESAMPLE_FACTOR as f32))?;
    let out = nn(b.cont(b.view_2d(
        out,
        RESAMPLE_FACTOR * l,
        1,
        (out_l * 4) as usize,
        ((2 * RESAMPLE_WIDTH * RESAMPLE_FACTOR) * 4) as usize,
    )))?;
    shaped(out, [RESAMPLE_FACTOR * l, 1, 1, 1], "ltx resampler output")
}

pub struct LtxVocoder {
    base: VocoderConfig,
    bwe: VocoderConfig,
    loaded: Vec<(String, Vec<f32>)>,
}

impl LtxVocoder {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let base = VocoderConfig::infer(&weights, "vocoder.vocoder")?;
        let bwe = VocoderConfig::infer(&weights, "vocoder.bwe_generator")?;
        let mut loaded = Vec::new();
        load_vocoder(&weights, &base, &mut loaded)?;
        load_vocoder(&weights, &bwe, &mut loaded)?;
        loaded.push((
            MEL_BASIS.to_string(),
            weights.f32_widened(MEL_BASIS, &[N_BINS, MEL])?,
        ));
        loaded.push((
            STFT_BASIS.to_string(),
            weights.f32_widened(STFT_BASIS, &[N_FFT, 1, 2 * N_BINS])?,
        ));
        loaded.push((RESAMPLER_KERNEL.to_string(), resampler_kernel()));
        Ok(Self { base, bwe, loaded })
    }

    pub fn base(&self) -> &VocoderConfig {
        &self.base
    }

    pub fn bwe(&self) -> &VocoderConfig {
        &self.bwe
    }

    /// Samples per channel for a `t`-frame mel: `160 * 3` per mel frame.
    pub fn output_len(&self, t: i64) -> i64 {
        t * self.base.hop() * RESAMPLE_FACTOR
    }

    /// Decode a `[mel=64, time=t, ch=2]` mel (ggml order) into a 48 kHz
    /// waveform `[samples=t * 480, ch=2]` in ggml `(time, channel)` order.
    pub fn decode(&self, ctx: &GgmlContext, mel: &[f32], t: i64) -> Result<Vec<f32>, WeightError> {
        if t < 1 {
            return Err(WeightError::Shape {
                name: "ltx vocoder decode dims".into(),
                expected: vec![1],
                got: vec![t],
            });
        }
        if mel.len() as i64 != MEL * t * 2 {
            return Err(WeightError::Shape {
                name: "ltx vocoder mel".into(),
                expected: vec![MEL, t, 2],
                got: vec![mel.len() as i64],
            });
        }
        let mut b = ctx.builder(VOCODER_GRAPH_MEM)?;
        let mut wmap = HashMap::new();
        for config in [&self.base, &self.bwe] {
            for tensor in vocoder_weight_tensors(config) {
                let ne = tensor.graph;
                let name = tensor.name;
                let created = nn(b.f32_4d(&name, ne[0], ne[1], ne[2], ne[3], true))?;
                wmap.insert(name, created);
            }
        }
        wmap.insert(
            MEL_BASIS.to_string(),
            nn(b.f32_2d(MEL_BASIS, N_BINS, MEL, true))?,
        );
        wmap.insert(
            STFT_BASIS.to_string(),
            nn(b.f32_4d(STFT_BASIS, N_FFT, 1, 1, 2 * N_BINS, true))?,
        );
        wmap.insert(
            RESAMPLER_KERNEL.to_string(),
            nn(b.f32_4d(RESAMPLER_KERNEL, RESAMPLE_TAPS, 1, 1, 1, true))?,
        );

        // Stereo mel `[64, t, 2]` -> `[t, 128]` with `k = channel * 64 + bin`.
        let mel_in = nn(b.f32_3d("ltx_vocoder_mel", MEL, t, 2, true))?;
        let mel4 = nn(b.reshape_4d(mel_in, MEL, t, 1, 2))?;
        let mel128 = nn(b.cont(b.permute(mel4, 1, 0, 2, 3)))?; // [t, 64, 1, 2]
        let mel128 = nn(b.reshape_2d(mel128, t, 2 * MEL))?;

        let wav16 = bigvgan(&b, &wmap, &self.base, mel128, t)?; // [160t, 2]
        let t16 = t * self.base.hop();
        let nb1 = (t16 * 4) as usize;
        let ch0 = nn(b.cont(b.view_2d(wav16, t16, 1, nb1, 0)))?;
        let ch1 = nn(b.cont(b.view_2d(wav16, t16, 1, nb1, (t16 * 4) as usize)))?;

        // The base hop 160 is a multiple of the STFT hop 80, so the waveform
        // never needs main's pad-to-multiple branch.
        let m0 = mel_stft(&b, &wmap, ch0, t16)?;
        let m1 = mel_stft(&b, &wmap, ch1, t16)?;
        let tm = (t16 + STFT_LEFT_PAD - N_FFT) / HOP + 1;
        let m0t = nn(b.cont(b.permute(m0, 1, 0, 2, 3)))?; // [tm, 64]
        let m1t = nn(b.cont(b.permute(m1, 1, 0, 2, 3)))?;
        let bwe_in = nn(b.cont(nn(b.concat(m0t, m1t, 1))?))?;
        let bwe_in = shaped(bwe_in, [tm, 2 * MEL, 1, 1], "ltx bwe mel")?;
        let residual = bigvgan(&b, &wmap, &self.bwe, bwe_in, tm)?;

        let s0 = resample3(&b, &wmap, ch0, t16)?;
        let s1 = resample3(&b, &wmap, ch1, t16)?;
        let skip = nn(b.cont(nn(b.concat(s0, s1, 1))?))?;
        let out = nn(b.add(skip, residual))?;
        let out = nn(b.clamp(out, -1.0, 1.0))?;
        let out = shaped(out, [self.output_len(t), 2, 1, 1], "ltx vocoder output")?;

        let mut graph = b.finish(out)?;
        for (name, data) in &self.loaded {
            graph.set_f32(name, data)?;
        }
        graph.set_f32("ltx_vocoder_mel", mel)?;
        let out = graph.compute()?;
        if let Some(detail) = crate::common::ops::non_finite("ltx vocoder", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// smoke writers (graph-vs-host tests)
// ---------------------------------------------------------------------------

fn lcg(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed as f32 / u32::MAX as f32) - 0.5
}

/// Deterministic smoke weights for `AudioDecoderConfig::smoke()`.
pub fn smoke_store(config: &AudioDecoderConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 43u32;
    let mut out = Vec::new();
    for conv in decoder_convs(config) {
        let k = if conv.one_by_one { 1 } else { 3 };
        let count = (k * k * conv.ic * conv.oc) as usize;
        out.push((
            format!("{}.conv.weight", conv.name),
            (0..count).map(|_| lcg(&mut seed) * 0.1).collect(),
        ));
        out.push((
            format!("{}.conv.bias", conv.name),
            (0..conv.oc).map(|_| lcg(&mut seed) * 0.1).collect(),
        ));
    }
    let stats = (FREQ * config.latent) as usize;
    out.push((
        AUDIO_MEAN.to_string(),
        (0..stats).map(|_| lcg(&mut seed) * 0.2).collect(),
    ));
    out.push((
        AUDIO_STD.to_string(),
        (0..stats).map(|_| 0.75 + lcg(&mut seed) * 0.25).collect(),
    ));
    out
}

fn write_store(
    path: &std::path::Path,
    store: &[(String, Vec<f32>)],
    shapes: &[Vec<u64>],
) -> Result<(), WeightError> {
    let bytes: Vec<Vec<u8>> = store
        .iter()
        .map(|(_, values)| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        })
        .collect();
    let tensors: Vec<crate::weights::SafeTensor<'_>> = store
        .iter()
        .zip(shapes)
        .zip(&bytes)
        .map(|(((name, _), shape), data)| crate::weights::SafeTensor {
            name,
            dtype: "F32",
            shape,
            data,
        })
        .collect();
    crate::weights::write_safetensors(path, &tensors)
}

/// Write the smoke audio decoder as a safetensors file (PyTorch shapes).
pub fn write_smoke(dir: &std::path::Path) -> Result<PathBuf, WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = AudioDecoderConfig::smoke();
    let store = smoke_store(&config);
    let convs = decoder_convs(&config);
    let shapes: Vec<Vec<u64>> = store
        .iter()
        .map(|(name, values)| {
            if let Some(conv) = convs
                .iter()
                .find(|conv| format!("{}.conv.weight", conv.name) == *name)
            {
                let k = if conv.one_by_one { 1 } else { 3 };
                vec![conv.oc as u64, conv.ic as u64, k, k]
            } else {
                vec![values.len() as u64]
            }
        })
        .collect();
    let path = dir.join("ltx_audio_vae.safetensors");
    write_store(&path, &store, &shapes)?;
    Ok(path)
}

/// Deterministic smoke weights for the two smoke vocoder configs.
pub fn vocoder_smoke_store(base: &VocoderConfig, bwe: &VocoderConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 47u32;
    let mut out = Vec::new();
    for config in [base, bwe] {
        for tensor in vocoder_weight_tensors(config) {
            let count = tensor.file.iter().product::<i64>() as usize;
            let values: Vec<f32> = if tensor.name.ends_with(".act.alpha") {
                (0..count).map(|_| 0.6 + lcg(&mut seed) * 0.4).collect()
            } else if tensor.name.ends_with(".act.beta") {
                (0..count).map(|_| lcg(&mut seed) * 0.4).collect()
            } else if tensor.name.ends_with(".filter") {
                let tap = 1.0 / FILTER_TAPS as f32;
                (0..count).map(|i| tap * (1.0 + 0.1 * (i as f32))).collect()
            } else if tensor.name.ends_with(".bias") {
                (0..count).map(|_| lcg(&mut seed) * 0.05).collect()
            } else {
                let scale = 0.5 / (count as f32).sqrt();
                (0..count).map(|_| lcg(&mut seed) * scale).collect()
            };
            out.push((tensor.name, values));
        }
    }
    out.push((
        MEL_BASIS.to_string(),
        (0..(N_BINS * MEL) as usize)
            .map(|_| 0.02 + lcg(&mut seed) * 0.02)
            .collect(),
    ));
    out.push((
        STFT_BASIS.to_string(),
        (0..(N_FFT * 2 * N_BINS) as usize)
            .map(|_| lcg(&mut seed) * 0.02)
            .collect(),
    ));
    out
}

/// PyTorch shape of one smoke-store entry.
fn vocoder_smoke_shape(
    name: &str,
    values: &[f32],
    base: &VocoderConfig,
    bwe: &VocoderConfig,
) -> Vec<u64> {
    if name == MEL_BASIS {
        return vec![MEL as u64, N_BINS as u64];
    }
    if name == STFT_BASIS {
        return vec![2 * N_BINS as u64, 1, N_FFT as u64];
    }
    if name.ends_with(".upsample.filter") || name.ends_with(".lowpass.filter") {
        return vec![1, 1, FILTER_TAPS as u64];
    }
    let config = if name.starts_with("vocoder.vocoder.") {
        base
    } else {
        bwe
    };
    let ch = config.channels();
    let p = &config.prefix;
    if name == format!("{p}.conv_pre.weight") {
        return vec![ch[0] as u64, 2 * MEL as u64, 7];
    }
    if name == format!("{p}.conv_post.weight") {
        return vec![2, *ch.last().expect("stages") as u64, 7];
    }
    if name.contains(".ups.") && name.ends_with(".weight") {
        let stage: usize = name
            .trim_end_matches(".weight")
            .rsplit('.')
            .next()
            .and_then(|index| index.parse().ok())
            .expect("ups stage index");
        return vec![
            ch[stage] as u64,
            ch[stage + 1] as u64,
            config.kernels[stage] as u64,
        ];
    }
    if (name.contains(".convs1.") || name.contains(".convs2.")) && name.ends_with(".weight") {
        let rb: usize = name
            .split(".resblocks.")
            .nth(1)
            .and_then(|rest| rest.split('.').next())
            .and_then(|index| index.parse().ok())
            .expect("resblock index");
        let j: usize = name
            .trim_end_matches(".weight")
            .rsplit('.')
            .next()
            .and_then(|index| index.parse().ok())
            .expect("dilation index");
        let c = ch[rb / RES_KERNELS.len() + 1];
        return vec![c as u64, c as u64, RES_KERNELS[j] as u64];
    }
    vec![values.len() as u64]
}

/// Write the smoke vocoders as a safetensors file (PyTorch shapes).
pub fn write_vocoder_smoke(dir: &std::path::Path) -> Result<PathBuf, WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let base = VocoderConfig::smoke("vocoder.vocoder")?;
    let bwe = VocoderConfig::smoke("vocoder.bwe_generator")?;
    let store = vocoder_smoke_store(&base, &bwe);
    let shapes: Vec<Vec<u64>> = store
        .iter()
        .map(|(name, values)| vocoder_smoke_shape(name, values, &base, &bwe))
        .collect();
    let path = dir.join("ltx_vocoder.safetensors");
    write_store(&path, &store, &shapes)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;

    fn at<'a>(store: &'a [(String, Vec<f32>)], name: &str) -> &'a [f32] {
        store
            .iter()
            .find(|(key, _)| key == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
            .as_slice()
    }

    fn silu_host(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn pixel_norm_host(x: &mut [f32], w: usize, h: usize, c: usize) {
        for hh in 0..h {
            for ww in 0..w {
                let base = ww + w * hh;
                let mut sum = 0f32;
                for cc in 0..c {
                    let v = x[base + w * h * cc];
                    sum += v * v;
                }
                let scale = 1.0 / (sum / c as f32 + PIXEL_NORM_EPS).sqrt();
                for cc in 0..c {
                    x[base + w * h * cc] *= scale;
                }
            }
        }
    }

    /// Causal 3x3 conv: two zero rows on top of the time axis, one zero
    /// column per side on the frequency axis. `x` is `[w, h, ic]` ggml order.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_causal_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        w: usize,
        h: usize,
        ic: usize,
        oc: usize,
    ) -> Vec<f32> {
        let weight = at(store, &format!("{prefix}.conv.weight"));
        let bias = at(store, &format!("{prefix}.conv.bias"));
        conv2d_host(weight, bias, x, w, h, ic, oc, 3, 2, 1)
    }

    /// `pad_top` zero rows on the time axis, `pad_w` zero columns per side.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_host(
        weight: &[f32],
        bias: &[f32],
        x: &[f32],
        w: usize,
        h: usize,
        ic: usize,
        oc: usize,
        k: usize,
        pad_top: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; w * h * oc];
        for o in 0..oc {
            for hh in 0..h {
                for ww in 0..w {
                    let mut acc = bias[o];
                    for kh in 0..k {
                        for kw in 0..k {
                            let sh = hh as i64 + kh as i64 - pad_top as i64;
                            let sw = ww as i64 + kw as i64 - pad_w as i64;
                            if sh < 0 || sh >= h as i64 || sw < 0 || sw >= w as i64 {
                                continue;
                            }
                            let (sh, sw) = (sh as usize, sw as usize);
                            let tap = kw + k * kh;
                            for i in 0..ic {
                                acc +=
                                    weight[tap + k * k * (i + ic * o)] * x[sw + w * (sh + h * i)];
                            }
                        }
                    }
                    out[ww + w * (hh + h * o)] = acc;
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn res_block_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        w: usize,
        h: usize,
        ic: usize,
        oc: usize,
    ) -> Vec<f32> {
        let mut y = x.to_vec();
        pixel_norm_host(&mut y, w, h, ic);
        for v in y.iter_mut() {
            *v = silu_host(*v);
        }
        let mut y = conv2d_causal_host(store, &format!("{prefix}.conv1"), &y, w, h, ic, oc);
        pixel_norm_host(&mut y, w, h, oc);
        for v in y.iter_mut() {
            *v = silu_host(*v);
        }
        let y = conv2d_causal_host(store, &format!("{prefix}.conv2"), &y, w, h, oc, oc);
        let residual = if ic != oc {
            let weight = at(store, &format!("{prefix}.nin_shortcut.conv.weight"));
            let bias = at(store, &format!("{prefix}.nin_shortcut.conv.bias"));
            conv2d_host(weight, bias, x, w, h, ic, oc, 1, 0, 0)
        } else {
            x.to_vec()
        };
        y.iter().zip(&residual).map(|(a, b)| a + b).collect()
    }

    /// x2 nearest on both axes, causal conv, drop the first time row.
    fn upsample_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        w: usize,
        h: usize,
        c: usize,
    ) -> Vec<f32> {
        let (ow, oh) = (2 * w, 2 * h);
        let mut up = vec![0f32; ow * oh * c];
        for cc in 0..c {
            for hh in 0..oh {
                for ww in 0..ow {
                    up[ww + ow * (hh + oh * cc)] = x[(ww / 2) + w * ((hh / 2) + h * cc)];
                }
            }
        }
        let y = conv2d_causal_host(store, prefix, &up, ow, oh, c, c);
        // drop the first time row
        let mut out = vec![0f32; ow * (oh - 1) * c];
        for cc in 0..c {
            out[cc * ow * (oh - 1)..(cc + 1) * ow * (oh - 1)]
                .copy_from_slice(&y[cc * ow * oh + ow..(cc + 1) * ow * oh]);
        }
        out
    }

    fn host_audio_decode(
        store: &[(String, Vec<f32>)],
        config: &AudioDecoderConfig,
        latent: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let (w, c0) = (FREQ as usize, config.latent as usize);
        let mean = at(store, AUDIO_MEAN);
        let std = at(store, AUDIO_STD);
        // file order is c * 16 + f; ggml latent is f + 16 * (tt + t * c)
        let mut x = vec![0f32; w * t * c0];
        for cc in 0..c0 {
            for tt in 0..t {
                for f in 0..w {
                    let idx = f + w * (tt + t * cc);
                    let stat = cc * w + f;
                    x[idx] = latent[idx] * std[stat] + mean[stat];
                }
            }
        }
        let (mut cw, mut ct) = (w, t);
        x = conv2d_causal_host(
            store,
            "audio_vae.decoder.conv_in",
            &x,
            cw,
            ct,
            c0,
            config.base as usize,
        );
        for block in ["block_1", "block_2"] {
            x = res_block_host(
                store,
                &format!("audio_vae.decoder.mid.{block}"),
                &x,
                cw,
                ct,
                config.base as usize,
                config.base as usize,
            );
        }
        for (index, ic, oc, upsample) in config.up_blocks() {
            let (ic, oc) = (ic as usize, oc as usize);
            for block in 0..3 {
                let (bic, boc) = if block == 0 { (ic, oc) } else { (oc, oc) };
                x = res_block_host(
                    store,
                    &format!("audio_vae.decoder.up.{index}.block.{block}"),
                    &x,
                    cw,
                    ct,
                    bic,
                    boc,
                );
            }
            if upsample {
                x = upsample_host(
                    store,
                    &format!("audio_vae.decoder.up.{index}.upsample.conv"),
                    &x,
                    cw,
                    ct,
                    oc,
                );
                cw *= 2;
                ct = 2 * ct - 1;
            }
        }
        pixel_norm_host(&mut x, cw, ct, config.tail as usize);
        for v in x.iter_mut() {
            *v = silu_host(*v);
        }
        conv2d_causal_host(
            store,
            "audio_vae.decoder.conv_out",
            &x,
            cw,
            ct,
            config.tail as usize,
            config.mel as usize,
        )
    }

    #[test]
    fn audio_decoder_graph_matches_the_host_reference() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-audio-vae-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_smoke(&dir).unwrap();
        let decoder = LtxAudioDecoder::open(&[path]).unwrap();
        assert_eq!(decoder.config(), AudioDecoderConfig::smoke());

        let t = 2i64;
        let config = decoder.config();
        let mut seed = 9u32;
        let latent: Vec<f32> = (0..(FREQ * t * config.latent) as usize)
            .map(|_| lcg(&mut seed))
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let out = decoder.decode(&ctx, &latent, t).unwrap();
        let mel_t = decoder.output_len(t);
        assert_eq!(mel_t, 5);
        assert_eq!(out.len(), (MEL * mel_t * 2) as usize);

        let store = smoke_store(&AudioDecoderConfig::smoke());
        let host = host_audio_decode(&store, &config, &latent, t as usize);
        assert_eq!(host.len(), out.len());
        let mut max_diff = 0f32;
        for (g, e) in out.iter().zip(&host) {
            max_diff = max_diff.max((g - e).abs());
        }
        assert!(max_diff < 1e-3, "graph vs host max diff {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------------
    // vocoder host reference
    // -----------------------------------------------------------------------

    /// 1D conv with symmetric zero padding. `weight` is `[k, ic, oc]` ggml.
    #[allow(clippy::too_many_arguments)]
    fn conv1d_host(
        weight: &[f32],
        bias: Option<&[f32]>,
        x: &[f32],
        t: usize,
        ic: usize,
        oc: usize,
        k: usize,
        stride: usize,
        pad: usize,
        dil: usize,
    ) -> Vec<f32> {
        let out_t = (t + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        let mut out = vec![0f32; out_t * oc];
        for o in 0..oc {
            for tt in 0..out_t {
                let mut acc = bias.map_or(0.0, |b| b[o]);
                for ki in 0..k {
                    let pos = tt as i64 * stride as i64 + ki as i64 * dil as i64 - pad as i64;
                    if pos < 0 || pos >= t as i64 {
                        continue;
                    }
                    for i in 0..ic {
                        acc += weight[ki + k * (i + ic * o)] * x[pos as usize + t * i];
                    }
                }
                out[tt + out_t * o] = acc;
            }
        }
        out
    }

    /// Transposed 1D conv with the symmetric `(k - rate) / 2` crop.
    /// `weight` is `[k, oc, ic]` ggml.
    fn ups_host(
        weight: &[f32],
        bias: &[f32],
        x: &[f32],
        t: usize,
        ic: usize,
        oc: usize,
        k: usize,
        rate: usize,
    ) -> Vec<f32> {
        let full_t = (t - 1) * rate + k;
        let mut full = vec![0f32; full_t * oc];
        for i in 0..ic {
            for tt in 0..t {
                let v = x[tt + t * i];
                for ki in 0..k {
                    for o in 0..oc {
                        full[tt * rate + ki + full_t * o] += v * weight[ki + k * (o + oc * i)];
                    }
                }
            }
        }
        let p = (k - rate) / 2;
        let out_t = t * rate;
        let mut out = vec![0f32; out_t * oc];
        for o in 0..oc {
            for tt in 0..out_t {
                out[tt + out_t * o] = full[tt + p + full_t * o] + bias[o];
            }
        }
        out
    }

    /// Depthwise 12-tap filter over every channel.
    fn dw_host(filt: &[f32], x: &[f32], l: usize, c: usize, stride: usize) -> Vec<f32> {
        let out_l = (l - FILTER_TAPS as usize) / stride + 1;
        let mut out = vec![0f32; out_l * c];
        for cc in 0..c {
            for tt in 0..out_l {
                let mut acc = 0f32;
                for ki in 0..FILTER_TAPS as usize {
                    acc += filt[ki] * x[tt * stride + ki + l * cc];
                }
                out[tt + out_l * cc] = acc;
            }
        }
        out
    }

    fn replicate_pad_host(x: &[f32], l: usize, c: usize, left: usize, right: usize) -> Vec<f32> {
        let out_l = l + left + right;
        let mut out = vec![0f32; out_l * c];
        for cc in 0..c {
            for tt in 0..out_l {
                let src = (tt as i64 - left as i64).clamp(0, l as i64 - 1) as usize;
                out[tt + out_l * cc] = x[src + l * cc];
            }
        }
        out
    }

    fn act1d_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        t: usize,
        c: usize,
    ) -> Vec<f32> {
        // zero-insert x2
        let mut up = vec![0f32; 2 * t * c];
        for cc in 0..c {
            for tt in 0..t {
                up[2 * tt + 2 * t * cc] = x[tt + t * cc];
            }
        }
        let taps = FILTER_TAPS as usize;
        let padded = replicate_pad_host(&up, 2 * t, c, taps / 2, taps / 2 - 1);
        let filt = at(store, &format!("{prefix}.upsample.filter"));
        let mut xf = dw_host(filt, &padded, 2 * t + taps - 1, c, 1);
        for v in xf.iter_mut() {
            *v *= 2.0;
        }
        // SnakeBeta with the load-time folds applied here too
        let alpha = at(store, &format!("{prefix}.act.alpha"));
        let beta = at(store, &format!("{prefix}.act.beta"));
        let mut xs = vec![0f32; 2 * t * c];
        for cc in 0..c {
            let a = alpha[cc].exp();
            let binv = 1.0 / (beta[cc].exp() + 1e-9);
            for tt in 0..2 * t {
                let v = xf[tt + 2 * t * cc];
                xs[tt + 2 * t * cc] = v + binv * (a * v).sin().powi(2);
            }
        }
        let padded = replicate_pad_host(&xs, 2 * t, c, taps / 2 - 1, taps / 2);
        let filt = at(store, &format!("{prefix}.downsample.lowpass.filter"));
        dw_host(filt, &padded, 2 * t + taps - 1, c, 2)
    }

    fn amp_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        t: usize,
        c: usize,
    ) -> Vec<f32> {
        let mut x = x.to_vec();
        for j in 0..RES_KERNELS.len() {
            let residual = x.clone();
            let k = RES_KERNELS[j] as usize;
            let d = DILATIONS[j] as usize;
            let y = act1d_host(store, &format!("{prefix}.acts1.{j}"), &x, t, c);
            let w1 = at(store, &format!("{prefix}.convs1.{j}.weight"));
            let b1 = at(store, &format!("{prefix}.convs1.{j}.bias"));
            let y = conv1d_host(w1, Some(b1), &y, t, c, c, k, 1, (k * d - d) / 2, d);
            let y = act1d_host(store, &format!("{prefix}.acts2.{j}"), &y, t, c);
            let w2 = at(store, &format!("{prefix}.convs2.{j}.weight"));
            let b2 = at(store, &format!("{prefix}.convs2.{j}.bias"));
            let y = conv1d_host(w2, Some(b2), &y, t, c, c, k, 1, k / 2, 1);
            x = y.iter().zip(&residual).map(|(a, b)| a + b).collect();
        }
        x
    }

    fn bigvgan_host(
        store: &[(String, Vec<f32>)],
        config: &VocoderConfig,
        mel: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let ch: Vec<usize> = config.channels().iter().map(|c| *c as usize).collect();
        let p = &config.prefix;
        let w = at(store, &format!("{p}.conv_pre.weight"));
        let bias = at(store, &format!("{p}.conv_pre.bias"));
        let mut x = conv1d_host(w, Some(bias), mel, t, 2 * MEL as usize, ch[0], 7, 1, 3, 1);
        let mut ct = t;
        for (stage, (&rate, &k)) in config.rates.iter().zip(&config.kernels).enumerate() {
            let (rate, k) = (rate as usize, k as usize);
            let w = at(store, &format!("{p}.ups.{stage}.weight"));
            let bias = at(store, &format!("{p}.ups.{stage}.bias"));
            x = ups_host(w, bias, &x, ct, ch[stage], ch[stage + 1], k, rate);
            ct *= rate;
            let mut acc: Option<Vec<f32>> = None;
            for j in 0..RES_KERNELS.len() {
                let r = amp_host(
                    store,
                    &format!("{p}.resblocks.{}", stage * RES_KERNELS.len() + j),
                    &x,
                    ct,
                    ch[stage + 1],
                );
                acc = Some(match acc {
                    None => r,
                    Some(a) => a.iter().zip(&r).map(|(u, v)| u + v).collect(),
                });
            }
            let scale = 1.0 / RES_KERNELS.len() as f32;
            x = acc.unwrap().iter().map(|v| v * scale).collect();
        }
        let last = *ch.last().unwrap();
        x = act1d_host(store, &format!("{p}.act_post"), &x, ct, last);
        let w = at(store, &format!("{p}.conv_post.weight"));
        let mut out = conv1d_host(w, None, &x, ct, last, 2, 7, 1, 3, 1);
        if config.clip {
            for v in out.iter_mut() {
                *v = v.clamp(-1.0, 1.0);
            }
        }
        out
    }

    /// `[l]` waveform -> `[64, t']` log-mel (`mel + 64 * t'` order).
    fn mel_stft_host(store: &[(String, Vec<f32>)], wav: &[f32]) -> Vec<f32> {
        let l = wav.len();
        let basis = at(store, STFT_BASIS); // [512, 1, 514] ggml: tap k + 512 * o
        let tp = (l + STFT_LEFT_PAD as usize - N_FFT as usize) / HOP as usize + 1;
        let bins = N_BINS as usize;
        let n_fft = N_FFT as usize;
        let mut mag_t = vec![0f32; bins * tp]; // [bins, t']
        for tt in 0..tp {
            for o in 0..2 * bins {
                let mut acc = 0f32;
                for ki in 0..n_fft {
                    let pos = tt * HOP as usize + ki;
                    let v = if pos < STFT_LEFT_PAD as usize {
                        0.0
                    } else {
                        wav[pos - STFT_LEFT_PAD as usize]
                    };
                    acc += basis[ki + n_fft * o] * v;
                }
                if o < bins {
                    mag_t[o + bins * tt] = acc * acc;
                } else {
                    mag_t[o - bins + bins * tt] += acc * acc;
                }
            }
        }
        for v in mag_t.iter_mut() {
            *v = v.sqrt();
        }
        let mel_basis = at(store, MEL_BASIS); // [257, 64] ggml: f + 257 * m
        let mut mel = vec![0f32; MEL as usize * tp];
        for tt in 0..tp {
            for m in 0..MEL as usize {
                let mut acc = 0f32;
                for f in 0..bins {
                    acc += mel_basis[f + bins * m] * mag_t[f + bins * tt];
                }
                mel[m + MEL as usize * tt] = acc.max(1e-5).ln();
            }
        }
        mel
    }

    /// `[l]` -> `[3l]` x3 Hann-sinc resample.
    fn resample3_host(wav: &[f32]) -> Vec<f32> {
        let l = wav.len();
        let width = RESAMPLE_WIDTH as usize;
        let ratio = RESAMPLE_FACTOR as usize;
        let lp = l + 2 * width;
        let padded = replicate_pad_host(wav, l, 1, width, width);
        // zero-insert
        let zi = ratio * (lp - 1) + 1;
        let mut up = vec![0f32; zi];
        for tt in 0..lp {
            up[ratio * tt] = padded[tt];
        }
        let kernel = resampler_kernel();
        let taps = RESAMPLE_TAPS as usize;
        let full_len = zi + 2 * (taps - 1);
        let mut xp = vec![0f32; full_len];
        xp[taps - 1..taps - 1 + zi].copy_from_slice(&up);
        let out_l = zi + taps - 1;
        let mut full = vec![0f32; out_l];
        for tt in 0..out_l {
            let mut acc = 0f32;
            for ki in 0..taps {
                acc += kernel[ki] * xp[tt + ki];
            }
            full[tt] = acc * ratio as f32;
        }
        full[2 * width * ratio..2 * width * ratio + ratio * l].to_vec()
    }

    fn vocoder_decode_host(
        store: &[(String, Vec<f32>)],
        base: &VocoderConfig,
        bwe: &VocoderConfig,
        mel: &[f32],
        t: usize,
    ) -> Vec<f32> {
        // [64, t, 2] -> [t, 128] with k = channel * 64 + bin
        let mel_bins = MEL as usize;
        let mut mel128 = vec![0f32; t * 2 * mel_bins];
        for cc in 0..2 {
            for tt in 0..t {
                for m in 0..mel_bins {
                    mel128[tt + t * (cc * mel_bins + m)] = mel[m + mel_bins * (tt + t * cc)];
                }
            }
        }
        let wav16 = bigvgan_host(store, base, &mel128, t); // [160t, 2]
        let t16 = t * base.hop() as usize;
        let ch0: Vec<f32> = wav16[..t16].to_vec();
        let ch1: Vec<f32> = wav16[t16..].to_vec();
        let m0 = mel_stft_host(store, &ch0); // [64, tm]
        let m1 = mel_stft_host(store, &ch1);
        let tm = (t16 + STFT_LEFT_PAD as usize - N_FFT as usize) / HOP as usize + 1;
        let mut bwe_in = vec![0f32; tm * 2 * mel_bins];
        for cc in 0..2 {
            let src = if cc == 0 { &m0 } else { &m1 };
            for tt in 0..tm {
                for m in 0..mel_bins {
                    bwe_in[tt + tm * (cc * mel_bins + m)] = src[m + mel_bins * tt];
                }
            }
        }
        let residual = bigvgan_host(store, bwe, &bwe_in, tm);
        let s0 = resample3_host(&ch0);
        let s1 = resample3_host(&ch1);
        let n = t * 480;
        assert_eq!(s0.len(), n);
        assert_eq!(residual.len(), 2 * n);
        let mut out = vec![0f32; 2 * n];
        for i in 0..n {
            out[i] = (s0[i] + residual[i]).clamp(-1.0, 1.0);
            out[n + i] = (s1[i] + residual[n + i]).clamp(-1.0, 1.0);
        }
        out
    }

    #[test]
    fn vocoder_graph_matches_the_host_reference() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-vocoder-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_vocoder_smoke(&dir).unwrap();
        let vocoder = LtxVocoder::open(&[path]).unwrap();
        let base = VocoderConfig::smoke("vocoder.vocoder").unwrap();
        let bwe = VocoderConfig::smoke("vocoder.bwe_generator").unwrap();
        assert_eq!(vocoder.base(), &base);
        assert_eq!(vocoder.bwe(), &bwe);

        let t = 2i64;
        let mut seed = 13u32;
        let mel: Vec<f32> = (0..(MEL * t * 2) as usize)
            .map(|_| lcg(&mut seed))
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let out = vocoder.decode(&ctx, &mel, t).unwrap();
        assert_eq!(out.len(), (t * 480 * 2) as usize);

        let store = vocoder_smoke_store(&base, &bwe);
        let host = vocoder_decode_host(&store, &base, &bwe, &mel, t as usize);
        assert_eq!(host.len(), out.len());
        let mut max_diff = 0f32;
        let mut worst = 0;
        for (i, (g, e)) in out.iter().zip(&host).enumerate() {
            let diff = (g - e).abs();
            if diff > max_diff {
                max_diff = diff;
                worst = i;
            }
        }
        assert!(
            max_diff < 1e-3,
            "graph vs host max diff {max_diff} at {worst}: graph {} host {}",
            out[worst],
            host[worst]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_fails_loud_on_missing_weights() {
        let dir =
            std::env::temp_dir().join(format!("danmo-ltx-audio-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.safetensors");
        crate::weights::write_safetensors(
            &path,
            &[crate::weights::SafeTensor {
                name: "dummy",
                dtype: "F32",
                shape: &[1],
                data: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let err = match LtxAudioDecoder::open(std::slice::from_ref(&path)) {
            Err(err) => err,
            Ok(_) => panic!("decoder accepted empty weights"),
        };
        assert!(err.to_string().contains("mean-of-means"), "{err}");
        let err = match LtxVocoder::open(std::slice::from_ref(&path)) {
            Err(err) => err,
            Ok(_) => panic!("vocoder accepted empty weights"),
        };
        assert!(err.to_string().contains("conv_pre"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
