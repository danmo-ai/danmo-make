//! HunyuanVideo 1.5 causal 3D VAE encode/decode over the Tencent
//! `vae/diffusion_pytorch_model.safetensors` native key layout (no remap).
//!
//! Semantics are a port of main's `hunyuan/vae_mlx.py`
//! (`AutoencoderKLConv3D`, diffusers `HunyuanVideo15`):
//! - Activations are `[W, H, F, C]` F32 (channels on ne3). The safetensors
//!   conv kernels are PyTorch `[OC, IC, KD, KH, KW]`; the flat bytes double as
//!   the ggml `[KW, KH, KD, IC * OC]` kernel with `IC * OC` packed `ic + IC *
//!   oc`. RMSNorm gammas are `[C, 1, 1, 1]` -> ggml `[1, 1, 1, C]`.
//! - `HunyuanVideo15CausalConv3d`: time pad `kt - 1` replicating the first
//!   frame, then symmetric replicate pad `k//2` on H and W, then a valid
//!   3x3x3 conv.
//! - `HunyuanVideo15RMSNorm`: `x / sqrt(sum(x^2, C) + 1e-6) * sqrt(C) *
//!   gamma`, which is ggml `rms_norm(eps = 1e-6 / C)` plus the gamma
//!   multiply (sum = C * mean, and the `sqrt(C)` cancels).
//! - Mid block: resnet, then single-head attention over the `F * H * W`
//!   tokens (scale `C^-0.5`), then resnet. The temporal mask is frame-prefix
//!   causal: token `(f, i)` sees frames `0..=f`. main materializes a full
//!   `[seq, seq]` bias; here frame `f` runs flash attention against the
//!   concatenated `0..=f` prefix, which is the same math without the mask.
//! - DCAE downsample: 3x3x3 conv to `out / factor` channels, then
//!   space-to-depth with channel packing `p = ((i1*r2 + i2)*r3 + i3)*c + ci`
//!   (channels fastest). A temporal step (`r1 = 2`) special-cases the first
//!   frame: it is rearranged with `r1 = 1` and its channels doubled. The
//!   shortcut rearranges the input and group-means `factor * in / out`
//!   channels (`group / 2` for the first frame). Upsample is the inverse:
//!   conv to `out * factor`, depth-to-space, first frame drops half its
//!   channels, and the shortcut is a channel repeat-interleave.
//! - Encoder: conv_in -> 5 down blocks (2 resnets each; spatial downsample
//!   on blocks 0..4, temporal on 2..4) -> mid -> output is
//!   `conv_out(norm_out + silu)` plus a parameter-free group-mean shortcut
//!   (`block_out[-1] -> 2 * latent`). Encode takes the Gaussian mode (first
//!   `latent` channels) times `scaling_factor` (1.03682).
//! - Decoder: `conv_in(latents / scaling)` plus a repeat-interleave shortcut
//!   (`latent -> block_out[0]`), mid, 5 up blocks (3 resnets each; temporal
//!   upsample on blocks 0..2, spatial on 0..4), norm_out, silu, conv_out,
//!   clipped to `[-1, 1]`.
//!
//! One graph holds the whole clip; there is no spatial tiling or temporal
//! chunking yet (main tiles at 720p). Encoder input frames must satisfy
//! `(T - 1) % 4 == 0`; decode maps `T -> 4 * (T - 1) + 1`.

use crate::runtime::{flash_attend, shape, GgmlContext, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;

const GRAPH_MEM: usize = 64 << 20;
const RMS_EPS: f32 = 1e-6;
/// Published `vae/config.json` `scaling_factor` (encode multiplies, decode divides).
pub const SCALING: f32 = 1.03682;
/// Resnets per encoder down block (diffusers `layers_per_block`).
const ENC_RES: usize = 2;
/// Resnets per decoder up block (`layers_per_block + 1`).
const DEC_RES: usize = 3;
/// Spatial downsample on encoder blocks `< 4`, temporal on blocks `>= 2`.
const ENC_SPATIAL: usize = 4;
const ENC_TEMPORAL: usize = 2;
/// Temporal upsample on decoder blocks `< 2`, spatial on blocks `< 4`.
const DEC_TEMPORAL: usize = 2;
const DEC_SPATIAL: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HunyuanVaeSpec {
    /// Latent channels. The encoder emits `2 * latent` (mean + logvar).
    pub latent: i64,
    /// Encoder block widths `[128, 256, 512, 1024, 1024]`.
    pub enc_blocks: [i64; 5],
    /// Decoder block widths `[1024, 1024, 512, 256, 128]`.
    pub dec_blocks: [i64; 5],
    pub scaling: f32,
}

impl HunyuanVaeSpec {
    /// The published `Tencent-Hunyuan/HunyuanVideo-1.5` VAE topology.
    pub fn published() -> Self {
        Self {
            latent: 32,
            enc_blocks: [128, 256, 512, 1024, 1024],
            dec_blocks: [1024, 1024, 512, 256, 128],
            scaling: SCALING,
        }
    }

    /// Same topology at 1/16 the width for the host-reference test.
    pub fn smoke() -> Self {
        Self {
            latent: 4,
            enc_blocks: [8, 16, 32, 32, 32],
            dec_blocks: [32, 32, 32, 16, 8],
            scaling: SCALING,
        }
    }

    /// Latent frame count for `t` pixel frames (`(t - 1) / 4 + 1`).
    pub fn encode_frames(t: i64) -> i64 {
        (t - 1) / 4 + 1
    }

    /// Pixel frame count for `t` latent frames (`4 * (t - 1) + 1`).
    pub fn decode_frames(t: i64) -> i64 {
        4 * (t - 1) + 1
    }

    /// Infer the topology from the native safetensors shapes. The block
    /// widths come from the RMSNorm gammas; the downsample/upsample factors
    /// from the conv channel ratios, and both must match the published
    /// spatial/temporal pattern or the file fails loud.
    pub fn infer(weights: &GgufWeights) -> Result<Self, WeightError> {
        let gamma_c = |name: &str| -> Result<i64, WeightError> {
            let ne = weights.shape(name)?;
            if ne[0] != 1 || ne[1] != 1 || ne[2] != 1 || ne[3] < 1 {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae {name} is {ne:?}, not [1, 1, 1, C]"
                )));
            }
            Ok(ne[3])
        };
        let mut enc = [0i64; 5];
        for (i, slot) in enc.iter_mut().enumerate() {
            *slot = gamma_c(&format!("encoder.down.{i}.block.0.norm1.gamma"))?;
            let second = gamma_c(&format!("encoder.down.{i}.block.1.norm1.gamma"))?;
            if second != *slot {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae encoder down block {i} mixes channels {slot} and {second}"
                )));
            }
            if weights.contains(&format!("encoder.down.{i}.block.2.norm1.gamma")) {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae encoder down block {i} has a third resnet; the topology carries {ENC_RES}"
                )));
            }
        }
        let conv_in = weights.shape("encoder.conv_in.conv.weight")?;
        if conv_in[..3] != [3, 3, 3] || conv_in[3] != 3 * enc[0] {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae encoder.conv_in.conv.weight is {conv_in:?}, expected [3, 3, 3, {}]",
                3 * enc[0]
            )));
        }
        let out_ch = weights.shape("encoder.conv_out.conv.bias")?[0];
        if out_ch < 2 || out_ch % 2 != 0 {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae encoder emits {out_ch} channels, not an even mean/logvar split"
            )));
        }
        let latent = out_ch / 2;
        let conv_out = weights.shape("encoder.conv_out.conv.weight")?;
        if conv_out[..3] != [3, 3, 3] || conv_out[3] != enc[4] * out_ch {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae encoder.conv_out.conv.weight is {conv_out:?}"
            )));
        }
        for name in ["block_1", "block_2"] {
            let mid = gamma_c(&format!("encoder.mid.{name}.norm1.gamma"))?;
            if mid != enc[4] {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae encoder mid {name} channels {mid} != {}",
                    enc[4]
                )));
            }
        }
        let q = weights.shape("encoder.mid.attn_1.q.weight")?;
        if q[..3] != [1, 1, 1] || q[3] != enc[4] * enc[4] {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae encoder.mid.attn_1.q.weight is {q:?}"
            )));
        }
        for i in 0..4 {
            let name = format!("encoder.down.{i}.downsample.conv.conv.weight");
            let conv = weights.shape(&name)?;
            if conv[..3] != [3, 3, 3] || conv[3] % enc[i] != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae {name} is {conv:?}, in channels {}",
                    enc[i]
                )));
            }
            let cm = conv[3] / enc[i];
            if cm < 1 || enc[i + 1] % cm != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae {name} packs {cm} channels for out {}",
                    enc[i + 1]
                )));
            }
            let factor = enc[i + 1] / cm;
            let want = if i >= ENC_TEMPORAL { 8 } else { 4 };
            if factor != want {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae encoder downsample {i} factor {factor}, expected {want}"
                )));
            }
        }
        if weights.contains("encoder.down.4.downsample.conv.conv.weight") {
            return Err(WeightError::NotBuilt(
                "hunyuan vae encoder block 4 downsamples; the topology stops at 4".into(),
            ));
        }
        let mut dec = [0i64; 5];
        for (i, slot) in dec.iter_mut().enumerate() {
            *slot = gamma_c(&format!("decoder.up.{i}.block.0.norm1.gamma"))?;
            for j in 1..DEC_RES {
                let other = gamma_c(&format!("decoder.up.{i}.block.{j}.norm1.gamma"))?;
                if other != *slot {
                    return Err(WeightError::NotBuilt(format!(
                        "hunyuan vae decoder up block {i} mixes channels {slot} and {other}"
                    )));
                }
            }
            if weights.contains(&format!("decoder.up.{i}.block.{DEC_RES}.norm1.gamma")) {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae decoder up block {i} has a fourth resnet; the topology carries {DEC_RES}"
                )));
            }
        }
        let dec_in = weights.shape("decoder.conv_in.conv.weight")?;
        if dec_in[..3] != [3, 3, 3] || dec_in[3] != latent * dec[0] {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae decoder.conv_in.conv.weight is {dec_in:?}, latent {latent}"
            )));
        }
        if dec[0] % latent != 0 {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae decoder shortcut repeats {latent} -> {}",
                dec[0]
            )));
        }
        for i in 0..4 {
            let name = format!("decoder.up.{i}.upsample.conv.conv.weight");
            let conv = weights.shape(&name)?;
            if conv[..3] != [3, 3, 3] || conv[3] % dec[i] != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae {name} is {conv:?}, in channels {}",
                    dec[i]
                )));
            }
            let packed = conv[3] / dec[i];
            if packed % dec[i + 1] != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae {name} emits {packed} for out {}",
                    dec[i + 1]
                )));
            }
            let factor = packed / dec[i + 1];
            let want = if i < DEC_TEMPORAL { 8 } else { 4 };
            if factor != want {
                return Err(WeightError::NotBuilt(format!(
                    "hunyuan vae decoder upsample {i} factor {factor}, expected {want}"
                )));
            }
        }
        if weights.contains("decoder.up.4.upsample.conv.conv.weight") {
            return Err(WeightError::NotBuilt(
                "hunyuan vae decoder block 4 upsamples; the topology stops at 4".into(),
            ));
        }
        let pixel = weights.shape("decoder.conv_out.conv.bias")?[0];
        if pixel != 3 {
            return Err(WeightError::NotBuilt(format!(
                "hunyuan vae decoder emits {pixel} pixel channels, expected 3"
            )));
        }
        Ok(Self {
            latent,
            enc_blocks: enc,
            dec_blocks: dec,
            scaling: SCALING,
        })
    }
}

fn push_conv(out: &mut Vec<(String, Vec<i64>)>, prefix: &str, kernel: i64, ic: i64, oc: i64) {
    out.push((
        format!("{prefix}.conv.weight"),
        vec![kernel, kernel, kernel, ic * oc],
    ));
    out.push((format!("{prefix}.conv.bias"), vec![oc]));
}

fn push_res(out: &mut Vec<(String, Vec<i64>)>, prefix: &str, c: i64) {
    for norm in ["norm1", "norm2"] {
        out.push((format!("{prefix}.{norm}.gamma"), vec![1, 1, 1, c]));
    }
    push_conv(out, &format!("{prefix}.conv1"), 3, c, c);
    push_conv(out, &format!("{prefix}.conv2"), 3, c, c);
}

fn push_mid(out: &mut Vec<(String, Vec<i64>)>, prefix: &str, c: i64) {
    push_res(out, &format!("{prefix}.block_1"), c);
    let attn = format!("{prefix}.attn_1");
    out.push((format!("{attn}.norm.gamma"), vec![1, 1, 1, c]));
    for proj in ["q", "k", "v", "proj_out"] {
        out.push((format!("{attn}.{proj}.weight"), vec![1, 1, 1, c * c]));
        out.push((format!("{attn}.{proj}.bias"), vec![c]));
    }
    push_res(out, &format!("{prefix}.block_2"), c);
}

/// `(name, ggml dims)` for every encoder weight, native safetensors names.
pub fn encoder_layout(spec: &HunyuanVaeSpec) -> Vec<(String, Vec<i64>)> {
    let mut out = Vec::new();
    push_conv(&mut out, "encoder.conv_in", 3, 3, spec.enc_blocks[0]);
    for i in 0..5 {
        for j in 0..ENC_RES {
            push_res(
                &mut out,
                &format!("encoder.down.{i}.block.{j}"),
                spec.enc_blocks[i],
            );
        }
        if i < ENC_SPATIAL {
            let factor = if i >= ENC_TEMPORAL { 8 } else { 4 };
            let cm = spec.enc_blocks[i + 1] / factor;
            push_conv(
                &mut out,
                &format!("encoder.down.{i}.downsample.conv"),
                3,
                spec.enc_blocks[i],
                cm,
            );
        }
    }
    push_mid(&mut out, "encoder.mid", spec.enc_blocks[4]);
    out.push((
        "encoder.norm_out.gamma".into(),
        vec![1, 1, 1, spec.enc_blocks[4]],
    ));
    push_conv(
        &mut out,
        "encoder.conv_out",
        3,
        spec.enc_blocks[4],
        2 * spec.latent,
    );
    out
}

/// `(name, ggml dims)` for every decoder weight, native safetensors names.
pub fn decoder_layout(spec: &HunyuanVaeSpec) -> Vec<(String, Vec<i64>)> {
    let mut out = Vec::new();
    push_conv(
        &mut out,
        "decoder.conv_in",
        3,
        spec.latent,
        spec.dec_blocks[0],
    );
    push_mid(&mut out, "decoder.mid", spec.dec_blocks[0]);
    for i in 0..5 {
        for j in 0..DEC_RES {
            push_res(
                &mut out,
                &format!("decoder.up.{i}.block.{j}"),
                spec.dec_blocks[i],
            );
        }
        if i < DEC_SPATIAL {
            let factor = if i < DEC_TEMPORAL { 8 } else { 4 };
            push_conv(
                &mut out,
                &format!("decoder.up.{i}.upsample.conv"),
                3,
                spec.dec_blocks[i],
                spec.dec_blocks[i + 1] * factor,
            );
        }
    }
    out.push((
        "decoder.norm_out.gamma".into(),
        vec![1, 1, 1, spec.dec_blocks[4]],
    ));
    push_conv(&mut out, "decoder.conv_out", 3, spec.dec_blocks[4], 3);
    out
}

fn load_parts(
    paths: &[PathBuf],
    layout: &[(String, Vec<i64>)],
) -> Result<Vec<(String, Vec<f32>)>, WeightError> {
    let weights = GgufWeights::open_parts(paths)?;
    let mut params = Vec::with_capacity(layout.len());
    for (name, dims) in layout {
        params.push((name.clone(), weights.f32_widened(name, dims)?));
    }
    Ok(params)
}

/// Open the VAE safetensors and load the encoder weights (F32 widened).
pub fn load_encoder(
    paths: &[PathBuf],
) -> Result<(HunyuanVaeSpec, Vec<(String, Vec<f32>)>), WeightError> {
    let weights = GgufWeights::open_parts(paths)?;
    let spec = HunyuanVaeSpec::infer(&weights)?;
    let params = load_parts(paths, &encoder_layout(&spec))?;
    Ok((spec, params))
}

/// Open the VAE safetensors and load the decoder weights (F32 widened).
pub fn load_decoder(
    paths: &[PathBuf],
) -> Result<(HunyuanVaeSpec, Vec<(String, Vec<f32>)>), WeightError> {
    let weights = GgufWeights::open_parts(paths)?;
    let spec = HunyuanVaeSpec::infer(&weights)?;
    let params = load_parts(paths, &decoder_layout(&spec))?;
    Ok((spec, params))
}

// ---------------------------------------------------------------------------
// graph fragments
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
        .ok_or_else(|| RuntimeError::Message(format!("hunyuan vae graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let actual = shape(tensor);
    if actual != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {actual:?}, the hunyuan vae expects {ne:?}"
        )));
    }
    Ok(tensor)
}

/// `ggml_permute` uses destination semantics: `ne[axis_i] = old ne[i]`.
fn permute_cont(b: &GraphBuilder, x: Tensor, ax: [i32; 4]) -> Result<Tensor, RuntimeError> {
    nn(b.cont(b.permute(x, ax[0], ax[1], ax[2], ax[3])))
}

/// Frame slice `[w, h, count, c]` of a contiguous `[w, h, f, c]` volume.
fn frames_view(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    f: i64,
    c: i64,
    start: i64,
    count: i64,
) -> Result<Tensor, RuntimeError> {
    nn(b.cont(b.view_4d(
        x,
        w,
        h,
        count,
        c,
        (w * 4) as usize,
        (w * h * 4) as usize,
        (w * h * f * 4) as usize,
        (start * w * h * 4) as usize,
    )))
}

/// 3x3x3 causal conv: time pad 2 replicating frame 0, symmetric replicate
/// pad 1 on H and W, valid conv, bias.
#[allow(clippy::too_many_arguments)]
fn causal_conv3d(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    f: i64,
    ic: i64,
    oc: i64,
) -> Result<Tensor, RuntimeError> {
    let first = frames_view(b, x, w, h, f, ic, 0, 1)?;
    let x = nn(b.concat(nn(b.concat(first, first, 2))?, x, 2))?;
    let f2 = f + 2;
    let row = |offset: usize| -> Result<Tensor, RuntimeError> {
        nn(b.cont(b.view_4d(
            x,
            w,
            1,
            f2,
            ic,
            (w * 4) as usize,
            (w * h * 4) as usize,
            (w * h * f2 * 4) as usize,
            offset,
        )))
    };
    let x = nn(b.concat(
        nn(b.concat(row(0)?, x, 1))?,
        row(((h - 1) * w * 4) as usize)?,
        1,
    ))?;
    let h2 = h + 2;
    let col = |offset: usize| -> Result<Tensor, RuntimeError> {
        nn(b.cont(b.view_4d(
            x,
            1,
            h2,
            f2,
            ic,
            (w * 4) as usize,
            (w * h2 * 4) as usize,
            (w * h2 * f2 * 4) as usize,
            offset,
        )))
    };
    let x = nn(b.concat(
        nn(b.concat(col(0)?, x, 0))?,
        col(((w - 1) * 4) as usize)?,
        0,
    ))?;
    let kernel = wt(weights, &format!("{prefix}.conv.weight"))?;
    let y = nn(b.conv3d_direct(
        kernel, x, 1, 1, 1, 0, 0, 0, 1, 1, 1, ic as i32, 1, oc as i32,
    ))?;
    let y = shaped(y, [w, h, f, oc], &format!("{prefix}.conv.weight"))?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.conv.bias"))?, 1, 1, 1, oc)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    shaped(y, [w, h, f, oc], &format!("{prefix}.conv.bias"))
}

/// `x / sqrt(sum(x^2, C) + 1e-6) * sqrt(C) * gamma` = `rms_norm(1e-6/C) * gamma`.
fn hunyuan_rms(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    name: &str,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let y = permute_cont(b, x, [1, 2, 3, 0])?; // [c, w, h, f]
    let y = nn(b.rms_norm(y, RMS_EPS / c as f32))?;
    let gamma = nn(b.cont(b.reshape_4d(wt(weights, name)?, c, 1, 1, 1)))?;
    let y = nn(b.mul(y, nn(b.repeat(gamma, y))?))?;
    permute_cont(b, y, [3, 0, 1, 2]) // [w, h, f, c]
}

fn silu(b: &GraphBuilder, x: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.silu(x))
}

#[allow(clippy::too_many_arguments)]
fn res_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    f: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let y = hunyuan_rms(b, weights, x, &format!("{prefix}.norm1.gamma"), c)?;
    let y = silu(b, y)?;
    let y = causal_conv3d(b, weights, y, &format!("{prefix}.conv1"), w, h, f, c, c)?;
    let y = hunyuan_rms(b, weights, y, &format!("{prefix}.norm2.gamma"), c)?;
    let y = silu(b, y)?;
    let y = causal_conv3d(b, weights, y, &format!("{prefix}.conv2"), w, h, f, c, c)?;
    nn(b.add(x, y))
}

/// 1x1x1 conv as a matmul on `[C, tokens]`: weight `[1, 1, 1, C*C]` packs
/// `i + C * o`, so reshaped `[C, C]` multiplies directly.
fn point_linear(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    t: Tensor,
    name: &str,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let w2 = nn(b.reshape_2d(wt(weights, &format!("{name}.weight"))?, c, c))?;
    let y = nn(b.mul_mat(w2, t))?;
    let bias = nn(b.cont(b.reshape_2d(wt(weights, &format!("{name}.bias"))?, c, 1)))?;
    nn(b.add(y, nn(b.repeat(bias, y))?))
}

/// Single-head attention over the `F * H * W` tokens with frame-prefix
/// causality: frame `f` flash-attends to the `0..=f` prefix (no mask).
fn attn_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    f: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let n = hunyuan_rms(b, weights, x, &format!("{prefix}.norm.gamma"), c)?;
    let t = permute_cont(b, n, [1, 2, 3, 0])?; // [c, w, h, f]
    let t = nn(b.reshape_2d(t, c, w * h * f))?;
    let q = point_linear(b, weights, t, &format!("{prefix}.q"), c)?;
    let k = point_linear(b, weights, t, &format!("{prefix}.k"), c)?;
    let v = point_linear(b, weights, t, &format!("{prefix}.v"), c)?;
    let hw = w * h;
    let row = (c * 4) as usize;
    let mut att: Option<Tensor> = None;
    for fr in 0..f {
        let qf = nn(b.reshape_3d(
            nn(b.view_2d(q, c, hw, row, (fr * hw * c * 4) as usize))?,
            c,
            1,
            hw,
        ))?;
        let kp = nn(b.reshape_3d(
            nn(b.view_2d(k, c, (fr + 1) * hw, row, 0))?,
            c,
            1,
            (fr + 1) * hw,
        ))?;
        let vp = nn(b.reshape_3d(
            nn(b.view_2d(v, c, (fr + 1) * hw, row, 0))?,
            c,
            1,
            (fr + 1) * hw,
        ))?;
        let a = flash_attend(b, qf, kp, vp, std::ptr::null_mut())?;
        let a = nn(b.reshape_2d(a, c, hw))?;
        att = Some(match att {
            None => a,
            Some(acc) => nn(b.concat(acc, a, 1))?,
        });
    }
    let att = match att {
        Some(att) => att,
        None => {
            return Err(RuntimeError::Message(
                "hunyuan vae attention needs frames".into(),
            ))
        }
    };
    let proj = point_linear(b, weights, att, &format!("{prefix}.proj_out"), c)?;
    let y = nn(b.reshape_4d(proj, c, w, h, f))?;
    let y = permute_cont(b, y, [3, 0, 1, 2])?; // [w, h, f, c]
    nn(b.add(x, y))
}

/// Channel group mean `[w, h, f, in_c] -> [w, h, f, out_c]`: output channel
/// `o` averages input channels `[o * g, (o + 1) * g)` (`g = in_c / out_c`).
fn group_mean(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    f: i64,
    in_c: i64,
    out_c: i64,
) -> Result<Tensor, RuntimeError> {
    if in_c < out_c || in_c % out_c != 0 {
        return Err(RuntimeError::Message(format!(
            "hunyuan vae group mean {in_c} -> {out_c} is not an integer group"
        )));
    }
    let g = in_c / out_c;
    let y = nn(b.reshape_4d(x, w * h * f, g, out_c, 1))?;
    let y = permute_cont(b, y, [1, 0, 2, 3])?; // [g, whf, out, 1]
    let y = nn(b.sum_rows(y))?;
    let y = nn(b.scale(y, (g as f32).recip()))?;
    nn(b.reshape_4d(y, w, h, f, out_c))
}

/// Space-to-depth `[W, H, F, C] -> [W/2, H/2, F/r1, r1*4*C]` with channel
/// packing `p = ((i1*2 + i2)*2 + i3)*c + ci` (main `_dcae_downsample_rearrange`).
fn rearrange_down(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    f: i64,
    c: i64,
    r1: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.reshape_4d(x, 2, w / 2, h, f * c))?; // [i3, w', h, (f, ci)]
    let y = permute_cont(b, y, [3, 0, 1, 2])?; // [w', h, fc, i3]
    let y = nn(b.reshape_4d(y, w / 2, 2, h / 2, f * c * 2))?; // [w', i2, h', (f, ci, i3)]
    let y = permute_cont(b, y, [0, 3, 1, 2])?; // [w', h', fc2, i2]
    let y = nn(b.reshape_4d(y, w / 2, h / 2, r1, (f / r1) * c * 4))?; // [w', h', i1, (f', ci, i3, i2)]
    let y = permute_cont(b, y, [0, 1, 3, 2])?; // [w', h', f'c4, i1]
    let y = nn(b.reshape_4d(y, w / 2, h / 2, f / r1, c * 4 * r1))?;
    shaped(
        y,
        [w / 2, h / 2, f / r1, c * 4 * r1],
        "hunyuan vae rearrange_down",
    )
}

/// Depth-to-space `[W, H, F, r1*4*C] -> [2W, 2H, F*r1, C]`, the inverse of
/// [`rearrange_down`] (main `_dcae_upsample_rearrange`).
fn rearrange_up(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    f: i64,
    c: i64,
    r1: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.reshape_4d(x, w, h * f * c, 2, r1 * 2))?; // [w, (h,f,ci), i3, (i2,i1)]
    let y = permute_cont(b, y, [1, 2, 0, 3])?; // [i3, w, hfc, i2i1] — sub-pixel axis first
    let y = nn(b.reshape_4d(y, w * 2, h, f, c * 2 * r1))?; // [2w, h, f, (ci,i2,i1)]
    let y = nn(b.reshape_4d(y, w * 2, h * f * c, 2, r1))?; // [2w, (h,f,ci), i2, i1]
    let y = permute_cont(b, y, [0, 2, 1, 3])?; // [2w, i2, hfc, i1]
    let y = nn(b.reshape_4d(y, w * 2, h * 2, f * c, r1))?; // [2w, 2h, (f,ci), i1]
    let y = nn(b.reshape_4d(y, w * 2 * h * 2, f, c, r1))?; // [wh, f, ci, i1]
    let y = permute_cont(b, y, [0, 2, 3, 1])?; // [wh, i1, f, ci]
    let y = nn(b.reshape_4d(y, w * 2, h * 2, f * r1, c))?;
    shaped(y, [w * 2, h * 2, f * r1, c], "hunyuan vae rearrange_up")
}

/// Channel repeat-interleave: out channel `ci * r + j` is input channel `ci`
/// (main `_repeat_interleave_channels`).
fn repeat_interleave(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    f: i64,
    c: i64,
    r: i64,
) -> Result<Tensor, RuntimeError> {
    if r <= 1 {
        return Ok(x);
    }
    let y = nn(b.reshape_4d(x, w * h * f, c, 1, 1))?;
    let y = nn(b.repeat_4d(y, w * h * f, c, r, 1))?;
    let y = permute_cont(b, y, [0, 2, 1, 3])?; // [whf, r, c, 1]
    let y = nn(b.reshape_4d(y, w, h, f, c * r))?;
    shaped(y, [w, h, f, c * r], "hunyuan vae repeat_interleave")
}

/// Encoder downsample: conv to `oc / factor`, DCAE space-to-depth, first
/// frame special-cased on temporal steps, plus the group-mean shortcut.
#[allow(clippy::too_many_arguments)]
fn enc_downsample(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    f: i64,
    ic: i64,
    oc: i64,
    temporal: bool,
) -> Result<(Tensor, i64), RuntimeError> {
    let factor = if temporal { 8 } else { 4 };
    let cm = oc / factor;
    let cv = causal_conv3d(b, weights, x, &format!("{prefix}.conv"), w, h, f, ic, cm)?;
    let (ow, oh) = (w / 2, h / 2);
    if !temporal {
        let main = rearrange_down(b, cv, w, h, f, cm, 1)?;
        let sc = rearrange_down(b, x, w, h, f, ic, 1)?;
        let sc = group_mean(b, sc, ow, oh, f, 4 * ic, oc)?;
        return Ok((nn(b.add(main, sc))?, f));
    }
    let first = rearrange_down(b, frames_view(b, cv, w, h, f, cm, 0, 1)?, w, h, 1, cm, 1)?;
    let first = nn(b.concat(first, first, 3))?;
    let next = rearrange_down(
        b,
        frames_view(b, cv, w, h, f, cm, 1, f - 1)?,
        w,
        h,
        f - 1,
        cm,
        2,
    )?;
    let main = nn(b.concat(first, next, 2))?;
    let x_first = rearrange_down(b, frames_view(b, x, w, h, f, ic, 0, 1)?, w, h, 1, ic, 1)?;
    let x_first = group_mean(b, x_first, ow, oh, 1, 4 * ic, oc)?;
    let x_next = rearrange_down(
        b,
        frames_view(b, x, w, h, f, ic, 1, f - 1)?,
        w,
        h,
        f - 1,
        ic,
        2,
    )?;
    let x_next = group_mean(b, x_next, ow, oh, (f - 1) / 2, 8 * ic, oc)?;
    let sc = nn(b.concat(x_first, x_next, 2))?;
    Ok((nn(b.add(main, sc))?, (f - 1) / 2 + 1))
}

/// Decoder upsample: conv to `oc * factor`, DCAE depth-to-space, first frame
/// halves channels on temporal steps, plus the repeat-interleave shortcut.
#[allow(clippy::too_many_arguments)]
fn dec_upsample(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    f: i64,
    ic: i64,
    oc: i64,
    temporal: bool,
) -> Result<(Tensor, i64), RuntimeError> {
    let factor = if temporal { 8 } else { 4 };
    let repeats = factor * oc / ic;
    let cv = causal_conv3d(
        b,
        weights,
        x,
        &format!("{prefix}.conv"),
        w,
        h,
        f,
        ic,
        oc * factor,
    )?;
    if !temporal {
        let main = rearrange_up(b, cv, w, h, f, oc, 1)?;
        let sc = repeat_interleave(b, x, w, h, f, ic, repeats)?;
        let sc = rearrange_up(b, sc, w, h, f, oc, 1)?;
        return Ok((nn(b.add(main, sc))?, f));
    }
    let first = rearrange_up(
        b,
        frames_view(b, cv, w, h, f, oc * 8, 0, 1)?,
        w,
        h,
        1,
        oc * 2,
        1,
    )?;
    let (w2, h2) = (w * 2, h * 2);
    let first = nn(b.cont(b.view_4d(
        first,
        w2,
        h2,
        1,
        oc,
        (w2 * 4) as usize,
        (w2 * h2 * 4) as usize,
        (w2 * h2 * 4) as usize,
        0,
    )))?;
    let next = rearrange_up(
        b,
        frames_view(b, cv, w, h, f, oc * 8, 1, f - 1)?,
        w,
        h,
        f - 1,
        oc,
        2,
    )?;
    let main = nn(b.concat(first, next, 2))?;
    let x_first = rearrange_up(b, frames_view(b, x, w, h, f, ic, 0, 1)?, w, h, 1, ic / 4, 1)?;
    let x_first = repeat_interleave(b, x_first, w2, h2, 1, ic / 4, repeats / 2)?;
    let x_next = rearrange_up(
        b,
        frames_view(b, x, w, h, f, ic, 1, f - 1)?,
        w,
        h,
        f - 1,
        ic / 8,
        2,
    )?;
    let x_next = repeat_interleave(b, x_next, w2, h2, 2 * (f - 1), ic / 8, repeats)?;
    let sc = nn(b.concat(x_first, x_next, 2))?;
    Ok((nn(b.add(main, sc))?, 2 * f - 1))
}

// ---------------------------------------------------------------------------
// towers
// ---------------------------------------------------------------------------

fn declare(
    b: &mut GraphBuilder,
    layout: &[(String, Vec<i64>)],
) -> Result<HashMap<String, Tensor>, RuntimeError> {
    let mut map = HashMap::with_capacity(layout.len());
    for (name, dims) in layout {
        let tensor = match dims.len() {
            1 => b.f32_1d(name, dims[0], true),
            4 => b.f32_4d(name, dims[0], dims[1], dims[2], dims[3], true),
            other => {
                return Err(RuntimeError::Message(format!(
                    "hunyuan vae weight {name} has {other} dims"
                )))
            }
        };
        map.insert(name.clone(), nn(tensor)?);
    }
    Ok(map)
}

fn encode_tower(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    spec: &HunyuanVaeSpec,
    pixels: Tensor,
    w: i64,
    h: i64,
    t: i64,
) -> Result<Tensor, RuntimeError> {
    let mut x = causal_conv3d(
        b,
        weights,
        pixels,
        "encoder.conv_in",
        w,
        h,
        t,
        3,
        spec.enc_blocks[0],
    )?;
    let (mut cw, mut ch, mut cf) = (w, h, t);
    for i in 0..5 {
        for j in 0..ENC_RES {
            x = res_block(
                b,
                weights,
                x,
                &format!("encoder.down.{i}.block.{j}"),
                cw,
                ch,
                cf,
                spec.enc_blocks[i],
            )?;
        }
        if i < ENC_SPATIAL {
            let temporal = i >= ENC_TEMPORAL;
            let (y, frames) = enc_downsample(
                b,
                weights,
                x,
                &format!("encoder.down.{i}.downsample"),
                cw,
                ch,
                cf,
                spec.enc_blocks[i],
                spec.enc_blocks[i + 1],
                temporal,
            )?;
            x = y;
            cw /= 2;
            ch /= 2;
            cf = frames;
        }
    }
    let cm = spec.enc_blocks[4];
    x = res_block(b, weights, x, "encoder.mid.block_1", cw, ch, cf, cm)?;
    x = attn_block(b, weights, x, "encoder.mid.attn_1", cw, ch, cf, cm)?;
    x = res_block(b, weights, x, "encoder.mid.block_2", cw, ch, cf, cm)?;
    let shortcut = group_mean(b, x, cw, ch, cf, cm, 2 * spec.latent)?;
    x = hunyuan_rms(b, weights, x, "encoder.norm_out.gamma", cm)?;
    x = silu(b, x)?;
    x = causal_conv3d(
        b,
        weights,
        x,
        "encoder.conv_out",
        cw,
        ch,
        cf,
        cm,
        2 * spec.latent,
    )?;
    nn(b.add(x, shortcut))
}

fn decode_tower(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    spec: &HunyuanVaeSpec,
    latents: Tensor,
    w: i64,
    h: i64,
    t: i64,
) -> Result<Tensor, RuntimeError> {
    let d = spec.dec_blocks;
    let x0 = causal_conv3d(
        b,
        weights,
        latents,
        "decoder.conv_in",
        w,
        h,
        t,
        spec.latent,
        d[0],
    )?;
    let shortcut = repeat_interleave(b, latents, w, h, t, spec.latent, d[0] / spec.latent)?;
    let mut x = nn(b.add(x0, shortcut))?;
    let (mut cw, mut ch, mut cf) = (w, h, t);
    x = res_block(b, weights, x, "decoder.mid.block_1", cw, ch, cf, d[0])?;
    x = attn_block(b, weights, x, "decoder.mid.attn_1", cw, ch, cf, d[0])?;
    x = res_block(b, weights, x, "decoder.mid.block_2", cw, ch, cf, d[0])?;
    for i in 0..5 {
        for j in 0..DEC_RES {
            x = res_block(
                b,
                weights,
                x,
                &format!("decoder.up.{i}.block.{j}"),
                cw,
                ch,
                cf,
                d[i],
            )?;
        }
        if i < DEC_SPATIAL {
            let temporal = i < DEC_TEMPORAL;
            let (y, frames) = dec_upsample(
                b,
                weights,
                x,
                &format!("decoder.up.{i}.upsample"),
                cw,
                ch,
                cf,
                d[i],
                d[i + 1],
                temporal,
            )?;
            x = y;
            cw *= 2;
            ch *= 2;
            cf = frames;
        }
    }
    x = hunyuan_rms(b, weights, x, "decoder.norm_out.gamma", d[4])?;
    x = silu(b, x)?;
    x = causal_conv3d(b, weights, x, "decoder.conv_out", cw, ch, cf, d[4], 3)?;
    nn(b.clamp(x, -1.0, 1.0))
}

fn bad_geometry(what: &str, got: Vec<i64>) -> WeightError {
    WeightError::Shape {
        name: format!("hunyuan vae {what}"),
        expected: vec![1],
        got,
    }
}

/// Encode `pixels` (`[W, H, T, 3]` F32 in `[-1, 1]`, w fastest) to latents
/// `[W/16, H/16, (T-1)/4+1, latent]` (Gaussian mode times `scaling_factor`).
pub fn encode(
    ctx: &GgmlContext,
    params: &[(String, Vec<f32>)],
    spec: &HunyuanVaeSpec,
    pixels: &[f32],
    w: i64,
    h: i64,
    t: i64,
) -> Result<Vec<f32>, WeightError> {
    if w < 16 || h < 16 || w % 16 != 0 || h % 16 != 0 {
        return Err(bad_geometry("encode spatial geometry", vec![w, h]));
    }
    if t < 1 || (t - 1) % 4 != 0 {
        return Err(bad_geometry(
            "encode frames (need (T - 1) % 4 == 0)",
            vec![t],
        ));
    }
    if pixels.len() != (w * h * t * 3) as usize {
        return Err(bad_geometry(
            "encode pixels",
            vec![pixels.len() as i64, w * h * t * 3],
        ));
    }
    let mut b = ctx.builder(GRAPH_MEM)?;
    let wmap = declare(&mut b, &encoder_layout(spec))?;
    let input = nn(b.f32_4d("pixels", w, h, t, 3, true))?;
    let out = encode_tower(&b, &wmap, spec, input, w, h, t)?;
    let out = nn(b.cont(out))?;
    let mut graph = b.finish(out)?;
    for (name, data) in params {
        graph.set_f32(name, data)?;
    }
    graph.set_f32("pixels", pixels)?;
    let moments = graph.compute()?;
    let (lw, lh, lf) = (w / 16, h / 16, HunyuanVaeSpec::encode_frames(t));
    let latent = spec.latent as usize;
    let plane = (lw * lh * lf) as usize;
    if moments.len() != plane * 2 * latent {
        return Err(WeightError::NotBuilt(format!(
            "hunyuan vae encoder returned {} values, expected {}",
            moments.len(),
            plane * 2 * latent
        )));
    }
    let mut out = vec![0f32; plane * latent];
    for c in 0..latent {
        for i in 0..plane {
            out[c * plane + i] = moments[c * plane + i] * spec.scaling;
        }
    }
    Ok(out)
}

/// Decode `latents` (`[W, H, T, latent]`, w fastest) to pixels
/// `[16W, 16H, 4(T-1)+1, 3]` clipped to `[-1, 1]`.
pub fn decode(
    ctx: &GgmlContext,
    params: &[(String, Vec<f32>)],
    spec: &HunyuanVaeSpec,
    latents: &[f32],
    w: i64,
    h: i64,
    t: i64,
) -> Result<Vec<f32>, WeightError> {
    if w < 1 || h < 1 || t < 1 {
        return Err(bad_geometry("decode geometry", vec![w, h, t]));
    }
    if latents.len() != (w * h * t * spec.latent) as usize {
        return Err(bad_geometry(
            "decode latents",
            vec![latents.len() as i64, w * h * t * spec.latent],
        ));
    }
    let scaled: Vec<f32> = latents.iter().map(|v| v / spec.scaling).collect();
    let mut b = ctx.builder(GRAPH_MEM)?;
    let wmap = declare(&mut b, &decoder_layout(spec))?;
    let input = nn(b.f32_4d("latents", w, h, t, spec.latent, true))?;
    let out = decode_tower(&b, &wmap, spec, input, w, h, t)?;
    let out = nn(b.cont(out))?;
    let mut graph = b.finish(out)?;
    for (name, data) in params {
        graph.set_f32(name, data)?;
    }
    graph.set_f32("latents", &scaled)?;
    let pixels = graph.compute()?;
    let expected = (w * 16 * h * 16 * HunyuanVaeSpec::decode_frames(t) * 3) as usize;
    if pixels.len() != expected {
        return Err(WeightError::NotBuilt(format!(
            "hunyuan vae decoder returned {} values, expected {expected}",
            pixels.len()
        )));
    }
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::{decode, decoder_layout, encode, encoder_layout, HunyuanVaeSpec, SCALING};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_safetensors, GgufWeights, SafeTensor};
    use std::collections::HashMap;

    fn tap(name: &str, index: usize, scale: f32) -> f32 {
        let mut hash = 2166136261u32;
        for byte in name.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
        }
        hash = hash.wrapping_mul(16777619) ^ index as u32;
        ((hash % 13) as f32 - 6.0) * scale
    }

    /// Deterministic weight value: small for convs, near-1 for gammas.
    fn weight_at(name: &str, index: usize) -> f32 {
        if name.ends_with(".gamma") {
            1.0 + tap(name, index, 0.02)
        } else if name.ends_with(".bias") {
            tap(name, index, 0.01)
        } else if name.contains(".attn_1.") {
            tap(name, index, 0.05)
        } else {
            tap(name, index, 0.02)
        }
    }

    fn params_for(layout: &[(String, Vec<i64>)]) -> Vec<(String, Vec<f32>)> {
        layout
            .iter()
            .map(|(name, dims)| {
                let len = dims.iter().product::<i64>() as usize;
                (name.clone(), (0..len).map(|i| weight_at(name, i)).collect())
            })
            .collect()
    }

    fn map_of(params: &[(String, Vec<f32>)]) -> HashMap<String, Vec<f32>> {
        params.iter().cloned().collect()
    }

    // -----------------------------------------------------------------------
    // host reference (NCTHW volumes, main `vae_mlx.py` semantics)
    // -----------------------------------------------------------------------

    #[derive(Clone)]
    struct Vol {
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        d: Vec<f32>,
    }

    impl Vol {
        fn new(c: usize, t: usize, h: usize, w: usize) -> Self {
            Self {
                c,
                t,
                h,
                w,
                d: vec![0.0; c * t * h * w],
            }
        }
        fn at(&self, c: usize, f: usize, y: usize, x: usize) -> f32 {
            self.d[((c * self.t + f) * self.h + y) * self.w + x]
        }
        fn set(&mut self, c: usize, f: usize, y: usize, x: usize, v: f32) {
            self.d[((c * self.t + f) * self.h + y) * self.w + x] = v;
        }
        /// ggml flat `[W, H, F, C]` order.
        fn ggml(&self) -> Vec<f32> {
            let mut out = vec![0f32; self.d.len()];
            for c in 0..self.c {
                for f in 0..self.t {
                    for y in 0..self.h {
                        for x in 0..self.w {
                            out[x + self.w * (y + self.h * (f + self.t * c))] = self.at(c, f, y, x);
                        }
                    }
                }
            }
            out
        }
    }

    fn at<'a>(map: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a [f32] {
        map.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    /// Causal conv3d: replicate time pad 2 in front, replicate spatial pad 1.
    /// The weight flat layout is `[KW, KH, KD, IC*OC]` = torch `[OC, IC, KD, KH, KW]`.
    fn h_conv3d(x: &Vol, wt: &[f32], bias: &[f32], oc: usize) -> Vol {
        let mut out = Vol::new(oc, x.t, x.h, x.w);
        for o in 0..oc {
            for f in 0..x.t {
                for yy in 0..x.h {
                    for xx in 0..x.w {
                        let mut acc = bias[o];
                        for i in 0..x.c {
                            for zd in 0..3 {
                                let sf = f as i64 + zd as i64 - 2;
                                let sf = sf.clamp(0, x.t as i64 - 1) as usize;
                                for ky in 0..3 {
                                    let sy = (yy as i64 + ky as i64 - 1).clamp(0, x.h as i64 - 1)
                                        as usize;
                                    for kx in 0..3 {
                                        let sx = (xx as i64 + kx as i64 - 1)
                                            .clamp(0, x.w as i64 - 1)
                                            as usize;
                                        acc += x.at(i, sf, sy, sx)
                                            * wt[(((o * x.c + i) * 3 + zd) * 3 + ky) * 3 + kx];
                                    }
                                }
                            }
                        }
                        out.set(o, f, yy, xx, acc);
                    }
                }
            }
        }
        out
    }

    fn h_rms(x: &mut Vol, gamma: &[f32]) {
        let scale = (x.c as f32).sqrt();
        for f in 0..x.t {
            for y in 0..x.h {
                for xx in 0..x.w {
                    let mut sum = 0f32;
                    for c in 0..x.c {
                        let v = x.at(c, f, y, xx);
                        sum += v * v;
                    }
                    let inv = scale / (sum + 1e-6).sqrt();
                    for c in 0..x.c {
                        x.set(c, f, y, xx, x.at(c, f, y, xx) * inv * gamma[c]);
                    }
                }
            }
        }
    }

    fn h_silu(x: &mut Vol) {
        for v in x.d.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    fn h_add(a: &Vol, b: &Vol) -> Vol {
        assert_eq!(a.d.len(), b.d.len());
        let mut out = a.clone();
        for (v, r) in out.d.iter_mut().zip(&b.d) {
            *v += r;
        }
        out
    }

    fn h_res(x: &Vol, prefix: &str, map: &HashMap<String, Vec<f32>>) -> Vol {
        let mut y = x.clone();
        h_rms(&mut y, at(map, &format!("{prefix}.norm1.gamma")));
        h_silu(&mut y);
        let mut y = h_conv3d(
            &y,
            at(map, &format!("{prefix}.conv1.conv.weight")),
            at(map, &format!("{prefix}.conv1.conv.bias")),
            x.c,
        );
        h_rms(&mut y, at(map, &format!("{prefix}.norm2.gamma")));
        h_silu(&mut y);
        let y = h_conv3d(
            &y,
            at(map, &format!("{prefix}.conv2.conv.weight")),
            at(map, &format!("{prefix}.conv2.conv.bias")),
            x.c,
        );
        h_add(x, &y)
    }

    /// Frame-prefix causal single-head attention (main `HunyuanVideo15AttnBlock`).
    fn h_attn(x: &Vol, prefix: &str, map: &HashMap<String, Vec<f32>>) -> Vol {
        let c = x.c;
        let hw = x.h * x.w;
        let seq = x.t * hw;
        let mut normed = x.clone();
        h_rms(&mut normed, at(map, &format!("{prefix}.norm.gamma")));
        let project = |proj: &str| -> Vec<f32> {
            let wt = at(map, &format!("{prefix}.{proj}.weight"));
            let bias = at(map, &format!("{prefix}.{proj}.bias"));
            let mut out = vec![0f32; c * seq];
            for o in 0..c {
                for f in 0..x.t {
                    for px in 0..hw {
                        let (y, xx) = (px / x.w, px % x.w);
                        let mut acc = bias[o];
                        for i in 0..c {
                            acc += normed.at(i, f, y, xx) * wt[i + c * o];
                        }
                        out[o * seq + f * hw + px] = acc;
                    }
                }
            }
            out
        };
        let q = project("q");
        let k = project("k");
        let v = project("v");
        let scale = (c as f32).sqrt().recip();
        let mut mixed = vec![0f32; c * seq];
        for f in 0..x.t {
            let prefix_tokens = (f + 1) * hw;
            for px in 0..hw {
                let token = f * hw + px;
                let mut scores = vec![0f32; prefix_tokens];
                let mut peak = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for i in 0..c {
                        acc += q[i * seq + token] * k[i * seq + j];
                    }
                    *score = acc * scale;
                    peak = peak.max(*score);
                }
                let mut denom = 0f32;
                for score in scores.iter_mut() {
                    *score = (*score - peak).exp();
                    denom += *score;
                }
                for i in 0..c {
                    let mut acc = 0f32;
                    for (j, score) in scores.iter().enumerate() {
                        acc += v[i * seq + j] * score;
                    }
                    mixed[i * seq + token] = acc / denom;
                }
            }
        }
        let wt = at(map, &format!("{prefix}.proj_out.weight"));
        let bias = at(map, &format!("{prefix}.proj_out.bias"));
        let mut out = x.clone();
        for o in 0..c {
            for f in 0..x.t {
                for px in 0..hw {
                    let token = f * hw + px;
                    let mut acc = bias[o];
                    for i in 0..c {
                        acc += mixed[i * seq + token] * wt[i + c * o];
                    }
                    let (y, xx) = (px / x.w, px % x.w);
                    let prev = out.at(o, f, y, xx);
                    out.set(o, f, y, xx, prev + acc);
                }
            }
        }
        out
    }

    /// `(c, t, h, w) -> (r1*4*c, t/r1, h/2, w/2)`, channel packing
    /// `p = ((i1*2 + i2)*2 + i3)*c + ci`.
    fn h_rearrange_down(x: &Vol, r1: usize) -> Vol {
        let mut out = Vol::new(r1 * 4 * x.c, x.t / r1, x.h / 2, x.w / 2);
        for ci in 0..x.c {
            for f in 0..x.t / r1 {
                for y in 0..x.h / 2 {
                    for xx in 0..x.w / 2 {
                        for i1 in 0..r1 {
                            for i2 in 0..2 {
                                for i3 in 0..2 {
                                    let p = ((i1 * 2 + i2) * 2 + i3) * x.c + ci;
                                    out.set(
                                        p,
                                        f,
                                        y,
                                        xx,
                                        x.at(ci, f * r1 + i1, y * 2 + i2, xx * 2 + i3),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// `(r1*4*c, t, h, w) -> (c, t*r1, 2h, 2w)`, the inverse packing.
    fn h_rearrange_up(x: &Vol, r1: usize) -> Vol {
        let c = x.c / (r1 * 4);
        let mut out = Vol::new(c, x.t * r1, x.h * 2, x.w * 2);
        for ci in 0..c {
            for f in 0..x.t {
                for y in 0..x.h {
                    for xx in 0..x.w {
                        for i1 in 0..r1 {
                            for i2 in 0..2 {
                                for i3 in 0..2 {
                                    let p = ((i1 * 2 + i2) * 2 + i3) * c + ci;
                                    out.set(
                                        ci,
                                        f * r1 + i1,
                                        y * 2 + i2,
                                        xx * 2 + i3,
                                        x.at(p, f, y, xx),
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Channel group mean: `out[o]` averages `in[o*g..(o+1)*g)`.
    fn h_group_mean(x: &Vol, out_c: usize) -> Vol {
        let g = x.c / out_c;
        let mut out = Vol::new(out_c, x.t, x.h, x.w);
        for o in 0..out_c {
            for f in 0..x.t {
                for y in 0..x.h {
                    for xx in 0..x.w {
                        let mut acc = 0f32;
                        for j in 0..g {
                            acc += x.at(o * g + j, f, y, xx);
                        }
                        out.set(o, f, y, xx, acc / g as f32);
                    }
                }
            }
        }
        out
    }

    /// `out[ci*r + j] = in[ci]`.
    fn h_repeat_interleave(x: &Vol, r: usize) -> Vol {
        let mut out = Vol::new(x.c * r, x.t, x.h, x.w);
        for ci in 0..x.c {
            for j in 0..r {
                for f in 0..x.t {
                    for y in 0..x.h {
                        for xx in 0..x.w {
                            out.set(ci * r + j, f, y, xx, x.at(ci, f, y, xx));
                        }
                    }
                }
            }
        }
        out
    }

    fn frame_slice(x: &Vol, start: usize, count: usize) -> Vol {
        let mut out = Vol::new(x.c, count, x.h, x.w);
        for c in 0..x.c {
            for f in 0..count {
                for y in 0..x.h {
                    for xx in 0..x.w {
                        out.set(c, f, y, xx, x.at(c, start + f, y, xx));
                    }
                }
            }
        }
        out
    }

    fn h_concat_frames(a: &Vol, b: &Vol) -> Vol {
        let mut out = Vol::new(a.c, a.t + b.t, a.h, a.w);
        for (src, off) in [(a, 0), (b, a.t)] {
            for c in 0..src.c {
                for f in 0..src.t {
                    for y in 0..src.h {
                        for x in 0..src.w {
                            out.set(c, off + f, y, x, src.at(c, f, y, x));
                        }
                    }
                }
            }
        }
        out
    }

    fn h_concat_channels(a: &Vol, b: &Vol) -> Vol {
        let mut out = Vol::new(a.c + b.c, a.t, a.h, a.w);
        for (src, off) in [(a, 0), (b, a.c)] {
            for c in 0..src.c {
                for f in 0..src.t {
                    for y in 0..src.h {
                        for x in 0..src.w {
                            out.set(off + c, f, y, x, src.at(c, f, y, x));
                        }
                    }
                }
            }
        }
        out
    }

    fn h_enc_downsample(
        x: &Vol,
        prefix: &str,
        oc: usize,
        temporal: bool,
        map: &HashMap<String, Vec<f32>>,
    ) -> Vol {
        let factor = if temporal { 8 } else { 4 };
        let h = h_conv3d(
            x,
            at(map, &format!("{prefix}.conv.conv.weight")),
            at(map, &format!("{prefix}.conv.conv.bias")),
            oc / factor,
        );
        if !temporal {
            let h = h_rearrange_down(&h, 1);
            let sc = h_group_mean(&h_rearrange_down(x, 1), oc);
            return h_add(&h, &sc);
        }
        let h_first = h_concat_channels(
            &h_rearrange_down(&frame_slice(&h, 0, 1), 1),
            &h_rearrange_down(&frame_slice(&h, 0, 1), 1),
        );
        let h_next = h_rearrange_down(&frame_slice(&h, 1, h.t - 1), 2);
        let main = h_concat_frames(&h_first, &h_next);
        let x_first = h_group_mean(&h_rearrange_down(&frame_slice(x, 0, 1), 1), oc);
        let x_next = h_group_mean(&h_rearrange_down(&frame_slice(x, 1, x.t - 1), 2), oc);
        h_add(&main, &h_concat_frames(&x_first, &x_next))
    }

    fn h_dec_upsample(
        x: &Vol,
        prefix: &str,
        oc: usize,
        temporal: bool,
        map: &HashMap<String, Vec<f32>>,
    ) -> Vol {
        let factor = if temporal { 8 } else { 4 };
        let repeats = factor * oc / x.c;
        let h = h_conv3d(
            x,
            at(map, &format!("{prefix}.conv.conv.weight")),
            at(map, &format!("{prefix}.conv.conv.bias")),
            oc * factor,
        );
        if !temporal {
            let main = h_rearrange_up(&h, 1);
            let sc = h_rearrange_up(&h_repeat_interleave(x, repeats), 1);
            return h_add(&main, &sc);
        }
        let h_first = h_rearrange_up(&frame_slice(&h, 0, 1), 1);
        let mut first = Vol::new(oc, 1, h_first.h, h_first.w);
        for c in 0..oc {
            for y in 0..h_first.h {
                for xx in 0..h_first.w {
                    first.set(c, 0, y, xx, h_first.at(c, 0, y, xx));
                }
            }
        }
        let main = h_concat_frames(&first, &h_rearrange_up(&frame_slice(&h, 1, h.t - 1), 2));
        let x_first = h_repeat_interleave(&h_rearrange_up(&frame_slice(x, 0, 1), 1), repeats / 2);
        let x_next = h_repeat_interleave(&h_rearrange_up(&frame_slice(x, 1, x.t - 1), 2), repeats);
        h_add(&main, &h_concat_frames(&x_first, &x_next))
    }

    fn h_encode(pixels: &Vol, spec: &HunyuanVaeSpec, map: &HashMap<String, Vec<f32>>) -> Vol {
        let mut x = h_conv3d(
            pixels,
            at(map, "encoder.conv_in.conv.weight"),
            at(map, "encoder.conv_in.conv.bias"),
            spec.enc_blocks[0] as usize,
        );
        for i in 0..5 {
            for j in 0..2 {
                x = h_res(&x, &format!("encoder.down.{i}.block.{j}"), map);
            }
            if i < 4 {
                x = h_enc_downsample(
                    &x,
                    &format!("encoder.down.{i}.downsample"),
                    spec.enc_blocks[i + 1] as usize,
                    i >= 2,
                    map,
                );
            }
        }
        x = h_res(&x, "encoder.mid.block_1", map);
        x = h_attn(&x, "encoder.mid.attn_1", map);
        x = h_res(&x, "encoder.mid.block_2", map);
        let shortcut = h_group_mean(&x, 2 * spec.latent as usize);
        h_rms(&mut x, at(map, "encoder.norm_out.gamma"));
        h_silu(&mut x);
        let x = h_conv3d(
            &x,
            at(map, "encoder.conv_out.conv.weight"),
            at(map, "encoder.conv_out.conv.bias"),
            2 * spec.latent as usize,
        );
        h_add(&x, &shortcut)
    }

    fn h_decode(latents: &Vol, spec: &HunyuanVaeSpec, map: &HashMap<String, Vec<f32>>) -> Vol {
        let d0 = spec.dec_blocks[0] as usize;
        let conv = h_conv3d(
            latents,
            at(map, "decoder.conv_in.conv.weight"),
            at(map, "decoder.conv_in.conv.bias"),
            d0,
        );
        let mut x = h_add(
            &conv,
            &h_repeat_interleave(latents, d0 / spec.latent as usize),
        );
        x = h_res(&x, "decoder.mid.block_1", map);
        x = h_attn(&x, "decoder.mid.attn_1", map);
        x = h_res(&x, "decoder.mid.block_2", map);
        for i in 0..5 {
            for j in 0..3 {
                x = h_res(&x, &format!("decoder.up.{i}.block.{j}"), map);
            }
            if i < 4 {
                x = h_dec_upsample(
                    &x,
                    &format!("decoder.up.{i}.upsample"),
                    spec.dec_blocks[i + 1] as usize,
                    i < 2,
                    map,
                );
            }
        }
        h_rms(&mut x, at(map, "decoder.norm_out.gamma"));
        h_silu(&mut x);
        let x = h_conv3d(
            &x,
            at(map, "decoder.conv_out.conv.weight"),
            at(map, "decoder.conv_out.conv.bias"),
            3,
        );
        let mut out = x.clone();
        for v in out.d.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        out
    }

    fn compare(got: &[f32], want: &[f32], what: &str) {
        assert_eq!(got.len(), want.len(), "{what} length");
        let mut worst = 0f32;
        for (g, w) in got.iter().zip(want) {
            worst = worst.max((g - w).abs());
        }
        assert!(worst < 2e-3, "{what} max abs {worst}");
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-hy-vae-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn published_geometry_matches_the_config() {
        let spec = HunyuanVaeSpec::published();
        assert_eq!(spec.latent, 32);
        assert_eq!(spec.enc_blocks, [128, 256, 512, 1024, 1024]);
        assert_eq!(spec.dec_blocks, [1024, 1024, 512, 256, 128]);
        assert_eq!(HunyuanVaeSpec::encode_frames(81), 21);
        assert_eq!(HunyuanVaeSpec::decode_frames(21), 81);
        assert_eq!(SCALING, 1.03682);
    }

    #[test]
    fn infer_reads_the_native_layout() {
        let dir = temp_dir("infer");
        let spec = HunyuanVaeSpec::smoke();
        let mut owned = Vec::new();
        for (name, dims) in encoder_layout(&spec)
            .into_iter()
            .chain(decoder_layout(&spec))
        {
            let data: Vec<f32> = (0..dims.iter().product::<i64>() as usize)
                .map(|i| weight_at(&name, i))
                .collect();
            let shape: Vec<u64> = dims.iter().rev().map(|d| *d as u64).collect();
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            owned.push((name, shape, bytes));
        }
        let tensors: Vec<SafeTensor> = owned
            .iter()
            .map(|(name, shape, bytes)| SafeTensor {
                name,
                dtype: "F32",
                shape,
                data: bytes,
            })
            .collect();
        let path = dir.join("diffusion_pytorch_model.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        let weights = GgufWeights::open_parts(&[path]).unwrap();
        let inferred = HunyuanVaeSpec::infer(&weights).unwrap();
        assert_eq!(inferred, spec);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_encoder_matches_the_host_reference() {
        let spec = HunyuanVaeSpec::smoke();
        let params = params_for(&encoder_layout(&spec));
        let (w, h, t) = (32i64, 32i64, 5i64);
        let mut pixels = Vol::new(3, t as usize, h as usize, w as usize);
        for (i, v) in pixels.d.iter_mut().enumerate() {
            *v = tap("pixels", i, 0.25);
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = encode(&ctx, &params, &spec, &pixels.ggml(), w, h, t).unwrap();
        let moments = h_encode(&pixels, &spec, &map_of(&params));
        let latent = spec.latent as usize;
        let mut want = Vol::new(latent, moments.t, moments.h, moments.w);
        for c in 0..latent {
            for f in 0..moments.t {
                for y in 0..moments.h {
                    for x in 0..moments.w {
                        want.set(c, f, y, x, moments.at(c, f, y, x) * spec.scaling);
                    }
                }
            }
        }
        assert_eq!((moments.c, moments.t, moments.h, moments.w), (8, 2, 2, 2));
        compare(&got, &want.ggml(), "hunyuan vae encode");
    }

    #[test]
    fn smoke_decoder_matches_the_host_reference() {
        let spec = HunyuanVaeSpec::smoke();
        let params = params_for(&decoder_layout(&spec));
        let (w, h, t) = (2i64, 2i64, 2i64);
        let mut latents = Vol::new(spec.latent as usize, t as usize, h as usize, w as usize);
        for (i, v) in latents.d.iter_mut().enumerate() {
            *v = tap("latents", i, 0.3);
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = decode(&ctx, &params, &spec, &latents.ggml(), w, h, t).unwrap();
        let mut scaled = latents.clone();
        for v in scaled.d.iter_mut() {
            *v /= spec.scaling;
        }
        let want = h_decode(&scaled, &spec, &map_of(&params));
        assert_eq!((want.c, want.t, want.h, want.w), (3, 5, 32, 32));
        compare(&got, &want.ggml(), "hunyuan vae decode");
    }

    fn graph_unary(
        input: &[f32],
        dims: [i64; 4],
        f: impl FnOnce(&crate::runtime::GraphBuilder, crate::runtime::Tensor) -> crate::runtime::Tensor,
    ) -> Vec<f32> {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(8 << 20).unwrap();
        let x = b.f32_4d("x", dims[0], dims[1], dims[2], dims[3], true);
        let out = super::nn(f(&b, x)).unwrap();
        let out = super::nn(b.cont(out)).unwrap();
        let mut graph = b.finish(out).unwrap();
        graph.set_f32("x", input).unwrap();
        graph.compute().unwrap()
    }

    #[test]
    fn rearrange_up_matches_host() {
        for r1 in [1i64, 2] {
            let (w, h, f, c) = (2i64, 2i64, 2i64, 3i64);
            let packed = r1 * 4 * c;
            let input: Vec<f32> = (0..(w * h * f * packed) as usize)
                .map(|i| tap("rz", i, 0.3))
                .collect();
            let got = graph_unary(&input, [w, h, f, packed], |b, x| {
                super::rearrange_up(b, x, w, h, f, c, r1).unwrap()
            });
            let mut xv = Vol::new(packed as usize, f as usize, h as usize, w as usize);
            for cix in 0..packed as usize {
                for ff in 0..f as usize {
                    for yy in 0..h as usize {
                        for xx in 0..w as usize {
                            xv.set(
                                cix,
                                ff,
                                yy,
                                xx,
                                input
                                    [xx + w as usize * (yy + h as usize * (ff + f as usize * cix))],
                            );
                        }
                    }
                }
            }
            let want = h_rearrange_up(&xv, r1 as usize);
            compare(&got, &want.ggml(), "rearrange_up");
        }
    }

    #[test]
    fn repeat_interleave_matches_host() {
        let (w, h, f, c, r) = (2i64, 3i64, 2i64, 3i64, 4i64);
        let input: Vec<f32> = (0..(w * h * f * c) as usize)
            .map(|i| tap("ri", i, 0.3))
            .collect();
        let got = graph_unary(&input, [w, h, f, c], |b, x| {
            super::repeat_interleave(b, x, w, h, f, c, r).unwrap()
        });
        let mut xv = Vol::new(c as usize, f as usize, h as usize, w as usize);
        for cix in 0..c as usize {
            for ff in 0..f as usize {
                for yy in 0..h as usize {
                    for xx in 0..w as usize {
                        xv.set(
                            cix,
                            ff,
                            yy,
                            xx,
                            input[xx + w as usize * (yy + h as usize * (ff + f as usize * cix))],
                        );
                    }
                }
            }
        }
        let want = h_repeat_interleave(&xv, r as usize);
        compare(&got, &want.ggml(), "repeat_interleave");
    }

    #[test]
    fn encode_rejects_bad_geometry() {
        let spec = HunyuanVaeSpec::smoke();
        let params = params_for(&encoder_layout(&spec));
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let pixels = vec![0f32; 32 * 32 * 5 * 3];
        assert!(encode(&ctx, &params, &spec, &pixels, 24, 32, 5).is_err());
        assert!(encode(&ctx, &params, &spec, &pixels, 32, 32, 6).is_err());
        assert!(encode(&ctx, &params, &spec, &pixels[..12], 32, 32, 5).is_err());
    }
}
