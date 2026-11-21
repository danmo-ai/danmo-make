//! SeedVR2 3D VAE over `T >= 1` frames. CausalConv3d replicates frame 0
//! `kt - 1` times in front; each temporal downsample maps `T` to
//! `(T - 1) / 2 + 1`, each temporal upsample doubles `T` except that a single
//! latent frame stays a single frame. Weights are the published
//! `ema_vae_fp16.safetensors` (PyTorch layout); the flat bytes double as the
//! ggml `[KW, KH, KD, IC * OC]` kernel and `[IC, OC]` linear without a repack.

use crate::runtime::{GgmlContext, GraphBuilder, RuntimeError, Tensor, shape};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;

const GRAPH_MEM: usize = 256 << 20;
const NORM_EPS: f32 = 1e-6;

#[derive(Clone)]
pub struct VaeSpec {
    pub block_out: Vec<i64>,
    pub enc_resnets: usize,
    pub dec_resnets: usize,
    pub latent: i64,
    pub temporal_down: usize,
    pub temporal_up: usize,
    pub groups: i32,
    pub scaling: f32,
}

/// The published `ema_vae_fp16.safetensors` architecture (what `infer`
/// recovers): `block_out [128, 256, 512, 512]`, 2 encoder and 3 decoder
/// resnets per block, latent 16, 2 temporal downs and ups, 32 groups,
/// scaling 0.9152.
impl VaeSpec {
    pub fn smoke() -> Self {
        Self {
            block_out: vec![8, 16, 16, 32],
            enc_resnets: 1,
            dec_resnets: 1,
            latent: 16,
            temporal_down: 1,
            temporal_up: 1,
            groups: 8,
            scaling: 0.9152,
        }
    }

    /// Infer the architecture from the weight names and shapes. GroupNorm
    /// groups are not stored in the file: they are the largest value up to
    /// 32 that divides every norm channel count. `scaling` is the published
    /// constant. Anything unexpected fails loud.
    pub fn infer(w: &GgufWeights) -> Result<Self, WeightError> {
        let names: Vec<String> = w.names().map(str::to_string).collect();
        let missing = |what: String| WeightError::NotBuilt(format!("seedvr2 vae has no {what}"));
        let mut blocks = 0usize;
        while names
            .iter()
            .any(|n| n.starts_with(&format!("encoder.down_blocks.{blocks}.")))
        {
            blocks += 1;
        }
        if blocks == 0 {
            return Err(missing("encoder.down_blocks".into()));
        }
        let mut block_out = Vec::with_capacity(blocks);
        for i in 0..blocks {
            let shape = w.shape(&format!("encoder.down_blocks.{i}.resnets.0.conv1.bias"))?;
            block_out.push(shape[0]);
        }
        let count_resnets = |prefix: &str| -> usize {
            let mut n = 0usize;
            while names
                .iter()
                .any(|name| name == &format!("{prefix}.resnets.{n}.norm1.weight"))
            {
                n += 1;
            }
            n
        };
        let enc_resnets = count_resnets("encoder.down_blocks.0");
        let dec_resnets = count_resnets(&format!("decoder.up_blocks.{}", blocks - 1));
        if enc_resnets == 0 || dec_resnets == 0 {
            return Err(missing("resnets".into()));
        }
        let latent = w.shape("encoder.conv_out.bias")?[0] / 2;
        let temporal_down = (0..blocks.saturating_sub(1))
            .filter(|&i| {
                w.shape(&format!(
                    "encoder.down_blocks.{i}.downsamplers.0.conv.weight"
                ))
                .map(|s| s[2] == 3)
                .unwrap_or(false)
            })
            .count();
        let temporal_up = (0..blocks.saturating_sub(1))
            .filter(|&i| {
                let channel = block_out[blocks - 1 - i];
                w.shape(&format!(
                    "decoder.up_blocks.{i}.upsamplers.0.upscale_conv.bias"
                ))
                .map(|s| s[0] == channel * 8)
                .unwrap_or(false)
            })
            .count();
        let mut common: i64 = 0;
        for name in &names {
            let norm = name.ends_with(".bias")
                && (name.contains("norm1")
                    || name.contains("norm2")
                    || name.contains("conv_norm_out")
                    || name.contains("group_norm"));
            if norm {
                let n = w.shape(name)?[0];
                common = if common == 0 {
                    n
                } else {
                    let (mut a, mut b) = (common, n);
                    while b != 0 {
                        (a, b) = (b, a % b);
                    }
                    a
                };
            }
        }
        let mut groups = common.min(32);
        while groups > 1 && common % groups != 0 {
            groups -= 1;
        }
        if groups < 1 {
            return Err(missing("a valid group norm divisor".into()));
        }
        let spec = Self {
            block_out,
            enc_resnets,
            dec_resnets,
            latent,
            temporal_down,
            temporal_up,
            groups: groups as i32,
            scaling: 0.9152,
        };
        let conv_in = w.shape("decoder.conv_in.weight")?;
        if conv_in[3] != spec.latent * spec.block_out[blocks - 1] {
            return Err(WeightError::NotBuilt(format!(
                "seedvr2 vae decoder.conv_in expects {} input tensors but latent {} x channels {} disagree",
                conv_in[3],
                spec.latent,
                spec.block_out[blocks - 1]
            )));
        }
        Ok(spec)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Vector(i64),
    Linear {
        inp: i64,
        out: i64,
    },
    Conv3d {
        kw: i64,
        kh: i64,
        kd: i64,
        ic: i64,
        oc: i64,
    },
}

struct Param {
    name: String,
    kind: Kind,
}

impl Param {
    fn shape(&self) -> Vec<i64> {
        match self.kind {
            Kind::Vector(n) => vec![n],
            Kind::Linear { inp, out } => vec![inp, out],
            Kind::Conv3d { kw, kh, kd, ic, oc } => vec![kw, kh, kd, ic * oc],
        }
    }
}

pub struct LoadedVae {
    encoder: Vec<(String, Vec<f32>)>,
    decoder: Vec<(String, Vec<f32>)>,
}

pub fn load(vae: &GgufWeights, spec: &VaeSpec) -> Result<LoadedVae, WeightError> {
    let mut encoder = Vec::new();
    for param in encoder_params(spec) {
        let data = vae.f32_widened(&param.name, &param.shape())?;
        encoder.push((param.name, data));
    }
    let mut decoder = Vec::new();
    for param in decoder_params(spec) {
        let data = vae.f32_widened(&param.name, &param.shape())?;
        decoder.push((param.name, data));
    }
    Ok(LoadedVae { encoder, decoder })
}

fn conv3d(out: &mut Vec<Param>, prefix: &str, kd: i64, ic: i64, oc: i64) {
    out.push(Param {
        name: format!("{prefix}.weight"),
        kind: Kind::Conv3d {
            kw: 3,
            kh: 3,
            kd,
            ic,
            oc,
        },
    });
    out.push(Param {
        name: format!("{prefix}.bias"),
        kind: Kind::Vector(oc),
    });
}

fn conv1x1(out: &mut Vec<Param>, prefix: &str, ic: i64, oc: i64) {
    out.push(Param {
        name: format!("{prefix}.weight"),
        kind: Kind::Conv3d {
            kw: 1,
            kh: 1,
            kd: 1,
            ic,
            oc,
        },
    });
    out.push(Param {
        name: format!("{prefix}.bias"),
        kind: Kind::Vector(oc),
    });
}

fn norm(out: &mut Vec<Param>, prefix: &str, channels: i64) {
    out.push(Param {
        name: format!("{prefix}.weight"),
        kind: Kind::Vector(channels),
    });
    out.push(Param {
        name: format!("{prefix}.bias"),
        kind: Kind::Vector(channels),
    });
}

fn resnet(out: &mut Vec<Param>, prefix: &str, in_ch: i64, out_ch: i64) {
    norm(out, &format!("{prefix}.norm1"), in_ch);
    conv3d(out, &format!("{prefix}.conv1"), 3, in_ch, out_ch);
    norm(out, &format!("{prefix}.norm2"), out_ch);
    conv3d(out, &format!("{prefix}.conv2"), 3, out_ch, out_ch);
    if in_ch != out_ch {
        conv1x1(out, &format!("{prefix}.conv_shortcut"), in_ch, out_ch);
    }
}

fn attention(out: &mut Vec<Param>, prefix: &str, channels: i64) {
    norm(out, &format!("{prefix}.group_norm"), channels);
    for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
        out.push(Param {
            name: format!("{prefix}.{proj}.weight"),
            kind: Kind::Linear {
                inp: channels,
                out: channels,
            },
        });
        out.push(Param {
            name: format!("{prefix}.{proj}.bias"),
            kind: Kind::Vector(channels),
        });
    }
}

fn mid_block(out: &mut Vec<Param>, prefix: &str, channels: i64) {
    resnet(out, &format!("{prefix}.resnets.0"), channels, channels);
    attention(out, &format!("{prefix}.attentions.0"), channels);
    resnet(out, &format!("{prefix}.resnets.1"), channels, channels);
}

fn encoder_params(spec: &VaeSpec) -> Vec<Param> {
    let mut out = Vec::new();
    let blocks = spec.block_out.len();
    conv3d(&mut out, "encoder.conv_in", 3, 3, spec.block_out[0]);
    let mut input = spec.block_out[0];
    for (i, &channel) in spec.block_out.iter().enumerate() {
        for j in 0..spec.enc_resnets {
            let in_ch = if j == 0 { input } else { channel };
            resnet(
                &mut out,
                &format!("encoder.down_blocks.{i}.resnets.{j}"),
                in_ch,
                channel,
            );
        }
        input = channel;
        if i + 1 != blocks {
            let temporal = i >= blocks - spec.temporal_down - 1;
            let kd = if temporal { 3 } else { 1 };
            conv3d(
                &mut out,
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                kd,
                channel,
                channel,
            );
        }
    }
    mid_block(&mut out, "encoder.mid_block", spec.block_out[blocks - 1]);
    norm(
        &mut out,
        "encoder.conv_norm_out",
        spec.block_out[blocks - 1],
    );
    conv3d(
        &mut out,
        "encoder.conv_out",
        3,
        spec.block_out[blocks - 1],
        2 * spec.latent,
    );
    out
}

fn decoder_params(spec: &VaeSpec) -> Vec<Param> {
    let mut out = Vec::new();
    let blocks = spec.block_out.len();
    let reversed: Vec<i64> = spec.block_out.iter().rev().copied().collect();
    conv3d(&mut out, "decoder.conv_in", 3, spec.latent, reversed[0]);
    mid_block(&mut out, "decoder.mid_block", reversed[0]);
    let mut input = reversed[0];
    for (i, &channel) in reversed.iter().enumerate() {
        for j in 0..spec.dec_resnets {
            let in_ch = if j == 0 { input } else { channel };
            resnet(
                &mut out,
                &format!("decoder.up_blocks.{i}.resnets.{j}"),
                in_ch,
                channel,
            );
        }
        input = channel;
        if i + 1 != blocks {
            let tf = if i < spec.temporal_up { 2 } else { 1 };
            conv1x1(
                &mut out,
                &format!("decoder.up_blocks.{i}.upsamplers.0.upscale_conv"),
                channel,
                channel * 4 * tf,
            );
            conv3d(
                &mut out,
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                3,
                channel,
                channel,
            );
        }
    }
    norm(&mut out, "decoder.conv_norm_out", reversed[blocks - 1]);
    conv3d(&mut out, "decoder.conv_out", 3, reversed[blocks - 1], 3);
    out
}

fn declare(b: &mut GraphBuilder, params: &[Param]) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    for param in params {
        let tensor = match param.kind {
            Kind::Vector(n) => b.f32_1d(&param.name, n, true),
            Kind::Linear { inp, out } => b.f32_2d(&param.name, inp, out, true),
            Kind::Conv3d { kw, kh, kd, ic, oc } => b.f32_4d(&param.name, kw, kh, kd, ic * oc, true),
        };
        map.insert(param.name.clone(), tensor);
    }
    map
}

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
        .ok_or_else(|| RuntimeError::Message(format!("seedvr2 vae graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let ne_tensor = shape(tensor);
    if ne_tensor != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {ne_tensor:?}, the seedvr2 vae expects {ne:?}"
        )));
    }
    Ok(tensor)
}

/// Replicate frame 0 `frames` times in front along the depth axis (ne2).
fn causal_pad(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    d: i64,
    c: i64,
    frames: i64,
) -> Result<Tensor, RuntimeError> {
    if frames == 0 {
        return Ok(x);
    }
    let frame0 = nn(b.view_4d(
        x,
        w,
        h,
        1,
        c,
        (w * 4) as usize,
        (w * h * 4) as usize,
        (w * h * d * 4) as usize,
        0,
    ))?;
    let pad = nn(b.repeat_4d(frame0, w, h, frames, c))?;
    nn(b.concat(pad, x, 2))
}

/// `w`/`h`/`d` are the input spatial size. The output size comes from the
/// kernel shape, `pad`, and the strides.
#[allow(clippy::too_many_arguments)]
fn apply_conv3d(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    kd: i64,
    stride: i64,
    stride_d: i64,
    pad: i32,
    oc: i64,
    w: i64,
    h: i64,
    d: i64,
    ic: i64,
) -> Result<Tensor, RuntimeError> {
    let x = if kd > 1 {
        nn(b.cont(causal_pad(b, x, w, h, d, ic, kd - 1)?))?
    } else {
        x
    };
    let kernel = wt(weights, &format!("{prefix}.weight"))?;
    let kernel_ne = shape(kernel);
    let y = nn(b.conv3d_direct(
        kernel,
        x,
        stride as i32,
        stride as i32,
        stride_d as i32,
        pad,
        pad,
        0,
        1,
        1,
        1,
        ic as i32,
        1,
        oc as i32,
    ))?;
    let w_out = (w + 2 * i64::from(pad) - kernel_ne[0]) / stride + 1;
    let h_out = (h + 2 * i64::from(pad) - kernel_ne[1]) / stride + 1;
    let d_out = if kd > 1 { (d - 1) / stride_d + 1 } else { d };
    let y = nn(b.cont(y))?;
    let y = shaped(y, [w_out, h_out, d_out, oc], &format!("{prefix}.weight"))?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.bias"))?, 1, 1, 1, oc)))?;
    let bias = nn(b.repeat(bias, y))?;
    nn(b.add(y, bias))
}

fn apply_norm(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    groups: i32,
    channels: i64,
    w: i64,
    h: i64,
    d: i64,
) -> Result<Tensor, RuntimeError> {
    // ggml group_norm groups along ne2; move channels there, frames to batch.
    let y = nn(b.cont(b.permute(x, 0, 1, 3, 2)))?;
    let y = nn(b.group_norm(y, groups, NORM_EPS))?;
    let scale =
        nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.weight"))?, 1, 1, channels, 1)))?;
    let y = nn(b.mul(y, nn(b.repeat(scale, y))?))?;
    let bias =
        nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.bias"))?, 1, 1, channels, 1)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    let y = nn(b.cont(b.permute(y, 0, 1, 3, 2)))?;
    shaped(y, [w, h, d, channels], &format!("{prefix}.weight"))
}

#[allow(clippy::too_many_arguments)]
fn apply_resnet(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    groups: i32,
    in_ch: i64,
    out_ch: i64,
    w: i64,
    h: i64,
    d: i64,
) -> Result<Tensor, RuntimeError> {
    let hidden = apply_norm(
        b,
        weights,
        x,
        &format!("{prefix}.norm1"),
        groups,
        in_ch,
        w,
        h,
        d,
    )?;
    let hidden = nn(b.silu(hidden))?;
    let hidden = apply_conv3d(
        b,
        weights,
        hidden,
        &format!("{prefix}.conv1"),
        3,
        1,
        1,
        1,
        out_ch,
        w,
        h,
        d,
        in_ch,
    )?;
    let hidden = apply_norm(
        b,
        weights,
        hidden,
        &format!("{prefix}.norm2"),
        groups,
        out_ch,
        w,
        h,
        d,
    )?;
    let hidden = nn(b.silu(hidden))?;
    let hidden = apply_conv3d(
        b,
        weights,
        hidden,
        &format!("{prefix}.conv2"),
        3,
        1,
        1,
        1,
        out_ch,
        w,
        h,
        d,
        out_ch,
    )?;
    let skip = if in_ch != out_ch {
        apply_conv3d(
            b,
            weights,
            x,
            &format!("{prefix}.conv_shortcut"),
            1,
            1,
            1,
            0,
            out_ch,
            w,
            h,
            d,
            in_ch,
        )?
    } else {
        x
    };
    let y = nn(b.add(skip, hidden))?;
    shaped(y, [w, h, d, out_ch], &format!("{prefix}.conv2.weight"))
}

fn linear(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    channels: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.mul_mat(wt(weights, &format!("{prefix}.weight"))?, x))?;
    let bias = nn(b.cont(b.reshape_2d(wt(weights, &format!("{prefix}.bias"))?, channels, 1)))?;
    let bias = nn(b.repeat(bias, y))?;
    nn(b.add(y, bias))
}

#[allow(clippy::too_many_arguments)]
fn apply_attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    groups: i32,
    channels: i64,
    w: i64,
    h: i64,
) -> Result<Tensor, RuntimeError> {
    let hidden = apply_norm(
        b,
        weights,
        x,
        &format!("{prefix}.group_norm"),
        groups,
        channels,
        w,
        h,
        1,
    )?;
    // [W, H, 1, C] to [C, W*H] tokens (channel fastest).
    let tokens = {
        let p = nn(b.cont(b.permute(hidden, 1, 2, 3, 0)))?;
        let p = nn(b.reshape_2d(p, channels, w * h))?;
        shaped(p, [channels, w * h, 1, 1], "seedvr2 vae attention tokens")?
    };
    let q = linear(b, weights, tokens, &format!("{prefix}.to_q"), channels)?;
    let k = linear(b, weights, tokens, &format!("{prefix}.to_k"), channels)?;
    let v = linear(b, weights, tokens, &format!("{prefix}.to_v"), channels)?;
    let scores = nn(b.mul_mat(k, q))?;
    let scores = nn(b.scale(scores, (channels as f32).sqrt().recip()))?;
    let probs = nn(b.soft_max(scores))?;
    // out[t, c] = sum_i probs[i, t] * v[c, i]: contract the key axis by
    // multiplying probs [keys, queries] with transposed v [W*H, C].
    let v_t = nn(b.cont(b.permute(v, 1, 0, 2, 3)))?;
    let mixed = nn(b.mul_mat(probs, v_t))?;
    // [W*H, C] to [C, W*H] for the output projection.
    let mixed = nn(b.cont(b.permute(mixed, 1, 0, 2, 3)))?;
    let mixed = linear(b, weights, mixed, &format!("{prefix}.to_out.0"), channels)?;
    // [C, W*H] back to [W, H, 1, C].
    let mixed = nn(b.reshape_4d(nn(b.cont(mixed))?, channels, w, h, 1))?;
    let mixed = nn(b.cont(b.permute(mixed, 3, 0, 1, 2)))?;
    let mixed = shaped(mixed, [w, h, 1, channels], "seedvr2 vae attention out")?;
    nn(b.add(x, mixed))
}

/// Spatial attention is per frame: slice each `[W, H, 1, C]` frame, attend,
/// and concat the results along the depth axis.
#[allow(clippy::too_many_arguments)]
fn apply_attention_frames(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    groups: i32,
    channels: i64,
    w: i64,
    h: i64,
    d: i64,
) -> Result<Tensor, RuntimeError> {
    if d == 1 {
        return apply_attention(b, weights, x, prefix, groups, channels, w, h);
    }
    let mut frames = Vec::with_capacity(d as usize);
    for f in 0..d {
        let frame = nn(b.cont(b.view_4d(
            x,
            w,
            h,
            1,
            channels,
            (w * 4) as usize,
            (w * h * 4) as usize,
            (w * h * d * 4) as usize,
            (f * w * h * 4) as usize,
        )))?;
        frames.push(apply_attention(
            b, weights, frame, prefix, groups, channels, w, h,
        )?);
    }
    let mut it = frames.into_iter();
    let first = it.next().expect("at least one frame");
    let mut out = first;
    for frame in it {
        out = nn(b.concat(out, frame, 2))?;
    }
    shaped(out, [w, h, d, channels], "seedvr2 vae attention frames")
}

/// Pixel shuffle: `upscale_conv` output channel `(((i*2 + j)*tf) + t)*C + c`
/// of input frame `f` lands at `(w*2 + j, h*2 + i)` of output frame
/// `f*tf + t`. A single-frame input keeps one output frame. `ggml_set`
/// scatters each plane with channels contiguous, then the assembled
/// `[C, W2, H2, F]` tensor is transposed back to `[W2, H2, F, C]`.
fn pixel_shuffle(
    b: &GraphBuilder,
    x: Tensor,
    channels: i64,
    tf: i64,
    w: i64,
    h: i64,
    d: i64,
) -> Result<Tensor, RuntimeError> {
    let w2 = w * 2;
    let h2 = h * 2;
    let frames_out = if d == 1 { 1 } else { d * tf };
    // A single input frame keeps one output frame: only the t = 0 slice of
    // the sub-pixel frame axis is used, matching the reference.
    let t_range = if d == 1 { 1 } else { tf };
    let mut out = b.blank_f32_4d(channels, w2, h2, frames_out);
    for f in 0..d {
        for t in 0..t_range {
            let frame_out = if d == 1 { 0 } else { f * tf + t };
            for i in 0..2 {
                for j in 0..2 {
                    let s = ((i * 2 + j) * tf + t) * channels;
                    let plane = nn(b.view_4d(
                        x,
                        w,
                        h,
                        channels,
                        1,
                        (w * 4) as usize,
                        (w * h * d * 4) as usize,
                        (w * h * d * 4) as usize,
                        ((s * (w * h * d) + f * (w * h)) * 4) as usize,
                    ))?;
                    // [W, H, C] to [C, W, H] so the scatter's dim 0 is contiguous.
                    let plane_t = nn(b.cont(b.permute(plane, 1, 2, 0, 3)))?;
                    out = nn(b.set(
                        out,
                        plane_t,
                        (channels * 2 * 4) as usize,
                        (channels * w2 * 2 * 4) as usize,
                        (channels * w2 * h2 * 4) as usize,
                        ((j + i * w2) * channels * 4 + frame_out * channels * w2 * h2 * 4) as usize,
                    ))?;
                }
            }
        }
    }
    // [C, W2, H2, F] to [W2, H2, F, C].
    let out = nn(b.cont(b.permute(out, 3, 0, 1, 2)))?;
    shaped(
        out,
        [w2, h2, frames_out, channels],
        "seedvr2 vae pixel shuffle",
    )
}

/// Latent frame count after the causal temporal downsamples of `encode_tile`.
pub fn encode_frames_out(spec: &VaeSpec, frames: i64) -> i64 {
    let blocks = spec.block_out.len();
    let mut d = frames;
    for i in 0..blocks.saturating_sub(1) {
        if i >= blocks - spec.temporal_down - 1 {
            d = (d - 1) / 2 + 1;
        }
    }
    d
}

/// Output frame count after the temporal upsamples of `decode_tile`.
/// `D*2` per upsample; a single latent frame stays a single frame.
pub fn decode_frames_out(spec: &VaeSpec, frames: i64) -> i64 {
    let blocks = spec.block_out.len();
    let mut d = frames;
    for i in 0..blocks.saturating_sub(1) {
        if i < spec.temporal_up {
            d = if d == 1 { 1 } else { d * 2 };
        }
    }
    d
}

/// Encode one tile. `image` is `[W, H, F, 3]` (w fastest, then h, frame,
/// channel) in `[-1, 1]`. Returns the scaled latent `[W/8, H/8, D, latent]`
/// where `D` follows the causal temporal downsamples.
pub fn encode_tile(
    ctx: &GgmlContext,
    loaded: &[(String, Vec<f32>)],
    spec: &VaeSpec,
    image: &[f32],
    w: i64,
    h: i64,
    frames: i64,
) -> Result<Vec<f32>, WeightError> {
    if frames < 1 {
        return Err(WeightError::Shape {
            name: "seedvr2 vae encode frames".into(),
            expected: vec![1],
            got: vec![frames],
        });
    }
    if image.len() != (w * h * frames * 3) as usize {
        return Err(WeightError::Shape {
            name: "seedvr2 vae encode tile".into(),
            expected: vec![w, h, frames, 3],
            got: vec![image.len() as i64],
        });
    }
    let params = encoder_params(spec);
    let mut builder = ctx.builder(GRAPH_MEM)?;
    let declared = declare(&mut builder, &params);
    let x = builder.f32_4d("seedvr2_vae_image", w, h, frames, 3, true);
    let blocks = spec.block_out.len();
    let mut x = apply_conv3d(
        &builder,
        &declared,
        x,
        "encoder.conv_in",
        3,
        1,
        1,
        1,
        spec.block_out[0],
        w,
        h,
        frames,
        3,
    )?;
    let mut cw = w;
    let mut ch = h;
    let mut cd = frames;
    let mut input = spec.block_out[0];
    for (i, &channel) in spec.block_out.iter().enumerate() {
        for j in 0..spec.enc_resnets {
            let in_ch = if j == 0 { input } else { channel };
            x = apply_resnet(
                &builder,
                &declared,
                x,
                &format!("encoder.down_blocks.{i}.resnets.{j}"),
                spec.groups,
                in_ch,
                channel,
                cw,
                ch,
                cd,
            )?;
        }
        input = channel;
        if i + 1 != blocks {
            let temporal = i >= blocks - spec.temporal_down - 1;
            let (kd, stride_d) = if temporal { (3, 2) } else { (1, 1) };
            let padded = nn(builder.pad_ext(x, 0, 1, 0, 1, 0, 0, 0, 0))?;
            x = apply_conv3d(
                &builder,
                &declared,
                padded,
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                kd,
                2,
                stride_d,
                0,
                channel,
                cw + 1,
                ch + 1,
                cd,
                channel,
            )?;
            cw /= 2;
            ch /= 2;
            if temporal {
                cd = (cd - 1) / 2 + 1;
            }
        }
    }
    x = apply_resnet(
        &builder,
        &declared,
        x,
        "encoder.mid_block.resnets.0",
        spec.groups,
        input,
        input,
        cw,
        ch,
        cd,
    )?;
    x = apply_attention_frames(
        &builder,
        &declared,
        x,
        "encoder.mid_block.attentions.0",
        spec.groups,
        input,
        cw,
        ch,
        cd,
    )?;
    x = apply_resnet(
        &builder,
        &declared,
        x,
        "encoder.mid_block.resnets.1",
        spec.groups,
        input,
        input,
        cw,
        ch,
        cd,
    )?;
    x = apply_norm(
        &builder,
        &declared,
        x,
        "encoder.conv_norm_out",
        spec.groups,
        input,
        cw,
        ch,
        cd,
    )?;
    x = nn(builder.silu(x))?;
    x = apply_conv3d(
        &builder,
        &declared,
        x,
        "encoder.conv_out",
        3,
        1,
        1,
        1,
        2 * spec.latent,
        cw,
        ch,
        cd,
        input,
    )?;
    // Keep the mean half and scale it.
    let mean = nn(builder.view_4d(
        x,
        cw,
        ch,
        cd,
        spec.latent,
        (cw * 4) as usize,
        (cw * ch * 4) as usize,
        (cw * ch * cd * 4) as usize,
        0,
    ))?;
    let latent = nn(builder.scale(mean, spec.scaling))?;
    let output = shaped(latent, [cw, ch, cd, spec.latent], "seedvr2 vae encode out")?;
    let mut graph = builder.finish(output)?;
    graph.set_f32("seedvr2_vae_image", image)?;
    for (name, data) in loaded {
        graph.set_f32(name, data)?;
    }
    let out = graph.compute()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err(WeightError::NotBuilt(
            "seedvr2 vae encode produced a non-finite latent".into(),
        ));
    }
    Ok(out)
}

/// Decode one tile. `latent` is `[W, H, D, latent]` (w fastest, then h,
/// frame, channel). Returns the `[W*8, H*8, F, 3]` image in `[-1, 1]` where
/// `F` follows the temporal upsamples (`D*2` per upsample, `1` stays `1`).
pub fn decode_tile(
    ctx: &GgmlContext,
    loaded: &[(String, Vec<f32>)],
    spec: &VaeSpec,
    latent: &[f32],
    w: i64,
    h: i64,
    frames: i64,
) -> Result<Vec<f32>, WeightError> {
    if frames < 1 {
        return Err(WeightError::Shape {
            name: "seedvr2 vae decode frames".into(),
            expected: vec![1],
            got: vec![frames],
        });
    }
    if latent.len() != (w * h * frames * spec.latent) as usize {
        return Err(WeightError::Shape {
            name: "seedvr2 vae decode tile".into(),
            expected: vec![w, h, frames, spec.latent],
            got: vec![latent.len() as i64],
        });
    }
    let params = decoder_params(spec);
    let mut builder = ctx.builder(GRAPH_MEM)?;
    let declared = declare(&mut builder, &params);
    let z = builder.f32_4d("seedvr2_vae_latent", w, h, frames, spec.latent, true);
    let z = nn(builder.scale(z, spec.scaling.recip()))?;
    let blocks = spec.block_out.len();
    let reversed: Vec<i64> = spec.block_out.iter().rev().copied().collect();
    let mut x = apply_conv3d(
        &builder,
        &declared,
        z,
        "decoder.conv_in",
        3,
        1,
        1,
        1,
        reversed[0],
        w,
        h,
        frames,
        spec.latent,
    )?;
    let mut cw = w;
    let mut ch = h;
    let mut cd = frames;
    let mut input = reversed[0];
    x = apply_resnet(
        &builder,
        &declared,
        x,
        "decoder.mid_block.resnets.0",
        spec.groups,
        input,
        input,
        cw,
        ch,
        cd,
    )?;
    x = apply_attention_frames(
        &builder,
        &declared,
        x,
        "decoder.mid_block.attentions.0",
        spec.groups,
        input,
        cw,
        ch,
        cd,
    )?;
    x = apply_resnet(
        &builder,
        &declared,
        x,
        "decoder.mid_block.resnets.1",
        spec.groups,
        input,
        input,
        cw,
        ch,
        cd,
    )?;
    for (i, &channel) in reversed.iter().enumerate() {
        for j in 0..spec.dec_resnets {
            let in_ch = if j == 0 { input } else { channel };
            x = apply_resnet(
                &builder,
                &declared,
                x,
                &format!("decoder.up_blocks.{i}.resnets.{j}"),
                spec.groups,
                in_ch,
                channel,
                cw,
                ch,
                cd,
            )?;
        }
        input = channel;
        if i + 1 != blocks {
            let tf = if i < spec.temporal_up { 2 } else { 1 };
            x = apply_conv3d(
                &builder,
                &declared,
                x,
                &format!("decoder.up_blocks.{i}.upsamplers.0.upscale_conv"),
                1,
                1,
                1,
                0,
                channel * 4 * tf,
                cw,
                ch,
                cd,
                channel,
            )?;
            x = pixel_shuffle(&builder, x, channel, tf, cw, ch, cd)?;
            cw *= 2;
            ch *= 2;
            cd = if cd == 1 { 1 } else { cd * tf };
            x = apply_conv3d(
                &builder,
                &declared,
                x,
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                3,
                1,
                1,
                1,
                channel,
                cw,
                ch,
                cd,
                channel,
            )?;
        }
    }
    x = apply_norm(
        &builder,
        &declared,
        x,
        "decoder.conv_norm_out",
        spec.groups,
        input,
        cw,
        ch,
        cd,
    )?;
    x = nn(builder.silu(x))?;
    x = apply_conv3d(
        &builder,
        &declared,
        x,
        "decoder.conv_out",
        3,
        1,
        1,
        1,
        3,
        cw,
        ch,
        cd,
        input,
    )?;
    let output = shaped(x, [cw, ch, cd, 3], "seedvr2 vae decode out")?;
    let mut graph = builder.finish(output)?;
    graph.set_f32("seedvr2_vae_latent", latent)?;
    for (name, data) in loaded {
        graph.set_f32(name, data)?;
    }
    let out = graph.compute()?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err(WeightError::NotBuilt(
            "seedvr2 vae decode produced a non-finite sample".into(),
        ));
    }
    Ok(out)
}

pub fn encoder_loaded(loaded: &LoadedVae) -> &[(String, Vec<f32>)] {
    &loaded.encoder
}

pub fn decoder_loaded(loaded: &LoadedVae) -> &[(String, Vec<f32>)] {
    &loaded.decoder
}

/// Write a deterministic smoke safetensors VAE for `VaeSpec::smoke()`.
/// Convs map the ggml `[KW, KH, KD, IC*OC]` back to PyTorch
/// `[OC, IC, KD, KH, KW]`; both flatten identically.
pub fn write_smoke_vae(dir: &std::path::Path) -> Result<std::path::PathBuf, WeightError> {
    use crate::weights::{SafeTensor, write_safetensors};
    let spec = VaeSpec::smoke();
    let mut names: Vec<String> = Vec::new();
    let mut shapes: Vec<Vec<u64>> = Vec::new();
    let mut values: Vec<Vec<u8>> = Vec::new();
    let mut seed = 1.0f32;
    for param in encoder_params(&spec)
        .into_iter()
        .chain(decoder_params(&spec))
    {
        let shape: Vec<u64> = match param.kind {
            Kind::Vector(n) => vec![n as u64],
            Kind::Linear { inp, out } => vec![out as u64, inp as u64],
            Kind::Conv3d { kw, kh, kd, ic, oc } => {
                vec![oc as u64, ic as u64, kd as u64, kh as u64, kw as u64]
            }
        };
        let n: usize = shape.iter().product::<u64>() as usize;
        let data: Vec<u8> = (0..n)
            .map(|i| (((i as f32 + seed) * 0.37).sin()) * 0.05)
            .flat_map(|v: f32| v.to_le_bytes())
            .collect();
        names.push(param.name);
        shapes.push(shape);
        values.push(data);
        seed += 1.0;
    }
    let tensors: Vec<SafeTensor> = names
        .iter()
        .zip(&shapes)
        .zip(&values)
        .map(|((name, shape), data)| SafeTensor {
            name: name.as_str(),
            dtype: "F32",
            shape: shape.as_slice(),
            data: data.as_slice(),
        })
        .collect();
    let path = dir.join("ema_vae_fp16.safetensors");
    write_safetensors(&path, &tensors)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;
    use crate::weights::{SafeTensor, write_safetensors};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn spec() -> VaeSpec {
        VaeSpec {
            block_out: vec![8, 16],
            enc_resnets: 1,
            dec_resnets: 1,
            latent: 2,
            temporal_down: 1,
            temporal_up: 1,
            groups: 8,
            scaling: 0.5,
        }
    }

    fn fill(
        name: &str,
        shape: &[u64],
        seed: f32,
        out: &mut Vec<SafeTensor<'static>>,
        store: &mut Vec<(String, Vec<f32>)>,
    ) {
        let n: usize = shape.iter().product::<u64>() as usize;
        let scale = 0.05;
        let data: Vec<f32> = (0..n)
            .map(|i| (((i as f32 + seed) * 0.37).sin()) * scale)
            .collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let name_static: &'static str = Box::leak(name.to_string().into_boxed_str());
        let shape_static: &'static [u64] = Box::leak(shape.to_vec().into_boxed_slice());
        let bytes_static: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        out.push(SafeTensor {
            name: name_static,
            dtype: "F32",
            shape: shape_static,
            data: bytes_static,
        });
        store.push((name.to_string(), data));
    }

    /// Write a smoke safetensors VAE from the graph parameter lists. Convs
    /// map `[KW, KH, KD, IC*OC]` back to PyTorch `[OC, IC, KD, KH, KW]`; both
    /// flatten identically, so the declared shape order is only relabeled.
    fn write_smoke(
        dir: &PathBuf,
    ) -> (
        std::sync::Arc<GgufWeights>,
        Vec<(String, Vec<f32>)>,
        Vec<(String, Vec<f32>)>,
    ) {
        let spec = spec();
        let mut tensors: Vec<SafeTensor<'static>> = Vec::new();
        let mut store: Vec<(String, Vec<f32>)> = Vec::new();
        let mut seed = 1.0f32;
        for param in encoder_params(&spec)
            .into_iter()
            .chain(decoder_params(&spec))
        {
            let shape: Vec<u64> = match param.kind {
                Kind::Vector(n) => vec![n as u64],
                Kind::Linear { inp, out } => vec![out as u64, inp as u64],
                Kind::Conv3d { kw, kh, kd, ic, oc } => {
                    vec![oc as u64, ic as u64, kd as u64, kh as u64, kw as u64]
                }
            };
            fill(&param.name, &shape, seed, &mut tensors, &mut store);
            seed += 1.0;
        }
        let path = dir.join("ema_vae_fp16.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        let vae = GgufWeights::open(&path).unwrap();
        let loaded = load(&vae, &spec).unwrap();
        (vae, loaded.encoder, loaded.decoder)
    }

    fn at<'a>(map: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a Vec<f32> {
        map.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    // ----- host reference -----

    fn h_conv3d(
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        ic: usize,
        wt: &[f32],
        bias: &[f32],
        kd: usize,
        stride_d: usize,
        stride_s: usize,
        pad: usize,
        oc: usize,
    ) -> (Vec<f32>, usize, usize, usize) {
        // causal temporal pad: replicate frame 0 kd-1 times in front
        let (mut x, mut d2) = (x.to_vec(), d);
        if kd > 1 {
            let mut padded = vec![0f32; w * h * (d + kd - 1) * ic];
            for c in 0..ic {
                for f in 0..(kd - 1) {
                    padded[c * w * h * (d + kd - 1) + f * w * h
                        ..c * w * h * (d + kd - 1) + (f + 1) * w * h]
                        .copy_from_slice(&x[c * w * h * d..c * w * h * d + w * h]);
                }
                padded[c * w * h * (d + kd - 1) + (kd - 1) * w * h
                    ..c * w * h * (d + kd - 1) + (kd - 1 + d) * w * h]
                    .copy_from_slice(&x[c * w * h * d..(c + 1) * w * h * d]);
            }
            x = padded;
            d2 = d + kd - 1;
        }
        let d_out = (d2 - kd) / stride_d + 1;
        let kw_eff = if wt.len() == oc * ic * kd * 9 { 3 } else { 1 };
        let kh_eff = kw_eff;
        let w_out = (w + 2 * pad - kw_eff) / stride_s + 1;
        let h_out = (h + 2 * pad - kh_eff) / stride_s + 1;
        let mut out = vec![0f32; w_out * h_out * d_out * oc];
        for o in 0..oc {
            for z in 0..d_out {
                for yy in 0..h_out {
                    for xx in 0..w_out {
                        let mut acc = bias[o];
                        for i in 0..ic {
                            for zd in 0..kd {
                                for ky in 0..kh_eff {
                                    for kx in 0..kw_eff {
                                        let sx = (xx * stride_s) as i64 + kx as i64 - pad as i64;
                                        let sy = (yy * stride_s) as i64 + ky as i64 - pad as i64;
                                        let sz = z * stride_d + zd;
                                        if sx < 0
                                            || sy < 0
                                            || sx >= w as i64
                                            || sy >= h as i64
                                            || sz >= d2
                                        {
                                            continue;
                                        }
                                        let xv = x
                                            [(i * d2 + sz) * w * h + sy as usize * w + sx as usize];
                                        let wv = wt[(((o * ic + i) * kd + zd) * kh_eff + ky)
                                            * kw_eff
                                            + kx];
                                        acc += xv * wv;
                                    }
                                }
                            }
                        }
                        out[(o * d_out + z) * w_out * h_out + yy * w_out + xx] = acc;
                    }
                }
            }
        }
        (out, w_out, h_out, d_out)
    }

    fn h_group_norm(
        x: &mut [f32],
        w: usize,
        h: usize,
        d: usize,
        c: usize,
        groups: usize,
        weight: &[f32],
        bias: &[f32],
    ) {
        let per = c / groups;
        let n = w * h * d;
        for f in 0..d {
            for g in 0..groups {
                let start = g * per;
                let mut mean = 0f32;
                for c in start..start + per {
                    for p in 0..w * h {
                        mean += x[c * n + f * w * h + p];
                    }
                }
                mean /= (per * w * h) as f32;
                let mut var = 0f32;
                for c in start..start + per {
                    for p in 0..w * h {
                        let v = x[c * n + f * w * h + p] - mean;
                        var += v * v;
                    }
                }
                var /= (per * w * h) as f32;
                let inv = (var + NORM_EPS).sqrt().recip();
                for c in start..start + per {
                    for p in 0..w * h {
                        let idx = c * n + f * w * h + p;
                        x[idx] = (x[idx] - mean) * inv * weight[c] + bias[c];
                    }
                }
            }
        }
    }

    fn h_silu(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    fn h_linear(x: &[f32], tokens: usize, c: usize, wt: &[f32], bias: &[f32]) -> Vec<f32> {
        // x [C, tokens] channel fastest; wt PyTorch [out, in]
        let mut out = vec![0f32; c * tokens];
        for t in 0..tokens {
            for o in 0..c {
                let mut acc = bias[o];
                for i in 0..c {
                    acc += wt[o * c + i] * x[i * tokens + t];
                }
                out[o * tokens + t] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn h_attention(
        x: &[f32],
        w: usize,
        h: usize,
        c: usize,
        groups: usize,
        map: &HashMap<String, Vec<f32>>,
        prefix: &str,
    ) -> Vec<f32> {
        let mut normed = x.to_vec();
        h_group_norm(
            &mut normed,
            w,
            h,
            1,
            c,
            groups,
            at(map, &format!("{prefix}.group_norm.weight")),
            at(map, &format!("{prefix}.group_norm.bias")),
        );
        // to tokens [C, W*H] (channel fastest)
        let tokens = w * h;
        let q = h_linear(
            &normed,
            tokens,
            c,
            at(map, &format!("{prefix}.to_q.weight")),
            at(map, &format!("{prefix}.to_q.bias")),
        );
        let k = h_linear(
            &normed,
            tokens,
            c,
            at(map, &format!("{prefix}.to_k.weight")),
            at(map, &format!("{prefix}.to_k.bias")),
        );
        let v = h_linear(
            &normed,
            tokens,
            c,
            at(map, &format!("{prefix}.to_v.weight")),
            at(map, &format!("{prefix}.to_v.bias")),
        );
        let scale = (c as f32).sqrt().recip();
        let mut out = vec![0f32; c * tokens];
        for tq in 0..tokens {
            let mut scores = vec![0f32; tokens];
            let mut max = f32::NEG_INFINITY;
            for tk in 0..tokens {
                let mut s = 0f32;
                for ch in 0..c {
                    s += q[ch * tokens + tq] * k[ch * tokens + tk];
                }
                scores[tk] = s * scale;
                max = max.max(scores[tk]);
            }
            let mut sum = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for ch in 0..c {
                let mut acc = 0f32;
                for tk in 0..tokens {
                    acc += v[ch * tokens + tk] * scores[tk] / sum;
                }
                out[ch * tokens + tq] = acc;
            }
        }
        let projected = h_linear(
            &out,
            tokens,
            c,
            at(map, &format!("{prefix}.to_out.0.weight")),
            at(map, &format!("{prefix}.to_out.0.bias")),
        );
        let mut result = x.to_vec();
        for i in 0..result.len() {
            result[i] += projected[i];
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn h_resnet(
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        in_ch: usize,
        out_ch: usize,
        groups: usize,
        map: &HashMap<String, Vec<f32>>,
        prefix: &str,
    ) -> Vec<f32> {
        let mut hidden = x.to_vec();
        h_group_norm(
            &mut hidden,
            w,
            h,
            d,
            in_ch,
            groups,
            at(map, &format!("{prefix}.norm1.weight")),
            at(map, &format!("{prefix}.norm1.bias")),
        );
        h_silu(&mut hidden);
        let (mut hidden, _, _, _) = h_conv3d(
            &hidden,
            w,
            h,
            d,
            in_ch,
            at(map, &format!("{prefix}.conv1.weight")),
            at(map, &format!("{prefix}.conv1.bias")),
            3,
            1,
            1,
            1,
            out_ch,
        );
        h_group_norm(
            &mut hidden,
            w,
            h,
            d,
            out_ch,
            groups,
            at(map, &format!("{prefix}.norm2.weight")),
            at(map, &format!("{prefix}.norm2.bias")),
        );
        h_silu(&mut hidden);
        let (hidden, _, _, _) = h_conv3d(
            &hidden,
            w,
            h,
            d,
            out_ch,
            at(map, &format!("{prefix}.conv2.weight")),
            at(map, &format!("{prefix}.conv2.bias")),
            3,
            1,
            1,
            1,
            out_ch,
        );
        let skip = if in_ch != out_ch {
            let (s, _, _, _) = h_conv3d(
                x,
                w,
                h,
                d,
                in_ch,
                at(map, &format!("{prefix}.conv_shortcut.weight")),
                at(map, &format!("{prefix}.conv_shortcut.bias")),
                1,
                1,
                1,
                0,
                out_ch,
            );
            s
        } else {
            x.to_vec()
        };
        hidden.iter().zip(skip.iter()).map(|(a, b)| a + b).collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what} length");
        let mut worst = 0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((x - y).abs());
        }
        assert!(worst < tol, "{what} max abs diff {worst}");
    }

    #[test]
    fn smoke_encoder_matches_the_host_reference() {
        let dir = std::env::temp_dir().join(format!("danmo-seedvr2-vae-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (_vae, enc, _dec) = write_smoke(&dir);
        let spec = spec();
        let map: HashMap<String, Vec<f32>> = enc.iter().cloned().collect();
        let (w, h) = (16usize, 16usize);
        let image: Vec<f32> = (0..w * h * 3).map(|i| (i as f32 * 0.11).sin()).collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = encode_tile(&ctx, &enc, &spec, &image, w as i64, h as i64, 1).unwrap();

        // host: conv_in -> block0 (resnet, temporal downsample) -> block1
        // (resnet) -> mid -> norm/silu/conv_out -> mean * scaling
        let (mut x, mut cw, mut ch, mut d) = h_conv3d(
            &image,
            w,
            h,
            1,
            3,
            at(&map, "encoder.conv_in.weight"),
            at(&map, "encoder.conv_in.bias"),
            3,
            1,
            1,
            1,
            8,
        );
        x = h_resnet(
            &x,
            cw,
            ch,
            d,
            8,
            8,
            8,
            &map,
            "encoder.down_blocks.0.resnets.0",
        );
        // downsample: pad (0,1) on H/W, kt=3 stride 2 (temporal_down=1 -> block 0 temporal)
        let mut padded = vec![0f32; (cw + 1) * (ch + 1) * d * 8];
        for c in 0..8 {
            for f in 0..d {
                for yy in 0..ch {
                    let src = (c * d + f) * cw * ch + yy * cw;
                    let dst = (c * d + f) * (cw + 1) * (ch + 1) + yy * (cw + 1);
                    padded[dst..dst + cw].copy_from_slice(&x[src..src + cw]);
                }
            }
        }
        let (x2, cw2, ch2, d2) = h_conv3d(
            &padded,
            cw + 1,
            ch + 1,
            d,
            8,
            at(&map, "encoder.down_blocks.0.downsamplers.0.conv.weight"),
            at(&map, "encoder.down_blocks.0.downsamplers.0.conv.bias"),
            3,
            2,
            2,
            0,
            8,
        );
        assert_eq!((cw2, ch2, d2), (cw / 2, ch / 2, 1));
        cw = cw2;
        ch = ch2;
        let mut x = h_resnet(
            &x2,
            cw,
            ch,
            d2,
            8,
            16,
            8,
            &map,
            "encoder.down_blocks.1.resnets.0",
        );
        x = h_resnet(
            &x,
            cw,
            ch,
            1,
            16,
            16,
            8,
            &map,
            "encoder.mid_block.resnets.0",
        );
        x = h_attention(&x, cw, ch, 16, 8, &map, "encoder.mid_block.attentions.0");
        x = h_resnet(
            &x,
            cw,
            ch,
            1,
            16,
            16,
            8,
            &map,
            "encoder.mid_block.resnets.1",
        );
        h_group_norm(
            &mut x,
            cw,
            ch,
            1,
            16,
            8,
            at(&map, "encoder.conv_norm_out.weight"),
            at(&map, "encoder.conv_norm_out.bias"),
        );
        h_silu(&mut x);
        let (x, _, _, _) = h_conv3d(
            &x,
            cw,
            ch,
            1,
            16,
            at(&map, "encoder.conv_out.weight"),
            at(&map, "encoder.conv_out.bias"),
            3,
            1,
            1,
            1,
            4,
        );
        let n = cw * ch;
        let want: Vec<f32> = (0..2 * n).map(|i| x[i] * spec.scaling).collect();
        close(&got, &want, 1e-3, "encoder");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_decoder_matches_the_host_reference() {
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-vae-dec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (_vae, _enc, dec) = write_smoke(&dir);
        let spec = spec();
        let map: HashMap<String, Vec<f32>> = dec.iter().cloned().collect();
        let (w, h) = (2usize, 2usize);
        let latent: Vec<f32> = (0..w * h * 2).map(|i| (i as f32 * 0.23).cos()).collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = decode_tile(&ctx, &dec, &spec, &latent, w as i64, h as i64, 1).unwrap();

        let mut z: Vec<f32> = latent.iter().map(|v| v / spec.scaling).collect();
        let (mut x, mut cw, mut ch, _) = h_conv3d(
            &z,
            w,
            h,
            1,
            2,
            at(&map, "decoder.conv_in.weight"),
            at(&map, "decoder.conv_in.bias"),
            3,
            1,
            1,
            1,
            16,
        );
        z.clear();
        x = h_resnet(
            &x,
            cw,
            ch,
            1,
            16,
            16,
            8,
            &map,
            "decoder.mid_block.resnets.0",
        );
        x = h_attention(&x, cw, ch, 16, 8, &map, "decoder.mid_block.attentions.0");
        x = h_resnet(
            &x,
            cw,
            ch,
            1,
            16,
            16,
            8,
            &map,
            "decoder.mid_block.resnets.1",
        );
        // up block 0: resnet 16 -> 16, temporal upsampler (tf=2)
        x = h_resnet(
            &x,
            cw,
            ch,
            1,
            16,
            16,
            8,
            &map,
            "decoder.up_blocks.0.resnets.0",
        );
        let (up, _, _, _) = h_conv3d(
            &x,
            cw,
            ch,
            1,
            16,
            at(&map, "decoder.up_blocks.0.upsamplers.0.upscale_conv.weight"),
            at(&map, "decoder.up_blocks.0.upsamplers.0.upscale_conv.bias"),
            1,
            1,
            1,
            0,
            128,
        );
        // pixel shuffle tf=2: channel ((i*2+j)*2)*16+c -> (x*2+j, y*2+i)
        let (w2, h2) = (cw * 2, ch * 2);
        let mut shuffled = vec![0f32; w2 * h2 * 16];
        for c in 0..16 {
            for i in 0..2usize {
                for j in 0..2usize {
                    let s = (i * 2 + j) * 2;
                    for yy in 0..ch {
                        for xx in 0..cw {
                            shuffled[c * w2 * h2 + (yy * 2 + i) * w2 + xx * 2 + j] =
                                up[(s * 16 + c) * cw * ch + yy * cw + xx];
                        }
                    }
                }
            }
        }
        let (x2, _, _, _) = h_conv3d(
            &shuffled,
            w2,
            h2,
            1,
            16,
            at(&map, "decoder.up_blocks.0.upsamplers.0.conv.weight"),
            at(&map, "decoder.up_blocks.0.upsamplers.0.conv.bias"),
            3,
            1,
            1,
            1,
            16,
        );
        cw = w2;
        ch = h2;
        // up block 1 drops 16 -> 8 through a conv_shortcut
        let mut x = h_resnet(
            &x2,
            cw,
            ch,
            1,
            16,
            8,
            8,
            &map,
            "decoder.up_blocks.1.resnets.0",
        );
        h_group_norm(
            &mut x,
            cw,
            ch,
            1,
            8,
            8,
            at(&map, "decoder.conv_norm_out.weight"),
            at(&map, "decoder.conv_norm_out.bias"),
        );
        h_silu(&mut x);
        let (want, _, _, _) = h_conv3d(
            &x,
            cw,
            ch,
            1,
            8,
            at(&map, "decoder.conv_out.weight"),
            at(&map, "decoder.conv_out.bias"),
            3,
            1,
            1,
            1,
            3,
        );
        close(&got, &want, 1e-3, "decoder");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn probe_conv3d_temporal_window() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(64 << 20).unwrap();
        // input [W=2, H=2, D=5, C=1], frame k filled with k+1.
        let x = b.f32_4d("x", 2, 2, 5, 1, true);
        // kernel [KW=1, KH=1, KD=3, IC*OC=1]: delta at kz=2 picks the last
        // frame of each window.
        let k = b.f32_4d("k", 1, 1, 3, 1, true);
        let y = nn(b.conv3d_direct(k, x, 1, 1, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1)).unwrap();
        let mut g = b.finish(y).unwrap();
        let mut data = vec![0f32; 2 * 2 * 5];
        for f in 0..5 {
            for p in 0..4 {
                data[f * 4 + p] = (f + 1) as f32;
            }
        }
        g.set_f32("x", &data).unwrap();
        g.set_f32("k", &[0.0, 0.0, 1.0]).unwrap();
        let out = g.compute().unwrap();
        assert_eq!(out, vec![3.0, 3.0, 3.0, 3.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn probe_causal_pad_replicates_frame_zero() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(64 << 20).unwrap();
        let x = b.f32_4d("x", 2, 2, 5, 1, true);
        let y = causal_pad(&b, x, 2, 2, 5, 1, 2).unwrap();
        let y = nn(b.cont(y)).unwrap();
        let mut g = b.finish(y).unwrap();
        let mut data = vec![0f32; 2 * 2 * 5];
        for f in 0..5 {
            for p in 0..4 {
                data[f * 4 + p] = (f + 1) as f32;
            }
        }
        g.set_f32("x", &data).unwrap();
        let out = g.compute().unwrap();
        let want: Vec<f32> = [1f32, 1., 1., 2., 3., 4., 5.]
            .into_iter()
            .flat_map(|v| [v; 4])
            .collect();
        assert_eq!(out, want);
    }

    #[test]
    fn infer_recovers_the_written_spec() {
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-vae-infer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_smoke_vae(&dir).unwrap();
        let weights = GgufWeights::open(&path).unwrap();
        let spec = VaeSpec::infer(&weights).unwrap();
        let want = VaeSpec::smoke();
        assert_eq!(spec.block_out, want.block_out);
        assert_eq!(spec.enc_resnets, want.enc_resnets);
        assert_eq!(spec.dec_resnets, want.dec_resnets);
        assert_eq!(spec.latent, want.latent);
        assert_eq!(spec.temporal_down, want.temporal_down);
        assert_eq!(spec.temporal_up, want.temporal_up);
        assert_eq!(spec.groups, want.groups);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn h_attention_frames(
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        c: usize,
        groups: usize,
        map: &HashMap<String, Vec<f32>>,
        prefix: &str,
    ) -> Vec<f32> {
        let plane = w * h;
        let mut out = x.to_vec();
        for f in 0..d {
            let mut frame = Vec::with_capacity(c * plane);
            for ch in 0..c {
                frame.extend_from_slice(&x[(ch * d + f) * plane..(ch * d + f) * plane + plane]);
            }
            let res = h_attention(&frame, w, h, c, groups, map, prefix);
            for ch in 0..c {
                out[(ch * d + f) * plane..(ch * d + f) * plane + plane]
                    .copy_from_slice(&res[ch * plane..(ch + 1) * plane]);
            }
        }
        out
    }

    /// host encode of the smoke spec over `f` frames; returns
    /// `[latent, D, hl, wl]` planar `[C][F][H][W]` scaled mean.
    fn h_encode(
        image: &[f32],
        w: usize,
        h: usize,
        f: usize,
        spec: &VaeSpec,
        map: &HashMap<String, Vec<f32>>,
    ) -> (Vec<f32>, usize, usize, usize) {
        let (mut x, mut cw, mut ch, mut d) = h_conv3d(
            image,
            w,
            h,
            f,
            3,
            at(map, "encoder.conv_in.weight"),
            at(map, "encoder.conv_in.bias"),
            3,
            1,
            1,
            1,
            8,
        );
        x = h_resnet(
            &x,
            cw,
            ch,
            d,
            8,
            8,
            8,
            map,
            "encoder.down_blocks.0.resnets.0",
        );
        let mut padded = vec![0f32; (cw + 1) * (ch + 1) * d * 8];
        for c in 0..8 {
            for fr in 0..d {
                for yy in 0..ch {
                    let src = (c * d + fr) * cw * ch + yy * cw;
                    let dst = (c * d + fr) * (cw + 1) * (ch + 1) + yy * (cw + 1);
                    padded[dst..dst + cw].copy_from_slice(&x[src..src + cw]);
                }
            }
        }
        let (x2, cw2, ch2, d2) = h_conv3d(
            &padded,
            cw + 1,
            ch + 1,
            d,
            8,
            at(map, "encoder.down_blocks.0.downsamplers.0.conv.weight"),
            at(map, "encoder.down_blocks.0.downsamplers.0.conv.bias"),
            3,
            2,
            2,
            0,
            8,
        );
        assert_eq!((cw2, ch2), (cw / 2, ch / 2));
        cw = cw2;
        ch = ch2;
        d = d2;
        let mut x = h_resnet(
            &x2,
            cw,
            ch,
            d,
            8,
            16,
            8,
            map,
            "encoder.down_blocks.1.resnets.0",
        );
        x = h_resnet(&x, cw, ch, d, 16, 16, 8, map, "encoder.mid_block.resnets.0");
        x = h_attention_frames(&x, cw, ch, d, 16, 8, map, "encoder.mid_block.attentions.0");
        x = h_resnet(&x, cw, ch, d, 16, 16, 8, map, "encoder.mid_block.resnets.1");
        h_group_norm(
            &mut x,
            cw,
            ch,
            d,
            16,
            8,
            at(map, "encoder.conv_norm_out.weight"),
            at(map, "encoder.conv_norm_out.bias"),
        );
        h_silu(&mut x);
        let (x, _, _, _) = h_conv3d(
            &x,
            cw,
            ch,
            d,
            16,
            at(map, "encoder.conv_out.weight"),
            at(map, "encoder.conv_out.bias"),
            3,
            1,
            1,
            1,
            4,
        );
        let plane = cw * ch;
        let latent = spec.latent as usize;
        let want: Vec<f32> = (0..latent * d * plane)
            .map(|i| {
                let px = i % plane;
                let fr = (i / plane) % d;
                let c = i / (plane * d);
                x[(c * d + fr) * plane + px] * spec.scaling
            })
            .collect();
        (want, cw, ch, d)
    }

    /// host decode of the smoke spec over `d` latent frames; returns
    /// `[3, F, h_out, w_out]` planar `[C][F][H][W]`.
    fn h_decode(
        latent: &[f32],
        w: usize,
        h: usize,
        d: usize,
        spec: &VaeSpec,
        map: &HashMap<String, Vec<f32>>,
    ) -> (Vec<f32>, usize, usize, usize) {
        let z: Vec<f32> = latent.iter().map(|v| v / spec.scaling).collect();
        let (mut x, mut cw, mut ch, d) = h_conv3d(
            &z,
            w,
            h,
            d,
            2,
            at(map, "decoder.conv_in.weight"),
            at(map, "decoder.conv_in.bias"),
            3,
            1,
            1,
            1,
            16,
        );
        x = h_resnet(&x, cw, ch, d, 16, 16, 8, map, "decoder.mid_block.resnets.0");
        x = h_attention_frames(&x, cw, ch, d, 16, 8, map, "decoder.mid_block.attentions.0");
        x = h_resnet(&x, cw, ch, d, 16, 16, 8, map, "decoder.mid_block.resnets.1");
        x = h_resnet(
            &x,
            cw,
            ch,
            d,
            16,
            16,
            8,
            map,
            "decoder.up_blocks.0.resnets.0",
        );
        let (up, _, _, _) = h_conv3d(
            &x,
            cw,
            ch,
            d,
            16,
            at(map, "decoder.up_blocks.0.upsamplers.0.upscale_conv.weight"),
            at(map, "decoder.up_blocks.0.upsamplers.0.upscale_conv.bias"),
            1,
            1,
            1,
            0,
            128,
        );
        // pixel shuffle tf=2 over d frames: channel ((i*2+j)*2+t)*16+c of
        // frame f lands at (x*2+j, y*2+i) of output frame f*2+t; d==1 keeps
        // one output frame from the t=0 slice.
        let (w2, h2) = (cw * 2, ch * 2);
        let f2 = if d == 1 { 1 } else { d * 2 };
        let mut shuffled = vec![0f32; 16 * f2 * w2 * h2];
        for c in 0..16 {
            for f in 0..d {
                let t_range: Vec<usize> = if d == 1 { vec![0] } else { vec![0, 1] };
                for t in t_range {
                    let fo = if d == 1 { 0 } else { f * 2 + t };
                    for i in 0..2usize {
                        for j in 0..2usize {
                            let s = ((i * 2 + j) * 2 + t) * 16 + c;
                            for yy in 0..ch {
                                for xx in 0..cw {
                                    shuffled[(c * f2 + fo) * w2 * h2
                                        + (yy * 2 + i) * w2
                                        + xx * 2
                                        + j] = up[(s * d + f) * cw * ch + yy * cw + xx];
                                }
                            }
                        }
                    }
                }
            }
        }
        let (x2, _, _, _) = h_conv3d(
            &shuffled,
            w2,
            h2,
            f2,
            16,
            at(map, "decoder.up_blocks.0.upsamplers.0.conv.weight"),
            at(map, "decoder.up_blocks.0.upsamplers.0.conv.bias"),
            3,
            1,
            1,
            1,
            16,
        );
        cw = w2;
        ch = h2;
        let mut x = h_resnet(
            &x2,
            cw,
            ch,
            f2,
            16,
            8,
            8,
            map,
            "decoder.up_blocks.1.resnets.0",
        );
        h_group_norm(
            &mut x,
            cw,
            ch,
            f2,
            8,
            8,
            at(map, "decoder.conv_norm_out.weight"),
            at(map, "decoder.conv_norm_out.bias"),
        );
        h_silu(&mut x);
        let (want, _, _, _) = h_conv3d(
            &x,
            cw,
            ch,
            f2,
            8,
            at(map, "decoder.conv_out.weight"),
            at(map, "decoder.conv_out.bias"),
            3,
            1,
            1,
            1,
            3,
        );
        (want, cw, ch, f2)
    }

    #[test]
    fn frame_counts_follow_the_temporal_strides() {
        let spec = spec();
        assert_eq!(encode_frames_out(&spec, 1), 1);
        assert_eq!(encode_frames_out(&spec, 4), 2);
        assert_eq!(encode_frames_out(&spec, 5), 3);
        assert_eq!(decode_frames_out(&spec, 1), 1);
        assert_eq!(decode_frames_out(&spec, 2), 4);
        assert_eq!(decode_frames_out(&spec, 3), 6);
    }

    #[test]
    fn multi_frame_encode_decode_matches_the_host_reference() {
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-vae-multi-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (_vae, enc, dec) = write_smoke(&dir);
        let spec = spec();
        let enc_map: HashMap<String, Vec<f32>> = enc.iter().cloned().collect();
        let dec_map: HashMap<String, Vec<f32>> = dec.iter().cloned().collect();
        let (w, h, f) = (16usize, 16usize, 4usize);
        let image: Vec<f32> = (0..w * h * f * 3)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got_enc = encode_tile(&ctx, &enc, &spec, &image, w as i64, h as i64, f as i64).unwrap();
        let (want_enc, lw, lh, d) = h_encode(&image, w, h, f, &spec, &enc_map);
        assert_eq!(d, 2);
        close(&got_enc, &want_enc, 1e-3, "multi-frame encoder");

        let got_dec =
            decode_tile(&ctx, &dec, &spec, &got_enc, lw as i64, lh as i64, d as i64).unwrap();
        let (want_dec, _, _, f_out) = h_decode(&want_enc, lw, lh, d, &spec, &dec_map);
        assert_eq!(f_out, 4);
        close(&got_dec, &want_dec, 1e-3, "multi-frame decoder");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn encode_is_causal_along_the_frame_axis() {
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-vae-causal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (_vae, enc, _dec) = write_smoke(&dir);
        let spec = spec();
        let (w, h, f) = (16usize, 16usize, 5usize);
        let image: Vec<f32> = (0..w * h * f * 3)
            .map(|i| (i as f32 * 0.11).sin())
            .collect();
        // Perturb the whole last frame; latent frames before it must not move.
        let mut perturbed = image.clone();
        let plane = w * h;
        for c in 0..3 {
            for px in 0..plane {
                perturbed[c * f * plane + 4 * plane + px] += 0.5;
            }
        }
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let base = encode_tile(&ctx, &enc, &spec, &image, w as i64, h as i64, f as i64).unwrap();
        let changed =
            encode_tile(&ctx, &enc, &spec, &perturbed, w as i64, h as i64, f as i64).unwrap();
        let d = encode_frames_out(&spec, f as i64) as usize;
        assert_eq!(d, 3);
        // one spatial downsample in the smoke spec (blocks = 2)
        let lw = w / 2;
        let lh = h / 2;
        let lplane = lw * lh;
        let latent = spec.latent as usize;
        let frame_slice = |v: &[f32], fr: usize| -> Vec<f32> {
            (0..latent * lplane)
                .map(|i| {
                    let px = i % lplane;
                    let c = i / lplane;
                    v[(c * d + fr) * lplane + px]
                })
                .collect()
        };
        let frame_diff = |fr: usize| -> f32 {
            frame_slice(&base, fr)
                .iter()
                .zip(frame_slice(&changed, fr))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        for fr in 0..2 {
            assert!(
                frame_diff(fr) < 1e-5,
                "latent frame {fr} must not depend on input frame 4 (diff {})",
                frame_diff(fr)
            );
        }
        assert!(
            frame_diff(2) > 1e-4,
            "latent frame 2 must see input frame 4 (diff {})",
            frame_diff(2)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
