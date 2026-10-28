//! ACE-Step AutoencoderOobleck decoder (audio VAE) via ggml.
//!
//! Topology matches `/tmp/acestep-cpp/vae.h`: conv1 → five upsample blocks (snake +
//! conv_transpose via `mul_mat` + `col2im_1d` + three ResUnits) → snake → conv2.
//! Upsample strides 10×6×4×4×2 = **1920×** per latent frame.
//!
//! **Output layout:** planar stereo `[ch0 (T samples), ch1 (T samples)]` with
//! `T = frames × 1920`. The ggml tensor is `[T, 2]` with channel 1 stored at
//! offset `T` (same as acestep.cpp tiled decode readback).
//!
//! **Weights:** real GGUF stores `weight_g` / `weight_v` for weight-norm convs;
//! load fuses to graph inputs. Smoke / layout-only bundles may ship pre-fused
//! `*.weight` tensors (F32 or F16) plus raw snake `alpha` / `beta` (exp applied at load).

use crate::runtime::{shape, GraphBuilder, GgmlContext, ReusedGraph, Tensor, TYPE_F16, TYPE_F32};
use crate::weights::{GgufWeights, WeightError};
use danmo_ggml_sys as g;

pub const LATENT_CHANNELS: i64 = 64;
pub const AUDIO_UPSAMPLE: i64 = 1920;
pub const OUT_CHANNELS: i64 = 2;

const BLOCK_STRIDES: [i64; 5] = [10, 6, 4, 4, 2];
const BLOCK_IN_CH: [i64; 5] = [2048, 1024, 512, 256, 128];
const BLOCK_OUT_CH: [i64; 5] = [1024, 512, 256, 128, 128];
const RES_DILATIONS: [i32; 3] = [1, 3, 9];

const GRAPH_MEM: usize = 512 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AceVaeConfig {
    pub latent_channels: i64,
}

impl AceVaeConfig {
    pub fn published() -> Self {
        Self {
            latent_channels: LATENT_CHANNELS,
        }
    }

    /// Same ranks as XL; use zero weights in unit tests.
    pub fn smoke() -> Self {
        Self::published()
    }
}

#[derive(Debug, Clone)]
pub struct VaeWeightSpec {
    pub name: String,
    pub ne: [i64; 4],
    /// Graph input ggml type (`TYPE_F32` or `TYPE_F16`).
    pub ggml_type: u32,
}

/// GGUF / safetensors names for layout checks (unfused weight-norm + snake params).
pub fn weight_layout(_config: AceVaeConfig) -> Vec<VaeWeightSpec> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<VaeWeightSpec>, name: &str, ne: [i64; 4]| {
        out.push(VaeWeightSpec {
            name: name.to_string(),
            ne,
            ggml_type: TYPE_F32,
        });
    };
    push(&mut out, "decoder.conv1.weight_g", [2048, 1, 1, 1]);
    push(&mut out, "decoder.conv1.weight_v", [7, 64, 2048, 1]);
    push(&mut out, "decoder.conv1.bias", [2048, 1, 1, 1]);
    for i in 0..5 {
        let ic = BLOCK_IN_CH[i];
        let oc = BLOCK_OUT_CH[i];
        let k = BLOCK_STRIDES[i] * 2;
        let p = format!("decoder.block.{i}");
        push(&mut out, &format!("{p}.snake1.alpha"), [1, ic, 1, 1]);
        push(&mut out, &format!("{p}.snake1.beta"), [1, ic, 1, 1]);
        push(&mut out, &format!("{p}.conv_t1.weight_g"), [ic, 1, 1, 1]);
        push(&mut out, &format!("{p}.conv_t1.weight_v"), [k, oc, ic, 1]);
        push(&mut out, &format!("{p}.conv_t1.bias"), [oc, 1, 1, 1]);
        for r in 1..=3 {
            let ru = format!("{p}.res_unit{r}");
            push(&mut out, &format!("{ru}.snake1.alpha"), [1, oc, 1, 1]);
            push(&mut out, &format!("{ru}.snake1.beta"), [1, oc, 1, 1]);
            push(&mut out, &format!("{ru}.conv1.weight_g"), [oc, 1, 1, 1]);
            push(&mut out, &format!("{ru}.conv1.weight_v"), [7, oc, oc, 1]);
            push(&mut out, &format!("{ru}.conv1.bias"), [oc, 1, 1, 1]);
            push(&mut out, &format!("{ru}.snake2.alpha"), [1, oc, 1, 1]);
            push(&mut out, &format!("{ru}.snake2.beta"), [1, oc, 1, 1]);
            push(&mut out, &format!("{ru}.conv2.weight_g"), [oc, 1, 1, 1]);
            push(&mut out, &format!("{ru}.conv2.weight_v"), [1, oc, oc, 1]);
            push(&mut out, &format!("{ru}.conv2.bias"), [oc, 1, 1, 1]);
        }
    }
    push(&mut out, "decoder.snake1.alpha", [1, 128, 1, 1]);
    push(&mut out, "decoder.snake1.beta", [1, 128, 1, 1]);
    push(&mut out, "decoder.conv2.weight_g", [2, 1, 1, 1]);
    push(&mut out, "decoder.conv2.weight_v", [7, 128, 2, 1]);
    out
}

/// Fused graph parameter names (what `load_weights` writes after weight-norm / snake prep).
pub fn graph_weight_layout(_config: AceVaeConfig) -> Vec<VaeWeightSpec> {
    let mut out = Vec::new();
    out.push(VaeWeightSpec {
        name: "decoder.conv1.weight".into(),
        ne: [7, 64, 2048, 1],
        ggml_type: TYPE_F16,
    });
    out.push(VaeWeightSpec {
        name: "decoder.conv1.bias".into(),
        ne: [2048, 1, 1, 1],
        ggml_type: TYPE_F32,
    });
    for i in 0..5 {
        let ic = BLOCK_IN_CH[i];
        let oc = BLOCK_OUT_CH[i];
        let k = BLOCK_STRIDES[i] * 2;
        let p = format!("decoder.block.{i}");
        out.push(VaeWeightSpec {
            name: format!("{p}.snake1.exp_a"),
            ne: [1, ic, 1, 1],
            ggml_type: TYPE_F32,
        });
        out.push(VaeWeightSpec {
            name: format!("{p}.snake1.inv_b"),
            ne: [1, ic, 1, 1],
            ggml_type: TYPE_F32,
        });
        out.push(VaeWeightSpec {
            name: format!("{p}.conv_t1.weight"),
            ne: [ic, k * oc, 1, 1],
            ggml_type: TYPE_F16,
        });
        out.push(VaeWeightSpec {
            name: format!("{p}.conv_t1.bias"),
            ne: [oc, 1, 1, 1],
            ggml_type: TYPE_F32,
        });
        for r in 1..=3 {
            let ru = format!("{p}.res_unit{r}");
            for (stem, ch) in [("snake1", oc), ("snake2", oc)] {
                out.push(VaeWeightSpec {
                    name: format!("{ru}.{stem}.exp_a"),
                    ne: [1, ch, 1, 1],
                    ggml_type: TYPE_F32,
                });
                out.push(VaeWeightSpec {
                    name: format!("{ru}.{stem}.inv_b"),
                    ne: [1, ch, 1, 1],
                    ggml_type: TYPE_F32,
                });
            }
            out.push(VaeWeightSpec {
                name: format!("{ru}.conv1.weight"),
                ne: [7, oc, oc, 1],
                ggml_type: TYPE_F16,
            });
            out.push(VaeWeightSpec {
                name: format!("{ru}.conv1.bias"),
                ne: [oc, 1, 1, 1],
                ggml_type: TYPE_F32,
            });
            out.push(VaeWeightSpec {
                name: format!("{ru}.conv2.weight"),
                ne: [1, oc, oc, 1],
                ggml_type: TYPE_F16,
            });
            out.push(VaeWeightSpec {
                name: format!("{ru}.conv2.bias"),
                ne: [oc, 1, 1, 1],
                ggml_type: TYPE_F32,
            });
        }
    }
    out.push(VaeWeightSpec {
        name: "decoder.snake1.exp_a".into(),
        ne: [1, 128, 1, 1],
        ggml_type: TYPE_F32,
    });
    out.push(VaeWeightSpec {
        name: "decoder.snake1.inv_b".into(),
        ne: [1, 128, 1, 1],
        ggml_type: TYPE_F32,
    });
    out.push(VaeWeightSpec {
        name: "decoder.conv2.weight".into(),
        ne: [7, 128, 2, 1],
        ggml_type: TYPE_F16,
    });
    out
}

pub(crate) struct ResUnitW {
    pub(crate) s1a: Tensor,
    pub(crate) s1b: Tensor,
    pub(crate) c1w: Tensor,
    pub(crate) c1b: Tensor,
    pub(crate) s2a: Tensor,
    pub(crate) s2b: Tensor,
    pub(crate) c2w: Tensor,
    pub(crate) c2b: Tensor,
    pub(crate) dilation: i32,
}

struct BlockW {
    sa: Tensor,
    sb: Tensor,
    ctw: Tensor,
    ctb: Tensor,
    stride: i32,
    kernel: i32,
    out_ch: i32,
    ru: [ResUnitW; 3],
}

struct VaeWeights {
    c1w: Tensor,
    c1b: Tensor,
    blocks: [BlockW; 5],
    sa: Tensor,
    sb: Tensor,
    c2w: Tensor,
}

/// Decode DiT latents `[frames, 64]` (time-major) to planar stereo audio.
pub fn decode(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    latent: &[f32],
    frames: i64,
) -> Result<Vec<f32>, WeightError> {
    let config = AceVaeConfig::published();
    if frames < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step VAE decode needs at least one latent frame".into(),
        ));
    }
    if config.latent_channels != LATENT_CHANNELS {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step VAE latent channels {} != published {LATENT_CHANNELS}",
            config.latent_channels
        )));
    }
    let expect_latent = (frames * config.latent_channels) as usize;
    if latent.len() != expect_latent {
        return Err(WeightError::Shape {
            name: "latent".into(),
            expected: vec![frames, config.latent_channels],
            got: vec![latent.len() as i64],
        });
    }
    let expect_audio = (frames * AUDIO_UPSAMPLE * OUT_CHANNELS) as usize;
    let mut session = AceVaeSession::build(ctx, config, frames)?;
    load_weights(&mut session, weights, config)?;
    let planar = session.run(latent, frames)?;
    if planar.len() != expect_audio {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step VAE output length {} != expected {expect_audio}",
            planar.len()
        )));
    }
    Ok(planar)
}

pub(crate) struct AceVaeSession {
    graph: ReusedGraph,
    frames: i64,
}

impl AceVaeSession {
    pub(crate) fn build(ctx: &GgmlContext, config: AceVaeConfig, frames: i64) -> Result<Self, WeightError> {
        let mut builder = ctx
            .builder(GRAPH_MEM)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let latent_in = builder.f32_2d("vae.latent", frames, config.latent_channels, true);
        let w = declare_weights(&mut builder, config)?;
        let output = build_decode_graph(&builder, &w, latent_in, config);
        let graph = builder
            .finish(output)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        Ok(Self { graph, frames })
    }

    fn run(&mut self, latent: &[f32], frames: i64) -> Result<Vec<f32>, WeightError> {
        if frames != self.frames {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE graph was built for T={} but decode asked for T={frames}",
                self.frames
            )));
        }
        // Time-major [t, c] → ggml [T, C] row-major t + c*T.
        let t = frames as usize;
        let c = LATENT_CHANNELS as usize;
        let mut packed = vec![0f32; t * c];
        for tt in 0..t {
            for cc in 0..c {
                packed[tt + cc * t] = latent[tt * c + cc];
            }
        }
        self.graph
            .set_f32("vae.latent", &packed)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        self.graph
            .compute_device()
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let raw = self
            .graph
            .peek_output()
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
        let [t_out, ch, ..] = shape(self.graph.output());
        if ch != OUT_CHANNELS {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE graph output channels {ch} != {OUT_CHANNELS}"
            )));
        }
        let expect_t = frames * AUDIO_UPSAMPLE;
        if t_out != expect_t {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE T_audio {t_out} != frames*1920 ({expect_t})"
            )));
        }
        if raw.len() as i64 != t_out * ch {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step VAE readback {} floats != {t_out}×{ch}",
                raw.len()
            )));
        }
        let t_usize = t_out as usize;
        let mut planar = vec![0f32; t_usize * 2];
        for i in 0..t_usize {
            planar[i] = raw[i];
            planar[t_usize + i] = raw[t_usize + i];
        }
        Ok(planar)
    }

    fn set_f32(&mut self, name: &str, data: &[f32]) -> Result<(), WeightError> {
        self.graph
            .set_f32(name, data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))
    }

    fn set_f16(&mut self, name: &str, data: &[f32]) -> Result<(), WeightError> {
        let mut bytes = vec![0u8; data.len() * 2];
        f32_row_to_f16(data, &mut bytes);
        self.graph
            .set_bytes(name, &bytes)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))
    }
}

fn declare_weights(b: &mut GraphBuilder, config: AceVaeConfig) -> Result<VaeWeights, WeightError> {
    if config.latent_channels != LATENT_CHANNELS {
        return Err(WeightError::NotBuilt(
            "ACE-Step VAE only supports 64-dim latents".into(),
        ));
    }
    let c1w = b.tensor_3d("decoder.conv1.weight", TYPE_F16, 7, 64, 2048, true);
    let c1b = b.tensor_1d("decoder.conv1.bias", TYPE_F32, 2048, true);
    let mut blocks: [BlockW; 5] = [
        block_placeholder(),
        block_placeholder(),
        block_placeholder(),
        block_placeholder(),
        block_placeholder(),
    ];
    for i in 0..5 {
        let ic = BLOCK_IN_CH[i];
        let oc = BLOCK_OUT_CH[i];
        let stride = BLOCK_STRIDES[i] as i32;
        let kernel = (stride * 2) as i32;
        let p = format!("decoder.block.{i}");
        let sa = b.tensor_2d(&format!("{p}.snake1.exp_a"), TYPE_F32, 1, ic, true);
        let sb = b.tensor_2d(&format!("{p}.snake1.inv_b"), TYPE_F32, 1, ic, true);
        let ctw = b.tensor_2d(
            &format!("{p}.conv_t1.weight"),
            TYPE_F16,
            ic,
            (kernel as i64) * oc,
            true,
        );
        let ctb = b.tensor_1d(&format!("{p}.conv_t1.bias"), TYPE_F32, oc, true);
        let mut ru = [res_unit_placeholder(), res_unit_placeholder(), res_unit_placeholder()];
        for (r_idx, dil) in RES_DILATIONS.iter().enumerate() {
            let r = r_idx + 1;
            let ru_p = format!("{p}.res_unit{r}");
            ru[r_idx] = declare_res_unit(b, &ru_p, oc, *dil)?;
        }
        blocks[i] = BlockW {
            sa,
            sb,
            ctw,
            ctb,
            stride,
            kernel,
            out_ch: oc as i32,
            ru,
        };
    }
    let sa = b.tensor_2d("decoder.snake1.exp_a", TYPE_F32, 1, 128, true);
    let sb = b.tensor_2d("decoder.snake1.inv_b", TYPE_F32, 1, 128, true);
    let c2w = b.tensor_3d("decoder.conv2.weight", TYPE_F16, 7, 128, 2, true);
    Ok(VaeWeights {
        c1w,
        c1b,
        blocks,
        sa,
        sb,
        c2w,
    })
}

pub(crate) fn declare_res_unit(
    b: &mut GraphBuilder,
    prefix: &str,
    ch: i64,
    dilation: i32,
) -> Result<ResUnitW, WeightError> {
    Ok(ResUnitW {
        s1a: b.tensor_2d(&format!("{prefix}.snake1.exp_a"), TYPE_F32, 1, ch, true),
        s1b: b.tensor_2d(&format!("{prefix}.snake1.inv_b"), TYPE_F32, 1, ch, true),
        c1w: b.tensor_3d(&format!("{prefix}.conv1.weight"), TYPE_F16, 7, ch, ch, true),
        c1b: b.tensor_1d(&format!("{prefix}.conv1.bias"), TYPE_F32, ch, true),
        s2a: b.tensor_2d(&format!("{prefix}.snake2.exp_a"), TYPE_F32, 1, ch, true),
        s2b: b.tensor_2d(&format!("{prefix}.snake2.inv_b"), TYPE_F32, 1, ch, true),
        c2w: b.tensor_3d(&format!("{prefix}.conv2.weight"), TYPE_F16, 1, ch, ch, true),
        c2b: b.tensor_1d(&format!("{prefix}.conv2.bias"), TYPE_F32, ch, true),
        dilation,
    })
}

fn block_placeholder() -> BlockW {
    BlockW {
        sa: std::ptr::null_mut(),
        sb: std::ptr::null_mut(),
        ctw: std::ptr::null_mut(),
        ctb: std::ptr::null_mut(),
        stride: 0,
        kernel: 0,
        out_ch: 0,
        ru: [
            res_unit_placeholder(),
            res_unit_placeholder(),
            res_unit_placeholder(),
        ],
    }
}

pub(crate) fn res_unit_placeholder() -> ResUnitW {
    ResUnitW {
        s1a: std::ptr::null_mut(),
        s1b: std::ptr::null_mut(),
        c1w: std::ptr::null_mut(),
        c1b: std::ptr::null_mut(),
        s2a: std::ptr::null_mut(),
        s2b: std::ptr::null_mut(),
        c2w: std::ptr::null_mut(),
        c2b: std::ptr::null_mut(),
        dilation: 1,
    }
}

fn build_decode_graph(
    b: &GraphBuilder,
    w: &VaeWeights,
    latent: Tensor,
    config: AceVaeConfig,
) -> Tensor {
    let _ = config;
    let mut x = conv1d_bias(b, w.c1w, Some(w.c1b), latent, 1, 3, 1);
    for blk in &w.blocks {
        x = snake(b, x, blk.sa, blk.sb);
        let pad = (blk.kernel - blk.stride) / 2;
        x = conv_t1d(b, blk.ctw, Some(blk.ctb), x, blk.stride, pad, blk.out_ch);
        for ru in &blk.ru {
            x = res_unit(b, ru, x);
        }
    }
    x = snake(b, x, w.sa, w.sb);
    conv1d_bias(b, w.c2w, None, x, 1, 3, 1)
}

pub(crate) fn snake(b: &GraphBuilder, x: Tensor, exp_a: Tensor, inv_b: Tensor) -> Tensor {
    let ax = b.mul(x, exp_a);
    let s = b.sin(ax);
    let s2 = b.sqr(s);
    let d = b.mul(s2, inv_b);
    b.add(x, d)
}

pub(crate) fn conv1d_bias(
    b: &GraphBuilder,
    weight: Tensor,
    bias: Option<Tensor>,
    x: Tensor,
    stride: i32,
    pad: i32,
    dilation: i32,
) -> Tensor {
    let mut y = b.conv_1d(weight, x, stride, pad, dilation);
    let sh = shape(y);
    y = b.reshape_2d(y, sh[0], sh[1]);
    if let Some(bias) = bias {
        let b2d = b.reshape_2d(bias, 1, shape(bias)[0]);
        y = b.add(y, b2d);
    }
    y
}

fn conv_t1d(
    b: &GraphBuilder,
    weight: Tensor,
    bias: Option<Tensor>,
    x: Tensor,
    stride: i32,
    pad: i32,
    oc: i32,
) -> Tensor {
    let xt = b.cont(b.transpose(x));
    let col = b.mul_mat(weight, xt);
    let mut y = b.col2im_1d(col, stride, oc, pad);
    if let Some(bias) = bias {
        let b2d = b.reshape_2d(bias, 1, oc as i64);
        y = b.add(y, b2d);
    }
    y
}

pub(crate) fn res_unit(b: &GraphBuilder, ru: &ResUnitW, x: Tensor) -> Tensor {
    let skip = x;
    let pad = 3 * ru.dilation;
    let mut y = snake(b, x, ru.s1a, ru.s1b);
    y = conv1d_bias(b, ru.c1w, Some(ru.c1b), y, 1, pad, ru.dilation);
    y = snake(b, y, ru.s2a, ru.s2b);
    y = conv1d_bias(b, ru.c2w, Some(ru.c2b), y, 1, 0, 1);
    b.add(skip, y)
}

pub fn load_weights(
    session: &mut AceVaeSession,
    weights: &GgufWeights,
    config: AceVaeConfig,
) -> Result<(), WeightError> {
    let _ = config;
    load_conv1d_fused(session, weights, "decoder.conv1", [7, 64, 2048])?;
    load_bias(session, weights, "decoder.conv1.bias", 2048)?;
    for i in 0..5 {
        let ic = BLOCK_IN_CH[i];
        let oc = BLOCK_OUT_CH[i];
        let k = BLOCK_STRIDES[i] * 2;
        let p = format!("decoder.block.{i}");
        load_snake_pair(session, weights, &format!("{p}.snake1"), ic)?;
        load_conv_t1_fused(session, weights, &format!("{p}.conv_t1"), ic, k, oc)?;
        load_bias(session, weights, &format!("{p}.conv_t1.bias"), oc)?;
        for r in 1..=3 {
            let ru = format!("{p}.res_unit{r}");
            load_snake_pair(session, weights, &format!("{ru}.snake1"), oc)?;
            load_conv1d_fused(session, weights, &format!("{ru}.conv1"), [7, oc, oc])?;
            load_bias(session, weights, &format!("{ru}.conv1.bias"), oc)?;
            load_snake_pair(session, weights, &format!("{ru}.snake2"), oc)?;
            load_conv1d_fused(session, weights, &format!("{ru}.conv2"), [1, oc, oc])?;
            load_bias(session, weights, &format!("{ru}.conv2.bias"), oc)?;
        }
    }
    load_snake_pair(session, weights, "decoder.snake1", 128)?;
    load_conv1d_fused(session, weights, "decoder.conv2", [7, 128, 2])?;
    Ok(())
}

fn load_bias(session: &mut AceVaeSession, weights: &GgufWeights, name: &str, len: i64) -> Result<(), WeightError> {
    let data = weights.f32_widened(name, &[len])?;
    session.set_f32(name, &data)
}

fn load_snake_pair(
    session: &mut AceVaeSession,
    weights: &GgufWeights,
    prefix: &str,
    channels: i64,
) -> Result<(), WeightError> {
    let exp_name = format!("{prefix}.exp_a");
    let inv_name = format!("{prefix}.inv_b");
    if weights.contains(&exp_name) && weights.contains(&inv_name) {
        let ne_a = weights.shape(&exp_name)?;
        let ne_b = weights.shape(&inv_name)?;
        let exp_a = weights.f32_widened(&exp_name, &[ne_a[0], ne_a[1], ne_a[2]])?;
        let inv_b = weights.f32_widened(&inv_name, &[ne_b[0], ne_b[1], ne_b[2]])?;
        session.set_f32(&exp_name, &exp_a)?;
        return session.set_f32(&inv_name, &inv_b);
    }
    let alpha_name = format!("{prefix}.alpha");
    let beta_name = format!("{prefix}.beta");
    let alpha = weights.f32_widened(&alpha_name, &[1, channels, 1])?;
    let beta = weights.f32_widened(&beta_name, &[1, channels, 1])?;
    let mut exp_a = Vec::with_capacity(channels as usize);
    let mut inv_b = Vec::with_capacity(channels as usize);
    for i in 0..channels as usize {
        let a = alpha.get(i).copied().unwrap_or(alpha[0]);
        let b = beta.get(i).copied().unwrap_or(beta[0]);
        if !a.is_finite() || !b.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "{prefix} snake alpha/beta is not finite"
            )));
        }
        exp_a.push(a.exp());
        inv_b.push((b.exp()).recip());
    }
    session.set_f32(&format!("{prefix}.exp_a"), &exp_a)?;
    session.set_f32(&format!("{prefix}.inv_b"), &inv_b)
}

fn load_conv1d_fused(
    session: &mut AceVaeSession,
    weights: &GgufWeights,
    prefix: &str,
    _ne: [i64; 3],
) -> Result<(), WeightError> {
    let fused = fuse_weight_norm(weights, prefix, false)?;
    let name = format!("{prefix}.weight");
    session.set_f16(&name, &fused)
}

fn load_conv_t1_fused(
    session: &mut AceVaeSession,
    weights: &GgufWeights,
    prefix: &str,
    ic: i64,
    kernel: i64,
    oc: i64,
) -> Result<(), WeightError> {
    let fused = fuse_weight_norm(weights, prefix, true)?;
    let expect = (ic * kernel * oc) as usize;
    if fused.len() != expect {
        return Err(WeightError::Shape {
            name: format!("{prefix}.weight"),
            expected: vec![ic, kernel * oc],
            got: vec![fused.len() as i64],
        });
    }
    session.set_f16(&format!("{prefix}.weight"), &fused)
}

fn ggml_dim0_and_count(ne: [i64; 4]) -> (i64, i64) {
    let mut n_dims = 1usize;
    for i in 1..4 {
        if ne[i] > 1 {
            n_dims = i + 1;
        }
    }
    let dim0 = ne[n_dims - 1];
    let count = ne[0] * ne[1] * ne[2] * if ne[3] > 1 { ne[3] } else { 1 };
    (dim0, count)
}

/// Fuse `weight_g` × normalized `weight_v`, or read pre-fused `{prefix}.weight`.
pub(crate) fn fuse_weight_norm(
    weights: &GgufWeights,
    prefix: &str,
    transpose_ct: bool,
) -> Result<Vec<f32>, WeightError> {
    let weight_name = format!("{prefix}.weight");
    if weights.contains(&weight_name) {
        let ne = weights.shape(&weight_name)?;
        return weights.f32_widened(&weight_name, &[ne[0], ne[1], ne[2]]);
    }
    let g_name = format!("{prefix}.weight_g");
    let v_name = format!("{prefix}.weight_v");
    if !weights.contains(&g_name) || !weights.contains(&v_name) {
        return Err(WeightError::MissingTensor {
            path: String::new(),
            name: format!("{prefix}.weight or {g_name}+{v_name}"),
        });
    }
    let v_ne = weights.shape(&v_name)?;
    let (dim0, n_el) = ggml_dim0_and_count(v_ne);
    let fan = n_el / dim0;
    let g = weights.f32_widened(&g_name, &[dim0])?;
    let v = weights.f32_widened(&v_name, &[v_ne[0], v_ne[1], v_ne[2]])?;
    if v.len() as i64 != n_el {
        return Err(WeightError::Shape {
            name: v_name,
            expected: vec![n_el],
            got: vec![v.len() as i64],
        });
    }
    let mut out = vec![0f32; v.len()];
    for d in 0..dim0 as usize {
        let gv = g.get(d).copied().unwrap_or(0.0);
        let mut nsq = 0f32;
        for i in 0..fan as usize {
            let vv = v[d * fan as usize + i];
            nsq += vv * vv;
        }
        let s = gv / (nsq.sqrt() + 1e-12);
        for i in 0..fan as usize {
            let vv = v[d * fan as usize + i];
            let fused = vv * s;
            if transpose_ct {
                out[i * dim0 as usize + d] = fused;
            } else {
                out[d * fan as usize + i] = fused;
            }
        }
    }
    Ok(out)
}

pub(crate) fn f32_row_to_f16(src: &[f32], dst_bytes: &mut [u8]) {
    debug_assert_eq!(src.len() * 2, dst_bytes.len());
    let mut f16 = vec![0u16; src.len()];
    unsafe {
        g::ggml_fp32_to_fp16_row(src.as_ptr(), f16.as_mut_ptr(), src.len() as i64);
    }
    for (i, v) in f16.iter().enumerate() {
        dst_bytes[i * 2] = (v & 0xff) as u8;
        dst_bytes[i * 2 + 1] = (v >> 8) as u8;
    }
}

/// Zero fused weights + neutral snake for layout-only smoke runs (no GGUF file).
pub fn load_smoke_zeros(session: &mut AceVaeSession) -> Result<(), WeightError> {
    for spec in graph_weight_layout(AceVaeConfig::smoke()) {
        let n = spec.ne[0] * spec.ne[1] * spec.ne[2] * spec.ne[3].max(1);
        if spec.ggml_type == TYPE_F16 {
            session.set_f16(&spec.name, &vec![0f32; n as usize])?;
        } else {
            let fill = if spec.name.ends_with(".exp_a") {
                vec![1f32; n as usize]
            } else if spec.name.ends_with(".inv_b") {
                vec![1f32; n as usize]
            } else {
                vec![0f32; n as usize]
            };
            session.set_f32(&spec.name, &fill)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        decode, graph_weight_layout, load_smoke_zeros, weight_layout, AceVaeConfig,
        AceVaeSession, AUDIO_UPSAMPLE, LATENT_CHANNELS,
    };
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd, GgufWeights};

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ace-vae-{name}-{}-{}",
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
    fn published_layout_lists_decoder_blocks() {
        let rows = weight_layout(AceVaeConfig::published());
        assert!(rows.iter().any(|r| r.name == "decoder.conv1.weight_v"));
        assert!(rows.iter().any(|r| r.name == "decoder.block.4.res_unit3.conv2.weight_v"));
        assert_eq!(graph_weight_layout(AceVaeConfig::published()).len(), 145);
    }

    #[test]
    fn smoke_decode_zero_latent_finite_planar_length() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let frames = 1i64;
        let mut session = AceVaeSession::build(&ctx, AceVaeConfig::smoke(), frames).unwrap();
        load_smoke_zeros(&mut session).unwrap();
        let latent = vec![0f32; frames as usize * LATENT_CHANNELS as usize];
        let audio = session.run(&latent, frames).unwrap();
        assert_eq!(audio.len(), (frames * AUDIO_UPSAMPLE * 2) as usize);
        assert!(audio.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn smoke_gguf_fused_weights_decode() {
        let dir = scratch("fused");
        let config = AceVaeConfig::smoke();
        let owned: Vec<(String, [i64; 4], Vec<f32>)> = graph_weight_layout(config)
            .into_iter()
            .map(|row| {
                let n = (row.ne[0] * row.ne[1] * row.ne[2] * row.ne[3].max(1)) as usize;
                let data = if row.name.ends_with(".exp_a") || row.name.ends_with(".inv_b") {
                    vec![1f32; n]
                } else {
                    vec![0f32; n]
                };
                (row.name, row.ne, data)
            })
            .collect();
        let tensors: Vec<F32Nd<'_>> = owned
            .iter()
            .map(|(name, ne, data)| F32Nd { name, ne: *ne, data })
            .collect();
        write_f32_nd(&dir.join("vae-smoke.gguf"), &tensors).unwrap();
        let weights = GgufWeights::open(&dir.join("vae-smoke.gguf")).unwrap();
        let latent = vec![0f32; LATENT_CHANNELS as usize];
        let audio = decode(&ctx_from_test(), &weights, &latent, 1).unwrap();
        assert_eq!(audio.len(), (AUDIO_UPSAMPLE * 2) as usize);
        assert!(audio.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn ctx_from_test() -> GgmlContext {
        GgmlContext::new(BackendKind::Cpu).unwrap()
    }
}
