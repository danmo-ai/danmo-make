//! LTX 2.3 video VAE decoder (`decoder.*` of the published video VAE
//! safetensors, BF16). Port of main's non-causal `LTX23VideoDecoder`
//! (`vae_mlx.py`):
//! - every conv is a 3x3x3 `Conv3dBlock` with symmetric replicate padding on
//!   the frame axis (frame 0 repeated in front, the last frame repeated at
//!   the back) and symmetric zero padding on the spatial axes;
//! - residual blocks are pre-activation:
//!   `x + conv2(silu(pixel_norm(conv1(silu(pixel_norm(x))))))` where
//!   `pixel_norm` is a weightless RMS norm across channels (eps 1e-8);
//! - the four DepthToSpaceUpsample convs are followed by pixel-shuffle
//!   `(sf, tf) = (2,2), (2,2), (1,2), (2,1)`; a temporal shuffle drops the
//!   first output frame, so `F` latent frames decode to `8F - 7`;
//! - the head is `pixel_norm -> silu -> conv_out` (48 channels) and a spatial
//!   unpatchify with patch 4 (channel order `(c, r_w, q_h)`, which differs
//!   from the shuffle order) yields the 3 output channels, 32x spatial;
//! - the latent is denormalized per channel first: `x * std + mean` with the
//!   top-level `per_channel_statistics.{mean,std}-of-means` tensors.
//!
//! PyTorch `[OC, IC, KD, KH, KW]` conv bytes double as the ggml
//! `[KW, KH, KD, IC * OC]` kernel without a repack; kernels are widened to
//! F32 because ggml's direct conv3d only takes F16/F32 kernels.

use crate::runtime::{shape, GgmlContext, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;

const GRAPH_MEM: usize = 256 << 20;
const PIXEL_NORM_EPS: f32 = 1e-8;
/// Residual blocks per ResStage (up_blocks 0 / 2 / 4 / 6 / 8).
const RES_COUNTS: [i64; 5] = [2, 2, 4, 6, 4];
/// (spatial, temporal) factors of the DepthToSpaceUpsample blocks 1/3/5/7.
const SHUFFLES: [(i64, i64); 4] = [(2, 2), (2, 2), (1, 2), (2, 1)];

pub(crate) const MEAN: &str = "per_channel_statistics.mean-of-means";
pub(crate) const STD: &str = "per_channel_statistics.std-of-means";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecoderConfig {
    /// Latent channels (128).
    pub latent: i64,
    /// conv_in output / stage 0 channels (1024).
    pub base: i64,
    /// Stage 2/4 channels (512).
    pub mid: i64,
    /// Stage 6 channels (256).
    pub low: i64,
    /// Stage 8 channels (128).
    pub tail: i64,
    /// conv_out output channels before the unpatchify (48).
    pub head: i64,
}

impl DecoderConfig {
    fn stage_channels(&self) -> [i64; 5] {
        [self.base, self.mid, self.mid, self.low, self.tail]
    }

    pub fn smoke() -> Self {
        Self {
            latent: 4,
            base: 8,
            mid: 8,
            low: 8,
            tail: 8,
            head: 48,
        }
    }

    /// Recover the architecture from the weight shapes. The structure is the
    /// published fixed one (res counts, shuffle factors); anything else fails
    /// loud.
    pub fn infer(w: &GgufWeights) -> Result<Self, WeightError> {
        let fail = |detail: String| WeightError::NotBuilt(format!("ltx video vae: {detail}"));
        let latent = w.shape(MEAN)?[0];
        if w.shape(STD)?[0] != latent || latent < 1 {
            return Err(fail(format!(
                "per-channel statistics disagree: mean {latent} std {}",
                w.shape(STD)?[0]
            )));
        }
        let conv_in = w.shape("decoder.conv_in.conv.weight")?;
        if conv_in[0..3] != [3, 3, 3] || conv_in[3] % latent != 0 {
            return Err(fail(format!(
                "decoder.conv_in.conv.weight shape {conv_in:?} is not [3, 3, 3, {latent} * base]"
            )));
        }
        let mut channels = vec![conv_in[3] / latent];
        for (stage, &count) in RES_COUNTS.iter().enumerate() {
            let ch = channels[stage];
            for block in 0..count {
                for conv in ["conv1", "conv2"] {
                    let name = format!(
                        "decoder.up_blocks.{}.res_blocks.{block}.{conv}.conv.weight",
                        stage * 2
                    );
                    let s = w.shape(&name)?;
                    if s != [3, 3, 3, ch * ch] {
                        return Err(fail(format!(
                            "{name} shape {s:?} is not [3, 3, 3, {ch} * {ch}]"
                        )));
                    }
                }
            }
            if stage < SHUFFLES.len() {
                let (sf, tf) = SHUFFLES[stage];
                let name = format!("decoder.up_blocks.{}.conv.conv.weight", stage * 2 + 1);
                let s = w.shape(&name)?;
                let div = sf * sf * tf;
                if s[0..3] != [3, 3, 3] || s[3] % (ch * div) != 0 {
                    return Err(fail(format!(
                        "{name} shape {s:?} does not fit a ({sf}, {tf}) shuffle from {ch} channels"
                    )));
                }
                channels.push(s[3] / (ch * div));
            }
        }
        if channels[1] != channels[2] {
            return Err(fail(format!(
                "stages 2 and 4 disagree on channels: {:?}",
                &channels[1..=2]
            )));
        }
        let conv_out = w.shape("decoder.conv_out.conv.weight")?;
        let tail = channels[4];
        if conv_out[0..3] != [3, 3, 3] || conv_out[3] % tail != 0 {
            return Err(fail(format!(
                "decoder.conv_out.conv.weight shape {conv_out:?} is not [3, 3, 3, {tail} * head]"
            )));
        }
        let head = conv_out[3] / tail;
        if head % 3 != 0 {
            return Err(fail(format!("conv_out head {head} is not 3 * patch^2")));
        }
        let patch2 = head / 3;
        let patch = (patch2 as f64).sqrt() as i64;
        if patch * patch != patch2 {
            return Err(fail(format!("conv_out head {head} is not 3 * patch^2")));
        }
        Ok(Self {
            latent,
            base: channels[0],
            mid: channels[1],
            low: channels[3],
            tail,
            head,
        })
    }

    /// Unpatchify patch: `head = 3 * patch^2`.
    fn patch(&self) -> i64 {
        ((self.head / 3) as f64).sqrt() as i64
    }
}

struct Conv {
    /// Module prefix; the tensors are `{prefix}.conv.{weight,bias}`.
    name: String,
    ic: i64,
    oc: i64,
}

fn convs(config: &DecoderConfig) -> Vec<Conv> {
    let stage_ch = config.stage_channels();
    let mut out = Vec::new();
    out.push(Conv {
        name: "decoder.conv_in".into(),
        ic: config.latent,
        oc: config.base,
    });
    for (stage, &ch) in stage_ch.iter().enumerate() {
        for block in 0..RES_COUNTS[stage] {
            for conv in ["conv1", "conv2"] {
                out.push(Conv {
                    name: format!("decoder.up_blocks.{}.res_blocks.{block}.{conv}", stage * 2),
                    ic: ch,
                    oc: ch,
                });
            }
        }
        if stage < SHUFFLES.len() {
            let (sf, tf) = SHUFFLES[stage];
            out.push(Conv {
                name: format!("decoder.up_blocks.{}.conv", stage * 2 + 1),
                ic: ch,
                oc: stage_ch[stage + 1] * sf * sf * tf,
            });
        }
    }
    out.push(Conv {
        name: "decoder.conv_out".into(),
        ic: config.tail,
        oc: config.head,
    });
    out
}

// ---------------------------------------------------------------------------
// graph
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
        .ok_or_else(|| RuntimeError::Message(format!("ltx video vae graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let actual = shape(tensor);
    if actual != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {actual:?}, the ltx video vae expects {ne:?}"
        )));
    }
    Ok(tensor)
}

/// Symmetric replicate pad on the frame axis: frame 0 once in front, the
/// last frame once at the back (`[w, h, d, c]` -> `[w, h, d + 2, c]`).
fn temporal_replicate_pad(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    d: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let nb1 = (w * 4) as usize;
    let nb2 = (w * h * 4) as usize;
    let nb3 = (w * h * d * 4) as usize;
    let first = nn(b.cont(b.view_4d(x, w, h, 1, c, nb1, nb2, nb3, 0)))?;
    let last = nn(b.cont(b.view_4d(x, w, h, 1, c, nb1, nb2, nb3, ((d - 1) * w * h * 4) as usize)))?;
    let y = nn(b.cont(nn(b.concat(first, x, 2))?))?;
    nn(b.cont(nn(b.concat(y, last, 2))?))
}

/// 3x3x3 conv, stride 1: replicate time pad + zero spatial pad 1.
#[allow(clippy::too_many_arguments)]
fn conv3d_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    d: i64,
    ic: i64,
    oc: i64,
) -> Result<Tensor, RuntimeError> {
    let x = temporal_replicate_pad(b, x, w, h, d, ic)?;
    let kernel = wt(weights, &format!("{prefix}.conv.weight"))?;
    let y = nn(b.conv3d_direct(
        kernel, x, 1, 1, 1, 1, 1, 0, 1, 1, 1, ic as i32, 1, oc as i32,
    ))?;
    let y = nn(b.cont(y))?;
    let y = shaped(y, [w, h, d, oc], &format!("{prefix}.conv.weight"))?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.conv.bias"))?, 1, 1, 1, oc)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    shaped(y, [w, h, d, oc], &format!("{prefix}.conv.bias"))
}

/// Weightless RMS norm across channels (ne3): move channels to ne0,
/// `rms_norm`, move back. `ggml_permute` uses destination semantics:
/// `ne[axis_i] = old ne[i]`.
fn pixel_norm(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    d: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.cont(b.permute(x, 1, 2, 3, 0)))?; // [c, w, h, d]
    let y = nn(b.rms_norm(y, PIXEL_NORM_EPS))?;
    let y = nn(b.cont(b.permute(y, 3, 0, 1, 2)))?; // [w, h, d, c]
    shaped(y, [w, h, d, c], "ltx video vae pixel_norm")
}

#[allow(clippy::too_many_arguments)]
fn res_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    w: i64,
    h: i64,
    d: i64,
    ch: i64,
) -> Result<Tensor, RuntimeError> {
    let y = pixel_norm(b, x, w, h, d, ch)?;
    let y = nn(b.silu(y))?;
    let y = conv3d_block(b, weights, y, &format!("{prefix}.conv1"), w, h, d, ch, ch)?;
    let y = pixel_norm(b, y, w, h, d, ch)?;
    let y = nn(b.silu(y))?;
    let y = conv3d_block(b, weights, y, &format!("{prefix}.conv2"), w, h, d, ch, ch)?;
    nn(b.add(x, y))
}

/// Depth-to-space: `out[w*sf+pw, h*sf+ph, d*tf+t, c] =
/// in[w, h, d, c*tf*sf^2 + t*sf^2 + ph*sf + pw]`. A temporal shuffle also
/// drops the first output frame (main drops it unconditionally on tf > 1).
#[allow(clippy::too_many_arguments)]
fn pixel_shuffle(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    d: i64,
    c: i64,
    sf: i64,
    tf: i64,
) -> Result<(Tensor, [i64; 4]), RuntimeError> {
    let y = nn(b.reshape_4d(x, w, h * d, sf, c * tf * sf))?;
    let y = nn(b.cont(b.permute(y, 1, 2, 0, 3)))?; // [pw, w, hd, c tf sf]
    let y = nn(b.reshape_3d(y, w * sf, h * d, c * tf * sf))?;
    let y = nn(b.reshape_4d(y, w * sf, h * d, sf, c * tf))?;
    let y = nn(b.cont(b.permute(y, 0, 2, 1, 3)))?; // [w sf, ph, hd, c tf]
    let y = nn(b.reshape_4d(y, w * sf, h * sf, d, c * tf))?;
    let y = nn(b.reshape_4d(y, w * sf * h * sf, d, tf, c))?;
    let y = nn(b.cont(b.permute(y, 0, 2, 1, 3)))?; // [wshs, t, d, c]
    let y = nn(b.reshape_4d(y, w * sf, h * sf, d * tf, c))?;
    let y = nn(b.cont(y))?;
    if tf > 1 {
        let out_d = d * tf - 1;
        let view = nn(b.view_4d(
            y,
            w * sf,
            h * sf,
            out_d,
            c,
            (w * sf * 4) as usize,
            (w * sf * h * sf * 4) as usize,
            (w * sf * h * sf * d * tf * 4) as usize,
            (w * sf * h * sf * 4) as usize,
        ))?;
        let y = nn(b.cont(view))?;
        return Ok((y, [w * sf, h * sf, out_d, c]));
    }
    Ok((y, [w * sf, h * sf, d * tf, c]))
}

/// Spatial unpatchify: `out[w*ps+r, h*ps+q, f, c] =
/// in[w, h, f, c*ps^2 + r*ps + q]` (channel order `(c, r_w, q_h)`).
fn unpatchify(
    b: &GraphBuilder,
    x: Tensor,
    w: i64,
    h: i64,
    d: i64,
    head: i64,
    ps: i64,
) -> Result<Tensor, RuntimeError> {
    let c = head / (ps * ps);
    let y = nn(b.reshape_4d(x, w, h * d, ps, ps * c))?;
    let y = nn(b.cont(b.permute(y, 0, 2, 1, 3)))?; // [w, q, hd, r c]
    let y = nn(b.reshape_4d(y, w, h * ps, d, ps * c))?;
    let y = nn(b.reshape_4d(y, w, h * ps * d, ps, c))?;
    let y = nn(b.cont(b.permute(y, 1, 2, 0, 3)))?; // [r, w, hdq, c]
    let y = nn(b.reshape_4d(y, w * ps, h * ps, d, c))?;
    nn(b.cont(y))
}

// ---------------------------------------------------------------------------
// decoder
// ---------------------------------------------------------------------------

pub struct LtxVideoDecoder {
    config: DecoderConfig,
    loaded: Vec<(String, Vec<f32>)>,
}

impl LtxVideoDecoder {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let config = DecoderConfig::infer(&weights)?;
        let mut loaded = Vec::new();
        for conv in convs(&config) {
            let weight = format!("{}.conv.weight", conv.name);
            let data = weights.f32_widened(&weight, &[3, 3, 3, conv.ic * conv.oc])?;
            loaded.push((weight, data));
            let bias = format!("{}.conv.bias", conv.name);
            let data = weights.f32_widened(&bias, &[conv.oc])?;
            loaded.push((bias, data));
        }
        for name in [MEAN, STD] {
            let data = weights.f32_widened(name, &[config.latent])?;
            loaded.push((name.to_string(), data));
        }
        Ok(Self { config, loaded })
    }

    pub fn config(&self) -> DecoderConfig {
        self.config
    }

    /// `(width, height, frames)` of the decoded pixels for a `w x h x f`
    /// latent.
    pub fn output_dims(&self, w: i64, h: i64, f: i64) -> (i64, i64, i64) {
        let mut frames = f;
        for &(_, tf) in &SHUFFLES {
            frames *= tf;
            if tf > 1 {
                frames -= 1;
            }
        }
        let spatial: i64 =
            SHUFFLES.iter().map(|&(sf, _)| sf).product::<i64>() * self.config.patch();
        (w * spatial, h * spatial, frames)
    }

    /// Decode a `[w, h, f, latent]` latent (ggml order) into
    /// `[w * 32, h * 32, 8f - 7, 3]` pixels in ggml order.
    pub fn decode(
        &self,
        ctx: &GgmlContext,
        latent: &[f32],
        w: i64,
        h: i64,
        f: i64,
    ) -> Result<Vec<f32>, WeightError> {
        let config = &self.config;
        if w < 1 || h < 1 || f < 1 {
            return Err(WeightError::Shape {
                name: "ltx video vae decode dims".into(),
                expected: vec![1, 1, 1],
                got: vec![w, h, f],
            });
        }
        if latent.len() as i64 != w * h * f * config.latent {
            return Err(WeightError::Shape {
                name: "ltx video vae latent".into(),
                expected: vec![w, h, f, config.latent],
                got: vec![latent.len() as i64],
            });
        }
        let mut b = ctx.builder(GRAPH_MEM)?;
        let mut wmap = HashMap::new();
        for conv in convs(config) {
            let weight = format!("{}.conv.weight", conv.name);
            wmap.insert(
                weight.clone(),
                nn(b.f32_4d(&weight, 3, 3, 3, conv.ic * conv.oc, true))?,
            );
            let bias = format!("{}.conv.bias", conv.name);
            wmap.insert(bias.clone(), nn(b.f32_1d(&bias, conv.oc, true))?);
        }
        for name in [MEAN, STD] {
            wmap.insert(name.to_string(), nn(b.f32_1d(name, config.latent, true))?);
        }
        let z = nn(b.f32_4d("ltx_vae_latent", w, h, f, config.latent, true))?;
        // Denormalize: x * std + mean per channel.
        let std = nn(b.cont(b.reshape_4d(wt(&wmap, STD)?, 1, 1, 1, config.latent)))?;
        let mean = nn(b.cont(b.reshape_4d(wt(&wmap, MEAN)?, 1, 1, 1, config.latent)))?;
        let z = nn(b.mul(z, nn(b.repeat(std, z))?))?;
        let mut x = nn(b.add(z, nn(b.repeat(mean, z))?))?;

        let stage_ch = config.stage_channels();
        let (mut cw, mut ch, mut cd) = (w, h, f);
        x = conv3d_block(
            &b,
            &wmap,
            x,
            "decoder.conv_in",
            cw,
            ch,
            cd,
            config.latent,
            config.base,
        )?;
        for (stage, &channels) in stage_ch.iter().enumerate() {
            for block in 0..RES_COUNTS[stage] {
                x = res_block(
                    &b,
                    &wmap,
                    x,
                    &format!("decoder.up_blocks.{}.res_blocks.{block}", stage * 2),
                    cw,
                    ch,
                    cd,
                    channels,
                )?;
            }
            if stage < SHUFFLES.len() {
                let (sf, tf) = SHUFFLES[stage];
                let oc = stage_ch[stage + 1] * sf * sf * tf;
                x = conv3d_block(
                    &b,
                    &wmap,
                    x,
                    &format!("decoder.up_blocks.{}.conv", stage * 2 + 1),
                    cw,
                    ch,
                    cd,
                    channels,
                    oc,
                )?;
                let (y, dims) = pixel_shuffle(&b, x, cw, ch, cd, stage_ch[stage + 1], sf, tf)?;
                x = y;
                cw = dims[0];
                ch = dims[1];
                cd = dims[2];
            }
        }
        x = pixel_norm(&b, x, cw, ch, cd, config.tail)?;
        x = nn(b.silu(x))?;
        x = conv3d_block(
            &b,
            &wmap,
            x,
            "decoder.conv_out",
            cw,
            ch,
            cd,
            config.tail,
            config.head,
        )?;
        let x = unpatchify(&b, x, cw, ch, cd, config.head, config.patch())?;
        let mut graph = b.finish(x)?;
        for (name, data) in &self.loaded {
            graph.set_f32(name, data)?;
        }
        graph.set_f32("ltx_vae_latent", latent)?;
        let out = graph.compute()?;
        if let Some(detail) = crate::common::ops::non_finite("ltx video vae", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// smoke writer (graph-vs-host tests)
// ---------------------------------------------------------------------------

/// Deterministic smoke weights for `DecoderConfig::smoke()`.
pub fn smoke_store(config: &DecoderConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 41u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let mut out = Vec::new();
    for conv in convs(config) {
        let count = (27 * conv.ic * conv.oc) as usize;
        out.push((
            format!("{}.conv.weight", conv.name),
            (0..count).map(|_| next() * 0.1).collect(),
        ));
        out.push((
            format!("{}.conv.bias", conv.name),
            (0..conv.oc).map(|_| next() * 0.1).collect(),
        ));
    }
    out.push((
        MEAN.to_string(),
        (0..config.latent).map(|_| next() * 0.2).collect(),
    ));
    out.push((
        STD.to_string(),
        (0..config.latent).map(|_| 0.75 + next() * 0.25).collect(),
    ));
    out
}

/// Write the smoke decoder as a safetensors file (PyTorch shapes).
pub fn write_smoke(dir: &std::path::Path) -> Result<PathBuf, WeightError> {
    write_smoke_with(dir, &DecoderConfig::smoke())
}

/// Write the smoke video VAE for an explicit config (session smoke bundle
/// shares the 128-channel latent with the DiT).
pub fn write_smoke_with(
    dir: &std::path::Path,
    config: &DecoderConfig,
) -> Result<PathBuf, WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = *config;
    let store = smoke_store(&config);
    let conv_list = convs(&config);
    let mut shapes: Vec<Vec<u64>> = Vec::new();
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for (name, values) in &store {
        let shape: Vec<u64> = if name.ends_with(".conv.weight") {
            let conv = conv_list
                .iter()
                .find(|conv| format!("{}.conv.weight", conv.name) == *name)
                .expect("smoke store and layout agree");
            vec![conv.oc as u64, conv.ic as u64, 3, 3, 3]
        } else {
            vec![values.len() as u64]
        };
        shapes.push(shape);
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(data);
    }
    let tensors: Vec<crate::weights::SafeTensor<'_>> = store
        .iter()
        .zip(&shapes)
        .zip(&bytes)
        .map(|(((name, _), shape), data)| crate::weights::SafeTensor {
            name,
            dtype: "F32",
            shape,
            data,
        })
        .collect();
    let path = dir.join("ltx_video_vae.safetensors");
    crate::weights::write_safetensors(&path, &tensors)?;
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

    fn pixel_norm_host(x: &mut [f32], w: usize, h: usize, d: usize, c: usize) {
        for dd in 0..d {
            for hh in 0..h {
                for ww in 0..w {
                    let base = ww + w * (hh + h * dd);
                    let mut sum = 0f32;
                    for cc in 0..c {
                        let v = x[base + w * h * d * cc];
                        sum += v * v;
                    }
                    let scale = 1.0 / (sum / c as f32 + PIXEL_NORM_EPS).sqrt();
                    for cc in 0..c {
                        x[base + w * h * d * cc] *= scale;
                    }
                }
            }
        }
    }

    /// 3x3x3 conv: replicate pad on the frame axis, zero pad on space.
    #[allow(clippy::too_many_arguments)]
    fn conv3d_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        ic: usize,
        oc: usize,
    ) -> Vec<f32> {
        let weight = at(store, &format!("{prefix}.conv.weight"));
        let bias = at(store, &format!("{prefix}.conv.bias"));
        let mut out = vec![0f32; w * h * d * oc];
        for o in 0..oc {
            for dd in 0..d {
                for hh in 0..h {
                    for ww in 0..w {
                        let mut acc = bias[o];
                        for kd in 0..3i64 {
                            for kh in 0..3i64 {
                                for kw in 0..3i64 {
                                    let sd = (dd as i64 + kd - 1).clamp(0, d as i64 - 1) as usize;
                                    let sh = hh as i64 + kh - 1;
                                    let sw = ww as i64 + kw - 1;
                                    if sh < 0 || sh >= h as i64 || sw < 0 || sw >= w as i64 {
                                        continue;
                                    }
                                    let (sh, sw) = (sh as usize, sw as usize);
                                    let tap = (kw + 3 * (kh + 3 * kd)) as usize;
                                    for i in 0..ic {
                                        acc += weight[tap + 27 * (i + ic * o)]
                                            * x[sw + w * (sh + h * (sd + d * i))];
                                    }
                                }
                            }
                        }
                        out[ww + w * (hh + h * (dd + d * o))] = acc;
                    }
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
        d: usize,
        ch: usize,
    ) -> Vec<f32> {
        let mut y = x.to_vec();
        pixel_norm_host(&mut y, w, h, d, ch);
        for v in y.iter_mut() {
            *v = silu_host(*v);
        }
        let mut y = conv3d_host(store, &format!("{prefix}.conv1"), &y, w, h, d, ch, ch);
        pixel_norm_host(&mut y, w, h, d, ch);
        for v in y.iter_mut() {
            *v = silu_host(*v);
        }
        let y = conv3d_host(store, &format!("{prefix}.conv2"), &y, w, h, d, ch, ch);
        x.iter().zip(&y).map(|(a, b)| a + b).collect()
    }

    /// Returns the shuffled tensor and its `[w, h, d]` (channels known).
    fn pixel_shuffle_host(
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        c: usize,
        sf: usize,
        tf: usize,
    ) -> (Vec<f32>, usize, usize, usize) {
        let (ow, oh, od) = (w * sf, h * sf, d * tf);
        let mut out = vec![0f32; ow * oh * od * c];
        for cc in 0..c {
            for dd in 0..d {
                for t in 0..tf {
                    for hh in 0..h {
                        for ph in 0..sf {
                            for ww in 0..w {
                                for pw in 0..sf {
                                    let cin = cc * tf * sf * sf + t * sf * sf + ph * sf + pw;
                                    let src = ww + w * (hh + h * (dd + d * cin));
                                    let dst = (ww * sf + pw)
                                        + ow * ((hh * sf + ph) + oh * ((dd * tf + t) + od * cc));
                                    out[dst] = x[src];
                                }
                            }
                        }
                    }
                }
            }
        }
        if tf > 1 {
            // drop the first frame
            let frame = ow * oh;
            let mut kept = vec![0f32; frame * (od - 1) * c];
            for cc in 0..c {
                kept[cc * frame * (od - 1)..(cc + 1) * frame * (od - 1)]
                    .copy_from_slice(&out[cc * frame * od + frame..(cc + 1) * frame * od]);
            }
            return (kept, ow, oh, od - 1);
        }
        (out, ow, oh, od)
    }

    fn unpatchify_host(
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        head: usize,
        ps: usize,
    ) -> Vec<f32> {
        let c = head / (ps * ps);
        let (ow, oh) = (w * ps, h * ps);
        let mut out = vec![0f32; ow * oh * d * c];
        for cc in 0..c {
            for dd in 0..d {
                for hh in 0..h {
                    for q in 0..ps {
                        for ww in 0..w {
                            for r in 0..ps {
                                let cin = cc * ps * ps + r * ps + q;
                                let src = ww + w * (hh + h * (dd + d * cin));
                                let dst = (ww * ps + r) + ow * ((hh * ps + q) + oh * (dd + d * cc));
                                out[dst] = x[src];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn host_decode(
        store: &[(String, Vec<f32>)],
        config: &DecoderConfig,
        latent: &[f32],
        w: usize,
        h: usize,
        f: usize,
    ) -> Vec<f32> {
        let latent_c = config.latent as usize;
        let mean = at(store, MEAN);
        let std = at(store, STD);
        let mut x: Vec<f32> = latent
            .iter()
            .enumerate()
            .map(|(i, v)| v * std[i / (w * h * f) % latent_c] + mean[i / (w * h * f) % latent_c])
            .collect();
        let stage_ch: Vec<usize> = config
            .stage_channels()
            .iter()
            .map(|c| *c as usize)
            .collect();
        let (mut cw, mut ch, mut cd) = (w, h, f);
        x = conv3d_host(
            store,
            "decoder.conv_in",
            &x,
            cw,
            ch,
            cd,
            latent_c,
            config.base as usize,
        );
        for (stage, &channels) in stage_ch.iter().enumerate() {
            for block in 0..RES_COUNTS[stage] {
                x = res_block_host(
                    store,
                    &format!("decoder.up_blocks.{}.res_blocks.{block}", stage * 2),
                    &x,
                    cw,
                    ch,
                    cd,
                    channels,
                );
            }
            if stage < SHUFFLES.len() {
                let (sf, tf) = SHUFFLES[stage];
                let oc = stage_ch[stage + 1] * (sf * sf * tf) as usize;
                x = conv3d_host(
                    store,
                    &format!("decoder.up_blocks.{}.conv", stage * 2 + 1),
                    &x,
                    cw,
                    ch,
                    cd,
                    channels,
                    oc,
                );
                let (y, nw, nh, nd) = pixel_shuffle_host(
                    &x,
                    cw,
                    ch,
                    cd,
                    stage_ch[stage + 1],
                    sf as usize,
                    tf as usize,
                );
                x = y;
                cw = nw;
                ch = nh;
                cd = nd;
            }
        }
        pixel_norm_host(&mut x, cw, ch, cd, config.tail as usize);
        for v in x.iter_mut() {
            *v = silu_host(*v);
        }
        let x = conv3d_host(
            store,
            "decoder.conv_out",
            &x,
            cw,
            ch,
            cd,
            config.tail as usize,
            config.head as usize,
        );
        unpatchify_host(
            &x,
            cw,
            ch,
            cd,
            config.head as usize,
            config.patch() as usize,
        )
    }

    #[test]
    fn graph_matches_the_host_reference() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-vae-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_smoke(&dir).unwrap();
        let decoder = LtxVideoDecoder::open(&[path]).unwrap();
        assert_eq!(decoder.config(), DecoderConfig::smoke());

        let (w, h, f) = (3i64, 2i64, 3i64);
        let (ow, oh, of) = decoder.output_dims(w, h, f);
        assert_eq!((ow, oh, of), (96, 64, 17));
        let config = decoder.config();
        let mut seed = 7u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        let latent: Vec<f32> = (0..(w * h * f * config.latent) as usize)
            .map(|_| next())
            .collect();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let out = decoder.decode(&ctx, &latent, w, h, f).unwrap();
        assert_eq!(out.len(), (ow * oh * of * 3) as usize);

        let store = smoke_store(&DecoderConfig::smoke());
        let host = host_decode(&store, &config, &latent, w as usize, h as usize, f as usize);
        assert_eq!(host.len(), out.len());
        let mut max_diff = 0f32;
        for (g, e) in out.iter().zip(&host) {
            max_diff = max_diff.max((g - e).abs());
        }
        assert!(max_diff < 2e-3, "graph vs host max diff {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_fails_loud_on_a_missing_conv_in() {
        let dir = std::env::temp_dir().join(format!("danmo-ltx-vae-empty-{}", std::process::id()));
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
        let err = match LtxVideoDecoder::open(&[path]) {
            Err(err) => err,
            Ok(_) => panic!("accepted empty weights"),
        };
        assert!(
            err.to_string().contains("per_channel_statistics")
                || err.to_string().contains("mean-of-means"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Standalone coverage for the two rearrangement chains: they are the
/// easiest place to swap a sub-pixel index without the host reference
/// noticing at the full-decoder level.
#[cfg(test)]
mod rearrange_tests {
    use super::*;
    use crate::runtime::BackendKind;

    #[test]
    fn pixel_shuffle_chain_matches_the_reference_mapping() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (w, h, d, c, sf, tf) = (2i64, 3i64, 2i64, 2i64, 2i64, 2i64);
        let total = c * tf * sf * sf;
        let mut b = ctx.builder(16 << 20).unwrap();
        let x = nn(b.f32_4d("x", w, h, d, total, true)).unwrap();
        let (y, dims) = pixel_shuffle(&b, x, w, h, d, c, sf, tf).unwrap();
        let mut g = b.finish(y).unwrap();
        let input: Vec<f32> = (0..(w * h * d * total) as usize)
            .map(|i| i as f32)
            .collect();
        g.set_f32("x", &input).unwrap();
        let out = g.compute().unwrap();
        let (ow, oh, od) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
        println!("dims {dims:?}");
        // host mapping
        let (ww, hh, dd, cc, sfu, tfu) = (
            w as usize,
            h as usize,
            d as usize,
            c as usize,
            sf as usize,
            tf as usize,
        );
        let (fw, fh, fd) = (ww * sfu, hh * sfu, dd * tfu);
        let mut host = vec![0f32; fw * fh * fd * cc];
        for c0 in 0..cc {
            for d0 in 0..dd {
                for t in 0..tfu {
                    for h0 in 0..hh {
                        for ph in 0..sfu {
                            for w0 in 0..ww {
                                for pw in 0..sfu {
                                    let cin = c0 * tfu * sfu * sfu + t * sfu * sfu + ph * sfu + pw;
                                    let src = w0 + ww * (h0 + hh * (d0 + dd * cin));
                                    let dst = (w0 * sfu + pw)
                                        + fw * ((h0 * sfu + ph) + fh * ((d0 * tfu + t) + fd * c0));
                                    host[dst] = input[src] as f32;
                                }
                            }
                        }
                    }
                }
            }
        }
        // drop first frame
        let frame = fw * fh;
        let mut kept = vec![0f32; frame * (fd - 1) * cc];
        for c0 in 0..cc {
            kept[c0 * frame * (fd - 1)..(c0 + 1) * frame * (fd - 1)]
                .copy_from_slice(&host[c0 * frame * fd + frame..(c0 + 1) * frame * fd]);
        }
        assert_eq!(out.len(), kept.len(), "len {} vs {}", out.len(), kept.len());
        let mut max_diff = 0f32;
        let mut bad = 0;
        for (i, (g0, h0)) in out.iter().zip(&kept).enumerate() {
            let diff = (g0 - h0).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.0 && bad < 10 {
                let (c0, rem) = (i / (ow * oh * od), i % (ow * oh * od));
                let d0 = rem / (ow * oh);
                let h0i = rem % (ow * oh) / ow;
                let w0 = rem % ow;
                println!("mismatch at w{w0} h{h0i} d{d0} c{c0}: graph {g0} host {h0}");
                bad += 1;
            }
        }
        assert!(max_diff == 0.0, "pixel_shuffle chain max diff {max_diff}");
    }

    #[test]
    fn unpatchify_chain_matches_the_reference_mapping() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (w, h, d, head, ps) = (2i64, 3i64, 2i64, 48i64, 4i64);
        let mut b = ctx.builder(16 << 20).unwrap();
        let x = nn(b.f32_4d("x", w, h, d, head, true)).unwrap();
        let y = unpatchify(&b, x, w, h, d, head, ps).unwrap();
        let dims = shape(y);
        let mut g = b.finish(y).unwrap();
        let input: Vec<f32> = (0..(w * h * d * head) as usize).map(|i| i as f32).collect();
        g.set_f32("x", &input).unwrap();
        let out = g.compute().unwrap();
        let (ow, oh, od, oc) = (
            dims[0] as usize,
            dims[1] as usize,
            dims[2] as usize,
            dims[3] as usize,
        );
        println!("dims {dims:?}");
        let (ww, hh, dd, psu) = (w as usize, h as usize, d as usize, ps as usize);
        let c = head as usize / (psu * psu);
        let (fw, fh) = (ww * psu, hh * psu);
        let mut host = vec![0f32; fw * fh * dd * c];
        for cc in 0..c {
            for d0 in 0..dd {
                for h0 in 0..hh {
                    for q in 0..psu {
                        for w0 in 0..ww {
                            for r in 0..psu {
                                let cin = cc * psu * psu + r * psu + q;
                                let src = w0 + ww * (h0 + hh * (d0 + dd * cin));
                                let dst =
                                    (w0 * psu + r) + fw * ((h0 * psu + q) + fh * (d0 + dd * cc));
                                host[dst] = input[src] as f32;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(out.len(), host.len());
        let mut max_diff = 0f32;
        let mut bad = 0;
        for (i, (g0, h0)) in out.iter().zip(&host).enumerate() {
            let diff = (g0 - h0).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.0 && bad < 10 {
                let (c0, rem) = (i / (ow * oh * od), i % (ow * oh * od));
                let d0 = rem / (ow * oh);
                let h0i = rem % (ow * oh) / ow;
                let w0 = rem % ow;
                println!("mismatch at w{w0} h{h0i} d{d0} c{c0}: graph {g0} host {h0}");
                bad += 1;
            }
        }
        assert!(max_diff == 0.0, "unpatchify chain max diff {max_diff}");
    }
}
