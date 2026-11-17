//! LTX 2.3 latent upsampler (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`,
//! BF16). Port of main's `LTX23LatentUpsampler` spatial-x2 variant
//! (`vae_mlx.py`), used between the two dev stages:
//! - `initial_conv` (3x3x3 Conv3d, zero pad 1 on every axis) ->
//!   `initial_norm` (GroupNorm) -> SiLU;
//! - 4 pre-upsample ResBlocks:
//!   `conv1 -> norm1 -> SiLU -> conv2 -> norm2 -> SiLU(x + residual)`;
//! - a per-frame Conv2d (zero pad 1) to `4 * mid` channels followed by a
//!   2x pixel shuffle with the `(c, p1, p2)` channel order (`p1` grows H,
//!   `p2` grows W), doubling H and W;
//! - 4 post-upsample ResBlocks and `final_conv` back to the latent channels.
//!
//! GroupNorm normalizes the whole clip as one sample (main runs the
//! upsampler on a `(1, F, H, W, C)` tensor and MLX GroupNorm reduces over
//! every spatial axis), so channels move to ne2 with the folded
//! `[w * h * d, 1, c, 1]` layout instead of the per-frame batch layout.
//!
//! PyTorch `[OC, IC, KD, KH, KW]` conv3d bytes double as the ggml
//! `[KW, KH, KD, IC * OC]` kernel; the Conv2d `[OC, IC, KH, KW]` bytes
//! double as `[KW, KH, IC, OC]`. Everything is widened to F32.

use crate::runtime::{shape, GgmlContext, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;

const GRAPH_MEM: usize = 256 << 20;
const NORM_EPS: f32 = 1e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UpscalerConfig {
    /// Latent channels in and out (128).
    pub latent: i64,
    /// Hidden channels (1024).
    pub mid: i64,
    /// ResBlocks per stage (4).
    pub blocks: i64,
    /// GroupNorm groups (32).
    pub groups: i32,
}

impl UpscalerConfig {
    pub fn smoke() -> Self {
        Self {
            latent: 2,
            mid: 32,
            blocks: 1,
            groups: 32,
        }
    }

    /// Recover the architecture from the weight shapes. Only the spatial x2
    /// variant (`upsampler.0.*` Conv2d) is supported; the rational x1.5 and
    /// temporal variants fail loud.
    pub fn infer(w: &GgufWeights) -> Result<Self, WeightError> {
        let fail = |detail: String| WeightError::NotBuilt(format!("ltx upscaler: {detail}"));
        let initial = w.shape("initial_conv.weight")?;
        if initial[0..3] != [3, 3, 3] {
            return Err(fail(format!(
                "initial_conv.weight shape {initial:?} is not a 3x3x3 conv"
            )));
        }
        let mid = w.shape("initial_conv.bias")?[0];
        let latent = w.shape("final_conv.bias")?[0];
        if initial[3] != latent * mid || latent < 1 || mid < 1 {
            return Err(fail(format!(
                "initial_conv.weight shape {initial:?} is not [3, 3, 3, {latent} * {mid}]"
            )));
        }
        let final_conv = w.shape("final_conv.weight")?;
        if final_conv != [3, 3, 3, mid * latent] {
            return Err(fail(format!(
                "final_conv.weight shape {final_conv:?} is not [3, 3, 3, {mid} * {latent}]"
            )));
        }
        let up = w.shape("upsampler.0.weight")?;
        if up != [3, 3, mid, 4 * mid] {
            return Err(fail(format!(
                "upsampler.0.weight shape {up:?} is not the spatial x2 [3, 3, {mid}, {}]",
                4 * mid
            )));
        }
        if w.shape("upsampler.0.bias")?[0] != 4 * mid {
            return Err(fail("upsampler.0.bias length mismatch".into()));
        }
        let mut blocks = 0i64;
        while w.contains(&format!("res_blocks.{blocks}.conv1.weight")) {
            blocks += 1;
        }
        if blocks == 0 {
            return Err(fail("no res_blocks".into()));
        }
        for stage in ["res_blocks", "post_upsample_res_blocks"] {
            for block in 0..blocks {
                for conv in ["conv1", "conv2"] {
                    let name = format!("{stage}.{block}.{conv}.weight");
                    let s = w.shape(&name)?;
                    if s != [3, 3, 3, mid * mid] {
                        return Err(fail(format!(
                            "{name} shape {s:?} is not [3, 3, 3, {mid} * {mid}]"
                        )));
                    }
                }
                for norm in ["norm1", "norm2"] {
                    for param in ["weight", "bias"] {
                        let name = format!("{stage}.{block}.{norm}.{param}");
                        if w.shape(&name)?[0] != mid {
                            return Err(fail(format!("{name} is not [{mid}]")));
                        }
                    }
                }
            }
        }
        for param in ["weight", "bias"] {
            if w.shape(&format!("initial_norm.{param}"))?[0] != mid {
                return Err(fail(format!("initial_norm.{param} is not [{mid}]")));
            }
        }
        let groups = if mid % 32 == 0 {
            32
        } else if mid % 16 == 0 {
            16
        } else {
            return Err(fail(format!(
                "mid channels {mid} support neither 32 nor 16 norm groups"
            )));
        };
        Ok(Self {
            latent,
            mid,
            blocks,
            groups,
        })
    }
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
        .ok_or_else(|| RuntimeError::Message(format!("ltx upscaler graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let actual = shape(tensor);
    if actual != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {actual:?}, the ltx upscaler expects {ne:?}"
        )));
    }
    Ok(tensor)
}

/// 3x3x3 conv, stride 1, zero pad 1 on every axis: `[w, h, d, ic]` stays
/// `[w, h, d, oc]`.
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
    let kernel = wt(weights, &format!("{prefix}.weight"))?;
    let y = nn(b.conv3d_direct(
        kernel, x, 1, 1, 1, 1, 1, 1, 1, 1, 1, ic as i32, 1, oc as i32,
    ))?;
    let y = nn(b.cont(y))?;
    let y = shaped(y, [w, h, d, oc], &format!("{prefix}.weight"))?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.bias"))?, 1, 1, 1, oc)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    shaped(y, [w, h, d, oc], &format!("{prefix}.bias"))
}

/// GroupNorm with the whole clip as one sample (main's MLX GroupNorm
/// reduces over every spatial axis): fold `[w, h, d, c]` into
/// `[w * h * d, 1, c, 1]` so ggml's per-ne3 grouping sees a single sample.
fn group_norm_clip(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    groups: i32,
    w: i64,
    h: i64,
    d: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.cont(b.reshape_4d(x, w * h * d, 1, c, 1)))?;
    let y = nn(b.group_norm(y, groups, NORM_EPS))?;
    let scale = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.weight"))?, 1, 1, c, 1)))?;
    let y = nn(b.mul(y, nn(b.repeat(scale, y))?))?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, &format!("{prefix}.bias"))?, 1, 1, c, 1)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    let y = nn(b.cont(b.reshape_4d(y, w, h, d, c)))?;
    shaped(y, [w, h, d, c], &format!("{prefix}.weight"))
}

/// `conv1 -> norm1 -> SiLU -> conv2 -> norm2 -> SiLU(x + residual)`.
#[allow(clippy::too_many_arguments)]
fn res_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    prefix: &str,
    groups: i32,
    w: i64,
    h: i64,
    d: i64,
    ch: i64,
) -> Result<Tensor, RuntimeError> {
    let y = conv3d_block(b, weights, x, &format!("{prefix}.conv1"), w, h, d, ch, ch)?;
    let y = group_norm_clip(
        b,
        weights,
        y,
        &format!("{prefix}.norm1"),
        groups,
        w,
        h,
        d,
        ch,
    )?;
    let y = nn(b.silu(y))?;
    let y = conv3d_block(b, weights, y, &format!("{prefix}.conv2"), w, h, d, ch, ch)?;
    let y = group_norm_clip(
        b,
        weights,
        y,
        &format!("{prefix}.norm2"),
        groups,
        w,
        h,
        d,
        ch,
    )?;
    nn(b.silu(nn(b.add(x, y))?))
}

/// Per-frame Conv2d to `4 * mid` channels plus the 2x pixel shuffle:
/// `out[w * 2 + p2, h * 2 + p1, c, n] = in[w, h, c * 4 + p1 * 2 + p2, n]`.
/// The clip runs as the conv batch (frames on ne3), then the layout returns
/// to `[2w, 2h, d, c]`.
#[allow(clippy::too_many_arguments)]
fn spatial_x2(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    w: i64,
    h: i64,
    d: i64,
    c: i64,
) -> Result<Tensor, RuntimeError> {
    // [w, h, d, c] -> [w, h, c, d]: frames become the conv batch.
    let y = nn(b.cont(b.permute(x, 0, 1, 3, 2)))?;
    let kernel = wt(weights, "upsampler.0.weight")?;
    let y = nn(b.conv2d_direct(kernel, y, 1, 1, 1, 1, 1, 1))?;
    let y = nn(b.cont(y))?;
    let y = shaped(y, [w, h, 4 * c, d], "upsampler.0.weight")?;
    let bias = nn(b.cont(b.reshape_4d(wt(weights, "upsampler.0.bias")?, 1, 1, 4 * c, 1)))?;
    let y = nn(b.add(y, nn(b.repeat(bias, y))?))?;
    // Pixel shuffle. ggml_permute uses destination semantics:
    // `ne[axis_i] = old ne[i]`.
    let y = nn(b.reshape_4d(y, w, h, 2, 2 * c * d))?; // ne2 = p2
    let y = nn(b.cont(b.permute(y, 1, 2, 0, 3)))?; // [p2, w, h, p1 c n]
    let y = nn(b.reshape_4d(y, w * 2, h, 2, c * d))?; // ne0 = (w, p2)
    let y = nn(b.cont(b.permute(y, 0, 2, 1, 3)))?; // [w*2, p1, h, c n]
    let y = nn(b.reshape_4d(y, w * 2, h * 2, c, d))?;
    // [2w, 2h, c, d] -> [2w, 2h, d, c].
    let y = nn(b.cont(b.permute(y, 0, 1, 3, 2)))?;
    shaped(y, [w * 2, h * 2, d, c], "ltx upscaler pixel shuffle")
}

// ---------------------------------------------------------------------------
// upsampler
// ---------------------------------------------------------------------------

fn graph_conv3d(
    b: &mut GraphBuilder,
    wmap: &mut HashMap<String, Tensor>,
    name: &str,
    ic: i64,
    oc: i64,
) -> Result<(), RuntimeError> {
    let weight = format!("{name}.weight");
    wmap.insert(
        weight.clone(),
        nn(b.f32_4d(&weight, 3, 3, 3, ic * oc, true))?,
    );
    let bias = format!("{name}.bias");
    wmap.insert(bias.clone(), nn(b.f32_1d(&bias, oc, true))?);
    Ok(())
}

fn graph_norm(
    b: &mut GraphBuilder,
    wmap: &mut HashMap<String, Tensor>,
    name: &str,
    channels: i64,
) -> Result<(), RuntimeError> {
    for param in ["weight", "bias"] {
        let tensor = format!("{name}.{param}");
        wmap.insert(tensor.clone(), nn(b.f32_1d(&tensor, channels, true))?);
    }
    Ok(())
}

fn load_conv3d(
    weights: &GgufWeights,
    loaded: &mut Vec<(String, Vec<f32>)>,
    name: &str,
    ic: i64,
    oc: i64,
) -> Result<(), WeightError> {
    let weight = format!("{name}.weight");
    loaded.push((
        weight.clone(),
        weights.f32_widened(&weight, &[3, 3, 3, ic * oc])?,
    ));
    let bias = format!("{name}.bias");
    loaded.push((bias.clone(), weights.f32_widened(&bias, &[oc])?));
    Ok(())
}

fn load_norm(
    weights: &GgufWeights,
    loaded: &mut Vec<(String, Vec<f32>)>,
    name: &str,
    channels: i64,
) -> Result<(), WeightError> {
    for param in ["weight", "bias"] {
        let tensor = format!("{name}.{param}");
        loaded.push((tensor.clone(), weights.f32_widened(&tensor, &[channels])?));
    }
    Ok(())
}

pub struct LtxLatentUpscaler {
    config: UpscalerConfig,
    loaded: Vec<(String, Vec<f32>)>,
}

impl LtxLatentUpscaler {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let config = UpscalerConfig::infer(&weights)?;
        let mut loaded = Vec::new();
        load_conv3d(
            &weights,
            &mut loaded,
            "initial_conv",
            config.latent,
            config.mid,
        )?;
        load_norm(&weights, &mut loaded, "initial_norm", config.mid)?;
        for stage in ["res_blocks", "post_upsample_res_blocks"] {
            for block in 0..config.blocks {
                for conv in ["conv1", "conv2"] {
                    load_conv3d(
                        &weights,
                        &mut loaded,
                        &format!("{stage}.{block}.{conv}"),
                        config.mid,
                        config.mid,
                    )?;
                }
                load_norm(
                    &weights,
                    &mut loaded,
                    &format!("{stage}.{block}.norm1"),
                    config.mid,
                )?;
                load_norm(
                    &weights,
                    &mut loaded,
                    &format!("{stage}.{block}.norm2"),
                    config.mid,
                )?;
            }
        }
        loaded.push((
            "upsampler.0.weight".to_string(),
            weights.f32_widened("upsampler.0.weight", &[3, 3, config.mid, 4 * config.mid])?,
        ));
        loaded.push((
            "upsampler.0.bias".to_string(),
            weights.f32_widened("upsampler.0.bias", &[4 * config.mid])?,
        ));
        load_conv3d(
            &weights,
            &mut loaded,
            "final_conv",
            config.mid,
            config.latent,
        )?;
        Ok(Self { config, loaded })
    }

    pub fn config(&self) -> UpscalerConfig {
        self.config
    }

    /// Upscale a `[w, h, d, latent]` latent (ggml order) into
    /// `[2w, 2h, d, latent]`.
    pub fn upscale(
        &self,
        ctx: &GgmlContext,
        latent: &[f32],
        w: i64,
        h: i64,
        d: i64,
    ) -> Result<Vec<f32>, WeightError> {
        let config = &self.config;
        if w < 1 || h < 1 || d < 1 {
            return Err(WeightError::Shape {
                name: "ltx upscaler dims".into(),
                expected: vec![1, 1, 1],
                got: vec![w, h, d],
            });
        }
        if latent.len() as i64 != w * h * d * config.latent {
            return Err(WeightError::Shape {
                name: "ltx upscaler latent".into(),
                expected: vec![w, h, d, config.latent],
                got: vec![latent.len() as i64],
            });
        }
        let mut b = ctx.builder(GRAPH_MEM)?;
        let mut wmap = HashMap::new();
        graph_conv3d(&mut b, &mut wmap, "initial_conv", config.latent, config.mid)?;
        graph_norm(&mut b, &mut wmap, "initial_norm", config.mid)?;
        for stage in ["res_blocks", "post_upsample_res_blocks"] {
            for block in 0..config.blocks {
                for conv in ["conv1", "conv2"] {
                    graph_conv3d(
                        &mut b,
                        &mut wmap,
                        &format!("{stage}.{block}.{conv}"),
                        config.mid,
                        config.mid,
                    )?;
                }
                graph_norm(
                    &mut b,
                    &mut wmap,
                    &format!("{stage}.{block}.norm1"),
                    config.mid,
                )?;
                graph_norm(
                    &mut b,
                    &mut wmap,
                    &format!("{stage}.{block}.norm2"),
                    config.mid,
                )?;
            }
        }
        wmap.insert(
            "upsampler.0.weight".to_string(),
            nn(b.f32_4d("upsampler.0.weight", 3, 3, config.mid, 4 * config.mid, true))?,
        );
        wmap.insert(
            "upsampler.0.bias".to_string(),
            nn(b.f32_1d("upsampler.0.bias", 4 * config.mid, true))?,
        );
        graph_conv3d(&mut b, &mut wmap, "final_conv", config.mid, config.latent)?;

        let mut x = nn(b.f32_4d("ltx_upsampler_latent", w, h, d, config.latent, true))?;
        x = conv3d_block(
            &b,
            &wmap,
            x,
            "initial_conv",
            w,
            h,
            d,
            config.latent,
            config.mid,
        )?;
        x = group_norm_clip(
            &b,
            &wmap,
            x,
            "initial_norm",
            config.groups,
            w,
            h,
            d,
            config.mid,
        )?;
        x = nn(b.silu(x))?;
        for block in 0..config.blocks {
            x = res_block(
                &b,
                &wmap,
                x,
                &format!("res_blocks.{block}"),
                config.groups,
                w,
                h,
                d,
                config.mid,
            )?;
        }
        x = spatial_x2(&b, &wmap, x, w, h, d, config.mid)?;
        let (w2, h2) = (w * 2, h * 2);
        for block in 0..config.blocks {
            x = res_block(
                &b,
                &wmap,
                x,
                &format!("post_upsample_res_blocks.{block}"),
                config.groups,
                w2,
                h2,
                d,
                config.mid,
            )?;
        }
        x = conv3d_block(
            &b,
            &wmap,
            x,
            "final_conv",
            w2,
            h2,
            d,
            config.mid,
            config.latent,
        )?;
        let mut graph = b.finish(x)?;
        for (name, data) in &self.loaded {
            graph.set_f32(name, data)?;
        }
        graph.set_f32("ltx_upsampler_latent", latent)?;
        let out = graph.compute()?;
        if let Some(detail) = crate::common::ops::non_finite("ltx upscaler", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// smoke writer (graph-vs-host tests)
// ---------------------------------------------------------------------------

/// Deterministic smoke weights for `UpscalerConfig::smoke()`.
pub fn smoke_store(config: &UpscalerConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 53u32;
    let mut store = Vec::new();
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    smoke_conv3d(
        &mut store,
        &mut next,
        "initial_conv",
        config.latent,
        config.mid,
    );
    store.push((
        "initial_norm.weight".to_string(),
        (0..config.mid).map(|_| 1.0 + next() * 0.1).collect(),
    ));
    store.push((
        "initial_norm.bias".to_string(),
        (0..config.mid as usize).map(|_| next() * 0.02).collect(),
    ));
    for stage in ["res_blocks", "post_upsample_res_blocks"] {
        for block in 0..config.blocks {
            for conv in ["conv1", "conv2"] {
                smoke_conv3d(
                    &mut store,
                    &mut next,
                    &format!("{stage}.{block}.{conv}"),
                    config.mid,
                    config.mid,
                );
            }
            for norm in ["norm1", "norm2"] {
                store.push((
                    format!("{stage}.{block}.{norm}.weight"),
                    (0..config.mid).map(|_| 1.0 + next() * 0.1).collect(),
                ));
                store.push((
                    format!("{stage}.{block}.{norm}.bias"),
                    (0..config.mid as usize).map(|_| next() * 0.02).collect(),
                ));
            }
        }
    }
    store.push((
        "upsampler.0.weight".into(),
        (0..(9 * config.mid * 4 * config.mid) as usize)
            .map(|_| next() * 0.05)
            .collect(),
    ));
    store.push((
        "upsampler.0.bias".into(),
        (0..(4 * config.mid) as usize)
            .map(|_| next() * 0.02)
            .collect(),
    ));
    smoke_conv3d(
        &mut store,
        &mut next,
        "final_conv",
        config.mid,
        config.latent,
    );
    store
}

fn smoke_conv3d(
    store: &mut Vec<(String, Vec<f32>)>,
    next: &mut dyn FnMut() -> f32,
    name: &str,
    ic: i64,
    oc: i64,
) {
    store.push((
        format!("{name}.weight"),
        (0..(27 * ic * oc) as usize)
            .map(|_| next() * 0.05)
            .collect(),
    ));
    store.push((
        format!("{name}.bias"),
        (0..oc as usize).map(|_| next() * 0.02).collect(),
    ));
}

/// Write the smoke upscaler as a safetensors file.
pub fn write_smoke(dir: &std::path::Path) -> Result<PathBuf, WeightError> {
    write_smoke_with(dir, &UpscalerConfig::smoke())
}

/// Write the smoke upscaler for an explicit config (session smoke bundle
/// shares the 128-channel latent with the DiT and the video VAE).
pub fn write_smoke_with(
    dir: &std::path::Path,
    config: &UpscalerConfig,
) -> Result<PathBuf, WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = *config;
    let store = smoke_store(&config);
    let mut shapes: Vec<Vec<u64>> = Vec::new();
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for (name, values) in &store {
        let shape: Vec<u64> = if name.ends_with(".weight") && name.contains("conv") {
            // conv3d kernels in PyTorch order [OC, IC, KD, KH, KW].
            let oc = values.len() as u64 / 27;
            let ic = if name == "initial_conv.weight" {
                config.latent as u64
            } else if name == "final_conv.weight" {
                config.mid as u64
            } else {
                config.mid as u64
            };
            vec![oc / ic, ic, 3, 3, 3]
        } else if name == "upsampler.0.weight" {
            vec![4 * config.mid as u64, config.mid as u64, 3, 3]
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
    let path = dir.join("upscaler_smoke.safetensors");
    crate::weights::write_safetensors(&path, &tensors)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;

    fn at<'a>(store: &'a [(String, Vec<f32>)], name: &str) -> &'a [f32] {
        &store
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
    }

    fn silu_host(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// `[w, h, d, c]` (channel-slowest) conv3d, zero pad 1. The kernel flat
    /// is `[3, 3, 3, ic * oc]` with the input channel fastest in the fold.
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
        let kernel = at(store, &format!("{prefix}.weight"));
        let bias = at(store, &format!("{prefix}.bias"));
        let mut out = vec![0f32; w * h * d * oc];
        for o in 0..oc {
            for zz in 0..d {
                for yy in 0..h {
                    for xx in 0..w {
                        let mut acc = bias[o];
                        for kz in 0..3i64 {
                            for ky in 0..3i64 {
                                for kx in 0..3i64 {
                                    let (sx, sy, sz) = (
                                        xx as i64 + kx - 1,
                                        yy as i64 + ky - 1,
                                        zz as i64 + kz - 1,
                                    );
                                    if sx < 0
                                        || sy < 0
                                        || sz < 0
                                        || sx >= w as i64
                                        || sy >= h as i64
                                        || sz >= d as i64
                                    {
                                        continue;
                                    }
                                    let tap = (kx + 3 * (ky + 3 * kz)) as usize;
                                    for i in 0..ic {
                                        acc += kernel[tap + 27 * (i + ic * o)]
                                            * x[sx as usize
                                                + w * (sy as usize + h * (sz as usize + d * i))];
                                    }
                                }
                            }
                        }
                        out[xx + w * (yy + h * (zz + d * o))] = acc;
                    }
                }
            }
        }
        out
    }

    /// Whole-clip GroupNorm on `[w, h, d, c]`.
    #[allow(clippy::too_many_arguments)]
    fn group_norm_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        groups: usize,
        c: usize,
    ) -> Vec<f32> {
        let scale = at(store, &format!("{prefix}.weight"));
        let bias = at(store, &format!("{prefix}.bias"));
        let per = c / groups;
        let n = x.len() / c;
        let mut out = vec![0f32; x.len()];
        for g in 0..groups {
            let mut mean = 0f32;
            for pos in 0..n {
                for i in g * per..(g + 1) * per {
                    mean += x[pos + n * i];
                }
            }
            mean /= (n * per) as f32;
            let mut var = 0f32;
            for pos in 0..n {
                for i in g * per..(g + 1) * per {
                    let v = x[pos + n * i] - mean;
                    var += v * v;
                }
            }
            let inv = 1.0 / (var / (n * per) as f32 + NORM_EPS).sqrt();
            for pos in 0..n {
                for i in g * per..(g + 1) * per {
                    out[pos + n * i] = (x[pos + n * i] - mean) * inv * scale[i] + bias[i];
                }
            }
        }
        out
    }

    /// `[w, h, c, d]` (batch-slowest) per-frame conv2d, zero pad 1. The
    /// kernel flat is `[3, 3, ic, oc]` with the input channel fastest.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        w: usize,
        h: usize,
        d: usize,
        ic: usize,
        oc: usize,
    ) -> Vec<f32> {
        let kernel = at(store, &format!("{prefix}.weight"));
        let bias = at(store, &format!("{prefix}.bias"));
        let mut out = vec![0f32; w * h * oc * d];
        for n in 0..d {
            for o in 0..oc {
                for yy in 0..h {
                    for xx in 0..w {
                        let mut acc = bias[o];
                        for ky in 0..3i64 {
                            for kx in 0..3i64 {
                                let (sx, sy) = (xx as i64 + kx - 1, yy as i64 + ky - 1);
                                if sx < 0 || sy < 0 || sx >= w as i64 || sy >= h as i64 {
                                    continue;
                                }
                                let tap = (kx + 3 * ky) as usize;
                                for i in 0..ic {
                                    acc += kernel[tap + 9 * (i + ic * o)]
                                        * x[sx as usize + w * (sy as usize + h * (i + ic * n))];
                                }
                            }
                        }
                        out[xx + w * (yy + h * (o + oc * n))] = acc;
                    }
                }
            }
        }
        out
    }

    fn res_block_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        groups: usize,
        w: usize,
        h: usize,
        d: usize,
        ch: usize,
    ) -> Vec<f32> {
        let y = conv3d_host(store, &format!("{prefix}.conv1"), x, w, h, d, ch, ch);
        let y = group_norm_host(store, &format!("{prefix}.norm1"), &y, groups, ch);
        let y: Vec<f32> = y.iter().map(|v| silu_host(*v)).collect();
        let y = conv3d_host(store, &format!("{prefix}.conv2"), &y, w, h, d, ch, ch);
        let y = group_norm_host(store, &format!("{prefix}.norm2"), &y, groups, ch);
        x.iter().zip(&y).map(|(a, b)| silu_host(a + b)).collect()
    }

    #[test]
    fn graph_matches_the_host_reference() {
        let config = UpscalerConfig::smoke();
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-upsmoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_smoke(&dir).unwrap();
        let upscaler = LtxLatentUpscaler::open(&[path]).unwrap();
        assert_eq!(upscaler.config(), config);

        let (w, h, d) = (3i64, 2i64, 2i64);
        let (wu, hu, du, c, m) = (
            w as usize,
            h as usize,
            d as usize,
            config.latent as usize,
            config.mid as usize,
        );
        let store = smoke_store(&config);
        let mut rng = 97u32;
        let mut next = move || {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng as f32 / u32::MAX as f32) - 0.5
        };
        let latent: Vec<f32> = (0..(w * h * d * config.latent) as usize)
            .map(|_| next())
            .collect();

        // host reference
        let mut x = conv3d_host(&store, "initial_conv", &latent, wu, hu, du, c, m);
        x = group_norm_host(&store, "initial_norm", &x, config.groups as usize, m);
        let mut x: Vec<f32> = x.iter().map(|v| silu_host(*v)).collect();
        for block in 0..config.blocks {
            x = res_block_host(
                &store,
                &format!("res_blocks.{block}"),
                &x,
                config.groups as usize,
                wu,
                hu,
                du,
                m,
            );
        }
        // per-frame conv2d + pixel shuffle: [w, h, d, c] -> [w, h, c, d]
        let mut framed = vec![0f32; wu * hu * m * du];
        for n in 0..du {
            for i in 0..m {
                for yy in 0..hu {
                    for xx in 0..wu {
                        framed[xx + wu * (yy + hu * (i + m * n))] =
                            x[xx + wu * (yy + hu * (n + du * i))];
                    }
                }
            }
        }
        let y = conv2d_host(&store, "upsampler.0", &framed, wu, hu, du, m, 4 * m);
        // out[2w, 2h, d, c] = shuffle of y[w, h, 4c, d]
        let mut x = vec![0f32; (2 * wu) * (2 * hu) * du * m];
        for n in 0..du {
            for cc in 0..m {
                for p1 in 0..2usize {
                    for p2 in 0..2usize {
                        for yy in 0..hu {
                            for xx in 0..wu {
                                let v = y[xx + wu * (yy + hu * (cc * 4 + p1 * 2 + p2 + 4 * m * n))];
                                x[xx * 2 + p2 + 2 * wu * (yy * 2 + p1 + 2 * hu * (n + du * cc))] =
                                    v;
                            }
                        }
                    }
                }
            }
        }
        for block in 0..config.blocks {
            x = res_block_host(
                &store,
                &format!("post_upsample_res_blocks.{block}"),
                &x,
                config.groups as usize,
                2 * wu,
                2 * hu,
                du,
                m,
            );
        }
        let host = conv3d_host(&store, "final_conv", &x, 2 * wu, 2 * hu, du, m, c);

        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let out = upscaler.upscale(&ctx, &latent, w, h, d).unwrap();
        assert_eq!(out.len(), host.len());
        let mut max_diff = 0f32;
        for (g, e) in out.iter().zip(&host) {
            max_diff = max_diff.max((g - e).abs());
        }
        assert!(max_diff < 2e-3, "graph vs host max diff {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_fails_loud_on_missing_weights() {
        let dir = std::env::temp_dir().join(format!("danmo-ltx-up-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.safetensors");
        let data = [0f32.to_le_bytes(), 0f32.to_le_bytes()].concat();
        let shape = [2u64];
        crate::weights::write_safetensors(
            &path,
            &[crate::weights::SafeTensor {
                name: "dummy.weight",
                dtype: "F32",
                shape: &shape,
                data: &data,
            }],
        )
        .unwrap();
        let err = match LtxLatentUpscaler::open(&[path]) {
            Err(err) => err,
            Ok(_) => panic!("accepted empty weights"),
        };
        assert!(err.to_string().contains("initial_conv"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
