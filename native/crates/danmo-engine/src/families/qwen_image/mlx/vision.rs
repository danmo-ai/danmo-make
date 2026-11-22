//! Qwen2.5-VL vision tower for Qwen-Image-Edit.
//!
//! Patch embed, windowed ViT, and the merger follow danmo-make
//! `edit_encoder_mlx.py` (`VisionTransformer`). Weights are
//! `encoder.visual.*` in the text-encoder bundle. Full attention runs on
//! blocks 7, 15, 23, and 31; every other block is windowed. The merger
//! uses exact `gelu`, not the DiT tanh approximation.

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use crate::weights::MlxWeights;

const PATCH: i32 = 14;
const MERGE: i32 = 2;
const TEMPORAL: i32 = 2;
const WINDOW: i32 = 112;
const HEADS: i32 = 16;
const FULL_ATTENTION: [u32; 4] = [7, 15, 23, 31];
const RMS_EPS: f32 = 1e-6;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VisionGrid {
    pub t: i32,
    pub h: i32,
    pub w: i32,
}

impl VisionGrid {
    pub(super) fn tokens(self) -> usize {
        (self.t * self.h * self.w / (MERGE * MERGE)) as usize
    }
}

/// Resize each RGB image the way `QwenVisionLanguageTokenizer` does, then
/// pack Qwen2.5-VL patches. `images` are interleaved RGB.
pub(super) fn preprocess(
    images: &[(&[u8], u32, u32)],
    picture_prefix: bool,
    vl_size: Option<(u32, u32)>,
) -> std::result::Result<(Vec<f32>, Vec<VisionGrid>), crate::weights::WeightError> {
    if images.is_empty() {
        return Err(crate::weights::WeightError::NotBuilt(
            "Qwen edit vision encode requires an image".into(),
        ));
    }
    let mut pixels = Vec::new();
    let mut grids = Vec::with_capacity(images.len());
    for (index, (rgb, width, height)) in images.iter().enumerate() {
        let (rgb, width, height) = if picture_prefix {
            let (dw, dh) = condition_size(*width, *height)?;
            (resize(*rgb, *width, *height, dw, dh)?, dw, dh)
        } else if index == 0 {
            let (dw, dh) = vl_size.ok_or_else(|| {
                crate::weights::WeightError::NotBuilt(
                    "Qwen edit single-image encode requires a VL size".into(),
                )
            })?;
            (resize(*rgb, *width, *height, dw, dh)?, dw, dh)
        } else {
            return Err(crate::weights::WeightError::NotBuilt(
                "Qwen edit without the picture prefix accepts one image".into(),
            ));
        };
        let (dw, dh) = smart_resize(height, width, PATCH * MERGE)?;
        let rgb = if dw != width || dh != height {
            resize(&rgb, width, height, dw, dh)?
        } else {
            rgb
        };
        let (patches, grid) = pack_patches(&rgb, dw, dh)?;
        pixels.extend(patches);
        grids.push(grid);
    }
    Ok((pixels, grids))
}

pub(super) fn forward(
    weights: &MlxWeights,
    pixels: &[f32],
    grids: &[VisionGrid],
    stream: &Stream,
) -> Result<Array> {
    if grids.is_empty() {
        return Err(Error::new("Qwen-Image vision", "no vision grids"));
    }
    let spec = vision_spec(weights)?;
    let tokens: i32 = grids.iter().map(|grid| grid.t * grid.h * grid.w).sum();
    let patch_dim = 3 * TEMPORAL * PATCH * PATCH;
    if pixels.len() != (tokens * patch_dim) as usize {
        return Err(Error::new(
            "Qwen-Image vision",
            format!(
                "pixels {} do not match {tokens} patches of {patch_dim}",
                pixels.len()
            ),
        ));
    }
    let weight = weights.tensor("encoder.visual.patch_embed.proj.weight")?;
    let dtype = weight.dtype();
    let hidden = Array::from_f32_slice(pixels, &[tokens, 3, TEMPORAL, PATCH, PATCH])?
        .astype(dtype, stream)?
        .transpose_axes(&[0, 2, 3, 4, 1], stream)?;
    let mut hidden = Array::conv3d(
        &hidden,
        weight,
        [TEMPORAL, PATCH, PATCH],
        [0, 0, 0],
        [1, 1, 1],
        1,
        stream,
    )?;
    hidden = hidden.reshape(&[tokens, spec.embed], stream)?;

    let grids_i: Vec<(i32, i32, i32)> = grids.iter().map(|grid| (grid.t, grid.h, grid.w)).collect();
    let (window, cu_window) = window_plan(&grids_i)
        .map_err(|error| Error::new("Qwen-Image vision", error.to_string()))?;
    let rotary = rotary_table(&grids_i, spec.head, stream)?;
    let unit = MERGE * MERGE;
    let groups = tokens / unit;
    if window.len() != groups as usize {
        return Err(Error::new(
            "Qwen-Image vision",
            format!(
                "window index {} does not match {groups} merge groups",
                window.len()
            ),
        ));
    }
    let order = Array::from_i32_slice(&window, &[groups])?;
    hidden = hidden
        .reshape(&[groups, unit, spec.embed], stream)?
        .take_axis(&order, 0, stream)?
        .reshape(&[tokens, spec.embed], stream)?;
    let rotary = rotary
        .reshape(&[groups, unit, spec.head], stream)?
        .take_axis(&order, 0, stream)?
        .reshape(&[tokens, spec.head], stream)?;
    let (cos, sin) = (rotary.cos(stream)?, rotary.sin(stream)?);

    let full_cu = cumulative_patches(&grids_i);
    for layer in 0..spec.layers {
        let cu = if FULL_ATTENTION.contains(&layer) {
            &full_cu
        } else {
            &cu_window
        };
        hidden = vision_block(weights, spec, layer, &hidden, &cos, &sin, cu, stream)?;
    }
    hidden = merger(weights, spec, &hidden, &grids_i, stream)?;
    let reverse = argsort(&window);
    let reverse = Array::from_i32_slice(&reverse, &[groups])?;
    hidden.take_axis(&reverse, 0, stream)
}

#[derive(Clone, Copy)]
struct VisionSpec {
    embed: i32,
    head: i32,
    layers: u32,
    hidden: i32,
}

fn vision_spec(weights: &MlxWeights) -> Result<VisionSpec> {
    let kernel = weights
        .tensor("encoder.visual.patch_embed.proj.weight")?
        .shape();
    if kernel.len() != 5
        || kernel[1] != TEMPORAL
        || kernel[2] != PATCH
        || kernel[3] != PATCH
        || kernel[4] != 3
    {
        return Err(Error::new(
            "Qwen-Image vision",
            format!("patch embed {kernel:?} is not [O, 2, 14, 14, 3]"),
        ));
    }
    let embed = kernel[0];
    if embed % HEADS != 0 {
        return Err(Error::new(
            "Qwen-Image vision",
            format!("embed {embed} does not split into {HEADS} heads"),
        ));
    }
    let mut layers = 0u32;
    while weights.has(&format!("encoder.visual.blocks.{layers}.norm1.weight")) {
        layers += 1;
    }
    if layers == 0 {
        return Err(Error::new("Qwen-Image vision", "no visual blocks"));
    }
    for index in FULL_ATTENTION {
        if index >= layers {
            return Err(Error::new(
                "Qwen-Image vision",
                format!("full-attention block {index} is past depth {layers}"),
            ));
        }
    }
    let merger = weights
        .tensor("encoder.visual.merger.mlp_1.weight")?
        .shape();
    if merger.len() != 2 {
        return Err(Error::new(
            "Qwen-Image vision",
            format!("merger mlp_1 {merger:?} is not a matrix"),
        ));
    }
    Ok(VisionSpec {
        embed,
        head: embed / HEADS,
        layers,
        hidden: merger[0],
    })
}

fn vision_block(
    weights: &MlxWeights,
    spec: VisionSpec,
    layer: u32,
    hidden: &Array,
    cos: &Array,
    sin: &Array,
    cu: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let path = format!("encoder.visual.blocks.{layer}");
    let norm = norm_weight(
        weights,
        &format!("{path}.norm1.weight"),
        hidden.dtype(),
        stream,
    )?;
    let attended = attention(
        weights,
        spec,
        &path,
        &hidden.rms_norm(Some(&norm), RMS_EPS, stream)?,
        cos,
        sin,
        cu,
        stream,
    )?;
    let hidden = Array::add(hidden, &attended, stream)?;
    let norm = norm_weight(
        weights,
        &format!("{path}.norm2.weight"),
        hidden.dtype(),
        stream,
    )?;
    let mlp = swiglu(
        weights,
        &path,
        &hidden.rms_norm(Some(&norm), RMS_EPS, stream)?,
        stream,
    )?;
    Array::add(&hidden, &mlp, stream)
}

fn attention(
    weights: &MlxWeights,
    spec: VisionSpec,
    path: &str,
    hidden: &Array,
    cos: &Array,
    sin: &Array,
    cu: &[i32],
    stream: &Stream,
) -> Result<Array> {
    let seq = hidden.shape()[0];
    let qkv = linear(weights, &format!("{path}.attn.qkv.weight"), hidden, stream)?;
    let qkv = qkv.reshape(&[seq, 3, HEADS, spec.head], stream)?;
    let q = qkv.slice(
        &[0, 0, 0, 0],
        &[seq, 1, HEADS, spec.head],
        &[1, 1, 1, 1],
        stream,
    )?;
    let k = qkv.slice(
        &[0, 1, 0, 0],
        &[seq, 2, HEADS, spec.head],
        &[1, 1, 1, 1],
        stream,
    )?;
    let v = qkv.slice(
        &[0, 2, 0, 0],
        &[seq, 3, HEADS, spec.head],
        &[1, 1, 1, 1],
        stream,
    )?;
    let q = q
        .reshape(&[seq, HEADS, spec.head], stream)?
        .transpose_axes(&[1, 0, 2], stream)?;
    let k = k
        .reshape(&[seq, HEADS, spec.head], stream)?
        .transpose_axes(&[1, 0, 2], stream)?;
    let v = v
        .reshape(&[seq, HEADS, spec.head], stream)?
        .transpose_axes(&[1, 0, 2], stream)?;
    let q = apply_rope(&q, cos, sin, stream)?;
    let k = apply_rope(&k, cos, sin, stream)?;
    let scale = 1.0 / (spec.head as f32).sqrt();
    let attended = if cu.len() > 2 {
        chunk_attention(&q, &k, &v, cu, scale, stream)?
    } else {
        sdpa(&q, &k, &v, scale, stream)?
    };
    let attended = attended
        .transpose_axes(&[1, 0, 2], stream)?
        .reshape(&[seq, spec.embed], stream)?;
    linear(
        weights,
        &format!("{path}.attn.proj.weight"),
        &attended,
        stream,
    )
}

fn apply_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    // input [heads, seq, dim], tables [seq, dim]
    let dtype = input.dtype();
    let input = input.astype(Dtype::Float32, stream)?;
    let dim = input.shape()[2];
    let parts = input.split_sections(&[dim / 2], -1, stream)?;
    let rotated =
        Array::concatenate_axis(&[parts[1].negative(stream)?, parts[0].clone()], -1, stream)?;
    let cos = cos.astype(Dtype::Float32, stream)?.expand_dims(0, stream)?;
    let sin = sin.astype(Dtype::Float32, stream)?.expand_dims(0, stream)?;
    let out = Array::add(
        &Array::multiply(&input, &cos, stream)?,
        &Array::multiply(&rotated, &sin, stream)?,
        stream,
    )?;
    out.astype(dtype, stream)
}

fn sdpa(q: &Array, k: &Array, v: &Array, scale: f32, stream: &Stream) -> Result<Array> {
    let dtype = q.dtype();
    let q = q.expand_dims(0, stream)?;
    let k = k.expand_dims(0, stream)?;
    let v = v.expand_dims(0, stream)?;
    Array::scaled_dot_product_attention(
        &q.astype(Dtype::Float32, stream)?,
        &k.astype(Dtype::Float32, stream)?,
        &v.astype(Dtype::Float32, stream)?,
        scale,
        "",
        None,
        stream,
    )?
    .astype(dtype, stream)?
    .squeeze_axis(0, stream)
}

fn chunk_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    cu: &[i32],
    scale: f32,
    stream: &Stream,
) -> Result<Array> {
    let mut parts = Vec::new();
    for pair in cu.windows(2) {
        let (start, stop) = (pair[0], pair[1]);
        if stop < start {
            return Err(Error::new(
                "Qwen-Image vision",
                format!("attention span {start}..{stop} is inverted"),
            ));
        }
        if stop == start {
            continue;
        }
        let q = q.slice(
            &[0, start, 0],
            &[HEADS, stop, q.shape()[2]],
            &[1, 1, 1],
            stream,
        )?;
        let k = k.slice(
            &[0, start, 0],
            &[HEADS, stop, k.shape()[2]],
            &[1, 1, 1],
            stream,
        )?;
        let v = v.slice(
            &[0, start, 0],
            &[HEADS, stop, v.shape()[2]],
            &[1, 1, 1],
            stream,
        )?;
        parts.push(sdpa(&q, &k, &v, scale, stream)?);
    }
    if parts.is_empty() {
        return Err(Error::new("Qwen-Image vision", "attention spans are empty"));
    }
    Array::concatenate_axis(&parts, 1, stream)
}

fn swiglu(weights: &MlxWeights, path: &str, hidden: &Array, stream: &Stream) -> Result<Array> {
    let gate = linear(
        weights,
        &format!("{path}.mlp.gate_proj.weight"),
        hidden,
        stream,
    )?
    .silu(stream)?;
    let up = linear(
        weights,
        &format!("{path}.mlp.up_proj.weight"),
        hidden,
        stream,
    )?;
    let mid = Array::multiply(&gate, &up, stream)?;
    linear(
        weights,
        &format!("{path}.mlp.down_proj.weight"),
        &mid,
        stream,
    )
}

fn merger(
    weights: &MlxWeights,
    spec: VisionSpec,
    hidden: &Array,
    grids: &[(i32, i32, i32)],
    stream: &Stream,
) -> Result<Array> {
    let norm = norm_weight(
        weights,
        "encoder.visual.merger.ln_q.weight",
        hidden.dtype(),
        stream,
    )?;
    let hidden = hidden.rms_norm(Some(&norm), RMS_EPS, stream)?;
    let merged_in = spec.embed * MERGE * MERGE;
    let mut parts = Vec::new();
    let mut offset = 0i32;
    for (t, h, w) in grids {
        let count = t * h * w;
        let rows = hidden.slice(&[offset, 0], &[offset + count, spec.embed], &[1, 1], stream)?;
        parts.push(rows.reshape(&[count / (MERGE * MERGE), merged_in], stream)?);
        offset += count;
    }
    let hidden = Array::concatenate_axis(&parts, 0, stream)?;
    let hidden = linear(
        weights,
        "encoder.visual.merger.mlp_0.weight",
        &hidden,
        stream,
    )?;
    let hidden = crate::common::mlx_ops::gelu(&hidden, stream)?;
    let hidden = linear(
        weights,
        "encoder.visual.merger.mlp_1.weight",
        &hidden,
        stream,
    )?;
    if hidden.shape()[1] != spec.hidden {
        return Err(Error::new(
            "Qwen-Image vision",
            format!("merger output {} is not {}", hidden.shape()[1], spec.hidden),
        ));
    }
    Ok(hidden)
}

fn rotary_table(grids: &[(i32, i32, i32)], head: i32, stream: &Stream) -> Result<Array> {
    let rotary_dim = head / 2;
    let mut positions = Vec::new();
    let mut max_grid = 1i32;
    for (t, h, w) in grids {
        if h % MERGE != 0 || w % MERGE != 0 {
            return Err(Error::new(
                "Qwen-Image vision",
                format!("grid {h}x{w} is not divisible by the merge size"),
            ));
        }
        max_grid = max_grid.max(*h).max(*w);
        let merge_h = h / MERGE;
        let merge_w = w / MERGE;
        for _ in 0..*t {
            for mh in 0..merge_h {
                for mw in 0..merge_w {
                    for sh in 0..MERGE {
                        for sw in 0..MERGE {
                            positions.push((mh * MERGE + sh, mw * MERGE + sw));
                        }
                    }
                }
            }
        }
    }
    let half = rotary_dim / 2;
    let base = Array::from_f32_slice(&[10000.0], &[])?;
    let dim = Array::from_f32_slice(&[rotary_dim as f32], &[])?;
    let exponents = Array::arange(0.0, f64::from(rotary_dim), 2.0, Dtype::Float32, stream)?;
    let inv = Array::divide(
        &Array::from_f32_slice(&[1.0], &[])?,
        &Array::power(&base, &Array::divide(&exponents, &dim, stream)?, stream)?,
        stream,
    )?;
    let pos = Array::arange(0.0, f64::from(max_grid), 1.0, Dtype::Float32, stream)?;
    let table = Array::multiply(
        &pos.expand_dims(1, stream)?,
        &inv.expand_dims(0, stream)?,
        stream,
    )?;
    table.eval()?;
    let table = table.to_vec_f32(stream)?;
    if table.len() != (max_grid * half) as usize {
        return Err(Error::new(
            "Qwen-Image vision",
            "rotary table length does not match the grid",
        ));
    }
    let mut angles = Vec::with_capacity(positions.len() * head as usize);
    let rows = positions.len() as i32;
    for (row, col) in &positions {
        let h = &table[(*row as usize) * half as usize..(*row as usize + 1) * half as usize];
        let w = &table[(*col as usize) * half as usize..(*col as usize + 1) * half as usize];
        angles.extend_from_slice(h);
        angles.extend_from_slice(w);
        angles.extend_from_slice(h);
        angles.extend_from_slice(w);
    }
    Array::from_f32_slice(&angles, &[rows, head])
}

fn linear(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let bias_name = name.replace(".weight", ".bias");
    let bias = if weights.has(&bias_name) {
        Some(weights.tensor(&bias_name)?)
    } else {
        None
    };
    let input = input.astype(weights.tensor(name)?.dtype(), stream)?;
    weights.linear(name)?.apply(&input, bias, stream)
}

fn norm_weight(weights: &MlxWeights, name: &str, dtype: Dtype, stream: &Stream) -> Result<Array> {
    weights.tensor(name)?.astype(dtype, stream)
}

/// Window reorder of merge-groups, plus the unique cumulative patch spans.
pub(super) fn window_plan(
    grids: &[(i32, i32, i32)],
) -> std::result::Result<(Vec<i32>, Vec<i32>), crate::weights::WeightError> {
    let window = WINDOW / PATCH / MERGE;
    let mut index = Vec::new();
    let mut cu = vec![0i32];
    let mut offset = 0i32;
    for (t, h, w) in grids {
        if h % MERGE != 0 || w % MERGE != 0 || *t < 1 {
            return Err(crate::weights::WeightError::NotBuilt(format!(
                "vision grid {t}x{h}x{w} is not a merged Qwen grid"
            )));
        }
        let llm_h = h / MERGE;
        let llm_w = w / MERGE;
        let pad_h = window - llm_h.rem_euclid(window);
        let pad_w = window - llm_w.rem_euclid(window);
        let nwh = (llm_h + pad_h) / window;
        let nww = (llm_w + pad_w) / window;
        let padded_h = llm_h + pad_h;
        let padded_w = llm_w + pad_w;
        for frame in 0..*t {
            let mut cells = vec![-100i32; (padded_h * padded_w) as usize];
            for row in 0..llm_h {
                for col in 0..llm_w {
                    cells[(row * padded_w + col) as usize] =
                        frame * llm_h * llm_w + row * llm_w + col;
                }
            }
            for wh in 0..nwh {
                for ww in 0..nww {
                    let mut valid = 0i32;
                    for sh in 0..window {
                        for sw in 0..window {
                            let row = wh * window + sh;
                            let col = ww * window + sw;
                            let value = cells[(row * padded_w + col) as usize];
                            if value >= 0 {
                                index.push(value + offset);
                                valid += 1;
                            }
                        }
                    }
                    let last = *cu.last().unwrap();
                    cu.push(last + valid * MERGE * MERGE);
                }
            }
        }
        offset += t * llm_h * llm_w;
    }
    let mut unique = vec![cu[0]];
    for value in cu.into_iter().skip(1) {
        if value != *unique.last().unwrap() {
            unique.push(value);
        }
    }
    Ok((index, unique))
}

fn cumulative_patches(grids: &[(i32, i32, i32)]) -> Vec<i32> {
    let mut cu = vec![0];
    let mut offset = 0;
    for (t, h, w) in grids {
        offset += t * h * w;
        cu.push(offset);
    }
    cu
}

fn argsort(index: &[i32]) -> Vec<i32> {
    let mut order: Vec<i32> = (0..index.len() as i32).collect();
    order.sort_by_key(|position| index[*position as usize]);
    order
}

pub(super) fn condition_size(
    width: u32,
    height: u32,
) -> std::result::Result<(u32, u32), crate::weights::WeightError> {
    area_size(width, height, 384.0 * 384.0, 32)
}

pub(super) fn vae_condition_size(
    width: u32,
    height: u32,
) -> std::result::Result<(u32, u32), crate::weights::WeightError> {
    area_size(width, height, 1024.0 * 1024.0, 32)
}

/// Output pixels when the request does not set a size: source, floored to 16.
pub(super) fn source_output_size(
    width: u32,
    height: u32,
) -> std::result::Result<(u32, u32), crate::weights::WeightError> {
    let width = width / 16 * 16;
    let height = height / 16 * 16;
    if width < 16 || height < 16 {
        return Err(crate::weights::WeightError::NotBuilt(format!(
            "source is smaller than 16px after alignment"
        )));
    }
    Ok((width.max(16), height.max(16)))
}

fn area_size(
    width: u32,
    height: u32,
    area: f64,
    multiple: u32,
) -> std::result::Result<(u32, u32), crate::weights::WeightError> {
    if width < 1 || height < 1 {
        return Err(crate::weights::WeightError::NotBuilt(
            "edit image size is empty".into(),
        ));
    }
    let ratio = f64::from(width) / f64::from(height);
    let side = py_round((area * ratio).sqrt() / f64::from(multiple)) * f64::from(multiple);
    let other = py_round((side / ratio) / f64::from(multiple)) * f64::from(multiple);
    let (width, height) = (side as u32, other as u32);
    if width < multiple || height < multiple {
        return Err(crate::weights::WeightError::NotBuilt(format!(
            "edit condition size {width}x{height} collapsed"
        )));
    }
    Ok((width, height))
}

pub(super) fn smart_resize(
    height: u32,
    width: u32,
    factor: i32,
) -> std::result::Result<(u32, u32), crate::weights::WeightError> {
    if height < 1 || width < 1 {
        return Err(crate::weights::WeightError::NotBuilt(
            "vision resize source is empty".into(),
        ));
    }
    let (height, width) = (f64::from(height), f64::from(width));
    if width.max(height) / width.min(height) > 200.0 {
        return Err(crate::weights::WeightError::NotBuilt(
            "vision image aspect ratio is above 200".into(),
        ));
    }
    let factor_f = f64::from(factor);
    let min_pixels = 56.0 * 56.0;
    let max_pixels = 28.0 * 28.0 * 1280.0;
    let mut h = py_round(height / factor_f) * factor_f;
    let mut w = py_round(width / factor_f) * factor_f;
    if h * w > max_pixels {
        let beta = ((height * width) / max_pixels).sqrt();
        h = (height / beta / factor_f).floor() * factor_f;
        w = (width / beta / factor_f).floor() * factor_f;
        h = h.max(factor_f);
        w = w.max(factor_f);
    } else if h * w < min_pixels {
        let beta = (min_pixels / (height * width)).sqrt();
        h = (height * beta / factor_f).ceil() * factor_f;
        w = (width * beta / factor_f).ceil() * factor_f;
    }
    Ok((w as u32, h as u32))
}

/// Python 3 `round` (half to even).
fn py_round(value: f64) -> f64 {
    let floor = value.floor();
    let diff = value - floor;
    if (diff - 0.5).abs() < 1e-8 {
        if (floor as i64).rem_euclid(2) == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

fn resize(
    rgb: &[u8],
    width: u32,
    height: u32,
    dw: u32,
    dh: u32,
) -> std::result::Result<Vec<u8>, crate::weights::WeightError> {
    crate::common::image::resize_bicubic(rgb, width, height, dw, dh)
}

fn pack_patches(
    rgb: &[u8],
    width: u32,
    height: u32,
) -> std::result::Result<(Vec<f32>, VisionGrid), crate::weights::WeightError> {
    let (width, height) = (width as i32, height as i32);
    if height % PATCH != 0 || width % PATCH != 0 {
        return Err(crate::weights::WeightError::NotBuilt(format!(
            "vision size {width}x{height} is not a multiple of {PATCH}"
        )));
    }
    let grid_h = height / PATCH;
    let grid_w = width / PATCH;
    if grid_h % MERGE != 0 || grid_w % MERGE != 0 {
        return Err(crate::weights::WeightError::NotBuilt(format!(
            "vision grid {grid_h}x{grid_w} is not divisible by {MERGE}"
        )));
    }
    let plane = (width * height) as usize;
    if rgb.len() != plane * 3 {
        return Err(crate::weights::WeightError::Shape {
            name: "rgb".into(),
            expected: vec![(plane * 3) as i64],
            got: vec![rgb.len() as i64],
        });
    }
    let merge_h = grid_h / MERGE;
    let merge_w = grid_w / MERGE;
    let patch_dim = (3 * TEMPORAL * PATCH * PATCH) as usize;
    let tokens = (grid_h * grid_w) as usize;
    let mut out = vec![0f32; tokens * patch_dim];
    for mh in 0..merge_h {
        for mw in 0..merge_w {
            for sh in 0..MERGE {
                for sw in 0..MERGE {
                    let patch_row = mh * MERGE + sh;
                    let patch_col = mw * MERGE + sw;
                    let seq = ((mh * merge_w + mw) * MERGE * MERGE + sh * MERGE + sw) as usize;
                    let mut cursor = seq * patch_dim;
                    for channel in 0..3 {
                        for _temporal in 0..TEMPORAL {
                            for py in 0..PATCH {
                                for px in 0..PATCH {
                                    let y = patch_row * PATCH + py;
                                    let x = patch_col * PATCH + px;
                                    let byte =
                                        rgb[((y * width + x) as usize) * 3 + channel as usize];
                                    let sample = byte as f32 / 255.0;
                                    out[cursor] =
                                        (sample - MEAN[channel as usize]) / STD[channel as usize];
                                    cursor += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((
        out,
        VisionGrid {
            t: 1,
            h: grid_h,
            w: grid_w,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        condition_size, smart_resize, source_output_size, vae_condition_size, window_plan,
    };

    #[test]
    fn condition_sizes_match_python_half_even_rounding() {
        assert_eq!(vae_condition_size(1000, 500).unwrap(), (1440, 704));
        assert_eq!(condition_size(1000, 500).unwrap(), (544, 256));
        assert_eq!(source_output_size(1000, 500).unwrap(), (992, 496));
        assert_eq!(smart_resize(256, 544, 28).unwrap(), (532, 252));
    }

    #[test]
    fn window_plan_matches_the_qwen_pad_even_when_the_grid_divides() {
        let (index, cu) = window_plan(&[(1, 28, 28)]).unwrap();
        assert_eq!(index.len(), 196);
        assert_eq!(
            &index[..16],
            &[0, 1, 2, 3, 14, 15, 16, 17, 28, 29, 30, 31, 42, 43, 44, 45]
        );
        assert_eq!(cu.first().copied(), Some(0));
        assert_eq!(cu[1], 64);
        assert_eq!(*cu.last().unwrap(), 28 * 28);
        let (both, _) = window_plan(&[(1, 28, 28), (1, 14, 28)]).unwrap();
        assert_eq!(both.len(), 294);
        assert_eq!(
            &both[both.len() - 8..],
            &[290, 291, 264, 265, 278, 279, 292, 293]
        );
    }
}
