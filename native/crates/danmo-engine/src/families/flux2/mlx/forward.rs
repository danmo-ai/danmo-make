//! Klein DiT forward. Names and math follow danmo-make
//! `families/flux2/transformer_mlx.py` and `common/ops/norm.py`.
//!
//! Double-stream modulation chunks are shift, scale, gate, then the MLP
//! triple. `AdaLayerNormContinuous` is the other order: scale, then shift.
//! Q/K RMS runs in f32 and casts back. A guidance embedder is added only when
//! `time_guidance_embed.guidance_linear_*` is in the checkpoint. Distilled
//! Klein bundles do not have those weights.

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use super::super::checkpoint::klein_layout;
use crate::weights::MlxWeights;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 2000.0;

#[derive(Debug, Clone, Copy)]
pub(super) struct KleinSpec {
    pub dim: i32,
    pub heads: i32,
    pub head: i32,
    pub in_channels: i32,
    pub joint: i32,
    pub freq: i32,
    pub doubles: u32,
    pub singles: u32,
    pub hidden: i32,
    pub guidance: bool,
}

pub(super) fn assert_contract(weights: &MlxWeights) -> Result<KleinSpec> {
    let names: Vec<&str> = weights.names().collect();
    let layout = klein_layout(names).map_err(|error| Error::new("Flux2 Klein", error))?;
    let (dim, in_channels) = matrix(weights, "x_embedder.weight")?;
    if in_channels != 128 {
        return Err(Error::new(
            "Flux2 Klein",
            format!("packed latent has {in_channels} channels, the VAE packs 128"),
        ));
    }
    let (context_out, joint) = matrix(weights, "context_embedder.weight")?;
    expect_eq("context_embedder.weight", context_out, dim)?;
    let (time_out, freq) = matrix(weights, "time_guidance_embed.linear_1.weight")?;
    expect_eq("time_guidance_embed.linear_1.weight", time_out, dim)?;
    if freq < 2 || freq % 2 != 0 {
        return Err(Error::new(
            "Flux2 Klein",
            format!("timestep frequency dim {freq} is not a positive even width"),
        ));
    }
    let (time2_out, time2_in) = matrix(weights, "time_guidance_embed.linear_2.weight")?;
    expect_eq("time_guidance_embed.linear_2.weight out", time2_out, dim)?;
    expect_eq("time_guidance_embed.linear_2.weight in", time2_in, dim)?;
    let guidance_in = weights
        .names()
        .any(|name| name.starts_with("time_guidance_embed.guidance_linear_1"));
    let guidance_out = weights
        .names()
        .any(|name| name.starts_with("time_guidance_embed.guidance_linear_2"));
    if guidance_in != guidance_out {
        return Err(Error::new(
            "Flux2 Klein",
            "guidance embedder is missing one of guidance_linear_1 or guidance_linear_2",
        ));
    }
    if guidance_in {
        let (gout, gin) = matrix(weights, "time_guidance_embed.guidance_linear_1.weight")?;
        expect_eq("guidance_linear_1 out", gout, dim)?;
        expect_eq("guidance_linear_1 in", gin, freq)?;
        let (gout, gin) = matrix(weights, "time_guidance_embed.guidance_linear_2.weight")?;
        expect_eq("guidance_linear_2 out", gout, dim)?;
        expect_eq("guidance_linear_2 in", gin, dim)?;
    }
    let (mod_out, mod_in) = matrix(weights, "double_stream_modulation_img.linear.weight")?;
    expect_eq("double_stream_modulation_img out", mod_out, dim * 6)?;
    expect_eq("double_stream_modulation_img in", mod_in, dim)?;
    let (mod_out, mod_in) = matrix(weights, "double_stream_modulation_txt.linear.weight")?;
    expect_eq("double_stream_modulation_txt out", mod_out, dim * 6)?;
    expect_eq("double_stream_modulation_txt in", mod_in, dim)?;
    let (mod_out, mod_in) = matrix(weights, "single_stream_modulation.linear.weight")?;
    expect_eq("single_stream_modulation out", mod_out, dim * 3)?;
    expect_eq("single_stream_modulation in", mod_in, dim)?;
    let (norm_out, norm_in) = matrix(weights, "norm_out.linear.weight")?;
    expect_eq("norm_out out", norm_out, dim * 2)?;
    expect_eq("norm_out in", norm_in, dim)?;
    let (proj_out, proj_in) = matrix(weights, "proj_out.weight")?;
    expect_eq("proj_out out", proj_out, in_channels)?;
    expect_eq("proj_out in", proj_in, dim)?;
    let (fused_out, fused_in) = matrix(
        weights,
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",
    )?;
    expect_eq("to_qkv_mlp_proj in", fused_in, dim)?;
    let qkv = dim * 3;
    if fused_out <= qkv || (fused_out - qkv) % 2 != 0 {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "to_qkv_mlp_proj out {fused_out} does not split into QKV {qkv} plus an even SwiGLU"
            ),
        ));
    }
    let hidden = (fused_out - qkv) / 2;
    let head = vector_len(weights, "transformer_blocks.0.attn.norm_q.weight")?;
    if head < 8 || head % 8 != 0 || dim % head != 0 {
        return Err(Error::new(
            "Flux2 Klein",
            format!("head dim {head} does not split width {dim} into four even RoPE axes"),
        ));
    }
    Ok(KleinSpec {
        dim: i32_dim(dim, "width")?,
        heads: i32_dim(dim / head, "heads")?,
        head: i32_dim(head, "head")?,
        in_channels: i32_dim(in_channels, "channels")?,
        joint: i32_dim(joint, "joint")?,
        freq: i32_dim(freq, "frequency")?,
        doubles: layout.double_blocks,
        singles: layout.single_blocks,
        hidden: i32_dim(hidden, "mlp")?,
        guidance: guidance_in,
    })
}

fn matrix(weights: &MlxWeights, name: &str) -> Result<(i64, i64)> {
    let Some((out, inn)) = weights.linear(name)?.out_in()? else {
        return Err(Error::new("Flux2 Klein", format!("{name} is not a matrix")));
    };
    if out < 1 || inn < 1 {
        return Err(Error::new(
            "Flux2 Klein",
            format!("{name} shape {out}x{inn} is empty"),
        ));
    }
    Ok((out, inn))
}

fn expect_eq(what: &str, got: i64, expected: i64) -> Result<()> {
    if got != expected {
        return Err(Error::new(
            "Flux2 Klein",
            format!("{what} is {got}, expected {expected}"),
        ));
    }
    Ok(())
}

fn i32_dim(value: i64, what: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::new("Flux2 Klein", format!("{what} {value} does not fit in i32")))
}

fn vector_len(weights: &MlxWeights, name: &str) -> Result<i64> {
    let shape = weights.tensor(name)?.shape();
    let len = shape.iter().map(|dim| i64::from(*dim)).product();
    if len < 1 {
        return Err(Error::new("Flux2 Klein", format!("{name} is empty")));
    }
    Ok(len)
}

/// Packed image token ids `[T, H, W, L]`. Create uses `T=0`; edit appends
/// reference tokens at `T = 10, 20, …` (diffusers / BFL).
pub(crate) const REF_T_SCALE: i32 = 10;

pub(super) fn image_token_ids(t: i32, grid_h: i32, grid_w: i32) -> Result<Vec<[i32; 4]>> {
    if grid_h < 1 || grid_w < 1 {
        return Err(Error::new("Flux2 Klein", "RoPE grid is empty"));
    }
    let mut ids = Vec::with_capacity((grid_h * grid_w) as usize);
    for row in 0..grid_h {
        for col in 0..grid_w {
            ids.push([t, row, col, 0]);
        }
    }
    Ok(ids)
}

/// One denoise step. `latent` is `[1, seq, C]` and may include reference tokens
/// after the target. Only the first `target_seq` tokens are returned (BFL /
/// diffusers Klein edit). `img_ids` length must equal `seq`.
pub(super) fn forward(
    weights: &MlxWeights,
    spec: KleinSpec,
    latent: &Array,
    text: &Array,
    timestep: &[f32],
    guidance: Option<&[f32]>,
    img_ids: &[[i32; 4]],
    target_seq: i32,
    stream: &Stream,
) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[2] != spec.in_channels {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "latent shape {shape:?} is not [1, seq, {}]",
                spec.in_channels
            ),
        ));
    }
    let seq_img = shape[1];
    if img_ids.len() != seq_img as usize {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "img_ids has {} entries for latent seq {seq_img}",
                img_ids.len()
            ),
        ));
    }
    if target_seq < 1 || target_seq > seq_img {
        return Err(Error::new(
            "Flux2 Klein",
            format!("target_seq {target_seq} is outside latent seq {seq_img}"),
        ));
    }
    let text_shape = text.shape();
    if text_shape.len() != 3 || text_shape[0] != 1 || text_shape[2] != spec.joint {
        return Err(Error::new(
            "Flux2 Klein",
            format!("text shape {text_shape:?} is not [1, seq, {}]", spec.joint),
        ));
    }
    let seq_txt = text_shape[1];
    if timestep.len() != spec.freq as usize {
        return Err(Error::new(
            "Flux2 Klein",
            format!(
                "timestep embedding has {} values, linear_1 expects {}",
                timestep.len(),
                spec.freq
            ),
        ));
    }

    let freq = Array::from_f32_slice(timestep, &[1, spec.freq])?;
    let temb = linear(
        weights,
        "time_guidance_embed.linear_1.weight",
        &freq,
        stream,
    )?;
    let mut temb = linear(
        weights,
        "time_guidance_embed.linear_2.weight",
        &temb.silu(stream)?,
        stream,
    )?;
    if spec.guidance {
        let Some(guidance) = guidance else {
            return Err(Error::new(
                "Flux2 Klein",
                "this bundle has a guidance embedder and guidance was not set",
            ));
        };
        if guidance.len() != spec.freq as usize {
            return Err(Error::new(
                "Flux2 Klein",
                format!(
                    "guidance embedding has {} values, guidance_linear_1 expects {}",
                    guidance.len(),
                    spec.freq
                ),
            ));
        }
        let guidance = Array::from_f32_slice(guidance, &[1, spec.freq])?;
        let guidance = linear(
            weights,
            "time_guidance_embed.guidance_linear_1.weight",
            &guidance,
            stream,
        )?;
        let guidance = linear(
            weights,
            "time_guidance_embed.guidance_linear_2.weight",
            &guidance.silu(stream)?,
            stream,
        )?;
        temb = Array::add(&temb, &guidance, stream)?;
    } else if guidance.is_some() {
        return Err(Error::new(
            "Flux2 Klein",
            "this bundle has no guidance embedder",
        ));
    }
    // Match danmo-make Flux2DiTMLX: temb.astype(bfloat16) after time embed.
    let temb = temb.astype(Dtype::BFloat16, stream)?;
    let dtype = temb.dtype();
    let latent = latent.astype(dtype, stream)?;
    let text = text.astype(dtype, stream)?;
    let mut img = linear(weights, "x_embedder.weight", &latent, stream)?;
    let mut txt = linear(weights, "context_embedder.weight", &text, stream)?;

    let conditioned = temb.silu(stream)?;
    let mod_img = linear(
        weights,
        "double_stream_modulation_img.linear.weight",
        &conditioned,
        stream,
    )?;
    let mod_txt = linear(
        weights,
        "double_stream_modulation_txt.linear.weight",
        &conditioned,
        stream,
    )?;
    let mod_single = linear(
        weights,
        "single_stream_modulation.linear.weight",
        &conditioned,
        stream,
    )?;
    let (img_cos, img_sin) = rope_from_ids(img_ids, spec.head, stream)?;
    let (txt_cos, txt_sin) = rope_table(seq_txt, 1, spec.head, true, stream)?;
    let cos = Array::concatenate_axis(&[txt_cos, img_cos], 0, stream)?;
    let sin = Array::concatenate_axis(&[txt_sin, img_sin], 0, stream)?;

    for index in 0..spec.doubles {
        (txt, img) = double_block(
            weights, spec, index, &txt, &img, &mod_txt, &mod_img, &cos, &sin, stream,
        )?;
    }
    let mut seq = Array::concatenate_axis(&[txt.clone(), img.clone()], 1, stream)?;
    for index in 0..spec.singles {
        seq = single_block(weights, spec, index, &seq, &mod_single, &cos, &sin, stream)?;
    }
    let img = slice_axis(&seq, 1, seq_txt, seq_txt + seq_img, stream)?;
    let normed = layer_norm(&img, stream)?;
    let out_mod = linear(weights, "norm_out.linear.weight", &conditioned, stream)?;
    // AdaLayerNormContinuous: the first half is scale, the second is shift.
    let scale = channel(&out_mod, 0, spec.dim, stream)?;
    let shift = channel(&out_mod, spec.dim, spec.dim, stream)?;
    let normed = scale_shift(&normed, &scale, &shift, stream)?;
    let velocity = linear(weights, "proj_out.weight", &normed, stream)?;
    if target_seq == seq_img {
        return Ok(velocity);
    }
    slice_axis(&velocity, 1, 0, target_seq, stream)
}

fn double_block(
    weights: &MlxWeights,
    spec: KleinSpec,
    index: u32,
    txt: &Array,
    img: &Array,
    mod_txt: &Array,
    mod_img: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let prefix = format!("transformer_blocks.{index}");
    let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
        mods(mod_img, spec.dim, stream)?;
    let (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
        mods(mod_txt, spec.dim, stream)?;
    let n_img = scale_shift(&layer_norm(img, stream)?, &scale_msa, &shift_msa, stream)?;
    let n_txt = scale_shift(
        &layer_norm(txt, stream)?,
        &c_scale_msa,
        &c_shift_msa,
        stream,
    )?;
    let (img_out, txt_out) =
        joint_attention(weights, spec, &prefix, &n_img, &n_txt, cos, sin, stream)?;
    let img = Array::add(img, &Array::multiply(&gate_msa, &img_out, stream)?, stream)?;
    let txt = Array::add(
        txt,
        &Array::multiply(&c_gate_msa, &txt_out, stream)?,
        stream,
    )?;
    let n_img = scale_shift(&layer_norm(&img, stream)?, &scale_mlp, &shift_mlp, stream)?;
    let n_txt = scale_shift(
        &layer_norm(&txt, stream)?,
        &c_scale_mlp,
        &c_shift_mlp,
        stream,
    )?;
    let img = Array::add(
        &img,
        &Array::multiply(
            &gate_mlp,
            &feed_forward(weights, &format!("{prefix}.ff"), &n_img, stream)?,
            stream,
        )?,
        stream,
    )?;
    let txt = Array::add(
        &txt,
        &Array::multiply(
            &c_gate_mlp,
            &feed_forward(weights, &format!("{prefix}.ff_context"), &n_txt, stream)?,
            stream,
        )?,
        stream,
    )?;
    Ok((txt, img))
}

fn joint_attention(
    weights: &MlxWeights,
    spec: KleinSpec,
    prefix: &str,
    img: &Array,
    txt: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let seq_txt = txt.shape()[1];
    let (q_img, k_img, v_img) = project_qkv(
        weights,
        img,
        &format!("{prefix}.attn.to_q.weight"),
        &format!("{prefix}.attn.to_k.weight"),
        &format!("{prefix}.attn.to_v.weight"),
        &format!("{prefix}.attn.norm_q.weight"),
        &format!("{prefix}.attn.norm_k.weight"),
        spec,
        stream,
    )?;
    let (q_txt, k_txt, v_txt) = project_qkv(
        weights,
        txt,
        &format!("{prefix}.attn.add_q_proj.weight"),
        &format!("{prefix}.attn.add_k_proj.weight"),
        &format!("{prefix}.attn.add_v_proj.weight"),
        &format!("{prefix}.attn.norm_added_q.weight"),
        &format!("{prefix}.attn.norm_added_k.weight"),
        spec,
        stream,
    )?;
    let q = Array::concatenate_axis(&[q_txt, q_img], 2, stream)?;
    let k = Array::concatenate_axis(&[k_txt, k_img], 2, stream)?;
    let v = Array::concatenate_axis(&[v_txt, v_img], 2, stream)?;
    let mixed = attend(&q, &k, &v, cos, sin, spec.head, stream)?;
    let txt = slice_axis(&mixed, 1, 0, seq_txt, stream)?;
    let img = slice_axis(&mixed, 1, seq_txt, mixed.shape()[1], stream)?;
    let img = linear(
        weights,
        &format!("{prefix}.attn.to_out.weight"),
        &img,
        stream,
    )?;
    let txt = linear(
        weights,
        &format!("{prefix}.attn.to_add_out.weight"),
        &txt,
        stream,
    )?;
    Ok((img, txt))
}

fn project_qkv(
    weights: &MlxWeights,
    input: &Array,
    q_name: &str,
    k_name: &str,
    v_name: &str,
    q_norm: &str,
    k_norm: &str,
    spec: KleinSpec,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let q = reshape_heads(&linear(weights, q_name, input, stream)?, spec, stream)?;
    let k = reshape_heads(&linear(weights, k_name, input, stream)?, spec, stream)?;
    let v = reshape_heads(&linear(weights, v_name, input, stream)?, spec, stream)?;
    let q = rms_heads(&q, weights.tensor(q_norm)?, stream)?;
    let k = rms_heads(&k, weights.tensor(k_norm)?, stream)?;
    Ok((q, k, v))
}

fn single_block(
    weights: &MlxWeights,
    spec: KleinSpec,
    index: u32,
    hidden: &Array,
    modulation: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let prefix = format!("single_transformer_blocks.{index}.attn");
    let (shift, scale, gate) = mods3(modulation, spec.dim, stream)?;
    let normed = scale_shift(&layer_norm(hidden, stream)?, &scale, &shift, stream)?;
    let proj = linear(
        weights,
        &format!("{prefix}.to_qkv_mlp_proj.weight"),
        &normed,
        stream,
    )?;
    let qkv_width = spec.dim * 3;
    let qkv = channel(&proj, 0, qkv_width, stream)?;
    let mlp = channel(&proj, qkv_width, spec.hidden * 2, stream)?;
    let q = channel(&qkv, 0, spec.dim, stream)?;
    let k = channel(&qkv, spec.dim, spec.dim, stream)?;
    let v = channel(&qkv, spec.dim * 2, spec.dim, stream)?;
    let q = rms_heads(
        &reshape_heads(&q, spec, stream)?,
        weights.tensor(&format!("{prefix}.norm_q.weight"))?,
        stream,
    )?;
    let k = rms_heads(
        &reshape_heads(&k, spec, stream)?,
        weights.tensor(&format!("{prefix}.norm_k.weight"))?,
        stream,
    )?;
    let v = reshape_heads(&v, spec, stream)?;
    let attended = attend(&q, &k, &v, cos, sin, spec.head, stream)?;
    let mlp = swiglu(&mlp, stream)?;
    let fused = Array::concatenate_axis(&[attended, mlp], 2, stream)?;
    let delta = linear(weights, &format!("{prefix}.to_out.weight"), &fused, stream)?;
    Array::add(hidden, &Array::multiply(&gate, &delta, stream)?, stream)
}

fn attend(
    q: &Array,
    k: &Array,
    v: &Array,
    cos: &Array,
    sin: &Array,
    head: i32,
    stream: &Stream,
) -> Result<Array> {
    let q = apply_rope(q, cos, sin, stream)?;
    let k = apply_rope(k, cos, sin, stream)?;
    let scale = 1.0 / (head as f32).sqrt();
    let shape = q.shape();
    let (batch, _heads, sequence, _dim) = (shape[0], shape[1], shape[2], shape[3]);
    Array::scaled_dot_product_attention(&q, &k, &v, scale, "", None, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, head * shape[1]], stream)
}

fn feed_forward(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let gate_up = linear(
        weights,
        &format!("{prefix}.linear_in.weight"),
        input,
        stream,
    )?;
    linear(
        weights,
        &format!("{prefix}.linear_out.weight"),
        &swiglu(&gate_up, stream)?,
        stream,
    )
}

fn swiglu(gate_up: &Array, stream: &Stream) -> Result<Array> {
    let width = *gate_up.shape().last().unwrap_or(&0);
    if width <= 0 || width % 2 != 0 {
        return Err(Error::new(
            "Flux2 Klein",
            format!("SwiGLU width {width} is not a positive even number"),
        ));
    }
    let half = width / 2;
    let gate = channel(gate_up, 0, half, stream)?.silu(stream)?;
    let up = channel(gate_up, half, half, stream)?;
    Array::multiply(&gate, &up, stream)
}

/// Six equal chunks: shift, scale, gate, shift_mlp, scale_mlp, gate_mlp.
fn mods(
    packed: &Array,
    dim: i32,
    stream: &Stream,
) -> Result<(Array, Array, Array, Array, Array, Array)> {
    Ok((
        channel(packed, 0, dim, stream)?,
        channel(packed, dim, dim, stream)?,
        channel(packed, dim * 2, dim, stream)?,
        channel(packed, dim * 3, dim, stream)?,
        channel(packed, dim * 4, dim, stream)?,
        channel(packed, dim * 5, dim, stream)?,
    ))
}

fn mods3(packed: &Array, dim: i32, stream: &Stream) -> Result<(Array, Array, Array)> {
    Ok((
        channel(packed, 0, dim, stream)?,
        channel(packed, dim, dim, stream)?,
        channel(packed, dim * 2, dim, stream)?,
    ))
}

fn scale_shift(x: &Array, scale: &Array, shift: &Array, stream: &Stream) -> Result<Array> {
    let one = Array::from_f32_slice(&[1.0], &[])?.astype(x.dtype(), stream)?;
    let scale = Array::add(&one, scale, stream)?;
    Array::add(&Array::multiply(x, &scale, stream)?, shift, stream)
}

fn layer_norm(x: &Array, stream: &Stream) -> Result<Array> {
    x.layer_norm(LN_EPS, stream)
}

fn rms_heads(x: &Array, scale: &Array, stream: &Stream) -> Result<Array> {
    let dtype = x.dtype();
    let x = x.astype(Dtype::Float32, stream)?;
    let scale = scale.astype(Dtype::Float32, stream)?;
    x.rms_norm(Some(&scale), RMS_EPS, stream)?
        .astype(dtype, stream)
}

fn reshape_heads(x: &Array, spec: KleinSpec, stream: &Stream) -> Result<Array> {
    let shape = x.shape();
    if shape.len() != 3 || shape[2] != spec.dim {
        return Err(Error::new(
            "Flux2 Klein",
            format!("QKV {shape:?} is not [B, S, {}]", spec.dim),
        ));
    }
    x.reshape(&[shape[0], shape[1], spec.heads, spec.head], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)
}

fn linear(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    weights.linear(name)?.apply(input, None, stream)
}

fn channel(x: &Array, start: i32, len: i32, stream: &Stream) -> Result<Array> {
    let axis = x.shape().len() - 1;
    slice_axis(x, axis as i32, start, start + len, stream)
}

fn slice_axis(x: &Array, axis: i32, start: i32, stop: i32, stream: &Stream) -> Result<Array> {
    let shape = x.shape();
    if axis < 0 || axis as usize >= shape.len() || start < 0 || stop > shape[axis as usize] {
        return Err(Error::new(
            "Flux2 Klein",
            format!("slice axis {axis} {start}..{stop} is outside {shape:?}"),
        ));
    }
    let mut begins = vec![0; shape.len()];
    let mut ends = shape;
    let strides = vec![1; begins.len()];
    begins[axis as usize] = start;
    ends[axis as usize] = stop;
    x.slice(&begins, &ends, &strides, stream)
}

/// `text` builds `[0, 0, 0, index]`. Otherwise `rows`/`cols` are the image grid
/// and each token is `[0, row, col, 0]` in row-major order.
fn rope_table(
    rows: i32,
    cols: i32,
    head: i32,
    text: bool,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let seq = if text { rows } else { rows * cols };
    if seq < 1 || cols < 1 || head % 4 != 0 {
        return Err(Error::new("Flux2 Klein", "RoPE grid is empty"));
    }
    let mut ids = Vec::with_capacity(seq as usize);
    for index in 0..seq {
        if text {
            ids.push([0, 0, 0, index]);
        } else {
            ids.push([0, index / cols, index % cols, 0]);
        }
    }
    rope_from_ids(&ids, head, stream)
}

fn rope_from_ids(ids: &[[i32; 4]], head: i32, stream: &Stream) -> Result<(Array, Array)> {
    if ids.is_empty() || head % 4 != 0 {
        return Err(Error::new("Flux2 Klein", "RoPE grid is empty"));
    }
    let axis = head / 4;
    let mut cos_axes = Vec::new();
    let mut sin_axes = Vec::new();
    for axis_index in 0..4 {
        let positions: Vec<i32> = ids.iter().map(|id| id[axis_index]).collect();
        let (cos, sin) = axis_table(&positions, axis, stream)?;
        cos_axes.push(cos);
        sin_axes.push(sin);
    }
    Ok((
        Array::concatenate_axis(&cos_axes, 1, stream)?,
        Array::concatenate_axis(&sin_axes, 1, stream)?,
    ))
}

fn axis_table(positions: &[i32], axis: i32, stream: &Stream) -> Result<(Array, Array)> {
    let theta = Array::from_f32_slice(&[ROPE_THETA], &[])?;
    let width = Array::from_f32_slice(&[axis as f32], &[])?;
    let pos = Array::from_i32_slice(positions, &[positions.len() as i32])?
        .astype(Dtype::Float32, stream)?
        .expand_dims(1, stream)?;
    let exponents = Array::arange(0.0, axis as f64, 2.0, Dtype::Float32, stream)?;
    let exponents = Array::divide(&exponents, &width, stream)?;
    let freq = Array::power(&theta, &exponents, stream)?;
    let freq = Array::divide(&Array::from_f32_slice(&[1.0], &[])?, &freq, stream)?
        .expand_dims(0, stream)?;
    let angles = Array::multiply(&pos, &freq, stream)?;
    Ok((angles.cos(stream)?, angles.sin(stream)?))
}

fn apply_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let shape = input.shape();
    let (batch, heads, sequence, dim) = (shape[0], shape[1], shape[2], shape[3]);
    let pairs = dim / 2;
    let input = input
        .astype(Dtype::Float32, stream)?
        .reshape(&[batch, heads, sequence, pairs, 2], stream)?;
    let cos = cos.reshape(&[1, 1, sequence, pairs], stream)?;
    let sin = sin.reshape(&[1, 1, sequence, pairs], stream)?;
    let real = pair_component(&input, 0, stream)?;
    let imag = pair_component(&input, 1, stream)?;
    let out_real = Array::subtract(
        &Array::multiply(&real, &cos, stream)?,
        &Array::multiply(&imag, &sin, stream)?,
        stream,
    )?;
    let out_imag = Array::add(
        &Array::multiply(&real, &sin, stream)?,
        &Array::multiply(&imag, &cos, stream)?,
        stream,
    )?;
    Array::stack(&[out_real, out_imag], -1, stream)?
        .reshape(&[batch, heads, sequence, dim], stream)?
        .astype(dtype, stream)
}

fn pair_component(input: &Array, index: i32, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let axis = shape.len() - 1;
    let mut start = vec![0; shape.len()];
    let mut stop = shape;
    let strides = vec![1; start.len()];
    start[axis] = index;
    stop[axis] = index + 1;
    input
        .slice(&start, &stop, &strides, stream)?
        .squeeze_axis(axis as i32, stream)
}

#[cfg(test)]
mod tests {
    use super::{apply_rope, image_token_ids, rope_table, slice_axis, REF_T_SCALE, ROPE_THETA};
    use danmo_mlx::{Array, Dtype, Stream};

    #[test]
    fn image_token_ids_use_t_then_row_major_hw() {
        let target = image_token_ids(0, 2, 3).unwrap();
        assert_eq!(target.len(), 6);
        assert_eq!(target[0], [0, 0, 0, 0]);
        assert_eq!(target[5], [0, 1, 2, 0]);
        let reference = image_token_ids(REF_T_SCALE, 2, 3).unwrap();
        assert_eq!(reference[0], [REF_T_SCALE, 0, 0, 0]);
        assert_eq!(reference[4], [REF_T_SCALE, 1, 1, 0]);
    }

    fn slice_second(table: &Array, stream: &Stream) -> Array {
        slice_axis(table, 0, 1, 2, stream).unwrap()
    }

    #[test]
    fn rope_matches_the_four_axis_pair_rotation() {
        let stream = Stream::cpu().unwrap();
        let head = 8i32;
        let (cos, sin) = rope_table(2, 1, head, true, &stream).unwrap();
        let cos = slice_second(&cos, &stream);
        let sin = slice_second(&sin, &stream);
        cos.eval().unwrap();
        sin.eval().unwrap();
        let values: Vec<f32> = (0..head as usize).map(|i| i as f32 * 0.1 - 0.3).collect();
        let input = Array::from_f32_slice(&values, &[1, 1, 1, head]).unwrap();
        let got = apply_rope(&input, &cos, &sin, &stream).unwrap();
        got.eval().unwrap();
        let got = got.to_vec_f32(&stream).unwrap();
        let ids = [0.0f32, 0.0, 0.0, 1.0];
        let axis = head / 4;
        let half = (head / 2) as usize;
        let mut expect = values.clone();
        let mut pair = 0usize;
        for axis_i in 0..4 {
            let n = (axis / 2) as usize;
            for i in 0..n {
                let scale = (2 * i) as f32 / axis as f32;
                let ang = ids[axis_i] * ROPE_THETA.powf(-scale);
                let c = ang.cos();
                let s = ang.sin();
                let i0 = pair * 2;
                let i1 = i0 + 1;
                let even = values[i0];
                let odd = values[i1];
                expect[i0] = even * c - odd * s;
                expect[i1] = odd * c + even * s;
                pair += 1;
            }
        }
        assert_eq!(pair, half);
        for (index, (got, want)) in got.iter().zip(&expect).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "index {index} got {got} want {want}"
            );
        }
        let _ = Dtype::Float32;
    }
}
