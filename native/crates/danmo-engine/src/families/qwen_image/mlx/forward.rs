//! Qwen-Image DiT forward. Names are the flat mlx-community layout
//! (`img_in`, `txt_norm`, `transformer_blocks.N.*`, `norm_out.linear`,
//! `proj_out`); math follows danmo-make `families/qwen/transformer_mlx.py`
//! and diffusers `QwenImageTransformer2DModel`.
//!
//! 60 double-stream blocks. Modulation triples are shift, scale, gate;
//! `AdaLayerNormContinuous` is scale then shift. Q/K RMS runs in F32 and
//! casts back. RoPE is complex-pair rotation on the 3-axis table
//! (frame 16 / height 56 / width 56, centered spatial positions).

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use crate::weights::MlxWeights;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10000.0;
const AXES: [i32; 3] = [16, 56, 56];
const TIMESTEP_FREQ: i32 = 256;

#[derive(Debug, Clone, Copy)]
pub(super) struct QwenSpec {
    pub dim: i32,
    pub heads: i32,
    pub head: i32,
    pub in_channels: i32,
    pub text_dim: i32,
    pub layers: u32,
}

pub(super) fn assert_contract(weights: &MlxWeights) -> Result<QwenSpec> {
    let (dim, in_channels) = matrix(weights, "img_in.weight")?;
    if in_channels != 64 {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("packed latent has {in_channels} channels, expected 64"),
        ));
    }
    let (txt_out, text_dim) = matrix(weights, "txt_in.weight")?;
    expect_eq("txt_in out", txt_out, dim)?;
    let norm = weights.tensor("txt_norm.weight")?.shape();
    if norm != vec![text_dim as i32] {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("txt_norm {norm:?} does not match text width {text_dim}"),
        ));
    }
    let (time_out, freq) = matrix(weights, "time_text_embed.timestep_embedder.linear_1.weight")?;
    expect_eq("timestep linear_1 out", time_out, dim)?;
    expect_eq("timestep linear_1 in", freq, i64::from(TIMESTEP_FREQ))?;
    let (time2_out, time2_in) =
        matrix(weights, "time_text_embed.timestep_embedder.linear_2.weight")?;
    expect_eq("timestep linear_2 out", time2_out, dim)?;
    expect_eq("timestep linear_2 in", time2_in, dim)?;
    let head = vector_len(weights, "transformer_blocks.0.attn.norm_q.weight")?;
    if head < 8 || dim % head != 0 {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("head dim {head} does not divide width {dim}"),
        ));
    }
    let head_pairs = head / 2;
    let axis_pairs: i64 = AXES.iter().map(|axis| i64::from(axis / 2)).sum();
    if head_pairs != axis_pairs {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("head pairs {head_pairs} do not match RoPE axes {AXES:?}"),
        ));
    }
    let (mod_out, mod_in) = matrix(weights, "transformer_blocks.0.img_mod_linear.weight")?;
    expect_eq("img_mod_linear out", mod_out, dim * 6)?;
    expect_eq("img_mod_linear in", mod_in, dim)?;
    let (mod_out, mod_in) = matrix(weights, "transformer_blocks.0.txt_mod_linear.weight")?;
    expect_eq("txt_mod_linear out", mod_out, dim * 6)?;
    expect_eq("txt_mod_linear in", mod_in, dim)?;
    let (norm_out, norm_in) = matrix(weights, "norm_out.linear.weight")?;
    expect_eq("norm_out out", norm_out, dim * 2)?;
    expect_eq("norm_out in", norm_in, dim)?;
    let (proj_out, proj_in) = matrix(weights, "proj_out.weight")?;
    expect_eq("proj_out in", proj_in, dim)?;
    expect_eq("proj_out out (patch2 x 16)", proj_out, 64)?;
    let mut layers = 0u32;
    while weights.has(&format!(
        "transformer_blocks.{layers}.img_mod_linear.weight"
    )) {
        layers += 1;
    }
    if layers == 0 {
        return Err(Error::new(
            "Qwen-Image DiT",
            "no transformer_blocks.* found",
        ));
    }
    Ok(QwenSpec {
        dim: i32_dim(dim, "width")?,
        heads: i32_dim(dim / head, "heads")?,
        head: i32_dim(head, "head")?,
        in_channels: i32_dim(in_channels, "channels")?,
        text_dim: i32_dim(text_dim, "text width")?,
        layers,
    })
}

fn matrix(weights: &MlxWeights, name: &str) -> Result<(i64, i64)> {
    let Some((out, inn)) = weights.linear(name)?.out_in()? else {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("{name} is not a matrix"),
        ));
    };
    if out < 1 || inn < 1 {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("{name} shape {out}x{inn} is empty"),
        ));
    }
    Ok((out, inn))
}

fn expect_eq(what: &str, got: i64, expected: i64) -> Result<()> {
    if got != expected {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("{what} is {got}, expected {expected}"),
        ));
    }
    Ok(())
}

fn i32_dim(value: i64, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        Error::new(
            "Qwen-Image DiT",
            format!("{what} {value} does not fit in i32"),
        )
    })
}

fn vector_len(weights: &MlxWeights, name: &str) -> Result<i64> {
    let shape = weights.tensor(name)?.shape();
    let len: i64 = shape.iter().map(|dim| i64::from(*dim)).product();
    if len < 1 {
        return Err(Error::new("Qwen-Image DiT", format!("{name} is empty")));
    }
    Ok(len)
}

/// One denoise step. `latent` is `[B, seq, 64]`, `text` is `[B, seq_txt,
/// text_dim]`, `sigma` is the flow-match sigma in `[0, 1]` (timestep features
/// scale it by 1000). Returns `[B, seq, 64]` velocity.
pub(super) fn forward(
    weights: &MlxWeights,
    spec: QwenSpec,
    latent: &Array,
    text: &Array,
    sigma: f32,
    rope: &RopeTables,
    txt_mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 3 || shape[2] != spec.in_channels {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!(
                "latent shape {shape:?} is not [B, seq, {}]",
                spec.in_channels
            ),
        ));
    }
    let batch = shape[0];
    let seq_img = shape[1];
    let text_shape = text.shape();
    if text_shape.len() != 3 || text_shape[0] != batch || text_shape[2] != spec.text_dim {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("text shape {text_shape:?} does not match latent batch {batch}"),
        ));
    }
    let seq_txt = text_shape[1];
    if rope.img_cos.shape()[0] != seq_img || rope.txt_cos.shape()[0] != seq_txt {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!(
                "RoPE tables img {} / txt {} do not match seq img {seq_img} / txt {seq_txt}",
                rope.img_cos.shape()[0],
                rope.txt_cos.shape()[0]
            ),
        ));
    }
    if !sigma.is_finite() {
        return Err(Error::new(
            "Qwen-Image DiT",
            format!("sigma {sigma} is not finite"),
        ));
    }

    let dtype = latent.dtype();
    let temb = timestep_embed(weights, sigma, dtype, stream)?;

    let mut img = linear(weights, "img_in.weight", latent, stream)?;
    let txt_norm = weights
        .tensor("txt_norm.weight")?
        .astype(text.dtype(), stream)?;
    let normed_text = text.rms_norm(Some(&txt_norm), RMS_EPS, stream)?;
    let mut txt = linear(weights, "txt_in.weight", &normed_text, stream)?;
    // One Metal command buffer for the whole 60-layer BF16 step times out.
    // Materialize each block so the buffer stays one layer.
    img.eval()?;
    txt.eval()?;

    let conditioned = temb.silu(stream)?;
    conditioned.eval()?;
    for index in 0..spec.layers {
        (txt, img) = double_block(
            weights,
            spec,
            index,
            &txt,
            &img,
            &conditioned,
            rope,
            txt_mask,
            stream,
        )?;
        txt.eval()?;
        img.eval()?;
    }
    // AdaLayerNormContinuous: scale first, shift second, x * (1 + scale) + shift.
    let normed = img.layer_norm(LN_EPS, stream)?;
    let out_mod = linear(weights, "norm_out.linear.weight", &conditioned, stream)?;
    let scale = channel(&out_mod, 0, spec.dim, stream)?;
    let shift = channel(&out_mod, spec.dim, spec.dim, stream)?;
    let one = Array::from_f32_slice(&[1.0], &[])?.astype(dtype, stream)?;
    let scale = Array::add(&one, &scale, stream)?.expand_dims(1, stream)?;
    let shift = shift.expand_dims(1, stream)?;
    let modulated = Array::add(&Array::multiply(&normed, &scale, stream)?, &shift, stream)?;
    linear(weights, "proj_out.weight", &modulated, stream)
}

fn double_block(
    weights: &MlxWeights,
    spec: QwenSpec,
    index: u32,
    txt: &Array,
    img: &Array,
    conditioned: &Array,
    rope: &RopeTables,
    txt_mask: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let prefix = format!("transformer_blocks.{index}");
    let img_mod = linear(
        weights,
        &format!("{prefix}.img_mod_linear.weight"),
        conditioned,
        stream,
    )?;
    let txt_mod = linear(
        weights,
        &format!("{prefix}.txt_mod_linear.weight"),
        conditioned,
        stream,
    )?;
    let (i_shift1, i_scale1, i_gate1, i_shift2, i_scale2, i_gate2) =
        mods(&img_mod, spec.dim, stream)?;
    let (t_shift1, t_scale1, t_gate1, t_shift2, t_scale2, t_gate2) =
        mods(&txt_mod, spec.dim, stream)?;

    let n_img = modulate(
        &img.layer_norm(LN_EPS, stream)?,
        &i_scale1,
        &i_shift1,
        stream,
    )?;
    let n_txt = modulate(
        &txt.layer_norm(LN_EPS, stream)?,
        &t_scale1,
        &t_shift1,
        stream,
    )?;
    let (img_attn, txt_attn) = joint_attention(
        weights, spec, &prefix, &n_img, &n_txt, rope, txt_mask, stream,
    )?;
    let img = gated_add(img, &i_gate1, &img_attn, stream)?;
    let txt = gated_add(txt, &t_gate1, &txt_attn, stream)?;

    let n_img = modulate(
        &img.layer_norm(LN_EPS, stream)?,
        &i_scale2,
        &i_shift2,
        stream,
    )?;
    let img_ff = feed_forward(weights, &format!("{prefix}.img_ff"), &n_img, stream)?;
    let img = gated_add(&img, &i_gate2, &img_ff, stream)?;
    let n_txt = modulate(
        &txt.layer_norm(LN_EPS, stream)?,
        &t_scale2,
        &t_shift2,
        stream,
    )?;
    let txt_ff = feed_forward(weights, &format!("{prefix}.txt_ff"), &n_txt, stream)?;
    let txt = gated_add(&txt, &t_gate2, &txt_ff, stream)?;
    Ok((txt, img))
}

fn joint_attention(
    weights: &MlxWeights,
    spec: QwenSpec,
    prefix: &str,
    img: &Array,
    txt: &Array,
    rope: &RopeTables,
    txt_mask: Option<&Array>,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let seq_txt = txt.shape()[1];
    let attn = format!("{prefix}.attn");
    let (q_img, k_img, v_img) = project_qkv(
        weights,
        img,
        &format!("{attn}.to_q.weight"),
        &format!("{attn}.to_k.weight"),
        &format!("{attn}.to_v.weight"),
        &format!("{attn}.norm_q.weight"),
        &format!("{attn}.norm_k.weight"),
        spec,
        stream,
    )?;
    let (q_txt, k_txt, v_txt) = project_qkv(
        weights,
        txt,
        &format!("{attn}.add_q_proj.weight"),
        &format!("{attn}.add_k_proj.weight"),
        &format!("{attn}.add_v_proj.weight"),
        &format!("{attn}.norm_added_q.weight"),
        &format!("{attn}.norm_added_k.weight"),
        spec,
        stream,
    )?;
    let q_img = apply_rope_bshd(&q_img, &rope.img_cos, &rope.img_sin, stream)?;
    let k_img = apply_rope_bshd(&k_img, &rope.img_cos, &rope.img_sin, stream)?;
    let q_txt = apply_rope_bshd(&q_txt, &rope.txt_cos, &rope.txt_sin, stream)?;
    let k_txt = apply_rope_bshd(&k_txt, &rope.txt_cos, &rope.txt_sin, stream)?;

    let q = Array::concatenate_axis(&[q_txt, q_img], 1, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = Array::concatenate_axis(&[k_txt, k_img], 1, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = Array::concatenate_axis(&[v_txt, v_img], 1, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let dtype = q.dtype();
    let mixed = Array::scaled_dot_product_attention(
        &q.astype(Dtype::Float32, stream)?,
        &k.astype(Dtype::Float32, stream)?,
        &v.astype(Dtype::Float32, stream)?,
        1.0 / (spec.head as f32).sqrt(),
        "",
        txt_mask,
        stream,
    )?
    .astype(dtype, stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?;
    let joint_seq = mixed.shape()[1];
    let mixed = mixed.reshape(&[mixed.shape()[0], joint_seq, spec.dim], stream)?;
    let txt_out = slice_axis(&mixed, 1, 0, seq_txt, stream)?;
    let img_out = slice_axis(&mixed, 1, seq_txt, joint_seq, stream)?;
    let img_out = linear(
        weights,
        &format!("{attn}.attn_to_out.0.weight"),
        &img_out,
        stream,
    )?;
    let txt_out = linear(
        weights,
        &format!("{attn}.to_add_out.weight"),
        &txt_out,
        stream,
    )?;
    Ok((img_out, txt_out))
}

fn project_qkv(
    weights: &MlxWeights,
    input: &Array,
    q_name: &str,
    k_name: &str,
    v_name: &str,
    q_norm: &str,
    k_norm: &str,
    spec: QwenSpec,
    stream: &Stream,
) -> Result<(Array, Array, Array)> {
    let shape = input.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let head_shape = [batch, sequence, spec.heads, spec.head];
    let q = linear(weights, q_name, input, stream)?.reshape(&head_shape, stream)?;
    let k = linear(weights, k_name, input, stream)?.reshape(&head_shape, stream)?;
    let v = linear(weights, v_name, input, stream)?.reshape(&head_shape, stream)?;
    let q = rms_heads(&q, weights.tensor(q_norm)?, stream)?;
    let k = rms_heads(&k, weights.tensor(k_norm)?, stream)?;
    Ok((q, k, v))
}

fn rms_heads(x: &Array, scale: &Array, stream: &Stream) -> Result<Array> {
    let dtype = x.dtype();
    let x = x.astype(Dtype::Float32, stream)?;
    let scale = scale.astype(Dtype::Float32, stream)?;
    x.rms_norm(Some(&scale), RMS_EPS, stream)?
        .astype(dtype, stream)
}

fn feed_forward(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let hidden = linear(weights, &format!("{prefix}.mlp_in.weight"), input, stream)?;
    let hidden = crate::common::mlx_ops::gelu_tanh(&hidden, stream)?;
    linear(
        weights,
        &format!("{prefix}.mlp_out.weight"),
        &hidden,
        stream,
    )
}

fn timestep_embed(
    weights: &MlxWeights,
    sigma: f32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array> {
    let features = super::super::ids::timestep_features(sigma, TIMESTEP_FREQ as usize);
    let freq = Array::from_f32_slice(&features, &[1, TIMESTEP_FREQ])?;
    let temb = linear(
        weights,
        "time_text_embed.timestep_embedder.linear_1.weight",
        &freq,
        stream,
    )?;
    let temb = linear(
        weights,
        "time_text_embed.timestep_embedder.linear_2.weight",
        &temb.silu(stream)?,
        stream,
    )?;
    temb.astype(dtype, stream)
}

/// Six equal chunks: shift, scale, gate, shift_mlp, scale_mlp, gate_mlp.
/// Each is `[B, 1, dim]` for broadcasting over the sequence.
fn mods(
    packed: &Array,
    dim: i32,
    stream: &Stream,
) -> Result<(Array, Array, Array, Array, Array, Array)> {
    let mut out = Vec::with_capacity(6);
    for index in 0..6 {
        out.push(channel(packed, index * dim, dim, stream)?.expand_dims(1, stream)?);
    }
    Ok((
        out[0].clone(),
        out[1].clone(),
        out[2].clone(),
        out[3].clone(),
        out[4].clone(),
        out[5].clone(),
    ))
}

fn modulate(x: &Array, scale: &Array, shift: &Array, stream: &Stream) -> Result<Array> {
    let one = Array::from_f32_slice(&[1.0], &[])?.astype(x.dtype(), stream)?;
    let scale = Array::add(&one, scale, stream)?;
    Array::add(&Array::multiply(x, &scale, stream)?, shift, stream)
}

fn gated_add(base: &Array, gate: &Array, delta: &Array, stream: &Stream) -> Result<Array> {
    Array::add(base, &Array::multiply(gate, delta, stream)?, stream)
}

fn linear(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    let bias_name = name.replace(".weight", ".bias");
    let bias = if weights.has(&bias_name) {
        Some(weights.tensor(&bias_name)?)
    } else {
        None
    };
    weights.linear(name)?.apply(input, bias, stream)
}

fn channel(x: &Array, start: i32, len: i32, stream: &Stream) -> Result<Array> {
    let axis = x.shape().len() - 1;
    slice_axis(x, axis as i32, start, start + len, stream)
}

fn slice_axis(x: &Array, axis: i32, start: i32, stop: i32, stream: &Stream) -> Result<Array> {
    let shape = x.shape();
    if axis < 0 || axis as usize >= shape.len() || start < 0 || stop > shape[axis as usize] {
        return Err(Error::new(
            "Qwen-Image DiT",
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

/// Complex-pair rotation on `[B, S, H, D]` with `[S, D/2]` tables, F32 math.
fn apply_rope_bshd(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let shape = input.shape();
    let (batch, sequence, heads, dim) = (shape[0], shape[1], shape[2], shape[3]);
    let pairs = dim / 2;
    let input = input
        .astype(Dtype::Float32, stream)?
        .reshape(&[batch, sequence, heads, pairs, 2], stream)?;
    let cos = cos.reshape(&[1, sequence, 1, pairs], stream)?;
    let sin = sin.reshape(&[1, sequence, 1, pairs], stream)?;
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
        .reshape(&[batch, sequence, heads, dim], stream)?
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

/// RoPE tables for one create geometry: `[seq_img, 64]` image and
/// `[seq_txt, 64]` text cos/sin. Built once per denoise run.
pub(super) struct RopeTables {
    pub img_cos: Array,
    pub img_sin: Array,
    pub txt_cos: Array,
    pub txt_sin: Array,
}

pub(super) fn rope_tables(
    grid_h: i64,
    grid_w: i64,
    seq_txt: usize,
    stream: &Stream,
) -> Result<RopeTables> {
    let ids = super::super::ids::position_ids(grid_h, grid_w, seq_txt as i64)
        .map_err(|error| Error::new("Qwen-Image DiT", error.to_string()))?;
    let (img_cos, img_sin) = rope_from_positions(&ids.image, stream)?;
    let (txt_cos, txt_sin) = rope_from_positions(&ids.text, stream)?;
    Ok(RopeTables {
        img_cos,
        img_sin,
        txt_cos,
        txt_sin,
    })
}

pub(super) fn rope_from_positions(
    positions: &[[i32; 3]],
    stream: &Stream,
) -> Result<(Array, Array)> {
    if positions.is_empty() {
        return Err(Error::new("Qwen-Image DiT", "RoPE position list is empty"));
    }
    let mut cos_axes = Vec::new();
    let mut sin_axes = Vec::new();
    for axis in 0..3 {
        let width = AXES[axis];
        let coords: Vec<i32> = positions.iter().map(|id| id[axis]).collect();
        let (cos, sin) = axis_table(&coords, width, stream)?;
        cos_axes.push(cos);
        sin_axes.push(sin);
    }
    Ok((
        Array::concatenate_axis(&cos_axes, 1, stream)?,
        Array::concatenate_axis(&sin_axes, 1, stream)?,
    ))
}

fn axis_table(positions: &[i32], width: i32, stream: &Stream) -> Result<(Array, Array)> {
    let theta = Array::from_f32_slice(&[ROPE_THETA], &[])?;
    let dim = Array::from_f32_slice(&[width as f32], &[])?;
    let pos = Array::from_i32_slice(positions, &[positions.len() as i32])?
        .astype(Dtype::Float32, stream)?
        .expand_dims(1, stream)?;
    let exponents = Array::arange(0.0, f64::from(width), 2.0, Dtype::Float32, stream)?;
    let exponents = Array::divide(&exponents, &dim, stream)?;
    let freq = Array::power(&theta, &exponents, stream)?;
    let freq = Array::divide(&Array::from_f32_slice(&[1.0], &[])?, &freq, stream)?
        .expand_dims(0, stream)?;
    let angles = Array::multiply(&pos, &freq, stream)?;
    Ok((angles.cos(stream)?, angles.sin(stream)?))
}

/// Padding bias for batched CFG: `[B, 1, 1, seq_txt + seq_img]`, 0 for valid
/// text keys and all image keys, -1e9 for padded text keys (main
/// `_convert_mask_for_qwen`).
/// Additive SDPA bias `[B, 1, 1, txt + img]`. Each batch row keeps its own
/// text keys; image keys stay visible. Matches mflux `_convert_mask_for_qwen`.
pub(super) fn joint_text_bias(
    rows_valid: &[Vec<bool>],
    seq_img: i32,
    stream: &Stream,
) -> Result<Array> {
    let Some(first) = rows_valid.first() else {
        return Err(Error::new("Qwen-Image DiT", "text mask has no batch row"));
    };
    let seq_txt = first.len() as i32;
    let joint = seq_txt + seq_img;
    let mut rows = Vec::with_capacity(rows_valid.len() * joint as usize);
    for valid in rows_valid {
        if valid.len() != seq_txt as usize {
            return Err(Error::new(
                "Qwen-Image DiT",
                format!("text mask row has {} keys, expected {seq_txt}", valid.len()),
            ));
        }
        for keep in valid {
            rows.push(if *keep { 0.0 } else { -1e9 });
        }
        rows.extend(std::iter::repeat_n(0.0, seq_img as usize));
    }
    let bias = Array::from_f32_slice(&rows, &[rows_valid.len() as i32, 1, 1, joint])?;
    let _ = stream;
    Ok(bias)
}

#[cfg(test)]
mod tests {
    use super::{joint_text_bias, rope_tables, AXES};
    use danmo_mlx::Stream;

    #[test]
    fn rope_tables_match_the_centered_diffusers_positions() {
        let stream = Stream::cpu().unwrap();
        let tables = rope_tables(2, 2, 2, &stream).unwrap();
        assert_eq!(tables.img_cos.shape(), vec![4, 64]);
        assert_eq!(tables.txt_cos.shape(), vec![2, 64]);
        tables.img_cos.eval().unwrap();
        let cos = tables.img_cos.to_vec_f32(&stream).unwrap();
        // First image token is [0, -1, -1]: frame angle 0, spatial -1 * omega0.
        assert!((cos[0] - 1.0).abs() < 1e-6);
        let pairs_per_axis: Vec<usize> = AXES.iter().map(|a| (a / 2) as usize).collect();
        let height_start = pairs_per_axis[0];
        let angle = -1.0f32 * 10000f32.powf(0.0);
        assert!((cos[height_start] - angle.cos()).abs() < 1e-5);
        // Text starts at max(h/2, w/2) = 1 on all axes.
        tables.txt_cos.eval().unwrap();
        let txt = tables.txt_cos.to_vec_f32(&stream).unwrap();
        assert!((txt[0] - 1.0f32.cos()).abs() < 1e-5);
    }

    #[test]
    fn joint_text_bias_keeps_each_branch_mask() {
        let stream = Stream::cpu().unwrap();
        let bias = joint_text_bias(&[vec![true, true], vec![true, false]], 1, &stream).unwrap();
        assert_eq!(bias.shape(), vec![2, 1, 1, 3]);
        bias.eval().unwrap();
        let values = bias.to_vec_f32(&stream).unwrap();
        assert_eq!(values, vec![0.0, 0.0, 0.0, 0.0, -1e9, 0.0]);
    }
}
