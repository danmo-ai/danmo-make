//! Qwen-Image joint DiT. This file only appends ggml nodes.
//!
//! Block order matches diffusers `QwenImageTransformerBlock` and main
//! `QwenTransformerBlock`: per-block shift/scale/gate, separate Q/K/V,
//! RMS on Q and K, RoPE, text-then-image attention, gated residual, tanh-GELU MLP.
//! Final `AdaLayerNormContinuous` chunks scale then shift.
//!
//! Attention scale is `1/sqrt(head_dim)` via `flash_attend`. The SD.cpp snapshot
//! at 90e87bc8 hardcodes `1/128` and a `1/32` scale on `to_out.0` / `to_add_out`
//! for a CUDA k-quant NaN. Those scales are not applied.
//! `zero_cond_t` (a second timestep selected per token) is not in this graph.

use super::ids::position_ids;
use super::weights::{weight_layout, QwenImageConfig, LN_EPS};
use crate::runtime::{flash_attend, rope_nd, GraphBuilder, RuntimeError, Tensor, TYPE_F32};
use std::collections::HashMap;

pub struct BuiltDit {
    pub output: Tensor,
}

pub fn build_dit(
    b: &mut GraphBuilder,
    config: QwenImageConfig,
    grid_w: i64,
    grid_h: i64,
    seq_txt: i64,
    types: &HashMap<String, u32>,
) -> Result<BuiltDit, RuntimeError> {
    if config.num_heads < 1
        || config.attn_head_dim < 1
        || config.inner_dim != config.num_heads * config.attn_head_dim
    {
        return Err(RuntimeError::Message(
            "qwen-image inner_dim must equal heads * head_dim".into(),
        ));
    }
    let axis_sum: i64 = config.axes.iter().sum();
    if axis_sum != config.attn_head_dim || config.axes.iter().any(|axis| *axis < 2 || axis % 2 != 0)
    {
        return Err(RuntimeError::Message(format!(
            "qwen-image rope axes {:?} do not sum to head {}",
            config.axes, config.attn_head_dim
        )));
    }
    if grid_w < 1 || grid_h < 1 || seq_txt < 1 {
        return Err(RuntimeError::Message(
            "qwen-image image grid or text length is empty".into(),
        ));
    }
    let seq_img = grid_w * grid_h;
    let mut weights = HashMap::new();
    for (name, ne0, ne1) in weight_layout(config) {
        let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
        let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name, tensor);
    }
    let latent = nn(b.f32_2d("latent", config.in_channels, seq_img, true))?;
    let txt = nn(b.f32_2d("txt", config.joint_attention_dim, seq_txt, true))?;
    let temb_freq = nn(b.f32_2d("temb_freq", config.freq_dim, 1, true))?;
    let mut rope_img = [std::ptr::null_mut(); 3];
    let mut rope_txt = [std::ptr::null_mut(); 3];
    for i in 0..3 {
        rope_img[i] = nn(b.i32_1d(&format!("rope_img_{i}"), seq_img, true))?;
        rope_txt[i] = nn(b.i32_1d(&format!("rope_txt_{i}"), seq_txt, true))?;
    }
    let output = forward(
        b, &weights, config, latent, txt, temb_freq, rope_img, rope_txt, seq_img, seq_txt,
    )?;
    Ok(BuiltDit { output })
}

fn forward(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: QwenImageConfig,
    latent: Tensor,
    context: Tensor,
    temb_freq: Tensor,
    rope_img: [Tensor; 3],
    rope_txt: [Tensor; 3],
    seq_img: i64,
    seq_txt: i64,
) -> Result<Tensor, RuntimeError> {
    let dim = config.inner_dim;
    let hidden = nn(b.silu(biased(
        b,
        temb_freq,
        wt(weights, "time_text_embed.timestep_embedder.linear_1.weight")?,
        wt(weights, "time_text_embed.timestep_embedder.linear_1.bias")?,
    )?))?;
    let temb = biased(
        b,
        hidden,
        wt(weights, "time_text_embed.timestep_embedder.linear_2.weight")?,
        wt(weights, "time_text_embed.timestep_embedder.linear_2.bias")?,
    )?;
    let mut img = biased(
        b,
        latent,
        wt(weights, "img_in.weight")?,
        wt(weights, "img_in.bias")?,
    )?;
    let normed_txt = nn(b.mul(
        nn(b.rms_norm(context, LN_EPS))?,
        wt(weights, "txt_norm.weight")?,
    ))?;
    let mut txt = biased(
        b,
        normed_txt,
        wt(weights, "txt_in.weight")?,
        wt(weights, "txt_in.bias")?,
    )?;
    for i in 0..config.num_layers {
        (txt, img) = block(
            b,
            weights,
            &format!("transformer_blocks.{i}"),
            txt,
            img,
            temb,
            rope_txt,
            rope_img,
            config,
            seq_txt,
            seq_img,
        )?;
    }
    let emb = biased(
        b,
        nn(b.silu(temb))?,
        wt(weights, "norm_out.linear.weight")?,
        wt(weights, "norm_out.linear.bias")?,
    )?;
    // AdaLayerNormContinuous stores scale, then shift.
    let scale = rows(b, emb, 0, dim, dim * 2, 1)?;
    let shift = rows(b, emb, dim, dim, dim * 2, 1)?;
    let normed = scale_shift(b, nn(b.norm(img, LN_EPS))?, scale, shift)?;
    biased(
        b,
        normed,
        wt(weights, "proj_out.weight")?,
        wt(weights, "proj_out.bias")?,
    )
}

fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    txt: Tensor,
    img: Tensor,
    temb: Tensor,
    rope_txt: [Tensor; 3],
    rope_img: [Tensor; 3],
    config: QwenImageConfig,
    seq_txt: i64,
    seq_img: i64,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let dim = config.inner_dim;
    let img_mod = biased(
        b,
        nn(b.silu(temb))?,
        wt(weights, &format!("{prefix}.img_mod.1.weight"))?,
        wt(weights, &format!("{prefix}.img_mod.1.bias"))?,
    )?;
    let txt_mod = biased(
        b,
        nn(b.silu(temb))?,
        wt(weights, &format!("{prefix}.txt_mod.1.weight"))?,
        wt(weights, &format!("{prefix}.txt_mod.1.bias"))?,
    )?;
    let (img_shift_msa, img_scale_msa, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp) =
        mods(b, img_mod, dim)?;
    let (txt_shift_msa, txt_scale_msa, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) =
        mods(b, txt_mod, dim)?;

    let img_n = scale_shift(b, nn(b.norm(img, LN_EPS))?, img_scale_msa, img_shift_msa)?;
    let txt_n = scale_shift(b, nn(b.norm(txt, LN_EPS))?, txt_scale_msa, txt_shift_msa)?;
    let (img_attn, txt_attn) = attention(
        b, weights, prefix, img_n, txt_n, rope_img, rope_txt, config, seq_img, seq_txt,
    )?;
    let img = gate_add(b, img, img_gate_msa, img_attn)?;
    let txt = gate_add(b, txt, txt_gate_msa, txt_attn)?;

    let img_n = scale_shift(b, nn(b.norm(img, LN_EPS))?, img_scale_mlp, img_shift_mlp)?;
    let txt_n = scale_shift(b, nn(b.norm(txt, LN_EPS))?, txt_scale_mlp, txt_shift_mlp)?;
    let img = gate_add(
        b,
        img,
        img_gate_mlp,
        mlp(b, weights, prefix, "img_mlp", img_n)?,
    )?;
    let txt = gate_add(
        b,
        txt,
        txt_gate_mlp,
        mlp(b, weights, prefix, "txt_mlp", txt_n)?,
    )?;
    Ok((txt, img))
}

fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    img: Tensor,
    txt: Tensor,
    rope_img: [Tensor; 3],
    rope_txt: [Tensor; 3],
    config: QwenImageConfig,
    seq_img: i64,
    seq_txt: i64,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let dim = config.inner_dim;
    let seq = seq_txt + seq_img;
    let (q_img, k_img, v_img) = qkv(b, weights, prefix, img, rope_img, config, seq_img, false)?;
    let (q_txt, k_txt, v_txt) = qkv(b, weights, prefix, txt, rope_txt, config, seq_txt, true)?;
    let q = nn(b.cont(b.concat(q_txt, q_img, 2)))?;
    let k = nn(b.cont(b.concat(k_txt, k_img, 2)))?;
    let v = nn(b.cont(b.concat(v_txt, v_img, 2)))?;
    let mixed = nn(b.reshape_2d(flash_attend(b, q, k, v, std::ptr::null_mut())?, dim, seq))?;
    let txt_tok = tokens(b, mixed, dim, 0, seq_txt)?;
    let img_tok = tokens(b, mixed, dim, seq_txt, seq_img)?;
    let img_out = biased(
        b,
        img_tok,
        wt(weights, &format!("{prefix}.attn.to_out.0.weight"))?,
        wt(weights, &format!("{prefix}.attn.to_out.0.bias"))?,
    )?;
    let txt_out = biased(
        b,
        txt_tok,
        wt(weights, &format!("{prefix}.attn.to_add_out.weight"))?,
        wt(weights, &format!("{prefix}.attn.to_add_out.bias"))?,
    )?;
    Ok((img_out, txt_out))
}

fn qkv(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
    rope: [Tensor; 3],
    config: QwenImageConfig,
    seq: i64,
    text: bool,
) -> Result<(Tensor, Tensor, Tensor), RuntimeError> {
    let (q_name, k_name, v_name, qn, kn) = if text {
        (
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.norm_added_q.weight",
            "attn.norm_added_k.weight",
        )
    } else {
        (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.norm_q.weight",
            "attn.norm_k.weight",
        )
    };
    let q = biased(
        b,
        x,
        wt(weights, &format!("{prefix}.{q_name}.weight"))?,
        wt(weights, &format!("{prefix}.{q_name}.bias"))?,
    )?;
    let k = biased(
        b,
        x,
        wt(weights, &format!("{prefix}.{k_name}.weight"))?,
        wt(weights, &format!("{prefix}.{k_name}.bias"))?,
    )?;
    let v = biased(
        b,
        x,
        wt(weights, &format!("{prefix}.{v_name}.weight"))?,
        wt(weights, &format!("{prefix}.{v_name}.bias"))?,
    )?;
    let head = config.attn_head_dim;
    let heads = config.num_heads;
    let q = split_heads(b, q, head, heads, seq)?;
    let k = split_heads(b, k, head, heads, seq)?;
    let v = split_heads(b, v, head, heads, seq)?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, LN_EPS))?,
        wt(weights, &format!("{prefix}.{qn}"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, LN_EPS))?,
        wt(weights, &format!("{prefix}.{kn}"))?,
    ))?;
    let q = rope_nd(b, q, &rope, &config.axes, config.rope_theta)?;
    let k = rope_nd(b, k, &rope, &config.axes, config.rope_theta)?;
    Ok((q, k, v))
}

fn mlp(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    side: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let hidden = nn(b.gelu(biased(
        b,
        x,
        wt(weights, &format!("{prefix}.{side}.net.0.proj.weight"))?,
        wt(weights, &format!("{prefix}.{side}.net.0.proj.bias"))?,
    )?))?;
    biased(
        b,
        hidden,
        wt(weights, &format!("{prefix}.{side}.net.2.weight"))?,
        wt(weights, &format!("{prefix}.{side}.net.2.bias"))?,
    )
}

/// Bind the position ids this graph declared. Create uses one frame at index 0.
pub fn bind_rope(
    graph: &crate::runtime::ReusedGraph,
    grid_w: i64,
    grid_h: i64,
    seq_txt: i64,
) -> Result<(), RuntimeError> {
    let ids = position_ids(grid_h, grid_w, seq_txt)?;
    for axis in 0..3 {
        let image: Vec<i32> = ids.image.iter().map(|row| row[axis]).collect();
        let text: Vec<i32> = ids.text.iter().map(|row| row[axis]).collect();
        graph.set_i32(&format!("rope_img_{axis}"), &image)?;
        graph.set_i32(&format!("rope_txt_{axis}"), &text)?;
    }
    Ok(())
}

fn mods(
    b: &GraphBuilder,
    packed: Tensor,
    dim: i64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor), RuntimeError> {
    let total = dim * 6;
    Ok((
        rows(b, packed, 0, dim, total, 1)?,
        rows(b, packed, dim, dim, total, 1)?,
        rows(b, packed, dim * 2, dim, total, 1)?,
        rows(b, packed, dim * 3, dim, total, 1)?,
        rows(b, packed, dim * 4, dim, total, 1)?,
        rows(b, packed, dim * 5, dim, total, 1)?,
    ))
}

fn rows(
    b: &GraphBuilder,
    tensor: Tensor,
    row0: i64,
    rows_n: i64,
    total_rows: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let view = b.view_2d(
        tensor,
        rows_n,
        seq,
        (total_rows as usize) * std::mem::size_of::<f32>(),
        (row0 as usize) * std::mem::size_of::<f32>(),
    );
    nn(b.cont(view))
}

fn tokens(
    b: &GraphBuilder,
    tensor: Tensor,
    dim: i64,
    token0: i64,
    count: i64,
) -> Result<Tensor, RuntimeError> {
    let view = b.view_2d(
        tensor,
        dim,
        count,
        (dim as usize) * std::mem::size_of::<f32>(),
        (token0 as usize) * (dim as usize) * std::mem::size_of::<f32>(),
    );
    nn(b.cont(view))
}

fn split_heads(
    b: &GraphBuilder,
    x: Tensor,
    head: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    nn(b.cont(b.reshape_3d(x, head, heads, seq)))
}

fn scale_shift(
    b: &GraphBuilder,
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
) -> Result<Tensor, RuntimeError> {
    let scaled = nn(b.mul(x, scale))?;
    let y = nn(b.add(x, scaled))?;
    nn(b.add(y, shift))
}

fn gate_add(
    b: &GraphBuilder,
    residual: Tensor,
    gate: Tensor,
    delta: Tensor,
) -> Result<Tensor, RuntimeError> {
    nn(b.add(residual, nn(b.mul(delta, gate))?))
}

fn biased(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.project(weight, x))?;
    nn(b.add(y, nn(b.repeat(bias, y))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}
