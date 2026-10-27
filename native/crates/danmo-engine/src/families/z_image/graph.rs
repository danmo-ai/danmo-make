//! Z-Image DiT. Tensor names are the unsloth / SD.cpp GGUF names.
//!
//! The block math matches Tongyi `transformer.py` and SD.cpp `z_image.hpp`:
//! fused qkv (q then k then v), RMSNorm, complex RoPE, SwiGLU, AdaLN
//! `x * (1 + scale)` and `tanh` gates, final LayerNorm, then negate.
//! Attention scale is `1/sqrt(head)`. SD.cpp's `1/128` rescales K and V and
//! divides the same factor back out; it is an overflow guard, not a second scale.
//! The w2 `1/128` scale is the same kind of cancel. Neither is applied.
//!
//! Sequence order is image then caption, which is the training order. SD.cpp
//! concatenates caption then image; with full attention the image tokens match.

use super::geom::{TokenGrid, FINAL_EPS, RMS_EPS};
use super::weights::{weight_layout, ZImageConfig};
use crate::runtime::{flash_attend, rope_nd, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{bind_lora_factors, LoraFactor};
use std::collections::HashMap;

pub struct BuiltDit {
    pub output: Tensor,
}

pub fn build_dit(
    b: &mut GraphBuilder,
    config: ZImageConfig,
    grid: TokenGrid,
    types: &HashMap<String, u32>,
    loras: &[LoraFactor],
) -> Result<BuiltDit, RuntimeError> {
    if config.heads < 1 || config.head_dim < 2 || grid.cap_real < 1 || grid.img_real < 1 {
        return Err(RuntimeError::Message(
            "z-image graph has an empty head, caption, or image".into(),
        ));
    }
    if config.axes.iter().sum::<i64>() != config.head_dim {
        return Err(RuntimeError::Message(
            "z-image rope axes do not sum to the head dim".into(),
        ));
    }
    let mut weights = HashMap::new();
    for (name, ne0, ne1) in weight_layout(config) {
        let ggml_type = types
            .get(&name)
            .copied()
            .unwrap_or(crate::runtime::TYPE_F32);
        let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name, tensor);
    }
    bind_lora_factors(b, &weights, loras)?;
    let latent = nn(b.f32_3d(
        "latent",
        grid.latent_w,
        grid.latent_h,
        config.in_channels,
        true,
    ))?;
    let txt = nn(b.f32_2d("txt", config.cap_dim, grid.cap_real, true))?;
    let temb = nn(b.f32_2d("temb_freq", config.freq, 1, true))?;
    let mut rope = [std::ptr::null_mut(); 3];
    for (axis, slot) in rope.iter_mut().enumerate() {
        *slot = nn(b.i32_1d(&format!("rope_axis_{axis}"), grid.unified, true))?;
    }
    let index = nn(b.i32_1d("patch_index", grid.img_real * config.token_dim(), true))?;
    let output = forward(b, &weights, config, grid, latent, txt, temb, rope, index)?;
    Ok(BuiltDit { output })
}

fn forward(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: ZImageConfig,
    grid: TokenGrid,
    latent: Tensor,
    txt: Tensor,
    temb: Tensor,
    rope: [Tensor; 3],
    patch_rows: Tensor,
) -> Result<Tensor, RuntimeError> {
    let t_emb = timestep(b, weights, temb)?;
    let image = embed_image(b, weights, config, grid, latent, patch_rows)?;
    let caption = embed_caption(b, weights, config, grid, txt)?;
    let image_rope = span(b, &rope, 0, grid.img_len)?;
    let caption_rope = span(b, &rope, grid.img_len, grid.cap_len)?;
    let mut image = image;
    for i in 0..config.refiners {
        image = block(
            b,
            weights,
            &format!("noise_refiner.{i}"),
            image,
            &image_rope,
            Some(t_emb),
            config,
            grid.img_len,
            true,
        )?;
    }
    let mut caption = caption;
    for i in 0..config.refiners {
        caption = block(
            b,
            weights,
            &format!("context_refiner.{i}"),
            caption,
            &caption_rope,
            None,
            config,
            grid.cap_len,
            false,
        )?;
    }
    let mut unified = nn(b.cont(b.concat(image, caption, 1)))?;
    for i in 0..config.layers {
        unified = block(
            b,
            weights,
            &format!("layers.{i}"),
            unified,
            &rope,
            Some(t_emb),
            config,
            grid.unified,
            true,
        )?;
    }
    let tokens = final_layer(b, weights, unified, t_emb)?;
    let real = rows(b, tokens, 0, grid.img_real, config.token_dim())?;
    let negated = nn(b.scale(real, -1.0))?;
    let flat_tokens = nn(b.reshape_2d(negated, 1, grid.img_real * config.token_dim()))?;
    let flat_latent = nn(b.reshape_2d(
        latent,
        1,
        grid.latent_w * grid.latent_h * config.in_channels,
    ))?;
    let scattered = nn(b.get_rows_back(flat_tokens, patch_rows, flat_latent))?;
    nn(b.reshape_3d(scattered, grid.latent_w, grid.latent_h, config.in_channels))
}

fn timestep(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    temb: Tensor,
) -> Result<Tensor, RuntimeError> {
    let hidden = add_bias(
        b,
        nn(linear(b, temb, wt(weights, "t_embedder.mlp.0.weight")?))?,
        wt(weights, "t_embedder.mlp.0.bias")?,
    )?;
    add_bias(
        b,
        nn(linear(
            b,
            nn(b.silu(hidden))?,
            wt(weights, "t_embedder.mlp.2.weight")?,
        ))?,
        wt(weights, "t_embedder.mlp.2.bias")?,
    )
}

fn embed_image(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: ZImageConfig,
    grid: TokenGrid,
    latent: Tensor,
    patch_rows: Tensor,
) -> Result<Tensor, RuntimeError> {
    let flat = nn(b.reshape_2d(
        latent,
        1,
        grid.latent_w * grid.latent_h * config.in_channels,
    ))?;
    let gathered = nn(b.get_rows(flat, patch_rows))?;
    let tokens = nn(b.reshape_2d(gathered, config.token_dim(), grid.img_real))?;
    let embedded = add_bias(
        b,
        nn(linear(b, tokens, wt(weights, "x_embedder.weight")?))?,
        wt(weights, "x_embedder.bias")?,
    )?;
    pad_tokens(
        b,
        embedded,
        wt(weights, "x_pad_token")?,
        config.dim,
        grid.img_pad,
    )
}

fn embed_caption(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: ZImageConfig,
    grid: TokenGrid,
    txt: Tensor,
) -> Result<Tensor, RuntimeError> {
    let normed = rms(b, txt, wt(weights, "cap_embedder.0.weight")?, RMS_EPS)?;
    let embedded = add_bias(
        b,
        nn(linear(b, normed, wt(weights, "cap_embedder.1.weight")?))?,
        wt(weights, "cap_embedder.1.bias")?,
    )?;
    pad_tokens(
        b,
        embedded,
        wt(weights, "cap_pad_token")?,
        config.dim,
        grid.cap_pad,
    )
}

fn pad_tokens(
    b: &GraphBuilder,
    tokens: Tensor,
    pad: Tensor,
    dim: i64,
    pad_len: i64,
) -> Result<Tensor, RuntimeError> {
    if pad_len == 0 {
        return Ok(tokens);
    }
    let canvas = b.blank_f32_2d(dim, pad_len);
    let repeated = nn(b.repeat(pad, canvas))?;
    nn(b.cont(b.concat(tokens, repeated, 1)))
}

fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
    rope: &[Tensor; 3],
    t_emb: Option<Tensor>,
    config: ZImageConfig,
    seq: i64,
    modulated: bool,
) -> Result<Tensor, RuntimeError> {
    let dim = config.dim;
    let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if modulated {
        let t_emb = t_emb
            .ok_or_else(|| RuntimeError::Message(format!("{prefix} is missing its adaLN input")))?;
        let packed = add_bias(
            b,
            nn(linear(
                b,
                t_emb,
                wt(weights, &format!("{prefix}.adaLN_modulation.0.weight"))?,
            ))?,
            wt(weights, &format!("{prefix}.adaLN_modulation.0.bias"))?,
        )?;
        // `modulate` applies `x * (1 + scale)`. The chunks stay the raw linear outputs.
        (
            chunk(b, packed, 0, dim)?,
            nn(b.tanh(chunk(b, packed, dim, dim)?))?,
            chunk(b, packed, dim * 2, dim)?,
            nn(b.tanh(chunk(b, packed, dim * 3, dim)?))?,
        )
    } else {
        let one = std::ptr::null_mut();
        (one, one, one, one)
    };
    let normed = rms(
        b,
        x,
        wt(weights, &format!("{prefix}.attention_norm1.weight"))?,
        RMS_EPS,
    )?;
    let normed = if modulated {
        modulate(b, normed, scale_msa)?
    } else {
        normed
    };
    let attn = attention(b, weights, prefix, normed, rope, config, seq)?;
    let attn = rms(
        b,
        attn,
        wt(weights, &format!("{prefix}.attention_norm2.weight"))?,
        RMS_EPS,
    )?;
    let x = if modulated {
        gate_add(b, x, gate_msa, attn)?
    } else {
        nn(b.add(x, attn))?
    };
    let normed = rms(
        b,
        x,
        wt(weights, &format!("{prefix}.ffn_norm1.weight"))?,
        RMS_EPS,
    )?;
    let normed = if modulated {
        modulate(b, normed, scale_mlp)?
    } else {
        normed
    };
    let ffn = feed_forward(b, weights, prefix, normed)?;
    let ffn = rms(
        b,
        ffn,
        wt(weights, &format!("{prefix}.ffn_norm2.weight"))?,
        RMS_EPS,
    )?;
    if modulated {
        gate_add(b, x, gate_mlp, ffn)
    } else {
        nn(b.add(x, ffn))
    }
}

fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
    rope: &[Tensor; 3],
    config: ZImageConfig,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let dim = config.dim;
    let head = config.head_dim;
    let heads = config.heads;
    let qkv = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.attention.qkv.weight"))?,
    ))?;
    let qkv = nn(b.reshape_3d(qkv, head, heads * 3, seq))?;
    let bytes = std::mem::size_of::<f32>();
    let nb1 = (head as usize) * bytes;
    let nb2 = nb1 * (heads as usize) * 3;
    let q = nn(b.cont(b.view_3d(qkv, head, heads, seq, nb1, nb2, 0)))?;
    let k = nn(b.cont(b.view_3d(qkv, head, heads, seq, nb1, nb2, (heads as usize) * nb1)))?;
    let v = nn(b.cont(b.view_3d(qkv, head, heads, seq, nb1, nb2, (heads as usize) * 2 * nb1)))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, RMS_EPS))?,
        wt(weights, &format!("{prefix}.attention.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, RMS_EPS))?,
        wt(weights, &format!("{prefix}.attention.k_norm.weight"))?,
    ))?;
    let q = rope_nd(b, q, rope, &config.axes, config.rope_theta)?;
    let k = rope_nd(b, k, rope, &config.axes, config.rope_theta)?;
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(out, dim, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.attention.out.weight"))?,
    ))
}

fn feed_forward(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let gate = nn(b.silu(nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.feed_forward.w1.weight"))?,
    ))?))?;
    let up = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.feed_forward.w3.weight"))?,
    ))?;
    let hidden = nn(b.mul(gate, up))?;
    nn(linear(
        b,
        hidden,
        wt(weights, &format!("{prefix}.feed_forward.w2.weight"))?,
    ))
}

fn final_layer(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    t_emb: Tensor,
) -> Result<Tensor, RuntimeError> {
    let scale = add_bias(
        b,
        nn(linear(
            b,
            nn(b.silu(t_emb))?,
            wt(weights, "final_layer.adaLN_modulation.1.weight")?,
        ))?,
        wt(weights, "final_layer.adaLN_modulation.1.bias")?,
    )?;
    let normed = nn(b.norm(x, FINAL_EPS))?;
    let shifted = modulate(b, normed, scale)?;
    add_bias(
        b,
        nn(linear(
            b,
            shifted,
            wt(weights, "final_layer.linear.weight")?,
        ))?,
        wt(weights, "final_layer.linear.bias")?,
    )
}

/// `x * (1 + scale)` with `scale` shaped `[dim, 1]`.
fn modulate(b: &GraphBuilder, x: Tensor, scale: Tensor) -> Result<Tensor, RuntimeError> {
    let scale = nn(b.repeat(scale, x))?;
    let scaled = nn(b.mul(x, scale))?;
    nn(b.add(x, scaled))
}

fn gate_add(
    b: &GraphBuilder,
    residual: Tensor,
    gate: Tensor,
    delta: Tensor,
) -> Result<Tensor, RuntimeError> {
    let gate = nn(b.repeat(gate, delta))?;
    nn(b.add(residual, nn(b.mul(delta, gate))?))
}

fn chunk(b: &GraphBuilder, packed: Tensor, row0: i64, rows: i64) -> Result<Tensor, RuntimeError> {
    let total = rows * 4;
    let view = b.view_2d(
        packed,
        rows,
        1,
        (total as usize) * std::mem::size_of::<f32>(),
        (row0 as usize) * std::mem::size_of::<f32>(),
    );
    nn(b.cont(view))
}

fn rows(
    b: &GraphBuilder,
    tensor: Tensor,
    token0: i64,
    count: i64,
    dim: i64,
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

fn span(
    b: &GraphBuilder,
    rope: &[Tensor; 3],
    start: i64,
    len: i64,
) -> Result<[Tensor; 3], RuntimeError> {
    let mut out = [std::ptr::null_mut(); 3];
    let offset = (start as usize) * std::mem::size_of::<i32>();
    for (src, dst) in rope.iter().zip(out.iter_mut()) {
        *dst = nn(b.cont(b.view_1d(*src, len, offset)))?;
    }
    Ok(out)
}

fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor, eps: f32) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, eps))?, weight))
}

fn add_bias(b: &GraphBuilder, x: Tensor, bias: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, nn(b.repeat(bias, x))?))
}

fn linear(b: &GraphBuilder, x: Tensor, w: Tensor) -> Tensor {
    b.project(w, x)
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
