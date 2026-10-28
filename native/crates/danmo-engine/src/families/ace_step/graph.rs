//! ACE-Step DiT topology from acestep.cpp `dit-graph.h`.
//!
//! Timestep sinusoids are host inputs (`temb_freq_t` / `temb_freq_r`) so the
//! graph stays free of `ggml_timestep_embedding`. `proj_in` / `proj_out` are
//! the load-time 2D ranks from `dit_load_proj_*_w`. SwiGLU is `silu(gate)*up`.

use super::weights::{AceStepConfig, TIME_FREQ_DIM};
use crate::runtime::{flash_attend, shape, GraphBuilder, RuntimeError, Tensor, TYPE_F32};
use std::collections::HashMap;

pub struct BuiltDit {
    pub output: Tensor,
    pub seq: i64,
}

/// 2D ranks the compute graph binds. File Conv1d rows are remapped at load.
pub fn graph_weight_layout(config: AceStepConfig) -> Vec<(String, i64, i64)> {
    let h = config.hidden;
    let ff = config.intermediate;
    let q = config.heads * config.head_dim;
    let kv = config.kv_heads * config.head_dim;
    let mut out = Vec::new();
    let push = |out: &mut Vec<(String, i64, i64)>, name: String, ne0: i64, ne1: i64| {
        out.push((name, ne0, ne1));
    };
    for stem in ["decoder.time_embed", "decoder.time_embed_r"] {
        push(&mut out, format!("{stem}.linear_1.weight"), TIME_FREQ_DIM, h);
        push(&mut out, format!("{stem}.linear_1.bias"), h, 1);
        push(&mut out, format!("{stem}.linear_2.weight"), h, h);
        push(&mut out, format!("{stem}.linear_2.bias"), h, 1);
        push(&mut out, format!("{stem}.time_proj.weight"), h, h * 6);
        push(&mut out, format!("{stem}.time_proj.bias"), h * 6, 1);
    }
    push(
        &mut out,
        "decoder.proj_in.1.weight".into(),
        config.in_channels * config.patch,
        h,
    );
    push(&mut out, "decoder.proj_in.1.bias".into(), h, 1);
    push(
        &mut out,
        "decoder.condition_embedder.weight".into(),
        config.condition_dim,
        h,
    );
    push(&mut out, "decoder.condition_embedder.bias".into(), h, 1);
    for layer in 0..config.layers {
        let p = format!("decoder.layers.{layer}");
        push(&mut out, format!("{p}.self_attn_norm.weight"), h, 1);
        push(&mut out, format!("{p}.self_attn.q_proj.weight"), h, q);
        push(&mut out, format!("{p}.self_attn.k_proj.weight"), h, kv);
        push(&mut out, format!("{p}.self_attn.v_proj.weight"), h, kv);
        push(
            &mut out,
            format!("{p}.self_attn.q_norm.weight"),
            config.head_dim,
            1,
        );
        push(
            &mut out,
            format!("{p}.self_attn.k_norm.weight"),
            config.head_dim,
            1,
        );
        push(&mut out, format!("{p}.self_attn.o_proj.weight"), q, h);
        push(&mut out, format!("{p}.cross_attn_norm.weight"), h, 1);
        push(&mut out, format!("{p}.cross_attn.q_proj.weight"), h, q);
        push(&mut out, format!("{p}.cross_attn.k_proj.weight"), h, kv);
        push(&mut out, format!("{p}.cross_attn.v_proj.weight"), h, kv);
        push(
            &mut out,
            format!("{p}.cross_attn.q_norm.weight"),
            config.head_dim,
            1,
        );
        push(
            &mut out,
            format!("{p}.cross_attn.k_norm.weight"),
            config.head_dim,
            1,
        );
        push(&mut out, format!("{p}.cross_attn.o_proj.weight"), q, h);
        push(&mut out, format!("{p}.mlp_norm.weight"), h, 1);
        push(&mut out, format!("{p}.mlp.gate_proj.weight"), h, ff);
        push(&mut out, format!("{p}.mlp.up_proj.weight"), h, ff);
        push(&mut out, format!("{p}.mlp.down_proj.weight"), ff, h);
        push(&mut out, format!("{p}.scale_shift_table"), h, 6);
    }
    push(&mut out, "decoder.norm_out.weight".into(), h, 1);
    push(&mut out, "decoder.scale_shift_table".into(), h, 2);
    push(
        &mut out,
        "decoder.proj_out.1.weight".into(),
        h,
        config.out_channels * config.patch,
    );
    push(
        &mut out,
        "decoder.proj_out.1.bias".into(),
        config.out_channels,
        1,
    );
    out
}

pub fn build_dit(
    b: &mut GraphBuilder,
    config: AceStepConfig,
    frames: i64,
    enc_seq: i64,
    types: &HashMap<String, u32>,
) -> Result<BuiltDit, RuntimeError> {
    if frames < 1 || frames % config.patch != 0 {
        return Err(RuntimeError::Message(format!(
            "ACE-Step frames {frames} must be a positive multiple of patch {}",
            config.patch
        )));
    }
    if enc_seq < 1 {
        return Err(RuntimeError::Message(
            "ACE-Step encoder sequence is empty".into(),
        ));
    }
    if config.heads < 1
        || config.kv_heads < 1
        || config.head_dim < 1
        || config.heads % config.kv_heads != 0
    {
        return Err(RuntimeError::Message(
            "ACE-Step attention head layout is empty".into(),
        ));
    }
    let seq = frames / config.patch;
    let mut weights = HashMap::new();
    for (name, ne0, ne1) in graph_weight_layout(config) {
        let ggml_type = types.get(&name).copied().unwrap_or(TYPE_F32);
        let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name, tensor);
    }
    let input = nn(b.f32_2d(
        "input_latents",
        config.in_channels,
        frames,
        true,
    ))?;
    let enc_hidden = nn(b.f32_2d(
        "enc_hidden",
        config.condition_dim,
        enc_seq,
        true,
    ))?;
    let temb_freq_t = nn(b.f32_2d("temb_freq_t", TIME_FREQ_DIM, 1, true))?;
    let temb_freq_r = nn(b.f32_2d("temb_freq_r", TIME_FREQ_DIM, 1, true))?;
    let positions = nn(b.i32_1d("positions", seq, true))?;
    let output = forward(
        b,
        &weights,
        config,
        input,
        enc_hidden,
        temb_freq_t,
        temb_freq_r,
        positions,
        seq,
        frames,
        enc_seq,
    )?;
    Ok(BuiltDit { output, seq })
}

fn forward(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    input: Tensor,
    enc_hidden: Tensor,
    temb_freq_t: Tensor,
    temb_freq_r: Tensor,
    positions: Tensor,
    seq: i64,
    frames: i64,
    enc_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let h = config.hidden;
    let (temb_t, tproj_t) = temb(b, weights, "decoder.time_embed", temb_freq_t)?;
    let (temb_r, tproj_r) = temb(b, weights, "decoder.time_embed_r", temb_freq_r)?;
    let temb = nn(b.add(temb_t, temb_r))?;
    let tproj = nn(b.add(tproj_t, tproj_r))?;
    let patched = nn(b.reshape_2d(
        input,
        config.in_channels * config.patch,
        seq,
    ))?;
    let mut hidden = add_bias(
        b,
        nn(linear(b, patched, wt(weights, "decoder.proj_in.1.weight")?))?,
        wt(weights, "decoder.proj_in.1.bias")?,
    )?;
    let enc = add_bias(
        b,
        nn(linear(
            b,
            enc_hidden,
            wt(weights, "decoder.condition_embedder.weight")?,
        ))?,
        wt(weights, "decoder.condition_embedder.bias")?,
    )?;
    for layer in 0..config.layers {
        hidden = layer_block(
            b,
            weights,
            config,
            layer,
            hidden,
            tproj,
            enc,
            positions,
            seq,
            enc_seq,
        )?;
    }
    let oss = nn(b.reshape_2d(wt(weights, "decoder.scale_shift_table")?, h * 2, 1))?;
    let out_shift = nn(b.add(chunk(b, oss, 0, h)?, temb))?;
    let out_scale = nn(b.add(chunk(b, oss, h, h)?, temb))?;
    let norm_out = rms(
        b,
        hidden,
        wt(weights, "decoder.norm_out.weight")?,
        config.rms_norm_eps,
    )?;
    let norm_out = adaln(b, norm_out, out_scale, out_shift)?;
    let projected = add_bias(
        b,
        nn(linear(
            b,
            norm_out,
            wt(weights, "decoder.proj_out.1.weight")?,
        ))?,
        wt(weights, "decoder.proj_out.1.bias")?,
    )?;
    nn(b.reshape_2d(projected, config.out_channels, frames))
}

fn temb(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    stem: &str,
    freq: Tensor,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let h = add_bias(
        b,
        nn(linear(
            b,
            freq,
            wt(weights, &format!("{stem}.linear_1.weight"))?,
        ))?,
        wt(weights, &format!("{stem}.linear_1.bias"))?,
    )?;
    let temb = add_bias(
        b,
        nn(linear(
            b,
            nn(b.silu(h))?,
            wt(weights, &format!("{stem}.linear_2.weight"))?,
        ))?,
        wt(weights, &format!("{stem}.linear_2.bias"))?,
    )?;
    let tproj = add_bias(
        b,
        nn(linear(
            b,
            nn(b.silu(temb))?,
            wt(weights, &format!("{stem}.time_proj.weight"))?,
        ))?,
        wt(weights, &format!("{stem}.time_proj.bias"))?,
    )?;
    Ok((temb, tproj))
}

fn layer_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    layer: i64,
    hidden: Tensor,
    tproj: Tensor,
    enc: Tensor,
    positions: Tensor,
    seq: i64,
    enc_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let h = config.hidden;
    let p = format!("decoder.layers.{layer}");
    let ss = nn(b.reshape_2d(wt(weights, &format!("{p}.scale_shift_table"))?, h * 6, 1))?;
    let adaln_pack = nn(b.add(ss, tproj))?;
    let shift_sa = chunk(b, adaln_pack, 0, h)?;
    let scale_sa = chunk(b, adaln_pack, h, h)?;
    let gate_sa = chunk(b, adaln_pack, h * 2, h)?;
    let shift_ffn = chunk(b, adaln_pack, h * 3, h)?;
    let scale_ffn = chunk(b, adaln_pack, h * 4, h)?;
    let gate_ffn = chunk(b, adaln_pack, h * 5, h)?;

    let norm_sa = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.self_attn_norm.weight"))?,
        config.rms_norm_eps,
    )?;
    let norm_sa = adaln(b, norm_sa, scale_sa, shift_sa)?;
    let sa = self_attn(b, weights, config, &p, norm_sa, positions, seq)?;
    let hidden = gate_add(b, hidden, sa, gate_sa)?;

    let norm_ca = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.cross_attn_norm.weight"))?,
        config.rms_norm_eps,
    )?;
    let ca = cross_attn(b, weights, config, &p, norm_ca, enc, seq, enc_seq)?;
    let hidden = nn(b.add(hidden, ca))?;

    let norm_ffn = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.mlp_norm.weight"))?,
        config.rms_norm_eps,
    )?;
    let norm_ffn = adaln(b, norm_ffn, scale_ffn, shift_ffn)?;
    let ffn = mlp(b, weights, &p, norm_ffn)?;
    gate_add(b, hidden, ffn, gate_ffn)
}

fn self_attn(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    prefix: &str,
    x: Tensor,
    positions: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let d = config.head_dim;
    let nh = config.heads;
    let nkv = config.kv_heads;
    let q = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.self_attn.q_proj.weight"))?,
    ))?;
    let k = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.self_attn.k_proj.weight"))?,
    ))?;
    let v = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.self_attn.v_proj.weight"))?,
    ))?;
    let q = nn(b.reshape_3d(q, d, nh, seq))?;
    let k = nn(b.reshape_3d(k, d, nkv, seq))?;
    let v = nn(b.reshape_3d(v, d, nkv, seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
    ))?;
    let q = nn(b.cont(b.rope_neox(q, positions, d as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, positions, d as i32, config.rope_theta)))?;
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(out, nh * d, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.self_attn.o_proj.weight"))?,
    ))
}

fn cross_attn(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: AceStepConfig,
    prefix: &str,
    x: Tensor,
    enc: Tensor,
    seq: i64,
    enc_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let d = config.head_dim;
    let nh = config.heads;
    let nkv = config.kv_heads;
    let q = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.cross_attn.q_proj.weight"))?,
    ))?;
    let k = nn(linear(
        b,
        enc,
        wt(weights, &format!("{prefix}.cross_attn.k_proj.weight"))?,
    ))?;
    let v = nn(linear(
        b,
        enc,
        wt(weights, &format!("{prefix}.cross_attn.v_proj.weight"))?,
    ))?;
    let q = nn(b.reshape_3d(q, d, nh, seq))?;
    let k = nn(b.reshape_3d(k, d, nkv, enc_seq))?;
    let v = nn(b.reshape_3d(v, d, nkv, enc_seq))?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.cross_attn.q_norm.weight"))?,
    ))?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, config.rms_norm_eps))?,
        wt(weights, &format!("{prefix}.cross_attn.k_norm.weight"))?,
    ))?;
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(out, nh * d, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.cross_attn.o_proj.weight"))?,
    ))
}

fn mlp(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let gate = nn(b.silu(nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.mlp.gate_proj.weight"))?,
    ))?))?;
    let up = nn(linear(
        b,
        x,
        wt(weights, &format!("{prefix}.mlp.up_proj.weight"))?,
    ))?;
    let hidden = nn(b.mul(gate, up))?;
    nn(linear(
        b,
        hidden,
        wt(weights, &format!("{prefix}.mlp.down_proj.weight"))?,
    ))
}

fn adaln(b: &GraphBuilder, norm: Tensor, scale: Tensor, shift: Tensor) -> Result<Tensor, RuntimeError> {
    let scale = nn(b.repeat(scale, norm))?;
    let shift = nn(b.repeat(shift, norm))?;
    let scaled = nn(b.add(norm, nn(b.mul(norm, scale))?))?;
    nn(b.add(scaled, shift))
}

fn gate_add(
    b: &GraphBuilder,
    residual: Tensor,
    delta: Tensor,
    gate: Tensor,
) -> Result<Tensor, RuntimeError> {
    let gate = nn(b.repeat(gate, delta))?;
    nn(b.add(residual, nn(b.mul(delta, gate))?))
}

fn chunk(b: &GraphBuilder, packed: Tensor, offset: i64, rows: i64) -> Result<Tensor, RuntimeError> {
    let dims = shape(packed);
    let total = dims[0];
    let view = b.view_2d(
        packed,
        rows,
        1,
        (total as usize) * std::mem::size_of::<f32>(),
        (offset as usize) * std::mem::size_of::<f32>(),
    );
    nn(b.cont(view))
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
        .ok_or_else(|| RuntimeError::Message(format!("ACE-Step graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}
