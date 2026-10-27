//! Flux2 MM-DiT topology. This file only appends ggml nodes.
//!
//! The block order matches `Flux2DiTMLX`: timestep embed, x/context embed,
//! double-stream joint blocks, single-stream blocks, AdaLN, proj_out.
//! Attention is per-head. Head dims divisible by 8 use four-axis RoPE
//! (`[head/4; 4]`, theta 2000) through `ggml_rope_ext`. Tensor names are the GGUF names.

use super::weights::{weight_layout, Flux2Config};
use crate::runtime::{
    flash_attend, rope_pair, GraphBuilder, RopeIn, RuntimeError, Tensor, TYPE_F32,
};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

pub struct BuiltDit {
    pub output: Tensor,
    pub rope: RopeIn,
}

/// Append the DiT. The caller owns the builder, reserves the graph, and runs it.
pub fn build_dit(
    b: &mut GraphBuilder,
    config: Flux2Config,
    grid_w: i64,
    grid_h: i64,
    seq_txt: i64,
    types: &HashMap<String, u32>,
) -> Result<BuiltDit, RuntimeError> {
    if config.num_heads < 1 || config.attn_head_dim < 1 {
        return Err(RuntimeError::Message(
            "flux2 attention heads are empty".into(),
        ));
    }
    if config.inner_dim != config.num_heads * config.attn_head_dim {
        return Err(RuntimeError::Message(
            "flux2 inner_dim must equal heads * head_dim".into(),
        ));
    }
    if grid_w < 1 || grid_h < 1 {
        return Err(RuntimeError::Message("flux2 image grid is empty".into()));
    }
    if seq_txt < 1 {
        return Err(RuntimeError::Message("flux2 text sequence is empty".into()));
    }
    let seq_img = grid_w * grid_h;
    let seq = seq_txt + seq_img;
    let axis_rope = config.attn_head_dim % 8 == 0;
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
    let rope = if axis_rope {
        let mut axes = [std::ptr::null_mut(); 4];
        for (i, slot) in axes.iter_mut().enumerate() {
            *slot = nn(b.i32_1d(&format!("rope_axis_{i}"), seq, true))?;
        }
        RopeIn {
            axis: true,
            pos: std::ptr::null_mut(),
            axes,
            theta: config.rope_theta,
        }
    } else {
        RopeIn {
            axis: false,
            pos: nn(b.i32_1d("token_index", seq, true))?,
            axes: [std::ptr::null_mut(); 4],
            theta: config.rope_theta,
        }
    };
    let output = forward(
        b, &weights, config, latent, txt, temb_freq, rope, seq_img, seq_txt,
    )?;
    Ok(BuiltDit { output, rope })
}

fn forward(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: Flux2Config,
    latent: Tensor,
    txt_in: Tensor,
    temb_freq: Tensor,
    rope: RopeIn,
    seq_img: i64,
    seq_txt: i64,
) -> Result<Tensor, RuntimeError> {
    let dim = config.inner_dim;
    let head = config.attn_head_dim;
    let heads = config.num_heads;
    let hidden = config.hidden();
    let seq = seq_txt + seq_img;

    let h = nn(b.silu(linear(
        b,
        temb_freq,
        wt(weights, "time_guidance_embed.linear_1.weight")?,
    )))?;
    let temb = nn(linear(
        b,
        h,
        wt(weights, "time_guidance_embed.linear_2.weight")?,
    ))?;
    let mod_img = nn(linear(
        b,
        nn(b.silu(temb))?,
        wt(weights, "double_stream_modulation_img.linear.weight")?,
    ))?;
    let mod_txt = nn(linear(
        b,
        nn(b.silu(temb))?,
        wt(weights, "double_stream_modulation_txt.linear.weight")?,
    ))?;
    let mut img = nn(linear(b, latent, wt(weights, "x_embedder.weight")?))?;
    let mut txt = nn(linear(b, txt_in, wt(weights, "context_embedder.weight")?))?;

    for i in 0..config.num_layers {
        let p = format!("transformer_blocks.{i}");
        (txt, img) = joint_block(
            b, weights, &p, txt, img, mod_txt, mod_img, rope, dim, head, heads, hidden, seq_txt,
            seq_img,
        )?;
    }

    let mut seq_h = nn(b.concat(txt, img, 1))?;
    let mod_single = nn(linear(
        b,
        nn(b.silu(temb))?,
        wt(weights, "single_stream_modulation.linear.weight")?,
    ))?;
    for i in 0..config.num_single_layers {
        let p = format!("single_transformer_blocks.{i}.attn");
        seq_h = single_block(
            b, weights, &p, seq_h, mod_single, rope, dim, head, heads, hidden, seq,
        )?;
    }
    let img = tokens(b, seq_h, dim, seq_txt, seq_img)?;
    let normed = nn(b.norm(img, LN_EPS))?;
    let out_mod = nn(linear(
        b,
        nn(b.silu(temb))?,
        wt(weights, "norm_out.linear.weight")?,
    ))?;
    let scale = features(b, out_mod, 0, dim, dim * 2, 1)?;
    let shift = features(b, out_mod, dim, dim, dim * 2, 1)?;
    let normed = scale_shift(b, normed, scale, shift)?;
    nn(linear(b, normed, wt(weights, "proj_out.weight")?))
}

fn joint_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    txt: Tensor,
    img: Tensor,
    mod_txt: Tensor,
    mod_img: Tensor,
    rope: RopeIn,
    dim: i64,
    head: i64,
    heads: i64,
    hidden: i64,
    seq_txt: i64,
    seq_img: i64,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let seq = seq_txt + seq_img;
    let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = mods(b, mod_img, dim)?;
    let (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
        mods(b, mod_txt, dim)?;
    let n_img = scale_shift(b, nn(b.norm(img, LN_EPS))?, scale_msa, shift_msa)?;
    let n_txt = scale_shift(b, nn(b.norm(txt, LN_EPS))?, c_scale_msa, c_shift_msa)?;
    let q_img = rms_heads(
        b,
        nn(linear(
            b,
            n_img,
            wt(weights, &format!("{prefix}.attn.to_q.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.attn.norm_q.weight"))?,
        head,
        heads,
        seq_img,
    )?;
    let k_img = rms_heads(
        b,
        nn(linear(
            b,
            n_img,
            wt(weights, &format!("{prefix}.attn.to_k.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.attn.norm_k.weight"))?,
        head,
        heads,
        seq_img,
    )?;
    let v_img = split_heads(
        b,
        nn(linear(
            b,
            n_img,
            wt(weights, &format!("{prefix}.attn.to_v.weight"))?,
        ))?,
        head,
        heads,
        seq_img,
    )?;
    let q_txt = rms_heads(
        b,
        nn(linear(
            b,
            n_txt,
            wt(weights, &format!("{prefix}.attn.add_q_proj.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.attn.norm_added_q.weight"))?,
        head,
        heads,
        seq_txt,
    )?;
    let k_txt = rms_heads(
        b,
        nn(linear(
            b,
            n_txt,
            wt(weights, &format!("{prefix}.attn.add_k_proj.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.attn.norm_added_k.weight"))?,
        head,
        heads,
        seq_txt,
    )?;
    let v_txt = split_heads(
        b,
        nn(linear(
            b,
            n_txt,
            wt(weights, &format!("{prefix}.attn.add_v_proj.weight"))?,
        ))?,
        head,
        heads,
        seq_txt,
    )?;
    let q = nn(b.cont(b.concat(q_txt, q_img, 2)))?;
    let k = nn(b.cont(b.concat(k_txt, k_img, 2)))?;
    let v = nn(b.cont(b.concat(v_txt, v_img, 2)))?;
    let mixed = attend(b, q, k, v, rope, dim, head, heads, seq)?;
    let txt_tok = tokens(b, mixed, dim, 0, seq_txt)?;
    let img_tok = tokens(b, mixed, dim, seq_txt, seq_img)?;
    let txt = gate_add(
        b,
        txt,
        c_gate_msa,
        nn(linear(
            b,
            txt_tok,
            wt(weights, &format!("{prefix}.attn.to_add_out.weight"))?,
        ))?,
    )?;
    let img = gate_add(
        b,
        img,
        gate_msa,
        nn(linear(
            b,
            img_tok,
            wt(weights, &format!("{prefix}.attn.to_out.weight"))?,
        ))?,
    )?;
    let n_img = scale_shift(b, nn(b.norm(img, LN_EPS))?, scale_mlp, shift_mlp)?;
    let n_txt = scale_shift(b, nn(b.norm(txt, LN_EPS))?, c_scale_mlp, c_shift_mlp)?;
    let img = gate_add(
        b,
        img,
        gate_mlp,
        ffn(
            b,
            n_img,
            wt(weights, &format!("{prefix}.ff.linear_in.weight"))?,
            wt(weights, &format!("{prefix}.ff.linear_out.weight"))?,
            hidden,
        )?,
    )?;
    let txt = gate_add(
        b,
        txt,
        c_gate_mlp,
        ffn(
            b,
            n_txt,
            wt(weights, &format!("{prefix}.ff_context.linear_in.weight"))?,
            wt(weights, &format!("{prefix}.ff_context.linear_out.weight"))?,
            hidden,
        )?,
    )?;
    Ok((txt, img))
}

fn single_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    seq_h: Tensor,
    mod_single: Tensor,
    rope: RopeIn,
    dim: i64,
    head: i64,
    heads: i64,
    hidden: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let shift = features(b, mod_single, 0, dim, dim * 3, 1)?;
    let scale = features(b, mod_single, dim, dim, dim * 3, 1)?;
    let gate = features(b, mod_single, dim * 2, dim, dim * 3, 1)?;
    let n = scale_shift(b, nn(b.norm(seq_h, LN_EPS))?, scale, shift)?;
    let total = dim * 3 + hidden * 2;
    let proj = nn(linear(
        b,
        n,
        wt(weights, &format!("{prefix}.to_qkv_mlp_proj.weight"))?,
    ))?;
    let qkv = features(b, proj, 0, dim * 3, total, seq)?;
    let mlp = features(b, proj, dim * 3, hidden * 2, total, seq)?;
    let q = rms_heads(
        b,
        features(b, qkv, 0, dim, dim * 3, seq)?,
        wt(weights, &format!("{prefix}.norm_q.weight"))?,
        head,
        heads,
        seq,
    )?;
    let k = rms_heads(
        b,
        features(b, qkv, dim, dim, dim * 3, seq)?,
        wt(weights, &format!("{prefix}.norm_k.weight"))?,
        head,
        heads,
        seq,
    )?;
    let v = split_heads(
        b,
        features(b, qkv, dim * 2, dim, dim * 3, seq)?,
        head,
        heads,
        seq,
    )?;
    let attn = attend(b, q, k, v, rope, dim, head, heads, seq)?;
    let mlp_out = nn(b.swiglu(mlp))?;
    let fused = nn(b.concat(attn, mlp_out, 0))?;
    let delta = nn(linear(
        b,
        fused,
        wt(weights, &format!("{prefix}.to_out.weight"))?,
    ))?;
    gate_add(b, seq_h, gate, delta)
}

fn attend(
    b: &GraphBuilder,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    rope: RopeIn,
    dim: i64,
    head: i64,
    _heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let (q, k) = rope_pair(b, q, k, head, &rope)?;
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    nn(b.reshape_2d(out, dim, seq))
}

#[cfg(test)]
fn flux2_cos_sin(
    head: i64,
    seq_txt: i64,
    grid_w: i64,
    grid_h: i64,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let axis = head / 4;
    let half = (head / 2) as usize;
    let seq_img = grid_w * grid_h;
    let seq = (seq_txt + seq_img) as usize;
    let mut cos = vec![0f32; half * seq];
    let mut sin = vec![0f32; half * seq];
    for s in 0..seq {
        let ids = if (s as i64) < seq_txt {
            [0f32, 0.0, 0.0, s as f32]
        } else {
            let i = s as i64 - seq_txt;
            let h = (i / grid_w) as f32;
            let w = (i % grid_w) as f32;
            [0.0, h, w, 0.0]
        };
        let mut pair = 0usize;
        for axis_i in 0..4 {
            let dim = axis;
            let n = (dim / 2) as usize;
            for i in 0..n {
                let scale = (2 * i) as f32 / dim as f32;
                let ang = ids[axis_i] * theta.powf(-scale);
                cos[pair + half * s] = ang.cos();
                sin[pair + half * s] = ang.sin();
                pair += 1;
            }
        }
    }
    (cos, sin)
}

fn ffn(
    b: &GraphBuilder,
    x: Tensor,
    w_in: Tensor,
    w_out: Tensor,
    _hidden: i64,
) -> Result<Tensor, RuntimeError> {
    let mid = nn(b.swiglu(nn(linear(b, x, w_in))?))?;
    nn(linear(b, mid, w_out))
}

fn mods(
    b: &GraphBuilder,
    packed: Tensor,
    dim: i64,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor), RuntimeError> {
    let total = dim * 6;
    Ok((
        features(b, packed, 0, dim, total, 1)?,
        features(b, packed, dim, dim, total, 1)?,
        features(b, packed, dim * 2, dim, total, 1)?,
        features(b, packed, dim * 3, dim, total, 1)?,
        features(b, packed, dim * 4, dim, total, 1)?,
        features(b, packed, dim * 5, dim, total, 1)?,
    ))
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

fn rms_heads(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    head: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let x = split_heads(b, x, head, heads, seq)?;
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, weight))
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

fn features(
    b: &GraphBuilder,
    tensor: Tensor,
    row0: i64,
    rows: i64,
    total_rows: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let view = b.view_2d(
        tensor,
        rows,
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

fn linear(b: &GraphBuilder, x: Tensor, w: Tensor) -> Tensor {
    b.mul_mat(w, x)
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

#[cfg(test)]
mod rope_tests {
    use super::flux2_cos_sin;
    use crate::runtime::{rope_pair, BackendKind, GgmlContext, RopeIn};

    #[test]
    fn height_axis_follows_the_image_row() {
        let (cos, sin) = flux2_cos_sin(8, 1, 1, 2, 2000.0);
        let half = 4usize;
        let s = 2usize;
        assert!((cos[half * s] - 1.0).abs() < 1e-5);
        assert!((cos[1 + half * s] - 1.0f32.cos()).abs() < 1e-5);
        assert!((sin[1 + half * s] - 1.0f32.sin()).abs() < 1e-5);
        assert!((cos[2 + half * s] - 1.0).abs() < 1e-5);
        assert!((sin[3 + half * s]).abs() < 1e-5);
    }

    #[test]
    fn rope_ext_matches_the_four_axis_table() {
        let head = 8i64;
        let heads = 1i64;
        let seq_txt = 1i64;
        let grid_w = 1i64;
        let grid_h = 2i64;
        let seq = seq_txt + grid_w * grid_h;
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(4 * 1024 * 1024).unwrap();
        let x = b.f32_3d("x", head, heads, seq, true);
        let mut axes = [std::ptr::null_mut(); 4];
        for (i, slot) in axes.iter_mut().enumerate() {
            *slot = b.i32_1d(&format!("rope_axis_{i}"), seq, true);
        }
        let rope = RopeIn {
            axis: true,
            pos: std::ptr::null_mut(),
            axes,
            theta: 2000.0,
        };
        let (y, _) = rope_pair(&b, x, x, head, &rope).unwrap();
        let mut graph = b.finish(y).unwrap();
        let values: Vec<f32> = (0..(head * heads * seq) as usize)
            .map(|i| (i as f32) * 0.1 - 0.3)
            .collect();
        graph.set_f32("x", &values).unwrap();
        let mut columns = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for s in 0..seq {
            let ids = if s < seq_txt {
                [0, 0, 0, s as i32]
            } else {
                let i = s - seq_txt;
                [0, (i / grid_w) as i32, (i % grid_w) as i32, 0]
            };
            for (column, id) in columns.iter_mut().zip(ids) {
                column.push(id);
            }
        }
        for (i, column) in columns.iter().enumerate() {
            graph.set_i32(&format!("rope_axis_{i}"), column).unwrap();
        }
        let got = graph.compute().unwrap();
        let (cos, sin) = flux2_cos_sin(head, seq_txt, grid_w, grid_h, 2000.0);
        let half = (head / 2) as usize;
        let mut expect = values.clone();
        for s in 0..seq as usize {
            for pair in 0..half {
                let i0 = pair * 2 + (head as usize) * s;
                let i1 = i0 + 1;
                let c = cos[pair + half * s];
                let sn = sin[pair + half * s];
                let even = values[i0];
                let odd = values[i1];
                expect[i0] = even * c - odd * sn;
                expect[i1] = odd * c + even * sn;
            }
        }
        for (i, (a, b)) in got.iter().zip(expect.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "index {i} got {a} expect {b}");
        }
    }
}
