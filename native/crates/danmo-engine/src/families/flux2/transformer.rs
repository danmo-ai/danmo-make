//! Flux2 MM-DiT. One cgraph is reserved and replayed for every Euler step.
//!
//! The block order matches `Flux2DiTMLX`: timestep embed, x/context embed,
//! double-stream joint blocks, single-stream blocks, AdaLN, proj_out.
//! Heads other than 1 fail loud. Tensor names are the GGUF names.

use super::weights::{weight_layout, Flux2Config};
use crate::common::ops::{self, euler_step, linear_sigmas};
use crate::runtime::{GgmlContext, GraphBuilder, ReusedGraph, RuntimeError};
use danmo_ggml_sys::ggml_tensor;
use std::collections::HashMap;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

pub struct Flux2Denoise {
    graph: ReusedGraph,
    pub steps_run: u32,
    seq_img: i64,
    seq_txt: i64,
    in_channels: i64,
    joint: i64,
    freq: i64,
}

impl Flux2Denoise {
    pub fn build(
        ctx: &GgmlContext,
        config: Flux2Config,
        seq_img: i64,
    ) -> Result<Self, RuntimeError> {
        if config.num_heads != 1 {
            return Err(RuntimeError::Message(format!(
                "flux2 attention for {} heads is not built",
                config.num_heads
            )));
        }
        if config.inner_dim != config.num_heads * config.attn_head_dim {
            return Err(RuntimeError::Message(
                "flux2 inner_dim must equal heads * head_dim".into(),
            ));
        }
        if seq_img < 1 {
            return Err(RuntimeError::Message(
                "flux2 image sequence is empty".into(),
            ));
        }
        let seq_txt = 1i64;
        let seq = seq_txt + seq_img;
        let mut b = ctx.builder(64 * 1024 * 1024)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in weight_layout(config) {
            let tensor = b.f32_2d(&name, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        let latent = nn(b.f32_2d("latent", config.in_channels, seq_img, true))?;
        let txt = nn(b.f32_2d("txt", config.joint_attention_dim, seq_txt, true))?;
        let temb_freq = nn(b.f32_2d("temb_freq", config.freq_dim, 1, true))?;
        let pos = nn(b.i32_1d("token_index", seq, true))?;
        let y = forward(
            &b, &weights, config, latent, txt, temb_freq, pos, seq_img, seq_txt,
        )?;
        let graph = b.finish(y)?;
        Ok(Self {
            graph,
            steps_run: 0,
            seq_img,
            seq_txt,
            in_channels: config.in_channels,
            joint: config.joint_attention_dim,
            freq: config.freq_dim,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    pub fn latent_len(&self) -> usize {
        (self.in_channels * self.seq_img) as usize
    }

    pub fn txt_len(&self) -> usize {
        (self.joint * self.seq_txt) as usize
    }

    pub fn denoise(
        &mut self,
        mut latent: Vec<f32>,
        txt: &[f32],
        steps: u32,
        width: u32,
        height: u32,
        shift: bool,
        guidance: f32,
        mut on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
        if latent.len() != self.latent_len() {
            return Err(RuntimeError::Message(format!(
                "latent has {} values, graph expects {}",
                latent.len(),
                self.latent_len()
            )));
        }
        if txt.len() != self.txt_len() {
            return Err(RuntimeError::Message(format!(
                "text embedding has {} values, graph expects {}",
                txt.len(),
                self.txt_len()
            )));
        }
        let positions: Vec<i32> = (0..self.seq_txt + self.seq_img).map(|i| i as i32).collect();
        self.graph.set_i32("token_index", &positions)?;
        let sigmas = linear_sigmas(steps, width, height, shift);
        let zeros = vec![0f32; txt.len()];
        for step in 0..steps {
            let t = sigmas[step as usize];
            let freq = ops::sinusoidal_timestep(t, self.freq as usize);
            self.graph.set_f32("temb_freq", &freq)?;
            self.graph.set_f32("latent", &latent)?;
            self.graph.set_f32("txt", txt)?;
            let cond = self.graph.compute()?;
            let velocity = if (guidance - 1.0).abs() < f32::EPSILON {
                cond
            } else {
                self.graph.set_f32("txt", &zeros)?;
                let uncond = self.graph.compute()?;
                ops::cfg_combine(&uncond, &cond, guidance)
            };
            let sigma_next = sigmas[step as usize + 1];
            euler_step(&mut latent, &velocity, t, sigma_next);
            self.steps_run += 1;
            on_step(step + 1, steps);
        }
        Ok(latent)
    }

    pub fn graph_computes(&self) -> u32 {
        self.graph.compute_count()
    }
}

fn forward(
    b: &GraphBuilder<'_>,
    weights: &HashMap<String, *mut ggml_tensor>,
    config: Flux2Config,
    latent: *mut ggml_tensor,
    txt_in: *mut ggml_tensor,
    temb_freq: *mut ggml_tensor,
    pos: *mut ggml_tensor,
    seq_img: i64,
    seq_txt: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let dim = config.inner_dim;
    let head = config.attn_head_dim;
    let heads = config.num_heads;
    let hidden = config.hidden();
    let seq = seq_txt + seq_img;
    let theta = config.rope_theta;

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
            b, weights, &p, txt, img, mod_txt, mod_img, pos, dim, head, heads, hidden, seq_txt,
            seq_img, theta,
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
            b, weights, &p, seq_h, mod_single, pos, dim, head, heads, hidden, seq, theta,
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
    b: &GraphBuilder<'_>,
    weights: &HashMap<String, *mut ggml_tensor>,
    prefix: &str,
    txt: *mut ggml_tensor,
    img: *mut ggml_tensor,
    mod_txt: *mut ggml_tensor,
    mod_img: *mut ggml_tensor,
    pos: *mut ggml_tensor,
    dim: i64,
    head: i64,
    heads: i64,
    hidden: i64,
    seq_txt: i64,
    seq_img: i64,
    theta: f32,
) -> Result<(*mut ggml_tensor, *mut ggml_tensor), RuntimeError> {
    let seq = seq_txt + seq_img;
    let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = mods(b, mod_img, dim)?;
    let (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) =
        mods(b, mod_txt, dim)?;
    let n_img = scale_shift(b, nn(b.norm(img, LN_EPS))?, scale_msa, shift_msa)?;
    let n_txt = scale_shift(b, nn(b.norm(txt, LN_EPS))?, c_scale_msa, c_shift_msa)?;
    let q_img = rms(
        b,
        linear(
            b,
            n_img,
            wt(weights, &format!("{prefix}.attn.to_q.weight"))?,
        ),
        wt(weights, &format!("{prefix}.attn.norm_q.weight"))?,
    )?;
    let k_img = rms(
        b,
        linear(
            b,
            n_img,
            wt(weights, &format!("{prefix}.attn.to_k.weight"))?,
        ),
        wt(weights, &format!("{prefix}.attn.norm_k.weight"))?,
    )?;
    let v_img = nn(linear(
        b,
        n_img,
        wt(weights, &format!("{prefix}.attn.to_v.weight"))?,
    ))?;
    let q_txt = rms(
        b,
        linear(
            b,
            n_txt,
            wt(weights, &format!("{prefix}.attn.add_q_proj.weight"))?,
        ),
        wt(weights, &format!("{prefix}.attn.norm_added_q.weight"))?,
    )?;
    let k_txt = rms(
        b,
        linear(
            b,
            n_txt,
            wt(weights, &format!("{prefix}.attn.add_k_proj.weight"))?,
        ),
        wt(weights, &format!("{prefix}.attn.norm_added_k.weight"))?,
    )?;
    let v_txt = nn(linear(
        b,
        n_txt,
        wt(weights, &format!("{prefix}.attn.add_v_proj.weight"))?,
    ))?;
    let q = nn(b.concat(q_txt, q_img, 1))?;
    let k = nn(b.concat(k_txt, k_img, 1))?;
    let v = nn(b.concat(v_txt, v_img, 1))?;
    let mixed = attend(b, q, k, v, pos, dim, head, heads, seq, theta)?;
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
    b: &GraphBuilder<'_>,
    weights: &HashMap<String, *mut ggml_tensor>,
    prefix: &str,
    seq_h: *mut ggml_tensor,
    mod_single: *mut ggml_tensor,
    pos: *mut ggml_tensor,
    dim: i64,
    head: i64,
    heads: i64,
    hidden: i64,
    seq: i64,
    theta: f32,
) -> Result<*mut ggml_tensor, RuntimeError> {
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
    let q = rms(
        b,
        features(b, qkv, 0, dim, dim * 3, seq)?,
        wt(weights, &format!("{prefix}.norm_q.weight"))?,
    )?;
    let k = rms(
        b,
        features(b, qkv, dim, dim, dim * 3, seq)?,
        wt(weights, &format!("{prefix}.norm_k.weight"))?,
    )?;
    let v = features(b, qkv, dim * 2, dim, dim * 3, seq)?;
    let attn = attend(b, q, k, v, pos, dim, head, heads, seq, theta)?;
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
    b: &GraphBuilder<'_>,
    q: *mut ggml_tensor,
    k: *mut ggml_tensor,
    v: *mut ggml_tensor,
    pos: *mut ggml_tensor,
    dim: i64,
    head: i64,
    heads: i64,
    seq: i64,
    theta: f32,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let q = nn(b.reshape_3d(q, head, heads, seq))?;
    let k = nn(b.reshape_3d(k, head, heads, seq))?;
    let q = nn(b.cont(b.rope(q, pos, head as i32, theta)))?;
    let k = nn(b.cont(b.rope(k, pos, head as i32, theta)))?;
    let q = nn(b.reshape_2d(q, dim, seq))?;
    let k = nn(b.reshape_2d(k, dim, seq))?;
    let scale = 1.0 / (head as f32).sqrt();
    let scores = nn(b.soft_max(b.scale(nn(b.mul_mat(k, q))?, scale)))?;
    let vt = nn(b.cont(b.transpose(v)))?;
    nn(b.mul_mat(vt, scores))
}

fn ffn(
    b: &GraphBuilder<'_>,
    x: *mut ggml_tensor,
    w_in: *mut ggml_tensor,
    w_out: *mut ggml_tensor,
    _hidden: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let mid = nn(b.swiglu(nn(linear(b, x, w_in))?))?;
    nn(linear(b, mid, w_out))
}

fn mods(
    b: &GraphBuilder<'_>,
    packed: *mut ggml_tensor,
    dim: i64,
) -> Result<
    (
        *mut ggml_tensor,
        *mut ggml_tensor,
        *mut ggml_tensor,
        *mut ggml_tensor,
        *mut ggml_tensor,
        *mut ggml_tensor,
    ),
    RuntimeError,
> {
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

fn rms(
    b: &GraphBuilder<'_>,
    x: *mut ggml_tensor,
    weight: *mut ggml_tensor,
) -> Result<*mut ggml_tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, weight))
}

fn scale_shift(
    b: &GraphBuilder<'_>,
    x: *mut ggml_tensor,
    scale: *mut ggml_tensor,
    shift: *mut ggml_tensor,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let scaled = nn(b.mul(x, scale))?;
    let y = nn(b.add(x, scaled))?;
    nn(b.add(y, shift))
}

fn gate_add(
    b: &GraphBuilder<'_>,
    residual: *mut ggml_tensor,
    gate: *mut ggml_tensor,
    delta: *mut ggml_tensor,
) -> Result<*mut ggml_tensor, RuntimeError> {
    nn(b.add(residual, nn(b.mul(delta, gate))?))
}

fn features(
    b: &GraphBuilder<'_>,
    tensor: *mut ggml_tensor,
    row0: i64,
    rows: i64,
    total_rows: i64,
    seq: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
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
    b: &GraphBuilder<'_>,
    tensor: *mut ggml_tensor,
    dim: i64,
    token0: i64,
    count: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let view = b.view_2d(
        tensor,
        dim,
        count,
        (dim as usize) * std::mem::size_of::<f32>(),
        (token0 as usize) * (dim as usize) * std::mem::size_of::<f32>(),
    );
    nn(b.cont(view))
}

fn linear(b: &GraphBuilder<'_>, x: *mut ggml_tensor, w: *mut ggml_tensor) -> *mut ggml_tensor {
    b.mul_mat(w, x)
}

fn wt(
    weights: &HashMap<String, *mut ggml_tensor>,
    name: &str,
) -> Result<*mut ggml_tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

fn nn(tensor: *mut ggml_tensor) -> Result<*mut ggml_tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}
