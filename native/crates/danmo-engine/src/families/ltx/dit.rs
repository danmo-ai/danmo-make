//! LTX 2.3 48-block joint audio/video DiT (`transformer_blocks.*` in the
//! published GGUF). Port of main's `LTX23Model` (`transformer_mlx.py`):
//! - `patchify_proj` / `audio_patchify_proj` embed the flattened latent
//!   voxels (`[latent, seq]` tokens, patch 1) into the video (4096) and audio
//!   (2048) towers; `proj_out` / `audio_proj_out` map back.
//! - Eight shared adaLN MLPs turn the scalar-sigma sinusoidal embedding
//!   (`sinusoidal_timestep(sigma * 1000, 256)` — both timestep multipliers
//!   are 1000 in the published config, so the A/V gate adaln shares the
//!   input) into per-block modulation vectors; each block adds its own
//!   scale-shift tables and applies `x * (1 + scale) + shift` over a
//!   weightless RMS norm (eps 1e-6).
//! - Every attention is `to_q/k/v` with bias, full-inner-dim q/k RMS norms,
//!   flash attention at `1/sqrt(head_dim)`, a per-(head, token)
//!   `2*sigmoid(to_gate_logits)` gate, and `to_out`. Self attentions use
//!   split RoPE over the pixel-space-second positions (video 3 axes
//!   `[20, 2048, 2048]`, audio 1 axis `[20]`, theta 10000, `theta^linspace`
//!   grid scaled by `pi/2`, zero-padded at the front when the axis split
//!   does not fill the half dim); text cross attentions use no rope; the
//!   A/V cross attentions rope q and k with the 1-axis cross tables.
//! - FFN is `gelu_tanh(net.0.proj)` -> `net.2`; the fused ggml `gelu` F16
//!   table is again avoided via the exact `x * sigmoid(2u)` form.
//! - The head is a weightless layer norm (eps 1e-6) modulated by the
//!   embedded timestep plus the top-level `scale_shift_table`, then the
//!   projection.
//! Per-token timesteps, attention masks, and perturbations (IC-LoRA/edit
//! paths) are not wired; the t2v sessions drive the scalar path.

use crate::runtime::{
    flash_attend, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

const GRAPH_MEM: usize = 256 << 20;
const NORM_EPS: f32 = 1e-6;
const ROPE_THETA: f64 = 10_000.0;
/// Published positional maxima (`positional_embedding_max_pos` /
/// `audio_positional_embedding_max_pos` in the GGUF config kv).
const VIDEO_MAX_POS: [f32; 3] = [20.0, 2048.0, 2048.0];
const AUDIO_MAX_POS: [f32; 1] = [20.0];
const CROSS_MAX_POS: [f32; 1] = [20.0];
/// Both timestep multipliers are 1000 in the published checkpoints
/// (`timestep_scale_multiplier`, `av_ca_timestep_scale_multiplier`), so the
/// sinusoidal input is `sinusoidal_timestep(sigma * 1000, t_dim)` and the
/// A/V gate adaLN shares it.
pub const TIMESTEP_SCALE: f32 = 1000.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TowerConfig {
    pub dim: i64,
    pub heads: i64,
    pub head_dim: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitConfig {
    /// Video tower (4096 / 32 heads / 128 head dim).
    pub video: TowerConfig,
    /// Audio tower (2048 / 32 / 64).
    pub audio: TowerConfig,
    /// A/V cross attention heads (32).
    pub cross_heads: i64,
    /// A/V cross attention head dim (64).
    pub cross_head_dim: i64,
    /// Joint blocks (48).
    pub layers: i64,
    /// Latent channels per token (128).
    pub latent: i64,
    /// Sinusoidal timestep width (256).
    pub t_dim: i64,
}

impl DitConfig {
    pub fn smoke() -> Self {
        Self {
            video: TowerConfig {
                dim: 64,
                heads: 4,
                head_dim: 16,
            },
            audio: TowerConfig {
                dim: 32,
                heads: 4,
                head_dim: 8,
            },
            cross_heads: 4,
            cross_head_dim: 8,
            layers: 2,
            latent: 8,
            t_dim: 16,
        }
    }

    fn cross_inner(&self) -> i64 {
        self.cross_heads * self.cross_head_dim
    }
}

// ---------------------------------------------------------------------------
// weight layout
// ---------------------------------------------------------------------------

pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) ne0: i64,
    pub(crate) ne1: i64,
}

/// Biases, norms, and scale-shift tables are F32 in the graph (widened on
/// load); the linear matrices keep their stored type (BF16 for the
/// published GGUF).
fn is_f32_weight(name: &str) -> bool {
    name.ends_with(".bias") || name.ends_with("norm.weight") || name.contains("scale_shift_table")
}

fn adaln_entries(out: &mut Vec<Entry>, prefix: &str, dim: i64, num_params: i64, t_dim: i64) {
    let emb = format!("{prefix}.emb.timestep_embedder");
    out.push(Entry {
        name: format!("{emb}.linear_1.weight"),
        ne0: t_dim,
        ne1: dim,
    });
    out.push(Entry {
        name: format!("{emb}.linear_1.bias"),
        ne0: dim,
        ne1: 1,
    });
    out.push(Entry {
        name: format!("{emb}.linear_2.weight"),
        ne0: dim,
        ne1: dim,
    });
    out.push(Entry {
        name: format!("{emb}.linear_2.bias"),
        ne0: dim,
        ne1: 1,
    });
    out.push(Entry {
        name: format!("{prefix}.linear.weight"),
        ne0: dim,
        ne1: dim * num_params,
    });
    out.push(Entry {
        name: format!("{prefix}.linear.bias"),
        ne0: dim * num_params,
        ne1: 1,
    });
}

#[allow(clippy::too_many_arguments)]
fn attn_entries(
    out: &mut Vec<Entry>,
    prefix: &str,
    query_dim: i64,
    kv_dim: i64,
    inner: i64,
    out_dim: i64,
    heads: i64,
) {
    for (proj, inp, outp) in [
        ("to_q", query_dim, inner),
        ("to_k", kv_dim, inner),
        ("to_v", kv_dim, inner),
        ("to_out.0", inner, out_dim),
    ] {
        out.push(Entry {
            name: format!("{prefix}.{proj}.weight"),
            ne0: inp,
            ne1: outp,
        });
        out.push(Entry {
            name: format!("{prefix}.{proj}.bias"),
            ne0: outp,
            ne1: 1,
        });
    }
    out.push(Entry {
        name: format!("{prefix}.to_gate_logits.weight"),
        ne0: query_dim,
        ne1: heads,
    });
    out.push(Entry {
        name: format!("{prefix}.to_gate_logits.bias"),
        ne0: heads,
        ne1: 1,
    });
    for norm in ["q_norm", "k_norm"] {
        out.push(Entry {
            name: format!("{prefix}.{norm}.weight"),
            ne0: inner,
            ne1: 1,
        });
    }
}

fn ff_entries(out: &mut Vec<Entry>, prefix: &str, dim: i64) {
    out.push(Entry {
        name: format!("{prefix}.net.0.proj.weight"),
        ne0: dim,
        ne1: dim * 4,
    });
    out.push(Entry {
        name: format!("{prefix}.net.0.proj.bias"),
        ne0: dim * 4,
        ne1: 1,
    });
    out.push(Entry {
        name: format!("{prefix}.net.2.weight"),
        ne0: dim * 4,
        ne1: dim,
    });
    out.push(Entry {
        name: format!("{prefix}.net.2.bias"),
        ne0: dim,
        ne1: 1,
    });
}

pub(crate) fn layout(config: &DitConfig) -> Vec<Entry> {
    let mut out = Vec::new();
    let (vd, ad) = (config.video.dim, config.audio.dim);
    let ci = config.cross_inner();
    for (name, inp, outp) in [
        ("patchify_proj", config.latent, vd),
        ("audio_patchify_proj", config.latent, ad),
        ("proj_out", vd, config.latent),
        ("audio_proj_out", ad, config.latent),
    ] {
        out.push(Entry {
            name: format!("{name}.weight"),
            ne0: inp,
            ne1: outp,
        });
        out.push(Entry {
            name: format!("{name}.bias"),
            ne0: outp,
            ne1: 1,
        });
    }
    out.push(Entry {
        name: "scale_shift_table".into(),
        ne0: vd,
        ne1: 2,
    });
    out.push(Entry {
        name: "audio_scale_shift_table".into(),
        ne0: ad,
        ne1: 2,
    });
    for (name, dim, np) in [
        ("adaln_single", vd, 9),
        ("audio_adaln_single", ad, 9),
        ("prompt_adaln_single", vd, 2),
        ("audio_prompt_adaln_single", ad, 2),
        ("av_ca_video_scale_shift_adaln_single", vd, 4),
        ("av_ca_audio_scale_shift_adaln_single", ad, 4),
        ("av_ca_a2v_gate_adaln_single", vd, 1),
        ("av_ca_v2a_gate_adaln_single", ad, 1),
    ] {
        adaln_entries(&mut out, name, dim, np, config.t_dim);
    }
    for i in 0..config.layers {
        let b = format!("transformer_blocks.{i}");
        attn_entries(
            &mut out,
            &format!("{b}.attn1"),
            vd,
            vd,
            vd,
            vd,
            config.video.heads,
        );
        attn_entries(
            &mut out,
            &format!("{b}.attn2"),
            vd,
            vd,
            vd,
            vd,
            config.video.heads,
        );
        attn_entries(
            &mut out,
            &format!("{b}.audio_attn1"),
            ad,
            ad,
            ad,
            ad,
            config.audio.heads,
        );
        attn_entries(
            &mut out,
            &format!("{b}.audio_attn2"),
            ad,
            ad,
            ad,
            ad,
            config.audio.heads,
        );
        attn_entries(
            &mut out,
            &format!("{b}.audio_to_video_attn"),
            vd,
            ad,
            ci,
            vd,
            config.cross_heads,
        );
        attn_entries(
            &mut out,
            &format!("{b}.video_to_audio_attn"),
            ad,
            vd,
            ci,
            ad,
            config.cross_heads,
        );
        ff_entries(&mut out, &format!("{b}.ff"), vd);
        ff_entries(&mut out, &format!("{b}.audio_ff"), ad);
        for (table, dim, n) in [
            ("scale_shift_table", vd, 9),
            ("audio_scale_shift_table", ad, 9),
            ("prompt_scale_shift_table", vd, 2),
            ("audio_prompt_scale_shift_table", ad, 2),
            ("scale_shift_table_a2v_ca_video", vd, 5),
            ("scale_shift_table_a2v_ca_audio", ad, 5),
        ] {
            out.push(Entry {
                name: format!("{b}.{table}"),
                ne0: dim,
                ne1: n,
            });
        }
    }
    out
}

// ---------------------------------------------------------------------------
// rope tables (host, mirrors main's `_precompute_rope_freqs` split path)
// ---------------------------------------------------------------------------

/// Split-RoPE `(cos, sin)` tables `[head_dim/2, heads, seq]`. `positions` is
/// token-major `[seq][num_axes]`. The `theta^linspace(0,1)` grid scaled by
/// pi/2 is built per axis, zero-padded at the front when
/// `num_axes * (inner / (2 * num_axes)) < inner / 2` (main pads the same
/// way), then folded to per-head halves.
pub(crate) fn rope_tables(
    positions: &[f32],
    num_axes: usize,
    max_pos: &[f32],
    inner: i64,
    heads: i64,
) -> (Vec<f32>, Vec<f32>) {
    let seq = positions.len() / num_axes;
    let head_half = (inner / heads / 2) as usize;
    let num_freqs = inner as usize / (2 * num_axes);
    let expected = inner as usize / 2;
    let total = num_freqs * num_axes;
    let pad = expected - total;
    let denom = num_freqs.saturating_sub(1).max(1) as f64;
    let mut cos = vec![0f32; head_half * heads as usize * seq];
    let mut sin = cos.clone();
    for s in 0..seq {
        let mut freqs = vec![0f64; expected];
        for (j, slot) in freqs.iter_mut().enumerate().skip(pad).take(total) {
            let axis = (j - pad) / num_freqs;
            let k = (j - pad) % num_freqs;
            let freq = ROPE_THETA.powf(k as f64 / denom) * std::f64::consts::FRAC_PI_2;
            let frac = positions[s * num_axes + axis] as f64 / max_pos[axis] as f64 * 2.0 - 1.0;
            *slot = freq * frac;
        }
        for h in 0..heads as usize {
            for d in 0..head_half {
                let f = freqs[h * head_half + d];
                let index = d + head_half * (h + heads as usize * s);
                cos[index] = f.cos() as f32;
                sin[index] = f.sin() as f32;
            }
        }
    }
    (cos, sin)
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
        .ok_or_else(|| RuntimeError::Message(format!("ltx dit graph is missing {name}")))
}

fn linear_bias(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.mul_mat(wt(weights, &format!("{prefix}.weight"))?, x))?;
    nn(b.add(y, wt(weights, &format!("{prefix}.bias"))?))
}

/// `normed * (1 + scale) + shift` with `[dim, 1]` broadcast vectors.
fn modulate(
    b: &GraphBuilder,
    normed: Tensor,
    scale: Tensor,
    shift: Tensor,
) -> Result<Tensor, RuntimeError> {
    nn(b.add(nn(b.add(normed, nn(b.mul(normed, scale))?))?, shift))
}

/// The reference split RoPE: rotate the two head-dim halves with the
/// per-(head, position, pair) tables (same layout as the connectors).
fn rope_split(
    b: &GraphBuilder,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    half: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let nb1 = (2 * half) as usize * 4;
    let nb2 = (2 * half * heads) as usize * 4;
    let first = nn(b.cont(b.view_3d(x, half, heads, seq, nb1, nb2, 0)))?;
    let second = nn(b.cont(b.view_3d(x, half, heads, seq, nb1, nb2, (half * 4) as usize)))?;
    let rot1 = nn(b.sub(nn(b.mul(first, cos))?, nn(b.mul(second, sin))?))?;
    let rot2 = nn(b.add(nn(b.mul(first, sin))?, nn(b.mul(second, cos))?))?;
    nn(b.concat(nn(b.cont(rot1))?, nn(b.cont(rot2))?, 0))
}

struct RopePair {
    cos: Tensor,
    sin: Tensor,
}

/// One `LTX23Attention`: q/k RMS norms, optional split RoPE on q and k,
/// flash attention, per-head gate, output projection. `stg` is the STG
/// blend gate: `out = attn * g + v * (1 - g)` (main passes the value tensor
/// through when the block's self attention is skipped); `None` keeps the
/// plain attention output.
#[allow(clippy::too_many_arguments)]
fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
    kv: Tensor,
    q_seq: i64,
    kv_seq: i64,
    inner: i64,
    heads: i64,
    head_dim: i64,
    rope: Option<&RopePair>,
    rope_k: Option<&RopePair>,
    stg: Option<Tensor>,
) -> Result<Tensor, RuntimeError> {
    let q = linear_bias(b, weights, &format!("{prefix}.to_q"), x)?;
    let q = nn(b.mul(
        nn(b.rms_norm(q, NORM_EPS))?,
        wt(weights, &format!("{prefix}.q_norm.weight"))?,
    ))?;
    let q = nn(b.cont(nn(b.reshape_3d(q, head_dim, heads, q_seq))?))?;
    let k = linear_bias(b, weights, &format!("{prefix}.to_k"), kv)?;
    let k = nn(b.mul(
        nn(b.rms_norm(k, NORM_EPS))?,
        wt(weights, &format!("{prefix}.k_norm.weight"))?,
    ))?;
    let k = nn(b.cont(nn(b.reshape_3d(k, head_dim, heads, kv_seq))?))?;
    let v = linear_bias(b, weights, &format!("{prefix}.to_v"), kv)?;
    let v = nn(b.cont(nn(b.reshape_3d(v, head_dim, heads, kv_seq))?))?;
    let half = head_dim / 2;
    let q = match rope {
        Some(pair) => rope_split(b, q, pair.cos, pair.sin, half, heads, q_seq)?,
        None => q,
    };
    let k = match rope_k.or(rope) {
        Some(pair) => rope_split(b, k, pair.cos, pair.sin, half, heads, kv_seq)?,
        None => k,
    };
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    // STG skip: blend the attention output with the value projection.
    let out = match stg {
        Some(g) => nn(b.add(v, nn(b.mul(nn(b.sub(out, v))?, nn(b.repeat(g, out))?))?))?,
        None => out,
    };
    // gate: 2 * sigmoid(to_gate_logits(x)), one scalar per (head, token).
    let gate = linear_bias(b, weights, &format!("{prefix}.to_gate_logits"), x)?;
    let gate = nn(b.scale(nn(b.sigmoid(gate))?, 2.0))?;
    let gate = nn(b.reshape_3d(gate, 1, heads, q_seq))?;
    let out = nn(b.mul(out, gate))?;
    let out = nn(b.reshape_2d(out, inner, q_seq))?;
    linear_bias(b, weights, &format!("{prefix}.to_out.0"), out)
}

/// adaLN MLP: `linear(silu(emb(t_emb)))`; returns `(params, embedded)`.
fn adaln(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    t_emb: Tensor,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let emb = format!("{prefix}.emb.timestep_embedder");
    let embedded = linear_bias(b, weights, &format!("{emb}.linear_1"), t_emb)?;
    let embedded = nn(b.silu(embedded))?;
    let embedded = linear_bias(b, weights, &format!("{emb}.linear_2"), embedded)?;
    let params = nn(b.silu(embedded))?;
    let params = linear_bias(b, weights, &format!("{prefix}.linear"), params)?;
    Ok((params, embedded))
}

/// `params [dim * n, 1]` plus the first n columns of the `[dim, n]` table
/// (main adds `table[:num_params]`; the a2v/v2a tables carry a 5th gate row
/// applied separately), split into n `[dim, 1]` columns.
fn mod_columns(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    table: &str,
    params: Tensor,
    dim: i64,
    n: i64,
) -> Result<Vec<Tensor>, RuntimeError> {
    let t = wt(weights, table)?;
    let t = nn(b.cont(b.view_2d(t, dim, n, (dim * 4) as usize, 0)))?;
    let p = nn(b.reshape_2d(params, dim, n))?;
    let p = nn(b.add(p, t))?;
    (0..n)
        .map(|i| {
            let col = nn(b.view_2d(p, dim, 1, (dim * 4) as usize, (i * dim * 4) as usize))?;
            nn(b.cont(col))
        })
        .collect()
}

/// One column of a `[dim, n]` table as `[dim, 1]`.
fn table_column(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    table: &str,
    dim: i64,
    col: i64,
) -> Result<Tensor, RuntimeError> {
    let t = wt(weights, table)?;
    nn(b.cont(b.view_2d(t, dim, 1, (dim * 4) as usize, (col * dim * 4) as usize)))
}

pub struct DitGraph {
    graph: ReusedGraph,
    config: DitConfig,
    v_seq: i64,
    a_seq: i64,
    t_seq: i64,
}

impl DitGraph {
    fn build(
        ctx: &GgmlContext,
        config: &DitConfig,
        v_seq: i64,
        a_seq: i64,
        t_seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(GRAPH_MEM)?;
        let mut weights = HashMap::new();
        for entry in layout(config) {
            let ggml_type = types.get(&entry.name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&entry.name, ggml_type, entry.ne0, entry.ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(entry.name, tensor);
        }
        let (vd, ad) = (config.video.dim, config.audio.dim);
        let ci = config.cross_inner();
        let vid = nn(b.f32_2d("video_latent", config.latent, v_seq, true))?;
        let aud = nn(b.f32_2d("audio_latent", config.latent, a_seq, true))?;
        let t_emb = nn(b.f32_2d("t_emb", config.t_dim, 1, true))?;
        let vtext = nn(b.f32_2d("video_text", vd, t_seq, true))?;
        let atext = nn(b.f32_2d("audio_text", ad, t_seq, true))?;
        let mut rope =
            |tag: &str, inner: i64, heads: i64, seq: i64| -> Result<RopePair, RuntimeError> {
                let half = inner / heads / 2;
                Ok(RopePair {
                    cos: nn(b.f32_3d(&format!("rope_{tag}_cos"), half, heads, seq, true))?,
                    sin: nn(b.f32_3d(&format!("rope_{tag}_sin"), half, heads, seq, true))?,
                })
            };
        let v_rope = rope("v", vd, config.video.heads, v_seq)?;
        let a_rope = rope("a", ad, config.audio.heads, a_seq)?;
        let cv_rope = rope("cv", ci, config.cross_heads, v_seq)?;
        let ca_rope = rope("ca", ci, config.cross_heads, a_seq)?;

        // Shared adaLN MLPs. The A/V gate embedding equals t_emb because
        // both timestep multipliers are 1000.
        let (v_params, v_embedded) = adaln(&b, &weights, "adaln_single", t_emb)?;
        let (a_params, a_embedded) = adaln(&b, &weights, "audio_adaln_single", t_emb)?;
        let (vp_params, _) = adaln(&b, &weights, "prompt_adaln_single", t_emb)?;
        let (ap_params, _) = adaln(&b, &weights, "audio_prompt_adaln_single", t_emb)?;
        let (avv_params, _) = adaln(&b, &weights, "av_ca_video_scale_shift_adaln_single", t_emb)?;
        let (ava_params, _) = adaln(&b, &weights, "av_ca_audio_scale_shift_adaln_single", t_emb)?;
        let (a2v_gate, _) = adaln(&b, &weights, "av_ca_a2v_gate_adaln_single", t_emb)?;
        let (v2a_gate, _) = adaln(&b, &weights, "av_ca_v2a_gate_adaln_single", t_emb)?;

        let mut video = linear_bias(&b, &weights, "patchify_proj", vid)?;
        let mut audio = linear_bias(&b, &weights, "audio_patchify_proj", aud)?;

        for i in 0..config.layers {
            let blk = format!("transformer_blocks.{i}");
            let v = mod_columns(
                &b,
                &weights,
                &format!("{blk}.scale_shift_table"),
                v_params,
                vd,
                9,
            )?;
            let a = mod_columns(
                &b,
                &weights,
                &format!("{blk}.audio_scale_shift_table"),
                a_params,
                ad,
                9,
            )?;
            let vp = mod_columns(
                &b,
                &weights,
                &format!("{blk}.prompt_scale_shift_table"),
                vp_params,
                vd,
                2,
            )?;
            let ap = mod_columns(
                &b,
                &weights,
                &format!("{blk}.audio_prompt_scale_shift_table"),
                ap_params,
                ad,
                2,
            )?;
            let avv = mod_columns(
                &b,
                &weights,
                &format!("{blk}.scale_shift_table_a2v_ca_video"),
                avv_params,
                vd,
                4,
            )?;
            let ava = mod_columns(
                &b,
                &weights,
                &format!("{blk}.scale_shift_table_a2v_ca_audio"),
                ava_params,
                ad,
                4,
            )?;
            let gate_a2v = nn(b.add(
                a2v_gate,
                table_column(
                    &b,
                    &weights,
                    &format!("{blk}.scale_shift_table_a2v_ca_video"),
                    vd,
                    4,
                )?,
            ))?;
            let gate_v2a = nn(b.add(
                v2a_gate,
                table_column(
                    &b,
                    &weights,
                    &format!("{blk}.scale_shift_table_a2v_ca_audio"),
                    ad,
                    4,
                )?,
            ))?;

            // video / audio self attention
            let ptb_vself = nn(b.f32_1d(&format!("ptb_vself_{i}"), 1, true))?;
            let ptb_aself = nn(b.f32_1d(&format!("ptb_aself_{i}"), 1, true))?;
            let ptb_a2v = nn(b.f32_1d(&format!("ptb_a2v_{i}"), 1, true))?;
            let ptb_v2a = nn(b.f32_1d(&format!("ptb_v2a_{i}"), 1, true))?;
            let normed = nn(b.rms_norm(video, NORM_EPS))?;
            let x = modulate(&b, normed, v[1], v[0])?;
            let sa = attention(
                &b,
                &weights,
                &format!("{blk}.attn1"),
                x,
                x,
                v_seq,
                v_seq,
                vd,
                config.video.heads,
                config.video.head_dim,
                Some(&v_rope),
                None,
                Some(ptb_vself),
            )?;
            video = nn(b.add(video, nn(b.mul(sa, v[2]))?))?;

            let normed = nn(b.rms_norm(audio, NORM_EPS))?;
            let x = modulate(&b, normed, a[1], a[0])?;
            let sa = attention(
                &b,
                &weights,
                &format!("{blk}.audio_attn1"),
                x,
                x,
                a_seq,
                a_seq,
                ad,
                config.audio.heads,
                config.audio.head_dim,
                Some(&a_rope),
                None,
                Some(ptb_aself),
            )?;
            audio = nn(b.add(audio, nn(b.mul(sa, a[2]))?))?;

            // text cross attention (no rope; prompt-modulated text)
            let normed = nn(b.rms_norm(video, NORM_EPS))?;
            let x = modulate(&b, normed, v[7], v[6])?;
            let text = modulate(&b, vtext, vp[1], vp[0])?;
            let ca = attention(
                &b,
                &weights,
                &format!("{blk}.attn2"),
                x,
                text,
                v_seq,
                t_seq,
                vd,
                config.video.heads,
                config.video.head_dim,
                None,
                None,
                None,
            )?;
            video = nn(b.add(video, nn(b.mul(ca, v[8]))?))?;

            let normed = nn(b.rms_norm(audio, NORM_EPS))?;
            let x = modulate(&b, normed, a[7], a[6])?;
            let text = modulate(&b, atext, ap[1], ap[0])?;
            let ca = attention(
                &b,
                &weights,
                &format!("{blk}.audio_attn2"),
                x,
                text,
                a_seq,
                t_seq,
                ad,
                config.audio.heads,
                config.audio.head_dim,
                None,
                None,
                None,
            )?;
            audio = nn(b.add(audio, nn(b.mul(ca, a[8]))?))?;

            // joint A/V cross attention
            let video_n3 = nn(b.rms_norm(video, NORM_EPS))?;
            let audio_n3 = nn(b.rms_norm(audio, NORM_EPS))?;
            let q = modulate(&b, video_n3, avv[0], avv[1])?;
            let kv = modulate(&b, audio_n3, ava[0], ava[1])?;
            let a2v = attention(
                &b,
                &weights,
                &format!("{blk}.audio_to_video_attn"),
                q,
                kv,
                v_seq,
                a_seq,
                ci,
                config.cross_heads,
                config.cross_head_dim,
                Some(&cv_rope),
                Some(&ca_rope),
                None,
            )?;
            let a2v = nn(b.mul(a2v, nn(b.repeat(ptb_a2v, a2v))?))?;
            video = nn(b.add(video, nn(b.mul(a2v, gate_a2v))?))?;

            let q = modulate(&b, audio_n3, ava[2], ava[3])?;
            let kv = modulate(&b, video_n3, avv[2], avv[3])?;
            let v2a = attention(
                &b,
                &weights,
                &format!("{blk}.video_to_audio_attn"),
                q,
                kv,
                a_seq,
                v_seq,
                ci,
                config.cross_heads,
                config.cross_head_dim,
                Some(&ca_rope),
                Some(&cv_rope),
                None,
            )?;
            let v2a = nn(b.mul(v2a, nn(b.repeat(ptb_v2a, v2a))?))?;
            audio = nn(b.add(audio, nn(b.mul(v2a, gate_v2a))?))?;

            // feed forwards
            let normed = nn(b.rms_norm(video, NORM_EPS))?;
            let x = modulate(&b, normed, v[4], v[3])?;
            let ff = linear_bias(&b, &weights, &format!("{blk}.ff.net.0.proj"), x)?;
            let ff = super::gelu_tanh(&b, ff)?;
            let ff = linear_bias(&b, &weights, &format!("{blk}.ff.net.2"), ff)?;
            video = nn(b.add(video, nn(b.mul(ff, v[5]))?))?;

            let normed = nn(b.rms_norm(audio, NORM_EPS))?;
            let x = modulate(&b, normed, a[4], a[3])?;
            let ff = linear_bias(&b, &weights, &format!("{blk}.audio_ff.net.0.proj"), x)?;
            let ff = super::gelu_tanh(&b, ff)?;
            let ff = linear_bias(&b, &weights, &format!("{blk}.audio_ff.net.2"), ff)?;
            audio = nn(b.add(audio, nn(b.mul(ff, a[5]))?))?;
        }

        let video_out = output_block(
            &b,
            &weights,
            video,
            v_embedded,
            "scale_shift_table",
            "proj_out",
            vd,
        )?;
        let audio_out = output_block(
            &b,
            &weights,
            audio,
            a_embedded,
            "audio_scale_shift_table",
            "audio_proj_out",
            ad,
        )?;
        let out = nn(b.concat(video_out, audio_out, 1))?;
        let graph = b.finish(out)?;
        Ok(Self {
            graph,
            config: *config,
            v_seq,
            a_seq,
            t_seq,
        })
    }

    /// Run one denoise step. Returns `(video_velocity, audio_velocity)`,
    /// each `[latent, seq]` flat in ggml order.
    pub fn forward(&mut self, step: &DitStep) -> Result<(Vec<f32>, Vec<f32>), WeightError> {
        let config = &self.config;
        let expect = [
            (
                "video_latent",
                step.video_latent.len() as i64,
                config.latent * self.v_seq,
            ),
            (
                "audio_latent",
                step.audio_latent.len() as i64,
                config.latent * self.a_seq,
            ),
            ("t_emb", step.t_emb.len() as i64, config.t_dim),
            (
                "video_text",
                step.video_text.len() as i64,
                config.video.dim * self.t_seq,
            ),
            (
                "audio_text",
                step.audio_text.len() as i64,
                config.audio.dim * self.t_seq,
            ),
        ];
        for (name, got, want) in expect {
            if got != want {
                return Err(WeightError::NotBuilt(format!(
                    "ltx dit {name} is {got} values, expected {want}"
                )));
            }
        }
        if step.video_positions.len() as i64 != self.v_seq * 3
            || step.audio_positions.len() as i64 != self.a_seq
        {
            return Err(WeightError::NotBuilt(format!(
                "ltx dit positions are video {} audio {}, expected {} and {}",
                step.video_positions.len(),
                step.audio_positions.len(),
                self.v_seq * 3,
                self.a_seq,
            )));
        }
        self.graph.set_f32("video_latent", &step.video_latent)?;
        self.graph.set_f32("audio_latent", &step.audio_latent)?;
        self.graph.set_f32("t_emb", &step.t_emb)?;
        self.graph.set_f32("video_text", &step.video_text)?;
        self.graph.set_f32("audio_text", &step.audio_text)?;
        for i in 0..config.layers {
            let gate = |skip: bool| if skip { 0.0f32 } else { 1.0f32 };
            self.graph.set_f32(
                &format!("ptb_vself_{i}"),
                &[gate(step.perturb.skip_video_self.contains(&i))],
            )?;
            self.graph.set_f32(
                &format!("ptb_aself_{i}"),
                &[gate(step.perturb.skip_audio_self.contains(&i))],
            )?;
            self.graph
                .set_f32(&format!("ptb_a2v_{i}"), &[gate(step.perturb.skip_av_cross)])?;
            self.graph
                .set_f32(&format!("ptb_v2a_{i}"), &[gate(step.perturb.skip_av_cross)])?;
        }
        let tables = [
            (
                "v",
                rope_tables(
                    &step.video_positions,
                    3,
                    &VIDEO_MAX_POS,
                    config.video.dim,
                    config.video.heads,
                ),
            ),
            (
                "a",
                rope_tables(
                    &step.audio_positions,
                    1,
                    &AUDIO_MAX_POS,
                    config.audio.dim,
                    config.audio.heads,
                ),
            ),
            (
                "cv",
                rope_tables(
                    &cross_positions(&step.video_positions),
                    1,
                    &CROSS_MAX_POS,
                    config.cross_inner(),
                    config.cross_heads,
                ),
            ),
            (
                "ca",
                rope_tables(
                    &step.audio_positions,
                    1,
                    &CROSS_MAX_POS,
                    config.cross_inner(),
                    config.cross_heads,
                ),
            ),
        ];
        for (tag, (cos, sin)) in &tables {
            self.graph.set_f32(&format!("rope_{tag}_cos"), cos)?;
            self.graph.set_f32(&format!("rope_{tag}_sin"), sin)?;
        }
        let out = self.graph.compute()?;
        if let Some(detail) = crate::common::ops::non_finite("ltx dit", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        let row = config.latent as usize;
        let v_tokens = self.v_seq as usize;
        let mut video = Vec::with_capacity(row * v_tokens);
        let mut audio = Vec::with_capacity(row * self.a_seq as usize);
        video.extend_from_slice(&out[..row * v_tokens]);
        audio.extend_from_slice(&out[row * v_tokens..]);
        Ok((video, audio))
    }
}

/// Axis 0 (frame seconds) of the video positions, for the cross rope.
fn cross_positions(video_positions: &[f32]) -> Vec<f32> {
    video_positions.iter().step_by(3).copied().collect()
}

/// Final norm + modulation + projection head for one tower.
fn output_block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    embedded: Tensor,
    table: &str,
    proj: &str,
    dim: i64,
) -> Result<Tensor, RuntimeError> {
    let shift = nn(b.add(table_column(b, weights, table, dim, 0)?, embedded))?;
    let scale = nn(b.add(table_column(b, weights, table, dim, 1)?, embedded))?;
    let normed = nn(b.norm(x, NORM_EPS))?;
    let x = modulate(b, normed, scale, shift)?;
    linear_bias(b, weights, proj, x)
}

/// Per-step inputs. All flats are ggml `[dim, seq]` order.
pub struct DitStep {
    pub video_latent: Vec<f32>,
    pub audio_latent: Vec<f32>,
    /// Host sinusoidal embedding `sinusoidal_timestep(sigma * 1000, t_dim)`.
    pub t_emb: Vec<f32>,
    pub video_text: Vec<f32>,
    pub audio_text: Vec<f32>,
    /// `[f, h, w]` per video token (seconds, pixel units), token-major.
    pub video_positions: Vec<f32>,
    /// Seconds per audio token.
    pub audio_positions: Vec<f32>,
    /// STG / modality perturbation gates (default: no perturbation).
    pub perturb: DitPerturb,
}

/// Perturbation gates for one denoise call (main
/// `perturbations_mlx`): skipping a block's self attention passes the value
/// projection through; skipping the A/V cross attention drops that residual
/// at every block.
#[derive(Clone, Debug, Default)]
pub struct DitPerturb {
    /// Block indices whose video self attention is skipped (STG).
    pub skip_video_self: Vec<i64>,
    /// Block indices whose audio self attention is skipped (STG).
    pub skip_audio_self: Vec<i64>,
    /// Skip both A/V cross attentions at every block (modality guidance).
    pub skip_av_cross: bool,
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

pub struct LtxDit {
    weights: Arc<GgufWeights>,
    config: DitConfig,
}

impl LtxDit {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let fail = |detail: String| WeightError::NotBuilt(format!("ltx dit: {detail}"));
        let patch = weights.shape("patchify_proj.weight")?;
        let latent = patch[0];
        let vd = patch[1];
        let audio_patch = weights.shape("audio_patchify_proj.weight")?;
        if audio_patch[0] != latent {
            return Err(fail(format!(
                "audio patchify input {} != video latent {latent}",
                audio_patch[0]
            )));
        }
        let ad = audio_patch[1];
        let mut layers = 0i64;
        while weights.contains(&format!("transformer_blocks.{layers}.attn1.to_q.weight")) {
            layers += 1;
        }
        if layers == 0 {
            return Err(fail("no transformer_blocks".into()));
        }
        let tower = |prefix: &str, dim: i64| -> Result<TowerConfig, WeightError> {
            let q = weights.shape(&format!("transformer_blocks.0.{prefix}.to_q.weight"))?;
            if q[0] != dim {
                return Err(fail(format!("{prefix} to_q input {} != {dim}", q[0])));
            }
            let gate = weights.shape(&format!(
                "transformer_blocks.0.{prefix}.to_gate_logits.weight"
            ))?;
            let heads = gate[1];
            if heads < 1 || q[1] % heads != 0 {
                return Err(fail(format!(
                    "{prefix} inner {} is not divisible by heads {heads}",
                    q[1]
                )));
            }
            let head_dim = q[1] / heads;
            if head_dim % 2 != 0 {
                return Err(fail(format!("{prefix} head dim {head_dim} is not even")));
            }
            Ok(TowerConfig {
                dim: q[1],
                heads,
                head_dim,
            })
        };
        let video = tower("attn1", vd)?;
        if video.dim != vd {
            return Err(fail(format!(
                "attn1 inner {} != patchify dim {vd}",
                video.dim
            )));
        }
        let audio = tower("audio_attn1", ad)?;
        if audio.dim != ad {
            return Err(fail(format!(
                "audio_attn1 inner {} != audio patchify dim {ad}",
                audio.dim
            )));
        }
        let cross_gate =
            weights.shape("transformer_blocks.0.audio_to_video_attn.to_gate_logits.weight")?;
        let cross_q = weights.shape("transformer_blocks.0.audio_to_video_attn.to_q.weight")?;
        if cross_q[0] != vd || cross_gate[0] != vd {
            return Err(fail(format!(
                "audio_to_video query dims {cross_q:?}/{cross_gate:?} != {vd}"
            )));
        }
        let cross_heads = cross_gate[1];
        if cross_heads < 1 || cross_q[1] % cross_heads != 0 {
            return Err(fail(format!(
                "audio_to_video inner {} is not divisible by heads {cross_heads}",
                cross_q[1]
            )));
        }
        let cross_head_dim = cross_q[1] / cross_heads;
        if cross_head_dim % 2 != 0 {
            return Err(fail(format!(
                "audio_to_video head dim {cross_head_dim} is not even"
            )));
        }
        let t_dim = weights.shape("adaln_single.emb.timestep_embedder.linear_1.weight")?[0];
        let config = DitConfig {
            video,
            audio,
            cross_heads,
            cross_head_dim,
            layers,
            latent,
            t_dim,
        };
        // The fixed structure: every declared tensor must exist.
        for entry in layout(&config) {
            if !weights.contains(&entry.name) {
                return Err(WeightError::MissingTensor {
                    path: "transformer".into(),
                    name: entry.name,
                });
            }
        }
        Ok(Self { weights, config })
    }

    pub fn config(&self) -> DitConfig {
        self.config
    }

    /// Build a reusable graph for one token geometry and load the weights
    /// into it. Replay `forward` per denoise step.
    pub fn build_graph(
        &self,
        ctx: &GgmlContext,
        v_seq: i64,
        a_seq: i64,
        t_seq: i64,
    ) -> Result<DitGraph, WeightError> {
        let entries = layout(&self.config);
        let mut types = HashMap::new();
        for entry in &entries {
            let ggml_type = if is_f32_weight(&entry.name) {
                TYPE_F32
            } else {
                let ggml_type = self.weights.ggml_type_of(&entry.name)?;
                if !matmul_type(ggml_type) {
                    return Err(WeightError::Type {
                        name: entry.name.clone(),
                        ggml_type,
                    });
                }
                let block = type_block(ggml_type);
                if block > 1 && entry.ne0 % block != 0 {
                    return Err(WeightError::NotBuilt(format!(
                        "{} length {} is not divisible by ggml block {block}",
                        entry.name, entry.ne0
                    )));
                }
                ggml_type
            };
            types.insert(entry.name.clone(), ggml_type);
        }
        let mut graph = DitGraph::build(ctx, &self.config, v_seq, a_seq, t_seq, &types)?;
        for entry in &entries {
            if is_f32_weight(&entry.name) {
                let values = self
                    .weights
                    .f32_widened(&entry.name, &[entry.ne0, entry.ne1])?;
                graph.graph.set_f32(&entry.name, &values)?;
            } else {
                let raw = self
                    .weights
                    .raw_exact(&entry.name, &[entry.ne0, entry.ne1])?;
                graph.graph.set_bytes(&entry.name, &raw)?;
            }
        }
        Ok(graph)
    }
}

// ---------------------------------------------------------------------------
// smoke writer (graph-vs-host tests)
// ---------------------------------------------------------------------------

/// Deterministic smoke weights for `DitConfig::smoke()`.
pub fn smoke_store(config: &DitConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 97u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let mut fill = |count: usize| (0..count).map(|_| next()).collect::<Vec<f32>>();
    layout(config)
        .into_iter()
        .map(|entry| {
            let scale = if entry.name.contains("scale_shift_table") {
                0.4
            } else {
                0.2
            };
            (
                entry.name,
                fill((entry.ne0 * entry.ne1) as usize)
                    .into_iter()
                    .map(|v| v * scale)
                    .collect(),
            )
        })
        .collect()
}

/// Write the smoke DiT as a typed GGUF.
pub fn write_smoke(dir: &std::path::Path) -> Result<PathBuf, WeightError> {
    write_smoke_with(dir, &DitConfig::smoke())
}

/// Write the smoke DiT GGUF for an explicit config (session smoke bundle).
pub fn write_smoke_with(dir: &std::path::Path, config: &DitConfig) -> Result<PathBuf, WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = *config;
    let store = smoke_store(&config);
    let entries = layout(&config);
    let mut bytes: Vec<Vec<u8>> = Vec::new();
    for (_, values) in &store {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(data);
    }
    let tensors: Vec<crate::weights::TypedTensor<'_>> = entries
        .iter()
        .zip(&store)
        .zip(&bytes)
        .map(|((entry, _), data)| crate::weights::TypedTensor {
            name: &entry.name,
            ne0: entry.ne0,
            ne1: entry.ne1,
            ggml_type: crate::runtime::TYPE_F32,
            data,
        })
        .collect();
    let path = dir.join("ltx_dit_smoke.gguf");
    crate::weights::write_typed_tensors(&path, &tensors)?;
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

    fn gelu_tanh_host(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
    }

    fn silu_host(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn rms_host(x: &[f32], w: Option<&[f32]>, dim: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0f32; dim * seq];
        for s in 0..seq {
            let mut sum = 0f32;
            for d in 0..dim {
                sum += x[d + dim * s] * x[d + dim * s];
            }
            let scale = 1.0 / (sum / dim as f32 + NORM_EPS).sqrt();
            for d in 0..dim {
                out[d + dim * s] = x[d + dim * s] * scale * w.map_or(1.0, |w| w[d]);
            }
        }
        out
    }

    fn layer_norm_host(x: &[f32], dim: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0f32; dim * seq];
        for s in 0..seq {
            let mut mean = 0f32;
            for d in 0..dim {
                mean += x[d + dim * s];
            }
            mean /= dim as f32;
            let mut var = 0f32;
            for d in 0..dim {
                let v = x[d + dim * s] - mean;
                var += v * v;
            }
            let scale = 1.0 / (var / dim as f32 + NORM_EPS).sqrt();
            for d in 0..dim {
                out[d + dim * s] = (x[d + dim * s] - mean) * scale;
            }
        }
        out
    }

    fn linear_bias_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        inp: usize,
        out: usize,
        seq: usize,
    ) -> Vec<f32> {
        let w = at(store, &format!("{prefix}.weight"));
        let bias = at(store, &format!("{prefix}.bias"));
        let mut y = vec![0f32; out * seq];
        for s in 0..seq {
            for o in 0..out {
                let mut acc = bias[o];
                for i in 0..inp {
                    acc += w[i + inp * o] * x[i + inp * s];
                }
                y[o + out * s] = acc;
            }
        }
        y
    }

    fn modulate_host(x: &[f32], scale: &[f32], shift: &[f32], dim: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0f32; dim * seq];
        for s in 0..seq {
            for d in 0..dim {
                out[d + dim * s] = x[d + dim * s] * (1.0 + scale[d]) + shift[d];
            }
        }
        out
    }

    fn adaln_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        t_emb: &[f32],
        dim: usize,
        num_params: usize,
        t_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let emb = format!("{prefix}.emb.timestep_embedder");
        let e = linear_bias_host(store, &format!("{emb}.linear_1"), t_emb, t_dim, dim, 1);
        let e: Vec<f32> = e.iter().map(|v| silu_host(*v)).collect();
        let e = linear_bias_host(store, &format!("{emb}.linear_2"), &e, dim, dim, 1);
        let p: Vec<f32> = e.iter().map(|v| silu_host(*v)).collect();
        let p = linear_bias_host(
            store,
            &format!("{prefix}.linear"),
            &p,
            dim,
            dim * num_params,
            1,
        );
        (p, e)
    }

    /// `(params + table)` split into n columns of `dim`.
    fn mod_columns_host(params: &[f32], table: &[f32], dim: usize, n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| params[d + dim * i] + table[d + dim * i])
                    .collect()
            })
            .collect()
    }

    fn rope_apply_host(
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        heads: usize,
        head_dim: usize,
        seq: usize,
    ) {
        let half = head_dim / 2;
        for s in 0..seq {
            for h in 0..heads {
                for d in 0..half {
                    let t = d + half * (h + heads * s);
                    let i1 = d + head_dim * (h + heads * s);
                    let i2 = i1 + half;
                    let (x1, x2) = (x[i1], x[i2]);
                    x[i1] = x1 * cos[t] - x2 * sin[t];
                    x[i2] = x1 * sin[t] + x2 * cos[t];
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_host(
        store: &[(String, Vec<f32>)],
        prefix: &str,
        x: &[f32],
        kv: &[f32],
        qd: usize,
        kd: usize,
        inner: usize,
        out_dim: usize,
        heads: usize,
        head_dim: usize,
        q_seq: usize,
        kv_seq: usize,
        rope_q: Option<(&[f32], &[f32])>,
        rope_k: Option<(&[f32], &[f32])>,
        stg: f32,
    ) -> Vec<f32> {
        let q = linear_bias_host(store, &format!("{prefix}.to_q"), x, qd, inner, q_seq);
        let mut q = rms_host(
            &q,
            Some(at(store, &format!("{prefix}.q_norm.weight"))),
            inner,
            q_seq,
        );
        let k = linear_bias_host(store, &format!("{prefix}.to_k"), kv, kd, inner, kv_seq);
        let mut k = rms_host(
            &k,
            Some(at(store, &format!("{prefix}.k_norm.weight"))),
            inner,
            kv_seq,
        );
        let v = linear_bias_host(store, &format!("{prefix}.to_v"), kv, kd, inner, kv_seq);
        if let Some((cos, sin)) = rope_q {
            rope_apply_host(&mut q, cos, sin, heads, head_dim, q_seq);
        }
        if let Some((cos, sin)) = rope_k {
            rope_apply_host(&mut k, cos, sin, heads, head_dim, kv_seq);
        }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0f32; inner * q_seq];
        for h in 0..heads {
            for qp in 0..q_seq {
                let mut scores = vec![0f32; kv_seq];
                for kp in 0..kv_seq {
                    let mut acc = 0f32;
                    for d in 0..head_dim {
                        acc +=
                            q[d + head_dim * (h + heads * qp)] * k[d + head_dim * (h + heads * kp)];
                    }
                    scores[kp] = acc * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for d in 0..head_dim {
                    let mut acc = 0f32;
                    for kp in 0..kv_seq {
                        acc += scores[kp] / sum * v[d + head_dim * (h + heads * kp)];
                    }
                    out[d + head_dim * (h + heads * qp)] = acc;
                }
            }
        }
        // STG skip: blend with the value projection before the head gate.
        if stg != 1.0 {
            for i in 0..out.len() {
                out[i] = out[i] * stg + v[i] * (1.0 - stg);
            }
        }
        let gate = linear_bias_host(
            store,
            &format!("{prefix}.to_gate_logits"),
            x,
            qd,
            heads,
            q_seq,
        );
        for s in 0..q_seq {
            for h in 0..heads {
                let g = 2.0 / (1.0 + (-gate[h + heads * s]).exp());
                for d in 0..head_dim {
                    out[d + head_dim * (h + heads * s)] *= g;
                }
            }
        }
        linear_bias_host(
            store,
            &format!("{prefix}.to_out.0"),
            &out,
            inner,
            out_dim,
            q_seq,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_host(
        store: &[(String, Vec<f32>)],
        config: &DitConfig,
        step: &DitStep,
        v_seq: usize,
        a_seq: usize,
        t_seq: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let vd = config.video.dim as usize;
        let ad = config.audio.dim as usize;
        let ci = config.cross_inner() as usize;
        let latent = config.latent as usize;
        let t_dim = config.t_dim as usize;
        let (v_params, v_embedded) = adaln_host(store, "adaln_single", &step.t_emb, vd, 9, t_dim);
        let (a_params, a_embedded) =
            adaln_host(store, "audio_adaln_single", &step.t_emb, ad, 9, t_dim);
        let (vp_params, _) = adaln_host(store, "prompt_adaln_single", &step.t_emb, vd, 2, t_dim);
        let (ap_params, _) = adaln_host(
            store,
            "audio_prompt_adaln_single",
            &step.t_emb,
            ad,
            2,
            t_dim,
        );
        let (avv_params, _) = adaln_host(
            store,
            "av_ca_video_scale_shift_adaln_single",
            &step.t_emb,
            vd,
            4,
            t_dim,
        );
        let (ava_params, _) = adaln_host(
            store,
            "av_ca_audio_scale_shift_adaln_single",
            &step.t_emb,
            ad,
            4,
            t_dim,
        );
        let (a2v_gate, _) = adaln_host(
            store,
            "av_ca_a2v_gate_adaln_single",
            &step.t_emb,
            vd,
            1,
            t_dim,
        );
        let (v2a_gate, _) = adaln_host(
            store,
            "av_ca_v2a_gate_adaln_single",
            &step.t_emb,
            ad,
            1,
            t_dim,
        );

        let (v_cos, v_sin) = rope_tables(
            &step.video_positions,
            3,
            &VIDEO_MAX_POS,
            config.video.dim,
            config.video.heads,
        );
        let (a_cos, a_sin) = rope_tables(
            &step.audio_positions,
            1,
            &AUDIO_MAX_POS,
            config.audio.dim,
            config.audio.heads,
        );
        let (cv_cos, cv_sin) = rope_tables(
            &cross_positions(&step.video_positions),
            1,
            &CROSS_MAX_POS,
            config.cross_inner(),
            config.cross_heads,
        );
        let (ca_cos, ca_sin) = rope_tables(
            &step.audio_positions,
            1,
            &CROSS_MAX_POS,
            config.cross_inner(),
            config.cross_heads,
        );

        let mut video = linear_bias_host(
            store,
            "patchify_proj",
            &step.video_latent,
            latent,
            vd,
            v_seq,
        );
        let mut audio = linear_bias_host(
            store,
            "audio_patchify_proj",
            &step.audio_latent,
            latent,
            ad,
            a_seq,
        );

        for i in 0..config.layers as usize {
            let blk = format!("transformer_blocks.{i}");
            let stg_v = if step.perturb.skip_video_self.contains(&(i as i64)) {
                0.0
            } else {
                1.0
            };
            let stg_a = if step.perturb.skip_audio_self.contains(&(i as i64)) {
                0.0
            } else {
                1.0
            };
            let cross = if step.perturb.skip_av_cross { 0.0 } else { 1.0 };
            let v = mod_columns_host(
                &v_params,
                at(store, &format!("{blk}.scale_shift_table")),
                vd,
                9,
            );
            let a = mod_columns_host(
                &a_params,
                at(store, &format!("{blk}.audio_scale_shift_table")),
                ad,
                9,
            );
            let vp = mod_columns_host(
                &vp_params,
                at(store, &format!("{blk}.prompt_scale_shift_table")),
                vd,
                2,
            );
            let ap = mod_columns_host(
                &ap_params,
                at(store, &format!("{blk}.audio_prompt_scale_shift_table")),
                ad,
                2,
            );
            let avv = mod_columns_host(
                &avv_params,
                at(store, &format!("{blk}.scale_shift_table_a2v_ca_video")),
                vd,
                4,
            );
            let ava = mod_columns_host(
                &ava_params,
                at(store, &format!("{blk}.scale_shift_table_a2v_ca_audio")),
                ad,
                4,
            );
            let gate_a2v: Vec<f32> = (0..vd)
                .map(|d| {
                    a2v_gate[d]
                        + at(store, &format!("{blk}.scale_shift_table_a2v_ca_video"))[d + vd * 4]
                })
                .collect();
            let gate_v2a: Vec<f32> = (0..ad)
                .map(|d| {
                    v2a_gate[d]
                        + at(store, &format!("{blk}.scale_shift_table_a2v_ca_audio"))[d + ad * 4]
                })
                .collect();
            let gate = |x: &mut [f32], g: &[f32], dim: usize, seq: usize| {
                for s in 0..seq {
                    for d in 0..dim {
                        x[d + dim * s] *= g[d];
                    }
                }
            };

            let x = modulate_host(&rms_host(&video, None, vd, v_seq), &v[1], &v[0], vd, v_seq);
            let mut sa = attention_host(
                store,
                &format!("{blk}.attn1"),
                &x,
                &x,
                vd,
                vd,
                vd,
                vd,
                config.video.heads as usize,
                config.video.head_dim as usize,
                v_seq,
                v_seq,
                Some((&v_cos, &v_sin)),
                Some((&v_cos, &v_sin)),
                stg_v,
            );
            gate(&mut sa, &v[2], vd, v_seq);
            for (h, o) in video.iter_mut().zip(&sa) {
                *h += o;
            }

            let x = modulate_host(&rms_host(&audio, None, ad, a_seq), &a[1], &a[0], ad, a_seq);
            let mut sa = attention_host(
                store,
                &format!("{blk}.audio_attn1"),
                &x,
                &x,
                ad,
                ad,
                ad,
                ad,
                config.audio.heads as usize,
                config.audio.head_dim as usize,
                a_seq,
                a_seq,
                Some((&a_cos, &a_sin)),
                Some((&a_cos, &a_sin)),
                stg_a,
            );
            gate(&mut sa, &a[2], ad, a_seq);
            for (h, o) in audio.iter_mut().zip(&sa) {
                *h += o;
            }

            let x = modulate_host(&rms_host(&video, None, vd, v_seq), &v[7], &v[6], vd, v_seq);
            let text = modulate_host(&step.video_text, &vp[1], &vp[0], vd, t_seq);
            let mut ca = attention_host(
                store,
                &format!("{blk}.attn2"),
                &x,
                &text,
                vd,
                vd,
                vd,
                vd,
                config.video.heads as usize,
                config.video.head_dim as usize,
                v_seq,
                t_seq,
                None,
                None,
                1.0,
            );
            gate(&mut ca, &v[8], vd, v_seq);
            for (h, o) in video.iter_mut().zip(&ca) {
                *h += o;
            }

            let x = modulate_host(&rms_host(&audio, None, ad, a_seq), &a[7], &a[6], ad, a_seq);
            let text = modulate_host(&step.audio_text, &ap[1], &ap[0], ad, t_seq);
            let mut ca = attention_host(
                store,
                &format!("{blk}.audio_attn2"),
                &x,
                &text,
                ad,
                ad,
                ad,
                ad,
                config.audio.heads as usize,
                config.audio.head_dim as usize,
                a_seq,
                t_seq,
                None,
                None,
                1.0,
            );
            gate(&mut ca, &a[8], ad, a_seq);
            for (h, o) in audio.iter_mut().zip(&ca) {
                *h += o;
            }

            let video_n3 = rms_host(&video, None, vd, v_seq);
            let audio_n3 = rms_host(&audio, None, ad, a_seq);
            let q = modulate_host(&video_n3, &avv[0], &avv[1], vd, v_seq);
            let kv = modulate_host(&audio_n3, &ava[0], &ava[1], ad, a_seq);
            let mut a2v = attention_host(
                store,
                &format!("{blk}.audio_to_video_attn"),
                &q,
                &kv,
                vd,
                ad,
                ci,
                vd,
                config.cross_heads as usize,
                config.cross_head_dim as usize,
                v_seq,
                a_seq,
                Some((&cv_cos, &cv_sin)),
                Some((&ca_cos, &ca_sin)),
                1.0,
            );
            gate(&mut a2v, &gate_a2v, vd, v_seq);
            for o in a2v.iter_mut() {
                *o *= cross;
            }
            for (h, o) in video.iter_mut().zip(&a2v) {
                *h += o;
            }

            let q = modulate_host(&audio_n3, &ava[2], &ava[3], ad, a_seq);
            let kv = modulate_host(&video_n3, &avv[2], &avv[3], vd, v_seq);
            let mut v2a = attention_host(
                store,
                &format!("{blk}.video_to_audio_attn"),
                &q,
                &kv,
                ad,
                vd,
                ci,
                ad,
                config.cross_heads as usize,
                config.cross_head_dim as usize,
                a_seq,
                v_seq,
                Some((&ca_cos, &ca_sin)),
                Some((&cv_cos, &cv_sin)),
                1.0,
            );
            gate(&mut v2a, &gate_v2a, ad, a_seq);
            for o in v2a.iter_mut() {
                *o *= cross;
            }
            for (h, o) in audio.iter_mut().zip(&v2a) {
                *h += o;
            }

            let x = modulate_host(&rms_host(&video, None, vd, v_seq), &v[4], &v[3], vd, v_seq);
            let ff = linear_bias_host(
                store,
                &format!("{blk}.ff.net.0.proj"),
                &x,
                vd,
                vd * 4,
                v_seq,
            );
            let ff: Vec<f32> = ff.iter().map(|v| gelu_tanh_host(*v)).collect();
            let mut ff =
                linear_bias_host(store, &format!("{blk}.ff.net.2"), &ff, vd * 4, vd, v_seq);
            gate(&mut ff, &v[5], vd, v_seq);
            for (h, o) in video.iter_mut().zip(&ff) {
                *h += o;
            }

            let x = modulate_host(&rms_host(&audio, None, ad, a_seq), &a[4], &a[3], ad, a_seq);
            let ff = linear_bias_host(
                store,
                &format!("{blk}.audio_ff.net.0.proj"),
                &x,
                ad,
                ad * 4,
                a_seq,
            );
            let ff: Vec<f32> = ff.iter().map(|v| gelu_tanh_host(*v)).collect();
            let mut ff = linear_bias_host(
                store,
                &format!("{blk}.audio_ff.net.2"),
                &ff,
                ad * 4,
                ad,
                a_seq,
            );
            gate(&mut ff, &a[5], ad, a_seq);
            for (h, o) in audio.iter_mut().zip(&ff) {
                *h += o;
            }
        }

        let output = |x: &[f32],
                      embedded: &[f32],
                      table: &str,
                      proj: &str,
                      dim: usize,
                      seq: usize|
         -> Vec<f32> {
            let t = at(store, table);
            let shift: Vec<f32> = (0..dim).map(|d| t[d] + embedded[d]).collect();
            let scale: Vec<f32> = (0..dim).map(|d| t[d + dim] + embedded[d]).collect();
            let x = modulate_host(&layer_norm_host(x, dim, seq), &scale, &shift, dim, seq);
            linear_bias_host(store, proj, &x, dim, config.latent as usize, seq)
        };
        let video_out = output(
            &video,
            &v_embedded,
            "scale_shift_table",
            "proj_out",
            vd,
            v_seq,
        );
        let audio_out = output(
            &audio,
            &a_embedded,
            "audio_scale_shift_table",
            "audio_proj_out",
            ad,
            a_seq,
        );
        (video_out, audio_out)
    }

    fn lcg(seed: u32) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f32 / u32::MAX as f32) - 0.5
        }
    }

    fn make_step(
        config: &DitConfig,
        v_seq: i64,
        a_seq: i64,
        t_seq: i64,
        sigma: f32,
        seed: u32,
    ) -> DitStep {
        let mut next = lcg(seed);
        let mut fill = |count: usize| (0..count).map(|_| next()).collect::<Vec<f32>>();
        DitStep {
            video_latent: fill((config.latent * v_seq) as usize),
            audio_latent: fill((config.latent * a_seq) as usize),
            t_emb: crate::common::ops::sinusoidal_timestep(
                sigma * TIMESTEP_SCALE,
                config.t_dim as usize,
            ),
            video_text: fill((config.video.dim * t_seq) as usize),
            audio_text: fill((config.audio.dim * t_seq) as usize),
            video_positions: (0..v_seq)
                .flat_map(|i| {
                    [
                        (i % 3) as f32 * 5.25 + 0.25,
                        (i % 5) as f32 * 400.5 + 16.0,
                        (i % 7) as f32 * 290.75 + 32.0,
                    ]
                })
                .collect(),
            audio_positions: (0..a_seq).map(|i| i as f32 * 0.31 + 0.02).collect(),
            perturb: DitPerturb::default(),
        }
    }

    #[test]
    fn graph_matches_the_host_reference() {
        let config = DitConfig::smoke();
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-dit-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_smoke(&dir).unwrap();
        let dit = LtxDit::open(&[path]).unwrap();
        assert_eq!(dit.config(), config);

        let (v_seq, a_seq, t_seq) = (5i64, 4i64, 6i64);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut graph = dit.build_graph(&ctx, v_seq, a_seq, t_seq).unwrap();
        let store = smoke_store(&config);
        // Replay the same graph twice with different steps (denoise replay).
        for (sigma, seed) in [(0.7f32, 11u32), (0.31, 23)] {
            let step = make_step(&config, v_seq, a_seq, t_seq, sigma, seed);
            let (video, audio) = graph.forward(&step).unwrap();
            let (video_host, audio_host) = forward_host(
                &store,
                &config,
                &step,
                v_seq as usize,
                a_seq as usize,
                t_seq as usize,
            );
            assert_eq!(video.len(), video_host.len());
            assert_eq!(audio.len(), audio_host.len());
            let mut max_diff = 0f32;
            for (g, h) in video
                .iter()
                .chain(&audio)
                .zip(video_host.iter().chain(&audio_host))
            {
                max_diff = max_diff.max((g - h).abs());
            }
            assert!(
                max_diff < 2e-3,
                "sigma {sigma}: graph vs host max diff {max_diff}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn perturbation_gates_match_the_host_reference() {
        let config = DitConfig::smoke();
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-dit-ptb-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = write_smoke(&dir).unwrap();
        let dit = LtxDit::open(&[path]).unwrap();
        let (v_seq, a_seq, t_seq) = (5i64, 4i64, 6i64);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut graph = dit.build_graph(&ctx, v_seq, a_seq, t_seq).unwrap();
        let store = smoke_store(&config);
        for perturb in [
            DitPerturb {
                skip_video_self: vec![1],
                skip_audio_self: vec![0],
                skip_av_cross: false,
            },
            DitPerturb {
                skip_video_self: Vec::new(),
                skip_audio_self: Vec::new(),
                skip_av_cross: true,
            },
        ] {
            let mut step = make_step(&config, v_seq, a_seq, t_seq, 0.7, 11);
            step.perturb = perturb.clone();
            let (video, audio) = graph.forward(&step).unwrap();
            let (video_host, audio_host) = forward_host(
                &store,
                &config,
                &step,
                v_seq as usize,
                a_seq as usize,
                t_seq as usize,
            );
            let mut max_diff = 0f32;
            for (g, h) in video
                .iter()
                .chain(&audio)
                .zip(video_host.iter().chain(&audio_host))
            {
                max_diff = max_diff.max((g - h).abs());
            }
            assert!(
                max_diff < 2e-3,
                "perturb {perturb:?}: graph vs host max diff {max_diff}"
            );
        }
        // The gates change the output: a perturbed replay differs from the
        // unperturbed one.
        let plain = make_step(&config, v_seq, a_seq, t_seq, 0.7, 11);
        let (video, _) = graph.forward(&plain).unwrap();
        let mut skipped = plain;
        skipped.perturb.skip_av_cross = true;
        let (video_skipped, _) = graph.forward(&skipped).unwrap();
        assert!(
            video
                .iter()
                .zip(&video_skipped)
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "skip_av_cross did not change the video output"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_fails_loud_on_missing_blocks() {
        let dir = std::env::temp_dir().join(format!("danmo-ltx-dit-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.gguf");
        crate::weights::write_f32_gguf(&path, "dummy.weight", 1, 1, &[0.0]).unwrap();
        let err = match LtxDit::open(&[path]) {
            Err(err) => err,
            Ok(_) => panic!("accepted empty weights"),
        };
        assert!(err.to_string().contains("patchify_proj"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
