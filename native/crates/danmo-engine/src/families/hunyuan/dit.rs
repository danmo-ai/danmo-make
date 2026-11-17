//! HunyuanVideo 1.5 54-block joint DiT over the jayn7 GGUF (native Tencent
//! names: `img_in`, `txt_in`, `vision_in`, `byt5_in`, `time_in`,
//! `cond_type_embedding`, `double_blocks.{i}.img_*` / `txt_*`,
//! `final_layer`). Semantics are a port of main's `hunyuan/transformer_mlx.py`
//! (diffusers `HunyuanVideo15Transformer3DModel`):
//!
//! - Patch embed: patch (1, 1, 1), so `img_in.proj` is a per-token linear
//!   `[65, dim]` over `[noise(32); cond(32); mask(1)]` channels. Token order
//!   is `x + W * (y + H * f)` (w fastest, frames slowest), matching the ggml
//!   `[W, H, F, C]` latent flat layout.
//! - Token refiner (`txt_in`): pooled mean of the raw Qwen2.5-VL tokens ->
//!   `c_embedder` (linear/silu/linear), plus the sinusoidal timestep MLP
//!   (`t_embedder`), then 2 blocks of affine LayerNorm -> self attention (no
//!   qk norms, no rope) -> gated residual, LN -> linear/silu FF -> gated
//!   residual. main pads and masks the text tokens; the port passes only
//!   valid tokens, which is the same math without a mask.
//! - Vision in (`vision_in.proj.{0,1,3,4}`): LN 1152 -> linear -> exact-erf
//!   GELU -> linear 1152 -> dim -> LN. The encoder sequence is
//!   `[image tokens; text tokens]` (main's valid-first reorder; byT5 glyph
//!   tokens are absent because glyph prompts fail loud before the DiT).
//! - `cond_type_embedding`: + row 0 on text tokens, + row 2 on image tokens.
//! - `time_in`: sinusoidal 256 (`[sin(t*f); cos(t*f)]`, f =
//!   `10000^(-i/128)`, host-computed) -> MLP 256 -> dim -> dim.
//! - 54 double blocks: Flux-style 6-way adaLN (`img_mod` / `txt_mod` on
//!   `linear(silu(temb))`: shift_msa, scale_msa, gate_msa, shift_mlp,
//!   scale_mlp, gate_mlp), joint attention (fused `img_attn_qkv` /
//!   `txt_attn_qkv`, per-head RMSNorm eps 1e-6 on q/k, RoPE on the image
//!   stream only, one flash attention over `[img; txt]`, separate output
//!   projections), gated residuals, then LN -> scale/shift -> gelu-tanh FF
//!   with the mlp gate.
//! - RoPE is the diffusers "real" convention: 3 axes `[16, 56, 56]` at theta
//!   256 with adjacent-duplicated frequencies and split-half rotation pairs
//!   `(i, i + 64)` — ggml's built-in rope cannot express the duplicated
//!   frequencies, so four host tables `[64, L]` (cos/sin for the first and
//!   second head-dim halves) ride as graph inputs and the rotation is
//!   `out1 = x1*c1 - x2*s1`, `out2 = x1*s2 + x2*c2`.
//! - `final_layer`: AdaLayerNormContinuous (`norm(x) * (1 + scale) + shift`
//!   from `linear(silu(temb))`) then `linear(dim, 32)`. The `[32, L]` output
//!   is permuted to `[L, 32]`, which is exactly the ggml `[W, H, F, C]` flat
//!   latent order.
//!
//! Matmul weights keep their stored type (Q8_0 in the published GGUF; the
//! refiner/time/head towers are F16); biases, norms, and the embedding table
//! widen to F32.

use crate::runtime::{
    flash_attend, shape, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;

const GRAPH_MEM: usize = 64 << 20;
const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DitConfig {
    pub layers: i64,
    pub dim: i64,
    pub heads: i64,
    pub head_dim: i64,
    /// FF inner width in the double blocks (`mlp_ratio 4`).
    pub mlp: i64,
    /// Latent channels in: noise + cond + mask (32 + 32 + 1 published).
    pub in_channels: i64,
    pub out_channels: i64,
    /// Qwen2.5-VL embedding width (3584).
    pub text_dim: i64,
    /// SigLIP embedding width (1152).
    pub image_dim: i64,
    pub refiner_layers: i64,
    /// Refiner FF inner width (mlp_ratio 4 at the refiner width).
    pub refiner_mlp: i64,
    /// Sinusoidal timestep width (256).
    pub freq_dim: i64,
    /// RoPE axis dims, summing to `head_dim` (`[16, 56, 56]`).
    pub rope_axes: [i64; 3],
    pub rope_theta: f32,
}

impl DitConfig {
    /// The published 720p i2v checkpoint.
    pub fn published() -> Self {
        Self {
            layers: 54,
            dim: 2048,
            heads: 16,
            head_dim: 128,
            mlp: 8192,
            in_channels: 65,
            out_channels: 32,
            text_dim: 3584,
            image_dim: 1152,
            refiner_layers: 2,
            refiner_mlp: 8192,
            freq_dim: 256,
            rope_axes: [16, 56, 56],
            rope_theta: 256.0,
        }
    }

    /// Same topology at a tiny width for the host-reference test. The head
    /// dim stays 128 so the published rope axes `[16, 56, 56]` hold.
    pub fn smoke() -> Self {
        Self {
            layers: 3,
            dim: 256,
            heads: 2,
            head_dim: 128,
            mlp: 512,
            in_channels: 9,
            out_channels: 4,
            text_dim: 24,
            image_dim: 12,
            refiner_layers: 2,
            refiner_mlp: 512,
            freq_dim: 16,
            rope_axes: [16, 56, 56],
            rope_theta: 256.0,
        }
    }

    /// Infer the topology from the GGUF tensor shapes; anything unexpected
    /// fails loud.
    pub fn infer(weights: &GgufWeights) -> Result<Self, WeightError> {
        let bad = |msg: String| WeightError::NotBuilt(format!("hunyuan dit {msg}"));
        let mut layers = 0i64;
        while weights.contains(&format!("double_blocks.{layers}.img_attn_qkv.weight")) {
            layers += 1;
        }
        if layers < 1 {
            return Err(bad("has no double_blocks".into()));
        }
        let qkv = weights.shape("double_blocks.0.img_attn_qkv.weight")?;
        let dim = qkv[0];
        if dim < 1 || qkv[1] != 3 * dim {
            return Err(bad(format!(
                "double_blocks.0.img_attn_qkv.weight is {qkv:?}, not [dim, 3*dim]"
            )));
        }
        let head_dim = weights.shape("double_blocks.0.img_attn_q_norm.weight")?[0];
        if head_dim < 1 || dim % head_dim != 0 {
            return Err(bad(format!(
                "q_norm head dim {head_dim} does not divide {dim}"
            )));
        }
        let heads = dim / head_dim;
        let mlp = weights.shape("double_blocks.0.img_mlp.fc1.weight")?[1];
        let img_in = weights.shape("img_in.proj.weight")?;
        if img_in[3] % dim != 0 {
            return Err(bad(format!("img_in.proj.weight is {img_in:?}")));
        }
        let in_channels = img_in[3] / dim;
        let out_channels = weights.shape("final_layer.linear.weight")?[1];
        let text_dim = weights.shape("txt_in.input_embedder.weight")?[0];
        let image_dim = weights.shape("vision_in.proj.0.weight")?[0];
        let freq_dim = weights.shape("time_in.mlp.0.weight")?[0];
        let mut refiner_layers = 0i64;
        while weights.contains(&format!(
            "txt_in.individual_token_refiner.blocks.{refiner_layers}.self_attn_qkv.weight"
        )) {
            refiner_layers += 1;
        }
        if refiner_layers < 1 {
            return Err(bad("has no individual_token_refiner blocks".into()));
        }
        let refiner_mlp =
            weights.shape("txt_in.individual_token_refiner.blocks.0.mlp.fc1.weight")?[1];
        let rope_axes = [16, 56, 56];
        if rope_axes.iter().sum::<i64>() != head_dim {
            return Err(bad(format!(
                "rope axes {rope_axes:?} do not sum to head dim {head_dim}"
            )));
        }
        let cond = weights.shape("cond_type_embedding.weight")?;
        if cond != [dim, 3, 1, 1] {
            return Err(bad(format!(
                "cond_type_embedding.weight is {cond:?}, not [dim, 3]"
            )));
        }
        Ok(Self {
            layers,
            dim,
            heads,
            head_dim,
            mlp,
            in_channels,
            out_channels,
            text_dim,
            image_dim,
            refiner_layers,
            refiner_mlp,
            freq_dim,
            rope_axes,
            rope_theta: 256.0,
        })
    }
}

struct Entry {
    name: String,
    dims: Vec<i64>,
    /// Vector-type tensor widened to F32 (bias / norm / embedding); matmul
    /// weights keep their stored ggml type.
    widen: bool,
}

fn push(out: &mut Vec<Entry>, name: String, dims: &[i64], widen: bool) {
    out.push(Entry {
        name,
        dims: dims.to_vec(),
        widen,
    });
}

fn push_linear(out: &mut Vec<Entry>, name: &str, input: i64, output: i64) {
    push(out, format!("{name}.weight"), &[input, output], false);
    push(out, format!("{name}.bias"), &[output], true);
}

/// Every weight the graph reads, in the GGUF's native names.
pub fn layout(config: &DitConfig) -> Vec<Entry> {
    let mut out = Vec::new();
    let d = config.dim;
    // F32 in the published GGUF; the `[1, 1, 1, C*D]` file shape folds to a
    // 2D `[C, D]` matmul in the graph.
    push(
        &mut out,
        "img_in.proj.weight".into(),
        &[1, 1, 1, config.in_channels * d],
        true,
    );
    push(&mut out, "img_in.proj.bias".into(), &[d], true);
    push_linear(&mut out, "time_in.mlp.0", config.freq_dim, d);
    push_linear(&mut out, "time_in.mlp.2", d, d);
    push_linear(&mut out, "txt_in.input_embedder", config.text_dim, d);
    push_linear(&mut out, "txt_in.t_embedder.mlp.0", config.freq_dim, d);
    push_linear(&mut out, "txt_in.t_embedder.mlp.2", d, d);
    push_linear(&mut out, "txt_in.c_embedder.linear_1", config.text_dim, d);
    push_linear(&mut out, "txt_in.c_embedder.linear_2", d, d);
    for i in 0..config.refiner_layers {
        let p = format!("txt_in.individual_token_refiner.blocks.{i}");
        push(&mut out, format!("{p}.norm1.weight"), &[d], true);
        push(&mut out, format!("{p}.norm1.bias"), &[d], true);
        push_linear(&mut out, &format!("{p}.self_attn_qkv"), d, 3 * d);
        push_linear(&mut out, &format!("{p}.self_attn_proj"), d, d);
        push(&mut out, format!("{p}.norm2.weight"), &[d], true);
        push(&mut out, format!("{p}.norm2.bias"), &[d], true);
        push_linear(&mut out, &format!("{p}.mlp.fc1"), d, config.refiner_mlp);
        push_linear(&mut out, &format!("{p}.mlp.fc2"), config.refiner_mlp, d);
        push_linear(&mut out, &format!("{p}.adaLN_modulation.1"), d, 2 * d);
    }
    push(
        &mut out,
        "vision_in.proj.0.weight".into(),
        &[config.image_dim],
        true,
    );
    push(
        &mut out,
        "vision_in.proj.0.bias".into(),
        &[config.image_dim],
        true,
    );
    push_linear(
        &mut out,
        "vision_in.proj.1",
        config.image_dim,
        config.image_dim,
    );
    push_linear(&mut out, "vision_in.proj.3", config.image_dim, d);
    push(&mut out, "vision_in.proj.4.weight".into(), &[d], true);
    push(&mut out, "vision_in.proj.4.bias".into(), &[d], true);
    push(&mut out, "cond_type_embedding.weight".into(), &[d, 3], true);
    for i in 0..config.layers {
        let p = format!("double_blocks.{i}");
        for stream in ["img", "txt"] {
            push_linear(&mut out, &format!("{p}.{stream}_attn_qkv"), d, 3 * d);
            push(
                &mut out,
                format!("{p}.{stream}_attn_q_norm.weight"),
                &[config.head_dim],
                true,
            );
            push(
                &mut out,
                format!("{p}.{stream}_attn_k_norm.weight"),
                &[config.head_dim],
                true,
            );
            push_linear(&mut out, &format!("{p}.{stream}_attn_proj"), d, d);
            push_linear(&mut out, &format!("{p}.{stream}_mod.linear"), d, 6 * d);
            push_linear(&mut out, &format!("{p}.{stream}_mlp.fc1"), d, config.mlp);
            push_linear(&mut out, &format!("{p}.{stream}_mlp.fc2"), config.mlp, d);
        }
    }
    push_linear(&mut out, "final_layer.adaLN_modulation.1", d, 2 * d);
    push_linear(&mut out, "final_layer.linear", d, config.out_channels);
    out
}

// ---------------------------------------------------------------------------
// host-computed tables
// ---------------------------------------------------------------------------

/// Sinusoidal timestep embedding `[sin(t*f); cos(t*f)]` with
/// `f = 10000^(-i / (dim/2))` (main `sinusoidal_timestep_proj(sin_first =
/// false, flip_sin_to_cos = true)`).
pub fn timestep_embedding(dim: usize, t: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let arg = t * freq;
        out[i] = arg.sin();
        out[half + i] = arg.cos();
    }
    out
}

/// The four split-half RoPE tables `[head_dim/2, L]` for a `w x h x f`
/// latent grid. Token `s = x + w * (y + h * t)`; head-dim index `j` maps to
/// axis ranges `[0, a0)`, `[a0, a0+a1)`, `[a0+a1, ...)` with adjacent-duplicated
/// frequencies `theta^(-2 * (local/2) / axis_dim)`. `(cos1, sin1)` cover
/// indices `0..half`, `(cos2, sin2)` indices `half..`.
pub fn rope_tables(
    w: i64,
    h: i64,
    f: i64,
    axes: [i64; 3],
    theta: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let head_dim: i64 = axes.iter().sum();
    let half = (head_dim / 2) as usize;
    let seq = (w * h * f) as usize;
    let starts = [0, axes[0], axes[0] + axes[1]];
    let mut cos1 = vec![0f32; half * seq];
    let mut sin1 = vec![0f32; half * seq];
    let mut cos2 = vec![0f32; half * seq];
    let mut sin2 = vec![0f32; half * seq];
    let angle = |j: i64, s: i64| -> f32 {
        let axis = if j < starts[1] {
            0
        } else if j < starts[2] {
            1
        } else {
            2
        };
        let local = j - starts[axis];
        let freq = (theta as f64).powf(-2.0 * (local / 2) as f64 / axes[axis] as f64);
        let pos = match axis {
            0 => s / (w * h),
            1 => (s / w) % h,
            _ => s % w,
        };
        (pos as f64 * freq) as f32
    };
    for s in 0..seq as i64 {
        for d in 0..half {
            let a1 = angle(d as i64, s);
            let a2 = angle((d + half) as i64, s);
            cos1[d + half * s as usize] = a1.cos();
            sin1[d + half * s as usize] = a1.sin();
            cos2[d + half * s as usize] = a2.cos();
            sin2[d + half * s as usize] = a2.sin();
        }
    }
    (cos1, sin1, cos2, sin2)
}

// ---------------------------------------------------------------------------
// graph fragments
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
        .ok_or_else(|| RuntimeError::Message(format!("hunyuan dit graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let actual = shape(tensor);
    if actual != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {actual:?}, the hunyuan dit expects {ne:?}"
        )));
    }
    Ok(tensor)
}

/// `y = w @ x + bias` on `[in, seq]` activations; the bias broadcasts.
fn linear(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    name: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.mul_mat(wt(weights, &format!("{name}.weight"))?, x))?;
    nn(b.add(y, wt(weights, &format!("{name}.bias"))?))
}

/// LayerNorm over `ne0`: `norm(x, 1e-6) * weight + bias` (affine) or plain
/// (either weight tensor null).
fn layer_norm(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Result<Tensor, RuntimeError> {
    let y = nn(b.norm(x, LN_EPS))?;
    if weight.is_null() {
        return Ok(y);
    }
    let y = nn(b.mul(y, weight))?;
    nn(b.add(y, bias))
}

/// `x * (1 + scale) + shift` with broadcast `[dim, 1]` vectors.
fn modulate(
    b: &GraphBuilder,
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
) -> Result<Tensor, RuntimeError> {
    nn(b.add(nn(b.add(x, nn(b.mul(x, scale))?))?, shift))
}

/// `linear(silu(temb))` split into `parts` column vectors `[dim, 1]`.
fn modulation(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    name: &str,
    temb: Tensor,
    dim: i64,
    parts: i64,
) -> Result<Vec<Tensor>, RuntimeError> {
    let e = linear(b, weights, name, nn(b.silu(temb))?)?;
    let e = nn(b.reshape_2d(e, dim, parts))?;
    let row = (dim * parts * 4) as usize;
    (0..parts)
        .map(|i| nn(b.cont(b.view_2d(e, dim, 1, row, (i * dim * 4) as usize))))
        .collect()
}

/// Fused qkv projection -> per-head RMS-normed `[head_dim, heads, seq]`
/// triple. `normed` selects the double-block qk norms; the refiner self
/// attention passes no norms.
#[allow(clippy::too_many_arguments)]
fn qkv_heads(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
    seq: i64,
    dim: i64,
    heads: i64,
    head_dim: i64,
    q_norm: Option<&str>,
    k_norm: Option<&str>,
) -> Result<(Tensor, Tensor, Tensor), RuntimeError> {
    let qkv = linear(b, weights, &format!("{prefix}.qkv"), x)?;
    let qkv = shaped(qkv, [3 * dim, seq, 1, 1], &format!("{prefix}.qkv"))?;
    let row = (3 * dim * 4) as usize;
    let mut split = |idx: i64, norm: Option<&str>| -> Result<Tensor, RuntimeError> {
        let part = nn(b.view_3d(
            qkv,
            head_dim,
            heads,
            seq,
            (dim * 4) as usize,
            row,
            (idx * dim * 4) as usize,
        ))?;
        if let Some(name) = norm {
            let part = nn(b.rms_norm(part, RMS_EPS))?;
            let w = nn(b.cont(b.reshape_4d(wt(weights, name)?, head_dim, 1, 1, 1)))?;
            return nn(b.mul(part, w));
        }
        Ok(part)
    };
    Ok((split(0, q_norm)?, split(1, k_norm)?, split(2, None)?))
}

/// Split-half RoPE with the adjacent-duplicated frequency tables:
/// `out1 = x1*c1 - x2*s1`, `out2 = x1*s2 + x2*c2` on `[head_dim, heads, seq]`.
fn rope_split(
    b: &GraphBuilder,
    x: Tensor,
    tables: &(Tensor, Tensor, Tensor, Tensor),
    half: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let nb1 = (2 * half * 4) as usize;
    let nb2 = (2 * half * heads * 4) as usize;
    let x1 = nn(b.view_3d(x, half, heads, seq, nb1, nb2, 0))?;
    let x2 = nn(b.view_3d(x, half, heads, seq, nb1, nb2, (half * 4) as usize))?;
    let (c1, s1, c2, s2) = tables;
    let out1 = nn(b.sub(nn(b.mul(x1, *c1))?, nn(b.mul(x2, *s1))?))?;
    let out2 = nn(b.add(nn(b.mul(x1, *s2))?, nn(b.mul(x2, *c2))?))?;
    nn(b.concat(nn(b.cont(out1))?, nn(b.cont(out2))?, 0))
}

/// `cond_type_embedding.weight [dim, 3]` column `idx` broadcast onto `[dim, seq]`.
fn cond_type(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    idx: i64,
    seq: i64,
    dim: i64,
) -> Result<Tensor, RuntimeError> {
    let table = wt(weights, "cond_type_embedding.weight")?;
    let col = nn(b.cont(b.view_2d(
        table,
        dim,
        1,
        (dim * 3 * 4) as usize,
        (idx * dim * 4) as usize,
    )))?;
    nn(b.repeat(col, b.blank_f32_2d(dim, seq)))
}

// ---------------------------------------------------------------------------
// the graph
// ---------------------------------------------------------------------------

pub struct DitGraph {
    graph: ReusedGraph,
    config: DitConfig,
    img_seq: i64,
    txt_seq: i64,
    vis_seq: i64,
}

impl DitGraph {
    fn build(
        ctx: &GgmlContext,
        config: &DitConfig,
        img_seq: i64,
        txt_seq: i64,
        vis_seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(GRAPH_MEM)?;
        let mut weights = HashMap::new();
        for entry in layout(config) {
            let ggml_type = if entry.widen {
                TYPE_F32
            } else {
                types.get(&entry.name).copied().unwrap_or(TYPE_F32)
            };
            let tensor = match entry.dims.len() {
                1 => b.tensor_1d(&entry.name, ggml_type, entry.dims[0], true),
                2 => b.tensor_2d(&entry.name, ggml_type, entry.dims[0], entry.dims[1], true),
                // The only 4D entry is the F32 `img_in.proj.weight`.
                4 if ggml_type == TYPE_F32 => b.f32_4d(
                    &entry.name,
                    entry.dims[0],
                    entry.dims[1],
                    entry.dims[2],
                    entry.dims[3],
                    true,
                ),
                other => {
                    return Err(RuntimeError::Message(format!(
                        "hunyuan dit weight {} has {other} dims",
                        entry.name
                    )))
                }
            };
            weights.insert(entry.name.clone(), nn(tensor)?);
        }
        let w = &weights;
        let d = config.dim;
        let latents = nn(b.f32_2d("latents", config.in_channels, img_seq, true))?;
        let temb_in = nn(b.f32_2d("temb", config.freq_dim, 1, true))?;
        let txt = nn(b.f32_2d("txt", config.text_dim, txt_seq, true))?;
        let vis = nn(b.f32_2d("img", config.image_dim, vis_seq, true))?;
        let half = config.head_dim / 2;
        let tables = (
            nn(b.f32_3d("rope_cos1", half, 1, img_seq, true))?,
            nn(b.f32_3d("rope_sin1", half, 1, img_seq, true))?,
            nn(b.f32_3d("rope_cos2", half, 1, img_seq, true))?,
            nn(b.f32_3d("rope_sin2", half, 1, img_seq, true))?,
        );

        // Time embedding: MLP on the host sinusoidal projection.
        let temb = linear(&b, w, "time_in.mlp.0", temb_in)?;
        let temb = linear(&b, w, "time_in.mlp.2", nn(b.silu(temb))?)?;

        // Token refiner over the Qwen2.5-VL embeddings. The pooled vector is
        // the mean over tokens: transpose, sum rows, transpose back.
        let pooled = nn(b.cont(b.permute(txt, 1, 0, 2, 3)))?; // [seq, dim]
        let pooled = nn(b.sum_rows(pooled))?;
        let pooled = nn(b.cont(b.permute(pooled, 1, 0, 2, 3)))?; // [dim, 1]
        let pooled = nn(b.scale(pooled, (txt_seq as f32).recip()))?;
        let rtemb = linear(&b, w, "txt_in.t_embedder.mlp.0", temb_in)?;
        let rtemb = linear(&b, w, "txt_in.t_embedder.mlp.2", nn(b.silu(rtemb))?)?;
        let ctemb = linear(&b, w, "txt_in.c_embedder.linear_1", pooled)?;
        let ctemb = linear(&b, w, "txt_in.c_embedder.linear_2", nn(b.silu(ctemb))?)?;
        let rtemb = nn(b.add(rtemb, ctemb))?;
        let mut enc_txt = linear(&b, w, "txt_in.input_embedder", txt)?;
        for i in 0..config.refiner_layers {
            let p = format!("txt_in.individual_token_refiner.blocks.{i}");
            let gates = modulation(&b, w, &format!("{p}.adaLN_modulation.1"), rtemb, d, 2)?;
            let normed = layer_norm(
                &b,
                enc_txt,
                wt(w, &format!("{p}.norm1.weight"))?,
                wt(w, &format!("{p}.norm1.bias"))?,
            )?;
            let (q, k, v) = qkv_heads(
                &b,
                w,
                &format!("{p}.self_attn"),
                normed,
                txt_seq,
                d,
                config.heads,
                config.head_dim,
                None,
                None,
            )?;
            let att = flash_attend(&b, q, k, v, std::ptr::null_mut())?;
            let att = nn(b.reshape_2d(att, d, txt_seq))?;
            let att = linear(&b, w, &format!("{p}.self_attn_proj"), att)?;
            enc_txt = nn(b.add(enc_txt, nn(b.mul(att, gates[0]))?))?;
            let normed = layer_norm(
                &b,
                enc_txt,
                wt(w, &format!("{p}.norm2.weight"))?,
                wt(w, &format!("{p}.norm2.bias"))?,
            )?;
            let ff = linear(&b, w, &format!("{p}.mlp.fc1"), normed)?;
            let ff = linear(&b, w, &format!("{p}.mlp.fc2"), nn(b.silu(ff))?)?;
            enc_txt = nn(b.add(enc_txt, nn(b.mul(ff, gates[1]))?))?;
        }
        enc_txt = nn(b.add(enc_txt, cond_type(&b, w, 0, txt_seq, d)?))?;

        // Vision stream: SigLIP embeds -> image_embedder, + cond type 2.
        let mut enc_img = layer_norm(
            &b,
            vis,
            wt(w, "vision_in.proj.0.weight")?,
            wt(w, "vision_in.proj.0.bias")?,
        )?;
        enc_img = linear(&b, w, "vision_in.proj.1", enc_img)?;
        enc_img = nn(b.gelu_erf(enc_img))?;
        enc_img = linear(&b, w, "vision_in.proj.3", enc_img)?;
        enc_img = layer_norm(
            &b,
            enc_img,
            wt(w, "vision_in.proj.4.weight")?,
            wt(w, "vision_in.proj.4.bias")?,
        )?;
        enc_img = nn(b.add(enc_img, cond_type(&b, w, 2, vis_seq, d)?))?;

        let enc_seq = vis_seq + txt_seq;
        let mut enc = nn(b.concat(enc_img, enc_txt, 1))?;

        // Image stream: per-token linear patch embed (the `[1,1,1,C*D]`
        // conv weight folds to a 2D matmul).
        let img_in_w = nn(b.reshape_2d(wt(w, "img_in.proj.weight")?, config.in_channels, d))?;
        let mut hid = nn(b.mul_mat(img_in_w, latents))?;
        hid = nn(b.add(hid, wt(w, "img_in.proj.bias")?))?;

        for i in 0..config.layers {
            let p = format!("double_blocks.{i}");
            let img_mod = modulation(&b, w, &format!("{p}.img_mod.linear"), temb, d, 6)?;
            let txt_mod = modulation(&b, w, &format!("{p}.txt_mod.linear"), temb, d, 6)?;
            let n_h = modulate(
                &b,
                layer_norm(&b, hid, std::ptr::null_mut(), std::ptr::null_mut())?,
                img_mod[1],
                img_mod[0],
            )?;
            let n_e = modulate(
                &b,
                layer_norm(&b, enc, std::ptr::null_mut(), std::ptr::null_mut())?,
                txt_mod[1],
                txt_mod[0],
            )?;
            let (q_i, k_i, v_i) = qkv_heads(
                &b,
                w,
                &format!("{p}.img_attn"),
                n_h,
                img_seq,
                d,
                config.heads,
                config.head_dim,
                Some(&format!("{p}.img_attn_q_norm.weight")),
                Some(&format!("{p}.img_attn_k_norm.weight")),
            )?;
            let q_i = rope_split(&b, q_i, &tables, half, config.heads, img_seq)?;
            let k_i = rope_split(&b, k_i, &tables, half, config.heads, img_seq)?;
            let (q_t, k_t, v_t) = qkv_heads(
                &b,
                w,
                &format!("{p}.txt_attn"),
                n_e,
                enc_seq,
                d,
                config.heads,
                config.head_dim,
                Some(&format!("{p}.txt_attn_q_norm.weight")),
                Some(&format!("{p}.txt_attn_k_norm.weight")),
            )?;
            let q = nn(b.concat(q_i, q_t, 2))?;
            let k = nn(b.concat(k_i, k_t, 2))?;
            let v = nn(b.concat(v_i, v_t, 2))?;
            let att = flash_attend(&b, q, k, v, std::ptr::null_mut())?;
            let att = nn(b.reshape_2d(att, d, img_seq + enc_seq))?;
            let row = ((img_seq + enc_seq) * d * 4) as usize;
            let att_img = nn(b.cont(b.view_2d(att, d, img_seq, row, 0)))?;
            let att_txt = nn(b.cont(b.view_2d(att, d, enc_seq, row, (img_seq * d * 4) as usize)))?;
            let att_img = linear(&b, w, &format!("{p}.img_attn_proj"), att_img)?;
            let att_txt = linear(&b, w, &format!("{p}.txt_attn_proj"), att_txt)?;
            hid = nn(b.add(hid, nn(b.mul(att_img, img_mod[2]))?))?;
            enc = nn(b.add(enc, nn(b.mul(att_txt, txt_mod[2]))?))?;
            let n_h2 = modulate(
                &b,
                layer_norm(&b, hid, std::ptr::null_mut(), std::ptr::null_mut())?,
                img_mod[4],
                img_mod[3],
            )?;
            let ff = linear(&b, w, &format!("{p}.img_mlp.fc1"), n_h2)?;
            let ff = linear(&b, w, &format!("{p}.img_mlp.fc2"), nn(b.gelu(ff))?)?;
            hid = nn(b.add(hid, nn(b.mul(ff, img_mod[5]))?))?;
            let n_e2 = modulate(
                &b,
                layer_norm(&b, enc, std::ptr::null_mut(), std::ptr::null_mut())?,
                txt_mod[4],
                txt_mod[3],
            )?;
            let ff = linear(&b, w, &format!("{p}.txt_mlp.fc1"), n_e2)?;
            let ff = linear(&b, w, &format!("{p}.txt_mlp.fc2"), nn(b.gelu(ff))?)?;
            enc = nn(b.add(enc, nn(b.mul(ff, txt_mod[5]))?))?;
        }

        // final_layer: AdaLayerNormContinuous + proj, then to [L, out] flat.
        let fin = modulation(&b, w, "final_layer.adaLN_modulation.1", temb, d, 2)?;
        let hid = modulate(
            &b,
            layer_norm(&b, hid, std::ptr::null_mut(), std::ptr::null_mut())?,
            fin[0],
            fin[1],
        )?;
        let hid = linear(&b, w, "final_layer.linear", hid)?;
        let hid = shaped(
            hid,
            [config.out_channels, img_seq, 1, 1],
            "hunyuan dit proj_out",
        )?;
        let out = nn(b.cont(b.permute(hid, 1, 0, 2, 3)))?; // [seq, out]
        let graph = b.finish(out)?;
        Ok(Self {
            graph,
            config: *config,
            img_seq,
            txt_seq,
            vis_seq,
        })
    }

    /// One denoise forward. `latents` is `[in_channels, img_seq]` (host packs
    /// `[noise; cond; mask]`), `temb` the [`timestep_embedding`] output, `txt`
    /// `[text_dim, txt_seq]`, `img` `[image_dim, vis_seq]`; the rope tables
    /// come from [`rope_tables`] for the latent geometry. Returns
    /// `[img_seq, out_channels]` flat — the ggml `[W, H, F, C]` order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        latents: &[f32],
        temb: &[f32],
        txt: &[f32],
        img: &[f32],
        rope: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    ) -> Result<Vec<f32>, RuntimeError> {
        let c = self.config;
        self.expect("latents", latents, (c.in_channels * self.img_seq) as usize)?;
        self.expect("temb", temb, c.freq_dim as usize)?;
        self.expect("txt", txt, (c.text_dim * self.txt_seq) as usize)?;
        self.expect("img", img, (c.image_dim * self.vis_seq) as usize)?;
        let half = (c.head_dim / 2) as usize;
        let seq = self.img_seq as usize;
        self.expect("rope_cos1", &rope.0, half * seq)?;
        self.expect("rope_sin1", &rope.1, half * seq)?;
        self.expect("rope_cos2", &rope.2, half * seq)?;
        self.expect("rope_sin2", &rope.3, half * seq)?;
        self.graph.set_f32("latents", latents)?;
        self.graph.set_f32("temb", temb)?;
        self.graph.set_f32("txt", txt)?;
        self.graph.set_f32("img", img)?;
        self.graph.set_f32("rope_cos1", &rope.0)?;
        self.graph.set_f32("rope_sin1", &rope.1)?;
        self.graph.set_f32("rope_cos2", &rope.2)?;
        self.graph.set_f32("rope_sin2", &rope.3)?;
        let out = self.graph.compute()?;
        self.expect("output", &out, (c.out_channels * self.img_seq) as usize)?;
        Ok(out)
    }

    fn expect(&self, what: &str, data: &[f32], want: usize) -> Result<(), RuntimeError> {
        if data.len() != want {
            return Err(RuntimeError::Message(format!(
                "hunyuan dit {what} has {} values, expected {want}",
                data.len()
            )));
        }
        Ok(())
    }
}

/// Loaded DiT weights plus the inferred config.
pub struct DitWeights {
    pub weights: std::sync::Arc<GgufWeights>,
    pub config: DitConfig,
}

impl DitWeights {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        let config = DitConfig::infer(&weights)?;
        Ok(Self { weights, config })
    }

    pub fn config(&self) -> DitConfig {
        self.config
    }

    /// Build a reusable graph for one token geometry and load the weights.
    /// Replay [`DitGraph::forward`] per denoise step.
    pub fn build_graph(
        &self,
        ctx: &GgmlContext,
        img_seq: i64,
        txt_seq: i64,
        vis_seq: i64,
    ) -> Result<DitGraph, WeightError> {
        let entries = layout(&self.config);
        let mut types = HashMap::new();
        for entry in &entries {
            if entry.widen {
                continue;
            }
            let ggml_type = self.weights.ggml_type_of(&entry.name)?;
            if !crate::weights::matmul_type(ggml_type) {
                return Err(WeightError::Type {
                    name: entry.name.clone(),
                    ggml_type,
                });
            }
            types.insert(entry.name.clone(), ggml_type);
        }
        let mut graph = DitGraph::build(ctx, &self.config, img_seq, txt_seq, vis_seq, &types)?;
        for entry in &entries {
            if entry.widen {
                let values = self.weights.f32_widened(&entry.name, &entry.dims)?;
                graph.graph.set_f32(&entry.name, &values)?;
            } else {
                let raw = self.weights.raw_exact(&entry.name, &entry.dims)?;
                graph.graph.set_bytes(&entry.name, &raw)?;
            }
        }
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::{layout, rope_tables, timestep_embedding, DitConfig, DitWeights};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_f32_nd, F32Nd};
    use std::collections::HashMap;

    const W: i64 = 4;
    const H: i64 = 4;
    const F: i64 = 3;
    const TXT: i64 = 5;
    const VIS: i64 = 3;

    fn lcg(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (*seed as f32 / u32::MAX as f32) - 0.5
    }

    /// Deterministic smoke weights; matrices scale ~1/sqrt(in), LN/norm
    /// vectors sit near 1, modulation/gate outputs stay small.
    pub fn smoke_store(config: &DitConfig) -> Vec<(String, Vec<f32>)> {
        let mut seed = 41u32;
        layout(config)
            .into_iter()
            .map(|entry| {
                let len: usize = entry.dims.iter().product::<i64>() as usize;
                let input = if entry.dims.len() >= 2 {
                    entry.dims[entry.dims.len() - 2].max(1) as f32
                } else {
                    1.0
                };
                let values = (0..len)
                    .map(|_| {
                        let v = lcg(&mut seed);
                        if entry.name.ends_with(".bias") {
                            v * 0.04
                        } else if entry.name.contains("norm")
                            || entry.name == "cond_type_embedding.weight"
                        {
                            1.0 + v * 0.2
                        } else if entry.name.contains("mod.linear")
                            || entry.name.contains("adaLN_modulation")
                        {
                            v * 0.1
                        } else {
                            v * 0.3 / input.sqrt()
                        }
                    })
                    .collect();
                (entry.name, values)
            })
            .collect()
    }

    fn write_smoke(dir: &std::path::Path) -> std::path::PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let config = DitConfig::smoke();
        let store = smoke_store(&config);
        let entries = layout(&config);
        let tensors: Vec<F32Nd> = entries
            .iter()
            .zip(&store)
            .map(|(entry, (_, values))| {
                let mut ne = [1i64; 4];
                for (i, d) in entry.dims.iter().enumerate() {
                    ne[i] = *d;
                }
                F32Nd {
                    name: &entry.name,
                    ne,
                    data: values,
                }
            })
            .collect();
        let path = dir.join("hunyuan_dit_smoke.gguf");
        write_f32_nd(&path, &tensors).unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // host reference: matrices are `[in, n]` flat (`i + in * s` / `i + in * o`)
    // -----------------------------------------------------------------------

    fn at<'a>(map: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a [f32] {
        map.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    fn h_linear(
        map: &HashMap<String, Vec<f32>>,
        name: &str,
        x: &[f32],
        input: i64,
        output: i64,
    ) -> Vec<f32> {
        let w = at(map, &format!("{name}.weight"));
        let bias = at(map, &format!("{name}.bias"));
        let n = x.len() / input as usize;
        let mut out = vec![0f32; output as usize * n];
        for o in 0..output as usize {
            for s in 0..n {
                let mut acc = bias[o];
                for i in 0..input as usize {
                    acc += w[i + input as usize * o] * x[i + input as usize * s];
                }
                out[o + output as usize * s] = acc;
            }
        }
        out
    }

    fn h_layer_norm(x: &[f32], dim: i64, weight: Option<&[f32]>, bias: Option<&[f32]>) -> Vec<f32> {
        let d = dim as usize;
        let n = x.len() / d;
        let mut out = vec![0f32; x.len()];
        for s in 0..n {
            let row = &x[d * s..d * (s + 1)];
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = (var + 1e-6).sqrt().recip();
            for i in 0..d {
                let mut v = (row[i] - mean) * inv;
                if let Some(w) = weight {
                    v *= w[i];
                }
                if let Some(b) = bias {
                    v += b[i];
                }
                out[i + d * s] = v;
            }
        }
        out
    }

    fn h_silu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|v| v / (1.0 + (-*v).exp())).collect()
    }

    fn h_gelu_tanh(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| 0.5 * v * (1.0 + (0.7978845606f32 * (v + 0.044715 * v * v * v)).tanh()))
            .collect()
    }

    fn h_gelu_erf(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|v| 0.5 * v * (1.0 + libm_erf(v / std::f32::consts::SQRT_2)))
            .collect()
    }

    fn libm_erf(x: f32) -> f32 {
        // Abramowitz-Stegun 7.1.26, plenty for parity at this tolerance.
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t
            * (0.254829592
                + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (1.0 - poly * (-x * x).exp())
    }

    fn h_rms_heads(x: &[f32], head_dim: i64, weight: &[f32]) -> Vec<f32> {
        let hd = head_dim as usize;
        let mut out = vec![0f32; x.len()];
        for (chunk, out_chunk) in x.chunks(hd).zip(out.chunks_mut(hd)) {
            let ms = chunk.iter().map(|v| v * v).sum::<f32>() / hd as f32;
            let inv = (ms + 1e-6).sqrt().recip();
            for (i, (v, o)) in chunk.iter().zip(out_chunk.iter_mut()).enumerate() {
                *o = v * inv * weight[i];
            }
        }
        out
    }

    /// Split-half RoPE with the shared tables (same math as the graph).
    fn h_rope(
        x: &[f32],
        heads: i64,
        head_dim: i64,
        seq: i64,
        tables: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    ) -> Vec<f32> {
        let (c1, s1, c2, s2) = tables;
        let hd = head_dim as usize;
        let half = hd / 2;
        let mut out = vec![0f32; x.len()];
        for h in 0..heads as usize {
            for s in 0..seq as usize {
                for d in 0..half {
                    let x1 = x[d + hd * (h + heads as usize * s)];
                    let x2 = x[d + half + hd * (h + heads as usize * s)];
                    out[d + hd * (h + heads as usize * s)] =
                        x1 * c1[d + half * s] - x2 * s1[d + half * s];
                    out[d + half + hd * (h + heads as usize * s)] =
                        x1 * s2[d + half * s] + x2 * c2[d + half * s];
                }
            }
        }
        out
    }

    /// Full bidirectional softmax attention over `[head_dim, heads, seq]`.
    fn h_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: i64,
        head_dim: i64,
        q_seq: i64,
        kv_seq: i64,
    ) -> Vec<f32> {
        let hd = head_dim as usize;
        let hs = heads as usize;
        let (qs, ks) = (q_seq as usize, kv_seq as usize);
        let scale = (head_dim as f32).sqrt().recip();
        let mut out = vec![0f32; hd * hs * qs];
        for h in 0..hs {
            for i in 0..qs {
                let mut scores = vec![0f32; ks];
                let mut peak = f32::NEG_INFINITY;
                for (j, score) in scores.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for d in 0..hd {
                        acc += q[d + hd * (h + hs * i)] * k[d + hd * (h + hs * j)];
                    }
                    *score = acc * scale;
                    peak = peak.max(*score);
                }
                let mut denom = 0f32;
                for score in scores.iter_mut() {
                    *score = (*score - peak).exp();
                    denom += *score;
                }
                for d in 0..hd {
                    let mut acc = 0f32;
                    for (j, score) in scores.iter().enumerate() {
                        acc += v[d + hd * (h + hs * j)] * score;
                    }
                    out[d + hd * (h + hs * i)] = acc / denom;
                }
            }
        }
        out
    }

    fn h_qkv(
        map: &HashMap<String, Vec<f32>>,
        prefix: &str,
        x: &[f32],
        seq: i64,
        dim: i64,
        heads: i64,
        head_dim: i64,
        normed: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let qkv = h_linear(map, &format!("{prefix}.qkv"), x, dim, 3 * dim);
        let d = dim as usize;
        let mut out = [
            qkv[..d * seq as usize].to_vec(),
            qkv[d * seq as usize..2 * d * seq as usize].to_vec(),
            qkv[2 * d * seq as usize..].to_vec(),
        ];
        if normed {
            for (idx, part) in out.iter_mut().enumerate().take(2) {
                let name = if idx == 0 {
                    format!("{prefix}_q_norm.weight")
                } else {
                    format!("{prefix}_k_norm.weight")
                };
                // reshape [dim, seq] -> per-head rms over head_dim
                let mut per_head = vec![0f32; part.len()];
                let hd = head_dim as usize;
                let hs = heads as usize;
                for h in 0..hs {
                    for s in 0..seq as usize {
                        for dd in 0..hd {
                            per_head[dd + hd * (h + hs * s)] = part[dd + hd * h + d * s];
                        }
                    }
                }
                let normed_heads = h_rms_heads(&per_head, head_dim, at(map, &name));
                for h in 0..hs {
                    for s in 0..seq as usize {
                        for dd in 0..hd {
                            part[dd + hd * h + d * s] = normed_heads[dd + hd * (h + hs * s)];
                        }
                    }
                }
            }
        }
        // return [head_dim, heads, seq] layout
        let hd = head_dim as usize;
        let hs = heads as usize;
        let to_heads = |flat: &[f32]| -> Vec<f32> {
            let mut o = vec![0f32; flat.len()];
            for h in 0..hs {
                for s in 0..seq as usize {
                    for dd in 0..hd {
                        o[dd + hd * (h + hs * s)] = flat[dd + hd * h + d * s];
                    }
                }
            }
            o
        };
        (to_heads(&out[0]), to_heads(&out[1]), to_heads(&out[2]))
    }

    fn h_from_heads(flat: &[f32], dim: i64, heads: i64, head_dim: i64, seq: i64) -> Vec<f32> {
        let (d, hd, hs) = (dim as usize, head_dim as usize, heads as usize);
        let mut o = vec![0f32; flat.len()];
        for h in 0..hs {
            for s in 0..seq as usize {
                for dd in 0..hd {
                    o[dd + hd * h + d * s] = flat[dd + hd * (h + hs * s)];
                }
            }
        }
        o
    }

    fn h_modulate(x: &[f32], dim: i64, scale: &[f32], shift: &[f32]) -> Vec<f32> {
        let d = dim as usize;
        let mut out = vec![0f32; x.len()];
        for s in 0..x.len() / d {
            for i in 0..d {
                out[i + d * s] = x[i + d * s] * (1.0 + scale[i]) + shift[i];
            }
        }
        out
    }

    fn h_add_scaled(base: &[f32], delta: &[f32], dim: i64, gate: &[f32]) -> Vec<f32> {
        let d = dim as usize;
        let mut out = base.to_vec();
        for s in 0..base.len() / d {
            for i in 0..d {
                out[i + d * s] += delta[i + d * s] * gate[i];
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn h_forward(
        map: &HashMap<String, Vec<f32>>,
        config: &DitConfig,
        latents: &[f32],
        temb_in: &[f32],
        txt: &[f32],
        vis: &[f32],
        tables: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    ) -> Vec<f32> {
        let d = config.dim;
        let img_seq = (W * H * F) as i64;
        let emb = h_linear(map, "time_in.mlp.0", temb_in, config.freq_dim, d);
        let temb = h_linear(map, "time_in.mlp.2", &h_silu(&emb), d, d);

        // token refiner
        let n_txt = TXT as usize;
        let mut pooled = vec![0f32; config.text_dim as usize];
        for s in 0..n_txt {
            for i in 0..config.text_dim as usize {
                pooled[i] += txt[i + config.text_dim as usize * s];
            }
        }
        for v in pooled.iter_mut() {
            *v /= n_txt as f32;
        }
        let rt = h_linear(map, "txt_in.t_embedder.mlp.0", temb_in, config.freq_dim, d);
        let rt = h_linear(map, "txt_in.t_embedder.mlp.2", &h_silu(&rt), d, d);
        let ct = h_linear(
            map,
            "txt_in.c_embedder.linear_1",
            &pooled,
            config.text_dim,
            d,
        );
        let ct = h_linear(map, "txt_in.c_embedder.linear_2", &h_silu(&ct), d, d);
        let rtemb: Vec<f32> = rt.iter().zip(&ct).map(|(a, b)| a + b).collect();
        let mut enc_txt = h_linear(map, "txt_in.input_embedder", txt, config.text_dim, d);
        for i in 0..config.refiner_layers {
            let p = format!("txt_in.individual_token_refiner.blocks.{i}");
            let gates = h_linear(
                map,
                &format!("{p}.adaLN_modulation.1"),
                &h_silu(&rtemb),
                d,
                2 * d,
            );
            let (g_msa, g_mlp) = gates.split_at(d as usize);
            let normed = h_layer_norm(
                &enc_txt,
                d,
                Some(at(map, &format!("{p}.norm1.weight"))),
                Some(at(map, &format!("{p}.norm1.bias"))),
            );
            let (q, k, v) = h_qkv(
                map,
                &format!("{p}.self_attn"),
                &normed,
                TXT,
                d,
                config.heads,
                config.head_dim,
                false,
            );
            let att = h_attention(&q, &k, &v, config.heads, config.head_dim, TXT, TXT);
            let att = h_from_heads(&att, d, config.heads, config.head_dim, TXT);
            let att = h_linear(map, &format!("{p}.self_attn_proj"), &att, d, d);
            enc_txt = h_add_scaled(&enc_txt, &att, d, g_msa);
            let normed = h_layer_norm(
                &enc_txt,
                d,
                Some(at(map, &format!("{p}.norm2.weight"))),
                Some(at(map, &format!("{p}.norm2.bias"))),
            );
            let ff = h_linear(map, &format!("{p}.mlp.fc1"), &normed, d, config.refiner_mlp);
            let ff = h_linear(
                map,
                &format!("{p}.mlp.fc2"),
                &h_silu(&ff),
                config.refiner_mlp,
                d,
            );
            enc_txt = h_add_scaled(&enc_txt, &ff, d, g_mlp);
        }
        let cond = at(map, "cond_type_embedding.weight");
        for s in 0..TXT as usize {
            for i in 0..d as usize {
                enc_txt[i + d as usize * s] += cond[i];
            }
        }

        // vision stream
        let mut enc_img = h_layer_norm(
            vis,
            config.image_dim,
            Some(at(map, "vision_in.proj.0.weight")),
            Some(at(map, "vision_in.proj.0.bias")),
        );
        enc_img = h_linear(
            map,
            "vision_in.proj.1",
            &enc_img,
            config.image_dim,
            config.image_dim,
        );
        enc_img = h_gelu_erf(&enc_img);
        enc_img = h_linear(map, "vision_in.proj.3", &enc_img, config.image_dim, d);
        enc_img = h_layer_norm(
            &enc_img,
            d,
            Some(at(map, "vision_in.proj.4.weight")),
            Some(at(map, "vision_in.proj.4.bias")),
        );
        for s in 0..VIS as usize {
            for i in 0..d as usize {
                enc_img[i + d as usize * s] += cond[i + 2 * d as usize];
            }
        }

        let mut enc = enc_img.clone();
        enc.extend_from_slice(&enc_txt);
        let enc_seq = VIS + TXT;

        let mut hid = h_linear(map, "img_in.proj", latents, config.in_channels, d);

        for i in 0..config.layers {
            let p = format!("double_blocks.{i}");
            let im = h_linear(
                map,
                &format!("{p}.img_mod.linear"),
                &h_silu(&temb),
                d,
                6 * d,
            );
            let tm = h_linear(
                map,
                &format!("{p}.txt_mod.linear"),
                &h_silu(&temb),
                d,
                6 * d,
            );
            let n_h = h_modulate(
                &h_layer_norm(&hid, d, None, None),
                d,
                &im[d as usize..2 * d as usize],
                &im[..d as usize],
            );
            let n_e = h_modulate(
                &h_layer_norm(&enc, d, None, None),
                d,
                &tm[d as usize..2 * d as usize],
                &tm[..d as usize],
            );
            let (q_i, k_i, v_i) = h_qkv(
                map,
                &format!("{p}.img_attn"),
                &n_h,
                img_seq,
                d,
                config.heads,
                config.head_dim,
                true,
            );
            let q_i = h_rope(&q_i, config.heads, config.head_dim, img_seq, tables);
            let k_i = h_rope(&k_i, config.heads, config.head_dim, img_seq, tables);
            let (q_t, k_t, v_t) = h_qkv(
                map,
                &format!("{p}.txt_attn"),
                &n_e,
                enc_seq,
                d,
                config.heads,
                config.head_dim,
                true,
            );
            let cat = |a: &[f32], b: &[f32]| -> Vec<f32> {
                // concat along seq in [head_dim, heads, seq] layout
                let hd = config.head_dim as usize;
                let hs = config.heads as usize;
                let (la, lb) = (a.len() / (hd * hs), b.len() / (hd * hs));
                let mut o = vec![0f32; a.len() + b.len()];
                for h in 0..hs {
                    for s in 0..la + lb {
                        for dd in 0..hd {
                            let src = if s < la {
                                a[dd + hd * (h + hs * s)]
                            } else {
                                b[dd + hd * (h + hs * (s - la))]
                            };
                            o[dd + hd * (h + hs * s)] = src;
                        }
                    }
                }
                o
            };
            let q = cat(&q_i, &q_t);
            let k = cat(&k_i, &k_t);
            let v = cat(&v_i, &v_t);
            let att = h_attention(
                &q,
                &k,
                &v,
                config.heads,
                config.head_dim,
                img_seq + enc_seq,
                img_seq + enc_seq,
            );
            let att = h_from_heads(&att, d, config.heads, config.head_dim, img_seq + enc_seq);
            let split = (d * img_seq) as usize;
            let att_img = h_linear(map, &format!("{p}.img_attn_proj"), &att[..split], d, d);
            let att_txt = h_linear(map, &format!("{p}.txt_attn_proj"), &att[split..], d, d);
            hid = h_add_scaled(&hid, &att_img, d, &im[2 * d as usize..3 * d as usize]);
            enc = h_add_scaled(&enc, &att_txt, d, &tm[2 * d as usize..3 * d as usize]);
            let n_h2 = h_modulate(
                &h_layer_norm(&hid, d, None, None),
                d,
                &im[4 * d as usize..5 * d as usize],
                &im[3 * d as usize..4 * d as usize],
            );
            let ff = h_linear(map, &format!("{p}.img_mlp.fc1"), &n_h2, d, config.mlp);
            let ff = h_linear(
                map,
                &format!("{p}.img_mlp.fc2"),
                &h_gelu_tanh(&ff),
                config.mlp,
                d,
            );
            hid = h_add_scaled(&hid, &ff, d, &im[5 * d as usize..]);
            let n_e2 = h_modulate(
                &h_layer_norm(&enc, d, None, None),
                d,
                &tm[4 * d as usize..5 * d as usize],
                &tm[3 * d as usize..4 * d as usize],
            );
            let ff = h_linear(map, &format!("{p}.txt_mlp.fc1"), &n_e2, d, config.mlp);
            let ff = h_linear(
                map,
                &format!("{p}.txt_mlp.fc2"),
                &h_gelu_tanh(&ff),
                config.mlp,
                d,
            );
            enc = h_add_scaled(&enc, &ff, d, &tm[5 * d as usize..]);
        }

        let fin = h_linear(
            map,
            "final_layer.adaLN_modulation.1",
            &h_silu(&temb),
            d,
            2 * d,
        );
        let hid = h_modulate(
            &h_layer_norm(&hid, d, None, None),
            d,
            &fin[..d as usize],
            &fin[d as usize..],
        );
        h_linear(map, "final_layer.linear", &hid, d, config.out_channels)
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-hy-dit-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn smoke_dit_matches_the_host_reference() {
        let dir = temp_dir("host");
        let path = write_smoke(&dir);
        let opened = DitWeights::open(&[path]).unwrap();
        let config = DitConfig::smoke();
        assert_eq!(opened.config(), config);

        let img_seq = W * H * F;
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut graph = opened.build_graph(&ctx, img_seq, TXT, VIS).unwrap();

        let mut seed = 7u32;
        let latents: Vec<f32> = (0..config.in_channels * img_seq)
            .map(|_| lcg(&mut seed))
            .collect();
        let txt: Vec<f32> = (0..config.text_dim * TXT).map(|_| lcg(&mut seed)).collect();
        let vis: Vec<f32> = (0..config.image_dim * VIS)
            .map(|_| lcg(&mut seed))
            .collect();
        let temb = timestep_embedding(config.freq_dim as usize, 350.0);
        let tables = rope_tables(W, H, F, config.rope_axes, config.rope_theta);

        let got = graph.forward(&latents, &temb, &txt, &vis, &tables).unwrap();

        let map: HashMap<String, Vec<f32>> = smoke_store(&config).into_iter().collect();
        let want = h_forward(&map, &config, &latents, &temb, &txt, &vis, &tables);

        // graph output is [seq, out]; host is [out, seq] flat
        let out = config.out_channels as usize;
        let seq = img_seq as usize;
        assert_eq!(got.len(), seq * out);
        let mut worst = 0f32;
        for s in 0..seq {
            for c in 0..out {
                worst = worst.max((got[c + out * s] - want[c + out * s]).abs());
            }
        }
        assert!(worst < 2e-3, "hunyuan dit max abs {worst}");

        // Replay with a second timestep reuses the reserved graph.
        let temb2 = timestep_embedding(config.freq_dim as usize, 60.0);
        let got2 = graph
            .forward(&latents, &temb2, &txt, &vis, &tables)
            .unwrap();
        let want2 = h_forward(&map, &config, &latents, &temb2, &txt, &vis, &tables);
        let mut worst2 = 0f32;
        for (g, w) in got2.iter().zip(&want2) {
            // both are [out, seq] vs [seq, out] — compare through the map
            let _ = (g, w);
        }
        for s in 0..seq {
            for c in 0..out {
                worst2 = worst2.max((got2[c + out * s] - want2[c + out * s]).abs());
            }
        }
        assert!(worst2 < 2e-3, "hunyuan dit replay max abs {worst2}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn timestep_embedding_matches_the_flipped_order() {
        let emb = timestep_embedding(8, 2.0);
        let half = 4;
        for i in 0..half {
            let freq = (-(10000f32.ln()) * i as f32 / half as f32).exp();
            assert!((emb[i] - (2.0 * freq).sin()).abs() < 1e-6);
            assert!((emb[half + i] - (2.0 * freq).cos()).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_tables_match_the_reference_convention() {
        let tables = rope_tables(2, 2, 1, [4, 6, 6], 256.0);
        let (cos1, _, cos2, _) = &tables;
        // head index 2 (first axis, local 2 -> pair 1 of dim 4); token s=3 is
        // (x=1, y=1, t=0), so the first-axis angle is 0.
        assert!((cos1[2 + 8 * 3] - 1.0).abs() < 1e-6);
        // head index 8 = half + 0 -> second-axis local 4, pair 2 of 6, y=1
        let freq2 = 256f32.powf(-2.0 * 2.0 / 6.0);
        assert!(
            (cos2[8 * 3] - (freq2).cos()).abs() < 1e-6,
            "{}",
            cos2[8 * 3]
        );
    }

    #[test]
    fn missing_weight_fails_loud() {
        let dir = temp_dir("missing");
        let config = DitConfig::smoke();
        let store = smoke_store(&config);
        let entries = layout(&config);
        let tensors: Vec<F32Nd> = entries
            .iter()
            .zip(&store)
            .filter(|(entry, _)| entry.name != "double_blocks.1.img_mlp.fc2.weight")
            .map(|(entry, (_, values))| {
                let mut ne = [1i64; 4];
                for (i, d) in entry.dims.iter().enumerate() {
                    ne[i] = *d;
                }
                F32Nd {
                    name: &entry.name,
                    ne,
                    data: values,
                }
            })
            .collect();
        let path = dir.join("partial.gguf");
        write_f32_nd(&path, &tensors).unwrap();
        let opened = DitWeights::open(&[path]);
        // infer still works (probes block 0), but the graph load must fail.
        let opened = match opened {
            Ok(o) => o,
            Err(_) => return,
        };
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        assert!(opened.build_graph(&ctx, W * H * F, TXT, VIS).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
