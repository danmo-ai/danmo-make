//! Wan DiT GGUF layout. Tensor names are the published `wan` arch names:
//! `blocks.N.{self_attn,cross_attn}.{q,k,v,o}` with full-dim RMS `norm_q` /
//! `norm_k`, `blocks.N.ffn.{0,2}` GELU-tanh MLP, `blocks.N.modulation` the six
//! adaLN tables, and a 5D `patch_embedding.weight` conv stored as
//! `[kw, kh, kt, in, dim]` (folded to 4D on read). The head is `head.head` +
//! `head.modulation`; `norm1` / `norm2` carry no weights. The GGUF has no
//! arch metadata, so the config is inferred from tensor shapes; an unknown
//! layout fails, it is never guessed.

use crate::weights::{GgufWeights, WeightError};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DitConfig {
    pub dim: i64,
    pub depth: usize,
    pub heads: i64,
    pub ffn_dim: i64,
    pub in_dim: i64,
    pub text_dim: i64,
    /// (t, h, w) patch; TI2V-5B is (1, 2, 2).
    pub patch: (i64, i64, i64),
}

impl DitConfig {
    /// Sinusoidal timestep width (`time_embedding.0` input).
    pub const FREQ_DIM: i64 = 256;
    /// umT5 captions are padded to this length.
    pub const TEXT_LEN: i64 = 512;
    /// Both the 5B (24 heads) and 14B (40 heads) Wan DiTs use 128-wide heads.
    pub const HEAD_DIM: i64 = 128;

    pub fn infer(w: &GgufWeights) -> Result<Self, WeightError> {
        let patch_emb = w.shape("patch_embedding.weight")?;
        let patch = (patch_emb[2], patch_emb[1], patch_emb[0]);
        // The 5D conv `[kw, kh, kt, in, dim]` folds into ne3 = in * dim;
        // `text_embedding.0` is Linear(text_dim -> dim) and pins dim down.
        let text = w.shape("text_embedding.0.weight")?;
        let text_dim = text[0];
        let dim = text[1];
        let folded = patch_emb[3];
        if dim <= 0 || dim % Self::HEAD_DIM != 0 || folded % dim != 0 {
            return Err(WeightError::NotBuilt(format!(
                "wan patch_embedding.weight is {patch_emb:?}, dim {dim} head_dim {}",
                Self::HEAD_DIM
            )));
        }
        let in_dim = folded / dim;
        let mut depth = 0usize;
        for name in w.names() {
            if let Some(rest) = name.strip_prefix("blocks.") {
                if let Some((index, _)) = rest.split_once('.') {
                    if let Ok(index) = index.parse::<usize>() {
                        depth = depth.max(index + 1);
                    }
                }
            }
        }
        if depth == 0 {
            return Err(WeightError::NotBuilt(
                "wan dit has no blocks.* tensors".into(),
            ));
        }
        let ffn = w.shape("blocks.0.ffn.0.weight")?;
        if ffn[0] != dim {
            return Err(WeightError::NotBuilt(format!(
                "wan blocks.0.ffn.0.weight is {ffn:?}, dim {dim}"
            )));
        }
        let text_dim = w.shape("text_embedding.0.weight")?[0];
        for required in [
            "blocks.0.modulation",
            "blocks.0.norm3.weight",
            "blocks.0.self_attn.norm_q.weight",
            "time_projection.1.weight",
            "head.head.weight",
            "head.modulation",
        ] {
            if !w.contains(required) {
                return Err(WeightError::NotBuilt(format!(
                    "wan dit is missing {required}"
                )));
            }
        }
        let modulation = w.shape("blocks.0.modulation")?;
        if modulation[0] != dim || modulation[1] != 6 {
            return Err(WeightError::NotBuilt(format!(
                "wan blocks.0.modulation is {modulation:?}, expected [dim, 6]"
            )));
        }
        if w.shape("time_projection.1.weight")?[1] != 6 * dim {
            return Err(WeightError::NotBuilt(
                "wan time_projection.1 is not 6 * dim".into(),
            ));
        }
        let head = w.shape("head.head.weight")?;
        if head[0] != dim || head[1] != in_dim * patch.0 * patch.1 * patch.2 {
            return Err(WeightError::NotBuilt(format!(
                "wan head.head is {head:?}, dim {dim} in_dim {in_dim} patch {patch:?}"
            )));
        }
        Ok(Self {
            dim,
            depth,
            heads: dim / Self::HEAD_DIM,
            ffn_dim: ffn[1],
            in_dim,
            text_dim,
            patch,
        })
    }
}

// ---- graph ----

use crate::runtime::{flash_attend, rope_nd, GraphBuilder, RuntimeError, Tensor};
use crate::weights::LoraFactor;
use std::collections::HashMap;

/// LayerNorm and qk RMSNorm eps (main `WanConfig.eps`).
pub const LN_EPS: f32 = 1e-6;
/// Factorized RoPE theta (`factorized_rope_concat_params` default).
pub const ROPE_THETA: f32 = 10000.0;

impl DitConfig {
    /// `in_dim * pt * ph * pw`, the patch embedding row width.
    pub fn patch_dim(&self) -> i64 {
        self.in_dim * self.patch.0 * self.patch.1 * self.patch.2
    }

    /// Factorized RoPE axis widths `[t, h, w]`; they sum to the head dim.
    pub fn rope_dims(&self) -> [i64; 3] {
        let d = Self::HEAD_DIM;
        let side = 2 * (d / 6);
        [d - 2 * side, side, side]
    }
}

/// Bias, norm, and modulation vectors stay F32 in the graph; the published
/// GGUF stores them F16 and the loader widens them. Everything else is a
/// matmul weight and keeps the stored ggml type.
pub fn vector_param(name: &str) -> bool {
    name.ends_with(".bias") || name.contains("norm") || name.ends_with("modulation")
}

pub fn weight_layout(config: DitConfig) -> Vec<(String, i64, i64)> {
    let dim = config.dim;
    let mut names = vec![
        ("patch_embedding.weight".into(), config.patch_dim(), dim),
        ("patch_embedding.bias".into(), dim, 1),
        ("text_embedding.0.weight".into(), config.text_dim, dim),
        ("text_embedding.0.bias".into(), dim, 1),
        ("text_embedding.2.weight".into(), dim, dim),
        ("text_embedding.2.bias".into(), dim, 1),
        ("time_embedding.0.weight".into(), DitConfig::FREQ_DIM, dim),
        ("time_embedding.0.bias".into(), dim, 1),
        ("time_embedding.2.weight".into(), dim, dim),
        ("time_embedding.2.bias".into(), dim, 1),
        ("time_projection.1.weight".into(), dim, 6 * dim),
        ("time_projection.1.bias".into(), 6 * dim, 1),
        ("head.head.weight".into(), dim, config.patch_dim()),
        ("head.head.bias".into(), config.patch_dim(), 1),
        ("head.modulation".into(), dim, 2),
    ];
    for i in 0..config.depth {
        let prefix = format!("blocks.{i}");
        names.push((format!("{prefix}.modulation"), dim, 6));
        names.push((format!("{prefix}.norm3.weight"), dim, 1));
        names.push((format!("{prefix}.norm3.bias"), dim, 1));
        for part in ["self_attn", "cross_attn"] {
            for leaf in ["q", "k", "v", "o"] {
                names.push((format!("{prefix}.{part}.{leaf}.weight"), dim, dim));
                names.push((format!("{prefix}.{part}.{leaf}.bias"), dim, 1));
            }
            names.push((format!("{prefix}.{part}.norm_q.weight"), dim, 1));
            names.push((format!("{prefix}.{part}.norm_k.weight"), dim, 1));
        }
        names.push((format!("{prefix}.ffn.0.weight"), dim, config.ffn_dim));
        names.push((format!("{prefix}.ffn.0.bias"), config.ffn_dim, 1));
        names.push((format!("{prefix}.ffn.2.weight"), config.ffn_dim, dim));
        names.push((format!("{prefix}.ffn.2.bias"), dim, 1));
    }
    names
}

/// One Wan DiT forward. Inputs: `tokens` `[patch_dim, seq]` (host-patchified
/// latent), `temb` `[freq_dim, 1]` (host sinusoidal embedding), `txt`
/// `[text_dim, 512]` (padded umT5 states), and the i32 RoPE position vectors
/// `pos_t` / `pos_h` / `pos_w` `[seq]`. The text embedding MLP and per-block
/// cross K/V run in-graph every step; at video token counts they are noise
/// next to self attention. Output: head tokens `[patch_dim, seq]`.
pub fn build_dit(
    b: &mut GraphBuilder,
    config: DitConfig,
    seq: i64,
    types: &HashMap<String, u32>,
    loras: &[LoraFactor],
) -> Result<Tensor, RuntimeError> {
    let dim = config.dim;
    if dim < 1 || dim % DitConfig::HEAD_DIM != 0 || seq < 1 {
        return Err(RuntimeError::Message(format!(
            "wan dit dim {dim} seq {seq} are not usable"
        )));
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
    crate::weights::bind_lora_factors(b, &weights, loras)?;
    let tokens = nn(b.f32_2d("tokens", config.patch_dim(), seq, true))?;
    let temb = nn(b.f32_2d("temb", DitConfig::FREQ_DIM, 1, true))?;
    let txt = nn(b.f32_2d("txt", config.text_dim, DitConfig::TEXT_LEN, true))?;
    let mut pos = [std::ptr::null_mut(); 3];
    for (axis, slot) in pos.iter_mut().enumerate() {
        *slot = nn(b.i32_1d(&format!("pos_{axis}"), seq, true))?;
    }
    let mut x = add_bias(
        b,
        linear(b, tokens, wt(&weights, "patch_embedding.weight")?),
        wt(&weights, "patch_embedding.bias")?,
    )?;
    let t_hidden = add_bias(
        b,
        linear(b, temb, wt(&weights, "time_embedding.0.weight")?),
        wt(&weights, "time_embedding.0.bias")?,
    )?;
    let e = add_bias(
        b,
        linear(
            b,
            nn(b.silu(t_hidden))?,
            wt(&weights, "time_embedding.2.weight")?,
        ),
        wt(&weights, "time_embedding.2.bias")?,
    )?;
    let e0 = add_bias(
        b,
        linear(b, nn(b.silu(e))?, wt(&weights, "time_projection.1.weight")?),
        wt(&weights, "time_projection.1.bias")?,
    )?;
    let e0 = nn(b.reshape_2d(e0, dim, 6))?;
    // text_embedding: Linear -> GELU(tanh) -> Linear, then it stays fixed.
    let context = add_bias(
        b,
        linear(b, txt, wt(&weights, "text_embedding.0.weight")?),
        wt(&weights, "text_embedding.0.bias")?,
    )?;
    let context = nn(b.gelu(context))?;
    let context = add_bias(
        b,
        linear(b, context, wt(&weights, "text_embedding.2.weight")?),
        wt(&weights, "text_embedding.2.bias")?,
    )?;
    let dims = config.rope_dims();
    for i in 0..config.depth {
        x = block(b, &weights, i, x, e0, context, &pos, dims, config, seq)?;
    }
    let head_mod = wt(&weights, "head.modulation")?;
    let shift = nn(b.add(column(b, head_mod, dim, 0)?, e))?;
    let scale = nn(b.add(column(b, head_mod, dim, 1)?, e))?;
    let normed = nn(b.norm(x, LN_EPS))?;
    let modulated = scale_shift(b, normed, scale, shift)?;
    add_bias(
        b,
        linear(b, modulated, wt(&weights, "head.head.weight")?),
        wt(&weights, "head.head.bias")?,
    )
}

fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    index: usize,
    x: Tensor,
    e0: Tensor,
    context: Tensor,
    pos: &[Tensor; 3],
    dims: [i64; 3],
    config: DitConfig,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let dim = config.dim;
    let prefix = format!("blocks.{index}");
    let modulation = nn(b.add(wt(weights, &format!("{prefix}.modulation"))?, e0))?;
    let e0v = column(b, modulation, dim, 0)?;
    let e1v = column(b, modulation, dim, 1)?;
    let e2v = column(b, modulation, dim, 2)?;
    let e3v = column(b, modulation, dim, 3)?;
    let e4v = column(b, modulation, dim, 4)?;
    let e5v = column(b, modulation, dim, 5)?;
    let normed = scale_shift(b, nn(b.norm(x, LN_EPS))?, e1v, e0v)?;
    let y = attention(
        b,
        weights,
        &format!("{prefix}.self_attn"),
        normed,
        normed,
        true,
        pos,
        dims,
        config,
        seq,
        seq,
    )?;
    let x = gate_add(b, x, e2v, y)?;
    let normed = affine_norm(
        b,
        nn(b.norm(x, LN_EPS))?,
        wt(weights, &format!("{prefix}.norm3.weight"))?,
        wt(&weights, &format!("{prefix}.norm3.bias"))?,
    )?;
    let y = attention(
        b,
        weights,
        &format!("{prefix}.cross_attn"),
        normed,
        context,
        false,
        pos,
        dims,
        config,
        seq,
        DitConfig::TEXT_LEN,
    )?;
    let x = nn(b.add(x, y))?;
    let normed = scale_shift(b, nn(b.norm(x, LN_EPS))?, e4v, e3v)?;
    let hidden = add_bias(
        b,
        linear(b, normed, wt(weights, &format!("{prefix}.ffn.0.weight"))?),
        wt(weights, &format!("{prefix}.ffn.0.bias"))?,
    )?;
    let y = add_bias(
        b,
        linear(
            b,
            nn(b.gelu(hidden))?,
            wt(weights, &format!("{prefix}.ffn.2.weight"))?,
        ),
        wt(weights, &format!("{prefix}.ffn.2.bias"))?,
    )?;
    gate_add(b, x, e5v, y)
}

/// `x` is `[dim, seq]`; with `self_rope` the k side is `x` itself plus the
/// factorized RoPE, otherwise it is the fixed text context (no RoPE).
fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    x: Tensor,
    kv: Tensor,
    self_rope: bool,
    pos: &[Tensor; 3],
    dims: [i64; 3],
    config: DitConfig,
    seq: i64,
    kv_seq: i64,
) -> Result<Tensor, RuntimeError> {
    let dim = config.dim;
    let head = DitConfig::HEAD_DIM;
    let heads = config.heads;
    let q = add_bias(
        b,
        linear(b, x, wt(weights, &format!("{prefix}.q.weight"))?),
        wt(weights, &format!("{prefix}.q.bias"))?,
    )?;
    let q = rms(b, q, wt(weights, &format!("{prefix}.norm_q.weight"))?)?;
    let k = add_bias(
        b,
        linear(b, kv, wt(weights, &format!("{prefix}.k.weight"))?),
        wt(weights, &format!("{prefix}.k.bias"))?,
    )?;
    let k = rms(b, k, wt(weights, &format!("{prefix}.norm_k.weight"))?)?;
    let v = add_bias(
        b,
        linear(b, kv, wt(weights, &format!("{prefix}.v.weight"))?),
        wt(weights, &format!("{prefix}.v.bias"))?,
    )?;
    let q = nn(b.reshape_3d(q, head, heads, seq))?;
    let k = nn(b.reshape_3d(k, head, heads, kv_seq))?;
    let v = nn(b.reshape_3d(v, head, heads, kv_seq))?;
    let (q, k) = if self_rope {
        (
            rope_nd(b, q, pos, &dims, ROPE_THETA)?,
            rope_nd(b, k, pos, &dims, ROPE_THETA)?,
        )
    } else {
        (q, k)
    };
    let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(out, dim, seq))?;
    add_bias(
        b,
        linear(b, mixed, wt(weights, &format!("{prefix}.o.weight"))?),
        wt(weights, &format!("{prefix}.o.bias"))?,
    )
}

/// Full-dim RMSNorm: `x / sqrt(mean(x^2) + eps) * weight` over `ne0 = dim`.
fn rms(b: &GraphBuilder, x: Tensor, weight: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, LN_EPS))?, nn(b.repeat(weight, x))?))
}

/// LayerNorm affine: `norm(x) * weight + bias`.
fn affine_norm(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Result<Tensor, RuntimeError> {
    let scaled = nn(b.mul(x, nn(b.repeat(weight, x))?))?;
    nn(b.add(scaled, nn(b.repeat(bias, x))?))
}

/// `x * (1 + scale) + shift` with `[dim, 1]` vectors.
fn scale_shift(
    b: &GraphBuilder,
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
) -> Result<Tensor, RuntimeError> {
    let scaled = nn(b.mul(x, nn(b.repeat(scale, x))?))?;
    let x = nn(b.add(x, scaled))?;
    nn(b.add(x, nn(b.repeat(shift, x))?))
}

/// `x + y * gate` with a `[dim, 1]` gate.
fn gate_add(b: &GraphBuilder, x: Tensor, gate: Tensor, y: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, nn(b.mul(y, nn(b.repeat(gate, y))?))?))
}

/// Column `j` of a `[dim, n]` table as a contiguous `[dim, 1]`.
fn column(b: &GraphBuilder, table: Tensor, dim: i64, j: i64) -> Result<Tensor, RuntimeError> {
    let bytes = std::mem::size_of::<f32>();
    let view = b.view_2d(
        table,
        dim,
        1,
        (dim as usize) * bytes,
        (j as usize) * (dim as usize) * bytes,
    );
    nn(b.cont(view))
}

fn linear(b: &GraphBuilder, x: Tensor, w: Tensor) -> Tensor {
    b.project(w, x)
}

fn add_bias(b: &GraphBuilder, x: Tensor, bias: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, nn(b.repeat(bias, x))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("wan dit graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

// ---- host helpers (session side) ----

/// `sinusoidal_embedding_1d(dim, t)`: `[cos(t*f); sin(t)*f)]` with
/// `f = 10000^(-i / (dim/2))` — the shared `ops::sinusoidal_timestep`.
pub fn sinusoidal_embedding(dim: usize, t: f32) -> Vec<f32> {
    crate::common::ops::sinusoidal_timestep(t, dim)
}

// ---- smoke writer (session smoke bundle) ----

fn smoke_values(name: &str, n: usize) -> Vec<f32> {
    if name.contains("norm") && name.ends_with(".weight") {
        return (0..n).map(|i| 1.0 + (i % 3) as f32 * 0.01).collect();
    }
    if name.ends_with("modulation") {
        return (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.01).collect();
    }
    if name.ends_with(".bias") {
        return (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.005).collect();
    }
    (0..n)
        .map(|i| (((i as f32) * 0.61 + 0.3).sin()) * 0.02)
        .collect()
}

/// Write a deterministic smoke DiT GGUF at `config`, skipping `skip` when
/// set (the missing-tensor test uses it). `patch_embedding.weight` is the
/// folded 4D `[pw, ph, pt, in*dim]` of the published 5D conv.
pub fn write_smoke(
    dir: &std::path::Path,
    config: DitConfig,
    skip: Option<&str>,
) -> Result<std::path::PathBuf, WeightError> {
    use crate::weights::{write_f32_nd, F32Nd};
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let mut tensors: Vec<(String, [i64; 4], Vec<f32>)> = Vec::new();
    for (name, ne0, ne1) in weight_layout(config) {
        if Some(name.as_str()) == skip {
            continue;
        }
        let ne = if name == "patch_embedding.weight" {
            [
                config.patch.2,
                config.patch.1,
                config.patch.0,
                config.in_dim * config.dim,
            ]
        } else {
            [ne0, ne1, 1, 1]
        };
        let n = ne.iter().product::<i64>() as usize;
        let data = smoke_values(&name, n);
        tensors.push((name, ne, data));
    }
    let views: Vec<F32Nd> = tensors
        .iter()
        .map(|(name, ne, data)| F32Nd {
            name: name.as_str(),
            ne: *ne,
            data: data.as_slice(),
        })
        .collect();
    let path = dir.join("wan-dit.gguf");
    write_f32_nd(&path, &views)?;
    Ok(path)
}

/// Per-token RoPE positions `[t, h, w]` for a patch grid `(f, h, w)`.
pub fn rope_positions(grid: (i64, i64, i64)) -> [Vec<i32>; 3] {
    let (f, h, w) = grid;
    let mut pos = [Vec::new(), Vec::new(), Vec::new()];
    for tf in 0..f {
        for th in 0..h {
            for tw in 0..w {
                pos[0].push(tf as i32);
                pos[1].push(th as i32);
                pos[2].push(tw as i32);
            }
        }
    }
    pos
}

/// `[C, T, H, W]` row-major latent -> token matrix `[patch_dim, L]`, feature
/// order `(w, h, t, c)` with `w` fastest — the GGUF conv fold order.
pub fn patchify(
    latent: &[f32],
    channels: i64,
    frames: i64,
    height: i64,
    width: i64,
    patch: (i64, i64, i64),
) -> Result<Vec<f32>, WeightError> {
    let (pt, ph, pw) = patch;
    if channels < 1 || frames % pt != 0 || height % ph != 0 || width % pw != 0 {
        return Err(WeightError::NotBuilt(format!(
            "wan latent [C,T,H,W]=[{channels},{frames},{height},{width}] is not divisible by patch {patch:?}"
        )));
    }
    let (fg, hg, wg) = (frames / pt, height / ph, width / pw);
    let pd = channels * pt * ph * pw;
    if latent.len() as i64 != channels * frames * height * width {
        return Err(WeightError::NotBuilt(format!(
            "wan latent has {} values, expected {}",
            latent.len(),
            channels * frames * height * width
        )));
    }
    let mut out = vec![0.0f32; (pd * fg * hg * wg) as usize];
    for tf in 0..fg {
        for th in 0..hg {
            for tw in 0..wg {
                let l = (tf * hg + th) * wg + tw;
                for ci in 0..channels {
                    for ti in 0..pt {
                        for hi in 0..ph {
                            for wi in 0..pw {
                                let j = wi + pw * (hi + ph * (ti + pt * ci));
                                let src = ((ci * frames + tf * pt + ti) * height + th * ph + hi)
                                    * width
                                    + tw * pw
                                    + wi;
                                out[(l * pd + j) as usize] = latent[src as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Head tokens `[patch_dim, L]` -> `[C, T, H, W]` row-major latent. The head
/// feature order is `(pt, ph, pw, c)` with `c` fastest.
pub fn unpatchify(
    tokens: &[f32],
    channels: i64,
    grid: (i64, i64, i64),
    patch: (i64, i64, i64),
) -> Result<Vec<f32>, WeightError> {
    let (pt, ph, pw) = patch;
    let (fg, hg, wg) = grid;
    let pd = channels * pt * ph * pw;
    let seq = fg * hg * wg;
    if tokens.len() as i64 != pd * seq {
        return Err(WeightError::NotBuilt(format!(
            "wan head tokens have {} values, expected {}",
            tokens.len(),
            pd * seq
        )));
    }
    let (frames, height, width) = (fg * pt, hg * ph, wg * pw);
    let mut out = vec![0.0f32; (channels * frames * height * width) as usize];
    for tf in 0..fg {
        for th in 0..hg {
            for tw in 0..wg {
                let l = (tf * hg + th) * wg + tw;
                for ti in 0..pt {
                    for hi in 0..ph {
                        for wi in 0..pw {
                            for ci in 0..channels {
                                let j = ((ti * ph + hi) * pw + wi) * channels + ci;
                                let dst = ((ci * frames + tf * pt + ti) * height + th * ph + hi)
                                    * width
                                    + tw * pw
                                    + wi;
                                out[dst as usize] = tokens[(l * pd + j) as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;
    use std::path::PathBuf;

    const DIM: usize = 128;
    const HEAD: usize = 128;
    const HEADS: usize = 1;
    const FFN: usize = 256;
    const IN_DIM: i64 = 2;
    const TEXT_DIM: i64 = 8;
    const TEXT_LEN: usize = 512;
    const GRID: (i64, i64, i64) = (2, 2, 2);
    const SEQ: usize = 8;

    fn smoke_config() -> DitConfig {
        DitConfig {
            dim: DIM as i64,
            depth: 2,
            heads: HEADS as i64,
            ffn_dim: FFN as i64,
            in_dim: IN_DIM,
            text_dim: TEXT_DIM,
            patch: (1, 2, 2),
        }
    }

    fn write_smoke(dir: &std::path::Path, skip: Option<&str>) -> PathBuf {
        super::write_smoke(dir, smoke_config(), skip).unwrap()
    }

    fn load_all(path: &std::path::Path, cfg: DitConfig) -> HashMap<String, Vec<f32>> {
        let w = GgufWeights::open(path).unwrap();
        let mut out = HashMap::new();
        for (name, ne0, ne1) in weight_layout(cfg) {
            let data = if name == "patch_embedding.weight" {
                w.f32_widened(
                    &name,
                    &[cfg.patch.2, cfg.patch.1, cfg.patch.0, cfg.in_dim * cfg.dim],
                )
                .unwrap()
            } else {
                w.f32_widened(&name, &[ne0, ne1]).unwrap()
            };
            out.insert(name, data);
        }
        out
    }

    // ----- host reference: plain f32, mirrors main `WanModelMLX` -----

    fn rlinear(
        w: &[f32],
        bias: &[f32],
        x: &[f32],
        in_dim: usize,
        out_dim: usize,
        rows: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; out_dim * rows];
        for r in 0..rows {
            for o in 0..out_dim {
                let mut acc = bias[o];
                for i in 0..in_dim {
                    acc += w[i + o * in_dim] * x[i + r * in_dim];
                }
                out[o + r * out_dim] = acc;
            }
        }
        out
    }

    fn ln(x: &[f32], dim: usize, rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + LN_EPS).sqrt();
            for d in 0..dim {
                out[r * dim + d] = (row[d] - mean) * inv;
            }
        }
        out
    }

    fn rms_full(x: &[f32], weight: &[f32], dim: usize, rows: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        for r in 0..rows {
            let row = &x[r * dim..(r + 1) * dim];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let inv = 1.0 / (ms + LN_EPS).sqrt();
            for d in 0..dim {
                out[r * dim + d] = row[d] * inv * weight[d];
            }
        }
        out
    }

    fn silu_ref(x: &[f32]) -> Vec<f32> {
        x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
    }

    fn gelu_tanh_ref(x: &[f32]) -> Vec<f32> {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        x.iter()
            .map(|v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
            .collect()
    }

    fn rope_ref(x: &mut [f32], seq: usize, pos: &[Vec<i32>; 3], dims: [i64; 3]) {
        let mut bounds = Vec::new();
        let mut start = 0i64;
        for axis in 0..3 {
            bounds.push((axis, start / 2, dims[axis]));
            start += dims[axis];
        }
        for l in 0..seq {
            for p in 0..HEAD / 2 {
                let (axis, j0, d_axis) = bounds
                    .iter()
                    .copied()
                    .find(|(_, j0, d)| (p as i64) >= *j0 && (p as i64) < j0 + d / 2)
                    .unwrap();
                let j = p as i64 - j0;
                let angle = pos[axis][l] as f32 * ROPE_THETA.powf(-2.0 * j as f32 / d_axis as f32);
                let (cos, sin) = (angle.cos(), angle.sin());
                let i0 = 2 * p + l * DIM;
                let i1 = i0 + 1;
                let (a, bv) = (x[i0], x[i1]);
                x[i0] = a * cos - bv * sin;
                x[i1] = a * sin + bv * cos;
            }
        }
    }

    fn attention_ref(q: &[f32], k: &[f32], v: &[f32], lq: usize, lk: usize) -> Vec<f32> {
        let scale = 1.0 / (HEAD as f32).sqrt();
        let mut out = vec![0.0f32; DIM * lq];
        for n in 0..HEADS {
            for i in 0..lq {
                let mut scores = vec![0.0f32; lk];
                for j in 0..lk {
                    let mut acc = 0.0f32;
                    for d in 0..HEAD {
                        acc += q[n * HEAD + d + i * DIM] * k[n * HEAD + d + j * DIM];
                    }
                    scores[j] = acc * scale;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for d in 0..HEAD {
                    let mut acc = 0.0f32;
                    for j in 0..lk {
                        acc += scores[j] / sum * v[n * HEAD + d + j * DIM];
                    }
                    out[n * HEAD + d + i * DIM] = acc;
                }
            }
        }
        out
    }

    /// `x * (1 + scale) + shift` with per-dim vectors.
    fn scale_shift_ref(x: &[f32], scale: &[f32], shift: &[f32], dim: usize) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, v)| v * (1.0 + scale[i % dim]) + shift[i % dim])
            .collect()
    }

    fn gate_add_ref(x: &[f32], gate: &[f32], y: &[f32], dim: usize) -> Vec<f32> {
        x.iter()
            .enumerate()
            .map(|(i, v)| v + y[i] * gate[i % dim])
            .collect()
    }

    fn reference(
        w: &HashMap<String, Vec<f32>>,
        tokens: &[f32],
        temb: &[f32],
        txt: &[f32],
    ) -> Vec<f32> {
        let cfg = smoke_config();
        let pd = cfg.patch_dim() as usize;
        let lin = |name: &str, x: &[f32], in_dim: usize, out_dim: usize, rows: usize| {
            rlinear(
                &w[&format!("{name}.weight")],
                &w[&format!("{name}.bias")],
                x,
                in_dim,
                out_dim,
                rows,
            )
        };
        let mut x = lin("patch_embedding", tokens, pd, DIM, SEQ);
        let t0 = lin(
            "time_embedding.0",
            temb,
            DitConfig::FREQ_DIM as usize,
            DIM,
            1,
        );
        let e = lin("time_embedding.2", &silu_ref(&t0), DIM, DIM, 1);
        let e0 = lin("time_projection.1", &silu_ref(&e), DIM, 6 * DIM, 1);
        let context = lin(
            "text_embedding.2",
            &gelu_tanh_ref(&lin(
                "text_embedding.0",
                txt,
                TEXT_DIM as usize,
                DIM,
                TEXT_LEN,
            )),
            DIM,
            DIM,
            TEXT_LEN,
        );
        let pos = rope_positions(GRID);
        let dims = smoke_config().rope_dims();
        for i in 0..cfg.depth {
            let prefix = format!("blocks.{i}");
            let modulation = &w[&format!("{prefix}.modulation")];
            let col = |j: usize| -> Vec<f32> {
                (0..DIM)
                    .map(|d| e0[j * DIM + d] + modulation[d + j * DIM])
                    .collect()
            };
            let (e0v, e1v, e2v, e3v, e4v, e5v) = (col(0), col(1), col(2), col(3), col(4), col(5));
            // self attention
            let n = ln(&x, DIM, SEQ);
            let n = scale_shift_ref(&n, &e1v, &e0v, DIM);
            let sa = format!("{prefix}.self_attn");
            let q = rms_full(
                &lin(&format!("{sa}.q"), &n, DIM, DIM, SEQ),
                &w[&format!("{sa}.norm_q.weight")],
                DIM,
                SEQ,
            );
            let k = rms_full(
                &lin(&format!("{sa}.k"), &n, DIM, DIM, SEQ),
                &w[&format!("{sa}.norm_k.weight")],
                DIM,
                SEQ,
            );
            let v = lin(&format!("{sa}.v"), &n, DIM, DIM, SEQ);
            let mut q = q;
            let mut k = k;
            rope_ref(&mut q, SEQ, &pos, dims);
            rope_ref(&mut k, SEQ, &pos, dims);
            let attn = attention_ref(&q, &k, &v, SEQ, SEQ);
            let y = lin(&format!("{sa}.o"), &attn, DIM, DIM, SEQ);
            let x1 = gate_add_ref(&x, &e2v, &y, DIM);
            // cross attention
            let n = ln(&x1, DIM, SEQ);
            let n3w = &w[&format!("{prefix}.norm3.weight")];
            let n3b = &w[&format!("{prefix}.norm3.bias")];
            let n: Vec<f32> = n
                .iter()
                .enumerate()
                .map(|(i, v)| v * n3w[i % DIM] + n3b[i % DIM])
                .collect();
            let ca = format!("{prefix}.cross_attn");
            let q = rms_full(
                &lin(&format!("{ca}.q"), &n, DIM, DIM, SEQ),
                &w[&format!("{ca}.norm_q.weight")],
                DIM,
                SEQ,
            );
            let k = rms_full(
                &lin(&format!("{ca}.k"), &context, DIM, DIM, TEXT_LEN),
                &w[&format!("{ca}.norm_k.weight")],
                DIM,
                TEXT_LEN,
            );
            let v = lin(&format!("{ca}.v"), &context, DIM, DIM, TEXT_LEN);
            let attn = attention_ref(&q, &k, &v, SEQ, TEXT_LEN);
            let y = lin(&format!("{ca}.o"), &attn, DIM, DIM, SEQ);
            let x2: Vec<f32> = x1.iter().zip(y.iter()).map(|(a, b)| a + b).collect();
            // ffn
            let n = ln(&x2, DIM, SEQ);
            let n = scale_shift_ref(&n, &e4v, &e3v, DIM);
            let hidden = lin(&format!("{prefix}.ffn.0"), &n, DIM, FFN, SEQ);
            let y = lin(
                &format!("{prefix}.ffn.2"),
                &gelu_tanh_ref(&hidden),
                FFN,
                DIM,
                SEQ,
            );
            x = gate_add_ref(&x2, &e5v, &y, DIM);
        }
        let hm = &w["head.modulation"];
        let shift: Vec<f32> = (0..DIM).map(|d| hm[d] + e[d]).collect();
        let scale: Vec<f32> = (0..DIM).map(|d| hm[DIM + d] + e[d]).collect();
        let n = ln(&x, DIM, SEQ);
        let n = scale_shift_ref(&n, &scale, &shift, DIM);
        lin("head.head", &n, DIM, pd, SEQ)
    }

    fn run_graph(
        graph: &mut crate::runtime::ReusedGraph,
        tokens: &[f32],
        temb: &[f32],
        txt: &[f32],
    ) -> Vec<f32> {
        graph.set_f32("tokens", tokens).unwrap();
        graph.set_f32("temb", temb).unwrap();
        graph.set_f32("txt", txt).unwrap();
        let pos = rope_positions(GRID);
        graph.set_i32("pos_0", &pos[0]).unwrap();
        graph.set_i32("pos_1", &pos[1]).unwrap();
        graph.set_i32("pos_2", &pos[2]).unwrap();
        graph.compute().unwrap()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn dit_matches_host_reference_and_replays() {
        let dir = std::env::temp_dir().join(format!("danmo-wan-dit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_smoke(&dir, None);
        let cfg = smoke_config();
        let weights = load_all(&path, cfg);
        // The ggml context owns the graph memory; it must outlive the graph.
        let ctx = crate::runtime::GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(64 * 1024 * 1024).unwrap();
        let out = build_dit(&mut b, cfg, SEQ as i64, &HashMap::new(), &[]).unwrap();
        let mut graph = b.finish(out).unwrap();
        for (name, data) in &weights {
            graph.set_f32(name, data).unwrap();
        }
        let latent: Vec<f32> = (0..(IN_DIM * 2 * 4 * 4) as usize)
            .map(|i| ((i as f32) * 0.37).sin() * 0.4)
            .collect();
        let tokens = patchify(&latent, IN_DIM, 2, 4, 4, cfg.patch).unwrap();
        let txt: Vec<f32> = (0..(TEXT_DIM as usize * TEXT_LEN))
            .map(|i| ((i as f32) * 0.11).cos() * 0.2)
            .collect();
        for (t, scale) in [(250.0f32, 1.0f32), (750.0, 0.5)] {
            let tokens: Vec<f32> = tokens.iter().map(|v| v * scale).collect();
            let temb = sinusoidal_embedding(DitConfig::FREQ_DIM as usize, t);
            let got = run_graph(&mut graph, &tokens, &temb, &txt);
            let want = reference(&weights, &tokens, &temb, &txt);
            assert_eq!(got.len(), want.len());
            let diff = max_diff(&got, &want);
            assert!(diff < 1e-3, "t {t}: max diff {diff}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn patchify_places_features_in_file_order() {
        // C=2, T=2, H=2, W=4, patch (1,2,2): grid (2,1,2), pd=8, L=4.
        let latent: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let tokens = patchify(&latent, 2, 2, 2, 4, (1, 2, 2)).unwrap();
        // token (tf, tw) = l = tf*2 + tw; feature j = wi + 2*hi + 4*ci
        // src = ((ci*2 + tf)*2 + hi)*4 + tw*2 + wi
        let expect = |l: usize, ci: i64, hi: i64, wi: i64| {
            let (tf, tw) = ((l / 2) as i64, (l % 2) as i64);
            ((ci * 2 + tf) * 2 + hi) * 4 + tw * 2 + wi
        };
        assert_eq!(tokens[0], expect(0, 0, 0, 0) as f32); // j=0
        assert_eq!(tokens[3], expect(0, 0, 1, 1) as f32); // j=3
        assert_eq!(tokens[4], expect(0, 1, 0, 0) as f32); // j=4: channel 1
        assert_eq!(tokens[8 + 7], expect(1, 1, 1, 1) as f32); // l=1, j=7
        assert_eq!(tokens[3 * 8 + 4], expect(3, 1, 0, 0) as f32); // l=3 (tf=1, tw=1)
                                                                  // unpatchify is the transpose mapping with c fastest in the head token.
        let head_tokens: Vec<f32> = (0..32).map(|i| (i + 100) as f32).collect();
        let back = unpatchify(&head_tokens, 2, (2, 1, 2), (1, 2, 2)).unwrap();
        // token l=(tf,tw), j = ((ti*ph+hi)*pw+wi)*c+ci; dst over [C,T,H,W]=[2,2,2,4]
        // j=0 l=0 -> c0 t0 h0 w0 = 100; j=1 l=0 -> c1 -> dst 16
        assert_eq!(back[0], 100.0);
        assert_eq!(back[16], 101.0);
        // j=2 -> wi=1, ci=0 -> w=1
        assert_eq!(back[1], 102.0);
        // j=4 -> hi=1 -> h=1 -> dst 4
        assert_eq!(back[4], 104.0);
        // l=3 -> tf=1, tw=1: j=7 -> ti0 hi1 wi1 ci1 -> dst = ((1*2+1)*2+1)*4 + 1*2+1 = 31
        assert_eq!(back[31], 100.0 + (3 * 8 + 7) as f32);
    }

    #[test]
    fn missing_tensor_fails_loud() {
        let dir =
            std::env::temp_dir().join(format!("danmo-wan-dit-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_smoke(&dir, Some("head.modulation"));
        let w = GgufWeights::open(&path).unwrap();
        let err = w.f32_widened("head.modulation", &[DIM as i64, 2]);
        assert!(err.is_err(), "missing head.modulation must fail");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
