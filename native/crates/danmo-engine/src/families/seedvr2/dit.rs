//! SeedVR2 MM-DiT one-shot upscale graph. Tensor names are the published
//! `seedvr2_ema_*` GGUF names. The caller feeds the 33 channel
//! `[noise; latent; mask]` input and a timestep; the single Euler step
//! (`t_norm == 1`, `s == 0`) is `latents - model_output` on the host.
//!
//! Config is inferred from the checkpoint: `.ada.all` blocks are the shared
//! MM layers, `proj_in_gate` marks SwiGLU (3B), `vid_out_ada` marks the
//! output modulation (3B), and rope `freqs` values above 1 mark pixel mode
//! (7B). `last_layer_vid_only` follows the SwiGLU split: both published
//! checkpoints agree with it. An unknown layout fails, it is never guessed.

use crate::runtime::{flash_attend, GgmlContext, GraphBuilder, RuntimeError, Tensor};
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};
use std::collections::HashMap;

const GRAPH_MEM: usize = 256 * 1024 * 1024;
const RMS_EPS: f32 = 1e-5;

#[derive(Debug, Clone)]
pub struct DitConfig {
    pub layers: usize,
    pub mm_layers: usize,
    pub dim: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub mlp_hidden: i64,
    pub txt_in_dim: i64,
    pub emb_sin: i64,
    pub swiglu: bool,
    pub output_ada: bool,
    pub last_layer_vid_only: bool,
    pub pixel_rope: bool,
    pub freqs_len: i64,
    pub vid_channels: i64,
    pub out_channels: i64,
    pub patch: i64,
    pub txt_len: i64,
}

impl DitConfig {
    pub fn inner(&self) -> i64 {
        self.heads * self.head_dim
    }

    pub fn patch_dim(&self) -> i64 {
        self.vid_channels * self.patch * self.patch
    }

    pub fn out_patch_dim(&self) -> i64 {
        self.out_channels * self.patch * self.patch
    }

    /// Smoke graph: two layers (one vid/txt, one shared), SwiGLU, output ada,
    /// lang rope. Head 8 gives rope `freq_dim` 2 and one frequency.
    pub fn smoke() -> Self {
        Self {
            layers: 2,
            mm_layers: 1,
            dim: 16,
            heads: 2,
            head_dim: 8,
            mlp_hidden: 24,
            txt_in_dim: 8,
            emb_sin: 8,
            swiglu: true,
            output_ada: true,
            last_layer_vid_only: true,
            pixel_rope: false,
            freqs_len: 1,
            vid_channels: 33,
            out_channels: 16,
            patch: 2,
            txt_len: 8,
        }
    }

    /// Infer the config from checkpoint tensor names, shapes, and rope freqs.
    pub fn infer(w: &GgufWeights, txt_len: i64) -> Result<Self, WeightError> {
        let mut layers = 0usize;
        let mut mm_layers = None;
        for name in w.names() {
            if let Some(rest) = name.strip_prefix("blocks.") {
                if let Some((index, _)) = rest.split_once('.') {
                    if let Ok(index) = index.parse::<usize>() {
                        layers = layers.max(index + 1);
                        if mm_layers.is_none() && rest.contains(".ada.all.") {
                            mm_layers = Some(index);
                        }
                    }
                }
            }
        }
        if layers == 0 {
            return Err(WeightError::NotBuilt(
                "seedvr2 dit has no blocks.* tensors".into(),
            ));
        }
        let mm_layers = mm_layers.unwrap_or(layers);
        let side = |i: usize| if i < mm_layers { "vid" } else { "all" };
        let ada = if mm_layers > 0 { "vid" } else { "all" };
        let dim = w.shape(&format!("blocks.0.ada.{ada}.attn_shift"))?[0];
        let head_dim = w.shape(&format!("blocks.0.attn.norm_q.{}.weight", side(0)))?[0];
        let qkv = w.shape(&format!("blocks.0.attn.proj_qkv.{}.weight", side(0)))?;
        if qkv[0] != dim || qkv[1] % (3 * head_dim) != 0 {
            return Err(WeightError::NotBuilt(format!(
                "seedvr2 blocks.0.attn.proj_qkv is {qkv:?}, dim {dim} head {head_dim}"
            )));
        }
        let heads = qkv[1] / (3 * head_dim);
        let swiglu = w.contains(&format!("blocks.0.mlp.{}.proj_in_gate.weight", side(0)));
        let mlp_hidden = w.shape(&format!("blocks.0.mlp.{}.proj_in.weight", side(0)))?[1];
        let output_ada = w.contains("vid_out_ada.out_shift") && w.contains("vid_out_norm.weight");
        let freqs_len = w.shape("blocks.0.attn.rope.rope.freqs")?[0];
        let freqs = w.f32_widened("blocks.0.attn.rope.rope.freqs", &[freqs_len])?;
        // lang freqs are `theta^(-2i/d)` <= 1; pixel freqs are `linspace(1, 128) * pi`.
        let pixel_rope = freqs.iter().any(|f| *f > 1.0);
        let txt_in_dim = w.shape("txt_in.weight")?[0];
        let patch_dim = w.shape("vid_in.proj.weight")?[0];
        let out_dim = w.shape("vid_out.proj.weight")?[1];
        let patch = 2;
        if patch_dim % (patch * patch) != 0 || out_dim % (patch * patch) != 0 {
            return Err(WeightError::NotBuilt(format!(
                "seedvr2 vid patch dims {patch_dim}/{out_dim} are not (1, 2, 2) patches"
            )));
        }
        let emb_sin = w.shape("emb_in.proj_in.weight")?[0];
        if w.shape("emb_in.proj_out.weight")?[1] != 6 * dim {
            return Err(WeightError::NotBuilt(
                "seedvr2 emb_in.proj_out is not 6 * dim".into(),
            ));
        }
        Ok(Self {
            layers,
            mm_layers,
            dim,
            heads,
            head_dim,
            mlp_hidden,
            txt_in_dim,
            emb_sin,
            swiglu,
            output_ada,
            // Both published checkpoints follow the SwiGLU/GELU split.
            last_layer_vid_only: swiglu,
            pixel_rope,
            freqs_len,
            vid_channels: patch_dim / (patch * patch),
            out_channels: out_dim / (patch * patch),
            patch,
            txt_len,
        })
    }

    fn side(&self, layer: usize) -> &'static str {
        if layer < self.mm_layers {
            "vid"
        } else {
            "all"
        }
    }
}

/// `(name, dims)` in ggml order: 2D is `[in, out]`, 1D is `[n]`.
pub fn weight_layout(config: &DitConfig) -> Vec<(String, Vec<i64>)> {
    fn push_linear(out: &mut Vec<(String, Vec<i64>)>, name: String, i: i64, o: i64, bias: bool) {
        out.push((name.clone(), vec![i, o]));
        if bias {
            out.push((name.replace(".weight", ".bias"), vec![o]));
        }
    }
    let mut out = Vec::new();
    let dim = config.dim;
    let head = config.head_dim;
    let inner = config.inner();
    let hidden = config.mlp_hidden;
    push_linear(
        &mut out,
        "vid_in.proj.weight".into(),
        config.patch_dim(),
        dim,
        true,
    );
    push_linear(
        &mut out,
        "txt_in.weight".into(),
        config.txt_in_dim,
        dim,
        true,
    );
    push_linear(
        &mut out,
        "emb_in.proj_in.weight".into(),
        config.emb_sin,
        dim,
        true,
    );
    push_linear(&mut out, "emb_in.proj_hid.weight".into(), dim, dim, true);
    push_linear(
        &mut out,
        "emb_in.proj_out.weight".into(),
        dim,
        6 * dim,
        true,
    );
    for i in 0..config.layers {
        let p = format!("blocks.{i}");
        let sides: Vec<&str> = if i < config.mm_layers {
            vec!["vid", "txt"]
        } else {
            vec!["all"]
        };
        for side in &sides {
            for layer in ["attn", "mlp"] {
                for mode in ["shift", "scale", "gate"] {
                    out.push((format!("{p}.ada.{side}.{layer}_{mode}"), vec![dim]));
                }
            }
            out.push((format!("{p}.attn.norm_q.{side}.weight"), vec![head]));
            out.push((format!("{p}.attn.norm_k.{side}.weight"), vec![head]));
            out.push((
                format!("{p}.attn.proj_qkv.{side}.weight"),
                vec![dim, 3 * inner],
            ));
            out.push((format!("{p}.attn.proj_out.{side}.weight"), vec![inner, dim]));
            out.push((format!("{p}.attn.proj_out.{side}.bias"), vec![dim]));
            if config.swiglu {
                out.push((
                    format!("{p}.mlp.{side}.proj_in_gate.weight"),
                    vec![dim, hidden],
                ));
                out.push((format!("{p}.mlp.{side}.proj_in.weight"), vec![dim, hidden]));
                out.push((format!("{p}.mlp.{side}.proj_out.weight"), vec![hidden, dim]));
            } else {
                out.push((format!("{p}.mlp.{side}.proj_in.weight"), vec![dim, hidden]));
                out.push((format!("{p}.mlp.{side}.proj_in.bias"), vec![hidden]));
                out.push((format!("{p}.mlp.{side}.proj_out.weight"), vec![hidden, dim]));
                out.push((format!("{p}.mlp.{side}.proj_out.bias"), vec![dim]));
            }
        }
        out.push((format!("{p}.attn.rope.rope.freqs"), vec![config.freqs_len]));
    }
    if config.output_ada {
        out.push(("vid_out_norm.weight".into(), vec![dim]));
        out.push(("vid_out_ada.out_shift".into(), vec![dim]));
        out.push(("vid_out_ada.out_scale".into(), vec![dim]));
    }
    push_linear(
        &mut out,
        "vid_out.proj.weight".into(),
        dim,
        config.out_patch_dim(),
        true,
    );
    out
}

pub enum TensorData {
    F32(Vec<f32>),
    Raw { bytes: Vec<u8>, ggml_type: u32 },
}

pub struct LoadedDit {
    pub tensors: Vec<(String, TensorData)>,
}

/// Read every weight in the layout. 1D params are widened to F32; matmuls
/// stay in their stored type. A missing tensor or a bad shape fails.
pub fn load(weights: &GgufWeights, config: &DitConfig) -> Result<LoadedDit, WeightError> {
    let mut tensors = Vec::new();
    for (name, dims) in weight_layout(config) {
        if dims.len() == 1 {
            tensors.push((
                name.clone(),
                TensorData::F32(weights.f32_widened(&name, &dims)?),
            ));
        } else {
            let ggml_type = weights.ggml_type_of(&name)?;
            if !matmul_type(ggml_type) {
                return Err(WeightError::Type {
                    name: name.clone(),
                    ggml_type,
                });
            }
            let block = type_block(ggml_type);
            if block > 1 && dims[0] % block != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "{name} length {} is not divisible by ggml block {block}",
                    dims[0]
                )));
            }
            tensors.push((
                name.clone(),
                TensorData::Raw {
                    bytes: weights.raw_exact(&name, &dims)?,
                    ggml_type,
                },
            ));
        }
    }
    Ok(LoadedDit { tensors })
}

/// Spatial window plan for one shift parity. `forward[i]` is the token index
/// that lands at partitioned position `i`; `reverse` undoes it. Windows are
/// `(start, len, [t, h, w])` in partitioned order.
pub struct WindowPlan {
    pub forward: Vec<i32>,
    pub reverse: Vec<i32>,
    pub windows: Vec<(i64, i64, [i64; 3])>,
}

/// Port of the reference `WindowPartitioner._make_windows` with
/// `window_size == (4, 3, 3)`. Only the parity shift changes the plan.
pub fn plan_windows(t: i64, h: i64, w: i64, shift: bool) -> WindowPlan {
    let scale = (3600.0 / (h * w) as f64).sqrt();
    let resized_h = (h as f64 * scale).round() as i64;
    let resized_w = (w as f64 * scale).round() as i64;
    let ceil_div = |a: i64, b: i64| a.div_euclid(b) + i64::from(a % b != 0);
    let wh = ceil_div(resized_h, 3);
    let ww = ceil_div(resized_w, 3);
    let wt = ceil_div(t.min(30), 4);
    let (st, sh, sw, nt, nh, nw) = if shift {
        let st = if wt < t { 0.5 } else { 0.0 };
        let sh = if wh < h { 0.5 } else { 0.0 };
        let sw = if ww < w { 0.5 } else { 0.0 };
        let nt = if st > 0.0 {
            ((t as f64 - st) / wt as f64).ceil() as i64 + 1
        } else {
            1
        };
        let nh = if sh > 0.0 {
            ((h as f64 - sh) / wh as f64).ceil() as i64 + 1
        } else {
            1
        };
        let nw = if sw > 0.0 {
            ((w as f64 - sw) / ww as f64).ceil() as i64 + 1
        } else {
            1
        };
        (st, sh, sw, nt, nh, nw)
    } else {
        (
            0.0,
            0.0,
            0.0,
            ceil_div(t, wt),
            ceil_div(h, wh),
            ceil_div(w, ww),
        )
    };
    let mut windows: Vec<([i64; 2], [i64; 2], [i64; 2])> = Vec::new();
    for iw in 0..nw {
        let w_start = (((iw as f64 - sw) * ww as f64) as i64).max(0);
        let w_end = ((((iw as f64 - sw) + 1.0) * ww as f64) as i64).min(w);
        if w_end <= w_start {
            continue;
        }
        for ih in 0..nh {
            let h_start = (((ih as f64 - sh) * wh as f64) as i64).max(0);
            let h_end = ((((ih as f64 - sh) + 1.0) * wh as f64) as i64).min(h);
            if h_end <= h_start {
                continue;
            }
            for it in 0..nt {
                let t_start = (((it as f64 - st) * wt as f64) as i64).max(0);
                let t_end = ((((it as f64 - st) + 1.0) * wt as f64) as i64).min(t);
                if t_end <= t_start {
                    continue;
                }
                windows.push(([t_start, t_end], [h_start, h_end], [w_start, w_end]));
            }
        }
    }
    let total = (t * h * w) as usize;
    let mut forward = Vec::with_capacity(total);
    let mut spans = Vec::with_capacity(windows.len());
    for ([t0, t1], [h0, h1], [w0, w1]) in &windows {
        let start = forward.len() as i64;
        for tt in *t0..*t1 {
            for yy in *h0..*h1 {
                for xx in *w0..*w1 {
                    forward.push((tt * h * w + yy * w + xx) as i32);
                }
            }
        }
        spans.push((
            start,
            forward.len() as i64 - start,
            [t1 - t0, h1 - h0, w1 - w0],
        ));
    }
    let mut reverse = vec![0i32; total];
    for (position, &token) in forward.iter().enumerate() {
        reverse[token as usize] = position as i32;
    }
    WindowPlan {
        forward,
        reverse,
        windows: spans,
    }
}

/// cos/sin tables `[head_dim, tokens]` in partitioned order plus the signed
/// swap matrix for `rotate_half`. The last `head_dim - rot_dim` channels pass
/// through (cos 1, sin 0).
pub struct RopeTables {
    pub vid_cos: [Vec<f32>; 2],
    pub vid_sin: [Vec<f32>; 2],
    pub txt_cos: Vec<f32>,
    pub txt_sin: Vec<f32>,
    pub swap: Vec<f32>,
}

fn axial_freqs(pos: &[f64], freqs: &[f32], per_axis: usize) -> Vec<f64> {
    // outer(pos, freqs) with each frequency repeated twice, `[len, per_axis]`.
    let mut out = vec![0.0f64; pos.len() * per_axis];
    for (t, &p) in pos.iter().enumerate() {
        for j in 0..per_axis / 2 {
            let angle = p * freqs[j] as f64;
            out[t * per_axis + 2 * j] = angle;
            out[t * per_axis + 2 * j + 1] = angle;
        }
    }
    out
}

/// Port of `RoPEModule._get_axial_freqs` + `_apply_mm_rope_3d` for one
/// window or the text span. Returns `[tokens][rot_dim]` angles.
fn window_angles(
    freqs: &[f32],
    pixel: bool,
    txt_len: i64,
    shape: [i64; 3],
    is_txt: bool,
) -> Vec<Vec<f64>> {
    let per_axis = 2 * freqs.len();
    let rot_dim = per_axis * 3;
    let positions = |n: i64| -> Vec<f64> {
        if pixel {
            (0..n)
                .map(|i| -1.0 + 2.0 * i as f64 / (n - 1).max(1) as f64)
                .collect()
        } else {
            (0..n).map(|i| i as f64).collect()
        }
    };
    if is_txt {
        // Text positions repeat the 1D axis across all three rope axes.
        let pos = positions(txt_len);
        let axis = axial_freqs(&pos, freqs, per_axis);
        return (0..txt_len as usize)
            .map(|t| {
                let mut row = Vec::with_capacity(rot_dim);
                for _ in 0..3 {
                    row.extend_from_slice(&axis[t * per_axis..(t + 1) * per_axis]);
                }
                row
            })
            .collect();
    }
    let [t, h, w] = shape;
    // lang: temporal starts after the text tokens. pixel: the temporal axis
    // is a linspace shared with the text span, sliced to the video frames.
    let pos_t: Vec<f64> = if pixel {
        let n = txt_len + t;
        (txt_len..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1).max(1) as f64)
            .collect()
    } else {
        (0..t).map(|i| (txt_len + i) as f64).collect()
    };
    let pos_h = positions(h);
    let pos_w = positions(w);
    let fa = axial_freqs(&pos_t, freqs, per_axis);
    let fb = axial_freqs(&pos_h, freqs, per_axis);
    let fc = axial_freqs(&pos_w, freqs, per_axis);
    let mut out = Vec::with_capacity((t * h * w) as usize);
    for tt in 0..t as usize {
        for yy in 0..h as usize {
            for xx in 0..w as usize {
                let mut row = Vec::with_capacity(rot_dim);
                row.extend_from_slice(&fa[tt * per_axis..(tt + 1) * per_axis]);
                row.extend_from_slice(&fb[yy * per_axis..(yy + 1) * per_axis]);
                row.extend_from_slice(&fc[xx * per_axis..(xx + 1) * per_axis]);
                out.push(row);
            }
        }
    }
    out
}

/// Build the rope tables for both shift parities. `plans[p]` is the window
/// plan for parity `p`; tables follow each plan's partitioned order.
pub fn rope_tables(config: &DitConfig, plans: &[WindowPlan; 2], freqs: &[f32]) -> RopeTables {
    let head = config.head_dim as usize;
    let per_axis = 2 * freqs.len();
    let rot_dim = per_axis * 3;
    let mut tables = RopeTables {
        vid_cos: [Vec::new(), Vec::new()],
        vid_sin: [Vec::new(), Vec::new()],
        txt_cos: vec![0.0; head * config.txt_len as usize],
        txt_sin: vec![0.0; head * config.txt_len as usize],
        swap: vec![0.0; head * head],
    };
    // Tables are `[head_dim, tokens]` flat (channel fastest per token).
    let fill_t = |angles: &[Vec<f64>]| -> (Vec<f32>, Vec<f32>) {
        let mut cos = vec![1.0f32; head * angles.len()];
        let mut sin = vec![0.0f32; head * angles.len()];
        for (t, row) in angles.iter().enumerate() {
            for (c, &angle) in row.iter().enumerate() {
                cos[t * head + c] = angle.cos() as f32;
                sin[t * head + c] = angle.sin() as f32;
            }
        }
        (cos, sin)
    };
    for parity in 0..2 {
        let plan = &plans[parity];
        // Per-window angles in partitioned order.
        let mut angles: Vec<Vec<f64>> = Vec::new();
        for (_, _, shape) in &plan.windows {
            angles.extend(window_angles(
                freqs,
                config.pixel_rope,
                config.txt_len,
                *shape,
                false,
            ));
        }
        let (cos, sin) = fill_t(&angles);
        tables.vid_cos[parity] = cos;
        tables.vid_sin[parity] = sin;
    }
    let txt_angles = window_angles(freqs, config.pixel_rope, config.txt_len, [1, 1, 1], true);
    let (cos, sin) = fill_t(&txt_angles);
    tables.txt_cos = cos;
    tables.txt_sin = sin;
    // rotate_half as a signed permutation: pairs (2i, 2i+1) in the rotated
    // span, identity outside.
    for c in 0..head {
        if c < rot_dim {
            let (src, sign) = if c % 2 == 0 {
                (c + 1, -1.0)
            } else {
                (c - 1, 1.0)
            };
            tables.swap[c * head + src] = sign;
        } else {
            tables.swap[c * head + c] = 1.0;
        }
    }
    tables
}

/// `(start, len)` of every window, per parity, plus the gather/scatter
/// indices and rope tables the graph consumes.
struct Plan {
    windows: [WindowPlan; 2],
    rope: RopeTables,
    patch_in: Vec<i32>,
    patch_out: Vec<i32>,
}

fn plan(config: &DitConfig, t: i64, w: i64, h: i64, freqs: &[f32]) -> Result<Plan, WeightError> {
    let patch = config.patch;
    if w % patch != 0 || h % patch != 0 {
        return Err(WeightError::NotBuilt(format!(
            "seedvr2 latent {w}x{h} is not divisible by the patch {patch}"
        )));
    }
    if t < 1 {
        return Err(WeightError::NotBuilt(format!(
            "seedvr2 dit needs at least one latent frame, got {t}"
        )));
    }
    let (hp, wp) = (h / patch, w / patch);
    let windows = [
        plan_windows(t, hp, wp, false),
        plan_windows(t, hp, wp, true),
    ];
    let rope = rope_tables(config, &windows, freqs);
    // Token order is `[T][H][W]` (frame outer, matching the reference
    // patchify); each token spans `(hh, ww, c)` with the channel fastest.
    let mut patch_in = Vec::with_capacity((t * hp * wp * config.patch_dim()) as usize);
    for tt in 0..t {
        for yy in 0..hp {
            for xx in 0..wp {
                for hh in 0..patch {
                    for ww in 0..patch {
                        for c in 0..config.vid_channels {
                            patch_in.push(
                                (xx * patch
                                    + ww
                                    + w * (yy * patch + hh)
                                    + w * h * tt
                                    + w * h * t * c) as i32,
                            );
                        }
                    }
                }
            }
        }
    }
    let mut patch_out = Vec::with_capacity((t * hp * wp * config.out_patch_dim()) as usize);
    for tt in 0..t {
        for yy in 0..hp {
            for xx in 0..wp {
                for hh in 0..patch {
                    for ww in 0..patch {
                        for c in 0..config.out_channels {
                            patch_out.push(
                                (xx * patch
                                    + ww
                                    + w * (yy * patch + hh)
                                    + w * h * tt
                                    + w * h * t * c) as i32,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(Plan {
        windows,
        rope,
        patch_in,
        patch_out,
    })
}

/// Sinusoidal timestep features `[emb_sin]`: `[sin(t * f), cos(t * f)]`.
pub fn timestep_features(timestep: f32, dim: i64) -> Vec<f32> {
    let half = (dim / 2) as usize;
    let mut out = Vec::with_capacity(dim as usize);
    for i in 0..half {
        let freq = (-(10000f64.ln()) * i as f64 / half as f64).exp();
        out.push((timestep as f64 * freq).sin() as f32);
    }
    for i in 0..half {
        let freq = (-(10000f64.ln()) * i as f64 / half as f64).exp();
        out.push((timestep as f64 * freq).cos() as f32);
    }
    out
}

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
        .ok_or_else(|| RuntimeError::Message(format!("seedvr2 dit graph is missing {name}")))
}

fn shaped(tensor: Tensor, ne: [i64; 4], what: &str) -> Result<Tensor, RuntimeError> {
    let got = crate::runtime::shape(tensor);
    if got != ne {
        return Err(RuntimeError::Message(format!(
            "{what} is {got:?}, the seedvr2 dit expects {ne:?}"
        )));
    }
    Ok(tensor)
}

fn add_bias(b: &GraphBuilder, x: Tensor, bias: Tensor) -> Result<Tensor, RuntimeError> {
    nn(b.add(x, bias))
}

/// `emb` is `[dim, 2, 3]`; column `(layer, mode)` is a `[dim, 1]` slice.
fn emb_col(
    b: &GraphBuilder,
    emb: Tensor,
    dim: i64,
    layer: i64,
    mode: i64,
) -> Result<Tensor, RuntimeError> {
    let offset = ((mode * 2 + layer) * dim * 4) as usize;
    nn(b.cont(b.view_2d(emb, dim, 1, (dim * 4) as usize, offset)))
}

fn modulate_in(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    emb: Tensor,
    prefix: &str,
    layer: &str,
    layer_idx: i64,
    dim: i64,
) -> Result<Tensor, RuntimeError> {
    let shift = add_bias(
        b,
        emb_col(b, emb, dim, layer_idx, 0)?,
        wt(weights, &format!("{prefix}.{layer}_shift"))?,
    )?;
    let scale = add_bias(
        b,
        emb_col(b, emb, dim, layer_idx, 1)?,
        wt(weights, &format!("{prefix}.{layer}_scale"))?,
    )?;
    nn(b.add(nn(b.mul(x, scale))?, shift))
}

fn modulate_out(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    emb: Tensor,
    prefix: &str,
    layer: &str,
    layer_idx: i64,
    dim: i64,
) -> Result<Tensor, RuntimeError> {
    let gate = add_bias(
        b,
        emb_col(b, emb, dim, layer_idx, 2)?,
        wt(weights, &format!("{prefix}.{layer}_gate"))?,
    )?;
    nn(b.mul(x, gate))
}

/// `y = x * cos + rotate_half(x) * sin`; tables are `[head, L]`, `x` is
/// `[head, heads, L]`.
fn apply_rope(
    b: &GraphBuilder,
    x: Tensor,
    cos_t: Tensor,
    sin_t: Tensor,
    swap: Tensor,
    head: i64,
    heads: i64,
    len: i64,
) -> Result<Tensor, RuntimeError> {
    let flat = nn(b.reshape_2d(x, head, heads * len))?;
    let rot = nn(b.reshape_3d(nn(b.mul_mat(swap, flat))?, head, heads, len))?;
    let cos_b = nn(b.reshape_3d(cos_t, head, 1, len))?;
    let sin_b = nn(b.reshape_3d(sin_t, head, 1, len))?;
    nn(b.add(nn(b.mul(x, cos_b))?, nn(b.mul(rot, sin_b))?))
}

/// Per-head RMS norm over `head` channels with the published weight.
fn head_norm(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    x: Tensor,
    name: &str,
) -> Result<Tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, wt(weights, name)?))
}

struct AttentionOut {
    vid: Tensor,
    txt: Tensor,
}

#[allow(clippy::too_many_arguments)]
fn attention(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: &DitConfig,
    plan: &Plan,
    layer: usize,
    vid: Tensor,
    txt: Tensor,
    swap: Tensor,
    cos_t: Tensor,
    sin_t: Tensor,
    cos_txt: Tensor,
    sin_txt: Tensor,
    fwd: Tensor,
    rev: Tensor,
) -> Result<AttentionOut, RuntimeError> {
    let p = format!("blocks.{layer}");
    let head = config.head_dim;
    let heads = config.heads;
    let inner = config.inner();
    let vid_len = crate::runtime::shape(vid)[1];
    let txt_len = config.txt_len;

    let split = |qkv: Tensor, len: i64| -> Result<(Tensor, Tensor, Tensor), RuntimeError> {
        let qkv = nn(b.reshape_3d(qkv, head, heads * 3, len))?;
        let nb1 = (head * 4) as usize;
        let nb2 = (head * heads * 3 * 4) as usize;
        let q = nn(b.cont(b.view_3d(qkv, head, heads, len, nb1, nb2, 0)))?;
        let k =
            nn(b.cont(b.view_3d(qkv, head, heads, len, nb1, nb2, (heads * head * 4) as usize)))?;
        let v = nn(b.cont(b.view_3d(
            qkv,
            head,
            heads,
            len,
            nb1,
            nb2,
            (2 * heads * head * 4) as usize,
        )))?;
        Ok((q, k, v))
    };

    let side = config.side(layer);
    let qkv_v = nn(b.project(
        wt(weights, &format!("{p}.attn.proj_qkv.{side}.weight"))?,
        vid,
    ))?;
    let (q_v, k_v, v_v) = split(qkv_v, vid_len)?;
    let txt_side = if side == "all" { "all" } else { "txt" };
    let qkv_t = nn(b.project(
        wt(weights, &format!("{p}.attn.proj_qkv.{txt_side}.weight"))?,
        txt,
    ))?;
    let (q_t, k_t, v_t) = split(qkv_t, txt_len)?;

    let q_v = head_norm(b, weights, q_v, &format!("{p}.attn.norm_q.{side}.weight"))?;
    let k_v = head_norm(b, weights, k_v, &format!("{p}.attn.norm_k.{side}.weight"))?;
    let q_t = head_norm(
        b,
        weights,
        q_t,
        &format!("{p}.attn.norm_q.{txt_side}.weight"),
    )?;
    let k_t = head_norm(
        b,
        weights,
        k_t,
        &format!("{p}.attn.norm_k.{txt_side}.weight"),
    )?;

    // Window partition of the video tokens.
    let gather = |x: Tensor| -> Result<Tensor, RuntimeError> {
        let flat = nn(b.reshape_2d(x, inner, vid_len))?;
        let got = nn(b.get_rows(flat, fwd))?;
        nn(b.reshape_3d(got, head, heads, vid_len))
    };
    let q_v = gather(q_v)?;
    let k_v = gather(k_v)?;
    let v_v = gather(v_v)?;

    // RoPE on video (partitioned order) and text.
    let q_v = apply_rope(b, q_v, cos_t, sin_t, swap, head, heads, vid_len)?;
    let k_v = apply_rope(b, k_v, cos_t, sin_t, swap, head, heads, vid_len)?;
    let q_t = apply_rope(b, q_t, cos_txt, sin_txt, swap, head, heads, txt_len)?;
    let k_t = apply_rope(b, k_t, cos_txt, sin_txt, swap, head, heads, txt_len)?;

    // Per window: concat text tokens, flash attention, split back.
    let mut vid_parts: Vec<Tensor> = Vec::new();
    let mut txt_sum: Option<Tensor> = None;
    let nb1 = (head * 4) as usize;
    let nb2 = (head * heads * 4) as usize;
    for (start, len, _) in &plan.windows[layer % 2].windows {
        let offset = (*start * inner * 4) as usize;
        let slice = |x: Tensor| -> Result<Tensor, RuntimeError> {
            nn(b.cont(b.view_3d(x, head, heads, *len, nb1, nb2, offset)))
        };
        let q_w = nn(b.cont(b.concat(slice(q_v)?, q_t, 2)))?;
        let k_w = nn(b.cont(b.concat(slice(k_v)?, k_t, 2)))?;
        let v_w = nn(b.cont(b.concat(slice(v_v)?, v_t, 2)))?;
        let out = flash_attend(b, q_w, k_w, v_w, std::ptr::null_mut())?;
        let out = shaped(
            out,
            [head, heads, len + txt_len, 1],
            "seedvr2 dit attention out",
        )?;
        let vid_o = nn(b.cont(b.view_3d(out, head, heads, *len, nb1, nb2, 0)))?;
        vid_parts.push(vid_o);
        let txt_o = nn(b.cont(b.view_3d(
            out,
            head,
            heads,
            txt_len,
            nb1,
            nb2,
            (*len * inner * 4) as usize,
        )))?;
        txt_sum = Some(match txt_sum {
            None => txt_o,
            Some(sum) => nn(b.add(sum, txt_o))?,
        });
    }
    let windows_n = plan.windows[layer % 2].windows.len() as f32;
    let mut vid_cat = vid_parts[0];
    for part in &vid_parts[1..] {
        vid_cat = nn(b.cont(b.concat(vid_cat, *part, 2)))?;
    }
    let txt_sum = txt_sum.ok_or(RuntimeError::Init)?;
    let txt_mean = nn(b.scale(txt_sum, windows_n.recip()))?;

    // Reverse the partition, project out.
    let vid_flat = nn(b.reshape_2d(vid_cat, inner, vid_len))?;
    let vid_rev = nn(b.get_rows(vid_flat, rev))?;
    let vid_out = nn(b.project(
        wt(weights, &format!("{p}.attn.proj_out.{side}.weight"))?,
        vid_rev,
    ))?;
    let vid_out = add_bias(
        b,
        vid_out,
        wt(weights, &format!("{p}.attn.proj_out.{side}.bias"))?,
    )?;
    let txt_flat = nn(b.reshape_2d(txt_mean, inner, txt_len))?;
    let txt_out = nn(b.project(
        wt(weights, &format!("{p}.attn.proj_out.{txt_side}.weight"))?,
        txt_flat,
    ))?;
    let txt_out = add_bias(
        b,
        txt_out,
        wt(weights, &format!("{p}.attn.proj_out.{txt_side}.bias"))?,
    )?;
    Ok(AttentionOut {
        vid: vid_out,
        txt: txt_out,
    })
}

/// tanh-approximated GELU without constant tensors:
/// `0.5x + 0.5x * tanh(0.7978845608 (x + 0.044715 x^3))`.
fn gelu_approx(b: &GraphBuilder, x: Tensor) -> Result<Tensor, RuntimeError> {
    let x3 = nn(b.mul(nn(b.mul(x, x))?, x))?;
    let inner = nn(b.add(x, nn(b.scale(x3, 0.044715))?))?;
    let t = nn(b.tanh(nn(b.scale(inner, 0.797_884_6))?))?;
    let half = nn(b.scale(x, 0.5))?;
    nn(b.add(half, nn(b.mul(half, t))?))
}

fn mlp(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: &DitConfig,
    prefix: &str,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    if config.swiglu {
        let gate = nn(b.project(wt(weights, &format!("{prefix}.proj_in_gate.weight"))?, x))?;
        let up = nn(b.project(wt(weights, &format!("{prefix}.proj_in.weight"))?, x))?;
        let hidden = nn(b.mul(nn(b.silu(gate))?, up))?;
        nn(b.project(wt(weights, &format!("{prefix}.proj_out.weight"))?, hidden))
    } else {
        let hidden = nn(b.project(wt(weights, &format!("{prefix}.proj_in.weight"))?, x))?;
        let hidden = add_bias(b, hidden, wt(weights, &format!("{prefix}.proj_in.bias"))?)?;
        let hidden = gelu_approx(b, hidden)?;
        let out = nn(b.project(wt(weights, &format!("{prefix}.proj_out.weight"))?, hidden))?;
        add_bias(b, out, wt(weights, &format!("{prefix}.proj_out.bias"))?)
    }
}

#[allow(clippy::too_many_arguments)]
fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: &DitConfig,
    plan: &Plan,
    inputs: &BlockInputs,
    layer: usize,
    vid: Tensor,
    txt: Tensor,
    emb: Tensor,
) -> Result<(Tensor, Tensor), RuntimeError> {
    let p = format!("blocks.{layer}");
    let dim = config.dim;
    let side = config.side(layer);
    let last = config.last_layer_vid_only && layer + 1 == config.layers;
    let ada_vid = format!("{p}.ada.{side}");
    let ada_txt = if side == "all" {
        ada_vid.clone()
    } else {
        format!("{p}.ada.txt")
    };

    let vid_n = nn(b.rms_norm(vid, RMS_EPS))?;
    let txt_n = nn(b.rms_norm(txt, RMS_EPS))?;
    let vid_n = modulate_in(b, weights, vid_n, emb, &ada_vid, "attn", 0, dim)?;
    let txt_n = if last {
        txt_n
    } else {
        modulate_in(b, weights, txt_n, emb, &ada_txt, "attn", 0, dim)?
    };
    let attn = attention(
        b,
        weights,
        config,
        plan,
        layer,
        vid_n,
        txt_n,
        inputs.swap,
        inputs.vid_cos[layer % 2],
        inputs.vid_sin[layer % 2],
        inputs.txt_cos,
        inputs.txt_sin,
        inputs.fwd[layer % 2],
        inputs.rev[layer % 2],
    )?;
    let vid_attn = modulate_out(b, weights, attn.vid, emb, &ada_vid, "attn", 0, dim)?;
    let vid = nn(b.add(vid, vid_attn))?;
    let txt = if last {
        txt
    } else {
        let txt_attn = modulate_out(b, weights, attn.txt, emb, &ada_txt, "attn", 0, dim)?;
        nn(b.add(txt, txt_attn))?
    };

    let vid_n = nn(b.rms_norm(vid, RMS_EPS))?;
    let vid_n = modulate_in(b, weights, vid_n, emb, &ada_vid, "mlp", 1, dim)?;
    let vid_m = mlp(b, weights, config, &format!("{p}.mlp.{side}"), vid_n)?;
    let vid_m = modulate_out(b, weights, vid_m, emb, &ada_vid, "mlp", 1, dim)?;
    let vid = nn(b.add(vid, vid_m))?;
    let txt = if last {
        txt
    } else {
        let txt_n = nn(b.rms_norm(txt, RMS_EPS))?;
        let txt_n = modulate_in(b, weights, txt_n, emb, &ada_txt, "mlp", 1, dim)?;
        let txt_side = if side == "all" { "all" } else { "txt" };
        let txt_m = mlp(b, weights, config, &format!("{p}.mlp.{txt_side}"), txt_n)?;
        let txt_m = modulate_out(b, weights, txt_m, emb, &ada_txt, "mlp", 1, dim)?;
        nn(b.add(txt, txt_m))?
    };
    Ok((vid, txt))
}

struct BlockInputs {
    swap: Tensor,
    vid_cos: [Tensor; 2],
    vid_sin: [Tensor; 2],
    txt_cos: Tensor,
    txt_sin: Tensor,
    fwd: [Tensor; 2],
    rev: [Tensor; 2],
}

/// Declare every weight in the layout: 1D as F32, matmuls in their stored
/// ggml type.
fn declare_weights(
    b: &mut GraphBuilder,
    config: &DitConfig,
    loaded: &LoadedDit,
) -> Result<HashMap<String, Tensor>, WeightError> {
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for (name, dims) in weight_layout(config) {
        let tensor = if dims.len() == 1 {
            b.f32_2d(&name, dims[0], 1, true)
        } else {
            let ggml_type = match loaded.tensors.iter().find(|(n, _)| n == &name) {
                Some((_, TensorData::Raw { ggml_type, .. })) => *ggml_type,
                _ => {
                    return Err(WeightError::NotBuilt(format!(
                        "{name} is not loaded as a matmul weight"
                    )))
                }
            };
            b.tensor_2d(&name, ggml_type, dims[0], dims[1], true)
        };
        if tensor.is_null() {
            return Err(WeightError::NotBuilt(format!(
                "seedvr2 dit cannot declare {name}"
            )));
        }
        weights.insert(name, tensor);
    }
    Ok(weights)
}

/// Run one SeedVR2 DiT forward over `t` latent frames. `vid` is the
/// `[W, H, T, vid_channels]` planar input (`[noise; latent; mask]`, channel
/// outer), `txt` the `[txt_in_dim, txt_len]` positive embedding. Returns the
/// `[W, H, T, out_channels]` planar velocity.
pub fn run(
    ctx: &GgmlContext,
    config: &DitConfig,
    loaded: &LoadedDit,
    vid: &[f32],
    w: i64,
    h: i64,
    t: i64,
    txt: &[f32],
    timestep: f32,
) -> Result<Vec<f32>, WeightError> {
    let (vid_data, txt_data) = (vid, txt);
    if vid_data.len() != (w * h * t * config.vid_channels) as usize {
        return Err(WeightError::Shape {
            name: "seedvr2 dit vid".into(),
            expected: vec![w, h, t, config.vid_channels],
            got: vec![vid.len() as i64],
        });
    }
    if txt_data.len() != (config.txt_in_dim * config.txt_len) as usize {
        return Err(WeightError::Shape {
            name: "seedvr2 dit txt".into(),
            expected: vec![config.txt_in_dim, config.txt_len],
            got: vec![txt.len() as i64],
        });
    }
    // Rope freqs are per-block weights; the published checkpoints repeat the
    // same vector, and a divergence fails loud instead of picking one.
    let freqs_name = |i: usize| format!("blocks.{i}.attn.rope.rope.freqs");
    let mut freqs: Option<Vec<f32>> = None;
    for i in 0..config.layers {
        let values = loaded
            .tensors
            .iter()
            .find_map(|(name, data)| match (name == &freqs_name(i), data) {
                (true, TensorData::F32(values)) => Some(values.clone()),
                _ => None,
            })
            .ok_or_else(|| WeightError::NotBuilt(format!("{} is not loaded", freqs_name(i))))?;
        match &freqs {
            None => freqs = Some(values),
            Some(first) if *first == values => {}
            Some(_) => {
                return Err(WeightError::NotBuilt(format!(
                    "{} diverges from blocks.0 rope freqs",
                    freqs_name(i)
                )))
            }
        }
    }
    let freqs =
        freqs.ok_or_else(|| WeightError::NotBuilt("seedvr2 dit has no rope freqs".into()))?;
    let plan = plan(config, t, w, h, &freqs)?;
    let tokens = t * (w / config.patch) * (h / config.patch);

    let mut b = ctx.builder(GRAPH_MEM).map_err(weight_err)?;
    let weights = declare_weights(&mut b, config, loaded)?;

    let vid_in =
        nn(b.f32_2d("vid", 1, w * h * t * config.vid_channels, true)).map_err(weight_err)?;
    let txt_in =
        nn(b.f32_2d("txt", config.txt_in_dim, config.txt_len, true)).map_err(weight_err)?;
    let temb = nn(b.f32_2d("temb", config.emb_sin, 1, true)).map_err(weight_err)?;
    let patch_in_idx =
        nn(b.i32_1d("patch_in_idx", tokens * config.patch_dim(), true)).map_err(weight_err)?;
    let patch_out_idx =
        nn(b.i32_1d("patch_out_idx", tokens * config.out_patch_dim(), true)).map_err(weight_err)?;
    let mut fwd = [std::ptr::null_mut(); 2];
    let mut rev = [std::ptr::null_mut(); 2];
    let mut vid_cos = [std::ptr::null_mut(); 2];
    let mut vid_sin = [std::ptr::null_mut(); 2];
    for parity in 0..2 {
        fwd[parity] = nn(b.i32_1d(&format!("fwd_{parity}"), tokens, true)).map_err(weight_err)?;
        rev[parity] = nn(b.i32_1d(&format!("rev_{parity}"), tokens, true)).map_err(weight_err)?;
        vid_cos[parity] =
            nn(b.f32_2d(&format!("rope_cos_{parity}"), config.head_dim, tokens, true))
                .map_err(weight_err)?;
        vid_sin[parity] =
            nn(b.f32_2d(&format!("rope_sin_{parity}"), config.head_dim, tokens, true))
                .map_err(weight_err)?;
    }
    let txt_cos =
        nn(b.f32_2d("rope_cos_txt", config.head_dim, config.txt_len, true)).map_err(weight_err)?;
    let txt_sin =
        nn(b.f32_2d("rope_sin_txt", config.head_dim, config.txt_len, true)).map_err(weight_err)?;
    let swap =
        nn(b.f32_2d("rope_swap", config.head_dim, config.head_dim, true)).map_err(weight_err)?;

    // vid patch embed
    let gathered = nn(b.get_rows(vid_in, patch_in_idx)).map_err(weight_err)?;
    let patches = nn(b.reshape_2d(gathered, config.patch_dim(), tokens)).map_err(weight_err)?;
    let mut vid =
        nn(b.project(wt(&weights, "vid_in.proj.weight")?, patches)).map_err(weight_err)?;
    vid = add_bias(&b, vid, wt(&weights, "vid_in.proj.bias")?).map_err(weight_err)?;
    let mut txt = nn(b.project(wt(&weights, "txt_in.weight")?, txt_in)).map_err(weight_err)?;
    txt = add_bias(&b, txt, wt(&weights, "txt_in.bias")?).map_err(weight_err)?;

    // timestep MLP
    let e = nn(b.project(wt(&weights, "emb_in.proj_in.weight")?, temb)).map_err(weight_err)?;
    let e = add_bias(&b, e, wt(&weights, "emb_in.proj_in.bias")?).map_err(weight_err)?;
    let e = nn(b.silu(e)).map_err(weight_err)?;
    let e = nn(b.project(wt(&weights, "emb_in.proj_hid.weight")?, e)).map_err(weight_err)?;
    let e = add_bias(&b, e, wt(&weights, "emb_in.proj_hid.bias")?).map_err(weight_err)?;
    let e = nn(b.silu(e)).map_err(weight_err)?;
    let e = nn(b.project(wt(&weights, "emb_in.proj_out.weight")?, e)).map_err(weight_err)?;
    let e = add_bias(&b, e, wt(&weights, "emb_in.proj_out.bias")?).map_err(weight_err)?;
    let emb = nn(b.reshape_3d(e, config.dim, 2, 3)).map_err(weight_err)?;

    let inputs = BlockInputs {
        swap,
        vid_cos,
        vid_sin,
        txt_cos,
        txt_sin,
        fwd,
        rev,
    };
    for layer in 0..config.layers {
        let (v, t) = block(&b, &weights, config, &plan, &inputs, layer, vid, txt, emb)?;
        vid = v;
        txt = t;
    }
    if config.output_ada {
        let normed = nn(b.rms_norm(vid, RMS_EPS)).map_err(weight_err)?;
        let normed = nn(b.mul(normed, wt(&weights, "vid_out_norm.weight")?)).map_err(weight_err)?;
        let shift = add_bias(
            &b,
            emb_col(&b, emb, config.dim, 0, 0)?,
            wt(&weights, "vid_out_ada.out_shift")?,
        )
        .map_err(weight_err)?;
        let scale = add_bias(
            &b,
            emb_col(&b, emb, config.dim, 0, 1)?,
            wt(&weights, "vid_out_ada.out_scale")?,
        )
        .map_err(weight_err)?;
        vid = nn(b.add(nn(b.mul(normed, scale))?, shift)).map_err(weight_err)?;
    }
    let out = nn(b.project(wt(&weights, "vid_out.proj.weight")?, vid)).map_err(weight_err)?;
    let out = add_bias(&b, out, wt(&weights, "vid_out.proj.bias")?).map_err(weight_err)?;
    let flat = nn(b.reshape_2d(out, 1, config.out_patch_dim() * tokens)).map_err(weight_err)?;
    let canvas = b.blank_f32_2d(1, w * h * t * config.out_channels);
    let scattered = nn(b.set_rows(canvas, flat, patch_out_idx)).map_err(weight_err)?;
    let output = shaped(
        nn(b.reshape_4d(scattered, w, h, t, config.out_channels)).map_err(weight_err)?,
        [w, h, t, config.out_channels],
        "seedvr2 dit out",
    )
    .map_err(weight_err)?;

    let mut graph = b.finish(output).map_err(weight_err)?;
    graph.set_f32("vid", vid_data).map_err(weight_err)?;
    graph.set_f32("txt", txt_data).map_err(weight_err)?;
    graph
        .set_f32("temb", &timestep_features(timestep, config.emb_sin))
        .map_err(weight_err)?;
    graph
        .set_i32("patch_in_idx", &plan.patch_in)
        .map_err(weight_err)?;
    graph
        .set_i32("patch_out_idx", &plan.patch_out)
        .map_err(weight_err)?;
    for parity in 0..2 {
        graph
            .set_i32(&format!("fwd_{parity}"), &plan.windows[parity].forward)
            .map_err(weight_err)?;
        graph
            .set_i32(&format!("rev_{parity}"), &plan.windows[parity].reverse)
            .map_err(weight_err)?;
        graph
            .set_f32(&format!("rope_cos_{parity}"), &plan.rope.vid_cos[parity])
            .map_err(weight_err)?;
        graph
            .set_f32(&format!("rope_sin_{parity}"), &plan.rope.vid_sin[parity])
            .map_err(weight_err)?;
    }
    graph
        .set_f32("rope_cos_txt", &plan.rope.txt_cos)
        .map_err(weight_err)?;
    graph
        .set_f32("rope_sin_txt", &plan.rope.txt_sin)
        .map_err(weight_err)?;
    graph
        .set_f32("rope_swap", &plan.rope.swap)
        .map_err(weight_err)?;
    for (name, data) in &loaded.tensors {
        match data {
            TensorData::F32(values) => graph.set_f32(name, values).map_err(weight_err)?,
            TensorData::Raw { bytes, .. } => graph.set_bytes(name, bytes).map_err(weight_err)?,
        }
    }
    let out = graph.compute().map_err(weight_err)?;
    if out.iter().any(|v| !v.is_finite()) {
        return Err(WeightError::NotBuilt(
            "seedvr2 dit produced a non-finite velocity".into(),
        ));
    }
    Ok(out)
}

fn weight_err(err: RuntimeError) -> WeightError {
    match err {
        RuntimeError::Message(message) => WeightError::NotBuilt(message),
        other => WeightError::NotBuilt(format!("seedvr2 dit graph failed: {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::BackendKind;
    use crate::weights::{write_f32_nd, F32Nd};
    use std::path::PathBuf;

    fn fill(
        name: &str,
        dims: &[i64],
        seed: f32,
        tensors: &mut Vec<F32Nd<'static>>,
        store: &mut Vec<(String, Vec<f32>)>,
    ) {
        let n = dims.iter().product::<i64>() as usize;
        let data: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 1.0) * 0.37 + seed).sin() * 0.05)
            .collect();
        let shape: [i64; 4] = {
            let mut s = [1i64; 4];
            for (i, d) in dims.iter().enumerate() {
                s[i] = *d;
            }
            s
        };
        let data_static: &'static [f32] = Box::leak(data.clone().into_boxed_slice());
        tensors.push(F32Nd {
            name: Box::leak(name.to_string().into_boxed_str()),
            ne: shape,
            data: data_static,
        });
        store.push((name.to_string(), data));
    }

    /// Write the smoke DiT as an F32 GGUF and read it back.
    fn write_smoke(
        dir: &PathBuf,
        config: &DitConfig,
    ) -> (std::sync::Arc<GgufWeights>, HashMap<String, Vec<f32>>) {
        let mut tensors: Vec<F32Nd<'static>> = Vec::new();
        let mut store: Vec<(String, Vec<f32>)> = Vec::new();
        let mut seed = 1.0f32;
        for (name, dims) in weight_layout(config) {
            fill(&name, &dims, seed, &mut tensors, &mut store);
            seed += 1.0;
        }
        // lang rope freqs: theta^(-2i/d) with d = 2 * freqs_len.
        let freqs: Vec<f32> = (0..config.freqs_len)
            .map(|i| 10000f32.powf(-2.0 * i as f32 / (2.0 * config.freqs_len as f32)))
            .collect();
        for (entry, _) in tensors.iter_mut().zip(store.iter_mut()) {
            if entry.name.ends_with("attn.rope.rope.freqs") {
                entry.data = Box::leak(freqs.clone().into_boxed_slice());
            }
        }
        for (name, data) in store.iter_mut() {
            if name.ends_with("attn.rope.rope.freqs") {
                *data = freqs.clone();
            }
        }
        let path = dir.join("seedvr2_smoke.gguf");
        write_f32_nd(&path, &tensors).unwrap();
        let weights = GgufWeights::open(&path).unwrap();
        let map: HashMap<String, Vec<f32>> = store.into_iter().collect();
        (weights, map)
    }

    fn at<'a>(map: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a Vec<f32> {
        map.get(name).unwrap_or_else(|| panic!("missing {name}"))
    }

    // ----- host reference (independent port of the reference MM-DiT) -----

    /// `x` is `[in, L]` flat in ggml order (`x[i + t * in]`); `w` is ggml
    /// `[in, out]` flat. Returns `[out, L]` in ggml order.
    fn h_linear(
        x: &[f32],
        l: usize,
        inp: usize,
        out: usize,
        w: &[f32],
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut y = vec![0.0; out * l];
        for t in 0..l {
            for o in 0..out {
                let mut acc = bias.map_or(0.0, |b| b[o]);
                for i in 0..inp {
                    acc += w[o * inp + i] * x[i + t * inp];
                }
                y[o + t * out] = acc;
            }
        }
        y
    }

    /// RMS norm over `c` channels of `x` `[c, l]` in ggml order.
    fn h_rms(x: &mut [f32], l: usize, c: usize, eps: f32) {
        for t in 0..l {
            let mut ss = 0.0;
            for i in 0..c {
                ss += x[i + t * c] * x[i + t * c];
            }
            let inv = (ss / c as f32 + eps).sqrt().recip();
            for i in 0..c {
                x[i + t * c] *= inv;
            }
        }
    }

    fn h_silu(x: &mut [f32]) {
        for v in x.iter_mut() {
            *v = *v / (1.0 + (-*v).exp());
        }
    }

    /// Independent port of `_make_windows((4,3,3))`.
    fn h_windows(t: i64, h: i64, w: i64, shift: bool) -> Vec<([i64; 2], [i64; 2], [i64; 2])> {
        let scale = (3600.0 / (h * w) as f64).sqrt();
        let rh = (h as f64 * scale).round() as i64;
        let rw = (w as f64 * scale).round() as i64;
        let cd = |a: i64, b: i64| (a + b - 1) / b;
        let (wh, ww, wt) = (cd(rh, 3), cd(rw, 3), cd(t.min(30), 4));
        let (st, sh, sw, nt, nh, nw) = if shift {
            let st = if wt < t { 0.5 } else { 0.0 };
            let sh = if wh < h { 0.5 } else { 0.0 };
            let sw = if ww < w { 0.5 } else { 0.0 };
            (
                st,
                sh,
                sw,
                if st > 0.0 {
                    ((t as f64 - st) / wt as f64).ceil() as i64 + 1
                } else {
                    1
                },
                if sh > 0.0 {
                    ((h as f64 - sh) / wh as f64).ceil() as i64 + 1
                } else {
                    1
                },
                if sw > 0.0 {
                    ((w as f64 - sw) / ww as f64).ceil() as i64 + 1
                } else {
                    1
                },
            )
        } else {
            (0.0, 0.0, 0.0, cd(t, wt), cd(h, wh), cd(w, ww))
        };
        let mut out = Vec::new();
        for iw in 0..nw {
            let w0 = (((iw as f64 - sw) * ww as f64) as i64).max(0);
            let w1 = ((((iw as f64 - sw) + 1.0) * ww as f64) as i64).min(w);
            if w1 <= w0 {
                continue;
            }
            for ih in 0..nh {
                let h0 = (((ih as f64 - sh) * wh as f64) as i64).max(0);
                let h1 = ((((ih as f64 - sh) + 1.0) * wh as f64) as i64).min(h);
                if h1 <= h0 {
                    continue;
                }
                for it in 0..nt {
                    let t0 = (((it as f64 - st) * wt as f64) as i64).max(0);
                    let t1 = ((((it as f64 - st) + 1.0) * wt as f64) as i64).min(t);
                    if t1 <= t0 {
                        continue;
                    }
                    out.push(([t0, t1], [h0, h1], [w0, w1]));
                }
            }
        }
        out
    }

    /// Window-local rope angles for one token, `[rot_dim]`.
    fn h_angles(freqs: &[f32], txt_len: i64, pos: (i64, i64, i64), is_txt: bool) -> Vec<f64> {
        let per_axis = 2 * freqs.len();
        let mut row = vec![0.0f64; per_axis * 3];
        let fill = |axis: usize, p: f64, row: &mut [f64]| {
            for j in 0..freqs.len() {
                let a = p * freqs[j] as f64;
                row[axis * per_axis + 2 * j] = a;
                row[axis * per_axis + 2 * j + 1] = a;
            }
        };
        if is_txt {
            for axis in 0..3 {
                fill(axis, pos.0 as f64, &mut row);
            }
        } else {
            fill(0, (txt_len + pos.0) as f64, &mut row);
            fill(1, pos.1 as f64, &mut row);
            fill(2, pos.2 as f64, &mut row);
        }
        row
    }

    fn h_rope_token(x: &mut [f32], angles: &[f64]) {
        let rot = angles.len();
        for i in 0..rot / 2 {
            let (c, s) = (angles[2 * i].cos() as f32, angles[2 * i].sin() as f32);
            let (a, b) = (x[2 * i], x[2 * i + 1]);
            x[2 * i] = a * c - b * s;
            x[2 * i + 1] = b * c + a * s;
        }
    }

    fn h_sdpa(q: &[f32], k: &[f32], v: &[f32], lq: usize, lk: usize, head: usize) -> Vec<f32> {
        // q [head, lq], k/v [head, lk], ggml order (`x[c + t * head]`).
        let scale = (head as f32).sqrt().recip();
        let mut out = vec![0.0; head * lq];
        for tq in 0..lq {
            let mut scores = vec![0.0f32; lk];
            let mut max = f32::NEG_INFINITY;
            for tk in 0..lk {
                let mut s = 0.0;
                for c in 0..head {
                    s += q[c + tq * head] * k[c + tk * head];
                }
                scores[tk] = s * scale;
                max = max.max(scores[tk]);
            }
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - max).exp();
                sum += *s;
            }
            for c in 0..head {
                let mut acc = 0.0;
                for tk in 0..lk {
                    acc += v[c + tk * head] * scores[tk] / sum;
                }
                out[c + tq * head] = acc;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn h_attention(
        map: &HashMap<String, Vec<f32>>,
        config: &DitConfig,
        freqs: &[f32],
        layer: usize,
        vid: &[f32],
        txt: &[f32],
        t: i64,
        hp: i64,
        wp: i64,
    ) -> (Vec<f32>, Vec<f32>) {
        let p = format!("blocks.{layer}");
        let side = config.side(layer);
        let txt_side = if side == "all" { "all" } else { "txt" };
        let (dim, head, heads) = (
            config.dim as usize,
            config.head_dim as usize,
            config.heads as usize,
        );
        let inner = head * heads;
        let lv = vid.len() / dim;
        let lt = config.txt_len as usize;
        let qkv_v = h_linear(
            vid,
            lv,
            dim,
            3 * inner,
            at(map, &format!("{p}.attn.proj_qkv.{side}.weight")),
            None,
        );
        let qkv_t = h_linear(
            txt,
            lt,
            dim,
            3 * inner,
            at(map, &format!("{p}.attn.proj_qkv.{txt_side}.weight")),
            None,
        );
        let windows = h_windows(t, hp, wp, layer % 2 == 1);
        // per-head q/k/v channels: [head][token][channel]
        let mut v_out = vec![0.0f32; dim * lv];
        let mut t_out = vec![0.0f32; dim * lt];
        for hd in 0..heads {
            // [head, token] slices for this head, ggml order
            let slice = |src: &[f32], part: usize, l: usize| -> Vec<f32> {
                let mut o = vec![0.0; head * l];
                for c in 0..head {
                    for t in 0..l {
                        o[c + t * head] = src[(part * inner + hd * head + c) + t * (3 * inner)];
                    }
                }
                o
            };
            let mut q_v = slice(&qkv_v, 0, lv);
            let mut k_v = slice(&qkv_v, 1, lv);
            let v_v = slice(&qkv_v, 2, lv);
            let mut q_t = slice(&qkv_t, 0, lt);
            let mut k_t = slice(&qkv_t, 1, lt);
            let v_t = slice(&qkv_t, 2, lt);
            // qk rms norm with weight
            let norm = |x: &mut [f32], name: &str| {
                h_rms(x, x.len() / head, head, RMS_EPS);
                let w = at(map, name);
                for t in 0..x.len() / head {
                    for c in 0..head {
                        x[c + t * head] *= w[c];
                    }
                }
            };
            norm(&mut q_v, &format!("{p}.attn.norm_q.{side}.weight"));
            norm(&mut k_v, &format!("{p}.attn.norm_k.{side}.weight"));
            norm(&mut q_t, &format!("{p}.attn.norm_q.{txt_side}.weight"));
            norm(&mut k_t, &format!("{p}.attn.norm_k.{txt_side}.weight"));
            // text rope (same for every window)
            for t in 0..lt {
                let angles = h_angles(freqs, config.txt_len, (t as i64, 0, 0), true);
                let mut qc: Vec<f32> = (0..head).map(|c| q_t[c + t * head]).collect();
                let mut kc: Vec<f32> = (0..head).map(|c| k_t[c + t * head]).collect();
                h_rope_token(&mut qc, &angles);
                h_rope_token(&mut kc, &angles);
                for c in 0..head {
                    q_t[c + t * head] = qc[c];
                    k_t[c + t * head] = kc[c];
                }
            }
            for ([t0, t1], [h0, h1], [w0, w1]) in &windows {
                // gather window tokens in (t, y, x) order
                let mut ids = Vec::new();
                for tt in *t0..*t1 {
                    for yy in *h0..*h1 {
                        for xx in *w0..*w1 {
                            ids.push(((tt * hp + yy) * wp + xx) as usize);
                        }
                    }
                }
                let lw = ids.len();
                let gather = |src: &[f32]| -> Vec<f32> {
                    let mut o = vec![0.0; head * lw];
                    for (pos, &tok) in ids.iter().enumerate() {
                        for c in 0..head {
                            o[c + pos * head] = src[c + tok * head];
                        }
                    }
                    o
                };
                let mut q_w = gather(&q_v);
                let mut k_w = gather(&k_v);
                let v_w = gather(&v_v);
                // window-local rope
                let mut local = Vec::new();
                for tt in *t0..*t1 {
                    for yy in *h0..*h1 {
                        for xx in *w0..*w1 {
                            local.push((tt - t0, yy - h0, xx - w0));
                        }
                    }
                }
                for (pos, &(pt, py, px)) in local.iter().enumerate() {
                    let angles = h_angles(freqs, config.txt_len, (pt, py, px), false);
                    let mut qc: Vec<f32> = (0..head).map(|c| q_w[c + pos * head]).collect();
                    let mut kc: Vec<f32> = (0..head).map(|c| k_w[c + pos * head]).collect();
                    h_rope_token(&mut qc, &angles);
                    h_rope_token(&mut kc, &angles);
                    for c in 0..head {
                        q_w[c + pos * head] = qc[c];
                        k_w[c + pos * head] = kc[c];
                    }
                }
                // concat text
                let cat = |a: &[f32], b: &[f32], la: usize, lb: usize| -> Vec<f32> {
                    let mut o = vec![0.0; head * (la + lb)];
                    for t in 0..la + lb {
                        for c in 0..head {
                            o[c + t * head] = if t < la {
                                a[c + t * head]
                            } else {
                                b[c + (t - la) * head]
                            };
                        }
                    }
                    o
                };
                let q_c = cat(&q_w, &q_t, lw, lt);
                let k_c = cat(&k_w, &k_t, lw, lt);
                let v_c = cat(&v_w, &v_t, lw, lt);
                let out = h_sdpa(&q_c, &k_c, &v_c, lw + lt, lw + lt, head);
                for (pos, &tok) in ids.iter().enumerate() {
                    for c in 0..head {
                        v_out[(hd * head + c) + tok * dim] = out[c + pos * head];
                    }
                }
                for t in 0..lt {
                    for c in 0..head {
                        t_out[(hd * head + c) + t * dim] +=
                            out[c + (lw + t) * head] / windows.len() as f32;
                    }
                }
            }
        }
        let v_proj = h_linear(
            &v_out,
            lv,
            inner,
            dim,
            at(map, &format!("{p}.attn.proj_out.{side}.weight")),
            Some(at(map, &format!("{p}.attn.proj_out.{side}.bias"))),
        );
        let t_proj = h_linear(
            &t_out,
            lt,
            inner,
            dim,
            at(map, &format!("{p}.attn.proj_out.{txt_side}.weight")),
            Some(at(map, &format!("{p}.attn.proj_out.{txt_side}.bias"))),
        );
        (v_proj, t_proj)
    }

    #[test]
    fn smoke_dit_matches_the_host_reference() {
        let config = DitConfig::smoke();
        let dir = std::env::temp_dir().join(format!("danmo-seedvr2-dit-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (weights, map) = write_smoke(&dir, &config);
        let inferred = DitConfig::infer(&weights, config.txt_len).unwrap();
        assert_eq!(inferred.layers, config.layers);
        assert_eq!(inferred.mm_layers, config.mm_layers);
        assert_eq!(inferred.dim, config.dim);
        assert_eq!(inferred.heads, config.heads);
        assert_eq!(inferred.head_dim, config.head_dim);
        assert!(inferred.swiglu);
        assert!(inferred.output_ada);
        assert!(inferred.last_layer_vid_only);
        assert!(!inferred.pixel_rope);

        let (w, h) = (48i64, 48i64);
        let vid: Vec<f32> = (0..(w * h * config.vid_channels) as usize)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();
        let txt: Vec<f32> = (0..(config.txt_in_dim * config.txt_len) as usize)
            .map(|i| (i as f32 * 0.031).cos() * 0.4)
            .collect();
        let loaded = load(&weights, &config).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = run(&ctx, &config, &loaded, &vid, w, h, 1, &txt, 1000.0).unwrap();
        let want = host_forward(&config, &map, &vid, &txt, 1, w, h);

        assert_eq!(got.len(), want.len());
        let mut worst = 0f32;
        for (g, w_) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w_).abs());
        }
        assert!(worst < 1e-4, "dit max abs diff {worst}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    struct HostTrace {
        vid0: Vec<f32>,
        txt0: Vec<f32>,
        emb: Vec<f32>,
        blocks: Vec<(Vec<f32>, Vec<f32>)>,
        out: Vec<f32>,
    }

    fn host_forward(
        config: &DitConfig,
        map: &HashMap<String, Vec<f32>>,
        vid: &[f32],
        txt: &[f32],
        t: i64,
        w: i64,
        h: i64,
    ) -> Vec<f32> {
        host_trace(config, map, vid, txt, t, w, h).out
    }

    fn host_trace(
        config: &DitConfig,
        map: &HashMap<String, Vec<f32>>,
        vid: &[f32],
        txt: &[f32],
        t: i64,
        w: i64,
        h: i64,
    ) -> HostTrace {
        let (hp, wp) = (h / 2, w / 2);
        let tokens = (t * hp * wp) as usize;
        let dim = config.dim as usize;
        let freqs = at(&map, "blocks.0.attn.rope.rope.freqs").clone();
        // patchify: patches[e + token * patch_dim] in ggml order, tokens
        // `[T][H][W]` frame outer.
        let mut patches = vec![0.0f32; config.patch_dim() as usize * tokens];
        for tt in 0..t {
            for yy in 0..hp {
                for xx in 0..wp {
                    let tok = (tt * hp * wp + yy * wp + xx) as usize;
                    let mut e = 0usize;
                    for hh in 0..2 {
                        for ww in 0..2 {
                            for c in 0..config.vid_channels {
                                patches[e + tok * config.patch_dim() as usize] = vid[(xx * 2
                                    + ww
                                    + w * (yy * 2 + hh)
                                    + w * h * tt
                                    + w * h * t * c)
                                    as usize];
                                e += 1;
                            }
                        }
                    }
                }
            }
        }
        let mut vid_t = h_linear(
            &patches,
            tokens,
            config.patch_dim() as usize,
            dim,
            at(&map, "vid_in.proj.weight"),
            Some(at(&map, "vid_in.proj.bias")),
        );
        let mut txt_t = h_linear(
            &txt,
            config.txt_len as usize,
            config.txt_in_dim as usize,
            dim,
            at(&map, "txt_in.weight"),
            Some(at(&map, "txt_in.bias")),
        );
        // timestep embedding
        let temb = timestep_features(1000.0, config.emb_sin);
        let mut e = h_linear(
            &temb,
            1,
            config.emb_sin as usize,
            dim,
            at(&map, "emb_in.proj_in.weight"),
            Some(at(&map, "emb_in.proj_in.bias")),
        );
        h_silu(&mut e);
        let mut e = h_linear(
            &e,
            1,
            dim,
            dim,
            at(&map, "emb_in.proj_hid.weight"),
            Some(at(&map, "emb_in.proj_hid.bias")),
        );
        h_silu(&mut e);
        let emb = h_linear(
            &e,
            1,
            dim,
            6 * dim,
            at(&map, "emb_in.proj_out.weight"),
            Some(at(&map, "emb_in.proj_out.bias")),
        );
        let emb_col = |layer: usize, mode: usize| -> Vec<f32> {
            emb[(mode * 2 + layer) * dim..(mode * 2 + layer + 1) * dim].to_vec()
        };
        let modulate = |x: &mut [f32], l: usize, ada: &str, layer: &str, li: usize| {
            let shift = at(&map, &format!("{ada}.{layer}_shift"));
            let scale = at(&map, &format!("{ada}.{layer}_scale"));
            let es = emb_col(li, 0);
            let ec = emb_col(li, 1);
            for t in 0..l {
                for c in 0..dim {
                    x[c + t * dim] = x[c + t * dim] * (ec[c] + scale[c]) + (es[c] + shift[c]);
                }
            }
        };
        let gate = |x: &mut [f32], l: usize, ada: &str, layer: &str, li: usize| {
            let g = at(&map, &format!("{ada}.{layer}_gate"));
            let eg = emb_col(li, 2);
            for t in 0..l {
                for c in 0..dim {
                    x[c + t * dim] *= eg[c] + g[c];
                }
            }
        };
        let vid0 = vid_t.clone();
        let txt0 = txt_t.clone();
        let mut blocks: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        for layer in 0..config.layers {
            let p = format!("blocks.{layer}");
            let side = config.side(layer);
            let last = config.last_layer_vid_only && layer + 1 == config.layers;
            let ada_v = format!("{p}.ada.{side}");
            let ada_t = if side == "all" {
                ada_v.clone()
            } else {
                format!("{p}.ada.txt")
            };
            let mut vid_n = vid_t.clone();
            h_rms(&mut vid_n, tokens, dim, RMS_EPS);
            let mut txt_n = txt_t.clone();
            h_rms(&mut txt_n, config.txt_len as usize, dim, RMS_EPS);
            modulate(&mut vid_n, tokens, &ada_v, "attn", 0);
            if !last {
                modulate(&mut txt_n, config.txt_len as usize, &ada_t, "attn", 0);
            }
            let (mut av, mut at_) =
                h_attention(&map, &config, &freqs, layer, &vid_n, &txt_n, t, hp, wp);
            gate(&mut av, tokens, &ada_v, "attn", 0);
            for i in 0..vid_t.len() {
                vid_t[i] += av[i];
            }
            if !last {
                gate(&mut at_, config.txt_len as usize, &ada_t, "attn", 0);
                for i in 0..txt_t.len() {
                    txt_t[i] += at_[i];
                }
            }
            let mut vid_n = vid_t.clone();
            h_rms(&mut vid_n, tokens, dim, RMS_EPS);
            modulate(&mut vid_n, tokens, &ada_v, "mlp", 1);
            let gate_w = h_linear(
                &vid_n,
                tokens,
                dim,
                config.mlp_hidden as usize,
                at(&map, &format!("{p}.mlp.{side}.proj_in_gate.weight")),
                None,
            );
            let up_w = h_linear(
                &vid_n,
                tokens,
                dim,
                config.mlp_hidden as usize,
                at(&map, &format!("{p}.mlp.{side}.proj_in.weight")),
                None,
            );
            let mut hidden = gate_w;
            h_silu(&mut hidden);
            for i in 0..hidden.len() {
                hidden[i] *= up_w[i];
            }
            let mut vid_m = h_linear(
                &hidden,
                tokens,
                config.mlp_hidden as usize,
                dim,
                at(&map, &format!("{p}.mlp.{side}.proj_out.weight")),
                None,
            );
            gate(&mut vid_m, tokens, &ada_v, "mlp", 1);
            for i in 0..vid_t.len() {
                vid_t[i] += vid_m[i];
            }
            if !last {
                let mut txt_n = txt_t.clone();
                h_rms(&mut txt_n, config.txt_len as usize, dim, RMS_EPS);
                modulate(&mut txt_n, config.txt_len as usize, &ada_t, "mlp", 1);
                let txt_side = if side == "all" { "all" } else { "txt" };
                let gate_w = h_linear(
                    &txt_n,
                    config.txt_len as usize,
                    dim,
                    config.mlp_hidden as usize,
                    at(&map, &format!("{p}.mlp.{txt_side}.proj_in_gate.weight")),
                    None,
                );
                let up_w = h_linear(
                    &txt_n,
                    config.txt_len as usize,
                    dim,
                    config.mlp_hidden as usize,
                    at(&map, &format!("{p}.mlp.{txt_side}.proj_in.weight")),
                    None,
                );
                let mut hidden = gate_w;
                h_silu(&mut hidden);
                for i in 0..hidden.len() {
                    hidden[i] *= up_w[i];
                }
                let mut txt_m = h_linear(
                    &hidden,
                    config.txt_len as usize,
                    config.mlp_hidden as usize,
                    dim,
                    at(&map, &format!("{p}.mlp.{txt_side}.proj_out.weight")),
                    None,
                );
                gate(&mut txt_m, config.txt_len as usize, &ada_t, "mlp", 1);
                for i in 0..txt_t.len() {
                    txt_t[i] += txt_m[i];
                }
            }
            blocks.push((vid_t.clone(), txt_t.clone()));
        }
        // output ada
        let mut vid_n = vid_t.clone();
        h_rms(&mut vid_n, tokens, dim, RMS_EPS);
        let norm_w = at(&map, "vid_out_norm.weight");
        let out_shift = at(&map, "vid_out_ada.out_shift");
        let out_scale = at(&map, "vid_out_ada.out_scale");
        let es = emb_col(0, 0);
        let ec = emb_col(0, 1);
        for t in 0..tokens {
            for c in 0..dim {
                vid_n[c + t * dim] = vid_n[c + t * dim] * norm_w[c] * (ec[c] + out_scale[c])
                    + (es[c] + out_shift[c]);
            }
        }
        let out = h_linear(
            &vid_n,
            tokens,
            dim,
            config.out_patch_dim() as usize,
            at(&map, "vid_out.proj.weight"),
            Some(at(&map, "vid_out.proj.bias")),
        );
        // unpatchify: out[e + token * out_patch_dim] in ggml order
        let mut want = vec![0.0f32; (w * h * t * config.out_channels) as usize];
        for tt in 0..t {
            for yy in 0..hp {
                for xx in 0..wp {
                    let tok = (tt * hp * wp + yy * wp + xx) as usize;
                    let mut e = 0usize;
                    for hh in 0..2 {
                        for ww in 0..2 {
                            for c in 0..config.out_channels {
                                want[(xx * 2 + ww + w * (yy * 2 + hh) + w * h * tt + w * h * t * c)
                                    as usize] = out[e + tok * config.out_patch_dim() as usize];
                                e += 1;
                            }
                        }
                    }
                }
            }
        }
        HostTrace {
            vid0,
            txt0,
            emb,
            blocks,
            out: want,
        }
    }

    /// Graph-side stage probe: 0 vid embed, 1 txt embed, 2 timestep emb,
    /// 3 + 2i vid after block i, 4 + 2i txt after block i.
    fn trace_run(
        ctx: &GgmlContext,
        config: &DitConfig,
        loaded: &LoadedDit,
        vid: &[f32],
        t: i64,
        w: i64,
        h: i64,
        txt: &[f32],
        stage: usize,
    ) -> Vec<f32> {
        let freqs: Vec<f32> = loaded
            .tensors
            .iter()
            .find_map(|(n, d)| match (n.as_str(), d) {
                ("blocks.0.attn.rope.rope.freqs", TensorData::F32(v)) => Some(v.clone()),
                _ => None,
            })
            .unwrap();
        let plan = plan(config, t, w, h, &freqs).unwrap();
        let tokens = t * (w / config.patch) * (h / config.patch);
        let mut b = ctx.builder(GRAPH_MEM).unwrap();
        let weights = declare_weights(&mut b, config, loaded).unwrap();
        let vid_in = nn(b.f32_2d("vid", 1, w * h * t * config.vid_channels, true)).unwrap();
        let txt_in = nn(b.f32_2d("txt", config.txt_in_dim, config.txt_len, true)).unwrap();
        let temb = nn(b.f32_2d("temb", config.emb_sin, 1, true)).unwrap();
        let patch_in_idx = nn(b.i32_1d("patch_in_idx", tokens * config.patch_dim(), true)).unwrap();
        let mut fwd = [std::ptr::null_mut(); 2];
        let mut rev = [std::ptr::null_mut(); 2];
        let mut vid_cos = [std::ptr::null_mut(); 2];
        let mut vid_sin = [std::ptr::null_mut(); 2];
        for parity in 0..2 {
            fwd[parity] = nn(b.i32_1d(&format!("fwd_{parity}"), tokens, true)).unwrap();
            rev[parity] = nn(b.i32_1d(&format!("rev_{parity}"), tokens, true)).unwrap();
            vid_cos[parity] =
                nn(b.f32_2d(&format!("rope_cos_{parity}"), config.head_dim, tokens, true)).unwrap();
            vid_sin[parity] =
                nn(b.f32_2d(&format!("rope_sin_{parity}"), config.head_dim, tokens, true)).unwrap();
        }
        let txt_cos = nn(b.f32_2d("rope_cos_txt", config.head_dim, config.txt_len, true)).unwrap();
        let txt_sin = nn(b.f32_2d("rope_sin_txt", config.head_dim, config.txt_len, true)).unwrap();
        let swap = nn(b.f32_2d("rope_swap", config.head_dim, config.head_dim, true)).unwrap();

        let gathered = nn(b.get_rows(vid_in, patch_in_idx)).unwrap();
        let patches = nn(b.reshape_2d(gathered, config.patch_dim(), tokens)).unwrap();
        let vid0 = nn(b.project(wt(&weights, "vid_in.proj.weight").unwrap(), patches)).unwrap();
        let vid0 = add_bias(&b, vid0, wt(&weights, "vid_in.proj.bias").unwrap()).unwrap();
        let txt0 = nn(b.project(wt(&weights, "txt_in.weight").unwrap(), txt_in)).unwrap();
        let txt0 = add_bias(&b, txt0, wt(&weights, "txt_in.bias").unwrap()).unwrap();
        let e = nn(b.project(wt(&weights, "emb_in.proj_in.weight").unwrap(), temb)).unwrap();
        let e = add_bias(&b, e, wt(&weights, "emb_in.proj_in.bias").unwrap()).unwrap();
        let e = nn(b.silu(e)).unwrap();
        let e = nn(b.project(wt(&weights, "emb_in.proj_hid.weight").unwrap(), e)).unwrap();
        let e = add_bias(&b, e, wt(&weights, "emb_in.proj_hid.bias").unwrap()).unwrap();
        let e = nn(b.silu(e)).unwrap();
        let e = nn(b.project(wt(&weights, "emb_in.proj_out.weight").unwrap(), e)).unwrap();
        let e = add_bias(&b, e, wt(&weights, "emb_in.proj_out.bias").unwrap()).unwrap();
        let emb = nn(b.reshape_3d(e, config.dim, 2, 3)).unwrap();

        let output = match stage {
            0 => vid0,
            1 => txt0,
            2 => e,
            _ => {
                let inputs = BlockInputs {
                    swap,
                    vid_cos,
                    vid_sin,
                    txt_cos,
                    txt_sin,
                    fwd,
                    rev,
                };
                let mut v = vid0;
                let mut t = txt0;
                let mut chosen = None;
                for layer in 0..config.layers {
                    let (nv, nt) =
                        block(&b, &weights, config, &plan, &inputs, layer, v, t, emb).unwrap();
                    v = nv;
                    t = nt;
                    if stage == 3 + 2 * layer {
                        chosen = Some(v);
                        break;
                    }
                    if stage == 4 + 2 * layer {
                        chosen = Some(t);
                        break;
                    }
                }
                chosen.unwrap_or_else(|| panic!("bad probe stage {stage}"))
            }
        };
        let mut graph = b.finish(output).unwrap();
        graph.set_f32("vid", vid).unwrap();
        graph.set_f32("txt", txt).unwrap();
        graph
            .set_f32("temb", &timestep_features(1000.0, config.emb_sin))
            .unwrap();
        graph.set_i32("patch_in_idx", &plan.patch_in).unwrap();
        for parity in 0..2 {
            graph
                .set_i32(&format!("fwd_{parity}"), &plan.windows[parity].forward)
                .unwrap();
            graph
                .set_i32(&format!("rev_{parity}"), &plan.windows[parity].reverse)
                .unwrap();
            graph
                .set_f32(&format!("rope_cos_{parity}"), &plan.rope.vid_cos[parity])
                .unwrap();
            graph
                .set_f32(&format!("rope_sin_{parity}"), &plan.rope.vid_sin[parity])
                .unwrap();
        }
        graph.set_f32("rope_cos_txt", &plan.rope.txt_cos).unwrap();
        graph.set_f32("rope_sin_txt", &plan.rope.txt_sin).unwrap();
        graph.set_f32("rope_swap", &plan.rope.swap).unwrap();
        for (name, data) in &loaded.tensors {
            match data {
                TensorData::F32(values) => graph.set_f32(name, values).unwrap(),
                TensorData::Raw { bytes, .. } => graph.set_bytes(name, bytes).unwrap(),
            }
        }
        graph.compute().unwrap()
    }

    #[test]
    fn dit_stage_trace_matches_host() {
        let config = DitConfig::smoke();
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-dit-probe-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (weights, map) = write_smoke(&dir, &config);
        let (w, h) = (4i64, 4i64);
        let vid: Vec<f32> = (0..(w * h * config.vid_channels) as usize)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();
        let txt: Vec<f32> = (0..(config.txt_in_dim * config.txt_len) as usize)
            .map(|i| (i as f32 * 0.031).cos() * 0.4)
            .collect();
        let loaded = load(&weights, &config).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let trace = host_trace(&config, &map, &vid, &txt, 1, w, h);
        let pairs: Vec<(String, Vec<f32>, Vec<f32>)> = vec![
            (
                "vid_embed".into(),
                trace_run(&ctx, &config, &loaded, &vid, 1, w, h, &txt, 0),
                trace.vid0.clone(),
            ),
            (
                "txt_embed".into(),
                trace_run(&ctx, &config, &loaded, &vid, 1, w, h, &txt, 1),
                trace.txt0.clone(),
            ),
            (
                "timestep_emb".into(),
                trace_run(&ctx, &config, &loaded, &vid, 1, w, h, &txt, 2),
                trace.emb.clone(),
            ),
        ];
        let mut pairs = pairs;
        for layer in 0..config.layers {
            pairs.push((
                format!("block{layer}_vid"),
                trace_run(&ctx, &config, &loaded, &vid, 1, w, h, &txt, 3 + 2 * layer),
                trace.blocks[layer].0.clone(),
            ));
            pairs.push((
                format!("block{layer}_txt"),
                trace_run(&ctx, &config, &loaded, &vid, 1, w, h, &txt, 4 + 2 * layer),
                trace.blocks[layer].1.clone(),
            ));
        }
        let mut worst_stage = String::new();
        let mut worst_diff = 0f32;
        for (name, got, want) in &pairs {
            assert_eq!(got.len(), want.len(), "{name} length");
            let mut diff = 0f32;
            for (g, w_) in got.iter().zip(want.iter()) {
                diff = diff.max((g - w_).abs());
            }
            eprintln!("stage {name}: max abs diff {diff}");
            if diff > worst_diff {
                worst_diff = diff;
                worst_stage = name.clone();
            }
        }
        assert!(
            worst_diff < 1e-4,
            "worst stage {worst_stage} diff {worst_diff}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_dit_tiny_grid() {
        let config = DitConfig::smoke();
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-dit-tiny-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (weights, map) = write_smoke(&dir, &config);
        let (w, h) = (4i64, 4i64);
        let vid: Vec<f32> = (0..(w * h * config.vid_channels) as usize)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();
        let txt: Vec<f32> = (0..(config.txt_in_dim * config.txt_len) as usize)
            .map(|i| (i as f32 * 0.031).cos() * 0.4)
            .collect();
        let loaded = load(&weights, &config).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = run(&ctx, &config, &loaded, &vid, w, h, 1, &txt, 1000.0).unwrap();
        let want = host_forward(&config, &map, &vid, &txt, 1, w, h);
        let mut worst = 0f32;
        for (g, w_) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w_).abs());
        }
        assert!(worst < 1e-4, "tiny dit max abs diff {worst}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn smoke_dit_multi_frame_matches_the_host_reference() {
        let config = DitConfig::smoke();
        let dir =
            std::env::temp_dir().join(format!("danmo-seedvr2-dit-frames-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (weights, map) = write_smoke(&dir, &config);
        // t = 5: the temporal window is 2 wide, so both parities split the
        // frame axis instead of degenerating to one window.
        let (w, h, t) = (16i64, 16i64, 5i64);
        let vid: Vec<f32> = (0..(w * h * t * config.vid_channels) as usize)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();
        let txt: Vec<f32> = (0..(config.txt_in_dim * config.txt_len) as usize)
            .map(|i| (i as f32 * 0.031).cos() * 0.4)
            .collect();
        let loaded = load(&weights, &config).unwrap();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let got = run(&ctx, &config, &loaded, &vid, w, h, t, &txt, 1000.0).unwrap();
        let want = host_forward(&config, &map, &vid, &txt, t, w, h);
        assert_eq!(got.len(), (w * h * t * config.out_channels) as usize);
        assert_eq!(got.len(), want.len());
        let mut worst = 0f32;
        for (g, w_) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w_).abs());
        }
        assert!(worst < 1e-4, "multi-frame dit max abs diff {worst}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
