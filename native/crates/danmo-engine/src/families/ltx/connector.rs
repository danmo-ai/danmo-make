//! LTX 2.3 text-embedding connectors: the dual projection from the Gemma
//! state stack plus the two 8-layer 1-D connector towers whose outputs
//! condition the video and audio towers of the DiT.
//!
//! Semantics are a port of main's `_GemmaFeaturesExtractor` /
//! `_TextEncoderConnector` (`text_encoder_mlx.py`):
//! - `video = Linear(stack * sqrt(video_dim / gemma_dim)) + bias` and the
//!   audio twin; the projection weights are the published connectors
//!   safetensors (`text_embedding_projection.*`), while the tower weights
//!   live inside the DiT GGUF as `{video,audio}_embeddings_connector.*`.
//! - Before the blocks, padding rows are replaced by the learnable registers
//!   tiled cyclically across the sequence (`_replace_padding_with_registers`
//!   with left padding: valid tokens sit at the tail).
//! - Each block: pre-RMS (no weight) -> attention (`to_q/k/v` with bias, q/k
//!   RMS norm over the full dim, the reference split RoPE tables
//!   (`theta^linspace(0,1,dim/2)` scaled by `pos/max_pos*2-1`), full
//!   attention, per-head gate `2*sigmoid(to_gate_logits)`, `to_out`) ->
//!   residual; pre-RMS -> `gelu_tanh(ff.net.0.proj)` -> `ff.net.2` ->
//!   residual.
//! - A final weightless RMS norm closes each tower.

use super::gemma::GemmaConditioning;
use crate::runtime::{
    flash_attend, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor, TYPE_F32,
};
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 10_000.0;
const ROPE_MAX_POS: i64 = 4096;
const REGISTERS: i64 = 128;
const DEPTH: i64 = 8;

#[derive(Clone, Copy, Debug)]
pub struct TowerConfig {
    pub dim: i64,
    pub heads: i64,
    pub head_dim: i64,
}

#[derive(Clone, Copy, Debug)]
pub struct ConnectorConfig {
    /// Gemma hidden size (3840); the projection scale divides by it.
    pub gemma_dim: i64,
    /// Stacked input width: `gemma_dim * states` (188160 for 49 states).
    pub stack_dim: i64,
    pub video: TowerConfig,
    pub audio: TowerConfig,
}

impl ConnectorConfig {
    pub fn smoke() -> Self {
        Self {
            gemma_dim: 8,
            stack_dim: 8 * 49,
            video: TowerConfig {
                dim: 8,
                heads: 2,
                head_dim: 4,
            },
            audio: TowerConfig {
                dim: 8,
                heads: 2,
                head_dim: 4,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// weight layout
// ---------------------------------------------------------------------------

/// Which file a connector weight lives in.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Source {
    /// The connectors safetensors (`text_embedding_projection.*`).
    Projection,
    /// The DiT GGUF (`{video,audio}_embeddings_connector.*`).
    Dit,
}

pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) ne0: i64,
    pub(crate) ne1: i64,
    pub(crate) source: Source,
}

fn tower_prefix(video: bool) -> &'static str {
    if video {
        "video_embeddings_connector"
    } else {
        "audio_embeddings_connector"
    }
}

pub(crate) fn layout(config: &ConnectorConfig) -> Vec<Entry> {
    let mut out = Vec::new();
    for (video, tower) in [(true, config.video), (false, config.audio)] {
        let p = tower_prefix(video);
        out.push(Entry {
            name: format!("{p}.learnable_registers"),
            ne0: tower.dim,
            ne1: REGISTERS,
            source: Source::Dit,
        });
        for i in 0..DEPTH {
            let b = format!("{p}.transformer_1d_blocks.{i}");
            for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
                out.push(Entry {
                    name: format!("{b}.attn1.{proj}.weight"),
                    ne0: tower.dim,
                    ne1: tower.dim,
                    source: Source::Dit,
                });
                out.push(Entry {
                    name: format!("{b}.attn1.{proj}.bias"),
                    ne0: tower.dim,
                    ne1: 1,
                    source: Source::Dit,
                });
            }
            for norm in ["q_norm", "k_norm"] {
                out.push(Entry {
                    name: format!("{b}.attn1.{norm}.weight"),
                    ne0: tower.dim,
                    ne1: 1,
                    source: Source::Dit,
                });
            }
            out.push(Entry {
                name: format!("{b}.attn1.to_gate_logits.weight"),
                ne0: tower.dim,
                ne1: tower.heads,
                source: Source::Dit,
            });
            out.push(Entry {
                name: format!("{b}.attn1.to_gate_logits.bias"),
                ne0: tower.heads,
                ne1: 1,
                source: Source::Dit,
            });
            out.push(Entry {
                name: format!("{b}.ff.net.0.proj.weight"),
                ne0: tower.dim,
                ne1: tower.dim * 4,
                source: Source::Dit,
            });
            out.push(Entry {
                name: format!("{b}.ff.net.0.proj.bias"),
                ne0: tower.dim * 4,
                ne1: 1,
                source: Source::Dit,
            });
            out.push(Entry {
                name: format!("{b}.ff.net.2.weight"),
                ne0: tower.dim * 4,
                ne1: tower.dim,
                source: Source::Dit,
            });
            out.push(Entry {
                name: format!("{b}.ff.net.2.bias"),
                ne0: tower.dim,
                ne1: 1,
                source: Source::Dit,
            });
        }
    }
    out
}

fn projection_layout(config: &ConnectorConfig) -> Vec<Entry> {
    vec![
        Entry {
            name: "text_embedding_projection.video_aggregate_embed.weight".into(),
            ne0: config.stack_dim,
            ne1: config.video.dim,
            source: Source::Projection,
        },
        Entry {
            name: "text_embedding_projection.video_aggregate_embed.bias".into(),
            ne0: config.video.dim,
            ne1: 1,
            source: Source::Projection,
        },
        Entry {
            name: "text_embedding_projection.audio_aggregate_embed.weight".into(),
            ne0: config.stack_dim,
            ne1: config.audio.dim,
            source: Source::Projection,
        },
        Entry {
            name: "text_embedding_projection.audio_aggregate_embed.bias".into(),
            ne0: config.audio.dim,
            ne1: 1,
            source: Source::Projection,
        },
    ]
}

/// Biases, norms, and registers are F32 in the graph (widened on load).
fn is_f32_weight(name: &str) -> bool {
    name.ends_with(".bias")
        || name.ends_with("norm.weight")
        || name.ends_with("learnable_registers")
}

// ---------------------------------------------------------------------------
// rope tables (host, mirrors main's `_precompute_connector_rope`)
// ---------------------------------------------------------------------------

/// `(cos, sin)` tables `[head_dim/2, heads, seq]` for one tower.
pub(crate) fn rope_tables(tower: &TowerConfig, seq: i64) -> (Vec<f32>, Vec<f32>) {
    let half = tower.head_dim / 2;
    let num_freqs = tower.dim / 2;
    let mut cos = vec![0f32; (half * tower.heads * seq) as usize];
    let mut sin = cos.clone();
    for f in 0..num_freqs {
        // theta^linspace(0, 1, num_freqs)
        let freq = ROPE_THETA.powf(f as f32 / (num_freqs - 1).max(1) as f32);
        let h = f / half;
        let d = f % half;
        for s in 0..seq {
            let frac = s as f32 / ROPE_MAX_POS as f32 * 2.0 - 1.0;
            let angle = freq * frac;
            let index = (d + half * h + half * tower.heads * s) as usize;
            cos[index] = angle.cos();
            sin[index] = angle.sin();
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
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
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

struct ConnectorGraph {
    graph: ReusedGraph,
    seq: i64,
    config: ConnectorConfig,
}

impl ConnectorGraph {
    fn build(
        ctx: &GgmlContext,
        config: &ConnectorConfig,
        seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut b = ctx.builder(64 * 1024 * 1024)?;
        let mut weights = HashMap::new();
        for entry in layout(config).into_iter().chain(projection_layout(config)) {
            // The registers are tiled on the host and arrive as an input.
            if entry.name.ends_with("learnable_registers") {
                continue;
            }
            let ggml_type = types.get(&entry.name).copied().unwrap_or(TYPE_F32);
            let tensor = b.tensor_2d(&entry.name, ggml_type, entry.ne0, entry.ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(entry.name, tensor);
        }
        let stacked = nn(b.f32_2d("stacked", config.stack_dim, seq, true))?;
        let valid = nn(b.f32_2d("valid_mask", 1, seq, true))?;
        let pad = nn(b.f32_2d("pad_mask", 1, seq, true))?;

        let video_scale = ((config.video.dim as f32) / (config.gemma_dim as f32)).sqrt();
        let video = nn(b.scale(stacked, video_scale))?;
        let video = linear_bias(
            &b,
            &weights,
            "text_embedding_projection.video_aggregate_embed",
            video,
        )?;
        let audio_scale = ((config.audio.dim as f32) / (config.gemma_dim as f32)).sqrt();
        let audio = nn(b.scale(stacked, audio_scale))?;
        let audio = linear_bias(
            &b,
            &weights,
            "text_embedding_projection.audio_aggregate_embed",
            audio,
        )?;

        let video = tower(&mut b, &weights, config, true, video, valid, pad, seq)?;
        let audio = tower(&mut b, &weights, config, false, audio, valid, pad, seq)?;
        let out = nn(b.concat(video, audio, 0))?;
        let graph = b.finish(out)?;
        Ok(Self {
            graph,
            seq,
            config: *config,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn tower(
    b: &mut GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: &ConnectorConfig,
    video: bool,
    projected: Tensor,
    valid: Tensor,
    pad: Tensor,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let (tower, tag) = if video {
        (config.video, "v")
    } else {
        (config.audio, "a")
    };
    let p = tower_prefix(video);
    let dim = tower.dim;
    let half = tower.head_dim / 2;
    let registers = nn(b.f32_2d(&format!("registers_tiled_{tag}"), dim, seq, true))?;
    let rope_cos = nn(b.f32_3d(&format!("rope_cos_{tag}"), half, tower.heads, seq, true))?;
    let rope_sin = nn(b.f32_3d(&format!("rope_sin_{tag}"), half, tower.heads, seq, true))?;
    // Registers replace the padding rows; valid rows keep the projection.
    let mut hidden = nn(b.add(nn(b.mul(projected, valid))?, nn(b.mul(registers, pad))?))?;
    for i in 0..DEPTH {
        let blk = format!("{p}.transformer_1d_blocks.{i}");
        let x = nn(b.rms_norm(hidden, RMS_EPS))?;
        let q = linear_bias(b, weights, &format!("{blk}.attn1.to_q"), x)?;
        let q = nn(b.mul(
            nn(b.rms_norm(q, RMS_EPS))?,
            wt(weights, &format!("{blk}.attn1.q_norm.weight"))?,
        ))?;
        let q = nn(b.cont(nn(b.reshape_3d(q, tower.head_dim, tower.heads, seq))?))?;
        let k = linear_bias(b, weights, &format!("{blk}.attn1.to_k"), x)?;
        let k = nn(b.mul(
            nn(b.rms_norm(k, RMS_EPS))?,
            wt(weights, &format!("{blk}.attn1.k_norm.weight"))?,
        ))?;
        let k = nn(b.cont(nn(b.reshape_3d(k, tower.head_dim, tower.heads, seq))?))?;
        let v = linear_bias(b, weights, &format!("{blk}.attn1.to_v"), x)?;
        let v = nn(b.cont(nn(b.reshape_3d(v, tower.head_dim, tower.heads, seq))?))?;
        let q = rope_split(b, q, rope_cos, rope_sin, half, tower.heads, seq)?;
        let k = rope_split(b, k, rope_cos, rope_sin, half, tower.heads, seq)?;
        let out = flash_attend(b, q, k, v, std::ptr::null_mut())?;
        // gate: 2 * sigmoid(to_gate_logits(x)), one scalar per (head, token).
        let gate = linear_bias(b, weights, &format!("{blk}.attn1.to_gate_logits"), x)?;
        let gate = nn(b.scale(nn(b.sigmoid(gate))?, 2.0))?;
        let gate = nn(b.reshape_3d(gate, 1, tower.heads, seq))?;
        let out = nn(b.mul(out, gate))?;
        let out = nn(b.reshape_2d(out, dim, seq))?;
        let out = linear_bias(b, weights, &format!("{blk}.attn1.to_out.0"), out)?;
        hidden = nn(b.add(hidden, out))?;
        let x = nn(b.rms_norm(hidden, RMS_EPS))?;
        let ff = linear_bias(b, weights, &format!("{blk}.ff.net.0.proj"), x)?;
        let ff = super::gelu_tanh(b, ff)?;
        let ff = linear_bias(b, weights, &format!("{blk}.ff.net.2"), ff)?;
        hidden = nn(b.add(hidden, ff))?;
    }
    nn(b.rms_norm(hidden, RMS_EPS))
}

/// The reference split RoPE: rotate the two head-dim halves with the
/// per-(head, position, pair) tables.
fn rope_split(
    b: &GraphBuilder,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    half: i64,
    heads: i64,
    seq: i64,
) -> Result<Tensor, RuntimeError> {
    let stride0 = half * 4;
    // x is `[2*half, heads, seq]` contiguous: the half split views keep the
    // original per-head and per-token strides.
    let nb1 = (2 * half) as usize * 4;
    let nb2 = (2 * half * heads) as usize * 4;
    let first = nn(b.cont(b.view_3d(x, half, heads, seq, nb1, nb2, 0)))?;
    let second = nn(b.cont(b.view_3d(x, half, heads, seq, nb1, nb2, stride0 as usize)))?;
    let rot1 = nn(b.sub(nn(b.mul(first, cos))?, nn(b.mul(second, sin))?))?;
    let rot2 = nn(b.add(nn(b.mul(first, sin))?, nn(b.mul(second, cos))?))?;
    nn(b.concat(nn(b.cont(rot1))?, nn(b.cont(rot2))?, 0))
}

// ---------------------------------------------------------------------------
// encoder
// ---------------------------------------------------------------------------

pub struct ConnectorEncoder {
    projection: Arc<GgufWeights>,
    dit: Arc<GgufWeights>,
    config: ConnectorConfig,
}

impl ConnectorEncoder {
    /// `connectors` is the published embeddings-connectors safetensors; `dit`
    /// is the DiT GGUF holding the `{video,audio}_embeddings_connector.*`
    /// towers.
    pub fn open(connectors: &[PathBuf], dit: &[PathBuf]) -> Result<Self, WeightError> {
        let projection = GgufWeights::open_parts(connectors)?;
        let dit = GgufWeights::open_parts(dit)?;
        let video_w = projection.shape("text_embedding_projection.video_aggregate_embed.weight")?;
        let audio_w = projection.shape("text_embedding_projection.audio_aggregate_embed.weight")?;
        if video_w[0] != audio_w[0] {
            return Err(WeightError::NotBuilt(format!(
                "connector projection inputs disagree: video {video_w:?} audio {audio_w:?}"
            )));
        }
        let stack_dim = video_w[0];
        let tower = |video: bool, dim: i64| -> Result<TowerConfig, WeightError> {
            let p = tower_prefix(video);
            let q = dit.shape(&format!("{p}.transformer_1d_blocks.0.attn1.to_q.weight"))?;
            if q[0] != dim || q[1] != dim {
                return Err(WeightError::NotBuilt(format!(
                    "{p} to_q shape {q:?} does not match the projection dim {dim}"
                )));
            }
            let gate = dit.shape(&format!(
                "{p}.transformer_1d_blocks.0.attn1.to_gate_logits.weight"
            ))?;
            let heads = gate[1];
            if heads < 1 || dim % heads != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "{p} dim {dim} is not divisible by heads {heads}"
                )));
            }
            let head_dim = dim / heads;
            if head_dim % 2 != 0 {
                return Err(WeightError::NotBuilt(format!(
                    "{p} head dim {head_dim} is not even (split RoPE)"
                )));
            }
            for i in 1..DEPTH {
                let name = format!("{p}.transformer_1d_blocks.{i}.attn1.to_q.weight");
                if !dit.contains(&name) {
                    return Err(WeightError::MissingTensor {
                        path: "transformer".into(),
                        name,
                    });
                }
            }
            let registers = dit.shape(&format!("{p}.learnable_registers"))?;
            if registers[0] != dim || registers[1] != REGISTERS {
                return Err(WeightError::NotBuilt(format!(
                    "{p} registers shape {registers:?} is not [{dim}, {REGISTERS}]"
                )));
            }
            Ok(TowerConfig {
                dim,
                heads,
                head_dim,
            })
        };
        let video = tower(true, video_w[1])?;
        let audio = tower(false, audio_w[1])?;
        // The stack width is gemma_dim * states; infer gemma_dim from the
        // published 49-state convention by failing loud on anything else.
        if stack_dim % 49 != 0 {
            return Err(WeightError::NotBuilt(format!(
                "connector stack dim {stack_dim} is not gemma_dim * 49"
            )));
        }
        Ok(Self {
            projection,
            dit,
            config: ConnectorConfig {
                gemma_dim: stack_dim / 49,
                stack_dim,
                video,
                audio,
            },
        })
    }

    pub fn config(&self) -> ConnectorConfig {
        self.config
    }

    /// Project + run both towers. Returns `(video, audio)`, each `[dim, seq]`
    /// flat in ggml order.
    pub fn encode(
        &self,
        ctx: &GgmlContext,
        conditioning: &GemmaConditioning,
    ) -> Result<(Vec<f32>, Vec<f32>), WeightError> {
        let config = &self.config;
        let seq = conditioning.seq;
        if conditioning.stacked.len() as i64 != config.stack_dim * seq
            || conditioning.mask.len() as i64 != seq
        {
            return Err(WeightError::NotBuilt(format!(
                "gemma stack {}x{seq} does not match the connector input {}x{seq}",
                conditioning.stacked.len() as i64 / seq,
                config.stack_dim,
            )));
        }
        let entries: Vec<Entry> = layout(config)
            .into_iter()
            .chain(projection_layout(config))
            .collect();
        let mut types = HashMap::new();
        for entry in &entries {
            if entry.name.ends_with("learnable_registers") {
                continue;
            }
            let store = match entry.source {
                Source::Projection => &self.projection,
                Source::Dit => &self.dit,
            };
            let ggml_type = if is_f32_weight(&entry.name) {
                TYPE_F32
            } else {
                let ggml_type = store.ggml_type_of(&entry.name)?;
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
        let mut graph = ConnectorGraph::build(ctx, config, seq, &types)?;
        for entry in &entries {
            if entry.name.ends_with("learnable_registers") {
                continue;
            }
            let store = match entry.source {
                Source::Projection => &self.projection,
                Source::Dit => &self.dit,
            };
            if is_f32_weight(&entry.name) {
                let values = store.f32_widened(&entry.name, &[entry.ne0, entry.ne1])?;
                graph.graph.set_f32(&entry.name, &values)?;
            } else {
                let raw = store.raw_exact(&entry.name, &[entry.ne0, entry.ne1])?;
                graph.graph.set_bytes(&entry.name, &raw)?;
            }
        }
        graph.graph.set_f32("stacked", &conditioning.stacked)?;
        graph.graph.set_f32("valid_mask", &conditioning.mask)?;
        let pad: Vec<f32> = conditioning.mask.iter().map(|v| 1.0 - v).collect();
        graph.graph.set_f32("pad_mask", &pad)?;
        for (video, tower) in [(true, config.video), (false, config.audio)] {
            let tag = if video { "v" } else { "a" };
            let p = tower_prefix(video);
            let registers = self
                .dit
                .f32_widened(&format!("{p}.learnable_registers"), &[tower.dim, REGISTERS])?;
            let mut tiled = vec![0f32; (tower.dim * seq) as usize];
            for s in 0..seq as usize {
                let src = s % REGISTERS as usize;
                let dst = s;
                tiled[dst * tower.dim as usize..(dst + 1) * tower.dim as usize].copy_from_slice(
                    &registers[src * tower.dim as usize..(src + 1) * tower.dim as usize],
                );
            }
            graph
                .graph
                .set_f32(&format!("registers_tiled_{tag}"), &tiled)?;
            let (cos, sin) = rope_tables(&tower, seq);
            graph.graph.set_f32(&format!("rope_cos_{tag}"), &cos)?;
            graph.graph.set_f32(&format!("rope_sin_{tag}"), &sin)?;
        }
        let out = graph.graph.compute()?;
        if let Some(detail) = crate::common::ops::non_finite("ltx connector", &out) {
            return Err(WeightError::NotBuilt(detail));
        }
        // ggml concat along dim 0 interleaves the towers per token: the
        // `[video + audio, seq]` output holds one token's video rows then its
        // audio rows. De-interleave back into per-tower `[dim, seq]` flats.
        let vd = config.video.dim as usize;
        let ad = config.audio.dim as usize;
        let row = vd + ad;
        let mut video = Vec::with_capacity(vd * seq as usize);
        let mut audio = Vec::with_capacity(ad * seq as usize);
        for s in 0..seq as usize {
            video.extend_from_slice(&out[s * row..s * row + vd]);
            audio.extend_from_slice(&out[s * row + vd..(s + 1) * row]);
        }
        Ok((video, audio))
    }
}

// ---------------------------------------------------------------------------
// smoke writer (graph-vs-host tests)
// ---------------------------------------------------------------------------

/// Deterministic smoke weights for both towers and the projection.
pub fn smoke_store(config: &ConnectorConfig) -> Vec<(String, Vec<f32>)> {
    let mut seed = 23u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let mut fill = |count: usize| (0..count).map(|_| next()).collect::<Vec<f32>>();
    layout(config)
        .into_iter()
        .chain(projection_layout(config))
        .map(|entry| (entry.name, fill((entry.ne0 * entry.ne1) as usize)))
        .collect()
}

/// Write the smoke connectors safetensors and the smoke DiT-side GGUF.
pub fn write_smoke(dir: &Path) -> Result<(PathBuf, PathBuf), WeightError> {
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Bundle {
        path: dir.display().to_string(),
        detail: format!("{err}"),
    })?;
    let config = ConnectorConfig::smoke();
    let store: HashMap<String, Vec<f32>> = smoke_store(&config).into_iter().collect();
    let to_bytes = |values: &[f32]| -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    };
    let proj_entries = projection_layout(&config);
    let proj_bytes: Vec<Vec<u8>> = proj_entries
        .iter()
        .map(|entry| to_bytes(&store[&entry.name]))
        .collect();
    // safetensors shapes are PyTorch order: [out, in] for matrices.
    let proj_shapes: Vec<Vec<u64>> = proj_entries
        .iter()
        .map(|entry| vec![entry.ne1 as u64, entry.ne0 as u64])
        .collect();
    let proj_tensors: Vec<crate::weights::SafeTensor<'_>> = proj_entries
        .iter()
        .zip(&proj_bytes)
        .zip(&proj_shapes)
        .map(|((entry, data), shape)| crate::weights::SafeTensor {
            name: &entry.name,
            dtype: "F32",
            shape,
            data,
        })
        .collect();
    let proj_path = dir.join("ltx_embeddings_connectors.safetensors");
    crate::weights::write_safetensors(&proj_path, &proj_tensors)?;

    let dit_entries = layout(&config);
    let dit_bytes: Vec<Vec<u8>> = dit_entries
        .iter()
        .map(|entry| to_bytes(&store[&entry.name]))
        .collect();
    let dit_tensors: Vec<crate::weights::TypedTensor<'_>> = dit_entries
        .iter()
        .zip(&dit_bytes)
        .map(|(entry, data)| crate::weights::TypedTensor {
            name: &entry.name,
            ne0: entry.ne0,
            ne1: entry.ne1,
            ggml_type: crate::runtime::TYPE_F32,
            data,
        })
        .collect();
    let dit_path = dir.join("ltx_dit_smoke.gguf");
    crate::weights::write_typed_tensors(&dit_path, &dit_tensors)?;
    Ok((proj_path, dit_path))
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

    fn rms_host(x: &[f32], w: Option<&[f32]>, dim: usize, seq: usize) -> Vec<f32> {
        let mut out = vec![0f32; dim * seq];
        for s in 0..seq {
            let mut sum = 0f32;
            for d in 0..dim {
                sum += x[d + dim * s] * x[d + dim * s];
            }
            let scale = 1.0 / (sum / dim as f32 + RMS_EPS).sqrt();
            for d in 0..dim {
                out[d + dim * s] = x[d + dim * s] * scale * w.map_or(1.0, |w| w[d]);
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

    /// Host port of one connector tower on `[dim, seq]` input.
    fn tower_host(
        store: &[(String, Vec<f32>)],
        tower: &TowerConfig,
        video: bool,
        projected: &[f32],
        mask: &[f32],
    ) -> Vec<f32> {
        let p = tower_prefix(video);
        let dim = tower.dim as usize;
        let heads = tower.heads as usize;
        let hd = tower.head_dim as usize;
        let half = hd / 2;
        let seq = mask.len();
        let registers = at(store, &format!("{p}.learnable_registers"));
        let mut hidden = vec![0f32; dim * seq];
        for s in 0..seq {
            for d in 0..dim {
                let reg = registers[d + dim * (s % REGISTERS as usize)];
                hidden[d + dim * s] = projected[d + dim * s] * mask[s] + reg * (1.0 - mask[s]);
            }
        }
        let (cos, sin) = rope_tables(tower, seq as i64);
        for i in 0..DEPTH as usize {
            let blk = format!("{p}.transformer_1d_blocks.{i}");
            let x = rms_host(&hidden, None, dim, seq);
            let q = linear_bias_host(store, &format!("{blk}.attn1.to_q"), &x, dim, dim, seq);
            let q = rms_host(
                &q,
                Some(at(store, &format!("{blk}.attn1.q_norm.weight"))),
                dim,
                seq,
            );
            let k = linear_bias_host(store, &format!("{blk}.attn1.to_k"), &x, dim, dim, seq);
            let k = rms_host(
                &k,
                Some(at(store, &format!("{blk}.attn1.k_norm.weight"))),
                dim,
                seq,
            );
            let v = linear_bias_host(store, &format!("{blk}.attn1.to_v"), &x, dim, dim, seq);
            // split rope on [hd, heads*seq] rows (head-major: hd*h + dim*s)
            let rope = |input: &[f32]| -> Vec<f32> {
                let mut out = input.to_vec();
                for s in 0..seq {
                    for h in 0..heads {
                        for d in 0..half {
                            let idx = d + half * h + half * heads * s;
                            let x1 = input[d + hd * h + dim * s];
                            let x2 = input[d + half + hd * h + dim * s];
                            out[d + hd * h + dim * s] = x1 * cos[idx] - x2 * sin[idx];
                            out[d + half + hd * h + dim * s] = x1 * sin[idx] + x2 * cos[idx];
                        }
                    }
                }
                out
            };
            let q = rope(&q);
            let k = rope(&k);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut mixed = vec![0f32; dim * seq];
            for h in 0..heads {
                for qp in 0..seq {
                    let mut scores = vec![0f32; seq];
                    for kp in 0..seq {
                        let mut acc = 0f32;
                        for d in 0..hd {
                            acc += q[d + hd * h + dim * qp] * k[d + hd * h + dim * kp];
                        }
                        scores[kp] = acc * scale;
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for kp in 0..seq {
                            acc += scores[kp] / sum * v[d + hd * h + dim * kp];
                        }
                        mixed[d + hd * h + dim * qp] = acc;
                    }
                }
            }
            // gate per (head, token)
            let gate = linear_bias_host(
                store,
                &format!("{blk}.attn1.to_gate_logits"),
                &x,
                dim,
                heads,
                seq,
            );
            for s in 0..seq {
                for h in 0..heads {
                    let g = 2.0 / (1.0 + (-gate[h + heads * s]).exp());
                    for d in 0..hd {
                        mixed[d + hd * h + dim * s] *= g;
                    }
                }
            }
            let out = linear_bias_host(
                store,
                &format!("{blk}.attn1.to_out.0"),
                &mixed,
                dim,
                dim,
                seq,
            );
            for (h, o) in hidden.iter_mut().zip(&out) {
                *h += o;
            }
            let x = rms_host(&hidden, None, dim, seq);
            let ff = linear_bias_host(
                store,
                &format!("{blk}.ff.net.0.proj"),
                &x,
                dim,
                dim * 4,
                seq,
            );
            let ff: Vec<f32> = ff.iter().map(|v| gelu_tanh_host(*v)).collect();
            let ff = linear_bias_host(store, &format!("{blk}.ff.net.2"), &ff, dim * 4, dim, seq);
            for (h, o) in hidden.iter_mut().zip(&ff) {
                *h += o;
            }
        }
        rms_host(&hidden, None, dim, seq)
    }

    #[test]
    fn graph_matches_the_host_reference() {
        let config = ConnectorConfig::smoke();
        let dir = std::env::temp_dir().join(format!(
            "danmo-ltx-conn-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let (proj_path, dit_path) = write_smoke(&dir).unwrap();
        let encoder = ConnectorEncoder::open(&[proj_path], &[dit_path]).unwrap();
        assert_eq!(encoder.config().stack_dim, config.stack_dim);

        let seq = 16i64;
        let valid = 10usize;
        let mut mask = vec![0f32; seq as usize];
        for v in mask[seq as usize - valid..].iter_mut() {
            *v = 1.0;
        }
        let mut seed = 5u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        let stacked: Vec<f32> = (0..(config.stack_dim * seq) as usize)
            .map(|_| next())
            .collect();
        // The Gemma stage multiplies the stack by the valid mask; mirror it.
        let masked: Vec<f32> = stacked
            .iter()
            .enumerate()
            .map(|(i, v)| v * mask[i / config.stack_dim as usize])
            .collect();
        let conditioning = GemmaConditioning {
            stacked: masked.clone(),
            mask: mask.clone(),
            seq,
            states: 3,
            dim: config.gemma_dim,
        };
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let (video, audio) = encoder.encode(&ctx, &conditioning).unwrap();

        // host reference
        let store = smoke_store(&config);
        let vscale = ((config.video.dim as f32) / (config.gemma_dim as f32)).sqrt();
        let scaled: Vec<f32> = masked.iter().map(|v| v * vscale).collect();
        let video_host = linear_bias_host(
            &store,
            "text_embedding_projection.video_aggregate_embed",
            &scaled,
            config.stack_dim as usize,
            config.video.dim as usize,
            seq as usize,
        );
        let video_host = tower_host(&store, &config.video, true, &video_host, &mask);
        let ascale = ((config.audio.dim as f32) / (config.gemma_dim as f32)).sqrt();
        let scaled: Vec<f32> = masked.iter().map(|v| v * ascale).collect();
        let audio_host = linear_bias_host(
            &store,
            "text_embedding_projection.audio_aggregate_embed",
            &scaled,
            config.stack_dim as usize,
            config.audio.dim as usize,
            seq as usize,
        );
        let audio_host = tower_host(&store, &config.audio, false, &audio_host, &mask);

        let mut max_diff = 0f32;
        for (g, h) in video
            .iter()
            .chain(&audio)
            .zip(video_host.iter().chain(&audio_host))
        {
            max_diff = max_diff.max((g - h).abs());
        }
        assert!(max_diff < 2e-3, "graph vs host max diff {max_diff}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_fails_loud_on_a_missing_tower_block() {
        let dir = std::env::temp_dir().join(format!("danmo-ltx-conn-empty-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let proj_path = dir.join("empty.safetensors");
        crate::weights::write_safetensors(
            &proj_path,
            &[crate::weights::SafeTensor {
                name: "dummy",
                dtype: "F32",
                shape: &[1],
                data: &[0, 0, 0, 0],
            }],
        )
        .unwrap();
        let dit_path = dir.join("empty.gguf");
        crate::weights::write_f32_gguf(&dit_path, "dummy.weight", 1, 1, &[0.0]).unwrap();
        let err = match ConnectorEncoder::open(&[proj_path], &[dit_path]) {
            Err(err) => err,
            Ok(_) => panic!("accepted empty weights"),
        };
        assert!(err.to_string().contains("video_aggregate_embed"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
