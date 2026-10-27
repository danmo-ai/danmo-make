//! Run the Qwen3 tower. Layer choice and the `thinking` flag come from the caller.

use super::graph::{build_qwen3, QwenTower};
use super::tokenizer::tokenize;
use crate::runtime::{GgmlContext, ReusedGraph, RuntimeError, TYPE_F32};
use crate::weights::{matmul_type, type_block, GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy)]
pub struct Qwen3Config {
    pub vocab: i64,
    pub hidden: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub intermediate: i64,
    pub rope_theta: f32,
}

impl Qwen3Config {
    pub fn smoke() -> Self {
        Self {
            vocab: 4,
            hidden: 8,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 16,
            rope_theta: 1_000_000.0,
        }
    }
}

pub struct Loaded {
    weights: GgufWeights,
    config: Qwen3Config,
}

impl Loaded {
    pub fn open(paths: &[PathBuf]) -> Result<Self, WeightError> {
        let weights = GgufWeights::open_parts(paths)?;
        if weights.names().any(|name| name.starts_with("blk.")) {
            return Err(WeightError::NotBuilt(
                "text encoder tensor names are blk.*; this graph reads layers.*".into(),
            ));
        }
        let config = infer_config(&weights)?;
        Ok(Self { weights, config })
    }

    pub fn config(&self) -> Qwen3Config {
        self.config
    }

    /// `layers` are 1-based hidden states, concatenated in the order given.
    /// `thinking` is forwarded to the chat string. The result width must equal `width`.
    pub fn encode(
        &self,
        ctx: &GgmlContext,
        tokenizer_json: &str,
        prompt: &str,
        layers: &[i64],
        thinking: bool,
        width: i64,
    ) -> Result<(Vec<f32>, i64), WeightError> {
        if layers.is_empty() {
            return Err(WeightError::NotBuilt(
                "text encoder captured no hidden states".into(),
            ));
        }
        for layer in layers {
            if *layer < 1 || *layer > self.config.layers {
                return Err(WeightError::NotBuilt(format!(
                    "text encoder layer {layer} is outside 1..={}",
                    self.config.layers
                )));
            }
        }
        let ids = tokenize(tokenizer_json, prompt, self.config.vocab, thinking)?;
        let embed = run_tower(ctx, &self.weights, self.config, layers, &ids)?;
        let seq = ids.len() as i64;
        if embed.len() as i64 != width * seq {
            return Err(WeightError::Shape {
                name: "prompt_embeds".into(),
                expected: vec![width, seq],
                got: vec![embed.len() as i64],
            });
        }
        Ok((embed, seq))
    }
}

fn run_tower(
    ctx: &GgmlContext,
    weights: &GgufWeights,
    config: Qwen3Config,
    selected: &[i64],
    ids: &[i32],
) -> Result<Vec<f32>, WeightError> {
    let seq = ids.len() as i64;
    let last = *selected.iter().max().unwrap_or(&0);
    let layout = layout_prefix(config, last);
    let mut types = HashMap::new();
    for (name, ne0, _) in &layout {
        let ggml_type = weights.ggml_type_of(name)?;
        check_stored_type(name, ggml_type, *ne0)?;
        types.insert(name.clone(), ggml_type);
    }
    let mut graph = Qwen3Encode::build(ctx, config, selected, seq, &types)?;
    for (name, ne0, ne1) in &layout {
        let raw = weights.raw_exact(name, &[*ne0, *ne1])?;
        graph.load_bytes(name, &raw)?;
    }
    Ok(graph.compute(ids)?)
}

struct Qwen3Encode {
    graph: ReusedGraph,
    seq: i64,
}

impl Qwen3Encode {
    fn build(
        ctx: &GgmlContext,
        config: Qwen3Config,
        selected: &[i64],
        seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        if seq < 1 {
            return Err(RuntimeError::Message("text sequence is empty".into()));
        }
        let last = *selected.iter().max().unwrap_or(&0);
        let layout = layout_prefix(config, last);
        let mut b = ctx.builder(64 * 1024 * 1024)?;
        let out = build_qwen3(
            &mut b,
            QwenTower {
                hidden: config.hidden,
                heads: config.heads,
                kv_heads: config.kv_heads,
                head_dim: config.head_dim,
                rope_theta: config.rope_theta,
            },
            selected,
            seq,
            &layout,
            types,
        )?;
        let graph = b.finish(out)?;
        Ok(Self { graph, seq })
    }

    fn load_bytes(&mut self, name: &str, data: &[u8]) -> Result<(), RuntimeError> {
        self.graph.set_bytes(name, data)
    }

    fn compute(&mut self, ids: &[i32]) -> Result<Vec<f32>, RuntimeError> {
        if ids.len() as i64 != self.seq {
            return Err(RuntimeError::Message(format!(
                "token ids has {} values, graph expects {}",
                ids.len(),
                self.seq
            )));
        }
        let positions: Vec<i32> = (0..self.seq).map(|i| i as i32).collect();
        let mut mask = vec![0f32; (self.seq * self.seq) as usize];
        for q in 0..self.seq {
            for k in 0..self.seq {
                if k > q {
                    mask[(k + q * self.seq) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        self.graph.set_i32("token_ids", ids)?;
        self.graph.set_i32("positions", &positions)?;
        self.graph.set_f32("attn_mask", &mask)?;
        self.graph.compute()
    }
}

fn infer_config(weights: &GgufWeights) -> Result<Qwen3Config, WeightError> {
    let embed = weights.shape("embed_tokens.weight")?;
    let q = weights.shape("layers.0.self_attn.q_proj.weight")?;
    let k = weights.shape("layers.0.self_attn.k_proj.weight")?;
    let norm = weights.shape("layers.0.self_attn.q_norm.weight")?;
    let gate = weights.shape("layers.0.mlp.gate_proj.weight")?;
    let hidden = embed[0];
    let vocab = embed[1];
    let head = norm[0];
    if hidden < 1 || vocab < 1 || head < 1 || q[0] != hidden || k[0] != hidden {
        return Err(WeightError::Shape {
            name: "embed_tokens.weight".into(),
            expected: vec![hidden, vocab],
            got: vec![q[0], q[1], k[0], k[1], head],
        });
    }
    if q[1] % head != 0 || k[1] % head != 0 {
        return Err(WeightError::NotBuilt(
            "text encoder head dim does not divide q/k width".into(),
        ));
    }
    let heads = q[1] / head;
    let kv_heads = k[1] / head;
    if kv_heads < 1 || heads % kv_heads != 0 || heads * head != hidden {
        return Err(WeightError::NotBuilt(format!(
            "text encoder heads {heads} kv {kv_heads} head {head} do not match hidden {hidden}"
        )));
    }
    let layers = count_layers(weights);
    if layers < 1 {
        return Err(WeightError::MissingTensor {
            path: "text_encoder".into(),
            name: "layers.0.self_attn.q_proj.weight".into(),
        });
    }
    Ok(Qwen3Config {
        vocab,
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim: head,
        intermediate: gate[1],
        rope_theta: 1_000_000.0,
    })
}

fn count_layers(weights: &GgufWeights) -> i64 {
    let mut n = 0i64;
    loop {
        let name = format!("layers.{n}.self_attn.q_proj.weight");
        if !weights.contains(&name) {
            break;
        }
        n += 1;
    }
    n
}

pub fn encoder_layout(config: Qwen3Config) -> Vec<(String, i64, i64)> {
    layout_prefix(config, config.layers)
}

fn layout_prefix(config: Qwen3Config, layers: i64) -> Vec<(String, i64, i64)> {
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    push("embed_tokens.weight".into(), config.hidden, config.vocab);
    let q_out = config.heads * config.head_dim;
    let kv_out = config.kv_heads * config.head_dim;
    for i in 0..layers {
        let p = format!("layers.{i}");
        push(format!("{p}.input_layernorm.weight"), config.hidden, 1);
        push(format!("{p}.self_attn.q_proj.weight"), config.hidden, q_out);
        push(
            format!("{p}.self_attn.k_proj.weight"),
            config.hidden,
            kv_out,
        );
        push(
            format!("{p}.self_attn.v_proj.weight"),
            config.hidden,
            kv_out,
        );
        push(format!("{p}.self_attn.o_proj.weight"), q_out, config.hidden);
        push(format!("{p}.self_attn.q_norm.weight"), config.head_dim, 1);
        push(format!("{p}.self_attn.k_norm.weight"), config.head_dim, 1);
        push(
            format!("{p}.post_attention_layernorm.weight"),
            config.hidden,
            1,
        );
        push(
            format!("{p}.mlp.gate_proj.weight"),
            config.hidden,
            config.intermediate,
        );
        push(
            format!("{p}.mlp.up_proj.weight"),
            config.hidden,
            config.intermediate,
        );
        push(
            format!("{p}.mlp.down_proj.weight"),
            config.intermediate,
            config.hidden,
        );
    }
    out
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if is_rms(name) {
        if ggml_type != TYPE_F32 {
            return Err(WeightError::NotBuilt(format!(
                "{name} is ggml type {ggml_type}. Norm weights stay F32"
            )));
        }
        return Ok(());
    }
    if !matmul_type(ggml_type) {
        return Err(WeightError::Type {
            name: name.to_string(),
            ggml_type,
        });
    }
    let block = type_block(ggml_type);
    if block > 1 && ne0 % block != 0 {
        return Err(WeightError::NotBuilt(format!(
            "{name} length {ne0} is not divisible by ggml block {block}"
        )));
    }
    Ok(())
}

pub(crate) fn is_rms(name: &str) -> bool {
    name.ends_with("layernorm.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight")
}
