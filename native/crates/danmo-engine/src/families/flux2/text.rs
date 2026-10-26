//! Qwen3 prompt tower for Flux2. Tensor names are the GGUF names.
//!
//! Hidden states after layers 9, 18, and 27 are concatenated when that matches
//! the DiT joint width. A shorter tower concatenates every layer output when
//! `hidden * layers` equals the joint width. `blk.*` names are not rewritten.

use super::weights::Flux2Config;
use crate::runtime::{BackendKind, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError};
use crate::weights::{
    matmul_type, type_block, write_f32_tensors, F32Tensor, GgufWeights, WeightError,
};
use danmo_ggml_sys::{ggml_tensor, ggml_type_GGML_TYPE_F32};
use std::collections::HashMap;
use std::path::Path;

const RMS_EPS: f32 = 1e-6;

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

pub fn encode_prompt(
    paths: &[std::path::PathBuf],
    tokenizer_json: &str,
    prompt: &str,
    joint: i64,
    backend: BackendKind,
) -> Result<(Vec<f32>, i64), WeightError> {
    let weights = GgufWeights::open_parts(paths)?;
    if weights.names().any(|name| name.starts_with("blk.")) {
        return Err(WeightError::NotBuilt(
            "text encoder tensor names are blk.*; this graph reads layers.*".into(),
        ));
    }
    let config = infer_config(&weights)?;
    let selected = select_layers(config.layers, config.hidden, joint)?;
    let ids = tokenize(tokenizer_json, prompt, config.vocab)?;
    let seq = ids.len() as i64;
    let last = *selected.iter().max().unwrap_or(&0);
    let layout = layout_prefix(config, last);
    let mut types = HashMap::new();
    for (name, ne0, _) in &layout {
        let ggml_type = weights.ggml_type_of(name)?;
        check_stored_type(name, ggml_type, *ne0)?;
        types.insert(name.clone(), ggml_type);
    }
    let ctx = GgmlContext::new(backend)?;
    let mut graph = Qwen3Encode::build(&ctx, config, &selected, seq, &types)?;
    for (name, ne0, ne1) in &layout {
        let raw = weights.raw_exact(name, &[*ne0, *ne1])?;
        graph.load_bytes(name, &raw)?;
    }
    let embed = graph.compute(&ids)?;
    if embed.len() as i64 != joint * seq {
        return Err(WeightError::Shape {
            name: "prompt_embeds".into(),
            expected: vec![joint, seq],
            got: vec![embed.len() as i64],
        });
    }
    Ok((embed, seq))
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
        let mut b = ctx.builder(64 * 1024 * 1024)?;
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in layout_prefix(config, last) {
            let ggml_type = types.get(&name).copied().unwrap_or(ggml_type_GGML_TYPE_F32);
            let tensor = b.tensor_2d(&name, ggml_type, ne0, ne1, true);
            if tensor.is_null() {
                return Err(RuntimeError::Init);
            }
            weights.insert(name, tensor);
        }
        let token_ids = nn(b.i32_1d("token_ids", seq, true))?;
        let pos = nn(b.i32_1d("positions", seq, true))?;
        let mask = nn(b.f32_2d("attn_mask", seq, seq, true))?;
        let mut hidden = nn(b.get_rows(wt(&weights, "embed_tokens.weight")?, token_ids))?;
        let last = *selected.iter().max().unwrap_or(&0);
        let mut captured: Vec<*mut ggml_tensor> = Vec::new();
        for layer in 0..last {
            hidden = encoder_layer(&b, &weights, config, layer, hidden, pos, mask, seq)?;
            let index = layer + 1;
            if selected.contains(&index) {
                captured.push(hidden);
            }
        }
        let mut out = *captured.first().ok_or_else(|| {
            RuntimeError::Message("text encoder captured no hidden states".into())
        })?;
        for extra in captured.iter().skip(1) {
            out = nn(b.concat(out, *extra, 0))?;
        }
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

fn encoder_layer(
    b: &GraphBuilder<'_>,
    weights: &HashMap<String, *mut ggml_tensor>,
    config: Qwen3Config,
    layer: i64,
    hidden: *mut ggml_tensor,
    pos: *mut ggml_tensor,
    mask: *mut ggml_tensor,
    seq: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let p = format!("layers.{layer}");
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.input_layernorm.weight"))?,
    )?;
    let attn = attention(b, weights, config, &p, normed, pos, mask, seq)?;
    let hidden = nn(b.add(hidden, attn))?;
    let normed = rms(
        b,
        hidden,
        wt(weights, &format!("{p}.post_attention_layernorm.weight"))?,
    )?;
    let mid = nn(b.mul(
        nn(b.silu(linear(
            b,
            normed,
            wt(weights, &format!("{p}.mlp.gate_proj.weight"))?,
        )))?,
        nn(linear(
            b,
            normed,
            wt(weights, &format!("{p}.mlp.up_proj.weight"))?,
        ))?,
    ))?;
    let mlp = nn(linear(
        b,
        mid,
        wt(weights, &format!("{p}.mlp.down_proj.weight"))?,
    ))?;
    nn(b.add(hidden, mlp))
}

fn attention(
    b: &GraphBuilder<'_>,
    weights: &HashMap<String, *mut ggml_tensor>,
    config: Qwen3Config,
    prefix: &str,
    hidden: *mut ggml_tensor,
    pos: *mut ggml_tensor,
    mask: *mut ggml_tensor,
    seq: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let head = config.head_dim;
    let heads = config.heads;
    let kv = config.kv_heads;
    let q = rms_heads(
        b,
        nn(linear(
            b,
            hidden,
            wt(weights, &format!("{prefix}.self_attn.q_proj.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
        head,
        heads,
        seq,
    )?;
    let k = rms_heads(
        b,
        nn(linear(
            b,
            hidden,
            wt(weights, &format!("{prefix}.self_attn.k_proj.weight"))?,
        ))?,
        wt(weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
        head,
        kv,
        seq,
    )?;
    let v = split_heads(
        b,
        nn(linear(
            b,
            hidden,
            wt(weights, &format!("{prefix}.self_attn.v_proj.weight"))?,
        ))?,
        head,
        kv,
        seq,
    )?;
    let q = nn(b.cont(b.rope_neox(q, pos, head as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.rope_neox(k, pos, head as i32, config.rope_theta)))?;
    let k = nn(b.cont(b.repeat(k, q)))?;
    let v = nn(b.cont(b.repeat(v, q)))?;
    let q_b = nn(b.cont(b.permute(q, 0, 2, 1, 3)))?;
    let k_b = nn(b.cont(b.permute(k, 0, 2, 1, 3)))?;
    let scale = 1.0 / (head as f32).sqrt();
    let scores = nn(b.soft_max_ext(nn(b.mul_mat(k_b, q_b))?, mask, scale))?;
    let v_b = nn(b.cont(b.permute(v, 1, 2, 0, 3)))?;
    let mixed = nn(b.mul_mat(v_b, scores))?;
    let mixed = nn(b.cont(b.permute(mixed, 0, 2, 1, 3)))?;
    let mixed = nn(b.reshape_2d(mixed, config.hidden, seq))?;
    nn(linear(
        b,
        mixed,
        wt(weights, &format!("{prefix}.self_attn.o_proj.weight"))?,
    ))
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

fn select_layers(layers: i64, hidden: i64, joint: i64) -> Result<Vec<i64>, WeightError> {
    if layers >= 27 && hidden.saturating_mul(3) == joint {
        return Ok(vec![9, 18, 27]);
    }
    if hidden.saturating_mul(layers) == joint {
        return Ok((1..=layers).collect());
    }
    Err(WeightError::NotBuilt(format!(
        "text encoder hidden {hidden} x {layers} layers is not joint {joint}"
    )))
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

fn tokenize(json: &str, prompt: &str, vocab_size: i64) -> Result<Vec<i32>, WeightError> {
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|err| WeightError::NotBuilt(format!("tokenizer.json is not json: {err}")))?;
    if value.get("model").is_some() {
        return Err(WeightError::NotBuilt("chat tokenizer is not built".into()));
    }
    let Some(vocab) = value.get("vocab").and_then(|item| item.as_object()) else {
        return Err(WeightError::NotBuilt("tokenizer has no vocab".into()));
    };
    let words: Vec<&str> = prompt
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect();
    if words.is_empty() {
        return Err(WeightError::NotBuilt("prompt has no tokens".into()));
    }
    let mut ids = Vec::with_capacity(words.len());
    for word in words {
        let Some(id) = vocab.get(word).and_then(|item| item.as_i64()) else {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer has no id for {word}"
            )));
        };
        if id < 0 || id >= vocab_size {
            return Err(WeightError::NotBuilt(format!(
                "tokenizer id {id} for {word} is outside vocab {vocab_size}"
            )));
        }
        ids.push(id as i32);
    }
    Ok(ids)
}

fn check_stored_type(name: &str, ggml_type: u32, ne0: i64) -> Result<(), WeightError> {
    if is_rms(name) {
        if ggml_type != ggml_type_GGML_TYPE_F32 {
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

fn is_rms(name: &str) -> bool {
    name.ends_with("layernorm.weight")
        || name.ends_with("q_norm.weight")
        || name.ends_with("k_norm.weight")
}

fn rms(
    b: &GraphBuilder<'_>,
    x: *mut ggml_tensor,
    weight: *mut ggml_tensor,
) -> Result<*mut ggml_tensor, RuntimeError> {
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, weight))
}

fn split_heads(
    b: &GraphBuilder<'_>,
    x: *mut ggml_tensor,
    head: i64,
    heads: i64,
    seq: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    nn(b.cont(b.reshape_3d(x, head, heads, seq)))
}

fn rms_heads(
    b: &GraphBuilder<'_>,
    x: *mut ggml_tensor,
    weight: *mut ggml_tensor,
    head: i64,
    heads: i64,
    seq: i64,
) -> Result<*mut ggml_tensor, RuntimeError> {
    let x = split_heads(b, x, head, heads, seq)?;
    nn(b.mul(nn(b.rms_norm(x, RMS_EPS))?, weight))
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

pub fn write_qwen3_bundle(dir: &Path, dit: Flux2Config) -> Result<(), WeightError> {
    let enc = Qwen3Config::smoke();
    if dit.joint_attention_dim != enc.hidden * enc.layers {
        return Err(WeightError::NotBuilt(format!(
            "joint {} does not match qwen3 smoke hidden {} x {} layers",
            dit.joint_attention_dim, enc.hidden, enc.layers
        )));
    }
    super::write_config_bundle(dir, dit)?;
    let owned: Vec<(String, i64, i64, Vec<f32>)> = encoder_layout(enc)
        .into_iter()
        .map(|(name, ne0, ne1)| {
            let data = if name == "embed_tokens.weight" {
                embed_rows(ne0, ne1)
            } else if is_rms(&name) {
                vec![1.0; (ne0 as usize) * (ne1 as usize)]
            } else {
                diagonal(ne0, ne1)
            };
            (name, ne0, ne1, data)
        })
        .collect();
    let tensors: Vec<F32Tensor<'_>> = owned
        .iter()
        .map(|(name, ne0, ne1, data)| F32Tensor {
            name,
            ne0: *ne0,
            ne1: *ne1,
            data,
        })
        .collect();
    write_f32_tensors(&dir.join("text_encoder.gguf"), &tensors)?;
    let tokenizer = r#"{"vocab":{"a":0,"cat":1,"dog":2}}
"#;
    std::fs::write(dir.join("tokenizer.json"), tokenizer).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    Ok(())
}

fn embed_rows(hidden: i64, vocab: i64) -> Vec<f32> {
    let mut data = vec![0f32; (hidden as usize) * (vocab as usize)];
    for token in 0..vocab {
        let row = token.min(hidden - 1) as usize;
        data[row + (token as usize) * (hidden as usize)] = 1.0;
    }
    data
}

fn diagonal(ne0: i64, ne1: i64) -> Vec<f32> {
    let mut data = vec![0f32; (ne0 as usize) * (ne1 as usize)];
    let span = ne0.min(ne1) as usize;
    for i in 0..span {
        data[i + i * ne0 as usize] = 0.05;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::{encode_prompt, write_qwen3_bundle};
    use crate::families::Flux2Config;
    use crate::runtime::BackendKind;

    #[test]
    fn cat_and_dog_prompts_embed_differently() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-qwen3-emb-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let mut dit = Flux2Config::from_dim(8);
        dit.joint_attention_dim = 16;
        write_qwen3_bundle(&dir, dit).unwrap();
        let tokenizer = std::fs::read_to_string(dir.join("tokenizer.json")).unwrap();
        let paths = vec![dir.join("text_encoder.gguf")];
        let (cat, seq) = encode_prompt(&paths, &tokenizer, "a cat", 16, BackendKind::Cpu).unwrap();
        let (dog, _) = encode_prompt(&paths, &tokenizer, "a dog", 16, BackendKind::Cpu).unwrap();
        assert_eq!(seq, 2);
        assert_eq!(cat.len(), 32);
        assert!(cat.iter().any(|v| v.abs() > 1e-4), "cat embed {cat:?}");
        assert_ne!(cat, dog);
        let err = encode_prompt(&paths, &tokenizer, "a missing", 16, BackendKind::Cpu).unwrap_err();
        assert!(err.to_string().contains("no id for missing"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
