//! ACE-Step 5Hz LM GGUF metadata retained for legacy smoke fixtures.
//! Product inference uses the safetensors MLX implementation.

use crate::weights::{GgufWeights, WeightError};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AceLmConfig {
    pub vocab: i64,
    pub hidden: i64,
    pub layers: i64,
    pub heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub intermediate: i64,
    pub rope_theta: f32,
}

impl AceLmConfig {
    /// Tiny ranks for CPU layout / greedy smoke (not a published card).
    pub fn smoke() -> Self {
        Self {
            vocab: 32,
            hidden: 8,
            layers: 1,
            heads: 2,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 16,
            rope_theta: 1_000_000.0,
        }
    }
}

pub fn weight_layout(config: AceLmConfig) -> Vec<(String, i64, i64)> {
    // ggml mul_mat: weight is [ne0=in, ne1=out] (same as common qwen3 / fsq).
    let h = config.hidden;
    let q = config.heads * config.head_dim;
    let k = config.kv_heads * config.head_dim;
    let ff = config.intermediate;
    let mut out = Vec::new();
    let mut push = |name: String, ne0: i64, ne1: i64| out.push((name, ne0, ne1));
    push("model.embed_tokens.weight".into(), h, config.vocab);
    push("model.norm.weight".into(), h, 1);
    for layer in 0..config.layers {
        let p = format!("model.layers.{layer}");
        push(format!("{p}.input_layernorm.weight"), h, 1);
        push(format!("{p}.self_attn.q_proj.weight"), h, q);
        push(format!("{p}.self_attn.k_proj.weight"), h, k);
        push(format!("{p}.self_attn.v_proj.weight"), h, k);
        push(format!("{p}.self_attn.q_norm.weight"), config.head_dim, 1);
        push(format!("{p}.self_attn.k_norm.weight"), config.head_dim, 1);
        push(format!("{p}.self_attn.o_proj.weight"), q, h);
        push(format!("{p}.post_attention_layernorm.weight"), h, 1);
        push(format!("{p}.mlp.gate_proj.weight"), h, ff);
        push(format!("{p}.mlp.up_proj.weight"), h, ff);
        push(format!("{p}.mlp.down_proj.weight"), ff, h);
    }
    out
}

pub fn open_lm_gguf(
    bundle: &Path,
) -> Result<(std::sync::Arc<GgufWeights>, AceLmConfig), WeightError> {
    let path = find_lm_gguf(bundle)?;
    let weights = GgufWeights::open(&path)?;
    let config = infer_config(&weights)?;
    for (name, ne0, ne1) in weight_layout(config) {
        let got = weights.shape(&name)?;
        if got[0] != ne0 || got[1] != ne1 {
            return Err(WeightError::Shape {
                name,
                expected: vec![ne0, ne1],
                got: got.to_vec(),
            });
        }
    }
    Ok((weights, config))
}

fn find_lm_gguf(bundle: &Path) -> Result<std::path::PathBuf, WeightError> {
    if bundle.is_file() {
        return if bundle.extension().and_then(|e| e.to_str()) == Some("gguf") {
            Ok(bundle.to_path_buf())
        } else {
            Err(WeightError::NotBuilt(format!(
                "ACE-Step LM path {} is not a GGUF",
                bundle.display()
            )))
        };
    }
    if !bundle.is_dir() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step LM bundle {} is missing",
            bundle.display()
        )));
    }
    // Prefer dedicated LM companion dirs (product install layout).
    for sub in ["lm", "companions/lm", "5hz-lm"] {
        let dir = bundle.join(sub);
        if dir.is_dir() {
            if let Ok(path) = find_lm_gguf_in_dir(&dir) {
                return Ok(path);
            }
        }
    }
    find_lm_gguf_in_dir(bundle)
}

/// Locate a single LM GGUF under `dir` (named `*lm*` / `*qwen*` preferred).
pub(crate) fn find_lm_gguf_in_dir(dir: &Path) -> Result<std::path::PathBuf, WeightError> {
    let mut named = Vec::new();
    let mut any = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })? {
        let path = entry
            .map_err(|err| WeightError::Open {
                path: dir.display().to_string(),
                detail: err.to_string(),
            })?
            .path();
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            continue;
        }
        any.push(path.clone());
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if name.contains("lm") || name.contains("qwen") {
            named.push(path);
        }
    }
    let mut matches = if named.is_empty() { any } else { named };
    match matches.len() {
        0 => Err(WeightError::NotBuilt(format!(
            "ACE-Step 5Hz LM GGUF is missing under {}",
            dir.display()
        ))),
        1 => Ok(matches.remove(0)),
        _ => Err(WeightError::NotBuilt(format!(
            "ACE-Step LM bundle {} has multiple GGUF files; keep a single LM card",
            dir.display()
        ))),
    }
}

fn infer_config(weights: &GgufWeights) -> Result<AceLmConfig, WeightError> {
    // Weights are [ne0=in, ne1=out]: q_proj [hidden, heads*head_dim].
    let embed = weights.shape("model.embed_tokens.weight")?;
    let q = weights.shape("model.layers.0.self_attn.q_proj.weight")?;
    let k = weights.shape("model.layers.0.self_attn.k_proj.weight")?;
    let qn = weights.shape("model.layers.0.self_attn.q_norm.weight")?;
    let gate = weights.shape("model.layers.0.mlp.gate_proj.weight")?;
    let hidden = embed[0];
    let vocab = embed[1];
    let head_dim = qn[0];
    if q[0] != hidden || k[0] != hidden || head_dim < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM GGUF layout could not be inferred".into(),
        ));
    }
    let heads = q[1] / head_dim;
    let kv_heads = k[1] / head_dim;
    let intermediate = gate[1];
    let mut layers = 0i64;
    while weights.contains(&format!("model.layers.{layers}.self_attn.q_proj.weight")) {
        layers += 1;
    }
    if layers < 1 || vocab < 1 || heads < 1 || kv_heads < 1 || intermediate < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step LM GGUF layout could not be inferred".into(),
        ));
    }
    Ok(AceLmConfig {
        vocab,
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim,
        intermediate,
        rope_theta: 1_000_000.0,
    })
}

/// Write a zero-weight smoke LM GGUF under `dir` (test helper).
#[cfg(test)]
pub fn write_smoke_lm(dir: &Path, config: AceLmConfig) -> Result<(), WeightError> {
    use crate::weights::{write_f32_nd, F32Nd};
    std::fs::create_dir_all(dir).map_err(|err| WeightError::Open {
        path: dir.display().to_string(),
        detail: err.to_string(),
    })?;
    let owned: Vec<(String, [i64; 4], Vec<f32>)> = weight_layout(config)
        .into_iter()
        .map(|(name, ne0, ne1)| (name, [ne0, ne1, 1, 1], vec![0f32; (ne0 * ne1) as usize]))
        .collect();
    let tensors: Vec<F32Nd<'_>> = owned
        .iter()
        .map(|(name, ne, data)| F32Nd {
            name,
            ne: *ne,
            data,
        })
        .collect();
    write_f32_nd(&dir.join("acestep-lm-smoke.gguf"), &tensors)
}

#[cfg(test)]
mod tests {
    use super::{weight_layout, AceLmConfig};

    #[test]
    fn smoke_layout_lists_tied_embed_head() {
        let layout = weight_layout(AceLmConfig::smoke());
        assert!(layout
            .iter()
            .any(|(n, _, _)| n == "model.embed_tokens.weight"));
        assert!(layout.iter().any(|(n, _, _)| n == "model.norm.weight"));
        assert!(layout
            .iter()
            .any(|(n, _, _)| n == "model.layers.0.self_attn.q_proj.weight"));
    }
}
