//! Qwen3 tower used by Flux2 Klein. The checkpoint has no `config.json` on the
//! installed bundles, so the shape is read from the weights and the RoPE theta
//! comes from the published Qwen3-4B / Qwen3-8B table. Anything else fails.

use danmo_mlx::{Error, Result};

use crate::common::qwen3::Qwen3Config;
use crate::weights::MlxWeights;

pub(super) const TEXT_LAYERS: [i64; 3] = [9, 18, 27];
pub(super) const TEXT_VOCAB: i64 = 151_936;

pub(super) fn text_config_from_weights(weights: &MlxWeights) -> Result<Qwen3Config> {
    let (vocab, hidden) = matrix(weights, "model.embed_tokens.weight")?;
    let layers = layer_count(weights)?;
    let (q_out, q_in) = matrix(weights, "model.layers.0.self_attn.q_proj.weight")?;
    let (k_out, k_in) = matrix(weights, "model.layers.0.self_attn.k_proj.weight")?;
    let (intermediate, gate_in) = matrix(weights, "model.layers.0.mlp.gate_proj.weight")?;
    if q_in != hidden || k_in != hidden || gate_in != hidden {
        return Err(Error::new(
            "Flux2 text encoder",
            format!("a projection in-dim is not the embedding width {hidden}"),
        ));
    }
    let head = vector_len(weights, "model.layers.0.self_attn.q_norm.weight")?;
    if head < 1 || q_out % head != 0 || k_out % head != 0 {
        return Err(Error::new(
            "Flux2 text encoder",
            format!("q_norm length {head} does not divide q {q_out} and k {k_out}"),
        ));
    }
    let heads = q_out / head;
    let kv_heads = k_out / head;
    let rope_theta = published_theta(vocab, hidden, layers, heads, kv_heads, head, intermediate)?;
    if layers < *TEXT_LAYERS.last().unwrap_or(&0) {
        return Err(Error::new(
            "Flux2 text encoder",
            format!("tower has {layers} layers, Flux2 reads layers {TEXT_LAYERS:?}"),
        ));
    }
    Ok(Qwen3Config {
        vocab,
        hidden,
        layers,
        heads,
        kv_heads,
        head_dim: head,
        intermediate,
        rope_theta,
    })
}

fn published_theta(
    vocab: i64,
    hidden: i64,
    layers: i64,
    heads: i64,
    kv_heads: i64,
    head: i64,
    intermediate: i64,
) -> Result<f32> {
    const TOWERS: &[(i64, i64, i64, i64, i64, i64, i64, f32)] = &[
        (151_936, 2560, 36, 32, 8, 128, 9728, 1_000_000.0),
        (151_936, 4096, 36, 32, 8, 128, 12288, 1_000_000.0),
    ];
    let measured = (vocab, hidden, layers, heads, kv_heads, head, intermediate);
    for (v, h, l, heads, kv, head, mid, theta) in TOWERS {
        if measured == (*v, *h, *l, *heads, *kv, *head, *mid) {
            return Ok(*theta);
        }
    }
    Err(Error::new(
        "Flux2 text encoder",
        format!(
            "vocab {vocab} hidden {hidden} layers {layers} heads {heads} kv {kv_heads} head {head} intermediate {intermediate} is not Qwen3-4B or Qwen3-8B"
        ),
    ))
}

fn layer_count(weights: &MlxWeights) -> Result<i64> {
    let mut seen = Vec::new();
    for name in weights.names() {
        let Some(rest) = name.strip_prefix("model.layers.") else {
            continue;
        };
        let Some(index) = rest.split('.').next() else {
            continue;
        };
        if rest.contains("self_attn.q_proj.weight") {
            let index: i64 = index.parse().map_err(|_| {
                Error::new(
                    "Flux2 text encoder",
                    format!("layer index {index} is not an integer"),
                )
            })?;
            seen.push(index);
        }
    }
    seen.sort_unstable();
    seen.dedup();
    if seen.is_empty() || seen.iter().copied().ne(0..seen.len() as i64) {
        return Err(Error::new(
            "Flux2 text encoder",
            format!("Qwen3 layer indices are not 0..{}", seen.len()),
        ));
    }
    Ok(seen.len() as i64)
}

fn matrix(weights: &MlxWeights, name: &str) -> Result<(i64, i64)> {
    let Some((out, inn)) = weights.linear(name)?.out_in()? else {
        return Err(Error::new(
            "Flux2 text encoder",
            format!("{name} is not a matrix"),
        ));
    };
    Ok((out, inn))
}

fn vector_len(weights: &MlxWeights, name: &str) -> Result<i64> {
    let shape = weights.tensor(name)?.shape();
    Ok(shape.iter().map(|dim| i64::from(*dim)).product())
}

#[cfg(test)]
mod tests {
    use super::published_theta;

    #[test]
    fn published_towers_are_qwen3_4b_and_8b() {
        assert_eq!(
            published_theta(151_936, 2560, 36, 32, 8, 128, 9728).unwrap(),
            1_000_000.0
        );
        assert_eq!(
            published_theta(151_936, 4096, 36, 32, 8, 128, 12288).unwrap(),
            1_000_000.0
        );
        let err = published_theta(151_936, 2048, 36, 32, 8, 128, 6144).unwrap_err();
        assert!(
            err.to_string().contains("not Qwen3-4B or Qwen3-8B"),
            "{err}"
        );
    }
}
