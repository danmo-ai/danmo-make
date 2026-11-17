use danmo_mlx::{Array, Error, LinearWeight, Result, Stream};

use super::attention::{self, AttentionWeights};
use super::{feed_forward, modulation};

pub struct BlockWeights<'a> {
    pub attention: AttentionWeights<'a>,
    pub attention_norm1: &'a Array,
    pub attention_norm2: &'a Array,
    pub ffn_norm1: &'a Array,
    pub ffn_norm2: &'a Array,
    pub w1: &'a LinearWeight,
    pub w2: &'a LinearWeight,
    pub w3: &'a LinearWeight,
    pub adaln: Option<(&'a LinearWeight, &'a Array)>,
}

pub fn forward(
    input: &Array,
    weights: BlockWeights<'_>,
    heads: i32,
    ids: &[[i32; 3]],
    axes: [i32; 3],
    theta: f32,
    eps: f32,
    timestep: Option<&Array>,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let modulation_parts = match weights.adaln {
        Some((weight, bias)) => {
            let timestep = timestep.ok_or_else(|| {
                Error::new(
                    "Z-Image block",
                    "modulated block needs a timestep embedding",
                )
            })?;
            let packed = weight.apply(timestep, Some(bias), stream)?;
            Some(modulation::adaln(&packed, input.shape()[2], stream)?)
        }
        None => None,
    };

    let mut x = input.rms_norm(Some(weights.attention_norm1), eps, stream)?;
    if let Some(parts) = &modulation_parts {
        x = modulation::modulate(&x, &parts.scale_attention, stream)?;
    }
    let attended = attention::forward(
        &x,
        weights.attention,
        heads,
        ids,
        axes,
        theta,
        eps,
        mask,
        stream,
    )?;
    let attended = attended.rms_norm(Some(weights.attention_norm2), eps, stream)?;
    let attended = match &modulation_parts {
        Some(parts) => Array::multiply(&attended, &parts.gate_attention, stream)?,
        None => attended,
    };
    let x = Array::add(input, &attended, stream)?;

    let mut residual = x.rms_norm(Some(weights.ffn_norm1), eps, stream)?;
    if let Some(parts) = &modulation_parts {
        residual = modulation::modulate(&residual, &parts.scale_ffn, stream)?;
    }
    let residual = feed_forward::forward(&residual, weights.w1, weights.w2, weights.w3, stream)?;
    let residual = residual.rms_norm(Some(weights.ffn_norm2), eps, stream)?;
    let residual = match &modulation_parts {
        Some(parts) => Array::multiply(&residual, &parts.gate_ffn, stream)?,
        None => residual,
    };
    Array::add(&x, &residual, stream)
}

#[cfg(test)]
mod tests {
    use super::{forward, BlockWeights};
    use crate::families::z_image::mlx::attention::AttentionWeights;
    use crate::families::z_image::weights::ZImageConfig;
    use danmo_mlx::{Array, DeviceType, LinearWeight, Result, Stream};
    use std::collections::HashMap;

    fn values(count: usize, modulus: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..count)
            .map(|index| (index % modulus) as f32 * scale - offset)
            .collect()
    }

    fn probe(device_type: DeviceType) -> Result<()> {
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let input_values = values(12, 11, 0.1, 0.4);
        let input = Array::from_f32_slice(&input_values, &[1, 2, 6])?;
        let q_values = values(36, 11, 0.03, 0.1);
        let k_values = values(36, 9, 0.025, -0.05);
        let v_values = values(36, 7, 0.04, 0.08);
        let out_values = values(36, 13, 0.02, -0.07);
        let norm_q_values = values(6, 5, 0.2, -0.2);
        let norm_k_values = values(6, 7, 0.1, -0.1);
        let attn_norm1_values = values(6, 5, 0.1, -0.1);
        let attn_norm2_values = values(6, 7, 0.08, -0.1);
        let ffn_norm1_values = values(6, 9, 0.06, -0.1);
        let ffn_norm2_values = values(6, 11, 0.04, -0.1);
        let w1_values = values(24, 13, 0.04, -0.2);
        let w2_values = values(24, 11, 0.03, -0.1);
        let w3_values = values(24, 9, 0.05, -0.15);
        let adaln_weight_values = values(48, 17, 0.02, -0.1);
        let adaln_bias_values = values(24, 19, 0.03, -0.1);
        let timestep_values = values(2, 3, 0.2, -0.1);

        let q = LinearWeight::dense(Array::from_f32_slice(&q_values, &[6, 6])?);
        let k = LinearWeight::dense(Array::from_f32_slice(&k_values, &[6, 6])?);
        let v = LinearWeight::dense(Array::from_f32_slice(&v_values, &[6, 6])?);
        let out = LinearWeight::dense(Array::from_f32_slice(&out_values, &[6, 6])?);
        let norm_q = Array::from_f32_slice(&norm_q_values, &[6])?;
        let norm_k = Array::from_f32_slice(&norm_k_values, &[6])?;
        let attn_norm1 = Array::from_f32_slice(&attn_norm1_values, &[6])?;
        let attn_norm2 = Array::from_f32_slice(&attn_norm2_values, &[6])?;
        let ffn_norm1 = Array::from_f32_slice(&ffn_norm1_values, &[6])?;
        let ffn_norm2 = Array::from_f32_slice(&ffn_norm2_values, &[6])?;
        let w1 = LinearWeight::dense(Array::from_f32_slice(&w1_values, &[4, 6])?);
        let w2 = LinearWeight::dense(Array::from_f32_slice(&w2_values, &[6, 4])?);
        let w3 = LinearWeight::dense(Array::from_f32_slice(&w3_values, &[4, 6])?);
        let adaln_weight =
            LinearWeight::dense(Array::from_f32_slice(&adaln_weight_values, &[24, 2])?);
        let adaln_bias = Array::from_f32_slice(&adaln_bias_values, &[24])?;
        let timestep = Array::from_f32_slice(&timestep_values, &[1, 2])?;
        let ids = [[1, 0, 0], [2, 1, 1]];

        let mut host_weights = HashMap::new();
        let mut fused = q_values.clone();
        fused.extend(&k_values);
        fused.extend(&v_values);
        host_weights.insert("layers.0.attention.qkv.weight".into(), fused);
        host_weights.insert("layers.0.attention.out.weight".into(), out_values);
        host_weights.insert("layers.0.attention.q_norm.weight".into(), norm_q_values);
        host_weights.insert("layers.0.attention.k_norm.weight".into(), norm_k_values);
        host_weights.insert("layers.0.attention_norm1.weight".into(), attn_norm1_values);
        host_weights.insert("layers.0.attention_norm2.weight".into(), attn_norm2_values);
        host_weights.insert("layers.0.ffn_norm1.weight".into(), ffn_norm1_values);
        host_weights.insert("layers.0.ffn_norm2.weight".into(), ffn_norm2_values);
        host_weights.insert("layers.0.feed_forward.w1.weight".into(), w1_values);
        host_weights.insert("layers.0.feed_forward.w2.weight".into(), w2_values);
        host_weights.insert("layers.0.feed_forward.w3.weight".into(), w3_values);
        host_weights.insert(
            "layers.0.adaLN_modulation.0.weight".into(),
            adaln_weight_values,
        );
        host_weights.insert("layers.0.adaLN_modulation.0.bias".into(), adaln_bias_values);
        let config = ZImageConfig {
            dim: 6,
            heads: 1,
            head_dim: 6,
            axes: [2, 2, 2],
            ffn: 4,
            adaln: 2,
            rope_theta: 256.0,
            ..ZImageConfig::smoke()
        };

        for modulated in [false, true] {
            let actual = forward(
                &input,
                BlockWeights {
                    attention: AttentionWeights {
                        to_q: &q,
                        to_k: &k,
                        to_v: &v,
                        to_out: &out,
                        norm_q: &norm_q,
                        norm_k: &norm_k,
                    },
                    attention_norm1: &attn_norm1,
                    attention_norm2: &attn_norm2,
                    ffn_norm1: &ffn_norm1,
                    ffn_norm2: &ffn_norm2,
                    w1: &w1,
                    w2: &w2,
                    w3: &w3,
                    adaln: modulated.then_some((&adaln_weight, &adaln_bias)),
                },
                1,
                &ids,
                [2, 2, 2],
                256.0,
                1e-5,
                modulated.then_some(&timestep),
                None,
                &stream,
            )?
            .to_vec_f32(&stream)?;
            let expected = crate::families::z_image::reference::block(
                &host_weights,
                "layers.0",
                &input_values,
                &ids,
                modulated.then_some(timestep_values.as_slice()),
                config,
                modulated,
            );
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1.5e-4, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn cpu_context_and_adaln_blocks_match_host_reference() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_context_and_adaln_blocks_match_host_reference() {
        probe(DeviceType::Gpu).unwrap();
    }
}
