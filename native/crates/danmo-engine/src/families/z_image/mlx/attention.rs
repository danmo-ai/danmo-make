use danmo_mlx::{Array, Error, Result, Stream};

pub struct AttentionWeights<'a> {
    pub to_q: &'a Array,
    pub to_k: &'a Array,
    pub to_v: &'a Array,
    pub to_out: &'a Array,
    pub norm_q: &'a Array,
    pub norm_k: &'a Array,
}

pub fn forward(
    input: &Array,
    weights: AttentionWeights<'_>,
    heads: i32,
    ids: &[[i32; 3]],
    axes: [i32; 3],
    theta: f32,
    eps: f32,
    mask: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    if shape.len() != 3 {
        return Err(Error::new("Z-Image attention", "expected [B, S, D] input"));
    }
    let (batch, sequence, dim) = (shape[0], shape[1], shape[2]);
    if heads <= 0 || dim % heads != 0 || ids.len() != sequence as usize {
        return Err(Error::new(
            "Z-Image attention",
            "invalid heads or position count",
        ));
    }
    let head_dim = dim / heads;
    if axes.iter().sum::<i32>() != head_dim
        || axes.iter().any(|width| *width <= 0 || width % 2 != 0)
    {
        return Err(Error::new(
            "Z-Image attention",
            "RoPE axes do not partition head dim",
        ));
    }

    let mut q = Array::linear(input, weights.to_q, None, stream)?
        .reshape(&[batch, sequence, heads, head_dim], stream)?;
    let mut k = Array::linear(input, weights.to_k, None, stream)?
        .reshape(&[batch, sequence, heads, head_dim], stream)?;
    let v = Array::linear(input, weights.to_v, None, stream)?
        .reshape(&[batch, sequence, heads, head_dim], stream)?;
    q = q.rms_norm(Some(weights.norm_q), eps, stream)?;
    k = k.rms_norm(Some(weights.norm_k), eps, stream)?;
    q = apply_rope(&q, ids, axes, theta, stream)?;
    k = apply_rope(&k, ids, axes, theta, stream)?;

    let q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = k.transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = v.transpose_axes(&[0, 2, 1, 3], stream)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let output = Array::scaled_dot_product_attention(&q, &k, &v, scale, "", mask, stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, dim], stream)?;
    Array::linear(&output, weights.to_out, None, stream)
}

fn apply_rope(
    input: &Array,
    ids: &[[i32; 3]],
    axes: [i32; 3],
    theta: f32,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (batch, sequence, heads, dim) = (shape[0], shape[1], shape[2], shape[3]);
    let pairs = dim / 2;
    let table = rope_table(ids, axes, theta)?.reshape(&[1, sequence, 1, pairs, 2], stream)?;
    let input = input.reshape(&[batch, sequence, heads, pairs, 2], stream)?;
    let input_real = component(&input, 0, stream)?;
    let input_imag = component(&input, 1, stream)?;
    let real = component(&table, 0, stream)?;
    let imag = component(&table, 1, stream)?;
    let output_real = Array::subtract(
        &Array::multiply(&input_real, &real, stream)?,
        &Array::multiply(&input_imag, &imag, stream)?,
        stream,
    )?;
    let output_imag = Array::add(
        &Array::multiply(&input_real, &imag, stream)?,
        &Array::multiply(&input_imag, &real, stream)?,
        stream,
    )?;
    Array::stack(&[output_real, output_imag], -1, stream)?
        .reshape(&[batch, sequence, heads, dim], stream)
}

fn component(input: &Array, index: i32, stream: &Stream) -> Result<Array> {
    let mut start = vec![0; input.shape().len()];
    let mut stop = input.shape();
    let strides = vec![1; input.shape().len()];
    let axis = start.len() - 1;
    start[axis] = index;
    stop[axis] = index + 1;
    input
        .slice(&start, &stop, &strides, stream)?
        .squeeze_axis(axis as i32, stream)
}

fn rope_table(ids: &[[i32; 3]], axes: [i32; 3], theta: f32) -> Result<Array> {
    let head_dim = axes.iter().sum::<i32>();
    let pairs = head_dim / 2;
    let mut values = vec![0.0; ids.len() * pairs as usize * 2];
    for (sequence, id) in ids.iter().enumerate() {
        let mut pair_offset = 0i32;
        for (axis, width) in axes.iter().enumerate() {
            let axis_pairs = width / 2;
            // main precomputes freq = theta^(-2k/d) once per pair, then
            // angle = position * freq. Iterative angle *= step drifts from
            // that at large positions (1024px grids) and breaks parity.
            for pair in 0..axis_pairs {
                let freq = theta.powf(-2.0 * pair as f32 / *width as f32);
                let angle = id[axis] as f32 * freq;
                let index = (sequence as i32 * pairs + pair_offset + pair) as usize * 2;
                values[index] = angle.cos();
                values[index + 1] = angle.sin();
            }
            pair_offset += axis_pairs;
        }
    }
    Array::from_f32_slice(&values, &[ids.len() as i32, pairs, 2])
}

#[cfg(test)]
mod tests {
    use super::{forward, AttentionWeights};
    use crate::families::z_image::weights::ZImageConfig;
    use danmo_mlx::{Array, DeviceType, Result, Stream};
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
        let q = Array::from_f32_slice(&q_values, &[6, 6])?;
        let k = Array::from_f32_slice(&k_values, &[6, 6])?;
        let v = Array::from_f32_slice(&v_values, &[6, 6])?;
        let out = Array::from_f32_slice(&out_values, &[6, 6])?;
        let norm_q = Array::from_f32_slice(&norm_q_values, &[6])?;
        let norm_k = Array::from_f32_slice(&norm_k_values, &[6])?;
        let ids = [[1, 0, 0], [2, 1, 1]];
        let actual = forward(
            &input,
            AttentionWeights {
                to_q: &q,
                to_k: &k,
                to_v: &v,
                to_out: &out,
                norm_q: &norm_q,
                norm_k: &norm_k,
            },
            1,
            &ids,
            [2, 2, 2],
            256.0,
            1e-5,
            None,
            &stream,
        )?
        .to_vec_f32(&stream)?;

        let mut fused = q_values.clone();
        fused.extend(&k_values);
        fused.extend(&v_values);
        let mut host_weights = HashMap::new();
        host_weights.insert("layers.0.attention.qkv.weight".to_string(), fused);
        host_weights.insert("layers.0.attention.out.weight".to_string(), out_values);
        host_weights.insert(
            "layers.0.attention.q_norm.weight".to_string(),
            norm_q_values,
        );
        host_weights.insert(
            "layers.0.attention.k_norm.weight".to_string(),
            norm_k_values,
        );
        let config = ZImageConfig {
            dim: 6,
            heads: 1,
            head_dim: 6,
            axes: [2, 2, 2],
            rope_theta: 256.0,
            ..ZImageConfig::smoke()
        };
        let expected = crate::families::z_image::reference::attention(
            &host_weights,
            "layers.0",
            &input_values,
            &ids,
            config,
        );
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 8e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    fn mask_probe(device_type: DeviceType) -> Result<()> {
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let queries = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
        let keys = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
        let values = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
        let mask = Array::from_f32_slice(
            &[0.0, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY],
            &[1, 1, 2, 2],
        )?;
        let output = Array::scaled_dot_product_attention(
            &queries,
            &keys,
            &values,
            std::f32::consts::FRAC_1_SQRT_2,
            "",
            Some(&mask),
            &stream,
        )?
        .to_vec_f32(&stream)?;
        assert_eq!(output, vec![1.0, 2.0, 1.0, 2.0]);
        Ok(())
    }

    #[test]
    fn cpu_additive_mask_excludes_invalid_keys() {
        mask_probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_additive_mask_excludes_invalid_keys() {
        mask_probe(DeviceType::Gpu).unwrap();
    }

    #[test]
    fn cpu_attention_matches_host_reference() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_attention_matches_host_reference() {
        probe(DeviceType::Gpu).unwrap();
    }
}
