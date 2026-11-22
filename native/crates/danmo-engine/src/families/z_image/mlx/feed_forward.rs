use danmo_mlx::{Array, LinearWeight, Result, Stream};

pub fn forward(
    input: &Array,
    w1: &LinearWeight,
    w2: &LinearWeight,
    w3: &LinearWeight,
    stream: &Stream,
) -> Result<Array> {
    let gate = w1.apply(input, None, stream)?.silu(stream)?;
    let projection = w3.apply(input, None, stream)?;
    let hidden = Array::multiply(&gate, &projection, stream)?;
    w2.apply(&hidden, None, stream)
}

#[cfg(test)]
mod tests {
    use super::super::host::linear;
    use super::forward;
    use danmo_mlx::{Array, DeviceType, LinearWeight, Result, Stream};

    fn probe(device_type: DeviceType) -> Result<()> {
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let input_values = vec![0.25, -0.5, 1.0, 0.75];
        let w1_values: Vec<f32> = (0..8).map(|index| (index % 5) as f32 * 0.1 - 0.2).collect();
        let w3_values: Vec<f32> = (0..8)
            .map(|index| (index % 7) as f32 * 0.05 - 0.1)
            .collect();
        let w2_values: Vec<f32> = (0..8).map(|index| (index % 3) as f32 * 0.1 - 0.1).collect();
        let input = Array::from_f32_slice(&input_values, &[1, 2, 2])?;
        let w1 = LinearWeight::dense(Array::from_f32_slice(&w1_values, &[4, 2])?);
        let w2 = LinearWeight::dense(Array::from_f32_slice(&w2_values, &[2, 4])?);
        let w3 = LinearWeight::dense(Array::from_f32_slice(&w3_values, &[4, 2])?);
        let output = forward(&input, &w1, &w2, &w3, &stream)?.to_vec_f32(&stream)?;

        let gate = linear(&input_values, &w1_values, None, 2, 2, 4);
        let projection = linear(&input_values, &w3_values, None, 2, 2, 4);
        let hidden: Vec<f32> = gate
            .iter()
            .zip(&projection)
            .map(|(gate, projection)| {
                let silu = gate / (1.0 + (-gate).exp());
                silu * projection
            })
            .collect();
        let expected = linear(&hidden, &w2_values, None, 2, 4, 2);
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn cpu_feed_forward_matches_main_formula() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_feed_forward_matches_main_formula() {
        probe(DeviceType::Gpu).unwrap();
    }
}
