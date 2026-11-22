use danmo_mlx::{Array, Dtype, Error, LinearWeight, Result, Stream};

pub fn embed(
    timestep: &Array,
    frequency_dim: usize,
    w1: &LinearWeight,
    b1: &Array,
    w2: &LinearWeight,
    b2: &Array,
    stream: &Stream,
) -> Result<Array> {
    if frequency_dim == 0 || frequency_dim % 2 != 0 {
        return Err(Error::new(
            "Z-Image timestep",
            format!("frequency dimension must be positive and even, got {frequency_dim}"),
        ));
    }
    let half = frequency_dim / 2;
    let indices = Array::arange(0.0, half as f64, 1.0, Dtype::Float32, stream)?;
    let scale = Array::from_f32_slice(&[-10000f64.ln() as f32], &[])?;
    let denominator = Array::from_f32_slice(&[half as f32], &[])?;
    let exponents = Array::divide(
        &Array::multiply(&indices, &scale, stream)?,
        &denominator,
        stream,
    )?;
    let frequencies = exponents.exp(stream)?.expand_dims(0, stream)?;
    let timestep = timestep
        .astype(Dtype::Float32, stream)?
        .reshape(&[1, 1], stream)?;
    let angles = Array::multiply(&timestep, &frequencies, stream)?;
    let input = Array::concatenate_axis(&[angles.cos(stream)?, angles.sin(stream)?], 1, stream)?;
    // main feeds the F32 sinusoidal embedding straight into the MLP: dense
    // matmul and QMM both promote to the input dtype, so no cast to the
    // weight's compute dtype happens here.
    let hidden = w1.apply(&input, Some(b1), stream)?.silu(stream)?;
    w2.apply(&hidden, Some(b2), stream)
}

#[cfg(test)]
mod tests {
    use super::super::host::linear;
    use super::embed;
    use danmo_mlx::{DeviceType, Result, Stream};

    fn probe(device_type: DeviceType) -> Result<()> {
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let w1_values: Vec<f32> = (0..32)
            .map(|index| (index % 7) as f32 * 0.02 - 0.05)
            .collect();
        let b1_values: Vec<f32> = (0..4).map(|index| index as f32 * 0.03 - 0.04).collect();
        let w2_values: Vec<f32> = (0..12)
            .map(|index| (index % 5) as f32 * 0.04 - 0.06)
            .collect();
        let b2_values = vec![0.02, -0.01, 0.03];
        let w1 =
            danmo_mlx::LinearWeight::dense(danmo_mlx::Array::from_f32_slice(&w1_values, &[4, 8])?);
        let b1 = danmo_mlx::Array::from_f32_slice(&b1_values, &[4])?;
        let w2 =
            danmo_mlx::LinearWeight::dense(danmo_mlx::Array::from_f32_slice(&w2_values, &[3, 4])?);
        let b2 = danmo_mlx::Array::from_f32_slice(&b2_values, &[3])?;
        let timestep = danmo_mlx::Array::from_f32_slice(&[500.0], &[1])?;
        let output = embed(&timestep, 8, &w1, &b1, &w2, &b2, &stream)?.to_vec_f32(&stream)?;

        let features = super::super::super::geom::timestep_features(500.0, 8);
        let hidden: Vec<f32> = linear(&features, &w1_values, Some(&b1_values), 1, 8, 4)
            .into_iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .collect();
        let expected = linear(&hidden, &w2_values, Some(&b2_values), 1, 4, 3);
        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn cpu_timestep_embedder_matches_main_formula() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_timestep_embedder_matches_main_formula() {
        probe(DeviceType::Gpu).unwrap();
    }
}
