use danmo_mlx::{Array, Error, Result, Stream};

pub struct AdaLnParts {
    pub scale_attention: Array,
    pub gate_attention: Array,
    pub scale_ffn: Array,
    pub gate_ffn: Array,
}

pub fn adaln(input: &Array, dim: i32, stream: &Stream) -> Result<AdaLnParts> {
    if dim <= 0 || input.shape().last() != Some(&(4 * dim)) {
        return Err(Error::new(
            "Z-Image AdaLN",
            format!(
                "expected trailing dimension {}, got {:?}",
                4 * dim,
                input.shape()
            ),
        ));
    }
    let batch = input.shape().first().copied().unwrap_or(1);
    let packed = input.reshape(&[batch, 1, 4 * dim], stream)?;
    let parts = packed.split_sections(&[dim, dim * 2, dim * 3], -1, stream)?;
    if parts.len() != 4 {
        return Err(Error::new(
            "Z-Image AdaLN",
            "expected four modulation parts",
        ));
    }
    let one = Array::from_f32_slice(&[1.0], &[])?;
    Ok(AdaLnParts {
        scale_attention: Array::add(&parts[0], &one, stream)?,
        gate_attention: parts[1].tanh(stream)?,
        scale_ffn: Array::add(&parts[2], &one, stream)?,
        gate_ffn: parts[3].tanh(stream)?,
    })
}

pub fn modulate(input: &Array, scale: &Array, stream: &Stream) -> Result<Array> {
    Array::multiply(input, scale, stream)
}

#[cfg(test)]
mod tests {
    use super::super::host;
    use super::{adaln, modulate};
    use danmo_mlx::{Array, DeviceType, Result, Stream};

    fn probe(device_type: DeviceType) -> Result<()> {
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let input_values: Vec<f32> = (0..8).map(|index| index as f32 * 0.25 - 0.75).collect();
        let input = Array::from_f32_slice(&input_values, &[1, 8])?;
        let parts = adaln(&input, 2, &stream)?;
        let outputs = [
            parts.scale_attention,
            parts.gate_attention,
            parts.scale_ffn,
            parts.gate_ffn,
        ];
        for (part, values) in outputs.iter().zip(input_values.chunks_exact(2)) {
            assert_eq!(part.shape(), vec![1, 1, 2]);
            let actual = part.to_vec_f32(&stream)?;
            for (index, actual) in actual.iter().enumerate() {
                let source = values[index];
                let expected = if std::ptr::eq(part, &outputs[0]) || std::ptr::eq(part, &outputs[2])
                {
                    1.0 + source
                } else {
                    source.tanh()
                };
                assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
            }
        }
        let normalized = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
        let scale = Array::from_f32_slice(&[2.0, -1.0], &[1, 1, 2])?;
        let output = modulate(&normalized, &scale, &stream)?.to_vec_f32(&stream)?;
        assert_eq!(output, vec![2.0, -2.0, 6.0, -4.0]);

        let weight = Array::from_f32_slice(&[0.5, 1.5], &[2])?;
        let normalized = normalized.rms_norm(Some(&weight), 1e-5, &stream)?;
        let actual = normalized.to_vec_f32(&stream)?;
        let expected = host::rms_norm(&[1.0, 2.0, 3.0, 4.0], &[0.5, 1.5], 2, 2, 1e-5);
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn cpu_adaln_matches_main_formula() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_adaln_matches_main_formula() {
        probe(DeviceType::Gpu).unwrap();
    }
}
