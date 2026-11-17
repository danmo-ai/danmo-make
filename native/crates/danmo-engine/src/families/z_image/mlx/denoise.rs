use danmo_mlx::{Array, Error, Result, Stream};

pub fn euler<F>(initial: &Array, sigmas: &[f32], stream: &Stream, mut velocity: F) -> Result<Array>
where
    F: FnMut(&Array, f32) -> Result<Array>,
{
    if sigmas.len() < 2 || sigmas.iter().any(|sigma| !sigma.is_finite()) {
        return Err(Error::new(
            "Z-Image denoise",
            "scheduler needs finite sigma endpoints",
        ));
    }
    let mut current = initial.clone();
    for step in 0..sigmas.len() - 1 {
        let output = velocity(&current, sigmas[step])?;
        let output = if output.dtype() == current.dtype() {
            output
        } else {
            output.astype(current.dtype(), stream)?
        };
        let delta = sigmas[step + 1] - sigmas[step];
        let delta = Array::from_f32_slice(&[delta], &[])?.astype(current.dtype(), stream)?;
        current = Array::add(&current, &Array::multiply(&output, &delta, stream)?, stream)?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::euler;
    use danmo_mlx::{Array, DeviceType, Result, Stream};

    fn probe(device_type: DeviceType) -> Result<()> {
        let stream = match device_type {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let initial = Array::from_f32_slice(&[1.0, -1.0], &[2])?;
        let output = euler(&initial, &[1.0, 0.75, 0.25, 0.0], &stream, |_, _| {
            Array::from_f32_slice(&[2.0, -1.0], &[2])
        })?;
        assert_eq!(output.dtype(), danmo_mlx::Dtype::Float32);
        let output = output.to_vec_f32(&stream)?;
        assert_eq!(output, vec![-1.0, 0.0]);
        let initial = Array::from_f32_slice(&[1.0, -1.0], &[2])?
            .astype(danmo_mlx::Dtype::BFloat16, &stream)?;
        let output = euler(&initial, &[1.0, 0.0], &stream, |_, _| {
            Array::from_f32_slice(&[2.0, -1.0], &[2])
        })?;
        assert_eq!(output.dtype(), danmo_mlx::Dtype::BFloat16);
        assert_eq!(output.to_vec_f32(&stream)?, vec![-1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn cpu_euler_matches_z_image_update() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_euler_matches_z_image_update() {
        probe(DeviceType::Gpu).unwrap();
    }
}
