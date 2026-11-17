use danmo_mlx::{Array, CancelToken, Error, Result, Stream};

pub fn euler<F>(initial: &Array, sigmas: &[f32], stream: &Stream, velocity: F) -> Result<Array>
where
    F: FnMut(&Array, f32) -> Result<Array>,
{
    euler_cancellable(initial, sigmas, stream, None, velocity)
}

/// Euler with a step-boundary cancellation check (plan §1.3): the token is
/// polled before each velocity call, so a cancelled run stops without
/// starting another forward. The in-flight forward itself is not interrupted;
/// that boundary is recorded in the plan.
pub fn euler_cancellable<F>(
    initial: &Array,
    sigmas: &[f32],
    stream: &Stream,
    cancel: Option<&CancelToken>,
    mut velocity: F,
) -> Result<Array>
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
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return Err(Error::new(
                "Z-Image denoise",
                format!("cancelled at step {step}"),
            ));
        }
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
    use super::{euler, euler_cancellable};
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

    #[test]
    fn cancellation_stops_at_the_step_boundary() {
        let stream = Stream::cpu().unwrap();
        let initial = Array::from_f32_slice(&[1.0, -1.0], &[2]).unwrap();
        let token = danmo_mlx::CancelToken::new();
        let mut calls = 0usize;
        let error = euler_cancellable(
            &initial,
            &[1.0, 0.75, 0.5, 0.25, 0.0],
            &stream,
            Some(&token),
            |_, _| {
                calls += 1;
                if calls == 2 {
                    token.cancel();
                }
                Array::from_f32_slice(&[2.0, -1.0], &[2])
            },
        )
        .err()
        .expect("cancelled run must fail");
        assert!(error.to_string().contains("cancelled at step 2"), "{error}");
        assert_eq!(calls, 2, "no forward may start after cancellation");
        // A fresh token runs to completion on the same inputs.
        let output = euler_cancellable(
            &initial,
            &[1.0, 0.75, 0.5, 0.25, 0.0],
            &stream,
            Some(&danmo_mlx::CancelToken::new()),
            |_, _| Array::from_f32_slice(&[2.0, -1.0], &[2]),
        )
        .unwrap();
        assert_eq!(output.to_vec_f32(&stream).unwrap(), vec![-1.0, 0.0]);
    }
}
