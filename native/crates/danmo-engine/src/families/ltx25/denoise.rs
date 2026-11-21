//! LTX 2.5 x0-prediction Euler updates.
//!
//! Both LTX 2.5 stages use the ancestral rectified-flow update. The joint
//! A/V denoise loop calls these steps; no image-pipeline velocity convention
//! is substituted.

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

pub const ANCESTRAL_ETA: f32 = 1.0;
pub const ANCESTRAL_S_NOISE: f32 = 1.0;

pub fn euler_x0(
    latent: &Array,
    prediction: &Array,
    sigma: f32,
    sigma_next: f32,
    stream: &Stream,
) -> Result<Array> {
    validate_inputs(latent, prediction, sigma, sigma_next)?;
    if sigma_next == 0.0 {
        return Ok(prediction.clone());
    }
    let direction = Array::divide(
        &Array::subtract(latent, prediction, stream)?,
        &scalar(sigma)?,
        stream,
    )?;
    Array::add(
        latent,
        &Array::multiply(&direction, &scalar(sigma_next - sigma)?, stream)?,
        stream,
    )
}

pub fn ancestral_euler_x0(
    latent: &Array,
    prediction: &Array,
    sigma: f32,
    sigma_next: f32,
    noise: &Array,
    stream: &Stream,
) -> Result<Array> {
    ancestral_euler_x0_with(
        latent,
        prediction,
        sigma,
        sigma_next,
        noise,
        ANCESTRAL_ETA,
        ANCESTRAL_S_NOISE,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn ancestral_euler_x0_with(
    latent: &Array,
    prediction: &Array,
    sigma: f32,
    sigma_next: f32,
    noise: &Array,
    eta: f32,
    s_noise: f32,
    stream: &Stream,
) -> Result<Array> {
    validate_inputs(latent, prediction, sigma, sigma_next)?;
    if latent.shape() != noise.shape() {
        return Err(Error::new(
            "LTX 2.5 ancestral denoise",
            format!(
                "noise shape {:?} does not match latent {:?}",
                noise.shape(),
                latent.shape()
            ),
        ));
    }
    if sigma_next >= sigma || !eta.is_finite() || eta < 0.0 || !s_noise.is_finite() {
        return Err(Error::new(
            "LTX 2.5 ancestral denoise",
            format!(
                "need sigma_next < sigma, eta >= 0, and finite s_noise; got {sigma} -> {sigma_next}, eta={eta}, s_noise={s_noise}"
            ),
        ));
    }
    if sigma_next == 0.0 {
        return Ok(prediction.clone());
    }
    let downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta;
    let sigma_down = sigma_next * downstep_ratio;
    let ratio = sigma_down / sigma;
    let alpha_next = 1.0 - sigma_next;
    let alpha_down = 1.0 - sigma_down;
    if alpha_down.abs() <= f32::EPSILON {
        return Err(Error::new(
            "LTX 2.5 ancestral denoise",
            "ancestral downstep has a singular alpha denominator",
        ));
    }
    let renoise = (sigma_next * sigma_next
        - sigma_down * sigma_down * alpha_next * alpha_next / (alpha_down * alpha_down))
        .max(0.0)
        .sqrt();

    let output_dtype = latent.dtype();
    let latent = latent.astype(Dtype::Float32, stream)?;
    let prediction = prediction.astype(Dtype::Float32, stream)?;
    let noise = noise.astype(Dtype::Float32, stream)?;
    let blended = Array::add(
        &Array::multiply(&latent, &scalar(ratio)?, stream)?,
        &Array::multiply(&prediction, &scalar(1.0 - ratio)?, stream)?,
        stream,
    )?;
    let downscaled = Array::multiply(&blended, &scalar(alpha_next / alpha_down)?, stream)?;
    let updated = Array::add(
        &downscaled,
        &Array::multiply(&noise, &scalar(s_noise * renoise)?, stream)?,
        stream,
    )?;
    updated.astype(output_dtype, stream)
}

fn validate_inputs(latent: &Array, prediction: &Array, sigma: f32, sigma_next: f32) -> Result<()> {
    if latent.shape() != prediction.shape() {
        return Err(Error::new(
            "LTX 2.5 denoise",
            format!(
                "prediction shape {:?} does not match latent {:?}",
                prediction.shape(),
                latent.shape()
            ),
        ));
    }
    if !sigma.is_finite() || !sigma_next.is_finite() || sigma <= 0.0 || sigma_next < 0.0 {
        return Err(Error::new(
            "LTX 2.5 denoise",
            format!("invalid sigma endpoints {sigma} -> {sigma_next}"),
        ));
    }
    Ok(())
}

fn scalar(value: f32) -> Result<Array> {
    Array::from_f32_slice(&[value], &[])
}

#[cfg(test)]
mod tests {
    use super::{ancestral_euler_x0, euler_x0};
    use danmo_mlx::{Array, DeviceType, Result, Stream};

    fn probe(device: DeviceType) -> Result<()> {
        let stream = match device {
            DeviceType::Cpu => Stream::cpu()?,
            DeviceType::Gpu => Stream::gpu()?,
        };
        let latent = Array::from_f32_slice(&[0.8, -0.4], &[2])?;
        let prediction = Array::from_f32_slice(&[0.1, 0.3], &[2])?;
        let noise = Array::from_f32_slice(&[0.25, -0.5], &[2])?;
        let sigma = 0.8f32;
        let sigma_next = 0.5f32;

        let euler =
            euler_x0(&latent, &prediction, sigma, sigma_next, &stream)?.to_vec_f32(&stream)?;
        for (got, want) in euler.iter().zip([0.5375, -0.1375]) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }

        let downstep_ratio = sigma_next / sigma;
        let sigma_down = sigma_next * downstep_ratio;
        let ratio = sigma_down / sigma;
        let alpha_next = 1.0 - sigma_next;
        let alpha_down = 1.0 - sigma_down;
        let renoise = (sigma_next * sigma_next
            - sigma_down.powi(2) * alpha_next.powi(2) / alpha_down.powi(2))
        .sqrt();
        let expected: Vec<f32> = [0.8f32, -0.4]
            .iter()
            .zip([0.1f32, 0.3])
            .zip([0.25f32, -0.5])
            .map(|((x, x0), n)| {
                let mixed = ratio * x + (1.0 - ratio) * x0;
                (alpha_next / alpha_down) * mixed + n * renoise
            })
            .collect();
        let ancestral =
            ancestral_euler_x0(&latent, &prediction, sigma, sigma_next, &noise, &stream)?
                .to_vec_f32(&stream)?;
        for (got, want) in ancestral.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        let terminal = euler_x0(&latent, &prediction, sigma, 0.0, &stream)?.to_vec_f32(&stream)?;
        assert_eq!(terminal, vec![0.1, 0.3]);
        Ok(())
    }

    #[test]
    fn cpu_x0_steps_match_the_host_reference() {
        probe(DeviceType::Cpu).unwrap();
    }

    #[cfg(feature = "mlx-metal")]
    #[test]
    fn metal_x0_steps_match_the_host_reference() {
        probe(DeviceType::Gpu).unwrap();
    }
}
