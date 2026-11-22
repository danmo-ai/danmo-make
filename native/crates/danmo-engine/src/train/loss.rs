//! Flow-Match SFT loss: target `v = ε − x0`, `x_t = (1−σ)x0 + σ·ε`.

use danmo_mlx::{Array, Result as MlxResult, Stream};

pub fn sample_noisy_latent(
    x0: &Array,
    sigma: f32,
    eps: &Array,
    stream: &Stream,
) -> MlxResult<Array> {
    let one_m = Array::scalar_f32(1.0 - sigma)?;
    let sig = Array::scalar_f32(sigma)?;
    let clean = Array::multiply(x0, &one_m, stream)?;
    let noise = Array::multiply(eps, &sig, stream)?;
    Array::add(&clean, &noise, stream)
}

/// Mean squared error between prediction and velocity target `ε − x0`.
pub fn flow_match_mse(pred: &Array, x0: &Array, eps: &Array, stream: &Stream) -> MlxResult<Array> {
    let target = Array::subtract(eps, x0, stream)?;
    let err = Array::subtract(pred, &target, stream)?;
    let sq = err.square(stream)?;
    sq.mean(false, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use danmo_mlx::Stream;

    #[test]
    fn perfect_velocity_prediction_is_zero_loss() {
        let stream = Stream::cpu().unwrap();
        let x0 = Array::from_f32_slice(&[1.0, -1.0, 0.5, 0.25], &[4]).unwrap();
        let eps = Array::from_f32_slice(&[0.0, 1.0, -0.5, 0.75], &[4]).unwrap();
        let target = Array::subtract(&eps, &x0, &stream).unwrap();
        let loss = flow_match_mse(&target, &x0, &eps, &stream).unwrap();
        loss.eval().unwrap();
        let v = loss.to_vec_f32(&stream).unwrap()[0];
        assert!(v.abs() < 1e-6, "loss={v}");
    }
}
