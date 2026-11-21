//! LongCat-Avatar DMD distilled sigma schedule and the 3-way disentangled
//! CFG combiner. Mirrors the behavior baseline exactly: text and audio
//! guidance each default to 4.0, and the scheduler consumes the negated
//! velocity (the DiT emits noise, the flow-matching step expects velocity).

use danmo_mlx::{Array, Result, Stream};

/// DMD distilled sigmas for `avatar-v1.5` (8-step schedule over 1000 train
/// timesteps), in decreasing order. Callers append the terminal `0.0`.
pub fn dmd_distilled_sigmas(num_train_timesteps: usize, num_distill_steps: usize) -> Vec<f32> {
    let step = num_train_timesteps / num_distill_steps;
    let mut indices: Vec<usize> = (0..num_distill_steps)
        .map(|i| ((i + 1) * step) as f32).map(|v| v.round() as usize)
        .collect();
    for idx in &mut indices {
        *idx = num_train_timesteps - *idx;
    }
    let denom = (num_train_timesteps - 1) as f32;
    let mut sigmas: Vec<f32> = indices
        .iter()
        .map(|idx| (num_train_timesteps - 1 - idx) as f32 / denom)
        .collect();
    sigmas.reverse();
    sigmas
}

/// FlowMatchEulerDiscreteScheduler applies its shift even to custom sigmas.
pub fn apply_scheduler_shift(sigmas: &mut [f32], shift: f32) {
    for sigma in sigmas {
        *sigma = shift * *sigma / (1.0 + (shift - 1.0) * *sigma);
    }
}

/// One Euler step over the flow-matching sigma schedule:
/// `x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * velocity`.
pub fn euler_step(latents: &Array, velocity: &Array, from: f32, to: f32, stream: &Stream) -> Result<Array> {
    let delta = Array::scalar_f32(to - from)?;
    let step = Array::multiply(velocity, &delta, stream)?;
    Array::add(latents, &step, stream)
}

/// `uncond + text_g * (cond - uncond_text) + audio_g * (uncond_text - uncond)`.
pub fn disentangled_cfg_combine(
    cond: &Array,
    uncond_text: &Array,
    uncond: &Array,
    text_guidance: f32,
    audio_guidance: f32,
    stream: &Stream,
) -> Result<Array> {
    let text_delta = Array::subtract(cond, uncond_text, stream)?;
    let text_term = Array::multiply(&text_delta, &Array::scalar_f32(text_guidance)?, stream)?;
    let audio_delta = Array::subtract(uncond_text, uncond, stream)?;
    let audio_term = Array::multiply(&audio_delta, &Array::scalar_f32(audio_guidance)?, stream)?;
    let combined = Array::add(uncond, &text_term, stream)?;
    Array::add(&combined, &audio_term, stream)
}

/// The DiT predicts noise; the scheduler expects velocity = -noise.
pub fn flip_velocity(noise_pred: &Array, stream: &Stream) -> Result<Array> {
    noise_pred.negative(stream)
}

#[cfg(test)]
mod tests {
    use super::{apply_scheduler_shift, dmd_distilled_sigmas};

    #[test]
    fn dmd_schedule_is_the_eight_step_baseline() {
        let sigmas = dmd_distilled_sigmas(1000, 8);
        let expected: Vec<f32> = [1.0, 0.8748749, 0.7497498, 0.6246246, 0.4994995, 0.3743744, 0.24924924, 0.124124125]
            .into_iter()
            .collect();
        assert_eq!(sigmas.len(), 8);
        for (got, want) in sigmas.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    #[test]
    fn scheduler_shift_is_applied_to_custom_dmd_sigmas() {
        let mut sigmas = dmd_distilled_sigmas(1000, 8);
        let unshifted = sigmas.clone();
        apply_scheduler_shift(&mut sigmas, 7.0);
        for (shifted, original) in sigmas.iter().zip(unshifted) {
            let expected = 7.0 * original / (1.0 + 6.0 * original);
            assert!((shifted - expected).abs() < 1e-6);
        }
        assert_eq!(sigmas[0], 1.0);
        assert!(sigmas[1] > 0.97);
        assert!(sigmas.last().unwrap() > &0.49);
    }
}
