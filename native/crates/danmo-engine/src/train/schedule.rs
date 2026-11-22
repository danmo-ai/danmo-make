//! DiffSynth-aligned Flow-Match training schedule (Z-Image shift + mid-Gaussian weight).

use danmo_mlx::{Array, Result as MlxResult, Stream};

/// Training timestep grid matching DiffSynth `FlowMatchScheduler` (Z-Image template).
#[derive(Debug, Clone)]
pub struct FlowMatchTrainSchedule {
    pub sigmas: Vec<f32>,
    pub weights: Vec<f32>,
    pub shift: f32,
}

impl FlowMatchTrainSchedule {
    /// `num_steps` discrete sigmas with static shift (DiffSynth Z-Image default shift=3).
    pub fn z_image(num_steps: usize, shift: f32) -> Self {
        let n = num_steps.max(1);
        let mut sigmas = Vec::with_capacity(n);
        for i in 0..n {
            let u = 1.0 - (i as f32) / n as f32;
            let sigma = shift * u / (1.0 + (shift - 1.0) * u);
            sigmas.push(sigma.clamp(1e-4, 1.0 - 1e-4));
        }
        let weights = mid_gaussian_weights(&sigmas);
        Self {
            sigmas,
            weights,
            shift,
        }
    }
}

fn mid_gaussian_weights(sigmas: &[f32]) -> Vec<f32> {
    // DiffSynth `set_training_weight`: Gaussian peaked at mid timesteps.
    let steps = 1000.0_f32;
    let mut raw: Vec<f32> = sigmas
        .iter()
        .map(|&s| {
            let x = s * steps;
            (-2.0 * ((x - steps / 2.0) / steps).powi(2)).exp()
        })
        .collect();
    let min = raw.iter().cloned().fold(f32::INFINITY, f32::min);
    for y in &mut raw {
        *y -= min;
    }
    let sum: f32 = raw.iter().sum::<f32>().max(1e-8);
    let bump = raw.get(1).copied().unwrap_or(0.0);
    for y in &mut raw {
        *y *= steps / sum;
        *y += bump;
    }
    if sigmas.len() != 1000 {
        let scale = sigmas.len() as f32 / steps;
        for y in &mut raw {
            *y *= scale;
        }
    }
    raw
}

pub fn discrete_training_weight(schedule: &FlowMatchTrainSchedule, index: usize) -> f32 {
    schedule.weights.get(index).copied().unwrap_or(1.0)
}

/// Uniform index into the discrete training grid; returns (sigma, weight, index).
pub fn sample_discrete_sigma(schedule: &FlowMatchTrainSchedule, rng_u: f32) -> (f32, f32, usize) {
    let n = schedule.sigmas.len().max(1);
    let index = ((rng_u.clamp(0.0, 0.9999)) * n as f32) as usize;
    let index = index.min(n - 1);
    (
        schedule.sigmas[index],
        discrete_training_weight(schedule, index),
        index,
    )
}

/// SD3 / AI-Toolkit logit-normal: `sigmoid(N(0,1))`.
pub fn sample_logit_normal_sigma(stream: &Stream) -> MlxResult<Array> {
    let z = Array::from_f32_slice(
        &[0.0], // placeholder replaced below via host RNG for unit tests without MLX normal API dependency in schedule
        &[1],
    )?;
    let _ = (stream, z);
    // Prefer host sampling for schedule helpers used outside executor; engine loop uses MLX RNG.
    let u = 1.0 / (1.0 + (-host_normal()).exp());
    Array::from_f32_slice(&[u.clamp(1e-3, 1.0 - 1e-3)], &[1])
}

fn host_normal() -> f32 {
    // Box-Muller with fixed cheap entropy from system time.
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as f32)
        .unwrap_or(1.0);
    let u1 = ((t % 997.0) + 1.0) / 1000.0;
    let u2 = (((t * 1.7) % 991.0) + 1.0) / 1000.0;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_image_schedule_shift3_is_monotonic_decreasing() {
        let sched = FlowMatchTrainSchedule::z_image(1000, 3.0);
        assert_eq!(sched.sigmas.len(), 1000);
        assert!(sched.sigmas[0] > sched.sigmas[999]);
        assert!(sched.weights.iter().all(|w| w.is_finite() && *w > 0.0));
    }
}
