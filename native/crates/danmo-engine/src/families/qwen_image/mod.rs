//! Qwen-Image. Create on a safetensors bundle runs the MLX session. The ggml
//! graph for create and edit is gone.

mod edit_session;
mod ids;
#[cfg(feature = "mlx")]
mod mlx;
mod session;

pub(crate) use edit_session::QwenImageEditFamily;
pub(crate) use session::QwenImageFamily;

use crate::common::ops::flow_match_shift_sigmas;
use crate::weights::WeightError;

/// mflux `LinearScheduler` for Qwen-Image-2512. The resolution shift is
/// `mu = m * (width * height / 256) + b` with base 0.5 at 256 tokens and max
/// 0.9 at 8192, then the schedule is rescaled so the last sigma is 0.02.
/// A request that sets a positive shift and leaves `use_empirical_mu` false
/// takes that static shift instead.
pub(super) fn schedule(
    steps: u32,
    width: u32,
    height: u32,
    vae_scale: u32,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
) -> Result<Vec<f32>, WeightError> {
    match scheduler.unwrap_or("flow_match_euler_discrete") {
        "linear" => Ok(qwen_sigmas(steps, width, height)),
        "flow_match_euler" | "flow_match_euler_discrete" => {
            if empirical_mu || scheduler_shift.is_none() {
                let scale = vae_scale.max(1);
                if width % scale != 0 || height % scale != 0 {
                    return Err(WeightError::NotBuilt(format!(
                        "size {width}x{height} is not divisible by vae scale {scale}"
                    )));
                }
                return Ok(qwen_sigmas(steps, width, height));
            }
            let shift = scheduler_shift.unwrap_or(0.0);
            if !shift.is_finite() || shift <= 0.0 {
                return Err(WeightError::NotBuilt(format!(
                    "scheduler shift {shift} is not a positive finite value"
                )));
            }
            Ok(flow_match_shift_sigmas(steps, shift))
        }
        other => Err(WeightError::NotBuilt(format!(
            "scheduler {other} is not built"
        ))),
    }
}

/// Qwen-Image-2512 flow schedule from mflux `ModelConfig` + `shift_sigmas`.
fn qwen_sigmas(steps: u32, width: u32, height: u32) -> Vec<f32> {
    const BASE_SHIFT: f64 = 0.5;
    const MAX_SHIFT: f64 = 0.9;
    const BASE_SEQ: f64 = 256.0;
    const MAX_SEQ: f64 = 8192.0;
    const TERMINAL: f64 = 0.02;
    let steps = steps.max(1);
    let seq = f64::from(width) * f64::from(height) / 256.0;
    let slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ - BASE_SEQ);
    let intercept = BASE_SHIFT - slope * BASE_SEQ;
    let exp_mu = (slope * seq + intercept).exp();
    let mut shifted = Vec::with_capacity(steps as usize);
    for index in 0..steps {
        let t = if steps == 1 {
            1.0
        } else {
            let end = 1.0 / f64::from(steps);
            1.0 + (end - 1.0) * f64::from(index) / f64::from(steps - 1)
        };
        shifted.push(exp_mu / (exp_mu + (1.0 / t - 1.0)));
    }
    let scale = (1.0 - shifted[shifted.len() - 1]) / (1.0 - TERMINAL);
    let mut sigmas: Vec<f32> = shifted
        .into_iter()
        .map(|sigma| (1.0 - (1.0 - sigma) / scale) as f32)
        .collect();
    sigmas.push(0.0);
    sigmas
}

#[cfg(test)]
mod tests {
    use super::schedule;

    #[test]
    fn published_schedule_matches_mflux_qwen_shift() {
        use crate::common::ops::flow_match_shift_sigmas;
        let card = schedule(4, 1024, 1024, 16, None, None, false).unwrap();
        let expect = [1.0, 0.76670946, 0.4556138, 0.02, 0.0];
        assert_eq!(card.len(), expect.len());
        for (got, want) in card.iter().zip(expect) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        let named = schedule(4, 1024, 1024, 16, Some("flow_match_euler"), Some(6.0), true).unwrap();
        assert_eq!(named, card);
        let shifted = schedule(
            4,
            1024,
            1024,
            16,
            Some("flow_match_euler_discrete"),
            Some(6.0),
            false,
        )
        .unwrap();
        assert_eq!(shifted, flow_match_shift_sigmas(4, 6.0));
        let linear = schedule(4, 1024, 768, 16, Some("linear"), None, false).unwrap();
        assert!((linear[linear.len() - 2] - 0.02).abs() < 1e-5, "{linear:?}");
        assert_eq!(*linear.last().unwrap(), 0.0);
        let bad = schedule(
            4,
            1024,
            1024,
            16,
            Some("flow_match_euler"),
            Some(0.0),
            false,
        );
        assert!(bad.unwrap_err().to_string().contains("scheduler shift"));
    }
}
