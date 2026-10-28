//! ACE-Step flow-match schedules from acestep.cpp / main.
//!
//! Turbo uses the published 8-step shift-3 table (shift on
//! `linspace(1, 0, 9)` without the terminal 0, so the last t is 0.3). SFT uses
//! `linspace(1, 0, steps)` then the same static shift (last t is 0). Schedules
//! do **not** append an extra trailing 0: the Euler loop treats the step after
//! the last published t as 0.

use crate::weights::WeightError;

/// Turbo 8-step shift-3 table (maolan / acestep.cpp / main).
pub const TURBO_TIMESTEPS: [f32; 8] = [
    1.0,
    0.954_545_44,
    0.9,
    0.833_333_3,
    0.75,
    0.642_857_13,
    0.5,
    0.3,
];

pub const TURBO_SHIFT: f32 = 3.0;
pub const SFT_DEFAULT_STEPS: u32 = 50;
pub const SFT_DEFAULT_SHIFT: f32 = 3.0;

/// `shift * t / (1 + (shift - 1) * t)` on a descending linspace.
pub fn shift_schedule(steps: u32, shift: f32) -> Result<Vec<f32>, WeightError> {
    if steps < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step schedule needs at least one step".into(),
        ));
    }
    if !shift.is_finite() || shift <= 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step scheduler shift {shift} is not a positive finite value"
        )));
    }
    let mut out = Vec::with_capacity(steps as usize);
    for index in 0..steps {
        let t = if steps == 1 {
            1.0
        } else {
            1.0 - (index as f32) / (steps as f32 - 1.0)
        };
        out.push(shift * t / (1.0 + (shift - 1.0) * t));
    }
    Ok(out)
}

pub fn turbo_schedule() -> Vec<f32> {
    TURBO_TIMESTEPS.to_vec()
}

pub fn sft_schedule(steps: u32, shift: Option<f32>) -> Result<Vec<f32>, WeightError> {
    shift_schedule(steps, shift.unwrap_or(SFT_DEFAULT_SHIFT))
}

#[cfg(test)]
mod tests {
    use super::{shift_schedule, turbo_schedule, TURBO_TIMESTEPS, TURBO_SHIFT};

    #[test]
    fn turbo_table_matches_the_published_eight_steps() {
        let got = turbo_schedule();
        assert_eq!(got.len(), 8);
        for (left, right) in got.iter().zip(TURBO_TIMESTEPS) {
            assert!((left - right).abs() < 1e-6, "{left} vs {right}");
        }
        // Turbo is shift-3 on linspace(1, 0, 9) without the terminal 0.
        for (index, &published) in TURBO_TIMESTEPS.iter().enumerate() {
            let t = 1.0 - (index as f32) / 8.0;
            let shifted = TURBO_SHIFT * t / (1.0 + (TURBO_SHIFT - 1.0) * t);
            assert!(
                (shifted - published).abs() < 1e-5,
                "step {index}: {shifted} vs {published}"
            );
        }
        let sft = shift_schedule(8, TURBO_SHIFT).unwrap();
        assert!((sft[0] - 1.0).abs() < 1e-6);
        assert!((sft[7] - 0.0).abs() < 1e-6, "{}", sft[7]);
    }
}
