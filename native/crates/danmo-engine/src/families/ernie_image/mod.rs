#[cfg(feature = "mlx")]
mod mlx;
mod session;

pub(crate) use session::ErnieImageFamily;

pub fn schedule(steps: u32, shift: f32) -> Result<Vec<f32>, String> {
    if steps == 0 || !shift.is_finite() || shift <= 0.0 {
        return Err(format!(
            "ERNIE steps {steps} or scheduler shift {shift} is invalid"
        ));
    }
    let mut sigmas = Vec::with_capacity(steps as usize + 1);
    for index in 0..steps {
        let sigma = 1.0 - index as f32 / steps as f32;
        sigmas.push(shift * sigma / (1.0 + (shift - 1.0) * sigma));
    }
    sigmas.push(0.0);
    Ok(sigmas)
}

#[cfg(test)]
mod tests {
    use super::schedule;

    #[test]
    fn schedule_matches_official_shifted_linspace() {
        let actual = schedule(4, 4.0).unwrap();
        let expected = [1.0, 0.9230769, 0.8, 0.5714286, 0.0];
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert!(schedule(0, 4.0).is_err());
        assert!(schedule(4, 0.0).is_err());
    }
}
