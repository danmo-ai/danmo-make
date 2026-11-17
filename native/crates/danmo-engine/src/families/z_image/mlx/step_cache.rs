use danmo_mlx::{Array, Result, Stream};

const COEFFICIENTS: [f64; 5] = [
    -898.9907628349583,
    367.7086118008557,
    -45.41511572598643,
    3.95114319842774,
    0.0,
];

#[derive(Debug, Clone)]
pub struct Decision {
    pub kind: &'static str,
    pub should_compute: bool,
    pub should_update_cache: bool,
    pub rel_l1: Option<f64>,
    pub predicted_distance: Option<f64>,
    pub accumulated_distance: f64,
}

pub struct StepCache {
    enabled: bool,
    threshold: f64,
    num_steps: usize,
    previous_mod_input: Option<Array>,
    cached_residual: Option<Array>,
    accumulated_distance: f64,
    pub computed_count: usize,
    pub skipped_count: usize,
    pub history: Vec<Decision>,
}

impl StepCache {
    pub fn z_image(num_steps: usize) -> Self {
        Self::new(num_steps, 0.11)
    }

    pub fn new(num_steps: usize, threshold: f64) -> Self {
        Self {
            enabled: num_steps > 8,
            threshold,
            num_steps,
            previous_mod_input: None,
            cached_residual: None,
            accumulated_distance: 0.0,
            computed_count: 0,
            skipped_count: 0,
            history: Vec::new(),
        }
    }

    pub fn gate(&mut self, mod_in: &Array, step_idx: usize, stream: &Stream) -> Result<Decision> {
        let decision = if !self.enabled || self.threshold <= 0.0 {
            self.computed("computed", false)
        } else if step_idx < 1 || step_idx >= self.num_steps - 1 {
            self.computed("forced", false)
        } else if mod_in
            .isfinite_all(stream)?
            .astype(danmo_mlx::Dtype::Float32, stream)?
            .to_vec_f32(stream)?[0]
            == 0.0
        {
            self.computed("numerical-miss", false)
        } else if self.previous_mod_input.is_none() {
            self.computed("computed", true)
        } else {
            let previous = self.previous_mod_input.as_ref().unwrap();
            let numerator = Array::subtract(mod_in, previous, stream)?
                .abs(stream)?
                .mean(false, stream)?
                .to_vec_f32(stream)?[0] as f64;
            let denominator = previous
                .abs(stream)?
                .mean(false, stream)?
                .to_vec_f32(stream)?[0] as f64;
            let rel_l1 = numerator / denominator.max(1e-12);
            let predicted = poly_eval(rel_l1).max(0.0);
            let accumulated = self.accumulated_distance + predicted;
            if accumulated < self.threshold {
                self.accumulated_distance = accumulated;
                Decision {
                    kind: "skipped",
                    should_compute: false,
                    should_update_cache: false,
                    rel_l1: Some(rel_l1),
                    predicted_distance: Some(predicted),
                    accumulated_distance: accumulated,
                }
            } else {
                self.accumulated_distance = 0.0;
                Decision {
                    kind: "computed",
                    should_compute: true,
                    should_update_cache: true,
                    rel_l1: Some(rel_l1),
                    predicted_distance: Some(predicted),
                    accumulated_distance: 0.0,
                }
            }
        };
        if decision.should_compute {
            self.computed_count += 1;
        } else {
            self.skipped_count += 1;
        }
        self.history.push(decision.clone());
        Ok(decision)
    }

    fn computed(&self, kind: &'static str, update: bool) -> Decision {
        Decision {
            kind,
            should_compute: true,
            should_update_cache: update,
            rel_l1: None,
            predicted_distance: None,
            accumulated_distance: self.accumulated_distance,
        }
    }

    pub fn cached_residual(&self) -> Option<&Array> {
        self.cached_residual.as_ref()
    }

    pub fn store_computed(&mut self, decision: &Decision, mod_in: &Array, residual: &Array) {
        if decision.should_update_cache {
            self.previous_mod_input = Some(mod_in.clone());
        }
        self.cached_residual = Some(residual.clone());
    }

    /// Drop the per-run gate/residual state so one resident model can serve
    /// consecutive generations (main's `reset_for_new_generation`).
    pub fn reset(&mut self) {
        self.previous_mod_input = None;
        self.cached_residual = None;
        self.accumulated_distance = 0.0;
        self.computed_count = 0;
        self.skipped_count = 0;
        self.history.clear();
    }
}

fn poly_eval(x: f64) -> f64 {
    let [c4, c3, c2, c1, c0] = COEFFICIENTS;
    ((((c4 * x) + c3) * x + c2) * x + c1) * x + c0
}

#[cfg(test)]
mod tests {
    use super::StepCache;
    use danmo_mlx::{Array, Stream};

    #[test]
    fn gate_computes_skips_and_preserves_previous_input() {
        let stream = Stream::cpu().unwrap();
        let first = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let second = Array::from_f32_slice(&[1.1, 2.1, 3.1], &[1, 3]).unwrap();
        let residual = Array::from_f32_slice(&[0.5, 0.5, 0.5], &[1, 3]).unwrap();
        let mut cache = StepCache::new(9, 1.0);
        let decision = cache.gate(&first, 0, &stream).unwrap();
        assert_eq!(decision.kind, "forced");
        cache.store_computed(&decision, &first, &residual);
        let decision = cache.gate(&second, 1, &stream).unwrap();
        assert_eq!(decision.kind, "computed");
        cache.store_computed(&decision, &second, &residual);
        let decision = cache.gate(&second, 2, &stream).unwrap();
        assert_eq!(decision.kind, "skipped");
        assert_eq!(cache.computed_count, 2);
        assert_eq!(cache.skipped_count, 1);
        assert!(cache.cached_residual().is_some());
        assert!(decision.rel_l1.unwrap() < 1e-6);
        let decision = cache.gate(&second, 8, &stream).unwrap();
        assert_eq!(decision.kind, "forced");
    }

    #[test]
    fn reset_clears_run_state_for_the_next_generation() {
        let stream = Stream::cpu().unwrap();
        let input = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let residual = Array::from_f32_slice(&[0.5, 0.5, 0.5], &[1, 3]).unwrap();
        let mut cache = StepCache::new(9, 1.0);
        let decision = cache.gate(&input, 1, &stream).unwrap();
        cache.store_computed(&decision, &input, &residual);
        assert_eq!(cache.computed_count, 1);
        assert!(cache.cached_residual().is_some());
        cache.reset();
        assert_eq!(cache.computed_count, 0);
        assert_eq!(cache.skipped_count, 0);
        assert!(cache.history.is_empty());
        assert!(cache.cached_residual().is_none());
        // After reset the first gated step must compute again, not skip
        // against the previous generation's input.
        let decision = cache.gate(&input, 1, &stream).unwrap();
        assert_eq!(decision.kind, "computed");
        assert!(decision.should_update_cache);
    }
}
