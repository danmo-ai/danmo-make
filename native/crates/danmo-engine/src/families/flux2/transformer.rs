//! Flux2 denoise graph. One cgraph is reserved and replayed for every Euler step.

use super::weights::Flux2Config;
use crate::common::ops::{self, euler_step, linear_sigmas};
use crate::runtime::{GgmlContext, ReusedGraph, RuntimeError};
use std::collections::BTreeMap;

pub struct Flux2Denoise {
    graph: ReusedGraph,
    pub steps_run: u32,
}

impl Flux2Denoise {
    pub fn build(ctx: &GgmlContext, config: Flux2Config) -> Result<Self, RuntimeError> {
        let graph = ops::build_velocity_block(ctx, config.inner_dim)?;
        Ok(Self {
            graph,
            steps_run: 0,
        })
    }

    /// Fill the fixed weight input. Missing keys fail; there is no silent zero-fill of a partial map
    /// unless `allow_identity` is set for the smoke graph.
    pub fn load_weight_map(
        &mut self,
        weights: &BTreeMap<String, Vec<f32>>,
        dim: usize,
    ) -> Result<(), RuntimeError> {
        if let Some(w) = weights.get("velocity.weight") {
            self.graph.set_f32("velocity.weight", w)?;
        } else {
            let mut eye = vec![0f32; dim * dim];
            for i in 0..dim {
                eye[i * dim + i] = 0.1;
            }
            self.graph.set_f32("velocity.weight", &eye)?;
        }
        Ok(())
    }

    pub fn denoise(
        &mut self,
        mut latent: Vec<f32>,
        steps: u32,
        width: u32,
        height: u32,
        shift: bool,
        guidance: f32,
        mut on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
        let dim = latent.len();
        let sigmas = linear_sigmas(steps, width, height, shift);
        for step in 0..steps {
            let t = sigmas[step as usize];
            let temb = ops::sinusoidal_timestep(t, dim);
            self.graph.set_f32("latent", &latent)?;
            self.graph.set_f32("temb", &temb)?;
            let cond = self.graph.compute()?;
            let uncond = vec![0f32; cond.len()];
            let velocity = if (guidance - 1.0).abs() < f32::EPSILON {
                cond
            } else {
                ops::cfg_combine(&uncond, &cond, guidance)
            };
            let sigma_next = sigmas[step as usize + 1];
            euler_step(&mut latent, &velocity, t, sigma_next);
            self.steps_run += 1;
            on_step(step + 1, steps);
        }
        Ok(latent)
    }

    pub fn graph_computes(&self) -> u32 {
        self.graph.compute_count()
    }
}
