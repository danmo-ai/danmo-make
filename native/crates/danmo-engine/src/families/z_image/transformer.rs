//! Z-Image denoise session. The topology lives in `graph`. The step loop does not
//! reuse Flux CFG: Z-Image is `cond + g * (cond - uncond)`, and guidance 0 is one forward.

use super::geom::{
    patch_index, positions_image_first, rope_columns, timestep_features, token_grid, TokenGrid,
};
use super::graph::build_dit;
use super::weights::ZImageConfig;
use crate::runtime::{GgmlContext, ReusedGraph, RuntimeError};
use std::collections::HashMap;

const STEP_MEM: usize = 8 * 1024 * 1024;
const GRAPH_MEM: usize = 64 * 1024 * 1024;

pub struct ZImageDenoise {
    graph: ReusedGraph,
    steps_run: u32,
    config: ZImageConfig,
    grid: TokenGrid,
}

impl ZImageDenoise {
    pub fn build(
        ctx: &GgmlContext,
        config: ZImageConfig,
        width: u32,
        height: u32,
        cap_real: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let grid = token_grid(width, height, cap_real, config.vae_scale)?;
        let mut builder = ctx.builder(GRAPH_MEM)?;
        let built = build_dit(&mut builder, config, grid, types)?;
        let graph = builder.finish(built.output)?;
        Ok(Self {
            graph,
            steps_run: 0,
            config,
            grid,
        })
    }

    pub fn load_bytes(&mut self, name: &str, data: &[u8]) -> Result<(), RuntimeError> {
        self.graph.set_bytes(name, data)
    }

    pub fn latent_len(&self) -> usize {
        (self.grid.latent_w * self.grid.latent_h * self.config.in_channels) as usize
    }

    pub fn txt_len(&self) -> usize {
        (self.config.cap_dim * self.grid.cap_real) as usize
    }

    pub fn denoise(
        &mut self,
        latent: Vec<f32>,
        txt: &[f32],
        uncond_txt: Option<&[f32]>,
        sigmas: &[f32],
        guidance: f32,
        mut on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
        self.bind(latent.as_slice(), txt)?;
        let mut euler = zimage_euler(&self.graph, self.grid, self.config.in_channels)?;
        if sigmas.len() < 2 {
            return Err(RuntimeError::Message(
                "z-image sigma schedule is shorter than the step count".into(),
            ));
        }
        let total = (sigmas.len() - 1) as u32;
        // The velocity is the same shape as the latent, so the allocator may
        // reuse that buffer for the graph output. Keep x on the euler graph,
        // which the forward does not write, and copy it back before each pass.
        euler.set_f32("latent", &latent)?;
        if let Some(uncond) = uncond_txt {
            if uncond.len() != txt.len() {
                return Err(RuntimeError::Message(format!(
                    "negative embedding has {} values, prompt has {}",
                    uncond.len(),
                    txt.len()
                )));
            }
        }
        let zeros = vec![0f32; txt.len()];
        let uncond_ref = uncond_txt.unwrap_or(zeros.as_slice());
        let mut text_is_cond = true;
        for index in 0..total {
            let sigma = sigmas[index as usize];
            self.set_timestep(sigma)?;
            if !text_is_cond {
                self.graph.set_f32("txt", txt)?;
                text_is_cond = true;
            }
            ReusedGraph::copy(euler.tensor("latent")?, self.graph.tensor("latent")?)?;
            self.graph.compute_device()?;
            ReusedGraph::copy(self.graph.output(), euler.tensor("vel_cond")?)?;
            if guidance.abs() < f32::EPSILON {
                ReusedGraph::copy(self.graph.output(), euler.tensor("vel_uncond")?)?;
            } else {
                self.graph.set_f32("txt", uncond_ref)?;
                text_is_cond = false;
                ReusedGraph::copy(euler.tensor("latent")?, self.graph.tensor("latent")?)?;
                self.graph.compute_device()?;
                ReusedGraph::copy(self.graph.output(), euler.tensor("vel_uncond")?)?;
            }
            euler.set_f32("guidance", &[guidance])?;
            euler.set_f32("dt", &[sigmas[index as usize + 1] - sigma])?;
            euler.compute_device()?;
            ReusedGraph::copy(euler.output(), euler.tensor("latent")?)?;
            on_step(index + 1, total);
        }
        self.steps_run = total;
        euler.read_output()
    }

    /// One conditional forward. The returned tensor is the velocity, still on the host.
    #[cfg(test)]
    pub fn forward_velocity(
        &mut self,
        latent: &[f32],
        txt: &[f32],
        sigma: f32,
    ) -> Result<Vec<f32>, RuntimeError> {
        self.bind(latent, txt)?;
        self.set_timestep(sigma)?;
        self.graph.compute()
    }

    pub fn graph_computes(&self) -> u32 {
        self.graph.compute_count()
    }

    #[cfg(test)]
    pub fn host_readbacks(&self) -> u32 {
        self.graph.host_readbacks()
    }

    #[cfg(test)]
    pub fn steps_run(&self) -> u32 {
        self.steps_run
    }

    fn bind(&self, latent: &[f32], txt: &[f32]) -> Result<(), RuntimeError> {
        if latent.len() != self.latent_len() {
            return Err(RuntimeError::Message(format!(
                "latent has {} values, graph expects {}",
                latent.len(),
                self.latent_len()
            )));
        }
        if txt.len() != self.txt_len() {
            return Err(RuntimeError::Message(format!(
                "text embedding has {} values, graph expects {}",
                txt.len(),
                self.txt_len()
            )));
        }
        self.graph.set_f32("latent", latent)?;
        self.graph.set_f32("txt", txt)?;
        let ids = positions_image_first(self.grid);
        let columns = rope_columns(&ids);
        for (axis, column) in columns.iter().enumerate() {
            self.graph.set_i32(&format!("rope_axis_{axis}"), column)?;
        }
        let index = patch_index(self.grid, self.config.in_channels);
        self.graph.set_i32("patch_index", &index)?;
        Ok(())
    }

    fn set_timestep(&self, sigma: f32) -> Result<(), RuntimeError> {
        let t = (1.0 - sigma) * self.config.t_scale;
        let freq = timestep_features(t, self.config.freq as usize);
        self.graph.set_f32("temb_freq", &freq)
    }
}

/// `latent + dt * (cond + g * (cond - uncond))` on `[W, H, C]`.
fn zimage_euler(
    parent: &ReusedGraph,
    grid: TokenGrid,
    channels: i64,
) -> Result<ReusedGraph, RuntimeError> {
    let mut b = parent.child(STEP_MEM)?;
    let latent = b.f32_3d("latent", grid.latent_w, grid.latent_h, channels, true);
    let cond = b.f32_3d("vel_cond", grid.latent_w, grid.latent_h, channels, true);
    let uncond = b.f32_3d("vel_uncond", grid.latent_w, grid.latent_h, channels, true);
    let guidance = b.f32_1d("guidance", 1, true);
    let dt = b.f32_1d("dt", 1, true);
    let diff = b.sub(cond, uncond);
    let guided = b.add(cond, b.mul(diff, b.repeat(guidance, diff)));
    let next = b.add(latent, b.mul(guided, b.repeat(dt, latent)));
    if next.is_null() {
        return Err(RuntimeError::Init);
    }
    b.finish(next)
}
