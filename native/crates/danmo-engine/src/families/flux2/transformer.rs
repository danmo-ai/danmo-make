//! Flux2 denoise session. The topology lives in `graph`; the step loop does not.

use super::graph::build_dit;
use super::weights::Flux2Config;
use crate::pipeline::denoise::{denoise_resident, flow_steps};
use crate::runtime::{GgmlContext, ReusedGraph, RuntimeError};
use std::collections::HashMap;

pub struct Flux2Denoise {
    graph: ReusedGraph,
    pub steps_run: u32,
    seq_img: i64,
    seq_txt: i64,
    in_channels: i64,
    joint: i64,
    freq: i64,
    grid_w: i64,
    grid_h: i64,
    axis_rope: bool,
}

impl Flux2Denoise {
    pub fn build(
        ctx: &GgmlContext,
        config: Flux2Config,
        grid_w: i64,
        grid_h: i64,
        seq_txt: i64,
    ) -> Result<Self, RuntimeError> {
        Self::build_typed(ctx, config, grid_w, grid_h, seq_txt, &HashMap::new())
    }

    /// `types` maps a GGUF tensor name to its stored ggml type. Missing names are F32.
    pub fn build_typed(
        ctx: &GgmlContext,
        config: Flux2Config,
        grid_w: i64,
        grid_h: i64,
        seq_txt: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut builder = ctx.builder(64 * 1024 * 1024)?;
        let built = build_dit(&mut builder, config, grid_w, grid_h, seq_txt, types)?;
        let graph = builder.finish(built.output)?;
        Ok(Self {
            graph,
            steps_run: 0,
            seq_img: grid_w * grid_h,
            seq_txt,
            in_channels: config.in_channels,
            joint: config.joint_attention_dim,
            freq: config.freq_dim,
            grid_w,
            grid_h,
            axis_rope: built.rope.axis,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    pub fn load_bytes(&mut self, name: &str, data: &[u8]) -> Result<(), RuntimeError> {
        self.graph.set_bytes(name, data)
    }

    pub fn latent_len(&self) -> usize {
        (self.in_channels * self.seq_img) as usize
    }

    pub fn txt_len(&self) -> usize {
        (self.joint * self.seq_txt) as usize
    }

    pub fn denoise(
        &mut self,
        latent: Vec<f32>,
        txt: &[f32],
        steps: u32,
        width: u32,
        height: u32,
        shift: bool,
        guidance: f32,
        on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
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
        self.bind_rope()?;
        let schedule = flow_steps(steps, width, height, shift, self.freq as usize);
        let latents = denoise_resident(
            &mut self.graph,
            &schedule,
            &latent,
            txt,
            guidance,
            self.in_channels,
            self.seq_img,
            on_step,
        )?;
        self.steps_run = steps;
        Ok(latents)
    }

    pub fn graph_computes(&self) -> u32 {
        self.graph.compute_count()
    }

    pub fn host_readbacks(&self) -> u32 {
        self.graph.host_readbacks()
    }

    fn bind_rope(&self) -> Result<(), RuntimeError> {
        let seq = self.seq_txt + self.seq_img;
        if self.axis_rope {
            let mut axes = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for s in 0..seq {
                let ids = if s < self.seq_txt {
                    [0, 0, 0, s as i32]
                } else {
                    let i = s - self.seq_txt;
                    let h = (i / self.grid_w) as i32;
                    let w = (i % self.grid_w) as i32;
                    if h >= self.grid_h as i32 {
                        return Err(RuntimeError::Message(
                            "image token is outside the rope grid".into(),
                        ));
                    }
                    [0, h, w, 0]
                };
                for (axis, id) in axes.iter_mut().zip(ids) {
                    axis.push(id);
                }
            }
            for (i, axis) in axes.iter().enumerate() {
                self.graph.set_i32(&format!("rope_axis_{i}"), axis)?;
            }
        } else {
            let positions: Vec<i32> = (0..seq).map(|i| i as i32).collect();
            self.graph.set_i32("token_index", &positions)?;
        }
        Ok(())
    }
}
