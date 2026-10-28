//! One Qwen-Image forward and the Euler loop around it.
//!
//! The velocity stays in packed `[in_channels, seq]` space. Guidance 1 is the
//! conditional pass alone (`eps_u + 1 * (eps_c - eps_u)` is `eps_c`). Any other
//! guidance needs the negative embedding, which waits on the tokenizer.

use super::graph::{bind_rope, build_dit};
use super::ids::timestep_features;
use super::weights::QwenImageConfig;
use crate::common::ops::non_finite;
use crate::runtime::{euler_graph, GgmlContext, ReusedGraph, RuntimeError};
use std::collections::HashMap;

const GRAPH_MEM: usize = 64 * 1024 * 1024;

pub struct QwenImageForward {
    graph: ReusedGraph,
    in_channels: i64,
    joint: i64,
    freq: i64,
    seq_img: i64,
    seq_txt: i64,
    grid_w: i64,
    grid_h: i64,
}

impl QwenImageForward {
    pub fn build(
        ctx: &GgmlContext,
        config: QwenImageConfig,
        grid_w: i64,
        grid_h: i64,
        seq_txt: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut builder = ctx.builder(GRAPH_MEM)?;
        let built = build_dit(&mut builder, config, grid_w, grid_h, seq_txt, types)?;
        let graph = builder.finish(built.output)?;
        Ok(Self {
            graph,
            in_channels: config.in_channels,
            joint: config.joint_attention_dim,
            freq: config.freq_dim,
            seq_img: grid_w * grid_h,
            seq_txt,
            grid_w,
            grid_h,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    pub fn load_bytes(&mut self, name: &str, data: &[u8]) -> Result<(), RuntimeError> {
        self.graph.set_bytes(name, data)
    }

    pub fn node_count(&self) -> i32 {
        self.graph.node_count()
    }

    pub fn graph_computes(&self) -> u32 {
        self.graph.compute_count()
    }

    pub fn latent_len(&self) -> usize {
        (self.in_channels * self.seq_img) as usize
    }

    pub fn forward(
        &mut self,
        latent: &[f32],
        txt: &[f32],
        temb_freq: &[f32],
    ) -> Result<Vec<f32>, RuntimeError> {
        let latent_len = (self.in_channels * self.seq_img) as usize;
        let txt_len = (self.joint * self.seq_txt) as usize;
        let freq_len = self.freq as usize;
        if latent.len() != latent_len || txt.len() != txt_len || temb_freq.len() != freq_len {
            return Err(RuntimeError::Message(format!(
                "qwen-image forward lengths latent {} txt {} freq {}, graph expects {latent_len} {txt_len} {freq_len}",
                latent.len(),
                txt.len(),
                temb_freq.len()
            )));
        }
        bind_rope(&self.graph, self.grid_w, self.grid_h, self.seq_txt)?;
        self.graph.set_f32("latent", latent)?;
        self.graph.set_f32("txt", txt)?;
        self.graph.set_f32("temb_freq", temb_freq)?;
        self.graph.compute()
    }

    /// Euler on the packed latent. `sigmas` includes the terminal 0, so the
    /// forward count is `sigmas.len() - 1`. Each forward embeds `sigmas[i]`.
    pub fn denoise(
        &mut self,
        latent: &[f32],
        txt: &[f32],
        sigmas: &[f32],
        guidance: f32,
        mut on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
        if !guidance.is_finite() || (guidance - 1.0).abs() >= f32::EPSILON {
            return Err(RuntimeError::Message(
                "cfg needs the negative embedding path".into(),
            ));
        }
        if sigmas.len() < 2 {
            return Err(RuntimeError::Message(
                "qwen-image sigma schedule is shorter than one step".into(),
            ));
        }
        let latent_len = (self.in_channels * self.seq_img) as usize;
        let txt_len = (self.joint * self.seq_txt) as usize;
        if latent.len() != latent_len || txt.len() != txt_len {
            return Err(RuntimeError::Message(format!(
                "qwen-image denoise lengths latent {} txt {}, graph expects {latent_len} {txt_len}",
                latent.len(),
                txt.len()
            )));
        }
        bind_rope(&self.graph, self.grid_w, self.grid_h, self.seq_txt)?;
        self.graph.set_f32("txt", txt)?;
        let mut euler = euler_graph(&self.graph, self.in_channels, self.seq_img)?;
        euler.set_f32("latent", latent)?;
        let total = (sigmas.len() - 1) as u32;
        for index in 0..total {
            let sigma = sigmas[index as usize];
            let freq = timestep_features(sigma, self.freq as usize);
            self.graph.set_f32("temb_freq", &freq)?;
            ReusedGraph::copy(euler.tensor("latent")?, self.graph.tensor("latent")?)?;
            self.graph.compute_device()?;
            let velocity = self.graph.peek_output()?;
            if let Some(detail) = non_finite(
                &format!("qwen-image velocity at step {}", index + 1),
                &velocity,
            ) {
                return Err(RuntimeError::Message(detail));
            }
            ReusedGraph::copy(self.graph.output(), euler.tensor("vel_cond")?)?;
            ReusedGraph::copy(self.graph.output(), euler.tensor("vel_uncond")?)?;
            euler.set_f32("guidance", &[1.0])?;
            euler.set_f32("dt", &[sigmas[index as usize + 1] - sigma])?;
            euler.compute_device()?;
            let next = euler.peek_output()?;
            if let Some(detail) = non_finite(
                &format!("qwen-image latent after step {}", index + 1),
                &next,
            ) {
                return Err(RuntimeError::Message(detail));
            }
            ReusedGraph::copy(euler.output(), euler.tensor("latent")?)?;
            on_step(index + 1, total);
        }
        euler.read_output()
    }
}
