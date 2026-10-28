//! One Qwen-Image forward. The step loop is not here.

use super::graph::{bind_rope, build_dit};
use super::weights::QwenImageConfig;
use crate::runtime::{GgmlContext, ReusedGraph, RuntimeError};
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

    pub fn node_count(&self) -> i32 {
        self.graph.node_count()
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
}
