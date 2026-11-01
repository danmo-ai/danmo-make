//! One Qwen-Image forward and the Euler loop around it.
//!
//! The velocity stays in packed `[in_channels, seq]` space. Guidance above 1
//! replays the same graph with the negative embedding and its own text mask,
//! then applies `eps_u + g * (eps_c - eps_u)`. Other guidance values use one
//! conditional pass.

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
    cond: Option<(i64, i64)>,
    target_seq: i64,
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
        Self::build_with_cond(ctx, config, grid_w, grid_h, seq_txt, types, None)
    }

    pub fn build_edit(
        ctx: &GgmlContext,
        config: QwenImageConfig,
        grid_w: i64,
        grid_h: i64,
        cond_w: i64,
        cond_h: i64,
        seq_txt: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        Self::build_with_cond(
            ctx,
            config,
            grid_w,
            grid_h,
            seq_txt,
            types,
            Some((cond_w, cond_h)),
        )
    }

    fn build_with_cond(
        ctx: &GgmlContext,
        config: QwenImageConfig,
        grid_w: i64,
        grid_h: i64,
        seq_txt: i64,
        types: &HashMap<String, u32>,
        cond: Option<(i64, i64)>,
    ) -> Result<Self, RuntimeError> {
        let cond_seq = cond.map(|(w, h)| w * h).unwrap_or(0);
        let target_seq = grid_w * grid_h;
        let mut builder = ctx.builder(GRAPH_MEM)?;
        let built = build_dit(
            &mut builder,
            config,
            grid_w,
            grid_h,
            seq_txt,
            types,
            cond_seq,
        )?;
        let graph = builder.finish(built.output)?;
        Ok(Self {
            graph,
            in_channels: config.in_channels,
            joint: config.joint_attention_dim,
            freq: config.freq_dim,
            seq_img: target_seq + cond_seq,
            seq_txt,
            grid_w,
            grid_h,
            cond,
            target_seq,
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
        let freq_len = (self.freq * if self.cond.is_some() { self.seq_img } else { 1 }) as usize;
        if latent.len() != latent_len || txt.len() != txt_len || temb_freq.len() != freq_len {
            return Err(RuntimeError::Message(format!(
                "qwen-image forward lengths latent {} txt {} freq {}, graph expects {latent_len} {txt_len} {freq_len}",
                latent.len(),
                txt.len(),
                temb_freq.len()
            )));
        }
        bind_rope(
            &self.graph,
            self.grid_w,
            self.grid_h,
            self.seq_txt,
            self.cond,
        )?;
        self.graph.set_f32("latent", latent)?;
        self.graph.set_f32("txt", txt)?;
        self.graph.set_f32("temb_freq", temb_freq)?;
        self.graph
            .set_f32("attn_mask", &self.attention_mask(self.seq_txt)?)?;
        self.graph.compute()
    }

    /// Euler on the packed latent. `sigmas` includes the terminal 0, so the
    /// forward count is `sigmas.len() - 1`. Each forward embeds `sigmas[i]`.
    pub fn denoise(
        &mut self,
        latent: &[f32],
        txt: &[f32],
        txt_tokens: i64,
        uncond: Option<(&[f32], i64)>,
        sigmas: &[f32],
        guidance: f32,
        mut on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
        if !guidance.is_finite() {
            return Err(RuntimeError::Message(
                "qwen-image guidance must be finite".into(),
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
        let cond_mask = self.attention_mask(txt_tokens)?;
        let uncond = if guidance > 1.0 {
            let (values, tokens) = uncond.ok_or_else(|| {
                RuntimeError::Message("qwen-image cfg needs a negative embedding".into())
            })?;
            if values.len() != txt_len {
                return Err(RuntimeError::Message(format!(
                    "qwen-image negative embedding has {} values, graph expects {txt_len}",
                    values.len()
                )));
            }
            Some((values, self.attention_mask(tokens)?))
        } else {
            None
        };
        bind_rope(
            &self.graph,
            self.grid_w,
            self.grid_h,
            self.seq_txt,
            self.cond,
        )?;
        self.graph.set_f32("txt", txt)?;
        self.graph.set_f32("attn_mask", &cond_mask)?;
        let mut euler = euler_graph(&self.graph, self.in_channels, self.seq_img)?;
        euler.set_f32("latent", latent)?;
        let total = (sigmas.len() - 1) as u32;
        let mut text_is_cond = true;
        for index in 0..total {
            let sigma = sigmas[index as usize];
            let freq = self.timestep_inputs(sigma);
            self.graph.set_f32("temb_freq", &freq)?;
            if !text_is_cond {
                self.graph.set_f32("txt", txt)?;
                self.graph.set_f32("attn_mask", &cond_mask)?;
                text_is_cond = true;
            }
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
            if let Some((negative, mask)) = uncond.as_ref() {
                self.graph.set_f32("txt", negative)?;
                self.graph.set_f32("attn_mask", mask)?;
                text_is_cond = false;
                ReusedGraph::copy(euler.tensor("latent")?, self.graph.tensor("latent")?)?;
                self.graph.compute_device()?;
                let velocity = self.graph.peek_output()?;
                if let Some(detail) = non_finite(
                    &format!("qwen-image negative velocity at step {}", index + 1),
                    &velocity,
                ) {
                    return Err(RuntimeError::Message(detail));
                }
                ReusedGraph::copy(self.graph.output(), euler.tensor("vel_uncond")?)?;
            } else {
                ReusedGraph::copy(self.graph.output(), euler.tensor("vel_uncond")?)?;
            }
            euler.set_f32("guidance", &[guidance])?;
            euler.set_f32("dt", &[sigmas[index as usize + 1] - sigma])?;
            euler.compute_device()?;
            let mut next = euler.peek_output()?;
            if self.cond.is_some() {
                let start = (self.in_channels * self.target_seq) as usize;
                next[start..].copy_from_slice(&latent[start..]);
            }
            if let Some(detail) = non_finite(
                &format!("qwen-image latent after step {}", index + 1),
                &next,
            ) {
                return Err(RuntimeError::Message(detail));
            }
            ReusedGraph::copy(euler.output(), euler.tensor("latent")?)?;
            on_step(index + 1, total);
        }
        let output = euler.read_output()?;
        if self.cond.is_some() {
            Ok(output[..(self.in_channels * self.target_seq) as usize].to_vec())
        } else {
            Ok(output)
        }
    }

    fn timestep_inputs(&self, sigma: f32) -> Vec<f32> {
        let base = timestep_features(sigma, self.freq as usize);
        if self.cond.is_none() {
            return base;
        }
        let mut out = vec![0.0; (self.freq * self.seq_img) as usize];
        for token in 0..self.target_seq {
            let start = token as usize * self.freq as usize;
            out[start..start + base.len()].copy_from_slice(&base);
        }
        out
    }

    fn attention_mask(&self, valid_txt: i64) -> Result<Vec<f32>, RuntimeError> {
        if valid_txt < 1 || valid_txt > self.seq_txt {
            return Err(RuntimeError::Message(format!(
                "qwen-image valid text length {valid_txt} is outside 1..={} ",
                self.seq_txt
            )));
        }
        let seq = self.seq_txt + self.seq_img;
        let mut mask = vec![0.0; (seq * seq) as usize];
        for query in 0..seq {
            for key in valid_txt..self.seq_txt {
                mask[(key + query * seq) as usize] = f32::NEG_INFINITY;
            }
        }
        Ok(mask)
    }
}
