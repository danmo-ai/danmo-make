//! ACE-Step DiT session: one forward and the guidance-1 Euler loop.
//!
//! Velocity is `[out_channels, T]`. Context latents stay packed into
//! `input_latents` ahead of `xt` each step. Schedules do not carry a trailing
//! 0; the last step uses `t_next = 0`.

use super::graph::{build_dit, graph_weight_layout};
use super::schedule::{sft_schedule, turbo_schedule, SFT_DEFAULT_STEPS};
use super::weights::{AceStepConfig, TIME_FREQ_DIM};
use crate::common::ops::{non_finite, sinusoidal_timestep};
use crate::runtime::{GgmlContext, ReusedGraph, RuntimeError};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;
use std::path::Path;

const GRAPH_MEM: usize = 64 * 1024 * 1024;

pub struct AceStepDit {
    graph: ReusedGraph,
    config: AceStepConfig,
    frames: i64,
    enc_seq: i64,
    seq: i64,
    pub steps_run: u32,
}

impl AceStepDit {
    pub fn build(
        ctx: &GgmlContext,
        config: AceStepConfig,
        frames: i64,
        enc_seq: i64,
    ) -> Result<Self, RuntimeError> {
        Self::build_typed(ctx, config, frames, enc_seq, &HashMap::new())
    }

    pub fn build_typed(
        ctx: &GgmlContext,
        config: AceStepConfig,
        frames: i64,
        enc_seq: i64,
        types: &HashMap<String, u32>,
    ) -> Result<Self, RuntimeError> {
        let mut builder = ctx.builder(GRAPH_MEM)?;
        let built = build_dit(&mut builder, config, frames, enc_seq, types)?;
        let graph = builder.finish(built.output)?;
        Ok(Self {
            graph,
            config,
            frames,
            enc_seq,
            seq: built.seq,
            steps_run: 0,
        })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    #[allow(dead_code)]
    pub fn load_bytes(&mut self, name: &str, data: &[u8]) -> Result<(), RuntimeError> {
        self.graph.set_bytes(name, data)
    }

    #[allow(dead_code)]
    pub fn node_count(&self) -> i32 {
        self.graph.node_count()
    }

    pub fn graph_computes(&self) -> u32 {
        self.graph.compute_count()
    }

    pub fn xt_len(&self) -> usize {
        (self.config.out_channels * self.frames) as usize
    }

    pub fn context_len(&self) -> usize {
        ((self.config.in_channels - self.config.out_channels) * self.frames) as usize
    }

    pub fn enc_len(&self) -> usize {
        (self.config.condition_dim * self.enc_seq) as usize
    }

    pub fn forward(
        &mut self,
        xt: &[f32],
        context: &[f32],
        enc_hidden: &[f32],
        t: f32,
        t_r: f32,
    ) -> Result<Vec<f32>, RuntimeError> {
        self.bind_step(xt, context, enc_hidden, t, t_r)?;
        self.graph.compute()
    }

    /// Guidance-1 Euler. `schedule` is descending and has no trailing 0.
    pub fn denoise(
        &mut self,
        xt: &[f32],
        context: &[f32],
        enc_hidden: &[f32],
        schedule: &[f32],
        guidance: f32,
        mut on_step: impl FnMut(u32, u32),
    ) -> Result<Vec<f32>, RuntimeError> {
        if !guidance.is_finite() || (guidance - 1.0).abs() >= f32::EPSILON {
            return Err(RuntimeError::Message(
                "ACE-Step CFG / APG needs the null condition path".into(),
            ));
        }
        if schedule.is_empty() {
            return Err(RuntimeError::Message(
                "ACE-Step schedule is empty".into(),
            ));
        }
        if xt.len() != self.xt_len()
            || context.len() != self.context_len()
            || enc_hidden.len() != self.enc_len()
        {
            return Err(RuntimeError::Message(format!(
                "ACE-Step denoise lengths xt {} context {} enc {}, graph expects {} {} {}",
                xt.len(),
                context.len(),
                enc_hidden.len(),
                self.xt_len(),
                self.context_len(),
                self.enc_len()
            )));
        }
        let mut xt = xt.to_vec();
        let total = schedule.len() as u32;
        for (index, &t_curr) in schedule.iter().enumerate() {
            if !t_curr.is_finite() {
                return Err(RuntimeError::Message(format!(
                    "ACE-Step schedule step {} is not finite",
                    index + 1
                )));
            }
            let t_next = if index + 1 < schedule.len() {
                schedule[index + 1]
            } else {
                0.0
            };
            let velocity = self.forward(&xt, context, enc_hidden, t_curr, t_curr)?;
            if let Some(detail) = non_finite(
                &format!("ACE-Step velocity at step {}", index + 1),
                &velocity,
            ) {
                return Err(RuntimeError::Message(detail));
            }
            let dt = t_next - t_curr;
            for (sample, vel) in xt.iter_mut().zip(velocity.iter()) {
                *sample += dt * *vel;
            }
            if let Some(detail) =
                non_finite(&format!("ACE-Step latent after step {}", index + 1), &xt)
            {
                return Err(RuntimeError::Message(detail));
            }
            on_step((index as u32) + 1, total);
        }
        self.steps_run = total;
        Ok(xt)
    }

    fn bind_step(
        &mut self,
        xt: &[f32],
        context: &[f32],
        enc_hidden: &[f32],
        t: f32,
        t_r: f32,
    ) -> Result<(), RuntimeError> {
        if xt.len() != self.xt_len()
            || context.len() != self.context_len()
            || enc_hidden.len() != self.enc_len()
        {
            return Err(RuntimeError::Message(format!(
                "ACE-Step forward lengths xt {} context {} enc {}, graph expects {} {} {}",
                xt.len(),
                context.len(),
                enc_hidden.len(),
                self.xt_len(),
                self.context_len(),
                self.enc_len()
            )));
        }
        if !t.is_finite() || !t_r.is_finite() {
            return Err(RuntimeError::Message(
                "ACE-Step timesteps are not finite".into(),
            ));
        }
        let positions: Vec<i32> = (0..self.seq as i32).collect();
        self.graph.set_i32("positions", &positions)?;
        self.graph.set_f32("enc_hidden", enc_hidden)?;
        self.graph.set_f32(
            "temb_freq_t",
            &sinusoidal_timestep(t * 1000.0, TIME_FREQ_DIM as usize),
        )?;
        self.graph.set_f32(
            "temb_freq_r",
            &sinusoidal_timestep((t - t_r) * 1000.0, TIME_FREQ_DIM as usize),
        )?;
        let packed = pack_input(context, xt, self.config, self.frames)?;
        self.graph.set_f32("input_latents", &packed)?;
        Ok(())
    }
}

fn pack_input(
    context: &[f32],
    xt: &[f32],
    config: AceStepConfig,
    frames: i64,
) -> Result<Vec<f32>, RuntimeError> {
    let ctx_ch = config.in_channels - config.out_channels;
    let oc = config.out_channels;
    let mut out = vec![0f32; (config.in_channels * frames) as usize];
    for t in 0..frames as usize {
        let ctx_off = t * ctx_ch as usize;
        let xt_off = t * oc as usize;
        let dst = t * config.in_channels as usize;
        out[dst..dst + ctx_ch as usize].copy_from_slice(&context[ctx_off..ctx_off + ctx_ch as usize]);
        out[dst + ctx_ch as usize..dst + config.in_channels as usize]
            .copy_from_slice(&xt[xt_off..xt_off + oc as usize]);
    }
    Ok(out)
}

/// Load DiT weights into a built graph. Conv1d `proj_*` rows are remapped to the
/// 2D ranks `dit-graph.h` expects after `dit_load_proj_*_w`.
pub fn load_weights(
    dit: &mut AceStepDit,
    weights: &GgufWeights,
    config: AceStepConfig,
) -> Result<(), WeightError> {
    for (name, ne0, ne1) in graph_weight_layout(config) {
        if name == "decoder.proj_in.1.weight" {
            let raw = weights.f32_exact(
                &name,
                &[config.patch, config.in_channels, config.hidden],
            )?;
            let remapped = remap_proj_in(&raw, config.patch, config.in_channels, config.hidden);
            dit.load_f32(&name, &remapped)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            continue;
        }
        if name == "decoder.proj_out.1.weight" {
            let raw = weights.f32_exact(
                &name,
                &[config.patch, config.out_channels, config.hidden],
            )?;
            let remapped = remap_proj_out(&raw, config.patch, config.out_channels, config.hidden);
            dit.load_f32(&name, &remapped)
                .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
            continue;
        }
        let data = weights.f32_exact(&name, &[ne0, ne1])?;
        dit.load_f32(&name, &data)
            .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    }
    Ok(())
}

pub fn remap_proj_in(raw: &[f32], patch: i64, in_ch: i64, hidden: i64) -> Vec<f32> {
    let mut data = vec![0f32; (in_ch * patch * hidden) as usize];
    for h in 0..hidden {
        for ic in 0..in_ch {
            for p in 0..patch {
                let src = (h * patch * in_ch + ic * patch + p) as usize;
                let dst = (h * in_ch * patch + p * in_ch + ic) as usize;
                data[dst] = raw[src];
            }
        }
    }
    data
}

pub fn remap_proj_out(raw: &[f32], patch: i64, out_ch: i64, hidden: i64) -> Vec<f32> {
    let mut data = vec![0f32; (out_ch * patch * hidden) as usize];
    for h in 0..hidden {
        for oc in 0..out_ch {
            for p in 0..patch {
                let src = (h * patch * out_ch + oc * patch + p) as usize;
                let dst = ((p * out_ch + oc) * hidden + h) as usize;
                data[dst] = raw[src];
            }
        }
    }
    data
}

/// Smoke / turbo / SFT schedule for a config. Turbo cards take the published
/// 8-step table; SFT takes `steps` with the default shift.
pub fn schedule_for(config: AceStepConfig, steps: Option<u32>) -> Result<Vec<f32>, WeightError> {
    if config.is_turbo {
        Ok(turbo_schedule())
    } else {
        sft_schedule(steps.unwrap_or(SFT_DEFAULT_STEPS), None)
    }
}

/// Build the DiT, load weights, run guidance-1 Euler on zero context / noise.
/// Product create still needs the text tower and VAE after this returns.
pub fn prove_smoke_denoise(
    ctx: &GgmlContext,
    bundle: &Path,
    config: AceStepConfig,
) -> Result<Vec<f32>, WeightError> {
    if config != AceStepConfig::smoke() {
        return Err(WeightError::NotBuilt(
            "ACE-Step XL DiT forward is not wired into audio create (smoke path only)".into(),
        ));
    }
    let path = super::dit_gguf(bundle)?;
    let loaded = GgufWeights::open(&path)?;
    super::check_layout(&loaded, config)?;
    let frames = config.patch * 2;
    let enc_seq = 2;
    let mut dit = AceStepDit::build(ctx, config, frames, enc_seq)
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    load_weights(&mut dit, &loaded, config)?;
    // Turbo table on the smoke card: eight short forwards.
    let schedule = schedule_for(
        AceStepConfig {
            is_turbo: true,
            ..config
        },
        None,
    )?;
    let xt = vec![0.1f32; dit.xt_len()];
    let context = vec![0f32; dit.context_len()];
    let enc = vec![0f32; dit.enc_len()];
    let latents = dit
        .denoise(&xt, &context, &enc, &schedule, 1.0, |_, _| ())
        .map_err(|err| WeightError::NotBuilt(err.to_string()))?;
    if let Some(detail) = non_finite("ACE-Step smoke latent", &latents) {
        return Err(WeightError::NotBuilt(detail));
    }
    Ok(latents)
}

#[cfg(test)]
mod tests {
    use super::{remap_proj_in, remap_proj_out, AceStepDit};
    use crate::families::ace_step::schedule::turbo_schedule;
    use crate::families::ace_step::weights::AceStepConfig;
    use crate::runtime::{BackendKind, GgmlContext};

    #[test]
    fn proj_in_remap_matches_acestep_index_order() {
        let patch = 2;
        let in_ch = 3;
        let hidden = 2;
        let mut raw = vec![0f32; (patch * in_ch * hidden) as usize];
        for h in 0..hidden {
            for ic in 0..in_ch {
                for p in 0..patch {
                    raw[(h * patch * in_ch + ic * patch + p) as usize] =
                        (h * 100 + ic * 10 + p) as f32;
                }
            }
        }
        let got = remap_proj_in(&raw, patch, in_ch, hidden);
        for h in 0..hidden {
            for ic in 0..in_ch {
                for p in 0..patch {
                    let dst = (h * in_ch * patch + p * in_ch + ic) as usize;
                    assert_eq!(got[dst], (h * 100 + ic * 10 + p) as f32);
                }
            }
        }
    }

    #[test]
    fn proj_out_remap_transposes_into_linear_rows() {
        let patch = 2;
        let out_ch = 2;
        let hidden = 3;
        let mut raw = vec![0f32; (patch * out_ch * hidden) as usize];
        for h in 0..hidden {
            for oc in 0..out_ch {
                for p in 0..patch {
                    raw[(h * patch * out_ch + oc * patch + p) as usize] =
                        (h * 100 + oc * 10 + p) as f32;
                }
            }
        }
        let got = remap_proj_out(&raw, patch, out_ch, hidden);
        for h in 0..hidden {
            for oc in 0..out_ch {
                for p in 0..patch {
                    let dst = ((p * out_ch + oc) * hidden + h) as usize;
                    assert_eq!(got[dst], (h * 100 + oc * 10 + p) as f32);
                }
            }
        }
    }

    #[test]
    fn smoke_forward_and_turbo_euler_run_on_cpu() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let config = AceStepConfig::smoke();
        let frames = config.patch * 2;
        let enc_seq = 2;
        let mut dit = AceStepDit::build(&ctx, config, frames, enc_seq).unwrap();
        for (name, ne0, ne1) in super::super::graph::graph_weight_layout(config) {
            let n = (ne0 * ne1) as usize;
            // Identity-ish fills keep attention / MLP finite.
            let data = if name.contains("norm") || name.contains("scale_shift") {
                vec![1.0f32; n]
            } else if name.ends_with(".bias") {
                vec![0.0f32; n]
            } else {
                let mut v = vec![0.0f32; n];
                let dim = ne0.min(ne1) as usize;
                for i in 0..dim {
                    v[i * ne0 as usize + i] = 0.05;
                }
                v
            };
            dit.load_f32(&name, &data).unwrap();
        }
        let xt = vec![0.1f32; dit.xt_len()];
        let context = vec![0.0f32; dit.context_len()];
        let enc = vec![0.0f32; dit.enc_len()];
        let velocity = dit.forward(&xt, &context, &enc, 1.0, 1.0).unwrap();
        assert_eq!(velocity.len(), dit.xt_len());
        assert!(velocity.iter().all(|v| v.is_finite()));
        let schedule = turbo_schedule();
        let latents = dit
            .denoise(&xt, &context, &enc, &schedule, 1.0, |_, _| ())
            .unwrap();
        assert_eq!(latents.len(), dit.xt_len());
        assert!(latents.iter().all(|v| v.is_finite()));
        assert_eq!(dit.steps_run, 8);
        assert!(dit.graph_computes() >= 9);
    }
}
