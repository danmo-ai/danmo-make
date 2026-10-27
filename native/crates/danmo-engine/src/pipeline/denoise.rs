//! Denoise loop. Inputs are written after the graph is built. Steps do not
//! read activations back to the host. The preview read happens once, at the end.

use crate::common::ops::flux_timestep_embed;
use crate::runtime::{euler_graph, DeviceStep, ReusedGraph, RuntimeError};

/// Replay `forward` once per conditional pass. Latents, CFG, and Euler stay on
/// the device. The host sees the latent only after the last step.
pub fn denoise_resident(
    forward: &mut ReusedGraph,
    steps: &[DeviceStep],
    latent0: &[f32],
    txt: &[f32],
    guidance: f32,
    channels: i64,
    seq: i64,
    mut on_step: impl FnMut(u32, u32),
) -> Result<Vec<f32>, RuntimeError> {
    let total = steps.len() as u32;
    if total == 0 {
        return Err(RuntimeError::Message("denoise has no steps".into()));
    }
    let mut euler = euler_graph(forward, channels, seq)?;
    forward.set_f32("latent", latent0)?;
    forward.set_f32("txt", txt)?;
    let zeros = vec![0f32; txt.len()];
    let mut text_is_cond = true;
    for (index, step) in steps.iter().enumerate() {
        forward.set_f32("temb_freq", &step.temb)?;
        if !text_is_cond {
            forward.set_f32("txt", txt)?;
            text_is_cond = true;
        }
        forward.compute_device()?;
        ReusedGraph::copy(forward.output(), euler.tensor("vel_cond")?)?;
        if (guidance - 1.0).abs() < f32::EPSILON {
            ReusedGraph::copy(forward.output(), euler.tensor("vel_uncond")?)?;
        } else {
            forward.set_f32("txt", &zeros)?;
            text_is_cond = false;
            forward.compute_device()?;
            ReusedGraph::copy(forward.output(), euler.tensor("vel_uncond")?)?;
        }
        if index == 0 {
            ReusedGraph::copy(forward.tensor("latent")?, euler.tensor("latent")?)?;
        }
        euler.set_f32("guidance", &[guidance])?;
        euler.set_f32("dt", &[step.sigma_next - step.sigma])?;
        euler.compute_device()?;
        ReusedGraph::copy(euler.output(), forward.tensor("latent")?)?;
        ReusedGraph::copy(euler.output(), euler.tensor("latent")?)?;
        on_step(index as u32 + 1, total);
    }
    euler.read_output()
}

/// Host timesteps for a caller-built sigma list. Fed into [`denoise_resident`].
/// `sigmas` includes the terminal 0, so the step count is `len - 1`.
pub fn flow_steps(sigmas: &[f32], freq_dim: usize) -> Result<Vec<DeviceStep>, RuntimeError> {
    if sigmas.len() < 2 {
        return Err(RuntimeError::Message(
            "denoise schedule needs at least two sigmas".into(),
        ));
    }
    Ok(sigmas
        .windows(2)
        .map(|pair| DeviceStep {
            sigma: pair[0],
            sigma_next: pair[1],
            temb: flux_timestep_embed(pair[0], freq_dim),
        })
        .collect())
}
