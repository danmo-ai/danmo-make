//! Denoise loop. Inputs are written after the graph is built. Each step peeks
//! the velocity and the updated latent so a non-finite sample fails at that step.

use crate::common::ops::{flux_timestep_embed, non_finite};
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
    // The host vector is the only intact copy. Gallocr may reuse the DiT
    // input buffer once `img_in` has read it, so the device tensor is not
    // read back after compute.
    forward.set_f32("latent", latent0)?;
    euler.set_f32("latent", latent0)?;
    forward.set_f32("txt", txt)?;
    let zeros = vec![0f32; txt.len()];
    let mut text_is_cond = true;
    let mut prev_velocity = String::from("none");
    let mut prev_latent = finite_span(latent0);
    for (index, step) in steps.iter().enumerate() {
        forward.set_f32("temb_freq", &step.temb)?;
        if !text_is_cond {
            forward.set_f32("txt", txt)?;
            text_is_cond = true;
        }
        forward.compute_device()?;
        let velocity = forward.peek_output()?;
        if let Some(detail) = non_finite(&format!("dit velocity at step {}", index + 1), &velocity)
        {
            return Err(RuntimeError::Message(format!(
                "{detail}; previous velocity {prev_velocity}; previous latent {prev_latent}"
            )));
        }
        prev_velocity = finite_span(&velocity);
        ReusedGraph::copy(forward.output(), euler.tensor("vel_cond")?)?;
        if (guidance - 1.0).abs() < f32::EPSILON {
            ReusedGraph::copy(forward.output(), euler.tensor("vel_uncond")?)?;
        } else {
            forward.set_f32("txt", &zeros)?;
            text_is_cond = false;
            forward.compute_device()?;
            let uncond = forward.peek_output()?;
            if let Some(detail) = non_finite(
                &format!("dit uncond velocity at step {}", index + 1),
                &uncond,
            ) {
                return Err(RuntimeError::Message(detail));
            }
            ReusedGraph::copy(forward.output(), euler.tensor("vel_uncond")?)?;
        }
        euler.set_f32("guidance", &[guidance])?;
        euler.set_f32("dt", &[step.sigma_next - step.sigma])?;
        euler.compute_device()?;
        let latent = euler.peek_output()?;
        if let Some(detail) = non_finite(&format!("latent after step {}", index + 1), &latent) {
            return Err(RuntimeError::Message(format!(
                "{detail}; velocity {prev_velocity}"
            )));
        }
        prev_latent = finite_span(&latent);
        ReusedGraph::copy(euler.output(), forward.tensor("latent")?)?;
        ReusedGraph::copy(euler.output(), euler.tensor("latent")?)?;
        on_step(index as u32 + 1, total);
    }
    euler.read_output()
}

fn finite_span(data: &[f32]) -> String {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut absmax = 0f32;
    let mut finite = 0usize;
    for value in data {
        if value.is_finite() {
            finite += 1;
            min = min.min(*value);
            max = max.max(*value);
            absmax = absmax.max(value.abs());
        }
    }
    if finite == 0 {
        return format!("no finite values out of {}", data.len());
    }
    format!("min {min} max {max} abs {absmax} over {finite}")
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
