//! Diffusion fragments shared by every family: flash attention, RoPE, in-graph
//! CFG, in-graph Euler, and direct conv. Families call these names.

use super::{shape, BackendKind, GgmlContext, GraphBuilder, ReusedGraph, RuntimeError, Tensor};
use std::time::Instant;

const PREVIEW_MEM: usize = 16 * 1024 * 1024;
const STEP_MEM: usize = 8 * 1024 * 1024;

/// One denoising step. Sigma values stay on the host; the activations do not.
pub struct DeviceStep {
    pub sigma: f32,
    pub sigma_next: f32,
    pub temb: Vec<f32>,
}

/// RoPE inputs already declared on the family graph.
#[derive(Clone, Copy)]
pub struct RopeIn {
    pub axis: bool,
    pub pos: Tensor,
    pub axes: [Tensor; 4],
    pub theta: f32,
}

pub struct DiffusionTiming {
    pub backend: String,
    pub device: String,
    pub steps: u32,
    pub elapsed_ms: u128,
    pub forward_readbacks: u32,
}

/// Fused attention. Query, key, and value arrive as `[head, heads, seq]`.
/// The mask, when present, is an F32 `[seq, seq]` tensor and is cast to F16.
/// Shapes that `ggml_flash_attn_ext` cannot take fail here.
pub fn flash_attend(
    b: &GraphBuilder,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_f32: Tensor,
) -> Result<Tensor, RuntimeError> {
    let q = nn(b.cont(b.permute(q, 0, 2, 1, 3)))?;
    let k = nn(b.cont(b.permute(k, 0, 2, 1, 3)))?;
    let v = nn(b.cont(b.permute(v, 0, 2, 1, 3)))?;
    let qs = shape(q);
    let ks = shape(k);
    let vs = shape(v);
    if qs[0] < 1 || qs[0] != ks[0] || qs[0] != vs[0] {
        return Err(RuntimeError::Message(format!(
            "flash attention head dims are q {} k {} v {}",
            qs[0], ks[0], vs[0]
        )));
    }
    if ks[2] < 1 || qs[2] % ks[2] != 0 || vs[2] < 1 || qs[2] % vs[2] != 0 {
        return Err(RuntimeError::Message(format!(
            "flash attention kv heads k {} v {} do not divide query heads {}",
            ks[2], vs[2], qs[2]
        )));
    }
    let mask = if mask_f32.is_null() {
        std::ptr::null_mut()
    } else {
        nn(b.cont(b.cast_f16(mask_f32)))?
    };
    let scale = 1.0 / (qs[0] as f32).sqrt();
    nn(b.flash_attn(q, k, v, mask, scale))
}

/// `ggml_rope_ext` on one head vector. Four-axis Flux2 positions are four
/// `rope_ext` calls, one per even-sized axis. A head that is not even fails.
pub fn rope_pair(
    b: &GraphBuilder,
    q: Tensor,
    k: Tensor,
    head: i64,
    rope: &RopeIn,
) -> Result<(Tensor, Tensor), RuntimeError> {
    if head < 2 || head % 2 != 0 {
        return Err(RuntimeError::Message(format!(
            "ggml rope_ext needs an even head dim, got {head}"
        )));
    }
    if rope.axis {
        if head % 8 != 0 {
            return Err(RuntimeError::Message(format!(
                "four-axis rope_ext needs a head dim divisible by 8, got {head}"
            )));
        }
        Ok((
            rope_axes(b, q, rope.axes, head, rope.theta)?,
            rope_axes(b, k, rope.axes, head, rope.theta)?,
        ))
    } else {
        Ok((
            nn(b.cont(b.rope(q, rope.pos, head as i32, rope.theta)))?,
            nn(b.cont(b.rope(k, rope.pos, head as i32, rope.theta)))?,
        ))
    }
}

fn rope_axes(
    b: &GraphBuilder,
    x: Tensor,
    axes: [Tensor; 4],
    head: i64,
    theta: f32,
) -> Result<Tensor, RuntimeError> {
    let dims = shape(x);
    let axis = head / 4;
    let heads = dims[1];
    let seq = dims[2];
    let bytes = std::mem::size_of::<f32>();
    let nb1 = (head as usize) * bytes;
    let nb2 = nb1 * (heads as usize);
    let mut acc: Tensor = std::ptr::null_mut();
    for i in 0..4 {
        let view = b.view_3d(
            x,
            axis,
            heads,
            seq,
            nb1,
            nb2,
            (i as usize) * (axis as usize) * bytes,
        );
        let view = nn(b.cont(view))?;
        let rotated = nn(b.cont(b.rope(view, axes[i], axis as i32, theta)))?;
        acc = if i == 0 {
            rotated
        } else {
            nn(b.cont(b.concat(acc, rotated, 0)))?
        };
    }
    Ok(acc)
}

/// Interleaved RoPE on unequal axes. `x` is `[head, heads, seq]`. Each axis is a
/// contiguous even slice and its own position vector. `ggml_rope_ext` uses
/// `theta^(-2i/n)` on that slice, which is the complex RoPE Z-Image and SD.cpp share.
pub fn rope_nd(
    b: &GraphBuilder,
    x: Tensor,
    pos: &[Tensor],
    dims: &[i64],
    theta: f32,
) -> Result<Tensor, RuntimeError> {
    if pos.len() != dims.len() || dims.is_empty() {
        return Err(RuntimeError::Message(
            "rope axes and position vectors have different lengths".into(),
        ));
    }
    let dims_x = shape(x);
    let head: i64 = dims.iter().sum();
    if dims_x[0] != head {
        return Err(RuntimeError::Message(format!(
            "rope head {} does not match axis sum {head}",
            dims_x[0]
        )));
    }
    let heads = dims_x[1];
    let seq = dims_x[2];
    let bytes = std::mem::size_of::<f32>();
    let nb1 = (head as usize) * bytes;
    let nb2 = nb1 * (heads as usize);
    let mut acc: Tensor = std::ptr::null_mut();
    let mut offset = 0usize;
    for (dim, p) in dims.iter().copied().zip(pos.iter().copied()) {
        if dim < 2 || dim % 2 != 0 {
            return Err(RuntimeError::Message(format!(
                "rope axis {dim} is not a positive even width"
            )));
        }
        let view = nn(b.cont(b.view_3d(x, dim, heads, seq, nb1, nb2, offset)))?;
        let rotated = nn(b.cont(b.rope(view, p, dim as i32, theta)))?;
        acc = if acc.is_null() {
            rotated
        } else {
            nn(b.cont(b.concat(acc, rotated, 0)))?
        };
        offset += (dim as usize) * bytes;
    }
    Ok(acc)
}

/// CFG and Euler on one reused graph: `x + dt * (u + g * (c - u))`.
pub(crate) fn euler_graph(
    parent: &ReusedGraph,
    channels: i64,
    seq: i64,
) -> Result<ReusedGraph, RuntimeError> {
    let mut b = parent.child(STEP_MEM)?;
    let latent = b.f32_2d("latent", channels, seq, true);
    let cond = b.f32_2d("vel_cond", channels, seq, true);
    let uncond = b.f32_2d("vel_uncond", channels, seq, true);
    let guidance = b.f32_1d("guidance", 1, true);
    let dt = b.f32_1d("dt", 1, true);
    let diff = b.sub(cond, uncond);
    let guided = b.add(uncond, b.mul(diff, b.repeat(guidance, diff)));
    let next = b.add(latent, b.mul(guided, b.repeat(dt, latent)));
    b.finish(nn(next)?)
}

/// 1x1 direct conv used by the preview decoder. Output is planar `[W, H, 3]`.
pub fn preview_conv(
    ctx: &GgmlContext,
    latent: &[f32],
    weight: &[f32],
    in_ch: i64,
    grid_w: i64,
    grid_h: i64,
) -> Result<Vec<f32>, RuntimeError> {
    let seq = grid_w * grid_h;
    if in_ch < 1 || grid_w < 1 || grid_h < 1 {
        return Err(RuntimeError::Message("conv preview grid is empty".into()));
    }
    if latent.len() as i64 != in_ch * seq {
        return Err(RuntimeError::Message(format!(
            "conv latent has {} values, graph expects {}",
            latent.len(),
            in_ch * seq
        )));
    }
    if weight.len() as i64 != in_ch * 3 {
        return Err(RuntimeError::Message(format!(
            "conv kernel has {} values, graph expects {}",
            weight.len(),
            in_ch * 3
        )));
    }
    let mut b = ctx.builder(PREVIEW_MEM)?;
    let x = b.f32_2d("latent", in_ch, seq, true);
    let w = b.f32_2d("kernel", in_ch, 3, true);
    let image = nn(b.reshape_3d(x, in_ch, grid_w, grid_h))?;
    let image = nn(b.cont(b.permute(image, 2, 0, 1, 3)))?;
    let kernel = nn(b.reshape_4d(w, 1, 1, in_ch, 3))?;
    let y = nn(b.conv2d_direct(kernel, image, 1, 1, 0, 0, 1, 1))?;
    let mut graph = b.finish(y)?;
    graph.set_f32("latent", latent)?;
    graph.set_f32("kernel", weight)?;
    graph.compute()
}

/// Time resident diffusion steps on one ggml device. Metal goes through
/// `ggml_backend_metal_init`. A missing device fails; nothing else runs the step.
pub fn measure_diffusion_steps(
    kind: BackendKind,
    steps: u32,
) -> Result<DiffusionTiming, RuntimeError> {
    let ctx = GgmlContext::new(kind)?;
    let device = ctx.device_name();
    let mut b = ctx.builder(PREVIEW_MEM)?;
    let w = b.f32_2d("w", 4, 4, true);
    let latent = b.f32_2d("latent", 4, 4, true);
    let txt = b.f32_2d("txt", 4, 1, true);
    let temb = b.f32_2d("temb_freq", 4, 1, true);
    let hidden = b.add(b.mul_mat(w, latent), b.repeat(temb, latent));
    let y = b.add(hidden, b.repeat(txt, latent));
    let mut forward = b.finish(nn(y)?)?;
    forward.set_f32(
        "w",
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )?;
    let steps_n = steps.max(1);
    let schedule: Vec<DeviceStep> = (0..steps_n)
        .map(|i| {
            let sigma = 1.0 - i as f32 / steps_n as f32;
            let sigma_next = 1.0 - (i + 1) as f32 / steps_n as f32;
            DeviceStep {
                sigma,
                sigma_next,
                temb: vec![0.01; 4],
            }
        })
        .collect();
    let started = Instant::now();
    let latents = crate::pipeline::denoise::denoise_resident(
        &mut forward,
        &schedule,
        &[0.1; 16],
        &[0.0; 4],
        1.0,
        4,
        4,
        |_, _| (),
    )?;
    if latents.len() != 16 || latents.iter().any(|v| !v.is_finite()) {
        return Err(RuntimeError::Message(
            "diffusion measure returned a non-finite latent".into(),
        ));
    }
    Ok(DiffusionTiming {
        backend: kind.name().into(),
        device,
        steps: steps_n,
        elapsed_ms: started.elapsed().as_millis(),
        forward_readbacks: forward.host_readbacks(),
    })
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_and_euler_stay_on_the_graph() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(STEP_MEM).unwrap();
        let latent = b.f32_2d("latent", 1, 1, true);
        let w = b.f32_2d("w", 1, 1, true);
        let txt = b.f32_2d("txt", 1, 1, true);
        let temb = b.f32_2d("temb_freq", 1, 1, true);
        let y = b.add(b.mul_mat(w, latent), b.add(txt, temb));
        let mut forward = b.finish(y).unwrap();
        forward.set_f32("w", &[1.0]).unwrap();
        let steps = [DeviceStep {
            sigma: 1.0,
            sigma_next: 0.5,
            temb: vec![0.0],
        }];
        let out = crate::pipeline::denoise::denoise_resident(
            &mut forward,
            &steps,
            &[1.0],
            &[3.0],
            2.0,
            1,
            1,
            |_, _| (),
        )
        .unwrap();
        assert_eq!(forward.host_readbacks(), 0);
        assert_eq!(forward.compute_count(), 2);
        let cond = 1.0 + 3.0;
        let uncond = 1.0 + 0.0;
        let vel = uncond + 2.0 * (cond - uncond);
        let expect = 1.0 + (0.5 - 1.0) * vel;
        assert!(
            (out[0] - expect).abs() < 1e-4,
            "got {out:?} expect {expect}"
        );
    }

    #[test]
    fn weight_values_survive_a_second_compute() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut b = ctx.builder(4 * 1024 * 1024).unwrap();
        let w = b.f32_2d("w", 32, 32, true);
        let x = b.f32_2d("x", 32, 1, true);
        let a = b.f32_2d("a", 32, 32, true);
        let c = b.f32_2d("c", 32, 32, true);
        let h = b.mul_mat(w, x);
        let sum = b.add(a, c);
        let extra = b.mul_mat(sum, x);
        let out = b.add(h, extra);
        let mut graph = b.finish(out).unwrap();
        let mut weight = vec![0f32; 32 * 32];
        for i in 0..32 {
            weight[i + i * 32] = 3.0;
        }
        graph.set_f32("w", &weight).unwrap();
        graph.set_f32("x", &vec![1.0; 32]).unwrap();
        graph.set_f32("a", &vec![0.0; 32 * 32]).unwrap();
        graph.set_f32("c", &vec![0.0; 32 * 32]).unwrap();
        let first = graph.compute().unwrap();
        let second = graph.compute().unwrap();
        assert_eq!(first.len(), 32);
        for (got, again) in first.iter().zip(second.iter()) {
            assert!((got - 3.0).abs() < 1e-4, "first {first:?}");
            assert!((again - 3.0).abs() < 1e-4, "second {second:?}");
        }
    }

    #[test]
    fn direct_conv_matches_the_channel_dot() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let latent = [1.0f32, 2.0, 3.0, 4.0];
        let weight = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let rgb = preview_conv(&ctx, &latent, &weight, 2, 2, 1).unwrap();
        assert_eq!(rgb.len(), 6);
        assert!((rgb[0] - 1.0).abs() < 1e-4, "{rgb:?}");
        assert!((rgb[1] - 3.0).abs() < 1e-4, "{rgb:?}");
        assert!((rgb[2] - 2.0).abs() < 1e-4, "{rgb:?}");
        assert!((rgb[3] - 4.0).abs() < 1e-4, "{rgb:?}");
    }
}
