//! Host sampler for the official ACE-Step 1.5 flow loop.
//!
//! Latents are `[batch, frames, channels]`, time-major inside each item.
//! The velocity callback sees one batch, never the CFG-doubled tensor.

use super::super::dcw::DcwSettings;
use crate::weights::WeightError;

const APG_MOMENTUM: f64 = -0.75;
const APG_NORM_THRESHOLD: f32 = 2.5;
const TURBO_SHIFTS: [f32; 3] = [1.0, 2.0, 3.0];
const TURBO_VALID_TIMESTEPS: [f64; 20] = [
    1.0,
    0.954_545_454_545_454_6,
    0.933_333_333_333_333_3,
    0.9,
    0.875,
    0.857_142_857_142_857_1,
    0.833_333_333_333_333_4,
    0.769_230_769_230_769_3,
    0.75,
    0.666_666_666_666_666_6,
    0.642_857_142_857_142_9,
    0.625,
    0.545_454_545_454_545_4,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.222_222_222_222_222_2,
    0.125,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// CFG-distilled 8-step (or 1–20 step) turbo DiT.
    Turbo,
    /// Base and SFT. CFG uses APG on the predictor and standard CFG on a Heun corrector.
    Base,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMethod {
    Ode,
    Sde,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerMode {
    Euler,
    Heun,
}

#[derive(Debug, Clone, Copy)]
pub struct SampleStep<'a> {
    pub latents: &'a [f32],
    pub timestep: f32,
    /// `false` is the null-condition forward used by CFG.
    pub conditional: bool,
    /// `true` after the cover blend has handed the step to the text2music context.
    pub text2music: bool,
}

pub struct SampleRequest<'a> {
    pub variant: ModelVariant,
    pub batch: usize,
    pub frames: usize,
    pub channels: usize,
    pub noise: &'a [f32],
    pub shift: f32,
    pub infer_steps: Option<u32>,
    pub timesteps: Option<&'a [f32]>,
    /// Already-built eval schedule. Skips shift tables and turbo timestep snapping.
    pub prepared: Option<&'a [f32]>,
    /// Exact cover-switch step. Overrides `cover_strength` when set.
    pub cover_steps_override: Option<usize>,
    pub method: StepMethod,
    pub sampler: SamplerMode,
    pub guidance: f32,
    /// Turbo checkpoints are CFG-distilled. Guidance above 1 fails instead of being ignored.
    pub cfg_distilled: bool,
    pub cfg_interval: (f32, f32),
    /// Fraction of steps that keep the cover context. `1` never switches.
    pub cover_strength: f32,
    /// `0` starts from pure noise. Positive values renoise `source` at the nearest timestep.
    pub cover_noise_strength: f32,
    pub source: Option<&'a [f32]>,
    pub velocity_norm_threshold: f32,
    pub velocity_ema_factor: f32,
    pub retake_variance: f32,
    pub retake_noise: Option<&'a [f32]>,
    pub dcw: Option<DcwSettings>,
    /// `[batch, frames]`. `true` keeps the generated latent; `false` is the preserved source region.
    pub repaint_mask: Option<&'a [bool]>,
    pub clean_source: Option<&'a [f32]>,
    /// Noise draw used to renoise preserved repaint frames. Defaults to the initial noise.
    pub repaint_noise: Option<&'a [f32]>,
    pub repaint_crossfade_frames: i32,
    pub repaint_injection_ratio: f32,
}

impl<'a> SampleRequest<'a> {
    pub fn new(
        variant: ModelVariant,
        batch: usize,
        frames: usize,
        channels: usize,
        noise: &'a [f32],
    ) -> Self {
        Self {
            variant,
            batch,
            frames,
            channels,
            noise,
            shift: 3.0,
            infer_steps: None,
            timesteps: None,
            prepared: None,
            cover_steps_override: None,
            method: StepMethod::Ode,
            sampler: SamplerMode::Euler,
            guidance: 1.0,
            cfg_distilled: variant == ModelVariant::Turbo,
            cfg_interval: (0.0, 1.0),
            cover_strength: 1.0,
            cover_noise_strength: 0.0,
            source: None,
            velocity_norm_threshold: 0.0,
            velocity_ema_factor: 0.0,
            retake_variance: 0.0,
            retake_noise: None,
            dcw: None,
            repaint_mask: None,
            clean_source: None,
            repaint_noise: None,
            repaint_crossfade_frames: 0,
            repaint_injection_ratio: 0.5,
        }
    }
}

pub fn sample<F, N>(
    request: &SampleRequest<'_>,
    sde_noise: N,
    velocity: F,
) -> Result<Vec<f32>, WeightError>
where
    F: FnMut(SampleStep<'_>) -> Result<Vec<f32>, WeightError>,
    N: FnMut() -> Result<Vec<f32>, WeightError>,
{
    sample_with(request, sde_noise, velocity, |_, _| {})
}

pub fn sample_with<F, N, S>(
    request: &SampleRequest<'_>,
    mut sde_noise: N,
    mut velocity: F,
    mut on_step: S,
) -> Result<Vec<f32>, WeightError>
where
    F: FnMut(SampleStep<'_>) -> Result<Vec<f32>, WeightError>,
    N: FnMut() -> Result<Vec<f32>, WeightError>,
    S: FnMut(u32, u32),
{
    validate(request)?;
    if request.sampler == SamplerMode::Heun && request.method == StepMethod::Sde {
        return Err(WeightError::NotBuilt(
            "ACE-Step Heun does not run with SDE".into(),
        ));
    }
    if request.cfg_distilled && request.guidance > 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step turbo is CFG-distilled and rejects guidance {}",
            request.guidance
        )));
    }

    let mut schedule = if let Some(prepared) = request.prepared {
        if prepared.is_empty() || prepared.iter().any(|t| !t.is_finite()) {
            return Err(WeightError::NotBuilt(
                "ACE-Step prepared schedule must be a non-empty list of finite timesteps".into(),
            ));
        }
        prepared.to_vec()
    } else {
        match request.variant {
            ModelVariant::Turbo => {
                turbo_schedule(request.shift, request.infer_steps, request.timesteps)?
            }
            ModelVariant::Base => {
                base_schedule(request.shift, request.infer_steps, request.timesteps)?
            }
        }
    };
    let count = request.batch * request.frames * request.channels;
    let drawn = mix_retake(request.noise, request)?;
    let mut xt = drawn.clone();
    if request.cover_noise_strength > 0.0 {
        let source = request.source.ok_or_else(|| {
            WeightError::NotBuilt("ACE-Step cover_noise_strength needs the source latent".into())
        })?;
        let evals = eval_timesteps(&schedule, request.variant);
        let nearest = nearest_timestep(&evals, 1.0 - request.cover_noise_strength);
        let start = evals
            .iter()
            .position(|t| *t == nearest)
            .expect("nearest timestep is taken from the schedule");
        xt = renoise(source, nearest, &drawn);
        schedule = match request.variant {
            ModelVariant::Turbo => schedule[start..].to_vec(),
            ModelVariant::Base => schedule[start..].to_vec(),
        };
    }

    let mut momentum = vec![0.0f64; count];
    let mut momentum_ready = false;
    let mut prev_vt: Option<Vec<f32>> = None;

    match request.variant {
        ModelVariant::Turbo => {
            let num_steps = schedule.len();
            let cover_steps = request
                .cover_steps_override
                .unwrap_or_else(|| (num_steps as f32 * request.cover_strength).floor() as usize);
            for step_idx in 0..num_steps {
                let t_curr = schedule[step_idx];
                let text2music = step_idx >= cover_steps;
                let mut vt = forward(
                    &xt,
                    t_curr,
                    true,
                    text2music,
                    request,
                    &mut velocity,
                    &mut momentum,
                    &mut momentum_ready,
                    prev_vt.as_deref(),
                )?;
                let xt_before = xt.clone();
                let vt_for_denoise = vt.clone();
                if step_idx + 1 == num_steps {
                    xt = axpy(&xt, &vt, -t_curr);
                    apply_dcw(request, &mut xt, &xt_before, &vt_for_denoise, t_curr)?;
                    on_step(num_steps as u32, num_steps as u32);
                    break;
                }
                let t_next = schedule[step_idx + 1];
                let t_after = t_next;
                match request.method {
                    StepMethod::Sde => {
                        let clean = axpy(&xt, &vt, -t_curr);
                        let fresh = sde_noise()?;
                        if fresh.len() != count {
                            return Err(WeightError::NotBuilt(format!(
                                "ACE-Step SDE noise length {} != {count}",
                                fresh.len()
                            )));
                        }
                        xt = renoise(&clean, t_next, &fresh);
                    }
                    StepMethod::Ode => match request.sampler {
                        SamplerMode::Heun => {
                            let dt = t_curr - t_next;
                            let predicted = axpy(&xt, &vt, -dt);
                            let vt2 = forward(
                                &predicted,
                                t_next,
                                true,
                                text2music,
                                request,
                                &mut velocity,
                                &mut momentum,
                                &mut momentum_ready,
                                Some(&vt),
                            )?;
                            vt = lerp(&vt, &vt2, 0.5);
                            xt = axpy(&xt, &vt, -dt);
                        }
                        SamplerMode::Euler => {
                            xt = axpy(&xt, &vt, -(t_curr - t_next));
                        }
                    },
                }
                apply_dcw(request, &mut xt, &xt_before, &vt_for_denoise, t_curr)?;
                prev_vt = Some(vt);
                inject_repaint(request, &mut xt, &drawn, t_after, step_idx, num_steps)?;
                on_step((step_idx as u32) + 1, num_steps as u32);
            }
        }
        ModelVariant::Base => {
            if schedule.len() < 2 {
                return Err(WeightError::NotBuilt(
                    "ACE-Step base schedule needs a start and an end".into(),
                ));
            }
            let intervals = schedule.len() - 1;
            let cover_steps = request
                .cover_steps_override
                .unwrap_or_else(|| (intervals as f32 * request.cover_strength).floor() as usize);
            for step_idx in 0..intervals {
                let t_curr = schedule[step_idx];
                let t_next = schedule[step_idx + 1];
                let text2music = step_idx >= cover_steps;
                let mut vt = forward(
                    &xt,
                    t_curr,
                    true,
                    text2music,
                    request,
                    &mut velocity,
                    &mut momentum,
                    &mut momentum_ready,
                    prev_vt.as_deref(),
                )?;
                let xt_before = xt.clone();
                let vt_for_denoise = vt.clone();
                let mut t_after = t_next;
                match request.method {
                    StepMethod::Sde => {
                        let clean = axpy(&xt, &vt, -t_curr);
                        let fresh = sde_noise()?;
                        if fresh.len() != count {
                            return Err(WeightError::NotBuilt(format!(
                                "ACE-Step SDE noise length {} != {count}",
                                fresh.len()
                            )));
                        }
                        t_after = 1.0 - ((step_idx + 1) as f32) / (intervals as f32);
                        xt = renoise(&clean, t_after, &fresh);
                    }
                    StepMethod::Ode => match request.sampler {
                        SamplerMode::Heun => {
                            let dt = t_curr - t_next;
                            let predicted = axpy(&xt, &vt, -dt);
                            let vt2 = heun_corrector(
                                &predicted,
                                t_next,
                                text2music,
                                request,
                                &mut velocity,
                                &vt,
                            )?;
                            vt = lerp(&vt, &vt2, 0.5);
                            xt = axpy(&xt, &vt, -dt);
                        }
                        SamplerMode::Euler => {
                            xt = axpy(&xt, &vt, -(t_curr - t_next));
                        }
                    },
                }
                apply_dcw(request, &mut xt, &xt_before, &vt_for_denoise, t_curr)?;
                prev_vt = Some(vt);
                inject_repaint(request, &mut xt, &drawn, t_after, step_idx, intervals)?;
                on_step((step_idx as u32) + 1, intervals as u32);
            }
        }
    }

    if let (Some(mask), Some(clean)) = (request.repaint_mask, request.clean_source) {
        if request.repaint_crossfade_frames > 0 {
            boundary_blend(
                &mut xt,
                clean,
                mask,
                request.batch,
                request.frames,
                request.channels,
                request.repaint_crossfade_frames as usize,
            )?;
        }
    }
    Ok(xt)
}

fn forward<F>(
    xt: &[f32],
    timestep: f32,
    conditional: bool,
    text2music: bool,
    request: &SampleRequest<'_>,
    velocity: &mut F,
    momentum: &mut [f64],
    momentum_ready: &mut bool,
    ema_prev: Option<&[f32]>,
) -> Result<Vec<f32>, WeightError>
where
    F: FnMut(SampleStep<'_>) -> Result<Vec<f32>, WeightError>,
{
    let cond = velocity(SampleStep {
        latents: xt,
        timestep,
        conditional: true,
        text2music,
    })?;
    expect_len("velocity", &cond, xt.len())?;
    let mut vt = if request.guidance > 1.0 && conditional {
        let uncond = velocity(SampleStep {
            latents: xt,
            timestep,
            conditional: false,
            text2music,
        })?;
        expect_len("null velocity", &uncond, xt.len())?;
        if in_cfg_interval(timestep, request.cfg_interval) {
            apg(
                &cond,
                &uncond,
                request.guidance,
                request.batch,
                request.frames,
                request.channels,
                momentum,
                momentum_ready,
            )?
        } else {
            cond
        }
    } else {
        cond
    };
    vt = stabilise(request, &vt, xt, ema_prev);
    Ok(vt)
}

fn heun_corrector<F>(
    predicted: &[f32],
    timestep: f32,
    text2music: bool,
    request: &SampleRequest<'_>,
    velocity: &mut F,
    predictor: &[f32],
) -> Result<Vec<f32>, WeightError>
where
    F: FnMut(SampleStep<'_>) -> Result<Vec<f32>, WeightError>,
{
    let cond = velocity(SampleStep {
        latents: predicted,
        timestep,
        conditional: true,
        text2music,
    })?;
    expect_len("Heun velocity", &cond, predicted.len())?;
    let mut vt = if request.guidance > 1.0 && in_cfg_interval(timestep, request.cfg_interval) {
        let uncond = velocity(SampleStep {
            latents: predicted,
            timestep,
            conditional: false,
            text2music,
        })?;
        expect_len("Heun null velocity", &uncond, predicted.len())?;
        cfg_forward(&cond, &uncond, request.guidance)
    } else if request.guidance > 1.0 {
        cond
    } else {
        cond
    };
    vt = stabilise(request, &vt, predicted, Some(predictor));
    Ok(vt)
}

fn apg(
    pred_cond: &[f32],
    pred_uncond: &[f32],
    guidance: f32,
    batch: usize,
    frames: usize,
    channels: usize,
    momentum: &mut [f64],
    ready: &mut bool,
) -> Result<Vec<f32>, WeightError> {
    let n = pred_cond.len();
    let mut diff = vec![0.0f64; n];
    for index in 0..n {
        diff[index] = f64::from(pred_cond[index]) - f64::from(pred_uncond[index]);
    }
    if !*ready {
        momentum.copy_from_slice(&diff);
        *ready = true;
    } else {
        for index in 0..n {
            momentum[index] = diff[index] + APG_MOMENTUM * momentum[index];
        }
    }
    let diff = momentum;
    let plane = frames * channels;
    for item in 0..batch {
        for channel in 0..channels {
            let mut norm_sq = 0.0f64;
            for frame in 0..frames {
                let value = diff[item * plane + frame * channels + channel];
                norm_sq += value * value;
            }
            let norm = norm_sq.sqrt();
            let scale = if norm == 0.0 {
                1.0
            } else {
                (f64::from(APG_NORM_THRESHOLD) / norm).min(1.0)
            };
            if scale < 1.0 {
                for frame in 0..frames {
                    let index = item * plane + frame * channels + channel;
                    diff[index] *= scale;
                }
            }
        }
    }
    let mut guided = vec![0.0f32; n];
    for item in 0..batch {
        for channel in 0..channels {
            let mut cond_sq = 0.0f64;
            for frame in 0..frames {
                let value = f64::from(pred_cond[item * plane + frame * channels + channel]);
                cond_sq += value * value;
            }
            let cond_norm = cond_sq.sqrt().max(1e-12);
            let mut dot = 0.0f64;
            for frame in 0..frames {
                let index = item * plane + frame * channels + channel;
                dot += diff[index] * f64::from(pred_cond[index]) / cond_norm;
            }
            for frame in 0..frames {
                let index = item * plane + frame * channels + channel;
                let direction = f64::from(pred_cond[index]) / cond_norm;
                let parallel = dot * direction;
                let orthogonal = diff[index] - parallel;
                let value = f64::from(pred_cond[index]) + (f64::from(guidance) - 1.0) * orthogonal;
                guided[index] = value as f32;
            }
        }
    }
    Ok(guided)
}

fn cfg_forward(cond: &[f32], uncond: &[f32], scale: f32) -> Vec<f32> {
    cond.iter()
        .zip(uncond)
        .map(|(cond, uncond)| uncond + scale * (cond - uncond))
        .collect()
}

fn stabilise(
    request: &SampleRequest<'_>,
    vt: &[f32],
    xt: &[f32],
    prev: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vt.to_vec();
    if request.velocity_norm_threshold > 0.0 {
        let plane = request.frames * request.channels;
        for item in 0..request.batch {
            let start = item * plane;
            let end = start + plane;
            let vt_norm = l2(&out[start..end]);
            let xt_norm = l2(&xt[start..end]) + 1e-10;
            let scale = (request.velocity_norm_threshold * xt_norm / (vt_norm + 1e-10)).min(1.0);
            for value in &mut out[start..end] {
                *value *= scale;
            }
        }
    }
    if request.velocity_ema_factor > 0.0 {
        if let Some(prev) = prev {
            let factor = request.velocity_ema_factor;
            for (value, previous) in out.iter_mut().zip(prev) {
                *value = (1.0 - factor) * *value + factor * *previous;
            }
        }
    }
    out
}

fn in_cfg_interval(timestep: f32, interval: (f32, f32)) -> bool {
    timestep >= interval.0 && timestep <= interval.1
}

fn turbo_schedule(
    shift: f32,
    infer_steps: Option<u32>,
    timesteps: Option<&[f32]>,
) -> Result<Vec<f32>, WeightError> {
    if let Some(timesteps) = timesteps {
        let mut values: Vec<f64> = timesteps.iter().map(|t| f64::from(*t)).collect();
        while values.last().is_some_and(|t| *t == 0.0) {
            values.pop();
        }
        if values.is_empty() {
            return Err(WeightError::NotBuilt(
                "ACE-Step turbo timesteps are empty after removing trailing zeros".into(),
            ));
        }
        if values.len() > 20 {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step turbo timesteps length {} exceeds 20",
                values.len()
            )));
        }
        return Ok(values
            .into_iter()
            .map(|t| nearest_valid_turbo(t) as f32)
            .collect());
    }
    if let Some(steps) = infer_steps {
        if steps < 1 || steps > 20 {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step turbo infer_steps {steps} must be in 1..=20"
            )));
        }
        if !shift.is_finite() || shift <= 0.0 {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step turbo shift {shift} must be positive"
            )));
        }
        return Ok((0..steps)
            .map(|index| {
                let t = 1.0 - f64::from(index) / f64::from(steps);
                shift_timestep(t, f64::from(shift)) as f32
            })
            .collect());
    }
    if !TURBO_SHIFTS
        .iter()
        .any(|published| (*published - shift).abs() < 1e-6)
    {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step turbo lookup shift {shift} must be 1, 2, or 3"
        )));
    }
    let shift = shift.round() as i32;
    Ok(match shift {
        1 => vec![1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2 => vec![
            1.0,
            0.933_333_3,
            0.857_142_87,
            0.769_230_8,
            0.666_666_7,
            0.545_454_56,
            0.4,
            0.222_222_22,
        ],
        _ => vec![
            1.0,
            0.954_545_44,
            0.9,
            0.833_333_3,
            0.75,
            0.642_857_13,
            0.5,
            0.3,
        ],
    })
}

fn base_schedule(
    shift: f32,
    infer_steps: Option<u32>,
    timesteps: Option<&[f32]>,
) -> Result<Vec<f32>, WeightError> {
    if !shift.is_finite() || shift <= 0.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step base shift {shift} must be positive"
        )));
    }
    if let Some(timesteps) = timesteps {
        if timesteps.len() < 2 {
            return Err(WeightError::NotBuilt(
                "ACE-Step base timesteps need a start and an end".into(),
            ));
        }
        return Ok(timesteps.to_vec());
    }
    let steps = infer_steps
        .ok_or_else(|| WeightError::NotBuilt("ACE-Step base sampling needs infer_steps".into()))?;
    if steps < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step base infer_steps must be at least 1".into(),
        ));
    }
    Ok((0..=steps)
        .map(|index| {
            let t = 1.0 - f64::from(index) / f64::from(steps);
            shift_timestep(t, f64::from(shift)) as f32
        })
        .collect())
}

fn shift_timestep(t: f64, shift: f64) -> f64 {
    if (shift - 1.0).abs() < 1e-12 {
        t
    } else {
        shift * t / (1.0 + (shift - 1.0) * t)
    }
}

fn nearest_valid_turbo(t: f64) -> f64 {
    TURBO_VALID_TIMESTEPS
        .into_iter()
        .min_by(|left, right| {
            (left - t)
                .abs()
                .partial_cmp(&(right - t).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1.0)
}

fn nearest_timestep(values: &[f32], target: f32) -> f32 {
    values
        .iter()
        .copied()
        .min_by(|left, right| {
            (left - target)
                .abs()
                .partial_cmp(&(right - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(target)
}

fn eval_timesteps(schedule: &[f32], variant: ModelVariant) -> Vec<f32> {
    match variant {
        ModelVariant::Turbo => schedule.to_vec(),
        ModelVariant::Base => schedule[..schedule.len() - 1].to_vec(),
    }
}

fn mix_retake(noise: &[f32], request: &SampleRequest<'_>) -> Result<Vec<f32>, WeightError> {
    if request.retake_variance == 0.0 {
        return Ok(noise.to_vec());
    }
    let retake = request.retake_noise.ok_or_else(|| {
        WeightError::NotBuilt("ACE-Step retake_variance needs a second noise draw".into())
    })?;
    expect_len("retake noise", retake, noise.len())?;
    let angle = f64::from(request.retake_variance) * std::f64::consts::FRAC_PI_2;
    let (sin, cos) = angle.sin_cos();
    Ok(noise
        .iter()
        .zip(retake)
        .map(|(noise, retake)| (cos * f64::from(*noise) + sin * f64::from(*retake)) as f32)
        .collect())
}

fn renoise(clean: &[f32], timestep: f32, noise: &[f32]) -> Vec<f32> {
    clean
        .iter()
        .zip(noise)
        .map(|(clean, noise)| timestep * *noise + (1.0 - timestep) * *clean)
        .collect()
}

fn axpy(x: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    x.iter().zip(v).map(|(x, v)| x + scale * v).collect()
}

fn lerp(left: &[f32], right: &[f32], right_weight: f32) -> Vec<f32> {
    left.iter()
        .zip(right)
        .map(|(left, right)| (1.0 - right_weight) * left + right_weight * right)
        .collect()
}

fn l2(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn apply_dcw(
    request: &SampleRequest<'_>,
    xt: &mut [f32],
    previous: &[f32],
    velocity: &[f32],
    timestep: f32,
) -> Result<(), WeightError> {
    let Some(settings) = request.dcw else {
        return Ok(());
    };
    let is_turbo = request.variant == ModelVariant::Turbo;
    let plane = request.frames * request.channels;
    for item in 0..request.batch {
        let start = item * plane;
        let end = start + plane;
        settings.apply(
            &mut xt[start..end],
            &previous[start..end],
            &velocity[start..end],
            timestep,
            request.frames as i64,
            request.channels as i64,
            is_turbo,
        )?;
    }
    Ok(())
}

fn inject_repaint(
    request: &SampleRequest<'_>,
    xt: &mut [f32],
    noise: &[f32],
    timestep: f32,
    step_idx: usize,
    steps: usize,
) -> Result<(), WeightError> {
    let (Some(mask), Some(clean)) = (request.repaint_mask, request.clean_source) else {
        return Ok(());
    };
    let noise = request.repaint_noise.unwrap_or(noise);
    let cutoff = (request.repaint_injection_ratio * steps as f32).round() as usize;
    if step_idx >= cutoff {
        return Ok(());
    }
    let plane = request.frames * request.channels;
    for item in 0..request.batch {
        for frame in 0..request.frames {
            if mask[item * request.frames + frame] {
                continue;
            }
            for channel in 0..request.channels {
                let index = item * plane + frame * request.channels + channel;
                let noised = timestep * noise[index] + (1.0 - timestep) * clean[index];
                xt[index] = noised;
            }
        }
    }
    Ok(())
}

fn boundary_blend(
    xt: &mut [f32],
    clean: &[f32],
    mask: &[bool],
    batch: usize,
    frames: usize,
    channels: usize,
    crossfade: usize,
) -> Result<(), WeightError> {
    let mut soft = vec![0.0f32; batch * frames];
    for (value, kept) in soft.iter_mut().zip(mask) {
        *value = if *kept { 1.0 } else { 0.0 };
    }
    for item in 0..batch {
        let row = &mask[item * frames..(item + 1) * frames];
        if row.iter().all(|kept| *kept) || row.iter().all(|kept| !*kept) {
            continue;
        }
        let Some(left) = row.iter().position(|kept| *kept) else {
            continue;
        };
        let right = row.iter().rposition(|kept| *kept).unwrap_or(left) + 1;
        let fade_start = left.saturating_sub(crossfade);
        let rise = left - fade_start;
        if rise > 0 {
            for offset in 0..rise {
                let span = rise + 1;
                soft[item * frames + fade_start + offset] = (offset as f32 + 1.0) / span as f32;
            }
        }
        let fade_end = (right + crossfade).min(frames);
        let fall = fade_end - right;
        if fall > 0 {
            for offset in 0..fall {
                let span = fall + 1;
                soft[item * frames + right + offset] = 1.0 - (offset as f32 + 1.0) / span as f32;
            }
        }
    }
    let plane = frames * channels;
    for item in 0..batch {
        for frame in 0..frames {
            let weight = soft[item * frames + frame];
            for channel in 0..channels {
                let index = item * plane + frame * channels + channel;
                xt[index] = weight * xt[index] + (1.0 - weight) * clean[index];
            }
        }
    }
    Ok(())
}

fn validate(request: &SampleRequest<'_>) -> Result<(), WeightError> {
    if request.batch == 0 || request.frames == 0 || request.channels == 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step latent shape must be non-zero".into(),
        ));
    }
    let count = request.batch * request.frames * request.channels;
    expect_len("noise", request.noise, count)?;
    if !(0.0..=1.0).contains(&request.cover_strength) || !request.cover_strength.is_finite() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cover strength {} must be in [0, 1]",
            request.cover_strength
        )));
    }
    if !(0.0..=1.0).contains(&request.cover_noise_strength)
        || !request.cover_noise_strength.is_finite()
    {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step cover_noise_strength {} must be in [0, 1]",
            request.cover_noise_strength
        )));
    }
    if !(0.0..=1.0).contains(&request.retake_variance) || !request.retake_variance.is_finite() {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step retake_variance {} must be in [0, 1]",
            request.retake_variance
        )));
    }
    if !request.guidance.is_finite() || request.guidance < 1.0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step guidance {} must be >= 1",
            request.guidance
        )));
    }
    if request.repaint_mask.is_some() != request.clean_source.is_some() {
        return Err(WeightError::NotBuilt(
            "ACE-Step repaint needs both the mask and the clean source latent".into(),
        ));
    }
    if let Some(mask) = request.repaint_mask {
        if mask.len() != request.batch * request.frames {
            return Err(WeightError::NotBuilt(format!(
                "ACE-Step repaint mask length {} != {}",
                mask.len(),
                request.batch * request.frames
            )));
        }
    }
    if let Some(clean) = request.clean_source {
        expect_len("repaint source", clean, count)?;
    }
    if request.repaint_crossfade_frames < 0 {
        return Err(WeightError::NotBuilt(
            "ACE-Step repaint crossfade must be >= 0".into(),
        ));
    }
    Ok(())
}

fn expect_len(name: &str, values: &[f32], expected: usize) -> Result<(), WeightError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(WeightError::NotBuilt(format!(
            "ACE-Step {name} length {} != {expected}",
            values.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refuse_noise() -> Result<Vec<f32>, WeightError> {
        Err(WeightError::NotBuilt("SDE noise was not expected".into()))
    }

    #[test]
    fn turbo_shift3_table_matches_the_published_eight_steps() {
        let schedule = turbo_schedule(3.0, None, None).unwrap();
        let published = [
            1.0,
            0.9545454545454546,
            0.9,
            0.8333333333333334,
            0.75,
            0.6428571428571429,
            0.5,
            0.3,
        ];
        assert_eq!(schedule.len(), 8);
        for (got, expected) in schedule.iter().zip(published) {
            assert!(
                (f64::from(*got) - expected).abs() < 1e-6,
                "{got} vs {expected}"
            );
        }
    }

    #[test]
    fn turbo_infer_steps_use_the_continuous_shift() {
        let schedule = turbo_schedule(1.0, Some(4), None).unwrap();
        assert_eq!(schedule, vec![1.0, 0.75, 0.5, 0.25]);
    }

    #[test]
    fn base_linspace_includes_the_terminal_zero() {
        let schedule = base_schedule(1.0, Some(4), None).unwrap();
        assert_eq!(schedule, vec![1.0, 0.75, 0.5, 0.25, 0.0]);
    }

    #[test]
    fn turbo_last_step_unwraps_x0() {
        let noise = [2.0f32, 2.0];
        let mut request = SampleRequest::new(ModelVariant::Turbo, 1, 2, 1, &noise);
        request.timesteps = Some(&[0.5]);
        let got = sample(&request, refuse_noise, |step| {
            assert!((step.timestep - 0.5).abs() < 1e-6);
            Ok(vec![1.0, 1.0])
        })
        .unwrap();
        assert_eq!(got, vec![1.5, 1.5]);
    }

    #[test]
    fn heun_with_constant_velocity_matches_euler_on_turbo() {
        let noise = [1.0f32, -1.0];
        let mut euler = SampleRequest::new(ModelVariant::Turbo, 1, 2, 1, &noise);
        euler.infer_steps = Some(2);
        euler.shift = 1.0;
        let mut heun = SampleRequest::new(ModelVariant::Turbo, 1, 2, 1, &noise);
        heun.infer_steps = Some(2);
        heun.shift = 1.0;
        heun.sampler = SamplerMode::Heun;
        let velocity = |step: SampleStep<'_>| {
            let _ = step;
            Ok(vec![0.25, -0.5])
        };
        let euler = sample(&euler, refuse_noise, velocity).unwrap();
        let heun = sample(&heun, refuse_noise, velocity).unwrap();
        for (left, right) in euler.iter().zip(&heun) {
            assert!((left - right).abs() < 1e-5, "{left} vs {right}");
        }
    }

    #[test]
    fn turbo_sde_renoises_with_the_next_schedule_value() {
        let noise = [1.0f32];
        let fresh = [3.0f32];
        let mut request = SampleRequest::new(ModelVariant::Turbo, 1, 1, 1, &noise);
        request.infer_steps = Some(2);
        request.shift = 1.0;
        request.method = StepMethod::Sde;
        let got = sample(&request, || Ok(fresh.to_vec()), |_step| Ok(vec![0.0])).unwrap();
        // step 0: pred_clean = noise, next t = 0.5 → 0.5*3 + 0.5*1 = 2
        // last step velocity 0 leaves it unchanged
        assert!((got[0] - 2.0).abs() < 1e-5, "{}", got[0]);
    }

    #[test]
    fn base_predictor_uses_apg_not_plain_cfg() {
        let noise = [0.0f32, 0.0];
        let mut request = SampleRequest::new(ModelVariant::Base, 1, 2, 1, &noise);
        request.infer_steps = Some(1);
        request.shift = 1.0;
        request.guidance = 2.0;
        let got = sample(&request, refuse_noise, |step| {
            if step.conditional {
                Ok(vec![1.0, 0.0])
            } else {
                Ok(vec![0.0, 1.0])
            }
        })
        .unwrap();
        // guided = [1, -1]; one Euler step of dt 1 from 0.
        assert!((got[0] + 1.0).abs() < 1e-5, "{}", got[0]);
        assert!((got[1] - 1.0).abs() < 1e-5, "{}", got[1]);
    }

    #[test]
    fn cfg_interval_skips_guidance_outside_the_window() {
        let noise = [0.0f32];
        let mut request = SampleRequest::new(ModelVariant::Base, 1, 1, 1, &noise);
        request.infer_steps = Some(1);
        request.shift = 1.0;
        request.guidance = 3.0;
        request.cfg_interval = (0.0, 0.5);
        let got = sample(&request, refuse_noise, |step| {
            if step.conditional {
                Ok(vec![1.0])
            } else {
                Ok(vec![100.0])
            }
        })
        .unwrap();
        // t = 1 is outside [0, 0.5], so the null forward is discarded and x0 = 0 - 1.
        assert!((got[0] + 1.0).abs() < 1e-5, "{}", got[0]);
    }

    #[test]
    fn retake_full_variance_is_the_second_draw() {
        let noise = [1.0f32];
        let retake = [4.0f32];
        let mut request = SampleRequest::new(ModelVariant::Turbo, 1, 1, 1, &noise);
        request.timesteps = Some(&[0.125]);
        request.retake_variance = 1.0;
        request.retake_noise = Some(&retake);
        let got = sample(&request, refuse_noise, |_| Ok(vec![0.0])).unwrap();
        assert!((got[0] - 4.0).abs() < 1e-5, "{}", got[0]);
    }

    #[test]
    fn cover_switch_happens_at_the_strength_boundary() {
        let noise = [0.0f32];
        let mut request = SampleRequest::new(ModelVariant::Turbo, 1, 1, 1, &noise);
        request.infer_steps = Some(2);
        request.shift = 1.0;
        request.cover_strength = 0.5;
        let mut flags = Vec::new();
        sample(&request, refuse_noise, |step| {
            flags.push(step.text2music);
            Ok(vec![0.0])
        })
        .unwrap();
        assert_eq!(flags, vec![false, true]);
    }

    #[test]
    fn heun_sde_and_turbo_cfg_fail_loud() {
        let noise = [0.0f32];
        let mut request = SampleRequest::new(ModelVariant::Turbo, 1, 1, 1, &noise);
        request.sampler = SamplerMode::Heun;
        request.method = StepMethod::Sde;
        assert!(sample(&request, refuse_noise, |_| Ok(vec![0.0])).is_err());
        request.method = StepMethod::Ode;
        request.guidance = 2.0;
        assert!(sample(&request, refuse_noise, |_| Ok(vec![0.0])).is_err());
    }
}
