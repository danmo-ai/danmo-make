//! ACE-Step repaint helpers (acestep.cpp `ops_build_context` + `sampler-repaint.h`).
//!
//! Region `[t0, t1)` is generated fresh (silence context + mask 1). Outside the
//! zone the DiT sees source latents (mask 0); after denoise we splice source
//! latents back outside the zone and optionally crossfade the boundaries.

use crate::weights::WeightError;

/// Map wall-clock region to latent frames (48 kHz / 1920 = 25 Hz).
pub fn region_to_frames(
    start_sec: f32,
    end_sec: f32,
    total_frames: i64,
) -> Result<(i64, i64), WeightError> {
    if !start_sec.is_finite() || !end_sec.is_finite() {
        return Err(WeightError::NotBuilt(
            "ACE-Step repaint region times must be finite".into(),
        ));
    }
    if end_sec <= start_sec {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step repaint region end {end_sec} must be > start {start_sec}"
        )));
    }
    if total_frames < 1 {
        return Err(WeightError::NotBuilt(
            "ACE-Step repaint needs at least one latent frame".into(),
        ));
    }
    let mut t0 = (start_sec * 48000.0 / 1920.0).round() as i64;
    let mut t1 = (end_sec * 48000.0 / 1920.0).round() as i64;
    if t0 < 0 {
        t0 = 0;
    }
    if t1 > total_frames {
        t1 = total_frames;
    }
    if t0 > total_frames {
        t0 = total_frames;
    }
    if t1 <= t0 {
        return Err(WeightError::NotBuilt(format!(
            "ACE-Step repaint region [{start_sec}, {end_sec}) maps to empty frames [{t0}, {t1})"
        )));
    }
    Ok((t0, t1))
}

/// Per-step inject: outside `[t0, t1)` replace xt with noised source
/// `t_next * noise + (1 - t_next) * src` (acestep.cpp `sampler_repaint_inject`).
pub fn inject_preserved(
    xt: &mut [f32],
    noise: &[f32],
    src: &[f32],
    frames: i64,
    out_channels: i64,
    t0: i64,
    t1: i64,
    injection_ratio: f32,
    step: u32,
    num_steps: u32,
    t_next: f32,
) {
    if t1 <= t0 || src.is_empty() {
        return;
    }
    let cutoff = (injection_ratio * num_steps as f32 + 0.5) as u32;
    if step >= cutoff {
        return;
    }
    let oc = out_channels as usize;
    let t_cover = src.len() / oc;
    for t in 0..frames as usize {
        if t >= t0 as usize && t < t1 as usize {
            continue;
        }
        let t_src = t.min(t_cover.saturating_sub(1));
        for c in 0..oc {
            let idx = t * oc + c;
            xt[idx] = t_next * noise[idx] + (1.0 - t_next) * src[t_src * oc + c];
        }
    }
}

/// Post-loop boundary blend (acestep.cpp `sampler_repaint_blend`).
pub fn blend_boundaries(
    output: &mut [f32],
    src: &[f32],
    frames: i64,
    out_channels: i64,
    t0: i64,
    t1: i64,
    crossfade_frames: i64,
) {
    if t1 <= t0 || crossfade_frames <= 0 || src.is_empty() {
        return;
    }
    let oc = out_channels as usize;
    let t_total = frames as usize;
    let t_cover = src.len() / oc;
    let cf = crossfade_frames as usize;
    let fade_start = t0.max(0) as usize;
    let fade_start = fade_start.saturating_sub(cf);
    let fade_end = ((t1 as usize) + cf).min(t_total);
    let t0u = t0 as usize;
    let t1u = t1 as usize;

    for t in fade_start..fade_end {
        if t >= t0u && t < t1u {
            continue;
        }
        let m = if t < t0u {
            let rl = t0u - fade_start;
            (t - fade_start + 1) as f32 / (rl + 1) as f32
        } else {
            let rl = fade_end - t1u;
            (fade_end - t) as f32 / (rl + 1) as f32
        };
        let t_src = t.min(t_cover.saturating_sub(1));
        for c in 0..oc {
            let idx = t * oc + c;
            output[idx] = m * output[idx] + (1.0 - m) * src[t_src * oc + c];
        }
    }
}

/// Hard latent splice: keep generated frames in `[t0, t1)`, copy source elsewhere.
pub fn splice_latents(
    output: &mut [f32],
    src: &[f32],
    frames: i64,
    out_channels: i64,
    t0: i64,
    t1: i64,
) {
    let oc = out_channels as usize;
    let copy_t = (src.len() / oc).min(frames as usize);
    for t in 0..copy_t {
        if t >= t0 as usize && t < t1 as usize {
            continue;
        }
        let dst = t * oc;
        let src_off = t * oc;
        output[dst..dst + oc].copy_from_slice(&src[src_off..src_off + oc]);
    }
}

#[cfg(test)]
mod tests {
    use super::{blend_boundaries, inject_preserved, region_to_frames, splice_latents};

    #[test]
    fn region_maps_seconds_to_25hz_frames() {
        let (t0, t1) = region_to_frames(1.0, 3.0, 200).unwrap();
        assert_eq!(t0, 25);
        assert_eq!(t1, 75);
        assert!(region_to_frames(3.0, 1.0, 200).is_err());
    }

    #[test]
    fn splice_keeps_zone_copies_outside() {
        let frames = 4i64;
        let oc = 2i64;
        let mut out = vec![9.0f32; (frames * oc) as usize];
        let mut src = vec![0.0f32; (frames * oc) as usize];
        for t in 0..frames as usize {
            src[t * 2] = t as f32;
            src[t * 2 + 1] = -(t as f32);
        }
        splice_latents(&mut out, &src, frames, oc, 1, 3);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], -0.0);
        assert_eq!(out[2], 9.0); // inside zone kept
        assert_eq!(out[4], 9.0);
        assert_eq!(out[6], 3.0);
    }

    #[test]
    fn inject_skips_after_cutoff() {
        let frames = 3i64;
        let oc = 1i64;
        let mut xt = vec![5.0f32; 3];
        let noise = vec![1.0f32; 3];
        let src = vec![2.0f32; 3];
        inject_preserved(&mut xt, &noise, &src, frames, oc, 1, 2, 0.5, 2, 4, 0.25);
        // cutoff = round(0.5*4)=2; step 2 >= cutoff → no change
        assert_eq!(xt, vec![5.0, 5.0, 5.0]);
        inject_preserved(&mut xt, &noise, &src, frames, oc, 1, 2, 0.5, 0, 4, 0.5);
        // outside zone t=0,2: 0.5*1 + 0.5*2 = 1.5; inside t=1 unchanged
        assert_eq!(xt[0], 1.5);
        assert_eq!(xt[1], 5.0);
        assert_eq!(xt[2], 1.5);
    }

    #[test]
    fn blend_ramps_outside_zone() {
        let frames = 6i64;
        let oc = 1i64;
        let mut out = vec![1.0f32; 6];
        let src = vec![0.0f32; 6];
        blend_boundaries(&mut out, &src, frames, oc, 2, 4, 2);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], 1.0);
        assert!(out[0] < 1.0 && out[0] > 0.0);
        assert!(out[5] < 1.0 && out[5] > 0.0);
    }
}
