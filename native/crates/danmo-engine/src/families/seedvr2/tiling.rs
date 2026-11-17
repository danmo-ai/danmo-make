//! Spatial VAE tiling for SeedVR2, matching the reference: 512 pixel tiles
//! (64 latent) with 64 pixel overlap (8 latent), cosine ramp blending, edge
//! tiles fully inside the overlap skipped. Anything that fits in one tile
//! runs a single untiled pass. Every tile carries all frames; the temporal
//! axis is never tiled.

use super::vae::{self, VaeSpec};
use crate::runtime::GgmlContext;
use crate::weights::WeightError;

const TILE_PX: i64 = 512;
const OVERLAP_PX: i64 = 64;
const SCALE: i64 = 8;

fn cos_ramp(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1).max(1) as f32;
            0.5 - 0.5 * (t * std::f32::consts::PI).cos()
        })
        .collect()
}

/// Slice `[tw, th, F, C]` out of a `[W, H, F, C]` buffer (w fastest, then h,
/// frame, channel).
fn slice_plane(
    src: &[f32],
    w: i64,
    h: i64,
    x: i64,
    y: i64,
    tw: i64,
    th: i64,
    frames: i64,
    c: i64,
) -> Vec<f32> {
    let (w, h, tw, th, frames, c) = (
        w as usize,
        h as usize,
        tw as usize,
        th as usize,
        frames as usize,
        c as usize,
    );
    let mut out = vec![0f32; tw * th * frames * c];
    for ch in 0..c {
        for f in 0..frames {
            for row in 0..th {
                let src_row =
                    ((y as usize + row) * w + x as usize) + f * w * h + ch * frames * w * h;
                let dst_row = row * tw + f * tw * th + ch * frames * tw * th;
                out[dst_row..dst_row + tw].copy_from_slice(&src[src_row..src_row + tw]);
            }
        }
    }
    out
}

/// Encode `[W, H, F, 3]` `[-1, 1]` to the scaled latent `[W/8, H/8, D, 16]`.
pub fn encode(
    ctx: &GgmlContext,
    loaded: &[(String, Vec<f32>)],
    spec: &VaeSpec,
    image: &[f32],
    w: i64,
    h: i64,
    frames: i64,
) -> Result<Vec<f32>, WeightError> {
    if w <= TILE_PX && h <= TILE_PX {
        return vae::encode_tile(ctx, loaded, spec, image, w, h, frames);
    }
    let h_lat = h / SCALE;
    let w_lat = w / SCALE;
    let latent_tile = TILE_PX / SCALE;
    let overlap = (OVERLAP_PX / SCALE).min(latent_tile - 1);
    let stride = (latent_tile - overlap).max(1);
    let ramp = cos_ramp(overlap as usize);
    let c = spec.latent as usize;
    let d = vae::encode_frames_out(spec, frames) as usize;
    let plane = (w_lat * h_lat) as usize;
    let mut acc = vec![0f32; plane * d * c];
    let mut count = vec![0f32; plane];
    let mut y = 0;
    while y < h_lat {
        let y_end = (y + latent_tile).min(h_lat);
        let mut x = 0;
        while x < w_lat {
            let x_end = (x + latent_tile).min(w_lat);
            if (y > 0 && y_end - y <= overlap) || (x > 0 && x_end - x <= overlap) {
                x += stride;
                continue;
            }
            let tile = slice_plane(
                image,
                w,
                h,
                x * SCALE,
                y * SCALE,
                (x_end - x) * SCALE,
                (y_end - y) * SCALE,
                frames,
                3,
            );
            let enc = vae::encode_tile(
                ctx,
                loaded,
                spec,
                &tile,
                (x_end - x) * SCALE,
                (y_end - y) * SCALE,
                frames,
            )?;
            let th = y_end - y;
            let tw = x_end - x;
            let tile_plane = (tw * th) as usize;
            let eff_h = th.min(h_lat - y);
            let eff_w = tw.min(w_lat - x);
            let ov_h = overlap.min(eff_h - 1).max(0);
            let ov_w = overlap.min(eff_w - 1).max(0);
            for row in 0..eff_h {
                let mut wh = 1.0f32;
                if ov_h > 0 {
                    if y > 0 && row < ov_h {
                        wh = ramp[row as usize];
                    }
                    if y_end < h_lat && row >= eff_h - ov_h {
                        wh = 1.0 - ramp[(row - (eff_h - ov_h)) as usize];
                    }
                }
                for col in 0..eff_w {
                    let mut ww = 1.0f32;
                    if ov_w > 0 {
                        if x > 0 && col < ov_w {
                            ww = ramp[col as usize];
                        }
                        if x_end < w_lat && col >= eff_w - ov_w {
                            ww = 1.0 - ramp[(col - (eff_w - ov_w)) as usize];
                        }
                    }
                    let w2d = wh * ww;
                    let dst_px = ((y + row) * w_lat + (x + col)) as usize;
                    let src_px = (row * tw + col) as usize;
                    count[dst_px] += w2d;
                    for ch in 0..c {
                        for f in 0..d {
                            acc[dst_px + f * plane + ch * d * plane] +=
                                enc[src_px + f * tile_plane + ch * d * tile_plane] * w2d;
                        }
                    }
                }
            }
            x += stride;
        }
        y += stride;
    }
    for px in 0..plane {
        let n = count[px].max(1e-6);
        for ch in 0..c {
            for f in 0..d {
                acc[px + f * plane + ch * d * plane] /= n;
            }
        }
    }
    Ok(acc)
}

/// Decode the scaled latent `[W, H, D, 16]` to `[W*8, H*8, F, 3]` `[-1, 1]`.
pub fn decode(
    ctx: &GgmlContext,
    loaded: &[(String, Vec<f32>)],
    spec: &VaeSpec,
    latent: &[f32],
    w: i64,
    h: i64,
    frames: i64,
) -> Result<Vec<f32>, WeightError> {
    let latent_tile = TILE_PX / SCALE;
    if w <= latent_tile && h <= latent_tile {
        return vae::decode_tile(ctx, loaded, spec, latent, w, h, frames);
    }
    let overlap = (OVERLAP_PX / SCALE).min(latent_tile - 1);
    let stride = (latent_tile - overlap).max(1);
    let ramp = cos_ramp(OVERLAP_PX as usize);
    let c = 3usize;
    let frames_out = vae::decode_frames_out(spec, frames) as usize;
    let w_out = w * SCALE;
    let h_out = h * SCALE;
    let plane = (w_out * h_out) as usize;
    let mut acc = vec![0f32; plane * frames_out * c];
    let mut count = vec![0f32; plane];
    let mut y = 0;
    while y < h {
        let y_end = (y + latent_tile).min(h);
        let mut x = 0;
        while x < w {
            let x_end = (x + latent_tile).min(w);
            if (y > 0 && y_end - y <= overlap) || (x > 0 && x_end - x <= overlap) {
                x += stride;
                continue;
            }
            let tile = slice_plane(
                latent,
                w,
                h,
                x,
                y,
                x_end - x,
                y_end - y,
                frames,
                spec.latent,
            );
            let dec = vae::decode_tile(ctx, loaded, spec, &tile, x_end - x, y_end - y, frames)?;
            let th_px = (y_end - y) * SCALE;
            let tw_px = (x_end - x) * SCALE;
            let tile_plane = (tw_px * th_px) as usize;
            let eff_h = th_px.min(h_out - y * SCALE);
            let eff_w = tw_px.min(w_out - x * SCALE);
            let ov_h = OVERLAP_PX.min(eff_h - 1).max(0);
            let ov_w = OVERLAP_PX.min(eff_w - 1).max(0);
            for row in 0..eff_h {
                let mut wh = 1.0f32;
                if ov_h > 0 {
                    if y > 0 && row < ov_h {
                        wh = ramp[row as usize];
                    }
                    if y_end < h && row >= eff_h - ov_h {
                        wh = 1.0 - ramp[(row - (eff_h - ov_h)) as usize];
                    }
                }
                for col in 0..eff_w {
                    let mut ww = 1.0f32;
                    if ov_w > 0 {
                        if x > 0 && col < ov_w {
                            ww = ramp[col as usize];
                        }
                        if x_end < w && col >= eff_w - ov_w {
                            ww = 1.0 - ramp[(col - (eff_w - ov_w)) as usize];
                        }
                    }
                    let w2d = wh * ww;
                    let dst_px = ((y * SCALE + row) * w_out + (x * SCALE + col)) as usize;
                    let src_px = (row * tw_px + col) as usize;
                    count[dst_px] += w2d;
                    for ch in 0..c {
                        for f in 0..frames_out {
                            acc[dst_px + f * plane + ch * frames_out * plane] +=
                                dec[src_px + f * tile_plane + ch * frames_out * tile_plane] * w2d;
                        }
                    }
                }
            }
            x += stride;
        }
        y += stride;
    }
    for px in 0..plane {
        let n = count[px].max(1e-6);
        for ch in 0..c {
            for f in 0..frames_out {
                acc[px + f * plane + ch * frames_out * plane] /= n;
            }
        }
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cos_ramp_is_a_symmetric_half_cosine() {
        let ramp = cos_ramp(8);
        assert_eq!(ramp.len(), 8);
        assert_eq!(ramp[0], 0.0);
        assert!((ramp[7] - 1.0).abs() < 1e-7);
        for i in 0..8 {
            assert!(ramp[i] >= 0.0 && ramp[i] <= 1.0);
            assert!(
                (ramp[i] + ramp[7 - i] - 1.0).abs() < 1e-6,
                "ramp[{i}] + ramp[{}] != 1",
                7 - i
            );
            if i > 0 {
                assert!(ramp[i] > ramp[i - 1]);
            }
        }
        assert!(cos_ramp(0).is_empty());
    }

    #[test]
    fn slice_plane_extracts_per_channel_windows() {
        // [W=4, H=3, C=2] planar: element (x, y, c) = c*12 + y*4 + x.
        let src: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let tile = slice_plane(&src, 4, 3, 1, 1, 2, 2, 1, 2);
        // channel 0: pixels (1,1),(2,1),(1,2),(2,2) -> 5,6,9,10
        // channel 1: offset 12 -> 17,18,21,22
        assert_eq!(tile, vec![5.0, 6.0, 9.0, 10.0, 17.0, 18.0, 21.0, 22.0]);
    }

    #[test]
    fn slice_plane_extracts_all_frames_per_tile() {
        // [W=2, H=2, F=2, C=2]: element (x, y, f, c) = c*8 + f*4 + y*2 + x.
        let src: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let tile = slice_plane(&src, 2, 2, 0, 0, 2, 2, 2, 2);
        assert_eq!(tile, src);
    }
}
