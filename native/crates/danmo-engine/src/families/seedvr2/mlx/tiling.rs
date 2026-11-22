//! Host spatial tiling around the MLX 3D VAE. Every tile carries the complete
//! temporal window; overlap blending uses the reference half-cosine ramps.

use danmo_mlx::{Array, Result, Stream};

use super::super::tiling::slice_plane;
use super::vae::{Vae, VaeStreamCache};

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

pub fn encode(
    vae: &Vae,
    image: &[f32],
    w: i64,
    h: i64,
    frames: i64,
    stream: &Stream,
) -> Result<Array> {
    encode_with_cache(vae, image, w, h, frames, stream, None)
}

pub fn encode_with_cache(
    vae: &Vae,
    image: &[f32],
    w: i64,
    h: i64,
    frames: i64,
    stream: &Stream,
    mut cache: Option<&mut VaeStreamCache>,
) -> Result<Array> {
    if w <= TILE_PX && h <= TILE_PX {
        let pixels = Array::from_f32_slice(image, &[1, 3, frames as i32, h as i32, w as i32])?;
        return vae.encode_tile_with_cache(&pixels, stream, cache);
    }
    let h_lat = h / SCALE;
    let w_lat = w / SCALE;
    let latent_tile = TILE_PX / SCALE;
    let overlap = (OVERLAP_PX / SCALE).min(latent_tile - 1);
    let stride = (latent_tile - overlap).max(1);
    let ramp = cos_ramp(overlap as usize);
    let channels = vae.spec().latent as usize;
    let depth = vae.encode_frames_out(frames) as usize;
    let plane = (w_lat * h_lat) as usize;
    let mut acc = vec![0f32; plane * depth * channels];
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
            let pixels = Array::from_f32_slice(
                &tile,
                &[
                    1,
                    3,
                    frames as i32,
                    ((y_end - y) * SCALE) as i32,
                    ((x_end - x) * SCALE) as i32,
                ],
            )?;
            let encoded = vae
                .encode_tile_with_cache(&pixels, stream, cache.as_deref_mut())?
                .to_vec_f32(stream)?;
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
                    let weight = wh * ww;
                    let dst_px = ((y + row) * w_lat + x + col) as usize;
                    let src_px = (row * tw + col) as usize;
                    count[dst_px] += weight;
                    for channel in 0..channels {
                        for frame in 0..depth {
                            acc[dst_px + frame * plane + channel * depth * plane] += encoded
                                [src_px + frame * tile_plane + channel * depth * tile_plane]
                                * weight;
                        }
                    }
                }
            }
            x += stride;
        }
        y += stride;
    }
    for px in 0..plane {
        let divisor = count[px].max(1e-6);
        for channel in 0..channels {
            for frame in 0..depth {
                acc[px + frame * plane + channel * depth * plane] /= divisor;
            }
        }
    }
    Array::from_f32_slice(
        &acc,
        &[1, channels as i32, depth as i32, h_lat as i32, w_lat as i32],
    )
}

pub fn decode(
    vae: &Vae,
    latent: &Array,
    w: i64,
    h: i64,
    frames: i64,
    stream: &Stream,
) -> Result<Array> {
    decode_with_cache(vae, latent, w, h, frames, stream, None)
}

pub fn decode_with_cache(
    vae: &Vae,
    latent: &Array,
    w: i64,
    h: i64,
    frames: i64,
    stream: &Stream,
    mut cache: Option<&mut VaeStreamCache>,
) -> Result<Array> {
    let latent_tile = TILE_PX / SCALE;
    if w <= latent_tile && h <= latent_tile {
        return vae.decode_tile_with_cache(latent, stream, cache);
    }
    let latent = latent.to_vec_f32(stream)?;
    let overlap = (OVERLAP_PX / SCALE).min(latent_tile - 1);
    let stride = (latent_tile - overlap).max(1);
    let ramp = cos_ramp(OVERLAP_PX as usize);
    let frames_out = vae.decode_frames_out(frames) as usize;
    let w_out = w * SCALE;
    let h_out = h * SCALE;
    let plane = (w_out * h_out) as usize;
    let mut acc = vec![0f32; plane * frames_out * 3];
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
                &latent,
                w,
                h,
                x,
                y,
                x_end - x,
                y_end - y,
                frames,
                vae.spec().latent,
            );
            let tile = Array::from_f32_slice(
                &tile,
                &[
                    1,
                    vae.spec().latent as i32,
                    frames as i32,
                    (y_end - y) as i32,
                    (x_end - x) as i32,
                ],
            )?;
            let decoded = vae
                .decode_tile_with_cache(&tile, stream, cache.as_deref_mut())?
                .to_vec_f32(stream)?;
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
                    let weight = wh * ww;
                    let dst_px = ((y * SCALE + row) * w_out + x * SCALE + col) as usize;
                    let src_px = (row * tw_px + col) as usize;
                    count[dst_px] += weight;
                    for channel in 0..3 {
                        for frame in 0..frames_out {
                            acc[dst_px + frame * plane + channel * frames_out * plane] += decoded
                                [src_px + frame * tile_plane + channel * frames_out * tile_plane]
                                * weight;
                        }
                    }
                }
            }
            x += stride;
        }
        y += stride;
    }
    for px in 0..plane {
        let divisor = count[px].max(1e-6);
        for channel in 0..3 {
            for frame in 0..frames_out {
                acc[px + frame * plane + channel * frames_out * plane] /= divisor;
            }
        }
    }
    Array::from_f32_slice(&acc, &[1, 3, frames_out as i32, h_out as i32, w_out as i32])
}
