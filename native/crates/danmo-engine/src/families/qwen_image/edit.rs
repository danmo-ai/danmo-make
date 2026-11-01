use crate::weights::WeightError;

const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
const PATCH: usize = 14;
const MERGE: usize = 2;
const TEMPORAL: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionGrid {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

pub struct VisionInput {
    pub values: Vec<f32>,
    pub grid: VisionGrid,
}

pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), WeightError> {
    if height == 0 || width == 0 || factor == 0 || min_pixels == 0 || max_pixels < min_pixels {
        return Err(WeightError::NotBuilt(
            "qwen edit resize dimensions are invalid".into(),
        ));
    }
    if height.max(width) > height.min(width) * 200 {
        return Err(WeightError::NotBuilt(format!(
            "qwen edit aspect ratio must be smaller than 200, got {height}x{width}"
        )));
    }
    let round = |value: usize| ((value + factor / 2) / factor) * factor;
    let mut h = round(height);
    let mut w = round(width);
    if h * w > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        w = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
    } else if h * w < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((h, w))
}

pub fn preprocess_rgb(rgb: &[u8], width: usize, height: usize) -> Result<VisionInput, WeightError> {
    if width == 0 || height == 0 || rgb.len() != width * height * 3 {
        return Err(WeightError::NotBuilt(format!(
            "qwen edit RGB length {} is not {width}×{height}×3",
            rgb.len()
        )));
    }
    let (resized_h, resized_w) =
        smart_resize(height, width, PATCH * MERGE, 56 * 56, 28 * 28 * 1280)?;
    let resized = if (resized_w, resized_h) == (width, height) {
        rgb.to_vec()
    } else {
        resize_bicubic(rgb, width, height, resized_w, resized_h)
    };
    let grid = VisionGrid {
        t: 1,
        h: resized_h / PATCH,
        w: resized_w / PATCH,
    };
    let mut values = Vec::with_capacity(grid.t * grid.h * grid.w * 3 * TEMPORAL * PATCH * PATCH);
    for gh in 0..grid.h / MERGE {
        for gw in 0..grid.w / MERGE {
            for mh in 0..MERGE {
                for mw in 0..MERGE {
                    for channel in 0..3 {
                        for _ in 0..TEMPORAL {
                            for ph in 0..PATCH {
                                for pw in 0..PATCH {
                                    let x = (gw * MERGE + mw) * PATCH + pw;
                                    let y = (gh * MERGE + mh) * PATCH + ph;
                                    let index = (y * resized_w + x) * 3 + channel;
                                    values.push(
                                        (resized[index] as f32 / 255.0 - CLIP_MEAN[channel])
                                            / CLIP_STD[channel],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(VisionInput { values, grid })
}

pub fn rgb_to_vae(rgb: &[u8]) -> Vec<f32> {
    rgb.iter()
        .map(|value| *value as f32 / 127.5 - 1.0)
        .collect()
}

/// Edit geometry from main `compute_qwen_edit_dimensions`. Output keeps the
/// source size floored to 16 unless the request sets an explicit size, in
/// which case a 1024² ratio-preserving target applies. The vision condition is
/// 384² and the VAE condition 1024², both ratio-preserving and 32-aligned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditDimensions {
    pub out_w: u32,
    pub out_h: u32,
    pub vl_w: u32,
    pub vl_h: u32,
    pub vae_w: u32,
    pub vae_h: u32,
}

pub fn compute_edit_dimensions(
    source_w: u32,
    source_h: u32,
    size: Option<(u32, u32)>,
) -> Result<EditDimensions, WeightError> {
    if source_w == 0 || source_h == 0 {
        return Err(WeightError::NotBuilt(
            "qwen edit source image is empty".into(),
        ));
    }
    let floor16 = |value: u32| (value / 16 * 16).max(16);
    let (out_w, out_h) = match size {
        None => (floor16(source_w), floor16(source_h)),
        Some((req_w, req_h)) => {
            let ratio = f64::from(source_w) / f64::from(source_h);
            let area = 1024.0 * 1024.0;
            let calc_w = py_round((area * ratio).sqrt() / 32.0) * 32.0;
            let calc_h = py_round(calc_w / ratio / 32.0) * 32.0;
            let use_w = if req_w > 0 { f64::from(req_w) } else { calc_w };
            let use_h = if req_h > 0 { f64::from(req_h) } else { calc_h };
            (
                floor16(use_w.max(0.0) as u32),
                floor16(use_h.max(0.0) as u32),
            )
        }
    };
    let ratio = f64::from(source_w) / f64::from(source_h);
    let scaled = |area: f64| {
        let w = py_round((area * ratio).sqrt() / 32.0) * 32.0;
        let h = py_round(w / ratio / 32.0) * 32.0;
        (w.max(32.0) as u32, h.max(32.0) as u32)
    };
    let (vl_w, vl_h) = scaled(384.0 * 384.0);
    let (vae_w, vae_h) = scaled(1024.0 * 1024.0);
    Ok(EditDimensions {
        out_w,
        out_h,
        vl_w,
        vl_h,
        vae_w,
        vae_h,
    })
}

/// Python `round` (banker's rounding) so the 32-alignment matches main.
fn py_round(value: f64) -> f64 {
    let floor = value.floor();
    let diff = value - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

pub fn resize_rgb(rgb: &[u8], width: u32, height: u32, target_w: u32, target_h: u32) -> Vec<u8> {
    resize_bicubic(
        rgb,
        width as usize,
        height as usize,
        target_w as usize,
        target_h as usize,
    )
}

fn resize_bicubic(
    source: &[u8],
    width: usize,
    height: usize,
    target_w: usize,
    target_h: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; target_w * target_h * 3];
    for y in 0..target_h {
        let source_y = (y as f32 + 0.5) * height as f32 / target_h as f32 - 0.5;
        let y0 = source_y.floor() as isize;
        for x in 0..target_w {
            let source_x = (x as f32 + 0.5) * width as f32 / target_w as f32 - 0.5;
            let x0 = source_x.floor() as isize;
            for channel in 0..3 {
                let mut value = 0.0;
                for ky in 0..4 {
                    let wy = cubic(source_y - (y0 + ky - 1) as f32);
                    for kx in 0..4 {
                        let wx = cubic(source_x - (x0 + kx - 1) as f32);
                        let sx = (x0 + kx - 1).clamp(0, width as isize - 1) as usize;
                        let sy = (y0 + ky - 1).clamp(0, height as isize - 1) as usize;
                        value += source[(sy * width + sx) * 3 + channel] as f32 * wx * wy;
                    }
                }
                out[(y * target_w + x) * 3 + channel] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn cubic(x: f32) -> f32 {
    let x = x.abs();
    if x <= 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_rules_match_main() {
        assert_eq!(
            smart_resize(28, 28, 28, 56 * 56, 28 * 28 * 1280).unwrap(),
            (56, 56)
        );
        assert_eq!(
            smart_resize(112, 112, 28, 56 * 56, 28 * 28 * 1280).unwrap(),
            (112, 112)
        );
        assert!(smart_resize(1, 1000, 28, 56 * 56, 28 * 28 * 1280).is_err());
        let (h, w) = smart_resize(2000, 2000, 28, 56 * 56, 28 * 28 * 1280).unwrap();
        assert!(h * w <= 28 * 28 * 1280);
    }

    #[test]
    fn patches_follow_the_qwen_merge_order() {
        let width = 56;
        let height = 56;
        let rgb: Vec<u8> = (0..width * height * 3)
            .map(|index| (index % 251) as u8)
            .collect();
        let input = preprocess_rgb(&rgb, width, height).unwrap();
        assert_eq!(input.grid, VisionGrid { t: 1, h: 4, w: 4 });
        assert_eq!(input.values.len(), 16 * 3 * 2 * 14 * 14);
        assert_eq!(
            input.values[0],
            (rgb[0] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0]
        );
        assert_eq!(
            input.values[3 * 2 * 14 * 14],
            (rgb[42] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0]
        );
    }

    #[test]
    fn vae_rgb_is_scaled_to_minus_one_plus_one() {
        assert_eq!(rgb_to_vae(&[0, 127, 255]).len(), 3);
        let values = rgb_to_vae(&[0, 255]);
        assert_eq!(values, [-1.0, 1.0]);
    }

    #[test]
    fn edit_dimensions_match_main() {
        let dims = compute_edit_dimensions(1024, 768, None).unwrap();
        assert_eq!(
            dims,
            EditDimensions {
                out_w: 1024,
                out_h: 768,
                vl_w: 448,
                vl_h: 320,
                vae_w: 1184,
                vae_h: 896,
            }
        );
        let square = compute_edit_dimensions(1024, 1024, None).unwrap();
        assert_eq!(
            square,
            EditDimensions {
                out_w: 1024,
                out_h: 1024,
                vl_w: 384,
                vl_h: 384,
                vae_w: 1024,
                vae_h: 1024,
            }
        );
        let small = compute_edit_dimensions(48, 32, None).unwrap();
        assert_eq!(
            small,
            EditDimensions {
                out_w: 48,
                out_h: 32,
                vl_w: 480,
                vl_h: 320,
                vae_w: 1248,
                vae_h: 832,
            }
        );
        let sized = compute_edit_dimensions(1024, 768, Some((768, 512))).unwrap();
        assert_eq!((sized.out_w, sized.out_h), (768, 512));
        let odd = compute_edit_dimensions(33, 17, None).unwrap();
        assert_eq!((odd.out_w, odd.out_h), (32, 16));
        assert!(compute_edit_dimensions(0, 16, None).is_err());
    }
}
