//! Edit source preparation shared by the img2img families. A source image is
//! floored to a multiple of `align` on each edge and center-cropped, then the
//! interleaved RGB bytes become a planar `[W, H, 3]` f32 tensor in `[-1, 1]`
//! (diffusers img2img normalization). No family ids appear here.

use crate::weights::WeightError;

/// Floor `width`/`height` to multiples of `align` and center-crop the
/// interleaved RGB buffer. Returns `(rgb, width, height)`.
pub fn center_crop_align(
    rgb: &[u8],
    width: u32,
    height: u32,
    align: u32,
) -> Result<(Vec<u8>, u32, u32), WeightError> {
    if align < 1 {
        return Err(WeightError::NotBuilt("align must be positive".into()));
    }
    let out_w = width / align * align;
    let out_h = height / align * align;
    if out_w < 1 || out_h < 1 {
        return Err(WeightError::NotBuilt(format!(
            "source {width}x{height} is smaller than the {align}px alignment"
        )));
    }
    let pixels = (width as usize) * (height as usize) * 3;
    if rgb.len() != pixels {
        return Err(WeightError::Shape {
            name: "rgb".into(),
            expected: vec![pixels as i64],
            got: vec![rgb.len() as i64],
        });
    }
    let off_x = (width - out_w) / 2;
    let off_y = (height - out_h) / 2;
    let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * 3];
    for y in 0..out_h {
        let src = ((off_y + y) as usize * width as usize + off_x as usize) * 3;
        let dst = y as usize * out_w as usize * 3;
        out[dst..dst + out_w as usize * 3].copy_from_slice(&rgb[src..src + out_w as usize * 3]);
    }
    Ok((out, out_w, out_h))
}

/// Interleaved RGB bytes to a planar `[W, H, 3]` f32 tensor in `[-1, 1]`.
/// The plane index is `x + y * width + channel * width * height`.
pub fn rgb_to_n11_planar(rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>, WeightError> {
    let w = width as usize;
    let h = height as usize;
    if rgb.len() != w * h * 3 {
        return Err(WeightError::Shape {
            name: "rgb".into(),
            expected: vec![(w * h * 3) as i64],
            got: vec![rgb.len() as i64],
        });
    }
    let plane = w * h;
    let mut out = vec![0f32; plane * 3];
    for y in 0..h {
        for x in 0..w {
            for channel in 0..3 {
                let byte = rgb[(y * w + x) * 3 + channel];
                out[x + y * w + channel * plane] = (byte as f32 / 255.0) * 2.0 - 1.0;
            }
        }
    }
    Ok(out)
}

/// PIL-style separable Lanczos-3 (`Image.Resampling.LANCZOS`) with antialias
/// widening on downscale. `rgb` is interleaved bytes; a same-size call copies.
pub fn resize_lanczos(
    rgb: &[u8],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, WeightError> {
    let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
    if sw < 1 || sh < 1 || dw < 1 || dh < 1 || rgb.len() != sw * sh * 3 {
        return Err(WeightError::Shape {
            name: "rgb".into(),
            expected: vec![(sw * sh * 3) as i64],
            got: vec![rgb.len() as i64],
        });
    }
    fn sinc(x: f32) -> f32 {
        if x.abs() < 1e-8 {
            1.0
        } else {
            let px = std::f32::consts::PI * x;
            px.sin() / px
        }
    }
    fn kernel(x: f32) -> f32 {
        let x = x.abs();
        if x >= 3.0 {
            0.0
        } else {
            sinc(x) * sinc(x / 3.0)
        }
    }
    fn weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = in_size as f32 / out_size as f32;
        let filter = scale.max(1.0);
        let support = 3.0 * filter;
        (0..out_size)
            .map(|o| {
                let center = (o as f32 + 0.5) * scale;
                let first = (center - support).ceil().max(0.0) as usize;
                let last = ((center + support).floor() as usize).min(in_size - 1);
                let mut ws = Vec::with_capacity(last + 1 - first);
                let mut sum = 0f32;
                for i in first..=last {
                    let w = kernel((i as f32 + 0.5 - center) / filter);
                    ws.push(w);
                    sum += w;
                }
                if sum.abs() > 1e-8 {
                    for w in &mut ws {
                        *w /= sum;
                    }
                }
                (first, ws)
            })
            .collect()
    }
    let wx = weights(sw, dw);
    let mut tmp = vec![0f32; dw * sh * 3];
    for y in 0..sh {
        for (ox, (first, ws)) in wx.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0f32;
                for (k, &w) in ws.iter().enumerate() {
                    acc += w * rgb[(y * sw + first + k) * 3 + c] as f32;
                }
                tmp[(y * dw + ox) * 3 + c] = acc;
            }
        }
    }
    let wy = weights(sh, dh);
    let mut out = vec![0u8; dw * dh * 3];
    for (oy, (first, ws)) in wy.iter().enumerate() {
        for x in 0..dw {
            for c in 0..3 {
                let mut acc = 0f32;
                for (k, &w) in ws.iter().enumerate() {
                    acc += w * tmp[((first + k) * dw + x) * 3 + c];
                }
                out[(oy * dw + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

/// PIL-style separable bicubic (`Image.Resampling.BICUBIC`, a = -0.5) with
/// antialias widening on downscale. `rgb` is interleaved bytes.
pub fn resize_bicubic(
    rgb: &[u8],
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
) -> Result<Vec<u8>, WeightError> {
    let (sw, sh, dw, dh) = (sw as usize, sh as usize, dw as usize, dh as usize);
    if sw < 1 || sh < 1 || dw < 1 || dh < 1 || rgb.len() != sw * sh * 3 {
        return Err(WeightError::Shape {
            name: "rgb".into(),
            expected: vec![(sw * sh * 3) as i64],
            got: vec![rgb.len() as i64],
        });
    }
    fn kernel(x: f32) -> f32 {
        let x = x.abs();
        if x < 1.0 {
            1.5 * x.powi(3) - 2.5 * x.powi(2) + 1.0
        } else if x < 2.0 {
            -0.5 * x.powi(3) + 2.5 * x.powi(2) - 4.0 * x + 2.0
        } else {
            0.0
        }
    }
    fn weights(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
        let scale = in_size as f32 / out_size as f32;
        let filter = scale.max(1.0);
        let support = 2.0 * filter;
        (0..out_size)
            .map(|o| {
                let center = (o as f32 + 0.5) * scale;
                let first = (center - support).ceil().max(0.0) as usize;
                let last = ((center + support).floor() as usize).min(in_size - 1);
                let mut ws = Vec::with_capacity(last + 1 - first);
                let mut sum = 0f32;
                for i in first..=last {
                    let w = kernel((i as f32 + 0.5 - center) / filter);
                    ws.push(w);
                    sum += w;
                }
                if sum.abs() > 1e-8 {
                    for w in &mut ws {
                        *w /= sum;
                    }
                }
                (first, ws)
            })
            .collect()
    }
    let wx = weights(sw, dw);
    let mut tmp = vec![0f32; dw * sh * 3];
    for y in 0..sh {
        for (ox, (first, ws)) in wx.iter().enumerate() {
            for c in 0..3 {
                let mut acc = 0f32;
                for (k, &w) in ws.iter().enumerate() {
                    acc += w * rgb[(y * sw + first + k) * 3 + c] as f32;
                }
                tmp[(y * dw + ox) * 3 + c] = acc;
            }
        }
    }
    let wy = weights(sh, dh);
    let mut out = vec![0u8; dw * dh * 3];
    for (oy, (first, ws)) in wy.iter().enumerate() {
        for x in 0..dw {
            for c in 0..3 {
                let mut acc = 0f32;
                for (k, &w) in ws.iter().enumerate() {
                    acc += w * tmp[((first + k) * dw + x) * 3 + c];
                }
                out[(oy * dw + x) * 3 + c] = acc.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{center_crop_align, resize_bicubic, resize_lanczos, rgb_to_n11_planar};

    #[test]
    fn bicubic_keeps_a_constant_image_constant() {
        let rgb = vec![77u8; 8 * 8 * 3];
        let up = resize_bicubic(&rgb, 8, 8, 16, 16).unwrap();
        assert!(up.iter().all(|&v| v == 77));
        let down = resize_bicubic(&rgb, 8, 8, 4, 4).unwrap();
        assert!(down.iter().all(|&v| v == 77));
        let same = resize_bicubic(&rgb, 8, 8, 8, 8).unwrap();
        assert_eq!(same, rgb);
        assert!(resize_bicubic(&rgb[..3], 8, 8, 4, 4).is_err());
    }

    #[test]
    fn lanczos_keeps_a_constant_image_constant() {
        let rgb = vec![77u8; 8 * 8 * 3];
        let up = resize_lanczos(&rgb, 8, 8, 16, 16).unwrap();
        assert!(up.iter().all(|&v| v == 77));
        let down = resize_lanczos(&rgb, 8, 8, 4, 4).unwrap();
        assert!(down.iter().all(|&v| v == 77));
        let same = resize_lanczos(&rgb, 8, 8, 8, 8).unwrap();
        assert_eq!(same, rgb);
        assert!(resize_lanczos(&rgb[..3], 8, 8, 4, 4).is_err());
    }

    #[test]
    fn crop_floors_and_centers() {
        // 5x3 source, each pixel is its x coordinate. Align 2 -> 4x2, crop x=0..? center: off_x=0 (5-4)/2=0, off_y=0 (3-2)/2=0.
        let mut rgb = vec![0u8; 5 * 3 * 3];
        for y in 0..3usize {
            for x in 0..5usize {
                rgb[(y * 5 + x) * 3] = (x + y * 10) as u8;
            }
        }
        let (out, w, h) = center_crop_align(&rgb, 5, 3, 2).unwrap();
        assert_eq!((w, h), (4, 2));
        assert_eq!(out[0], 0);
        assert_eq!(out[3], 1);
        assert_eq!(out[4 * 3], 10);
        let err = center_crop_align(&rgb, 5, 3, 8).unwrap_err();
        assert!(err.to_string().contains("alignment"), "{err}");
    }

    #[test]
    fn crop_odd_remainder_centers() {
        // 7x4 -> align 4 gives 4x4 with off_x = 1.
        let mut rgb = vec![0u8; 7 * 4 * 3];
        for y in 0..4usize {
            for x in 0..7usize {
                rgb[(y * 7 + x) * 3] = x as u8;
            }
        }
        let (out, w, h) = center_crop_align(&rgb, 7, 4, 4).unwrap();
        assert_eq!((w, h), (4, 4));
        assert_eq!(out[0], 1);
        assert_eq!(out[3], 2);
    }

    #[test]
    fn n11_planar_maps_bytes() {
        let rgb = [0u8, 128, 255, 10, 20, 30];
        let out = rgb_to_n11_planar(&rgb, 2, 1).unwrap();
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], -1.0);
        assert!((out[1] - ((10.0 / 255.0) * 2.0 - 1.0) as f32).abs() < 1e-7);
        assert!((out[2] - (128.0 / 255.0 * 2.0 - 1.0) as f32).abs() < 1e-7);
        assert_eq!(out[4], 1.0);
    }
}
