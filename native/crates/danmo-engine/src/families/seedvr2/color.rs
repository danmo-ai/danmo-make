//! SeedVR2 output color correction: wavelet reconstruction against the
//! preprocessed source, then CIE LAB histogram matching on a/b with a
//! luminance blend. Port of `preprocess_mlx.SeedVR2Util` (5 level wavelet,
//! 3x3 binomial kernel at stride `2^level`, stable-rank histogram match).

use crate::weights::WeightError;

const LEVELS: usize = 5;
const LUMINANCE_WEIGHT: f32 = 0.8;
const KERNEL: [[f32; 3]; 3] = [
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625],
];

/// Correct `content` (`[W, H, 3]` planar, `[-1, 1]`) against `style` of the
/// same shape. Returns the corrected planes in `[-1, 1]`.
pub fn correct(
    content: &[f32],
    style: &[f32],
    w: usize,
    h: usize,
) -> Result<Vec<f32>, WeightError> {
    let plane = w * h;
    if content.len() != plane * 3 || style.len() != plane * 3 {
        return Err(WeightError::Shape {
            name: "seedvr2 color correction".into(),
            expected: vec![(plane * 3) as i64],
            got: vec![content.len() as i64 + style.len() as i64],
        });
    }
    let (content_high, _) = decompose(content, w, h);
    let (_, style_low) = decompose(style, w, h);
    let mut merged = vec![0f32; plane * 3];
    for i in 0..plane * 3 {
        merged[i] = (content_high[i] + style_low[i]).clamp(-1.0, 1.0);
    }
    let content_lab: Vec<[f32; 3]> = (0..plane)
        .map(|px| rgb_to_lab(pixel01(&merged, px, plane)))
        .collect();
    let style_lab: Vec<[f32; 3]> = (0..plane)
        .map(|px| rgb_to_lab(pixel01(style, px, plane)))
        .collect();
    let matched_a = hist_match(
        &content_lab.iter().map(|lab| lab[1]).collect::<Vec<_>>(),
        &style_lab.iter().map(|lab| lab[1]).collect::<Vec<_>>(),
    );
    let matched_b = hist_match(
        &content_lab.iter().map(|lab| lab[2]).collect::<Vec<_>>(),
        &style_lab.iter().map(|lab| lab[2]).collect::<Vec<_>>(),
    );
    let matched_l = hist_match(
        &content_lab.iter().map(|lab| lab[0]).collect::<Vec<_>>(),
        &style_lab.iter().map(|lab| lab[0]).collect::<Vec<_>>(),
    );
    let mut out = vec![0f32; plane * 3];
    for px in 0..plane {
        let l = LUMINANCE_WEIGHT * content_lab[px][0] + (1.0 - LUMINANCE_WEIGHT) * matched_l[px];
        let rgb = lab_to_rgb([l, matched_a[px], matched_b[px]]);
        for c in 0..3 {
            let unit = rgb[c].clamp(0.0, 1.0);
            out[px + c * plane] = unit * 2.0 - 1.0;
        }
    }
    Ok(out)
}

fn pixel01(planes: &[f32], px: usize, plane: usize) -> [f32; 3] {
    let mut rgb = [0f32; 3];
    for c in 0..3 {
        rgb[c] = ((planes[px + c * plane] + 1.0) * 0.5).clamp(0.0, 1.0);
    }
    rgb
}

/// 3x3 binomial blur with taps at stride `radius`, edge padding. The radius
/// is clamped to `max(1, min(h, w) // 8)` like the reference.
fn blur(image: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let radius = radius.max(1).min((w.min(h) / 8).max(1));
    let mut out = vec![0f32; image.len()];
    let at = |img: &[f32], x: i64, y: i64, c: usize| -> f32 {
        let x = x.clamp(0, w as i64 - 1) as usize;
        let y = y.clamp(0, h as i64 - 1) as usize;
        img[x + y * w + c * w * h]
    };
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0f32;
                for (ky, dy) in (-1i64..=1).enumerate() {
                    for (kx, dx) in (-1i64..=1).enumerate() {
                        acc += KERNEL[ky][kx]
                            * at(
                                image,
                                x as i64 + dx * radius as i64,
                                y as i64 + dy * radius as i64,
                                c,
                            );
                    }
                }
                out[x + y * w + c * w * h] = acc;
            }
        }
    }
    out
}

/// `(high, low)` after `LEVELS` rounds of `high += cur - blur(cur)`.
fn decompose(image: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
    let mut high = vec![0f32; image.len()];
    let mut cur = image.to_vec();
    for level in 0..LEVELS {
        let low = blur(&cur, w, h, 1 << level);
        for i in 0..cur.len() {
            high[i] += cur[i] - low[i];
        }
        cur = low;
    }
    (high, cur)
}

/// Stable-rank histogram match: the k-th smallest source value becomes the
/// k-th smallest reference value.
fn hist_match(source: &[f32], reference: &[f32]) -> Vec<f32> {
    let mut src_idx: Vec<usize> = (0..source.len()).collect();
    src_idx.sort_by(|&a, &b| {
        source[a]
            .partial_cmp(&source[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut ref_sorted = reference.to_vec();
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = vec![0f32; source.len()];
    for (rank, &src) in src_idx.iter().enumerate() {
        out[src] = ref_sorted[rank];
    }
    out
}

fn srgb_to_linear(x: f32) -> f32 {
    if x > 0.04045 {
        ((x + 0.055) / 1.055).powf(2.4)
    } else {
        x / 12.92
    }
}

fn linear_to_srgb(x: f32) -> f32 {
    if x > 0.003_130_8 {
        1.055 * x.max(0.0).powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * x
    }
}

const RGB_TO_XYZ: [[f32; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175],
    [0.019_333_9, 0.119_192, 0.950_304_1],
];
const XYZ_TO_RGB: [[f32; 3]; 3] = [
    [3.240_454_2, -1.537_138_5, -0.498_531_4],
    [-0.969_266, 1.876_010_8, 0.041_556],
    [0.055_643_4, -0.204_025_9, 1.057_225_2],
];

fn rgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let lin = rgb.map(srgb_to_linear);
    let mut xyz = [0f32; 3];
    for (i, row) in RGB_TO_XYZ.iter().enumerate() {
        xyz[i] = row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2];
    }
    xyz[0] /= 0.95047;
    xyz[2] /= 1.08883;
    let eps3 = (6.0f32 / 29.0).powi(3);
    let kappa = (29.0f32 / 3.0).powi(3);
    let f = |v: f32| {
        if v > eps3 {
            v.cbrt()
        } else {
            (kappa * v + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(xyz[0]), f(xyz[1]), f(xyz[2]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn lab_to_rgb(lab: [f32; 3]) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    let eps = 6.0f32 / 29.0;
    let kappa = (29.0f32 / 3.0).powi(3);
    let inv = |f: f32| {
        if f > eps {
            f.powi(3)
        } else {
            (116.0 * f - 16.0) / kappa
        }
    };
    let xyz = [inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883];
    let mut rgb = [0f32; 3];
    for (i, row) in XYZ_TO_RGB.iter().enumerate() {
        rgb[i] = linear_to_srgb(row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]);
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0f32; w * h * 3];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    out[x + y * w + c * w * h] =
                        ((x + y + c) as f32 / (w + h + 3) as f32) * 2.0 - 1.0;
                }
            }
        }
        out
    }

    #[test]
    fn blur_preserves_a_constant_field() {
        let img = vec![0.4f32; 16 * 16 * 3];
        let out = blur(&img, 16, 16, 2);
        for v in out {
            assert!((v - 0.4).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn decomposition_reconstructs_the_input() {
        let img = gradient(24, 16);
        let (high, low) = decompose(&img, 24, 16);
        for (i, &v) in img.iter().enumerate() {
            assert!((high[i] + low[i] - v).abs() < 1e-4, "index {i}");
        }
    }

    #[test]
    fn hist_match_maps_ranks_onto_the_reference() {
        let source = vec![0.5, -1.0, 2.0, 0.0];
        let reference = vec![10.0, 20.0, 30.0, 40.0];
        let out = hist_match(&source, &reference);
        assert_eq!(out, vec![30.0, 10.0, 40.0, 20.0]);
    }

    #[test]
    fn lab_round_trip_is_close() {
        for rgb in [
            [0.2, 0.5, 0.8],
            [0.9, 0.1, 0.3],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ] {
            let back = lab_to_rgb(rgb_to_lab(rgb));
            for c in 0..3 {
                assert!((back[c] - rgb[c]).abs() < 1e-4, "{rgb:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn correction_keeps_style_luminance_direction() {
        let content = gradient(32, 32);
        // A genuinely darker style: compressed and shifted down in [-1, 1].
        let style: Vec<f32> = gradient(32, 32).iter().map(|v| v * 0.4 - 0.4).collect();
        let out = correct(&content, &style, 32, 32).unwrap();
        let mean = |data: &[f32]| data.iter().sum::<f32>() / data.len() as f32;
        // The corrected image is pulled toward the darker style.
        assert!(mean(&out) < mean(&content) + 1e-3);
        for v in &out {
            assert!(v.is_finite() && (-1.0..=1.0).contains(v));
        }
    }
}
