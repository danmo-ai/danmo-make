//! SeedVR2 output color correction: wavelet reconstruction against the
//! preprocessed source, then CIE LAB histogram matching on a/b with a
//! luminance blend. Port of `preprocess_mlx.SeedVR2Util` (5 level wavelet,
//! 3x3 binomial kernel at stride `2^level`, stable-rank histogram match).

use crate::weights::WeightError;

#[cfg(feature = "mlx")]
use danmo_mlx::{Array, Dtype, Result as MlxResult, Stream};

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

#[cfg(feature = "mlx")]
pub fn correct_mlx(
    content: &[f32],
    style: &[f32],
    w: usize,
    h: usize,
    stream: &Stream,
) -> MlxResult<Vec<f32>> {
    // Host planes are CHW `[c][y][x]`; conv / LAB want NHWC.
    let content = planar_to_nhwc(content, w, h, stream)?;
    let style = planar_to_nhwc(style, w, h, stream)?;
    let (content_high, _) = decompose_mlx(&content, stream)?;
    let (_, style_low) = decompose_mlx(&style, stream)?;
    let merged = Array::add(&content_high, &style_low, stream)?.clip(-1.0, 1.0, stream)?;
    // Match CPU pixel01: ((x + 1) * 0.5).clamp(0, 1). Without the 0.5, mid/high
    // tones clip to 1.0 and the LAB path blows out (severe overexposure).
    let content01 = Array::multiply(
        &Array::add(&merged, &scalar_mlx(1.0)?, stream)?,
        &scalar_mlx(0.5)?,
        stream,
    )?
    .clip(0.0, 1.0, stream)?;
    let style01 = Array::multiply(
        &Array::add(&style, &scalar_mlx(1.0)?, stream)?,
        &scalar_mlx(0.5)?,
        stream,
    )?
    .clip(0.0, 1.0, stream)?;
    let content_lab = rgb_to_lab_mlx(&content01, stream)?;
    let style_lab = rgb_to_lab_mlx(&style01, stream)?;
    let matched_a = hist_match_mlx(
        channel(&content_lab, 1, stream)?,
        channel(&style_lab, 1, stream)?,
        stream,
    )?;
    let matched_b = hist_match_mlx(
        channel(&content_lab, 2, stream)?,
        channel(&style_lab, 2, stream)?,
        stream,
    )?;
    let matched_l = hist_match_mlx(
        channel(&content_lab, 0, stream)?,
        channel(&style_lab, 0, stream)?,
        stream,
    )?;
    let luminance = Array::add(
        &Array::multiply(
            &channel(&content_lab, 0, stream)?,
            &scalar_mlx(LUMINANCE_WEIGHT)?,
            stream,
        )?,
        &Array::multiply(&matched_l, &scalar_mlx(1.0 - LUMINANCE_WEIGHT)?, stream)?,
        stream,
    )?;
    let rgb = lab_to_rgb_mlx(&luminance, &matched_a, &matched_b, stream)?.clip(0.0, 1.0, stream)?;
    let output = Array::subtract(
        &Array::multiply(&rgb, &scalar_mlx(2.0)?, stream)?,
        &scalar_mlx(1.0)?,
        stream,
    )?;
    // NHWC → CHW planar for the host consumer.
    output
        .transpose_axes(&[0, 3, 1, 2], stream)?
        .to_vec_f32(stream)
}

#[cfg(feature = "mlx")]
fn planar_to_nhwc(planes: &[f32], w: usize, h: usize, stream: &Stream) -> MlxResult<Array> {
    Array::from_f32_slice(planes, &[1, 3, h as i32, w as i32])?
        .transpose_axes(&[0, 2, 3, 1], stream)
}

#[cfg(feature = "mlx")]
fn scalar_mlx(value: f32) -> MlxResult<Array> {
    Array::scalar_f32(value)
}

#[cfg(feature = "mlx")]
fn channel(input: &Array, index: i32, stream: &Stream) -> MlxResult<Array> {
    let shape = input.shape();
    input.slice(
        &[0, 0, 0, index],
        &[shape[0], shape[1], shape[2], index + 1],
        &[1, 1, 1, 1],
        stream,
    )
}

#[cfg(feature = "mlx")]
fn blur_mlx(input: &Array, radius: usize, stream: &Stream) -> MlxResult<Array> {
    let shape = input.shape();
    // Match the CPU blur: 3x3 taps at stride `radius` with edge clamp.
    // Edge-pad by `radius`, then a dilated 3x3 (dilation=radius, pad=0) keeps
    // spatial size identical. A non-dilated 3x3 on the same pad grows by
    // 2*(radius-1) and breaks wavelet subtract (e.g. 5120x3072 → 5122x3074).
    let radius = radius
        .max(1)
        .min((shape[1].min(shape[2]) as usize / 8).max(1)) as i32;
    let padded = input.pad(
        &[1, 2],
        &[radius, radius],
        &[radius, radius],
        0.0,
        "edge",
        stream,
    )?;
    let mut values = vec![0.0f32; 3 * 27];
    for c in 0..3 {
        for (ky, _dy) in (-1i64..=1).enumerate() {
            for (kx, _dx) in (-1i64..=1).enumerate() {
                values[c * 27 + ky * 9 + kx * 3 + c] = KERNEL[ky][kx];
            }
        }
    }
    let weight = Array::from_f32_slice(&values, &[3, 3, 3, 3])?;
    Array::conv2d(
        &padded,
        &weight,
        [1, 1],
        [0, 0],
        [radius, radius],
        1,
        stream,
    )
}

#[cfg(feature = "mlx")]
fn decompose_mlx(input: &Array, stream: &Stream) -> MlxResult<(Array, Array)> {
    let mut high = input.zeros_like(stream)?;
    let mut current = input.clone();
    for level in 0..LEVELS {
        let low = blur_mlx(&current, 1 << level, stream)?;
        high = Array::add(&high, &Array::subtract(&current, &low, stream)?, stream)?;
        current = low;
    }
    Ok((high, current))
}

#[cfg(feature = "mlx")]
fn hist_match_mlx(source: Array, reference: Array, stream: &Stream) -> MlxResult<Array> {
    // CPU: sort reference by value, map source ranks onto those values.
    let shape = source.shape();
    let n = shape.iter().map(|d| *d as i64).product::<i64>() as i32;
    let source = source.reshape(&[n], stream)?;
    let reference = reference.reshape(&[n], stream)?;
    let source_order = source.argsort(stream)?.astype(Dtype::Int32, stream)?;
    let inverse = source_order.argsort(stream)?.astype(Dtype::Int32, stream)?;
    let reference_order = reference.argsort(stream)?.astype(Dtype::Int32, stream)?;
    let reference_sorted = reference.take_axis(&reference_order, 0, stream)?;
    reference_sorted
        .take_axis(&inverse, 0, stream)?
        .reshape(&shape, stream)
}

/// Right-multiply matrix for NHWC `[..., 3] @ M`. CPU does `M @ rgb`, so store `Mᵀ`.
#[cfg(feature = "mlx")]
fn color_matrix(rows: &[[f32; 3]; 3], stream: &Stream) -> MlxResult<Array> {
    let mut transposed = [0f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            transposed[i * 3 + j] = rows[j][i];
        }
    }
    Array::from_f32_slice(&transposed, &[3, 3])?.astype(Dtype::Float32, stream)
}

#[cfg(feature = "mlx")]
fn srgb_to_linear_mlx(input: &Array, stream: &Stream) -> MlxResult<Array> {
    let shifted = Array::divide(
        &Array::add(input, &scalar_mlx(0.055)?, stream)?,
        &scalar_mlx(1.055)?,
        stream,
    )?;
    let power = Array::power(&shifted, &scalar_mlx(2.4)?, stream)?;
    let linear = Array::divide(input, &scalar_mlx(12.92)?, stream)?;
    let condition = Array::greater(input, &scalar_mlx(0.04045)?, stream)?;
    Array::select(&condition, &power, &linear, stream)
}

#[cfg(feature = "mlx")]
fn linear_to_srgb_mlx(input: &Array, stream: &Stream) -> MlxResult<Array> {
    let positive = Array::maximum(input, &Array::scalar_f32(0.0)?, stream)?;
    let power = Array::subtract(
        &Array::multiply(
            &Array::power(&positive, &scalar_mlx(1.0 / 2.4)?, stream)?,
            &scalar_mlx(1.055)?,
            stream,
        )?,
        &scalar_mlx(0.055)?,
        stream,
    )?;
    let linear = Array::multiply(input, &scalar_mlx(12.92)?, stream)?;
    let condition = Array::greater(input, &scalar_mlx(0.003_130_8)?, stream)?;
    Array::select(&condition, &power, &linear, stream)
}

#[cfg(feature = "mlx")]
fn rgb_to_lab_mlx(rgb: &Array, stream: &Stream) -> MlxResult<Array> {
    let linear = srgb_to_linear_mlx(rgb, stream)?;
    let xyz = Array::matmul(&linear, &color_matrix(&RGB_TO_XYZ, stream)?, stream)?;
    let xyz = Array::divide(
        &xyz,
        &Array::from_f32_slice(&[0.95047, 1.0, 1.08883], &[1, 3])?,
        stream,
    )?;
    let eps3 = (6.0f32 / 29.0).powi(3);
    let kappa = (29.0f32 / 3.0).powi(3);
    let cubic = Array::power(&xyz, &scalar_mlx(1.0 / 3.0)?, stream)?;
    let linear = Array::divide(
        &Array::add(
            &Array::multiply(&xyz, &scalar_mlx(kappa)?, stream)?,
            &scalar_mlx(16.0)?,
            stream,
        )?,
        &scalar_mlx(116.0)?,
        stream,
    )?;
    let condition = Array::greater(&xyz, &scalar_mlx(eps3)?, stream)?;
    let f = Array::select(&condition, &cubic, &linear, stream)?;
    let fx = channel(&f, 0, stream)?;
    let fy = channel(&f, 1, stream)?;
    let fz = channel(&f, 2, stream)?;
    let l = Array::subtract(
        &Array::multiply(&fy, &scalar_mlx(116.0)?, stream)?,
        &scalar_mlx(16.0)?,
        stream,
    )?;
    let a = Array::multiply(
        &Array::subtract(&fx, &fy, stream)?,
        &scalar_mlx(500.0)?,
        stream,
    )?;
    let b = Array::multiply(
        &Array::subtract(&fy, &fz, stream)?,
        &scalar_mlx(200.0)?,
        stream,
    )?;
    // channel() keeps a trailing size-1 axis; concat on the last axis → NHWC.
    Array::concatenate_axis(&[l, a, b], 3, stream)
}

#[cfg(feature = "mlx")]
fn lab_to_rgb_mlx(l: &Array, a: &Array, b: &Array, stream: &Stream) -> MlxResult<Array> {
    let fy = Array::divide(
        &Array::add(l, &scalar_mlx(16.0)?, stream)?,
        &scalar_mlx(116.0)?,
        stream,
    )?;
    let fx = Array::add(&Array::divide(a, &scalar_mlx(500.0)?, stream)?, &fy, stream)?;
    let fz = Array::subtract(&fy, &Array::divide(b, &scalar_mlx(200.0)?, stream)?, stream)?;
    let eps = 6.0f32 / 29.0;
    let kappa = (29.0f32 / 3.0).powi(3);
    let inverse = |value: &Array| -> MlxResult<Array> {
        let cubic = Array::power(value, &scalar_mlx(3.0)?, stream)?;
        let linear = Array::divide(
            &Array::subtract(
                &Array::multiply(value, &scalar_mlx(116.0)?, stream)?,
                &scalar_mlx(16.0)?,
                stream,
            )?,
            &scalar_mlx(kappa)?,
            stream,
        )?;
        let condition = Array::greater(value, &scalar_mlx(eps)?, stream)?;
        Array::select(&condition, &cubic, &linear, stream)
    };
    let x = Array::multiply(&inverse(&fx)?, &scalar_mlx(0.95047)?, stream)?;
    let y = inverse(&fy)?;
    let z = Array::multiply(&inverse(&fz)?, &scalar_mlx(1.08883)?, stream)?;
    let xyz = Array::concatenate_axis(&[x, y, z], 3, stream)?;
    let rgb = Array::matmul(&xyz, &color_matrix(&XYZ_TO_RGB, stream)?, stream)?;
    linear_to_srgb_mlx(&rgb, stream)
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
    #[cfg(feature = "mlx")]
    use danmo_mlx::Array;

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

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_blur_keeps_spatial_shape_for_dilated_radii() {
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 blur shape", |_, stream, _, _| {
                let w = 48i32;
                let h = 64i32;
                let zeros = vec![0.0f32; (h * w * 3) as usize];
                // NHWC like correct_mlx.
                let input = Array::from_f32_slice(&zeros, &[1, h, w, 3]).unwrap();
                for radius in [1usize, 2, 4, 8, 16] {
                    let out = blur_mlx(&input, radius, stream).unwrap();
                    assert_eq!(
                        out.shape(),
                        input.shape(),
                        "radius {radius} changed spatial shape"
                    );
                }
                // Production 5120x3072 is too large here; 64x48 still hits radius=16.
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_color_correct_accepts_tall_aspect() {
        // Repro the broadcast failure path: tall frame + 5 wavelet levels.
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 color shape", |_, stream, _, _| {
                let (w, h) = (48usize, 80usize);
                let content = gradient(w, h);
                let style: Vec<f32> = content.iter().map(|v| v * 0.5).collect();
                let out = correct_mlx(&content, &style, w, h, stream).unwrap();
                assert_eq!(out.len(), w * h * 3);
                assert!(out.iter().all(|v| v.is_finite()));
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_wavelet_merge_matches_cpu() {
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 wavelet parity", |_, stream, _, _| {
                let (w, h) = (32usize, 24usize);
                let content = gradient(w, h);
                let style: Vec<f32> = content.iter().map(|v| v * 0.4 - 0.3).collect();
                let (ch, _) = decompose(&content, w, h);
                let (_, sl) = decompose(&style, w, h);
                let mut cpu = vec![0f32; content.len()];
                for i in 0..cpu.len() {
                    cpu[i] = (ch[i] + sl[i]).clamp(-1.0, 1.0);
                }
                let content_a = planar_to_nhwc(&content, w, h, stream).unwrap();
                let style_a = planar_to_nhwc(&style, w, h, stream).unwrap();
                let (ch_m, _) = decompose_mlx(&content_a, stream).unwrap();
                let (_, sl_m) = decompose_mlx(&style_a, stream).unwrap();
                let merged = Array::add(&ch_m, &sl_m, stream)
                    .unwrap()
                    .clip(-1.0, 1.0, stream)
                    .unwrap()
                    .transpose_axes(&[0, 3, 1, 2], stream)
                    .unwrap()
                    .to_vec_f32(stream)
                    .unwrap();
                let mut max_abs = 0.0f32;
                for (a, b) in cpu.iter().zip(&merged) {
                    max_abs = max_abs.max((a - b).abs());
                }
                eprintln!("wavelet merge max_abs={max_abs:.6}");
                assert!(max_abs < 1e-4, "wavelet merge max_abs {max_abs}");
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_lab_roundtrip_matches_cpu() {
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 lab roundtrip", |_, stream, _, _| {
                let (w, h) = (16usize, 12usize);
                let rgb01: Vec<f32> = gradient(w, h)
                    .into_iter()
                    .map(|v| ((v + 1.0) * 0.5).clamp(0.0, 1.0))
                    .collect();
                let mut cpu = vec![0f32; rgb01.len()];
                let plane = w * h;
                for px in 0..plane {
                    let mut rgb = [0f32; 3];
                    for c in 0..3 {
                        rgb[c] = rgb01[px + c * plane];
                    }
                    let back = lab_to_rgb(rgb_to_lab(rgb));
                    for c in 0..3 {
                        cpu[px + c * plane] = back[c];
                    }
                }
                let rgb_a = planar_to_nhwc(&rgb01, w, h, stream).unwrap();
                let lab = rgb_to_lab_mlx(&rgb_a, stream).unwrap();
                let l = channel(&lab, 0, stream).unwrap();
                let a = channel(&lab, 1, stream).unwrap();
                let b = channel(&lab, 2, stream).unwrap();
                let back = lab_to_rgb_mlx(&l, &a, &b, stream)
                    .unwrap()
                    .transpose_axes(&[0, 3, 1, 2], stream)
                    .unwrap()
                    .to_vec_f32(stream)
                    .unwrap();
                let mut max_abs = 0.0f32;
                for (x, y) in cpu.iter().zip(&back) {
                    max_abs = max_abs.max((x - y).abs());
                }
                eprintln!("lab roundtrip max_abs={max_abs:.6}");
                assert!(max_abs < 2e-3, "lab roundtrip max_abs {max_abs}");
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_hist_match_matches_cpu() {
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 hist parity", |_, stream, _, _| {
                let source = vec![0.5f32, -1.0, 2.0, 0.0, 0.25, -0.5, 1.5, 0.1];
                let reference = vec![10.0f32, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 5.0];
                let cpu = hist_match(&source, &reference);
                let src = Array::from_f32_slice(&source, &[1, 2, 4, 1]).unwrap();
                let refer = Array::from_f32_slice(&reference, &[1, 2, 4, 1]).unwrap();
                let mlx = hist_match_mlx(src, refer, stream)
                    .unwrap()
                    .to_vec_f32(stream)
                    .unwrap();
                let mut max_abs = 0.0f32;
                for (a, b) in cpu.iter().zip(&mlx) {
                    max_abs = max_abs.max((a - b).abs());
                }
                eprintln!("hist_match max_abs={max_abs:.6} cpu={cpu:?} mlx={mlx:?}");
                assert!(max_abs < 1e-5, "hist_match max_abs {max_abs}");
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_color_correct_matches_cpu() {
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 color parity", |_, stream, _, _| {
                let (w, h) = (32usize, 24usize);
                let content = gradient(w, h);
                let style: Vec<f32> = content.iter().map(|v| v * 0.4 - 0.3).collect();
                let cpu = correct(&content, &style, w, h).unwrap();
                let mlx = correct_mlx(&content, &style, w, h, stream).unwrap();
                assert_eq!(cpu.len(), mlx.len());
                let mut max_abs = 0.0f32;
                let mut mse = 0.0f64;
                for (a, b) in cpu.iter().zip(&mlx) {
                    let d = (a - b).abs();
                    max_abs = max_abs.max(d);
                    mse += f64::from(d * d);
                }
                mse /= cpu.len() as f64;
                eprintln!("seedvr2 color cpu↔mlx mse={mse:.8} max_abs={max_abs:.6}");
                assert!(
                    max_abs < 0.05,
                    "cpu↔mlx max_abs {max_abs} too large (mse={mse})"
                );
                Ok(())
            })
            .unwrap();
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn mlx_blur_matches_cpu_on_gradient() {
        let executor = crate::runtime::mlx_executor().unwrap();
        executor
            .execute("seedvr2 blur parity", |_, stream, _, _| {
                let (w, h) = (32usize, 24usize);
                let img = gradient(w, h);
                for radius in [1usize, 2, 4] {
                    let cpu = blur(&img, w, h, radius);
                    let nhwc = planar_to_nhwc(&img, w, h, stream).unwrap();
                    let mlx = blur_mlx(&nhwc, radius, stream)
                        .unwrap()
                        .transpose_axes(&[0, 3, 1, 2], stream)
                        .unwrap()
                        .to_vec_f32(stream)
                        .unwrap();
                    let mut max_abs = 0.0f32;
                    for (a, b) in cpu.iter().zip(&mlx) {
                        max_abs = max_abs.max((a - b).abs());
                    }
                    eprintln!("blur radius={radius} max_abs={max_abs:.6}");
                    assert!(max_abs < 1e-4, "blur radius {radius} max_abs {max_abs}");
                }
                Ok(())
            })
            .unwrap();
    }
}
