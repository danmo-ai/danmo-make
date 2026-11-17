use danmo_mlx::{AffineQuantization, Array, Dtype, LinearWeight, Result, Stream};

const OUT: i32 = 48;
const IN: i32 = 128;
const ROWS: i32 = 5;

fn values(count: i32, modulus: i32, scale: f32, offset: f32) -> Vec<f32> {
    (0..count)
        .map(|index| (index % modulus) as f32 * scale - offset)
        .collect()
}

fn max_error(lhs: &[f32], rhs: &[f32]) -> f32 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}

fn affine_probe(dtype: Dtype, bits: i32, group_size: i32) -> Result<()> {
    let stream = Stream::cpu()?;
    let dense =
        Array::from_f32_slice(&values(OUT * IN, 29, 0.013, 0.17), &[OUT, IN])?.astype(dtype, &stream)?;
    let input =
        Array::from_f32_slice(&values(ROWS * IN, 23, 0.021, 0.2), &[ROWS, IN])?.astype(dtype, &stream)?;
    let bias = Array::from_f32_slice(&values(OUT, 7, 0.05, 0.1), &[OUT])?.astype(dtype, &stream)?;
    let packed = dense.quantize_affine(group_size, bits, &stream)?;
    let reference = LinearWeight::Dense(packed.dequantize(group_size, bits, &stream)?);
    let quantized =
        LinearWeight::affine(packed, AffineQuantization::new(bits, group_size)?)?;
    assert_eq!(quantized.compute_dtype(), dtype);
    assert!(quantized.dense().is_none());

    let actual = quantized
        .apply(&input, Some(&bias), &stream)?
        .to_vec_f32(&stream)?;
    let expected = reference
        .apply(&input, Some(&bias), &stream)?
        .to_vec_f32(&stream)?;
    assert_eq!(actual.len(), (ROWS * OUT) as usize);
    let tolerance = if dtype == Dtype::Float32 { 1e-4 } else { 0.1 };
    let error = max_error(&actual, &expected);
    assert!(
        error <= tolerance,
        "{bits}-bit group {group_size} {dtype:?}: in-kernel dequant differs from dense by {error}"
    );

    let dense_bytes = (OUT * IN) as usize * if dtype == Dtype::Float32 { 4 } else { 2 };
    let scale_bytes = if dtype == Dtype::Float32 { 4 } else { 2 };
    let expected_bytes = (OUT * IN * bits / 8) as usize
        + 2 * (OUT * IN / group_size) as usize * scale_bytes;
    assert_eq!(quantized.nbytes(), expected_bytes);
    assert!(quantized.nbytes() < dense_bytes);
    Ok(())
}

#[test]
fn affine_linear_matches_dequantized_dense_reference() {
    for dtype in [Dtype::Float32, Dtype::BFloat16, Dtype::Float16] {
        for (bits, group_size) in [(4, 64), (8, 64), (4, 32), (4, 128)] {
            affine_probe(dtype, bits, group_size).unwrap();
        }
    }
}

#[test]
fn affine_weight_rejects_mismatched_quantization() {
    let stream = Stream::cpu().unwrap();
    let dense = Array::from_f32_slice(&values(OUT * IN, 29, 0.013, 0.17), &[OUT, IN]).unwrap();
    let packed = dense.quantize_affine(64, 4, &stream).unwrap();
    let wrong_bits = LinearWeight::affine(packed.clone(), AffineQuantization::new(8, 64).unwrap());
    assert!(wrong_bits.is_err());
    let wrong_group = LinearWeight::affine(packed, AffineQuantization::new(4, 32).unwrap());
    assert!(wrong_group.is_err());
    assert!(AffineQuantization::new(3, 64).is_err());
    assert!(AffineQuantization::new(4, 16).is_err());
}

#[test]
fn affine_weight_rejects_dense_tensor_as_packed() {
    let stream = Stream::cpu().unwrap();
    let dense = Array::from_f32_slice(&values(OUT * IN, 29, 0.013, 0.17), &[OUT, IN]).unwrap();
    let mut packed = dense.quantize_affine(64, 4, &stream).unwrap();
    packed.weight = dense;
    assert!(LinearWeight::affine(packed, AffineQuantization::new(4, 64).unwrap()).is_err());
}
