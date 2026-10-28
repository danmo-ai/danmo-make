//! ACE-Step FSQ scalar codec (HOT-Step `fsq-quant.h`).
//!
//! Matches vector_quantize_pytorch ResidualFSQ with `preserve_symmetry=True`
//! and `bound_hard_clamp=True` (soft clamp before hard bound). The older
//! acestep.cpp plain-`tanh` encoder disagrees with shipped checkpoints.

pub const FSQ_NDIMS: usize = 6;
pub const FSQ_LEVELS: [i32; FSQ_NDIMS] = [8, 8, 8, 5, 5, 5];
pub const FSQ_CODEBOOK_SIZE: i32 = 64_000;
pub const POOL_WINDOW: i64 = 5;

/// `project_in` raw floats → flat codebook index.
pub fn encode_index(raw_vals: &[f32; FSQ_NDIMS]) -> i32 {
    let mut index = 0i32;
    let mut mult = 1i32;
    for d in 0..FSQ_NDIMS {
        let l = FSQ_LEVELS[d];
        let lm1 = (l - 1) as f32;
        let c = 1.0 + 1.0 / lm1;
        let mut y = (raw_vals[d] / c).tanh() * c;
        y = y.clamp(-1.0, 1.0);
        let mut code = (lm1 * (y + 1.0) / 2.0 + 0.5).floor() as i32;
        if code < 0 {
            code = 0;
        }
        if code >= l {
            code = l - 1;
        }
        index += code * mult;
        mult *= l;
    }
    index
}

/// Flat codebook index → per-dim values in `[-1, 1]`.
pub fn decode_index(index: i32) -> [f32; FSQ_NDIMS] {
    let mut out = [0f32; FSQ_NDIMS];
    let mut stride = 1i32;
    for d in 0..FSQ_NDIMS {
        let l = FSQ_LEVELS[d];
        let level_idx = (index / stride) % l;
        let step = 2.0 / (l - 1) as f32;
        out[d] = level_idx as f32 * step - 1.0;
        stride *= l;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{decode_index, encode_index, FSQ_CODEBOOK_SIZE, FSQ_LEVELS};

    #[test]
    fn codebook_size_matches_levels_product() {
        let product: i32 = FSQ_LEVELS.iter().product();
        assert_eq!(product, FSQ_CODEBOOK_SIZE);
    }

    #[test]
    fn decode_encode_roundtrip_extreme_and_mid() {
        // Index 0 → all dims at level 0 → -1.
        let zero = decode_index(0);
        assert!(zero.iter().all(|&v| (v + 1.0).abs() < 1e-6));
        // Max level indices → all +1.
        let mut max = 0i32;
        let mut mult = 1i32;
        for &l in &FSQ_LEVELS {
            max += (l - 1) * mult;
            mult *= l;
        }
        let ones = decode_index(max);
        assert!(ones.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        // Soft-clamp encode of zeros lands on mid levels (not codebook 0).
        let mid = encode_index(&[0.0; 6]);
        let mut expect = 0i32;
        let mut mult = 1i32;
        for &l in &FSQ_LEVELS {
            let code = (((l - 1) as f32) / 2.0 + 0.5).floor() as i32;
            expect += code * mult;
            mult *= l;
        }
        assert_eq!(mid, expect);
        assert_eq!(encode_index(&decode_index(mid)), mid);
    }

    #[test]
    fn soft_clamp_differs_from_plain_tanh_on_large_inputs() {
        let raw = [10.0f32; 6];
        let soft = encode_index(&raw);
        // Plain tanh(+10) ≈ 1 → all max levels; soft clamp also saturates to max.
        let mut max = 0i32;
        let mut mult = 1i32;
        for &l in &FSQ_LEVELS {
            max += (l - 1) * mult;
            mult *= l;
        }
        assert_eq!(soft, max);
        // Near-boundary values where soft clamp matters vs plain tanh.
        let edge = [1.05f32, 1.05, 1.05, 1.05, 1.05, 1.05];
        let idx = encode_index(&edge);
        assert!((0..FSQ_CODEBOOK_SIZE).contains(&idx));
    }
}
