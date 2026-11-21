//! LTX-family position-dependent rotary tables for joint audio/video tokens.
//! This is a neutral numerical algorithm; token layouts and positions come from
//! `av_latent`, while model dimensions remain family configuration.

use danmo_mlx::{Array, Result, Stream};

#[derive(Clone)]
pub struct RopeTables {
    pub cos: Array,
    pub sin: Array,
}

pub fn split_rope_tables(
    positions: &[Vec<f32>],
    inner_dim: usize,
    num_heads: usize,
    theta: f32,
    max_positions: &[f32],
    stream: &Stream,
) -> Result<RopeTables> {
    let dimensions = positions.first().map_or(0, Vec::len);
    if dimensions == 0
        || max_positions.len() != dimensions
        || inner_dim % num_heads != 0
        || (inner_dim / num_heads) % 2 != 0
    {
        return Err(danmo_mlx::Error::new(
            "LTX A/V RoPE",
            format!(
                "invalid positions={dimensions}, inner_dim={inner_dim}, heads={num_heads}, max_positions={max_positions:?}"
            ),
        ));
    }
    if positions
        .iter()
        .any(|position| position.len() != dimensions)
    {
        return Err(danmo_mlx::Error::new(
            "LTX A/V RoPE",
            "ragged position vectors",
        ));
    }
    let head_half = inner_dim / (2 * num_heads);
    let frequencies = inner_dim / (2 * dimensions);
    let expected = inner_dim / 2;
    let produced = frequencies * dimensions;
    if produced > expected {
        return Err(danmo_mlx::Error::new(
            "LTX A/V RoPE",
            "frequency grid exceeds split head capacity",
        ));
    }
    let padding = expected - produced;
    let mut angles = Vec::with_capacity(positions.len() * expected);
    for position in positions {
        angles.extend(std::iter::repeat(0.0).take(padding));
        for frequency in 0..frequencies {
            let exponent = frequency as f32 / (frequencies.saturating_sub(1).max(1)) as f32;
            let grid = theta.powf(exponent) * std::f32::consts::FRAC_PI_2;
            for axis in 0..dimensions {
                let fraction = position[axis] / max_positions[axis];
                angles.push(grid * (2.0 * fraction - 1.0));
            }
        }
    }
    let shape = [
        1,
        positions.len() as i32,
        num_heads as i32,
        head_half as i32,
    ];
    let angles = Array::from_f32_slice(&angles, &shape)?.transpose_axes(&[0, 2, 1, 3], stream)?;
    Ok(RopeTables {
        cos: angles.cos(stream)?,
        sin: angles.sin(stream)?,
    })
}

#[cfg(test)]
mod tests {
    use super::split_rope_tables;
    use danmo_mlx::Stream;

    #[test]
    fn video_grid_has_left_padding_and_frequency_major_angles() {
        let stream = Stream::cpu().unwrap();
        let table = split_rope_tables(
            &[vec![0.5, 16.0, 32.0]],
            24,
            2,
            10000.0,
            &[20.0, 2048.0, 2048.0],
            &stream,
        )
        .unwrap();
        assert_eq!(table.cos.shape(), &[1, 2, 1, 6]);
        let cos = table.cos.to_vec_f32(&stream).unwrap();
        let sin = table.sin.to_vec_f32(&stream).unwrap();
        // inner_dim/(2*3)=4 frequencies, producing 12 values per token;
        // expected inner_dim/2=12, so there is no padding in this small probe.
        assert_eq!(cos.len(), 12);
        let first = std::f32::consts::FRAC_PI_2 * (2.0 * 0.5 / 20.0 - 1.0);
        assert!((cos[0] - first.cos()).abs() < 1e-6);
        assert!((sin[0] - first.sin()).abs() < 1e-6);
    }
}
