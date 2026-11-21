//! Neutral MLX tensor algorithms shared by model families.
//! Model dimensions, masks, and position policies stay with the caller.

use danmo_mlx::{Array, Dtype, Result, Stream};

/// Exact `mlx.nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2`.
pub fn gelu(input: &Array, stream: &Stream) -> Result<Array> {
    let scaled = Array::multiply(input, &scalar(1.0 / std::f32::consts::SQRT_2)?, stream)?;
    let cdf = Array::add(&scalar(1.0)?, &scaled.erf(stream)?, stream)?;
    let half = Array::multiply(input, &scalar(0.5)?, stream)?;
    Array::multiply(&half, &cdf, stream)
}

pub fn gelu_tanh(input: &Array, stream: &Stream) -> Result<Array> {
    let square = Array::multiply(input, input, stream)?;
    let cube = Array::multiply(&square, input, stream)?;
    let correction = Array::multiply(&cube, &scalar(0.044715)?, stream)?;
    let inner = Array::add(input, &correction, stream)?;
    let scaled = Array::multiply(
        &inner,
        &scalar((2.0 / std::f32::consts::PI).sqrt())?,
        stream,
    )?;
    let one_plus = Array::add(&scalar(1.0)?, &scaled.tanh(stream)?, stream)?;
    let half = Array::multiply(input, &scalar(0.5)?, stream)?;
    Array::multiply(&half, &one_plus, stream)
}

/// Split-half rotary update for `[..., N, D]`; tables end in `D/2`.
pub fn apply_split_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let dim = *shape.last().unwrap_or(&0);
    if dim <= 0 || dim % 2 != 0 {
        return Err(danmo_mlx::Error::new(
            "MLX split RoPE",
            format!("last dimension must be positive and even, got {shape:?}"),
        ));
    }
    let half = dim / 2;
    let first = slice_last(input, 0, half, stream)?;
    let second = slice_last(input, half, dim, stream)?;
    let rotated_first = Array::subtract(
        &Array::multiply(&first, cos, stream)?,
        &Array::multiply(&second, sin, stream)?,
        stream,
    )?;
    let rotated_second = Array::add(
        &Array::multiply(&first, sin, stream)?,
        &Array::multiply(&second, cos, stream)?,
        stream,
    )?;
    Array::concatenate_axis(&[rotated_first, rotated_second], -1, stream)
}

pub fn slice_last(input: &Array, start_at: i32, stop_at: i32, stream: &Stream) -> Result<Array> {
    let mut start = vec![0; input.shape().len()];
    let mut stop = input.shape();
    let strides = vec![1; input.shape().len()];
    let axis = start.len() - 1;
    start[axis] = start_at;
    stop[axis] = stop_at;
    input.slice(&start, &stop, &strides, stream)
}

/// `[1, 1, N, D/2]` tables. Only `rotated_dims` receive a nonzero angle;
/// the remaining channels rotate by zero, matching proportional partial RoPE.
pub fn rope_table_1d(
    sequence: usize,
    head_dim: usize,
    rotated_dims: usize,
    base: f32,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if head_dim == 0 || head_dim % 2 != 0 || rotated_dims > head_dim || rotated_dims % 2 != 0 {
        return Err(danmo_mlx::Error::new(
            "MLX RoPE table",
            format!("invalid head_dim={head_dim}, rotated_dims={rotated_dims}"),
        ));
    }
    let mut cos = Vec::with_capacity(sequence * head_dim / 2);
    let mut sin = Vec::with_capacity(cos.capacity());
    for position in 0..sequence {
        for index in 0..head_dim / 2 {
            let angle = if index < rotated_dims / 2 {
                position as f32 * base.powf(-((2 * index) as f32) / head_dim as f32)
            } else {
                0.0
            };
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let shape = [1, 1, sequence as i32, (head_dim / 2) as i32];
    let cos = Array::from_f32_slice(&cos, &shape)?.astype(Dtype::Float32, stream)?;
    let sin = Array::from_f32_slice(&sin, &shape)?.astype(Dtype::Float32, stream)?;
    Ok((cos, sin))
}

pub fn timestep_embedding(timesteps: &Array, dim: usize, max_period: f32) -> Result<Array> {
    if dim == 0 || dim % 2 != 0 {
        return Err(danmo_mlx::Error::new(
            "MLX timestep embedding",
            format!("embedding dimension must be positive and even, got {dim}"),
        ));
    }
    let times = timesteps.to_vec_f32(&Stream::cpu()?)?;
    let half = dim / 2;
    let scale = max_period.ln() / half as f32;
    let mut values = Vec::with_capacity(times.len() * dim);
    for time in &times {
        for index in 0..half {
            values.push((*time * (-scale * index as f32).exp()).cos());
        }
        for index in 0..half {
            values.push((*time * (-scale * index as f32).exp()).sin());
        }
    }
    Array::from_f32_slice(&values, &[times.len() as i32, dim as i32])
}

pub fn scalar(value: f32) -> Result<Array> {
    Array::from_f32_slice(&[value], &[])
}

#[cfg(test)]
mod tests {
    use super::{apply_split_rope, gelu_tanh, rope_table_1d};
    use danmo_mlx::{Array, Stream};

    #[test]
    fn split_rope_and_gelu_match_host_math() {
        let stream = Stream::cpu().unwrap();
        let input = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
        let (cos, sin) = rope_table_1d(1, 4, 2, 10.0, &stream).unwrap();
        // position zero leaves the input unchanged, including the inactive pair.
        let rotated = apply_split_rope(&input, &cos, &sin, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        assert_eq!(rotated, vec![1.0, 2.0, 3.0, 4.0]);
        let gelu = gelu_tanh(&input, &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        for (got, x) in gelu.iter().zip([1.0f32, 2.0, 3.0, 4.0]) {
            let want = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x * x * x)).tanh());
            assert!((got - want).abs() < 2e-5, "{got} != {want}");
        }
    }
}
