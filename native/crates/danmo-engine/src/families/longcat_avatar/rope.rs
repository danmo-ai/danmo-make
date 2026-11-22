//! LongCat 3D RoPE for video DiT tokens (Avatar variant semantics; the
//! reference-insertion branches that adjust the temporal grid are out of
//! scope for single-talker ATI2V and are not encoded).
//!
//! Head dim splits as `dim_t = head_dim - 4*(head_dim/6)`,
//! `dim_h = dim_w = 2*(head_dim/6)` (128 → 44/42/42). Rotation pairs adjacent
//! channels (`rotate_half` over pairs), computed in F32 and cast back.

use danmo_mlx::{Array, Error, Result, Stream};

/// Precomputed cosine/sine tables `[1, 1, T*H*W, head_dim]` in F32.
pub struct RopeTables {
    pub cos: Array,
    pub sin: Array,
    pub head_dim: usize,
}

impl RopeTables {
    pub fn build(
        frames: usize,
        height: usize,
        width: usize,
        head_dim: usize,
        stream: &Stream,
    ) -> Result<Self> {
        if head_dim == 0 || head_dim % 8 != 0 {
            return Err(Error::new(
                "longcat_avatar RoPE",
                format!("head_dim {head_dim} must be a positive multiple of 8"),
            ));
        }
        let dim_t = head_dim - 4 * (head_dim / 6);
        let dim_h = 2 * (head_dim / 6);
        let dim_w = 2 * (head_dim / 6);
        let base = 10_000f32;
        let axis_freqs = |dim: usize, length: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(length * dim);
            for position in 0..length {
                for index in 0..dim / 2 {
                    let exponent = (2 * index) as f32 / dim as f32;
                    let freq = 1.0 / base.powf(exponent);
                    let angle = position as f32 * freq;
                    out.push(angle);
                    out.push(angle);
                }
            }
            out
        };
        let freqs_t = axis_freqs(dim_t, frames);
        let freqs_h = axis_freqs(dim_h, height);
        let freqs_w = axis_freqs(dim_w, width);
        let mut cos = Vec::with_capacity(frames * height * width * head_dim);
        let mut sin = Vec::with_capacity(cos.capacity());
        for t in 0..frames {
            for h in 0..height {
                for w in 0..width {
                    let t_row = &freqs_t[t * dim_t..(t + 1) * dim_t];
                    let h_row = &freqs_h[h * dim_h..(h + 1) * dim_h];
                    let w_row = &freqs_w[w * dim_w..(w + 1) * dim_w];
                    for angle in t_row.iter().chain(h_row).chain(w_row) {
                        cos.push(angle.cos());
                        sin.push(angle.sin());
                    }
                }
            }
        }
        let sequence = (frames * height * width) as i32;
        let shape = [1, 1, sequence, head_dim as i32];
        Ok(Self {
            cos: Array::from_f32_slice(&cos, &shape)?,
            sin: Array::from_f32_slice(&sin, &shape)?,
            head_dim,
        })
    }

    /// Rotate `q`/`k` of shape `[B, heads, T*H*W, head_dim]` in place-value.
    pub fn apply(&self, q: &Array, k: &Array, stream: &Stream) -> Result<(Array, Array)> {
        let rotate = |x: &Array| -> Result<Array> {
            let dtype = x.dtype();
            let xf = x.astype(danmo_mlx::Dtype::Float32, stream)?;
            let rotated = rotate_half(&xf, stream)?;
            let out = Array::add(
                &Array::multiply(&xf, &self.cos, stream)?,
                &Array::multiply(&rotated, &self.sin, stream)?,
                stream,
            )?;
            out.astype(dtype, stream)
        };
        Ok((rotate(q)?, rotate(k)?))
    }
}

/// Pair-wise `[-x2, x1]` rotation over the last axis.
fn rotate_half(x: &Array, stream: &Stream) -> Result<Array> {
    let shape = x.shape();
    let last = *shape.last().unwrap_or(&0);
    if last <= 0 || last % 2 != 0 {
        return Err(Error::new(
            "longcat_avatar RoPE",
            format!("last dimension must be positive and even, got {shape:?}"),
        ));
    }
    let mut pair_shape = shape.to_vec();
    *pair_shape.last_mut().unwrap() = last / 2;
    pair_shape.push(2);
    let pairs = x.reshape(&pair_shape, stream)?;
    let rank = pair_shape.len();
    let zeros = vec![0; rank];
    let mut stop = pair_shape.clone();
    stop[rank - 1] = 1;
    let ones = vec![1; rank];
    let x1 = pairs
        .slice(&zeros, &stop, &ones, stream)?
        .squeeze_axis(-1, stream)?;
    let mut start2 = zeros.clone();
    start2[rank - 1] = 1;
    let mut stop2 = pair_shape.clone();
    stop2[rank - 1] = 2;
    let x2 = pairs
        .slice(&start2, &stop2, &ones, stream)?
        .squeeze_axis(-1, stream)?;
    let stacked = Array::stack(&[x2.negative(stream)?, x1], -1, stream)?;
    stacked.reshape(&shape, stream)
}

#[cfg(test)]
mod tests {
    use super::RopeTables;

    #[test]
    fn tables_cover_the_whole_grid() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let tables = RopeTables::build(3, 4, 5, 128, &stream).unwrap();
        assert_eq!(tables.cos.shape(), [1, 1, 60, 128]);
    }
}
