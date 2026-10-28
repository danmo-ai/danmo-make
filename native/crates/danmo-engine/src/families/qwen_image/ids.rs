//! Position ids for one Qwen-Image frame.
//!
//! `origin/main` `QwenEmbedRopeMLX` copies diffusers `QwenEmbedRope(scale_rope=True)`.
//! Spatial axis `n` is `[-left, ..., -1, 0, 1, ..., right)` with `right = n / 2`
//! and `left = n - right`. The frame axis of the first image is 0. Text ids
//! are `start + j` on all three axes, and `start = max(height / 2, width / 2)`.
//!
//! SD.cpp `gen_qwen_image_ids` uses `linspace(-n/2, -n/2 + n - 1, n)` and
//! `txt_id_start = max(h_len, w_len)`. Even grids match. Odd grids and the
//! text start do not. The weights were trained with the diffusers formula.

use crate::runtime::RuntimeError;

/// Text positions, then image positions. Attention concatenates in that order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeIds {
    pub text: Vec<[i32; 3]>,
    pub image: Vec<[i32; 3]>,
}

impl RopeIds {
    pub fn attention_order(&self) -> Vec<[i32; 3]> {
        self.text
            .iter()
            .copied()
            .chain(self.image.iter().copied())
            .collect()
    }
}

pub fn position_ids(grid_h: i64, grid_w: i64, seq_txt: i64) -> Result<RopeIds, RuntimeError> {
    if grid_h < 1 || grid_w < 1 || seq_txt < 1 {
        return Err(RuntimeError::Message(
            "qwen-image rope grid or text length is empty".into(),
        ));
    }
    let height = spatial_axis(grid_h);
    let width = spatial_axis(grid_w);
    let start = (grid_h / 2).max(grid_w / 2) as i32;
    let text = (0..seq_txt)
        .map(|j| {
            let id = start + j as i32;
            [id, id, id]
        })
        .collect();
    let mut image = Vec::with_capacity((grid_h * grid_w) as usize);
    for row in 0..grid_h {
        for col in 0..grid_w {
            image.push([0, height[row as usize], width[col as usize]]);
        }
    }
    Ok(RopeIds { text, image })
}

/// Centered positions. `n = 4` is `[-2, -1, 0, 1]`. `n = 3` is `[-2, -1, 0]`.
pub fn spatial_axis(n: i64) -> Vec<i32> {
    let right = n / 2;
    let left = n - right;
    let mut ids = Vec::with_capacity(n as usize);
    for i in 0..left {
        ids.push(-((left - i) as i32));
    }
    for i in 0..right {
        ids.push(i as i32);
    }
    ids
}

/// Cosine in the first half, sine in the second. Argument is `t * 1000 * freq`.
///
/// Diffusers `Timesteps(flip_sin_to_cos=True, scale=1000)` and main
/// `sinusoidal_timestep_proj(..., sin_first=True, flip_sin_to_cos=True, scale=1000)`.
/// `sin_first` writes sine then cosine; the flip swaps the halves.
pub fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0f32; dim];
    let half = dim / 2;
    if half == 0 {
        return embedding;
    }
    let scaled = t * 1000.0;
    for i in 0..half {
        let freq = (-10000f32.ln() * (i as f32) / (half as f32)).exp();
        let arg = scaled * freq;
        embedding[i] = arg.cos();
        embedding[half + i] = arg.sin();
    }
    embedding
}

#[cfg(test)]
mod tests {
    use super::{position_ids, spatial_axis, timestep_features};

    /// SD.cpp `h_start = -n/2`, then `linspace(start, start + n - 1, n)`.
    fn sdcpp_spatial(n: i64) -> Vec<i32> {
        let start = -(n / 2);
        (0..n).map(|i| (start + i) as i32).collect()
    }

    #[test]
    fn even_grid_matches_sdcpp_and_text_starts_at_half() {
        assert_eq!(spatial_axis(4), vec![-2, -1, 0, 1]);
        assert_eq!(spatial_axis(2), vec![-1, 0]);
        assert_eq!(sdcpp_spatial(4), spatial_axis(4));
        assert_eq!(sdcpp_spatial(2), spatial_axis(2));

        let ids = position_ids(4, 2, 3).unwrap();
        assert_eq!(ids.text, vec![[2, 2, 2], [3, 3, 3], [4, 4, 4]]);
        // SD.cpp `txt_id_start = max(h_len, w_len)` is 4 here, then 4, 5, 6.
        assert_ne!(ids.text[0][0], 4i32.max(2));
        assert_eq!(
            ids.image,
            vec![
                [0, -2, -1],
                [0, -2, 0],
                [0, -1, -1],
                [0, -1, 0],
                [0, 0, -1],
                [0, 0, 0],
                [0, 1, -1],
                [0, 1, 0],
            ]
        );
        assert_eq!(ids.attention_order()[0], [2, 2, 2]);
        assert_eq!(ids.attention_order()[3], [0, -2, -1]);
    }

    #[test]
    fn odd_axis_is_the_diffusers_center_not_the_sdcpp_linspace() {
        assert_eq!(spatial_axis(3), vec![-2, -1, 0]);
        assert_eq!(sdcpp_spatial(3), vec![-1, 0, 1]);
        assert_ne!(spatial_axis(3), sdcpp_spatial(3));
        assert_eq!(spatial_axis(1), vec![-1]);
        assert_eq!(sdcpp_spatial(1), vec![0]);
    }

    #[test]
    fn timestep_table_is_cos_then_sin_of_t_times_1000() {
        // t = 0.001 so the scaled argument is exactly 1.
        let got = timestep_features(0.001, 2);
        assert!((got[0] - 1f32.cos()).abs() < 1e-5);
        assert!((got[1] - 1f32.sin()).abs() < 1e-5);
        assert!((got[0] - 0.001f32.cos()).abs() > 0.1);
    }
}
