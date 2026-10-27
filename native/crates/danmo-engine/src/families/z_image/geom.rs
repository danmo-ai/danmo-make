//! Token geometry shared by the ggml graph and the SD.cpp check.
//!
//! Positions follow Tongyi `ZImageTransformer2DModel` and SD.cpp `gen_z_image_ids`:
//! caption ids are `(i + 1, 0, 0)` across the padded caption, image ids are
//! `(cap_len + 1, row, col)`, and image pad ids are `(0, 0, 0)`.

use crate::runtime::RuntimeError;

pub const SEQ_MULTI_OF: i64 = 32;
pub const PATCH: i64 = 2;
pub const ROPE_THETA: f32 = 256.0;
pub const T_SCALE: f32 = 1000.0;
pub const FREQ_DIM: i64 = 256;
pub const MID_DIM: i64 = 1024;
pub const VAE_SCALE: i64 = 8;
pub const RMS_EPS: f32 = 1e-5;
pub const FINAL_EPS: f32 = 1e-6;

/// Trailing pad so `len` is a multiple of `multiple`. SD.cpp `Rope::bound_mod`.
pub fn pad_len(len: i64, multiple: i64) -> i64 {
    if multiple <= 0 {
        return 0;
    }
    (multiple - (len % multiple)) % multiple
}

/// Tongyi `int(dim / 3 * 8)`. At dim 3840 this is 10240.
#[cfg(test)]
pub fn ffn_hidden(dim: i64) -> i64 {
    (dim as f64 / 3.0 * 8.0) as i64
}

/// SD.cpp rounds that width up to `multiple_of` (256). At 3840 the round is a no-op.
#[cfg(test)]
pub fn sdcpp_ffn(dim: i64, multiple_of: i64) -> i64 {
    let hidden = (dim as f64 * (8.0 / 3.0)) as i64;
    let multiple = multiple_of.max(1);
    multiple * ((hidden + multiple - 1) / multiple)
}

/// Head 128 is the published Z-Image split. Head 8 is the smoke graph.
pub fn axes_for_head(head: i64) -> Result<[i64; 3], RuntimeError> {
    match head {
        128 => Ok([32, 48, 48]),
        8 => Ok([2, 2, 4]),
        other => Err(RuntimeError::Message(format!(
            "z-image rope axes are 32+48+48 for head 128; head {other} has no axis table"
        ))),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TokenGrid {
    pub latent_w: i64,
    pub latent_h: i64,
    pub tokens_w: i64,
    pub tokens_h: i64,
    pub img_real: i64,
    pub img_pad: i64,
    pub img_len: i64,
    pub cap_real: i64,
    pub cap_pad: i64,
    pub cap_len: i64,
    pub unified: i64,
}

pub fn token_grid(
    width: u32,
    height: u32,
    cap_real: i64,
    vae_scale: i64,
) -> Result<TokenGrid, RuntimeError> {
    if vae_scale < 1 || cap_real < 1 {
        return Err(RuntimeError::Message(
            "z-image latent scale or caption length is empty".into(),
        ));
    }
    let width = width as i64;
    let height = height as i64;
    if width % vae_scale != 0 || height % vae_scale != 0 {
        return Err(RuntimeError::Message(format!(
            "size {width}x{height} is not divisible by vae scale {vae_scale}"
        )));
    }
    let latent_w = width / vae_scale;
    let latent_h = height / vae_scale;
    if latent_w % PATCH != 0 || latent_h % PATCH != 0 {
        return Err(RuntimeError::Message(format!(
            "latent {latent_w}x{latent_h} is not divisible by patch {PATCH}"
        )));
    }
    let tokens_w = latent_w / PATCH;
    let tokens_h = latent_h / PATCH;
    let img_real = tokens_w * tokens_h;
    let img_pad = pad_len(img_real, SEQ_MULTI_OF);
    let cap_pad = pad_len(cap_real, SEQ_MULTI_OF);
    let img_len = img_real + img_pad;
    let cap_len = cap_real + cap_pad;
    Ok(TokenGrid {
        latent_w,
        latent_h,
        tokens_w,
        tokens_h,
        img_real,
        img_pad,
        img_len,
        cap_real,
        cap_pad,
        cap_len,
        unified: img_len + cap_len,
    })
}

/// Image tokens, then caption tokens. This is the training order.
pub fn positions_image_first(grid: TokenGrid) -> Vec<[i32; 3]> {
    let mut ids = Vec::with_capacity(grid.unified as usize);
    for token in 0..grid.img_real {
        let row = token / grid.tokens_w;
        let col = token % grid.tokens_w;
        ids.push([(grid.cap_len + 1) as i32, row as i32, col as i32]);
    }
    for _ in 0..grid.img_pad {
        ids.push([0, 0, 0]);
    }
    for index in 0..grid.cap_len {
        ids.push([(index + 1) as i32, 0, 0]);
    }
    ids
}

/// SD.cpp `gen_z_image_ids` concatenates caption, then image. The ids themselves match.
#[cfg(test)]
pub fn positions_text_first(grid: TokenGrid) -> Vec<[i32; 3]> {
    let image_first = positions_image_first(grid);
    let (image, caption) = image_first.split_at(grid.img_len as usize);
    let mut ids = Vec::with_capacity(image_first.len());
    ids.extend_from_slice(caption);
    ids.extend_from_slice(image);
    ids
}

pub fn rope_columns(ids: &[[i32; 3]]) -> [Vec<i32>; 3] {
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    for id in ids {
        for axis in 0..3 {
            axes[axis].push(id[axis]);
        }
    }
    axes
}

/// Row index into a flattened `[W, H, C]` latent for each real patch element.
/// Within a token, channel is fastest, then patch x, then patch y.
pub fn patch_index(grid: TokenGrid, channels: i64) -> Vec<i32> {
    let mut index = Vec::with_capacity((grid.img_real * PATCH * PATCH * channels) as usize);
    for row in 0..grid.tokens_h {
        for col in 0..grid.tokens_w {
            for ph in 0..PATCH {
                for pw in 0..PATCH {
                    for channel in 0..channels {
                        let x = col * PATCH + pw;
                        let y = row * PATCH + ph;
                        let src = x + y * grid.latent_w + channel * grid.latent_w * grid.latent_h;
                        index.push(src as i32);
                    }
                }
            }
        }
    }
    index
}

/// Tongyi `TimestepEmbedder.timestep_embedding`: cos then sin, period 10000.
/// The caller passes `t = (1 - sigma) * t_scale`.
pub fn timestep_features(t: f32, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0f32; dim];
    let half = dim / 2;
    if half == 0 {
        return embedding;
    }
    for i in 0..half {
        let freq = (-10000f32.ln() * (i as f32) / (half as f32)).exp();
        let arg = t * freq;
        embedding[i] = arg.cos();
        embedding[half + i] = arg.sin();
    }
    embedding
}
