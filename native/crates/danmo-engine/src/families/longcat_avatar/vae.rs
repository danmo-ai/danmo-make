//! LongCat-Avatar Wan 2.1 causal 3D VAE (`AutoencoderKLWan`, `is_residual=false`).
//!
//! Encoder: reference frame `[1, 3, 1, H, W]` → latent mean `[1, z, 1, h, w]`
//! (chunked causal path; the reference uses the first 1-frame chunk).
//! Decoder: denormalized latents → `[1, 3, 1 + 4*(T-1), H, W]` one latent
//! frame at a time with the feat-cache, so the "Rep" upsample3d cadence
//! (first latent → 1 frame, each next → 2 per stage) matches the baseline.
//!
//! All conv weights stay in the checkpoint's PT layout `(O, I, kD, kH, kW)` /
//! `(O, I, kH, kW)` and are folded into a per-time-step 2D conv weight of
//! shape `(O, kH, kW, kD*I)` exactly like the baseline — no key renaming.

use danmo_mlx::{Array, Error, Result, Stream};

use super::config::VaeConfig;
use crate::weights::{load_mlx_file, MlxWeights, WeightDtype};

const CACHE_T: usize = 2;

pub struct WanVae {
    config: VaeConfig,
    weights: MlxWeights,
}

enum CacheSlot {
    Empty,
    Frames(Array),
    /// First upsample3d call skips `time_conv` (baseline "Rep" sentinel).
    Rep,
}

struct Cache {
    slots: Vec<CacheSlot>,
    index: usize,
}

impl Cache {
    fn new(slots: usize) -> Self {
        Self {
            slots: (0..slots).map(|_| CacheSlot::Empty).collect(),
            index: 0,
        }
    }
}

impl WanVae {
    pub fn open(component_dir: &Path2, config: &VaeConfig, stream: &Stream) -> Result<Self> {
        let weights_path = component_dir.join("diffusion_pytorch_model.safetensors");
        let weights = load_mlx_file(&weights_path, WeightDtype::Native, None)?;
        let vae = Self {
            config: config.clone(),
            weights,
        };
        vae.assert_topology()?;
        let _ = stream;
        Ok(vae)
    }

    fn assert_topology(&self) -> Result<()> {
        for name in [
            "encoder.conv_in.weight",
            "decoder.conv_in.weight",
            "quant_conv.weight",
            "post_quant_conv.weight",
        ] {
            if !self.weights.has(name) {
                return Err(Error::new(
                    "longcat_avatar VAE topology",
                    format!("missing tensor {name}"),
                ));
            }
        }
        let conv_in = self.weights.tensor("encoder.conv_in.weight")?.shape();
        if conv_in.len() != 5 || conv_in[4] != 3 {
            return Err(Error::new(
                "longcat_avatar VAE topology",
                format!("encoder.conv_in.weight is {conv_in:?}, expected (O, kD, kH, kW, 3)"),
            ));
        }
        Ok(())
    }

    /// Video `[1, 3, T, H, W]` in `[-1, 1]` → normalized latent `[1, z, T_lat, h, w]`.
    pub fn encode_normalized(&self, video: &Array, stream: &Stream) -> Result<Array> {
        let mut cache = Cache::new(self.encoder_cache_slots());
        let t = video.shape()[2] as usize;
        let chunks = 1 + (t.saturating_sub(1)) / 4;
        let mut out: Option<Array> = None;
        for i in 0..chunks {
            cache.index = 0;
            let chunk = if i == 0 {
                video.slice(
                    &[0, 0, 0, 0, 0],
                    &[1, 3, 1, video.shape()[3], video.shape()[4]],
                    &[1, 1, 1, 1, 1],
                    stream,
                )?
            } else {
                let from = 1 + 4 * (i - 1);
                let to = (1 + 4 * i).min(t);
                video.slice(
                    &[0, 0, from as i32, 0, 0],
                    &[1, 3, to as i32, video.shape()[3], video.shape()[4]],
                    &[1, 1, 1, 1, 1],
                    stream,
                )?
            };
            let encoded = self.encoder_forward(&chunk, &mut cache, stream)?;
            out = Some(match out {
                None => encoded,
                Some(prev) => Array::concatenate_axis(&[prev, encoded], 2, stream)?,
            });
        }
        let out =
            out.ok_or_else(|| Error::new("longcat_avatar VAE", "encoder produced no frames"))?;
        let mixed = self.causal_conv3d("quant_conv", &out, None, stream)?;
        let mu = mixed.slice(
            &[0, 0, 0, 0, 0],
            &[
                1,
                self.config.z_dim as i32,
                mixed.shape()[2],
                mixed.shape()[3],
                mixed.shape()[4],
            ],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        Ok(self.normalize(&mu, stream)?)
    }

    /// Denormalized latent `[1, z, T, h, w]` → video `[1, 3, frames, H, W]`.
    pub fn decode(&self, denormalized: &Array, stream: &Stream) -> Result<Array> {
        let x = self.causal_conv3d("post_quant_conv", denormalized, None, stream)?;
        let mut cache = Cache::new(self.decoder_cache_slots());
        let frames = x.shape()[2] as usize;
        let mut out: Option<Array> = None;
        for i in 0..frames {
            cache.index = 0;
            let chunk = x.slice(
                &[0, 0, i as i32, 0, 0],
                &[1, x.shape()[1], i as i32 + 1, x.shape()[3], x.shape()[4]],
                &[1, 1, 1, 1, 1],
                stream,
            )?;
            let decoded = self.decoder_forward(&chunk, &mut cache, stream)?;
            out = Some(match out {
                None => decoded,
                Some(prev) => Array::concatenate_axis(&[prev, decoded], 2, stream)?,
            });
        }
        let out =
            out.ok_or_else(|| Error::new("longcat_avatar VAE", "decoder produced no frames"))?;
        out.clip(-1.0, 1.0, stream)
    }

    pub fn denormalize(&self, latents: &Array, stream: &Stream) -> Result<Array> {
        let mean = self.channel_vector(&self.config.latents_mean, stream)?;
        let std = self.channel_vector(&self.config.latents_std, stream)?;
        let scaled = Array::multiply(latents, &std, stream)?;
        Array::add(&scaled, &mean, stream)
    }

    fn normalize(&self, mu: &Array, stream: &Stream) -> Result<Array> {
        let mean = self.channel_vector(&self.config.latents_mean, stream)?;
        let std = self.channel_vector(&self.config.latents_std, stream)?;
        let centered = Array::subtract(mu, &mean, stream)?;
        Array::divide(&centered, &std, stream)
    }

    fn channel_vector(&self, values: &[f32], _stream: &Stream) -> Result<Array> {
        Array::from_f32_slice(values, &[1, self.config.z_dim as i32, 1, 1, 1])
    }

    fn encoder_cache_slots(&self) -> usize {
        let mut count = 1; // conv_in
        let dim_mult = &self.config.dim_mult;
        for (i, _) in dim_mult.iter().enumerate() {
            count += self.config.num_res_blocks * 2;
            if i != dim_mult.len() - 1 {
                if self.config.temperal_downsample[i] {
                    count += 1;
                }
            }
        }
        count += 2 * 2; // mid_block resnets (2 blocks)
        count += 1; // conv_out
        count
    }

    fn decoder_cache_slots(&self) -> usize {
        let mut count = 1; // conv_in
        count += 2 * 2; // mid_block resnets
        let temporal_upsample: Vec<bool> = self
            .config
            .temperal_downsample
            .iter()
            .rev()
            .copied()
            .collect();
        for (i, _) in self.config.dim_mult.iter().enumerate() {
            count += (self.config.num_res_blocks + 1) * 2;
            if i != self.config.dim_mult.len() - 1 && temporal_upsample[i] {
                count += 1;
            }
        }
        count += 1; // conv_out
        count
    }

    // ------------------------------------------------------------------
    // encoder / decoder towers
    // ------------------------------------------------------------------

    fn encoder_forward(&self, x: &Array, cache: &mut Cache, stream: &Stream) -> Result<Array> {
        let mut x = self.causal_conv3d("encoder.conv_in", x, Some(cache), stream)?;
        let dim_mult = &self.config.dim_mult;
        let mut stage = 0usize;
        for i in 0..dim_mult.len() {
            for res in 0..self.config.num_res_blocks {
                x = self.residual_block(
                    &format!("encoder.down_blocks.{stage}"),
                    &x,
                    cache,
                    stream,
                )?;
                stage += 1;
                let _ = res;
            }
            if i != dim_mult.len() - 1 {
                let mode = if self.config.temperal_downsample[i] {
                    "downsample3d"
                } else {
                    "downsample2d"
                };
                x = self.resample(
                    &format!("encoder.down_blocks.{stage}"),
                    &x,
                    mode,
                    Some(cache),
                    stream,
                )?;
                stage += 1;
            }
        }
        x = self.mid_block("encoder.mid_block", &x, cache, stream)?;
        x = self
            .rms_norm("encoder.norm_out", &x, false, stream)?
            .silu(stream)?;
        self.causal_conv3d("encoder.conv_out", &x, Some(cache), stream)
    }

    fn decoder_forward(&self, x: &Array, cache: &mut Cache, stream: &Stream) -> Result<Array> {
        let mut x = self.causal_conv3d("decoder.conv_in", x, Some(cache), stream)?;
        x = self.mid_block("decoder.mid_block", &x, cache, stream)?;
        let temporal_upsample: Vec<bool> = self
            .config
            .temperal_downsample
            .iter()
            .rev()
            .copied()
            .collect();
        for (i, _) in self.config.dim_mult.iter().enumerate() {
            let up_flag = i != self.config.dim_mult.len() - 1;
            let mode = if up_flag {
                Some(if temporal_upsample[i] {
                    "upsample3d"
                } else {
                    "upsample2d"
                })
            } else {
                None
            };
            x = self.up_block(&format!("decoder.up_blocks.{i}"), &x, mode, cache, stream)?;
        }
        x = self
            .rms_norm("decoder.norm_out", &x, false, stream)?
            .silu(stream)?;
        self.causal_conv3d("decoder.conv_out", &x, Some(cache), stream)
    }

    fn mid_block(
        &self,
        prefix: &str,
        x: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array> {
        let x = self.residual_block(&format!("{prefix}.resnets.0"), x, cache, stream)?;
        let x = self.attention_block(&format!("{prefix}.attentions.0"), &x, stream)?;
        self.residual_block(&format!("{prefix}.resnets.1"), &x, cache, stream)
    }

    fn up_block(
        &self,
        prefix: &str,
        x: &Array,
        upsample_mode: Option<&str>,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array> {
        let mut out = x.clone();
        for index in 0..=self.config.num_res_blocks {
            out = self.residual_block(&format!("{prefix}.resnets.{index}"), &out, cache, stream)?;
        }
        if let Some(mode) = upsample_mode {
            out = self.resample(
                &format!("{prefix}.upsamplers.0"),
                &out,
                mode,
                Some(cache),
                stream,
            )?;
        }
        Ok(out)
    }

    // ------------------------------------------------------------------
    // blocks
    // ------------------------------------------------------------------

    fn residual_block(
        &self,
        prefix: &str,
        x: &Array,
        cache: &mut Cache,
        stream: &Stream,
    ) -> Result<Array> {
        let shortcut = if self.weights.has(&format!("{prefix}.conv_shortcut.weight")) {
            self.causal_conv3d(&format!("{prefix}.conv_shortcut"), x, None, stream)?
        } else {
            x.clone()
        };
        let h = self
            .rms_norm(&format!("{prefix}.norm1"), x, false, stream)?
            .silu(stream)?;
        let h = self.causal_conv3d(&format!("{prefix}.conv1"), &h, Some(cache), stream)?;
        let h = self
            .rms_norm(&format!("{prefix}.norm2"), &h, false, stream)?
            .silu(stream)?;
        let h = self.causal_conv3d(&format!("{prefix}.conv2"), &h, Some(cache), stream)?;
        Array::add(&shortcut, &h, stream)
    }

    fn attention_block(&self, prefix: &str, x: &Array, stream: &Stream) -> Result<Array> {
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        // [B, C, T, H, W] -> per-frame [B*T, H, W, C]
        let frames = x
            .transpose_axes(&[0, 2, 3, 4, 1], stream)?
            .reshape(&[b * t, h, w, c], stream)?;
        let normed = self.rms_norm(&format!("{prefix}.norm"), &frames, true, stream)?;
        let attn_qkv = self
            .weights
            .tensor(&format!("{prefix}.to_qkv.weight"))?
            .shape();
        if attn_qkv.last().copied() != Some(c) {
            return Err(Error::new(
                "longcat_avatar VAE",
                format!(
                    "{prefix}.to_qkv.weight is {attn_qkv:?}; expected channels-last (3C, 1, 1, C={c})"
                ),
            ));
        }
        let qkv = self.conv2d(&format!("{prefix}.to_qkv"), &normed, 1, stream)?;
        // Single-head spatial attention: [B*T, 3C, H, W] -> [3, B*T, HW, C]
        let qkv = qkv
            .reshape(&[b * t, h * w, 3, c], stream)?
            .transpose_axes(&[2, 0, 1, 3], stream)?;
        // Drop the leading projection axis, then add the 1-head axis for SDPA.
        let head = |from: i32| -> Result<Array> {
            let part = qkv.slice(
                &[from, 0, 0, 0],
                &[from + 1, b * t, h * w, c],
                &[1, 1, 1, 1],
                stream,
            )?;
            part.squeeze_axis(0, stream)?.expand_dims(1, stream)
        };
        let (q, k, v) = (head(0)?, head(1)?, head(2)?);
        let scale = (c as f32).powf(-0.5);
        let out = Array::scaled_dot_product_attention(&q, &k, &v, scale, "", None, stream)?
            .squeeze_axis(1, stream)?
            .reshape(&[b * t, h, w, c], stream)?;
        let out = self.conv2d(&format!("{prefix}.proj"), &out, 1, stream)?;
        let out = out
            .reshape(&[b, t, h, w, c], stream)?
            .transpose_axes(&[0, 4, 1, 2, 3], stream)?;
        Array::add(x, &out, stream)
    }

    fn resample(
        &self,
        prefix: &str,
        x: &Array,
        mode: &str,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        match mode {
            "upsample3d" => {
                let cache = cache.ok_or_else(|| {
                    Error::new(
                        "longcat_avatar VAE",
                        "upsample3d requires the feature cache",
                    )
                })?;
                let mut x = x.clone();
                let idx = cache.index;
                let slot_empty = matches!(cache.slots[idx], CacheSlot::Empty);
                if slot_empty {
                    // First call skips time_conv entirely ("Rep" sentinel).
                    cache.slots[idx] = CacheSlot::Rep;
                    cache.index += 1;
                } else {
                    let tail = x.slice(
                        &[0, 0, t - CACHE_T as i32, 0, 0],
                        &shape,
                        &[1, 1, 1, 1, 1],
                        stream,
                    )?;
                    let cache_x = if (CACHE_T as i32) > t {
                        let one =
                            x.slice(&[0, 0, t - 1, 0, 0], &shape, &[1, 1, 1, 1, 1], stream)?;
                        let pad = match &cache.slots[idx] {
                            CacheSlot::Rep => one.zeros_like(stream)?,
                            CacheSlot::Frames(prev) => prev.slice(
                                &[0, 0, prev.shape()[2] - 1, 0, 0],
                                &prev.shape(),
                                &[1, 1, 1, 1, 1],
                                stream,
                            )?,
                            CacheSlot::Empty => one.zeros_like(stream)?,
                        };
                        Array::concatenate_axis(&[pad, one], 2, stream)?
                    } else {
                        tail.clone()
                    };
                    let rep = matches!(cache.slots[idx], CacheSlot::Rep);
                    x = if rep {
                        self.causal_conv3d(&format!("{prefix}.time_conv"), &x, None, stream)?
                    } else {
                        self.causal_conv3d_cached(
                            &format!("{prefix}.time_conv"),
                            &x,
                            cache_x,
                            stream,
                        )?
                    };
                    cache.slots[idx] = CacheSlot::Frames(tail);
                    cache.index += 1;
                    // Double T via reshape-and-stack.
                    let t2 = x.shape()[2];
                    x = x
                        .reshape(&[b, 2, c, t2, h, w], stream)?
                        .transpose_axes(&[0, 2, 1, 3, 4, 5], stream)?;
                    // baseline: stack([x0, x1], axis=3) — interleave along time
                    let x0 = x.slice(
                        &[0, 0, 0, 0, 0, 0],
                        &[b, c, 1, t2, h, w],
                        &[1, 1, 1, 1, 1, 1],
                        stream,
                    )?;
                    let x1 = x.slice(
                        &[0, 0, 1, 0, 0, 0],
                        &[b, c, 2, t2, h, w],
                        &[1, 1, 1, 1, 1, 1],
                        stream,
                    )?;
                    x = Array::stack(&[x0, x1], 3, stream)?
                        .reshape(&[b, c, t2 * 2, h, w], stream)?;
                }
                self.resample_spatial_up(prefix, &x, stream)
            }
            "upsample2d" => self.resample_spatial_up(prefix, x, stream),
            "downsample2d" => self.resample_spatial_down(prefix, x, stream),
            "downsample3d" => {
                let mut x = self.resample_spatial_down(prefix, x, stream)?;
                if let Some(cache) = cache {
                    let idx = cache.index;
                    if matches!(cache.slots[idx], CacheSlot::Empty) {
                        cache.slots[idx] = CacheSlot::Frames(x.clone());
                        cache.index += 1;
                    } else {
                        let prev = match &cache.slots[idx] {
                            CacheSlot::Frames(prev) => prev.slice(
                                &[0, 0, prev.shape()[2] - 1, 0, 0],
                                &prev.shape(),
                                &[1, 1, 1, 1, 1],
                                stream,
                            )?,
                            _ => {
                                return Err(Error::new(
                                    "longcat_avatar VAE",
                                    "downsample3d cache slot is not a frame window",
                                ))
                            }
                        };
                        let tail = x.slice(
                            &[0, 0, x.shape()[2] - 1, 0, 0],
                            &x.shape(),
                            &[1, 1, 1, 1, 1],
                            stream,
                        )?;
                        x = self.causal_conv3d_cached(
                            &format!("{prefix}.time_conv"),
                            &x,
                            prev,
                            stream,
                        )?;
                        cache.slots[idx] = CacheSlot::Frames(tail);
                        cache.index += 1;
                    }
                } else {
                    x = self.causal_conv3d(&format!("{prefix}.time_conv"), &x, None, stream)?;
                }
                Ok(x)
            }
            other => Err(Error::new(
                "longcat_avatar VAE",
                format!("unknown resample mode {other}"),
            )),
        }
    }

    /// Nearest 2x spatial upsample + 3x3 Conv2d (channel-halving at decode).
    fn resample_spatial_up(&self, prefix: &str, x: &Array, stream: &Stream) -> Result<Array> {
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let frames = x.transpose_axes(&[0, 2, 3, 4, 1], stream)?;
        let frames = frames.reshape(&[b * t, h, w, c], stream)?;
        let up = frames
            .repeat_axis(2, 1, stream)?
            .repeat_axis(2, 2, stream)?;
        let conv = self.conv2d(&format!("{prefix}.resample.1"), &up, 1, stream)?;
        let c_out = conv.shape()[3];
        Ok(conv
            .reshape(&[b, t, h * 2, w * 2, c_out], stream)?
            .transpose_axes(&[0, 4, 1, 2, 3], stream)?)
    }

    /// Zero-pad right/bottom 1 + stride-2 3x3 Conv2d (channel-keeping).
    fn resample_spatial_down(&self, prefix: &str, x: &Array, stream: &Stream) -> Result<Array> {
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let frames = x
            .transpose_axes(&[0, 2, 3, 4, 1], stream)?
            .reshape(&[b * t, h, w, c], stream)?;
        let padded = frames.pad(&[1, 2], &[0, 0], &[1, 1], 0.0, "constant", stream)?;
        // The right/bottom pad above IS the conv's padding; do not pad twice.
        let weight = self
            .weights
            .tensor(&format!("{prefix}.resample.1.weight"))?;
        let conv = Array::conv2d(&padded, weight, [2, 2], [0, 0], [1, 1], 1, stream)?;
        let (h_out, w_out, c_out) = (conv.shape()[1], conv.shape()[2], conv.shape()[3]);
        Ok(conv
            .reshape(&[b, t, h_out, w_out, c_out], stream)?
            .transpose_axes(&[0, 4, 1, 2, 3], stream)?)
    }

    // ------------------------------------------------------------------
    // primitives
    // ------------------------------------------------------------------

    /// Channel-first L2 normalize × sqrt(dim) × gamma (`WanRMS_norm`).
    /// `images=true` is the per-frame `[B*T, H, W, C]` layout used by the
    /// attention block; `false` is the 5D `[B, C, T, H, W]` layout.
    fn rms_norm(&self, prefix: &str, x: &Array, images: bool, stream: &Stream) -> Result<Array> {
        let gamma = self.weights.tensor(&format!("{prefix}.gamma"))?;
        let shape = x.shape();
        let dim = if images {
            *shape.last().unwrap_or(&1)
        } else {
            shape[1]
        };
        let axis = if images { 3 } else { 1 };
        // norm = sqrt(max(sum(x^2), 1e-12)) — mean×dim keeps the exact sum.
        let sum_sq = Array::multiply(x, x, stream)?
            .mean_axes(&[axis], true, stream)?
            .pipe(
                |m, s| Array::multiply(&m, &Array::scalar_f32(dim as f32)?, s),
                stream,
            )?;
        let norm = Array::maximum(&sum_sq, &Array::scalar_f32(1e-12)?, stream)?.sqrt(stream)?;
        let normalized = Array::divide(x, &norm, stream)?;
        let scale = Array::scalar_f32((dim as f32).sqrt())?;
        let scaled = Array::multiply(&normalized, &scale, stream)?;
        let broadcast_shape = if images {
            vec![dim]
        } else {
            vec![dim, 1, 1, 1]
        };
        let gamma = gamma.reshape(&broadcast_shape, stream)?;
        Array::multiply(&scaled, &gamma, stream)
    }

    /// Causal conv3d without cache (fresh chunk start).
    fn causal_conv3d(
        &self,
        prefix: &str,
        x: &Array,
        cache: Option<&mut Cache>,
        stream: &Stream,
    ) -> Result<Array> {
        let cached = match cache {
            Some(cache) => {
                let idx = cache.index;
                let shape = x.shape();
                let tail_len = (CACHE_T as i32).min(shape[2]);
                let tail = x.slice(
                    &[0, 0, shape[2] - tail_len, 0, 0],
                    &shape,
                    &[1, 1, 1, 1, 1],
                    stream,
                )?;
                let cache_x = if tail_len < CACHE_T as i32 {
                    match &cache.slots[idx] {
                        CacheSlot::Frames(prev) => {
                            let one = prev.slice(
                                &[0, 0, prev.shape()[2] - 1, 0, 0],
                                &prev.shape(),
                                &[1, 1, 1, 1, 1],
                                stream,
                            )?;
                            Some(Array::concatenate_axis(&[one, tail.clone()], 2, stream)?)
                        }
                        _ => Some(tail.clone()),
                    }
                } else {
                    Some(tail.clone())
                };
                let result = match &cache.slots[idx] {
                    CacheSlot::Frames(prev) => {
                        self.causal_conv3d_cached(prefix, x, prev.clone(), stream)?
                    }
                    _ => self.causal_conv3d_full(prefix, x, None, stream)?,
                };
                cache.slots[idx] = CacheSlot::Frames(
                    cache_x
                        .ok_or_else(|| Error::new("longcat_avatar VAE", "cache window missing"))?,
                );
                cache.index += 1;
                result
            }
            None => self.causal_conv3d_full(prefix, x, None, stream)?,
        };
        Ok(cached)
    }

    /// Causal conv3d with an explicit cache window prepended along time.
    fn causal_conv3d_cached(
        &self,
        prefix: &str,
        x: &Array,
        cache_x: Array,
        stream: &Stream,
    ) -> Result<Array> {
        self.causal_conv3d_full(prefix, x, Some(cache_x), stream)
    }

    /// Core sliding-window conv: the bundle stores conv3d kernels in MLX
    /// channels-last layout `(O, kD, kH, kW, I)`. Fold to
    /// `(O, kH, kW, kD*I)` and run one 2D conv per time step; spatial padding
    /// is 1 for 3x3 kernels (1x1x1 mixers take no padding).
    fn causal_conv3d_full(
        &self,
        prefix: &str,
        x: &Array,
        cache_x: Option<Array>,
        stream: &Stream,
    ) -> Result<Array> {
        let weight = self.weights.tensor(&format!("{prefix}.weight"))?;
        let wshape = weight.shape();
        let (o, kd, kh, kw, i) = (wshape[0], wshape[1], wshape[2], wshape[3], wshape[4]);
        let mut x = x.clone();
        let causal_pad = kd - 1;
        let prepend = if let Some(cache_x) = cache_x {
            if causal_pad > 0 {
                let extra = causal_pad - cache_x.shape()[2];
                let mut parts = Vec::with_capacity(3);
                if extra > 0 {
                    let zero_shape = {
                        let mut shape = x.shape();
                        shape[2] = extra;
                        shape
                    };
                    parts.push(zeros(&zero_shape, x.dtype(), stream)?);
                }
                parts.push(cache_x);
                parts.push(x);
                Array::concatenate_axis(&parts, 2, stream)?
            } else {
                x
            }
        } else if causal_pad > 0 {
            let zero_shape = {
                let mut shape = x.shape();
                shape[2] = causal_pad;
                shape
            };
            let zeros = zeros(&zero_shape, x.dtype(), stream)?;
            Array::concatenate_axis(&[zeros, x], 2, stream)?
        } else {
            x
        };
        let x = prepend;
        let spatial_pad = (kh - 1) / 2;
        let x = if spatial_pad > 0 {
            x.pad(
                &[3, 4],
                &[spatial_pad, spatial_pad],
                &[spatial_pad, spatial_pad],
                0.0,
                "constant",
                stream,
            )?
        } else {
            x
        };
        // [B, C, T, H, W] -> [B, T, H, W, C]
        let x = x.transpose_axes(&[0, 2, 3, 4, 1], stream)?;
        let shape = x.shape();
        let (b, t, h, w, c_in) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        if c_in != i {
            return Err(Error::new(
                "longcat_avatar VAE",
                format!("{prefix}: input channels {c_in} != weight input {i}"),
            ));
        }
        let t_out = (t - kd) / 1 + 1;
        // Fold weight: (O, kD, kH, kW, I) -> (O, kH, kW, kD*I)
        let w2d = weight
            .transpose_axes(&[0, 2, 3, 1, 4], stream)?
            .reshape(&[o, kh, kw, kd * i], stream)?;
        let bias = self.weights.tensor(&format!("{prefix}.bias"))?;
        let mut outputs = Vec::with_capacity(t_out as usize);
        for t_i in 0..t_out {
            let window = x.slice(
                &[0, t_i, 0, 0, 0],
                &[b, t_i + kd, h, w, c_in],
                &[1, 1, 1, 1, 1],
                stream,
            )?;
            // [B, kD, H, W, I] -> [B, H, W, kD*I]
            let window = window
                .transpose_axes(&[0, 2, 3, 1, 4], stream)?
                .reshape(&[b, h, w, kd * i], stream)?;
            let out = Array::conv2d(&window, &w2d, [1, 1], [0, 0], [1, 1], 1, stream)?;
            let out = Array::add(&out, bias, stream)?;
            outputs.push(out);
        }
        let stacked = Array::stack(&outputs, 1, stream)?; // [B, T_out, H, W, O]
        stacked.transpose_axes(&[0, 4, 1, 2, 3], stream)
    }

    /// 2D conv on channels-last `[B, H, W, C]`; the checkpoint already stores
    /// `(O, kH, kW, I)` channels-last kernels. Padding is `(k-1)/2`.
    fn conv2d(&self, prefix: &str, x: &Array, stride: i32, stream: &Stream) -> Result<Array> {
        let weight = self.weights.tensor(&format!("{prefix}.weight"))?;
        let wshape = weight.shape();
        let (kh, kw) = (wshape[1], wshape[2]);
        let padding = ((kh - 1) / 2, (kw - 1) / 2);
        let out = Array::conv2d(
            x,
            weight,
            [stride, stride],
            [padding.0, padding.1],
            [1, 1],
            1,
            stream,
        )?;
        if self.weights.has(&format!("{prefix}.bias")) {
            let bias = self.weights.tensor(&format!("{prefix}.bias"))?;
            return Array::add(&out, bias, stream);
        }
        Ok(out)
    }
}

type Path2 = std::path::Path;

fn zeros(shape: &[i32], dtype: danmo_mlx::Dtype, stream: &Stream) -> Result<Array> {
    Array::from_f32_slice(&[0.0], &[])?
        .astype(dtype, stream)?
        .broadcast_to(shape, stream)
}

trait Pipe: Sized {
    fn pipe<T>(self, f: impl FnOnce(&Self, &Stream) -> Result<T>, stream: &Stream) -> Result<T>;
}
impl Pipe for Array {
    fn pipe<T>(self, f: impl FnOnce(&Self, &Stream) -> Result<T>, stream: &Stream) -> Result<T> {
        f(&self, stream)
    }
}
