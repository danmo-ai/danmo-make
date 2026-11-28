//! LongCat-Video-Avatar 1.5 DiT (`LongCatVideoAvatarTransformer3DModel`).
//!
//! Single-talker ATI2V path: patchify → AdaLN self-attn (3D RoPE) → text
//! cross-attn (variable-length packed) → audio cross-attn (per-frame tokens
//! from `audio_proj`, AdaLN-gated) → SwiGLU FFN → AdaLN final head.
//! AdaLN modulation, norms, and residual accumulation run in F32 and cast
//! back, matching the baseline's `*_FP32` convention.
//!
//! Video-continuation features (Reference Skip Q-slicing, MultiTalk routing,
//! KV cache) are not part of this migration slice and fail loudly via the
//! session if requested — nothing silently falls back.

use danmo_mlx::{Array, Dtype, Error, LinearWeight, Result, Stream};

use super::config::DitConfig;
use super::rope::RopeTables;
use crate::weights::{load_mlx_directory, MlxWeights, WeightDtype};

pub struct AvatarDit {
    config: DitConfig,
    weights: MlxWeights,
}

/// DiT inputs assembled by the session for one denoise call.
pub struct DitInputs {
    /// `[B, 16, T_lat, h, w]` (reference latent + noise).
    pub latents: Array,
    /// `[B]` timesteps in scheduler units (sigma × 1000).
    pub timestep: Array,
    /// `[B, 1, 512, 4096]` umT5 embeds (uncond/cond concatenated when B=2).
    pub text_embeds: Array,
    /// Number of valid text tokens per batch row.
    pub text_lengths: Vec<usize>,
    /// `[B, 1, 1, N_text]` text keep mask for `text_tokens_zero_pad`.
    pub text_masks: Array,
    /// `[B, T_audio, 5, 5, 1280]` windowed audio features.
    pub audio_embs: Array,
    pub num_cond_latents: usize,
}

impl AvatarDit {
    pub fn open(component_dir: &std::path::Path, config: &DitConfig) -> Result<Self> {
        let quantization = config.quantization.map(|q| danmo_mlx::AffineQuantization {
            bits: q.bits,
            group_size: q.group_size,
        });
        let weights = load_mlx_directory(component_dir, WeightDtype::Native, quantization)?;
        let dit = Self {
            config: config.clone(),
            weights,
        };
        dit.assert_topology()?;
        Ok(dit)
    }

    fn assert_topology(&self) -> Result<()> {
        for name in [
            "x_embedder.proj.weight",
            "t_embedder.mlp.0.weight",
            "t_embedder.mlp.2.weight",
            "y_embedder.y_proj.0.weight",
            "y_embedder.y_proj.2.weight",
            "audio_proj.proj1.weight",
            "audio_proj.proj1_vf.weight",
            "audio_proj.proj2.weight",
            "audio_proj.proj3.weight",
            "final_layer.linear.weight",
            "final_layer.adaLN_modulation.1.weight",
        ] {
            if !self.weights.has(name) {
                return Err(Error::new(
                    "longcat_avatar DiT topology",
                    format!("missing tensor {name}"),
                ));
            }
        }
        let proj = self.weights.tensor("x_embedder.proj.weight")?.shape();
        // Bundle kernels are MLX channels-last: (O, kD, kH, kW, I).
        let expect = [
            self.config.hidden_size as i32,
            self.config.patch_size[0] as i32,
            self.config.patch_size[1] as i32,
            self.config.patch_size[2] as i32,
            self.config.in_channels as i32,
        ];
        if proj != expect {
            return Err(Error::new(
                "longcat_avatar DiT topology",
                format!("x_embedder.proj.weight is {proj:?}, expected {expect:?} (O,kD,kH,kW,I)"),
            ));
        }
        for layer in 0..self.config.depth {
            let prefix = format!("blocks.{layer}");
            for suffix in [
                "adaLN_modulation.1.weight",
                "audio_adaLN_modulation.1.weight",
                "attn.qkv.weight",
                "attn.proj.weight",
                "cross_attn.q_linear.weight",
                "cross_attn.kv_linear.weight",
                "audio_cross_attn.q_linear.weight",
                "audio_cross_attn.kv_linear.weight",
                "ffn.w1.weight",
                "ffn.w2.weight",
                "ffn.w3.weight",
                "pre_crs_attn_norm.weight",
                "pre_video_crs_attn_norm.weight",
            ] {
                let name = format!("{prefix}.{suffix}");
                if !self.weights.has(&name) {
                    return Err(Error::new(
                        "longcat_avatar DiT topology",
                        format!("missing tensor {name}"),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn forward(&self, inputs: &DitInputs, stream: &Stream) -> Result<Array> {
        let latents = inputs.latents.shape();
        if latents.len() != 5 {
            return Err(Error::new(
                "longcat_avatar DiT",
                format!("latents must be [B, C, T, H, W], got {latents:?}"),
            ));
        }
        let (b, c, t, h, w) = (latents[0], latents[1], latents[2], latents[3], latents[4]);
        if c != self.config.in_channels as i32 {
            return Err(Error::new(
                "longcat_avatar DiT",
                format!("latent channels {c} != {}", self.config.in_channels),
            ));
        }
        let [pt, ph, pw] = self.config.patch_size;
        if t % pt as i32 != 0 || h % ph as i32 != 0 || w % pw as i32 != 0 {
            return Err(Error::new(
                "longcat_avatar DiT",
                format!("latent grid {t}x{h}x{w} is not divisible by patch {pt}x{ph}x{pw}"),
            ));
        }
        let (n_t, n_h, n_w) = (t / pt as i32, h / ph as i32, w / pw as i32);
        let hidden = self.config.hidden_size as i32;

        let dtype = self.weights.tensor("x_embedder.proj.weight")?.dtype();
        let tag = |stage: &str, e: Error| {
            Error::new("longcat_avatar DiT forward", format!("{stage}: {e}"))
        };
        // ---- patchify [B, C, T, H, W] -> [B, N, hidden] ----
        let mut tokens = self
            .patch_embed(&inputs.latents, dtype, stream)
            .map_err(|e| tag("patch_embed", e))?;
        // ---- timestep embed [B, N_t, tembed] (F32) ----
        let temb = self
            .timestep_embed(&inputs.timestep, n_t, b, inputs.num_cond_latents, stream)
            .map_err(|e| tag("timestep_embed", e))?;
        // ---- caption embed [B, 1, N_text, hidden] ----
        let caption = self
            .caption_embed(&inputs.text_embeds, dtype, stream)
            .map_err(|e| tag("caption_embed", e))?;
        let caption = if self.config.text_tokens_zero_pad {
            let mask = inputs
                .text_masks
                .reshape(&[b, 1, inputs.text_embeds.shape()[2], 1], stream)?
                .astype(dtype, stream)?;
            Array::multiply(&caption, &mask, stream)?
        } else {
            caption
        };

        // ---- audio per-frame context tokens [B*N_t, 32, 768] ----
        let audio = self
            .audio_context(&inputs.audio_embs, n_t, b, stream)
            .map_err(|e| tag("audio_context", e))?;

        let tokens_per_frame = (n_t * n_h * n_w) / n_t;
        let rope = RopeTables::build(
            n_t as usize,
            n_h as usize,
            n_w as usize,
            self.config.head_dim(),
            stream,
        )?;

        for layer in 0..self.config.depth {
            tokens = self.block(
                layer,
                &tokens,
                &caption,
                &inputs.text_lengths,
                &temb,
                &audio,
                &rope,
                [n_t, n_h, n_w],
                tokens_per_frame,
                inputs.num_cond_latents,
                dtype,
                stream,
            )?;
        }

        // ---- final head ----
        let tokens = self.final_layer(&tokens, &temb, [n_t, n_h, n_w], b, hidden, stream)?;
        self.unpatchify(&tokens, [n_t, n_h, n_w], stream)
    }

    // ------------------------------------------------------------------
    // embedders
    // ------------------------------------------------------------------

    /// 3D patchify as per-frame 2D convs (patch temporal is 1 for this model).
    fn patch_embed(&self, latents: &Array, dtype: Dtype, stream: &Stream) -> Result<Array> {
        let weight = self.weights.tensor("x_embedder.proj.weight")?;
        let bias = self.weights.tensor("x_embedder.proj.bias")?;
        let shape = weight.shape();
        let (o, kd, kh, kw, i) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        if kd != 1 {
            return Err(Error::new(
                "longcat_avatar DiT",
                format!("x_embedder temporal patch {kd} is not 1"),
            ));
        }
        // Channels-last (O, 1, kH, kW, I) -> (O, kH, kW, I)
        let w2d = weight.reshape(&[o, kh, kw, i], stream)?;
        let x = latents.astype(dtype, stream)?;
        let shape = x.shape();
        let (b, c, t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        // [B, C, T, H, W] -> [B*T, H, W, C]
        let frames = x
            .transpose_axes(&[0, 2, 3, 4, 1], stream)?
            .reshape(&[b * t, h, w, c], stream)?;
        let out = Array::conv2d(&frames, &w2d, [kh, kw], [0, 0], [1, 1], 1, stream)?;
        let out = Array::add(&out, bias, stream)?;
        let (h_out, w_out) = (out.shape()[1], out.shape()[2]);
        out.reshape(&[b, t * h_out * w_out, o], stream)
    }

    fn unpatchify(&self, tokens: &Array, grid: [i32; 3], stream: &Stream) -> Result<Array> {
        let [n_t, n_h, n_w] = grid;
        let [pt, ph, pw] = self.config.patch_size;
        let out_c = self.config.out_channels as i32;
        let b = tokens.shape()[0];
        let x = tokens.reshape(
            &[b, n_t, n_h, n_w, pt as i32, ph as i32, pw as i32, out_c],
            stream,
        )?;
        let x = x.transpose_axes(&[0, 7, 1, 4, 2, 5, 3, 6], stream)?;
        x.reshape(
            &[b, out_c, n_t * pt as i32, n_h * ph as i32, n_w * pw as i32],
            stream,
        )?
        .astype(Dtype::Float32, stream)
    }

    /// Sinusoidal + MLP timestep embedding. Each batch scalar is broadcast
    /// across `n_t`, except conditioning latents which use timestep zero.
    fn timestep_embed(
        &self,
        timestep: &Array,
        n_t: i32,
        b: i32,
        num_cond_latents: usize,
        stream: &Stream,
    ) -> Result<Array> {
        let dim = self.config.frequency_embedding_size;
        let times = timestep.astype(Dtype::Float32, stream)?;
        let per_batch = times.to_vec_f32(&Stream::cpu()?)?;
        if per_batch.len() as i32 != b {
            return Err(Error::new(
                "longcat_avatar DiT",
                format!(
                    "timestep has {} values, expected one per batch row ({b})",
                    per_batch.len()
                ),
            ));
        }
        let mut flat = Vec::with_capacity(per_batch.len() * n_t as usize);
        for value in &per_batch {
            for frame in 0..n_t as usize {
                flat.push(if frame < num_cond_latents {
                    0.0
                } else {
                    *value
                });
            }
        }
        let times = flat;
        let half = dim / 2;
        let max_period = 10_000f32;
        let mut freq = Vec::with_capacity(times.len() * dim);
        for time in &times {
            for index in 0..half {
                let scale = (-(max_period.ln()) * index as f32 / half as f32).exp();
                freq.push(time * scale);
            }
        }
        let mut embedding = Vec::with_capacity(times.len() * dim);
        for row in freq.chunks(half) {
            for angle in row {
                embedding.push(angle.cos());
            }
            for angle in row {
                embedding.push(angle.sin());
            }
        }
        let freq = Array::from_f32_slice(&embedding, &[times.len() as i32, dim as i32])?;
        let l0 = self.linear("t_embedder.mlp.0", &freq, stream)?;
        let act = l0.silu(stream)?;
        let temb = self.linear("t_embedder.mlp.2", &act, stream)?;
        temb.reshape(&[b, n_t, self.config.adaln_tembed_dim as i32], stream)
    }

    fn caption_embed(&self, text: &Array, dtype: Dtype, stream: &Stream) -> Result<Array> {
        let x = text.astype(dtype, stream)?;
        let x = self.linear("y_embedder.y_proj.0", &x, stream)?;
        let x = crate::common::mlx_ops::gelu_tanh(&x, stream)?;
        self.linear("y_embedder.y_proj.2", &x, stream)
    }

    fn linear(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        let weight: &LinearWeight = self.weights.linear(&format!("{prefix}.weight"))?;
        let bias_name = format!("{prefix}.bias");
        let bias = if self.weights.has(&bias_name) {
            Some(self.weights.tensor(&bias_name)?)
        } else {
            None
        };
        weight.apply(input, bias, stream)
    }

    // ------------------------------------------------------------------
    // audio path
    // ------------------------------------------------------------------

    /// `[B, T_audio, 5, 5, 1280]` → `[B*N_t, context_tokens, output_dim]`.
    fn audio_context(&self, audio: &Array, n_t: i32, b: i32, stream: &Stream) -> Result<Array> {
        let shape = audio.shape();
        let w = self.config.audio_window;
        let v = self.config.vae_scale;
        let middle = w / 2;
        let first = audio.slice(
            &[0, 0, 0, 0, 0],
            &[shape[0], 1, shape[2], shape[3], shape[4]],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        let latter = audio.slice(
            &[0, 1, 0, 0, 0],
            &[shape[0], shape[1], shape[2], shape[3], shape[4]],
            &[1, 1, 1, 1, 1],
            stream,
        )?;
        // Re-window per vae_scale chunk (mirrors baseline `_prepare_audio_hidden_states`).
        let (bz, t_l, w_, s_, c_) = (
            latter.shape()[0],
            latter.shape()[1],
            latter.shape()[2],
            latter.shape()[3],
            latter.shape()[4],
        );
        let n_chunks = t_l / v as i32;
        if n_chunks * v as i32 != t_l {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "audio frames {t_l} (total {} with head) are not 1 + vae_scale({v}) × latent frames",
                    shape[1]
                ),
            ));
        }
        if w_ != w as i32
            || s_ != self.config.audio_block as i32
            || c_ != self.config.audio_channel as i32
        {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "audio window must be [W={w}, S={}, C={}], got [W={w_}, S={s_}, C={c_}]",
                    self.config.audio_block, self.config.audio_channel
                ),
            ));
        }
        let latter = latter.reshape(&[bz, n_chunks, v as i32, w_, s_, c_], stream)?;
        let l_first = latter
            .slice(
                &[0, 0, 0, 0, 0, 0],
                &[bz, n_chunks, 1, (middle + 1) as i32, s_, c_],
                &[1, 1, 1, 1, 1, 1],
                stream,
            )?
            .reshape(&[bz, n_chunks, (middle + 1) as i32, s_, c_], stream)?;
        let l_mid = if v > 2 {
            latter
                .slice(
                    &[0, 0, 1, middle as i32, 0, 0],
                    &[bz, n_chunks, (v - 1) as i32, (middle + 1) as i32, s_, c_],
                    &[1, 1, 1, 1, 1, 1],
                    stream,
                )?
                .reshape(&[bz, n_chunks, (v - 2) as i32, s_, c_], stream)?
        } else {
            Array::from_f32_slice(&[0.0], &[])?
                .astype(latter.dtype(), stream)?
                .broadcast_to(&[bz, n_chunks, 0, s_, c_], stream)?
        };
        let l_last = latter
            .slice(
                &[0, 0, (v - 1) as i32, middle as i32, 0, 0],
                &[bz, n_chunks, v as i32, w_, s_, c_],
                &[1, 1, 1, 1, 1, 1],
                stream,
            )?
            .reshape(&[bz, n_chunks, (w - middle) as i32, s_, c_], stream)?;
        let window_total = (middle + 1) + (v - 2) + (w - middle);
        if window_total != w + v - 1 {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "re-window sum {window_total} != W+v-1={} (W={w}, v={v})",
                    w + v - 1
                ),
            ));
        }
        let latter_s = Array::concatenate_axis(&[l_first, l_mid, l_last], 2, stream)?;
        if latter_s.shape()[2] != (w + v - 1) as i32 {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "re-windowed audio is {:?}, expected window {} on axis 2",
                    latter_s.shape(),
                    w + v - 1
                ),
            ));
        }
        // audio_proj: first-frame proj1, subsequent proj1_vf, shared proj2→proj3
        let ctx = self.audio_proj(&first, &latter_s, stream)?;
        if ctx.shape()[1] != n_t {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "audio context produced {} rows, expected {n_t}",
                    ctx.shape()[1]
                ),
            ));
        }
        ctx.reshape(
            &[
                b * n_t,
                self.config.context_tokens as i32,
                self.config.output_dim as i32,
            ],
            stream,
        )
    }

    /// `AudioProjModel`: first frame `proj1`, subsequent frames `proj1_vf`,
    /// shared ReLU `proj2`, `proj3`, output LayerNorm.
    fn audio_proj(&self, first: &Array, latter: &Array, stream: &Stream) -> Result<Array> {
        let flatten = |x: &Array| -> Result<Array> {
            let s = x.shape();
            x.reshape(&[s[0] * s[1], s[2] * s[3] * s[4]], stream)
        };
        let relu = |x: &Array, stream: &Stream| -> Result<Array> {
            Array::maximum(x, &Array::scalar_f32(0.0)?, stream)
        };
        let first_flat = flatten(first)?;
        let latter_flat = flatten(latter)?;
        // proj1 input = W*S*C = 5*5*1280 = 32000; proj1_vf = (W+v-1)*S*C = 8*5*1280.
        // Fail loud on a drifted audio window instead of a broadcast error downstream.
        let expect_first =
            (self.config.audio_window * self.config.audio_block * self.config.audio_channel) as i32;
        let expect_vf = ((self.config.audio_window + self.config.vae_scale - 1)
            * self.config.audio_block
            * self.config.audio_channel) as i32;
        if first_flat.shape()[1] != expect_first {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "first-frame audio window flattens to {} features, audio_proj.proj1 expects {expect_first}",
                    first_flat.shape()[1]
                ),
            ));
        }
        if latter_flat.shape()[1] != expect_vf {
            return Err(Error::new(
                "longcat_avatar DiT audio",
                format!(
                    "re-windowed audio flattens to {} features, audio_proj.proj1_vf expects {expect_vf} (latter shape {:?})",
                    latter_flat.shape()[1],
                    latter.shape()
                ),
            ));
        }
        let ae = relu(
            &self.linear("audio_proj.proj1", &first_flat, stream)?,
            stream,
        )?;
        let ae_vf = relu(
            &self.linear("audio_proj.proj1_vf", &latter_flat, stream)?,
            stream,
        )?;
        let intermediate = self.config.intermediate_dim as i32;
        let ae = ae.reshape(&[first.shape()[0], first.shape()[1], intermediate], stream)?;
        let ae_vf = ae_vf.reshape(
            &[latter.shape()[0], latter.shape()[1], intermediate],
            stream,
        )?;
        let joined = Array::concatenate_axis(&[ae, ae_vf], 1, stream)?;
        let s = joined.shape();
        let flat = joined.reshape(&[s[0] * s[1], intermediate], stream)?;
        let mid = relu(&self.linear("audio_proj.proj2", &flat, stream)?, stream)?;
        let ctx = self.linear("audio_proj.proj3", &mid, stream)?.reshape(
            &[
                s[0] * s[1],
                self.config.context_tokens as i32,
                self.config.output_dim as i32,
            ],
            stream,
        )?;
        // Output LayerNorm (`audio_proj.norm`).
        let normed = ctx.layer_norm(1e-5, stream)?;
        let scaled = Array::multiply(
            &normed,
            self.weights.tensor("audio_proj.norm.weight")?,
            stream,
        )?;
        let ctx = Array::add(
            &scaled,
            self.weights.tensor("audio_proj.norm.bias")?,
            stream,
        )?;
        ctx.reshape(
            &[
                s[0],
                s[1],
                self.config.context_tokens as i32,
                self.config.output_dim as i32,
            ],
            stream,
        )
    }

    // ------------------------------------------------------------------
    // block
    // ------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        index: usize,
        x: &Array,
        caption: &Array,
        text_lengths: &[usize],
        temb: &Array,
        audio: &Array,
        rope: &RopeTables,
        grid: [i32; 3],
        tokens_per_frame: i32,
        num_cond_latents: usize,
        dtype: Dtype,
        stream: &Stream,
    ) -> Result<Array> {
        let prefix = format!("blocks.{index}");
        let hidden = self.config.hidden_size as i32;
        let b = x.shape()[0];
        let n = x.shape()[1];
        let t = grid[0];
        let ncl = num_cond_latents as i32;

        // ---- AdaLN params (F32) ----
        let ada = self
            .linear(
                &format!("{prefix}.adaLN_modulation.1"),
                &temb.silu(stream)?,
                stream,
            )?
            .astype(Dtype::Float32, stream)?
            .expand_dims(2, stream)?; // [B, T, 1, 6*C]
        let adas = ada.split_sections(
            &[hidden, 2 * hidden, 3 * hidden, 4 * hidden, 5 * hidden],
            -1,
            stream,
        )?;
        let [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
            <[Array; 6]>::try_from(adas).map_err(|_| {
                Error::new("longcat_avatar DiT", "adaLN split did not produce 6 parts")
            })?;

        let audio_t = if ncl > 0 {
            let s = temb.shape();
            temb.slice(&[0, ncl, 0], &s, &[1, 1, 1], stream)?
        } else {
            temb.clone()
        };
        let audio_ada = self
            .linear(
                &format!("{prefix}.audio_adaLN_modulation.1"),
                &audio_t.silu(stream)?,
                stream,
            )?
            .astype(Dtype::Float32, stream)?
            .expand_dims(2, stream)?;
        let audio_parts = audio_ada.split_sections(&[hidden, 2 * hidden], -1, stream)?;
        let [audio_shift, audio_scale, audio_gate] =
            <[Array; 3]>::try_from(audio_parts).map_err(|_| {
                Error::new(
                    "longcat_avatar DiT",
                    "audio adaLN split did not produce 3 parts",
                )
            })?;

        // ---- self-attn with 3D RoPE ----
        let modulated = modulate(&shift_msa, &scale_msa, x, [b, t, n / t, hidden], stream)?;
        let attn_out = self
            .self_attention(
                &prefix,
                &modulated,
                rope,
                grid,
                tokens_per_frame,
                ncl,
                dtype,
                stream,
            )
            .map_err(|e| stage_err(index, "self_attn", e))?;
        let x = gated_residual(x, &attn_out, &gate_msa, [b, t, n / t, hidden], stream)
            .map_err(|e| stage_err(index, "self_attn_residual", e))?;

        // ---- text cross-attn (residual only) ----
        let normed = self
            .layer_norm_affine(&format!("{prefix}.pre_crs_attn_norm"), &x, stream)
            .map_err(|e| stage_err(index, "text_norm", e))?;
        let x = self
            .text_cross_attention(
                &prefix,
                &x,
                &normed,
                caption,
                text_lengths,
                tokens_per_frame,
                ncl,
                stream,
            )
            .map_err(|e| stage_err(index, "text_cross_attn", e))?;

        // ---- audio cross-attn (AdaLN-gated output) ----
        let normed = self
            .layer_norm_affine(&format!("{prefix}.pre_video_crs_attn_norm"), &x, stream)
            .map_err(|e| stage_err(index, "audio_norm", e))?;
        let x = self
            .audio_cross_attention(
                &prefix,
                &x,
                &normed,
                audio,
                grid,
                tokens_per_frame,
                ncl,
                &audio_shift,
                &audio_scale,
                &audio_gate,
                stream,
            )
            .map_err(|e| stage_err(index, "audio_cross_attn", e))?;

        // ---- FFN (SwiGLU) ----
        let modulated = modulate(&shift_mlp, &scale_mlp, &x, [b, t, n / t, hidden], stream)
            .map_err(|e| stage_err(index, "ffn_modulate", e))?;
        let ffn_out = self
            .ffn(&prefix, &modulated, stream)
            .map_err(|e| stage_err(index, "ffn", e))?;
        gated_residual(&x, &ffn_out, &gate_mlp, [b, t, n / t, hidden], stream)
            .map_err(|e| stage_err(index, "ffn_residual", e))
    }

    fn self_attention(
        &self,
        prefix: &str,
        x: &Array,
        rope: &RopeTables,
        grid: [i32; 3],
        tokens_per_frame: i32,
        ncl: i32,
        dtype: Dtype,
        stream: &Stream,
    ) -> Result<Array> {
        let b = x.shape()[0];
        let n = x.shape()[1];
        let heads = self.config.num_heads as i32;
        let head_dim = self.config.head_dim() as i32;
        let qkv = self.linear(&format!("{prefix}.attn.qkv"), x, stream)?;
        let qkv = qkv
            .reshape(&[b, n, 3, heads, head_dim], stream)?
            .transpose_axes(&[2, 0, 3, 1, 4], stream)?;
        let q = qkv
            .slice(
                &[0, 0, 0, 0, 0],
                &[1, b, heads, n, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let k = qkv
            .slice(
                &[1, 0, 0, 0, 0],
                &[2, b, heads, n, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let v = qkv
            .slice(
                &[2, 0, 0, 0, 0],
                &[3, b, heads, n, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let q = rms_norm_head(
            &q,
            self.weights
                .tensor(&format!("{prefix}.attn.q_norm.weight"))?,
            stream,
        )?;
        let k = rms_norm_head(
            &k,
            self.weights
                .tensor(&format!("{prefix}.attn.k_norm.weight"))?,
            stream,
        )?;
        let (q, k) = rope.apply(&q, &k, stream)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let sdpa = |q: &Array, k: &Array, v: &Array| -> Result<Array> {
            Array::scaled_dot_product_attention(q, k, v, scale, "", None, stream)
        };
        let out = if ncl > 0 {
            let cond = ncl * tokens_per_frame;
            let q_c = q.slice(
                &[0, 0, 0, 0],
                &[b, heads, cond, head_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let k_c = k.slice(
                &[0, 0, 0, 0],
                &[b, heads, cond, head_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let v_c = v.slice(
                &[0, 0, 0, 0],
                &[b, heads, cond, head_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let x_c = sdpa(&q_c, &k_c, &v_c)?;
            let q_n = q.slice(
                &[0, 0, cond, 0],
                &[b, heads, n, head_dim],
                &[1, 1, 1, 1],
                stream,
            )?;
            let x_n = sdpa(&q_n, &k, &v)?;
            Array::concatenate_axis(&[x_c, x_n], 2, stream)?
        } else {
            sdpa(&q, &k, &v)?
        };
        let out = out
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[b, n, heads * head_dim], stream)?
            .astype(dtype, stream)?;
        self.linear(&format!("{prefix}.attn.proj"), &out, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn text_cross_attention(
        &self,
        prefix: &str,
        residual: &Array,
        normed: &Array,
        caption: &Array,
        text_lengths: &[usize],
        tokens_per_frame: i32,
        ncl: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let b = normed.shape()[0];
        let n = normed.shape()[1];
        let hidden = self.config.hidden_size as i32;
        let heads = self.config.num_heads as i32;
        let head_dim = self.config.head_dim() as i32;
        let start = if ncl > 0 { ncl * tokens_per_frame } else { 0 };
        let x_noise = normed.slice(&[0, start, 0], &normed.shape(), &[1, 1, 1], stream)?;
        // Pack visual queries across batch: [1, B*N, C]
        let x_packed = x_noise.reshape(&[1, b * (n - start), hidden], stream)?;
        let q = self
            .linear(&format!("{prefix}.cross_attn.q_linear"), &x_packed, stream)?
            .reshape(&[1, b * (n - start), heads, head_dim], stream)?;
        let q = rms_norm_head(
            &q,
            self.weights
                .tensor(&format!("{prefix}.cross_attn.q_norm.weight"))?,
            stream,
        )?;
        // Pack text K/V: take each row's valid tokens and concatenate.
        let mut kv_parts = Vec::with_capacity(text_lengths.len());
        for (row, valid) in text_lengths.iter().enumerate() {
            let row_embeds = caption.slice(
                &[row as i32, 0, 0, 0],
                &[row as i32 + 1, 1, *valid as i32, hidden],
                &[1, 1, 1, 1],
                stream,
            )?;
            kv_parts.push(row_embeds);
        }
        let packed_text = Array::concatenate_axis(&kv_parts, 2, stream)?.reshape(
            &[1, text_lengths.iter().sum::<usize>() as i32, hidden],
            stream,
        )?;
        let kv = self.linear(
            &format!("{prefix}.cross_attn.kv_linear"),
            &packed_text,
            stream,
        )?;
        let total_kv = kv.shape()[1];
        let kv = kv
            .reshape(&[1, total_kv, 2, heads, head_dim], stream)?
            .transpose_axes(&[2, 0, 3, 1, 4], stream)?;
        let k = kv
            .slice(
                &[0, 0, 0, 0, 0],
                &[1, 1, heads, total_kv, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let v = kv
            .slice(
                &[1, 0, 0, 0, 0],
                &[2, 1, heads, total_kv, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let k = rms_norm_head(
            &k,
            self.weights
                .tensor(&format!("{prefix}.cross_attn.k_norm.weight"))?,
            stream,
        )?;
        // Block-diagonal additive mask so each batch row only sees its own text.
        let mut mask = vec![f32::NEG_INFINITY; (b * (n - start) * total_kv) as usize];
        let mut q_offset = 0usize;
        let mut kv_offset = 0usize;
        for valid in text_lengths {
            for qi in 0..(n - start) as usize {
                let row_base = (q_offset + qi) * total_kv as usize;
                for ki in 0..*valid {
                    mask[row_base + kv_offset + ki] = 0.0;
                }
            }
            q_offset += (n - start) as usize;
            kv_offset += valid;
        }
        let mask = Array::from_f32_slice(&mask, &[1, 1, b * (n - start), total_kv])?
            .astype(q.dtype(), stream)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = packed_text_attention(&q, &k, &v, &mask, scale, stream)?
            .reshape(&[b, n - start, hidden], stream)?;
        let out = self.linear(&format!("{prefix}.cross_attn.proj"), &out, stream)?;
        let out = if ncl > 0 {
            let zeros = Array::from_f32_slice(&[0.0], &[])?
                .astype(out.dtype(), stream)?
                .broadcast_to(&[b, start, hidden], stream)?;
            Array::concatenate_axis(&[zeros, out], 1, stream)?
        } else {
            out
        };
        let out = out.astype(Dtype::Float32, stream)?;
        let residual = residual.astype(Dtype::Float32, stream)?;
        let summed = Array::add(&residual, &out, stream)?;
        summed.astype(normed.dtype(), stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn audio_cross_attention(
        &self,
        prefix: &str,
        x: &Array,
        normed: &Array,
        audio: &Array,
        grid: [i32; 3],
        tokens_per_frame: i32,
        ncl: i32,
        audio_shift: &Array,
        audio_scale: &Array,
        audio_gate: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let b = normed.shape()[0];
        let n = normed.shape()[1];
        let hidden = self.config.hidden_size as i32;
        let heads = self.config.num_heads as i32;
        let head_dim = self.config.head_dim() as i32;
        let t = grid[0];
        let n_noise_frames = t - ncl;
        let start = ncl * tokens_per_frame;
        let x_noise = normed.slice(&[0, start, 0], &normed.shape(), &[1, 1, 1], stream)?;
        // Per-frame queries: [B*T_noise, S, C]
        let x_frames = x_noise.reshape(&[b * n_noise_frames, tokens_per_frame, hidden], stream)?;
        let q = self
            .linear(
                &format!("{prefix}.audio_cross_attn.q_linear"),
                &x_frames,
                stream,
            )?
            .reshape(
                &[b * n_noise_frames, tokens_per_frame, heads, head_dim],
                stream,
            )?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let q = rms_norm_head(
            &q,
            self.weights
                .tensor(&format!("{prefix}.audio_cross_attn.q_norm.weight"))?,
            stream,
        )?;
        // Drop the cond frames from the per-frame audio tokens.
        let audio_noise = if ncl > 0 {
            let per_frame = audio.reshape(
                &[
                    b,
                    t,
                    self.config.context_tokens as i32,
                    self.config.output_dim as i32,
                ],
                stream,
            )?;
            let kept =
                per_frame.slice(&[0, ncl, 0, 0], &per_frame.shape(), &[1, 1, 1, 1], stream)?;
            kept.reshape(
                &[
                    b * n_noise_frames,
                    self.config.context_tokens as i32,
                    self.config.output_dim as i32,
                ],
                stream,
            )?
        } else {
            audio.clone()
        };
        let kv = self.linear(
            &format!("{prefix}.audio_cross_attn.kv_linear"),
            &audio_noise,
            stream,
        )?;
        let tokens = kv.shape()[1];
        let kv = kv
            .reshape(&[b * n_noise_frames, tokens, 2, heads, head_dim], stream)?
            .transpose_axes(&[2, 0, 3, 1, 4], stream)?;
        let k = kv
            .slice(
                &[0, 0, 0, 0, 0],
                &[1, b * n_noise_frames, heads, tokens, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let v = kv
            .slice(
                &[1, 0, 0, 0, 0],
                &[2, b * n_noise_frames, heads, tokens, head_dim],
                &[1, 1, 1, 1, 1],
                stream,
            )?
            .squeeze_axis(0, stream)?;
        let k = rms_norm_head(
            &k,
            self.weights
                .tensor(&format!("{prefix}.audio_cross_attn.k_norm.weight"))?,
            stream,
        )?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = Array::scaled_dot_product_attention(&q, &k, &v, scale, "", None, stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?
            .reshape(&[b * n_noise_frames, tokens_per_frame, hidden], stream)?;
        let out = self.linear(&format!("{prefix}.audio_cross_attn.proj"), &out, stream)?;
        // AdaLN-gate the noise region, zero the cond region, add in F32.
        let out = out.reshape(&[b, n_noise_frames * tokens_per_frame, hidden], stream)?;
        let gated = gated_residual_audio(
            &out,
            audio_shift,
            audio_scale,
            audio_gate,
            [b, n_noise_frames, tokens_per_frame, hidden],
            stream,
        )?;
        let gated = if ncl > 0 {
            let zeros = Array::from_f32_slice(&[0.0], &[])?
                .astype(Dtype::Float32, stream)?
                .broadcast_to(&[b, start, hidden], stream)?;
            Array::concatenate_axis(&[zeros, gated], 1, stream)?
        } else {
            gated
        };
        let summed = Array::add(&x.astype(Dtype::Float32, stream)?, &gated, stream)?;
        summed.astype(x.dtype(), stream)
    }

    fn ffn(&self, prefix: &str, x: &Array, stream: &Stream) -> Result<Array> {
        let gate = self
            .linear(&format!("{prefix}.ffn.w1"), x, stream)?
            .silu(stream)?;
        let up = self.linear(&format!("{prefix}.ffn.w3"), x, stream)?;
        let mid = Array::multiply(&gate, &up, stream)?;
        self.linear(&format!("{prefix}.ffn.w2"), &mid, stream)
    }

    fn layer_norm_affine(&self, prefix: &str, x: &Array, stream: &Stream) -> Result<Array> {
        let weight = self.weights.tensor(&format!("{prefix}.weight"))?;
        let bias = self.weights.tensor(&format!("{prefix}.bias"))?;
        let normed = x.astype(Dtype::Float32, stream)?.layer_norm(1e-6, stream)?;
        let scaled = Array::multiply(&normed, &weight.astype(Dtype::Float32, stream)?, stream)?;
        let shifted = Array::add(&scaled, &bias.astype(Dtype::Float32, stream)?, stream)?;
        shifted.astype(x.dtype(), stream)
    }

    // ------------------------------------------------------------------
    // final head
    // ------------------------------------------------------------------

    fn final_layer(
        &self,
        x: &Array,
        temb: &Array,
        grid: [i32; 3],
        b: i32,
        hidden: i32,
        stream: &Stream,
    ) -> Result<Array> {
        let n = x.shape()[1];
        let t = grid[0];
        let ada = self
            .linear(
                "final_layer.adaLN_modulation.1",
                &temb.silu(stream)?,
                stream,
            )?
            .astype(Dtype::Float32, stream)?
            .expand_dims(2, stream)?;
        let parts = ada.split_sections(&[hidden], -1, stream)?;
        let [shift, scale] = <[Array; 2]>::try_from(parts).map_err(|_| {
            Error::new(
                "longcat_avatar DiT",
                "final adaLN split did not produce 2 parts",
            )
        })?;
        let modulated = modulate(&shift, &scale, x, [b, t, n / t, hidden], stream)?;
        self.linear("final_layer.linear", &modulated, stream)
    }
}

/// Tag a DiT sub-stage so a broadcast/shape failure names its block and stage
/// instead of surfacing as an anonymous mlx_add error.
fn stage_err(block: usize, stage: &str, error: Error) -> Error {
    Error::new(
        "longcat_avatar DiT",
        format!("block {block} {stage}: {error}"),
    )
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// AdaLN-Zero modulation in F32: `ln(x) * (scale + 1) + shift` per frame.
/// `frame_shape` is `[B, T, tokens_per_frame, C]`.
fn modulate(
    shift: &Array,
    scale: &Array,
    x: &Array,
    frame_shape: [i32; 4],
    stream: &Stream,
) -> Result<Array> {
    let dtype = x.dtype();
    let [b, t, s, c] = frame_shape;
    let normed = x
        .reshape(&[b, t, s, c], stream)?
        .astype(Dtype::Float32, stream)?
        .layer_norm(1e-6, stream)?;
    let one = Array::scalar_f32(1.0)?;
    let scale = Array::add(scale, &one, stream)?;
    let out = Array::add(&Array::multiply(&normed, &scale, stream)?, shift, stream)?;
    out.reshape(&[b, t * s, c], stream)?.astype(dtype, stream)
}

/// F32 residual add with AdaLN gate: `x + gate * branch`.
fn gated_residual(
    x: &Array,
    branch: &Array,
    gate: &Array,
    frame_shape: [i32; 4],
    stream: &Stream,
) -> Result<Array> {
    let dtype = x.dtype();
    let [b, t, s, c] = frame_shape;
    let gated = Array::multiply(
        &branch
            .reshape(&[b, t, s, c], stream)?
            .astype(Dtype::Float32, stream)?,
        &gate.astype(Dtype::Float32, stream)?,
        stream,
    )?;
    let summed = Array::add(
        &x.astype(Dtype::Float32, stream)?,
        &gated.reshape(&[b, t * s, c], stream)?,
        stream,
    )?;
    summed.astype(dtype, stream)
}

/// Audio branch gate: F32 LN → `× (scale + 1) + shift` → `× gate`.
fn gated_residual_audio(
    branch: &Array,
    shift: &Array,
    scale: &Array,
    gate: &Array,
    frame_shape: [i32; 4],
    stream: &Stream,
) -> Result<Array> {
    let [b, t, s, c] = frame_shape;
    let normed = branch
        .reshape(&[b, t, s, c], stream)?
        .astype(Dtype::Float32, stream)?
        .layer_norm(1e-6, stream)?;
    let one = Array::scalar_f32(1.0)?;
    let scale = Array::add(scale, &one, stream)?;
    let modulated = Array::add(&Array::multiply(&normed, &scale, stream)?, shift, stream)?;
    Array::multiply(&modulated, gate, stream)?.reshape(&[b, t * s, c], stream)
}

fn packed_text_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: &Array,
    scale: f32,
    stream: &Stream,
) -> Result<Array> {
    let q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    Array::scaled_dot_product_attention(&q, k, v, scale, "", Some(mask), stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)
}

/// Per-head RMS norm in F32 with the checkpoint's per-head-dim weight.
fn rms_norm_head(x: &Array, weight: &Array, stream: &Stream) -> Result<Array> {
    let dtype = x.dtype();
    let normed = x
        .astype(Dtype::Float32, stream)?
        .rms_norm(None, 1e-6, stream)?;
    let scaled = Array::multiply(&normed, &weight.astype(Dtype::Float32, stream)?, stream)?;
    scaled.astype(dtype, stream)
}

#[cfg(test)]
mod tests {
    use super::packed_text_attention;
    use danmo_mlx::{Array, Stream};

    #[test]
    fn packed_text_attention_uses_heads_first_layout() {
        let stream = Stream::cpu().unwrap();
        let q = Array::from_f32_slice(&vec![0.1; 6 * 2 * 4], &[1, 6, 2, 4]).unwrap();
        let k = Array::from_f32_slice(&vec![0.2; 2 * 5 * 4], &[1, 2, 5, 4]).unwrap();
        let v = Array::from_f32_slice(&vec![0.3; 2 * 5 * 4], &[1, 2, 5, 4]).unwrap();
        let mask = Array::from_f32_slice(&vec![0.0; 6 * 5], &[1, 1, 6, 5]).unwrap();

        let out = packed_text_attention(&q, &k, &v, &mask, 0.5, &stream).unwrap();

        assert_eq!(out.shape(), [1, 6, 2, 4]);
        assert!(out
            .to_vec_f32(&stream)
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
}
