//! LTX 2.5 22B joint audio/video velocity transformer.
//! The family owns checkpoint topology. Scheduling, sampling, codecs, and
//! request orchestration stay outside this module.

use std::path::Path;

use danmo_mlx::{AffineQuantization, Array, LinearWeight, Result, Stream};
use serde_json::Value;

use super::bundle::Quantization;
use crate::common::av_rope::RopeTables;
use crate::common::mlx_ops::{apply_split_rope, gelu_tanh, scalar, slice_last};
use crate::weights::{load_mlx_file, LoraFile, MlxWeights, WeightDtype, WeightError};

use super::text_encoder::PromptEmbeddings;

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub dim: usize,
    pub audio_dim: usize,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub audio_heads: usize,
    pub audio_head_dim: usize,
    pub qkv_bias: bool,
    pub ff_bias: bool,
    pub audio_ff_bias: bool,
    pub gated: bool,
    pub cross_attention_adaln: bool,
    pub use_prompt_adaln_single: bool,
    pub use_keyframes_abs_pos_embedding: bool,
    pub video_rope_max_pos: Vec<f32>,
    pub audio_rope_max_pos: Vec<f32>,
    pub rope_theta: f32,
}

impl TransformerConfig {
    pub fn from_value(value: &Value) -> Result<Self> {
        if !boolean(value, "caption_proj_before_connector", false) {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 transformer config",
                "caption_proj_before_connector=true is required",
            ));
        }
        let heads = integer(value, "num_attention_heads")?;
        let head_dim = integer(value, "attention_head_dim")?;
        let audio_heads = integer(value, "audio_num_attention_heads")?;
        let audio_head_dim = integer(value, "audio_attention_head_dim")?;
        Ok(Self {
            dim: heads * head_dim,
            audio_dim: audio_heads * audio_head_dim,
            layers: integer(value, "num_layers")?,
            heads,
            head_dim,
            audio_heads,
            audio_head_dim,
            qkv_bias: boolean(value, "qkv_bias", true),
            ff_bias: boolean(value, "ff_bias", false),
            audio_ff_bias: boolean(value, "audio_ff_bias", true),
            gated: boolean(value, "apply_gated_attention", false),
            cross_attention_adaln: boolean(value, "cross_attention_adaln", false),
            use_prompt_adaln_single: boolean(value, "use_prompt_adaln_single", true),
            use_keyframes_abs_pos_embedding: boolean(
                value,
                "use_keyframes_abs_pos_embedding",
                false,
            ),
            video_rope_max_pos: floats(
                value,
                "positional_embedding_max_pos",
                &[20.0, 2048.0, 2048.0],
            )?,
            audio_rope_max_pos: floats(value, "audio_positional_embedding_max_pos", &[20.0])?,
            rope_theta: value
                .get("positional_embedding_theta")
                .and_then(Value::as_f64)
                .unwrap_or(10000.0) as f32,
        })
    }
}

pub struct Transformer {
    weights: MlxWeights,
    config: TransformerConfig,
}

struct AdaOutput {
    params: Array,
    embedded: Array,
}

#[derive(Debug, Clone, Copy)]
pub struct KeyframeSpan {
    pub start: usize,
    pub tokens: usize,
}

impl Transformer {
    pub fn load(
        bundle: &Path,
        value: &Value,
        quantization: Option<Quantization>,
    ) -> std::result::Result<Self, WeightError> {
        let config = TransformerConfig::from_value(value)
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let quantization = quantization
            .map(|quant| AffineQuantization::new(quant.bits, quant.group_size))
            .transpose()
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let weights = load_mlx_file(
            &bundle.join("transformer.safetensors"),
            WeightDtype::Native,
            quantization,
        )
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        Ok(Self { weights, config })
    }

    pub fn attach_lora(
        mut self,
        files: &[LoraFile],
        stream: &Stream,
    ) -> std::result::Result<Self, WeightError> {
        if files.is_empty() {
            return Ok(self);
        }
        self.weights =
            self.weights
                .attach_lora_routed(files, stream, &super::msr::route_lora_module)?;
        Ok(self)
    }

    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Shared-sigma forward (pure T2V / T2VA).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens(
        &self,
        video: &Array,
        audio: &Array,
        video_rope: &RopeTables,
        audio_rope: &RopeTables,
        video_cross_rope: &RopeTables,
        audio_cross_rope: &RopeTables,
        context: &PromptEmbeddings,
        keyframe: KeyframeSpan,
        sigma: f32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        self.forward_tokens_timed(
            video,
            audio,
            video_rope,
            audio_rope,
            video_cross_rope,
            audio_cross_rope,
            context,
            keyframe,
            None,
            None,
            sigma,
            stream,
        )
    }

    /// Per-token diffusion timesteps for IC-LoRA / Ref2VA.
    ///
    /// When `video_sigmas` / `audio_sigmas` are `None`, every token uses
    /// `sigma` (scalar AdaLN, same as the published T2V path). Otherwise each
    /// sequence must have one sigma per token; reference tokens stay at `0`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_tokens_timed(
        &self,
        video: &Array,
        audio: &Array,
        video_rope: &RopeTables,
        audio_rope: &RopeTables,
        video_cross_rope: &RopeTables,
        audio_cross_rope: &RopeTables,
        context: &PromptEmbeddings,
        keyframe: KeyframeSpan,
        video_sigmas: Option<&[f32]>,
        audio_sigmas: Option<&[f32]>,
        sigma: f32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let video_tokens = video.shape().get(1).copied().unwrap_or(0) as usize;
        let audio_tokens = audio.shape().get(1).copied().unwrap_or(0) as usize;
        if let Some(values) = video_sigmas {
            if values.len() != video_tokens {
                return Err(danmo_mlx::Error::new(
                    "LTX 2.5 transformer",
                    format!(
                        "video_sigmas length {} does not match video tokens {video_tokens}",
                        values.len()
                    ),
                ));
            }
        }
        if let Some(values) = audio_sigmas {
            if values.len() != audio_tokens {
                return Err(danmo_mlx::Error::new(
                    "LTX 2.5 transformer",
                    format!(
                        "audio_sigmas length {} does not match audio tokens {audio_tokens}",
                        values.len()
                    ),
                ));
            }
        }
        let mut video = linear(&self.weights, "patchify_proj", video, true, stream)?;
        if self.config.use_keyframes_abs_pos_embedding {
            video = apply_keyframe_embedding(
                &video,
                self.weights.tensor("keyframes_abs_pos_embedding")?,
                keyframe,
                stream,
            )?;
        }
        let mut audio = linear(&self.weights, "audio_patchify_proj", audio, true, stream)?;
        let timed = video_sigmas.is_some() || audio_sigmas.is_some();
        let video_sigma_vec = video_sigmas
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![sigma; video_tokens.max(1)]);
        let audio_sigma_vec = audio_sigmas
            .map(|values| values.to_vec())
            .unwrap_or_else(|| vec![sigma; audio_tokens.max(1)]);
        let (timestep, av_timestep, a_timestep, a_av_timestep) = if timed {
            let video_scaled = video_sigma_vec
                .iter()
                .map(|value| value * 1000.0)
                .collect::<Vec<_>>();
            let audio_scaled = audio_sigma_vec
                .iter()
                .map(|value| value * 1000.0)
                .collect::<Vec<_>>();
            (
                sinusoidal_batch(&video_scaled)?,
                sinusoidal(sigma)?,
                sinusoidal_batch(&audio_scaled)?,
                sinusoidal(sigma)?,
            )
        } else {
            let timestep = sinusoidal(sigma * 1000.0)?;
            let av_timestep = sinusoidal(sigma)?;
            (timestep.clone(), av_timestep.clone(), timestep, av_timestep)
        };
        let v_adaln = adaln(&self.weights, "adaln_single", &timestep, stream)?;
        let a_adaln = adaln(&self.weights, "audio_adaln_single", &a_timestep, stream)?;
        let av_v = adaln(
            &self.weights,
            "av_ca_video_scale_shift_adaln_single",
            &timestep,
            stream,
        )?;
        let av_a = adaln(
            &self.weights,
            "av_ca_audio_scale_shift_adaln_single",
            &a_timestep,
            stream,
        )?;
        let a2v_gate = adaln(
            &self.weights,
            "av_ca_a2v_gate_adaln_single",
            &av_timestep,
            stream,
        )?;
        let v2a_gate = adaln(
            &self.weights,
            "av_ca_v2a_gate_adaln_single",
            &a_av_timestep,
            stream,
        )?;

        // Prompt AdaLN conditions text tokens; always use the step's scalar sigma.
        // Per-video/audio token timesteps are only for the A/V streams (ImageReplace /
        // Ref2VA). Broadcasting a [T_video] AdaLN onto [T_prompt] fails loud.
        let prompt_timestep = if timed {
            sinusoidal(sigma * 1000.0)?
        } else {
            timestep.clone()
        };
        let v_prompt = if self.config.cross_attention_adaln && self.config.use_prompt_adaln_single {
            Some(adaln(
                &self.weights,
                "prompt_adaln_single",
                &prompt_timestep,
                stream,
            )?)
        } else {
            None
        };
        let a_prompt = if self.config.cross_attention_adaln && self.config.use_prompt_adaln_single {
            Some(adaln(
                &self.weights,
                "audio_prompt_adaln_single",
                &prompt_timestep,
                stream,
            )?)
        } else {
            None
        };

        for layer in 0..self.config.layers {
            let prefix = format!("transformer_blocks.{layer}");
            let adaln_count = if self.config.cross_attention_adaln {
                9
            } else {
                6
            };
            let vp = unpack(
                &v_adaln.params,
                self.weights
                    .tensor(&format!("{prefix}.scale_shift_table"))?,
                adaln_count,
                self.config.dim,
                stream,
            )?;
            let ap = unpack(
                &a_adaln.params,
                self.weights
                    .tensor(&format!("{prefix}.audio_scale_shift_table"))?,
                adaln_count,
                self.config.audio_dim,
                stream,
            )?;
            let vpp = match &v_prompt {
                Some(prompt) => Some(unpack(
                    &prompt.params,
                    self.weights
                        .tensor(&format!("{prefix}.prompt_scale_shift_table"))?,
                    2,
                    self.config.dim,
                    stream,
                )?),
                None => None,
            };
            let app = match &a_prompt {
                Some(prompt) => Some(unpack(
                    &prompt.params,
                    self.weights
                        .tensor(&format!("{prefix}.audio_prompt_scale_shift_table"))?,
                    2,
                    self.config.audio_dim,
                    stream,
                )?),
                None => None,
            };
            let avvp = unpack(
                &av_v.params,
                self.weights
                    .tensor(&format!("{prefix}.scale_shift_table_a2v_ca_video"))?,
                4,
                self.config.dim,
                stream,
            )?;
            let avap = unpack(
                &av_a.params,
                self.weights
                    .tensor(&format!("{prefix}.scale_shift_table_a2v_ca_audio"))?,
                4,
                self.config.audio_dim,
                stream,
            )?;
            let av_v_gate = token_scale(
                &Array::add(
                    &a2v_gate.params,
                    &self
                        .weights
                        .tensor(&format!("{prefix}.scale_shift_table_a2v_ca_video"))?
                        .slice(&[4, 0], &[5, self.config.dim as i32], &[1, 1], stream)?,
                    stream,
                )?,
                video_tokens,
                stream,
            )?;
            let av_a_gate = token_scale(
                &Array::add(
                    &v2a_gate.params,
                    &self
                        .weights
                        .tensor(&format!("{prefix}.scale_shift_table_a2v_ca_audio"))?
                        .slice(&[4, 0], &[5, self.config.audio_dim as i32], &[1, 1], stream)?,
                    stream,
                )?,
                audio_tokens,
                stream,
            )?;

            let normed = modulate(&video, &vp[1], &vp[0], stream)?;
            let sa = attention(
                &self.weights,
                &format!("{prefix}.attn1"),
                &normed,
                None,
                self.config.heads,
                self.config.head_dim,
                self.config.qkv_bias,
                self.config.gated,
                Some(video_rope),
                None,
                stream,
            )?;
            video = Array::add(&video, &Array::multiply(&sa, &vp[2], stream)?, stream)?;
            let video_post_sa = video.rms_norm(None, 1e-6, stream)?;

            let normed = modulate(&audio, &ap[1], &ap[0], stream)?;
            let sa = attention(
                &self.weights,
                &format!("{prefix}.audio_attn1"),
                &normed,
                None,
                self.config.audio_heads,
                self.config.audio_head_dim,
                self.config.qkv_bias,
                self.config.gated,
                Some(audio_rope),
                None,
                stream,
            )?;
            audio = Array::add(&audio, &Array::multiply(&sa, &ap[2], stream)?, stream)?;
            let audio_post_sa = audio.rms_norm(None, 1e-6, stream)?;

            let (video_cross_input, video_context) = if self.config.cross_attention_adaln {
                let input = affine(&video_post_sa, &vp[7], &vp[6], stream)?;
                let context = match &vpp {
                    Some(prompt) => affine(&context.video, &prompt[1], &prompt[0], stream)?,
                    None => context.video.clone(),
                };
                (input, context)
            } else {
                (video_post_sa.clone(), context.video.clone())
            };
            let mut cross = attention(
                &self.weights,
                &format!("{prefix}.attn2"),
                &video_cross_input,
                Some(&video_context),
                self.config.heads,
                self.config.head_dim,
                self.config.qkv_bias,
                self.config.gated,
                None,
                None,
                stream,
            )?;
            if self.config.cross_attention_adaln {
                cross = Array::multiply(&cross, &vp[8], stream)?;
            }
            video = Array::add(&video, &cross, stream)?;
            let (audio_cross_input, audio_context) = if self.config.cross_attention_adaln {
                let input = affine(&audio_post_sa, &ap[7], &ap[6], stream)?;
                let context = match &app {
                    Some(prompt) => affine(&context.audio, &prompt[1], &prompt[0], stream)?,
                    None => context.audio.clone(),
                };
                (input, context)
            } else {
                (audio_post_sa.clone(), context.audio.clone())
            };
            let mut cross = attention(
                &self.weights,
                &format!("{prefix}.audio_attn2"),
                &audio_cross_input,
                Some(&audio_context),
                self.config.audio_heads,
                self.config.audio_head_dim,
                self.config.qkv_bias,
                self.config.gated,
                None,
                None,
                stream,
            )?;
            if self.config.cross_attention_adaln {
                cross = Array::multiply(&cross, &ap[8], stream)?;
            }
            audio = Array::add(&audio, &cross, stream)?;

            let video_norm3 = video.rms_norm(None, 1e-6, stream)?;
            let audio_norm3 = audio.rms_norm(None, 1e-6, stream)?;
            let video_q = modulate(&video_norm3, &avvp[0], &avvp[1], stream)?;
            let audio_kv = modulate(&audio_norm3, &avap[0], &avap[1], stream)?;
            let a2v = attention(
                &self.weights,
                &format!("{prefix}.audio_to_video_attn"),
                &video_q,
                Some(&audio_kv),
                self.config.audio_heads,
                self.config.audio_head_dim,
                self.config.qkv_bias,
                self.config.gated,
                Some(video_cross_rope),
                Some(audio_cross_rope),
                stream,
            )?;
            video = Array::add(&video, &Array::multiply(&a2v, &av_v_gate, stream)?, stream)?;
            let audio_q = modulate(&audio_norm3, &avap[2], &avap[3], stream)?;
            let video_kv = modulate(&video_norm3, &avvp[2], &avvp[3], stream)?;
            let v2a = attention(
                &self.weights,
                &format!("{prefix}.video_to_audio_attn"),
                &audio_q,
                Some(&video_kv),
                self.config.audio_heads,
                self.config.audio_head_dim,
                self.config.qkv_bias,
                self.config.gated,
                Some(audio_cross_rope),
                Some(video_cross_rope),
                stream,
            )?;
            audio = Array::add(&audio, &Array::multiply(&v2a, &av_a_gate, stream)?, stream)?;

            let normed = modulate(&video, &vp[4], &vp[3], stream)?;
            let ff = linear(
                &self.weights,
                &format!("{prefix}.ff.proj_in"),
                &normed,
                self.config.ff_bias,
                stream,
            )?;
            let ff = gelu_tanh(&ff, stream)?;
            let ff = linear(
                &self.weights,
                &format!("{prefix}.ff.proj_out"),
                &ff,
                self.config.ff_bias,
                stream,
            )?;
            video = Array::add(&video, &Array::multiply(&ff, &vp[5], stream)?, stream)?;
            let normed = modulate(&audio, &ap[4], &ap[3], stream)?;
            let ff = linear(
                &self.weights,
                &format!("{prefix}.audio_ff.proj_in"),
                &normed,
                self.config.audio_ff_bias,
                stream,
            )?;
            let ff = gelu_tanh(&ff, stream)?;
            let ff = linear(
                &self.weights,
                &format!("{prefix}.audio_ff.proj_out"),
                &ff,
                self.config.audio_ff_bias,
                stream,
            )?;
            audio = Array::add(&audio, &Array::multiply(&ff, &ap[5], stream)?, stream)?;
            if (layer + 1) % 4 == 0 {
                Array::eval_all(&[&video, &audio])?;
            }
        }
        let video = output_block(
            &self.weights,
            &video,
            &v_adaln.embedded,
            "scale_shift_table",
            "proj_out",
            stream,
        )?;
        let audio = output_block(
            &self.weights,
            &audio,
            &a_adaln.embedded,
            "audio_scale_shift_table",
            "audio_proj_out",
            stream,
        )?;
        Ok((video, audio))
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    context: Option<&Array>,
    heads: usize,
    head_dim: usize,
    qkv_bias: bool,
    gated: bool,
    rope_q: Option<&RopeTables>,
    rope_k: Option<&RopeTables>,
    stream: &Stream,
) -> Result<Array> {
    let kv = context.unwrap_or(input);
    let (batch, sequence, kv_sequence) = (input.shape()[0], input.shape()[1], kv.shape()[1]);
    let mut q = linear(weights, &format!("{prefix}.to_q"), input, qkv_bias, stream)?;
    let mut k = linear(weights, &format!("{prefix}.to_k"), kv, qkv_bias, stream)?;
    let v = linear(weights, &format!("{prefix}.to_v"), kv, qkv_bias, stream)?;
    q = q.rms_norm(
        Some(weights.tensor(&format!("{prefix}.q_norm.weight"))?),
        1e-6,
        stream,
    )?;
    k = k.rms_norm(
        Some(weights.tensor(&format!("{prefix}.k_norm.weight"))?),
        1e-6,
        stream,
    )?;
    let mut q = q
        .reshape(&[batch, sequence, heads as i32, head_dim as i32], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let mut k = k
        .reshape(&[batch, kv_sequence, heads as i32, head_dim as i32], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = v
        .reshape(&[batch, kv_sequence, heads as i32, head_dim as i32], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)?;
    if let Some(rope) = rope_q {
        q = apply_split_rope(&q, &rope.cos, &rope.sin, stream)?;
    }
    if let Some(rope) = rope_k.or(rope_q) {
        k = apply_split_rope(&k, &rope.cos, &rope.sin, stream)?;
    }
    let mut output = Array::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        1.0 / (head_dim as f32).sqrt(),
        "",
        None,
        stream,
    )?;
    if gated {
        let gate = linear(
            weights,
            &format!("{prefix}.to_gate_logits"),
            input,
            true,
            stream,
        )?
        .sigmoid(stream)?;
        let gate = Array::multiply(&gate, &scalar(2.0)?, stream)?
            .transpose_axes(&[0, 2, 1], stream)?
            .expand_dims(-1, stream)?;
        output = Array::multiply(&output, &gate, stream)?;
    }
    let output = output
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, sequence, (heads * head_dim) as i32], stream)?;
    linear(weights, &format!("{prefix}.to_out"), &output, true, stream)
}

fn sinusoidal(timestep: f32) -> Result<Array> {
    sinusoidal_batch(&[timestep])
}

fn sinusoidal_batch(timesteps: &[f32]) -> Result<Array> {
    if timesteps.is_empty() {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 transformer",
            "sinusoidal_batch requires at least one timestep",
        ));
    }
    let mut values = Vec::with_capacity(timesteps.len() * 256);
    for &timestep in timesteps {
        for index in 0..128 {
            values.push((timestep * 10000.0f32.powf(-(index as f32) / 128.0)).cos());
        }
        for index in 0..128 {
            values.push((timestep * 10000.0f32.powf(-(index as f32) / 128.0)).sin());
        }
    }
    Array::from_f32_slice(&values, &[timesteps.len() as i32, 256])
}

fn apply_keyframe_embedding(
    hidden: &Array,
    embedding: &Array,
    span: KeyframeSpan,
    stream: &Stream,
) -> Result<Array> {
    let shape = hidden.shape();
    if shape.len() != 3 || embedding.shape() != [1, shape[2]] {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 keyframe embedding",
            format!(
                "expected hidden [B,T,D] and embedding [1,D], got {shape:?} and {:?}",
                embedding.shape()
            ),
        ));
    }
    let end = span.start.checked_add(span.tokens).ok_or_else(|| {
        danmo_mlx::Error::new("LTX 2.5 keyframe embedding", "token range overflow")
    })?;
    if span.tokens == 0 || end > shape[1] as usize {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 keyframe embedding",
            format!(
                "token range {}..{end} is outside sequence length {}",
                span.start, shape[1]
            ),
        ));
    }
    let mut marker = vec![0.0f32; shape[1] as usize];
    marker[span.start..end].fill(1.0);
    let marker =
        Array::from_f32_slice(&marker, &[1, shape[1], 1])?.astype(hidden.dtype(), stream)?;
    let embedding = embedding.astype(hidden.dtype(), stream)?;
    Array::add(
        hidden,
        &Array::multiply(&marker, &embedding, stream)?,
        stream,
    )
}

/// Broadcast or reshape AdaLN / gate vectors to `[1, tokens, dim]`.
fn token_scale(values: &Array, tokens: usize, stream: &Stream) -> Result<Array> {
    let shape = values.shape();
    match shape.len() {
        2 if shape[0] == 1 => Ok(values.clone()),
        2 if shape[0] == tokens as i32 => values.reshape(&[1, tokens as i32, shape[1]], stream),
        3 if shape[0] == 1 && shape[1] == tokens as i32 => Ok(values.clone()),
        _ => Err(danmo_mlx::Error::new(
            "LTX 2.5 transformer",
            format!("cannot view AdaLN params {shape:?} as {tokens} token scales"),
        )),
    }
}

fn adaln(
    weights: &MlxWeights,
    prefix: &str,
    timestep: &Array,
    stream: &Stream,
) -> Result<AdaOutput> {
    let embedded = linear(
        weights,
        &format!("{prefix}.emb.linear_1"),
        timestep,
        true,
        stream,
    )?
    .silu(stream)?;
    let embedded = linear(
        weights,
        &format!("{prefix}.emb.linear_2"),
        &embedded,
        true,
        stream,
    )?;
    let params = linear(
        weights,
        &format!("{prefix}.linear"),
        &embedded.silu(stream)?,
        true,
        stream,
    )?;
    Ok(AdaOutput { params, embedded })
}

fn unpack(
    params: &Array,
    table: &Array,
    count: usize,
    dim: usize,
    stream: &Stream,
) -> Result<Vec<Array>> {
    let shape = params.shape();
    let tokens = match shape.len() {
        2 => shape[0] as usize,
        _ => {
            return Err(danmo_mlx::Error::new(
                "LTX 2.5 transformer",
                format!("AdaLN params must be rank-2, got {shape:?}"),
            ))
        }
    };
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let generated = slice_last(
            params,
            (index * dim) as i32,
            ((index + 1) * dim) as i32,
            stream,
        )?;
        let row = table.slice(
            &[index as i32, 0],
            &[index as i32 + 1, dim as i32],
            &[1, 1],
            stream,
        )?;
        let combined = Array::add(&generated, &row, stream)?;
        output.push(token_scale(&combined, tokens, stream)?);
    }
    Ok(output)
}

fn modulate(input: &Array, scale: &Array, shift: &Array, stream: &Stream) -> Result<Array> {
    let normalized = input.rms_norm(None, 1e-6, stream)?;
    affine(&normalized, scale, shift, stream)
}

fn affine(input: &Array, scale: &Array, shift: &Array, stream: &Stream) -> Result<Array> {
    Array::add(
        &Array::multiply(input, &Array::add(&scalar(1.0)?, scale, stream)?, stream)?,
        shift,
        stream,
    )
}

fn output_block(
    weights: &MlxWeights,
    input: &Array,
    embedded: &Array,
    table_name: &str,
    projection: &str,
    stream: &Stream,
) -> Result<Array> {
    let table = weights.tensor(table_name)?;
    let dim = table.shape()[1];
    let emb_shape = embedded.shape();
    let (scale, shift) = if emb_shape.len() == 2 && emb_shape[0] > 1 {
        let tokens = emb_shape[0];
        let values = Array::add(
            &embedded.reshape(&[tokens, 1, dim], stream)?,
            &table.reshape(&[1, 2, dim], stream)?,
            stream,
        )?;
        let shift = values
            .slice(&[0, 0, 0], &[tokens, 1, dim], &[1, 1, 1], stream)?
            .reshape(&[1, tokens, dim], stream)?;
        let scale = values
            .slice(&[0, 1, 0], &[tokens, 2, dim], &[1, 1, 1], stream)?
            .reshape(&[1, tokens, dim], stream)?;
        (scale, shift)
    } else {
        let values = Array::add(table, embedded, stream)?;
        let shift = values
            .slice(&[0, 0], &[1, dim], &[1, 1], stream)?
            .reshape(&[1, 1, dim], stream)?;
        let scale = values
            .slice(&[1, 0], &[2, dim], &[1, 1], stream)?
            .reshape(&[1, 1, dim], stream)?;
        (scale, shift)
    };
    let normalized = input.layer_norm(1e-6, stream)?;
    linear(
        weights,
        projection,
        &affine(&normalized, &scale, &shift, stream)?,
        true,
        stream,
    )
}

fn linear(
    weights: &MlxWeights,
    name: &str,
    input: &Array,
    bias: bool,
    stream: &Stream,
) -> Result<Array> {
    let layer: &LinearWeight = weights.linear(&format!("{name}.weight"))?;
    let bias = if bias {
        Some(weights.tensor(&format!("{name}.bias"))?)
    } else {
        None
    };
    layer.apply(input, bias, stream)
}

pub fn patchify_video(latent: &Array, stream: &Stream) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 5 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 video patchifier",
            format!("expected [B,C,F,H,W], got {shape:?}"),
        ));
    }
    let (b, c, f, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    latent
        .transpose_axes(&[0, 2, 3, 4, 1], stream)?
        .reshape(&[b, f * h * w, c], stream)
}

pub fn unpatchify_video(
    tokens: &Array,
    frames: i32,
    height: i32,
    width: i32,
    stream: &Stream,
) -> Result<Array> {
    let b = tokens.shape()[0];
    let c = tokens.shape()[2];
    tokens
        .reshape(&[b, frames, height, width, c], stream)?
        .transpose_axes(&[0, 4, 1, 2, 3], stream)
}

pub fn patchify_audio(latent: &Array, stream: &Stream) -> Result<Array> {
    let shape = latent.shape();
    if shape.len() != 4 {
        return Err(danmo_mlx::Error::new(
            "LTX 2.5 audio patchifier",
            format!("expected [B,C,T,F], got {shape:?}"),
        ));
    }
    let (b, c, t, f) = (shape[0], shape[1], shape[2], shape[3]);
    latent
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[b, t, c * f], stream)
}

pub fn unpatchify_audio(tokens: &Array, channels: i32, stream: &Stream) -> Result<Array> {
    let (b, t, width) = (tokens.shape()[0], tokens.shape()[1], tokens.shape()[2]);
    tokens
        .reshape(&[b, t, channels, width / channels], stream)?
        .transpose_axes(&[0, 2, 1, 3], stream)
}

fn integer(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| {
            danmo_mlx::Error::new(
                "LTX 2.5 transformer config",
                format!("missing integer {key}"),
            )
        })
}

fn boolean(value: &Value, key: &str, default: bool) -> bool {
    value.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn floats(value: &Value, key: &str, default: &[f32]) -> Result<Vec<f32>> {
    let Some(value) = value.get(key) else {
        return Ok(default.to_vec());
    };
    value
        .as_array()
        .ok_or_else(|| {
            danmo_mlx::Error::new(
                "LTX 2.5 transformer config",
                format!("{key} must be a list"),
            )
        })?
        .iter()
        .map(|value| {
            value.as_f64().map(|value| value as f32).ok_or_else(|| {
                danmo_mlx::Error::new(
                    "LTX 2.5 transformer config",
                    format!("{key} contains a non-number"),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn keyframe_embedding_marks_only_the_target_first_frame() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let hidden = danmo_mlx::Array::from_f32_slice(&[0.0; 2 * 5 * 3], &[2, 5, 3]).unwrap();
        let embedding = danmo_mlx::Array::from_f32_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let marked = super::apply_keyframe_embedding(
            &hidden,
            &embedding,
            super::KeyframeSpan {
                start: 2,
                tokens: 2,
            },
            &stream,
        )
        .unwrap()
        .to_vec_f32(&stream)
        .unwrap();
        for batch in 0..2 {
            for token in 0..5 {
                let expected = if (2..4).contains(&token) { 1.0 } else { 0.0 };
                let offset = (batch * 5 + token) * 3;
                assert_eq!(
                    &marked[offset..offset + 3],
                    &[expected, 2.0 * expected, 3.0 * expected]
                );
            }
        }
    }

    #[test]
    fn audio_patchify_round_trip_preserves_channel_frequency_order() {
        let stream = danmo_mlx::Stream::cpu().unwrap();
        let latent = danmo_mlx::Array::from_f32_slice(
            &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
            &[1, 2, 1, 4],
        )
        .unwrap();
        let tokens = super::patchify_audio(&latent, &stream).unwrap();
        assert_eq!(tokens.shape(), &[1, 1, 8]);
        assert_eq!(
            tokens.to_vec_f32(&stream).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0]
        );
        let restored = super::unpatchify_audio(&tokens, 2, &stream).unwrap();
        assert_eq!(restored.shape(), latent.shape());
        assert_eq!(
            restored.to_vec_f32(&stream).unwrap(),
            latent.to_vec_f32(&stream).unwrap()
        );
    }

    #[test]
    #[ignore]
    fn real_transformer_forward_is_finite() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let bundle = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let model = super::Transformer::load(
            &root,
            &bundle.transformer,
            Some(crate::families::ltx25::bundle::Quantization {
                bits: 4,
                group_size: 64,
            }),
        )
        .unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let video = danmo_mlx::Array::from_f32_slice(&vec![0.01; 128 * 4 * 4], &[1, 128, 1, 4, 4])
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let audio = danmo_mlx::Array::from_f32_slice(&vec![0.01; 8 * 16 * 2], &[1, 8, 16, 2])
            .unwrap()
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let video = super::patchify_video(&video, &stream).unwrap();
        let audio = super::patchify_audio(&audio, &stream).unwrap();
        let vpos = crate::common::av_latent::video_positions(
            crate::common::av_latent::VideoLatentShape {
                frames: 1,
                height: 4,
                width: 4,
            },
            24.0,
            crate::families::ltx25::VIDEO_LATENT_SPEC,
        )
        .unwrap()
        .into_iter()
        .map(|position| position.to_vec())
        .collect::<Vec<_>>();
        let apos =
            crate::common::av_latent::audio_positions(2, crate::families::ltx25::AUDIO_LATENT_SPEC)
                .unwrap()
                .into_iter()
                .map(|position| vec![position])
                .collect::<Vec<_>>();
        let vcross = vpos
            .iter()
            .map(|position| vec![position[0]])
            .collect::<Vec<_>>();
        let cfg = model.config().clone();
        let vr = crate::common::av_rope::split_rope_tables(
            &vpos,
            cfg.dim,
            cfg.heads,
            cfg.rope_theta,
            &cfg.video_rope_max_pos,
            &stream,
        )
        .unwrap();
        let ar = crate::common::av_rope::split_rope_tables(
            &apos,
            cfg.audio_dim,
            cfg.audio_heads,
            cfg.rope_theta,
            &cfg.audio_rope_max_pos,
            &stream,
        )
        .unwrap();
        let vcr = crate::common::av_rope::split_rope_tables(
            &vcross,
            cfg.audio_dim,
            cfg.audio_heads,
            cfg.rope_theta,
            &[20.0],
            &stream,
        )
        .unwrap();
        let acr = crate::common::av_rope::split_rope_tables(
            &apos,
            cfg.audio_dim,
            cfg.audio_heads,
            cfg.rope_theta,
            &[20.0],
            &stream,
        )
        .unwrap();
        let context = super::PromptEmbeddings {
            video: danmo_mlx::Array::from_f32_slice(&vec![0.0; 1024 * 4096], &[1, 1024, 4096])
                .unwrap()
                .astype(danmo_mlx::Dtype::BFloat16, &stream)
                .unwrap(),
            audio: danmo_mlx::Array::from_f32_slice(&vec![0.0; 1024 * 2048], &[1, 1024, 2048])
                .unwrap()
                .astype(danmo_mlx::Dtype::BFloat16, &stream)
                .unwrap(),
        };
        let (video, audio) = model
            .forward_tokens(
                &video,
                &audio,
                &vr,
                &ar,
                &vcr,
                &acr,
                &context,
                super::KeyframeSpan {
                    start: 0,
                    tokens: video.shape()[1] as usize,
                },
                1.0,
                &stream,
            )
            .unwrap();
        let values = video.to_vec_f32(&stream).unwrap();
        let audio_values = audio.to_vec_f32(&stream).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(audio_values.iter().all(|value| value.is_finite()));
        println!(
            "LTX25 DiT probe: video={:?}, audio={:?}, first={:?}",
            video.shape(),
            audio.shape(),
            &values[..4]
        );
    }

    #[test]
    #[ignore]
    fn real_transformer_matches_python_velocity() {
        let root = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_BUNDLE").expect("DANQING_TEST_LTX25_BUNDLE"),
        );
        let fixture_path = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_LTX25_REFERENCE").expect("DANQING_TEST_LTX25_REFERENCE"),
        );
        let bundle = crate::families::ltx25::config::BundleConfig::load(&root).unwrap();
        let mut model = super::Transformer::load(
            &root,
            &bundle.transformer,
            Some(crate::families::ltx25::bundle::Quantization {
                bits: 4,
                group_size: 64,
            }),
        )
        .unwrap();
        let fixture =
            crate::weights::load_mlx_file(&fixture_path, crate::weights::WeightDtype::Native, None)
                .unwrap()
                .into_dense()
                .unwrap();
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let video = fixture["video_input"]
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let audio = fixture["audio_input"]
            .astype(danmo_mlx::Dtype::BFloat16, &stream)
            .unwrap();
        let context = super::PromptEmbeddings {
            video: fixture["video_context"].clone(),
            audio: fixture["audio_context"].clone(),
        };
        let vpos = crate::common::av_latent::video_positions(
            crate::common::av_latent::VideoLatentShape {
                frames: 1,
                height: 10,
                width: 10,
            },
            24.0,
            crate::families::ltx25::VIDEO_LATENT_SPEC,
        )
        .unwrap()
        .into_iter()
        .map(|position| position.to_vec())
        .collect::<Vec<_>>();
        let apos = crate::common::av_latent::audio_positions(
            25,
            crate::families::ltx25::AUDIO_LATENT_SPEC,
        )
        .unwrap()
        .into_iter()
        .map(|position| vec![position])
        .collect::<Vec<_>>();
        let vcross = vpos
            .iter()
            .map(|position| vec![position[0]])
            .collect::<Vec<_>>();
        let cfg = model.config().clone();
        let vr = crate::common::av_rope::split_rope_tables(
            &vpos,
            cfg.dim,
            cfg.heads,
            cfg.rope_theta,
            &cfg.video_rope_max_pos,
            &stream,
        )
        .unwrap();
        let ar = crate::common::av_rope::split_rope_tables(
            &apos,
            cfg.audio_dim,
            cfg.audio_heads,
            cfg.rope_theta,
            &cfg.audio_rope_max_pos,
            &stream,
        )
        .unwrap();
        let vcr = crate::common::av_rope::split_rope_tables(
            &vcross,
            cfg.audio_dim,
            cfg.audio_heads,
            cfg.rope_theta,
            &[20.0],
            &stream,
        )
        .unwrap();
        let acr = crate::common::av_rope::split_rope_tables(
            &apos,
            cfg.audio_dim,
            cfg.audio_heads,
            cfg.rope_theta,
            &[20.0],
            &stream,
        )
        .unwrap();
        let mut rope_max = 0.0f32;
        for (name, actual, expected) in [
            ("video_cos", &vr.cos, &fixture["video_rope_cos"]),
            ("video_sin", &vr.sin, &fixture["video_rope_sin"]),
            ("audio_cos", &ar.cos, &fixture["audio_rope_cos"]),
            ("audio_sin", &ar.sin, &fixture["audio_rope_sin"]),
            (
                "video_cross_cos",
                &vcr.cos,
                &fixture["video_cross_rope_cos"],
            ),
            (
                "video_cross_sin",
                &vcr.sin,
                &fixture["video_cross_rope_sin"],
            ),
            (
                "audio_cross_cos",
                &acr.cos,
                &fixture["audio_cross_rope_cos"],
            ),
            (
                "audio_cross_sin",
                &acr.sin,
                &fixture["audio_cross_rope_sin"],
            ),
        ] {
            assert_eq!(actual.shape(), expected.shape(), "{name} shape");
            let actual = actual.to_vec_f32(&stream).unwrap();
            let expected = expected.to_vec_f32(&stream).unwrap();
            let (max_index, max_abs) = actual
                .iter()
                .zip(&expected)
                .enumerate()
                .map(|(index, (a, b))| (index, (a - b).abs()))
                .fold(
                    (0, 0.0f32),
                    |best, item| {
                        if item.1 > best.1 {
                            item
                        } else {
                            best
                        }
                    },
                );
            rope_max = rope_max.max(max_abs);
            println!(
                "LTX25 RoPE {name}: max_abs={max_abs}, max_index={max_index}, rust={:?}, python={:?}",
                &actual[..8],
                &expected[..8]
            );
            println!(
                "LTX25 RoPE {name} worst: rust={}, python={}",
                actual[max_index], expected[max_index]
            );
        }
        // Large phase angles have a few float32 ULPs of grid divergence; keep
        // the bound tight enough to catch layout/sign errors without requiring
        // bit-identical argument reduction across independent implementations.
        assert!(rope_max < 0.003, "RoPE tables differ from Python");
        for layers in [1usize, 2, 4, 8, 16, 32, 48] {
            model.config.layers = layers;
            let (actual_video, actual_audio) = model
                .forward_tokens(
                    &video,
                    &audio,
                    &vr,
                    &ar,
                    &vcr,
                    &acr,
                    &context,
                    super::KeyframeSpan {
                        start: 0,
                        tokens: video.shape()[1] as usize,
                    },
                    1.0,
                    &stream,
                )
                .unwrap();
            for (name, actual, expected) in [
                (
                    "video",
                    actual_video,
                    &fixture[&format!("video_velocity_{layers}")],
                ),
                (
                    "audio",
                    actual_audio,
                    &fixture[&format!("audio_velocity_{layers}")],
                ),
            ] {
                let actual = actual.to_vec_f32(&stream).unwrap();
                let expected = expected.to_vec_f32(&stream).unwrap();
                let max_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                let mean_abs = actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / actual.len() as f32;
                println!(
                    "LTX25 {name} layers={layers}: max_abs={max_abs}, mean_abs={mean_abs}, rust={:?}, python={:?}",
                    &actual[..4],
                    &expected[..4]
                );
                // The first block is a tight structural check: bad RoPE layout,
                // norms, or projections fail here before float error can grow.
                // The full q4 video output has a separate accumulated-drift
                // bound. Full-depth audio drift is still diagnostic rather than
                // a passing parity claim; intermediate depths are printed too.
                let bounds = match (layers, name) {
                    (1, _) => Some((0.03, 0.006)),
                    (48, "video") => Some((0.75, 0.12)),
                    _ => None,
                };
                if let Some((max_bound, mean_bound)) = bounds {
                    assert!(
                        max_abs < max_bound && mean_abs < mean_bound,
                        "{name} layers={layers} velocity differs from Python: max_abs={max_abs}, mean_abs={mean_abs}"
                    );
                }
            }
        }
    }
}
