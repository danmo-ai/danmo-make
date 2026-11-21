//! SeedVR2 MM-DiT on MLX. Dense checkpoint tensors keep their native dtype;
//! normalization and rotary tables accumulate in F32.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

use danmo_mlx::{Array, Dtype, Result, Stream};

use super::super::dit::{DitConfig, plan_windows, timestep_features, window_angles};
use crate::common::mlx_ops::gelu_tanh;
use crate::weights::{MlxWeights, WeightDtype, WeightError, load_mlx_file};

const RMS_EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RopeKey {
    shape: [i64; 3],
    head: i32,
    text_offset: i64,
    pixel: bool,
    is_txt: bool,
}

pub struct Dit {
    weights: MlxWeights,
    config: DitConfig,
    freqs: Vec<f32>,
    rope_cache: RefCell<BTreeMap<RopeKey, (Array, Array)>>,
}

impl Dit {
    pub fn load(path: &Path, txt_len: i64) -> std::result::Result<Self, WeightError> {
        let weights = load_mlx_file(path, WeightDtype::Native, None)
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        let config = infer_config(&weights, txt_len)?;
        let freqs = read_freqs(&weights, &config)?;
        Ok(Self {
            weights,
            config,
            freqs,
            rope_cache: RefCell::new(BTreeMap::new()),
        })
    }

    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    pub fn nbytes(&self) -> usize {
        self.weights.nbytes()
    }

    fn rope_tables(
        &self,
        shape: [i64; 3],
        is_txt: bool,
        head: i32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let key = RopeKey {
            shape,
            head,
            text_offset: if is_txt {
                self.config.txt_len
            } else {
                video_text_offset(&self.config)
            },
            pixel: self.config.pixel_rope,
            is_txt,
        };
        if let Some(tables) = self.rope_cache.borrow().get(&key) {
            return Ok(tables.clone());
        }
        let angles = window_angles(
            &self.freqs,
            key.pixel,
            key.text_offset,
            key.shape,
            key.is_txt,
        );
        let tables = rope_arrays(&angles, head, stream)?;
        self.rope_cache
            .borrow_mut()
            .insert(key, (tables.0.clone(), tables.1.clone()));
        Ok(tables)
    }

    /// `vid` is `[1,33,T,H,W]`, `txt` is `[1,58,5120]` for the published 7B.
    pub fn forward(
        &self,
        vid: &Array,
        txt: &Array,
        timestep: f32,
        stream: &Stream,
    ) -> Result<Array> {
        let shape = vid.shape();
        if shape.len() != 5 || shape[0] != 1 || shape[1] != self.config.vid_channels as i32 {
            return Err(danmo_mlx::Error::new(
                "SeedVR2 DiT input",
                format!(
                    "expected [1,{},T,H,W], got {shape:?}",
                    self.config.vid_channels
                ),
            ));
        }
        if shape[3] % 2 != 0 || shape[4] % 2 != 0 || shape[2] < 1 {
            return Err(danmo_mlx::Error::new(
                "SeedVR2 DiT input",
                format!("latent shape {shape:?} is not patchable by (1,2,2)"),
            ));
        }
        let (t, h, w) = (shape[2], shape[3], shape[4]);
        let (hp, wp) = (h / 2, w / 2);
        let tokens = t * hp * wp;
        let dim = self.config.dim as i32;
        let dtype = vid.dtype();

        let patches = vid
            .reshape(&[1, shape[1], t, 1, hp, 2, wp, 2], stream)?
            .transpose_axes(&[0, 2, 4, 6, 3, 5, 7, 1], stream)?
            .reshape(&[1, tokens, self.config.patch_dim() as i32], stream)?;
        let mut vid = linear(&self.weights, "vid_in.proj", &patches, true, stream)?
            .reshape(&[1, tokens, dim], stream)?;
        let mut txt = linear(&self.weights, "txt_in", txt, true, stream)?;

        let temb = Array::from_f32_slice(
            &timestep_features(timestep, self.config.emb_sin),
            &[1, self.config.emb_sin as i32],
        )?
        .astype(dtype, stream)?;
        let mut emb = linear(&self.weights, "emb_in.proj_in", &temb, true, stream)?.silu(stream)?;
        emb = linear(&self.weights, "emb_in.proj_hid", &emb, true, stream)?.silu(stream)?;
        emb = linear(&self.weights, "emb_in.proj_out", &emb, true, stream)?
            .reshape(&[1, dim, 2, 3], stream)?;

        for layer in 0..self.config.layers {
            let (next_vid, next_txt) = self.block(layer, &vid, &txt, &emb, t, hp, wp, stream)?;
            vid = next_vid;
            txt = next_txt;
            Array::eval_all(&[&vid, &txt])?;
        }

        if self.config.output_ada {
            vid = vid.rms_norm(
                Some(self.weights.tensor("vid_out_norm.weight")?),
                RMS_EPS,
                stream,
            )?;
            let shift = modulation(&self.weights, "vid_out_ada.out", &emb, 0, 0, stream)?;
            let scale = modulation(&self.weights, "vid_out_ada.out", &emb, 0, 1, stream)?;
            vid = Array::add(&Array::multiply(&vid, &scale, stream)?, &shift, stream)?;
        }
        let out = linear(&self.weights, "vid_out.proj", &vid, true, stream)?;
        out.reshape(
            &[1, t, hp, wp, 1, 2, 2, self.config.out_channels as i32],
            stream,
        )?
        .transpose_axes(&[0, 7, 1, 4, 2, 5, 3, 6], stream)?
        .reshape(&[1, self.config.out_channels as i32, t, h, w], stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        layer: usize,
        vid: &Array,
        txt: &Array,
        emb: &Array,
        t: i32,
        hp: i32,
        wp: i32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let prefix = format!("blocks.{layer}");
        let side = if layer < self.config.mm_layers {
            "vid"
        } else {
            "all"
        };
        let txt_side = if side == "all" { "all" } else { "txt" };
        let last = self.config.last_layer_vid_only && layer + 1 == self.config.layers;
        let ada_vid = format!("{prefix}.ada.{side}");
        let ada_txt = format!("{prefix}.ada.{txt_side}");

        let mut vid_n = vid.rms_norm(None, RMS_EPS, stream)?;
        vid_n = modulate_in(&self.weights, &ada_vid, "attn", emb, 0, &vid_n, stream)?;
        let txt_n = txt.rms_norm(None, RMS_EPS, stream)?;
        let txt_n = if last {
            txt_n
        } else {
            modulate_in(&self.weights, &ada_txt, "attn", emb, 0, &txt_n, stream)?
        };
        let (attn_vid, attn_txt) =
            self.attention(layer, side, txt_side, &vid_n, &txt_n, t, hp, wp, stream)?;
        let attn_vid = modulate_out(&self.weights, &ada_vid, "attn", emb, 0, &attn_vid, stream)?;
        let mut vid = Array::add(vid, &attn_vid, stream)?;
        let mut txt = if last {
            txt.clone()
        } else {
            let attn_txt =
                modulate_out(&self.weights, &ada_txt, "attn", emb, 0, &attn_txt, stream)?;
            Array::add(txt, &attn_txt, stream)?
        };

        let vid_n = vid.rms_norm(None, RMS_EPS, stream)?;
        let vid_n = modulate_in(&self.weights, &ada_vid, "mlp", emb, 1, &vid_n, stream)?;
        let vid_m = self.mlp(&format!("{prefix}.mlp.{side}"), &vid_n, stream)?;
        let vid_m = modulate_out(&self.weights, &ada_vid, "mlp", emb, 1, &vid_m, stream)?;
        vid = Array::add(&vid, &vid_m, stream)?;
        if !last {
            let txt_n = txt.rms_norm(None, RMS_EPS, stream)?;
            let txt_n = modulate_in(&self.weights, &ada_txt, "mlp", emb, 1, &txt_n, stream)?;
            let txt_m = self.mlp(&format!("{prefix}.mlp.{txt_side}"), &txt_n, stream)?;
            let txt_m = modulate_out(&self.weights, &ada_txt, "mlp", emb, 1, &txt_m, stream)?;
            txt = Array::add(&txt, &txt_m, stream)?;
        }
        Ok((vid, txt))
    }

    fn mlp(&self, prefix: &str, input: &Array, stream: &Stream) -> Result<Array> {
        if self.config.swiglu {
            let gate = linear(
                &self.weights,
                &format!("{prefix}.proj_in_gate"),
                input,
                false,
                stream,
            )?
            .silu(stream)?;
            let up = linear(
                &self.weights,
                &format!("{prefix}.proj_in"),
                input,
                false,
                stream,
            )?;
            let hidden = Array::multiply(&gate, &up, stream)?;
            linear(
                &self.weights,
                &format!("{prefix}.proj_out"),
                &hidden,
                false,
                stream,
            )
        } else {
            let hidden = linear(
                &self.weights,
                &format!("{prefix}.proj_in"),
                input,
                true,
                stream,
            )?;
            let hidden = gelu_tanh(&hidden, stream)?;
            linear(
                &self.weights,
                &format!("{prefix}.proj_out"),
                &hidden,
                true,
                stream,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: usize,
        side: &str,
        txt_side: &str,
        vid: &Array,
        txt: &Array,
        t: i32,
        hp: i32,
        wp: i32,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        let prefix = format!("blocks.{layer}");
        let heads = self.config.heads as i32;
        let head = self.config.head_dim as i32;
        let inner = heads * head;
        let tokens = t * hp * wp;
        let txt_len = txt.shape()[1];
        let qkv_vid = linear(
            &self.weights,
            &format!("{prefix}.attn.proj_qkv.{side}"),
            vid,
            false,
            stream,
        )?
        .reshape(&[tokens, 3, heads, head], stream)?;
        let qkv_txt = linear(
            &self.weights,
            &format!("{prefix}.attn.proj_qkv.{txt_side}"),
            txt,
            false,
            stream,
        )?
        .reshape(&[txt_len, 3, heads, head], stream)?;

        let plan = plan_windows(t as i64, hp as i64, wp as i64, layer % 2 == 1);
        let forward = Array::from_i32_slice(&plan.forward, &[tokens])?;
        let reverse = Array::from_i32_slice(&plan.reverse, &[tokens])?;
        let qkv_vid = qkv_vid.take_axis(&forward, 0, stream)?;
        let q_v = head_norm(
            &self.weights,
            &format!("{prefix}.attn.norm_q.{side}.weight"),
            &part(&qkv_vid, 0, None, stream)?,
            stream,
        )?;
        let k_v = head_norm(
            &self.weights,
            &format!("{prefix}.attn.norm_k.{side}.weight"),
            &part(&qkv_vid, 1, None, stream)?,
            stream,
        )?;
        let v_v = part(&qkv_vid, 2, None, stream)?;
        let q_t = head_norm(
            &self.weights,
            &format!("{prefix}.attn.norm_q.{txt_side}.weight"),
            &part(&qkv_txt, 0, None, stream)?,
            stream,
        )?;
        let k_t = head_norm(
            &self.weights,
            &format!("{prefix}.attn.norm_k.{txt_side}.weight"),
            &part(&qkv_txt, 1, None, stream)?,
            stream,
        )?;
        let v_t = part(&qkv_txt, 2, None, stream)?;
        let (q_t, k_t) = if self.config.pixel_rope {
            (q_t, k_t)
        } else {
            let (cos, sin) = self.rope_tables([1, 1, 1], true, head, stream)?;
            (
                apply_rope(&q_t, &cos, &sin, stream)?,
                apply_rope(&k_t, &cos, &sin, stream)?,
            )
        };

        let mut video_outputs = Vec::with_capacity(plan.windows.len());
        let mut text_sum: Option<Array> = None;
        for (start, len, window_shape) in &plan.windows {
            let range = Some((*start as i32, (*start + *len) as i32));
            let q_window = token_part(&q_v, range, stream)?;
            let k_window = token_part(&k_v, range, stream)?;
            let v_window = token_part(&v_v, range, stream)?;
            let (cos, sin) = self.rope_tables(*window_shape, false, head, stream)?;
            let q_window = apply_rope(&q_window, &cos, &sin, stream)?;
            let k_window = apply_rope(&k_window, &cos, &sin, stream)?;
            let len = *len as i32;
            let q = Array::concatenate_axis(&[q_window, q_t.clone()], 0, stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .expand_dims(0, stream)?;
            let k = Array::concatenate_axis(&[k_window, k_t.clone()], 0, stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .expand_dims(0, stream)?;
            let v = Array::concatenate_axis(&[v_window, v_t.clone()], 0, stream)?
                .transpose_axes(&[1, 0, 2], stream)?
                .expand_dims(0, stream)?;
            let output = Array::scaled_dot_product_attention(
                &q,
                &k,
                &v,
                1.0 / (head as f32).sqrt(),
                "",
                None,
                stream,
            )?
            .squeeze_axis(0, stream)?
            .transpose_axes(&[1, 0, 2], stream)?;
            video_outputs.push(output.slice(
                &[0, 0, 0],
                &[len, heads, head],
                &[1, 1, 1],
                stream,
            )?);
            let text = output.slice(
                &[len, 0, 0],
                &[len + txt_len, heads, head],
                &[1, 1, 1],
                stream,
            )?;
            text_sum = Some(match text_sum {
                None => text,
                Some(sum) => Array::add(&sum, &text, stream)?,
            });
        }
        let mut vid = Array::concatenate_axis(&video_outputs, 0, stream)?
            .reshape(&[tokens, inner], stream)?
            .take_axis(&reverse, 0, stream)?;
        vid = linear(
            &self.weights,
            &format!("{prefix}.attn.proj_out.{side}"),
            &vid,
            true,
            stream,
        )?
        .reshape(&[1, tokens, inner], stream)?;
        let count = plan.windows.len() as f32;
        let txt = text_sum.ok_or_else(|| {
            danmo_mlx::Error::new("SeedVR2 attention", "window plan produced no windows")
        })?;
        let txt = Array::multiply(&txt, &Array::scalar_f32(count.recip())?, stream)?
            .reshape(&[txt_len, inner], stream)?;
        let txt = linear(
            &self.weights,
            &format!("{prefix}.attn.proj_out.{txt_side}"),
            &txt,
            true,
            stream,
        )?
        .reshape(&[1, txt_len, inner], stream)?;
        Ok((vid, txt))
    }
}

fn video_text_offset(config: &DitConfig) -> i64 {
    if config.pixel_rope { 0 } else { config.txt_len }
}

fn infer_config(weights: &MlxWeights, txt_len: i64) -> std::result::Result<DitConfig, WeightError> {
    let mut layers = 0usize;
    let mut mm_layers = None;
    for name in weights.names() {
        if let Some(rest) = name.strip_prefix("blocks.") {
            if let Some((index, _)) = rest.split_once('.') {
                if let Ok(index) = index.parse::<usize>() {
                    layers = layers.max(index + 1);
                    if mm_layers.is_none() && rest.contains(".ada.all.") {
                        mm_layers = Some(index);
                    }
                }
            }
        }
    }
    if layers == 0 {
        return Err(WeightError::NotBuilt(
            "seedvr2 MLX DiT has no blocks.* tensors".into(),
        ));
    }
    let mm_layers = mm_layers.unwrap_or(layers);
    let side = if mm_layers > 0 { "vid" } else { "all" };
    let shape = |name: &str| -> std::result::Result<Vec<i32>, WeightError> {
        weights
            .tensor(name)
            .map(|tensor| tensor.shape())
            .map_err(|error| WeightError::NotBuilt(error.to_string()))
    };
    let dim = shape(&format!("blocks.0.ada.{side}.attn_shift"))?[0] as i64;
    let head_dim = shape(&format!("blocks.0.attn.norm_q.{side}.weight"))?[0] as i64;
    let qkv = shape(&format!("blocks.0.attn.proj_qkv.{side}.weight"))?;
    if qkv[1] as i64 != dim || qkv[0] as i64 % (3 * head_dim) != 0 {
        return Err(WeightError::NotBuilt(format!(
            "seedvr2 MLX qkv shape {qkv:?} disagrees with dim={dim}, head_dim={head_dim}"
        )));
    }
    let heads = qkv[0] as i64 / (3 * head_dim);
    let swiglu = weights.has(&format!("blocks.0.mlp.{side}.proj_in_gate.weight"));
    let mlp_hidden = shape(&format!("blocks.0.mlp.{side}.proj_in.weight"))?[0] as i64;
    let output_ada = weights.has("vid_out_norm.weight") && weights.has("vid_out_ada.out_shift");
    let freqs_len = shape("blocks.0.attn.rope.rope.freqs")?[0] as i64;
    let freqs = weights
        .tensor("blocks.0.attn.rope.rope.freqs")
        .and_then(|tensor| tensor.to_vec_f32(&Stream::cpu()?))
        .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let pixel_rope = freqs.iter().any(|value| *value > 1.0);
    let patch = 2;
    let patch_dim = shape("vid_in.proj.weight")?[1] as i64;
    let out_dim = shape("vid_out.proj.weight")?[0] as i64;
    if patch_dim % (patch * patch) != 0 || out_dim % (patch * patch) != 0 {
        return Err(WeightError::NotBuilt(format!(
            "seedvr2 MLX patch dims {patch_dim}/{out_dim} are invalid"
        )));
    }
    let emb_sin = shape("emb_in.proj_in.weight")?[1] as i64;
    if shape("emb_in.proj_out.weight")?[0] as i64 != 6 * dim {
        return Err(WeightError::NotBuilt(
            "seedvr2 MLX emb_in.proj_out is not 6 * dim".into(),
        ));
    }
    Ok(DitConfig {
        layers,
        mm_layers,
        dim,
        heads,
        head_dim,
        mlp_hidden,
        txt_in_dim: shape("txt_in.weight")?[1] as i64,
        emb_sin,
        swiglu,
        output_ada,
        last_layer_vid_only: swiglu,
        pixel_rope,
        freqs_len,
        vid_channels: patch_dim / (patch * patch),
        out_channels: out_dim / (patch * patch),
        patch,
        txt_len,
    })
}

fn read_freqs(
    weights: &MlxWeights,
    config: &DitConfig,
) -> std::result::Result<Vec<f32>, WeightError> {
    let stream = Stream::cpu().map_err(|error| WeightError::NotBuilt(error.to_string()))?;
    let mut result = None;
    for layer in 0..config.layers {
        let name = format!("blocks.{layer}.attn.rope.rope.freqs");
        let values = weights
            .tensor(&name)
            .and_then(|tensor| tensor.to_vec_f32(&stream))
            .map_err(|error| WeightError::NotBuilt(error.to_string()))?;
        match &result {
            None => result = Some(values),
            Some(first) if *first == values => {}
            Some(_) => {
                return Err(WeightError::NotBuilt(format!(
                    "{name} diverges from blocks.0 RoPE frequencies"
                )));
            }
        }
    }
    result.ok_or_else(|| WeightError::NotBuilt("seedvr2 MLX DiT has no RoPE frequencies".into()))
}

fn linear(
    weights: &MlxWeights,
    prefix: &str,
    input: &Array,
    bias: bool,
    stream: &Stream,
) -> Result<Array> {
    let bias_name = format!("{prefix}.bias");
    let bias = if bias && weights.has(&bias_name) {
        Some(weights.tensor(&bias_name)?)
    } else {
        None
    };
    weights
        .linear(&format!("{prefix}.weight"))?
        .apply(input, bias, stream)
}

fn part(
    input: &Array,
    component: i32,
    range: Option<(i32, i32)>,
    stream: &Stream,
) -> Result<Array> {
    let shape = input.shape();
    let (start, stop) = range.unwrap_or((0, shape[0]));
    input
        .slice(
            &[start, component, 0, 0],
            &[stop, component + 1, shape[2], shape[3]],
            &[1, 1, 1, 1],
            stream,
        )?
        .squeeze_axis(1, stream)
}

fn token_part(input: &Array, range: Option<(i32, i32)>, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (start, stop) = range.unwrap_or((0, shape[0]));
    input.slice(
        &[start, 0, 0],
        &[stop, shape[1], shape[2]],
        &[1, 1, 1],
        stream,
    )
}

fn head_norm(weights: &MlxWeights, name: &str, input: &Array, stream: &Stream) -> Result<Array> {
    input.rms_norm(Some(weights.tensor(name)?), RMS_EPS, stream)
}

fn modulation(
    weights: &MlxWeights,
    prefix: &str,
    emb: &Array,
    layer: i32,
    mode: i32,
    stream: &Stream,
) -> Result<Array> {
    let suffix = match mode {
        0 => "shift",
        1 => "scale",
        2 => "gate",
        _ => unreachable!(),
    };
    let value = emb.slice(
        &[0, 0, layer, mode],
        &[1, emb.shape()[1], layer + 1, mode + 1],
        &[1, 1, 1, 1],
        stream,
    )?;
    let value = value.reshape(&[1, 1, emb.shape()[1]], stream)?;
    Array::add(
        &value,
        weights.tensor(&format!("{prefix}_{suffix}"))?,
        stream,
    )
}

fn modulate_in(
    weights: &MlxWeights,
    prefix: &str,
    layer: &str,
    emb: &Array,
    layer_idx: i32,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let base = format!("{prefix}.{layer}");
    let shift = modulation(weights, &base, emb, layer_idx, 0, stream)?;
    let scale = modulation(weights, &base, emb, layer_idx, 1, stream)?;
    Array::add(&Array::multiply(input, &scale, stream)?, &shift, stream)
}

fn modulate_out(
    weights: &MlxWeights,
    prefix: &str,
    layer: &str,
    emb: &Array,
    layer_idx: i32,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let gate = modulation(
        weights,
        &format!("{prefix}.{layer}"),
        emb,
        layer_idx,
        2,
        stream,
    )?;
    Array::multiply(input, &gate, stream)
}

fn rope_arrays(angles: &[Vec<f64>], head: i32, stream: &Stream) -> Result<(Array, Array)> {
    let mut cos = vec![1.0f32; angles.len() * head as usize];
    let mut sin = vec![0.0f32; cos.len()];
    for (token, row) in angles.iter().enumerate() {
        for (channel, angle) in row.iter().enumerate() {
            cos[token * head as usize + channel] = angle.cos() as f32;
            sin[token * head as usize + channel] = angle.sin() as f32;
        }
    }
    let shape = [angles.len() as i32, 1, head];
    let cos = Array::from_f32_slice(&cos, &shape)?.astype(Dtype::Float32, stream)?;
    let sin = Array::from_f32_slice(&sin, &shape)?.astype(Dtype::Float32, stream)?;
    Ok((cos, sin))
}

fn apply_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let shape = input.shape();
    let (tokens, heads, head) = (shape[0], shape[1], shape[2]);
    let dtype = input.dtype();
    let pairs = input
        .astype(Dtype::Float32, stream)?
        .reshape(&[tokens, heads, head / 2, 2], stream)?;
    let first = pairs.slice(
        &[0, 0, 0, 0],
        &[tokens, heads, head / 2, 1],
        &[1, 1, 1, 1],
        stream,
    )?;
    let second = pairs.slice(
        &[0, 0, 0, 1],
        &[tokens, heads, head / 2, 2],
        &[1, 1, 1, 1],
        stream,
    )?;
    let rotated = Array::stack(&[second.negative(stream)?, first], 3, stream)?
        .reshape(&[tokens, heads, head], stream)?;
    let output = Array::add(
        &Array::multiply(&pairs.reshape(&[tokens, heads, head], stream)?, cos, stream)?,
        &Array::multiply(&rotated, sin, stream)?,
        stream,
    )?;
    output.astype(dtype, stream)
}

#[cfg(all(test, feature = "mlx-metal"))]
mod tests {
    #[test]
    #[ignore = "loads the installed 7B DiT and a main stage fixture"]
    fn real_dit_stages_match_main() {
        let bundle = std::path::PathBuf::from(
            std::env::var("DANQING_TEST_SEEDVR2_BUNDLE").expect("DANQING_TEST_SEEDVR2_BUNDLE"),
        );
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                std::env::var("DANQING_TEST_SEEDVR2_DIT_STAGES")
                    .expect("DANQING_TEST_SEEDVR2_DIT_STAGES"),
            )
            .unwrap(),
        )
        .unwrap();
        let values = |name: &str| -> (Vec<i32>, Vec<f32>) {
            let shape = reference[name]["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap() as i32)
                .collect();
            let values = reference[name]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            (shape, values)
        };
        let (txt_shape, txt_values) = values("txt_pos");
        let stream = danmo_mlx::Stream::gpu().unwrap();
        let model = super::Dit::load(
            &bundle.join("seedvr2_ema_7b_fp16.safetensors"),
            txt_shape[1] as i64,
        )
        .unwrap();
        let (input_shape, input_values) = values("model_input");
        let input = danmo_mlx::Array::from_f32_slice(&input_values, &input_shape).unwrap();
        let txt = danmo_mlx::Array::from_f32_slice(&txt_values, &txt_shape).unwrap();
        let (t, h, w) = (input_shape[2], input_shape[3], input_shape[4]);
        let (hp, wp) = (h / 2, w / 2);
        let tokens = t * hp * wp;
        let dim = model.config.dim as i32;
        let patches = input
            .reshape(&[1, input_shape[1], t, 1, hp, 2, wp, 2], &stream)
            .unwrap()
            .transpose_axes(&[0, 2, 4, 6, 3, 5, 7, 1], &stream)
            .unwrap()
            .reshape(&[1, tokens, model.config.patch_dim() as i32], &stream)
            .unwrap();
        let mut vid = super::linear(&model.weights, "vid_in.proj", &patches, true, &stream)
            .unwrap()
            .reshape(&[1, tokens, dim], &stream)
            .unwrap();
        let mut stages = vec![("vid_embed", vid.clone())];
        let mut txt = super::linear(&model.weights, "txt_in", &txt, true, &stream).unwrap();
        stages.push(("txt_embed", txt.clone()));
        let temb = danmo_mlx::Array::from_f32_slice(
            &super::timestep_features(1000.0, model.config.emb_sin),
            &[1, model.config.emb_sin as i32],
        )
        .unwrap();
        let mut emb = super::linear(&model.weights, "emb_in.proj_in", &temb, true, &stream)
            .unwrap()
            .silu(&stream)
            .unwrap();
        emb = super::linear(&model.weights, "emb_in.proj_hid", &emb, true, &stream)
            .unwrap()
            .silu(&stream)
            .unwrap();
        emb = super::linear(&model.weights, "emb_in.proj_out", &emb, true, &stream)
            .unwrap()
            .reshape(&[1, dim, 2, 3], &stream)
            .unwrap();
        stages.push(("emb", emb.clone()));
        let vid_n = vid.rms_norm(None, super::RMS_EPS, &stream).unwrap();
        let txt_n = txt.rms_norm(None, super::RMS_EPS, &stream).unwrap();
        let vid_n = super::modulate_in(
            &model.weights,
            "blocks.0.ada.vid",
            "attn",
            &emb,
            0,
            &vid_n,
            &stream,
        )
        .unwrap();
        let txt_n = super::modulate_in(
            &model.weights,
            "blocks.0.ada.txt",
            "attn",
            &emb,
            0,
            &txt_n,
            &stream,
        )
        .unwrap();
        stages.push(("block_0_attn_in_vid", vid_n.clone()));
        stages.push(("block_0_attn_in_txt", txt_n.clone()));
        let heads = model.config.heads as i32;
        let head = model.config.head_dim as i32;
        let qkv_vid = super::linear(
            &model.weights,
            "blocks.0.attn.proj_qkv.vid",
            &vid_n,
            false,
            &stream,
        )
        .unwrap()
        .reshape(&[tokens, 3, heads, head], &stream)
        .unwrap();
        let qkv_txt = super::linear(
            &model.weights,
            "blocks.0.attn.proj_qkv.txt",
            &txt_n,
            false,
            &stream,
        )
        .unwrap()
        .reshape(&[txt.shape()[1], 3, heads, head], &stream)
        .unwrap();
        let plan = super::plan_windows(t as i64, hp as i64, wp as i64, false);
        let forward = danmo_mlx::Array::from_i32_slice(&plan.forward, &[tokens]).unwrap();
        let qkv_vid = qkv_vid.take_axis(&forward, 0, &stream).unwrap();
        let q_vid = super::head_norm(
            &model.weights,
            "blocks.0.attn.norm_q.vid.weight",
            &super::part(&qkv_vid, 0, None, &stream).unwrap(),
            &stream,
        )
        .unwrap();
        let k_vid = super::head_norm(
            &model.weights,
            "blocks.0.attn.norm_k.vid.weight",
            &super::part(&qkv_vid, 1, None, &stream).unwrap(),
            &stream,
        )
        .unwrap();
        stages.push(("block_0_q_norm_vid", q_vid.clone()));
        stages.push(("block_0_k_norm_vid", k_vid.clone()));
        let mut angle_values = Vec::new();
        let mut q_windows = Vec::new();
        let mut k_windows = Vec::new();
        for (start, len, window_shape) in &plan.windows {
            let angles = super::window_angles(
                &model.freqs,
                model.config.pixel_rope,
                super::video_text_offset(&model.config),
                *window_shape,
                false,
            );
            angle_values.extend(
                angles
                    .iter()
                    .flat_map(|row| row.iter().map(|value| *value as f32)),
            );
            let (cos, sin) = super::rope_arrays(&angles, head, &stream).unwrap();
            let range = Some((*start as i32, (*start + *len) as i32));
            q_windows.push(
                super::apply_rope(
                    &super::token_part(&q_vid, range, &stream).unwrap(),
                    &cos,
                    &sin,
                    &stream,
                )
                .unwrap(),
            );
            k_windows.push(
                super::apply_rope(
                    &super::token_part(&k_vid, range, &stream).unwrap(),
                    &cos,
                    &sin,
                    &stream,
                )
                .unwrap(),
            );
        }
        stages.push((
            "block_0_rope_angles",
            danmo_mlx::Array::from_f32_slice(&angle_values, &[tokens, 60]).unwrap(),
        ));
        let q_vid = danmo_mlx::Array::concatenate_axis(&q_windows, 0, &stream).unwrap();
        let k_vid = danmo_mlx::Array::concatenate_axis(&k_windows, 0, &stream).unwrap();
        stages.push(("block_0_q_rope_vid", q_vid));
        stages.push(("block_0_k_rope_vid", k_vid));
        let q_txt = super::head_norm(
            &model.weights,
            "blocks.0.attn.norm_q.txt.weight",
            &super::part(&qkv_txt, 0, None, &stream).unwrap(),
            &stream,
        )
        .unwrap();
        let k_txt = super::head_norm(
            &model.weights,
            "blocks.0.attn.norm_k.txt.weight",
            &super::part(&qkv_txt, 1, None, &stream).unwrap(),
            &stream,
        )
        .unwrap();
        stages.push(("block_0_q_txt", q_txt));
        stages.push(("block_0_k_txt", k_txt));
        let (attn_vid, attn_txt) = model
            .attention(0, "vid", "txt", &vid_n, &txt_n, t, hp, wp, &stream)
            .unwrap();
        stages.push(("block_0_attn_raw_vid", attn_vid.clone()));
        stages.push(("block_0_attn_raw_txt", attn_txt.clone()));
        let attn_vid = super::modulate_out(
            &model.weights,
            "blocks.0.ada.vid",
            "attn",
            &emb,
            0,
            &attn_vid,
            &stream,
        )
        .unwrap();
        let attn_txt = super::modulate_out(
            &model.weights,
            "blocks.0.ada.txt",
            "attn",
            &emb,
            0,
            &attn_txt,
            &stream,
        )
        .unwrap();
        vid = danmo_mlx::Array::add(&vid, &attn_vid, &stream).unwrap();
        txt = danmo_mlx::Array::add(&txt, &attn_txt, &stream).unwrap();
        stages.push(("block_0_after_attn_vid", vid.clone()));
        stages.push(("block_0_after_attn_txt", txt.clone()));
        let vid_n = vid.rms_norm(None, super::RMS_EPS, &stream).unwrap();
        let txt_n = txt.rms_norm(None, super::RMS_EPS, &stream).unwrap();
        let vid_n = super::modulate_in(
            &model.weights,
            "blocks.0.ada.vid",
            "mlp",
            &emb,
            1,
            &vid_n,
            &stream,
        )
        .unwrap();
        let txt_n = super::modulate_in(
            &model.weights,
            "blocks.0.ada.txt",
            "mlp",
            &emb,
            1,
            &txt_n,
            &stream,
        )
        .unwrap();
        stages.push(("block_0_mlp_in_vid", vid_n.clone()));
        let mlp_vid = model.mlp("blocks.0.mlp.vid", &vid_n, &stream).unwrap();
        let mlp_txt = model.mlp("blocks.0.mlp.txt", &txt_n, &stream).unwrap();
        stages.push(("block_0_mlp_raw_vid", mlp_vid.clone()));
        stages.push(("block_0_mlp_raw_txt", mlp_txt.clone()));
        let mlp_vid = super::modulate_out(
            &model.weights,
            "blocks.0.ada.vid",
            "mlp",
            &emb,
            1,
            &mlp_vid,
            &stream,
        )
        .unwrap();
        let mlp_txt = super::modulate_out(
            &model.weights,
            "blocks.0.ada.txt",
            "mlp",
            &emb,
            1,
            &mlp_txt,
            &stream,
        )
        .unwrap();
        vid = danmo_mlx::Array::add(&vid, &mlp_vid, &stream).unwrap();
        txt = danmo_mlx::Array::add(&txt, &mlp_txt, &stream).unwrap();
        stages.push(("block_0_vid", vid.clone()));
        stages.push(("block_0_txt", txt.clone()));
        for layer in 1..model.config.layers {
            let (next_vid, next_txt) = model
                .block(layer, &vid, &txt, &emb, t, hp, wp, &stream)
                .unwrap();
            vid = next_vid;
            txt = next_txt;
            if matches!(layer, 0 | 1 | 35) {
                stages.push((
                    Box::leak(format!("block_{layer}_vid").into_boxed_str()) as &str,
                    vid.clone(),
                ));
                stages.push((
                    Box::leak(format!("block_{layer}_txt").into_boxed_str()) as &str,
                    txt.clone(),
                ));
            }
        }
        if model.config.output_ada {
            vid = vid
                .rms_norm(
                    Some(model.weights.tensor("vid_out_norm.weight").unwrap()),
                    super::RMS_EPS,
                    &stream,
                )
                .unwrap();
            let shift =
                super::modulation(&model.weights, "vid_out_ada.out", &emb, 0, 0, &stream).unwrap();
            let scale =
                super::modulation(&model.weights, "vid_out_ada.out", &emb, 0, 1, &stream).unwrap();
            vid = danmo_mlx::Array::add(
                &danmo_mlx::Array::multiply(&vid, &scale, &stream).unwrap(),
                &shift,
                &stream,
            )
            .unwrap();
        }
        let out = super::linear(&model.weights, "vid_out.proj", &vid, true, &stream).unwrap();
        let velocity = out
            .reshape(
                &[1, t, hp, wp, 1, 2, 2, model.config.out_channels as i32],
                &stream,
            )
            .unwrap()
            .transpose_axes(&[0, 7, 1, 4, 2, 5, 3, 6], &stream)
            .unwrap()
            .reshape(&[1, model.config.out_channels as i32, t, h, w], &stream)
            .unwrap();
        stages.push(("velocity", velocity));
        for (name, actual) in stages {
            let actual = actual.to_vec_f32(&stream).unwrap();
            let (_, expected) = values(name);
            assert_eq!(actual.len(), expected.len(), "{name} length");
            let mut mse = 0.0f64;
            let mut max_abs = 0.0f32;
            let mut energy = 0.0f64;
            for (actual, expected) in actual.iter().zip(&expected) {
                let difference = actual - expected;
                mse += f64::from(difference * difference);
                max_abs = max_abs.max(difference.abs());
                energy += f64::from(expected * expected);
            }
            mse /= actual.len() as f64;
            let rel_l2 = (mse * actual.len() as f64 / energy.max(1e-20)).sqrt();
            eprintln!("dit {name}: mse={mse:.9} max_abs={max_abs:.6} rel_l2={rel_l2:.6}");
            assert!(rel_l2 < 0.01, "{name} relative L2 {rel_l2} exceeds 1%");
        }
    }
}
