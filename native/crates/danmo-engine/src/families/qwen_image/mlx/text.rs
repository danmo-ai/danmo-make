//! Qwen-Image text conditioning: Qwen2.5-VL decoder trunk + image prompt
//! template, matching danmo-make `families/qwen/text_encoder_mlx.py`.
//!
//! The checkpoint keys are `encoder.*` (mlx-community pre-remapped layout).
//! Text-only positions put the same index on all three MRoPE axes, so the
//! section picker collapses to plain half-split RoPE. The final RMS-normed
//! hidden state loses the 34-token template prefix; the remaining rows are
//! the DiT context (mask all ones).
//!
//! The tower keeps the checkpoint dtype (production policy). main promotes
//! the quantized text tower to F32 by widening dense params; that widening
//! is an intentional deviation and is not reproduced here.

use std::path::Path;

use danmo_mlx::{Array, Dtype, Error, Result, Stream};

use crate::weights::{MlxWeights, WeightError};

pub(super) const PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
pub(super) const MAX_LENGTH: usize = 1058;
pub(super) const DROP_IDX: usize = 34;

/// Qwen-Image-Edit system prompt. The VL encoder drops this prefix
/// (`edit_template_start_idx = 64`) before the DiT sees the hidden states.
pub(super) const EDIT_SYSTEM: &str = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n";
pub(super) const EDIT_DROP_IDX: usize = 64;
const IMAGE_PAD: &str = "<|image_pad|>";
const RMS_EPS: f32 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

#[derive(Debug, Clone, Copy)]
pub(super) struct TextSpec {
    pub vocab: i64,
    pub hidden: i32,
    pub layers: u32,
    pub heads: i32,
    pub kv_heads: i32,
    pub head: i32,
}

pub(super) fn text_spec(weights: &MlxWeights) -> Result<TextSpec> {
    let embedding = weights.tensor("encoder.embed_tokens.weight")?;
    let shape = embedding.shape();
    if shape.len() != 2 {
        return Err(Error::new(
            "Qwen-Image text",
            format!("embed_tokens shape {shape:?} is not [vocab, hidden]"),
        ));
    }
    let (vocab, hidden) = (i64::from(shape[0]), shape[1]);
    let (q_out, q_in) = matrix(weights, "encoder.layers.0.self_attn.q_proj.weight")?;
    if q_in != i64::from(hidden) || q_out != i64::from(hidden) {
        return Err(Error::new(
            "Qwen-Image text",
            format!("q_proj {q_out}x{q_in} does not match hidden {hidden}"),
        ));
    }
    let (k_out, k_in) = matrix(weights, "encoder.layers.0.self_attn.k_proj.weight")?;
    if k_in != i64::from(hidden) || k_out < 1 || q_out % k_out != 0 {
        return Err(Error::new(
            "Qwen-Image text",
            format!("k_proj {k_out}x{k_in} does not divide q width {q_out}"),
        ));
    }
    // Qwen2.5-VL 7B geometry: head 128. Derive heads from the projection
    // widths and fail when they do not divide evenly.
    let head = 128i32;
    if hidden % head != 0 || (k_out as i32) % head != 0 {
        return Err(Error::new(
            "Qwen-Image text",
            format!("hidden {hidden} / kv {k_out} do not split into head {head}"),
        ));
    }
    let mut layers = 0u32;
    while weights.has(&format!("encoder.layers.{layers}.input_layernorm.weight")) {
        layers += 1;
    }
    if layers == 0 {
        return Err(Error::new("Qwen-Image text", "no encoder.layers.* found"));
    }
    Ok(TextSpec {
        vocab,
        hidden,
        layers,
        heads: hidden / head,
        kv_heads: k_out as i32 / head,
        head,
    })
}

fn matrix(weights: &MlxWeights, name: &str) -> Result<(i64, i64)> {
    let Some((out, inn)) = weights.linear(name)?.out_in()? else {
        return Err(Error::new(
            "Qwen-Image text",
            format!("{name} is not a matrix"),
        ));
    };
    Ok((out, inn))
}

/// Fill the image prompt template and tokenize with the bundle's
/// `tokenizer/tokenizer.json` (Qwen2 byte-level BPE, NFC). Truncation matches
/// main `max_length=1058`.
pub(crate) fn prompt_ids(
    bundle: &Path,
    prompt: &str,
    vocab: i64,
) -> std::result::Result<Vec<i32>, WeightError> {
    let filled = PROMPT_TEMPLATE.replacen("{}", prompt, 1);
    let mut ids = crate::common::qwen3::encode_plain(bundle, &filled, vocab)?;
    if ids.len() > MAX_LENGTH {
        ids.truncate(MAX_LENGTH);
    }
    if ids.len() <= DROP_IDX {
        return Err(WeightError::NotBuilt(format!(
            "prompt tokenized to {} tokens, the template prefix alone is {DROP_IDX}",
            ids.len()
        )));
    }
    Ok(ids)
}

/// Encode one prompt (no padding). Returns `[1, seq, hidden]` BF16 context
/// rows after the template prefix drop.
pub(super) fn encode_prompt(
    weights: &MlxWeights,
    spec: TextSpec,
    ids: &[i32],
    stream: &Stream,
) -> Result<Array> {
    encode_ids(weights, spec, ids, None, DROP_IDX, stream)
}

/// Edit prompt: token embeddings, with `<|image_pad|>` rows replaced by the
/// vision tower, then the 64-token system prefix is dropped.
pub(super) fn encode_edit_prompt(
    weights: &MlxWeights,
    spec: TextSpec,
    ids: &[i32],
    image_pad: i32,
    image_embeds: &Array,
    stream: &Stream,
) -> Result<Array> {
    encode_ids(
        weights,
        spec,
        ids,
        Some((image_pad, image_embeds)),
        EDIT_DROP_IDX,
        stream,
    )
}

/// Picture-prefix edit text, or the single vision-token template.
pub(super) fn edit_prompt_text(prompt: &str, images: usize, picture_prefix: bool) -> String {
    let user = if picture_prefix {
        let mut pictures = String::new();
        for index in 1..=images {
            pictures.push_str(&format!(
                "Picture {index}: <|vision_start|><|image_pad|><|vision_end|>"
            ));
        }
        format!("{pictures}{prompt}")
    } else {
        format!("<|vision_start|><|image_pad|><|vision_end|>{prompt}")
    };
    format!("{EDIT_SYSTEM}{user}<|im_end|>\n<|im_start|>assistant\n")
}

/// Replace each `<|image_pad|>` id with `counts[i]` copies, in order.
pub(super) fn expand_image_pads(
    ids: &[i32],
    image_pad: i32,
    counts: &[usize],
) -> std::result::Result<Vec<i32>, WeightError> {
    let found = ids.iter().filter(|id| **id == image_pad).count();
    if found != counts.len() {
        return Err(WeightError::NotBuilt(format!(
            "edit prompt has {found} image placeholders, {images} images",
            images = counts.len()
        )));
    }
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut image = 0usize;
    for id in ids {
        if *id == image_pad {
            let count = counts[image];
            if count == 0 {
                return Err(WeightError::NotBuilt(
                    "vision grid produced no image tokens".into(),
                ));
            }
            out.extend(std::iter::repeat_n(image_pad, count));
            image += 1;
        } else {
            out.push(*id);
        }
    }
    Ok(out)
}

pub(super) fn image_pad_id(bundle: &Path, vocab: i64) -> std::result::Result<i32, WeightError> {
    let ids = crate::common::qwen3::encode_plain(bundle, IMAGE_PAD, vocab)?;
    match ids.as_slice() {
        [id] => Ok(*id),
        other => Err(WeightError::NotBuilt(format!(
            "<|image_pad|> tokenized to {other:?}, expected one id"
        ))),
    }
}

fn encode_ids(
    weights: &MlxWeights,
    spec: TextSpec,
    ids: &[i32],
    image_embeds: Option<(i32, &Array)>,
    drop_idx: usize,
    stream: &Stream,
) -> Result<Array> {
    if ids.is_empty() {
        return Err(Error::new("Qwen-Image text", "prompt has no tokens"));
    }
    let sequence = ids.len() as i32;
    let indices = Array::from_i32_slice(ids, &[sequence])?;
    let embedding = weights.embedding("encoder.embed_tokens.weight", stream)?;
    let mut hidden = embedding
        .take_axis(&indices, 0, stream)?
        .reshape(&[1, sequence, spec.hidden], stream)?;
    if let Some((image_pad, image_embeds)) = image_embeds {
        hidden = splice_image_rows(&hidden, ids, image_pad, image_embeds, spec.hidden, stream)?;
    }
    let dtype = hidden.dtype();
    let (cos, sin) = position_embeddings(spec, sequence, dtype, stream)?;
    let mask = causal_mask(sequence, stream)?;
    for layer in 0..spec.layers {
        hidden = layer_forward(weights, spec, layer, &hidden, &mask, &cos, &sin, stream)?;
    }
    let norm = weights
        .tensor("encoder.norm.weight")?
        .astype(hidden.dtype(), stream)?;
    let hidden = hidden.rms_norm(Some(&norm), RMS_EPS, stream)?;
    if sequence as usize <= drop_idx {
        return Err(Error::new(
            "Qwen-Image text",
            format!("sequence {sequence} does not exceed the {drop_idx}-token prefix"),
        ));
    }
    let trimmed = hidden.slice(
        &[0, drop_idx as i32, 0],
        &[1, sequence, spec.hidden],
        &[1, 1, 1],
        stream,
    )?;
    trimmed.astype(Dtype::BFloat16, stream)
}

fn splice_image_rows(
    hidden: &Array,
    ids: &[i32],
    pad: i32,
    image_embeds: &Array,
    width: i32,
    stream: &Stream,
) -> Result<Array> {
    let image_shape = image_embeds.shape();
    if image_shape.len() != 2 || image_shape[1] != width {
        return Err(Error::new(
            "Qwen-Image text",
            format!("image embeds {image_shape:?} are not [tokens, {width}]"),
        ));
    }
    let pads = ids.iter().filter(|id| **id == pad).count() as i32;
    if pads != image_shape[0] {
        return Err(Error::new(
            "Qwen-Image text",
            format!(
                "prompt has {pads} image tokens, vision produced {}",
                image_shape[0]
            ),
        ));
    }
    let image_embeds = image_embeds.astype(hidden.dtype(), stream)?;
    let mut parts: Vec<Array> = Vec::new();
    let mut cursor = 0i32;
    let mut image_row = 0i32;
    while cursor < ids.len() as i32 {
        if ids[cursor as usize] == pad {
            let start = cursor;
            while (cursor as usize) < ids.len() && ids[cursor as usize] == pad {
                cursor += 1;
            }
            let count = cursor - start;
            parts.push(
                image_embeds
                    .slice(
                        &[image_row, 0],
                        &[image_row + count, width],
                        &[1, 1],
                        stream,
                    )?
                    .reshape(&[1, count, width], stream)?,
            );
            image_row += count;
        } else {
            let start = cursor;
            while (cursor as usize) < ids.len() && ids[cursor as usize] != pad {
                cursor += 1;
            }
            parts.push(hidden.slice(&[0, start, 0], &[1, cursor, width], &[1, 1, 1], stream)?);
        }
    }
    Array::concatenate_axis(&parts, 1, stream)
}

fn layer_forward(
    weights: &MlxWeights,
    spec: TextSpec,
    layer: u32,
    hidden: &Array,
    mask: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let path = format!("encoder.layers.{layer}");
    let norm = norm_weight(
        weights,
        &path,
        "input_layernorm.weight",
        hidden.dtype(),
        stream,
    )?;
    let normed = hidden.rms_norm(Some(&norm), RMS_EPS, stream)?;
    let attended = attention(weights, spec, &path, &normed, mask, cos, sin, stream)?;
    let hidden = Array::add(hidden, &attended, stream)?;
    let post = norm_weight(
        weights,
        &path,
        "post_attention_layernorm.weight",
        hidden.dtype(),
        stream,
    )?;
    let normed = hidden.rms_norm(Some(&post), RMS_EPS, stream)?;
    let gate = linear(weights, &path, "mlp.gate_proj.weight", &normed, stream)?.silu(stream)?;
    let up = linear(weights, &path, "mlp.up_proj.weight", &normed, stream)?;
    let mid = Array::multiply(&gate, &up, stream)?;
    let mlp = linear(weights, &path, "mlp.down_proj.weight", &mid, stream)?;
    Ok(Array::add(&hidden, &mlp, stream)?)
}

fn attention(
    weights: &MlxWeights,
    spec: TextSpec,
    path: &str,
    hidden: &Array,
    mask: &Array,
    cos: &Array,
    sin: &Array,
    stream: &Stream,
) -> Result<Array> {
    let shape = hidden.shape();
    let (batch, sequence) = (shape[0], shape[1]);
    let mut q = linear(weights, path, "self_attn.q_proj.weight", hidden, stream)?
        .reshape(&[batch, sequence, spec.heads, spec.head], stream)?;
    let mut k = linear(weights, path, "self_attn.k_proj.weight", hidden, stream)?
        .reshape(&[batch, sequence, spec.kv_heads, spec.head], stream)?;
    let v = linear(weights, path, "self_attn.v_proj.weight", hidden, stream)?
        .reshape(&[batch, sequence, spec.kv_heads, spec.head], stream)?;
    q = apply_rope(&q, cos, sin, stream)?;
    k = apply_rope(&k, cos, sin, stream)?;
    let groups = spec.heads / spec.kv_heads;
    let mut k = k;
    let mut v = v;
    if groups > 1 {
        k = k.repeat_axis(groups, 2, stream)?;
        v = v.repeat_axis(groups, 2, stream)?;
    }
    let q = q.transpose_axes(&[0, 2, 1, 3], stream)?;
    let k = k.transpose_axes(&[0, 2, 1, 3], stream)?;
    let v = v.transpose_axes(&[0, 2, 1, 3], stream)?;
    let dtype = q.dtype();
    let output = Array::scaled_dot_product_attention(
        &q.astype(Dtype::Float32, stream)?,
        &k.astype(Dtype::Float32, stream)?,
        &v.astype(Dtype::Float32, stream)?,
        1.0 / (spec.head as f32).sqrt(),
        "",
        Some(mask),
        stream,
    )?
    .astype(dtype, stream)?
    .transpose_axes(&[0, 2, 1, 3], stream)?
    .reshape(&[batch, sequence, spec.hidden], stream)?;
    linear(weights, path, "self_attn.o_proj.weight", &output, stream)
}

fn apply_rope(input: &Array, cos: &Array, sin: &Array, stream: &Stream) -> Result<Array> {
    let dtype = input.dtype();
    let input = input.astype(Dtype::Float32, stream)?;
    let dim = input.shape()[3];
    let parts = input.split_sections(&[dim / 2], -1, stream)?;
    let rotated =
        Array::concatenate_axis(&[parts[1].negative(stream)?, parts[0].clone()], -1, stream)?;
    let out = Array::add(
        &Array::multiply(&input, cos, stream)?,
        &Array::multiply(&rotated, sin, stream)?,
        stream,
    )?;
    out.astype(dtype, stream)
}

/// MRoPE with equal positions on all three axes is plain half-split RoPE.
/// Tables are `[1, 1, S, head]`, built with MLX ops in F32 like main.
fn position_embeddings(
    spec: TextSpec,
    sequence: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let dim = spec.head;
    let arange = Array::arange(0.0, f64::from(dim), 2.0, Dtype::Float32, stream)?;
    let dim_scalar = Array::from_f32_slice(&[dim as f32], &[])?;
    let exponent = Array::divide(&arange, &dim_scalar, stream)?;
    let base = Array::from_f32_slice(&[ROPE_THETA], &[])?;
    let power = Array::power(&base, &exponent, stream)?;
    let one = Array::from_f32_slice(&[1.0], &[])?;
    let inv_freq = Array::divide(&one, &power, stream)?;
    let positions = Array::arange(0.0, f64::from(sequence), 1.0, Dtype::Float32, stream)?;
    let freqs = Array::multiply(
        &positions.expand_dims(1, stream)?,
        &inv_freq.expand_dims(0, stream)?,
        stream,
    )?;
    let emb = Array::concatenate_axis(&[freqs.clone(), freqs], -1, stream)?;
    let cos = emb
        .cos(stream)?
        .expand_dims(0, stream)?
        .expand_dims(2, stream)?;
    let sin = emb
        .sin(stream)?
        .expand_dims(0, stream)?
        .expand_dims(2, stream)?;
    Ok((cos.astype(dtype, stream)?, sin.astype(dtype, stream)?))
}

fn causal_mask(sequence: i32, stream: &Stream) -> Result<Array> {
    let n = sequence as usize;
    let mut mask = vec![f32::NEG_INFINITY; n * n];
    for query in 0..n {
        for key in 0..=query {
            mask[query * n + key] = 0.0;
        }
    }
    let mask = Array::from_f32_slice(&mask, &[1, 1, sequence, sequence])?;
    let _ = stream;
    Ok(mask)
}

fn norm_weight(
    weights: &MlxWeights,
    path: &str,
    name: &str,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array> {
    let tensor = weights.tensor(&format!("{path}.{name}"))?;
    if tensor.dtype() == dtype {
        Ok(tensor.clone())
    } else {
        tensor.astype(dtype, stream)
    }
}

fn linear(
    weights: &MlxWeights,
    path: &str,
    name: &str,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let full = format!("{path}.{name}");
    let bias_name = full.replace(".weight", ".bias");
    let bias = if weights.has(&bias_name) {
        Some(weights.tensor(&bias_name)?)
    } else {
        None
    };
    weights.linear(&full)?.apply(input, bias, stream)
}

#[cfg(test)]
mod tests {
    use super::{
        edit_prompt_text, expand_image_pads, prompt_ids, DROP_IDX, EDIT_DROP_IDX, MAX_LENGTH,
        PROMPT_TEMPLATE,
    };

    #[test]
    fn template_prefix_is_the_published_34_tokens() {
        // The system describe prompt plus the user/assistant scaffolding.
        assert!(PROMPT_TEMPLATE.starts_with("<|im_start|>system\n"));
        assert!(PROMPT_TEMPLATE.ends_with("<|im_start|>assistant\n"));
        assert_eq!(DROP_IDX, 34);
        assert_eq!(EDIT_DROP_IDX, 64);
        assert_eq!(MAX_LENGTH, 1058);
        let prefixed = edit_prompt_text("make it blue", 2, true);
        assert!(prefixed.contains("Picture 1: <|vision_start|><|image_pad|><|vision_end|>"));
        assert!(
            prefixed.contains("Picture 2: <|vision_start|><|image_pad|><|vision_end|>make it blue")
        );
        let plain = edit_prompt_text("make it blue", 1, false);
        assert!(plain.contains("<|vision_start|><|image_pad|><|vision_end|>make it blue"));
        assert!(!plain.contains("Picture 1:"));
        let expanded = expand_image_pads(&[1, 9, 2, 9], 9, &[2, 1]).unwrap();
        assert_eq!(expanded, vec![1, 9, 9, 2, 9]);
    }

    #[test]
    fn missing_tokenizer_fails_loud() {
        let dir = std::env::temp_dir().join(format!("qwen-image-text-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let err = prompt_ids(&dir, "a red apple", 152_064).unwrap_err();
        assert!(err.to_string().contains("tokenizer"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
