//! HunyuanVideo 1.5 i2v vision conditioning: the FLUX.1-Redux SigLIP tower
//! (main `image_encoder_mlx.py`). 27 layers, hidden 1152, 16 heads of 72,
//! intermediate 4304, patch 14 at 384px (729 tokens), LayerNorm eps 1e-6,
//! quick-gelu (`x * sigmoid(1.702x)`, main's `gelu_pytorch_tanh`), and a final
//! post_layernorm. Preprocess is `SiglipImageProcessor`: PIL bicubic to
//! 384x384, /255, then (x - 0.5) / 0.5. The weight names are the safetensors
//! names (`vision_model.*`); there is no remap.

use crate::common::image::resize_bicubic;
use crate::runtime::{flash_attend, GraphBuilder, RuntimeError, Tensor, TYPE_F32};
use crate::weights::{GgufWeights, WeightError};
use std::collections::HashMap;

const LN_EPS: f32 = 1e-6;
const QUICK_GELU_SCALE: f32 = 1.702;

#[derive(Debug, Clone, Copy)]
pub struct SiglipConfig {
    pub hidden: i64,
    pub layers: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub intermediate: i64,
    pub patch: i64,
    pub image: i64,
}

impl SiglipConfig {
    /// FLUX.1-Redux `image_encoder/config.json`.
    pub fn published() -> Self {
        Self {
            hidden: 1152,
            layers: 27,
            heads: 16,
            head_dim: 72,
            intermediate: 4304,
            patch: 14,
            image: 384,
        }
    }

    /// Two-layer smoke tower: patch 2 on a 4x4 image gives 4 tokens.
    pub fn smoke() -> Self {
        Self {
            hidden: 8,
            layers: 2,
            heads: 2,
            head_dim: 4,
            intermediate: 16,
            patch: 2,
            image: 4,
        }
    }

    pub fn patches(&self) -> i64 {
        (self.image / self.patch) * (self.image / self.patch)
    }
}

/// `(name, ggml dims)`. The patch conv kernel is `[KW, KH, IC, OC]` (the
/// safetensors `[OC, IC, KH, KW]` reversed); matmul weights are `[in, out]`.
pub fn siglip_layout(config: SiglipConfig) -> Vec<(String, Vec<i64>)> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<(String, Vec<i64>)>, name: String, dims: &[i64]| {
        out.push((name, dims.to_vec()));
    };
    push(
        &mut out,
        "vision_model.embeddings.patch_embedding.weight".into(),
        &[config.patch, config.patch, 3, config.hidden],
    );
    push(
        &mut out,
        "vision_model.embeddings.patch_embedding.bias".into(),
        &[config.hidden],
    );
    push(
        &mut out,
        "vision_model.embeddings.position_embedding.weight".into(),
        &[config.hidden, config.patches()],
    );
    for layer in 0..config.layers {
        let p = format!("vision_model.encoder.layers.{layer}");
        for norm in ["layer_norm1", "layer_norm2"] {
            push(&mut out, format!("{p}.{norm}.weight"), &[config.hidden]);
            push(&mut out, format!("{p}.{norm}.bias"), &[config.hidden]);
        }
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            push(
                &mut out,
                format!("{p}.self_attn.{proj}.weight"),
                &[config.hidden, config.hidden],
            );
            push(
                &mut out,
                format!("{p}.self_attn.{proj}.bias"),
                &[config.hidden],
            );
        }
        push(
            &mut out,
            format!("{p}.mlp.fc1.weight"),
            &[config.hidden, config.intermediate],
        );
        push(
            &mut out,
            format!("{p}.mlp.fc1.bias"),
            &[config.intermediate],
        );
        push(
            &mut out,
            format!("{p}.mlp.fc2.weight"),
            &[config.intermediate, config.hidden],
        );
        push(&mut out, format!("{p}.mlp.fc2.bias"), &[config.hidden]);
    }
    push(
        &mut out,
        "vision_model.post_layernorm.weight".into(),
        &[config.hidden],
    );
    push(
        &mut out,
        "vision_model.post_layernorm.bias".into(),
        &[config.hidden],
    );
    out
}

pub fn layout_matches(weights: &GgufWeights, config: SiglipConfig) -> bool {
    siglip_layout(config).iter().all(|(name, dims)| {
        let mut expected = [1i64; 4];
        for (axis, dim) in dims.iter().enumerate() {
            expected[axis] = *dim;
        }
        weights
            .shape(name)
            .map(|got| got == expected)
            .unwrap_or(false)
    })
}

pub struct SiglipForward {
    graph: crate::runtime::ReusedGraph,
    config: SiglipConfig,
}

impl SiglipForward {
    pub fn build(
        ctx: &crate::runtime::GgmlContext,
        config: SiglipConfig,
    ) -> Result<Self, RuntimeError> {
        if config.hidden < 1
            || config.heads < 1
            || config.head_dim < 2
            || config.heads * config.head_dim != config.hidden
            || config.image % config.patch != 0
        {
            return Err(RuntimeError::Message(
                "siglip head geometry does not match the hidden width".into(),
            ));
        }
        let mut builder = ctx.builder(32 * 1024 * 1024)?;
        let output = build_tower(&mut builder, config)?;
        let graph = builder.finish(output)?;
        Ok(Self { graph, config })
    }

    pub fn load_f32(&mut self, name: &str, data: &[f32]) -> Result<(), RuntimeError> {
        self.graph.set_f32(name, data)
    }

    /// `pixels` is the planar `[W, H, 3]` f32 buffer from [`preprocess`].
    /// Returns `[hidden, patches]` with the hidden dim on `ne0`.
    pub fn compute(&mut self, pixels: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        let expected = (self.config.image * self.config.image * 3) as usize;
        if pixels.len() != expected {
            return Err(RuntimeError::Message(format!(
                "siglip pixels {} is not {}x{}x3",
                pixels.len(),
                self.config.image,
                self.config.image
            )));
        }
        self.graph.set_f32("pixels", pixels)?;
        let out = self.graph.compute()?;
        let expected = (self.config.hidden * self.config.patches()) as usize;
        if out.len() != expected {
            return Err(RuntimeError::Message(format!(
                "siglip tower returned {} values, expected {}",
                out.len(),
                expected
            )));
        }
        Ok(out)
    }
}

/// Interleaved RGB → bicubic 384x384 → planar `[W, H, 3]` f32 in `[-1, 1]`
/// (SiglipImageProcessor rescale 1/255, mean/std 0.5).
pub fn preprocess(
    rgb: &[u8],
    width: u32,
    height: u32,
    config: SiglipConfig,
) -> Result<Vec<f32>, WeightError> {
    let resized = resize_bicubic(rgb, width, height, config.image as u32, config.image as u32)?;
    let side = config.image as usize;
    let plane = side * side;
    let mut out = vec![0f32; plane * 3];
    for y in 0..side {
        for x in 0..side {
            for channel in 0..3 {
                let byte = resized[(y * side + x) * 3 + channel];
                out[x + y * side + channel * plane] = (byte as f32 / 255.0 - 0.5) / 0.5;
            }
        }
    }
    Ok(out)
}

/// RGB source image → SigLIP `[hidden, patches]` embeddings (hidden on `ne0`).
pub fn encode(
    ctx: &crate::runtime::GgmlContext,
    weights: &GgufWeights,
    config: SiglipConfig,
    rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, WeightError> {
    if !layout_matches(weights, config) {
        return Err(WeightError::NotBuilt(
            "SigLIP image encoder layout does not match the tower config".into(),
        ));
    }
    let pixels = preprocess(rgb, width, height, config)?;
    let mut tower = SiglipForward::build(ctx, config)?;
    for (name, dims) in siglip_layout(config) {
        let data = weights.f32_widened(&name, &dims)?;
        tower.load_f32(&name, &data)?;
    }
    Ok(tower.compute(&pixels)?)
}

fn build_tower(b: &mut GraphBuilder, config: SiglipConfig) -> Result<Tensor, RuntimeError> {
    let mut weights = HashMap::new();
    for (name, dims) in siglip_layout(config) {
        let tensor = match dims.len() {
            1 => b.tensor_1d(&name, TYPE_F32, dims[0], true),
            2 => b.tensor_2d(&name, TYPE_F32, dims[0], dims[1], true),
            4 => b.f32_4d(&name, dims[0], dims[1], dims[2], dims[3], true),
            other => {
                return Err(RuntimeError::Message(format!(
                    "siglip weight {name} has {other} dims"
                )))
            }
        };
        if tensor.is_null() {
            return Err(RuntimeError::Init);
        }
        weights.insert(name, tensor);
    }
    let pixels = nn(b.f32_4d("pixels", config.image, config.image, 3, 1, true))?;
    let kernel = wt(&weights, "vision_model.embeddings.patch_embedding.weight")?;
    let conv = nn(b.conv2d_direct(
        kernel,
        pixels,
        config.patch as i32,
        config.patch as i32,
        0,
        0,
        1,
        1,
    ))?;
    // [grid, grid, hidden, 1] → [hidden, grid, grid] → [hidden, patches]; the
    // token index is `px + grid * py`, the row-major patch order. ggml_permute
    // takes destination axes: ne[1] = old ne[0], ne[2] = old ne[1], ne[0] =
    // old ne[2].
    let mut x = nn(b.reshape_2d(
        nn(b.cont(b.permute(conv, 1, 2, 0, 3)))?,
        config.hidden,
        config.patches(),
    ))?;
    let patch_bias = wt(&weights, "vision_model.embeddings.patch_embedding.bias")?;
    x = nn(b.add(x, nn(b.repeat(patch_bias, x))?))?;
    x = nn(b.add(
        x,
        wt(
            &weights,
            "vision_model.embeddings.position_embedding.weight",
        )?,
    ))?;
    for layer in 0..config.layers {
        x = block(b, &weights, config, layer, x)?;
    }
    layer_norm(
        b,
        x,
        wt(&weights, "vision_model.post_layernorm.weight")?,
        wt(&weights, "vision_model.post_layernorm.bias")?,
    )
}

fn block(
    b: &GraphBuilder,
    weights: &HashMap<String, Tensor>,
    config: SiglipConfig,
    layer: i64,
    x: Tensor,
) -> Result<Tensor, RuntimeError> {
    let p = format!("vision_model.encoder.layers.{layer}");
    let normed = layer_norm(
        b,
        x,
        wt(weights, &format!("{p}.layer_norm1.weight"))?,
        wt(weights, &format!("{p}.layer_norm1.bias"))?,
    )?;
    let seq = config.patches();
    let heads = |w: &str, bias: &str| -> Result<Tensor, RuntimeError> {
        let projected = biased(b, normed, wt(weights, w)?, wt(weights, bias)?)?;
        nn(b.cont(b.reshape_3d(projected, config.head_dim, config.heads, seq)))
    };
    let q = heads(
        &format!("{p}.self_attn.q_proj.weight"),
        &format!("{p}.self_attn.q_proj.bias"),
    )?;
    let k = heads(
        &format!("{p}.self_attn.k_proj.weight"),
        &format!("{p}.self_attn.k_proj.bias"),
    )?;
    let v = heads(
        &format!("{p}.self_attn.v_proj.weight"),
        &format!("{p}.self_attn.v_proj.bias"),
    )?;
    // Full bidirectional attention: no mask.
    let mixed = flash_attend(b, q, k, v, std::ptr::null_mut())?;
    let mixed = nn(b.reshape_2d(mixed, config.hidden, seq))?;
    let attn = biased(
        b,
        mixed,
        wt(weights, &format!("{p}.self_attn.out_proj.weight"))?,
        wt(weights, &format!("{p}.self_attn.out_proj.bias"))?,
    )?;
    let x = nn(b.add(x, attn))?;
    let normed = layer_norm(
        b,
        x,
        wt(weights, &format!("{p}.layer_norm2.weight"))?,
        wt(weights, &format!("{p}.layer_norm2.bias"))?,
    )?;
    let h = biased(
        b,
        normed,
        wt(weights, &format!("{p}.mlp.fc1.weight"))?,
        wt(weights, &format!("{p}.mlp.fc1.bias"))?,
    )?;
    let h = nn(b.mul(h, nn(b.sigmoid(b.scale(h, QUICK_GELU_SCALE)))?))?;
    let h = biased(
        b,
        h,
        wt(weights, &format!("{p}.mlp.fc2.weight"))?,
        wt(weights, &format!("{p}.mlp.fc2.bias"))?,
    )?;
    nn(b.add(x, h))
}

fn layer_norm(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Result<Tensor, RuntimeError> {
    let normed = nn(b.norm(x, LN_EPS))?;
    let scaled = nn(b.mul(normed, nn(b.repeat(weight, normed))?))?;
    nn(b.add(scaled, nn(b.repeat(bias, scaled))?))
}

fn biased(
    b: &GraphBuilder,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
) -> Result<Tensor, RuntimeError> {
    let projected = nn(b.mul_mat(weight, x))?;
    nn(b.add(projected, nn(b.repeat(bias, projected))?))
}

fn wt(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor, RuntimeError> {
    weights
        .get(name)
        .copied()
        .filter(|tensor| !tensor.is_null())
        .ok_or_else(|| RuntimeError::Message(format!("graph is missing {name}")))
}

fn nn(tensor: Tensor) -> Result<Tensor, RuntimeError> {
    if tensor.is_null() {
        Err(RuntimeError::Init)
    } else {
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::{layout_matches, preprocess, siglip_layout, SiglipConfig, SiglipForward};
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{write_safetensors, GgufWeights, SafeTensor};
    use std::collections::HashMap;

    fn tap(name: &str, index: usize) -> f32 {
        let mut hash = 2166136261u32;
        for byte in name.bytes() {
            hash = hash.wrapping_mul(16777619) ^ u32::from(byte);
        }
        hash = hash.wrapping_mul(16777619) ^ index as u32;
        ((hash % 7) as f32 - 3.0) * 0.05
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn write_tower(dir: &std::path::Path, config: SiglipConfig) -> std::path::PathBuf {
        let mut tensors = Vec::new();
        let mut owned = Vec::new();
        for (name, dims) in siglip_layout(config) {
            // dims are ggml order; the safetensors shape is the reverse.
            let data: Vec<f32> = (0..dims.iter().product::<i64>() as usize)
                .map(|i| tap(&name, i))
                .collect();
            let shape: Vec<u64> = dims.iter().rev().map(|d| *d as u64).collect();
            owned.push((name, shape, f32_bytes(&data)));
        }
        for (name, shape, bytes) in &owned {
            tensors.push(SafeTensor {
                name,
                dtype: "F32",
                shape,
                data: bytes,
            });
        }
        let path = dir.join("model.safetensors");
        write_safetensors(&path, &tensors).unwrap();
        path
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-hy-siglip-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn published_geometry_matches_the_redux_config() {
        let config = SiglipConfig::published();
        assert_eq!(config.hidden, 1152);
        assert_eq!(config.layers, 27);
        assert_eq!(config.heads * config.head_dim, 1152);
        assert_eq!(config.intermediate, 4304);
        assert_eq!(config.patches(), 729);
    }

    #[test]
    fn smoke_tower_matches_the_host_reference() {
        let dir = temp_dir("host");
        let config = SiglipConfig::smoke();
        let path = write_tower(&dir, config);
        let weights = GgufWeights::open_parts(&[path]).unwrap();
        assert!(layout_matches(&weights, config));
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut tower = SiglipForward::build(&ctx, config).unwrap();
        let mut host_weights = HashMap::new();
        for (name, dims) in siglip_layout(config) {
            let data = weights.f32_widened(&name, &dims).unwrap();
            tower.load_f32(&name, &data).unwrap();
            host_weights.insert(name, data);
        }
        let side = config.image as usize;
        let mut rgb = vec![0u8; side * side * 3];
        for (i, byte) in rgb.iter_mut().enumerate() {
            *byte = ((i * 37 + 11) % 256) as u8;
        }
        let pixels = preprocess(&rgb, side as u32, side as u32, config).unwrap();
        let got = tower.compute(&pixels).unwrap();
        let expect = host_tower(&host_weights, config, &pixels);
        assert_eq!(got.len(), expect.len());
        let mut worst = 0f32;
        for (got, want) in got.iter().zip(&expect) {
            worst = worst.max((got - want).abs());
        }
        assert!(worst < 2e-3, "siglip tower max abs {worst}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn at(data: &[f32], width: usize, row: usize, col: usize) -> f32 {
        data[row + width * col]
    }

    fn host_tower(
        weights: &HashMap<String, Vec<f32>>,
        config: SiglipConfig,
        pixels: &[f32],
    ) -> Vec<f32> {
        let hidden = config.hidden as usize;
        let patch = config.patch as usize;
        let grid = (config.image / config.patch) as usize;
        let patches = grid * grid;
        // Patch conv as a matmul: kernel row (c, ky, kx) is index
        // kx + patch*ky + patch*patch*c in the ggml [KW, KH, IC, OC] layout.
        let kernel = &weights["vision_model.embeddings.patch_embedding.weight"];
        let bias = &weights["vision_model.embeddings.patch_embedding.bias"];
        let mut x = vec![0f32; hidden * patches];
        for py in 0..grid {
            for px in 0..grid {
                let token = px + grid * py;
                for o in 0..hidden {
                    let mut acc = bias[o];
                    for c in 0..3 {
                        for ky in 0..patch {
                            for kx in 0..patch {
                                let krow = kx + patch * ky + patch * patch * c;
                                let pixel = pixels[(px * patch + kx)
                                    + (py * patch + ky) * config.image as usize
                                    + c * (config.image as usize).pow(2)];
                                acc += kernel[krow + patch * patch * 3 * o] * pixel;
                            }
                        }
                    }
                    x[o + hidden * token] = acc;
                }
            }
        }
        let pos = &weights["vision_model.embeddings.position_embedding.weight"];
        for i in 0..x.len() {
            x[i] += pos[i];
        }
        for layer in 0..config.layers {
            x = host_block(weights, config, layer, &x, patches);
        }
        host_ln(
            &x,
            hidden,
            patches,
            &weights["vision_model.post_layernorm.weight"],
            &weights["vision_model.post_layernorm.bias"],
        )
    }

    fn host_block(
        weights: &HashMap<String, Vec<f32>>,
        config: SiglipConfig,
        layer: i64,
        x: &[f32],
        seq: usize,
    ) -> Vec<f32> {
        let p = format!("vision_model.encoder.layers.{layer}");
        let hidden = config.hidden as usize;
        let head = config.head_dim as usize;
        let heads = config.heads as usize;
        let normed = host_ln(
            x,
            hidden,
            seq,
            &weights[&format!("{p}.layer_norm1.weight")],
            &weights[&format!("{p}.layer_norm1.bias")],
        );
        let project = |name: &str| -> Vec<f32> {
            host_linear(
                &normed,
                hidden,
                &weights[&format!("{p}.self_attn.{name}.weight")],
                hidden,
                seq,
                Some(&weights[&format!("{p}.self_attn.{name}.bias")]),
            )
        };
        let q = project("q_proj");
        let k = project("k_proj");
        let v = project("v_proj");
        let scale = 1.0 / (head as f32).sqrt();
        let mut mixed = vec![0f32; hidden * seq];
        for h in 0..heads {
            for qt in 0..seq {
                let mut scores = vec![0f32; seq];
                let mut top = f32::NEG_INFINITY;
                for kt in 0..seq {
                    let mut dot = 0f32;
                    for d in 0..head {
                        dot += at(&q, head, d, h + heads * qt) * at(&k, head, d, h + heads * kt);
                    }
                    scores[kt] = dot * scale;
                    top = top.max(scores[kt]);
                }
                let mut denom = 0f32;
                for score in &mut scores {
                    *score = (*score - top).exp();
                    denom += *score;
                }
                for kt in 0..seq {
                    let w = scores[kt] / denom;
                    for d in 0..head {
                        mixed[d + head * (h + heads * qt)] += w * at(&v, head, d, h + heads * kt);
                    }
                }
            }
        }
        let attn = host_linear(
            &mixed,
            hidden,
            &weights[&format!("{p}.self_attn.out_proj.weight")],
            hidden,
            seq,
            Some(&weights[&format!("{p}.self_attn.out_proj.bias")]),
        );
        let mut x: Vec<f32> = x.iter().zip(&attn).map(|(a, b)| a + b).collect();
        let normed = host_ln(
            &x,
            hidden,
            seq,
            &weights[&format!("{p}.layer_norm2.weight")],
            &weights[&format!("{p}.layer_norm2.bias")],
        );
        let inter = config.intermediate as usize;
        let h = host_linear(
            &normed,
            hidden,
            &weights[&format!("{p}.mlp.fc1.weight")],
            inter,
            seq,
            Some(&weights[&format!("{p}.mlp.fc1.bias")]),
        );
        let h: Vec<f32> = h.iter().map(|v| *v / (1.0 + (-1.702 * *v).exp())).collect();
        let h = host_linear(
            &h,
            inter,
            &weights[&format!("{p}.mlp.fc2.weight")],
            hidden,
            seq,
            Some(&weights[&format!("{p}.mlp.fc2.bias")]),
        );
        for (value, add) in x.iter_mut().zip(&h) {
            *value += add;
        }
        x
    }

    fn host_linear(
        x: &[f32],
        input: usize,
        weight: &[f32],
        output: usize,
        seq: usize,
        bias: Option<&Vec<f32>>,
    ) -> Vec<f32> {
        let mut y = vec![0f32; output * seq];
        for token in 0..seq {
            for o in 0..output {
                let mut acc = bias.map(|row| row[o]).unwrap_or(0.0);
                for i in 0..input {
                    acc += weight[i + input * o] * x[i + input * token];
                }
                y[o + output * token] = acc;
            }
        }
        y
    }

    fn host_ln(x: &[f32], width: usize, seq: usize, weight: &[f32], bias: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        for token in 0..seq {
            let mut mean = 0f32;
            for h in 0..width {
                mean += x[h + width * token];
            }
            mean /= width as f32;
            let mut var = 0f32;
            for h in 0..width {
                let d = x[h + width * token] - mean;
                var += d * d;
            }
            var /= width as f32;
            let inv = 1.0 / (var + 1e-6).sqrt();
            for h in 0..width {
                y[h + width * token] = (x[h + width * token] - mean) * inv * weight[h] + bias[h];
            }
        }
        y
    }
}
