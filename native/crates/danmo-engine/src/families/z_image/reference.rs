//! Host forward used to check the ggml graph. The math is Tongyi / `main`, which
//! is the same complex RoPE, SwiGLU, AdaLN, and `1/sqrt(head)` attention SD.cpp uses.
//! `text_first` is SD.cpp's caption-then-image order.

use super::geom::{positions_image_first, timestep_features, TokenGrid, FINAL_EPS, PATCH, RMS_EPS};
use super::weights::ZImageConfig;
use std::collections::HashMap;

pub fn velocity(
    weights: &HashMap<String, Vec<f32>>,
    config: ZImageConfig,
    grid: TokenGrid,
    latent: &[f32],
    txt: &[f32],
    sigma: f32,
    text_first: bool,
) -> Vec<f32> {
    let t = (1.0 - sigma) * config.t_scale;
    let freq = timestep_features(t, config.freq as usize);
    let t_emb = timestep_mlp(weights, config, &freq);
    let ids = positions_image_first(grid);
    let (image_ids, caption_ids) = ids.split_at(grid.img_len as usize);
    let mut image = embed_image(weights, config, grid, latent);
    let mut caption = embed_caption(weights, config, grid, txt);
    for i in 0..config.refiners {
        image = block(
            weights,
            &format!("noise_refiner.{i}"),
            &image,
            image_ids,
            Some(&t_emb),
            config,
            true,
        );
        caption = block(
            weights,
            &format!("context_refiner.{i}"),
            &caption,
            caption_ids,
            None,
            config,
            false,
        );
    }
    let (unified, unified_ids) = if text_first {
        (
            concat_seq(&caption, &image, config.dim as usize),
            concat_ids(caption_ids, image_ids),
        )
    } else {
        (
            concat_seq(&image, &caption, config.dim as usize),
            concat_ids(image_ids, caption_ids),
        )
    };
    let mut unified = unified;
    for i in 0..config.layers {
        unified = block(
            weights,
            &format!("layers.{i}"),
            &unified,
            &unified_ids,
            Some(&t_emb),
            config,
            true,
        );
    }
    let tokens = final_layer(weights, config, &unified, &t_emb);
    let start = if text_first { grid.cap_len as usize } else { 0 };
    let real = slice_tokens(
        &tokens,
        config.token_dim() as usize,
        start,
        grid.img_real as usize,
    );
    unpatchify(&real, grid, config.in_channels as usize)
}

fn timestep_mlp(
    weights: &HashMap<String, Vec<f32>>,
    config: ZImageConfig,
    freq: &[f32],
) -> Vec<f32> {
    let hidden = add_bias(
        &linear(
            freq,
            w(weights, "t_embedder.mlp.0.weight"),
            config.freq as usize,
            config.mid as usize,
            1,
        ),
        w(weights, "t_embedder.mlp.0.bias"),
        config.mid as usize,
        1,
    );
    let hidden = silu(&hidden);
    add_bias(
        &linear(
            &hidden,
            w(weights, "t_embedder.mlp.2.weight"),
            config.mid as usize,
            config.adaln as usize,
            1,
        ),
        w(weights, "t_embedder.mlp.2.bias"),
        config.adaln as usize,
        1,
    )
}

fn embed_image(
    weights: &HashMap<String, Vec<f32>>,
    config: ZImageConfig,
    grid: TokenGrid,
    latent: &[f32],
) -> Vec<f32> {
    let tokens = patchify(latent, grid, config.in_channels as usize);
    let embedded = add_bias(
        &linear(
            &tokens,
            w(weights, "x_embedder.weight"),
            config.token_dim() as usize,
            config.dim as usize,
            grid.img_real as usize,
        ),
        w(weights, "x_embedder.bias"),
        config.dim as usize,
        grid.img_real as usize,
    );
    pad_seq(
        &embedded,
        config.dim as usize,
        w(weights, "x_pad_token"),
        grid.img_pad as usize,
    )
}

fn embed_caption(
    weights: &HashMap<String, Vec<f32>>,
    config: ZImageConfig,
    grid: TokenGrid,
    txt: &[f32],
) -> Vec<f32> {
    let normed = rms(
        txt,
        w(weights, "cap_embedder.0.weight"),
        config.cap_dim as usize,
        grid.cap_real as usize,
        RMS_EPS,
    );
    let embedded = add_bias(
        &linear(
            &normed,
            w(weights, "cap_embedder.1.weight"),
            config.cap_dim as usize,
            config.dim as usize,
            grid.cap_real as usize,
        ),
        w(weights, "cap_embedder.1.bias"),
        config.dim as usize,
        grid.cap_real as usize,
    );
    pad_seq(
        &embedded,
        config.dim as usize,
        w(weights, "cap_pad_token"),
        grid.cap_pad as usize,
    )
}

fn block(
    weights: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x: &[f32],
    ids: &[[i32; 3]],
    t_emb: Option<&[f32]>,
    config: ZImageConfig,
    modulated: bool,
) -> Vec<f32> {
    let dim = config.dim as usize;
    let seq = x.len() / dim;
    let (scale_msa, gate_msa, scale_mlp, gate_mlp) = if modulated {
        let packed = add_bias(
            &linear(
                t_emb.expect("adaln"),
                w(weights, &format!("{prefix}.adaLN_modulation.0.weight")),
                config.adaln as usize,
                dim * 4,
                1,
            ),
            w(weights, &format!("{prefix}.adaLN_modulation.0.bias")),
            dim * 4,
            1,
        );
        (
            chunk(&packed, 0, dim),
            tanh_vec(&chunk(&packed, dim, dim)),
            chunk(&packed, dim * 2, dim),
            tanh_vec(&chunk(&packed, dim * 3, dim)),
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
    let normed = rms(
        x,
        w(weights, &format!("{prefix}.attention_norm1.weight")),
        dim,
        seq,
        RMS_EPS,
    );
    let normed = if modulated {
        modulate(&normed, &scale_msa, dim, seq)
    } else {
        normed
    };
    let attn = attention(weights, prefix, &normed, ids, config);
    let attn = rms(
        &attn,
        w(weights, &format!("{prefix}.attention_norm2.weight")),
        dim,
        seq,
        RMS_EPS,
    );
    let x = if modulated {
        gate_add(x, &gate_msa, &attn, dim, seq)
    } else {
        add(x, &attn)
    };
    let normed = rms(
        &x,
        w(weights, &format!("{prefix}.ffn_norm1.weight")),
        dim,
        seq,
        RMS_EPS,
    );
    let normed = if modulated {
        modulate(&normed, &scale_mlp, dim, seq)
    } else {
        normed
    };
    let ffn = swiglu(weights, prefix, &normed, config);
    let ffn = rms(
        &ffn,
        w(weights, &format!("{prefix}.ffn_norm2.weight")),
        dim,
        seq,
        RMS_EPS,
    );
    if modulated {
        gate_add(&x, &gate_mlp, &ffn, dim, seq)
    } else {
        add(&x, &ffn)
    }
}

fn attention(
    weights: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x: &[f32],
    ids: &[[i32; 3]],
    config: ZImageConfig,
) -> Vec<f32> {
    let dim = config.dim as usize;
    let head = config.head_dim as usize;
    let heads = config.heads as usize;
    let seq = x.len() / dim;
    let qkv = linear(
        x,
        w(weights, &format!("{prefix}.attention.qkv.weight")),
        dim,
        dim * 3,
        seq,
    );
    let (mut q, mut k, v) = split_qkv(&qkv, head, heads, seq);
    q = rms_heads(
        &q,
        w(weights, &format!("{prefix}.attention.q_norm.weight")),
        head,
        heads,
        seq,
        RMS_EPS,
    );
    k = rms_heads(
        &k,
        w(weights, &format!("{prefix}.attention.k_norm.weight")),
        head,
        heads,
        seq,
        RMS_EPS,
    );
    rope_seq(
        &mut q,
        ids,
        config.axes,
        config.rope_theta,
        head,
        heads,
        seq,
    );
    rope_seq(
        &mut k,
        ids,
        config.axes,
        config.rope_theta,
        head,
        heads,
        seq,
    );
    let mixed = attend(&q, &k, &v, head, heads, seq);
    linear(
        &mixed,
        w(weights, &format!("{prefix}.attention.out.weight")),
        dim,
        dim,
        seq,
    )
}

fn swiglu(
    weights: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x: &[f32],
    config: ZImageConfig,
) -> Vec<f32> {
    let dim = config.dim as usize;
    let hidden = config.ffn as usize;
    let seq = x.len() / dim;
    let gate = linear(
        x,
        w(weights, &format!("{prefix}.feed_forward.w1.weight")),
        dim,
        hidden,
        seq,
    );
    let up = linear(
        x,
        w(weights, &format!("{prefix}.feed_forward.w3.weight")),
        dim,
        hidden,
        seq,
    );
    let mut mixed = vec![0f32; gate.len()];
    for (slot, (gate, up)) in mixed.iter_mut().zip(gate.iter().zip(up.iter())) {
        *slot = silu_scalar(*gate) * up;
    }
    linear(
        &mixed,
        w(weights, &format!("{prefix}.feed_forward.w2.weight")),
        hidden,
        dim,
        seq,
    )
}

fn final_layer(
    weights: &HashMap<String, Vec<f32>>,
    config: ZImageConfig,
    x: &[f32],
    t_emb: &[f32],
) -> Vec<f32> {
    let dim = config.dim as usize;
    let seq = x.len() / dim;
    let token = config.token_dim() as usize;
    let activated = silu(t_emb);
    let scale = add_bias(
        &linear(
            &activated,
            w(weights, "final_layer.adaLN_modulation.1.weight"),
            config.adaln as usize,
            dim,
            1,
        ),
        w(weights, "final_layer.adaLN_modulation.1.bias"),
        dim,
        1,
    );
    let normed = layer_norm(x, dim, seq, FINAL_EPS);
    let shifted = modulate(&normed, &scale, dim, seq);
    add_bias(
        &linear(
            &shifted,
            w(weights, "final_layer.linear.weight"),
            dim,
            token,
            seq,
        ),
        w(weights, "final_layer.linear.bias"),
        token,
        seq,
    )
}

fn patchify(latent: &[f32], grid: TokenGrid, channels: usize) -> Vec<f32> {
    let token_dim = (PATCH * PATCH) as usize * channels;
    let width = grid.latent_w as usize;
    let height = grid.latent_h as usize;
    let mut tokens = vec![0f32; token_dim * grid.img_real as usize];
    let mut token = 0usize;
    for row in 0..grid.tokens_h as usize {
        for col in 0..grid.tokens_w as usize {
            let mut element = 0usize;
            for ph in 0..PATCH as usize {
                for pw in 0..PATCH as usize {
                    for channel in 0..channels {
                        let x = col * PATCH as usize + pw;
                        let y = row * PATCH as usize + ph;
                        let src = x + y * width + channel * width * height;
                        tokens[element + token * token_dim] = latent[src];
                        element += 1;
                    }
                }
            }
            token += 1;
        }
    }
    tokens
}

fn unpatchify(tokens: &[f32], grid: TokenGrid, channels: usize) -> Vec<f32> {
    let token_dim = (PATCH * PATCH) as usize * channels;
    let width = grid.latent_w as usize;
    let height = grid.latent_h as usize;
    let mut latent = vec![0f32; width * height * channels];
    for token in 0..grid.img_real as usize {
        let row = token / grid.tokens_w as usize;
        let col = token % grid.tokens_w as usize;
        let mut element = 0usize;
        for ph in 0..PATCH as usize {
            for pw in 0..PATCH as usize {
                for channel in 0..channels {
                    let x = col * PATCH as usize + pw;
                    let y = row * PATCH as usize + ph;
                    let dst = x + y * width + channel * width * height;
                    latent[dst] = -tokens[element + token * token_dim];
                    element += 1;
                }
            }
        }
    }
    latent
}

fn split_qkv(qkv: &[f32], head: usize, heads: usize, seq: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let dim = head * heads;
    let mut q = vec![0f32; dim * seq];
    let mut k = vec![0f32; dim * seq];
    let mut v = vec![0f32; dim * seq];
    for s in 0..seq {
        for h in 0..heads {
            for d in 0..head {
                let src = d + h * head;
                q[d + h * head + s * dim] = qkv[src + s * dim * 3];
                k[d + h * head + s * dim] = qkv[(src + dim) + s * dim * 3];
                v[d + h * head + s * dim] = qkv[(src + dim * 2) + s * dim * 3];
            }
        }
    }
    (q, k, v)
}

fn attend(q: &[f32], k: &[f32], v: &[f32], head: usize, heads: usize, seq: usize) -> Vec<f32> {
    let scale = 1.0 / (head as f32).sqrt();
    let dim = head * heads;
    let mut out = vec![0f32; dim * seq];
    for h in 0..heads {
        let mut scores = vec![0f32; seq * seq];
        for i in 0..seq {
            let mut max = f32::NEG_INFINITY;
            for j in 0..seq {
                let mut dot = 0f32;
                for d in 0..head {
                    dot += q[d + h * head + i * dim] * k[d + h * head + j * dim];
                }
                let score = dot * scale;
                scores[j + i * seq] = score;
                max = max.max(score);
            }
            let mut sum = 0f32;
            for j in 0..seq {
                let e = (scores[j + i * seq] - max).exp();
                scores[j + i * seq] = e;
                sum += e;
            }
            for d in 0..head {
                let mut acc = 0f32;
                for j in 0..seq {
                    acc += scores[j + i * seq] / sum * v[d + h * head + j * dim];
                }
                out[d + h * head + i * dim] = acc;
            }
        }
    }
    out
}

fn rope_seq(
    x: &mut [f32],
    ids: &[[i32; 3]],
    axes: [i64; 3],
    theta: f32,
    head: usize,
    heads: usize,
    seq: usize,
) {
    let dim = head * heads;
    for s in 0..seq {
        for h in 0..heads {
            let start = h * head + s * dim;
            let mut offset = 0usize;
            for (axis, id) in axes.iter().zip(ids[s]) {
                let width = *axis as usize;
                apply_rope(&mut x[start + offset..start + offset + width], id, theta);
                offset += width;
            }
        }
    }
}

fn apply_rope(values: &mut [f32], pos: i32, theta: f32) {
    let n = values.len();
    let mut angle = pos as f32;
    let step = theta.powf(-2.0 / n as f32);
    for pair in 0..n / 2 {
        let cos = angle.cos();
        let sin = angle.sin();
        let i = pair * 2;
        let x0 = values[i];
        let x1 = values[i + 1];
        values[i] = x0 * cos - x1 * sin;
        values[i + 1] = x0 * sin + x1 * cos;
        angle *= step;
    }
}

fn linear(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize, seq: usize) -> Vec<f32> {
    let mut y = vec![0f32; out_dim * seq];
    for s in 0..seq {
        for o in 0..out_dim {
            let mut acc = 0f32;
            for k in 0..in_dim {
                acc += w[k + o * in_dim] * x[k + s * in_dim];
            }
            y[o + s * out_dim] = acc;
        }
    }
    y
}

fn add_bias(x: &[f32], bias: &[f32], dim: usize, seq: usize) -> Vec<f32> {
    let mut y = x.to_vec();
    for s in 0..seq {
        for d in 0..dim {
            y[d + s * dim] += bias[d];
        }
    }
    y
}

fn rms(x: &[f32], weight: &[f32], dim: usize, seq: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for s in 0..seq {
        let mut sum = 0f32;
        for d in 0..dim {
            let v = x[d + s * dim];
            sum += v * v;
        }
        let scale = 1.0 / (sum / dim as f32 + eps).sqrt();
        for d in 0..dim {
            y[d + s * dim] = x[d + s * dim] * scale * weight[d];
        }
    }
    y
}

fn rms_heads(
    x: &[f32],
    weight: &[f32],
    head: usize,
    heads: usize,
    seq: usize,
    eps: f32,
) -> Vec<f32> {
    let dim = head * heads;
    let mut y = vec![0f32; x.len()];
    for s in 0..seq {
        for h in 0..heads {
            let mut sum = 0f32;
            for d in 0..head {
                let v = x[d + h * head + s * dim];
                sum += v * v;
            }
            let scale = 1.0 / (sum / head as f32 + eps).sqrt();
            for d in 0..head {
                y[d + h * head + s * dim] = x[d + h * head + s * dim] * scale * weight[d];
            }
        }
    }
    y
}

fn layer_norm(x: &[f32], dim: usize, seq: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for s in 0..seq {
        let mut sum = 0f32;
        for d in 0..dim {
            sum += x[d + s * dim];
        }
        let mean = sum / dim as f32;
        let mut var = 0f32;
        for d in 0..dim {
            let centered = x[d + s * dim] - mean;
            var += centered * centered;
        }
        var /= dim as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for d in 0..dim {
            y[d + s * dim] = (x[d + s * dim] - mean) * scale;
        }
    }
    y
}

fn modulate(x: &[f32], scale: &[f32], dim: usize, seq: usize) -> Vec<f32> {
    let mut y = vec![0f32; x.len()];
    for s in 0..seq {
        for d in 0..dim {
            y[d + s * dim] = x[d + s * dim] * (1.0 + scale[d]);
        }
    }
    y
}

fn gate_add(residual: &[f32], gate: &[f32], delta: &[f32], dim: usize, seq: usize) -> Vec<f32> {
    let mut y = vec![0f32; residual.len()];
    for s in 0..seq {
        for d in 0..dim {
            let i = d + s * dim;
            y[i] = residual[i] + delta[i] * gate[d];
        }
    }
    y
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().copied().map(silu_scalar).collect()
}

fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn tanh_vec(x: &[f32]) -> Vec<f32> {
    x.iter().copied().map(f32::tanh).collect()
}

fn chunk(packed: &[f32], start: usize, len: usize) -> Vec<f32> {
    packed[start..start + len].to_vec()
}

fn pad_seq(tokens: &[f32], dim: usize, pad: &[f32], pad_len: usize) -> Vec<f32> {
    let mut out = tokens.to_vec();
    for _ in 0..pad_len {
        out.extend_from_slice(&pad[..dim]);
    }
    out
}

fn concat_seq(left: &[f32], right: &[f32], dim: usize) -> Vec<f32> {
    let left_seq = left.len() / dim;
    let right_seq = right.len() / dim;
    let mut out = vec![0f32; dim * (left_seq + right_seq)];
    for s in 0..left_seq {
        for d in 0..dim {
            out[d + s * dim] = left[d + s * dim];
        }
    }
    for s in 0..right_seq {
        for d in 0..dim {
            out[d + (left_seq + s) * dim] = right[d + s * dim];
        }
    }
    out
}

fn concat_ids(left: &[[i32; 3]], right: &[[i32; 3]]) -> Vec<[i32; 3]> {
    let mut ids = left.to_vec();
    ids.extend_from_slice(right);
    ids
}

fn slice_tokens(tokens: &[f32], dim: usize, start: usize, count: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim * count];
    for s in 0..count {
        for d in 0..dim {
            out[d + s * dim] = tokens[d + (start + s) * dim];
        }
    }
    out
}

fn w<'a>(weights: &'a HashMap<String, Vec<f32>>, name: &str) -> &'a [f32] {
    weights
        .get(name)
        .unwrap_or_else(|| panic!("host reference is missing {name}"))
}

#[cfg(test)]
mod tests {
    use super::super::geom::{
        axes_for_head, ffn_hidden, pad_len, patch_index, positions_image_first,
        positions_text_first, sdcpp_ffn, timestep_features, token_grid, PATCH,
    };
    use super::super::transformer::ZImageDenoise;
    use super::super::weights::{weight_layout, ZImageConfig};
    use super::super::write_smoke_bundle;
    use super::velocity;
    use crate::common::ops::linear_sigmas;
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::GgufWeights;
    use std::collections::HashMap;

    #[test]
    fn published_constants_match_tongyi_and_sdcpp() {
        assert_eq!(pad_len(1, 32), 31);
        assert_eq!(pad_len(16, 32), 16);
        assert_eq!(pad_len(32, 32), 0);
        assert_eq!(ffn_hidden(3840), 10240);
        assert_eq!(sdcpp_ffn(3840, 256), 10240);
        assert_eq!(axes_for_head(128).unwrap(), [32, 48, 48]);
        assert!(axes_for_head(64).is_err());
        let emb = timestep_features(1000.0, 4);
        let second = 1000.0 * 10000f32.powf(-0.5);
        assert!((emb[0] - 1000f32.cos()).abs() < 1e-5);
        assert!((emb[1] - second.cos()).abs() < 1e-5, "cos {}", emb[1]);
        assert!((emb[2] - 1000f32.sin()).abs() < 1e-5);
        assert!((emb[3] - second.sin()).abs() < 1e-5);
    }

    #[test]
    fn sdcpp_position_ids_for_an_8_latent() {
        let grid = token_grid(64, 64, 1, 8).unwrap();
        let h_len = (8 + PATCH / 2) / PATCH;
        assert_eq!(h_len, grid.tokens_h);
        assert_eq!(grid.img_real, 16);
        assert_eq!(grid.cap_len, 32);
        assert_eq!(grid.img_pad, 16);
        let ids = positions_image_first(grid);
        assert_eq!(ids[0], [33, 0, 0]);
        assert_eq!(ids[1], [33, 0, 1]);
        assert_eq!(ids[4], [33, 1, 0]);
        assert_eq!(ids[15], [33, 3, 3]);
        assert_eq!(ids[16], [0, 0, 0]);
        assert_eq!(ids[31], [0, 0, 0]);
        assert_eq!(ids[32], [1, 0, 0]);
        assert_eq!(ids[63], [32, 0, 0]);
        let text_first = positions_text_first(grid);
        assert_eq!(text_first[0], [1, 0, 0]);
        assert_eq!(text_first[31], [32, 0, 0]);
        assert_eq!(text_first[32], [33, 0, 0]);
        assert_eq!(text_first[47], [33, 3, 3]);
        let index = patch_index(grid, 4);
        let mut seen = vec![false; index.len()];
        for src in index {
            assert!(src >= 0 && (src as usize) < seen.len());
            assert!(!seen[src as usize], "patch index repeats {src}");
            seen[src as usize] = true;
        }
        assert!(seen.iter().all(|flag| *flag));
    }

    #[test]
    fn sdcpp_kv_scale_and_w2_scale_cancel() {
        let q = [0.2f32, -0.1, 0.4, 0.3];
        let keys = [[0.5f32, 0.1, -0.2, 0.7], [0.1, 0.2, 0.0, -0.3]];
        let values = [[0.3f32, -0.4, 0.2, 0.1], [0.0, 0.2, 0.5, -0.1]];
        let plain = attend_one(&q, &keys, &values, 1.0);
        let guarded = attend_one(&q, &keys, &values, 1.0 / 128.0);
        for (got, exp) in guarded.iter().zip(plain) {
            assert!((got - exp).abs() < 1e-5, "{guarded:?} vs {plain:?}");
        }
        let x = [0.3f32, -0.2, 0.5, 0.1];
        let weight = [0.2f32, 0.1, -0.3, 0.4, 0.5, -0.1, 0.2, 0.3];
        let plain = mat(4, 2, &x, &weight, 1.0);
        let scaled = mat(4, 2, &x, &weight, 1.0 / 128.0);
        for (got, exp) in scaled.iter().zip(plain) {
            assert!((got - exp).abs() < 1e-5, "{scaled:?}");
        }
    }

    #[test]
    fn text_first_image_tokens_match_image_first() {
        let (weights, config, grid, latent, txt, sigma) = smoke_case();
        let image_first = velocity(&weights, config, grid, &latent, &txt, sigma, false);
        let text_first = velocity(&weights, config, grid, &latent, &txt, sigma, true);
        assert_close(&image_first, &text_first, 1e-3);
    }

    #[test]
    fn ggml_velocity_matches_the_host_reference() {
        let dir = scratch("z-image-parity");
        write_smoke_bundle(&dir).unwrap();
        let (weights, config, grid, latent, txt, sigma) = smoke_case_from(&dir);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let types = HashMap::new();
        let loaded = GgufWeights::open(&dir.join("transformer.gguf")).unwrap();
        let mut denoise =
            ZImageDenoise::build(&ctx, config, 64, 64, grid.cap_real, &types).unwrap();
        for (name, ne0, ne1) in weight_layout(config) {
            let raw = loaded.raw_exact(&name, &[ne0, ne1]).unwrap();
            denoise.load_bytes(&name, &raw).unwrap();
        }
        let got = denoise.forward_velocity(&latent, &txt, sigma).unwrap();
        let expect = velocity(&weights, config, grid, &latent, &txt, sigma, false);
        let worst_forward = worst_abs(&got, &expect);
        assert!(
            worst_forward < 5e-3,
            "forward worst {worst_forward} ggml {:?} host {:?}",
            &got[..8],
            &expect[..8]
        );
        let mut session =
            ZImageDenoise::build(&ctx, config, 64, 64, grid.cap_real, &types).unwrap();
        for (name, ne0, ne1) in weight_layout(config) {
            let raw = loaded.raw_exact(&name, &[ne0, ne1]).unwrap();
            session.load_bytes(&name, &raw).unwrap();
        }
        let sigmas = linear_sigmas(1, 64, 64, true);
        let stepped = session
            .denoise(latent.clone(), &txt, &sigmas, 0.0, |_, _| ())
            .unwrap();
        assert_eq!(session.graph_computes(), 1);
        assert_eq!(session.host_readbacks(), 0);
        assert_eq!(session.steps_run(), 1);
        let dt = sigmas[1] - sigmas[0];
        let euler: Vec<f32> = latent
            .iter()
            .zip(&expect)
            .map(|(x, v)| x + dt * v)
            .collect();
        let worst_step = worst_abs(&stepped, &euler);
        assert!(
            worst_step < 5e-3,
            "step worst {worst_step} dt {dt} stepped {:?} euler {:?} latent {:?}",
            &stepped[..8],
            &euler[..8],
            &latent[..8]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn attend_one(q: &[f32], keys: &[[f32; 4]], values: &[[f32; 4]], kv_scale: f32) -> [f32; 4] {
        let head = q.len() as f32;
        let softmax_scale = (1.0 / head.sqrt()) / kv_scale;
        let mut scores = Vec::new();
        let mut max = f32::NEG_INFINITY;
        for key in keys {
            let mut dot = 0f32;
            for d in 0..q.len() {
                dot += q[d] * (key[d] * kv_scale);
            }
            let score = dot * softmax_scale;
            max = max.max(score);
            scores.push(score);
        }
        let mut sum = 0f32;
        for score in &mut scores {
            *score = (*score - max).exp();
            sum += *score;
        }
        let mut out = [0f32; 4];
        for (score, value) in scores.iter().zip(values) {
            for d in 0..4 {
                out[d] += score / sum * (value[d] * kv_scale);
            }
        }
        for slot in &mut out {
            *slot /= kv_scale;
        }
        out
    }

    fn mat(in_dim: usize, out_dim: usize, x: &[f32], w: &[f32], scale: f32) -> Vec<f32> {
        let mut y = vec![0f32; out_dim];
        for o in 0..out_dim {
            let mut acc = 0f32;
            for k in 0..in_dim {
                acc += w[k + o * in_dim] * (x[k] * scale);
            }
            y[o] = acc / scale;
        }
        y
    }

    fn smoke_case() -> (
        HashMap<String, Vec<f32>>,
        ZImageConfig,
        super::super::geom::TokenGrid,
        Vec<f32>,
        Vec<f32>,
        f32,
    ) {
        let dir = scratch("z-image-case");
        write_smoke_bundle(&dir).unwrap();
        let case = smoke_case_from(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        case
    }

    fn smoke_case_from(
        dir: &std::path::Path,
    ) -> (
        HashMap<String, Vec<f32>>,
        ZImageConfig,
        super::super::geom::TokenGrid,
        Vec<f32>,
        Vec<f32>,
        f32,
    ) {
        let config = ZImageConfig::smoke();
        let grid = token_grid(64, 64, 1, 8).unwrap();
        let loaded = GgufWeights::open(&dir.join("transformer.gguf")).unwrap();
        let mut weights = HashMap::new();
        for (name, ne0, ne1) in weight_layout(config) {
            weights.insert(name.clone(), loaded.f32_exact(&name, &[ne0, ne1]).unwrap());
        }
        let text = GgufWeights::open(&dir.join("text_encoder.gguf")).unwrap();
        let txt = text
            .f32_exact("token_embed.weight", &[config.cap_dim, 1])
            .unwrap();
        let latent: Vec<f32> = (0..grid.latent_w * grid.latent_h * config.in_channels)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.02)
            .collect();
        let sigma = linear_sigmas(1, 64, 64, true)[0];
        (weights, config, grid, latent, txt, sigma)
    }

    fn worst_abs(got: &[f32], expect: &[f32]) -> f32 {
        assert_eq!(got.len(), expect.len());
        got.iter()
            .zip(expect)
            .map(|(g, e)| (g - e).abs())
            .fold(0f32, f32::max)
    }

    fn assert_close(got: &[f32], expect: &[f32], tol: f32) {
        assert_eq!(got.len(), expect.len());
        let mut worst = 0f32;
        for (g, e) in got.iter().zip(expect) {
            worst = worst.max((g - e).abs());
        }
        assert!(
            got.iter().all(|v| v.is_finite()) && worst < tol,
            "worst abs {worst} tol {tol} got {got:?}"
        );
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }
}
