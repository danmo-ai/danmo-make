//! Qwen-Image create path. The DiT graph reads bare diffusers names.
//!
//! The text encoder is the Qwen2.5-VL trunk in `text.rs`, not the shared Qwen3
//! tower. The VAE is the 3D causal decoder in `vae.rs`, not AutoencoderKL.
//! Create tokenizes, encodes, steps a packed latent at guidance 1, unpacks to
//! 16 channels, scales, and decodes frame 0.

mod graph;
mod ids;
mod text;
mod tokenizer;
mod transformer;
mod vae;
mod weights;

use crate::common::ops::{
    empirical_mu_sigmas, flow_match_shift_sigmas, linear_sigmas, normal_latent,
};
use crate::runtime::GgmlContext;
use crate::weights::{assert_gguf_bundle, GgufWeights, LoraFile, WeightError};
use std::collections::HashMap;
use std::path::Path;

use weights::{check_layout, config_from_weights, weight_layout};

pub fn execute(
    ctx: &GgmlContext,
    bundle: &Path,
    prompt: &str,
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    seed: Option<i64>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
    negative: &str,
    loras: &[LoraFile],
    on_progress: impl FnMut(u32, u32),
) -> Result<crate::families::DitImageRun, WeightError> {
    if !negative.trim().is_empty() {
        return Err(WeightError::NotBuilt(
            "negative prompt is not built for this family".into(),
        ));
    }
    if !loras.is_empty() {
        return Err(WeightError::NotBuilt(
            "lora inference graph is not built".into(),
        ));
    }
    let files = assert_gguf_bundle(bundle)?;
    let loaded = GgufWeights::open_parts(&files.transformer)?;
    let config = config_from_weights(&loaded)?;
    let scale = config.vae_scale.max(1);
    if width % scale != 0 || height % scale != 0 {
        return Err(WeightError::NotBuilt(format!(
            "size {width}x{height} is not divisible by vae scale {scale}"
        )));
    }
    check_layout(&loaded, config)?;
    if loaded
        .names()
        .any(|name| name.contains("__index_timestep_zero__"))
    {
        return Err(WeightError::NotBuilt(
            "zero_cond_t edit path is not built".into(),
        ));
    }
    if config.in_channels != 64 || config.out_channels != 16 || config.patch != 2 {
        return Err(WeightError::NotBuilt(format!(
            "qwen packed create needs in_channels 64 out_channels 16 patch 2, got {} {} {}",
            config.in_channels, config.out_channels, config.patch
        )));
    }
    if !guidance.is_finite() || (guidance - 1.0).abs() >= f32::EPSILON {
        return Err(WeightError::NotBuilt(
            "cfg needs the negative embedding path".into(),
        ));
    }
    let sigmas = schedule(
        steps,
        width,
        height,
        scale,
        scheduler,
        scheduler_shift,
        empirical_mu,
    )?;
    let text_encoder = GgufWeights::open_parts(&files.text_encoder)?;
    let text_config = text::config_from_weights(&text_encoder)?;
    if text_config.hidden != config.joint_attention_dim {
        return Err(WeightError::NotBuilt(format!(
            "text hidden {} is not the DiT joint width {}",
            text_config.hidden, config.joint_attention_dim
        )));
    }
    let ids = tokenizer::prompt_ids(bundle, prompt, text_config.vocab)?;
    let txt = text::encode(ctx, &text_encoder, text_config, &ids)?;
    let seq_txt = (txt.len() as i64) / text_config.hidden;
    if seq_txt < 1 || txt.len() as i64 != text_config.hidden * seq_txt {
        return Err(WeightError::NotBuilt(format!(
            "text embedding length {} is not a multiple of hidden {}",
            txt.len(),
            text_config.hidden
        )));
    }
    let grid_w = (width / scale) as i64;
    let grid_h = (height / scale) as i64;
    let layout = weight_layout(config);
    let mut types = HashMap::new();
    for (name, _, _) in &layout {
        let ggml_type = loaded.ggml_type_of(name)?;
        types.insert(name.clone(), ggml_type);
    }
    let mut denoise =
        transformer::QwenImageForward::build(ctx, config, grid_w, grid_h, seq_txt, &types)?;
    for (name, ne0, ne1) in &layout {
        let raw = loaded.raw_exact(name, &[*ne0, *ne1])?;
        denoise.load_bytes(name, &raw)?;
    }
    let latent = match seed {
        Some(seed) => normal_latent(seed, denoise.latent_len()),
        None => vec![0.1f32; denoise.latent_len()],
    };
    let latents = denoise.denoise(&latent, &txt, &sigmas, guidance, on_progress)?;
    if let Some(detail) = crate::common::ops::non_finite("qwen-image latent", &latents) {
        return Err(WeightError::NotBuilt(detail));
    }
    let png = vae::decode_packed(ctx, &files.vae, &latents, grid_w, grid_h, width, height)?;
    Ok(crate::families::DitImageRun {
        latents,
        graph_computes: denoise.graph_computes(),
        preview_png: png,
    })
}

/// Published card: `requires_sigma_shift` is true and no `scheduler_shift` is
/// stored, so both Euler names take empirical mu on
/// `(width / vae_scale) * (height / vae_scale)`. A request that sets a positive
/// shift and leaves `use_empirical_mu` false takes the static shift instead.
/// `linear` uses the same resolution shift as `LinearScheduler`.
fn schedule(
    steps: u32,
    width: u32,
    height: u32,
    vae_scale: u32,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
) -> Result<Vec<f32>, WeightError> {
    match scheduler.unwrap_or("flow_match_euler_discrete") {
        "linear" => Ok(linear_sigmas(steps, width, height, true)),
        "flow_match_euler" | "flow_match_euler_discrete" => {
            if empirical_mu || scheduler_shift.is_none() {
                let scale = vae_scale.max(1);
                if width % scale != 0 || height % scale != 0 {
                    return Err(WeightError::NotBuilt(format!(
                        "size {width}x{height} is not divisible by vae scale {scale}"
                    )));
                }
                let seq = (width / scale) * (height / scale);
                return Ok(empirical_mu_sigmas(steps, seq));
            }
            let shift = scheduler_shift.unwrap_or(0.0);
            if !shift.is_finite() || shift <= 0.0 {
                return Err(WeightError::NotBuilt(format!(
                    "scheduler shift {shift} is not a positive finite value"
                )));
            }
            Ok(flow_match_shift_sigmas(steps, shift))
        }
        other => Err(WeightError::NotBuilt(format!(
            "scheduler {other} is not built"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::execute;
    use super::ids::timestep_features;
    use super::schedule;
    use super::transformer::QwenImageForward;
    use super::weights::{
        check_layout, config_from_weights, stored_f32, weight_layout, QwenImageConfig,
    };
    use crate::runtime::{BackendKind, GgmlContext};
    use crate::weights::{
        write_f32_gguf, write_f32_nd, write_f32_tensors, F32Nd, F32Tensor, GgufWeights, WeightError,
    };
    use std::collections::HashMap;
    use std::path::Path;

    fn layer_norm(token: &[f32], eps: f32) -> Vec<f32> {
        let n = token.len() as f32;
        let mean = token.iter().copied().sum::<f32>() / n;
        let var = token
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>()
            / n;
        let scale = 1.0 / (var + eps).sqrt();
        token.iter().map(|value| (value - mean) * scale).collect()
    }

    fn identity(n: i64) -> Vec<f32> {
        let n = n as usize;
        let mut data = vec![0f32; n * n];
        for i in 0..n {
            data[i + i * n] = 1.0;
        }
        data
    }

    fn fill(name: &str, ne0: i64, ne1: i64) -> Vec<f32> {
        let n = (ne0 as usize) * (ne1 as usize);
        if name == "img_in.weight" || name == "txt_in.weight" || name == "proj_out.weight" {
            if ne0 == ne1 {
                return identity(ne0);
            }
            let mut data = vec![0f32; n];
            for i in 0..ne0.min(ne1) as usize {
                data[i + i * ne0 as usize] = 1.0;
            }
            return data;
        }
        if stored_f32(name) && !name.ends_with(".bias") {
            return vec![1.0; n];
        }
        vec![0.0; n]
    }

    #[test]
    fn zero_gate_residual_is_layer_norm_of_the_latent() {
        let config = QwenImageConfig::smoke();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut forward = QwenImageForward::build(&ctx, config, 2, 1, 2, &HashMap::new()).unwrap();
        for (name, ne0, ne1) in weight_layout(config) {
            let data = fill(&name, ne0, ne1);
            forward.load_f32(&name, &data).unwrap();
        }
        let latent = vec![
            0.2, -0.4, 0.6, -0.1, 0.3, 0.8, -0.7, 0.5, 1.0, -1.5, 0.25, 0.0, 2.0, -0.2, 0.4, -0.8,
        ];
        let txt = vec![0.0; (config.joint_attention_dim * 2) as usize];
        let freq = vec![0.0; config.freq_dim as usize];
        let got = forward.forward(&latent, &txt, &freq).unwrap();
        assert!(got.iter().all(|value| value.is_finite()), "{got:?}");
        let mut expect = Vec::new();
        for token in latent.chunks(config.inner_dim as usize) {
            expect.extend(layer_norm(token, 1e-6));
        }
        assert_eq!(got.len(), expect.len());
        for (i, (left, right)) in got.iter().zip(expect.iter()).enumerate() {
            assert!(
                (left - right).abs() < 1e-4,
                "index {i} got {left} expect {right}"
            );
        }
    }

    #[test]
    fn published_depth_reserves_above_the_old_node_cap() {
        let mut config = QwenImageConfig::smoke();
        config.num_layers = 60;
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let forward = QwenImageForward::build(&ctx, config, 1, 1, 1, &HashMap::new())
            .expect("qwen-image depth reserves");
        let nodes = forward.node_count();
        assert!(
            nodes > 8192,
            "60-layer smoke used {nodes} nodes, which still fits in 8192"
        );
        assert!(
            nodes <= 20480,
            "60-layer smoke used {nodes} nodes and needs a larger graph"
        );
    }

    #[test]
    fn published_constants_match_diffusers_and_main() {
        let config = QwenImageConfig::published();
        assert_eq!(config.axes, [16, 56, 56]);
        assert_eq!(config.axes.iter().sum::<i64>(), config.attn_head_dim);
        assert_eq!(config.rope_theta, 10000.0);
        assert_eq!(config.patch, 2);
        assert_eq!(config.num_layers, 60);
        assert_eq!(config.num_heads, 24);
        assert_eq!(config.in_channels, 64);
        assert_eq!(config.out_channels, 16);
        assert_eq!(config.proj_out_dim(), 64);
        assert_eq!(config.joint_attention_dim, 3584);
        assert_eq!(config.inner_dim, 3072);
        assert_eq!(config.freq_dim, 256);
        assert_eq!(config.mlp_hidden(), 12288);
        let names: Vec<String> = weight_layout(QwenImageConfig::smoke())
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(names
            .iter()
            .any(|name| name.ends_with("img_mlp.net.0.proj.weight")));
        assert!(names
            .iter()
            .any(|name| name.ends_with("img_mlp.net.2.weight")));
        assert!(names
            .iter()
            .any(|name| name.ends_with("attn.to_out.0.weight")));
        assert!(names.iter().any(|name| name.ends_with("attn.to_q.bias")));
        assert!(!names.iter().any(|name| name.contains("mlp_in")));
        assert!(!names
            .iter()
            .any(|name| name.contains("model.diffusion_model")));
        assert!(!names.iter().any(|name| name.contains("img_mod_linear")));
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "danmo-qwen-{name}-{}-{}",
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

    fn write_bundle(dir: &Path, config: QwenImageConfig, extra: Option<(&str, i64, i64)>) {
        let layout = weight_layout(config);
        let owned: Vec<(String, i64, i64, Vec<f32>)> = layout
            .iter()
            .map(|(name, ne0, ne1)| {
                let n = (*ne0 as usize) * (*ne1 as usize);
                (name.clone(), *ne0, *ne1, vec![0f32; n])
            })
            .collect();
        let mut tensors: Vec<F32Tensor<'_>> = owned
            .iter()
            .map(|(name, ne0, ne1, data)| F32Tensor {
                name,
                ne0: *ne0,
                ne1: *ne1,
                data,
            })
            .collect();
        if let Some((name, ne0, ne1)) = extra {
            let marker = [0f32; 1];
            tensors.push(F32Tensor {
                name,
                ne0,
                ne1,
                data: &marker,
            });
            write_f32_tensors(&dir.join("transformer.gguf"), &tensors).unwrap();
        } else {
            write_f32_tensors(&dir.join("transformer.gguf"), &tensors).unwrap();
        }
        write_f32_gguf(
            &dir.join("text_encoder.gguf"),
            "token_embed.weight",
            4,
            1,
            &[0.0; 4],
        )
        .unwrap();
        write_f32_gguf(
            &dir.join("vae.gguf"),
            "decoder.conv.weight",
            4,
            3,
            &[0.0; 12],
        )
        .unwrap();
        std::fs::write(dir.join("tokenizer.json"), "{}\n").unwrap();
        std::fs::write(
            dir.join("bundle.manifest.json"),
            r#"{
  "format": "gguf",
  "quant": "q4_k",
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#,
        )
        .unwrap();
    }

    fn must_err(result: Result<crate::families::DitImageRun, WeightError>) -> WeightError {
        match result {
            Err(err) => err,
            Ok(_) => panic!("qwen-image execute returned a picture"),
        }
    }

    fn loader_config() -> QwenImageConfig {
        QwenImageConfig {
            in_channels: 8,
            out_channels: 2,
            inner_dim: 8,
            num_heads: 1,
            attn_head_dim: 8,
            axes: [2, 2, 4],
            num_layers: 1,
            joint_attention_dim: 8,
            freq_dim: 8,
            patch: 2,
            rope_theta: 10000.0,
            vae_scale: 16,
        }
    }

    #[test]
    fn loader_reads_the_file_shapes_and_refuses_the_towers() {
        let dir = scratch("bundle");
        let config = loader_config();
        write_bundle(&dir, config, None);
        let loaded = GgufWeights::open(&dir.join("transformer.gguf")).unwrap();
        let read = config_from_weights(&loaded).unwrap();
        assert_eq!(read.in_channels, 8);
        assert_eq!(read.out_channels, 2);
        assert_eq!(read.patch, 2);
        assert_eq!(read.num_layers, 1);
        assert_eq!(read.axes, [2, 2, 4]);
        check_layout(&loaded, read).unwrap();
        let packed = scratch("packed");
        write_bundle(&packed, create_smoke_config(), None);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let err = must_err(execute(
            &ctx,
            &packed,
            "a red cube",
            4,
            16,
            16,
            1.0,
            Some(1),
            None,
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        let message = err.to_string();
        assert!(message.contains("Qwen2.5-VL"), "{message}");
        assert!(!message.contains("step loop"), "{message}");
        let cfg = must_err(execute(
            &ctx,
            &packed,
            "a red cube",
            4,
            16,
            16,
            4.0,
            Some(1),
            None,
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        assert!(
            cfg.to_string().contains("cfg needs the negative embedding"),
            "{cfg}"
        );
    }

    #[test]
    fn zero_cond_t_and_linear_and_a_bad_edge_fail() {
        let dir = scratch("edit");
        write_bundle(
            &dir,
            create_smoke_config(),
            Some(("__index_timestep_zero__", 1, 1)),
        );
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let err = must_err(execute(
            &ctx,
            &dir,
            "a red cube",
            4,
            16,
            16,
            1.0,
            Some(1),
            Some("flow_match_euler_discrete"),
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        assert!(
            matches!(err, WeightError::NotBuilt(ref message) if message.contains("zero_cond_t")),
            "{err}"
        );
        let clean = scratch("linear");
        write_bundle(&clean, create_smoke_config(), None);
        let err = must_err(execute(
            &ctx,
            &clean,
            "a red cube",
            4,
            16,
            16,
            1.0,
            Some(1),
            Some("linear"),
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        assert!(err.to_string().contains("Qwen2.5-VL"), "{err}");
        let err = must_err(execute(
            &ctx,
            &clean,
            "a red cube",
            4,
            16,
            16,
            1.0,
            Some(1),
            Some("unipc"),
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        assert!(
            err.to_string().contains("scheduler unipc is not built"),
            "{err}"
        );
        let err = must_err(execute(
            &ctx,
            &dir,
            "a red cube",
            4,
            15,
            16,
            1.0,
            Some(1),
            None,
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        assert!(err.to_string().contains("vae scale"), "{err}");
    }

    #[test]
    fn published_schedule_is_empirical_mu_until_a_static_shift_is_set() {
        use crate::common::ops::{empirical_mu_sigmas, flow_match_shift_sigmas, linear_sigmas};
        let card = schedule(4, 1024, 1024, 16, None, None, false).unwrap();
        assert_eq!(card, empirical_mu_sigmas(4, 64 * 64));
        let named = schedule(4, 1024, 1024, 16, Some("flow_match_euler"), Some(6.0), true).unwrap();
        assert_eq!(named, card);
        let shifted = schedule(
            4,
            1024,
            1024,
            16,
            Some("flow_match_euler_discrete"),
            Some(6.0),
            false,
        )
        .unwrap();
        assert_eq!(shifted, flow_match_shift_sigmas(4, 6.0));
        let linear = schedule(4, 1024, 768, 16, Some("linear"), None, false).unwrap();
        assert_eq!(linear, linear_sigmas(4, 1024, 768, true));
        let bad = schedule(
            4,
            1024,
            1024,
            16,
            Some("flow_match_euler"),
            Some(0.0),
            false,
        );
        assert!(bad.unwrap_err().to_string().contains("scheduler shift"));
    }

    #[test]
    fn one_euler_step_matches_the_conditional_forward() {
        let config = QwenImageConfig::smoke();
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut forward = QwenImageForward::build(&ctx, config, 2, 1, 2, &HashMap::new()).unwrap();
        for (name, ne0, ne1) in weight_layout(config) {
            forward.load_f32(&name, &fill(&name, ne0, ne1)).unwrap();
        }
        let latent = vec![
            0.2, -0.4, 0.6, -0.1, 0.3, 0.8, -0.7, 0.5, 1.0, -1.5, 0.25, 0.0, 2.0, -0.2, 0.4, -0.8,
        ];
        let txt = vec![0.0; (config.joint_attention_dim * 2) as usize];
        let sigma = 0.5f32;
        let freq = timestep_features(sigma, config.freq_dim as usize);
        let velocity = forward.forward(&latent, &txt, &freq).unwrap();
        let mut expect = latent.clone();
        for (value, vel) in expect.iter_mut().zip(&velocity) {
            *value += (0.0 - sigma) * vel;
        }
        let mut steps = 0u32;
        let got = forward
            .denoise(&latent, &txt, &[sigma, 0.0], 1.0, |done, _| steps = done)
            .unwrap();
        assert_eq!(steps, 1);
        assert_eq!(got.len(), expect.len());
        for (i, (left, right)) in got.iter().zip(&expect).enumerate() {
            assert!(
                (left - right).abs() < 1e-4,
                "index {i} got {left} expect {right}"
            );
        }
        let cfg = forward
            .denoise(&latent, &txt, &[sigma, 0.0], 4.0, |_, _| {})
            .unwrap_err();
        assert!(cfg.to_string().contains("negative embedding"), "{cfg}");
    }

    fn create_smoke_config() -> QwenImageConfig {
        QwenImageConfig {
            in_channels: 64,
            out_channels: 16,
            inner_dim: 8,
            num_heads: 1,
            attn_head_dim: 8,
            axes: [2, 2, 4],
            num_layers: 1,
            joint_attention_dim: 8,
            freq_dim: 8,
            patch: 2,
            rope_theta: 10000.0,
            vae_scale: 16,
        }
    }

    fn write_create_smoke(dir: &Path) {
        let dit = create_smoke_config();
        let owned: Vec<(String, i64, i64, Vec<f32>)> = weight_layout(dit)
            .into_iter()
            .map(|(name, ne0, ne1)| {
                let data = fill(&name, ne0, ne1);
                (name, ne0, ne1, data)
            })
            .collect();
        let tensors: Vec<F32Tensor<'_>> = owned
            .iter()
            .map(|(name, ne0, ne1, data)| F32Tensor {
                name,
                ne0: *ne0,
                ne1: *ne1,
                data,
            })
            .collect();
        write_f32_tensors(&dir.join("transformer.gguf"), &tensors).unwrap();

        let text = super::text::Qwen25VlConfig::create_smoke();
        let text_owned: Vec<(String, i64, i64, Vec<f32>)> = super::text::text_layout(text)
            .into_iter()
            .map(|(name, ne0, ne1)| {
                let n = (ne0 as usize) * (ne1 as usize);
                let data = if name == "model.embed_tokens.weight" {
                    let mut rows = vec![0f32; n];
                    for token in 0..ne1 as usize {
                        for h in 0..ne0 as usize {
                            rows[h + ne0 as usize * token] = ((token + h) % 5) as f32 * 0.01;
                        }
                    }
                    rows
                } else if name.ends_with(".weight") && !name.contains("proj") {
                    vec![1.0; n]
                } else {
                    vec![0.0; n]
                };
                (name, ne0, ne1, data)
            })
            .collect();
        let text_tensors: Vec<F32Tensor<'_>> = text_owned
            .iter()
            .map(|(name, ne0, ne1, data)| F32Tensor {
                name,
                ne0: *ne0,
                ne1: *ne1,
                data,
            })
            .collect();
        write_f32_tensors(&dir.join("text_encoder.gguf"), &text_tensors).unwrap();

        let spec = super::vae::DecoderSpec::create_smoke();
        let vae_rows = super::vae::frame0_tensors(spec);
        let vae_owned: Vec<(String, [i64; 4], u8, Vec<f32>)> = vae_rows
            .into_iter()
            .map(|row| {
                let count = if row.rank == 1 {
                    row.ne[0] as usize
                } else {
                    row.ne.iter().product::<i64>() as usize
                };
                let data = if row.name.ends_with(".gamma") {
                    vec![1.0; count]
                } else {
                    vec![0.0; count]
                };
                (row.name, row.ne, row.rank, data)
            })
            .collect();
        let vae_tensors: Vec<F32Nd<'_>> = vae_owned
            .iter()
            .map(|(name, ne, rank, data)| {
                let shape = if *rank == 1 { [ne[0], 1, 1, 1] } else { *ne };
                F32Nd {
                    name,
                    ne: shape,
                    data,
                }
            })
            .collect();
        write_f32_nd(&dir.join("vae.gguf"), &vae_tensors).unwrap();

        let tok = dir.join("tokenizer");
        std::fs::create_dir_all(&tok).unwrap();
        let (vocab, _header, config) = {
            // Keep specials at 256/257 for the byte-level smoke vocab.
            let mut vocab = String::from("{");
            for byte in 0u16..256 {
                if byte > 0 {
                    vocab.push(',');
                }
                let token = crate::common::bpe::byte_token(byte as u8).to_string();
                vocab.push_str(&serde_json::to_string(&token).unwrap());
                vocab.push(':');
                vocab.push_str(&byte.to_string());
            }
            vocab.push_str(",\"<|im_start|>\":256,\"<|im_end|>\":257}");
            let config = r#"{
  "added_tokens_decoder": {
    "256": {"content": "<|im_start|>", "special": true},
    "257": {"content": "<|im_end|>", "special": true}
  }
}"#;
            (vocab, "#version: 0.2\n", config)
        };
        std::fs::write(tok.join("vocab.json"), vocab).unwrap();
        std::fs::write(tok.join("merges.txt"), "#version: 0.2\n").unwrap();
        std::fs::write(tok.join("tokenizer_config.json"), config).unwrap();
        std::fs::write(
            dir.join("bundle.manifest.json"),
            r#"{
  "format": "gguf",
  "quant": "q4_k",
  "files": {
    "transformer": "transformer.gguf",
    "text_encoder": "text_encoder.gguf",
    "vae": "vae.gguf"
  }
}
"#,
        )
        .unwrap();
    }

    #[test]
    fn smoke_create_encodes_steps_and_decodes_a_png() {
        let dir = scratch("create");
        write_create_smoke(&dir);
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut steps_seen = 0u32;
        let out = execute(
            &ctx,
            &dir,
            "hi",
            2,
            16,
            16,
            1.0,
            Some(1),
            Some("flow_match_euler_discrete"),
            None,
            false,
            "",
            &[],
            |done, total| {
                steps_seen = done;
                assert_eq!(total, 2);
            },
        )
        .expect("qwen smoke create");
        assert_eq!(steps_seen, 2);
        assert_eq!(out.latents.len(), 64);
        assert!(out.latents.iter().all(|value| value.is_finite()));
        assert!(out.graph_computes >= 2);
        assert!(out.preview_png.starts_with(b"\x89PNG"));
        assert!(out.preview_png.len() > 100);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gelu_matches_the_tanh_approximation() {
        let ctx = GgmlContext::new(BackendKind::Cpu).unwrap();
        let mut builder = ctx.builder(4 * 1024 * 1024).unwrap();
        let x = builder.f32_2d("x", 4, 1, true);
        let y = builder.gelu(x);
        let mut graph = builder.finish(y).unwrap();
        let input = [-1.0f32, 0.0, 0.5, 2.0];
        graph.set_f32("x", &input).unwrap();
        let got = graph.compute().unwrap();
        for (got, x) in got.iter().zip(input) {
            let x3 = x * x * x;
            let expect = 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x3)).tanh());
            // The CPU build may evaluate GELU through the f16 table (`GGML_GELU_FP16`).
            // That table is still the tanh approximation. `gelu_quick` at 2 is about 0.02 away.
            assert!((got - expect).abs() < 1e-3, "{got} vs {expect}");
        }
    }
}
