//! Qwen-Image create path. The DiT graph reads bare diffusers names.
//!
//! The text encoder is the Qwen2.5-VL trunk in `text.rs`, not the shared Qwen3
//! tower. The VAE is the 3D causal decoder in `vae.rs`, not AutoencoderKL.
//! The Euler loop steps a packed latent once text embeddings exist. Create
//! tokenizes `tokenizer/vocab.json`, then stops until the text-encoder file
//! has `model.embed_tokens.weight`. The packed latent is unpacked to 16
//! channels before the frame-0 decoder.

mod graph;
mod ids;
mod text;
mod tokenizer;
mod transformer;
mod vae;
mod weights;

use crate::common::ops::{empirical_mu_sigmas, flow_match_shift_sigmas, linear_sigmas};
use crate::runtime::GgmlContext;
use crate::weights::{assert_gguf_bundle, GgufWeights, LoraFile, WeightError};
use std::path::Path;

use weights::{check_layout, config_from_weights};

pub fn execute(
    _ctx: &GgmlContext,
    bundle: &Path,
    _prompt: &str,
    steps: u32,
    width: u32,
    height: u32,
    guidance: f32,
    _seed: Option<i64>,
    scheduler: Option<&str>,
    scheduler_shift: Option<f32>,
    empirical_mu: bool,
    negative: &str,
    loras: &[LoraFile],
    _on_progress: impl FnMut(u32, u32),
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
    // The constructor stays reachable from this entry. Requests return before it runs.
    link_dit();
    if loaded
        .names()
        .any(|name| name.contains("__index_timestep_zero__"))
    {
        return Err(WeightError::NotBuilt(
            "zero_cond_t edit path is not built".into(),
        ));
    }
    let _sigmas = schedule(
        steps,
        width,
        height,
        scale,
        scheduler,
        scheduler_shift,
        empirical_mu,
    )?;
    if !guidance.is_finite() || (guidance - 1.0).abs() >= f32::EPSILON {
        return Err(WeightError::NotBuilt(
            "cfg needs the negative embedding and the Qwen2.5-VL tokenizer is not built".into(),
        ));
    }
    let _ids = tokenizer::prompt_ids(bundle, _prompt, text::Qwen25VlConfig::published().vocab)?;
    let text_encoder = GgufWeights::open_parts(&files.text_encoder)?;
    if !text_encoder
        .names()
        .any(|name| name == "model.embed_tokens.weight")
    {
        return Err(WeightError::NotBuilt(
            "Qwen2.5-VL text encoder is missing model.embed_tokens.weight".into(),
        ));
    }
    Err(WeightError::NotBuilt(
        "Qwen2.5-VL text encoder weights are not loaded into the tower".into(),
    ))
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

/// Referenced from `execute` so the graphs stay in the non-test crate.
/// The branch does not run: create still stops on the tokenizer.
fn link_dit() {
    if false {
        let Ok(ctx) = GgmlContext::new(crate::runtime::BackendKind::Cpu) else {
            return;
        };
        let _ = weights::QwenImageConfig::published();
        let _ = ids::timestep_features(0.0, 2);
        let _ = ids::spatial_axis(1);
        if let Ok(ids) = ids::position_ids(1, 1, 1) {
            let _ = ids.attention_order();
        }
        let Ok(mut forward) = transformer::QwenImageForward::build(
            &ctx,
            weights::QwenImageConfig::smoke(),
            1,
            1,
            1,
            &std::collections::HashMap::new(),
        ) else {
            return;
        };
        let _ = forward.load_f32("img_in.weight", &[]);
        let _ = forward.node_count();
        let _ = forward.forward(&[], &[], &[]);
        let _ = forward.denoise(&[], &[], &[1.0, 0.0], 1.0, |_, _| {});
        let _ = text::PROMPT_TEMPLATE.len() + text::MAX_LENGTH;
        let _ = text::ggml_mrope_axis(0, [16, 24, 24]);
        let _ = text::hf_mrope_cos(
            [0, 0, 0],
            128,
            text::Qwen25VlConfig::published().mrope,
            1_000_000.0,
        );
        let _ = text::trim_system_prefix(&[0.0], 1, 0);
        let _ = text::text_layout(text::Qwen25VlConfig::published());
        if let Ok(mut tower) = text::TextForward::build(&ctx, text::Qwen25VlConfig::smoke(), 1) {
            let _ = tower.load_f32("model.norm.weight", &[]);
            let _ = tower.compute(&[], &[]);
        }
        let _ = tokenizer::prompt_ids(Path::new("."), "", 1);
        let _ = vae::unpack_latents(&[], 0, 0);
        let _ = vae::scale_latents(&[], 0, 0, 0);
        let _ = vae::required_names(vae::DecoderSpec::published());
        let _ = vae::decoder_dims(96, [1, 2, 4, 4]);
        if let Ok(mut decode) = vae::VaeDecode::build(&ctx, vae::DecoderSpec::smoke(), 1, 1) {
            let _ = decode.load_f32("decoder.conv_out.bias", &[]);
            let _ = decode.decode(&[]);
        }
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
    use crate::weights::{write_f32_gguf, write_f32_tensors, F32Tensor, GgufWeights, WeightError};
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
            return identity(ne0);
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
            None,
            None,
            false,
            "",
            &[],
            |_, _| {},
        ));
        let message = err.to_string();
        assert!(
            message.contains("Qwen2.5-VL tokenizer is missing"),
            "{message}"
        );
        assert!(!message.contains("step loop"), "{message}");
        let cfg = must_err(execute(
            &ctx,
            &dir,
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
            loader_config(),
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
        write_bundle(&clean, loader_config(), None);
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
        assert!(
            err.to_string().contains("Qwen2.5-VL tokenizer is missing"),
            "{err}"
        );
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
        assert!(cfg.to_string().contains("tokenizer"), "{cfg}");
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
