//! LoRA factors on published graph parameter names.
//!
//! A factor is an in-graph residual `scale * B(Ax)` added to `Wx`. The base
//! weight stays quantized. Suffixes and wrapper prefixes are stripped; module
//! names are not rewritten onto another module tree. A name that is not a
//! graph parameter fails.

use super::{GgufWeights, WeightError};
use crate::runtime::{GraphBuilder, RuntimeError, Tensor};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct LoraFile {
    pub id: String,
    pub path: PathBuf,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct LoraFactor {
    pub target: String,
    pub in_dim: i64,
    pub rank: i64,
    pub out_dim: i64,
    pub scale: f32,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Down,
    Up,
    Alpha,
}

/// Map one published key onto a graph parameter. `lora_base_name` stays strict
/// and does not accept these suffixes.
pub(crate) fn resolve_lora_key(
    key: &str,
    names: &HashSet<String>,
) -> Result<(String, Part), WeightError> {
    let (module, part) = split_lora_suffix(key)?;
    let module = strip_wrappers(&module);
    if names.contains(&module) {
        return Ok((module, part));
    }
    let weighted = format!("{module}.weight");
    if names.contains(&weighted) {
        return Ok((weighted, part));
    }
    Err(WeightError::NotBuilt(format!(
        "lora {key} targets {module}, which is not a graph parameter"
    )))
}

pub fn load_lora_factors(
    files: &[LoraFile],
    layout: &[(String, i64, i64)],
) -> Result<Vec<LoraFactor>, WeightError> {
    let names: HashSet<String> = layout.iter().map(|(name, _, _)| name.clone()).collect();
    let shapes: HashMap<&str, (i64, i64)> = layout
        .iter()
        .map(|(name, ne0, ne1)| (name.as_str(), (*ne0, *ne1)))
        .collect();
    let mut factors = Vec::new();
    for file in files {
        if !file.weight.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "lora {} weight is not a finite number",
                file.id
            )));
        }
        factors.extend(load_file(file, &names, &shapes)?);
    }
    Ok(factors)
}

pub fn bind_lora_factors(
    b: &mut GraphBuilder,
    weights: &HashMap<String, Tensor>,
    factors: &[LoraFactor],
) -> Result<(), RuntimeError> {
    for (index, factor) in factors.iter().enumerate() {
        let weight = weights
            .get(&factor.target)
            .copied()
            .filter(|tensor| !tensor.is_null())
            .ok_or_else(|| {
                RuntimeError::Message(format!(
                    "lora targets {}, which is not a graph parameter",
                    factor.target
                ))
            })?;
        b.attach_lora(
            weight,
            index,
            factor.in_dim,
            factor.rank,
            factor.out_dim,
            factor.scale,
        )?;
    }
    Ok(())
}

fn load_file(
    file: &LoraFile,
    names: &HashSet<String>,
    shapes: &HashMap<&str, (i64, i64)>,
) -> Result<Vec<LoraFactor>, WeightError> {
    let weights = GgufWeights::open(&file.path)
        .map_err(|err| WeightError::NotBuilt(format!("lora {}: {err}", file.id)))?;
    let keys: Vec<String> = weights.names().map(str::to_string).collect();
    let mut groups: HashMap<String, Group> = HashMap::new();
    let mut order = Vec::new();
    for key in &keys {
        let (target, part) = resolve_lora_key(key, names)?;
        if !groups.contains_key(&target) {
            order.push(target.clone());
        }
        let group = groups.entry(target.clone()).or_default();
        match part {
            Part::Alpha => {
                if group.alpha.is_some() {
                    return Err(WeightError::NotBuilt(format!(
                        "lora {key} duplicates alpha for {target}"
                    )));
                }
                group.alpha = Some(read_alpha(&weights, key)?);
            }
            Part::Down | Part::Up => {
                let (matrix, ne0, ne1) = read_matrix(&weights, key)?;
                let slot = if part == Part::Down {
                    &mut group.down
                } else {
                    &mut group.up
                };
                if slot.is_some() {
                    let which = if part == Part::Down { "down" } else { "up" };
                    return Err(WeightError::NotBuilt(format!(
                        "lora {key} duplicates the {which} factor for {target}"
                    )));
                }
                *slot = Some(Matrix {
                    ne0,
                    ne1,
                    data: matrix,
                });
            }
        }
    }
    let mut factors = Vec::new();
    for target in order {
        let group = groups.remove(&target).unwrap_or_default();
        let Some((in_dim, out_dim)) = shapes.get(target.as_str()).copied() else {
            return Err(WeightError::NotBuilt(format!(
                "lora {target} targets {target}, which is not a graph parameter"
            )));
        };
        let down = group.down.ok_or_else(|| {
            WeightError::NotBuilt(format!("lora {target} is missing the down factor"))
        })?;
        let up = group.up.ok_or_else(|| {
            WeightError::NotBuilt(format!("lora {target} is missing the up factor"))
        })?;
        if down.ne1 < 1 || down.ne1 != up.ne0 || down.ne0 != in_dim || up.ne1 != out_dim {
            return Err(WeightError::Shape {
                name: target.clone(),
                expected: vec![in_dim, down.ne1, out_dim],
                got: vec![down.ne0, down.ne1, up.ne0, up.ne1],
            });
        }
        let rank = down.ne1;
        let alpha = group.alpha.unwrap_or(rank as f32);
        if !alpha.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "lora {target} alpha is not a finite number"
            )));
        }
        let scale = file.weight * alpha / (rank as f32);
        if !scale.is_finite() {
            return Err(WeightError::NotBuilt(format!(
                "lora {target} scale is not a finite number"
            )));
        }
        factors.push(LoraFactor {
            target,
            in_dim,
            rank,
            out_dim,
            scale,
            a: down.data,
            b: up.data,
        });
    }
    Ok(factors)
}

struct Matrix {
    ne0: i64,
    ne1: i64,
    data: Vec<f32>,
}

#[derive(Default)]
struct Group {
    down: Option<Matrix>,
    up: Option<Matrix>,
    alpha: Option<f32>,
}

fn read_matrix(weights: &GgufWeights, key: &str) -> Result<(Vec<f32>, i64, i64), WeightError> {
    let shape = weights.shape(key)?;
    if shape[2] != 1 || shape[3] != 1 {
        return Err(WeightError::NotBuilt(format!("lora {key} is not a matrix")));
    }
    let (ne0, ne1) = (shape[0], shape[1]);
    let data = weights.f32_widened(key, &[ne0, ne1])?;
    Ok((data, ne0, ne1))
}

fn read_alpha(weights: &GgufWeights, key: &str) -> Result<f32, WeightError> {
    let shape = weights.shape(key)?;
    let elems = shape
        .iter()
        .try_fold(1i64, |acc, dim| acc.checked_mul(*dim));
    if elems != Some(1) {
        return Err(WeightError::NotBuilt(format!(
            "lora {key} alpha is not a scalar"
        )));
    }
    let data = weights.f32_widened(key, &shape)?;
    let value = data.first().copied().unwrap_or(f32::NAN);
    if !value.is_finite() {
        return Err(WeightError::NotBuilt(format!(
            "lora {key} alpha is not a finite number"
        )));
    }
    Ok(value)
}

fn split_lora_suffix(key: &str) -> Result<(String, Part), WeightError> {
    const MARKS: &[(&str, Part)] = &[
        (".lora_down.weight", Part::Down),
        (".lora_up.weight", Part::Up),
        (".lora_A.weight", Part::Down),
        (".lora_B.weight", Part::Up),
        (".lora_a.weight", Part::Down),
        (".lora_b.weight", Part::Up),
        (".lora_down", Part::Down),
        (".lora_up", Part::Up),
        (".lora_A", Part::Down),
        (".lora_B", Part::Up),
        (".lora_a", Part::Down),
        (".lora_b", Part::Up),
        (".alpha", Part::Alpha),
    ];
    for (suffix, part) in MARKS {
        if let Some(module) = key.strip_suffix(suffix) {
            return Ok((module.to_string(), *part));
        }
    }
    Err(WeightError::NotBuilt(format!(
        "lora {key} is not a down, up, or alpha tensor"
    )))
}

fn strip_wrappers(module: &str) -> String {
    let mut name = module.trim().trim_matches('.').to_string();
    while name.contains("..") {
        name = name.replace("..", ".");
    }
    for prefix in [
        "pipe.transformer.",
        "z_image_transformer.",
        "diffusion_model.",
        "transformer.",
        "model.",
        "base_model.model.",
    ] {
        if let Some(rest) = name.strip_prefix(prefix) {
            name = rest.to_string();
        }
    }
    name = name.replace(".base_model.model.", ".");
    name = name.replace(".default.", ".");
    if let Some(rest) = name.strip_suffix(".default") {
        name = rest.trim_end_matches('.').to_string();
    }
    name.trim_matches('.').to_string()
}

#[cfg(test)]
mod tests {
    use super::{load_lora_factors, resolve_lora_key, LoraFile, Part};
    use crate::weights::{write_safetensors, SafeTensor};
    use std::collections::HashSet;

    fn names() -> HashSet<String> {
        ["double_blocks.0.img_attn.qkv.weight", "img_in.weight"]
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn fused_qkv_suffix_hits_the_graph_parameter() {
        let (target, part) = resolve_lora_key(
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
            &names(),
        )
        .unwrap();
        assert_eq!(target, "double_blocks.0.img_attn.qkv.weight");
        assert_eq!(part, Part::Down);
        let (target, part) = resolve_lora_key(
            "pipe.transformer.base_model.model.double_blocks.0.img_attn.qkv.lora_up.weight",
            &names(),
        )
        .unwrap();
        assert_eq!(target, "double_blocks.0.img_attn.qkv.weight");
        assert_eq!(part, Part::Up);
        let (target, _) = resolve_lora_key("base_model.model.img_in.lora_A", &names()).unwrap();
        assert_eq!(target, "img_in.weight");
    }

    #[test]
    fn diffusers_split_q_is_not_a_graph_parameter() {
        let err =
            resolve_lora_key("transformer_blocks.0.attn.to_q.lora_A.weight", &names()).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("targets transformer_blocks.0.attn.to_q")
                && message.contains("not a graph parameter"),
            "{message}"
        );
    }

    #[test]
    fn a_pair_needs_both_factors_and_scales_by_alpha_over_rank() {
        let dir = std::env::temp_dir().join(format!(
            "danmo-lora-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let down_only = dir.join("down.safetensors");
        let down = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
        write_safetensors(
            &down_only,
            &[SafeTensor {
                name: "diffusion_model.img_in.lora_A.weight",
                dtype: "F32",
                shape: &[1, 4],
                data: &down,
            }],
        )
        .unwrap();
        let layout = [("img_in.weight".into(), 4, 4)];
        let err = load_lora_factors(
            &[LoraFile {
                id: "portrait".into(),
                path: down_only,
                weight: 0.8,
            }],
            &layout,
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing the up factor"), "{err}");

        let both = dir.join("both.safetensors");
        let up = f32_bytes(&[0.5, 0.0, 0.0, 0.25]);
        let alpha = f32_bytes(&[2.0]);
        write_safetensors(
            &both,
            &[
                SafeTensor {
                    name: "diffusion_model.img_in.lora_A.weight",
                    dtype: "F32",
                    shape: &[1, 4],
                    data: &down,
                },
                SafeTensor {
                    name: "diffusion_model.img_in.lora_B.weight",
                    dtype: "F32",
                    shape: &[4, 1],
                    data: &up,
                },
                SafeTensor {
                    name: "diffusion_model.img_in.alpha",
                    dtype: "F32",
                    shape: &[],
                    data: &alpha,
                },
            ],
        )
        .unwrap();
        let factors = load_lora_factors(
            &[LoraFile {
                id: "portrait".into(),
                path: both,
                weight: 0.8,
            }],
            &layout,
        )
        .unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].target, "img_in.weight");
        assert_eq!(factors[0].in_dim, 4);
        assert_eq!(factors[0].rank, 1);
        assert_eq!(factors[0].out_dim, 4);
        assert!(
            (factors[0].scale - 1.6).abs() < 1e-6,
            "{}",
            factors[0].scale
        );
        let stray = dir.join("stray.safetensors");
        let bias = f32_bytes(&[1.0]);
        write_safetensors(
            &stray,
            &[SafeTensor {
                name: "diffusion_model.img_in.bias",
                dtype: "F32",
                shape: &[1],
                data: &bias,
            }],
        )
        .unwrap();
        let err = load_lora_factors(
            &[LoraFile {
                id: "portrait".into(),
                path: stray,
                weight: 1.0,
            }],
            &layout,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("not a down, up, or alpha"),
            "{err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}
