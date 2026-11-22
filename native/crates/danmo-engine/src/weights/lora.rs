//! LoRA factors on published graph parameter names.
//!
//! A factor is an in-graph residual `scale * B(Ax)` added to `Wx`. The base
//! weight stays quantized. Suffixes and wrapper prefixes are stripped. A
//! caller may remap the remaining module stem onto the graph name; this
//! loader does not invent that map. A name that is not a graph parameter fails.

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
pub(crate) enum Part {
    Down,
    Up,
    Alpha,
}

/// Where one LoRA module lands on the graph.
///
/// `Split` is a fused matrix (Flux2 QKV) that becomes several parameters.
/// `Skip` drops a module that is not a linear, such as an RMS norm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraRoute {
    One(String),
    Split(Vec<String>),
    Skip,
}

/// Map one published key onto a graph parameter. `lora_base_name` stays strict
/// and does not accept these suffixes.
pub(crate) fn resolve_lora_key(
    key: &str,
    names: &HashSet<String>,
) -> Result<(String, Part), WeightError> {
    resolve_lora_key_mapped(key, names, &|module| Ok(module.to_string()))
}

pub(crate) fn resolve_lora_key_mapped(
    key: &str,
    names: &HashSet<String>,
    remap: &dyn Fn(&str) -> Result<String, WeightError>,
) -> Result<(String, Part), WeightError> {
    let (module, part) = split_lora_suffix(key)?;
    let module = strip_wrappers(&module);
    let module = remap(&module)?;
    for candidate in [
        module.clone(),
        format!("{module}.weight"),
        format!("decoder.{module}"),
        format!("decoder.{module}.weight"),
    ] {
        if names.contains(&candidate) {
            return Ok((candidate, part));
        }
    }
    Err(WeightError::NotBuilt(format!(
        "lora {key} targets {module}, which is not a graph parameter"
    )))
}

pub fn load_lora_factors(
    files: &[LoraFile],
    layout: &[(String, i64, i64)],
) -> Result<Vec<LoraFactor>, WeightError> {
    load_lora_factors_with_default_alpha(files, layout, None)
}

pub fn load_lora_factors_mapped(
    files: &[LoraFile],
    layout: &[(String, i64, i64)],
    remap: &dyn Fn(&str) -> Result<String, WeightError>,
) -> Result<Vec<LoraFactor>, WeightError> {
    load_lora_factors_with_default_alpha_mapped(files, layout, None, remap)
}

pub fn load_lora_factors_with_default_alpha(
    files: &[LoraFile],
    layout: &[(String, i64, i64)],
    default_alpha: Option<f32>,
) -> Result<Vec<LoraFactor>, WeightError> {
    load_lora_factors_with_default_alpha_mapped(files, layout, default_alpha, &|module| {
        Ok(module.to_string())
    })
}

fn load_lora_factors_with_default_alpha_mapped(
    files: &[LoraFile],
    layout: &[(String, i64, i64)],
    default_alpha: Option<f32>,
    remap: &dyn Fn(&str) -> Result<String, WeightError>,
) -> Result<Vec<LoraFactor>, WeightError> {
    load_lora_factors_routed(files, layout, default_alpha, &|module| {
        Ok(LoraRoute::One(remap(module)?))
    })
}

pub(crate) fn load_lora_factors_routed(
    files: &[LoraFile],
    layout: &[(String, i64, i64)],
    default_alpha: Option<f32>,
    route: &dyn Fn(&str) -> Result<LoraRoute, WeightError>,
) -> Result<Vec<LoraFactor>, WeightError> {
    if default_alpha.is_some_and(|alpha| !alpha.is_finite()) {
        return Err(WeightError::NotBuilt(
            "lora default alpha is not a finite number".into(),
        ));
    }
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
        factors.extend(load_file(file, &names, &shapes, default_alpha, route)?);
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
    default_alpha: Option<f32>,
    route: &dyn Fn(&str) -> Result<LoraRoute, WeightError>,
) -> Result<Vec<LoraFactor>, WeightError> {
    let weights = GgufWeights::open(&file.path)
        .map_err(|err| WeightError::NotBuilt(format!("lora {}: {err}", file.id)))?;
    let keys: Vec<String> = weights.names().map(str::to_string).collect();
    let mut groups: HashMap<String, Group> = HashMap::new();
    let mut splits: HashMap<String, SplitGroup> = HashMap::new();
    let mut order = Vec::new();
    let mut split_order = Vec::new();
    let mut skipped = 0u32;
    for key in &keys {
        let (module, part) = match split_lora_suffix(key) {
            Ok(parts) => parts,
            Err(error) => {
                // Non LoRA-A/B tensors (e.g. LTX MSR slot embeddings) may be
                // skipped by the family route before this becomes a hard error.
                let stripped = strip_wrappers(key);
                if matches!(route(key)?, LoraRoute::Skip)
                    || matches!(route(&stripped)?, LoraRoute::Skip)
                {
                    skipped += 1;
                    continue;
                }
                return Err(error);
            }
        };
        let module = strip_wrappers(&module);
        match route(&module)? {
            LoraRoute::Skip => {
                skipped += 1;
                continue;
            }
            LoraRoute::Split(targets) => {
                if targets.len() < 2 {
                    return Err(WeightError::NotBuilt(format!(
                        "lora {key} split needs at least two targets"
                    )));
                }
                if !splits.contains_key(&module) {
                    split_order.push(module.clone());
                    splits.insert(
                        module.clone(),
                        SplitGroup {
                            targets,
                            down: None,
                            up: None,
                            alpha: None,
                        },
                    );
                }
                let group = splits.get_mut(&module).expect("split group");
                assign_part(
                    key,
                    &module,
                    part,
                    &weights,
                    &mut group.down,
                    &mut group.up,
                    &mut group.alpha,
                )?;
            }
            LoraRoute::One(mapped) => {
                let target = lookup_parameter(&mapped, names)?;
                if !groups.contains_key(&module) {
                    order.push(module.clone());
                }
                let group = groups.entry(module.clone()).or_default();
                if group.target.is_empty() {
                    group.target = target;
                } else if group.target != target {
                    return Err(WeightError::NotBuilt(format!(
                        "lora {key} resolves {module} to both {} and {target}",
                        group.target
                    )));
                }
                let label = group.target.clone();
                assign_part(
                    key,
                    &label,
                    part,
                    &weights,
                    &mut group.down,
                    &mut group.up,
                    &mut group.alpha,
                )?;
            }
        }
    }
    let mut factors = Vec::new();
    for module in order {
        let group = groups.remove(&module).unwrap_or_default();
        let label = if group.target.is_empty() {
            module
        } else {
            group.target.clone()
        };
        factors.push(finish_factor(
            label,
            group.down,
            group.up,
            group.alpha,
            shapes,
            file,
            default_alpha,
        )?);
    }
    for module in split_order {
        let group = splits.remove(&module).expect("split group");
        factors.extend(finish_split(
            &module,
            group,
            names,
            shapes,
            file,
            default_alpha,
        )?);
    }
    if factors.is_empty() && skipped > 0 {
        return Err(WeightError::NotBuilt(format!(
            "lora {} has no linear factors after dropping non-linear keys",
            file.id
        )));
    }
    Ok(factors)
}

fn assign_part(
    key: &str,
    label: &str,
    part: Part,
    weights: &GgufWeights,
    down: &mut Option<Matrix>,
    up: &mut Option<Matrix>,
    alpha: &mut Option<f32>,
) -> Result<(), WeightError> {
    match part {
        Part::Alpha => {
            if alpha.is_some() {
                return Err(WeightError::NotBuilt(format!(
                    "lora {key} duplicates alpha for {label}"
                )));
            }
            *alpha = Some(read_alpha(weights, key)?);
        }
        Part::Down | Part::Up => {
            let (matrix, ne0, ne1) = read_matrix(weights, key)?;
            let slot = if part == Part::Down { down } else { up };
            if slot.is_some() {
                let which = if part == Part::Down { "down" } else { "up" };
                return Err(WeightError::NotBuilt(format!(
                    "lora {key} duplicates the {which} factor for {label}"
                )));
            }
            *slot = Some(Matrix {
                ne0,
                ne1,
                data: matrix,
            });
        }
    }
    Ok(())
}

fn finish_factor(
    target: String,
    down: Option<Matrix>,
    up: Option<Matrix>,
    alpha: Option<f32>,
    shapes: &HashMap<&str, (i64, i64)>,
    file: &LoraFile,
    default_alpha: Option<f32>,
) -> Result<LoraFactor, WeightError> {
    let Some((in_dim, out_dim)) = shapes.get(target.as_str()).copied() else {
        return Err(WeightError::NotBuilt(format!(
            "lora {target} targets {target}, which is not a graph parameter"
        )));
    };
    let down = down.ok_or_else(|| {
        WeightError::NotBuilt(format!("lora {target} is missing the down factor"))
    })?;
    let up =
        up.ok_or_else(|| WeightError::NotBuilt(format!("lora {target} is missing the up factor")))?;
    if down.ne1 < 1 || down.ne1 != up.ne0 || down.ne0 != in_dim || up.ne1 != out_dim {
        return Err(WeightError::Shape {
            name: target,
            expected: vec![in_dim, down.ne1, out_dim],
            got: vec![down.ne0, down.ne1, up.ne0, up.ne1],
        });
    }
    let rank = down.ne1;
    let alpha = alpha
        .or(default_alpha)
        .or(adapter_alpha(&file.path)?)
        .unwrap_or(rank as f32);
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
    Ok(LoraFactor {
        target,
        in_dim,
        rank,
        out_dim,
        scale,
        a: down.data,
        b: up.data,
    })
}

fn finish_split(
    module: &str,
    group: SplitGroup,
    names: &HashSet<String>,
    shapes: &HashMap<&str, (i64, i64)>,
    file: &LoraFile,
    default_alpha: Option<f32>,
) -> Result<Vec<LoraFactor>, WeightError> {
    let parts = group.targets.len() as i64;
    let down = group.down.ok_or_else(|| {
        WeightError::NotBuilt(format!("lora {module} is missing the down factor"))
    })?;
    let up = group
        .up
        .ok_or_else(|| WeightError::NotBuilt(format!("lora {module} is missing the up factor")))?;
    if parts < 2 || up.ne1 % parts != 0 {
        return Err(WeightError::NotBuilt(format!(
            "lora {module} fuses {parts} projections but the up matrix out {} is not divisible by {parts}",
            up.ne1
        )));
    }
    // A shared down factor keeps its rank. A stacked down factor (rank divisible
    // by the projection count) is cut the same way as the up matrix. That is
    // the danmo-make `remap_flux2_lora_keys` rule.
    let split_down = down.ne1 % parts == 0;
    let mut factors = Vec::with_capacity(group.targets.len());
    for (index, mapped) in group.targets.iter().enumerate() {
        let target = lookup_parameter(mapped, names)?;
        let up_i = slice_ne1(&up, parts, index as i64);
        let down_i = if split_down {
            slice_ne1(&down, parts, index as i64)
        } else {
            down.clone()
        };
        factors.push(finish_factor(
            target,
            Some(down_i),
            Some(up_i),
            group.alpha,
            shapes,
            file,
            default_alpha,
        )?);
    }
    Ok(factors)
}

fn lookup_parameter(module: &str, names: &HashSet<String>) -> Result<String, WeightError> {
    for candidate in [
        module.to_string(),
        format!("{module}.weight"),
        format!("decoder.{module}"),
        format!("decoder.{module}.weight"),
    ] {
        if names.contains(&candidate) {
            return Ok(candidate);
        }
    }
    Err(WeightError::NotBuilt(format!(
        "lora targets {module}, which is not a graph parameter"
    )))
}

fn slice_ne1(matrix: &Matrix, parts: i64, index: i64) -> Matrix {
    let chunk = matrix.ne1 / parts;
    let start = index * chunk;
    let mut data = Vec::with_capacity((matrix.ne0 * chunk) as usize);
    for row in start..(start + chunk) {
        let base = (row * matrix.ne0) as usize;
        let end = base + matrix.ne0 as usize;
        data.extend_from_slice(&matrix.data[base..end]);
    }
    Matrix {
        ne0: matrix.ne0,
        ne1: chunk,
        data,
    }
}

#[derive(Clone)]
struct Matrix {
    ne0: i64,
    ne1: i64,
    data: Vec<f32>,
}

#[derive(Default)]
struct Group {
    target: String,
    down: Option<Matrix>,
    up: Option<Matrix>,
    alpha: Option<f32>,
}

struct SplitGroup {
    targets: Vec<String>,
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

fn adapter_alpha(path: &std::path::Path) -> Result<Option<f32>, WeightError> {
    let Some(dir) = path.parent() else {
        return Ok(None);
    };
    for name in ["adapter_config.json", "lora_config.json"] {
        let config = dir.join(name);
        if !config.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&config).map_err(|error| WeightError::Open {
            path: config.display().to_string(),
            detail: error.to_string(),
        })?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            WeightError::NotBuilt(format!(
                "lora config {} is not json: {error}",
                config.display()
            ))
        })?;
        let Some(alpha) = value.get("lora_alpha").or_else(|| value.get("alpha")) else {
            return Ok(None);
        };
        let number = alpha.as_f64().ok_or_else(|| {
            WeightError::NotBuilt(format!(
                "lora config {} alpha is not a number",
                config.display()
            ))
        })? as f32;
        if !number.is_finite() || number <= 0.0 {
            return Err(WeightError::NotBuilt(format!(
                "lora config {} alpha {number} is not a positive finite number",
                config.display()
            )));
        }
        return Ok(Some(number));
    }
    Ok(None)
}

fn split_lora_suffix(key: &str) -> Result<(String, Part), WeightError> {
    const MARKS: &[(&str, Part)] = &[
        (".lora_down.default.weight", Part::Down),
        (".lora_up.default.weight", Part::Up),
        (".lora_A.default.weight", Part::Down),
        (".lora_B.default.weight", Part::Up),
        (".lora_a.default.weight", Part::Down),
        (".lora_b.default.weight", Part::Up),
        (".lora_down.weight", Part::Down),
        (".lora_up.weight", Part::Up),
        (".lora_A.weight", Part::Down),
        (".lora_B.weight", Part::Up),
        (".lora_a.weight", Part::Down),
        (".lora_b.weight", Part::Up),
        (".lora_down.default", Part::Down),
        (".lora_up.default", Part::Up),
        (".lora_A.default", Part::Down),
        (".lora_B.default", Part::Up),
        (".lora_a.default", Part::Down),
        (".lora_b.default", Part::Up),
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
        "pipe.dit.",
        "z_image_transformer.",
        "diffusion_model.",
        "transformer.",
        "dit.",
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
    use super::{
        load_lora_factors, load_lora_factors_mapped, load_lora_factors_routed,
        load_lora_factors_with_default_alpha, resolve_lora_key, LoraFile, LoraRoute, Part,
    };
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
    fn ace_prefixes_and_default_alpha_match_main() {
        let dir = std::env::temp_dir().join(format!("danmo-ace-lora-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("ace.safetensors");
        let down = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
        write_safetensors(
            &path,
            &[
                SafeTensor {
                    name: "pipe.dit.layers.0.self_attn.q_proj.lora_A.weight",
                    dtype: "F32",
                    shape: &[1, 4],
                    data: &down,
                },
                SafeTensor {
                    name: "pipe.dit.layers.0.self_attn.q_proj.lora_B.weight",
                    dtype: "F32",
                    shape: &[4, 1],
                    data: &down,
                },
            ],
        )
        .unwrap();
        let layout = [("decoder.layers.0.self_attn.q_proj.weight".into(), 4, 4)];
        let factors = load_lora_factors_with_default_alpha(
            &[LoraFile {
                id: "ace".into(),
                path,
                weight: 0.5,
            }],
            &layout,
            Some(128.0),
        )
        .unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].target, layout[0].0);
        assert_eq!(factors[0].scale, 64.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peft_default_suffix_uses_adapter_config_alpha() {
        let dir = std::env::temp_dir().join(format!("danmo-peft-lora-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adapter_model.safetensors");
        let down = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
        write_safetensors(
            &path,
            &[
                SafeTensor {
                    name:
                        "base_model.model.decoder.layers.0.self_attn.q_proj.lora_A.default.weight",
                    dtype: "F32",
                    shape: &[1, 4],
                    data: &down,
                },
                SafeTensor {
                    name:
                        "base_model.model.decoder.layers.0.self_attn.q_proj.lora_B.default.weight",
                    dtype: "F32",
                    shape: &[4, 1],
                    data: &down,
                },
            ],
        )
        .unwrap();
        std::fs::write(dir.join("adapter_config.json"), r#"{"lora_alpha": 32}"#).unwrap();
        let layout = [("decoder.layers.0.self_attn.q_proj.weight".into(), 4, 4)];
        let factors = load_lora_factors(
            &[LoraFile {
                id: "peft".into(),
                path,
                weight: 0.5,
            }],
            &layout,
        )
        .unwrap();
        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].target, layout[0].0);
        assert_eq!(factors[0].scale, 16.0);
        let _ = std::fs::remove_dir_all(&dir);
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

    #[test]
    fn module_remap_lands_on_the_graph_name() {
        let dir = std::env::temp_dir().join(format!("danmo-remap-lora-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("adapter.safetensors");
        let down = f32_bytes(&[1.0, 0.0, 0.0, 1.0]);
        write_safetensors(
            &path,
            &[
                SafeTensor {
                    name: "diffusion_model.layers.0.attention.to_out.0.lora_A.weight",
                    dtype: "F32",
                    shape: &[1, 4],
                    data: &down,
                },
                SafeTensor {
                    name: "diffusion_model.layers.0.attention.to_out.0.lora_B.weight",
                    dtype: "F32",
                    shape: &[4, 1],
                    data: &down,
                },
            ],
        )
        .unwrap();
        let layout = [("layers.0.attention.to_out.weight".into(), 4, 4)];
        let factors = load_lora_factors_mapped(
            &[LoraFile {
                id: "zimage".into(),
                path,
                weight: 1.0,
            }],
            &layout,
            &|module| Ok(module.replace(".to_out.0", ".to_out")),
        )
        .unwrap();
        assert_eq!(factors[0].target, "layers.0.attention.to_out.weight");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fused_qkv_splits_the_up_matrix_and_shares_the_down_factor() {
        let dir = std::env::temp_dir().join(format!("danmo-split-lora-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fused.safetensors");
        let down = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let up: Vec<f32> = (0..12).map(|value| value as f32).collect();
        let up_bytes = f32_bytes(&up);
        write_safetensors(
            &path,
            &[
                SafeTensor {
                    name: "double_blocks.0.img_attn.qkv.lora_A.weight",
                    dtype: "F32",
                    shape: &[1, 4],
                    data: &down,
                },
                SafeTensor {
                    name: "double_blocks.0.img_attn.qkv.lora_B.weight",
                    dtype: "F32",
                    shape: &[12, 1],
                    data: &up_bytes,
                },
                SafeTensor {
                    name: "double_blocks.0.img_attn.norm.lora_A.weight",
                    dtype: "F32",
                    shape: &[1, 4],
                    data: &down,
                },
                SafeTensor {
                    name: "double_blocks.0.img_attn.norm.lora_B.weight",
                    dtype: "F32",
                    shape: &[4, 1],
                    data: &down,
                },
            ],
        )
        .unwrap();
        let layout = [
            ("to_q.weight".into(), 4, 4),
            ("to_k.weight".into(), 4, 4),
            ("to_v.weight".into(), 4, 4),
        ];
        let factors = load_lora_factors_routed(
            &[LoraFile {
                id: "fused".into(),
                path,
                weight: 1.0,
            }],
            &layout,
            None,
            &|module| {
                if module.ends_with("img_attn.qkv") {
                    Ok(LoraRoute::Split(vec![
                        "to_q".into(),
                        "to_k".into(),
                        "to_v".into(),
                    ]))
                } else if module.contains("norm") {
                    Ok(LoraRoute::Skip)
                } else {
                    Ok(LoraRoute::One(module.to_string()))
                }
            },
        )
        .unwrap();
        assert_eq!(factors.len(), 3);
        assert_eq!(factors[0].target, "to_q.weight");
        assert_eq!(factors[1].target, "to_k.weight");
        assert_eq!(factors[2].target, "to_v.weight");
        for factor in &factors {
            assert_eq!(factor.in_dim, 4);
            assert_eq!(factor.rank, 1);
            assert_eq!(factor.out_dim, 4);
            assert_eq!(factor.a, vec![1.0, 2.0, 3.0, 4.0]);
            assert!((factor.scale - 1.0).abs() < 1e-6);
        }
        assert_eq!(factors[0].b, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(factors[1].b, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(factors[2].b, vec![8.0, 9.0, 10.0, 11.0]);
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
