//! MLX safetensors bundles. Affine-quantized linears (`{base}.weight` packed
//! uint32 + `{base}.scales` + `{base}.biases`) load as one packed
//! `LinearWeight` under `{base}.weight` and are never dequantized here.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use danmo_mlx::{
    AffineQuantization, Array, Dtype, Error, LinearWeight, QuantizedArray, Result, SafeTensors,
    Stream,
};

/// Storage dtype for dense floating tensors. Packed weights and their
/// scales/biases always keep the checkpoint layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    /// Production default: keep the checkpoint dtype (BF16/FP16).
    Native,
    /// Widen dense tensors to F32. Only for reproducing main in parity tests;
    /// production loads cannot select it.
    #[cfg(test)]
    F32,
}

#[derive(Clone, Default)]
pub struct MlxWeights {
    tensors: BTreeMap<String, Entry>,
    quantization: Option<AffineQuantization>,
}

/// A dense entry keeps its array and a linear view sharing the same buffer, so
/// either accessor works without copying. Packed entries only have the view.
#[derive(Clone)]
enum Entry {
    Dense { tensor: Array, linear: LinearWeight },
    Packed(LinearWeight),
}

impl Entry {
    fn as_linear(&self) -> &LinearWeight {
        match self {
            Self::Dense { linear, .. } | Self::Packed(linear) => linear,
        }
    }

    fn dense(tensor: Array) -> Self {
        let linear = LinearWeight::dense(tensor.clone());
        Self::Dense { tensor, linear }
    }

    fn nbytes(&self) -> usize {
        match self {
            Self::Dense { tensor, .. } => tensor.nbytes(),
            Self::Packed(linear) => linear.nbytes(),
        }
    }
}

impl MlxWeights {
    pub fn from_dense(tensors: BTreeMap<String, Array>) -> Self {
        Self {
            tensors: tensors
                .into_iter()
                .map(|(name, tensor)| (name, Entry::dense(tensor)))
                .collect(),
            quantization: None,
        }
    }

    /// A tensor read as-is (norm, embedding, bias, conv kernel).
    pub fn tensor(&self, name: &str) -> Result<&Array> {
        match self.tensors.get(name) {
            Some(Entry::Dense { tensor, .. }) => Ok(tensor),
            Some(Entry::Packed(_)) => Err(Error::new(
                "MLX weights",
                format!("{name} is affine-quantized, but it is not used as a linear weight"),
            )),
            None => Err(Error::new("MLX weights", format!("missing tensor {name}"))),
        }
    }

    pub fn embedding(&self, name: &str, stream: &Stream) -> Result<Array> {
        match self.tensors.get(name) {
            Some(Entry::Dense { tensor, .. }) => Ok(tensor.clone()),
            Some(Entry::Packed(linear)) => linear.dequantize_embedding(stream),
            None => Err(Error::new("MLX weights", format!("missing tensor {name}"))),
        }
    }

    /// A `[out, in]` linear weight, dense or packed.
    pub fn linear(&self, name: &str) -> Result<&LinearWeight> {
        match self.tensors.get(name) {
            Some(Entry::Dense { linear, .. } | Entry::Packed(linear)) => Ok(linear),
            None => Err(Error::new("MLX weights", format!("missing tensor {name}"))),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Replace/insert one linear entry (test construction of mixed
    /// dense/packed weight sets).
    #[cfg(test)]
    pub fn with_linear(mut self, name: String, linear: LinearWeight) -> Self {
        self.tensors.insert(name, Entry::Packed(linear));
        self
    }

    #[cfg(test)]
    pub fn with_tensor(mut self, name: String, tensor: Array) -> Self {
        self.tensors.insert(name, Entry::dense(tensor));
        self
    }

    /// Dequantize one packed embedding into a resident dense table.
    ///
    /// `nn.Embedding` has no packed gather. Doing the full dequant inside
    /// every encode rebuilds a vocab-sized tensor; the warm gate keeps that
    /// table and gathers rows. Already-dense embeddings are unchanged.
    pub fn materialize_embedding(self, name: &str, stream: &Stream) -> Result<Self> {
        let packed = self.packed_names().any(|packed| packed == name);
        if !packed {
            return Ok(self);
        }
        let dense = self.linear(name)?.dequantize_embedding(stream)?;
        dense.eval()?;
        Ok(self.with_tensor_production(name.to_string(), dense))
    }

    /// Replace a dense tensor in place (layout normalization such as the
    /// mflux NHWC conv kernels). Fails loud on packed or missing entries.
    pub fn with_tensor_production(mut self, name: String, tensor: Array) -> Self {
        self.tensors.insert(name, Entry::dense(tensor));
        self
    }

    /// Names of the linears that stay packed in memory.
    pub fn packed_names(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .iter()
            .filter(|(_, entry)| matches!(entry, Entry::Packed(_)))
            .map(|(name, _)| name.as_str())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn quantization(&self) -> Option<AffineQuantization> {
        self.quantization
    }

    /// Resident bytes of every tensor, packed linears counted packed.
    pub fn nbytes(&self) -> usize {
        self.tensors.values().map(Entry::nbytes).sum()
    }

    /// Rename by a complete `source -> target` map (a family remap).
    pub fn rename(mut self, names: &BTreeMap<String, String>) -> Result<Self> {
        let mut output = BTreeMap::new();
        for (source, target) in names {
            let tensor = self
                .tensors
                .remove(source)
                .ok_or_else(|| Error::new("MLX weights rename", format!("missing {source}")))?;
            if output.insert(target.clone(), tensor).is_some() {
                return Err(Error::new(
                    "MLX weights rename",
                    format!("duplicate renamed tensor {target}"),
                ));
            }
        }
        if let Some(name) = self.tensors.keys().next() {
            return Err(Error::new(
                "MLX weights rename",
                format!("no target for tensor {name}"),
            ));
        }
        Ok(Self {
            tensors: output,
            quantization: self.quantization,
        })
    }

    #[cfg(test)]
    pub fn into_dequantized(self, stream: &Stream) -> Result<BTreeMap<String, Array>> {
        self.tensors
            .into_iter()
            .map(|(name, entry)| {
                let tensor = match entry {
                    Entry::Dense { tensor, .. } => tensor,
                    Entry::Packed(linear) => linear.dequantize_embedding(stream)?,
                };
                Ok((name, tensor))
            })
            .collect()
    }

    /// Attach LoRA residuals onto the linears those files name.
    ///
    /// The base matrices stay shared with the caller. The returned map is a
    /// new slot whose `apply` adds `scale * B(Ax)`, the same residual the
    /// ggml graph attaches. An empty file list returns `self`.
    pub fn attach_lora(
        self,
        files: &[super::LoraFile],
        stream: &Stream,
    ) -> std::result::Result<Self, super::WeightError> {
        if files.is_empty() {
            return Ok(self);
        }
        let layout = self.linear_layout();
        let factors = super::load_lora_factors(files, &layout)?;
        let mut grouped: BTreeMap<String, Vec<danmo_mlx::LoraResidual>> = BTreeMap::new();
        for factor in factors {
            let dtype = self
                .linear(&factor.target)
                .map_err(|error| super::WeightError::NotBuilt(error.to_string()))?
                .compute_dtype();
            let a = lora_matrix(&factor.a, factor.rank, factor.in_dim, dtype, stream)?;
            let b = lora_matrix(&factor.b, factor.out_dim, factor.rank, dtype, stream)?;
            grouped
                .entry(factor.target)
                .or_default()
                .push(danmo_mlx::LoraResidual {
                    a,
                    b,
                    scale: factor.scale,
                });
        }
        let mut weights = self;
        for (name, residuals) in grouped {
            let linear = weights
                .linear(&name)
                .map_err(|error| super::WeightError::NotBuilt(error.to_string()))?
                .with_loras(residuals);
            weights = weights
                .replace_linear(name, linear)
                .map_err(|error| super::WeightError::NotBuilt(error.to_string()))?;
        }
        Ok(weights)
    }

    fn linear_layout(&self) -> Vec<(String, i64, i64)> {
        self.tensors
            .iter()
            .filter_map(|(name, entry)| {
                let (out, inn) = entry.as_linear().out_in().ok().flatten()?;
                if out < 1 || inn < 1 {
                    return None;
                }
                Some((name.clone(), inn, out))
            })
            .collect()
    }

    fn replace_linear(mut self, name: String, linear: LinearWeight) -> Result<Self> {
        let entry = match self.tensors.remove(&name) {
            Some(Entry::Dense { tensor, .. }) => Entry::Dense { tensor, linear },
            Some(Entry::Packed(_)) => Entry::Packed(linear),
            None => return Err(Error::new("MLX LoRA", format!("missing linear {name}"))),
        };
        self.tensors.insert(name, entry);
        Ok(self)
    }

    /// Dense map for consumers that have no linear layers to quantize.
    pub fn into_dense(self) -> Result<BTreeMap<String, Array>> {
        self.tensors
            .into_iter()
            .map(|(name, tensor)| match tensor {
                Entry::Dense { tensor, .. } => Ok((name, tensor)),
                Entry::Packed(_) => Err(Error::new(
                    "MLX weights",
                    format!(
                        "{name} is affine-quantized; this component only accepts dense weights"
                    ),
                )),
            })
            .collect()
    }
}

fn lora_matrix(
    data: &[f32],
    rows: i64,
    cols: i64,
    dtype: Dtype,
    stream: &Stream,
) -> std::result::Result<Array, super::WeightError> {
    if rows < 1
        || cols < 1
        || rows > i32::MAX as i64
        || cols > i32::MAX as i64
        || data.len() != (rows * cols) as usize
    {
        return Err(super::WeightError::NotBuilt(format!(
            "lora matrix {rows}x{cols} has {} values",
            data.len()
        )));
    }
    let array = Array::from_f32_slice(data, &[rows as i32, cols as i32])
        .map_err(|error| super::WeightError::NotBuilt(error.to_string()))?;
    let array = array
        .astype(dtype, stream)
        .map_err(|error| super::WeightError::NotBuilt(error.to_string()))?;
    array
        .eval()
        .map_err(|error| super::WeightError::NotBuilt(error.to_string()))?;
    Ok(array)
}

/// Load every `*.safetensors` shard in `directory`.
///
/// `expected` is the registry's declared quantization. When it is set, the
/// bundle must be affine-quantized with exactly that bits/group size; when it
/// is `None`, packed layers are still accepted but need shard metadata
/// `quantization_level` whenever their shapes fit both 4 and 8 bits.
pub fn load_directory(
    directory: &Path,
    dtype: WeightDtype,
    expected: Option<AffineQuantization>,
) -> Result<MlxWeights> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(directory).map_err(|error| {
        Error::new(
            "MLX safetensors directory",
            format!("{}: {error}", directory.display()),
        )
    })? {
        let path = entry
            .map_err(|error| {
                Error::new(
                    "MLX safetensors directory",
                    format!("{}: {error}", directory.display()),
                )
            })?
            .path();
        if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(Error::new(
            "MLX safetensors directory",
            format!("no safetensors shards in {}", directory.display()),
        ));
    }

    let io = Stream::cpu()?;
    let mut raw = BTreeMap::new();
    let mut metadata_bits = None;
    for path in &paths {
        let tensors = SafeTensors::load(path, &io)?;
        if let Some(bits) = shard_bits(&tensors, path)? {
            if metadata_bits.is_some_and(|previous| previous != bits) {
                return Err(Error::new(
                    "MLX safetensors directory",
                    format!(
                        "{} declares quantization_level={bits}, another shard declares {}",
                        path.display(),
                        metadata_bits.unwrap()
                    ),
                ));
            }
            metadata_bits = Some(bits);
        }
        for name in tensors.names()? {
            let tensor = tensors
                .get(&name)?
                .ok_or_else(|| Error::new("MLX safetensors", format!("missing {name}")))?;
            if raw.insert(name.clone(), tensor).is_some() {
                return Err(Error::new(
                    "MLX safetensors directory",
                    format!("duplicate tensor {name}"),
                ));
            }
        }
    }
    if let (Some(expected), Some(bits)) = (expected, metadata_bits) {
        if expected.bits != bits {
            return Err(Error::new(
                "MLX affine quantization",
                format!(
                    "registry declares {}-bit weights, bundle metadata quantization_level={bits}",
                    expected.bits
                ),
            ));
        }
    }
    group(raw, dtype, expected, metadata_bits, &io)
}

fn shard_bits(tensors: &SafeTensors, path: &Path) -> Result<Option<i32>> {
    let Some(value) = tensors.metadata("quantization_level")? else {
        return Ok(None);
    };
    match value.trim() {
        "" | "None" | "null" => Ok(None),
        "4" => Ok(Some(4)),
        "8" => Ok(Some(8)),
        other => Err(Error::new(
            "MLX safetensors metadata",
            format!(
                "{} has quantization_level={other:?}; MLX affine bundles use 4 or 8",
                path.display()
            ),
        )),
    }
}

#[cfg_attr(not(test), allow(unused_variables))]
fn group(
    mut raw: BTreeMap<String, Array>,
    dtype: WeightDtype,
    expected: Option<AffineQuantization>,
    metadata_bits: Option<i32>,
    stream: &Stream,
) -> Result<MlxWeights> {
    let bases: BTreeSet<String> = raw
        .keys()
        .filter_map(|name| name.strip_suffix(".scales"))
        .map(str::to_string)
        .collect();
    if expected.is_some() && bases.is_empty() {
        return Err(Error::new(
            "MLX affine quantization",
            "registry declares quantized weights, but the bundle has no *.scales tensors",
        ));
    }

    let mut tensors = BTreeMap::new();
    let mut quantization = None;
    for base in &bases {
        let weight_name = format!("{base}.weight");
        let take = |raw: &mut BTreeMap<String, Array>, name: &str| {
            raw.remove(name).ok_or_else(|| {
                Error::new(
                    "MLX affine quantization",
                    format!("{base}.scales has no sibling {name}"),
                )
            })
        };
        let packed = QuantizedArray {
            weight: take(&mut raw, &weight_name)?,
            scales: take(&mut raw, &format!("{base}.scales"))?,
            biases: take(&mut raw, &format!("{base}.biases"))?,
        };
        let layer = infer(&weight_name, &packed, expected, metadata_bits)?;
        if quantization.is_some_and(|previous| previous != layer) {
            return Err(Error::new(
                "MLX affine quantization",
                format!(
                    "{weight_name} is {layer:?}, other layers are {:?}; mixed quantization is not supported",
                    quantization.unwrap()
                ),
            ));
        }
        quantization = Some(layer);
        tensors.insert(
            weight_name,
            Entry::Packed(LinearWeight::affine(packed, layer)?),
        );
    }

    for (name, tensor) in raw {
        if tensor.dtype() == Dtype::UInt32 {
            return Err(Error::new(
                "MLX affine quantization",
                format!("{name} is packed uint32 without *.scales and *.biases"),
            ));
        }
        let tensor = match dtype {
            WeightDtype::Native => tensor,
            #[cfg(test)]
            WeightDtype::F32 => {
                if tensor.dtype().is_floating() && tensor.dtype() != Dtype::Float32 {
                    tensor.astype(Dtype::Float32, stream)?
                } else {
                    tensor
                }
            }
        };
        tensors.insert(name, Entry::dense(tensor));
    }
    Ok(MlxWeights {
        tensors,
        quantization,
    })
}

/// Recover bits and group size from shapes. A packed row of `p` uint32 words
/// holds `p * 32 / bits` inputs split into `scales.shape[1]` groups; when both
/// 4 and 8 bits fit, metadata or the registry must choose.
fn infer(
    name: &str,
    packed: &QuantizedArray,
    expected: Option<AffineQuantization>,
    metadata_bits: Option<i32>,
) -> Result<AffineQuantization> {
    let weight = packed.weight.shape();
    let scales = packed.scales.shape();
    if weight.len() != 2 || scales.len() != 2 || scales[1] <= 0 {
        return Err(Error::new(
            "MLX affine quantization",
            format!("{name}: weight {weight:?} and scales {scales:?} are not a 2-D affine layer"),
        ));
    }
    let candidates: Vec<AffineQuantization> = [8, 4]
        .into_iter()
        .filter_map(|bits| {
            let inputs = weight[1] as i64 * 32 / bits as i64;
            (inputs % scales[1] as i64 == 0)
                .then(|| AffineQuantization::new(bits, (inputs / scales[1] as i64) as i32).ok())
                .flatten()
        })
        .collect();
    let bits = expected.map(|value| value.bits).or(metadata_bits);
    let chosen: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|candidate| bits.map_or(true, |bits| candidate.bits == bits))
        .collect();
    let layer = match chosen.as_slice() {
        [layer] => *layer,
        [] => {
            return Err(Error::new(
                "MLX affine quantization",
                format!(
                    "{name}: weight {weight:?} with scales {scales:?} does not fit {} affine packing",
                    bits.map_or("4- or 8-bit".to_string(), |bits| format!("{bits}-bit"))
                ),
            ))
        }
        _ => {
            return Err(Error::new(
                "MLX affine quantization",
                format!(
                    "{name}: weight {weight:?} with scales {scales:?} fits both 4 and 8 bits; set shard metadata quantization_level or the registry quantization"
                ),
            ))
        }
    };
    if let Some(expected) = expected {
        if expected != layer {
            return Err(Error::new(
                "MLX affine quantization",
                format!("{name}: bundle is {layer:?}, registry declares {expected:?}"),
            ));
        }
    }
    Ok(layer)
}

#[cfg(test)]
mod tests {
    use super::{load_directory, WeightDtype};
    use danmo_mlx::{save_safetensors, AffineQuantization, Array, Dtype, LinearWeight, Stream};
    use std::path::PathBuf;

    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            // Parallel tests on macOS can observe identical coarse clock
            // readings, so the directory name needs a process-wide counter:
            // a shared scratch dir made shards from different tests collide
            // ("duplicate tensor") and one test's Drop deleted another's files.
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "danmo-mlx-weights-{}-{}-{unique}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn dense(out: i32, inputs: i32, stream: &Stream) -> Array {
        let values: Vec<f32> = (0..out * inputs)
            .map(|index| (index % 31) as f32 * 0.011 - 0.16)
            .collect();
        Array::from_f32_slice(&values, &[out, inputs])
            .unwrap()
            .astype(Dtype::BFloat16, stream)
            .unwrap()
    }

    fn write_bundle(root: &std::path::Path, bits: i32, metadata: Option<&str>) -> Array {
        let stream = Stream::cpu().unwrap();
        let linear = dense(32, 256, &stream);
        let packed = linear.quantize_affine(64, bits, &stream).unwrap();
        let norm = Array::from_f32_slice(&[1.0; 256], &[256])
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
        let bias = Array::from_f32_slice(&[0.5; 32], &[32])
            .unwrap()
            .astype(Dtype::BFloat16, &stream)
            .unwrap();
        let metadata: Vec<(&str, &str)> = metadata
            .map(|bits| vec![("quantization_level", bits)])
            .unwrap_or_default();
        save_safetensors(
            &root.join("model-00001.safetensors"),
            &[
                ("proj.weight", &packed.weight),
                ("proj.scales", &packed.scales),
                ("proj.biases", &packed.biases),
                ("proj.bias", &bias),
            ],
            &metadata,
        )
        .unwrap();
        save_safetensors(
            &root.join("model-00002.safetensors"),
            &[("norm.weight", &norm)],
            &metadata,
        )
        .unwrap();
        packed.dequantize(64, bits, &stream).unwrap()
    }

    #[test]
    fn packed_linears_stay_packed_and_match_dequantized_reference() {
        let scratch = Scratch::new();
        let reference = write_bundle(&scratch.0, 4, Some("4"));
        let weights = load_directory(&scratch.0, WeightDtype::F32, None).unwrap();
        let expected = AffineQuantization::new(4, 64).unwrap();
        assert_eq!(weights.quantization(), Some(expected));
        let names: Vec<_> = weights.names().collect();
        assert_eq!(names, vec!["norm.weight", "proj.bias", "proj.weight"]);

        let proj = weights.linear("proj.weight").unwrap();
        assert_eq!(
            weights.packed_names().collect::<Vec<_>>(),
            vec!["proj.weight"]
        );
        assert_eq!(proj.compute_dtype(), Dtype::BFloat16);
        assert!(weights.tensor("proj.weight").is_err());
        assert_eq!(
            weights.tensor("norm.weight").unwrap().dtype(),
            Dtype::Float32,
            "F32 parity policy widens dense tensors only"
        );

        let packed_bytes = 32 * 256 / 2 + 2 * (32 * 256 / 64) * 2;
        assert_eq!(proj.nbytes(), packed_bytes);
        assert_eq!(weights.nbytes(), packed_bytes + 256 * 4 + 32 * 4);

        let stream = Stream::cpu().unwrap();
        let input = Array::from_f32_slice(
            &(0..3 * 256)
                .map(|index| (index % 17) as f32 * 0.03 - 0.2)
                .collect::<Vec<_>>(),
            &[3, 256],
        )
        .unwrap()
        .astype(Dtype::BFloat16, &stream)
        .unwrap();
        let bias = weights.tensor("proj.bias").unwrap();
        let actual = proj
            .apply(&input, Some(bias), &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        let expected = LinearWeight::dense(reference)
            .apply(&input, Some(bias), &stream)
            .unwrap()
            .to_vec_f32(&stream)
            .unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() < 0.1, "{actual} != {expected}");
        }
    }

    #[test]
    fn native_policy_keeps_checkpoint_dtype() {
        let scratch = Scratch::new();
        write_bundle(&scratch.0, 8, Some("8"));
        let weights = load_directory(&scratch.0, WeightDtype::Native, None).unwrap();
        assert_eq!(
            weights.quantization(),
            Some(AffineQuantization::new(8, 64).unwrap())
        );
        assert_eq!(
            weights.tensor("norm.weight").unwrap().dtype(),
            Dtype::BFloat16
        );
    }

    #[test]
    fn ambiguous_packing_without_metadata_fails_loud() {
        let scratch = Scratch::new();
        write_bundle(&scratch.0, 4, None);
        let error = load_directory(&scratch.0, WeightDtype::Native, None)
            .err()
            .expect("4-bit/g64 and 8-bit/g32 share shapes");
        assert!(
            error.to_string().contains("fits both 4 and 8 bits"),
            "{error}"
        );
        let weights = load_directory(
            &scratch.0,
            WeightDtype::Native,
            Some(AffineQuantization::new(4, 64).unwrap()),
        )
        .unwrap();
        assert_eq!(
            weights.quantization(),
            Some(AffineQuantization::new(4, 64).unwrap())
        );
    }

    #[test]
    fn registry_and_metadata_conflicts_fail_loud() {
        let scratch = Scratch::new();
        write_bundle(&scratch.0, 4, Some("4"));
        assert!(load_directory(
            &scratch.0,
            WeightDtype::Native,
            Some(AffineQuantization::new(8, 32).unwrap())
        )
        .is_err());
        assert!(load_directory(
            &scratch.0,
            WeightDtype::Native,
            Some(AffineQuantization::new(4, 128).unwrap())
        )
        .is_err());
    }

    #[test]
    fn registry_quantization_requires_packed_bundle() {
        let scratch = Scratch::new();
        let stream = Stream::cpu().unwrap();
        let linear = dense(32, 256, &stream);
        save_safetensors(
            &scratch.0.join("model.safetensors"),
            &[("proj.weight", &linear)],
            &[],
        )
        .unwrap();
        let error = load_directory(
            &scratch.0,
            WeightDtype::Native,
            Some(AffineQuantization::new(4, 64).unwrap()),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("no *.scales"), "{error}");
        let weights = load_directory(&scratch.0, WeightDtype::Native, None).unwrap();
        assert!(weights.quantization().is_none());
        assert_eq!(weights.packed_names().count(), 0);
        assert!(weights.tensor("proj.weight").is_ok());
    }

    #[test]
    fn incomplete_affine_layer_fails_loud() {
        let scratch = Scratch::new();
        let stream = Stream::cpu().unwrap();
        let packed = dense(32, 256, &stream)
            .quantize_affine(64, 4, &stream)
            .unwrap();
        save_safetensors(
            &scratch.0.join("model.safetensors"),
            &[
                ("proj.weight", &packed.weight),
                ("proj.scales", &packed.scales),
            ],
            &[("quantization_level", "4")],
        )
        .unwrap();
        let error = load_directory(&scratch.0, WeightDtype::Native, None)
            .err()
            .unwrap();
        assert!(error.to_string().contains("proj.biases"), "{error}");
    }

    #[test]
    fn missing_directory_is_an_error_not_a_panic() {
        let scratch = Scratch::new();
        let missing = scratch.0.join("absent");
        assert!(load_directory(&missing, WeightDtype::Native, None).is_err());
        assert!(load_directory(&scratch.0, WeightDtype::Native, None).is_err());
    }

    #[test]
    fn attach_lora_adds_the_residual() {
        let stream = Stream::cpu().unwrap();
        let weight = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("proj.weight".into(), weight);
        let weights = super::MlxWeights::from_dense(tensors);
        let scratch = Scratch::new();
        let path = scratch.0.join("adapter_model.safetensors");
        let down: Vec<u8> = [1.0f32, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let up: Vec<u8> = [1.0f32, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        crate::weights::write_safetensors(
            &path,
            &[
                crate::weights::SafeTensor {
                    name: "proj.lora_A.weight",
                    dtype: "F32",
                    shape: &[1, 2],
                    data: &down,
                },
                crate::weights::SafeTensor {
                    name: "proj.lora_B.weight",
                    dtype: "F32",
                    shape: &[2, 1],
                    data: &up,
                },
            ],
        )
        .unwrap();
        let adapted = weights
            .attach_lora(
                &[crate::weights::LoraFile {
                    id: "unit".into(),
                    path,
                    weight: 1.0,
                }],
                &stream,
            )
            .unwrap();
        let input = Array::from_f32_slice(&[1.0, 0.0], &[1, 2]).unwrap();
        let output = adapted
            .linear("proj.weight")
            .unwrap()
            .apply(&input, None, &stream)
            .unwrap();
        output.eval().unwrap();
        let values = output.to_vec_f32(&stream).unwrap();
        assert_eq!(values, vec![2.0, 0.0]);
    }
}
