//! Local quantization of a dense safetensors bundle into a `q4-mixed` bundle.
//!
//! The registry never downloads a quantized package. A `kind: derived` version
//! names its dense base plus a recipe; the installer downloads the dense base
//! and this module rewrites the large linear matrices into packed MLX affine
//! weights while the sensitive tensors (embeddings, norms, modulations,
//! final projections) stay at their checkpoint dtype.
//!
//! Output layout is the published MLX layout the loader already reads:
//! `{base}.weight` (packed uint32), `{base}.scales`, `{base}.biases`, shard
//! metadata `quantization_level`, and a `bundle.manifest.json` carrying the
//! recipe id/hash and the source layout digest.
//!
//! The loader rejects a bundle whose packed layers disagree on bits/group
//! size, so one recipe quantizes every selected layer at exactly one
//! `(bits, group_size)`. A recipe that cannot be honored fails loud.

use std::fs;
use std::path::{Path, PathBuf};

use danmo_mlx::{Array, Error, Result, Stream};

/// One conversion failure: `{path}: {detail}` under a stable operation tag.
fn fail(operation: &'static str, path: &Path, detail: impl Into<String>) -> Error {
    Error::new(operation, format!("{}: {}", path.display(), detail.into()))
}

/// Quantization applied to every selected linear of one bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
pub struct QuantizeLevel {
    pub bits: i32,
    pub group_size: i32,
}

/// One registry recipe: which components to rewrite and which tensors stay dense.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct QuantRecipe {
    pub id: String,
    pub version: u32,
    pub family: String,
    /// Relative component directories rewritten in place. `"."` is the root.
    #[serde(default = "default_components")]
    pub components: Vec<String>,
    /// Tensor-name globs that keep their checkpoint dtype.
    #[serde(default)]
    pub keep_dense: Vec<String>,
    /// Matrices smaller than this many elements stay dense.
    #[serde(default = "default_min_elements")]
    pub min_elements: u64,
    pub quantize: QuantizeLevel,
}

fn default_components() -> Vec<String> {
    vec![".".to_string()]
}

fn default_min_elements() -> u64 {
    65_536
}

impl QuantRecipe {
    pub fn parse(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .map_err(|error| fail("quantization recipe", path, error.to_string()))?;
        let recipe: Self = serde_json::from_str(&text)
            .map_err(|error| fail("quantization recipe", path, error.to_string()))?;
        recipe.validate(path)?;
        Ok(recipe)
    }

    fn validate(&self, path: &Path) -> Result<()> {
        let reject = |detail: String| fail("quantization recipe", path, detail);
        if self.id.trim().is_empty() {
            return Err(reject("recipe id is empty".into()));
        }
        if self.quantize.bits != 4 && self.quantize.bits != 8 {
            return Err(reject(format!(
                "recipe {} declares {} bits; MLX affine accepts 4 or 8",
                self.id, self.quantize.bits
            )));
        }
        if ![32, 64, 128].contains(&self.quantize.group_size) {
            return Err(reject(format!(
                "recipe {} declares group_size {}; MLX affine accepts 32, 64, or 128",
                self.id, self.quantize.group_size
            )));
        }
        if self.components.is_empty() {
            return Err(reject(format!("recipe {} has no components", self.id)));
        }
        for component in &self.components {
            if component.contains("..") || Path::new(component).is_absolute() {
                return Err(reject(format!(
                    "recipe {} component {component} escapes the bundle",
                    self.id
                )));
            }
        }
        if self
            .keep_dense
            .iter()
            .any(|pattern| pattern.trim().is_empty())
        {
            return Err(reject(format!(
                "recipe {} has an empty keep_dense glob",
                self.id
            )));
        }
        Ok(())
    }

    /// Digest of the recipe file bytes; the manifest carries it so a recipe
    /// edit invalidates an already-converted bundle.
    pub fn hash_at(path: &Path) -> Result<String> {
        let bytes =
            fs::read(path).map_err(|error| fail("quantization recipe", path, error.to_string()))?;
        Ok(sha256_hex(&bytes))
    }

    fn keeps_dense(&self, name: &str) -> bool {
        self.keep_dense
            .iter()
            .any(|pattern| glob_match(pattern, name))
    }
}

/// What a conversion actually did, for logs and the manifest.
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct QuantizeReport {
    pub shards: usize,
    pub tensors: usize,
    pub quantized: usize,
    pub dense: usize,
}

/// Convert `src` into `dst`. `dst` must not exist yet: a half-written bundle
/// never looks installed.
pub fn quantize_bundle(
    src: &Path,
    dst: &Path,
    recipe: &QuantRecipe,
    cancel: &(dyn Fn() -> bool + Sync),
    log: &(dyn Fn(&str) + Sync),
) -> Result<QuantizeReport> {
    if !src.is_dir() {
        return Err(fail("quantization", src, "dense source is not a directory"));
    }
    if dst.exists() {
        return Err(fail(
            "quantization",
            dst,
            "quantized output already exists; remove it before converting",
        ));
    }
    // Conversion is a build step: every read, pack, and write runs on a CPU
    // stream. A GPU stream would leave lazy `Load` primitives on the arrays and
    // `save_safetensors` cannot evaluate them.
    let stream = Stream::cpu()?;
    let mut report = QuantizeReport::default();
    fs::create_dir_all(dst)
        .map_err(|error| fail("quantization", dst, format!("create output: {error}")))?;

    // Rewrite each declared component shard by shard, then copy everything else
    // (VAE, tokenizer, configs) verbatim.
    for component in &recipe.components {
        if cancel() {
            return Err(cancelled());
        }
        let relative = normalize_component(component);
        let root = if relative.as_os_str().is_empty() {
            src.to_path_buf()
        } else {
            src.join(&relative)
        };
        if !root.is_dir() {
            return Err(fail(
                "quantization",
                src,
                format!("component {} is missing", relative.display()),
            ));
        }
        let out_root = if relative.as_os_str().is_empty() {
            dst.to_path_buf()
        } else {
            dst.join(&relative)
        };
        let shards = safetensors_in(&root)?;
        if shards.is_empty() {
            return Err(fail(
                "quantization",
                &root,
                "component has no safetensors shard",
            ));
        }
        fs::create_dir_all(&out_root)
            .map_err(|error| fail("quantization", &out_root, format!("create output: {error}")))?;
        for shard in &shards {
            if cancel() {
                return Err(cancelled());
            }
            let file_name = shard
                .file_name()
                .map(PathBuf::from)
                .ok_or_else(|| fail("quantization", shard, "shard has no file name"))?;
            let out_path = out_root.join(&file_name);
            let outcome = rewrite_shard(shard, &out_path, recipe, &stream, &mut report)?;
            log(&format!(
                "{}: {} tensors, {} quantized, {} dense",
                out_path.display(),
                outcome.0,
                outcome.1,
                outcome.2
            ));
        }
        // Component-local non-weight files (config.json, tokenizer) and any
        // file in a subdirectory of the component.
        copy_non_shards(&root, &out_root)?;
    }
    copy_rest(src, dst, recipe)?;
    Ok(report)
}

fn cancelled() -> Error {
    Error::new("quantization", "quantization cancelled")
}

fn normalize_component(component: &str) -> PathBuf {
    if component == "." || component.is_empty() {
        PathBuf::new()
    } else {
        PathBuf::from(component)
    }
}

fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    let entries =
        fs::read_dir(dir).map_err(|error| fail("quantization", dir, error.to_string()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Rewrite one shard. Returns `(tensors, quantized, dense)`.
fn rewrite_shard(
    shard: &Path,
    out_path: &Path,
    recipe: &QuantRecipe,
    stream: &Stream,
    report: &mut QuantizeReport,
) -> Result<(usize, usize, usize)> {
    let tensors = danmo_mlx::SafeTensors::load(shard, stream)?;
    let names = tensors.names()?;
    let mut values: Vec<(String, Array)> = Vec::with_capacity(names.len() * 3);
    let mut quantized = 0usize;
    let mut dense = 0usize;
    for name in &names {
        let Some(tensor) = tensors.get(name)? else {
            return Err(fail(
                "quantization",
                shard,
                format!("shard listed {name} but did not return it"),
            ));
        };
        let base = name.strip_suffix(".weight");
        let shape = tensor.shape();
        let eligible = base.is_some()
            && shape.len() == 2
            && tensor.dtype().is_floating()
            && !recipe.keeps_dense(name)
            && (shape[0] as u64) * (shape[1] as u64) >= recipe.min_elements;
        if !eligible {
            values.push((name.clone(), tensor));
            dense += 1;
            continue;
        }
        let base = base.expect("eligible checked the suffix");
        let inputs = shape[1] as i64;
        let group = recipe.quantize.group_size as i64;
        if inputs % group != 0 {
            return Err(fail(
                "quantization",
                shard,
                format!(
                    "{name} has {inputs} inputs, which is not a multiple of group_size {group}; \
                     add a keep_dense rule for this layer or change the recipe"
                ),
            ));
        }
        let packed = super::mlx::pack_affine(
            &tensor,
            recipe.quantize.group_size,
            recipe.quantize.bits,
            stream,
        )?;
        values.push((format!("{base}.weight"), packed.weight));
        values.push((format!("{base}.scales"), packed.scales));
        values.push((format!("{base}.biases"), packed.biases));
        quantized += 1;
    }
    let refs: Vec<(&str, &Array)> = values
        .iter()
        .map(|(name, array)| (name.as_str(), array))
        .collect();
    let level = recipe.quantize.bits.to_string();
    let metadata: Vec<(&str, &str)> = if quantized > 0 {
        vec![("quantization_level", level.as_str())]
    } else {
        Vec::new()
    };
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            fail(
                "quantization",
                parent,
                format!("create output dir: {error}"),
            )
        })?;
    }
    danmo_mlx::save_safetensors(out_path, &refs, &metadata)?;
    report.shards += 1;
    report.tensors += values.len();
    report.quantized += quantized;
    report.dense += dense;
    Ok((values.len(), quantized, dense))
}

/// Copy non-weight files beside the rewritten shards (configs, tokenizers).
fn copy_non_shards(from: &Path, to: &Path) -> Result<()> {
    let entries =
        fs::read_dir(from).map_err(|error| fail("quantization", from, error.to_string()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_file() {
            if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
                || name == "bundle.manifest.json"
            {
                continue;
            }
            fs::copy(&path, to.join(&name))
                .map_err(|error| fail("quantization", &path, format!("copy: {error}")))?;
        }
    }
    Ok(())
}

/// Copy every source entry that no recipe component owns.
fn copy_rest(src: &Path, dst: &Path, recipe: &QuantRecipe) -> Result<()> {
    let owned: Vec<PathBuf> = recipe
        .components
        .iter()
        .map(|component| normalize_component(component))
        .collect();
    // `"."` already rewrote every root shard and copied root sidecars; only
    // nested directories (tokenizer, text_encoder, …) still need a copy.
    let owns_root = owned
        .iter()
        .any(|component| component.as_os_str().is_empty());
    let entries =
        fs::read_dir(src).map_err(|error| fail("quantization", src, error.to_string()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let relative = PathBuf::from(&name);
        if owned.iter().any(|component| component == &relative) {
            continue;
        }
        if name == "bundle.manifest.json" {
            continue;
        }
        if owns_root && path.is_file() {
            continue;
        }
        let target = dst.join(&name);
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            fs::copy(&path, &target)
                .map_err(|error| fail("quantization", &path, format!("copy: {error}")))?;
        }
    }
    Ok(())
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)
        .map_err(|error| fail("quantization", to, format!("create dir: {error}")))?;
    let entries =
        fs::read_dir(from).map_err(|error| fail("quantization", from, error.to_string()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if name == ".install-hook-incomplete" {
            continue;
        }
        let target = to.join(&name);
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            fs::copy(&path, &target)
                .map_err(|error| fail("quantization", &path, format!("copy: {error}")))?;
        }
    }
    Ok(())
}

/// Layout fingerprint of the dense source: sorted `(relative path, size)`
/// pairs. It is not a content hash; it catches a re-published or re-laid-out
/// base without reading tens of gigabytes twice.
pub fn source_digest(src: &Path) -> Result<String> {
    let mut rows: Vec<String> = Vec::new();
    collect_sizes(src, src, &mut rows)?;
    rows.sort();
    let body = rows.join("\n");
    Ok(sha256_hex(body.as_bytes()))
}

fn collect_sizes(root: &Path, dir: &Path, rows: &mut Vec<String>) -> Result<()> {
    let entries =
        fs::read_dir(dir).map_err(|error| fail("quantization", dir, error.to_string()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sizes(root, &path, rows)?;
            continue;
        }
        let relative = path.strip_prefix(root).unwrap_or(&path);
        let size = entry.metadata().map(|meta| meta.len()).unwrap_or(0);
        rows.push(format!("{}\t{size}", relative.display()));
    }
    Ok(())
}

/// `bundle.manifest.json` payload for a converted bundle. `layers` is omitted
/// when the producer did not count tensors (a family converter with its own
/// writer).
pub fn manifest_value(
    recipe: &QuantRecipe,
    recipe_hash: &str,
    src: &Path,
    digest: &str,
    report: &QuantizeReport,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "quant": format!("q{}-mixed", recipe.quantize.bits),
        "source": {
            "path": src.display().to_string(),
            "layout_digest": digest,
        },
        "recipe": {
            "id": recipe.id,
            "version": recipe.version,
            "hash": recipe_hash,
        },
        "quantization": {
            "bits": recipe.quantize.bits,
            "group_size": recipe.quantize.group_size,
            "scheme": "mlx_affine",
        }
    });
    if report.quantized > 0 || report.dense > 0 {
        body["layers"] = serde_json::json!({
            "quantized": report.quantized,
            "dense": report.dense,
        });
    }
    body
}

pub const MANIFEST_FORMAT: &str = "danmo-mlx-quantized";
pub const MANIFEST_VERSION: u32 = 1;

/// Fail loud when a converted bundle's manifest names a different recipe than
/// the one on disk, or when the format is not one this build understands.
///
/// When `recipe` is `Some`, this asserts the path is a finished quantized
/// bundle: missing or non-quantized manifests are errors. `None` only checks
/// an existing quantized manifest if one is present (dense trees stay Ok).
pub fn verify_quantized_bundle(root: &Path, recipe: Option<(&QuantRecipe, &str)>) -> Result<()> {
    let path = root.join("bundle.manifest.json");
    if !path.is_file() {
        return match recipe {
            Some(_) => Err(fail(
                "quantization",
                root,
                "quantized bundle has no bundle.manifest.json; re-convert the bundle",
            )),
            None => Ok(()),
        };
    }
    let text = fs::read_to_string(&path)
        .map_err(|error| fail("quantization", &path, error.to_string()))?;
    let body: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        fail(
            "quantization",
            &path,
            format!("manifest is not JSON: {error}"),
        )
    })?;
    if body.get("format").and_then(|value| value.as_str()) != Some(MANIFEST_FORMAT) {
        return match recipe {
            Some(_) => Err(fail(
                "quantization",
                &path,
                format!(
                    "bundle.manifest.json is not a {MANIFEST_FORMAT} manifest; re-convert the bundle"
                ),
            )),
            None => Ok(()),
        };
    }
    let version = body
        .get("format_version")
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            fail(
                "quantization",
                &path,
                "quantized manifest has no format_version",
            )
        })?;
    if version != MANIFEST_VERSION as u64 {
        return Err(fail(
            "quantization",
            &path,
            format!(
                "quantized manifest format_version {version} is not {MANIFEST_VERSION}; re-convert the bundle"
            ),
        ));
    }
    let Some((recipe, hash)) = recipe else {
        return Ok(());
    };
    let declared_id = body
        .pointer("/recipe/id")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    let declared_hash = body
        .pointer("/recipe/hash")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    if declared_id != recipe.id {
        return Err(fail(
            "quantization manifest",
            &path,
            format!(
                "bundle was converted with recipe {declared_id}, registry asks for {}",
                recipe.id
            ),
        ));
    }
    if declared_hash != hash {
        return Err(fail(
            "quantization manifest",
            &path,
            format!(
                "recipe {} changed since this bundle was converted; re-convert the bundle",
                recipe.id
            ),
        ));
    }
    Ok(())
}

/// Minimal glob: `*` spans any run of characters, `?` one character.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut resume) = (usize::MAX, 0usize);
    while t < text.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == text[t]) {
            p += 1;
            t += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = p;
            resume = t;
            p += 1;
        } else if star != usize::MAX {
            p = star + 1;
            resume += 1;
            t = resume;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == '*' {
        p += 1;
    }
    p == pattern.len()
}

/// SHA-256 of `bytes`. In-tree so digests do not pull a new dependency.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            w[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = w[index - 15].rotate_right(7)
                ^ w[index - 15].rotate_right(18)
                ^ (w[index - 15] >> 3);
            let s1 = w[index - 2].rotate_right(17)
                ^ w[index - 2].rotate_right(19)
                ^ (w[index - 2] >> 10);
            w[index] = w[index - 16]
                .wrapping_add(s0)
                .wrapping_add(w[index - 7])
                .wrapping_add(s1);
        }
        let mut a = state;
        for index in 0..64 {
            let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
            let ch = (a[4] & a[5]) ^ ((!a[4]) & a[6]);
            let temp1 = a[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[index])
                .wrapping_add(w[index]);
            let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
            let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
            let temp2 = s0.wrapping_add(maj);
            a[7] = a[6];
            a[6] = a[5];
            a[5] = a[4];
            a[4] = a[3].wrapping_add(temp1);
            a[3] = a[2];
            a[2] = a[1];
            a[1] = a[0];
            a[0] = temp1.wrapping_add(temp2);
        }
        for index in 0..8 {
            state[index] = state[index].wrapping_add(a[index]);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
}

/// Names the installer and CLI use to find a recipe on disk.
pub fn recipe_paths(workspace: &Path, repo_root: &Path, id: &str) -> Vec<PathBuf> {
    let file = format!("{id}.json");
    vec![
        workspace.join("config").join("quant_recipes").join(&file),
        repo_root
            .join("default_config")
            .join("quant_recipes")
            .join(&file),
    ]
}

/// Resolve a recipe by id (or an explicit path) and return it with its hash.
pub fn load_recipe(
    workspace: &Path,
    repo_root: &Path,
    id_or_path: &str,
) -> Result<(QuantRecipe, String)> {
    let explicit = Path::new(id_or_path);
    if explicit.is_file() {
        let recipe = QuantRecipe::parse(explicit)?;
        let hash = QuantRecipe::hash_at(explicit)?;
        return Ok((recipe, hash));
    }
    let mut missing = Vec::new();
    for candidate in recipe_paths(workspace, repo_root, id_or_path) {
        if candidate.is_file() {
            let recipe = QuantRecipe::parse(&candidate)?;
            let hash = QuantRecipe::hash_at(&candidate)?;
            return Ok((recipe, hash));
        }
        missing.push(candidate.display().to_string());
    }
    Err(Error::new(
        "quantization recipe",
        format!(
            "quantization recipe {id_or_path} is missing ({})",
            missing.join(", ")
        ),
    ))
}

#[cfg(test)]
pub(crate) mod fixture {
    use super::*;
    use danmo_mlx::Dtype;

    /// Unique scratch directory per test, cleaned up before use.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "danmo-quantize-{name}-{}-{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    pub(crate) fn dense_matrix(out: i32, inputs: i32, stream: &Stream) -> Array {
        let values: Vec<f32> = (0..(out * inputs))
            .map(|index| ((index % 37) as f32) * 0.013 - 0.24)
            .collect();
        Array::from_f32_slice(&values, &[out, inputs])
            .unwrap()
            .astype(Dtype::BFloat16, stream)
            .unwrap()
    }

    /// Tiny dense bundle: one big linear plus a norm and an embedding that a
    /// recipe can leave dense.
    pub(crate) fn write_dense(root: &Path, stream: &Stream) {
        let component = root.join("transformer");
        fs::create_dir_all(&component).unwrap();
        let big = dense_matrix(256, 256, stream);
        let norm = Array::from_f32_slice(&[1.0f32; 256], &[256]).unwrap();
        let embed = dense_matrix(128, 128, stream);
        danmo_mlx::save_safetensors(
            &component.join("model.safetensors"),
            &[
                ("blocks.0.attn.weight", &big),
                ("blocks.0.norm.weight", &norm),
                ("x_embedder.weight", &embed),
            ],
            &[],
        )
        .unwrap();
        fs::write(component.join("config.json"), b"{}").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;

    #[test]
    fn glob_matches_names() {
        assert!(glob_match("*embedder*", "x_embedder.weight"));
        assert!(glob_match("*.weight", "a.b.weight"));
        assert!(!glob_match("*.bias", "a.b.weight"));
        assert!(glob_match(
            "transformer.blocks.?.attn.weight",
            "transformer.blocks.0.attn.weight"
        ));
        assert!(!glob_match("blocks.?", "blocks.10"));
    }

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 200]),
            "c2a908d98f5df987ade41b5fce213067efbcc21ef2240212a41e54b5e7c28ae5".to_string()
        );
    }

    #[test]
    fn recipe_rejects_bad_bits() {
        let dir = std::env::temp_dir().join(format!("danmo-recipe-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        fs::write(
            &path,
            r#"{"id":"x","version":1,"family":"z_image","quantize":{"bits":6,"group_size":32}}"#,
        )
        .unwrap();
        let error = QuantRecipe::parse(&path).unwrap_err().to_string();
        assert!(error.contains("4 or 8"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip_quantizes_big_linears_and_keeps_sensitive_tensors_dense() {
        let root = scratch("roundtrip");
        let src = root.join("dense");
        let dst = root.join("q4");
        let stream = Stream::cpu().unwrap();
        write_dense(&src, &stream);

        let recipe = QuantRecipe {
            id: "test-q4-mixed-v1".into(),
            version: 1,
            family: "z_image".into(),
            components: vec!["transformer".into()],
            keep_dense: vec!["*norm*".into(), "*embedder*".into()],
            min_elements: 1024,
            quantize: QuantizeLevel {
                bits: 4,
                group_size: 64,
            },
        };
        let report = quantize_bundle(&src, &dst, &recipe, &|| false, &|_message: &str| {}).unwrap();
        assert_eq!(report.quantized, 1, "only the plain linear is quantized");
        assert!(
            report.dense >= 2,
            "norm and embedder stay dense: {report:?}"
        );

        // The published MLX layout: packed weight + scales + biases.
        let out = danmo_mlx::SafeTensors::load(&dst.join("transformer/model.safetensors"), &stream)
            .unwrap();
        let names = out.names().unwrap();
        assert!(names.iter().any(|name| name == "blocks.0.attn.weight"));
        assert!(names.iter().any(|name| name == "blocks.0.attn.scales"));
        assert!(names.iter().any(|name| name == "blocks.0.attn.biases"));
        assert!(
            !names
                .iter()
                .any(|name| name == "blocks.0.attn.scales.scales"),
            "{names:?}"
        );
        assert_eq!(
            out.metadata("quantization_level").unwrap().as_deref(),
            Some("4")
        );
        // Sensitive tensors kept their checkpoint dtype.
        let norm = out.get("blocks.0.norm.weight").unwrap().unwrap();
        assert_eq!(norm.shape(), vec![256]);
        // Component-local config files are copied.
        assert!(dst.join("transformer/config.json").is_file());

        // The loader must accept the converted bundle at the recipe's level.
        let weights = crate::weights::load_mlx_directory(
            &dst.join("transformer"),
            crate::weights::WeightDtype::Native,
            Some(danmo_mlx::AffineQuantization::new(4, 64).unwrap()),
        )
        .unwrap();
        assert_eq!(
            weights.quantization(),
            Some(danmo_mlx::AffineQuantization::new(4, 64).unwrap())
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn root_component_rewrites_shards_and_copies_nested_dirs() {
        // Flat families (LTX after ingest) declare `components: ["."]`.
        let root = scratch("root-component");
        let src = root.join("dense");
        let dst = root.join("q4");
        let stream = Stream::cpu().unwrap();
        fs::create_dir_all(src.join("text_encoder")).unwrap();
        let big = dense_matrix(256, 256, &stream);
        let norm = Array::from_f32_slice(&[1.0f32; 256], &[256]).unwrap();
        danmo_mlx::save_safetensors(
            &src.join("transformer.safetensors"),
            &[
                ("blocks.0.attn.weight", &big),
                ("blocks.0.norm.weight", &norm),
            ],
            &[],
        )
        .unwrap();
        fs::write(src.join("bundle_config.json"), b"{}").unwrap();
        fs::write(src.join("text_encoder/config.json"), b"{\"ok\":true}").unwrap();

        let recipe = QuantRecipe {
            id: "flat-q4-mixed-v1".into(),
            version: 1,
            family: "ltx25".into(),
            components: vec![".".into()],
            keep_dense: vec!["*norm*".into()],
            min_elements: 1024,
            quantize: QuantizeLevel {
                bits: 4,
                group_size: 64,
            },
        };
        quantize_bundle(&src, &dst, &recipe, &|| false, &|_message: &str| {}).unwrap();

        let out =
            danmo_mlx::SafeTensors::load(&dst.join("transformer.safetensors"), &stream).unwrap();
        assert!(out
            .names()
            .unwrap()
            .iter()
            .any(|name| name == "blocks.0.attn.scales"));
        assert_eq!(
            out.metadata("quantization_level").unwrap().as_deref(),
            Some("4")
        );
        assert_eq!(
            fs::read_to_string(dst.join("bundle_config.json")).unwrap(),
            "{}"
        );
        assert_eq!(
            fs::read_to_string(dst.join("text_encoder/config.json")).unwrap(),
            "{\"ok\":true}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn ineligible_input_width_fails_loud() {
        let root = scratch("width");
        let src = root.join("dense");
        let stream = Stream::cpu().unwrap();
        let component = src.join("transformer");
        fs::create_dir_all(&component).unwrap();
        // 100 inputs is not a multiple of group_size 64.
        let odd = dense_matrix(128, 100, &stream);
        danmo_mlx::save_safetensors(
            &component.join("model.safetensors"),
            &[("blocks.0.attn.weight", &odd)],
            &[],
        )
        .unwrap();
        let recipe = QuantRecipe {
            id: "test".into(),
            version: 1,
            family: "z_image".into(),
            components: vec!["transformer".into()],
            keep_dense: Vec::new(),
            min_elements: 64,
            quantize: QuantizeLevel {
                bits: 4,
                group_size: 64,
            },
        };
        let error = quantize_bundle(&src, &root.join("q4"), &recipe, &|| false, &|_m: &str| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a multiple of group_size"), "{error}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn manifest_round_trips_and_rejects_a_changed_recipe() {
        let root = scratch("manifest");
        let stream = Stream::cpu().unwrap();
        let src = root.join("dense");
        write_dense(&src, &stream);
        let recipe = QuantRecipe {
            id: "z-image-q4-mixed-v1".into(),
            version: 1,
            family: "z_image".into(),
            components: vec!["transformer".into()],
            keep_dense: vec!["*norm*".into()],
            min_elements: 1024,
            quantize: QuantizeLevel {
                bits: 4,
                group_size: 64,
            },
        };
        let dst = root.join("q4");
        let report = quantize_bundle(&src, &dst, &recipe, &|| false, &|_message: &str| {}).unwrap();
        let digest = source_digest(&src).unwrap();
        let body = manifest_value(&recipe, "hash-v1", &src, &digest, &report);
        fs::write(
            dst.join("bundle.manifest.json"),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();

        assert!(verify_quantized_bundle(&dst, Some((&recipe, "hash-v1"))).is_ok());
        let error = verify_quantized_bundle(&dst, Some((&recipe, "hash-v2")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("re-convert"), "{error}");

        let other = QuantRecipe {
            id: "different-recipe".into(),
            ..recipe.clone()
        };
        let error = verify_quantized_bundle(&dst, Some((&other, "hash-v1")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("different-recipe"), "{error}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn derived_output_must_not_already_exist() {
        let root = scratch("existing");
        let stream = Stream::cpu().unwrap();
        let src = root.join("dense");
        write_dense(&src, &stream);
        let dst = root.join("q4");
        fs::create_dir_all(&dst).unwrap();
        let recipe = QuantRecipe {
            id: "test".into(),
            version: 1,
            family: "z_image".into(),
            components: vec!["transformer".into()],
            keep_dense: Vec::new(),
            min_elements: 1024,
            quantize: QuantizeLevel {
                bits: 4,
                group_size: 64,
            },
        };
        let error = quantize_bundle(&src, &dst, &recipe, &|| false, &|_m: &str| {})
            .unwrap_err()
            .to_string();
        assert!(error.contains("already exists"), "{error}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn verify_with_recipe_rejects_a_missing_manifest() {
        let root = scratch("no-manifest");
        fs::create_dir_all(&root).unwrap();
        let recipe = QuantRecipe {
            id: "test".into(),
            version: 1,
            family: "z_image".into(),
            components: vec!["transformer".into()],
            keep_dense: Vec::new(),
            min_elements: 1024,
            quantize: QuantizeLevel {
                bits: 4,
                group_size: 64,
            },
        };
        let error = verify_quantized_bundle(&root, Some((&recipe, "hash")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("bundle.manifest.json"), "{error}");
        assert!(verify_quantized_bundle(&root, None).is_ok());
        let _ = fs::remove_dir_all(&root);
    }
}
