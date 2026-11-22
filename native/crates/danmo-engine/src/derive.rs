//! Registry-driven local quantization.
//!
//! A registry version with `"kind": "derived"` is not a download: it names a
//! dense base version plus a recipe. The installer downloads the dense base,
//! then calls [`run_derive`] to write the `q4-mixed` bundle. This path is the
//! only producer of quantized bundles; nothing downloads a pre-quantized
//! package. Families may still own a layout ingest (LTX split Comfy pack) on
//! the dense base; they do not ship a parallel quantizer.
//!
//! Everything runs on the process MLX executor, so arrays never leave its
//! thread. The engine surface here takes paths and returns plain values, which
//! is what the download worker and the CLI both need.

use std::path::{Path, PathBuf};

use crate::weights::{
    load_recipe, manifest_value, quantize_bundle, recipe_paths, source_digest,
    verify_quantized_bundle, QuantizeReport,
};

/// Where the derived bundle for a model/version lives, resolved against the
/// workspace. Mirrors how the installer resolves every other `local_path`.
pub fn derived_layout(workspace: &Path, base_local_path: &str, version_key: &str) -> PathBuf {
    let base = if Path::new(base_local_path).is_absolute() {
        PathBuf::from(base_local_path)
    } else {
        workspace.join(base_local_path)
    };
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("bundle");
    let suffix = if version_key.is_empty() {
        "q4-mixed".to_string()
    } else {
        version_key.to_string()
    };
    let target = format!("{name}-{suffix}");
    base.with_file_name(target)
}

/// Marker the download worker drops while install hooks run. Mirrored here so
/// derive can tell an installer-precreated empty directory from a finished
/// quantized bundle without depending on `danmo-resolve`.
const INSTALL_HOOK_INCOMPLETE: &str = ".install-hook-incomplete";

/// Resolve a `derived` version against its dense base and convert it.
///
/// `base_local_path` and `derived_local_path` are registry `local_path` values
/// (workspace-relative or absolute). Returns the conversion report.
pub fn run_derive(
    workspace: &Path,
    repo_root: &Path,
    base_local_path: &str,
    derived_local_path: &str,
    recipe_id: &str,
    cancel: Box<dyn Fn() -> bool + Send + Sync>,
    log: Box<dyn Fn(&str) + Send + Sync>,
) -> Result<QuantizeReport, String> {
    let src = resolve(workspace, base_local_path);
    let dst = resolve(workspace, derived_local_path);
    if !src.is_dir() {
        return Err(format!(
            "derived quantization needs the dense base at {}; install it first",
            src.display()
        ));
    }
    let (recipe, hash) =
        load_recipe(workspace, repo_root, recipe_id).map_err(|error| error.to_string())?;
    // The installer may pre-create `dst` with only `.install-hook-incomplete`.
    // That must not count as a finished convert (previously verify no-op'd on
    // a missing manifest and the task was marked completed).
    if dst.exists() {
        if derived_target_reusable(&dst, &recipe, &hash)? {
            log(&format!(
                "{} already exists; leaving the converted bundle in place",
                dst.display()
            ));
            return Ok(QuantizeReport::default());
        }
        log(&format!(
            "removing incomplete derived target at {} before converting",
            dst.display()
        ));
        std::fs::remove_dir_all(&dst).map_err(|error| {
            format!(
                "remove incomplete derived target {}: {error}",
                dst.display()
            )
        })?;
    }
    log(&format!(
        "converting {} -> {} with recipe {} ({} bit, group {})",
        src.display(),
        dst.display(),
        recipe.id,
        recipe.quantize.bits,
        recipe.quantize.group_size
    ));
    let digest = source_digest(&src).map_err(|error| error.to_string())?;
    let executor =
        crate::runtime::mlx_executor().map_err(|error| format!("MLX executor: {error}"))?;
    let recipe_for_job = recipe.clone();
    let source = src.clone();
    let target = dst.clone();
    let cancel = std::sync::Arc::new(cancel);
    let log = std::sync::Arc::new(log);
    let cancel_job = cancel.clone();
    let log_job = log.clone();
    let outcome = executor
        .execute("derive quantization", move |_, _stream, _cache, _events| {
            quantize_bundle(
                &source,
                &target,
                &recipe_for_job,
                cancel_job.as_ref(),
                log_job.as_ref(),
            )
        })
        .map_err(|error| error.to_string())?;
    let body = manifest_value(&recipe, &hash, &src, &digest, &outcome);
    write_manifest(&dst, &body)?;
    Ok(outcome)
}

/// A finished quantized tree can be reused. An installer stub (incomplete
/// marker and/or no quantized manifest) must be rebuilt. A tree that looks
/// finished but fails recipe verification fails loud — do not silently wipe a
/// foreign or stale complete bundle.
fn derived_target_reusable(
    dst: &Path,
    recipe: &crate::weights::QuantRecipe,
    hash: &str,
) -> Result<bool, String> {
    if dst.join(INSTALL_HOOK_INCOMPLETE).is_file() {
        return Ok(false);
    }
    if !dst.join("bundle.manifest.json").is_file() {
        return Ok(false);
    }
    match verify_quantized_bundle(dst, Some((recipe, hash))) {
        Ok(()) => Ok(true),
        Err(error) => Err(error.to_string()),
    }
}

/// Locate a recipe for diagnostics (CLI `--print-recipe` and server logs).
pub fn find_recipe(workspace: &Path, repo_root: &Path, id: &str) -> Result<PathBuf, String> {
    recipe_paths(workspace, repo_root, id)
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| format!("quantization recipe {id} is missing"))
}

/// Write the manifest for a bundle a family converter produced (LTX assembles a
/// split Comfy pack, so it cannot go through the in-place rewriter). The
/// manifest still carries the recipe id/hash, so a recipe edit invalidates the
/// bundle exactly like a generically converted one.
pub fn write_derived_manifest(
    workspace: &Path,
    repo_root: &Path,
    base_local_path: &str,
    derived_local_path: &str,
    recipe_id: &str,
) -> Result<(), String> {
    let src = resolve(workspace, base_local_path);
    let dst = resolve(workspace, derived_local_path);
    if !dst.is_dir() {
        return Err(format!(
            "{} was not produced by the family converter",
            dst.display()
        ));
    }
    let (recipe, hash) =
        load_recipe(workspace, repo_root, recipe_id).map_err(|error| error.to_string())?;
    let digest = source_digest(&src).map_err(|error| error.to_string())?;
    let report = QuantizeReport::default();
    let body = manifest_value(&recipe, &hash, &src, &digest, &report);
    write_manifest(&dst, &body)
}

fn resolve(workspace: &Path, path: &str) -> PathBuf {
    let path = PathBuf::from(path);
    if path.is_absolute() {
        path
    } else {
        workspace.join(path)
    }
}

fn write_manifest(root: &Path, body: &serde_json::Value) -> Result<(), String> {
    std::fs::create_dir_all(root).map_err(|error| error.to_string())?;
    let mut body = body.clone();
    // Stamp a dense-style `files` listing so version_ready / UI component
    // checks agree with install manifests. Missing roles stay omitted.
    if body.get("files").is_none() {
        if let Some(files) = scan_bundle_files(root) {
            body["files"] = files;
        }
    }
    let mut bytes = serde_json::to_vec_pretty(&body).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    std::fs::write(root.join("bundle.manifest.json"), bytes).map_err(|error| error.to_string())
}

fn scan_bundle_files(root: &Path) -> Option<serde_json::Value> {
    let mut map = serde_json::Map::new();
    for role in ["transformer", "text_encoder", "vae"] {
        let dir = root.join(role);
        if !dir.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut files: Vec<String> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("safetensors"))
            })
            .filter_map(|path| {
                path.strip_prefix(root)
                    .ok()
                    .map(|relative| relative.display().to_string())
            })
            .collect();
        files.sort();
        if files.is_empty() {
            continue;
        }
        let value = if files.len() == 1 {
            serde_json::Value::String(files.remove(0))
        } else {
            serde_json::Value::Array(files.into_iter().map(serde_json::Value::String).collect())
        };
        map.insert(role.to_string(), value);
    }
    if !map.contains_key("transformer") {
        return None;
    }
    Some(serde_json::Value::Object(map))
}

/// Registry hook entry point. The download worker hands the hook an absolute
/// dense base and the recipe id (or an absolute recipe path); the target is
/// the directory being installed.
pub(crate) fn ingest_from_hook(
    bundle_root: &Path,
    hook_spec: &serde_json::Value,
    cancel: Box<dyn Fn() -> bool + Send + Sync>,
    log: Box<dyn Fn(&str) + Send + Sync>,
) -> Result<(), String> {
    let base = hook_spec
        .get("base")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "derive_quantize hook needs an absolute dense base path".to_string())?;
    let recipe = hook_spec
        .get("recipe")
        .and_then(|value| value.as_str())
        .filter(|text| !text.is_empty())
        .ok_or_else(|| "derive_quantize hook needs a recipe id".to_string())?;
    let workspace = hook_spec
        .get("workspace")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    let repo_root = hook_spec
        .get("repo_root")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    run_derive(
        Path::new(workspace),
        Path::new(repo_root),
        base,
        &bundle_root.display().to_string(),
        recipe,
        cancel,
        log,
    )
    .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::derived_layout;
    use crate::weights::fixture::{scratch, write_dense};
    use std::path::Path;

    #[test]
    fn derived_layout_puts_the_bundle_beside_the_dense_base() {
        let workspace = std::path::Path::new("/w");
        let path = derived_layout(workspace, "models/Image/z-image-turbo-fp16", "q4-mixed");
        assert_eq!(
            path,
            std::path::Path::new("/w/models/Image/z-image-turbo-fp16-q4-mixed")
        );
    }

    fn write_recipe(root: &Path, id: &str) {
        let dir = root.join("default_config/quant_recipes");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(format!("{id}.json")),
            serde_json::json!({
                "id": id,
                "version": 1,
                "family": "z_image",
                "components": ["transformer"],
                "keep_dense": ["*norm*"],
                "min_elements": 1024,
                "quantize": { "bits": 4, "group_size": 64 }
            })
            .to_string(),
        )
        .unwrap();
    }

    #[test]
    fn run_derive_writes_a_manifest_and_refuses_to_reconvert() {
        use danmo_mlx::Stream;
        let root = scratch("run");
        let workspace = root.join("workspace");
        write_dense(
            &workspace.join("models/Image/dense"),
            &Stream::cpu().unwrap(),
        );
        write_recipe(&root, "z-image-q4-mixed-v1");
        let base = "models/Image/dense";
        let target = "models/Image/dense-q4-mixed";

        let report = super::run_derive(
            &workspace,
            &root,
            base,
            target,
            "z-image-q4-mixed-v1",
            Box::new(|| false),
            Box::new(|_message: &str| {}),
        )
        .unwrap();
        assert_eq!(
            report.quantized, 2,
            "attention and embedder matrices quantized: {report:?}"
        );
        assert_eq!(report.dense, 1, "the norm stays dense: {report:?}");

        let derived = workspace.join(target);
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(derived.join("bundle.manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["format"], super::super::weights::MANIFEST_FORMAT);
        assert_eq!(manifest["recipe"]["id"], "z-image-q4-mixed-v1");
        assert!(manifest["source"]["layout_digest"].as_str().is_some());
        assert!(
            manifest["files"]["transformer"].is_string()
                || manifest["files"]["transformer"].is_array(),
            "derived manifest should stamp a files listing: {manifest}"
        );

        // A second run reuses the converted bundle instead of rewriting it.
        let again = super::run_derive(
            &workspace,
            &root,
            base,
            target,
            "z-image-q4-mixed-v1",
            Box::new(|| false),
            Box::new(|_message: &str| {}),
        )
        .unwrap();
        assert_eq!(again.shards, 0, "second run is a no-op: {again:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_derive_requires_the_dense_base() {
        let root = scratch("missing");
        write_recipe(&root, "z-image-q4-mixed-v1");
        let error = super::run_derive(
            &root,
            &root,
            "models/Image/absent",
            "models/Image/absent-q4-mixed",
            "z-image-q4-mixed-v1",
            Box::new(|| false),
            Box::new(|_message: &str| {}),
        )
        .unwrap_err();
        assert!(error.contains("install it first"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_derive_rebuilds_an_installer_precreated_stub() {
        use danmo_mlx::Stream;
        let root = scratch("stub");
        let workspace = root.join("workspace");
        write_dense(
            &workspace.join("models/Image/dense"),
            &Stream::cpu().unwrap(),
        );
        write_recipe(&root, "z-image-q4-mixed-v1");
        let target = workspace.join("models/Image/dense-q4-mixed");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::write(target.join(super::INSTALL_HOOK_INCOMPLETE), b"pending").unwrap();

        let report = super::run_derive(
            &workspace,
            &root,
            "models/Image/dense",
            "models/Image/dense-q4-mixed",
            "z-image-q4-mixed-v1",
            Box::new(|| false),
            Box::new(|_message: &str| {}),
        )
        .unwrap();
        assert!(
            report.quantized > 0,
            "stub must be rebuilt, not treated as done: {report:?}"
        );
        assert!(target.join("bundle.manifest.json").is_file());
        assert!(!target.join(super::INSTALL_HOOK_INCOMPLETE).is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_derive_rebuilds_an_empty_leftover_directory() {
        use danmo_mlx::Stream;
        let root = scratch("empty-dst");
        let workspace = root.join("workspace");
        write_dense(
            &workspace.join("models/Image/dense"),
            &Stream::cpu().unwrap(),
        );
        write_recipe(&root, "z-image-q4-mixed-v1");
        let target = workspace.join("models/Image/dense-q4-mixed");
        std::fs::create_dir_all(&target).unwrap();

        let report = super::run_derive(
            &workspace,
            &root,
            "models/Image/dense",
            "models/Image/dense-q4-mixed",
            "z-image-q4-mixed-v1",
            Box::new(|| false),
            Box::new(|_message: &str| {}),
        )
        .unwrap();
        assert!(report.quantized > 0, "empty leftover must rebuild: {report:?}");
        assert!(target.join("bundle.manifest.json").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn run_derive_rejects_an_unknown_recipe() {
        use danmo_mlx::Stream;
        let root = scratch("norecipe");
        write_dense(&root.join("dense"), &Stream::cpu().unwrap());
        let error = super::run_derive(
            &root,
            &root,
            "dense",
            "dense-q4-mixed",
            "no-such-recipe",
            Box::new(|| false),
            Box::new(|_message: &str| {}),
        )
        .unwrap_err();
        assert!(error.contains("no-such-recipe is missing"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
