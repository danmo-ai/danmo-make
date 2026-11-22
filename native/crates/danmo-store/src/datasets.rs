//! Filesystem LoRA datasets: DiffSynth-compatible `train.jsonl` + `meta.json`.

use danmo_contracts::{LoraDatasetMeta, LoraDatasetRow};
use serde_json::json;
use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use crate::StoreError;

pub fn datasets_root(project_root: &Path) -> PathBuf {
    project_root.join("datasets")
}

pub fn dataset_dir(project_root: &Path, dataset_id: &str) -> PathBuf {
    datasets_root(project_root).join(dataset_id)
}

fn meta_path(dir: &Path) -> PathBuf {
    dir.join("meta.json")
}

fn jsonl_path(dir: &Path) -> PathBuf {
    dir.join("train.jsonl")
}

fn images_dir(dir: &Path) -> PathBuf {
    dir.join("images")
}

pub fn list_datasets(project_root: &Path) -> Result<Vec<serde_json::Value>, StoreError> {
    let root = datasets_root(project_root);
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut items = Vec::new();
    for entry in fs::read_dir(&root).map_err(|e| StoreError::Message(e.to_string()))? {
        let entry = entry.map_err(|e| StoreError::Message(e.to_string()))?;
        if !entry.path().is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        if let Ok(summary) = summarize_dataset(project_root, &id) {
            items.push(summary);
        }
    }
    items.sort_by(|a, b| {
        b.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .cmp(a.get("id").and_then(|v| v.as_str()).unwrap_or(""))
    });
    Ok(items)
}

fn skipped_from_meta(meta: &serde_json::Value) -> Vec<String> {
    meta.get("skipped_images")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn read_meta_json(dir: &Path) -> Result<serde_json::Value, StoreError> {
    let text =
        fs::read_to_string(meta_path(dir)).map_err(|e| StoreError::Message(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| StoreError::Message(e.to_string()))
}

pub fn summarize_dataset(
    project_root: &Path,
    dataset_id: &str,
) -> Result<serde_json::Value, StoreError> {
    let dir = dataset_dir(project_root, dataset_id);
    let raw = read_meta_json(&dir)?;
    let rows = read_train_jsonl(&jsonl_path(&dir))?;
    let present: std::collections::HashSet<String> = rows.iter().map(|r| r.image.clone()).collect();
    let skipped: Vec<String> = skipped_from_meta(&raw)
        .into_iter()
        .filter(|rel| present.contains(rel))
        .collect();
    let cover = rows
        .iter()
        .find(|r| !r.image.is_empty() && dir.join(&r.image).is_file())
        .map(|r| r.image.clone())
        .unwrap_or_default();
    Ok(json!({
        "id": dataset_id,
        "name": raw.get("name").and_then(|v| v.as_str()).unwrap_or(dataset_id),
        "kind": raw.get("kind").and_then(|v| v.as_str()).unwrap_or("concept"),
        "trigger_word": raw.get("trigger_word").and_then(|v| v.as_str()).unwrap_or(""),
        "default_prompt": raw.get("default_prompt").and_then(|v| v.as_str()).unwrap_or(""),
        "nsfw": raw.get("nsfw").and_then(|v| v.as_bool()).unwrap_or(false),
        "face_crop": raw.get("face_crop").and_then(|v| v.as_str()).unwrap_or("auto"),
        "image_count": rows.len(),
        "training_image_count": rows.len().saturating_sub(skipped.len()),
        "skipped_images": skipped,
        "cover_image": cover,
    }))
}

pub fn get_dataset_detail(
    project_root: &Path,
    dataset_id: &str,
) -> Result<serde_json::Value, StoreError> {
    let dir = dataset_dir(project_root, dataset_id);
    let mut summary = summarize_dataset(project_root, dataset_id)?;
    let rows = read_train_jsonl(&jsonl_path(&dir))?;
    let images: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            json!({
                "file": row.image,
                "prompt": row.prompt,
                "exists": dir.join(&row.image).is_file(),
            })
        })
        .collect();
    if let Some(obj) = summary.as_object_mut() {
        obj.insert("images".into(), json!(images));
    }
    Ok(summary)
}

pub fn load_dataset_meta(
    project_root: &Path,
    dataset_id: &str,
) -> Result<LoraDatasetMeta, StoreError> {
    let path = meta_path(&dataset_dir(project_root, dataset_id));
    if !path.is_file() {
        return Err(StoreError::Message(format!(
            "dataset {dataset_id} not found"
        )));
    }
    let text = fs::read_to_string(&path).map_err(|e| StoreError::Message(e.to_string()))?;
    let mut meta: LoraDatasetMeta =
        serde_json::from_str(&text).map_err(|e| StoreError::Message(e.to_string()))?;
    if meta.id != dataset_id {
        return Err(StoreError::Message(format!(
            "dataset meta id {:?} does not match directory {dataset_id}",
            meta.id
        )));
    }
    let rows = read_train_jsonl(&jsonl_path(&dataset_dir(project_root, dataset_id)))?;
    meta.image_count = rows.len() as u32;
    Ok(meta)
}

pub fn create_dataset(
    project_root: &Path,
    name: &str,
    kind: &str,
    trigger_word: &str,
) -> Result<LoraDatasetMeta, StoreError> {
    let kind = kind.trim().to_ascii_lowercase();
    if kind != "concept" && kind != "style" {
        return Err(StoreError::Message(format!(
            "dataset kind must be concept|style, got {kind:?}"
        )));
    }
    let id = format!("ds_{}", &uuid_like()[..12]);
    let dir = dataset_dir(project_root, &id);
    fs::create_dir_all(images_dir(&dir)).map_err(|e| StoreError::Message(e.to_string()))?;
    let meta = LoraDatasetMeta {
        id: id.clone(),
        name: name.trim().to_string(),
        kind,
        trigger_word: trigger_word.trim().to_string(),
        default_prompt: String::new(),
        nsfw: false,
        face_crop: "auto".into(),
        image_count: 0,
    };
    write_meta(&dir, &meta)?;
    write_train_jsonl(&jsonl_path(&dir), &[])?;
    Ok(meta)
}

pub fn update_dataset_meta(
    project_root: &Path,
    dataset_id: &str,
    name: Option<&str>,
    kind: Option<&str>,
    trigger_word: Option<&str>,
    default_prompt: Option<&str>,
    nsfw: Option<bool>,
    face_crop: Option<&str>,
) -> Result<LoraDatasetMeta, StoreError> {
    let mut meta = load_dataset_meta(project_root, dataset_id)?;
    if let Some(name) = name {
        meta.name = name.trim().to_string();
    }
    if let Some(kind) = kind {
        let kind = kind.trim().to_ascii_lowercase();
        if kind != "concept" && kind != "style" {
            return Err(StoreError::Message(format!(
                "dataset kind must be concept|style, got {kind:?}"
            )));
        }
        meta.kind = kind;
    }
    if let Some(trigger_word) = trigger_word {
        meta.trigger_word = trigger_word.trim().to_string();
    }
    if let Some(default_prompt) = default_prompt {
        meta.default_prompt = default_prompt.to_string();
    }
    if let Some(nsfw) = nsfw {
        meta.nsfw = nsfw;
    }
    if let Some(face_crop) = face_crop {
        meta.face_crop = face_crop.trim().to_string();
    }
    write_meta(&dataset_dir(project_root, dataset_id), &meta)?;
    Ok(meta)
}

pub fn delete_dataset(project_root: &Path, dataset_id: &str) -> Result<(), StoreError> {
    let dir = dataset_dir(project_root, dataset_id);
    if !dir.is_dir() {
        return Err(StoreError::Message(format!(
            "dataset {dataset_id} not found"
        )));
    }
    fs::remove_dir_all(&dir).map_err(|e| StoreError::Message(e.to_string()))
}

pub fn read_dataset_rows(
    project_root: &Path,
    dataset_id: &str,
) -> Result<Vec<LoraDatasetRow>, StoreError> {
    let _ = load_dataset_meta(project_root, dataset_id)?;
    read_train_jsonl(&jsonl_path(&dataset_dir(project_root, dataset_id)))
}

/// Training pairs with caption policy applied.
///
/// - `concept`: every image uses `unified_caption` (must contain trigger when set).
/// - `style`: per-row prompts; empty row falls back to `unified_caption`.
/// - Rows listed in `meta.skipped_images` are excluded.
pub fn load_training_pairs(
    project_root: &Path,
    dataset_id: &str,
    unified_caption: &str,
) -> Result<(LoraDatasetMeta, Vec<(PathBuf, String)>), StoreError> {
    let meta = load_dataset_meta(project_root, dataset_id)?;
    let dir = dataset_dir(project_root, dataset_id);
    let raw = read_meta_json(&dir)?;
    let skipped: std::collections::HashSet<String> = skipped_from_meta(&raw).into_iter().collect();
    let rows = read_dataset_rows(project_root, dataset_id)?;
    let active: Vec<_> = rows
        .into_iter()
        .filter(|r| !skipped.contains(&r.image))
        .collect();
    if active.len() < 3 {
        return Err(StoreError::Message(format!(
            "Dataset must contain at least 3 training images (have {})",
            active.len()
        )));
    }
    let caption = unified_caption.trim();
    if caption.is_empty() {
        return Err(StoreError::Message(
            "dataset default_prompt (unified caption) is required".into(),
        ));
    }
    if meta.kind == "concept" {
        let trigger = meta.trigger_word.trim();
        if !trigger.is_empty()
            && !caption
                .to_ascii_lowercase()
                .contains(&trigger.to_ascii_lowercase())
        {
            return Err(StoreError::Message(format!(
                "concept caption must contain trigger_word {trigger:?}"
            )));
        }
    }
    let mut pairs = Vec::with_capacity(active.len());
    for row in active {
        let rel = row.image.trim();
        if rel.is_empty() {
            return Err(StoreError::Message(
                "train.jsonl row missing image path".into(),
            ));
        }
        let path = dir.join(rel);
        if !path.is_file() {
            return Err(StoreError::Message(format!("dataset image missing: {rel}")));
        }
        let prompt = if meta.kind == "concept" {
            caption.to_string()
        } else {
            let row_prompt = row.prompt.trim();
            if row_prompt.is_empty() {
                caption.to_string()
            } else {
                row_prompt.to_string()
            }
        };
        pairs.push((path, prompt));
    }
    Ok((meta, pairs))
}

/// Import DiffSynth `metadata.csv` (`image,prompt`) into `train.jsonl`.
pub fn import_metadata_csv(
    project_root: &Path,
    dataset_id: &str,
    csv_text: &str,
) -> Result<LoraDatasetMeta, StoreError> {
    let _ = load_dataset_meta(project_root, dataset_id)?;
    let mut rows = Vec::new();
    for (index, line) in csv_text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if index == 0 && line.to_ascii_lowercase().starts_with("image") {
            continue;
        }
        let (image, prompt) = split_csv_image_prompt(line).ok_or_else(|| {
            StoreError::Message(format!("invalid metadata.csv line {}: {line}", index + 1))
        })?;
        rows.push(LoraDatasetRow { image, prompt });
    }
    if rows.is_empty() {
        return Err(StoreError::Message(
            "metadata.csv produced no image rows".into(),
        ));
    }
    let dir = dataset_dir(project_root, dataset_id);
    write_train_jsonl(&jsonl_path(&dir), &rows)?;
    load_dataset_meta(project_root, dataset_id)
}

pub fn append_image_row(
    project_root: &Path,
    dataset_id: &str,
    relative_image: &str,
    prompt: &str,
) -> Result<LoraDatasetMeta, StoreError> {
    let dir = dataset_dir(project_root, dataset_id);
    let mut rows = read_train_jsonl(&jsonl_path(&dir))?;
    rows.push(LoraDatasetRow {
        image: relative_image.trim().to_string(),
        prompt: prompt.to_string(),
    });
    write_train_jsonl(&jsonl_path(&dir), &rows)?;
    load_dataset_meta(project_root, dataset_id)
}

/// Save uploaded bytes under `images/` and append a train.jsonl row.
pub fn add_image_bytes(
    project_root: &Path,
    dataset_id: &str,
    filename: &str,
    bytes: &[u8],
    prompt: &str,
) -> Result<serde_json::Value, StoreError> {
    let _ = load_dataset_meta(project_root, dataset_id)?;
    let dir = dataset_dir(project_root, dataset_id);
    let images = images_dir(&dir);
    fs::create_dir_all(&images).map_err(|e| StoreError::Message(e.to_string()))?;
    let safe = sanitize_filename(filename);
    let stem = Path::new(&safe)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("img");
    let ext = Path::new(&safe)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("jpg");
    let unique = format!("{stem}_{}.{}", &uuid_like()[..8], ext);
    let rel = format!("images/{unique}");
    let dest = dir.join(&rel);
    fs::write(&dest, bytes).map_err(|e| StoreError::Message(e.to_string()))?;
    let mut rows = read_train_jsonl(&jsonl_path(&dir))?;
    rows.push(LoraDatasetRow {
        image: rel,
        prompt: prompt.to_string(),
    });
    write_train_jsonl(&jsonl_path(&dir), &rows)?;
    get_dataset_detail(project_root, dataset_id)
}

pub fn delete_dataset_image(
    project_root: &Path,
    dataset_id: &str,
    relative: &str,
) -> Result<serde_json::Value, StoreError> {
    let dir = dataset_dir(project_root, dataset_id);
    let rel = relative.trim().trim_start_matches('/');
    if rel.is_empty() || rel.contains("..") {
        return Err(StoreError::Message("invalid image path".into()));
    }
    let mut rows = read_train_jsonl(&jsonl_path(&dir))?;
    let before = rows.len();
    rows.retain(|r| r.image != rel);
    if rows.len() == before {
        return Err(StoreError::Message(format!("image {rel} not in dataset")));
    }
    write_train_jsonl(&jsonl_path(&dir), &rows)?;
    let path = dir.join(rel);
    if path.is_file() {
        let _ = fs::remove_file(&path);
    }
    // Drop from skipped list if present.
    let mut raw = read_meta_json(&dir)?;
    if let Some(arr) = raw.get_mut("skipped_images").and_then(|v| v.as_array_mut()) {
        arr.retain(|v| v.as_str() != Some(rel));
    }
    fs::write(
        meta_path(&dir),
        serde_json::to_string_pretty(&raw).map_err(|e| StoreError::Message(e.to_string()))?,
    )
    .map_err(|e| StoreError::Message(e.to_string()))?;
    get_dataset_detail(project_root, dataset_id)
}

pub fn set_skipped_images(
    project_root: &Path,
    dataset_id: &str,
    files: &[String],
) -> Result<serde_json::Value, StoreError> {
    let dir = dataset_dir(project_root, dataset_id);
    let rows = read_train_jsonl(&jsonl_path(&dir))?;
    let present: std::collections::HashSet<&str> = rows.iter().map(|r| r.image.as_str()).collect();
    let skipped: Vec<String> = files
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && present.contains(s.as_str()))
        .collect();
    let mut raw = read_meta_json(&dir)?;
    if let Some(obj) = raw.as_object_mut() {
        obj.insert("skipped_images".into(), json!(skipped));
    }
    fs::write(
        meta_path(&dir),
        serde_json::to_string_pretty(&raw).map_err(|e| StoreError::Message(e.to_string()))?,
    )
    .map_err(|e| StoreError::Message(e.to_string()))?;
    get_dataset_detail(project_root, dataset_id)
}

pub fn import_asset_files(
    project_root: &Path,
    dataset_id: &str,
    source_paths: &[(PathBuf, String)],
    default_prompt: &str,
) -> Result<serde_json::Value, StoreError> {
    if source_paths.is_empty() {
        return Err(StoreError::Message("no assets to import".into()));
    }
    for (src, name) in source_paths {
        if !src.is_file() {
            return Err(StoreError::Message(format!(
                "asset file missing: {}",
                src.display()
            )));
        }
        let bytes = fs::read(src).map_err(|e| StoreError::Message(e.to_string()))?;
        let _ = add_image_bytes(project_root, dataset_id, name, &bytes, default_prompt)?;
    }
    get_dataset_detail(project_root, dataset_id)
}

/// Structural health (no face detector / VLM yet). Enough for the UI gate.
pub fn dataset_health_report(
    project_root: &Path,
    dataset_id: &str,
) -> Result<serde_json::Value, StoreError> {
    let detail = get_dataset_detail(project_root, dataset_id)?;
    let image_count = detail
        .get("training_image_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = detail
        .get("image_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let missing = detail
        .get("images")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|img| img.get("exists") == Some(&json!(false)))
                .count()
        })
        .unwrap_or(0);
    let mut hints = Vec::new();
    let mut score = 80i64;
    let level = if image_count < 3 {
        score = 20;
        hints.push(json!({
            "code": "too_few_images",
            "level": "error",
            "message": format!("Need at least 3 training images (have {image_count})"),
        }));
        "error"
    } else if missing > 0 {
        score = 40;
        hints.push(json!({
            "code": "missing_files",
            "level": "error",
            "message": format!("{missing} listed image(s) are missing on disk"),
        }));
        "error"
    } else if image_count < 8 {
        hints.push(json!({
            "code": "small_set",
            "level": "warn",
            "message": "Small dataset; 8–20 varied portraits usually lock identity better",
        }));
        "warn"
    } else {
        "ok"
    };
    Ok(json!({
        "level": level,
        "score": score,
        "kind": detail.get("kind").cloned().unwrap_or(json!("concept")),
        "stats": {
            "image_count": total,
            "training_image_count": image_count,
            "missing_files": missing,
        },
        "hints": hints,
        "faces": { "available": false, "rows": [] },
        "vision_available": false,
        "vlm_audited": false,
    }))
}

fn sanitize_filename(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("image.jpg");
    base.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn split_csv_image_prompt(line: &str) -> Option<(String, String)> {
    if let Some((image, prompt)) = line.split_once(',') {
        Some((image.trim().to_string(), prompt.trim().to_string()))
    } else {
        None
    }
}

fn read_train_jsonl(path: &Path) -> Result<Vec<LoraDatasetRow>, StoreError> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path).map_err(|e| StoreError::Message(e.to_string()))?;
    let mut rows = Vec::new();
    for line in std::io::BufReader::new(file).lines() {
        let line = line.map_err(|e| StoreError::Message(e.to_string()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: LoraDatasetRow =
            serde_json::from_str(line).map_err(|e| StoreError::Message(e.to_string()))?;
        rows.push(row);
    }
    Ok(rows)
}

fn write_train_jsonl(path: &Path, rows: &[LoraDatasetRow]) -> Result<(), StoreError> {
    let mut file = fs::File::create(path).map_err(|e| StoreError::Message(e.to_string()))?;
    for row in rows {
        let line = serde_json::to_string(row).map_err(|e| StoreError::Message(e.to_string()))?;
        writeln!(file, "{line}").map_err(|e| StoreError::Message(e.to_string()))?;
    }
    Ok(())
}

fn write_meta(dir: &Path, meta: &LoraDatasetMeta) -> Result<(), StoreError> {
    let skipped = read_meta_json(dir)
        .ok()
        .map(|raw| skipped_from_meta(&raw))
        .unwrap_or_default();
    let value = json!({
        "id": meta.id,
        "name": meta.name,
        "kind": meta.kind,
        "trigger_word": meta.trigger_word,
        "default_prompt": meta.default_prompt,
        "nsfw": meta.nsfw,
        "face_crop": meta.face_crop,
        "skipped_images": skipped,
    });
    fs::write(
        meta_path(dir),
        serde_json::to_string_pretty(&value).map_err(|e| StoreError::Message(e.to_string()))?,
    )
    .map_err(|e| StoreError::Message(e.to_string()))
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn concept_dataset_roundtrip_and_unified_caption() {
        let root = env::temp_dir().join(format!("danmo-ds-{}", uuid_like()));
        let meta = create_dataset(&root, "face", "concept", "ohwx").unwrap();
        let dir = dataset_dir(&root, &meta.id);
        let img = images_dir(&dir).join("a.jpg");
        fs::write(&img, b"fake").unwrap();
        fs::write(images_dir(&dir).join("b.jpg"), b"fake").unwrap();
        fs::write(images_dir(&dir).join("c.jpg"), b"fake").unwrap();
        write_train_jsonl(
            &jsonl_path(&dir),
            &[
                LoraDatasetRow {
                    image: "images/a.jpg".into(),
                    prompt: "ignored".into(),
                },
                LoraDatasetRow {
                    image: "images/b.jpg".into(),
                    prompt: "".into(),
                },
                LoraDatasetRow {
                    image: "images/c.jpg".into(),
                    prompt: "".into(),
                },
            ],
        )
        .unwrap();
        let err = load_training_pairs(&root, &meta.id, "a person").unwrap_err();
        assert!(err.to_string().contains("trigger_word"));
        let (_meta, pairs) = load_training_pairs(&root, &meta.id, "ohwx person smiling").unwrap();
        assert_eq!(pairs.len(), 3);
        assert!(pairs.iter().all(|(_, p)| p == "ohwx person smiling"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn imports_diffsynth_metadata_csv() {
        let root = env::temp_dir().join(format!("danmo-csv-{}", uuid_like()));
        let meta = create_dataset(&root, "style", "style", "").unwrap();
        super::import_metadata_csv(
            &root,
            &meta.id,
            "image,prompt\nimages/a.png,red car\nimages/b.png,blue car\n",
        )
        .unwrap();
        let rows = read_dataset_rows(&root, &meta.id).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].prompt, "red car");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn upload_skip_and_detail_shape() {
        let root = env::temp_dir().join(format!("danmo-up-{}", uuid_like()));
        let meta = create_dataset(&root, "face", "concept", "ohwx").unwrap();
        let _ = add_image_bytes(&root, &meta.id, "a.jpg", b"fake-a", "").unwrap();
        let _ = add_image_bytes(&root, &meta.id, "b.jpg", b"fake-b", "").unwrap();
        let detail = add_image_bytes(&root, &meta.id, "c.jpg", b"fake-c", "").unwrap();
        assert_eq!(detail["image_count"], 3);
        assert!(detail["images"].as_array().unwrap().len() == 3);
        assert!(!detail["cover_image"].as_str().unwrap().is_empty());
        let first = detail["images"][0]["file"].as_str().unwrap().to_string();
        let skipped = set_skipped_images(&root, &meta.id, &[first.clone()]).unwrap();
        assert_eq!(skipped["training_image_count"], 2);
        let health = dataset_health_report(&root, &meta.id).unwrap();
        assert_eq!(health["level"], "error"); // <3 active training images
        let _ = set_skipped_images(&root, &meta.id, &[]).unwrap();
        let health_ok = dataset_health_report(&root, &meta.id).unwrap();
        assert_ne!(health_ok["level"], "error");
        let _ = fs::remove_dir_all(&root);
    }
}
