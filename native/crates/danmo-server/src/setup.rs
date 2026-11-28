//! Hardware-aware model recommendations for first-run onboarding and Quick Setup.
//!
//! Everything here is driven by the factory registry (`catalog.media`,
//! `catalog.category`, `catalog.recommended`, `distribution.versions.*.size`) —
//! no family or engine ids are special-cased. A slot with no compatible model
//! reports an explicit `reason` instead of silently dropping out of the list.

use danmo_catalog::Catalog;
use serde_json::{json, Value};

/// Below this reference memory the image slot prefers the smaller base model.
const MEMORY_TIER_LOW: f64 = 16.0;
/// Above this reference memory the tier is `high`.
const MEMORY_TIER_MID: f64 = 32.0;
/// Share of reference memory a recommended download may occupy.
const MEMORY_BUDGET_RATIO: f64 = 0.70;
/// Used when the host memory probe fails; reported as `memory_detected: false`.
const UNKNOWN_MEMORY_GB: f64 = 16.0;

/// Version preference when several fit the memory budget. `q4-mixed` is the
/// locally built quantization, so it wins the low-memory tier.
const VERSION_PREF: &[&str] = &[
    "q4-mixed", "mlx-q4", "int4", "mlx-q8", "int8", "fp16", "bf16",
];

struct SlotSpec {
    key: &'static str,
    media: &'static str,
    category: &'static str,
    action: &'static str,
}

const SLOTS: &[SlotSpec] = &[
    SlotSpec {
        key: "image",
        media: "image",
        category: "base_models",
        action: "create",
    },
    SlotSpec {
        key: "video",
        media: "video",
        category: "video_models",
        action: "create",
    },
    SlotSpec {
        key: "audio",
        media: "audio",
        category: "music_models",
        action: "create",
    },
    SlotSpec {
        key: "llm",
        media: "llm",
        category: "llm_models",
        action: "chat",
    },
    SlotSpec {
        key: "vlm",
        media: "llm",
        category: "vlm_models",
        action: "describe",
    },
];

/// Backends this build can actually run. Registry backends are the `mlx` family;
/// a build without the feature must report slots as unavailable, not as empty.
pub fn available_backends() -> Vec<String> {
    if cfg!(feature = "mlx") {
        vec!["mlx".to_string()]
    } else {
        Vec::new()
    }
}

/// `hw.memsize`-backed total memory on macOS, `/proc/meminfo` elsewhere.
pub fn host_memory() -> (f64, f64) {
    #[cfg(target_os = "macos")]
    {
        return macos_memory();
    }
    #[cfg(not(target_os = "macos"))]
    {
        proc_memory()
    }
}

#[cfg(target_os = "macos")]
fn macos_memory() -> (f64, f64) {
    let mut size: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = b"hw.memsize\0";
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const libc::c_char,
            &mut size as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || size == 0 {
        tracing::warn!(
            "[setup] hw.memsize probe failed (rc={rc}); reporting memory_detected=false"
        );
        return (0.0, 0.0);
    }
    // macOS exposes no cheap "available" counter here; 0 means unknown.
    (bytes_to_gb(size), 0.0)
}

#[cfg(not(target_os = "macos"))]
fn proc_memory() -> (f64, f64) {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        tracing::warn!("[setup] /proc/meminfo unreadable; reporting memory_detected=false");
        return (0.0, 0.0);
    };
    let mut total = 0f64;
    let mut avail = 0f64;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = kib_to_gb(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail = kib_to_gb(rest);
        }
    }
    (total, avail)
}

#[cfg(not(target_os = "macos"))]
fn kib_to_gb(text: &str) -> f64 {
    bytes_to_gb(
        text.split_whitespace()
            .next()
            .and_then(|n| n.parse::<u64>().ok())
            .unwrap_or(0)
            * 1024,
    )
}

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0 / 1024.0
}
fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// Parse `"4.3GB"` / `"约 71GB 源文件 → 约 25GB 转换后"` / `"512MB"` to GB.
pub fn parse_human_size_to_gb(value: Option<&Value>) -> Option<f64> {
    let raw = match value {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Number(number)) => number.to_string(),
        _ => return None,
    };
    let cleaned: String = raw
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| !matches!(c, ',' | '~' | '≈' | ' '))
        .collect();
    let start = cleaned.find(|c: char| c.is_ascii_digit())?;
    let rest = &cleaned[start..];
    let digits = rest
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>();
    let amount: f64 = digits.parse().ok()?;
    if !(amount > 0.0) {
        return None;
    }
    let unit = rest[digits.len()..].trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
    if unit.starts_with("tb") || unit.starts_with('t') {
        Some(amount * 1024.0)
    } else if unit.starts_with("mb") || unit.starts_with('m') {
        Some(amount / 1024.0)
    } else {
        Some(amount)
    }
}

fn memory_tier(reference_gb: f64) -> &'static str {
    if reference_gb <= MEMORY_TIER_LOW {
        "low"
    } else if reference_gb <= MEMORY_TIER_MID {
        "mid"
    } else {
        "high"
    }
}

fn preferred_ids(slot: &str, tier: &str) -> &'static [&'static str] {
    match (slot, tier) {
        ("image", "low") => &["flux2-klein-4b", "z-image-turbo"],
        ("image", _) => &["z-image-turbo", "flux2-klein-4b"],
        ("video", _) => &["wan-2.2-ti2v-5b"],
        ("audio", _) => &["ace-step-xl-sft"],
        ("llm", _) => &["qwen3.5-4b"],
        ("vlm", _) => &["qwen3-vl-4b-instruct"],
        _ => &[],
    }
}

fn model_backends(entry: &Value) -> Vec<String> {
    let rows = entry
        .pointer("/runtime/backends")
        .or_else(|| entry.get("backends"))
        .and_then(|value| value.as_array());
    match rows {
        Some(rows) => rows
            .iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect(),
        // No declared backends means the MLX runtime, matching the registry default.
        None => vec!["mlx".to_string()],
    }
}

fn slot_candidates<'a>(
    models: &'a serde_json::Map<String, Value>,
    spec: &SlotSpec,
    backends: &[String],
) -> Vec<(&'a String, &'a Value)> {
    let mut out: Vec<(&String, &Value)> = models
        .iter()
        .filter(|(_, entry)| {
            let media = entry.pointer("/catalog/media").and_then(|v| v.as_str());
            let category = entry.pointer("/catalog/category").and_then(|v| v.as_str());
            if media != Some(spec.media) || category != Some(spec.category) {
                return false;
            }
            if entry
                .pointer("/catalog/recommended")
                .and_then(|v| v.as_bool())
                != Some(true)
            {
                return false;
            }
            if entry
                .pointer("/catalog/commercial_use_allowed")
                .and_then(|v| v.as_bool())
                != Some(true)
            {
                return false;
            }
            if entry
                .get("actions")
                .and_then(|actions| actions.get(spec.action))
                .is_none()
            {
                return false;
            }
            model_backends(entry)
                .iter()
                .any(|backend| backends.iter().any(|want| want == backend))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(b.0));
    out
}

fn version_estimated_gb(version: &Value) -> f64 {
    let size_gb = parse_human_size_to_gb(version.get("size")).unwrap_or(0.0);
    let min_gb = version
        .get("min_unified_memory_gb")
        .and_then(|value| value.as_f64())
        .filter(|value| *value > 0.0)
        .unwrap_or(0.0);
    size_gb.max(min_gb)
}

fn version_rank(key: &str) -> usize {
    VERSION_PREF
        .iter()
        .position(|preferred| *preferred == key)
        .unwrap_or(VERSION_PREF.len())
}

/// Pick the largest version that fits 70% of reference memory, preferring the
/// MLX quantized keys on ties. If nothing fits, return the smallest with a warning.
fn pick_version(
    versions: &serde_json::Map<String, Value>,
    ref_gb: f64,
) -> Option<(String, Value, Option<&'static str>)> {
    let budget = ref_gb * MEMORY_BUDGET_RATIO;
    let mut ranked: Vec<(String, Value, f64)> = versions
        .iter()
        .filter(|(_, version)| version.is_object())
        .map(|(key, version)| (key.clone(), version.clone(), version_estimated_gb(version)))
        .collect();
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|a, b| {
        version_rank(&a.0)
            .cmp(&version_rank(&b.0))
            .then_with(|| b.2.total_cmp(&a.2))
            .then_with(|| a.0.cmp(&b.0))
    });
    let fitting: Vec<&(String, Value, f64)> = ranked
        .iter()
        .filter(|row| row.2 <= 0.0 || row.2 <= budget)
        .collect();
    if let Some(best) = fitting.iter().max_by(|a, b| a.2.total_cmp(&b.2)) {
        return Some((best.0.clone(), best.1.clone(), None));
    }
    let smallest = ranked
        .iter()
        .filter(|row| row.2 > 0.0)
        .min_by(|a, b| a.2.total_cmp(&b.2))?;
    Some((
        smallest.0.clone(),
        smallest.1.clone(),
        Some("insufficient_memory"),
    ))
}

fn slot_recommendation(
    models: &serde_json::Map<String, Value>,
    spec: &SlotSpec,
    ref_gb: f64,
    tier: &str,
    backends: &[String],
) -> Value {
    let unavailable = |reason: &str| {
        json!({
            "slot": spec.key,
            "status": "unavailable",
            "model_id": Value::Null,
            "version_key": Value::Null,
            "estimated_gb": Value::Null,
            "warning": Value::Null,
            "reason": reason,
            "installed": false,
            "name": {},
            "version_name": {},
            "size_human": "",
        })
    };

    let candidates = slot_candidates(models, spec, backends);
    if candidates.is_empty() {
        if !backends.iter().any(|backend| backend == "mlx") {
            return unavailable("mlx_required");
        }
        return unavailable("no_compatible_models");
    }

    let preferred = preferred_ids(spec.key, tier);
    let mut ordered: Vec<(&String, &Value)> = Vec::new();
    for id in preferred {
        if let Some(entry) = candidates.iter().find(|(key, _)| key.as_str() == *id) {
            ordered.push(*entry);
        }
    }
    for entry in &candidates {
        if !preferred.iter().any(|id| *id == entry.0.as_str()) {
            ordered.push(*entry);
        }
    }

    let (model_id, entry) = ordered[0];
    let Some(versions) = entry
        .pointer("/distribution/versions")
        .and_then(|value| value.as_object())
    else {
        return unavailable("no_versions");
    };
    let Some((version_key, version, warning)) = pick_version(versions, ref_gb) else {
        return unavailable("no_versions");
    };

    let estimated = version_estimated_gb(&version);
    json!({
        "slot": spec.key,
        "status": if warning.is_some() { "warning" } else { "recommended" },
        "model_id": model_id,
        "version_key": version_key,
        "estimated_gb": if estimated > 0.0 { json!(round2(estimated)) } else { Value::Null },
        "warning": warning,
        "reason": Value::Null,
        "installed": false,
        "name": entry.pointer("/catalog/name").cloned().unwrap_or_else(|| json!({})),
        "version_name": version.get("name").cloned().unwrap_or_else(|| json!({})),
        "size_human": version.get("size").and_then(|v| v.as_str()).unwrap_or(""),
    })
}

fn reference_memory_gb(memory_gb: f64, mlx_memory_limit: f64) -> f64 {
    let cap = if mlx_memory_limit > 0.0 {
        mlx_memory_limit
    } else {
        memory_gb
    };
    if memory_gb > 0.0 && cap > 0.0 {
        memory_gb.min(cap)
    } else if memory_gb > 0.0 {
        memory_gb
    } else if cap > 0.0 {
        cap
    } else {
        UNKNOWN_MEMORY_GB
    }
}

/// Full onboarding / Quick Setup payload.
///
/// `readiness` maps a model id to "every published version is on disk"; it is
/// supplied by the caller so this module stays free of workspace I/O.
pub fn recommendations(
    catalog: &Catalog,
    memory_gb: f64,
    memory_available_gb: f64,
    mlx_memory_limit: f64,
    readiness: &dyn Fn(&str) -> bool,
) -> Value {
    let backends = available_backends();
    let primary_backend = backends.first().cloned().unwrap_or_default();
    let ref_gb = reference_memory_gb(memory_gb, mlx_memory_limit);
    let tier = memory_tier(ref_gb);

    let models = catalog
        .root()
        .get("models")
        .and_then(|value| value.as_object())
        .cloned()
        .unwrap_or_default();

    let mut slots = Vec::with_capacity(SLOTS.len());
    for spec in SLOTS {
        let mut row = slot_recommendation(&models, spec, ref_gb, tier, &backends);
        if let Some(model_id) = row.get("model_id").and_then(|v| v.as_str()) {
            let installed = readiness(model_id);
            if let Some(obj) = row.as_object_mut() {
                obj.insert("installed".into(), json!(installed));
            }
        }
        slots.push(row);
    }

    json!({
        "runtime": if cfg!(feature = "mlx") { "mlx" } else { "unavailable" },
        "weights": "safetensors",
        "versions": ["fp16", "bf16", "mlx-bf16", "mlx-q8", "mlx-q4"],
        "reference_memory_gb": round2(ref_gb),
        "memory_tier": tier,
        "memory_detected": memory_gb > 0.0,
        "available_backends": backends,
        "primary_backend": primary_backend,
        "system": {
            "platform": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "memory_gb": round2(memory_gb),
            "memory_available_gb": round2(memory_available_gb),
            "mlx_memory_limit": mlx_memory_limit,
        },
        "slots": slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn factory_catalog() -> Catalog {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../default_config/models_registry.json");
        Catalog::open(&path).expect("factory registry")
    }

    fn slot<'a>(payload: &'a Value, key: &str) -> &'a Value {
        payload["slots"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["slot"] == key)
            .expect("slot present")
    }

    #[test]
    fn parses_registry_size_strings() {
        assert_eq!(parse_human_size_to_gb(Some(&json!("4GB"))), Some(4.0));
        assert_eq!(parse_human_size_to_gb(Some(&json!("31.4GB"))), Some(31.4));
        let mb = parse_human_size_to_gb(Some(&json!("512MB"))).unwrap();
        assert!((mb - 0.5).abs() < 1e-9, "{mb}");
        let prose =
            parse_human_size_to_gb(Some(&json!("约 71GB 源文件 → 约 25GB 转换后"))).unwrap();
        assert!((prose - 71.0).abs() < 1e-9, "{prose}");
        assert_eq!(parse_human_size_to_gb(Some(&json!(""))), None);
        assert_eq!(parse_human_size_to_gb(None), None);
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn low_memory_picks_a_quantized_image_version() {
        let catalog = factory_catalog();
        let payload = recommendations(&catalog, 16.0, 8.0, 16.0, &|_| false);
        let image = slot(&payload, "image");
        assert_eq!(payload["memory_tier"], "low");
        assert!(image["model_id"].is_string(), "{image}");
        let version = image["version_key"].as_str().unwrap();
        assert!(
            ["q4-mixed", "mlx-q4", "mlx-q8", "int4", "int8"].contains(&version),
            "{version}"
        );
    }

    #[cfg(feature = "mlx")]
    #[test]
    fn high_memory_recommends_a_published_image_base_model() {
        let catalog = factory_catalog();
        let payload = recommendations(&catalog, 96.0, 64.0, 0.0, &|_| false);
        let image = slot(&payload, "image");
        assert_eq!(payload["memory_tier"], "high");
        let model = image["model_id"].as_str().unwrap();
        assert!(
            model == "z-image-turbo" || model == "flux2-klein-4b",
            "{model}"
        );
        assert!(image["version_key"].is_string());
    }

    #[cfg(not(feature = "mlx"))]
    #[test]
    fn without_mlx_image_slot_is_explicitly_unavailable() {
        let catalog = factory_catalog();
        let payload = recommendations(&catalog, 64.0, 32.0, 0.0, &|_| false);
        let image = slot(&payload, "image");
        assert_eq!(image["status"], "unavailable");
        assert_eq!(image["reason"], "mlx_required");
        assert!(image["model_id"].is_null());
    }

    #[test]
    fn every_slot_reports_a_model_or_an_explicit_reason() {
        let catalog = factory_catalog();
        let payload = recommendations(&catalog, 64.0, 32.0, 0.0, &|_| false);
        for row in payload["slots"].as_array().unwrap() {
            let has_model = row["model_id"].is_string();
            let has_reason = row["reason"].is_string();
            assert!(has_model ^ has_reason, "slot {row}");
        }
    }

    #[test]
    fn installed_flag_follows_the_readiness_probe() {
        let catalog = factory_catalog();
        let payload = recommendations(&catalog, 64.0, 32.0, 0.0, &|id| id == "z-image-turbo");
        let image = slot(&payload, "image");
        if image["model_id"] == "z-image-turbo" {
            assert_eq!(image["installed"], true);
        }
        let video = slot(&payload, "video");
        assert_eq!(video["installed"], false);
    }

    #[test]
    fn mlx_cap_drives_the_reference_budget() {
        assert_eq!(reference_memory_gb(128.0, 24.0), 24.0);
        assert_eq!(reference_memory_gb(128.0, 0.0), 128.0);
        assert_eq!(reference_memory_gb(0.0, 0.0), UNKNOWN_MEMORY_GB);
    }
}
