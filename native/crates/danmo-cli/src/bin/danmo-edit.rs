//! danmo-edit — 图像编辑 CLI（对应 POST /api/images/edits）。
//!
//! 用法:
//!   danmo-edit --model flux2-klein-4b --operation rewrite --source-image in.png \
//!       --prompt "make the apple green" --source-fidelity 0.35 --output out.png
//!   danmo-edit --model z-image-turbo --operation rewrite --source-image in.png \
//!       --prompt "..." --lora user-lora-abc:0.9 --output out.png

use serde_json::json;

fn main() {
    std::process::exit(match run() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("[cli] error: {error}");
            1
        }
    });
}

fn run() -> Result<(), String> {
    let args = danmo_cli::Args::parse_env()?;
    if args.has("help") {
        println!("{}", USAGE);
        return Ok(());
    }
    let host = danmo_cli::Host::new(&args)?;
    let model = args.require("model")?;
    let mut request = serde_json::Map::new();
    request.insert("model".into(), json!(model));
    request.insert("operation".into(), json!(args.require("operation")?));
    request.insert(
        "source_asset_id".into(),
        json!(args.require("source-image")?),
    );
    request.insert("prompt".into(), json!(args.require("prompt")?));
    request.insert(
        "negative_prompt".into(),
        json!(args.get("negative-prompt").unwrap_or("")),
    );
    if let Some(mask) = args.get("mask-image") {
        request.insert("mask_asset_id".into(), json!(mask));
    }
    let references: Vec<&str> = args.all("reference-image");
    if !references.is_empty() {
        request.insert("reference_asset_ids".into(), json!(references));
    }
    danmo_cli::apply_options(
        &mut request,
        &args,
        &[
            "source-fidelity",
            "size",
            "steps",
            "guidance",
            "seed",
            "scheduler",
        ],
    )?;
    if let Some(metadata) = args.get("metadata") {
        let metadata: serde_json::Value =
            serde_json::from_str(metadata).map_err(|error| format!("--metadata: {error}"))?;
        if !metadata.is_object() {
            return Err("--metadata must be a JSON object".into());
        }
        request.insert("metadata".into(), metadata);
    }
    let loras = danmo_cli::collect_loras(&host, &args)?;
    if !loras.is_empty() {
        request.insert("adapters".into(), json!(loras));
    }
    danmo_cli::run_task(&host, "image.edit", &model, json!(request))
}

const USAGE: &str =
    "danmo-edit --model ID[:version] --operation OP --source-image PATH --prompt TEXT --output FILE
  OP: rewrite | retouch | extend (family-dependent)
  [--mask-image PATH] [--reference-image PATH]... [--source-fidelity F]
  [--size WxH] [--steps N] [--guidance F] [--seed N] [--scheduler ID]
  [--metadata JSON] [--lora ID|PATH[:WEIGHT]]... [--workspace DIR] [--registry FILE]
  --metadata keys already set are kept; the model card fills the rest.";
