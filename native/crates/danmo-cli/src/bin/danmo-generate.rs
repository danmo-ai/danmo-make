//! danmo-generate — 文生图 CLI（对应 POST /api/images/generations）。
//!
//! 用法:
//!   danmo-generate --model z-image-turbo --prompt "a mountain" --output out.png
//!   danmo-generate --model flux2-klein-4b --prompt "a cat" --steps 4 --seed 1 --output cat.png
//!   danmo-generate --model z-image-turbo --prompt "..." \
//!       --lora user-lora-abc:0.9 --controlnet z-image-turbo-fun-controlnet-union \
//!       --control-image ref.png --controlnet-strength 0.8 --output out.png

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
    request.insert("prompt".into(), json!(args.require("prompt")?));
    request.insert(
        "negative_prompt".into(),
        json!(args.get("negative-prompt").unwrap_or("")),
    );
    if args.has("no-pe") {
        request.insert("use_pe".into(), json!(false));
    }
    danmo_cli::apply_options(
        &mut request,
        &args,
        &[
            "size",
            "steps",
            "guidance",
            "seed",
            "scheduler",
            "scheduler-shift",
        ],
    )?;
    let loras = danmo_cli::collect_loras(&host, &args)?;
    if !loras.is_empty() {
        request.insert("adapters".into(), json!(loras));
    }
    if let Some(controlnet) = args.get("controlnet") {
        let image = args.require("control-image")?;
        let weight = args.f32("controlnet-strength")?.unwrap_or(0.8);
        request.insert(
            "structural_guide".into(),
            json!({
                "asset_id": image,
                "model_id": controlnet,
                "weight": weight,
            }),
        );
    }
    danmo_cli::run_task(&host, "image.generation", &model, json!(request))
}

const USAGE: &str = "danmo-generate --model ID[:version] --prompt TEXT --output FILE
  [--negative-prompt TEXT] [--size WxH] [--steps N] [--guidance F] [--seed N]
  [--scheduler ID] [--scheduler-shift F] [--no-pe]
  [--lora ID|PATH[:WEIGHT]]... (weight 0..2, default 1.0)
  [--controlnet MODEL --control-image PATH [--controlnet-strength F]]
  [--workspace DIR] [--registry FILE]";
