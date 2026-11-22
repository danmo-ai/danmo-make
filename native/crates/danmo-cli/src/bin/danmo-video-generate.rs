//! danmo-video-generate — 文生视频 CLI（对应 POST /api/videos/generations）。
//!
//! 用法:
//!   danmo-video-generate --model ltx-2.5-distilled:mlx-q4 \
//!       --prompt "a red cube rotates on a desk" --num-frames 25 --fps 24 --output out.mp4

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
    danmo_cli::apply_options(
        &mut request,
        &args,
        &[
            "size",
            "steps",
            "guidance",
            "shift",
            "seed",
            "num-frames",
            "fps",
        ],
    )?;
    let loras = danmo_cli::collect_loras(&host, &args)?;
    if !loras.is_empty() {
        request.insert("adapters".into(), json!(loras));
    }
    danmo_cli::run_task(&host, "video.generation", &model, json!(request))
}

const USAGE: &str = "danmo-video-generate --model ID[:version] --prompt TEXT --output FILE
  [--negative-prompt TEXT] [--size WxH] [--steps N] [--guidance F] [--shift F]
  [--seed N] [--num-frames N] [--fps N] [--lora ID|PATH[:WEIGHT]]...
  [--workspace DIR] [--registry FILE]";
