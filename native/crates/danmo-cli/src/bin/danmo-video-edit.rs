//! danmo-video-edit — 视频编辑 / i2v animate CLI（对应 POST /api/videos/edits）。
//!
//! 用法:
//!   danmo-video-edit --model ltx-2.5-distilled:mlx-q4 --source-image first.png \
//!       --prompt "the cube rotates" --num-frames 25 --output out.mp4

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
    request.insert(
        "source_asset_id".into(),
        json!(args.require("source-image")?),
    );
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
    danmo_cli::run_task(&host, "video.edit", &model, json!(request))
}

const USAGE: &str =
    "danmo-video-edit --model ID[:version] --source-image PATH --prompt TEXT --output FILE
  [--negative-prompt TEXT] [--size WxH] [--steps N] [--guidance F] [--shift F]
  [--seed N] [--num-frames N] [--fps N] [--workspace DIR] [--registry FILE]";
