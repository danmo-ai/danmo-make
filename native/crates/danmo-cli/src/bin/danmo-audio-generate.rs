//! danmo-audio-generate — 音乐/歌曲生成 CLI（对应 POST /api/audio/generations）。
//!
//! 用法:
//!   danmo-audio-generate --model ace-step-1.5 --prompt "epic orchestral" \
//!       --duration 30 --output out.wav
//!   danmo-audio-generate --model ace-step-1.5 --prompt "lofi" \
//!       --lyrics-file song.lrc --vocal-language zh --output song.wav

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
    let lyrics = match args.get("lyrics-file") {
        Some(path) => {
            std::fs::read_to_string(path).map_err(|error| format!("lyrics file {path}: {error}"))?
        }
        None => args.get("lyrics").unwrap_or("").to_string(),
    };
    let mut request = serde_json::Map::new();
    request.insert("model".into(), json!(model));
    request.insert("prompt".into(), json!(args.require("prompt")?));
    request.insert("lyrics".into(), json!(lyrics));
    if args.has("instrumental") {
        request.insert("instrumental".into(), json!(true));
    }
    danmo_cli::apply_options(
        &mut request,
        &args,
        &[
            "duration",
            "seed",
            "steps",
            "guidance",
            "vocal-language",
            "bpm",
            "keyscale",
            "timesignature",
            "audio-format",
        ],
    )?;
    danmo_cli::run_task(&host, "audio.generation", &model, json!(request))
}

const USAGE: &str = "danmo-audio-generate --model ID[:version] --prompt TEXT --output FILE
  [--lyrics TEXT | --lyrics-file PATH] [--instrumental]
  [--duration SEC] [--seed N] [--steps N] [--guidance F]
  [--vocal-language LANG] [--bpm N] [--keyscale TEXT] [--timesignature TEXT]
  [--audio-format wav|mp3|flac] [--workspace DIR] [--registry FILE]";
