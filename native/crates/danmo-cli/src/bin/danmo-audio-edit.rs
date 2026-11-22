//! danmo-audio-edit — 音频编辑 CLI（对应 POST /api/audio/edits：cover / repaint / lego / …）。
//!
//! 用法:
//!   danmo-audio-edit --model ace-step-1.5 --task cover --source-audio song.wav \
//!       --prompt "lofi cover" --lyrics-file cover.lrc --output cover.wav

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
    request.insert(
        "task".into(),
        json!(args.get("task").unwrap_or("cover-nofsq")),
    );
    request.insert("prompt".into(), json!(args.require("prompt")?));
    request.insert("lyrics".into(), json!(lyrics));
    request.insert("source_audio".into(), json!(args.require("source-audio")?));
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
            "source-fidelity",
            "audio-format",
        ],
    )?;
    danmo_cli::run_task(&host, "audio.edit", &model, json!(request))
}

const USAGE: &str =
    "danmo-audio-edit --model ID[:version] --source-audio PATH --prompt TEXT --output FILE
  [--task cover|cover-nofsq|repaint|lego|extract|complete]
  [--lyrics TEXT | --lyrics-file PATH] [--instrumental]
  [--duration SEC] [--seed N] [--steps N] [--guidance F] [--source-fidelity F]
  [--vocal-language LANG] [--bpm N] [--audio-format wav|mp3|flac]
  [--workspace DIR] [--registry FILE]";
