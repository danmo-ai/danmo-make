//! danmo-quantize — 把完整权重（dense safetensors）按 recipe 转成本地量化包。
//!
//! 这是唯一的生产者：注册中心不再下载第三方预量化权重，安装流程和应用内「转量化」
//! 都走同一条引擎路径（`danmo_engine::run_derive`）。
//!
//! 用法:
//!   danmo-quantize --base models/Image/z-image-turbo-fp16 \
//!       --output models/Image/z-image-turbo-q4-mixed \
//!       --recipe z-image-q4-mixed-v1
//!   danmo-quantize --base ... --list-recipes
//!   danmo-quantize --base ... --recipe z-image-q4-mixed-v1 --print-recipe

fn main() {
    std::process::exit(match run() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("[danmo-quantize] error: {error}");
            1
        }
    });
}

const USAGE: &str = "\
danmo-quantize — 本地量化（dense → q4-mixed）

  --base <dir>         完整权重目录（必填）
  --output <dir>       量化输出目录（默认 <base>-q4-mixed）
  --recipe <id|path>   量化 recipe（默认按 family 推导，或取 --list-recipes 中的 id）
  --workspace <dir>    workspace 根（默认 DANQING_WORKSPACE 或当前目录）
  --repo-root <dir>    factory 根（默认 DANQING_REPO_ROOT 或自动探测）
  --list-recipes       列出可用 recipe
  --print-recipe       打印解析到的 recipe 路径后退出
  --json               以 JSON 输出转换报告
  --help               显示本帮助
";

fn run() -> Result<(), String> {
    let args = danmo_cli::Args::parse_env()?;
    if args.has("help") {
        println!("{USAGE}");
        return Ok(());
    }
    let workspace = danmo_cli::workspace_root(&args)?;
    let repo_root = danmo_cli::repo_root(&args)?;
    if args.has("list-recipes") {
        for row in danmo_cli::list_recipes(&workspace, &repo_root)? {
            println!("{row}");
        }
        return Ok(());
    }
    let base = args.require("base")?;
    let base_path = danmo_cli::resolve_path(&workspace, &base);
    if !base_path.is_dir() {
        return Err(format!("{} is not a directory", base_path.display()));
    }
    let recipe_id = match args.get("recipe") {
        Some(recipe) => recipe.to_string(),
        None => danmo_cli::default_recipe_for(&base_path)?,
    };
    if args.has("print-recipe") {
        println!(
            "{}",
            danmo_cli::find_recipe(&workspace, &repo_root, &recipe_id)?
        );
        return Ok(());
    }
    let output = match args.get("output") {
        Some(output) => output.to_string(),
        None => danmo_cli::default_output(&base_path),
    };
    let report = danmo_cli::run_quantize(&workspace, &repo_root, &base_path, &output, &recipe_id)?;
    if args.has("json") {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "base": base_path.display().to_string(),
                "output": output,
                "recipe": recipe_id,
                "shards": report.shards,
                "tensors": report.tensors,
                "quantized": report.quantized,
                "dense": report.dense,
            }))
            .map_err(|error| error.to_string())?
        );
    } else {
        println!(
            "{} -> {} (recipe {recipe_id}): {} shards, {} quantized, {} dense",
            base_path.display(),
            output,
            report.shards,
            report.quantized,
            report.dense
        );
    }
    Ok(())
}
