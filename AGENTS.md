# AGENTS.md — Danmo Make

Operational guide for people and coding agents working in this repo. Product overview: [README.md](README.md).

Architecture: [docs/engine_architecture.md](docs/engine_architecture.md). Migration status and remaining work: [docs/rust_mlx_migration_todo.md](docs/rust_mlx_migration_todo.md). Parity and acceptance evidence: `tests/parity/evidence/`.

The server is the Rust binary `danqing-api`. Use `make dev` (or `scripts/start.sh`) for the web app; there is no Python API server.

## Naming

Product / UI / release artifacts: **Danmo Make** / `danmo-make-*`. Runtime identifiers stay `danqing-*` (binary `danqing-api`, bundle id `com.danqing.studio.desktop`, env `DANQING_*`). The command-line creation tools are product surface and use the `danmo-*` prefix (`danmo-generate`, `danmo-edit`, `danmo-audio-generate`, `danmo-audio-edit`, `danmo-video-generate`, `danmo-video-edit`).

## Layout

| Item | Location |
|------|----------|
| Rust workspace | `native/Cargo.toml` (the root `Cargo.toml` has no workspace members) |
| API binary | `native/crates/danmo-server` (`danqing-api`) |
| Engine | `native/crates/danmo-engine` |
| Contracts / catalog | `native/crates/danmo-contracts`, `native/crates/danmo-catalog` |
| Bundle / LoRA resolution | `native/crates/danmo-resolve` (shared by server and CLI) |
| Persistence / queue | `native/crates/danmo-store`, `native/crates/danmo-schedule` |
| MLX | `native/crates/danmo-mlx-sys` + `native/crates/danmo-mlx` (`native/third_party/mlx-c`, `native/third_party/mlx`) |
| ggml (vendor, retiring) | `native/crates/danmo-ggml-sys` (`native/third_party/ggml`) — not a product runtime |
| CLI | `native/crates/danmo-cli` (six `danmo-*` binaries) |
| Frontend | `frontend/` (Vue 3, TypeScript, Vite, Pinia; port 5800) |
| Desktop | `desktop/` (Tauri 2; starts the `danqing-api` sidecar) |
| Factory configuration | `default_config/` (registry, presets, locales, LoRA examples) |
| Development / packaging | `scripts/`, `Makefile` |
| Generated artifacts | `out/` (frontend, sidecar, desktop, logs, run state) |
| Benchmarks / parity evidence | `tests/benchmark/`, `tests/parity/` |

Frontend UI dependencies `@danqing/dq-shell`, `@danqing/dq-tokens`, and `@danqing/dq-ui` are local `file:` packages from the sibling `dq-ui` checkout; see `frontend/package.json`.

## Development

Dev ports: backend **7800**, frontend **5800**. Override them with `DQ_BACKEND_PORT` and `DQ_FRONTEND_PORT`.

```bash
make dev                 # scripts/start.sh: danqing-api + Vite HMR
make stop
make dev-desktop         # Tauri + Vite + API
SKIP_BACKEND=1 make dev-desktop  # use an API that is already running
```

A source build needs Rust, Node.js/npm, CMake, and the native submodules. If MLX sources are missing:

```bash
git submodule update --init --recursive native/third_party/mlx native/third_party/mlx-c
```

`make dev` uses `metal,mlx-metal` on macOS. The API and Vite logs are under `out/run/`; use `make stop` rather than leaving orphaned processes. When replacing a running `danqing-api`, wait until port 7800 is free before starting the new process. Exit 143 is the previous process receiving SIGTERM. Do not restart while a task is actually running.

Workspace selection is handled by `scripts/workspace_env.sh`: explicit `DANQING_WORKSPACE`, then the pointer under `DANQING_USER_DATA_DIR` (default `~/.danmo-make`), then the repo root. The script also supplies defaults for `DANQING_DB`, `DANQING_OUTPUT`, and `DANQING_REGISTRY`.

## Verification

```bash
make test                # cargo test --manifest-path native/Cargo.toml
make lint                # cargo fmt --all -- --check
make frontend-typecheck
make frontend-build      # writes out/frontend/dist/
make frontend-canvas-unit
```

`cargo test` takes one Cargo package filter. Extra names after `--` are libtest filters. `--exact` needs the full test path. From the repo root, pass `--manifest-path native/Cargo.toml`; the root manifest is not the Rust workspace. `danmo-server` has no lib target.

Engine and family code must not import `danmo-mlx-sys`. MLX is opt-in: `mlx`, `mlx-metal`, `mlx-cuda`. The API build that serves macOS is `--features metal,mlx-metal`; Linux CUDA uses `--features mlx-cuda`.

```bash
cargo test --manifest-path native/Cargo.toml -p danmo-mlx
cargo test --manifest-path native/Cargo.toml -p danmo-engine --features mlx-metal --lib
cargo test --manifest-path native/Cargo.toml -p danmo-mlx --features metal
cargo test --manifest-path native/Cargo.toml -p danmo-mlx --features cuda
```

Switching between a prebuilt MLX and the source build needs `cargo clean -p danmo-mlx-sys` first. A stale `@rpath/libmlx.dylib` looks like a numeric failure. `mlx-c` is pinned at 0.8.0 and MLX at 0.32.4. Host-reference F32 probes on NAX set `MLX_ENABLE_TF32=0`. Leave the default on for real model runs.

## CLI

The CLI binaries run generation without the HTTP server. Registry and workspace resolution are shared with the server via `danmo-resolve`; task mapping uses `danmo_engine::engine_request`; execution uses the same `Engine::execute` path.

```bash
cargo run --manifest-path native/Cargo.toml -p danmo-cli --features mlx-metal --bin danmo-generate -- \
  --model z-image-turbo --prompt "a red apple" --output out.png
```

CLI sources are absolute or workspace-relative file paths, not asset ids. `--lora` accepts a file path, a user LoRA id from `{workspace}/config/user_loras.json`, or a registry card, with optional `:weight`. Every binary supports `--help`.

## Gotchas

- **No product Python.** Serving, engine, and training do not use a Python runtime. Desktop starts `danqing-api` only (no thin uvicorn / venv). Vendor Python under `native/third_party/` stays for MLX/ggml builds; development shell helpers may still use system `python3` for JSON parsing.
- **LoRA training runs the full Rust MLX pipeline** (VAE latent cache → Qwen3 TE → differentiable 30-block DiT with per-block `mx.checkpoint` → AdamW → DiffSynth-keyed adapter export). Z-Image / Z-Image-Turbo only; non-MLX builds return an error; quantized base tiers are rejected at submit (training needs a dense bundle). Remaining gates before "done": face detection for `face_crop=on`, per-tensor gradient evidence, checkpoint-resume E2E, a full-length 2500-step run, and the CUDA gate — see `docs/rust_mlx_migration_todo.md`. Do not revive a Python trainer. Downloaded LoRA inference remains supported.
- **Fail loud.** A missing capability, tensor, or backend is an error. Do not add a second runtime, a weaker path, or a silent default. CPU is a real device only when it was selected for the process.
- Families do not import each other. Family-id dispatch belongs in `families/mod.rs`. `pipeline`, `session`, `common`, `runtime`, `weights`, and server adapters do not branch on family ids.
- Do not invent Hugging Face repo ids. An unpublished version key is absent and generation fails.
- Do not return MLX `Array` values off the executor thread, and do not call `Executor::execute` from inside an executor job.
- Do not mark migration status done until that model/version/action has parity, performance, and E2E evidence. Metal and CUDA are separate gates. Do not mark quant `metal_performance` passed.

## Weights

MLX bundles are **safetensors**. Production loading keeps each dense tensor at its checkpoint dtype. A registry version is either `kind: "dense"` (the only download: `fp16` / `bf16` / `fp32`) or `kind: "derived"` (a dense `base` plus a recipe, converted on this machine). There is no pre-quantized download tier; the registry publishes one `q4-mixed` version per base model and it is a local build product.

`default_config/quant_recipes/<recipe>.json` holds the layer policy: `keep_dense` globs for embeddings, norms, modulations, and final projections, a `min_elements` floor, and one `{bits, group_size}` pair. The converter in `weights/quantize.rs` (driven by `derive.rs`) writes the published MLX affine layout — packed `{base}.weight` plus `{base}.scales` / `{base}.biases` — and stamps `bundle.manifest.json` with the recipe id, a hash of the recipe file, and a layout digest of the dense source. A changed recipe or re-published base fails the load with a re-convert instruction.

Packed affine weights stay packed inside `LinearWeight`; the loader rejects a bundle whose packed layers disagree on bits/group size. An install that lists `components` writes `bundle.manifest.json` after the last file lands.

LoRA files are resolved to adapter paths by the server or CLI and applied by the shared weight layer. A target that is not a parameter fails the task.

Layout, cache keys, and the dtype policy: [docs/engine_architecture.md](docs/engine_architecture.md).

## Backend selection and packaging

The product generation path is **MLX only**: build `danqing-api` with `metal,mlx-metal` (macOS) or `mlx-cuda` (Linux). There is no runtime Python stack and no GGUF product path. `DANQING_MLX_DEVICE` is not implemented; do not document it as a working switch. `GET /api/system/health` returns the selected backend, the reason, and the probe summary. Task logs carry a `[backend]` prefix.

`scripts/build_mlx_sidecar.sh` builds the release sidecar into `out/sidecar/danqing-api/` with `metal,mlx-metal` on macOS and `mlx-cuda` on Linux when `nvcc` is present (CUDA toolkit only; no GPU needed to compile). Override with `DANQING_SIDECAR_FEATURES`. CI uses the NVIDIA CUDA devel container and `scripts/ci_install_cuda_ubuntu.sh` for env only. MLX packs also stage `libmlx` / `libmlxc` (and `mlx.metallib` on macOS) next to `danqing-api` and rely on `@executable_path` / `$ORIGIN` LC_RPATH — a bare binary without those libs dies at launch with `Library not loaded: @rpath/libmlxc.dylib` / `no LC_RPATH's found`.

```bash
make pack-mlx-sidecar
make pack-macos-desktop
make pack-linux-desktop
make pack-linux-server
make pack-windows-sidecar  # sidecar only; Windows desktop packaging is not wired
```
