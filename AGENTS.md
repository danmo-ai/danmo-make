# AGENTS.md — Danmo Make

Operational guide for people and coding agents working in this repo. Product overview: [README.md](README.md).

Architecture: [docs/engine_architecture.md](docs/engine_architecture.md). Migration work still open: [docs/rust_mlx_migration_todo.md](docs/rust_mlx_migration_todo.md). Per model/version/action status: [docs/rust_mlx_migration_matrix.json](docs/rust_mlx_migration_matrix.json).

The server is the Rust binary `danqing-api`. Do not start it with `scripts/start.sh`.

## Naming

Product / UI / release artifacts: **Danmo Make** / `danmo-make-*`. Runtime identifiers stay `danqing-*` (binary `danqing-api`, bundle id `com.danqing.studio.desktop`, env `DANQING_*`). The command-line creation tools are product surface and use the `danmo-*` prefix (`danmo-generate`, `danmo-edit`, `danmo-audio-generate`, `danmo-audio-edit`, `danmo-video-generate`, `danmo-video-edit`).

## Layout

| Item | Location |
|------|----------|
| API binary | `native/crates/danmo-server` (`danqing-api`) |
| Engine | `native/crates/danmo-engine` |
| MLX | `native/crates/danmo-mlx-sys` + `native/crates/danmo-mlx` (`native/third_party/mlx-c`, `mlx`) |
| ggml (vendor, retiring) | `native/crates/danmo-ggml-sys` (`native/third_party/ggml`) — not a product runtime |
| Contracts / catalog / queue | `danmo-contracts`, `danmo-catalog`, `danmo-schedule`, `danmo-store` |
| Bundle / LoRA resolution | `danmo-resolve` (shared by server and CLI) |
| CLI | `danmo-cli` (`danmo-generate`, `danmo-edit`, `danmo-audio-generate`, `danmo-audio-edit`, `danmo-video-generate`, `danmo-video-edit`) |
| Frontend | `frontend/` (Vue 3, Vite, port 5800) |
| Registry | `default_config/models_registry.json` |
| Desktop | `desktop/` |

Dev ports: backend **7800**, frontend **5800**.

```bash
make dev          # cargo run danqing-api + Vite
make test         # cargo test --manifest-path native/Cargo.toml
make pack-mlx-sidecar
```

The CLI binaries run generation without the HTTP server (registry + workspace resolution is shared with the server via `danmo-resolve`; task mapping via `danmo_engine::engine_request`):

```bash
cargo run --manifest-path native/Cargo.toml -p danmo-cli --features mlx-metal --bin danmo-generate -- \
  --model z-image-turbo --prompt "a red apple" --output out.png
```

CLI sources are file paths (there is no asset store); `--lora` accepts a file path, a user LoRA id from `config/user_loras.json`, or a registry card, with optional `:weight`.

`cargo test` takes one Cargo filter. Extra names after `--` are libtest filters. `--exact` needs the full test path. From the repo root, pass `--manifest-path native/Cargo.toml` or the feature flags are rejected. `danmo-server` has no lib target.

Engine and family code must not import `danmo-mlx-sys`. MLX is opt-in: `mlx`, `mlx-metal`, `mlx-cuda`. The API build that serves this machine is `--features metal,mlx-metal`.

```bash
cargo test --manifest-path native/Cargo.toml -p danmo-mlx
cargo test --manifest-path native/Cargo.toml -p danmo-engine --features mlx-metal --lib
cargo test --manifest-path native/Cargo.toml -p danmo-mlx --features metal
cargo test --manifest-path native/Cargo.toml -p danmo-mlx --features cuda
```

Switching between a prebuilt MLX and the source build needs `cargo clean -p danmo-mlx-sys` first. A stale `@rpath/libmlx.dylib` looks like a numeric failure. `mlx-c` is pinned at 0.8.0 and MLX at 0.32.4. Host-reference F32 probes on NAX set `MLX_ENABLE_TF32=0`. Leave the default on for real model runs.

When replacing a running `danqing-api`, wait until port 7800 is free before starting the new process. Exit 143 is the previous process receiving SIGTERM. Do not restart while a task is actually running.

## Gotchas

- **No product Python.** Serving/engine/training Python trees are gone. Desktop starts `danqing-api` only (no thin uvicorn / venv). Vendor Python under `native/third_party/` stays for MLX/ggml builds.
- **LoRA training deferred.** Submit returns 501 until a Rust MLX trainer exists. Do not revive a Python trainer as a temporary path. Downloaded LoRA inference remains supported.
- **Fail loud.** A missing capability, tensor, or backend is an error. Do not add a second runtime, a weaker path, or a silent default. CPU is a real device only when it was selected for the process.
- Families do not import each other. The only family match is `families/mod.rs`. `pipeline`, `session`, `common`, and `runtime` do not branch on family ids.
- Do not invent Hugging Face repo ids. An unpublished version key is absent and generation fails.
- Do not return MLX `Array` values off the executor thread, and do not call `Executor::execute` from inside an executor job.
- Do not mark a migration row done in the matrix or the todo until that model/version/action has parity, performance, and E2E evidence. Do not mark `quant metal_performance` passed.

## Weights

MLX bundles are **safetensors**. Production loading keeps each dense tensor at its checkpoint dtype. Registry version keys include `fp16`, `bf16`, `fp32`, `int8`, `int4`, `mlx-q4`, `mlx-q8`; a label is not a tensor-dtype guarantee. Packed affine weights stay packed inside `LinearWeight`. An install that lists `components` writes `bundle.manifest.json` after the last file lands.

LoRA files are resolved to adapter paths by the server and applied by the shared weight layer. A target that is not a parameter fails the task.

Layout, cache keys, and the dtype policy: [docs/engine_architecture.md](docs/engine_architecture.md).

## Backend selection (MLX)

The product path is **MLX only**: build `danqing-api` with `metal,mlx-metal` (macOS) or `mlx-cuda` (Linux). There is no runtime Python stack and no GGUF product path. `GET /api/system/health` returns the selected backend, the reason, and the probe summary. Task logs carry a `[backend]` prefix.

`scripts/build_mlx_sidecar.sh` builds the release sidecar with `metal,mlx-metal` on macOS and `mlx-cuda` on Linux when `nvcc` is present (CUDA toolkit only; no GPU needed to compile). Override with `DANQING_SIDECAR_FEATURES`. CI uses the NVIDIA CUDA devel container and `scripts/ci_install_cuda_ubuntu.sh` for env only.
