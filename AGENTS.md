# AGENTS.md — Danmo Make

Operational guide. User overview: [README.md](README.md).

The Python backend, MLX engine, PyInstaller sidecar, and Python CLIs are gone. The server is the Rust binary `danqing-api`.

## Naming

Product / UI / release artifacts: **Danmo Make** / `danmo-make-*`. Runtime identifiers stay `danqing-*` (binary `danqing-api`, bundle id `com.danqing.studio.desktop`, env `DANQING_*`).

## Layout

| Item | Location |
|------|----------|
| API binary | `native/crates/danmo-server` (`danqing-api`) |
| Engine | `native/crates/danmo-engine` |
| ggml bindings | `native/crates/danmo-ggml-sys` (`native/third_party/ggml`) |
| Contracts / catalog / queue | `danmo-contracts`, `danmo-catalog`, `danmo-schedule`, `danmo-store` |
| Frontend | `frontend/` (Vue 3, Vite, port 5800) |
| Registry | `default_config/models_registry.json` |
| Desktop | `desktop/` |

Dev ports: backend **7800**, frontend **5800**.

```bash
make dev          # cargo run danqing-api + Vite
make test         # cargo test --manifest-path native/Cargo.toml
make pack-ggml-sidecar
```

## Weights

Runtime files are GGUF only. Users re-download. Safetensors and MLX `fp16` / `int8` / `int4` / `mlx-q4` / `mlx-q8` bundles are not scanned, converted, or loaded.

- Bundle directory: `transformer.gguf`, `text_encoder.gguf`, `vae.gguf`, `tokenizer.json`, `bundle.manifest.json`. Large towers may be sharded (`role-00001-of-000NN.gguf`) and are mmap'd per file. Duplicate tensor names across shards fail.
- Version keys: `bf16`, `q8_0`, `q4_k`. The API drops every other key. VAE stays F16/BF16 in all tiers. Inside a quantized file, norm / bias / modulation stay F32 or F16.
- `load` mmaps by tensor name. The name in the file is the graph parameter name. There is no `remap_*` and no `sanitize()`. A missing tensor or a shape mismatch fails the task with a re-download error.
- LoRA is GGUF. The matrix name is the base parameter plus `.lora_a` or `.lora_b`. Other suffixes are not rewritten.
- Do not invent Hugging Face repo ids. Until a GGUF is published, that version key is absent and generation fails loud.

`Q8_0` is about half the F16 weight bytes. `Q4_K` is about 30%. Activations do not shrink.

## Engine rules

- Pipelines do not branch on `family == ...`. The factory table in `families/mod.rs` is the only family match.
- Phases stay resolve → load → encode → schedule → infer → decode → persist.
- One cgraph is reserved and replayed. Steps only replace inputs.
- `import` of ggml stays inside `danmo-ggml-sys` and `danmo-engine` runtime/weight code.
- Fail loud. No identity weight fill when a file was requested.
- Flux2 currently runs a smoke velocity block (`velocity.weight`) so CPU tests can load a tiny GGUF. It is not the full MM-DiT. Other families return `GraphNotBuilt` after the bundle check.

## Backend selection

`DANQING_GGML_BACKEND=cpu|cuda|vulkan|metal`. The binary must be built with the matching cargo feature (`cuda`, `vulkan`, `metal`). A missing feature fails at context init. `scripts/build_ggml_sidecar.sh` enables Metal on macOS, CUDA when `nvcc` or `CUDA_PATH` is present, and Vulkan when the SDK is present.
