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

Long form: [docs/engine_architecture.md](docs/engine_architecture.md).

- Fail loud. No filled-in weights, no truncated prompt, no second CPU backend for an operator the chosen device cannot run. CPU as `DANQING_GGML_BACKEND` is a real device.
- One `GgmlContext` owns the device. The whole graph stays on it. Families add models. A new chip backend is a `GgmlContext` backend, not a family.
- Families are isolated. A family does not import another family. `families/mod.rs` is the only match on a family id. Pipelines do not branch on `family == ...`.
- Shared modules (`common/qwen3`, `common/vae`, `common/png`, `common/ops`, `runtime`) do not know family ids. Callers pass layer indices, a thinking flag, or a `VaeSpec`. Weight-file names such as `layers.*` and `decoder.*` are the file format.
- A difference that needs many callbacks or mode switches stays in the family. Denoise does: Flux2 steps `x + dt * (u + g * (c - u))` on `[channels, seq]`; Z-Image steps `x + dt * (c + g * (c - u))` on `[W, H, C]`.
- Phases stay resolve → load → encode → schedule → infer → decode → persist. One cgraph is reserved and replayed.
- ggml calls stay in `danmo-ggml-sys` and `danmo-engine` runtime / weight code.
- Built image graphs: `flux2` (MM-DiT; Qwen layers 9/18/27; packed-latent batch norm, unpatch, post-quant, then AutoencoderKL) and `z_image` (DiT; Qwen penultimate state; AutoencoderKL when `decoder.conv_in.weight` is present). A one-tensor `decoder.conv.weight` file stays the 1×1 preview. Every other family fails after the bundle check.

## Backend selection

`DANQING_GGML_BACKEND=cpu|cuda|vulkan|metal`. The binary must be built with the matching cargo feature (`cuda`, `vulkan`, `metal`). A missing feature fails at context init. `scripts/build_ggml_sidecar.sh` enables Metal on macOS, CUDA when `nvcc` or `CUDA_PATH` is present, and Vulkan when the SDK is present.
