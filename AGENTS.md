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

The DiT is GGUF. The text encoder and VAE are GGUF or the published safetensors. MLX `fp16` / `int8` / `int4` / `mlx-q4` / `mlx-q8` directories are not scanned, converted, or loaded. FLUX.2 Klein 4B and Z-Image installs copy the ModelScope text encoder, VAE, and tokenizer into that version's own directory. A directory that already contains a GGUF is not treated as an old MLX pack because of the safetensors companions. An install that lists `components` writes `bundle.manifest.json` into that directory after the last file lands. `quant` is the version key. `transformer` is the planned GGUF. `text_encoder` and `vae` are the planned weight files under those directories.

- Bundle directory: GGUF DiT, GGUF or safetensors text encoder and VAE, `tokenizer.json`, `bundle.manifest.json`. Large towers may be sharded and are mmap'd per file. Duplicate tensor names across shards fail. A safetensors text encoder whose first key is `model.embed_tokens.weight` is read with that prefix on every lookup. `blk.*` names fail.
- Version keys: `bf16`, `q8_0`, `q4_k`. The API drops every other key. Published VAE tensors are BF16 safetensors and are widened to F32 for the decoder graph. Inside a quantized DiT, norm scales stay F32. Matmul accepts F32, F16, BF16, Q8_0, and Q4_K.
- `load` mmaps by tensor name. The name in the file is the graph parameter name. There is no `remap_*` and no `sanitize()`. Flux2 reads the published fused names (`double_blocks`, `single_blocks`, `img_in`, `txt_in`, `time_in`, `final_layer`). Z-Image reads the published names already in its weight table. A missing tensor or a shape mismatch fails the task.
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
- Schedules live in `common/ops`. Flux2's two flow-match names use empirical mu, and the timestep is cosine then sine of `sigma * 1000`. Z-Image treats `flow_match_euler_discrete` as the same scheduler: static shift unless `use_empirical_mu` is set. `linear` keeps the resolution shift.
- Phases stay resolve → load → encode → schedule → infer → decode → persist. One cgraph is reserved and replayed.
- ggml calls stay in `danmo-ggml-sys` and `danmo-engine` runtime / weight code.
- Built image graphs: `flux2` (MM-DiT; Qwen layers 9/18/27; packed-latent batch norm, unpatch, post-quant, then AutoencoderKL) and `z_image` (DiT; Qwen penultimate state; AutoencoderKL when `decoder.conv_in.weight` is present). A one-tensor `decoder.conv.weight` file stays the 1×1 preview. Every other family fails after the bundle check.

## Backend selection

`DANQING_GGML_BACKEND=cpu|cuda|vulkan|metal`. The binary must be built with the matching cargo feature (`cuda`, `vulkan`, `metal`). A missing feature fails at context init. `scripts/build_ggml_sidecar.sh` enables Metal on macOS, CUDA when `nvcc` or `CUDA_PATH` is present, and Vulkan when the SDK is present.
