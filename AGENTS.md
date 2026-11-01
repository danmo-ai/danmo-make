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

- Bundle directory: GGUF DiT, GGUF or safetensors text encoder and VAE, `tokenizer.json`, `bundle.manifest.json`. Large towers may be sharded and are mmap'd per file. Duplicate tensor names across shards fail. A safetensors text encoder whose first key is `model.embed_tokens.weight` is read with that prefix on every lookup. `blk.*` names fail. Upscale bundles (SeedVR2) drop the tokenizer and text encoder: DiT GGUF plus VAE safetensors only, validated by `assert_upscale_bundle`. The fixed positive text embedding is `pos_emb.safetensors` — a bundle-local copy (root or `data/`) wins, otherwise the packaged default under `families/seedvr2/data/`.
- The SeedVR2 VAE safetensors stores 5D PyTorch conv3d kernels `[OC, IC, KD, KH, KW]`; the reader folds them to the ggml `[KW, KH, KD, IC*OC]` view without a repack. The SeedVR2 DiT GGUF keeps vectors, biases, and embeddings in F16; they are widened to F32 on load while matmuls stay Q8_0. Its VAE spec (`block_out`, resnet counts, latent channels, temporal up/down, group-norm groups) is inferred from the weight names and shapes, never from a registry flag.
- Version keys: `bf16`, `q8_0`, `q4_k`. The API drops every other key. Published VAE tensors are BF16 safetensors and are widened to F32 for the decoder graph. Inside a quantized DiT, norm scales stay F32. Matmul accepts F32, F16, BF16, Q8_0, and Q4_K.
- `load` mmaps by tensor name. The name in the file is the graph parameter name. There is no `remap_*` and no `sanitize()`. Flux2 reads the published fused names (`double_blocks`, `single_blocks`, `img_in`, `txt_in`, `time_in`, `final_layer`). Z-Image reads the published names already in its weight table. A missing tensor or a shape mismatch fails the task.
- LoRA is GGUF. The matrix name is the base parameter plus `.lora_a` or `.lora_b`. Other suffixes are not rewritten.
- Do not invent Hugging Face repo ids. Until a GGUF is published, that version key is absent and generation fails loud.

`Q8_0` is about half the F16 weight bytes. `Q4_K` is about 30%. Activations do not shrink.

## Engine rules

Long form: [docs/engine_architecture.md](docs/engine_architecture.md).

- Fail loud. No filled-in weights, no truncated prompt, no second CPU backend for an operator the chosen device cannot run. CPU as `DANQING_GGML_BACKEND` is a real device. Invalid configured backends fail startup; only an unset backend defaults to CPU.
- Engine creates one `GgmlContext` per task and injects it into every family session. Text encoders, companion LMs, DiT, FSQ, and VAE do not open another device. Families add models. A new chip backend is a `GgmlContext` backend, not a family.
- Families are isolated. A family does not import another family. `families/mod.rs` is the only match on a family id. `pipeline`, `session`, `common`, and `runtime` do not branch on family ids or contain family implementation types.
- Shared modules (`common/qwen3`, `common/vae`, `common/png`, `common/ops`, `runtime`) do not know family ids. Callers pass layer indices, a thinking flag, or a `VaeSpec`. Weight-file names such as `layers.*` and `decoder.*` are the file format.
- A difference that needs many callbacks or mode switches stays in the family. Denoise does: Flux2 steps `x + dt * (u + g * (c - u))` on `[channels, seq]`; Z-Image steps `x + dt * (c + g * (c - u))` on `[W, H, C]`.
- Schedules live in `common/ops`. Flux2's two flow-match names use empirical mu, and the timestep is cosine then sine of `sigma * 1000`. Z-Image treats `flow_match_euler_discrete` as the same scheduler: static shift unless `use_empirical_mu` is set. `linear` keeps the resolution shift.
- Standard DiT phases stay resolve → load → encode → schedule → infer → decode → persist. Job and GenerationFacade shapes use their documented shared phase order. One cgraph is reserved and replayed per denoise session.
- Server integrates through `ResolvePort`, `ArtifactSink`, and `RunObserver`. Generic `Artifact` / `EngineResult` types carry PNG, WAV, sidecars, metadata, and timings; family outputs do not leak into pipeline or server types.
- ggml calls stay in `danmo-ggml-sys` and `danmo-engine` runtime / weight code.
- Built graphs: `flux2`, `z_image`, and `qwen_image` image create; `z_image` image create with a union controlnet structural guide (diffusers Z-Image control flow: LANCZOS-resized control image VAE-encoded into a 33-channel context — control latents 16 + preserve mask 1 + inpaint latents 16 — embedded by `control_all_x_embedder.2-1`, refined by `control_noise_refiner` / `control_layers` blocks whose zero-init-style `after_proj` hints scaled by `structural_guide.weight` are added into the main noise refiner and even-numbered layers; inpaint pairs use NEAREST mask downsampling and `rgb*(1-mask)` source latents; weight ≤ 0 skips the guide; adapter weights use the diffusers split names `attention.to_q/to_k/to_v/to_out.0`); `qwen_image` image edit (single-reference rewrite, guidance ≥ 1, target/condition latent concat, VAE-encoded conditioning); `flux2` and `z_image` image edit (rewrite img2img: 16px-aligned center crop, AutoencoderKL encode, `init_timestep = max(1, int(steps * source_fidelity))` with default fidelity 0.6, sigma blend, denoise from that timestep); `seedvr2` image upscale (PIL-style bicubic resize by short-edge scale with optional softness pre-downscale, even align, pad 16, tiled 3D VAE encode, one MM-DiT Euler step at t=1000 on the `[noise; latent; mask]` 33-channel input, `latents - model_output`, tiled decode, wavelet + LAB color correction against the preprocessed source); `ace_step` audio create, cover/repaint/lego/extract/complete edits, applicable LoRA, Haar DCW, output quality metadata, and 5Hz LM. ACE-Step batches 1–8 outputs, applies vocal/instrumental conditioning, normalizes waveforms, and emits wav/flac/mp3/opus/aac through ffmpeg with off/mild/strong denoise. Near-silent output fails. Optional `ACESTEP_PMI_SCORING=1` adds real LM log-prob scoring. Every other family/action fails with `GraphNotBuilt` before queueing or when called directly. `real-esrgan-x4plus` stays in that bucket on purpose: no GGUF is published, its registry `versions` is empty, and both the HTTP gate (`ensure_available`) and the legacy `upscale` path fail loud. `z-image-turbo-fun-controlnet-union` likewise keeps an empty `versions` until an official GGUF is published; the structural-guide HTTP gate validates the guide (z_image family only, category `controlnets`, canny/depth/pose/hed/mlsd/scribble/gray/auto types, weight 0–2, paired inpaint source/mask) and resolution of the unpublished adapter fails loud. `diffrhythm-v2` has no published GGUF either (upstream ships safetensors only) and stays `GraphNotBuilt` like `real-esrgan-x4plus`. The assistant routes (`GET /v1/llm/model`, `GET /v1/vision/model`, `POST /api/chat/enhance-prompt`, `POST /v1/chat/completions`) read `llm_inference_*` settings: with `llm_inference_provider=openai_compatible` plus a configured base URL and API key they proxy the configured cloud model (non-stream JSON and streaming SSE passthrough, 502 on remote failure); every other configuration answers 503 fail loud — builtin local LLM inference is not built.
- Immutable mmap weights use the single-slot LRU + TTL cache in `weights/cache.rs`. Qwen3 embeddings use the independent four-entry cache in `common/conditioning.rs`; both are invalidated by path/length/mtime revisions. Mutable denoise sessions and per-run state are never shared.
- `runtime/capacity.rs` reports actual gallocr graph bytes, cached weights, conditioning bytes, and device memory. The first Metal task runs `runtime/probe.rs`, comparing flash attention against an independent host reference and logging timing. Small head dimensions are zero-padded to 64 and sliced back on the same device; unsupported paths fail rather than moving to CPU.

## Backend selection

`DANQING_GGML_BACKEND=auto|cpu|cuda|vulkan|metal`. The binary must be built with the matching cargo feature (`cuda`, `vulkan`, `metal`). A missing feature fails at startup or context init. `auto` is the default: startup tries only compiled backends, macOS orders Metal → CUDA → Vulkan → CPU, other platforms order CUDA → Vulkan → Metal → CPU, and each candidate must pass the attention probe. CPU remains the last fallback; selection never changes during a task. Health returns the selected backend, selection reason, and probe summary. The System Info UI displays them, and all task logs carry a `[backend]` prefix. `scripts/build_ggml_sidecar.sh` enables Metal on macOS, CUDA when `nvcc` or `CUDA_PATH` is present, and Vulkan when the SDK is present.
