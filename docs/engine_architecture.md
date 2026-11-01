# Danmo Make Engine Architecture

The engine is Rust on ggml, under `native/`. The server binary is `danqing-api`. DiT weights are GGUF; text encoders and VAEs are GGUF or published safetensors. Tensor names in weight files are graph parameter names.

Operational rules: [`AGENTS.md`](../AGENTS.md).

---

## 1. Principles

1. **Fail loud.** Missing tensors, wrong shapes, unknown actions/schedulers/options, unavailable backends, unbuilt graphs, and persistence failures return errors. Nothing is silently replaced, truncated, or moved to a fallback device.
2. **One device for the whole task.** Engine creates a `GgmlContext` for the selected backend (`cpu`, `cuda`, `vulkan`, or `metal`) and injects it into every family session. Text encoding, companion LMs, DiT, FSQ, and VAE do not open another device.
3. **Families add models.** A new chip is a `GgmlContext` backend, not a family or another engine tree.
4. **Families are isolated.** A family does not import another family. `families/mod.rs` is the only family-ID dispatch point.
5. **Shared code is family-neutral.** `pipeline`, `session`, `common`, and `runtime` do not match on family IDs or contain family implementation types.
6. **Keep shared code small.** Neutral parameters belong in shared code. Different tensor ranks, CFG formulas, denoise loops, DiT graphs, and readout policies stay in the family.
7. **Phases are explicit.** Standard DiT runs resolve → load → encode → schedule → infer → decode → persist. Job and facade shapes keep their own documented phase order.
8. **No key remap.** Missing or duplicated tensors fail. GGUF names bind directly to graph parameters. Safetensors dimensions are reversed into ggml order without moving bytes.

Architecture tests in `architecture.rs` check registry/factory synchronization, family-neutral shared layers, and the prohibition on direct ggml calls from production family code.

## 2. Dependency direction

```text
Vue / desktop
  → danmo-server HTTP + DTO adapters
  → Engine
  → Session + ResolvedRun + ExecutionContext
  → shared Dit / Job / Facade pipeline
  → families/mod.rs
  → family driver + family-owned session
  → common components and algorithms
  → runtime (GgmlContext, GraphBuilder, ggml operators)
  → danmo-ggml-sys
```

Server integrations are ports, not engine dependencies:

```text
ResolvePort   → Catalog / workspace / downloads / source assets
ArtifactSink  → atomic output writes + Scheduler / AssetStore + lineage
RunObserver   → task logs, progress, SSE, and phase events
```

`pipeline`, `session`, `common`, and `runtime` never import server or store crates. Production uses only the new Engine facade. The old free-function pipeline remains solely as a `cfg(test)` numerical reference in `pipeline/legacy.rs`.

## 3. Execution contracts

- `ResolvedRun`: model, version, family, action, bundle, and backend.
- `ExecutionContext`: task ID, cancellation token, observer, and start time. Every phase and inference progress callback uses it.
- `EngineRequest`: strongly typed common request fields plus opaque `family_options` at the boundary. A family parses its own options with unknown fields denied.
- `Artifact`: primary bytes, MIME/type, metadata, lineage, dimensions/duration, and nested sidecars. PNG, WAV, LRC, and lyric text use one persistence path.
- `EngineOutput` / `EngineResult`: neutral metrics, timings, metadata, artifacts, asset IDs, and output paths. Public pipeline types do not contain ACE-Step output structures.
- `EngineError`: invalid requests, unsupported actions/options, unknown families, unbuilt graphs, backend failures, cancellation, observer, resolution, weight, runtime, and persistence errors.

## 4. Family shapes and extension

`FamilySpec` contains stable metadata: ID, media, shape, LoRA support, and scheduler names. `families/mod.rs` maps a family/action to a driver.

| Shape | Shared phase order | Family owns |
|---|---|---|
| `DitPipeline` | resolve → load → encode → schedule → infer → decode → persist | DiT graph, weights, text policy, scheduler policy, denoise, VAE/readout |
| `JobPipeline` | resolve → load → infer(job) → persist | Non-standard upscale/job execution |
| `GenerationFacade` | resolve → prepare → infer → persist | Preparation, companion models, internal encode/schedule/decode substeps |

`DitImageFamily::load` returns a `DitImageSession`. The shared pipeline calls `encode`, `schedule`, `infer`, and `decode` in order. `FacadeFamily::prepare` returns a `FacadeSession`; ACE-Step create/edit/LM uses this interface. The facade test adds a second audio implementation without changing pipeline code.

A new family changes only:

1. Its registry entry in `default_config/models_registry.json`.
2. `families/<id>/` and its tests.
3. One factory registration in `families/mod.rs`.

It does not change pipeline, session, common, server, or public DTOs. Registry-declared capabilities and implemented drivers are checked before a task is queued. Unbuilt families remain visible but fail with `GraphNotBuilt`; they never return placeholder success.

## 5. Shared components and algorithms

- `common/qwen3`: Qwen3 tokenizer and tower. Callers pass 1-based layer indices, a thinking flag, and output width.
- `common/vae`: AutoencoderKL decoder configured by `VaeSpec`.
- `common/png`: RGB PNG output.
- `common/ops`: sigma schedules, deterministic normal noise, sinusoidal embeddings, and non-finite checks.
- `runtime`: device session, graph construction/replay, attention probes, small-head padding, capacity snapshots, RoPE, group norm, nearest upsample, preview convolution, Euler/CFG primitives, and element-size access.
- `common/conditioning`: a four-entry LRU for Qwen3 embeddings, keyed by backend, layer selection, width, token IDs, and weight-file revision.
- `weights`: GGUF/safetensors loading, LoRA binding, FP16 conversion, and mmap cache.

Flux2 and Z-Image share Qwen3 and AutoencoderKL with different parameters. Qwen-Image keeps its Qwen2.5-VL trunk and 3D causal VAE in its family because their topology differs. Flux2, Z-Image, Qwen-Image, and ACE-Step keep their own denoise loops where layout or guidance formulas differ.

Reusable components must move to `common` before another family consumes them. A family must not import an encoder, VAE, scheduler, or other implementation from another family.

## 6. ACE-Step boundary

All ACE-specific behavior lives in `families/ace_step/`:

- Create/edit/LM option schemas and mode parsing.
- 5Hz LM expansion and companion-model selection through `ResolvePort`.
- Source WAV decoding, 48 kHz validation, source latents, cover/repaint policies.
- WAV, lyric text, LRC, alignment metadata, and phase timings converted to generic artifacts/results.

The server only maps DTO fields mechanically and supplies source bytes. It does not know `CoverMode`, `LmMode`, `CreateOutput`, or ACE persistence rules. ACE receives the same `GgmlContext` as the rest of the task.

## 7. Weights and cache

A bundle contains a GGUF DiT, GGUF or published safetensors text encoder and VAE, tokenizer files, and `bundle.manifest.json`. Shards are mmap'd per file. Duplicate tensor names across shards fail.

Version keys are `bf16`, `q8_0`, and `q4_k`. Norm scales remain F32 inside quantized DiTs. Matmul accepts F32, F16, BF16, Q8_0, Q4_K, Q5_K, and Q6_K where supported by the runtime. VAE inputs are widened at the load boundary when required.

`weights/cache.rs` provides a single-slot LRU + 30-minute TTL cache for immutable mmap weights. Cache keys include paths, file lengths, and modification times. A changed file is reopened rather than reused. Every run owns its graph inputs, latent state, schedule, cancellation, progress, and outputs; mutable denoise sessions are not shared. Graph reserve or memory-budget failure is an error, never a reason to select another model, quant, or backend.

`common/conditioning.rs` is a separate four-entry LRU for Qwen3 embeddings. Its key includes the same file revision plus backend, selected layers, output width, and token IDs. It never shares entries across model or backend revisions.

`runtime/capacity.rs` registers actual gallocr graph buffer sizes and subtracts them on graph drop. Engine logs and returns weight, conditioning, graph, managed, available, and total device bytes in the capacity metadata. `runtime/probe.rs` validates flash attention against an independent host implementation and records host/backend timings on the first Metal task. Small head dimensions are padded to 64 and sliced back on the same device; unsupported paths fail instead of falling back to CPU.

## 8. Server, queue, and failure behavior

`danmo-server` maps HTTP DTOs to `EngineRequest`, implements engine ports, and translates results into task JSON. A single worker owns execution. Cancellation is registered in `danmo-schedule`, delivered through `ExecutionContext`, and recorded as `cancelled` rather than `failed`.

Invalid `DANQING_HTTP_PORT`, invalid `DANQING_GGML_BACKEND`, missing backend features, registry errors, and DB initialization errors fail startup. `DANQING_GGML_BACKEND` also accepts `auto` and defaults to it. Auto tries only compiled backends, using Metal → CUDA → Vulkan → CPU on macOS and CUDA → Vulkan → Metal → CPU elsewhere. Every candidate must initialize and pass the attention probe; CPU is the final fallback. Selection is fixed for the process and never changes during a task. Health returns the selected backend, reason, and probe summary, the System Info UI displays them, and every task log has a `[backend]` prefix. Dequeue and task-finish failures are logged and surfaced instead of being converted into an empty queue or ignored.

The assistant routes (`/v1/llm/model`, `/v1/vision/model`, `/api/chat/enhance-prompt`, `/v1/chat/completions`) mirror the legacy OpenAI-compatible surface. Model info reports the resolved registry id, provider, availability, and think flags. Generation requires `llm_inference_provider=openai_compatible` with a base URL and API key in settings: requests are proxied to `{base_url}/chat/completions` with the configured cloud model, including streaming SSE passthrough, and remote failures return 502. Without that configuration the routes answer 503 fail loud; builtin local LLM inference is not built.

## 9. Current graphs and migration status

| Family | Implemented GGML actions |
|---|---|
| `flux2` | Image create, applicable LoRA, rewrite img2img edit (VAE encode + sigma blend at `init_timestep`) |
| `z_image` | Image create, applicable LoRA, rewrite img2img edit (VAE encode + sigma blend at `init_timestep`), union controlnet structural guide (33-channel VAE-encoded context, diffusers control refiner/layer hints into the main branch, optional inpaint pair) |
| `qwen_image` | Image create, masked positive/negative CFG, single-reference edit (rewrite) |
| `seedvr2` | Image upscale: bicubic preprocess (softness down→up, even align, pad 16), tiled 3D VAE encode, one MM-DiT windowed-attention Euler step at t=1000, tiled decode, wavelet + LAB color correction |
| `ace_step` | Audio create/edit batches, vocal/instrumental conditioning, applicable LoRA, Haar DCW, normalized wav/flac/mp3/opus/aac + denoise and quality metadata, 5Hz LM with optional PMI |
| Other registered families | No graph yet; explicit `GraphNotBuilt` |

The `main` MLX implementation is a source of phase fixtures and model behavior, not a runtime dependency. A family/action becomes available only after its real graph, persistence, and end-to-end verification are complete.

## 10. Verification

```bash
cargo test --manifest-path native/Cargo.toml --offline --workspace
make test
make frontend-build
```

Verification covers architecture boundaries, a second facade family, invalid configurations, missing weights/tensors, cancellation, cache reuse/eviction, real HTTP task execution, artifact persistence, and CPU generation. CUDA, Vulkan, and Metal require matching cargo features and runtime verification on their respective platforms; build-only verification does not imply runtime support.

Metal verification uses `DANQING_GGML_BACKEND=metal cargo test --manifest-path native/Cargo.toml --offline -p danmo-engine --features metal --lib`. On Apple M5 Max, the attention probe reported max abs error 0.000008, about 2.284 ms for the host reference and 0.253 ms for Metal flash. The small-head fallback also matches the host reference, and a real Metal HTTP image task completes with persisted assets.
