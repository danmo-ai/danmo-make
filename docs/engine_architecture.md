# Danmo Make Engine Architecture

This is the target engine after the MLX migration. The server binary is `danqing-api`. Compute is MLX. Weights are published safetensors.

How to run the tree today: [`AGENTS.md`](../AGENTS.md). What is still unfinished: [`rust_mlx_migration_todo.md`](rust_mlx_migration_todo.md).

---

## 1. Principles

1. **Fail loud.** Missing tensors, wrong shapes, unknown actions, schedulers, options, an unavailable device, an unbuilt family action, and persistence failures return errors. Nothing is silently replaced, truncated, or moved to another device.
2. **One device for the whole task.** The process picks Metal, CUDA, or CPU at startup. Text encoders, companion LMs, DiT, FSQ, and VAE on that task share it.
3. **Families add models.** A new chip is a build of the MLX runtime, not a family and not a second engine. Family math does not branch on Metal versus CUDA.
4. **Families are isolated.** A family does not import another family. `families/mod.rs` is the only family-id dispatch.
5. **Shared code is family-neutral.** `pipeline`, `session`, `common`, `runtime`, and `weights` do not match on family ids and do not contain family implementation types. Server adapters do not contain family session or output types. Known short-term exceptions (video kind gates, ControlNet family checks, training family hardcodes) are tracked in the migration todo and must converge on registry / `FamilySpec` metadata.
6. **Keep shared code small.** Move code into `common` when at least two families share the math and differ only by configuration. One user, or a different topology, stays in the family. Weight-file names are the file format, not a family id.
7. **Phases are explicit.** Standard DiT runs resolve → load → encode → schedule → infer → decode → persist. Job and facade shapes keep their own phase order. Training keeps its own phase order (below). A flow that needs many callbacks stays a facade, not a pile of switches inside the DiT pipeline.
8. **Missing weights fail.** Duplicate tensor names fail. The shared weight layer may normalize a published layout (prefix, NHWC kernels, LoRA suffixes). A family does not grow a second loader or a private LoRA remap.

The dependency guard is: only `danmo-mlx` depends on `danmo-mlx-sys`. Families and `common` do not touch C handles or unsafe FFI. `architecture.rs` checks registry and factory sync, and that shared layers stay family-neutral (including `weights/` once the guard is extended).

## 2. Dependency direction

```text
Vue / desktop
  → danmo-server HTTP + DTO adapters
  → Engine
  → Session + ResolvedRun + ExecutionContext
  → shared Dit / Job / Facade pipeline
  → families/mod.rs
  → family driver + family-owned session
  → common components
  → runtime (Executor, Array, Stream, Device)
  → danmo-mlx
  → danmo-mlx-sys
  → mlx-c
  → MLX C++
  → Metal or CUDA
```

Server integrations are ports. The engine does not import server or store crates:

```text
ResolvePort   → catalog / workspace / downloads / source assets
ArtifactSink  → atomic output writes, scheduler, asset store, lineage
RunObserver   → task logs, progress, SSE, phase events
```

`mlx-c` is the MLX C API. `mlx-rs` is a reading reference, not a runtime dependency. A C++ capability mlx-c does not expose goes through one `danmo-mlx-native` shim (**target; not implemented**). That shim has no family math and no family strings.

## 3. Execution contracts

- `ResolvedRun`: model, version, family, action, bundle, and device.
- `ExecutionContext`: task id, cancellation, observer, and start time. Every phase and progress callback uses it.
- `EngineRequest`: typed common fields plus opaque `family_options`. The family parses its options and rejects unknown fields.
- `Artifact`: primary bytes or a controlled file payload, MIME, metadata, lineage, dimensions or duration, and nested sidecars. PNG, WAV, LRC, and lyric text share one persistence path. Large media is a file manifest, not base64 inside JSON.
- `EngineOutput` / `EngineResult`: metrics, timings, metadata, artifacts, asset ids, and output paths. Public pipeline types do not contain a family's output struct.
- `EngineError`: invalid requests, unsupported actions, unknown families, unbuilt actions, device failures, cancellation, observer, resolution, weight, runtime, and persistence errors.

Pipeline and session traits are device-neutral. They do not take a compute-graph context.

## 4. Family shapes

`FamilySpec` is stable metadata: id, media, shape, capability flags (`lora_inference`, `structural_guide`, `lora_training`), and scheduler names. `families/mod.rs` maps a family and action to one driver; its training match (`run_family_lora_training`) is the only family-id branch for LoRA training.

| Shape | Shared phase order | Family owns | Status |
|---|---|---|---|
| `DitPipeline` | resolve → load → encode → schedule → infer → decode → persist | DiT, weights, text policy, scheduler policy, denoise, readout | Shared runner in `pipeline/dit.rs` |
| `GenerationFacade` | resolve → prepare → infer → persist | Preparation, companion models, and the encode / schedule / decode steps inside | Shared runner in `pipeline/facade.rs` |
| `JobPipeline` | resolve → load → infer(job) → persist | Non-standard upscale or job execution | **Retained shape; no shared `pipeline/job.rs` yet.** `FamilyDriver::Job` fails loud. SeedVR2 and Real-ESRGAN are driven as Facades; every `FamilySpec.shape` must match its driver (Facade today, or Job once a shared job runner exists). |

`DitImageFamily::load` returns a `DitImageSession`. The shared pipeline calls `encode`, `schedule`, `infer`, and `decode` in that order. `FacadeFamily::prepare` returns a `FacadeSession`. A facade still has named substeps.

**Training** is not a generation shape. Phase order is resolve → load → encode cache (VAE / text) → train loop → export adapter. Training jobs share the MLX executor for array work but must not sit behind the single generation queue worker forever; the control plane may run them on a dedicated blocking path until a second worker exists. Exported adapters load through the shared weight / LoRA layer. Family-specific train hooks (presets, DiT forward under grad) stay in `families/<id>/train/`.

A new family changes only:

1. Its registry entry in `default_config/models_registry.json`.
2. `families/<id>/` and its tests.
3. One factory row in `families/mod.rs`.
4. The capability matrix row for each model, version, and action.

It does not change pipeline, session, common, server, or public DTOs, and it does not edit another family. Registry capabilities and implemented drivers are checked before a task is queued. An unbuilt action stays visible and fails before it runs. Orphan family trees that are not wired in `families/mod.rs` are forbidden; delete them rather than leave parallel source.

Denoise formulas that differ stay in the family. Flux2 steps `x + dt * (u + g * (c - u))` on `[channels, seq]`. Z-Image steps `x + dt * (c + g * (c - u))` on `[W, H, C]`. Schedules themselves live in `common/ops`.

## 5. Shared components

- `common/qwen3`: tokenizer and tower. Callers pass layer indices, a thinking flag, and output width.
- `common/qwen25vl`: shared trunk. Callers choose the layer output, including an intermediate layer with no final norm.
- `common/vae`: AutoencoderKL configured by `VaeSpec`, including tiling and a stream cache when more than one family needs them. **VAE rule:** only topologies that match and differ by configuration enter `common/vae`. Causal / 3D / product-specific VAEs stay in the family until a second identical topology appears.
- `common/png`, `common/image`, `common/ffmpeg`: pixels, resize and color, and process-spawned media.
- `common/ops`: sigma schedules, deterministic normal noise, sinusoidal embeddings, and non-finite checks.
- `weights`: safetensors loading, the dtype policy, and LoRA binding. Same family-neutral rule as `pipeline` / `session` / `common` / `runtime`: no family-id dispatch; layout helpers are named by published format or capability, not by family id.
- `runtime`: the MLX executor, device and stream, compile, attention dispatch, and capacity.

Attention layout, masks, and the request struct can live in `common`. The Metal or CUDA kernel stays in the runtime.

Before code moves into `common`, name the real callers, the part that is identical, the parameters that differ, the part that stays in the family, the shapes and dtypes, and the fixture that does not mention a family id. A neutral interface may exist before the second caller. A second framework may not.

## 6. MLX runtime

```text
families / common → danmo-mlx → danmo-mlx-sys → mlx-c → MLX C++ → Metal or CUDA
```

`danmo-mlx-sys` is the only crate that touches bindgen, CMake, and raw handles. `danmo-mlx` owns RAII arrays, devices, streams, dtypes, errors, compile, quantization, and IO.

Metal and CUDA are separate build features and separate artifacts (`mlx-metal`, `mlx-cuda`). There is no in-process switch. Family code does not contain `if metal` or `if cuda`. Device selection today is compile-time: a build with `mlx-metal` or `mlx-cuda` uses GPU; otherwise CPU. `DANQING_MLX_DEVICE=auto|gpu|cpu` is the **target** startup knob (read once; `gpu` fails if missing; `cpu` only when asked) and is **not implemented** yet — do not document callers as if it already exists.

One process owns one long-lived executor thread for generation. That thread creates arrays, streams, compiled functions, and the model cache. `Array`, `Device`, and `Stream` are thread-affine and are not sent off the executor. HTTP and Tokio threads pass plain data, paths, bytes, and cancellation tokens. Nested `Executor::execute` deadlocks. Cancellation is checked at step, token, tile, and chunk boundaries. A panic or a worker crash fails the task. It does not retry on another device. Every MLX family (including video facades) must run generation work through this executor; creating a private `Stream::gpu()` on the task thread is a structural violation.

The local assistant uses its own worker (**target; builtin path is still a stub**). A chat completion does not wait behind a long denoise. Training may share the MLX executor for compute but must not block the generation queue indefinitely; see §4. The control plane assigns each worker a memory budget and records it.

Compiled closures are shape-specialized. Tables that do not depend on the traced tensors, such as RoPE, are computed outside the closure. A family does not keep a global or thread-local weight cache.

Custom ops prefer stock MLX operators. A missing op is an explicit unavailable path, or a pair of Metal and CUDA kernels behind the runtime. Attention dispatch as native / general / custom / unavailable is the **target** runtime surface; families currently call stock MLX SDPA directly. `mlx-mfa` is a Metal reference, not a runtime dependency.

`runtime` reports active bytes, cache bytes, peak bytes, and the device limit (**target for product health; MLX APIs for active/peak exist, cache/limit and HTTP exposure are incomplete**). An attention probe compares the device kernel with an independent host reference. An unsupported path fails on that device. CUDA correctness needs at least a compile-time smoke gate under `mlx-cuda`; Metal-only tests do not cover it.

## 7. Weights, LoRA, and cache

`weights/mlx.rs` loads every safetensors shard in a directory. Version keys are the published ones: `fp16`, `bf16`, `fp32`, `int8`, `int4`, `mlx-q4`, `mlx-q8`. An unfinished install does not enter the cache.

`WeightDtype::Native` is the production policy and it is not a family branch. Each dense tensor keeps the checkpoint dtype: BF16, FP16, or F32. `WeightDtype::F32` exists only under `cfg(test)` so a parity fixture can reproduce a widened reference. Production cannot select it. Norms, RoPE, and timestep tables may be cast inside the op. A registry label such as `fp16` is not evidence of the tensor dtype.

Affine int4 and int8 stay packed: uint32 codes, scales, and biases live in an opaque `LinearWeight`. `apply` runs quantized matmul. The architecture guard rejects a path that dequantizes that matrix into a resident dense copy. Bits and group size are checked. Quantized norms or convolutions that the published loader widens do so in this weight layer, not in the family.

LoRA is a residual on that same `apply`: `y = Wx + scale * B(Ax)`, for both dense and packed linears. The shared loader matches published suffixes (`.lora_a` / `.lora_A` / `.lora_down`, including PEFT `.default`) onto weight names. `MlxWeights::attach_lora` attaches them. The cached base weights are not rewritten. Scale is strength times alpha over rank. Alpha comes from the tensor, else `lora_alpha` in `adapter_config.json` or `lora_config.json`, else the rank. A target that is not a 2-D linear fails.

The executor owns weights, compiled functions, and conditioning arrays. Cache identity includes shard path, length, mtime, manifest generation, tokenizer and config revision, device, dtype policy, and the LoRA set. The budget is in bytes, not only in entry count. Conditioning keys include every input that changes the output: token ids, negative prompt, layers, width, thinking flag, mask, and guidance. A host-byte cache exists only when a measurement shows it pays for itself. Mutable denoise state is never shared across tasks.

## 8. Server integration

`danmo-server` maps HTTP DTOs to `EngineRequest`, implements the ports, and writes task JSON. One worker owns generation execution. Cancellation is registered in `danmo-schedule`, delivered through `ExecutionContext`, and stored as `cancelled`. Training may use a separate blocking path until it has its own worker; downloads use a separate store. An `EngineRouter` that unifies capability-driven routing is **target**, not current code.

Invalid `DANQING_HTTP_PORT`, a missing MLX Cargo feature (`mlx-metal` / `mlx-cuda`), or a database init error fails startup. Registry JSON is opened on demand today; **startup-time registry validation is a tracked gap** and should fail loud once wired. Dequeue and task-finish failures are logged. They are not turned into an empty queue.

The active control plane is the only writer of tasks and assets for the generation queue. A worker returns events and a file manifest. SSE progress is currently polled from the DB; realtime push is unfinished.

Server adapters must not grow family-id branches. Availability gates call `danmo_engine::ensure_available`, capability checks read `FamilySpec`, and the training family comes from the catalog entry. The task-kind → `EngineRequest` mapping lives in `danmo_engine::engine_request` (`adapter.rs`) and is shared by every host.

A second host is the `danmo-cli` crate (`danmo-generate`, `danmo-edit`, `danmo-audio-generate`, `danmo-audio-edit`, `danmo-video-generate`, `danmo-video-edit`). It runs the same `Engine::execute` path without HTTP: installed-bundle and LoRA weight resolution is shared with the server through `danmo-resolve`; sources are file paths, not asset ids; artifacts persist to `--output`. The CLI is not a second engine and adds no family branches.

Product assistant routes (`/v1/llm/model`, `/v1/vision/model`, `/api/chat/enhance-prompt`, `/api/assistant/run`) call an internal LLM client for in-app enhance, creative-assistant, and related helpers only. The client may use the local MLX assistant worker (**target; builtin is a stub that fails loud**) or an OpenAI-compatible backend from settings; it is not a public chat proxy. A missing local model and a missing cloud configuration both fail the request.
