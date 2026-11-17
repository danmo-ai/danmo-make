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

A bundle contains a GGUF DiT, GGUF or published safetensors text encoder and VAE, tokenizer files, and `bundle.manifest.json`. Shards are mmap'd per file. Duplicate tensor names across shards fail. The GGUF tensor index is parsed by the engine itself (the ggml gguf reader refuses tensors past four dimensions); dims beyond four fold into `ne3` without moving bytes, so the published Wan 5D `patch_embedding.weight` loads as `[KW, KH, KD, IC*OC]`.

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
| `seedvr2` | Image upscale: bicubic preprocess (softness down→up, even align, pad 16), tiled 3D VAE encode, one MM-DiT windowed-attention Euler step at t=1000, tiled decode, wavelet + LAB color correction. Video upscale: ffmpeg PPM frame extraction, `temporal_window` chunks (default 5, min 4, tail repeats the last frame), causal 3D VAE over time (`D = (F-1)/2+1` per temporal down, decode crops `t_dec > t_in`, fails on `t_dec < t_in`), MM-DiT generalized to `[W, H, T, C]` with frame-outer `[T][H][W]` tokens, per-chunk seed `(seed + start*1009) & 0x7FFFFFFF`, per-frame crop + color correction, ffmpeg MP4 mux keeping the source audio |
| `wan` | Video create (t2v) end to end: prepare applies the TI2V-5B defaults (40 steps, CFG 5, shift 5, 81 frames at 16fps, empty negative prompt → the official Chinese sample_neg_prompt), enforces `(frames-1) % 4 == 0` and floors pixels to 32 (VAE 16 × patch 2), and checks the GGUF bundle (DiT + umT5 + VAE + tokenizer); the umT5 graph (unigram Viterbi tokenizer, unscaled bidirectional attention with per-layer relative position bias, gelu-erf gated FFN, F32 activations, host-verified, output zero-padded to 512 rows), the causal 3D VAE decoder (per-latent-frame replays, in-graph cache advance, first/steady chunk graphs, DupUp3D shortcuts, unpatchify 2, spec inferred from the folded shapes, published 48-channel normalization constants; no spatial tiling yet), and the DiT graph (patchify/adaLN/RoPE/flash-attn/cross-attn/FFN/head, host-verified across replays) are built; the session denoises with WanFlowUniPC (solver_order 2, reference sigma shift, `trunc(sigma*1000)` timesteps, standard CFG skipped at guidance 1) and muxes PNG frames to MP4 without an audio track. Video LoRA is not wired (the video contract has no adapters field). i2v / animate / 14B dual-model fail loud |
| `ltx` | Video create skeleton: prepare applies the LTX 2.3 reference defaults (dev 28 steps / CFG 3.0, distilled 8 stage-1 steps with guidance forced to 1 — the fixed 3 stage-2 refine steps land with the two-stage session — 97 frames at 24fps, 512x768), enforces `(frames-1) % 8 == 0` and floors pixels to 32 (vae_scale 32), and checks the extended bundle via `assert_ltx_bundle` (GGUF DiT + Gemma-3 text encoder + video VAE + audio VAE + embeddings connectors + tokenizer; both variants also require the spatial x2 latent upscaler and the distilled `transformer_stage2` DiT for the two-stage pipeline). The Gemma-3 text encoder (SentencePiece-BPE tokenizer, 48-block tower with per-head q/k RMS norms, local/global NEOX RoPE, masked 49-state stack) and the dual connectors (scaled projection, cyclic register padding, 8-block towers with split-RoPE tables and per-head gates, exact-tanh GELU) are built and host-verified. The video VAE decoder is built and host-verified: non-causal 3x3x3 convs (symmetric replicate time pad, zero space pad), pixel_norm (weightless per-pixel RMS norm, eps 1e-8) + SiLU pre-activation residual blocks, four depth-to-space pixel-shuffles (2,2)/(2,2)/(1,2)/(2,1) that drop the first frame after a temporal shuffle (F latent frames -> 8F-7), per-channel denormalize, and the 48-channel head with the patch-4 spatial unpatchify (channel order `(c, r_w, q_h)`); BF16 conv bytes double as the ggml `[KW, KH, KD, IC*OC]` kernel, widened to F32. The 48-block joint A/V DiT is built and host-verified: patchify projections into the 4096/2048 towers, eight shared adaLN MLPs on the `sinusoidal_timestep(sigma*1000, 256)` embedding (both timestep multipliers are 1000) with per-block scale-shift tables (the a2v/v2a tables' 5th row is the cross gate), q/k full-dim RMS norms + per-head `2*sigmoid` gates in all six attentions, split RoPE via host tables (video 3-axis `[20,2048,2048]`, audio `[20]`, cross 1-axis; theta-10000 `theta^linspace` grid * pi/2 with front zero-pad), prompt-adaln text modulation, exact-tanh GELU FFNs, layer-norm output heads; BF16 matmuls + F32 biases/norms/tables, one cgraph replayed per step (verified across two sigmas). Per-token timesteps, attention masks, and IC-LoRA perturbations are not wired. The audio VAE decoder and the vocoder are built and host-verified: causal 3x3 convs (two zero rows on top of the time axis, one zero column per side on the frequency axis), pixel_norm (eps 1e-6) + SiLU pre-activation residual blocks with 1x1 nin shortcuts, reverse-order up blocks whose nearest-x2 upsample convs drop the first time row (T -> 4T-3, 16 -> 64 mels), per-channel denormalize reordered from the file's `c*16+f` to the ggml `f+16c` layout; the BigVGAN v2 base vocoder (transposed 1D conv stages `[5,2,2,2,2,2]` with the symmetric `(k-r)/2` crop as a view, three averaged AMPBlock1 resblocks per stage — dilations 1/3/5, kernels 3/7/11, depthwise 12-tap anti-alias filters with the channels in the batch axis of a direct 2D conv for exact F32, log-scale SnakeBeta folds at load — act_post, bias-free conv_post, clip ±1), the x3 Hann-sinc resampler (host-built 43-tap kernel), the BWE log-mel front end (causal STFT conv n_fft 512 hop 80, mel matmul, log clamp 1e-5), and the BWE generator `[6,5,2,2,2]`; the 48 kHz output is `clamp(skip + bwe_residual, -1, 1)`, and the base hop 160 is always a multiple of the STFT hop 80 so shapes are static per mel length. The dev t2v session runs end to end: stage 1 denoises at half resolution under the token-adaptive `ltx2_schedule` (linspace sigmas, `tokens*(1.1/3072)+(0.95-1024*1.1/3072)` shift, stretched to the 0.1 terminal) with the dev multimodal guider — four DiT forwards per step (cond / negative / STG block-28 via the per-block perturbation gate inputs / modality via the a2v-v2a gates), `cond + (cfg-1)(cond-neg) + (cond-ptb) + 2(cond-mod)` with the 0.7 population-variance rescale, video cfg = request guidance, audio cfg fixed 7.0, `x0 = x - sigma*v` unwrap, plain Euler; the stage-1 tokens are denormalized with the video VAE per-channel statistics, run through the spatial x2 latent upscaler (zero-pad 3x3x3 convs, clip-wide GroupNorm, pixel-shuffle x2 conv, SiLU post-residual blocks), renormalized, and repatchified at full resolution; stage 2 renoises to sigma 0.909375 and refines for the fixed 3 `STAGE_2_SIGMAS` steps with the distilled DiT — dev loads the `transformer_stage2` component, distilled reuses its own transformer (main `_weight_stem`) — no guidance; the video VAE decodes pixels, the audio VAE + vocoder decode the 48 kHz stereo track, and ffmpeg muxes PNG frames + WAV into the MP4 without `-shortest` (the decoded audio is slightly shorter than frames/fps and every generated frame must land in the container). The distilled variant runs the same two-stage session: stage 1 slices `DISTILLED_SIGMAS[..steps+1]` (8 steps by default) with a single forward per step and no negative prompt (guidance forced to 1), stage 2 is identical |
| `hunyuan` | Video edit (i2v "animate") Facade skeleton: prepare applies the HunyuanVideo 1.5 reference defaults (720x1280, 81 frames at 24fps with `(frames-1) % 4 == 0`, 50 steps, guidance 6.0 — forced to 1 on the cfg-distilled checkpoint — shift 7.0, pixels floored to vae_scale 16) and validates the bundle via `assert_hunyuan_bundle` (jayn7 GGUF DiT with the native Tencent names `double_blocks` / `img_in` / `txt_in` / `byt5_in` / `vision_in` / `time_in` / `cond_type_embedding` / `final_layer`, Tencent video VAE safetensors, Qwen2.5-VL-7B text encoder safetensors + tokenizer, FLUX.1-Redux SigLIP image encoder via the `image_encoder` component role); quoted glyph text fails loud (the byT5 glyph encoder has no published GGUF or safetensors); the Qwen2.5-VL text encoder is built (video system message rendered without `<|im_end|>`, 1108-token cap, `hidden_states[-3]` = layer-25 output with no final norm via `TowerOutput::AfterLayer` on the shared `common::qwen25vl` trunk, 108-token crop, host-verified, safetensors shards); the SigLIP image encoder is built (FLUX.1-Redux tower: 27 layers, hidden 1152, 16x72 heads, inter 4304, patch 14 at 384px, LN eps 1e-6, quick-gelu, post_layernorm, `vision_model.*` safetensors names, bicubic + 0.5/0.5 preprocess, `conv2d_direct` patch embed permuted to row-major tokens, host-verified); the causal 3D VAE encoder/decoder are built (native safetensors key layout, replicate causal padding, per-frame-prefix flash attention instead of main's full `[seq, seq]` causal mask, DCAE rearranges with first-frame special cases, group-mean / repeat-interleave shortcuts, encode mode x 1.03682, decode / 1.03682 + clip, one whole-clip graph with no tiling yet, host-verified); infer fails loud until the 54-block joint DiT lands. Video create (t2v) has no published GGUF and stays `GraphNotBuilt` |
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
