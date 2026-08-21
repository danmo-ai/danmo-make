# Danmo Make Engine Architecture

**Single source of truth** for engine layout, plugin model, new-family integration, and CI gates.

Operational product rules: [`AGENTS.md`](../AGENTS.md). Cursor enforcement: [`.cursor/rules/model-migration.mdc`](../.cursor/rules/model-migration.mdc).

---

## 1. Governance anchors

| Constraint | Meaning |
|------------|---------|
| **Fail loud** | Missing backend, component, remap, or shape mismatch → `RuntimeError` / clear API error + task log |
| **No silent downgrade** | No weak CLI fallback, unknown-model shim, or partial key match loading |
| **Registry retained** | `default_config/models_registry.json` is the product catalog; profiles shrink duplication only |
| **Engine net LOC** | Refactors under `backend/engine/` should net-delete or stay neutral |
| **MLX-only honesty** | Registry `backends: ["mlx"]` must match implementation (Metal or mlx[cuda]); missing capability fails at task entry |
| **No parallel trees** | No `vae_codecs/`, `video_codecs/`, or `common/{vae,text_encoders,...}/` wrapper dirs |

**Explicit non-goals:** directory scan replacing registry; runtime structure inference; undocumented fallback toggles; ComfyUI-style open node canvas; pipeline as arbitrary graph executor (GenerationGraph is internal observability only).

---

## 2. Request path

```
REST / CLI
  → contracts + IImageEngine / IVideoEngine / IAudioEngine
  → TaskScheduler
  → DanQing*Engine → engine_dispatch
  → Session (image / video / audio / upscale) — phases + trace spans
  → phased helpers (*_create_phases, image_edit_phases, …)
  → FamilyPlugin + backbone load + inference (L2 strategies)
  → RuntimeContext + TransformerBase + common/
  → asset store
```

Generation families in `FAMILY_CONFIG_MAP` without a registered `FamilyPlugin` **fail loud** at dispatch. Unsupported action/model combinations fail loud via `_require_session_route()` — no silent pipeline fallback.

**Four facts**

| Fact | Source |
|------|--------|
| Product | `models_registry.json` |
| Structure | `model_configs.py` + Transformer code |
| Wiring | `_transformer_registry.py` + codec registries |
| Assets | bundle + `bundle.manifest.json` |

---

## 3. Engine layout

```
backend/
├── catalog/            # schema v3 loader, CatalogResponse DTO, migrate script
├── observability/      # RunTrace, error_codes, graph_runtime, diagnostic API
└── engine/
    ├── cache.py, lineage.py, codecs.py
    ├── contracts/      # pipeline/family runtime semantics
    ├── sessions/       # ImageSession, VideoSession, AudioSession, UpscaleSession
    │   ├── engine_dispatch.py   # Session vs pipeline routing
    │   ├── session_routing.py   # routes_to_*_session helpers
    │   └── _phases/             # resolve, encode, schedule, infer, decode, persist
    ├── protocols/      # Backbone, VAE, TextEncoder Protocol stubs
    ├── platform/       # PlatformSession (device session stub)
    ├── registry/       # family_registry + bootstrap_family_plugins()
    ├── runtime/        # MLXContext, mlx_runtime, mlx_dtype
    ├── common/         # __init__.py facade only at root
    │   ├── ops/        # attention, norm, embeddings, schedulers, cfg_batch
    │   ├── model/      # TransformerBase, DelegatingDiTStem
    │   ├── bundle/     # weights I/O, layout, lora_mlx
    │   └── codecs/     # cross-family TE/VAE (T5, CLIP, Qwen3, AutoencoderKL)
    ├── inference/      # L2 paradigms: denoise / AR / two-stage / job (no engine/paradigms/)
    ├── llm/            # Assistant: prompt sanitize, vision, describe-node (not DiT inference)
    ├── families/<id>/  # transformer, weights, plugin.py (≤8 logical units)
    ├── pipelines/      # phased helpers; fallback entry for non-plugin models
    ├── _transformer_registry.py
    ├── vae_codec_registry.py
    └── video_codec_registry.py
```

**Import map**

| Need | Import from |
|------|-------------|
| DiT primitives | `backend.engine.common.ops.*` |
| Transformer base | `backend.engine.common.model.base` |
| Bundle / LoRA | `backend.engine.common.bundle.*` |
| Cross-family codec | `backend.engine.codecs` or `common.codecs.*` |
| Pipeline semantics | `backend.engine.contracts` |
| Model cache | `backend.engine.cache` |
| MLX helpers | `backend.engine.runtime.mlx_runtime` |
| Family-only (e.g. Qwen VAE) | `backend.engine.families.qwen.*` |

**`common/` rules:** root may only contain `__init__.py`. Allowed subpackages: `ops`, `model`, `bundle`, `codecs`. Single-family code lives in `families/<id>/`.

---

## 4. MLX runtime & plugin shapes

### 4.1 RuntimeContext tiers

Engine is **MLX-only**: one API on macOS **Metal** and Linux **mlx[cuda]**. No `CudaContext` / torch CUDA under `backend/engine/`.

- **Tier 1:** hot path uses `ctx.*` (Linear, RMSNorm, attention helpers) via `MLXContext`.
- **Tier 2:** device-specific blocks in `*_mlx.py` when `ctx` cannot express required APIs.
- **`import mlx`** only in `runtime/` or `*_mlx.py`.
- **`import torch`** is **forbidden** under `backend/engine/` (`make check-engine-imports` / `--rule mlx-torch`).
- Do **not** add `*_cuda.py` or torch CUDA parallel trees.

### 4.2 Go-style stem (one logical unit)

`stem.py` + `stem_mlx.py` count as **1** unit. Registry-facing class lives in `transformer.py` or `stem.py`; implementation classes in `*DiTMLX` avoid name clashes. Do not scaffold or expect `stem_cuda.py`.

### 4.3 Shape A / B / C

| Shape | When | Entry | Template family |
|-------|------|-------|-----------------|
| **A — DiT + Pipeline** | Standard denoise `forward(latents, t, txt_embeds=…)` | `ImagePipeline` / `VideoPipeline` | **flux2** |
| **B — Job pipeline** | Non-standard denoise (upscale, MM-DiT job) | `stem.py` + `stem_mlx.py` | **seedvr2** |
| **C — Generation facade** | End-to-end audio / full stack | `generation.py` | **ace_step** |

Default image family files: `transformer.py`, optional `text_encoder.py`, `weights.py`. Do **not** mirror upstream directory trees.

---

## 5. Registration (Shape A)

Complete **all five** before merge:

1. `default_config/models_registry.json` — `family`, `engine`, `actions`, `backends`, bilingual `name`/`description`
2. `make sync-models-registry`
3. `backend/engine/config/model_configs.py` — dataclass + `FAMILY_CONFIG_MAP`
4. `backend/engine/_transformer_registry.py` — `_TRANSFORMER`, `_TEXT_ENCODER`, optional `_IMAGE_LORA_MERGE`; DiT remap via `sanitize()` on the class
5. `backend/engine/vae_codec_registry.py` — only if VAE `_class_name` ≠ generic `AutoencoderKL`

**VAE**

- Generic: `AutoencoderKL` + `common/codecs/vae/decoder.py`
- Family codec: register in `vae_codec_registry.py`
- Long mapping table: `families/<family>/vae/` (e.g. Qwen-Image)

**Weights:** `remap_<family>_weights` output keys must match flat `_param_map` exactly. Override `sanitize()` on the Transformer; no silent key drops.

**Pipeline:** registry-driven hooks and polymorphism only — no long-term `if family == "..."` in pipelines.

---

## 6. New model integration checklist

### Design

- [ ] Read upstream for math reference only; do not copy vendor tree layout
- [ ] Confirm `media`, `actions`, `backends`; fail loud on gaps
- [ ] Pick template: Shape A → flux2; B → seedvr2; C → ace_step
- [ ] Optional: `python scripts/scaffold_image_family.py --family NAME --class ClassName`

### Wiring

- [ ] Update registry + `make sync-models-registry`
- [ ] `model_configs.py`, `_transformer_registry.py`
- [ ] Bundle contract in `backend/core/bundle_manifest.py` when install validation applies
- [ ] Pipelines call `assert_media_bundle_ready()` from `common/bundle/layout.py`

### Implementation

- [ ] Reuse `common/ops` before family-local attention/norm/embeddings
- [ ] Family ≤8 logical units; no parallel `mlx/` / `runtime/` subtrees under family
- [ ] Nested MLX `parameters()` → explicit flat `_param_map` matching remap keys
- [ ] Explicit errors on shape/key mismatch

### Validation

```bash
python -m py_compile <touched files>
make verify-engine-stack
bin/danqing-generate --model <id> --prompt "test"   # or media-specific CLI
```

Desktop: new modules must be reachable from `scripts/build_desktop.py`.

### Reuse matrix (check first)

| Component | Path |
|-----------|------|
| Attention | `common/ops/attention.py` |
| RoPE / embeddings | `common/ops/embeddings.py` |
| Norm / AdaLN | `common/ops/norm.py` |
| CFG batching | `common/ops/cfg_batch.py` |
| Scheduler | `common/ops/schedulers.py` |
| MLX eval/load | `runtime/mlx_runtime.py` |
| Pipeline contracts | `engine/contracts` |

### Anti-patterns (reject)

- Family-local copies of generic SelfAttention/RMSNorm
- Silent load/inference fallbacks
- Model behavior in API routes or pipeline `family` branches
- New engine codec wrapper directories (`vae_codecs/`, `video_codecs/`)

---

## 7. CI gates

Implementation: [`scripts/check_engine_governance.py`](../scripts/check_engine_governance.py) + [`scripts/engine_governance_allowlist.txt`](../scripts/engine_governance_allowlist.txt).

### 7.1 Commands

| Gate | Command |
|------|---------|
| Full stack | `make verify-engine-stack` |
| All engine rules | `make check-engine-governance` |
| Imports | `make check-engine-imports` |
| MLX / no-torch | `make check-engine-governance --rule mlx-torch` |
| Family layout / budget | `make check-engine-family-layout` |
| Weight key parity | `make check-engine-governance --rule parity` |
| Registry contracts | `make check-engine-governance --rule registry` |
| Docs (single architecture file) | `make check-engine-governance --rule docs` |
| Pipeline family branches | `make check-engine-governance --rule pipeline-family` |
| Consistency | `make check-consistency` |

### 7.2 Rules summary

| Rule | Checks |
|------|--------|
| `imports` | No raw `mlx` outside `runtime` / `*_mlx`; no `torch` under `backend/engine/` |
| `mlx-torch` | No `torch` or leftover `*_cuda` imports under engine (MLX-only) |
| `layout` | No forbidden family subtrees; no `*_cuda.py`; `common/` subpackage layout; ≤8 logical units |
| `primitives` | No duplicate SelfAttention/RMSNorm classes in families |
| `attention` | Prefer `common/ops/attention` over raw `ctx.attention` |
| `sdpa` / `rope` / `modulation` | Centralized helpers in `common/ops` |
| `registry` | Hunyuan and registry contract invariants |
| `parity` | `remap_*` keys vs Transformer `_param_map` |
| `docs` | Only `docs/engine_architecture.md` under `docs/` |
| `pipeline-family` | No `family ==` / `family !=` branches in `pipelines/` |

### 7.3 Validation layers

```
L1 Static   verify-engine-stack + governance + consistency
L2 Load     manifest + key parity + shape check (fail at load)
L3 Generate bench-eval-smoke / bench-eval-case / CLI smoke
```

Family changes: at least L1 + L2 + one smoke case for that family. Bench FAIL must not merge via fallback paths.

User-visible failures: HTTP/CLI error + task log/SSE + i18n in `default_config/locales/`.

---

## 8. Parity debugging (bench regressions)

1. Run one case: `make bench-eval-case ID=<model>:<prompt>:<action>`.
2. Classify: **runtime** (crash/keys) → **semantic** (contract) → **numeric** (tensor math).
3. Lock invariants: seed, steps, scheduler, CFG, latent shape/dtype — log before diffing weights.
4. Check **contract layer** first: scheduler, guidance, noise layout, text encoder, tokenizer template kwargs.
5. Isolate stages: text encode → scheduler sigmas → initial noise → single denoise step → VAE decode.
6. Prefer centralized contract entrypoints over inline pipeline branches.

---

## 9. Optional backlog (non-blocking)

| Item | Notes |
|------|-------|
| Registry audit CLI | Profile inheritance / duplicate params report |
| TE convergence | Go-style pairs under `common/codecs/text_encoders/` (`stem.py` + `stem_mlx.py`) |
| Reuse metrics | `make check-engine-governance --report reuse` (common + inference import counts) |
| `register_image_family()` | `scripts/register_image_family.py` — wiring checklist after scaffold |

**Landed (2026-06):** Flux2 VAE encode → `vae_codec_registry`; `require_entry_family()`; FIBO edit hook `_IMAGE_EDIT_EXTRA_COND`; Hunyuan CFG batch encode; `codecs.py` used from `image_pipeline`; Wan UMT5 → `create_video_t5_encoder()`; LTX distilled timesteps → `video_apply_ltx_distilled_scheduler_timesteps()`; video upscale → `video_upscale_registry` + `VideoUpscaleSession`; PIL→MP4 → `common/codecs/vae/video_io.save_pil_frames_to_mp4`; LTX mlx-forge weight restore → `prepare_video_transformer_weights()`; SeedVR2 upscale load → `upscale_job_registry.get_upscale_pipeline_loader()`.

---

## 10. Sessions + plugins (landed)

**Status:** Production path (2026-06). Create/edit/upscale across image, video, and audio generation families route through Sessions.

### 10.1 Layer model

| Layer | Path | Role |
|-------|------|------|
| **Sessions** | `engine/sessions/` | `MediaSession` + `session_prepare`, `ResolvedRun` / `MediaRunContext`, phased orchestration |
| **Inference (L2)** | `engine/inference/` | Algorithm strategies: diffusion loop, flow-matching, job, AR, two-stage |
| **Family plugin** | `families/<id>/plugin.py` | `FamilyPlugin` bundle: spec, backbone factory, hook flags |
| **Codecs + ops** | `common/codecs/`, `common/ops/` | Shared TE/VAE/math reuse pool |
| **Image ops** | `pipelines/image_run_common.py` | Resolve / encode / schedule / VAE / preview (`ImagePipeline` is ctx holder only) |
| **Video ops** | `pipelines/video_run_common.py` | Resolve / encode / schedule / VAE / two-stage (`VideoPipeline` is ctx holder only) |
| **Platform** | `engine/platform/`, `engine/runtime/` | Device session, kernel factories (`*_mlx` only; Metal / mlx[cuda]) |
| **Catalog** | `backend/catalog/` | On-disk schema v3 ≠ `GET /api/registry` DTO (`CatalogResponse`) |
| **Observability** | `backend/observability/` | `RunTrace`, SSE `trace`, `GET /graph`, `POST /diagnose`; dev `GET /diagnostic` |

**Dependency rule:** `sessions → inference → (family plugin | codecs) → platform`. No `family ==` in `sessions/`. `observability/` does not import `families/`.

**Encode / schedule (v3 landed shape):** text encode, VAE encode, and scheduler setup live in `build_*_run_context(resolved=…)` inside `pipelines/*_phases.py` (helpers in `image_run_common.py` / `video_run_common.py`). Session `ResolvedRun` is the single registry/bundle source — phased builders must not re-`parse_model_version` or `require()` the entry. Optional per-family `FamilyPlugin.encode_conditioning` remains on the protocol; production encode uses `build_*_run_context` only. Phase trace spans: `_phases/trace.py` (`phase_trace_span`).

**Infer (single chain):** every `*RunContext` exposes `session_infer()`; production uses `infer_phase(..., run_ctx=ctx)` → `session_infer()` → `execute_*` → `inference/`. Legacy top-level `run_*` / `run_*_from_context` orchestrators are removed — only `run_*_phased` in `phased_create.py` + `build_*_run_context` remain.

### 10.2 Dispatch

- **`engine/sessions/engine_dispatch.py`** — `dispatch_image_create/edit/upscale`, `dispatch_video_create/edit`, `dispatch_audio_create/edit`.
- **`routes_to_*_session()`** — per-media routing when a `FamilyPlugin` is registered.
- **`assert_generation_family_has_plugin()`** — fail loud when `FAMILY_CONFIG_MAP` family lacks `bootstrap_family_plugins()` registration.
- **`_require_session_route()`** — fail loud when registry actions disallow the Session path.

### 10.3 Plugin backbone load

Shared loaders peel model cache wiring from phased encode:

| Media | Loader | Backbone helper |
|-------|--------|-----------------|
| Image | `pipelines/image_model_load.py` | `families/_image_backbone.py` |
| Video | `pipelines/video_model_load.py` | `families/_video_backbone.py` |
| Audio | `pipelines/audio_model_load.py`, `audio_persist.py` | `families/_audio_backbone.py` |
| Upscale | `pipelines/upscale_model_load.py` | `families/_upscale_backbone.py` |

`load_plugin_phase()` calls `backbone.load()` + `after_load()` under `load_backbone` span; encode phases reuse `plugin_*_if_ready` helpers.

### 10.4 Registered plugin families

Image create/edit: `flux2`, `z_image`, `qwen` (`qwen_image`), `flux1`, `fibo`, `ernie_image`. Video: `wan`, `ltx`, `hunyuan`. Audio: `ace_step`, `diffrhythm`. Upscale (image): `seedvr2`. Video upscale (job): `hunyuan` (1080p latent SR), `seedvr2` (spatiotemporal file video).

New Shape-A image family: add `plugin.py`, register in `registry/bootstrap.py`, implement phased create via `image_create_phases.py` — **no new pipeline class**.

### 10.5 Catalog v3

- On-disk: `default_config/models_registry.json` `schema_version: 3` (`catalog` / `runtime` / `ui` / `distribution` nesting).
- Migration: `make sync-models-registry` auto-migrates workspace v2 copies via `backend.catalog.migrate_v2`.
- Engine/governance reads flat model records via `backend.catalog.loader.expand_catalog_document()`.

### 10.6 Validation

```bash
make verify-engine-stack
# Bench smoke (image): make bench-eval-smoke
```

### 10.7 Observability (product)

- **SSE** `event: trace` on `GET /api/tasks/{id}/stream` — live pipeline graph snapshots (`RunTrace.set_update_callback`).
- **REST** `GET /api/tasks/{id}/graph` — graph for task log **Pipeline** tab.
- **REST** `POST /api/tasks/{id}/diagnose` — local LLM summary over diagnostic bundle (requires LLM sidecar).
- **Dev** `GET /api/tasks/{id}/diagnostic` — full bundle for agents/scripts.

| Media | Paradigm | Session phased entry |
|-------|----------|----------------------|
| Image create | `diffusion` | `run_image_create_phased` → `execute_create_denoise` |
| Image edit (rewrite) | `diffusion` | `run_image_edit_phased` → `execute_image_edit_denoise` |
| Image edit (fill) | `diffusion` | `run_image_edit_phased` → `ImageFillEditRunContext` + `execute_image_fill_edit_denoise` |
| Image edit (qwen VL) | `diffusion` | `run_image_edit_phased` → `QwenImageEditRunContext` + `execute_qwen_image_edit_denoise` |
| Video | `diffusion` | `run_video_create_phased` → `execute_video_denoise` |
| Video (LTX) | `two_stage` | `run_video_*_phased` → `run_family_video_generator_paradigm` |
| Audio create | `flow_matching` / `block_ar` | `run_audio_create_phased` → `run_audio_waveform_paradigm` |
| Audio edit | `flow_matching` | `run_audio_edit_phased` → `run_audio_edit_handler_paradigm` |
| Upscale (image) | `job` | `run_upscale_create_phased` → `run_upscale_job` |
| Video upscale (Hunyuan SR) | `job` | `run_video_upscale_phased` → `run_hunyuan_1080p_sr` |
| Video upscale (SeedVR2) | `job` | `run_video_upscale_phased` → `run_seedvr2_video_upscale` (ffmpeg extract/mux) |

All session paths: `infer_phase(..., run_ctx=ctx)` → `ctx.session_infer()` → L2 `inference/` (or family `execute_*`) → `persist_*_phase` / `traced_persist`. `*RunContext` types extend `MediaRunContext` in `sessions/_context.py`; session resolve/load/pipeline wiring is shared via `sessions/_prepare.py`.

---

## 11. Completed refactor archive

Phase 0–6, DX8–15 (registry profiles, manifest, common subpackages, contracts, runtime helpers, cache/lineage, codec facade), and engine v3 Phases 0–4 (sessions, catalog v3, plugin backbone, dispatch) are **landed**. This file is the single engine architecture document — no separate v2/v3 plan files under `docs/`.

---

## 12. v3 closure (2026-06)

### 12.1 Layer map (canonical)

| Layer | Path | Notes |
|-------|------|-------|
| Session (L1) | `engine/sessions/` | Product orchestration + `engine_dispatch.py` |
| Inference (L2) | `engine/inference/` | **Paradigm algorithms live here** (no `engine/paradigms/` package) |
| Family plugin | `families/<id>/plugin.py` | Wiring + `FamilySpec` from catalog |
| Catalog | `backend/catalog/` | On-disk `families` block drives `FamilySpec` |
| LLM assistant | `engine/llm/` | Prompt/lyrics sanitize, vision, describe-node — **not** generation inference |
| Observability | `backend/observability/` | RunTrace, graph, diagnose |

**Dependency rule:** `sessions → inference → (family plugin | codecs) → platform`. Session routes use **registry `actions`** + `FamilyPlugin` registration — no `family ==` string checks.

### 12.2 FamilySpec source of truth

- **Runtime:** `backend.catalog.family_spec_loader.family_spec_from_catalog()` (reads `models_registry.json` → `families.<id>`).
- **Migration seeding only:** `family_spec_from_model_config()` in the same module (v2→v3 `migrate_v2`).

### 12.3 Pipeline holders (naming)

| Media | Holder class | Ops |
|-------|--------------|-----|
| Image | `ImagePipeline` | `image_run_common.py` + `*_phases.py` |
| Video | `VideoPipeline` | `video_run_common.py` + `*_phases.py` |
| Audio | `AudioPipeline` (`music_pipeline` alias) | `audio_*_phases.py` |
| Image upscale | `UpscalePipeline` (`ImageUpscalePipeline` alias) | `upscale_*_phases.py` |
| Video upscale | `VideoUpscalePipeline` | `video_upscale_*_phases.py` |

Encode + schedule run in `build_*_run_context(resolved=…)` inside phased builders — not `_phases/encode.py`.

### 12.4 New image family DX

```bash
python scripts/scaffold_image_family.py --family MY_FAMILY --class MyTransformer
python scripts/register_image_family.py --family MY_FAMILY
make sync-models-registry && make verify-engine-stack
```

Scaffold emits `transformer.py`, `transformer_mlx.py`, `weights.py`, **`plugin.py`**.

### 12.5 Family layout exceptions

| Family | Exception |
|--------|-----------|
| **ace_step** (Shape C) | `audio/`, `lm/`, `vae/`, `vocals/`, `quality/` subtrees — documented, not a template for image DiT |
| **qwen** | `vae/` 3D VAE codec subtree — long mapping table, registry via `vae_codec_registry` |

### 12.6 Quantized DiT inference (MLX)

Registry versions with `quantization.bits` (4/8, `mlx_affine`) load DiT weights into
`QuantizedLinear` for low-VRAM inference — shared path for **local derived** and
**pre-downloaded** bundles. Resolution: `backend/engine/common/bundle/quant_inference.py`;
loader: `backend/engine/common/model/quantized_load_mlx.py`;
LoRA on quantized DiT: `backend/engine/common/model/quantized_lora_mlx.py` (merge → re-quantize touched layers).
TE/VAE (optional): `resolve_component_inference_weight_mode()`; local `convert_model` may also
quantize `text_encoder/` / `vae/` when registry sets `quantization.<component>.bits` (reference:
`flux2-klein-4b` derived `int4`/`int8`); Qwen3/Flux2 TE loads affine weights via registry +
`build_zimage_mlx_encoder`; VAE decode releases after `vae_forward_to_pil` when
`vae_release_after_decode` (default true). Quantized DiT is MLX-only (Metal / mlx[cuda]).
Design and rollout status:
[`docs/plans/quantized-inference-memory.md`](plans/quantized-inference-memory.md).

### 12.7 Ongoing (non-blocking)

| Item | Status |
|------|--------|
| Registry audit CLI | §9 backlog |
| TE Go-style pairs under `common/codecs/text_encoders/` | §9 backlog |
| Flux1 ControlNet | Registry `backends: ["mlx"]` on Fill/ControlNet rows |
| Engine net LOC reduction pass | Dedicated refactor PR (merge phases helpers where safe) |
| Qwen VAE file consolidation | Optional; high risk — keep subtree until remap tooling exists |
