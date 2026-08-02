# AGENTS.md — Danmo Make v4

Operational guide for contributors and coding agents. User-facing overview: [README.md](README.md) (English default, bilingual).

---

## What this is

Danmo Make — plugin-style **image / video** generation on **MLX** (Apple Silicon) and **CUDA** (NVIDIA). Split stack: FastAPI + Vue 3 SPA + CLI + SQLite. Models are declared in JSON and implemented under `backend/engine/families/`.

---

## Quick reference

| Item | Location |
|------|----------|
| API entry | `backend/main.py` |
| Frontend entry | `frontend/index.html` → Vite → `frontend/src/` |
| Infinite canvas | `frontend/src/components/studio/InfiniteCanvas.vue` + `useCanvasStore.ts` |
| Canvas sessions API | `backend/api/routes/canvas.py` → `/api/canvas/sessions` |
| Canvas edge/staging utils | `frontend/src/utils/canvasEdges.ts` (`make frontend-canvas-unit`) |
| Production UI build | `out/frontend/dist/` (`frontend/vite.config.ts`) |
| Launch / stop | `make dev` / `make start` / `make stop` → `scripts/start.sh`, `scripts/stop.sh` |
| Dev ports | Backend **7800**, frontend **5800** (`78xx`/`58xx`, suffix `00` = Studio) |
| Model registry (git / factory) | `default_config/models_registry.json` |
| Workspace pointer (local, gitignored) | `default_config/workspace.pointer.json` |
| Model registry (runtime) | `{workspace}/config/models_registry.json` — sync via `make sync-models-registry` |
| Family configs | `backend/engine/config/model_configs.py` |
| Transformer registry | `backend/engine/_transformer_registry.py` |
| API contracts | `backend/core/contracts.py` |
| Media interfaces | `backend/core/media_interfaces.py` (`IImageEngine`, `IVideoEngine`) |
| Pipeline runtime contracts | `backend/engine/contracts/` |
| Engine dispatch | `backend/engine/sessions/engine_dispatch.py` |
| Engine sessions | `backend/engine/sessions/` (`ImageSession`, `VideoSession`, …) |
| Catalog loader + DTO | `backend/catalog/` |
| Task observability | `backend/observability/` — SSE `trace`, `GET /graph`, `POST /diagnose`; dev `GET /diagnostic` |
| Model cache | `backend/engine/cache.py` |
| LLM assistant (engine) | `backend/engine/llm/` — sanitize / vision; not generation inference |
| Catalog `families` → FamilySpec | `backend/catalog/family_spec_loader.py` |
| Asset lineage | `backend/engine/lineage.py` |
| Runtime | `backend/engine/runtime/` (`MLXContext`, `CudaContext`, `mlx_runtime`, `mlx_dtype`) |
| Task kinds | `backend/core/task_kinds.py` (do not hardcode kind strings) |
| Cursor rules | `.cursor/rules/model-migration.mdc`, `.cursor/rules/no-silent-degrade.mdc`, `.cursor/rules/models-registry-maintain.mdc` |
| Engine architecture (single doc) | `docs/engine_architecture.md` |
| Bundle layout (T5 paths, ready assert) | `backend/engine/common/bundle/layout.py` |
| Pipeline progress + graph step logs | `backend/engine/pipelines/pipeline_progress.py` |
| Registry profiles (expand / shrink) | `backend/core/registry_profiles.py` |
| Bundle manifest + family contracts | `backend/core/bundle_manifest.py` |
| Desktop | `desktop/`, `make dev-desktop`, `make pack-macos-desktop` / `pack-linux-desktop` / `pack-windows-desktop` |

### Hardcoded paths

- Models: `./models/`
- LoRAs: `./models/Lora/`
- Outputs: `./outputs/`
- Factory / pointer: `default_config/` (`models_registry.json`, `presets.json`, `workspace.pointer.json`); runtime settings/registry under `{workspace}/config/`
- DB: `db/studio.db` (WAL; no runtime `ALTER` — reset DB + `outputs/` if schema drifts)

### Environment

- **Python** 3.11+, venv at `.venv/`
- **Key packages**: `fastapi`, `uvicorn`, `mlx`, `Pillow`, `transformers`, `safetensors` (+ `torch` when using CUDA)
- **Benchmark venv**: `tests/benchmark/venv/` via `make bench-setup` (isolated from app venv)
- **Platform detect**: `backend/engine/platform.py` — `["mlx"]` on darwin arm64 with mlx; `["cuda"]` when `torch.cuda.is_available()`; **fail loud** if a model’s `backends` entry is not satisfied

---

## Governance anchors (must follow)

Three product-level constraints (also in `.cursor/rules/*.mdc`):

1. **Layering / plugin model** — Reuse `backend/engine/common/`; new models = registry + `model_configs` + `families/<family>/` + `_transformer_registry`. **No** `family == …` branches in `ImagePipeline` / `VideoPipeline` for business logic.
2. **Contract API + CLI** — Routes/CLI only through contracts + `IImageEngine` / `IVideoEngine`. Per-model behavior via registry `actions` / `parameters` and Transformer polymorphism + Hooks — **not** `model_id` switches in `backend/api/routes/` or `backend/cli/`.
3. **RuntimeContext** — Hot paths use `ctx.*`; literal `import mlx` only in `backend/engine/runtime/` or `*_mlx.py` / `*_cuda.py`; literal `import torch` only in `*_cuda.py` (plus `runtime/cuda.py`). **MLX hot path (`*_mlx.py`) must not import torch or `*_cuda` modules** — native MLX or fail loud (`make check-engine-governance --rule mlx-torch`). Missing backend → **explicit error**, no silent fallback.

| Dimension | Acceptance |
|-----------|------------|
| Plugin | New image family: JSON + `model_configs` + `families/<family>/` + registry; no new Pipeline `family` branches |
| Family size | `families/<family>/` ≤ **8 logical units**; `stem.py` + `stem_mlx.py` + `stem_cuda.py` = **1** unit |
| API/CLI | Extend contracts + route/CLI first; REST and CLI stay aligned |
| Dual platform | Multi-`backends` models must run on each declared runtime or fail loud; MLX hot path no torch |
| Engine LOC | Refactors under `backend/engine/` should be **net delete or neutral** (bugfix exceptions documented in PR); no new `vae_codecs/` / `video_codecs/` wrapper trees — use `vae_codec_registry.py` / `video_codec_registry.py` only |

### Fail loud (default)

No silent degradation on generation, load, or registry parse failures. Use `RuntimeError` / clear HTTP errors + `t(key, locale)` + task logs/SSE. Documented user-consent toggles are the **only** allowed fallback (off by default). See `.cursor/rules/no-silent-degrade.mdc`.

---

## Architecture

### Layering

```
REST API / CLI
    ↓  contracts + interfaces only
TaskScheduler (global single queue, serial worker)
    ↓  IImageEngine / IVideoEngine / IAudioEngine
DanQingImageEngine / DanQingVideoEngine / DanQingAudioEngine
    ↓  engine_dispatch → Session (fail loud if no FamilyPlugin route)
ImageSession / VideoSession / AudioSession / UpscaleSession
    ↓  phased helpers + FamilyPlugin backbone
    ↓  RuntimeContext + TransformerBase + common/
V3TaskStore + SQLiteAssetStore
```

### Modular directories

| Path | Role |
|------|------|
| `backend/api/routes/` | REST by media (`images`, `videos`, `tasks`, `assets`, `registry`, …) |
| `backend/cli/` | `bin/danqing-*`, mirrors REST |
| `backend/scheduler/` | `TaskScheduler` |
| `backend/catalog/` | Registry schema v3 loader, `CatalogResponse`, migrate script |
| `backend/observability/` | RunTrace, SSE trace, graph/diagnose API; dev diagnostic bundle |
| `backend/engine/sessions/` | Orchestration + `engine_dispatch.py` |
| `backend/engine/pipelines/` | Phased helpers + pipeline helper objects (ctx/registry bundle) |
| `backend/engine/families/<family>/` | Per-family transformer, weights, `plugin.py` |
| `backend/engine/runtime/` | MLX / CUDA contexts |
| `backend/engine/common/` | Subpackages: ops, model, bundle, codecs (root = facade only) |
| `backend/engine/cache.py` | ModelCache (LRU + TTL) |
| `backend/engine/lineage.py` | Asset lineage helpers |
| `backend/persistence/` | SQLite stores |
| `backend/core/` | Interfaces, contracts, container, i18n |
| `backend/services/` | Settings, download |

### Replaceable components

- **RuntimeContext** — `MLXContext` / `CudaContext`
- **Scheduler** — `FlowMatchEulerScheduler`, `LinearScheduler`, …
- **VAEDecoder** — `scaling_factor`, `shift_factor`, `pytorch_compatible`
- **TextEncoder** — T5, Qwen3, CLIP, … (by `config.encoder_type`)
- **ModelCache** — LRU in engine layer

### Model as plugin

Every DiT on `ImagePipeline` / `VideoPipeline` **extends `TransformerBase`** (`backend/engine/common/model/base.py`): unified `forward` / `load_weights` / `parameters`, optional Hooks, default `refine_cfg_noise`.

**Pipeline.run()** (registry-driven, no family branches):

1. Resolve model → family + `ModelConfig`
2. Text encoder from `config.encoder_type`
3. Scheduler from registry `parameters.scheduler`
4. Initial latents from `config.vae_scale`
5. Denoise loop: `model(latents, t, txt_embeds=…, sigmas=…)`; CFG + optional `refine_cfg_noise`
6. VAE decode (flags from weights/config, not `family` string)
7. `asset_store.create_from_file`

**Invariant**: Prefer changing registry, config, transformer, remap, registry wiring only. Pipeline changes = new **registry-driven** switches or polymorphic call sites only.

### Transformer hooks

| Hook | When | Typical use |
|------|------|-------------|
| `after_load_weights(bundle_root)` | After `load_weights` | LoRA merge |
| `prepare_conditioning(request, bundle)` | After text encode | ControlNet cond dict |
| `before_denoise(latents, timesteps, sigmas, **cond)` | Before denoise loop | Inject control signal |
| `step_callback(step_idx, latents, noise_pred)` | Each step | Dynamic strength / logging |

`refine_cfg_noise` is **polymorphic** (not a hook), called when registry `enable_cfg_renorm` and CFG apply.

### Pipeline rules

- Scalars/enums in **registry + `model_configs`**, not `if family` in Pipeline.
- Shape/operator math in **Transformer** methods.
- Hooks for **optional cross-cutting** features only.
- Do not register `actions` the model cannot serve; `IImageEngine.supports` rejects at entry.
- Weight-key / config flags for variants (e.g. flux2-style VAE prep), not `family` strings.

### Family layout (migration)

Default per family: `transformer.py`, `text_encoder.py` (if needed), `weights.py`. Dual platform: Go-style `stem.py` / `stem_mlx.py` / `stem_cuda.py` (one logical unit). See `.cursor/rules/model-migration.mdc`.

**Weights**: Implement `remap_<family>_weights` in `weights.py` and override `sanitize()` on the DiT (stem or inner impl) to call it before `load_weights` applies tensors. VAE may use long tables. MLX nested `parameters()` → explicit flat `_param_map` matching remap keys.

### Implemented families (indicative)

`flux1`, `flux2`, `z_image`, `fibo`, `qwen` (`qwen_image` in registry), `seedvr2`, `ltx`, `wan`, `ace_step`, … — source of truth is `_transformer_registry.py` + `models_registry.json`.

---

## New model checklist

Extended execution checklist: `docs/engine_architecture.md` §5–§6.

1. **`default_config/models_registry.json`** — then `make sync-models-registry` to `{workspace}/config/models_registry.json`. Fields: `family`, `engine`, `actions`, `parameters`, `versions`, `backends`, bilingual `name`/`description`. First workspace setup copies from `default_config/` if missing; reset via settings only (no startup merge). See `.cursor/rules/models-registry-maintain.mdc`.
2. **`backend/engine/config/model_configs.py`** — dataclass + `FAMILY_CONFIG_MAP`
3. **`backend/engine/families/<family>/transformer.py`** — registry-facing stem (`DelegatingDiTStem` or `TransformerBase`); hooks here, math in `transformer_mlx.py`
4. **`backend/engine/families/<family>/weights.py`** — `remap_<family>_weights` (if needed); wire via DiT `sanitize()`
5. **`backend/engine/_transformer_registry.py`** — `_TRANSFORMER`, `_TEXT_ENCODER`, optional `_IMAGE_LORA_MERGE` (image/video DiT)

**Audio (ACE-Step)** — no `ImagePipeline` / `_TRANSFORMER` row: use `MusicPipeline` + `backend/engine/families/ace_step/generation.py` (public entry; MLX/CUDA dispatch inside family). Register `AceStepConfig` in `model_configs.py`; map registry `actions.create` → `task_kinds.AUDIO_GENERATION` via `task_kind_for_registry_action()`.

**Verify**

```bash
python -m py_compile <touched files>
bin/danqing-generate --model <id> --prompt "test"    # image
bin/danqing-audio-generate --model ace-step-xl-sft --prompt "test" --duration 10 --output /tmp/t.wav
make bench-eval-smoke     # image eval L1+L2 (smoke profile)
make bench-eval           # image eval full prompt matrix
make verify-engine-stack
```

---

## Configuration

### Registry (`schema_version: 3`)

- Top-level `engines`: `danqing-image`, `danqing-video`, `danqing-audio`
- Per model: `media` (`image` \| `video` \| `audio`), **`actions`** (not legacy `capabilities`)
- Image actions: `create`, `rewrite`, `retouch`, `extend`, `upscale`
- Video actions: `create`, `animate`
- `parameters`: typed (`int`, `float`, `enum`, `bool`, `object`)
- `name` / `description`: `{ "zh", "en" }`

### Presets

`{workspace}/config/presets.json` (seeded from `default_config/presets.json`) — each preset needs **`applies_to`** and **`media_scope`** (`image` \| `video`). UI filters by tab + scope.

### App settings

`{workspace}/config/.app_config.json` — `language`, `theme`, `default_model`, `mlx_memory_limit`, `queue_image_first`, …

---

## CLI vs REST API

| CLI | REST | Engine |
|-----|------|--------|
| `bin/danqing-generate` | `POST /api/images/generations` | `IImageEngine.generate()` |
| `bin/danqing-edit` | `POST /api/images/edits` | `IImageEngine.edit()` |
| `bin/danqing-upscale` | `POST /api/images/upscales` | `IImageEngine.upscale()` |
| `bin/danqing-video-generate` | `POST /api/videos/generations` | `IVideoEngine.generate()` |
| `bin/danqing-video-edit` | `POST /api/videos/edits` | `IVideoEngine.edit()` |
| `bin/danqing-video-upscale` | `POST /api/videos/upscales` | `IVideoEngine.upscale()` |

Engines: `ModelRegistry` + `EngineRegistry` → `DanQingImageEngine` / `DanQingVideoEngine`; runtime from registry `backends` via `_resolve_runtime`.

---

## API endpoints (catalog)

### Images / video

- `POST /api/images/generations` | `edits` | `upscales`
- `POST /api/videos/generations` | `edits` | `upscales`

### Tasks / queue

- `GET /api/tasks` (= `/api/tasks/list`) — `limit`, `offset`, `kind`, `status`, `since`
- `GET /api/tasks/{id}` — `queue_position`, `estimated_*`, `model`, `error`, …
- `GET /api/tasks/{id}/logs` — paginated logs
- `PATCH /api/tasks/{id}` — **`queued` only**: `{ "priority": "normal" \| "high" }`
- `DELETE /api/tasks/{id}` — cancel
- `GET /api/tasks/{id}/stream` — SSE: `log`, `progress`, `trace`, `status`, `result`, `done`
- `GET /api/tasks/{id}/graph` — pipeline DAG snapshot (product UI)
- `POST /api/tasks/{id}/diagnose` — LLM task diagnosis (local LLM required)
- `GET /api/tasks/{id}/diagnostic` — dev/agent diagnostic bundle
- `GET /api/queue` — snapshot; `estimated_wait_seconds` on queued items

`queue_image_first` in settings: image tasks dequeue before video.

### Assets / gallery

- `POST /api/assets` | `GET /api/assets` | `GET /api/assets/{id}/file` | `…/thumbnail`
- `POST /api/assets/reconcile` — `{ "dry_run": true }` default
- `GET /api/gallery/images` | `POST /api/gallery/upload` | `DELETE /api/gallery/image?path=asset:{id}`
- Task metadata: steps, guidance, seed, width/height; video `duration_seconds` (ffprobe or `num_frames/fps`)
- Lineage fields on assets: `parent_asset_id`, `relation_type` (canvas edges + gallery lineage)

### Canvas sessions

- `GET /api/canvas/sessions?media=image|video|audio` — list sessions
- `GET /api/canvas/sessions/{id}` — session row (`state`: items, viewport, staging, overlays, edges, `composer_snapshot`)
- `POST /api/canvas/sessions` — create (`media`, `title`, optional `state`)
- `PUT /api/canvas/sessions/{id}` — update title/state
- `DELETE /api/canvas/sessions/{id}` — delete
- `POST /api/chat/describe-node` — AI node note (vision when available)

### Registry / models / system

- `GET /api/registry` — full registry + `_index`
- `GET /api/models` — filter `media`, `action`, `installed`
- `GET /api/models/{id}` | `POST …/install` | `POST /api/models/install-batch` | `DELETE …/versions/{version_key}`
- `GET /api/presets` (read); writes via `/api/settings/presets`
- `GET /api/adapters` — LoRA index; `for_model` filter
- `GET /api/system/health` — `mlx` / `cuda` probe, GPU memory
- `GET /api/system/metrics` | `GET /api/settings/system`

### Audio

- `POST /api/audios/generations` — `AUDIO_GENERATION` → `DanQingAudioEngine` → `MusicPipeline` (ACE-Step `ace-step-xl-sft`, MLX + CUDA)
- `POST /api/audios/edits` — `AUDIO_EDIT` (registry actions `cover` / `repaint`; engine must declare support)

Route sources: `backend/api/routes/*.py`. Live docs: `http://localhost:7800/docs`.

---

## Running

```bash
make dev                 # uvicorn --reload + Vite HMR
make start / make stop   # same as dev / stop scripts

make frontend-dev        # :5800 → proxy /api to :7800 (if running backend separately)
make frontend-build      # → out/frontend/dist/

make dev-desktop         # Tauri + Vite HMR (SKIP_BACKEND=1 for external API)
make pack-macos-desktop  # macOS MLX .app/.dmg
make pack-linux-desktop  # Linux CUDA AppImage/.deb
make pack-windows-desktop # Windows CUDA NSIS (on Windows)
make pack-linux-server   # Linux CUDA server tar.gz (headless)
```

---

## Makefile targets

| Target | Purpose |
|--------|---------|
| `bench-setup` | Benchmark venv (PickScore judge) |
| `bench-eval` / `bench-eval-smoke` / `bench-eval-case ID=` | image L1+L2 eval |
| `bench-eval-calibrate` | write golden PickScore baselines |
| `test-engine-unit` | `scripts/test_engine_unit.py` |
| `check-consistency` | registry/routes/i18n |
| `check-engine-rules` | unified engine governance (`check_engine_governance.py`) |
| `check-engine-imports` | alias: `--rule imports` |
| `check-engine-mlx-torch` | alias: `--rule mlx-torch` — `*_mlx.py` must not depend on torch / `*_cuda` |
| `check-models-registry-contracts` | alias: `--rule registry` |
| `check-engine-governance` | engine rules + consistency |
| `verify-engine-stack` | governance + unit tests |
| `check-engine-family-layout` | alias: `--rule layout` |
| `lint` | Python syntax |
| `frontend-install` / `frontend-dev` / `frontend-build` / `frontend-typecheck` / `frontend-canvas-unit` | frontend |
| `dev-desktop` | Tauri desktop dev (FastAPI + Vite HMR; aligned with danmo-work/inbox) |
| `pack-macos-desktop` | macOS Tauri `.app` / `.dmg` (MLX sidecar) |
| `pack-linux-desktop` | Linux Tauri AppImage / `.deb` (CUDA sidecar) |
| `pack-windows-desktop` | Windows CUDA NSIS installer (venv + Tauri; on Windows) |
| `pack-linux-server` | Linux CUDA `.tar.gz` headless server |
| `pack-windows-server` | Windows CUDA `.zip` headless server (optional) |
| `clean` | `scripts/clean_build.py` |

Makefile pattern: `pack-<platform>-desktop` (Tauri) or `pack-<platform>-server` (archives). Steps: `venv` \| `sidecar` \| `shell` \| `archive`. Legacy aliases (`pack-windows-desktop-release`, `desktop-bundle`, …) remain.

---

## Frontend (Vue 3 + Vite + TypeScript)

- Locales: `frontend/src/locales/zh.json`, `en.json`
- Vue I18n 9, `legacy: false`; use `useI18n()` in script setup
- Template: `$t('key')`; script: `$tt` from `@/utils/i18n`
- Model/preset names: `$mn`, `$md`, `$mvn`, `$pn` on `globalProperties`
- Storage keys: `@/utils/storage` → `DQ_STORAGE` (`dq-studio.*.v4`)
- API: `@/utils/api` (`export const api`)
- Stores: `@/stores/registry`, `@/stores/tasks` (queue poll + SSE)
- Memory soft warning: `@/composables/memoryHint` (`warnIfRiskyMemory`)
- Navigation: Vue Router `router.push({ name: '…' })` — no `window.DQStudioNav`
- Global task queue: `App.vue` + `TopNav.vue` + `useTasksStore`; `openGlobalTaskQueue()` from `@/utils/appEvents`

### Frontend UI (`@danqing/dq-ui`)

- No Element Plus: templates use `Dq*` only; tokens from `@danqing/dq-tokens` (`--dq-*`).
- Shell layout: `App.vue` → `.dq-app-header` + `.dq-app-main`; `TopNav` is native `<nav class="dq-top-nav-menu">`.
- Governance: `make check-frontend-governance` (or `check-ep-boundary` / `check-theme-legacy` aliases). See `frontend/DQ-UI.md`.

### App.vue routing

`activePage` is a **local `ref`** synced from `route.name` via `watch` — do not bind cross-module route refs directly.

### Infinite canvas (Studio Canvas)

Gallery **grid** vs **canvas** toggle on image/video/audio create views (`StudioGalleryFilters`). Canvas is not a separate asset store — nodes reference gallery `asset:` paths; layout and composer bindings live in per-media canvas sessions (SQLite via `CanvasSessionStore`).

| Piece | Path |
|-------|------|
| Orchestrator | `InfiniteCanvas.vue` |
| Per-media store | `composables/useCanvasStore.ts` (singleton per `image` \| `video` \| `audio`) |
| Viewport / items / edges | `CanvasViewport.vue`, `CanvasItem.vue`, `CanvasEdges.vue` |
| Toolbars / panels | `CanvasToolbar.vue`, `CanvasLayerPanel.vue`, `CanvasSessionGraph.vue`, `CanvasLineageSidebar.vue` |
| Staging / edges math | `utils/canvasStaging.ts`, `utils/canvasEdges.ts` |
| Import/export | `utils/canvasImport.ts`, `utils/canvasExport.ts` |
| Create views | `ImageCreateView.vue`, `VideoCreateView.vue`, `AudioCreateView.vue` |

Workflow: import → select node → Composer fills params → generate lands in **staging** → lineage edges from `parent_asset_id` / `relation_type`. User-facing shortcuts: [README.md](README.md) → Infinite canvas workflow.

---

## Backend i18n

- `backend/core/i18n.py` — `t(key, locale, **params)`, `resolve_locale(Accept-Language)`
- Messages: `default_config/locales/zh.json`, `en.json`
- Registry/presets: bilingual objects in JSON (not legacy `name_en` fields)

---

## Desktop packaging

- Dev: `make dev-desktop` (`scripts/start_desktop.sh`; Vite via Tauri `beforeDevCommand`)
- Release: `make pack-macos-desktop` \| `pack-linux-desktop` \| `pack-windows-desktop`  
  (scripts: `pack_desktop_macos.sh` / `pack_desktop_linux.sh` / `pack_desktop_windows.sh` — same names as danmo-work/inbox)
- Artifacts: `out/frontend/dist/`, `out/sidecar/danqing-api/`, `out/desktop/bundle/`, `out/dist/*.tar.gz` (Linux server)
- macOS: `DANQING_PYINSTALLER_PROFILE=mlx` (no torch / `*_cuda`)
- Linux/Windows desktop: `DANQING_PYINSTALLER_PROFILE=cuda` (no MLX / `*_mlx`)
- Headless server archives: `make pack-linux-server` / `pack-windows-server`
- CI: `.github/workflows/release.yml` (macOS dmg + Linux AppImage/deb + Linux tar.gz + Windows NSIS)
- Sidecar env: `DANQING_HTTP_HOST`, `DANQING_HTTP_PORT`, `DANQING_USER_DATA_DIR`
- New engine modules must be reachable from `scripts/build_sidecar.py` / PyInstaller hooks

---

## Gotchas

- **First run is slow** — model load into GPU memory
- **DB schema** — no migrations; delete `db/studio.db` (+ outputs) to reset
- **`scripts/start.sh`** — browser dev: venv + uvicorn --reload + Vite; **`scripts/start_desktop.sh`** — Tauri + Vite HMR; release UI path is `out/frontend/dist/` (see `backend/main.py::_resolve_frontend_static_dir`)
- **Add model** — full 5-step checklist above, not registry-only
- **OpenAI-compatible API** (if added later) — separate `/v1/...` adapter; must delegate to same contracts/engines; do not replace resource-style `/api/images/*` routes
- **Contributors** — run `make verify-engine-stack` before PR
- **Structural guide (FLUX.1 create)** — `ImageGenerationRequest.structural_guide` only on `flux1*` + text-to-image; Canny/Depth/Redux preprocess in pipeline; companion LoRA auto-injected for Canny/Depth. UI: `useStructuralGuide.ts` + Composer advanced; not combinable with reference img2img.
- **FLUX Fill (retouch/extend)** — `flux-fill-controlnet` only; `ImagePipeline._run_flux1_fill_edit` (384-dim patch concat). Retouch needs `mask_asset_id`; extend uses `ExtendSpec`. CLI: `danqing-edit --operation retouch|extend` with `--mask-image` / `--extend-directions`.
- **ControlNet CUDA** — structural guide + Fill are **MLX-only** today (`backend/engine/families/flux1/structural.py`; registry `backends: ["mlx"]` on controlnet rows). CUDA paths are **placeholders** until a unified batch (`families/flux1/transformer_cuda.py` + pipeline hooks). Fail loud on `CudaContext`; do not silent-fallback.

---

## Reference

- [README.md](README.md) — user documentation (EN + 中文)
- [default_config/models_registry.json](default_config/models_registry.json)
- [backend/core/interfaces.py](backend/core/interfaces.py)
- [tests/benchmark/README.md](tests/benchmark/README.md)
