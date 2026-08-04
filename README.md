# Danmo Make

Language: **English** | [中文](README_zh.md)

Local **image / video / audio** studio on **MLX** (Apple Silicon) and **CUDA** (NVIDIA). Split stack: FastAPI + Vue 3 SPA + CLI + SQLite, with bilingual UI and **fail-loud** errors (no silent downgrades).

Product name: **Danmo Make**. Runtime / CLI / sidecar ids stay `danqing-*` / `DanQing*Engine` for upgrade compatibility ([AGENTS.md](AGENTS.md) → Naming boundary).

| | |
|---|---|
| **Contributors / coding agents** | [AGENTS.md](AGENTS.md) |
| **Desktop (Tauri 2)** | [desktop/README.md](desktop/README.md) |
| **Engine architecture** | [docs/engine_architecture.md](docs/engine_architecture.md) |
| **Image benchmarks** | [tests/benchmark/README.md](tests/benchmark/README.md) |
| **Releases** | [GitHub Releases](https://github.com/danmo-ai/danmo-make/releases) |

---

## Features

- **Dual runtime** — `MLXContext` on Apple Silicon; `CudaContext` when PyTorch CUDA is available (per-model `backends` in the registry).
- **Plugin models** — New families = registry JSON + `model_configs` + `families/<family>/` + `_transformer_registry`; pipelines stay family-agnostic.
- **Contract API + CLI** — Routes/CLI only through contracts + `IImageEngine` / `IVideoEngine` / `IAudioEngine`.
- **Global queue** — One serial worker for image / video / audio; SSE progress, priority, persistent logs.
- **Studio UI** — Vue 3 + Vite + TypeScript + `@danqing/dq-ui`; Create / Gallery / Models / Settings; bilingual registry names.
- **Infinite canvas** — Grid and canvas share one asset library; sessions, lineage edges, staging, composer bindings.
- **Audio** — ACE-Step music generation (`danqing-audio`, MLX + CUDA) via `MusicPipeline`.
- **MCP for agents** — Streamable HTTP at `/mcp/` (tools: `list_models`, `generate_*`, `edit_*`, …). Pair with **Danmo Work** builtin `danmo-make` expert (bound-only).
- **Desktop apps** — macOS MLX `.dmg`; Windows **portable zip** (CUDA thin, first-run setup); Linux CUDA **server** tar.gz.

---

## Requirements

| Platform | Notes |
|----------|--------|
| **macOS (Apple Silicon)** | Primary; MLX via Metal |
| **Linux / Windows + NVIDIA** | CUDA thin desktop/server; first launch installs torch into a local runtime venv |
| **Python** | 3.11+ (repo `.venv/` for web/dev) |
| **RAM** | 32 GB+ recommended for large models |
| **Node.js** | Frontend + desktop packaging |
| **ffmpeg / ffprobe** | Recommended for video thumbnails / duration |

Missing backends or unsupported actions **fail loudly** — no silent CLI/model fallback.

---

## Quick start

### Install (web / CLI)

```bash
git clone https://github.com/danmo-ai/danmo-make.git
cd danmo-make

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run (web)

```bash
make dev      # uvicorn --reload (:7800) + Vite HMR (:5800)
# make start / make stop
```

- UI: **http://localhost:5800** (proxies `/api` → :7800)
- API / Swagger: **http://localhost:7800/docs**
- MCP: **http://127.0.0.1:7800/mcp/** (trailing slash)

### Dev ports (Danmo line)

| Product | Backend | Frontend |
|---------|---------|----------|
| **Make (this repo)** | 7800 | 5800 |
| Work (Teams) | 7801 | 5801 |
| Inbox (Mail) | 7802 | 5802 |

Override: `DQ_BACKEND_PORT`, `DQ_FRONTEND_PORT`.

### Desktop

```bash
make pack-macos-desktop    # .app / .dmg (MLX sidecar)
make pack-windows-desktop  # *-portable.zip (CUDA thin; build on Windows)
make pack-linux-server     # CUDA thin server tar.gz
```

Windows users: unzip the **portable** build to a short path (e.g. `C:\DanmoMake`) and run the exe — not an NSIS installer. First launch opens runtime setup (torch not in the zip).

Details: [desktop/README.md](desktop/README.md).

### CLI

```bash
bin/danqing-generate --model z-image-turbo --prompt "a cat on a windowsill"
bin/danqing-edit --model <id> --image input.png --prompt "add a hat" --operation rewrite
bin/danqing-video-generate --model <id> --prompt "ocean waves at sunset"
bin/danqing-audio-generate --model ace-step-xl-sft --prompt "lofi beat" --duration 10 --output /tmp/t.wav
bin/danqing-mcp   # stdio MCP host; discovers port via env / ~/.danmo-make/api.port / 7800
```

CLI ↔ REST: [AGENTS.md](AGENTS.md#cli-vs-rest-api).

---

## Data layout

| Path | Role |
|------|------|
| `~/.danmo-make/` | Control plane: workspace pointer, app config, logs, `api.port`, CUDA `runtime-venv` |
| `{workspace}/config/` | Runtime `models_registry.json`, presets (seeded from `default_config/`) |
| `{workspace}/models/` | Weights / LoRAs |
| `{workspace}/outputs/` | Generations |
| `{workspace}/db/studio.db` | Tasks + assets (SQLite WAL) |

Dev (`make dev`, no pointer): media root can stay the repo (`./models`, `./outputs`, …); control plane is still `~/.danmo-make`.

Override control plane: `DANQING_USER_DATA_DIR`.

---

## Agent / MCP

With the API up:

| | |
|---|---|
| Endpoint | `http://127.0.0.1:7800/mcp/` (streamable HTTP; `/mcp` is normalized to `/mcp/`) |
| Stdio | `bin/danqing-mcp` |
| Auth | Loopback free; remote needs `Authorization: Bearer <key>` — **separate** HTTP vs MCP keys (**Settings → Integrations**) |

`list_models` filters by API **action** (`generate` / `edit` / `upscale`; aliases `create`→`generate`). Cards expose `actions` and `type`; LoRA adapters are omitted. Use `action=generate` before `generate_image` / `generate_video`.

Danmo Work: enable builtin connector **danmo-make** + expert (bound-only, not ambient).

Env overrides: `DANQING_HTTP_API_KEY`, `DANQING_MCP_API_KEY`, `DANQING_MCP_BASE_URL`.

---

## Studio ↔ model `actions`

Tabs only list models that declare the required registry action (seeded from `default_config/models_registry.json`).

### Image

| Tab | Registry action | API |
|-----|-----------------|-----|
| Text-to-image | `create` | `POST /api/images/generations` |
| Instruct / reference edit | `rewrite` | `POST /api/images/edits` |
| Inpaint | `retouch` | `POST /api/images/edits` |
| Outpaint | `extend` | `POST /api/images/edits` |
| Upscale | `upscale` | `POST /api/images/upscales` |

API / MCP surface maps `create`→`generate`, rewrite/retouch/extend→`edit`.

### Video

| Tab | Registry action | API |
|-----|-----------------|-----|
| Text-to-video | `create` | `POST /api/videos/generations` |
| Image-to-video | `animate` | `POST /api/videos/edits` |

### Audio

| Action | API |
|--------|-----|
| Text-to-music (`create`) | `POST /api/audios/generations` |
| Cover / repaint | `POST /api/audios/edits` |

### Infinite canvas (short)

Create → **Canvas** view: import (`I`), generate into staging (`S` snap), lineage edges (`E` / `Y` / `G`). Sessions via `/api/canvas/sessions`. See [AGENTS.md](AGENTS.md) for the full shortcut table.

### ControlNet (FLUX.1)

Structural guide on text-to-image for `flux1*` (Canny / Depth / Redux); Fill model for retouch/extend. MLX-first today; CUDA paths fail loud if unimplemented. Details in [AGENTS.md](AGENTS.md) → Gotchas.

---

## Project layout

```
danmo-make/
├── backend/
│   ├── api/routes/       # REST
│   ├── mcp/              # MCP tools + model cards
│   ├── cli/              # bin/danqing-*
│   ├── core/             # contracts, interfaces, i18n
│   ├── engine/           # pipelines, families, runtime, common
│   ├── persistence/      # SQLite stores
│   ├── scheduler/        # TaskScheduler
│   └── main.py
├── frontend/             # Vue 3 SPA
├── desktop/              # Tauri 2 shell
├── bin/                  # CLI + danqing-mcp
├── default_config/       # factory registry / presets / locales
├── scripts/              # pack, gates, release
├── tests/benchmark/
├── docs/
└── out/                  # build artifacts (gitignored)
```

---

## Architecture

```
REST / MCP / CLI
        ↓
TaskScheduler  (single global queue)
        ↓
DanQingImageEngine / DanQingVideoEngine / DanQingAudioEngine
        ↓
Pipelines + FamilyPlugin + RuntimeContext (MLX | CUDA)
        ↓
V3TaskStore + SQLiteAssetStore
```

Add a model: registry → `model_configs` → `families/<family>/` → `weights.py` → `_transformer_registry`. Checklist: [AGENTS.md](AGENTS.md#new-model-checklist).

---

## Release artifacts (CI)

Tag `v*` → [`.github/workflows/release.yml`](.github/workflows/release.yml):

| Platform | Artifact |
|----------|----------|
| macOS Apple Silicon | MLX `.dmg` / `.app` |
| Windows x64 | CUDA thin **`*-portable.zip`** |
| Linux x64 | CUDA thin **server** `.tar.gz` |

```bash
make pack-macos-desktop
make pack-windows-desktop    # on Windows
make pack-linux-server
```

Do not mix MLX and CUDA in one bundle.

---

## Configuration

**App settings** — under the control plane / workspace (`~/.danmo-make/…` or `{workspace}/config/.app_config.json`): language, theme, default model, memory limit, queue policy, integration API keys.

**Registry** — `{workspace}/config/models_registry.json` (`schema_version: 3`). Sync factory → workspace: `make sync-models-registry`.

**Useful env**:

```bash
HF_ENDPOINT=https://hf-mirror.com
MLX_METAL_MEMORY_LIMIT=120
DANQING_USER_DATA_DIR=~/.danmo-make
# Prefer Settings → Integrations over plaintext keys:
# DANQING_HTTP_API_KEY=...
# DANQING_MCP_API_KEY=...
```

---

## Development

| Command | Purpose |
|---------|---------|
| `make dev` / `stop` | API + Vite |
| `make dev-desktop` | Tauri + Vite (`SKIP_BACKEND=1` if API already up) |
| `make frontend-build` | → `out/frontend/dist/` |
| `make verify-engine-stack` | Governance + engine unit tests |
| `make check-consistency` | Registry / routes / i18n / frontend gates |
| `make bench-eval-smoke` | Image eval smoke |
| `make clean` | Remove `out/` |

---

## API overview

| Area | Endpoints |
|------|-----------|
| Images | `POST /api/images/generations` \| `edits` \| `upscales` |
| Videos | `POST /api/videos/generations` \| `edits` \| `upscales` |
| Audios | `POST /api/audios/generations` \| `edits` |
| Tasks | `GET/PATCH/DELETE /api/tasks/{id}`, SSE `…/stream`, `…/diagnose` |
| Assets / canvas | `/api/assets`, `/api/canvas/sessions` |
| Models | `/api/models`, `/api/registry` |
| MCP | `/mcp/` |
| System | `/api/system/health`, `/api/settings/*` |

Interactive docs: **http://localhost:7800/docs**

---

## License

MIT
