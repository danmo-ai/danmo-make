# Danmo Make

Language: **English** | [中文](README_zh.md)

Local **image / video / audio** studio. The API and engine are **Rust** on **ggml** (Metal, CUDA, Vulkan, CPU). The UI is a Vue 3 SPA. Weights are **GGUF** only (`bf16`, `q8_0`, `q4_k`); safetensors and MLX bundles are not loaded. **Fail-loud** on a missing graph or a missing GGUF file.

Product name: **Danmo Make**. Runtime / sidecar ids stay `danqing-*` ([AGENTS.md](AGENTS.md) → Naming boundary).

| | |
|---|---|
| **Contributors / coding agents** | [AGENTS.md](AGENTS.md) |
| **Desktop (Tauri 2)** | [desktop/README.md](desktop/README.md) |
| **Engine architecture** | [docs/engine_architecture.md](docs/engine_architecture.md) |
| **Releases** | [GitHub Releases](https://github.com/danmo-ai/danmo-make/releases) |

---

## Features

- **ggml runtime** — one graph, replayed per step, on Metal, CUDA, Vulkan, or CPU.
- **GGUF weights** — download `bf16`, `q8_0`, or `q4_k`. Tensor names are graph parameter names. There is no remap. VAE stays F16/BF16. Old safetensors and MLX packs are not read.
- **Contract API** — Vue talks to `danqing-api` (`POST /api/images/generations` and the sibling routes). Families that do not have a graph yet fail the task.
- **Global queue** — One serial worker for image / video / audio; SSE progress, priority, persistent logs.
- **Studio UI** — Vue 3 + Vite + TypeScript + `@danqing/dq-ui`; Create / Gallery / Models / Settings; bilingual registry names.
- **Infinite canvas** — Grid and canvas share one asset library; sessions, lineage edges, staging, composer bindings.
- **Audio** — ACE-Step and DiffRhythm are registered; their graphs are not built yet, so generation fails loud.
- **Desktop apps** — macOS ggml Metal `.dmg`; Linux ggml desktop / server tar.gz. Windows sidecar is Vulkan (CUDA when the toolkit is present).

---

## Requirements

| Platform | Notes |
|----------|--------|
| **macOS (Apple Silicon)** | Metal backend |
| **Linux** | CPU; CUDA if `nvcc` is present; Vulkan if the SDK is present |
| **Windows** | Vulkan sidecar; CUDA when `CUDA_PATH` or `nvcc` is present |
| **Rust** | stable toolchain (`rustup`) |
| **RAM** | 32 GB+ recommended for large models |
| **Node.js** | Frontend + desktop packaging |

Missing backends or unsupported actions **fail loudly** — no silent CLI/model fallback.

---

## Install from release (recommended)

Download from [GitHub Releases](https://github.com/danmo-ai/danmo-make/releases). Windows packs are temporarily unsupported.

### macOS (Apple Silicon) — `.dmg`

1. Open the DMG and drag **Danmo Make.app** into Applications.
2. First launch may be blocked (unsigned / quarantine). Prefer **Control-click → Open**, or:

```bash
xattr -dr com.apple.quarantine "/Applications/Danmo Make.app"
```

More detail: [desktop/README.md](desktop/README.md) → quarantine tip.

### Linux (x86_64 + NVIDIA) — thin server `.tar.gz`

CI ships **`danmo-make-linux-ggml-x86_64-<version>.tar.gz`** (headless `danqing-api` + built UI). `./run.sh` starts that binary. The device is `DANQING_GGML_BACKEND` (`cpu`, `cuda`, `vulkan`, `metal`). A backend the binary was not built with fails at startup.

```bash
tar -xzf danmo-make-linux-ggml-x86_64-*.tar.gz
cd danmo-make-linux-ggml-x86_64-*

# optional:
# export DANQING_HTTP_HOST=0.0.0.0
# export DANQING_HTTP_PORT=7800
# export DANQING_GGML_BACKEND=cuda

./run.sh
```

- UI / API: **http://127.0.0.1:7800** · docs **/docs**

Linux **desktop** (AppImage / `.deb`) is packable locally (`make pack-linux-desktop`) but **not** attached by current CI.

---

## Quick start (from source)

For contributors / local development. End users: use **Install from release** above.

### Install (web)

```bash
git clone --recurse-submodules https://github.com/danmo-ai/danmo-make.git
cd danmo-make
# Rust stable: https://rustup.rs/
# Node.js for the Vue app
```

### Run (web)

```bash
make dev      # danqing-api (:7800) + Vite HMR (:5800)
# make start / make stop
```

- UI: **http://localhost:5800** (proxies `/api` → :7800)
- API: **http://127.0.0.1:7800/api/system/health**

### Dev ports (Danmo line)

| Product | Backend | Frontend |
|---------|---------|----------|
| **Make (this repo)** | 7800 | 5800 |
| Work (Teams) | 7801 | 5801 |
| Inbox (Mail) | 7802 | 5802 |

Override: `DQ_BACKEND_PORT`, `DQ_FRONTEND_PORT`.

### Pack desktop / server (from source)

```bash
make pack-macos-desktop    # .app / .dmg (ggml Metal sidecar)
make pack-linux-desktop    # AppImage / .deb (ggml sidecar; local)
make pack-linux-server     # headless ggml danqing-api + UI
```

Windows desktop pack is not wired. `make pack-windows-sidecar` builds the ggml sidecar. Details: [desktop/README.md](desktop/README.md).

Generation goes through the HTTP API (`POST /api/images/generations` and the sibling routes). There is no `bin/danqing-*` CLI.

---

## Data layout

| Path | Role |
|------|------|
| `~/.danmo-make/` | Control plane: workspace pointer, app config, logs, `api.port` |
| `{workspace}/config/` | Runtime `models_registry.json`, presets (seeded from `default_config/`) |
| `{workspace}/models/` | Weights / LoRAs |
| `{workspace}/outputs/` | Generations |
| `{workspace}/db/studio.db` | Tasks + assets (SQLite WAL) |

Dev (`make dev`, no pointer): media root can stay the repo (`./models`, `./outputs`, …); control plane is still `~/.danmo-make`.

Override control plane: `DANQING_USER_DATA_DIR`.

---

## Agent / MCP

The streamable HTTP route `/mcp/` and `bin/danqing-mcp` are not in this build. Settings can still store a separate MCP key. Agents talk to the HTTP API. Loopback skips auth; a remote client needs `Authorization: Bearer` with the HTTP key from **Settings → Integrations** (`DANQING_HTTP_API_KEY`).

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

Registry `create` is `POST /api/images/generations`. `rewrite`, `retouch`, and `extend` are `POST /api/images/edits`.

### Video

| Tab | Registry action | API |
|-----|-----------------|-----|
| Text-to-video | `create` | `POST /api/videos/generations` |
| Image-to-video | `animate` | `POST /api/videos/edits` |

**MiniMax H3 (`minimax-h3-fl2va`)** — registered. Its ggml graph is not built, so a generate task fails loud.

### Audio

| Action | API |
|--------|-----|
| Text-to-music (`create`) | `POST /api/audios/generations` |
| Cover | `POST /api/audios/edits` |

### Infinite canvas (short)

Create → **Canvas** view: import (`I`), generate into staging (`S` snap), lineage edges (`E` / `Y` / `G`). Sessions via `/api/canvas/sessions`. See [AGENTS.md](AGENTS.md) for the full shortcut table.

### ControlNet (FLUX.1)

Structural guide and Fill are not built. A request that sets them fails the task. See [docs/engine_architecture.md](docs/engine_architecture.md).

---

## Project layout

```
danmo-make/
├── native/               # Rust workspace: ggml runtime, engine, danqing-api
├── frontend/             # Vue 3 SPA
├── desktop/              # Tauri 2 shell
├── default_config/       # factory registry / presets / locales
├── scripts/              # pack and dev
├── docs/
└── out/                  # build artifacts (gitignored)
```

---

## Architecture

```
Vue
  ↓ HTTP
danqing-api (Axum)
  ↓
Scheduler (one worker, SQLite tasks)
  ↓
Image / Video / Audio pipeline
  ↓
Family graph on ggml
  ↓
GGUF mmap by tensor name
```

Image graphs that run today are Flux2 (MM-DiT) and Z-Image (DiT). They share the Qwen3 tower, the PNG writer, and, for Z-Image, the AutoencoderKL decoder. Shared code takes parameters and does not know the family id. Families do not import each other. Every other family fails loud after the bundle check. A registry version that is still `fp16` / `mlx-q4` / `mlx-q8` fails the task and asks for a GGUF re-download. There is no key remap. Principles: [docs/engine_architecture.md](docs/engine_architecture.md).

---

## Release artifacts (CI)

Tag `v*` → [`.github/workflows/release.yml`](.github/workflows/release.yml).

| Platform | CI artifact |
|----------|-------------|
| macOS Apple Silicon | ggml Metal `.dmg` |
| Linux x86_64 | **`danmo-make-linux-ggml-*.tar.gz`** |
| Windows | `make pack-windows-sidecar` (Vulkan; CUDA when the toolkit is present) |

```bash
make pack-macos-desktop
make pack-linux-desktop
make pack-linux-server
make pack-ggml-sidecar
```

---

## Configuration

**App settings** — under the control plane / workspace (`~/.danmo-make/…` or `{workspace}/config/.app_config.json`): language, theme, default model, memory limit, queue policy, integration API keys.

**Registry** — `default_config/models_registry.json` (`schema_version: 3`). The API only advertises version keys `bf16`, `q8_0`, and `q4_k`. Older keys are ignored.

**Useful env**:

```bash
DANQING_HTTP_HOST=127.0.0.1
DANQING_HTTP_PORT=7800
DANQING_DB=db/studio.db
DANQING_REGISTRY=default_config/models_registry.json
DANQING_GGML_BACKEND=cpu
```

---

## Development

| Command | Purpose |
|---------|---------|
| `make dev` / `stop` | `danqing-api` + Vite |
| `make dev-desktop` | Tauri + Vite (`SKIP_BACKEND=1` if API already up) |
| `make frontend-build` | → `out/frontend/dist/` |
| `make test` | `cargo test` for the ggml backend |
| `make pack-ggml-sidecar` | Release `danqing-api` |
| `make clean` | `cargo clean` and remove `out/` |

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
| System | `/api/system/health`, `/api/settings/*` |

Interactive docs: **http://localhost:7800/docs**

---

## License

MIT
