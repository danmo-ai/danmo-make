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

CI ships **`danmo-make-linux-ggml-x86_64-<version>.tar.gz`** (headless `danqing-api` + built UI).

```bash
tar -xzf danmo-make-linux-ggml-x86_64-*.tar.gz
cd danmo-make-linux-ggml-x86_64-*

# optional:
# export DANQING_HTTP_HOST=0.0.0.0
# export DANQING_HTTP_PORT=7800

./run.sh
```

- UI / API: **http://127.0.0.1:7800** · docs **/docs** · MCP **/mcp/**
- Repair: `./run.sh --repair-runtime` · reinstall: `./run.sh --reinstall-runtime` · status: `./run.sh --status-runtime`
- Bundle also includes `README.txt` and `danqing-runtime-setup`

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
make pack-macos-desktop    # .app / .dmg (MLX Metal sidecar)
make pack-linux-desktop    # AppImage / .deb (mlx[cuda]; local)
make pack-linux-server     # Linux MLX thin server tar.gz
```

Windows is temporarily unsupported. Details: [desktop/README.md](desktop/README.md).

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
| `~/.danmo-make/` | Control plane: workspace pointer, app config, logs, `api.port`, optional Linux `runtime-venv` |
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

**MiniMax H3 (`minimax-h3-fl2va`)** — joint audio+video FL2VA on MLX (Metal / mlx[cuda], **mlx >= 0.32.3**, **mlx-vlm >= 0.7.6**, **transformers >= 5.19**). Correctness aligned with [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx) (scheduler `t=1−σ`, aspect-normalized RoPE, keyframe posterior sample). Generation is **compute-bound**; mlx-q4/q8 mainly reduces footprint. `balanced` / `quality` / `oracle` run the full Euler trajectory. `draft` only enables Turbo LoRA (6–8 steps recommended). Non-turbo clips are usually run at 8–16 steps; the registry default stays 30 so the sampling integral does not change silently. Denoiser reuse, fewer DiT layers, and a smaller internal canvas are lossy and are not turned on by presets. Tests: `make test-h3-parity`; smoke benchmark (installed bundle): `make bench-h3-smoke`.

### Audio

| Action | API |
|--------|-----|
| Text-to-music (`create`) | `POST /api/audios/generations` |
| Cover | `POST /api/audios/edits` |

### Infinite canvas (short)

Create → **Canvas** view: import (`I`), generate into staging (`S` snap), lineage edges (`E` / `Y` / `G`). Sessions via `/api/canvas/sessions`. See [AGENTS.md](AGENTS.md) for the full shortcut table.

### ControlNet (FLUX.1)

Structural guide on text-to-image for `flux1*` (Canny / Depth / Redux); Fill model for retouch/extend. MLX stack only. Details in [AGENTS.md](AGENTS.md) → Gotchas.

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

Flux2 is the only graph that runs today, and it is a smoke velocity block checked against a tiny GGUF, not the full MM-DiT. Every other family fails loud with “no ggml graph”. A model whose registry versions are still `fp16` / `mlx-q4` / `mlx-q8` fails the task and asks for a GGUF re-download. There is no key remap.

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
| MCP | `/mcp/` |
| System | `/api/system/health`, `/api/settings/*` |

Interactive docs: **http://localhost:7800/docs**

---

## License

MIT
