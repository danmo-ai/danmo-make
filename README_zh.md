# Danmo Make

Language: [English](README.md) | **中文**

本地 **图像 / 视频 / 音频** 创作工作室。API 与引擎是 **Rust + ggml**（Metal / CUDA / Vulkan / CPU）。界面是 Vue 3。权重只认 **GGUF**（`bf16`、`q8_0`、`q4_k`）。safetensors 与 MLX 包不会加载。缺图或缺文件就 **显式失败**。

产品名：**Danmo Make**。运行时标识仍是 `danqing-*`（见 [AGENTS.md](AGENTS.md)）。

| | |
|---|---|
| **贡献者 / Coding Agent** | [AGENTS.md](AGENTS.md) |
| **桌面版（Tauri 2）** | [desktop/README.md](desktop/README.md) |
| **引擎架构** | [docs/engine_architecture.md](docs/engine_architecture.md) |
| **发布包** | [GitHub Releases](https://github.com/danmo-ai/danmo-make/releases) |

---

## 特性

- **ggml 运行时** — 一张图反复算，后端是 Metal、CUDA、Vulkan 或 CPU。
- **GGUF 权重** — 只下载 `bf16`、`q8_0`、`q4_k`。张量名就是图参数名，没有 remap。VAE 保持 F16/BF16。
- **契约 API** — Vue 调 `danqing-api`。还没建图的族会让任务失败。
- **全局单队列** — 图像 / 视频 / 音频串行；SSE 进度、优先级、日志落库。
- **Studio UI** — Vue 3 + Vite + TypeScript + `@danqing/dq-ui`；创作 / 图库 / 模型 / 设置。
- **无限画布** — 网格与画布共用资产库；会话、谱系、落点、创作器绑定。
- **音频** — ACE-Step 与 DiffRhythm 已注册；图还没建，生成会显式失败。
- **Agent** — 走 HTTP API。`/mcp/` 和 `bin/danqing-mcp` 这版没有接上。
- **桌面发布** — macOS ggml Metal `.dmg`；Linux ggml 桌面 / 服务端 tar.gz。Windows 桌面包未接上；sidecar 用 `make pack-windows-sidecar`。

---

## 环境要求

| 平台 | 说明 |
|------|------|
| **macOS（Apple Silicon）** | Metal |
| **Linux** | CPU；有 `nvcc` 则 CUDA；有 Vulkan SDK 则 Vulkan |
| **Windows** | Vulkan sidecar；有 `CUDA_PATH` 或 `nvcc` 再加 CUDA |
| **Rust** | stable（`rustup`） |
| **内存** | 大模型建议 32 GB+ |
| **Node.js** | 前端与桌面打包 |

缺后端或不支持的 action **明确报错**，不做静默兼容回退。

---

## 安装发布包（推荐）

从 [GitHub Releases](https://github.com/danmo-ai/danmo-make/releases) 下载。Windows 包暂不提供。

### macOS（Apple Silicon）— `.dmg`

1. 打开 DMG，将 **Danmo Make.app** 拖入「应用程序」。
2. 首次打开可能被拦截（未公证 / 隔离属性）。可用 **按住 Control 点按 → 打开**，或：

```bash
xattr -dr com.apple.quarantine "/Applications/Danmo Make.app"
```

更多说明见 [desktop/README.md](desktop/README.md) →「已损坏，无法打开」。

### Linux（x86_64 + NVIDIA）— thin 服务端 `.tar.gz`

CI 产物为 **`danmo-make-linux-ggml-x86_64-<version>.tar.gz`**（无头 `danqing-api` + 已构建的界面）。`./run.sh` 启动这个二进制。设备由 `DANQING_GGML_BACKEND` 决定（`cpu`、`cuda`、`vulkan`、`metal`）。二进制没编进的后端会在启动时失败。

```bash
tar -xzf danmo-make-linux-ggml-x86_64-*.tar.gz
cd danmo-make-linux-ggml-x86_64-*

# 可选：
# export DANQING_HTTP_HOST=0.0.0.0
# export DANQING_HTTP_PORT=7800
# export DANQING_GGML_BACKEND=cuda

./run.sh
```

- 界面 / API：**http://127.0.0.1:7800** · 文档 **/docs**

Linux **桌面包**（AppImage / `.deb`）可本地打包（`make pack-linux-desktop`），**当前 CI 不挂载**。

---

## 快速开始（源码）

贡献者 / 本地开发用。普通用户请看上方 **安装发布包**。

### 安装（Web / CLI）

```bash
git clone --recurse-submodules https://github.com/danmo-ai/danmo-make.git
cd danmo-make
```

### 运行（Web）

```bash
make dev      # danqing-api (:7800) + Vite HMR (:5800)
# make start / make stop
```

- UI：**http://localhost:5800**（`/api` → :7800）
- API / Swagger：**http://localhost:7800/docs**

### 开发端口（Danmo 产品线）

| 产品 | 后端 | 前端 |
|------|------|------|
| **Make（本仓库）** | 7800 | 5800 |
| Work（Teams） | 7801 | 5801 |
| Inbox（Mail） | 7802 | 5802 |

可用 `DQ_BACKEND_PORT` / `DQ_FRONTEND_PORT` 覆盖。

### 从源码打包桌面 / 服务端

```bash
make pack-macos-desktop    # .app / .dmg（ggml Metal sidecar）
make pack-linux-desktop    # AppImage / .deb（ggml sidecar；本地）
make pack-linux-server     # 无头 ggml danqing-api + 界面
```

Windows 桌面包未接线。`make pack-windows-sidecar` 只打 ggml sidecar。详见 [desktop/README.md](desktop/README.md)。

生成走 HTTP API（`POST /api/images/generations` 及同组路由）。没有 `bin/danqing-*` 命令行。

---

## 数据布局

| 路径 | 作用 |
|------|------|
| `~/.danmo-make/` | 控制面：工作区指针、应用配置、日志、`api.port` |
| `{workspace}/config/` | 运行时注册表 / 预设（从 `default_config/` 种子） |
| `{workspace}/models/` | 权重 / LoRA |
| `{workspace}/outputs/` | 生成物 |
| `{workspace}/db/studio.db` | 任务 + 资产（SQLite WAL） |

开发（`make dev`、无指针）时媒体根可留在仓库（`./models` 等）；控制面仍为 `~/.danmo-make`。

覆盖控制面：`DANQING_USER_DATA_DIR`。

---

## Agent / MCP

这版没有 `/mcp/`，也没有 `bin/danqing-mcp`。设置里仍可以单独存一把 MCP 钥匙。Agent 走 HTTP API。本机 loopback 免密；远程要带 `Authorization: Bearer`，钥匙在 **设置 → 集成**（`DANQING_HTTP_API_KEY`）。

---

## 创作页 ↔ 模型 `actions`

标签页只列出工作区注册表中声明了对应 action 的模型。

### 图像

| 标签页 | 注册表 action | API |
|--------|---------------|-----|
| 文生图 | `create` | `POST /api/images/generations` |
| 指令 / 参考改图 | `rewrite` | `POST /api/images/edits` |
| 局部重绘 | `retouch` | `POST /api/images/edits` |
| 扩图 | `extend` | `POST /api/images/edits` |
| 放大 | `upscale` | `POST /api/images/upscales` |

注册表 `create` 对应 `POST /api/images/generations`。`rewrite`、`retouch`、`extend` 对应 `POST /api/images/edits`。

### 视频

| 标签页 | 注册表 action | API |
|--------|---------------|-----|
| 文生视频 | `create` | `POST /api/videos/generations` |
| 图生视频 | `animate` | `POST /api/videos/edits` |

**MiniMax H3（`minimax-h3-fl2va`）** — 已注册。ggml 图还没建，生成任务会显式失败。

### 音频

| Action | API |
|--------|-----|
| 文生音乐（`create`） | `POST /api/audios/generations` |
| 翻唱 / 局部重绘 | `POST /api/audios/edits` |

### 无限画布（摘要）

创作页 → **画布**：导入（`I`）、结果进落点（`S` 贴靠）、谱系（`E` / `Y` / `G`）。会话见 `/api/canvas/sessions`。快捷键全表见 [AGENTS.md](AGENTS.md)。

### ControlNet（FLUX.1）

结构引导和 Fill 还没建。带上这些参数的请求会让任务失败。见 [docs/engine_architecture.md](docs/engine_architecture.md)。

---

## 项目结构

```
danmo-make/
├── native/               # Rust：ggml 运行时、引擎、danqing-api
├── frontend/             # Vue 3
├── desktop/              # Tauri 2
├── default_config/
├── scripts/
├── docs/
└── out/
```

---

## 架构

```
Vue
  ↓ HTTP
danqing-api
  ↓
单 worker 队列（SQLite）
  ↓
图像 / 视频 / 音频流水线
  ↓
ggml 图
  ↓
按张量名 mmap GGUF
```

现在能跑的图像图是 Flux2（MM-DiT）和 Z-Image（DiT）。它们共用 Qwen3 文本塔、PNG 写出和 AutoencoderKL 解码器。Flux2 先对 packed latent 做 batch norm、2×2 unpatch 和 post-quant，再以 32 通道、scaling 1、shift 0 进入解码器。Z-Image 传入 16 通道、scaling 0.3611、shift 0.1159。只有 `decoder.conv.weight` 一张量的文件仍走 1×1 预览。共享代码只收参数，不认识家族 id。家族之间不互相引用。其余家族在检查完 bundle 之后显式失败。注册表里还是 `fp16` / `mlx-q4` / `mlx-q8` 的版本会要求重新下载 GGUF。没有 remap。原则见 [docs/engine_architecture.md](docs/engine_architecture.md)。

---

## 发版产物（CI）

打 `v*` tag → [`.github/workflows/release.yml`](.github/workflows/release.yml)。

| 平台 | CI 产物 |
|------|---------|
| macOS Apple Silicon | ggml Metal `.dmg` |
| Linux x86_64 | **`danmo-make-linux-ggml-*.tar.gz`** |
| Windows | `make pack-windows-sidecar`（Vulkan；有 CUDA 工具链再加 CUDA） |

```bash
make pack-macos-desktop
make pack-linux-desktop
make pack-linux-server
make pack-ggml-sidecar
```

---

## 配置

**应用设置** — 控制面 / 工作区下的 `.app_config.json`：语言、主题、默认模型、显存上限、队列策略、集成 API Key 等。

**注册表** — `{workspace}/config/models_registry.json`（`schema_version: 3`）。同步出厂默认：`make sync-models-registry`。

**常用环境变量**：

```bash
DANQING_HTTP_HOST=127.0.0.1
DANQING_HTTP_PORT=7800
DANQING_DB=db/studio.db
DANQING_REGISTRY=default_config/models_registry.json
DANQING_GGML_BACKEND=cpu
DANQING_USER_DATA_DIR=~/.danmo-make
# 优先在「设置 → 集成」创建钥匙，而不是长期写明文：
# DANQING_HTTP_API_KEY=...
# DANQING_MCP_API_KEY=...
```

---

## 开发

| 命令 | 用途 |
|------|------|
| `make dev` / `stop` | `danqing-api` + Vite |
| `make dev-desktop` | Tauri + Vite（已有 API 时 `SKIP_BACKEND=1`） |
| `make frontend-build` | → `out/frontend/dist/` |
| `make test` | ggml 后端的 `cargo test` |
| `make pack-ggml-sidecar` | 发布用 `danqing-api` |
| `make clean` | `cargo clean` 并清掉 `out/` |

---

## API 概览

| 领域 | 端点 |
|------|------|
| 图像 | `POST /api/images/generations` \| `edits` \| `upscales` |
| 视频 | `POST /api/videos/generations` \| `edits` \| `upscales` |
| 音频 | `POST /api/audios/generations` \| `edits` |
| 任务 | `GET/PATCH/DELETE /api/tasks/{id}`，SSE `…/stream`，`…/diagnose` |
| 资产 / 画布 | `/api/assets`，`/api/canvas/sessions` |
| 模型 | `/api/models`，`/api/registry` |
| 系统 | `/api/system/health`，`/api/settings/*` |

交互文档：**http://localhost:7800/docs**

---

## 许可证

MIT
