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
- **音频** — ACE-Step 文生音乐（`danqing-audio`，MLX）。
- **MCP（给 Agent）** — `/mcp/` streamable HTTP；可与 **Danmo Work** 内置 `danmo-make` 专家（仅绑定，非 Ambient）配合。
- **桌面发布** — macOS MLX `.dmg`；Linux mlx[cuda] 桌面 / **服务端** tar.gz。Windows 打包暂不支持。

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

CI 产物为 **`danmo-make-linux-mlx-x86_64-<version>.tar.gz`**（内置 API + 前端静态资源）。需要 NVIDIA 驱动；**首次** `./run.sh` 会把 **mlx[cuda]** 装到 `~/.danmo-make/runtime-venv`（控制台进度）。无 CPU 回退。

```bash
tar -xzf danmo-make-linux-mlx-x86_64-*.tar.gz
cd danmo-make-linux-mlx-x86_64-*

# 可选：
# export DANQING_USER_DATA_DIR=$HOME/.danmo-make
# export DANQING_PIP_MIRROR=tuna    # official | tuna | aliyun
# export DANQING_HTTP_HOST=0.0.0.0
# export DANQING_HTTP_PORT=7800

./run.sh
```

- 界面 / API：**http://127.0.0.1:7800** · 文档 **/docs** · MCP **/mcp/**
- 修复：`./run.sh --repair-runtime` · 重装：`./run.sh --reinstall-runtime` · 状态：`./run.sh --status-runtime`
- 包内另有 `README.txt`、`danqing-runtime-setup`

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
- MCP：**http://127.0.0.1:7800/mcp/**（需尾斜杠）

### 开发端口（Danmo 产品线）

| 产品 | 后端 | 前端 |
|------|------|------|
| **Make（本仓库）** | 7800 | 5800 |
| Work（Teams） | 7801 | 5801 |
| Inbox（Mail） | 7802 | 5802 |

可用 `DQ_BACKEND_PORT` / `DQ_FRONTEND_PORT` 覆盖。

### 从源码打包桌面 / 服务端

```bash
make pack-macos-desktop    # .app / .dmg（MLX Metal sidecar）
make pack-linux-desktop    # AppImage / .deb（mlx[cuda]；本地）
make pack-linux-server     # Linux MLX thin 服务端 tar.gz
```

Windows 暂不支持。详见 [desktop/README.md](desktop/README.md)。

### CLI

```bash
bin/danqing-generate --model z-image-turbo --prompt "窗台上的猫"
bin/danqing-edit --model <id> --image input.png --prompt "加一顶帽子" --operation rewrite
bin/danqing-video-generate --model <id> --prompt "夕阳下的海浪"
bin/danqing-audio-generate --model ace-step-xl-sft --prompt "lofi" --duration 10 --output /tmp/t.wav
bin/danqing-mcp   # stdio MCP；经环境变量 / ~/.danmo-make/api.port / 7800 发现端口
```

CLI ↔ REST：[AGENTS.md](AGENTS.md#cli-vs-rest-api)。

---

## 数据布局

| 路径 | 作用 |
|------|------|
| `~/.danmo-make/` | 控制面：工作区指针、应用配置、日志、`api.port`、可选 Linux `runtime-venv` |
| `{workspace}/config/` | 运行时注册表 / 预设（从 `default_config/` 种子） |
| `{workspace}/models/` | 权重 / LoRA |
| `{workspace}/outputs/` | 生成物 |
| `{workspace}/db/studio.db` | 任务 + 资产（SQLite WAL） |

开发（`make dev`、无指针）时媒体根可留在仓库（`./models` 等）；控制面仍为 `~/.danmo-make`。

覆盖控制面：`DANQING_USER_DATA_DIR`。

---

## Agent / MCP

API 启动后：

| | |
|---|---|
| 端点 | `http://127.0.0.1:7800/mcp/`（streamable HTTP；`/mcp` 会归一到 `/mcp/`） |
| Stdio | `bin/danqing-mcp` |
| 鉴权 | 本机 loopback 免密；远程需 `Authorization: Bearer <key>` — HTTP 与 MCP **两套钥匙**（**设置 → 集成**） |

`list_models` 按 API **action** 过滤（`generate` / `edit` / `upscale`；别名 `create`→`generate`）。卡片含 `actions` / `type`，不含 LoRA。文生图/视频前请传 `action=generate`。

Danmo Work：启用内置连接器 **danmo-make** + 专家（仅绑定，非 Ambient）。

环境变量：`DANQING_HTTP_API_KEY`、`DANQING_MCP_API_KEY`、`DANQING_MCP_BASE_URL`。

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

API / MCP：`create`→`generate`，rewrite/retouch/extend→`edit`。

### 视频

| 标签页 | 注册表 action | API |
|--------|---------------|-----|
| 文生视频 | `create` | `POST /api/videos/generations` |
| 图生视频 | `animate` | `POST /api/videos/edits` |

**MiniMax H3（`minimax-h3-fl2va`）** — MLX 音视频联合 FL2VA（Metal / mlx[cuda]，**mlx >= 0.32.3**，**mlx-vlm >= 0.7.6**，**transformers >= 5.19**）。正确性对标 [PipeNetwork/minimax-h3-mlx](https://github.com/PipeNetwork/minimax-h3-mlx)（scheduler `t=1−σ`、宽高比归一 RoPE、关键帧 posterior 采样）。生成 **算力瓶颈**；mlx-q4/q8 主要省显存/内存。`balanced` / `quality` / `oracle` 走完整 Euler。`draft` 只打开 Turbo LoRA（建议 6–8 步）。非 Turbo 常用 8–16 步；注册表默认仍是 30 步，避免静默改变采样积分。复用速度场、减少 DiT 层、缩小内部画布都是有损开关，质量预设不会打开。测试：`make test-h3-parity`；冒烟基准（需已安装 bundle）：`make bench-h3-smoke`。

### 音频

| Action | API |
|--------|-----|
| 文生音乐（`create`） | `POST /api/audios/generations` |
| 翻唱 / 局部重绘 | `POST /api/audios/edits` |

### 无限画布（摘要）

创作页 → **画布**：导入（`I`）、结果进落点（`S` 贴靠）、谱系（`E` / `Y` / `G`）。会话见 `/api/canvas/sessions`。快捷键全表见 [AGENTS.md](AGENTS.md)。

### ControlNet（FLUX.1）

`flux1*` 文生图结构引导（Canny / Depth / Redux）；Fill 用于重绘/扩图。仅 MLX 栈。详见 [AGENTS.md](AGENTS.md) → Gotchas。

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

没有 remap。注册表里还是 `fp16` / `mlx-q4` / `mlx-q8` 的模型会要求重新下载 GGUF。目前只有 flux2 的冒烟去噪图能跑，其余族会明确失败。

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
HF_ENDPOINT=https://hf-mirror.com
MLX_METAL_MEMORY_LIMIT=120
DANQING_USER_DATA_DIR=~/.danmo-make
# 优先在「设置 → 集成」创建钥匙，而不是长期写明文：
# DANQING_HTTP_API_KEY=...
# DANQING_MCP_API_KEY=...
```

---

## 开发

| 命令 | 用途 |
|------|------|
| `make dev` / `stop` | API + Vite |
| `make dev-desktop` | Tauri + Vite（已有 API 时 `SKIP_BACKEND=1`） |
| `make frontend-build` | → `out/frontend/dist/` |
| `make verify-engine-stack` | 引擎治理 + 单元测试 |
| `make check-consistency` | 注册表 / 路由 / i18n / 前端门禁 |
| `make bench-eval-smoke` | 图像评估冒烟 |
| `make clean` | 清理 `out/` |

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
| MCP | `/mcp/` |
| 系统 | `/api/system/health`，`/api/settings/*` |

交互文档：**http://localhost:7800/docs**

---

## 许可证

MIT
