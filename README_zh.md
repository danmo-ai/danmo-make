# Danmo Make

Language: [English](README.md) | **中文**

本地 **图像 / 视频 / 音频** 创作工作室，基于 **MLX**（Apple Silicon Metal；Linux **mlx[cuda]**）。技术栈：FastAPI + Vue 3 SPA + CLI + SQLite，界面双语，错误 **显式失败**（不静默降级）。**Windows 暂不支持。**

产品名：**Danmo Make**。引擎 / CLI / sidecar 技术标识仍为 `danqing-*` / `DanQing*Engine`（见 [AGENTS.md](AGENTS.md) → Naming boundary）。

| | |
|---|---|
| **贡献者 / Coding Agent** | [AGENTS.md](AGENTS.md) |
| **桌面版（Tauri 2）** | [desktop/README.md](desktop/README.md) |
| **引擎架构** | [docs/engine_architecture.md](docs/engine_architecture.md) |
| **图像基准** | [tests/benchmark/README.md](tests/benchmark/README.md) |
| **发布包** | [GitHub Releases](https://github.com/danmo-ai/danmo-make/releases) |

---

## 特性

- **MLX 运行时** — 单一 `MLXContext`：Apple Silicon 用 Metal，Linux NVIDIA 用 **mlx[cuda]**（注册表 `backends: ["mlx"]`）。
- **模型即插件** — 新模型 = 注册表 JSON + `model_configs` + `families/<family>/` + `_transformer_registry`；Pipeline 不写 `family` 业务分支。
- **契约化 API / CLI** — 只经 contracts + `IImageEngine` / `IVideoEngine` / `IAudioEngine`。
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
| **macOS（Apple Silicon）** | 主目标；MLX / Metal — `pip install -r requirements-macos.txt` |
| **Linux + NVIDIA** | MLX / **mlx[cuda]** — `pip install -r requirements-linux.txt` |
| **Windows** | 暂不支持 |
| **Python** | 3.11+（Web/开发用仓库根 `.venv/`） |
| **内存** | 大模型建议 32 GB+ |
| **Node.js** | 前端与桌面打包 |
| **ffmpeg / ffprobe** | 建议安装（缩略图 / 时长） |

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
git clone https://github.com/danmo-ai/danmo-make.git
cd danmo-make

python3.11 -m venv .venv
source .venv/bin/activate
# macOS Apple Silicon：
pip install -r requirements-macos.txt
# Linux（NVIDIA / mlx[cuda]）：
# pip install -r requirements-linux.txt
```

### 运行（Web）

```bash
make dev      # uvicorn --reload (:7800) + Vite HMR (:5800)
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
├── backend/
│   ├── api/routes/       # REST
│   ├── mcp/              # MCP 工具与模型卡片
│   ├── cli/              # bin/danqing-*
│   ├── core/             # contracts / interfaces / i18n
│   ├── engine/           # pipelines / families / runtime / common
│   ├── persistence/
│   ├── scheduler/
│   └── main.py
├── frontend/
├── desktop/              # Tauri 2
├── bin/
├── default_config/
├── scripts/
├── tests/benchmark/
├── docs/
└── out/                  # 构建产物（不进 git）
```

---

## 架构

```
REST / MCP / CLI
        ↓
TaskScheduler  (全局单队列)
        ↓
DanQingImageEngine / DanQingVideoEngine / DanQingAudioEngine
        ↓
Pipelines + FamilyPlugin + RuntimeContext (MLX)
        ↓
V3TaskStore + SQLiteAssetStore
```

加模型：注册表 → `model_configs` → `families/<family>/` → `weights.py` → `_transformer_registry`。清单：[AGENTS.md](AGENTS.md#new-model-checklist)。

---

## 发版产物（CI）

打 `v*` tag → [`.github/workflows/release.yml`](.github/workflows/release.yml)。**安装步骤见** [安装发布包](#安装发布包推荐)。

| 平台 | CI 产物 |
|------|---------|
| macOS Apple Silicon | MLX `.dmg`（Metal） |
| Linux x86_64 | **`danmo-make-linux-mlx-x86_64-*.tar.gz`**（thin 服务端；首次 `./run.sh` 装 mlx[cuda]） |
| Windows | 暂不支持 |

本地打包（并非全部由 CI 挂载）：

```bash
make pack-macos-desktop
make pack-linux-desktop    # AppImage / .deb — 目前仅本地
make pack-linux-server
```

全平台统一 MLX 栈（Metal 或 mlx[cuda]），无独立 torch CUDA 引擎。

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
