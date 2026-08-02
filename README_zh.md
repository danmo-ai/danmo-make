# Danmo Make v4

Language: [English](README.md) | **中文**

插件化图像/视频生成产品，基于 **MLX**（Apple Silicon）与 **CUDA**（NVIDIA）双后端。前后端分离：FastAPI + Vue 3 SPA，REST 与 CLI 语义一致，SQLite 持久化任务与资产，界面与错误信息支持 **中英文**。

产品名为 **Danmo Make**。引擎 / CLI / sidecar 等技术标识仍为 `danqing-*` / `DanQing*Engine`（见 [AGENTS.md](AGENTS.md) → Naming boundary）。

| | |
|---|---|
| **贡献者 / Agent 文档** | [AGENTS.md](AGENTS.md) |
| **桌面版（Tauri 2）** | [desktop/README.md](desktop/README.md) |
| **引擎架构** | [docs/engine_architecture.md](docs/engine_architecture.md) |
| **图像基准测试** | [tests/benchmark/README.md](tests/benchmark/README.md) |

---

## 特性

- **双运行时** — Apple Silicon 上 `MLXContext`；安装 PyTorch CUDA 时可用 `CudaContext`（以注册表 `backends` 为准）。
- **分层架构** — REST / CLI → `TaskScheduler` → `DanQing*Engine` → `ImagePipeline` / `VideoPipeline` → `RuntimeContext` → SQLite。
- **模型即插件** — 新模型族只需改注册表 JSON、`model_configs`、`families/<family>/`、`_transformer_registry.py`；Pipeline 骨架不写 `family` 分支。
- **契约化 API** — 路由与 CLI 通过 `backend/core/contracts.py` 与 `IImageEngine` / `IVideoEngine` 进入引擎；路由逻辑不写模型分支。
- **全局单队列** — 图像/视频（及音频占位）串行执行；SSE 进度、优先级、队列位置、日志落库。
- **四大模块** — **创作**（按模型 `actions` 过滤）、**图库**（SQLite `assets`）、**模型**（安装/删除权重）、**设置**（预设、队列策略、系统状态）。
- **无限画布**（图像 / 视频 / 音频创作页）— 画廊 **网格** 与 **画布** 两种视图共用作品库；画布会话持久化排版、谱系连线与创作器状态（按媒介隔离）。

### 无限画布工作流

在 **创作页 → 画布视图**（顶部画廊条切换）：

1. **导入** — 右下「导入作品」（`I`）、画廊悬停「添加到画布」，或网格多选后切画布。
2. **迭代** — 选中节点，底部 **创作器** 灌参；浮动工具栏精修 / 分支 / 翻唱等。
3. **生成** — 新结果落入 **生成落点**（橙色框）；`S` 贴靠选中节点。
4. **谱系** — 父子连线（`E`）、会话图谱（`G`）、谱系侧栏（`Y`）；单击定位、双击关闭并跳转。
5. **会话** — 左上会话栏切换/新建/重命名（`/api/canvas/sessions` 同步）。

| 快捷键 | 功能 |
|--------|------|
| `I` | 导入作品选择器 |
| `S` | 落点贴靠选中节点 |
| `R` | 区域引导（落点 + 叠加链接） |
| `L` / `G` / `E` | 图层 / 会话图谱 / 谱系连线 |
| `Y` | 谱系侧栏 |
| `F2` | 重命名选中节点 |
| `Esc` | 关闭面板 → 清除选择 |
| 空格拖移 | 平移视口 |

设置中的 **生成自动加入画布** 可在画廊视图下仍将结果落入当前会话落点区。

### ControlNet / 结构引导（FLUX.1）

Invoke 风格 **结构条件**（仅 FLUX.1 基底，如 `flux1-dev`）：

1. **模型页** — 安装 `flux1-dev` 与 ControlNet 包（`flux-canny-controlnet`、`flux-depth-controlnet`、`flux-redux` 等）。**Depth** 另需 `depth-pro` 工具模型；**Canny/Depth/Redux 预处理** 使用 OpenCV（Canny）或 **PyTorch**（Depth Pro + SigLIP/Redux，CPU）。
2. **创作器** — 高级参数选 ControlNet + 强度；绑定 **结构引导图**（须为画廊资产）。切换 ControlNet 会套用注册表推荐参数（如 Canny/Depth CFG ≈ 30）。
3. **画布** — 选中节点 →「结构引导分支」或「用作结构引导」；CTRL 叠加层与创作器同步。
4. **生成** — 请求携带 `structural_guide`（`model_id`、`asset_id`、`type`、`weight`）：
   - **Canny / Depth** — 预处理 → VAE 编码 → 128 通道 patch 拼接 + 配套 LoRA（`flux1-canny-dev-lora` / `flux1-depth-dev-lora`）。
   - **Redux** — SigLIP + redux MLP 令牌并入 T5 上下文（无 patch embed）。
   - **Fill**（`flux-fill-controlnet`）— 仅局部重绘/扩图（retouch/extend），文生图不可用。

结构引导不能与参考图 img2img 同请求并用；绑定引导图时谱系为 `relation_type: controlnet`。

### 创作页 ↔ 模型 `actions`

创作页的标签页仅列出工作区 `config/models_registry.json`（出厂默认在 `default_config/`）中声明对应 `action` 的模型。

#### 图像创作

| 标签页 | 所需 action | API |
|--------|-------------|-----|
| 文生图 | `create` | `POST /api/images/generations` |
| 参考图改图 | `rewrite` | `POST /api/images/edits` (`operation: rewrite`) |
| 指令修图 | `rewrite` | `POST /api/images/edits` (`operation: rewrite`) |
| 局部重绘 | `retouch` | `POST /api/images/edits` (`operation: retouch`) |
| 扩图 | `extend` | `POST /api/images/edits` (`operation: extend`) |
| 放大 | `upscale` | `POST /api/images/upscales` |

#### 视频创作

| 标签页 | 所需 action | API |
|--------|-------------|-----|
| 文生视频 | `create` | `POST /api/videos/generations` |
| 图生视频 | `animate` | `POST /api/videos/edits` |

#### 音频

音频接口已入队，但**无推理后端**时会在任务日志中**显式失败**（不静默降级）。

---

## 环境要求

| 平台 | 说明 |
|------|------|
| **macOS（Apple Silicon）** | 主要目标；MLX 通过 Metal。`make dev` 面向 macOS + Python 3.11 |
| **Linux / Windows + NVIDIA** | CUDA 路径（需 `torch` + CUDA）；部分 family 尚无 `*_cuda.py` 时会**明确报错**，不静默降级 |
| **Python** | 3.11+（仓库根 `.venv/`） |
| **内存** | 大模型建议 32 GB 及以上 |
| **Node.js** | 前端开发与桌面打包 |
| **ffmpeg / ffprobe** | 视频缩略图与时长元数据（建议安装） |

---

## 快速开始

### 安装

```bash
git clone <仓库地址> danmo-make
cd danmo-make

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 运行（Web）

```bash
# 开发：uvicorn --reload + Vite HMR（一键启停）
make dev
# 或 make start / make stop
```

浏览器打开 **http://localhost:5800**（Vite，代理 `/api` → :7800），API 文档 **/docs**。

### 开发端口（Danmo 产品线）

后端 **`78xx`**，前端 **`58xx`** — 后两位相同 = 同一项目。三个仓库可同时 `make dev`。

| 项目 | 后端 | 前端（Vite） |
|------|------|---------------|
| **Studio** | 7800 | 5800 |
| Teams | 7801 | 5801 |
| Mail | 7802 | 5802 |

可通过 `DQ_BACKEND_PORT`、`DQ_FRONTEND_PORT` 覆盖（见 `scripts/out_paths.sh`）。

### 发布

```bash
make pack-macos-desktop          # macOS .app / .dmg → out/desktop/bundle/
make pack-linux-server           # Linux CUDA tar.gz → out/dist/
make pack-windows-desktop-release # Windows NSIS（在 Windows 上）
```

### CLI 示例

```bash
bin/danqing-generate --model flux2-klein-9b --prompt "窗台上的猫"
bin/danqing-edit --model <id> --image input.png --prompt "加一顶帽子" --operation rewrite
bin/danqing-upscale --model <id> --image input.png
bin/danqing-video-generate --model <id> --prompt "夕阳下的海浪"
```

完整 CLI ↔ REST 对照：[AGENTS.md](AGENTS.md#cli-vs-rest-api)。

### 模型磁盘布局

| 路径 | 用途 |
|------|------|
| `./models/` | 模型权重（`.safetensors`、`.bin`、diffusers 布局、`model_index.json`） |
| `./models/Lora/` | LoRA 权重 |
| `./outputs/` | 生成文件 |
| `./db/studio.db` | 任务 + 资产（SQLite WAL） |

从 **模型** 页面安装权重，或通过 `POST /api/models/{id}/install`（下载进度 SSE）。

### 前端开发（热更新）

`make dev` 同时启动 API（:7800，--reload）与 Vite（:5800）。也可以单独运行：

```bash
make frontend-dev   # Vite，端口 5800，代理 /api → :7800
```

### 基准测试（可选）

使用 `tests/benchmark/venv/` 下的隔离虚拟环境：

```bash
make bench-setup
make bench-eval-smoke    # 图像模型评估（L1 + ImageReward，快速）
make bench-eval          # 完整提示词矩阵
make verify-engine-stack # 引擎规则检查 + 单元测试
```

---

## 项目结构

```
danmo-make/
├── backend/
│   ├── api/routes/          # REST（images, videos, tasks, assets, registry, …）
│   ├── cli/                 # bin/danqing-*（与 REST 对应）
│   ├── core/                # contracts, interfaces, DI container, i18n
│   ├── engine/
│   │   ├── pipelines/       # ImagePipeline, VideoPipeline, …
│   │   ├── families/        # 各模型族 transformer（flux1, flux2, z_image, ltx, …）
│   │   ├── runtime/         # MLXContext, CudaContext（仅此处可 import mlx/torch）
│   │   └── common/          # VAE, schedulers, text encoders, TransformerBase
│   ├── persistence/         # V3TaskStore, SQLiteAssetStore
│   ├── scheduler/           # 全局 TaskScheduler
│   └── main.py              # FastAPI 入口
├── frontend/                # Vue 3 + Vite + TypeScript
├── desktop/                 # Tauri 2 壳
├── bin/                     # danqing-* CLI
├── default_config/          # 出厂 models_registry, presets, locales, workspace.pointer
├── scripts/                 # 构建、lint 检查、桌面打包
├── tests/benchmark/         # 图像评估基准测试（L1+L2）
├── docs/                    # engine_architecture.md（引擎架构单文档）
├── models/  outputs/  db/
└── out/                     # 构建产物（不进 git）
    ├── frontend/dist/       # Vite 生产构建
    ├── sidecar/             # PyInstaller danqing-api
    └── desktop/bundle/      # .app / .dmg
```

---

## 架构概要

```
REST API (FastAPI)  ||  CLI (bin/danqing-*)
         ↓                    ↓
     TaskScheduler  (全局单队列，串行执行)
         ↓
 DanQingImageEngine / DanQingVideoEngine / DanQingAudioEngine
         ↓
 ImagePipeline / VideoPipeline  (注册表驱动组装线)
         ↓
 RuntimeContext (MLX | CUDA) + TransformerBase 模型族 + 公共组件
         ↓
 V3TaskStore + SQLiteAssetStore
```

**新增模型五步**：注册表 JSON → `model_configs.py` → `families/<family>/transformer.py` → `weights.py` (`remap_*`) → `_transformer_registry.py`。详见 [AGENTS.md](AGENTS.md#new-model-checklist)。

---

## 桌面版

各平台 sidecar 保持精简 — **不在一个发布包中混入 MLX 和 CUDA**：

| 平台 | Profile | 后端 | Make 目标 |
|------|---------|------|-----------|
| macOS（Apple Silicon） | `mlx` | MLX / Metal | `make pack-macos-desktop` |
| Linux x86_64 server | `cuda` | PyTorch CUDA | `make pack-linux-server` |
| Windows x64 desktop | `cuda` | PyTorch CUDA | `make pack-windows-desktop-release` |

```bash
make pack-macos-desktop           # 仅 MLX .dmg
make pack-linux-server            # CUDA 服务端 .tar.gz
make pack-windows-desktop-release # CUDA NSIS（在 Windows 上）
```

GitHub tag 构建使用同样的分离策略（`.github/workflows/release.yml`）。

详见 [desktop/README.md](desktop/README.md)。

---

## 配置

**应用设置** — `{workspace}/config/.app_config.json`：

```json
{
  "language": "en",
  "theme": "dark",
  "default_model": "flux2-klein-9b",
  "mlx_memory_limit": 120,
  "queue_image_first": true
}
```

**模型注册表** — `{workspace}/config/models_registry.json`（`schema_version: 3`；出厂默认在 `default_config/`）：嵌套 `catalog` / `runtime` / `ui` / `distribution`；API 通过 `GET /api/registry` 返回 `CatalogResponse` DTO。

**环境变量**（可选 `.env`）：

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_ENABLE_HF_TRANSFER=1
MLX_METAL_DEVICE_ONLY=1
MLX_METAL_MEMORY_LIMIT=120
```

---

## 开发

| 命令 | 用途 |
|------|------|
| `make dev` / `make start` / `make stop` | 开发：uvicorn --reload + Vite HMR |
| `make pack-macos-desktop` | macOS 桌面发布 |
| `make pack-linux-server` | Linux 服务端发布 |
| `make frontend-dev` | Vite 开发服务器 |
| `make frontend-build` | 生产 UI → `out/frontend/dist/` |
| `make frontend-typecheck` | `vue-tsc` |
| `make frontend-canvas-unit` | 画布边/落点工具自检 |
| `make check-consistency` | 注册表 / 路由 / i18n + 前端规则（含画布单元测试） |
| `make check-engine-imports` | mlx/torch 导入边界 |
| `make lint` | Python 语法检查 |
| `make clean` | 删除 `out/` 构建树 |

后端热重载：`python3 -m uvicorn backend.main:app --reload --port 7800`

---

## API 概览

| 领域 | 端点 |
|------|------|
| 图像 | `POST /api/images/generations`、`edits`、`upscales` |
| 视频 | `POST /api/videos/generations`、`edits` |
| 任务 | `GET/PATCH/DELETE /api/tasks/{id}`、`GET …/stream`（SSE）、`GET /api/queue` |
| 资产 | `GET/POST /api/assets`、`…/file`、`…/thumbnail`、`POST …/reconcile` |
| 模型 | `GET /api/models`、`POST /api/models/{id}/install`，注册表 `GET /api/registry` |
| 系统 | `GET /api/system/health`、`GET /api/settings/system` |

交互式文档：**http://localhost:7800/docs**

---

## 许可证

MIT
