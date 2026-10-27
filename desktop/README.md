# Danmo Make — Tauri 2 desktop shell

Language: 中文 below · English summary first.

Native **Tauri 2** window. The API is the Rust binary `danqing-api` (ggml). Engine rules: [docs/engine_architecture.md](../docs/engine_architecture.md).

| Platform | Runtime | Artifact |
|----------|---------|----------|
| **macOS** (Apple Silicon) | ggml Metal sidecar | `.app` / `.dmg` |
| **Linux x64 desktop** | ggml sidecar | AppImage / `.deb` (local pack; CI may omit) |
| **Linux server** | ggml `danqing-api` + built UI | `danmo-make-<os>-ggml-<arch>-<version>.tar.gz` |
| **Windows desktop** | not wired (`pack_desktop_windows.sh` exits) | sidecar only: `make pack-windows-sidecar` |

Product name: Danmo Make. Technical ids remain `danqing-*` ([AGENTS.md](../AGENTS.md)).

---

# Danmo Make — Tauri 2 桌面壳

本目录是 **Tauri 2** 窗口。API 是 Rust 二进制 `danqing-api`（ggml）。引擎原则见 [docs/engine_architecture.md](../docs/engine_architecture.md)。

- **macOS**：ggml Metal sidecar
- **Linux 桌面**：ggml sidecar（AppImage / `.deb`，本地打包）
- **Linux 服务端**：`danqing-api` + 已构建界面，`./run.sh` 直接启动二进制
- **Windows 桌面**：未接线。sidecar 用 `make pack-windows-sidecar`

产品名为 Danmo Make；技术 id 仍为 `danqing-*`（见仓库根 [AGENTS.md](../AGENTS.md)）。

## 构建产物目录

打包产物在仓库根 **`out/`** 下（`scripts/out_paths.sh`）：

| 路径 | 内容 |
|------|------|
| `out/frontend/dist/` | Vite 生产构建 |
| `out/sidecar/danqing-api/` | ggml sidecar |
| `out/desktop/bundle/` | `.dmg` / AppImage / `.deb` |
| `out/desktop/cargo/` | Cargo 中间产物 |

## 前置

- **Rust**（`cargo` 在 PATH）
- **Node.js**（`npm`）
- Linux 桌面另需 WebKitGTK 等（本地 `make pack-linux-desktop`；CI 暂未构建 Linux 桌面）

## 开发（`make dev-desktop`）

```bash
make dev-desktop
# 已有 API 时：SKIP_BACKEND=1 make dev-desktop
```

- `cargo run -p danmo-server` 启动 `danqing-api`（`:7800`）
- Tauri `beforeDevCommand` 启动 Vite HMR（`:5800`）
- 原生窗口加载 Vite；`Ctrl+C` / `make stop` 清理

## 发布构建

```bash
make pack-macos-desktop      # Darwin arm64 · ggml Metal · .app/.dmg
make pack-linux-desktop      # Linux x86_64 · ggml · AppImage/.deb（本地）
make pack-linux-server       # 无头 ggml API + 界面
make pack-windows-sidecar    # 仅 sidecar；桌面包未接线
```

| 脚本 | Make 目标 |
|------|-----------|
| `scripts/pack_desktop_macos.sh` | `pack-macos-desktop` |
| `scripts/pack_desktop_linux.sh` | `pack-linux-desktop` |
| `scripts/package_ggml_server.sh` | `pack-linux-server` |
| `scripts/pack_desktop_windows.sh` | 立即退出，并提示改打 sidecar |

顺序：`out/frontend/dist` → `scripts/build_ggml_sidecar.sh` → Tauri → `out/desktop/bundle/`。服务端包不走 Tauri，`./run.sh` 执行 `danqing-api`。

设备由 `DANQING_GGML_BACKEND` 选择（`cpu`、`cuda`、`vulkan`、`metal`）。二进制没编进的后端在启动时失败。

## 运行时环境变量

| 变量 | 说明 |
|------|------|
| `DANQING_HTTP_HOST` | 服务端包默认 `127.0.0.1` |
| `DANQING_HTTP_PORT` | 默认 `7800`；Tauri 开发时注入空闲端口 |
| `DANQING_GGML_BACKEND` | `cpu` / `cuda` / `vulkan` / `metal` |
| `DANQING_REGISTRY` | 注册表路径 |
| `DANQING_DB` | SQLite 路径 |
| `DANQING_USER_DATA_DIR` | 控制面根目录，默认 `~/.danmo-make` |

## 安装后提示「已损坏，无法打开」

从浏览器 / GitHub Release 下载的 `.dmg` 会带 **隔离属性**（quarantine），且当前构建 **未做 Apple 公证**，系统常误报为「损坏」。应用本身通常没问题。

**任选一种方式：**

1. **右键打开**：在「应用程序」里找到 **Danmo Make** → 按住 Control 点按 → **打开** → 再点 **打开**（仅首次）。
2. **去掉隔离属性**（把路径换成你的 `.app` 实际位置）：

```bash
xattr -dr com.apple.quarantine "/Applications/Danmo Make.app"
```

发布构建会在 `make pack-macos-desktop` 末尾对 `.app` 做 **ad-hoc 签名**，减轻该问题；正式分发仍建议配置 **Developer ID + 公证（notarize）**。
