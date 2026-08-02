# Danmo Make — Tauri 2 桌面壳

本目录提供 **Tauri 2** 原生窗口；业务仍由 **FastAPI REST**（PyInstaller 打包的 `danqing-api` sidecar）提供。

Makefile / 脚本命名与 [danmo-work](../DanQing-Teams) / [danmo-inbox](../DanQing-Mail) 对齐。

## 构建产物目录

所有打包产物在仓库根目录 **`out/`** 下（见 `scripts/out_paths.py`）：

| 路径 | 内容 |
|------|------|
| `out/frontend/dist/` | Vite 生产构建 |
| `out/sidecar/danqing-api/` | PyInstaller sidecar |
| `out/desktop/bundle/` | 平台安装包（`.dmg` / AppImage / `.deb` / NSIS） |
| `out/desktop/cargo/` | Cargo 中间产物（可清理） |

清理：`make clean` 或 `python scripts/clean_build.py`

## 前置

- **Rust**（`cargo` 在 PATH）
- 仓库根 **Python 3.11 + `.venv`**，已安装 **PyInstaller**
- **Node.js**（`npm`）
- Linux 桌面另需 WebKitGTK 等（见 CI `build-linux-desktop`）

## 开发（`make dev-desktop`）

与 danmo-work / danmo-inbox 相同：

```bash
make dev-desktop
# 已有 API 时：SKIP_BACKEND=1 make dev-desktop
```

- 启动 FastAPI（`:7800`，`--reload`）
- Tauri `beforeDevCommand` 启动 Vite HMR（`:5800`）
- 原生窗口加载 Vite；`Ctrl+C` / `make stop` 清理

## 发布构建

```bash
make pack-macos-desktop      # Darwin arm64 · MLX · .app/.dmg
make pack-linux-desktop      # Linux x86_64 · CUDA · AppImage/.deb
make pack-windows-desktop    # Windows x86_64 · CUDA · NSIS（须在 Windows 上跑）
```

等价脚本（与 work/inbox 同名）：

| 脚本 | Make 目标 |
|------|-----------|
| `scripts/pack_desktop_macos.sh` | `pack-macos-desktop` |
| `scripts/pack_desktop_linux.sh` | `pack-linux-desktop` |
| `scripts/pack_desktop_windows.sh` | `pack-windows-desktop` |

顺序：`out/frontend/dist` → PyInstaller sidecar → Tauri bundle → `out/desktop/bundle/`。

| 平台 | Sidecar profile | 产物 |
|------|-----------------|------|
| macOS Apple Silicon | **MLX**（无 torch） | `.app` / `.dmg` |
| Linux x86_64 desktop | **CUDA**（无 MLX） | AppImage / `.deb` |
| Windows x64 desktop | **CUDA**（无 MLX） | NSIS `*-setup.exe` |

另可选无界面服务端包：`make pack-linux-server` / `make pack-windows-server`。

`DANQING_PYINSTALLER_PROFILE`：`mlx`（macOS）或 `cuda`（Linux/Windows）。禁止在同一发布包中混装 MLX + CUDA。

CI（`.github/workflows/release.yml`）在打 `v*` tag 时并行构建：macOS `.dmg`、Linux 桌面 AppImage/deb、Linux CUDA `.tar.gz`、Windows NSIS。

## 运行时环境变量（sidecar）

| 变量 | 说明 |
|------|------|
| `DANQING_HTTP_HOST` | 默认 `0.0.0.0`；Tauri 设为 `127.0.0.1` |
| `DANQING_HTTP_PORT` | Tauri 选空闲端口并注入 |
| `DANQING_USER_DATA_DIR` | 可写数据根（models / outputs / db / config） |

## 安装后提示「已损坏，无法打开」

从浏览器 / GitHub Release 下载的 `.dmg` 会带 **隔离属性**（quarantine），且当前构建 **未做 Apple 公证**，系统常误报为「损坏」。应用本身通常没问题。

**任选一种方式：**

1. **右键打开**：在「应用程序」里找到 **Danmo Make** → 按住 Control 点按 → **打开** → 再点 **打开**（仅首次）。
2. **去掉隔离属性**（把路径换成你的 `.app` 实际位置）：

```bash
xattr -dr com.apple.quarantine "/Applications/Danmo Make.app"
```

发布构建会在 `make pack-macos-desktop` 末尾对 `.app` 做 **ad-hoc 签名**，减轻该问题；正式分发仍建议配置 **Developer ID + 公证（notarize）**。
