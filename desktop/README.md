# Danmo Make — Tauri 2 桌面壳

本目录提供 **Tauri 2** 原生窗口；业务仍由 **FastAPI REST**（PyInstaller 打包的 `danqing-api` sidecar）提供。

## 构建产物目录

所有打包产物在仓库根目录 **`out/`** 下（见 `scripts/out_paths.py`）：

| 路径 | 内容 |
|------|------|
| `out/frontend/dist/` | Vite 生产构建 |
| `out/sidecar/danqing-api/` | PyInstaller sidecar |
| `out/desktop/bundle/` | `.app` / `.dmg`（发布产物） |
| `out/desktop/cargo/` | Cargo 中间产物（可清理） |

清理：`make clean` 或 `python scripts/clean_build.py`

## 前置

- **Rust**（`cargo` 在 PATH）
- 仓库根 **Python 3.11 + `.venv`**，已安装 **PyInstaller**
- **Node.js**（`npm`）

## 开发（不打包 sidecar）

1. 终端 A：仓库根 `make dev`（API :7800，Vite :5800）
2. 终端 B：`cd desktop && npm install && npm run dev`

## 发布构建

```bash
make pack-macos-desktop
# 或: ./scripts/build_desktop.sh
```

顺序：`out/frontend/dist` → `out/sidecar/danqing-api` → Tauri bundle。

| 平台 | 命令 | Sidecar profile |
|------|------|-----------------|
| macOS (Apple Silicon) | `make pack-macos-desktop` | **MLX**（无 torch） |
| Linux x86_64 server | `make pack-linux-server` | **CUDA**（无 MLX） |
| Windows x64 desktop | `make pack-windows-desktop-release` | **CUDA**（无 MLX） |

`DANQING_PYINSTALLER_PROFILE`：`mlx`（macOS）或 `cuda`（Linux/Windows）。禁止在同一发布包中混装 MLX + CUDA。

Windows 需在 **Windows 本机** 构建；产物为 `out/desktop/bundle/nsis/*-setup.exe`。
CI（`.github/workflows/release.yml`）在打 `v*` tag 时并行构建三端：macOS `.dmg`、Linux CUDA `.tar.gz`、Windows NSIS `.exe`。

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

若仍在 DMG 卷宗里安装：

```bash
xattr -dr com.apple.quarantine "/Volumes/Danmo Make/Danmo Make.app"
```

然后拖入「应用程序」再启动。

发布构建会在 `make pack-macos-desktop` 末尾对 `.app` 做 **ad-hoc 签名**（`scripts/tauri_build.sh`），减轻该问题；正式分发仍建议配置 **Developer ID + 公证（notarize）**。

## `bundle_dmg.sh` / DMG 打包失败

若 Tauri 在 `Running bundle_dmg.sh` 处失败或长时间卡住（常见于 **macOS 15+**），多为 create-dmg 的 **Finder AppleScript** 与系统不兼容。`scripts/tauri_build.sh` 已设置 `CI=true`，让 Tauri 对 DMG 使用 `--skip-jenkins`（功能正常，窗口布局为默认样式）。

若仍有残留挂载导致失败，可先执行：

```bash
hdiutil detach "/Volumes/Danmo Make" -force 2>/dev/null || true
find out/desktop/cargo -name 'rw.*.dmg' -delete
make desktop-tauri
```
