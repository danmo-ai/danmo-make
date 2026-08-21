# Scripts

仓库根目录执行；路径常量见 `out_paths.py` / `out_paths.sh`。

## 日常开发

| 脚本 | 说明 |
|------|------|
| `start.sh` / `stop.sh` | `make dev` / `make stop`（依赖 `dev_process.sh`） |
| `sync_workspace_registry.py` | `make sync-models-registry` |
| `repair_asset_paths.py` | 修复 DB 中资产路径 |

## 质量门禁

| 脚本 | Make 目标 |
|------|-----------|
| `check_engine_governance.py` | `check-engine-rules`（全部 engine 规则）；`--rule imports\|layout\|…` 跑单项 |
| `check_frontend_governance.py` | `check-frontend-governance`；`--rule ep\|theme\|ui` |
| `check_consistency.py` | `check-consistency`（注册表/路由/i18n + 调用 frontend governance） |
| `test_engine_unit.py` | `test-engine-unit` |
| `make_lint.py` | `lint` |

Allowlist：`engine_governance_allowlist.txt`（`# --- imports ---` 等分段）。

聚合：`make check-engine-governance` = `check-engine-rules` + `check-consistency`；`make verify-engine-stack` 在此基础上加单元测试。

旧 Make 目标（`check-engine-imports`、`check-ep-boundary` 等）仍可用，内部转发到上述两个脚本。

## 桌面 / 服务端打包

| 脚本 | 说明 |
|------|------|
| `build_desktop.sh` | 前端 + `make pack-macos-desktop` |
| `build_sidecar.py` | PyInstaller sidecar |
| `prune_sidecar.py` | sidecar 瘦身（MLX） |
| `pyinstaller_common.py` / `pyinstaller_runtime_hook.py` / `pyinstaller_hooks/` | 冻结打包（profile=`mlx`） |
| `prepare_tauri_resources.py` / `set_desktop_version.py` / `stage_desktop_bundle.py` | Tauri |
| `tauri_build_macos.sh` / `tauri_build.sh` / `tauri_build.py` | Tauri 构建 |
| `package_linux_cuda_release.py` / `package_cuda_thin.py` | Linux MLX thin 发行包（`danmo-make-linux-mlx-*`） |
| `clean_build.py` | `make clean` |

## 可选工具

| 脚本 | 说明 |
|------|------|
| `scaffold_image_family.py` | 新图像 family 脚手架 |
