# Scripts

Run from the repo root. Paths live in `out_paths.sh`.

| Script | Role |
|--------|------|
| `start.sh` / `stop.sh` | `make dev` / `make stop` |
| `start_desktop.sh` | `make dev-desktop` |
| `build_mlx_sidecar.sh` | Release `danqing-api` (MLX: `metal,mlx-metal` or `mlx-cuda`) into `out/sidecar/danqing-api/` |
| `prepare_tauri_resources.sh` | Copy that sidecar into the Tauri bundle |
| `package_mlx_server.sh` | Headless server tar.gz (`danmo-make-linux-mlx-cuda-*.tar.gz`, etc.) |
| `pack_desktop_macos.sh` / `pack_desktop_linux.sh` | Desktop packs |
| `ci_import_apple_cert.sh` | CI: import Developer ID p12 into a temp keychain |
| `ci_install_cuda_ubuntu.sh` | Linux CI: apt CUDA 12.9, or env-only when `DANQING_CUDA_SKIP_APT_INSTALL=1` (CUDA container) |
| `gh_set_release_secrets.sh` | Interactive `gh secret set` for `APPLE_*` (see `make set-release-secrets`) |
| `release_version.sh` | Bump desktop version, tag, push |
| `set_desktop_version.sh` | Write the version into Tauri and desktop Cargo.toml |
