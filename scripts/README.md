# Scripts

Run from the repo root. Paths live in `out_paths.sh`.

| Script | Role |
|--------|------|
| `start.sh` / `stop.sh` | `make dev` / `make stop` |
| `start_desktop.sh` | `make dev-desktop` |
| `build_ggml_sidecar.sh` | Release `danqing-api` into `out/sidecar/danqing-api/` |
| `prepare_tauri_resources.sh` | Copy that sidecar into the Tauri bundle |
| `package_ggml_server.sh` | Headless tar.gz |
| `pack_desktop_macos.sh` / `pack_desktop_linux.sh` | Desktop packs |
| `release_version.sh` | Bump desktop version, tag, push |
| `set_desktop_version.sh` | Write the version into Tauri and desktop Cargo.toml |
