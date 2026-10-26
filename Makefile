.PHONY: help clean dev start stop dev-desktop test lint \
	frontend-install frontend-dev frontend-build frontend-typecheck frontend-canvas-unit \
	pack-prereqs pack-ggml-sidecar pack-windows-sidecar \
	pack-macos-desktop pack-linux-desktop pack-linux-server

RELEASE_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
FRONTEND_DIR := $(CURDIR)/frontend

help:
	@echo "Danmo Make — Rust ggml backend + Vue"
	@echo ""
	@echo "  dev | start            danqing-api (ggml) + Vite (:7800 / :5800)"
	@echo "  dev-desktop            Tauri webview (SKIP_BACKEND=1 for an external API)"
	@echo "  stop                   Stop dev processes"
	@echo "  test                   cargo test (native/)"
	@echo "  lint                   cargo fmt --check"
	@echo "  frontend-install | frontend-dev | frontend-build | frontend-typecheck | frontend-canvas-unit"
	@echo "  pack-ggml-sidecar      release danqing-api for this host"
	@echo "  pack-windows-sidecar   same binary; Vulkan, CUDA when the toolkit is present"
	@echo "  pack-macos-desktop     macOS arm64 .app/.dmg"
	@echo "  pack-linux-desktop     Linux x86_64 AppImage/.deb"
	@echo "  pack-linux-server      headless tar.gz (ggml binary + frontend)"

dev start:
	@chmod +x scripts/*.sh
	@./scripts/start.sh

dev-desktop:
	@chmod +x scripts/*.sh
	@./scripts/start_desktop.sh

stop:
	@chmod +x scripts/*.sh
	@./scripts/stop.sh

test:
	cargo test --manifest-path native/Cargo.toml

lint:
	cd native && cargo fmt --all -- --check

clean:
	cargo clean --manifest-path native/Cargo.toml
	rm -rf out

frontend-install:
	cd $(FRONTEND_DIR) && npm install

frontend-dev: frontend-install
	cd $(FRONTEND_DIR) && npm run dev

frontend-build: frontend-install
	cd $(FRONTEND_DIR) && npm run build

frontend-typecheck: frontend-install
	cd $(FRONTEND_DIR) && npm run typecheck

frontend-canvas-unit: frontend-install
	cd $(FRONTEND_DIR) && npm run canvas-unit

pack-prereqs:
	@command -v npm >/dev/null 2>&1 || (printf '%s\n' 'npm not found. Install Node.js: https://nodejs.org/' >&2; exit 1)
	@command -v cargo >/dev/null 2>&1 || (printf '%s\n' 'cargo not found. Install Rust: https://rustup.rs/' >&2; exit 1)
	@echo "pack-prereqs OK (npm + cargo)"

pack-ggml-sidecar:
	bash scripts/build_ggml_sidecar.sh

pack-windows-sidecar: pack-ggml-sidecar

pack-macos-desktop: pack-prereqs
	@chmod +x scripts/*.sh
	@RELEASE_VERSION=$(RELEASE_VERSION) ./scripts/pack_desktop_macos.sh

pack-linux-desktop: pack-prereqs
	@chmod +x scripts/*.sh
	@RELEASE_VERSION=$(RELEASE_VERSION) ./scripts/pack_desktop_linux.sh

pack-linux-server: frontend-build
	@chmod +x scripts/*.sh
	@RELEASE_VERSION=$(RELEASE_VERSION) ./scripts/package_ggml_server.sh
