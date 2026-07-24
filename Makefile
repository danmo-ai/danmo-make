.PHONY: help clean clean-download-cache lint dev start stop test test-integration \
	frontend-install frontend-dev frontend-build frontend-typecheck frontend-canvas-unit \
	bench-setup bench-download-judge bench-eval bench-eval-smoke bench-eval-case bench-eval-calibrate \
	calibrate-teacache-smoke calibrate-teacache-run calibrate-teacache-fit \
	chapter-parse-bench chapter-parse-bench-test \
	check-consistency check-models-registry-contracts check-ep-boundary check-theme-legacy check-ui-compat check-engine-rules check-engine-imports check-engine-family-layout check-engine-family-primitives check-engine-attention-paths check-engine-sdpa-paths check-engine-rope-paths check-engine-modulation-paths check-frontend-governance check-weight-parity check-engine-governance verify-engine-stack \
	sync-models-registry \
	strip-el-tokens test-engine-unit \
	pack-prereqs \
	pack-macos-desktop-sidecar pack-macos-desktop-shell pack-macos-desktop \
	pack-linux-server-venv pack-linux-server-sidecar pack-linux-server-archive pack-linux-server \
	pack-windows-venv pack-windows-sidecar \
	pack-windows-server-archive pack-windows-server \
	pack-windows-desktop-shell pack-windows-desktop pack-windows-desktop-release \
	desktop-prereqs desktop-sidecar desktop-tauri desktop-bundle \
	linux-cuda-venv linux-cuda-sidecar release-linux-cuda-tar release-linux-cuda \
	windows-cuda-venv windows-cuda-sidecar windows-cuda-desktop-sidecar \
	windows-desktop-tauri windows-desktop-bundle release-windows-desktop \
	release-windows-cuda-zip release-windows-cuda

# Windows Git Bash / MSYS: use Scripts/python.exe so pack-windows-* works via make.
ifeq ($(OS),Windows_NT)
  PYTHON := .venv/Scripts/python.exe
  BENCH_PY := tests/benchmark/venv/Scripts/python.exe
  BENCH_PIP := tests/benchmark/venv/Scripts/pip.exe
  BENCH_BIN := tests/benchmark/venv/Scripts
else
  PYTHON := .venv/bin/python3
  BENCH_PY := tests/benchmark/venv/bin/python3
  BENCH_PIP := tests/benchmark/venv/bin/pip
  BENCH_BIN := tests/benchmark/venv/bin
endif
BENCH_OUT := tests/benchmark/outputs
OUT_DIR := $(CURDIR)/out

# Release packaging (see scripts/out_paths.py)
# Naming: pack-<platform>-<product>-<step>
#   platform: macos | linux | windows
#   product:  desktop (Tauri) | server (API only, zip/tar)
#   step:     venv | sidecar | shell | archive | bundle | release
RELEASE_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu124

# ============================================================================
# Benchmark (independent venv: tests/benchmark/venv)
# ============================================================================

BENCH_PYTHON ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)

bench-setup:
	$(BENCH_PYTHON) -m venv tests/benchmark/venv
	$(BENCH_PIP) install -r tests/benchmark/requirements.txt
	@echo "Benchmark venv ready. Run: make bench-download-judge (ModelScope) then make bench-eval-smoke"

bench-download-judge:
	$(PYTHON) -m tests.benchmark download-judge

bench-download-judge-force:
	$(PYTHON) -m tests.benchmark download-judge --force

bench-eval:
	$(BENCH_PY) -m tests.benchmark eval --all --profile full

bench-eval-smoke:
	$(BENCH_PY) -m tests.benchmark eval --all --profile smoke

bench-eval-case:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make bench-eval-case ID=<model>:<prompt>:<action>"; \
		exit 2; \
	fi
	$(BENCH_PY) -m tests.benchmark eval --case $(ID)

bench-eval-calibrate:
	$(BENCH_PY) -m tests.benchmark eval --all --profile full --calibrate

TEACACHE_OUT := tests/benchmark/outputs/teacache

# TeaCache calibration — offline fit smoke (no GPU) or probe run (requires installed model).
calibrate-teacache-smoke:
	@mkdir -p $(TEACACHE_OUT)
	$(PYTHON) scripts/calibrate_teacache.py fit \
		--trace tests/fixtures/teacache/flux1_probe_trace.sample.json \
		--write-report $(TEACACHE_OUT)/flux1_smoke_report.json
	$(PYTHON) scripts/calibrate_teacache.py fit \
		--trace tests/fixtures/teacache/wan_probe_trace.sample.json \
		--write-report $(TEACACHE_OUT)/wan_smoke_report.json
	@echo "TeaCache calibration smoke OK -> $(TEACACHE_OUT)/"

calibrate-teacache-run:
	@if [ -z "$(MODEL)" ] || [ -z "$(PROMPT)" ]; then \
		echo "Usage: make calibrate-teacache-run MODEL=flux1-dev PROMPT='a mountain' [STEPS=28] [SIZE=512x512] [SEED=42]"; \
		echo "       Optional: OUTPUT=... TRACE=... TARGET_SKIP_RATE=0.35"; \
		exit 2; \
	fi
	@mkdir -p $(TEACACHE_OUT)
	$(PYTHON) scripts/calibrate_teacache.py run \
		--model $(MODEL) \
		--prompt "$(PROMPT)" \
		--steps $(or $(STEPS),28) \
		--output $(or $(OUTPUT),$(TEACACHE_OUT)/$(MODEL)_probe.png) \
		--trace $(or $(TRACE),$(TEACACHE_OUT)/$(MODEL)_probe_trace.json) \
		--target-skip-rate $(or $(TARGET_SKIP_RATE),0.35) \
		$(if $(SIZE),--size $(SIZE),) \
		$(if $(SEED),--seed $(SEED),) \
		$(if $(GUIDANCE),--guidance $(GUIDANCE),)

calibrate-teacache-fit:
	@if [ -z "$(TRACE)" ]; then \
		echo "Usage: make calibrate-teacache-fit TRACE=path/to/trace.json [FAMILY=] [STEPS=] [TARGET_SKIP_RATE=0.35]"; \
		exit 2; \
	fi
	@mkdir -p $(TEACACHE_OUT)
	$(PYTHON) scripts/calibrate_teacache.py fit \
		--trace $(TRACE) \
		--target-skip-rate $(or $(TARGET_SKIP_RATE),0.35) \
		$(if $(FAMILY),--family $(FAMILY),) \
		$(if $(STEPS),--steps $(STEPS),) \
		$(if $(FIT_COEFFICIENTS),--fit-coefficients,) \
		--write-report $(or $(REPORT),$(TEACACHE_OUT)/fit_report.json)

# Fixed long-video chapter parse speed/quality cases (wukong + rainy_night; requires local LLM).
CASE ?= all
RUNS ?= 1
CHAPTER_PARSE_BENCH_OUT := tests/benchmark/outputs/chapter_parse_bench.json

chapter-parse-bench:
	@mkdir -p tests/benchmark/outputs
	PYTHONPATH=. $(PYTHON) -m tests.chapter_parse_benchmark --case $(CASE) --runs $(RUNS) --out $(CHAPTER_PARSE_BENCH_OUT)

chapter-parse-bench-test:
	PYTHONPATH=. $(PYTHON) tests/chapter_parse_benchmark_test.py

# ============================================================================
# Frontend
# ============================================================================

FRONTEND_DIR := $(CURDIR)/frontend

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

# ============================================================================
# Dev server
# ============================================================================

dev start:
	@chmod +x scripts/*.sh
	@./scripts/start.sh

stop:
	@chmod +x scripts/*.sh
	@./scripts/stop.sh

test: verify-engine-stack

test-integration:
	PYTHONPATH=. $(PYTHON) tests/script_parse_integration.py ScriptParseIntegrationTests.test_decompose_and_expand_e2e ScriptParseIntegrationTests.test_wukong_decompose_and_expand_e2e

# ============================================================================
# Quality gates
# ============================================================================

check-consistency:
	$(PYTHON) scripts/check_consistency.py

sync-models-registry:
	$(PYTHON) scripts/sync_workspace_registry.py

ENGINE_GOV := scripts/check_engine_governance.py
FRONTEND_GOV := scripts/check_frontend_governance.py

check-frontend-governance:
	$(PYTHON) $(FRONTEND_GOV)

check-ep-boundary:
	$(PYTHON) $(FRONTEND_GOV) --rule ep

check-theme-legacy:
	$(PYTHON) $(FRONTEND_GOV) --rule theme

check-ui-compat:
	$(PYTHON) $(FRONTEND_GOV) --rule ui

check-canvas-utils:
	$(PYTHON) $(FRONTEND_GOV) --rule canvas

strip-el-tokens:
	$(PYTHON) scripts/strip_el_tokens.py

check-engine-rules:
	$(PYTHON) $(ENGINE_GOV)

check-engine-imports:
	$(PYTHON) $(ENGINE_GOV) --rule imports

check-engine-mlx-torch:
	$(PYTHON) $(ENGINE_GOV) --rule mlx-torch

check-engine-family-layout:
	$(PYTHON) $(ENGINE_GOV) --rule layout

check-engine-family-primitives:
	$(PYTHON) $(ENGINE_GOV) --rule primitives

check-engine-attention-paths:
	$(PYTHON) $(ENGINE_GOV) --rule attention

check-engine-sdpa-paths:
	$(PYTHON) $(ENGINE_GOV) --rule sdpa

check-engine-rope-paths:
	$(PYTHON) $(ENGINE_GOV) --rule rope

check-engine-modulation-paths:
	$(PYTHON) $(ENGINE_GOV) --rule modulation

check-models-registry-contracts:
	$(PYTHON) $(ENGINE_GOV) --rule registry

check-weight-parity:
	$(PYTHON) $(ENGINE_GOV) --rule parity

report-registry-audit:
	$(PYTHON) $(ENGINE_GOV) --report registry

report-family-budget:
	$(PYTHON) $(ENGINE_GOV) --report family-budget

report-family-reuse:
	$(PYTHON) $(ENGINE_GOV) --report reuse

check-engine-governance: check-engine-rules check-consistency check-weight-parity
	@echo "Engine governance suite OK"

verify-engine-stack: check-engine-governance test-engine-unit calibrate-teacache-smoke
	@echo "Engine stack verification OK"

test-engine-unit:
	PYTHONPATH=. $(PYTHON) scripts/test_engine_unit.py

lint:
	$(PYTHON) scripts/make_lint.py
	@echo "Lint OK"

clean:
	$(PYTHON) scripts/clean_build.py

clean-download-cache:
	$(PYTHON) scripts/clean_download_caches.py

# ============================================================================
# Release packaging — pack-<platform>-<product>-<step>
# ============================================================================

pack-prereqs:
	@command -v npm >/dev/null 2>&1 || (printf '%s\n' 'npm not found. Install Node.js: https://nodejs.org/' >&2; exit 1)
	@command -v cargo >/dev/null 2>&1 || (printf '%s\n' 'cargo not found. Install Rust: https://rustup.rs/' >&2; exit 1)
	@echo "pack-prereqs OK (npm + cargo)"

# --- macOS desktop (MLX sidecar only) — build on Darwin arm64 ---

export DANQING_PYINSTALLER_PROFILE ?= mlx

pack-macos-desktop-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=mlx $(PYTHON) scripts/build_sidecar.py

pack-macos-desktop-shell: pack-prereqs
	@./scripts/tauri_build_macos.sh

pack-macos-desktop: frontend-build pack-macos-desktop-sidecar pack-macos-desktop-shell
	@echo "pack-macos-desktop (MLX) -> $(OUT_DIR)/desktop/bundle/"

# --- Linux server (CUDA sidecar only) — build on Linux x86_64 ---

pack-linux-server-venv:
	@test -d .venv || python3.11 -m venv .venv || python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url $(TORCH_INDEX_URL)
	$(PYTHON) -m pip install -r requirements-cuda.txt pyinstaller

pack-linux-server-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=cuda $(PYTHON) scripts/build_sidecar.py

pack-linux-server-archive: pack-linux-server-sidecar
	RELEASE_VERSION=$(RELEASE_VERSION) $(PYTHON) scripts/package_linux_cuda_release.py --version $(RELEASE_VERSION)
	@echo "pack-linux-server-archive -> $(OUT_DIR)/dist/danqing-studio-linux-cuda-x86_64-$(RELEASE_VERSION).tar.gz"

pack-linux-server: pack-linux-server-venv pack-linux-server-archive

# --- Windows desktop (CUDA sidecar only) — build on Windows x86_64 ---

pack-windows-venv:
	@test -d .venv || py -3.11 -m venv .venv || python -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch torchvision --index-url $(TORCH_INDEX_URL)
	$(PYTHON) -m pip install -r requirements-cuda.txt pyinstaller

pack-windows-sidecar: frontend-build
	DANQING_PYINSTALLER_PROFILE=cuda $(PYTHON) scripts/build_sidecar.py

pack-windows-server-archive: pack-windows-sidecar
	RELEASE_VERSION=$(RELEASE_VERSION) $(PYTHON) scripts/package_windows_cuda_release.py --version $(RELEASE_VERSION)
	@echo "pack-windows-server-archive -> $(OUT_DIR)/dist/danqing-studio-windows-cuda-x86_64-$(RELEASE_VERSION).zip"

pack-windows-server: pack-windows-venv pack-windows-server-archive

pack-windows-desktop-shell: pack-prereqs
	$(PYTHON) scripts/tauri_build.py --platform windows

pack-windows-desktop: pack-windows-sidecar pack-windows-desktop-shell
	@echo "pack-windows-desktop -> $(OUT_DIR)/desktop/bundle/"

pack-windows-desktop-release: pack-windows-venv pack-windows-desktop

# ============================================================================
# Deprecated aliases (old names → pack-*)
# ============================================================================

desktop-prereqs: pack-prereqs
desktop-sidecar: pack-macos-desktop-sidecar
desktop-tauri: pack-macos-desktop-shell
desktop-bundle: pack-macos-desktop

linux-cuda-venv: pack-linux-server-venv
linux-cuda-sidecar: pack-linux-server-sidecar
release-linux-cuda-tar: pack-linux-server-archive
release-linux-cuda: pack-linux-server

windows-cuda-venv: pack-windows-venv
windows-cuda-sidecar: pack-windows-sidecar
windows-cuda-desktop-sidecar: pack-windows-sidecar
windows-desktop-tauri: pack-windows-desktop-shell
windows-desktop-bundle: pack-windows-desktop
release-windows-desktop: pack-windows-desktop-release
release-windows-cuda-zip: pack-windows-server-archive
release-windows-cuda: pack-windows-server

# ============================================================================
# Help
# ============================================================================

help:
	@echo "Danmo Make v4 — Makefile"
	@echo ""
	@echo "Benchmark:"
	@echo "  bench-setup / bench-download-judge / bench-eval / bench-eval-smoke / bench-eval-case"
	@echo "  calibrate-teacache-smoke — offline TeaCache fit replay (no GPU)"
	@echo "  calibrate-teacache-run MODEL=... PROMPT=... [STEPS=28] — probe rel_l1 + suggest threshold"
	@echo "  calibrate-teacache-fit TRACE=... — replay saved trace JSON"
	@echo "  chapter-parse-bench [CASE=all|wukong|rainy_night] [RUNS=1] — fixed LLM parse speed cases"
	@echo "  chapter-parse-bench-test — unittest gates (skips if no LLM)"
	@echo ""
	@echo "Frontend:  frontend-install | frontend-dev | frontend-build | frontend-typecheck | frontend-canvas-unit"
	@echo "Dev:       dev | start | stop"
	@echo "Desktop:   (deprecated — use Tauri desktop via pack-macos-desktop)"
	@echo "Test:      test | test-integration"
	@echo "Quality:   lint | check-*"
	@echo "Clean:     clean | clean-download-cache"
	@echo ""
	@echo "Release (pack-<platform>-<product>):"
	@echo "  macOS desktop (MLX):     pack-macos-desktop"
	@echo "  Linux server (CUDA):     pack-linux-server"
	@echo "  Windows desktop (CUDA):  pack-windows-desktop-release   (on Windows)"
	@echo "  Windows server zip:      pack-windows-server              (optional)"
	@echo ""
	@echo "Steps (when not using all-in-one targets above):"
	@echo "  pack-prereqs                  npm + cargo for Tauri"
	@echo "  pack-macos-desktop-sidecar    PyInstaller MLX sidecar"
	@echo "  pack-macos-desktop-shell      Tauri .app / .dmg"
	@echo "  pack-linux-server-venv        CUDA venv (Linux)"
	@echo "  pack-linux-server-sidecar     PyInstaller CUDA sidecar"
	@echo "  pack-linux-server-archive     .tar.gz"
	@echo "  pack-windows-venv             CUDA venv (Windows)"
	@echo "  pack-windows-sidecar          PyInstaller CUDA sidecar"
	@echo "  pack-windows-desktop-shell    Tauri NSIS installer"
	@echo "  pack-windows-server-archive   .zip (headless)"
