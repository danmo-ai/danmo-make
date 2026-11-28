#!/usr/bin/env bash
# Build the danqing-api sidecar for desktop / server packs.
# Desktop launcher expects the binary under out/sidecar/danqing-api/.
#
# Features:
#   macOS  → metal,mlx-metal
#   Linux  → mlx-cuda when nvcc is on PATH (toolkit only; no GPU needed to compile)
#   Windows → vulkan (+ cuda if CUDA_PATH is set)
#
# Override: DANQING_SIDECAR_FEATURES=mlx-cuda,metal,... (comma-separated).
#
# macOS / Linux MLX packs also stage libmlx / libmlxc (and mlx.metallib on
# Darwin) next to the binary and ensure an @executable_path / $ORIGIN rpath so
# dyld/ld.so can resolve @rpath deps outside the build tree.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEATURES=()
OS_NAME="$(uname -s 2>/dev/null || echo unknown)"

if [[ -n "${DANQING_SIDECAR_FEATURES:-}" ]]; then
  IFS=',' read -r -a FEATURES <<< "${DANQING_SIDECAR_FEATURES}"
else
  case "$OS_NAME" in
    MINGW*|MSYS*|CYGWIN*)
      FEATURES+=(vulkan)
      if [[ -n "${CUDA_PATH:-}" ]] || command -v nvcc >/dev/null 2>&1; then
        FEATURES+=(mlx-cuda)
      fi
      ;;
    Darwin)
      FEATURES+=(metal mlx-metal)
      ;;
    Linux)
      if command -v nvcc >/dev/null 2>&1; then
        FEATURES+=(mlx-cuda)
      else
        echo "WARNING: nvcc not found — building danqing-api without mlx-cuda (no MLX GPU backend)" >&2
      fi
      ;;
  esac
fi

ARGS=()
if [[ ${#FEATURES[@]} -gt 0 ]]; then
  IFS=,
  ARGS+=(--features "${FEATURES[*]}")
  unset IFS
  echo "sidecar features: ${FEATURES[*]}"
else
  echo "sidecar features: (default / none)"
fi

if [[ "$OS_NAME" == Linux && "${CI:-}" == "true" ]]; then
  if [[ " ${FEATURES[*]} " != *mlx-cuda* ]]; then
    echo "ERROR: CI Linux release must build danqing-api with mlx-cuda (set DANQING_SIDECAR_FEATURES=mlx-cuda)" >&2
    exit 1
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: mlx-cuda sidecar build requires nvcc on PATH" >&2
    exit 1
  fi
fi

cargo build --release --manifest-path "$ROOT/native/Cargo.toml" -p danmo-server "${ARGS[@]}"
mkdir -p "$ROOT/out/sidecar/danqing-api"
SRC="$ROOT/native/target/release/danqing-api"
DEST="$ROOT/out/sidecar/danqing-api"
# Wipe previous staged runtime libs so a feature flip does not leave stale dylibs.
rm -rf "$DEST"
mkdir -p "$DEST"

if [[ "$OS_NAME" == MINGW* || "$OS_NAME" == MSYS* || "$OS_NAME" == CYGWIN* ]]; then
  cp "${SRC}.exe" "$DEST/danqing-api.exe"
else
  cp "$SRC" "$DEST/danqing-api"
  chmod +x "$DEST/danqing-api"
fi

needs_mlx_runtime=0
case " ${FEATURES[*]} " in
  *" mlx-metal "*|*" mlx-cuda "*|*" mlx "*) needs_mlx_runtime=1 ;;
esac

# Newest danmo-mlx-sys out/lib that contains libmlxc (shared by Metal / CUDA builds).
resolve_mlx_lib_dir() {
  local target_dir="${CARGO_TARGET_DIR:-$ROOT/native/target}"
  local candidates=()
  local d
  while IFS= read -r -d '' d; do
    if [[ -f "$d/libmlxc.dylib" || -f "$d/libmlxc.so" ]]; then
      candidates+=("$d")
    fi
  done < <(find "$target_dir" -path '*/danmo-mlx-sys-*/out/lib' -type d -print0 2>/dev/null)

  if [[ ${#candidates[@]} -eq 0 && -n "${DANQING_MLX_PREBUILT:-}" ]]; then
    if [[ -d "${DANQING_MLX_PREBUILT}/lib" ]]; then
      echo "${DANQING_MLX_PREBUILT}/lib"
      return 0
    fi
  fi
  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi

  local newest="" newest_mtime=0 m
  for d in "${candidates[@]}"; do
    m=$(stat -c %Y "$d" 2>/dev/null || stat -f %m "$d" 2>/dev/null || echo 0)
    if (( m >= newest_mtime )); then
      newest_mtime=$m
      newest=$d
    fi
  done
  echo "$newest"
}

# Copy shared libs (+ metallib) next to danqing-api and pin runtime rpaths.
stage_mlx_runtime() {
  local dest="$1"
  local lib_dir metal_dir
  if ! lib_dir="$(resolve_mlx_lib_dir)"; then
    echo "ERROR: could not locate libmlxc next to danmo-mlx-sys build output" >&2
    echo "       (expected under \$CARGO_TARGET_DIR/.../danmo-mlx-sys-*/out/lib)" >&2
    exit 1
  fi
  echo "sidecar MLX libs: $lib_dir"

  local f
  for f in "$lib_dir"/libmlx*.dylib "$lib_dir"/libmlx*.so "$lib_dir"/libmlx*.so.* \
           "$lib_dir"/libjaccl*.dylib "$lib_dir"/libjaccl*.so "$lib_dir"/libjaccl*.so.*; do
    [[ -e "$f" ]] || continue
    cp -a "$f" "$dest/"
  done

  metal_dir="$(dirname "$lib_dir")/metal"
  local metallib=""
  local candidate
  for candidate in \
    "$metal_dir/mlx.metallib" \
    "$lib_dir/mlx.metallib" \
    "${DANQING_MLX_PREBUILT:+${DANQING_MLX_PREBUILT}/lib/mlx.metallib}" \
    "${DANQING_MLX_PREBUILT:+${DANQING_MLX_PREBUILT}/share/mlx/mlx.metallib}"; do
    [[ -n "$candidate" && -f "$candidate" ]] || continue
    metallib="$candidate"
    break
  done
  if [[ -n "$metallib" ]]; then
    cp -a "$metallib" "$dest/"
  elif [[ "$OS_NAME" == Darwin ]]; then
    echo "ERROR: mlx.metallib missing (looked under $metal_dir, $lib_dir, DANQING_MLX_PREBUILT)" >&2
    exit 1
  fi

  if [[ "$OS_NAME" == Darwin ]]; then
    if [[ ! -f "$dest/libmlxc.dylib" || ! -f "$dest/libmlx.dylib" ]]; then
      echo "ERROR: staged sidecar is missing libmlx.dylib / libmlxc.dylib" >&2
      ls -la "$dest" >&2 || true
      exit 1
    fi
    # Binary: prefer colocated libs via @executable_path (absolute build rpaths die after pack).
    if command -v install_name_tool >/dev/null 2>&1 && command -v otool >/dev/null 2>&1; then
      if ! otool -l "$dest/danqing-api" | grep -A2 LC_RPATH | grep -Fq '@executable_path'; then
        install_name_tool -add_rpath '@executable_path' "$dest/danqing-api"
      fi
      for f in "$dest"/*.dylib; do
        [[ -f "$f" ]] || continue
        if ! otool -l "$f" | grep -A2 LC_RPATH | grep -Fq '@loader_path'; then
          install_name_tool -add_rpath '@loader_path' "$f" 2>/dev/null || true
        fi
      done
      if ! otool -l "$dest/danqing-api" | grep -q LC_RPATH; then
        echo "ERROR: danqing-api has no LC_RPATH after staging (dyld cannot resolve @rpath/libmlxc.dylib)" >&2
        exit 1
      fi
    else
      echo "WARNING: install_name_tool/otool missing — rely on link-time @executable_path rpath" >&2
    fi
  elif [[ "$OS_NAME" == Linux ]]; then
    if [[ ! -f "$dest/libmlxc.so" && ! -e "$dest/libmlxc.so.0" ]]; then
      # Some builds emit only libmlxc.so without a soname symlink.
      if ! compgen -G "$dest/libmlxc.so*" >/dev/null; then
        echo "ERROR: staged sidecar is missing libmlxc.so*" >&2
        ls -la "$dest" >&2 || true
        exit 1
      fi
    fi
    if command -v patchelf >/dev/null 2>&1; then
      # Ensure $ORIGIN even if the link-time rpath was stripped.
      local current
      current="$(patchelf --print-rpath "$dest/danqing-api" 2>/dev/null || true)"
      case ":$current:" in
        *':$ORIGIN:'*) ;;
        *)
          if [[ -n "$current" ]]; then
            patchelf --set-rpath "\$ORIGIN:$current" "$dest/danqing-api"
          else
            patchelf --set-rpath '$ORIGIN' "$dest/danqing-api"
          fi
          ;;
      esac
    fi
  fi
}

if [[ "$needs_mlx_runtime" == 1 ]]; then
  if [[ "$OS_NAME" == Darwin || "$OS_NAME" == Linux ]]; then
    stage_mlx_runtime "$DEST"
  fi
fi

echo "sidecar: $DEST/ (${FEATURES[*]:-none})"
