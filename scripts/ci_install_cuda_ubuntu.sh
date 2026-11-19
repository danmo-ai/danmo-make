#!/usr/bin/env bash
# Install CUDA toolkit + cuDNN for compile-only MLX CUDA builds (no GPU required).
# Matches MLX docs: cuda-toolkit 12.9 + libcudnn9-dev-cuda-12 + BLAS/LAPACK.
# Usage: bash scripts/ci_install_cuda_ubuntu.sh
set -euo pipefail

if [[ "$(uname -s)" != Linux ]]; then
  echo "ci_install_cuda_ubuntu.sh is for Linux only" >&2
  exit 1
fi

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) CUDA_REPO_ARCH=x86_64 ;;
  aarch64|arm64) CUDA_REPO_ARCH=sbsa ;;
  *)
    echo "unsupported arch for CUDA apt repo: $ARCH" >&2
    exit 1
    ;;
esac

# Pin 12.9: MLX rejects CUDA Toolkit 13.1 specifically; 12.x is the documented path.
CUDA_MAJOR_MINOR="${DANQING_CUDA_VERSION:-12-9}"
CUDA_HOME_DIR="/usr/local/cuda-${CUDA_MAJOR_MINOR/-/.}"

echo "==> NVIDIA CUDA keyring (ubuntu2204/${CUDA_REPO_ARCH})"
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${CUDA_REPO_ARCH}/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

echo "==> apt install cuda-compiler/libraries + cuDNN + BLAS"
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  "cuda-compiler-${CUDA_MAJOR_MINOR}" \
  "cuda-libraries-dev-${CUDA_MAJOR_MINOR}" \
  "cuda-nvrtc-dev-${CUDA_MAJOR_MINOR}" \
  libcudnn9-dev-cuda-12 \
  libblas-dev \
  liblapack-dev \
  liblapacke-dev \
  libopenblas-dev

if [[ ! -x "${CUDA_HOME_DIR}/bin/nvcc" ]]; then
  # Some apt layouts use /usr/local/cuda symlink only.
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME_DIR=/usr/local/cuda
  else
    echo "nvcc missing under ${CUDA_HOME_DIR} or /usr/local/cuda" >&2
    exit 1
  fi
fi

export PATH="${CUDA_HOME_DIR}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME_DIR}"
export CUDA_PATH="${CUDA_HOME_DIR}"
# Compile for common datacenter/consumer arches without a live GPU.
export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-75;80;86;89;90}"
export CUDAARCHS="${CUDAARCHS:-75;80;86;89;90}"
# MLX cuda backend queries a physical GPU when unset; pin archs for compile-only CI.
export MLX_CUDA_ARCHITECTURES="${MLX_CUDA_ARCHITECTURES:-${CMAKE_CUDA_ARCHITECTURES}}"

if [[ -n "${GITHUB_PATH:-}" ]]; then
  echo "${CUDA_HOME_DIR}/bin" >> "$GITHUB_PATH"
fi
if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "CUDA_HOME=${CUDA_HOME}"
    echo "CUDA_PATH=${CUDA_PATH}"
    echo "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    echo "CUDAARCHS=${CUDAARCHS}"
    echo "MLX_CUDA_ARCHITECTURES=${MLX_CUDA_ARCHITECTURES}"
  } >> "$GITHUB_ENV"
fi

echo "==> nvcc report"
nvcc --version
echo "CUDA_HOME=${CUDA_HOME}"
