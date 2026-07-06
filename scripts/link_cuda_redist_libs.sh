#!/usr/bin/env bash
# Create soname symlinks for CUDA 12.8 redist libs (Salad /dev/dxg path).
# On devel images /usr/local/cuda/lib64 may be the same directory as CUDA_LIB;
# never overwrite .so.12 with an absolute path to itself.
set -euo pipefail

CUDA_LIB="${1:-/usr/local/cuda-12.8/targets/x86_64-linux/lib}"
LIB64="${2:-/usr/local/cuda/lib64}"

if [[ ! -d "${CUDA_LIB}" ]]; then
    echo "[link_cuda_redist] missing CUDA_LIB=${CUDA_LIB}" >&2
    exit 1
fi

cd "${CUDA_LIB}"

link_soname() {
    local versioned="$1"
    local soname="$2"
    if [[ -f "${versioned}" ]]; then
        ln -sf "${versioned}" "${soname}"
    fi
}

link_soname libcudart.so.12.8.57 libcudart.so.12
link_soname libcudart.so.12.8.57 libcudart.so
link_soname libnvrtc.so.12.8.61 libnvrtc.so.12
link_soname libcublas.so.12.8.3.14 libcublas.so.12
link_soname libcublasLt.so.12.8.3.14 libcublasLt.so.12
link_soname libnvJitLink.so.12.8.61 libnvJitLink.so.12

cuda_lib_real="$(readlink -f "${CUDA_LIB}")"
lib64_real="$(readlink -f "${LIB64}" 2>/dev/null || true)"
if [[ -z "${lib64_real}" || "${lib64_real}" != "${cuda_lib_real}" ]]; then
    mkdir -p "${LIB64}"
    for lib in libcudart libnvrtc libcublas libcublasLt libnvJitLink; do
        ln -sf "${CUDA_LIB}/${lib}.so.12" "${LIB64}/${lib}.so.12"
    done
    ln -sf "${CUDA_LIB}/libcudart.so.12" "${LIB64}/libcudart.so"
fi

resolved="$(readlink -f "${CUDA_LIB}/libcudart.so.12")"
if [[ ! -f "${resolved}" ]]; then
    echo "[link_cuda_redist] broken libcudart.so.12 -> ${resolved}" >&2
    exit 1
fi
