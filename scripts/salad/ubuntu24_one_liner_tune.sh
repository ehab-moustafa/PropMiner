#!/usr/bin/env bash
# One-shot production TUNE bootstrap (ubuntu:24.04). Same setup as
# ubuntu24_one_liner_fast.sh but runs the offline autotune instead of mining:
#   batch x graph_batch x cluster_m x carveout at production shape.
# No wallet required. Writes ~/.cache/propminer/autotune.json and prints the winner.
#
# Usage (Salad/vast startup, same shape as the mine one-liner):
#   ... curl .../scripts/salad/ubuntu24_one_liner_tune.sh | bash
#
# Tunables (env, all optional):
#   TUNE_SECONDS=15     seconds per candidate (bigger = slower but more accurate)
#   TUNE_REPEATS=3      repeats per candidate (worst dropped)
#   PROPMINER_N_CAP=131072
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

TAG="${PROPMINER_RELEASE_TAG:-continuous}"
REPO="${PROPMINER_RELEASE_REPO:-ehab-moustafa/PropMiner}"
ASSET="${PROPMINER_RELEASE_ASSET:-propminer-rtx5090-linux-amd64.tar.gz}"
RELEASE_URL="${PROPMINER_RELEASE_URL:-https://github.com/${REPO}/releases/download/${TAG}/${ASSET}}"

CUDA_ROOT="/usr/local/cuda-12.8/targets/x86_64-linux"
CUDA_LIB="${CUDA_ROOT}/lib"
CUDA_LIB64="/usr/local/cuda/lib64"
PM_DIR="/opt/propminer"
NVIDIA_REDIST="https://developer.download.nvidia.com/compute/cuda/redist"

export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
export PROPMINER_AUTOTUNE_REPEATS="${TUNE_REPEATS:-3}"
TUNE_SECONDS="${TUNE_SECONDS:-15}"

fail_keepalive() {
    echo "[tune] FATAL: $*" >&2
    echo "[tune] Container staying alive for logs (sleep infinity)..." >&2
    exec sleep infinity
}

curl_pkg() {
    local name="$1" version="$2"
    local out="/tmp/${name}.tar.xz"
    local url="${NVIDIA_REDIST}/${name}/linux-x86_64/${name}-linux-x86_64-${version}-archive.tar.xz"
    echo "[cuda] ${name}"
    curl -fsSL --retry 5 --retry-delay 3 -o "${out}" "${url}"
    tar -xf "${out}" --strip-components=1 -C "${CUDA_ROOT}"
    rm -f "${out}"
}

link_cuda_redist() {
    [[ -d "${CUDA_LIB}" ]] || fail_keepalive "missing CUDA lib dir ${CUDA_LIB}"
    cd "${CUDA_LIB}"
    if [[ ! -f libcudart.so.12.8.57 && ! -L libcudart.so.12 ]]; then
        fail_keepalive "libcudart not found under ${CUDA_LIB}"
    fi
    [[ -f libcudart.so.12.8.57 ]] && ln -sf libcudart.so.12.8.57 libcudart.so.12
    [[ -f libcudart.so.12.8.57 ]] && ln -sf libcudart.so.12.8.57 libcudart.so
    mkdir -p "${CUDA_LIB64}"
    ln -sf "${CUDA_LIB}/libcudart.so.12" "${CUDA_LIB64}/libcudart.so.12"
    ln -sf "${CUDA_LIB}/libcudart.so.12" "${CUDA_LIB64}/libcudart.so"
    [[ -f "$(readlink -f "${CUDA_LIB}/libcudart.so.12")" ]] || \
        fail_keepalive "broken libcudart.so.12 symlink under ${CUDA_LIB}"
}

setup_runtime_env() {
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
    export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export LD_LIBRARY_PATH="${PM_DIR}:${CUDA_LIB}:${CUDA_LIB64}:${LD_LIBRARY_PATH:-}"

    if [[ -e /dev/dxg ]]; then
        echo "[env] WSL2 detected (/dev/dxg)"
        local wsl_libs=() so f
        for so in \
            /usr/lib/x86_64-linux-gnu/libdxcore.so \
            /usr/lib/wsl/drivers/*/libnvdxgdmal.so.1 \
            /usr/lib/wsl/drivers/*/libcuda_loader.so \
            /usr/lib/wsl/drivers/*/libnvidia-ml_loader.so \
            /usr/lib/wsl/drivers/*/libcuda.so.1.1; do
            for f in ${so}; do
                [[ -f "${f}" ]] && wsl_libs+=("${f}")
            done
        done
        if [[ ${#wsl_libs[@]} -gt 0 ]]; then
            export LD_PRELOAD
            LD_PRELOAD="$(IFS=:; echo "${wsl_libs[*]}")"
            echo "[env] LD_PRELOAD set for WSL2"
        fi
        export LD_LIBRARY_PATH="${PM_DIR}:${CUDA_LIB}:${CUDA_LIB64}:/usr/lib/wsl/drivers:${LD_LIBRARY_PATH}"
    elif [[ -e /dev/nvidia0 ]]; then
        echo "[env] Native Linux NVIDIA detected (/dev/nvidia0)"
    else
        echo "[env] WARN: no /dev/dxg or /dev/nvidia0 — GPU may not work"
    fi
}

apt-get update
apt-get install -y curl ca-certificates libssl3 xz-utils

mkdir -p "${CUDA_ROOT}" "${CUDA_LIB64}" "${PM_DIR}"

curl_pkg cuda_cudart 12.8.57
link_cuda_redist

echo "[release] downloading ${ASSET}"
if ! curl -fsSL --retry 5 --retry-delay 3 -o /tmp/propminer-release.tar.gz "${RELEASE_URL}"; then
    fail_keepalive "release download failed — check https://github.com/${REPO}/releases/tag/${TAG}"
fi
tar xzf /tmp/propminer-release.tar.gz -C "${PM_DIR}" || fail_keepalive "tar extract failed for ${ASSET}"
chmod +x "${PM_DIR}/propminer"

setup_runtime_env

echo "[tune] Starting production autotune (version=$(cat "${PM_DIR}/VERSION" 2>/dev/null || echo unknown))"
echo "[tune] seconds/candidate=${TUNE_SECONDS} repeats=${PROPMINER_AUTOTUNE_REPEATS} n_cap=${PROPMINER_N_CAP}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true

TUNE_LOG="/tmp/propminer-tune.log"
(cd "${PM_DIR}" && LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${LD_PRELOAD:-}" \
    ./propminer --tune-autotune "${TUNE_SECONDS}" --rtx5090 --gpus 0 2>&1 | tee "${TUNE_LOG}") || true

echo ""
echo "===== TUNE RESULT ====="
grep -E "\[autotune\] Winner|Cached at" "${TUNE_LOG}" || echo "[tune] no winner line found (check ${TUNE_LOG})"
echo "----- autotune.json -----"
cat "${HOME}/.cache/propminer/autotune.json" 2>/dev/null || echo "[tune] no cache file at ${HOME}/.cache/propminer/autotune.json"
echo "========================="
echo "[tune] Done. Copy winners into env for mining: PROPMINER_BATCH / PROPMINER_GRAPH_BATCH / PEARL_GEMM_CONSUMER_CLUSTER_M"
echo "[tune] Container staying alive so you can read logs..."
exec sleep infinity
