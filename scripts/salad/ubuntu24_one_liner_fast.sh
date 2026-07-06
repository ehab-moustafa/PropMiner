#!/usr/bin/env bash
# Recommended SRBMiner-style Salad bootstrap: ubuntu:24.04 only.
#
# Downloads CUDA 12.8 cudart from NVIDIA (~1.3 MB) plus the small GitHub
# release tarball (~1.2 MB). PropMiner uses custom GEMM cubins — not libcublas.
# libcuda.so.1 comes from the Salad host driver (WSL /dev/dxg).
#
# Salad setup:
#   Image: ubuntu:24.04
#   Env:   PROPMINER_WALLET=krxYOURUSER  (optional: PROPMINER_WORKER=rig01)
#   Startup command: paste the single line from ubuntu24_one_liner_fast.oneline
#
# Local test:
#   PROPMINER_WALLET=krxTEST.worker bash scripts/salad/ubuntu24_one_liner_fast.sh
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

TAG="${PROPMINER_RELEASE_TAG:-continuous}"
REPO="${PROPMINER_RELEASE_REPO:-ehab-moustafa/PropMiner}"
ASSET="${PROPMINER_RELEASE_ASSET:-propminer-rtx5090-linux-amd64.tar.gz}"
RELEASE_URL="${PROPMINER_RELEASE_URL:-https://github.com/${REPO}/releases/download/${TAG}/${ASSET}}"

CUDA_DEST="/usr/local/cuda-12.8/targets/x86_64-linux/lib"
CUDA_LIB64="/usr/local/cuda/lib64"
PM_DIR="/opt/propminer"
NVIDIA_REDIST="https://developer.download.nvidia.com/compute/cuda/redist"

POOL="${PROPMINER_POOL:-}"
if [[ -z "${POOL}" ]]; then
    POOL="${PROPMINER_POOL_FALLBACK:-prl.kryptex.network:443,prl-eu.kryptex.network:443}"
fi

fail_keepalive() {
    echo "[salad] FATAL: $*" >&2
    echo "[salad] Container staying alive for logs (sleep infinity)..." >&2
    exec sleep infinity
}

curl_pkg() {
    local name="$1"
    local version="$2"
    local out="/tmp/${name}.tar.xz"
    local url="${NVIDIA_REDIST}/${name}/linux-x86_64/${name}-linux-x86_64-${version}-archive.tar.xz"
    echo "[cuda] ${name}"
    curl -fsSL --retry 5 --retry-delay 3 -o "${out}" "${url}"
    tar -xf "${out}" --strip-components=1 -C "${CUDA_DEST}"
    rm -f "${out}"
}

link_cuda_redist() {
    cd "${CUDA_DEST}"
    ln -sf libcudart.so.12.8.57 libcudart.so.12
    ln -sf libcudart.so.12.8.57 libcudart.so

    mkdir -p "${CUDA_LIB64}"
    ln -sf "${CUDA_DEST}/libcudart.so.12" "${CUDA_LIB64}/libcudart.so.12"
    ln -sf "${CUDA_DEST}/libcudart.so.12" "${CUDA_LIB64}/libcudart.so"

    if [[ ! -f "$(readlink -f "${CUDA_DEST}/libcudart.so.12")" ]]; then
        fail_keepalive "broken libcudart.so.12 symlink under ${CUDA_DEST}"
    fi
}

setup_runtime_env() {
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
    export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export PEARL_GEMM_CONSUMER_CLUSTER_M="${PEARL_GEMM_CONSUMER_CLUSTER_M:-2}"
    export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"
    export LD_LIBRARY_PATH="${CUDA_DEST}:${CUDA_LIB64}:${PM_DIR}:${LD_LIBRARY_PATH:-}"

    if [[ -e /dev/dxg ]]; then
        echo "[env] WSL2 detected (/dev/dxg)"
        export LD_LIBRARY_PATH="/usr/lib/wsl/drivers:${LD_LIBRARY_PATH}"
        local wsl_libs=()
        local so
        for so in \
            /usr/lib/x86_64-linux-gnu/libdxcore.so \
            /usr/lib/wsl/drivers/*/libnvdxgdmal.so.1 \
            /usr/lib/wsl/drivers/*/libcuda_loader.so \
            /usr/lib/wsl/drivers/*/libnvidia-ml_loader.so \
            /usr/lib/wsl/drivers/*/libcuda.so.1.1; do
            local f
            for f in ${so}; do
                [[ -f "${f}" ]] && wsl_libs+=("${f}")
            done
        done
        if [[ ${#wsl_libs[@]} -gt 0 ]]; then
            export LD_PRELOAD
            LD_PRELOAD="$(IFS=:; echo "${wsl_libs[*]}")"
            echo "[env] LD_PRELOAD set for WSL2"
        fi
    elif [[ -e /dev/nvidia0 ]]; then
        echo "[env] Native Linux NVIDIA detected"
    else
        echo "[env] No GPU device nodes; trying bundled CUDA runtime"
    fi
}

resolve_wallet() {
    local wallet="${PROPMINER_WALLET:-}"
    local worker="${PROPMINER_WORKER:-}"
    if [[ -z "${worker}" && "${wallet}" == *.* ]]; then
        worker="${wallet#*.}"
        wallet="${wallet%%.*}"
        echo "[mine] Parsed wallet.worker -> wallet=${wallet} worker=${worker}"
    fi
    if [[ -z "${wallet}" ]]; then
        fail_keepalive "set PROPMINER_WALLET (e.g. krxYOURUSER.worker1)"
    fi
    WALLET_RESOLVED="${wallet}"
    WORKER_RESOLVED="${worker}"
}

apt-get update
apt-get install -y curl ca-certificates libssl3

mkdir -p "${CUDA_DEST}" "${CUDA_LIB64}" "${PM_DIR}"

curl_pkg cuda_cudart 12.8.57
link_cuda_redist

echo "[release] downloading ${ASSET}"
if ! curl -fsSL --retry 5 --retry-delay 3 -o /tmp/propminer-release.tar.gz "${RELEASE_URL}"; then
    fail_keepalive "release download failed — check https://github.com/${REPO}/releases/tag/${TAG}"
fi
if ! tar xzf /tmp/propminer-release.tar.gz -C "${PM_DIR}"; then
    fail_keepalive "tar extract failed for ${ASSET}"
fi
chmod +x "${PM_DIR}/propminer"

setup_runtime_env
resolve_wallet

GPUS="${PROPMINER_GPUS:-0}"
RESTART_ON_EXIT="${PROPMINER_RESTART_ON_EXIT:-1}"
MINER_ARGS=(--rtx5090 --gpus "${GPUS}" --pool "${POOL}" --wallet "${WALLET_RESOLVED}")
if [[ -n "${WORKER_RESOLVED}" ]]; then
    MINER_ARGS+=(--worker "${WORKER_RESOLVED}")
fi

echo "[salad] Starting propminer (version=$(cat "${PM_DIR}/VERSION" 2>/dev/null || echo unknown))"
echo "[salad] wallet=${WALLET_RESOLVED} worker=${WORKER_RESOLVED:-<default>}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true

run_once() {
    (cd "${PM_DIR}" && LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${LD_PRELOAD:-}" \
        ./propminer "${MINER_ARGS[@]}")
}

if [[ "${RESTART_ON_EXIT}" == "0" ]]; then
    run_once || fail_keepalive "propminer exited"
    exit 0
fi

while true; do
    set +e
    run_once
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 && -n "${PROPMINER_ONCE:-}" ]]; then
        echo "[mine] clean exit (PROPMINER_ONCE=1)"
        exit 0
    fi
    echo "[mine] propminer exited rc=${rc}; restart in 10s"
    sleep 10
done
