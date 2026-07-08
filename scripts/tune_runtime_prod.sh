#!/usr/bin/env bash
# RTX 5090 / Stratum production runtime autotune.
#
# Default (PROPMINER_TUNE_ISOLATED=1): process-isolated FULL sweep
#   batch × graph_batch × cluster × graph on/off — each combo is its own process.
#
# Quick (PROPMINER_TUNE_QUICK=1): batch-only sweep (legacy tune_runtime_safe.sh).
#
# Legacy (PROPMINER_TUNE_ISOLATED=0): monolithic --tune-autotune → autotune.json
#   (fragile on wedge-prone Blackwell driver stacks).
#
# Usage: ./scripts/tune_runtime_prod.sh [seconds_per_combo] [repeats]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build_remote_test}"
SECONDS_PER="${1:-15}"
REPEATS="${2:-3}"

if [[ "${PROPMINER_TUNE_ISOLATED:-1}" == "1" ]]; then
    if [[ "${PROPMINER_TUNE_QUICK:-0}" == "1" ]]; then
        echo "[autotune] Quick batch-only sweep (PROPMINER_TUNE_QUICK=1)"
        exec "${ROOT}/scripts/tune_runtime_safe.sh" "${SECONDS_PER}"
    fi
    echo "[autotune] Full process-isolated sweep (batch × graph_batch × cluster × graph on/off)"
    echo "[autotune] Set PROPMINER_TUNE_QUICK=1 for batch-only, PROPMINER_TUNE_ISOLATED=0 for legacy"
    exec "${ROOT}/scripts/tune_runtime_full.sh" "${SECONDS_PER}"
fi

source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

mkdir -p "${BUILD_DIR}"

if [[ ! -x "${BUILD_DIR}/propminer" ]]; then
    echo "[autotune] Building propminer..."
    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc 2>/dev/null || echo 4)"
fi

export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
export PROPMINER_AUTOTUNE_REPEATS="${REPEATS}"
unset PROPMINER_BATCH PROPMINER_GRAPH_BATCH PEARL_GEMM_CONSUMER_CLUSTER_M \
      PROPMINER_BATCH_TUNE PROPMINER_AUTOTUNE || true

LOG="${BUILD_DIR}/runtime_autotune.log"
echo "[autotune] Legacy monolithic sweep (${SECONDS_PER}s x ${REPEATS} repeats; Stratum shape)"
run_propminer "${BUILD_DIR}/propminer" \
    --tune-autotune "${SECONDS_PER}" --rtx5090 --gpus 0 \
    2>&1 | tee "${LOG}"

echo "[autotune] Log: ${LOG}"
echo "[autotune] Cache: ${HOME}/.cache/propminer/autotune.json"
