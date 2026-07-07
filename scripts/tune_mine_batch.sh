#!/usr/bin/env bash
# DEPRECATED — use ./scripts/tune_runtime_prod.sh or PROPMINER_MODE=tune-prod
# (unified batch + graph_batch + cluster autotune).
#
# Usage:
#   ./scripts/tune_mine_batch.sh [seconds_per_candidate] [repeats]
#
# Writes ~/.cache/propminer/mine_batch.json and logs to build/mine_batch_tune.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build}"
SECONDS_PER="${1:-12}"
REPEATS="${2:-2}"

source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

mkdir -p "${BUILD_DIR}"

if [[ ! -x "${BUILD_DIR}/propminer" ]]; then
    echo "[batch-tune] Building propminer..."
    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DCMAKE_BUILD_TYPE=Release \
        > "${BUILD_DIR}/cmake-batch-tune.log" 2>&1
    cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc)" \
        > "${BUILD_DIR}/build-batch-tune.log" 2>&1
fi

export PEARL_GEMM_CONSUMER_CLUSTER_M="${PEARL_GEMM_CONSUMER_CLUSTER_M:-2}"
export PROPMINER_BATCH_TUNE_REPEATS="${REPEATS}"
unset PROPMINER_BATCH || true
unset PROPMINER_AUTOTUNE || true

echo "[batch-tune] Sweeping mine batch (${SECONDS_PER}s min/candidate, ${REPEATS} repeats)..."
run_propminer "${BUILD_DIR}/propminer" \
    --tune-mine-batch "${SECONDS_PER}" --rtx5090 --gpus 0 \
    2>&1 | tee "${BUILD_DIR}/mine_batch_tune.log"

echo "[batch-tune] Cache: $(python3 - <<'PY' 2>/dev/null || echo ~/.cache/propminer/mine_batch.json
import os
xdg = os.environ.get("XDG_CACHE_HOME")
home = os.environ.get("HOME", "/tmp")
print((xdg or f"{home}/.cache") + "/propminer/mine_batch.json")
PY
)"
