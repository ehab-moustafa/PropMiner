#!/usr/bin/env bash
# RTX 5090 cluster_m sweep + full runtime autotune comparison.
#
# Usage:
#   ./scripts/tune_cluster_sweep.sh [seconds_per_cluster] [cluster_repeats] [autotune_seconds]
#
# Phase 1: cluster_m in {1, 2, 4} — kernel does NOT support 3 (falls back to 1).
# Phase 2: full PROPMINER autotune (M/N, batch, graph, cluster_m, carveout).
#
# Logs: build/cluster_sweep.log
# Apply winner: export PEARL_GEMM_CONSUMER_CLUSTER_M=N
# Or use cached autotune: PROPMINER_AUTOTUNE=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build}"
CLUSTER_SECONDS="${1:-20}"
CLUSTER_REPEATS="${2:-3}"
AUTOTUNE_SECONDS="${3:-12}"
BENCH_BATCH="${PROPMINER_BENCH_BATCH:-4}"

source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

export PROPMINER_CLUSTER_TUNE_REPEATS="${CLUSTER_REPEATS}"
export PROPMINER_AUTOTUNE_REPEATS=3
export PROPMINER_BENCH_BATCH="${BENCH_BATCH}"
unset PEARL_GEMM_CONSUMER_CLUSTER_M || true

LOG="${BUILD_DIR}/cluster_sweep.log"
mkdir -p "${BUILD_DIR}"

if [[ ! -x "${BUILD_DIR}/propminer" ]]; then
  echo "[cluster-sweep] Building propminer..."
  cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROP_MINER_CUDA_ARCH=blackwell \
    -DCMAKE_CUDA_ARCHITECTURES=120a
  cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
fi

{
  echo "===== PropMiner cluster sweep $(date) ====="
  echo "cluster_seconds=${CLUSTER_SECONDS} repeats=${CLUSTER_REPEATS} batch=${BENCH_BATCH}"
  echo ""
  echo "=== Phase 1: cluster_m sweep {1,2,4} ==="
  "${BUILD_DIR}/propminer" --tune-cluster "${CLUSTER_SECONDS}" --rtx5090 --gpus 0
  echo ""
  echo "=== Phase 2: full runtime autotune (${AUTOTUNE_SECONDS}s x 3) ==="
  export PROPMINER_AUTOTUNE_REPEATS=3
  "${BUILD_DIR}/propminer" --tune-autotune "${AUTOTUNE_SECONDS}" --rtx5090 --gpus 0
  echo ""
  echo "===== Done ====="
  echo "Compare Phase 1 vs Phase 2 GMAC/s above."
  echo "Prod options:"
  echo "  export PEARL_GEMM_CONSUMER_CLUSTER_M=<winner>"
  echo "  PROPMINER_AUTOTUNE=1  # uses ~/.cache/propminer/autotune.json"
} 2>&1 | tee "${LOG}"

echo "[cluster-sweep] Log: ${LOG}"
