#!/usr/bin/env bash
# Nsight Compute profiling helper for steady-state Pearl GEMM on 5090.
# Run on the GPU box after a warm bench/mine session.
#
# Usage: ./scripts/profile_gemm_ncu.sh [gpu_index]
set -euo pipefail

GPU="${1:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/ncu_gemm_$(date +%Y%m%d_%H%M%S)"

if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu not found; install CUDA Nsight Compute" >&2
    exit 1
fi

mkdir -p "${OUT}"

echo "[ncu] Warmup + profile GPU ${GPU} (60s bench)..."
cd "${ROOT}/build"
PROPMINER_BENCH_JSON=1 ./propminer --bench 60 --rtx5090 --gpus "${GPU}" \
    > "${OUT}/bench.log" 2>&1 || true

echo "[ncu] Capture (adjust kernel name if needed)..."
ncu --set full \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name "regex:transcript_gemm" \
    --launch-skip 3 --launch-count 5 \
    -o "${OUT}/gemm" \
    ./propminer --bench 30 --rtx5090 --gpus "${GPU}" \
    2>&1 | tee "${OUT}/ncu.log"

echo "[ncu] Report: ${OUT}/gemm.ncu-rep"
echo "[ncu] Key metrics: sm__pipe_tensor_cycles_active, sm__warps_active.avg.pct_of_peak_sustained_active"
