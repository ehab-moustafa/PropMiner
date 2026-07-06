#!/usr/bin/env bash
# Validate host spin-wait + GPU-timer hashrate changes in gpu_worker.cpp
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== PropMiner host sync + GPU hashrate validation =="

echo "[1/4] Grep: spin wait + GPU timing events..."
rg -q 'spin_wait_batch_event' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "spin_wait_batch_event missing"
rg -q 'batch_start_event' "${ROOT}/src/host/pearl/gpu_worker.h" \
  || fail "batch_start_event missing from gpu_worker.h"
rg -q 'cudaEventElapsedTime' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "cudaEventElapsedTime missing"
rg 'cudaEventCreateWithFlags.*batch_done|batch_done.*cudaEventDisableTiming' \
  "${ROOT}/src/host/pearl/gpu_worker.cpp" 2>/dev/null && \
  fail "batch_done_event still uses cudaEventDisableTiming"
echo "  OK"

echo "[2/4] Grep: proof/mining paths unchanged..."
rg -q 'batch_seed_start' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "batch_seed_start missing"
rg -q 'upload_next_seed_async' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "seed conveyor missing"
rg -q 'iter_batch_graph_launch_ex' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "graph launch path missing"
echo "  OK"

echo "[3/4] Host-only reference tests..."
"${ROOT}/scripts/local_host_tests.sh"
echo "  OK"

echo "[4/4] CMake build (optional)..."
if command -v nvcc >/dev/null 2>&1; then
  BUILD="${ROOT}/build"
  cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROP_MINER_CUDA_ARCH=blackwell \
    -DCMAKE_CUDA_ARCHITECTURES=120a
  cmake --build "${BUILD}" --target propminer -j"${JOBS}"
  echo "  OK: propminer built"
  echo ""
  echo "Hardware gate (RTX 5090):"
  echo "  ${BUILD}/propminer --self-test --rtx5090 --gpus 0"
  echo "  ${BUILD}/propminer --bench 60 --rtx5090 --gpus 0"
else
  echo "  SKIP: nvcc not found"
fi

echo ""
echo "PASS: host sync + GPU hashrate validation complete."
