#!/usr/bin/env bash
# Validate SeedGenerator dead-code removal; live GpuWorker pinned async seed path intact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "== PropMiner SeedGenerator cleanup validation =="

echo "[1/4] Grep: no SeedGenerator in source/build..."
if rg -n 'SeedGenerator|seed_generator' \
    "${ROOT}/src" "${ROOT}/CMakeLists.txt" 2>/dev/null; then
  fail "SeedGenerator references remain in src/ or CMakeLists.txt"
fi
for f in seed_generator.h seed_generator.cpp; do
  [[ ! -f "${ROOT}/src/host/pearl/${f}" ]] || fail "file still exists: ${f}"
done
echo "  OK: no dead-code references"

echo "[2/4] Grep: live GpuWorker seed path..."
rg -q 'upload_next_seed_async' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "upload_next_seed_async missing from gpu_worker.cpp"
rg -q 'pinned_seed_host_' "${ROOT}/src/host/pearl/gpu_worker.h" \
  || fail "pinned_seed_host_ missing from gpu_worker.h"
rg -q 'seed_copy_stream_' "${ROOT}/src/host/pearl/gpu_worker.h" \
  || fail "seed_copy_stream_ missing from gpu_worker.h"
rg -q 'seed_base_' "${ROOT}/src/host/pearl/gpu_worker.h" \
  || fail "seed_base_ missing from gpu_worker.h"
rg -q 'batch_seed_start' "${ROOT}/src/host/pearl/gpu_worker.cpp" \
  || fail "batch_seed_start missing from gpu_worker.cpp"
rg -q 'sigma_seed' "${ROOT}/src/host/pearl/sigma_context.cpp" \
  || fail "sigma_seed missing from sigma_context.cpp"
echo "  OK: pinned async seed path present"

echo "[3/4] Host-only reference tests..."
"${ROOT}/scripts/local_host_tests.sh"
echo "  OK: ref_pearl / host tests passed"

echo "[4/4] CMake build (optional)..."
if command -v nvcc >/dev/null 2>&1; then
  BUILD="${ROOT}/build"
  cmake -S "${ROOT}" -B "${BUILD}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROP_MINER_CUDA_ARCH=blackwell \
    -DCMAKE_CUDA_ARCHITECTURES=120a
  cmake --build "${BUILD}" --target propminer -j"${JOBS}"
  if nm -C "${BUILD}/propminer" 2>/dev/null | grep -qi SeedGenerator; then
    fail "SeedGenerator symbol in binary"
  fi
  echo "  OK: propminer built; no SeedGenerator symbols"
  echo ""
  echo "Hardware gate (RTX 5090):"
  echo "  ${BUILD}/propminer --self-test --rtx5090 --gpus 0"
  echo "  ${BUILD}/propminer --bench 10 --rtx5090 --gpus 0"
else
  echo "  SKIP: nvcc not found (run self-test on RTX 5090 box)"
fi

echo ""
echo "PASS: SeedGenerator cleanup validation complete."
