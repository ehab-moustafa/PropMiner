#!/usr/bin/env bash
# Build PropMiner for RTX 5090 and run a benchmark.
#
# Usage:
#   ./scripts/build_and_benchmark.sh [BENCH_SECONDS]
#
# Requirements:
#   - Linux with CUDA 12.8+ and nvcc
#   - RTX 5090 (sm_120a)
#   - Rust toolchain (cargo) for pearl-mining-capi
#   - CMake >= 3.24, make, git

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
BENCH_SECONDS="${1:-300}"

mkdir -p "${BUILD_DIR}"

# Ensure CUTLASS submodule is present.
if [[ ! -f "${ROOT}/third_party/pearl-gemm/third_party/cutlass/include/cutlass/cutlass.h" ]]; then
    echo "[build] Fetching CUTLASS v4.4.0..."
    git clone --depth 1 --branch v4.4.0 https://github.com/NVIDIA/cutlass.git \
        "${ROOT}/third_party/pearl-gemm/third_party/cutlass" || true
fi

# Configure for native Blackwell sm_120a with RTX 5090 knobs.
# Hard constraints:
#   - Tile 128x256x128 maps to the 170-SM occupancy grid.
#   - STAGES=2 with KBLOCK=128 fits shared memory while keeping the Tensor
#     Core datapath fed.
#   - SWIZZLE_BITS=3 is the Blackwell-native swizzle from Alpha tuning.
echo "[build] Configuring CMake for PEARL_GEMM_ARCH=blackwell (sm_120a)..."
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROP_MINER_CUDA_ARCH=blackwell \
    -DCMAKE_CUDA_ARCHITECTURES=120a \
    -DPEARL_GEMM_BLACKWELL_BM=128 \
    -DPEARL_GEMM_BLACKWELL_BN=256 \
    -DPEARL_GEMM_BLACKWELL_KBLOCK=128 \
    -DPEARL_GEMM_BLACKWELL_STAGES=2 \
    -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS=3 \
    -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS=1 \
    -DPEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async

# Build the miner binary (this also builds libpearl_gemm_capi.so and libpearl_mining_capi.so).
echo "[build] Building propminer..."
cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc)"

# Verify architecture.
echo "[build] Built binaries:"
ls -lh "${BUILD_DIR}/propminer" "${BUILD_DIR}/libpearl_gemm_capi.so" "${BUILD_DIR}/libpearl_mining_capi.so" || true

# Run the built-in benchmark on GPU 0 with the RTX 5090 hard-coded profile.
echo "[bench] Running ${BENCH_SECONDS}s benchmark with --rtx5090 profile..."
RESULTS_DIR="${ROOT}/results"
mkdir -p "${RESULTS_DIR}"
LOG="${RESULTS_DIR}/benchmark_$(date +%Y%m%d_%H%M%S).log"
cd "${BUILD_DIR}"
PROPMINER_BENCH_JSON=1 ./propminer --bench "${BENCH_SECONDS}" --rtx5090 --gpus 0 \
    2>&1 | tee "${LOG}"

JSON_LINE="$(grep '"tmad_per_sec"' "${LOG}" | tail -1 || true)"
if [[ -n "${JSON_LINE}" ]]; then
    echo "${JSON_LINE}" >> "${RESULTS_DIR}/bench_history.jsonl"
    echo "[bench] Appended to ${RESULTS_DIR}/bench_history.jsonl"
fi

echo "[bench] Log: ${LOG}"
echo "[bench] Done."
