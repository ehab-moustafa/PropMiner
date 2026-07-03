#!/usr/bin/env bash
# Linux-only helper to sweep Blackwell-specific pearl-gemm Makefile knobs.
# Builds a matrix of (BM x BN x KBLOCK x STAGES x SWIZZLE_BITS x MIN_BLOCKS x LOAD_POLICY)
# variants and runs the built-in PropMiner speed-test against each.
#
# Usage:
#   ./scripts/tune_blackwell_knobs.sh [seconds_per_variant]
#
# The script expects to run in a CUDA-capable environment and will leave the
# fastest variant's .so staged as build/libpearl_gemm_capi.so.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build"
SECONDS_PER_VARIANT="${1:-10}"

mkdir -p "${BUILD_DIR}"

# Knob search space.  Keep it dense enough to matter but small enough to finish.
BMS=(128 256)
BNS=(128 256)
KBLOCKS=(64 128)
STAGES_LIST=(2 3 4)
SWIZZLES=(2 3 4)
MIN_BLOCKS_LIST=(1 2)
LOAD_POLICIES=(cp_async tma)

BEST_FILE=""
BEST_RATE=0

for BM in "${BMS[@]}"; do
for BN in "${BNS[@]}"; do
for KBLOCK in "${KBLOCKS[@]}"; do
for STAGES in "${STAGES_LIST[@]}"; do
for SWIZZLE in "${SWIZZLES[@]}"; do
for MIN_BLOCKS in "${MIN_BLOCKS_LIST[@]}"; do
for LOAD_POLICY in "${LOAD_POLICIES[@]}"; do
    LABEL="bw-${BM}x${BN}-k${KBLOCK}-s${STAGES}-sw${SWIZZLE}-mb${MIN_BLOCKS}-${LOAD_POLICY}"
    echo "[tune] Building ${LABEL}..."

    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPEARL_GEMM_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DPEARL_GEMM_BLACKWELL_BM="${BM}" \
        -DPEARL_GEMM_BLACKWELL_BN="${BN}" \
        -DPEARL_GEMM_BLACKWELL_KBLOCK="${KBLOCK}" \
        -DPEARL_GEMM_BLACKWELL_STAGES="${STAGES}" \
        -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${SWIZZLE}" \
        -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS="${MIN_BLOCKS}" \
        -DPEARL_GEMM_BLACKWELL_LOAD_POLICY="${LOAD_POLICY}" \
        -DCMAKE_BUILD_TYPE=Release \
        > "${BUILD_DIR}/cmake-${LABEL}.log" 2>&1

    cmake --build "${BUILD_DIR}" --target propminer -j$(nproc) \
        > "${BUILD_DIR}/build-${LABEL}.log" 2>&1

    RATE=$("${BUILD_DIR}/propminer" --speed-test-seconds "${SECONDS_PER_VARIANT}" \
        --batch-size 16 2>/dev/null | tail -1 | grep -oE '[0-9.]+' | head -1 || true)

    if [[ -z "${RATE}" || "${RATE}" == "0" ]]; then
        echo "[tune] ${LABEL}: failed or zero rate"
        continue
    fi

    echo "[tune] ${LABEL}: ${RATE} H/s"
    RATE_INT=$(printf '%.0f' "${RATE}")
    if (( RATE_INT > BEST_RATE )); then
        BEST_RATE=${RATE_INT}
        BEST_FILE="${BUILD_DIR}/libpearl_gemm_capi.so"
        cp "${BUILD_DIR}/libpearl_gemm_capi.so" "${BUILD_DIR}/libpearl_gemm_capi_${LABEL}.so"
        echo "[tune] ${LABEL}: new best (${RATE} H/s)"
    fi
done
done
done
done
done
done
done

if [[ -n "${BEST_FILE}" ]]; then
    cp "${BEST_FILE}" "${BUILD_DIR}/libpearl_gemm_capi.so"
    echo "[tune] Staged best variant: ${BEST_FILE} (${BEST_RATE} H/s)"
else
    echo "[tune] No variant succeeded"
    exit 1
fi
