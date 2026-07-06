#!/usr/bin/env bash
# RTX 5090 / Blackwell kernel-knob sweep (compile-time KBLOCK/STAGES/SWIZZLE/MIN_BLOCKS).
#
# Usage:
#   ./scripts/tune_blackwell_knobs.sh [seconds_per_repeat] [repeats]
#
# Leaves the best self-test-passing libpearl_gemm_capi.so in BUILD_DIR and writes
# ~/.cache/propminer/kernel_knobs.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build}"
SECONDS_PER_REPEAT="${1:-15}"
REPEATS="${2:-3}"
BENCH_BATCH="${PROPMINER_BENCH_BATCH:-4}"
RESULTS_DIR="${PROPMINER_RESULTS_DIR:-${BUILD_DIR}/knob_sweep_results}"

source "${ROOT}/scripts/setup_cuda_env.sh"
source "${ROOT}/scripts/tune_kernel_knobs_common.sh"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"
setup_cuda_runtime_env

echo "[knob-tune] RTX 5090 kernel knob sweep (${SECONDS_PER_REPEAT}s x ${REPEATS} repeats, batch=${BENCH_BATCH})"

KBLOCKS=(64 128)
STAGES_LIST=(2 3)
SWIZZLES=(2 3)
MIN_BLOCKS_LIST=(1 2)
LOAD_POLICY="cp_async"

BEST_LABEL=""
BEST_MANIFEST=""
BEST_RATE=0
BEST_SO=""

for KBLOCK in "${KBLOCKS[@]}"; do
for STAGES in "${STAGES_LIST[@]}"; do
    if ! tune_knob_smem_ok "${KBLOCK}" "${STAGES}"; then
        echo "[knob-tune] skip k${KBLOCK}-s${STAGES}: smem=$((384*KBLOCK*STAGES)) > budget"
        continue
    fi
for SWIZZLE in "${SWIZZLES[@]}"; do
for MIN_BLOCKS in "${MIN_BLOCKS_LIST[@]}"; do
    LABEL="bw-128x256-k${KBLOCK}-s${STAGES}-sw${SWIZZLE}-mb${MIN_BLOCKS}-${LOAD_POLICY}"
    MANIFEST="$(tune_knob_manifest "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}")"
    echo "[knob-tune] Building ${LABEL} (${MANIFEST})..."

    if ! cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DPEARL_GEMM_BLACKWELL_BM=128 \
        -DPEARL_GEMM_BLACKWELL_BN=256 \
        -DPEARL_GEMM_BLACKWELL_KBLOCK="${KBLOCK}" \
        -DPEARL_GEMM_BLACKWELL_STAGES="${STAGES}" \
        -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${SWIZZLE}" \
        -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS="${MIN_BLOCKS}" \
        -DPEARL_GEMM_BLACKWELL_LOAD_POLICY="${LOAD_POLICY}" \
        -DCMAKE_BUILD_TYPE=Release \
        > "${RESULTS_DIR}/cmake-${LABEL}.log" 2>&1; then
        echo "[knob-tune] ${LABEL}: cmake failed"
        continue
    fi

    if ! cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc)" \
        > "${RESULTS_DIR}/build-${LABEL}.log" 2>&1; then
        echo "[knob-tune] ${LABEL}: build failed"
        continue
    fi

    if ! tune_knob_self_test "${BUILD_DIR}/propminer" "${RESULTS_DIR}" "${LABEL}"; then
        echo "[knob-tune] ${LABEL}: self-test FAILED (skip bench)"
        continue
    fi

    MEAN_RATE="$(tune_knob_bench_variant "${BUILD_DIR}/propminer" \
        "${SECONDS_PER_REPEAT}" "${REPEATS}" "${BENCH_BATCH}" \
        "${RESULTS_DIR}" "${LABEL}")"
    RATE_INT="$(tune_knob_rate_to_int "${MEAN_RATE}")"
    echo "[knob-tune] ${LABEL}: mean=${MEAN_RATE} H/s (${REPEATS} repeats)"

    if (( RATE_INT <= BEST_RATE )); then
        continue
    fi

    if ! tune_knob_self_test "${BUILD_DIR}/propminer" "${RESULTS_DIR}" "${LABEL}"; then
        echo "[knob-tune] ${LABEL}: self-test FAILED (discarded)"
        continue
    fi

    BEST_RATE=${RATE_INT}
    BEST_LABEL="${LABEL}"
    BEST_MANIFEST="${MANIFEST}"
    BEST_SO="${BUILD_DIR}/libpearl_gemm_capi_${LABEL}.so"
    cp "${BUILD_DIR}/libpearl_gemm_capi.so" "${BEST_SO}"
    echo "[knob-tune] ${LABEL}: new best (${MEAN_RATE} H/s, self-test OK)"
done
done
done
done

if [[ -z "${BEST_SO}" ]]; then
    echo "[knob-tune] ERROR: no variant passed build+bench+self-test"
    exit 1
fi

cp "${BEST_SO}" "${BUILD_DIR}/libpearl_gemm_capi.so"
echo "[knob-tune] Staged best: ${BEST_LABEL} (${BEST_RATE} H/s) -> ${BUILD_DIR}/libpearl_gemm_capi.so"

KBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-k([0-9]+)-.*/\1/')"
SBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-s([0-9]+)-.*/\1/')"
SWBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-sw([0-9]+)-.*/\1/')"
MBBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-mb([0-9]+)-.*/\1/')"
KEY="$(tune_knob_gpu_cache_key)"
tune_knob_write_cache "${KEY}" "${KBEST}" "${SBEST}" "${SWBEST}" "${MBBEST}" \
    "${LOAD_POLICY}" "${BEST_MANIFEST}" "${BEST_RATE}" "1"
echo "[knob-tune] Cached winner to $(tune_knob_cache_path)"
echo "[knob-tune] Winner manifest: ${BEST_MANIFEST}"
