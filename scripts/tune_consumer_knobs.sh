#!/usr/bin/env bash
# Consumer path kernel-knob sweep (compile-time consumer tile shapes + knobs).
#
# The blackwell sweep (tune_blackwell_knobs.sh) covers GeForce v2/B200 paths.
# This script covers the consumer path (PEARL_GEMM_KERNEL=consumer) which is
# used as fallback when v2/v1 aren't compiled or explicitly requested.
#
# Sweeps:
#   Tile shapes: 128x256, 256x128, 128x128, 256x256
#   KBLOCK:      32, 64, 128
#   STAGES:      2, 3, 4
#   SWIZZLE:     2, 3
#   MIN_BLOCKS:  1, 2
#
# Usage:
#   ./scripts/tune_consumer_knobs.sh [seconds_per_repeat] [repeats]
#
# Leaves the best self-test-passing libpearl_gemm_capi.so in BUILD_DIR and
# writes ~/.cache/propminer/consumer_knobs.json
#
# Master TSV: logs every single run to build/consumer_knob_sweep_results/consumer_sweep_all_runs.tsv
# for post-hoc analysis, dashboards, and comparing all configs.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build}"
SECONDS_PER_REPEAT="${1:-15}"
REPEATS="${2:-3}"
BENCH_BATCH="${PROPMINER_BENCH_BATCH:-4}"
RESULTS_DIR="${PROPMINER_RESULTS_DIR:-${BUILD_DIR}/consumer_knob_sweep_results}"

source "${ROOT}/scripts/setup_cuda_env.sh"
source "${ROOT}/scripts/tune_kernel_knobs_common.sh"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"
setup_cuda_runtime_env

# Master TSV: every single run logged for post-hoc analysis / dashboards
MASTER_TSV="${RESULTS_DIR}/consumer_sweep_all_runs.tsv"
printf '%s\n' \
  'tile_shape	bM	bN	kblock	stages	swizzle	min_blocks	load_policy	self_test	status	tmad_per_sec	mean_rate	repeats' \
  > "${MASTER_TSV}"

echo "[consumer-knob-tune] Consumer path kernel knob sweep"
echo "[consumer-knob-tune] ${SECONDS_PER_REPEAT}s x ${REPEATS} repeats, batch=${BENCH_BATCH}"
echo "[consumer-knob-tune] Full results: ${MASTER_TSV}"

# Tile shapes to sweep (bM x bN)
TILE_SHAPES=(
    "128x256"
    "256x128"
    "128x128"
    "256x256"
)

KBLOCKS=(32 64 128)
STAGES_LIST=(2 3 4)
SWIZZLES=(2 3)
MIN_BLOCKS_LIST=(1 2)
LOAD_POLICY="cp_async"

BEST_LABEL=""
BEST_MANIFEST=""
BEST_RATE=0
BEST_SO=""
BEST_CONFIG=""

for TILE in "${TILE_SHAPES[@]}"; do
    bM="${TILE%x*}"
    bN="${TILE#*x}"

    for KBLOCK in "${KBLOCKS[@]}"; do
    for STAGES in "${STAGES_LIST[@]}"; do
        if ! tune_knob_smem_ok "${KBLOCK}" "${STAGES}"; then
            smem=$((384 * KBLOCK * STAGES))
            echo "[consumer-knob-tune] skip ${TILE}-k${KBLOCK}-s${STAGES}: smem=${smem} > ${TUNE_KNOB_SMEM_MAX_BYTES}"
            continue
        fi

        for SWIZZLE in "${SWIZZLES[@]}"; do
        for MIN_BLOCKS in "${MIN_BLOCKS_LIST[@]}"; do
            LABEL="c-${TILE}-k${KBLOCK}-s${STAGES}-sw${SWIZZLE}-mb${MIN_BLOCKS}-${LOAD_POLICY}"
            MANIFEST="$(tune_knob_manifest "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}")"
            echo "[consumer-knob-tune] Building ${LABEL} (${MANIFEST})..."

            SELF_TEST="fail"
            STATUS="skipped"
            MEAN_RATE="0"
            TMAD_PER_SEC=0

            if ! cmake -S "${ROOT}" -B "${BUILD_DIR}" \
                -DPROP_MINER_CUDA_ARCH=blackwell \
                -DCMAKE_CUDA_ARCHITECTURES=120a \
                -DPEARL_GEMM_BLACKWELL_BM="${bM}" \
                -DPEARL_GEMM_BLACKWELL_BN="${bN}" \
                -DPEARL_GEMM_CONSUMER_BM="${bM}" \
                -DPEARL_GEMM_CONSUMER_BN="${bN}" \
                -DPEARL_GEMM_CONSUMER_KBLOCK="${KBLOCK}" \
                -DPEARL_GEMM_CONSUMER_STAGES="${STAGES}" \
                -DPEARL_GEMM_CONSUMER_SWIZZLE="${SWIZZLE}" \
                -DPEARL_GEMM_CONSUMER_MIN_BLOCKS="${MIN_BLOCKS}" \
                -DPEARL_GEMM_BLACKWELL_KBLOCK="${KBLOCK}" \
                -DPEARL_GEMM_BLACKWELL_STAGES="${STAGES}" \
                -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${SWIZZLE}" \
                -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS="${MIN_BLOCKS}" \
                -DPEARL_GEMM_BLACKWELL_LOAD_POLICY="${LOAD_POLICY}" \
                -DCMAKE_BUILD_TYPE=Release \
                > "${RESULTS_DIR}/cmake-${LABEL}.log" 2>&1; then
                echo "[consumer-knob-tune] ${LABEL}: cmake failed"
                STATUS="cmake_failed"
                SELF_TEST="N/A"
                printf '%s\n' "${TILE}" "${bM}" "${bN}" "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}" "${SELF_TEST}" "${STATUS}" "${TMAD_PER_SEC}" "${MEAN_RATE}" "${REPEATS}" >> "${MASTER_TSV}"
                continue
            fi

            if ! cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc)" \
                > "${RESULTS_DIR}/build-${LABEL}.log" 2>&1; then
                echo "[consumer-knob-tune] ${LABEL}: build failed"
                STATUS="build_failed"
                SELF_TEST="N/A"
                printf '%s\n' "${TILE}" "${bM}" "${bN}" "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}" "${SELF_TEST}" "${STATUS}" "${TMAD_PER_SEC}" "${MEAN_RATE}" "${REPEATS}" >> "${MASTER_TSV}"
                continue
            fi

            if ! tune_knob_self_test "${BUILD_DIR}/propminer" "${RESULTS_DIR}" "${LABEL}"; then
                echo "[consumer-knob-tune] ${LABEL}: self-test FAILED (skip bench)"
                SELF_TEST="fail"
                STATUS="selftest_fail"
                TMAD_PER_SEC=0
                printf '%s\n' "${TILE}" "${bM}" "${bN}" "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}" "${SELF_TEST}" "${STATUS}" "${TMAD_PER_SEC}" "${MEAN_RATE}" "${REPEATS}" >> "${MASTER_TSV}"
                continue
            fi

            SELF_TEST="pass"
            MEAN_RATE="$(tune_knob_bench_variant "${BUILD_DIR}/propminer" \
                "${SECONDS_PER_REPEAT}" "${REPEATS}" "${BENCH_BATCH}" \
                "${RESULTS_DIR}" "${LABEL}")"
            TMAD_PER_SEC="$(tune_knob_rate_to_int "${MEAN_RATE}")"
            STATUS="ok"
            echo "[consumer-knob-tune] ${LABEL}: mean=${MEAN_RATE} H/s (${REPEATS} repeats)"
            printf '%s\n' "${TILE}" "${bM}" "${bN}" "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}" "${SELF_TEST}" "${STATUS}" "${TMAD_PER_SEC}" "${MEAN_RATE}" "${REPEATS}" >> "${MASTER_TSV}"

            if (( TMAD_PER_SEC <= BEST_RATE )); then
                continue
            fi

            if ! tune_knob_self_test "${BUILD_DIR}/propminer" "${RESULTS_DIR}" "${LABEL}"; then
                echo "[consumer-knob-tune] ${LABEL}: self-test FAILED (discarded)"
                SELF_TEST="fail"
                STATUS="selftest_fail_after_bench"
                printf '%s\n' "${TILE}" "${bM}" "${bN}" "${KBLOCK}" "${STAGES}" "${SWIZZLE}" "${MIN_BLOCKS}" "${LOAD_POLICY}" "${SELF_TEST}" "${STATUS}" "${TMAD_PER_SEC}" "${MEAN_RATE}" "${REPEATS}" >> "${MASTER_TSV}"
                continue
            fi

            BEST_RATE=${TMAD_PER_SEC}
            BEST_LABEL="${LABEL}"
            BEST_MANIFEST="${MANIFEST}"
            BEST_SO="${BUILD_DIR}/libpearl_gemm_capi_${LABEL}.so"
            BEST_CONFIG="bm=${bM} bn=${bN} kblock=${KBLOCK} stages=${STAGES} swizzle=${SWIZZLE} min_blocks=${MIN_BLOCKS}"
            cp "${BUILD_DIR}/libpearl_gemm_capi.so" "${BEST_SO}"
            echo "[consumer-knob-tune] ${LABEL}: new best (${MEAN_RATE} H/s, self-test OK)"
        done
        done
        done
    done
    done
done

if [[ -z "${BEST_SO}" ]]; then
    echo "[consumer-knob-tune] ERROR: no variant passed build+bench+self-test"
    exit 1
fi

cp "${BEST_SO}" "${BUILD_DIR}/libpearl_gemm_capi.so"
echo "[consumer-knob-tune] Staged best: ${BEST_LABEL} (${BEST_RATE} H/s) -> ${BUILD_DIR}/libpearl_gemm_capi.so"
echo "[consumer-knob-tune] Best config: ${BEST_CONFIG}"

KBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-k([0-9]+)-.*/\1/')"
SBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-s([0-9]+)-.*/\1/')"
SWBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-sw([0-9]+)-.*/\1/')"
MBBEST="$(echo "${BEST_LABEL}" | sed -E 's/.*-mb([0-9]+)-.*/\1/')"

# Extract tile shape from label
TILEBEST="$(echo "${BEST_LABEL}" | sed -E 's/c-(.*)-k.*/\1/')"
BTILEBEST="$(echo "${TILEBEST}" | sed -E 's/x/ /')"
BEST_BM="$(echo "${BTILEBEST}" | awk '{print $1}')"
BEST_BN="$(echo "${BTILEBEST}" | awk '{print $2}')"

KEY="$(tune_knob_gpu_cache_key)"
CACHE_FILE="$(tune_knob_cache_path)"
CACHE_DIR="$(dirname "${CACHE_FILE}")"
mkdir -p "${CACHE_DIR}"
CONSUMER_CACHE="${CACHE_DIR}/consumer_knobs.json"

# Write consumer-specific cache
{
    if [[ -f "${CONSUMER_CACHE}" ]]; then
        grep -v "^# consumer-knob-tune" "${CONSUMER_CACHE}" 2>/dev/null || true
    fi
    echo "# consumer-knob-tune ${BEST_LABEL} ${BEST_RATE}"
    echo "bm=${BEST_BM} bn=${BEST_BN} kblock=${KBEST} stages=${SBEST} swizzle=${SWBEST} min_blocks=${MBBEST} manifest=${BEST_MANIFEST} rate=${BEST_RATE}"
} > "${CONSUMER_CACHE}"

echo "[consumer-knob-tune] Consumer cache: ${CONSUMER_CACHE}"
echo "[consumer-knob-tune] Winner manifest: ${BEST_MANIFEST}"
echo "[consumer-knob-tune] Total runs logged: $(wc -l < "${MASTER_TSV}") (including header)"
