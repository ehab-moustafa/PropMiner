#!/usr/bin/env bash
# Process-isolated runtime batch sweep — survives GPU/driver wedges with retries.
#
# Each combo runs as its own short-lived `propminer --bench` process under
# `timeout`. On wedge/stall/timeout the child is killed; we retry up to
# PROPMINER_TUNE_MAX_RETRIES (default 3) with a cooldown before marking FAILED.
#
# This replaces the fragile single-process `--tune-autotune` sweep on wedge-prone
# Blackwell stacks (driver 580.x / sm_120a) where a wedged CUDA context cannot
# be recovered in-process.
#
# Usage: ./scripts/tune_runtime_safe.sh [seconds_per_combo] [batch_list]
#
# Env:
#   PROPMINER_N_CAP                 N ceiling (must match mining). Default 131072.
#   PROPMINER_TUNE_MAX_RETRIES      Attempts per combo before FAILED (default 3).
#   PROPMINER_TUNE_RETRY_COOLDOWN_SEC Cooldown between retries (default 3).
#   PROPMINER_TUNE_SAFE_GRAPH       1 = try CUDA graphs (default 1 = on).
#   PROPMINER_BUILD_DIR             Build dir with propminer.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build_remote_test}"
BIN="${BUILD_DIR}/propminer"
PER="${1:-15}"
BATCHES="${2:-1 2 4 6 8 10 12 16 20 24 32}"

MAX_RETRIES="${PROPMINER_TUNE_MAX_RETRIES:-3}"
COOLDOWN_SEC="${PROPMINER_TUNE_RETRY_COOLDOWN_SEC:-3}"
export PROPMINER_STALL_RESTART_MS="${PROPMINER_STALL_RESTART_MS:-12000}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
USE_GRAPH="${PROPMINER_TUNE_SAFE_GRAPH:-1}"
export PEARL_GEMM_CONSUMER_CLUSTER_M="${PEARL_GEMM_CONSUMER_CLUSTER_M:-4}"
TIMEOUT_S=$(( PER + 45 ))

source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

if [[ ! -x "${BIN}" ]]; then
    echo "[safe-tune] Building propminer (missing ${BIN})..."
    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc 2>/dev/null || echo 4)"
fi

RESULTS="${BUILD_DIR}/tune_safe_results.txt"
: > "${RESULTS}"

echo "===== PropMiner resilient process-isolated batch sweep ====="
echo "[safe-tune] bin=${BIN}"
echo "[safe-tune] N_CAP=${PROPMINER_N_CAP} per=${PER}s timeout=${TIMEOUT_S}s"
echo "[safe-tune] retries=${MAX_RETRIES} cooldown=${COOLDOWN_SEC}s graphs=${USE_GRAPH}"
echo "[safe-tune] batches: ${BATCHES}"
echo ""

best_batch=""
best_rate=0
failed_count=0
wedged_count=0

gpu_cooldown() {
    pkill -f "${BIN}" 2>/dev/null || true
    sleep 1
    # Best-effort GPU reset between retries (may fail on some hosts — harmless).
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --gpu-reset -i 0 >/dev/null 2>&1 || true
    fi
    sleep "${COOLDOWN_SEC}"
}

run_one_attempt() {
    local batch="$1"
    local out
    local graph_env=()
    if [[ "${USE_GRAPH}" != "1" ]]; then
        graph_env=(PROPMINER_BENCH_NO_GRAPH=1)
    fi
    out="$(timeout -k 5 "${TIMEOUT_S}" env \
        "${graph_env[@]}" \
        PROPMINER_BENCH_JSON=1 \
        PROPMINER_BENCH_BATCH="${batch}" \
        PROPMINER_GRAPH_BATCH="${batch}" \
        PEARL_GEMM_CONSUMER_CLUSTER_M="${PEARL_GEMM_CONSUMER_CLUSTER_M}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        "${BIN}" --bench "${PER}" --rtx5090 --gpus 0 2>&1)"
    local rc=$?
    if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
        echo "WEDGED(timeout)"
        return
    fi
    if [[ ${rc} -eq 42 ]]; then
        echo "WEDGED(stall)"
        return
    fi
    local rate
    rate="$(printf '%s\n' "${out}" \
        | grep -o '"tmad_per_sec"[^,}]*' \
        | grep -oE '[0-9]+(\.[0-9]+)?' \
        | tail -1)"
    if [[ -z "${rate}" ]]; then
        echo "NO_RESULT(rc=${rc})"
        return
    fi
    echo "${rate}"
}

run_one_with_retries() {
    local batch="$1"
    local attempt=1
    local res=""
    while (( attempt <= MAX_RETRIES )); do
        res="$(run_one_attempt "${batch}")"
        case "${res}" in
            WEDGED*|NO_RESULT*)
                if (( attempt < MAX_RETRIES )); then
                    echo "[safe-tune]   batch=${batch} attempt ${attempt}/${MAX_RETRIES}: ${res} — retrying after cooldown" >&2
                    gpu_cooldown
                    ((attempt++))
                    continue
                fi
                echo "${res}"
                return
                ;;
            *)
                if (( attempt > 1 )); then
                    echo "[safe-tune]   batch=${batch} recovered on attempt ${attempt}/${MAX_RETRIES}" >&2
                fi
                echo "${res}"
                return
                ;;
        esac
    done
    echo "${res}"
}

for b in ${BATCHES}; do
    printf '[safe-tune] batch=%-3s ... ' "${b}"
    res="$(run_one_with_retries "${b}")"
    case "${res}" in
        WEDGED*)
            echo "${res} (failed after ${MAX_RETRIES} attempts)"
            echo "batch=${b} result=${res} attempts=${MAX_RETRIES}" >> "${RESULTS}"
            ((wedged_count++)) || true
            ;;
        NO_RESULT*)
            echo "${res} (failed after ${MAX_RETRIES} attempts)"
            echo "batch=${b} result=${res} attempts=${MAX_RETRIES}" >> "${RESULTS}"
            ((failed_count++)) || true
            ;;
        *)
            echo "${res} TMAD/s"
            echo "batch=${b} tmad_per_sec=${res}" >> "${RESULTS}"
            if awk "BEGIN{exit !(${res} > ${best_rate})}"; then
                best_rate="${res}"
                best_batch="${b}"
            fi
            ;;
    esac
done

echo ""
echo "===== Safe sweep complete ====="
echo "[safe-tune] wedged=${wedged_count} no_result=${failed_count}"
sort -t= -k3 -g "${RESULTS}" 2>/dev/null || cat "${RESULTS}"
echo ""

if [[ -z "${best_batch}" ]]; then
    echo "[safe-tune] ERROR: every combo failed after ${MAX_RETRIES} attempts each."
    echo "[safe-tune] Try a different RunPod instance/driver, or raise PROPMINER_TUNE_MAX_RETRIES."
    exit 1
fi

graph_line="PROPMINER_BENCH_NO_GRAPH=1"
graph_batch_line="PROPMINER_GRAPH_BATCH=1"
if [[ "${USE_GRAPH}" == "1" ]]; then
    graph_line="# graphs enabled during sweep"
    graph_batch_line="PROPMINER_GRAPH_BATCH=${best_batch}"
fi

echo "[safe-tune] WINNER: batch=${best_batch} -> ${best_rate} TMAD/s"
echo ""
echo "[safe-tune] Mine on the fleet with (N_CAP must match this sweep):"
echo "  PROPMINER_N_CAP=${PROPMINER_N_CAP}"
echo "  PROPMINER_BATCH=${best_batch}"
echo "  ${graph_batch_line}"
echo "  ${graph_line}"
echo "  PEARL_GEMM_CONSUMER_CLUSTER_M=${PEARL_GEMM_CONSUMER_CLUSTER_M}"
echo "  PROPMINER_STALL_RESTART_MS=30000"
echo "  PROPMINER_STALL_RESTART_DELAY_SEC=3"
echo "  PROPMINER_AUTOTUNE=0"
echo "  PROPMINER_USE_TUNE_CACHE=0"
echo ""
echo "[safe-tune] Results table: ${RESULTS}"

ENV_OUT="${HOME}/.cache/propminer/tune_safe_result.env"
mkdir -p "$(dirname "${ENV_OUT}")"
{
    echo "PROPMINER_N_CAP=${PROPMINER_N_CAP}"
    echo "PROPMINER_BATCH=${best_batch}"
    echo "${graph_batch_line}"
    [[ "${USE_GRAPH}" != "1" ]] && echo "PROPMINER_BENCH_NO_GRAPH=1"
    echo "PEARL_GEMM_CONSUMER_CLUSTER_M=${PEARL_GEMM_CONSUMER_CLUSTER_M}"
    echo "PROPMINER_STALL_RESTART_MS=30000"
    echo "PROPMINER_STALL_RESTART_DELAY_SEC=3"
    echo "PROPMINER_AUTOTUNE=0"
    echo "PROPMINER_USE_TUNE_CACHE=0"
} > "${ENV_OUT}"
echo "[safe-tune] Wrote mining env: ${ENV_OUT}"
