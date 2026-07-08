#!/usr/bin/env bash
# Process-isolated runtime batch sweep — survives GPU/driver wedges.
#
# Why this exists:
#   `propminer --tune-autotune` runs every candidate combo inside ONE process.
#   On some Blackwell driver stacks (e.g. 580.x / CUDA 13 on sm_120a) a CUDA
#   stream can wedge (100% util, ~100W, no forward progress). A wedged context
#   cannot be recovered in-process, so a single bad combo hangs (or _Exit()s)
#   the whole sweep and no autotune.json is produced.
#
#   This script instead runs EACH combo as its own short-lived `propminer
#   --bench` process guarded by `timeout`. If a combo wedges, only that child
#   is killed; the sweep records it as failed and continues to the next combo.
#
# Result: a recommended PROPMINER_BATCH / PROPMINER_GRAPH_BATCH for mining,
#   plus a per-combo results table. Mining consumes these via env directly
#   (no per-UUID autotune.json needed), which is what the fleet wants anyway.
#
# Usage: ./scripts/tune_runtime_safe.sh [seconds_per_combo] [batch_list]
#   ./scripts/tune_runtime_safe.sh                # 15s per combo, default batches
#   ./scripts/tune_runtime_safe.sh 20 "1 2 4 8"   # custom
#
# Env:
#   PROPMINER_N_CAP           N ceiling (must match mining). Default 131072.
#   PROPMINER_TUNE_SAFE_GRAPH 1 = also try CUDA graphs (default 0 = graphs off,
#                             the safe path on wedge-prone drivers).
#   PROPMINER_BUILD_DIR       Build dir with propminer. Default build_remote_test.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build_remote_test}"
BIN="${BUILD_DIR}/propminer"
PER="${1:-15}"
BATCHES="${2:-1 2 4 6 8 10 12 16 20 24 32}"

# Abandon a wedged combo quickly instead of waiting the full 30s watchdog.
export PROPMINER_STALL_RESTART_MS="${PROPMINER_STALL_RESTART_MS:-12000}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"

USE_GRAPH="${PROPMINER_TUNE_SAFE_GRAPH:-0}"

# Hard wall-clock cap per combo: bench window + build/warmup + watchdog margin.
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

echo "===== PropMiner process-isolated safe batch sweep ====="
echo "[safe-tune] bin=${BIN}"
echo "[safe-tune] N_CAP=${PROPMINER_N_CAP} per=${PER}s timeout=${TIMEOUT_S}s graphs=${USE_GRAPH}"
echo "[safe-tune] batches: ${BATCHES}"
echo ""

best_batch=""
best_rate=0

run_one() {
    # $1 = batch. Prints the parsed tmad_per_sec (or empty on fail).
    local batch="$1"
    local out
    local graph_env=()
    if [[ "${USE_GRAPH}" != "1" ]]; then
        graph_env=(PROPMINER_BENCH_NO_GRAPH=1)
    fi
    # Each combo is its own process; `timeout` SIGKILLs a wedge (-k grace).
    out="$(timeout -k 5 "${TIMEOUT_S}" env \
        "${graph_env[@]}" \
        PROPMINER_BENCH_JSON=1 \
        PROPMINER_BENCH_BATCH="${batch}" \
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
    # Parse "tmad_per_sec": <number> from the JSON bench line.
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

for b in ${BATCHES}; do
    printf '[safe-tune] batch=%-3s ... ' "${b}"
    res="$(run_one "${b}")"
    case "${res}" in
        WEDGED*|NO_RESULT*)
            echo "${res} (skipped)"
            echo "batch=${b} result=${res}" >> "${RESULTS}"
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
sort -t= -k3 -g "${RESULTS}" 2>/dev/null || cat "${RESULTS}"
echo ""

if [[ -z "${best_batch}" ]]; then
    echo "[safe-tune] ERROR: every combo wedged or produced no result."
    echo "[safe-tune] This pod's GPU/driver may be unstable — try a different instance."
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
echo "  PEARL_GEMM_CONSUMER_CLUSTER_M=1"
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
    echo "PEARL_GEMM_CONSUMER_CLUSTER_M=1"
    echo "PROPMINER_AUTOTUNE=0"
    echo "PROPMINER_USE_TUNE_CACHE=0"
} > "${ENV_OUT}"
echo "[safe-tune] Wrote mining env: ${ENV_OUT}"
