#!/usr/bin/env bash
# Process-isolated FULL runtime autotune — every batch × graph_batch × cluster combo
# gets its own short-lived `propminer --bench` (kill + retry on wedge).
#
# Replaces the batch-only tune_runtime_safe.sh for production tuning.
#
# Usage: ./scripts/tune_runtime_full.sh [seconds_per_combo]
#
# Env:
#   PROPMINER_N_CAP              N ceiling (default 131072)
#   PROPMINER_TUNE_MAX_RETRIES   Retries per combo on wedge (default 3)
#   PROPMINER_TUNE_GRAPHS        both | on | off (default both)
#   PROPMINER_TUNE_CLUSTERS      0 = cluster 1 only; 1 = sweep 1,2,4 (default 1)
#   PROPMINER_TUNE_CARVEOUT      1 = also sweep carveout -1,50,80 (default 0)
#   PROPMINER_BUILD_DIR          Build dir with propminer
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build_remote_test}"
BIN="${BUILD_DIR}/propminer"
PER="${1:-15}"
BATCHES="${PROPMINER_TUNE_BATCHES:-1 2 4 6 8 10 12 14 16 20 24 28 32 40 48}"

MAX_RETRIES="${PROPMINER_TUNE_MAX_RETRIES:-3}"
COOLDOWN_SEC="${PROPMINER_TUNE_RETRY_COOLDOWN_SEC:-3}"
GRAPH_MODE="${PROPMINER_TUNE_GRAPHS:-both}"
SWEEP_CLUSTERS="${PROPMINER_TUNE_CLUSTERS:-1}"
SWEEP_CARVEOUT="${PROPMINER_TUNE_CARVEOUT:-0}"
export PROPMINER_STALL_RESTART_MS="${PROPMINER_STALL_RESTART_MS:-12000}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
TIMEOUT_S=$(( PER + 45 ))

source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

if [[ ! -x "${BIN}" ]]; then
    echo "[full-tune] Building propminer (missing ${BIN})..."
    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc 2>/dev/null || echo 4)"
fi

RESULTS="${BUILD_DIR}/tune_full_results.txt"
: > "${RESULTS}"

graph_batch_candidates() {
    local batch="$1"
    local g
    for g in 1 2 4 8 16 32; do
        if (( g <= batch && batch % g == 0 )); then
            echo "${g}"
        fi
    done
}

CLUSTER_MS=(1)
if [[ "${SWEEP_CLUSTERS}" == "1" ]]; then
    CLUSTER_MS=(1 2 4)
fi

CARVEOUTS=(-1)
if [[ "${SWEEP_CARVEOUT}" == "1" ]]; then
    CARVEOUTS=(-1 50 80)
fi

GRAPH_OPTS=()
case "${GRAPH_MODE}" in
    on)  GRAPH_OPTS=(1) ;;
    off) GRAPH_OPTS=(0) ;;
    *)   GRAPH_OPTS=(1 0) ;;
esac

combo_count=0
for _b in ${BATCHES}; do
    for use_graph in "${GRAPH_OPTS[@]}"; do
        if [[ "${use_graph}" == "1" ]]; then
            mapfile -t gbs < <(graph_batch_candidates "${_b}")
        else
            gbs=(1)
        fi
        for _gb in "${gbs[@]}"; do
            for _c in "${CLUSTER_MS[@]}"; do
                for _cv in "${CARVEOUTS[@]}"; do
                    ((combo_count++)) || true
                done
            done
        done
    done
done

echo "===== PropMiner FULL process-isolated runtime autotune ====="
echo "[full-tune] bin=${BIN}"
echo "[full-tune] N_CAP=${PROPMINER_N_CAP} per=${PER}s timeout=${TIMEOUT_S}s combos=${combo_count}"
echo "[full-tune] graphs=${GRAPH_MODE} clusters=${CLUSTER_MS[*]} carveout_sweep=${SWEEP_CARVEOUT}"
echo "[full-tune] batches: ${BATCHES}"
echo ""

best_rate=0
best_batch=""
best_graph_batch=""
best_cluster=1
best_carveout=-1
best_use_graph=1
failed_count=0
wedged_count=0
done_count=0

gpu_cooldown() {
    pkill -f "${BIN}" 2>/dev/null || true
    sleep 1
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --gpu-reset -i 0 >/dev/null 2>&1 || true
    fi
    sleep "${COOLDOWN_SEC}"
}

run_one_attempt() {
    local batch="$1" graph_batch="$2" cluster="$3" carveout="$4" use_graph="$5"
    local out
    local extra_env=()
    if [[ "${use_graph}" != "1" ]]; then
        extra_env+=(PROPMINER_BENCH_NO_GRAPH=1)
    fi
    if [[ "${carveout}" -ge 0 ]]; then
        extra_env+=(PEARL_GEMM_CONSUMER_CARVEOUT="${carveout}")
    fi
    out="$(timeout -k 5 "${TIMEOUT_S}" env \
        "${extra_env[@]}" \
        PROPMINER_BENCH_JSON=1 \
        PROPMINER_BENCH_BATCH="${batch}" \
        PROPMINER_GRAPH_BATCH="${graph_batch}" \
        PEARL_GEMM_CONSUMER_CLUSTER_M="${cluster}" \
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
    local batch="$1" graph_batch="$2" cluster="$3" carveout="$4" use_graph="$5"
    local attempt=1 res=""
    while (( attempt <= MAX_RETRIES )); do
        res="$(run_one_attempt "${batch}" "${graph_batch}" "${cluster}" "${carveout}" "${use_graph}")"
        case "${res}" in
            WEDGED*|NO_RESULT*)
                if (( attempt < MAX_RETRIES )); then
                    gpu_cooldown
                    ((attempt++))
                    continue
                fi
                echo "${res}"
                return
                ;;
            *)
                echo "${res}"
                return
                ;;
        esac
    done
    echo "${res}"
}

for batch in ${BATCHES}; do
    for use_graph in "${GRAPH_OPTS[@]}"; do
        if [[ "${use_graph}" == "1" ]]; then
            mapfile -t graph_batches < <(graph_batch_candidates "${batch}")
        else
            graph_batches=(1)
        fi
        for graph_batch in "${graph_batches[@]}"; do
            for cluster in "${CLUSTER_MS[@]}"; do
                for carveout in "${CARVEOUTS[@]}"; do
                    ((done_count++)) || true
                    graph_label=$([[ "${use_graph}" == "1" ]] && echo on || echo off)
                    printf '[full-tune] [%d/%d] batch=%-2s graph=%-3s gb=%-2s cluster=%s carveout=%s ... ' \
                        "${done_count}" "${combo_count}" "${batch}" "${graph_label}" \
                        "${graph_batch}" "${cluster}" "${carveout}"
                    res="$(run_one_with_retries "${batch}" "${graph_batch}" "${cluster}" "${carveout}" "${use_graph}")"
                    case "${res}" in
                        WEDGED*)
                            echo "${res}"
                            echo "batch=${batch} graph=${graph_label} graph_batch=${graph_batch} cluster_m=${cluster} carveout=${carveout} result=${res}" >> "${RESULTS}"
                            ((wedged_count++)) || true
                            ;;
                        NO_RESULT*)
                            echo "${res}"
                            echo "batch=${batch} graph=${graph_label} graph_batch=${graph_batch} cluster_m=${cluster} carveout=${carveout} result=${res}" >> "${RESULTS}"
                            ((failed_count++)) || true
                            ;;
                        *)
                            echo "${res} TMAD/s"
                            echo "batch=${batch} graph=${graph_label} graph_batch=${graph_batch} cluster_m=${cluster} carveout=${carveout} tmad_per_sec=${res}" >> "${RESULTS}"
                            if awk "BEGIN{exit !(${res} > ${best_rate})}"; then
                                best_rate="${res}"
                                best_batch="${batch}"
                                best_graph_batch="${graph_batch}"
                                best_cluster="${cluster}"
                                best_carveout="${carveout}"
                                best_use_graph="${use_graph}"
                            fi
                            ;;
                    esac
                done
            done
        done
    done
done

echo ""
echo "===== Full sweep complete ====="
echo "[full-tune] wedged=${wedged_count} no_result=${failed_count}"
sort -t= -k6 -g "${RESULTS}" 2>/dev/null | tail -20
echo ""

if [[ -z "${best_batch}" ]]; then
    echo "[full-tune] ERROR: every combo failed."
    exit 1
fi

echo "[full-tune] WINNER: batch=${best_batch} graph_batch=${best_graph_batch} cluster_m=${best_cluster} graph=$([[ ${best_use_graph} == 1 ]] && echo on || echo off) -> ${best_rate} TMAD/s"
echo ""

ENV_OUT="${HOME}/.cache/propminer/tune_full_result.env"
mkdir -p "$(dirname "${ENV_OUT}")"
{
    echo "PROPMINER_N_CAP=${PROPMINER_N_CAP}"
    echo "PROPMINER_BATCH=${best_batch}"
    echo "PROPMINER_GRAPH_BATCH=${best_graph_batch}"
    [[ "${best_use_graph}" != "1" ]] && echo "PROPMINER_BENCH_NO_GRAPH=1"
    echo "PEARL_GEMM_CONSUMER_CLUSTER_M=${best_cluster}"
    [[ "${best_carveout}" -ge 0 ]] && echo "PEARL_GEMM_CONSUMER_CARVEOUT=${best_carveout}"
    echo "PROPMINER_STALL_RESTART_MS=30000"
    echo "PROPMINER_STALL_RESTART_DELAY_SEC=3"
    echo "PROPMINER_AUTOTUNE=0"
    echo "PROPMINER_USE_TUNE_CACHE=0"
} > "${ENV_OUT}"

# Symlink for scripts that read tune_safe_result.env
ln -sf "${ENV_OUT}" "${HOME}/.cache/propminer/tune_safe_result.env"

echo "[full-tune] Mine on the fleet with:"
cat "${ENV_OUT}"
echo ""
echo "[full-tune] Results: ${RESULTS}"
echo "[full-tune] Env: ${ENV_OUT}"
