#!/usr/bin/env bash
# Process-isolated FULL runtime autotune:
#   N × batch × graph_batch × cluster × graph on/off
# Each combo = own `propminer --bench` process (kill + retry on wedge).
#
# Outputs:
#   tune_full_raw.tsv      — every run + GPU/CPU/RAM/VRAM telemetry
#   tune_full_results.txt  — human-readable lines (grep-friendly)
#   tune_full_summary.txt  — global winner + best per N
#   ~/.cache/propminer/tune_full_result.env — fleet mining env (global winner)
#
# Usage: ./scripts/tune_runtime_full.sh [seconds_per_combo]
#
# Env:
#   PROPMINER_TUNE_N_VALUES      N values to compare (default: 32768 65536 131072 262144)
#   PROPMINER_TUNE_SWEEP_N       1 = sweep N values (default); 0 = single PROPMINER_N_CAP only
#   PROPMINER_N_CAP              Used when PROPMINER_TUNE_SWEEP_N=0 (default 131072)
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
SWEEP_N="${PROPMINER_TUNE_SWEEP_N:-1}"
N_VALUES="${PROPMINER_TUNE_N_VALUES:-32768 65536 131072 262144}"
export PROPMINER_STALL_RESTART_MS="${PROPMINER_STALL_RESTART_MS:-12000}"
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

RAW_TSV="${BUILD_DIR}/tune_full_raw.tsv"
RESULTS="${BUILD_DIR}/tune_full_results.txt"
SUMMARY="${BUILD_DIR}/tune_full_summary.txt"
WINNERS_N="${BUILD_DIR}/tune_full_winners_by_n.tsv"
: > "${RAW_TSV}"
: > "${RESULTS}"
: > "${SUMMARY}"
: > "${WINNERS_N}"

printf '%s\n' \
  'n_cap	batch	graph	graph_batch	cluster_m	carveout	status	tmad_per_sec	tops_pct	m	n	k	gpu_util_pct	gpu_mem_util_pct	gpu_temp_c	gpu_power_w	gpu_power_limit_w	vram_used_mb	vram_total_mb	cpu_util_pct	ram_used_mb	ram_total_mb' \
  >> "${RAW_TSV}"
printf '%s\n' 'n_cap	best_batch	best_graph_batch	best_cluster	best_graph	tmad_per_sec' >> "${WINNERS_N}"

N_LIST=()
if [[ "${SWEEP_N}" == "1" ]]; then
    for n in ${N_VALUES}; do N_LIST+=("${n}"); done
else
    export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
    N_LIST=("${PROPMINER_N_CAP}")
fi

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

combos_per_n=0
for _b in ${BATCHES}; do
    for use_graph in "${GRAPH_OPTS[@]}"; do
        if [[ "${use_graph}" == "1" ]]; then
            mapfile -t _gbs < <(graph_batch_candidates "${_b}")
        else
            _gbs=(1)
        fi
        for _gb in "${_gbs[@]}"; do
            for _c in "${CLUSTER_MS[@]}"; do
                for _cv in "${CARVEOUTS[@]}"; do
                    ((combos_per_n++)) || true
                done
            done
        done
    done
done
total_combos=$(( combos_per_n * ${#N_LIST[@]} ))

echo "===== PropMiner FULL process-isolated runtime autotune ====="
echo "[full-tune] bin=${BIN}"
echo "[full-tune] per=${PER}s timeout=${TIMEOUT_S}s"
echo "[full-tune] N values: ${N_LIST[*]} (${#N_LIST[@]} shapes)"
echo "[full-tune] combos per N=${combos_per_n} total=${total_combos}"
echo "[full-tune] graphs=${GRAPH_MODE} clusters=${CLUSTER_MS[*]} carveout_sweep=${SWEEP_CARVEOUT}"
echo "[full-tune] batches: ${BATCHES}"
echo "[full-tune] raw log: ${RAW_TSV}"
echo ""

best_rate=0
best_n=""
best_batch=""
best_graph_batch=""
best_cluster=1
best_carveout=-1
best_use_graph=1
failed_count=0
wedged_count=0
done_count=0

declare -A best_rate_n best_batch_n best_gb_n best_cluster_n best_graph_n

gpu_cooldown() {
    pkill -f "${BIN}" 2>/dev/null || true
    sleep 1
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --gpu-reset -i 0 >/dev/null 2>&1 || true
    fi
    sleep "${COOLDOWN_SEC}"
}

# Fallback host telemetry when bench JSON is missing (wedge/timeout).
sample_host_telemetry() {
    local gpu_util=-1 gpu_mem=-1 gpu_temp=-1 gpu_pwr=-1 gpu_pl=-1
    local vram_u=-1 vram_t=-1
    if command -v nvidia-smi >/dev/null 2>&1; then
        IFS=',' read -r gpu_util gpu_mem gpu_temp gpu_pwr gpu_pl vram_u vram_t < <(
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,memory.used,memory.total \
                --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    fi
    local cpu=-1 ram_u=-1 ram_t=-1
    if [[ -r /proc/stat ]]; then
        read -r _ user nice system idle _ < /proc/stat
        cpu=0
    fi
    ram_t=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo -1)
    ram_u=$(awk '/MemAvailable/ {print int(($2)/1024)}' /proc/meminfo 2>/dev/null || echo -1)
    if [[ "${ram_u}" != -1 && "${ram_t}" != -1 ]]; then
        ram_u=$(( ram_t - ram_u ))
    fi
    printf '%s\n' "${gpu_util} ${gpu_mem} ${gpu_temp} ${gpu_pwr} ${gpu_pl} ${vram_u} ${vram_t} ${cpu} ${ram_u} ${ram_t}"
}

append_tsv_from_json() {
    local n_cap="$1" batch="$2" graph_label="$3" graph_batch="$4"
    local cluster="$5" carveout="$6" status="$7" json="$8"
    TUNE_JSON_LINE="${json}" TUNE_N_CAP="${n_cap}" TUNE_BATCH="${batch}" \
    TUNE_GRAPH="${graph_label}" TUNE_GB="${graph_batch}" TUNE_CLUSTER="${cluster}" \
    TUNE_CARVEOUT="${carveout}" TUNE_STATUS="${status}" \
    python3 <<'PY' >> "${RAW_TSV}"
import json, os
def g(j, k, d=-1):
    v = j.get(k, d)
    return "" if v is None else v
raw = os.environ.get("TUNE_JSON_LINE", "")
j = {}
if raw.strip().startswith("{"):
    try:
        j = json.loads(raw)
    except json.JSONDecodeError:
        pass
fields = [
    os.environ["TUNE_N_CAP"], os.environ["TUNE_BATCH"], os.environ["TUNE_GRAPH"],
    os.environ["TUNE_GB"], os.environ["TUNE_CLUSTER"], os.environ["TUNE_CARVEOUT"],
    os.environ["TUNE_STATUS"],
    g(j, "tmad_per_sec", ""), g(j, "tops_pct", ""), g(j, "m", ""), g(j, "n", ""), g(j, "k", ""),
    g(j, "gpu_util_pct"), g(j, "gpu_mem_util_pct"), g(j, "gpu_temp_c"),
    g(j, "gpu_power_w"), g(j, "gpu_power_limit_w"),
    g(j, "vram_used_mb"), g(j, "vram_total_mb"),
    g(j, "cpu_util_pct"), g(j, "ram_used_mb"), g(j, "ram_total_mb"),
]
print("\t".join(str(x) for x in fields))
PY
}

append_tsv_wedge() {
    local n_cap="$1" batch="$2" graph_label="$3" graph_batch="$4"
    local cluster="$5" carveout="$6" status="$7"
    read -r gpu_util gpu_mem gpu_temp gpu_pwr gpu_pl vram_u vram_t cpu ram_u ram_t \
        < <(sample_host_telemetry)
    printf '%s\n' \
      "${n_cap}	${batch}	${graph_label}	${graph_batch}	${cluster}	${carveout}	${status}						${gpu_util}	${gpu_mem}	${gpu_temp}	${gpu_pwr}	${gpu_pl}	${vram_u}	${vram_t}	${cpu}	${ram_u}	${ram_t}" \
      >> "${RAW_TSV}"
}

LAST_BENCH_JSON=""
LAST_BENCH_OUT=""

run_one_attempt() {
    local n_cap="$1" batch="$2" graph_batch="$3" cluster="$4" carveout="$5" use_graph="$6"
    local out extra_env=()
    if [[ "${use_graph}" != "1" ]]; then
        extra_env+=(PROPMINER_BENCH_NO_GRAPH=1)
    fi
    if [[ "${carveout}" -ge 0 ]]; then
        extra_env+=(PEARL_GEMM_CONSUMER_CARVEOUT="${carveout}")
    fi
    out="$(timeout -k 5 "${TIMEOUT_S}" env \
        "${extra_env[@]}" \
        PROPMINER_BENCH_JSON=1 \
        PROPMINER_N_CAP="${n_cap}" \
        PROPMINER_BENCH_BATCH="${batch}" \
        PROPMINER_GRAPH_BATCH="${graph_batch}" \
        PEARL_GEMM_CONSUMER_CLUSTER_M="${cluster}" \
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
        "${BIN}" --bench "${PER}" --rtx5090 --gpus 0 2>&1)"
    local rc=$?
    LAST_BENCH_OUT="${out}"
    LAST_BENCH_JSON="$(printf '%s\n' "${out}" | grep '"tmad_per_sec"' | tail -1 || true)"
    if [[ ${rc} -eq 124 || ${rc} -eq 137 ]]; then
        echo "WEDGED(timeout)"
        return
    fi
    if [[ ${rc} -eq 42 ]]; then
        echo "WEDGED(stall)"
        return
    fi
    local rate
    rate="$(printf '%s\n' "${LAST_BENCH_JSON}" \
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
    local n_cap="$1" batch="$2" graph_batch="$3" cluster="$4" carveout="$5" use_graph="$6"
    local attempt=1 res=""
    while (( attempt <= MAX_RETRIES )); do
        res="$(run_one_attempt "${n_cap}" "${batch}" "${graph_batch}" "${cluster}" "${carveout}" "${use_graph}")"
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

update_best() {
    local n_cap="$1" rate="$2" batch="$3" graph_batch="$4" cluster="$5"
    local carveout="$6" use_graph="$7" graph_label="$8"
    if awk "BEGIN{exit !(${rate} > ${best_rate_n[$n_cap]:-0})}"; then
        best_rate_n[$n_cap]="${rate}"
        best_batch_n[$n_cap]="${batch}"
        best_gb_n[$n_cap]="${graph_batch}"
        best_cluster_n[$n_cap]="${cluster}"
        best_graph_n[$n_cap]="${graph_label}"
    fi
    if awk "BEGIN{exit !(${rate} > ${best_rate})}"; then
        best_rate="${rate}"
        best_n="${n_cap}"
        best_batch="${batch}"
        best_graph_batch="${graph_batch}"
        best_cluster="${cluster}"
        best_carveout="${carveout}"
        best_use_graph="${use_graph}"
    fi
}

for n_cap in "${N_LIST[@]}"; do
    echo "[full-tune] === N_CAP=${n_cap} ==="
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
                        printf '[full-tune] [%d/%d] N=%-6s batch=%-2s graph=%-3s gb=%-2s cluster=%s ... ' \
                            "${done_count}" "${total_combos}" "${n_cap}" "${batch}" \
                            "${graph_label}" "${graph_batch}" "${cluster}"
                        res="$(run_one_with_retries "${n_cap}" "${batch}" "${graph_batch}" "${cluster}" "${carveout}" "${use_graph}")"
                        case "${res}" in
                            WEDGED*)
                                echo "${res}"
                                append_tsv_wedge "${n_cap}" "${batch}" "${graph_label}" \
                                    "${graph_batch}" "${cluster}" "${carveout}" "${res}"
                                echo "n_cap=${n_cap} batch=${batch} graph=${graph_label} graph_batch=${graph_batch} cluster_m=${cluster} carveout=${carveout} result=${res}" >> "${RESULTS}"
                                ((wedged_count++)) || true
                                ;;
                            NO_RESULT*)
                                echo "${res}"
                                append_tsv_wedge "${n_cap}" "${batch}" "${graph_label}" \
                                    "${graph_batch}" "${cluster}" "${carveout}" "${res}"
                                echo "n_cap=${n_cap} batch=${batch} graph=${graph_label} graph_batch=${graph_batch} cluster_m=${cluster} carveout=${carveout} result=${res}" >> "${RESULTS}"
                                ((failed_count++)) || true
                                ;;
                            *)
                                echo "${res} TMAD/s"
                                append_tsv_from_json "${n_cap}" "${batch}" "${graph_label}" \
                                    "${graph_batch}" "${cluster}" "${carveout}" "ok" \
                                    "${LAST_BENCH_JSON}"
                                echo "n_cap=${n_cap} batch=${batch} graph=${graph_label} graph_batch=${graph_batch} cluster_m=${cluster} carveout=${carveout} tmad_per_sec=${res}" >> "${RESULTS}"
                                update_best "${n_cap}" "${res}" "${batch}" "${graph_batch}" \
                                    "${cluster}" "${carveout}" "${use_graph}" "${graph_label}"
                                ;;
                        esac
                    done
                done
            done
        done
    done
done

{
    echo "# PropMiner full tune summary — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# total_combos=${total_combos} wedged=${wedged_count} failed=${failed_count}"
    echo "# raw_tsv=${RAW_TSV}"
    echo ""
    if [[ -n "${best_n}" ]]; then
        echo "GLOBAL_WINNER n_cap=${best_n} batch=${best_batch} graph_batch=${best_graph_batch} cluster_m=${best_cluster} graph=$([[ ${best_use_graph} == 1 ]] && echo on || echo off) tmad_per_sec=${best_rate}"
    fi
    echo ""
    echo "BEST_PER_N:"
    for n_cap in "${N_LIST[@]}"; do
        if [[ -n "${best_rate_n[$n_cap]:-}" ]]; then
            echo "  n_cap=${n_cap} batch=${best_batch_n[$n_cap]} graph_batch=${best_gb_n[$n_cap]} cluster_m=${best_cluster_n[$n_cap]} graph=${best_graph_n[$n_cap]} tmad_per_sec=${best_rate_n[$n_cap]}"
            printf '%s\n' "${n_cap}	${best_batch_n[$n_cap]}	${best_gb_n[$n_cap]}	${best_cluster_n[$n_cap]}	${best_graph_n[$n_cap]}	${best_rate_n[$n_cap]}" >> "${WINNERS_N}"
        fi
    done
} | tee "${SUMMARY}"

echo ""
echo "===== Full sweep complete ====="
echo "[full-tune] wedged=${wedged_count} no_result=${failed_count}"
echo "[full-tune] Top 20 by TMAD/s:"
grep 'tmad_per_sec=' "${RESULTS}" | sort -t= -k7 -g | tail -20 || true
echo ""

if [[ -z "${best_n}" ]]; then
    echo "[full-tune] ERROR: every combo failed."
    exit 1
fi

echo "[full-tune] GLOBAL WINNER: N=${best_n} batch=${best_batch} graph_batch=${best_graph_batch} cluster_m=${best_cluster} -> ${best_rate} TMAD/s"

ENV_OUT="${HOME}/.cache/propminer/tune_full_result.env"
mkdir -p "$(dirname "${ENV_OUT}")"
{
    echo "PROPMINER_N_CAP=${best_n}"
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

echo ""
echo "[full-tune] Files:"
echo "  raw (all runs + telemetry): ${RAW_TSV}"
echo "  summary:                  ${SUMMARY}"
echo "  winners per N:            ${WINNERS_N}"
echo "  fleet env (global best):  ${ENV_OUT}"
cat "${ENV_OUT}"
