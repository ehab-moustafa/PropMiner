#!/usr/bin/env bash
# Shared helpers for Blackwell kernel-knob sweeps (KBLOCK/STAGES/SWIZZLE/MIN_BLOCKS).
# Sourced by tune_blackwell_knobs.sh and remote_test_kit.sh (bench + sweep).
[[ -n "${_TUNE_KNOB_COMMON_LOADED:-}" ]] && return 0
_TUNE_KNOB_COMMON_LOADED=1
set -euo pipefail

tune_knob_manifest() {
    local kblock="${1}" stages="${2}" swizzle="${3}" min_blocks="${4}" load_policy="${5}"
    echo "k${kblock}-s${stages}-sw${swizzle}-mb${min_blocks}-${load_policy}"
}

tune_knob_smem_bytes() {
    local kblock="${1}" stages="${2}"
    echo $((384 * kblock * stages))
}

# Shared memory per CTA must fit RTX 5090 ~164 KiB budget (leave margin).
tune_knob_smem_ok() {
    local kblock="${1}" stages="${2}"
    local smem
    smem="$(tune_knob_smem_bytes "${kblock}" "${stages}")"
    [[ "${smem}" -le 163840 ]]
}

tune_knob_extract_bench_rate() {
    local logfile="${1}"
    local rate incomplete json_line
    incomplete="$(grep 'benchmark incomplete:' "${logfile}" 2>/dev/null | tail -1 || true)"
    if [[ -n "${incomplete}" ]]; then
        echo "0"
        return
    fi
    json_line="$(grep '"tmad_per_sec"' "${logfile}" 2>/dev/null | tail -1 || true)"
    if [[ -n "${json_line}" ]]; then
        rate="$(echo "${json_line}" | sed -n 's/.*"tmad_per_sec":\([0-9.eE+-]*\).*/\1/p')"
        if [[ -n "${rate}" && "${rate}" != "0" ]]; then
            echo "${rate}"
            return
        fi
    fi
    rate="$(grep '^benchmark: ' "${logfile}" 2>/dev/null | tail -1 \
        | sed -n 's/.*TMAD\/s=\([0-9.eE+-]*\).*/\1/p' || true)"
    if [[ -n "${rate}" && "${rate}" != "0" ]]; then
        echo "${rate}"
        return
    fi
    rate="$(grep 'benchmark complete:' "${logfile}" 2>/dev/null | tail -1 \
        | sed -E 's/.*benchmark complete:[[:space:]]*([0-9.eE+-]+).*/\1/' || true)"
    if [[ -n "${rate}" && "${rate}" != "0" ]]; then
        echo "${rate}"
        return
    fi
    if [[ -f "${PROPMINER_LOG_DIR:-.}/propminer_stderr.log" ]]; then
        rate="$(grep 'TMAD/s=' "${PROPMINER_LOG_DIR}/propminer_stderr.log" 2>/dev/null \
            | tail -1 | sed -n 's/.*TMAD\/s=\([0-9.eE+-]*\).*/\1/p' || true)"
    fi
    echo "${rate:-0}"
}

# Round TMAD/s floats (e.g. 300.5) for bash integer comparisons — not protocol H/s.
tune_knob_rate_to_int() {
    awk -v r="${1:-0}" 'BEGIN {
        gsub(/[^0-9.eE+-].*/, "", r)
        if (r == "" || r == "+" || r == "-") r = "0"
        printf "%.0f", r + 0
    }'
}

tune_knob_trimmed_mean() {
    local -n _rates="${1}"
    local n="${#_rates[@]}"
    if [[ "${n}" -eq 0 ]]; then
        echo "0"
        return
    fi
    if [[ "${n}" -gt 2 ]]; then
        local sorted=()
        local r
        for r in "${_rates[@]}"; do sorted+=("${r}"); done
        IFS=$'\n' sorted=($(printf '%s\n' "${sorted[@]}" | sort -n))
        unset IFS
        local sum=0
        local i
        for ((i = 1; i < n; ++i)); do
            sum=$(awk -v s="${sum}" -v v="${sorted[$i]}" 'BEGIN { print s + v }')
        done
        awk -v s="${sum}" -v c="$((n - 1))" 'BEGIN { if (c > 0) print s / c; else print 0 }'
        return
    fi
    local sum=0
    local r
    for r in "${_rates[@]}"; do
        sum=$(awk -v s="${sum}" -v v="${r}" 'BEGIN { print s + v }')
    done
    awk -v s="${sum}" -v c="${n}" 'BEGIN { if (c > 0) print s / c; else print 0 }'
}

tune_knob_bench_variant() {
    local propminer="${1}" seconds="${2}" repeats="${3}" batch="${4}" logdir="${5}" label="${6}"
    local rates=()
    local i log rc
    mkdir -p "${logdir}"
    export PROPMINER_BENCH_BATCH="${batch}"
    unset PROPMINER_BENCH_NO_GRAPH || true
    for ((i = 1; i <= repeats; ++i)); do
        log="${logdir}/bench-${label}-r${i}.log"
        set +o pipefail
        run_propminer "${propminer}" --bench "${seconds}" --rtx5090 --gpus 0 \
            > "${log}" 2>&1 || rc=$?
        set -o pipefail
        rates+=("$(tune_knob_extract_bench_rate "${log}")")
        sleep 2
    done
    tune_knob_trimmed_mean rates
}

tune_knob_self_test() {
    local propminer="${1}" logdir="${2}" label="${3}"
    local log="${logdir}/selftest-${label}.log"
    mkdir -p "${logdir}"
    if run_propminer "${propminer}" --self-test --rtx5090 --gpus 0 > "${log}" 2>&1; then
        return 0
    fi
    return 1
}

tune_knob_cache_path() {
    if [[ -n "${PROPMINER_KERNEL_KNOB_CACHE:-}" ]]; then
        echo "${PROPMINER_KERNEL_KNOB_CACHE}"
        return
    fi
    local dir="${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/propminer"
    mkdir -p "${dir}"
    echo "${dir}/kernel_knobs.json"
}

tune_knob_write_cache() {
    local key="${1}" kblock="${2}" stages="${3}" swizzle="${4}" min_blocks="${5}"
    local load_policy="${6}" manifest="${7}" hashrate="${8}" self_test="${9}"
    local cache
    cache="$(tune_knob_cache_path)"
    local lines=()
    if [[ -f "${cache}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            [[ -z "${line}" || "${line}" == \#* ]] && { lines+=("${line}"); continue; }
            [[ "${line%% *}" == "${key}" ]] && continue
            lines+=("${line}")
        done < "${cache}"
    fi
    {
        for line in "${lines[@]}"; do
            [[ -n "${line}" ]] && echo "${line}"
        done
        echo "${key} ${kblock},${stages},${swizzle},${min_blocks},${load_policy},${manifest},${hashrate},${self_test}"
    } > "${cache}"
}

tune_knob_gpu_cache_key() {
    python3 - <<'PY' 2>/dev/null || echo "unknown-gpu-blackwell"
import os, subprocess, re
uuid = ""
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,compute_cap,driver_version",
         "--format=csv,noheader"],
        text=True, stderr=subprocess.DEVNULL)
    line = out.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    uuid = re.sub(r"[^a-fA-F0-9]", "", parts[0])[:32]
    cap = parts[1].replace(".", ".") if len(parts) > 1 else "0.0"
    drv = re.sub(r"[^0-9]", "", parts[2]) if len(parts) > 2 else "0"
    arch = os.environ.get("PEARL_GEMM_ARCH", "blackwell")
    print(f"{uuid}-{cap}-drv{drv}-{arch}")
except Exception:
    print("unknown-gpu-blackwell")
PY
}
