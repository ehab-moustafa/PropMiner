#!/usr/bin/env bash
# GPU smoke tests for RunPod RTX 5090 (native Linux). Run after build, before long tune.
#
# Usage:
#   ./scripts/runpod_gpu_smoke.sh [seconds_per_bench]
#
# Env:
#   PROPMINER_BUILD_DIR   default: ./build
#   PROPMINER_SKIP_BUILD  1 = skip cmake build
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROPMINER_BUILD_DIR:-${ROOT}/build}"
BIN="${BUILD_DIR}/propminer"
PER="${1:-8}"
LOG="${BUILD_DIR}/runpod_smoke.log"

source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

echo "===== RunPod GPU smoke tests =====" | tee "${LOG}"
echo "[smoke] build_dir=${BUILD_DIR} bench=${PER}s" | tee -a "${LOG}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[smoke] FAIL: nvidia-smi not found (need GPU pod)" | tee -a "${LOG}" >&2
    exit 1
fi
nvidia-smi -L | tee -a "${LOG}"

echo "[smoke] host-only tests..." | tee -a "${LOG}"
"${ROOT}/scripts/local_host_tests.sh" | tee -a "${LOG}"

echo "[smoke] tune script dry-run..." | tee -a "${LOG}"
"${ROOT}/scripts/test_tune_full_dry.sh" | tee -a "${LOG}"

if [[ "${PROPMINER_SKIP_BUILD:-0}" != "1" && ! -x "${BIN}" ]]; then
    echo "[smoke] building propminer..." | tee -a "${LOG}"
    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120a \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc 2>/dev/null || echo 4)"
fi
if [[ ! -x "${BIN}" ]]; then
    echo "[smoke] FAIL: ${BIN} missing" | tee -a "${LOG}" >&2
    exit 1
fi

export PROPMINER_USE_STRATUM=1
export PROPMINER_SKIP_KNOB_TUNE=1
export PROPMINER_STALL_RESTART_MS=12000
export CUDA_MODULE_LOADING=EAGER
export CUDA_DEVICE_MAX_CONNECTIONS=1

failures=0
check_bench() {
    local label="$1"
    shift
    local out="${BUILD_DIR}/smoke_${label}.out"
    echo "[smoke] bench ${label} (${PER}s)..." | tee -a "${LOG}"
    set +e
    timeout "$(( PER + 90 ))" "$@" 2>&1 | tee "${out}"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "[smoke] FAIL ${label}: exit ${rc}" | tee -a "${LOG}" >&2
        ((failures++)) || true
        return
    fi
    local json
    json="$(grep -E '^\{"ts":' "${out}" | tail -1 || true)"
    if [[ -z "${json}" ]]; then
        echo "[smoke] FAIL ${label}: no bench JSON line" | tee -a "${LOG}" >&2
        ((failures++)) || true
        return
    fi
    python3 - "${json}" <<'PY'
import json, sys
j = json.loads(sys.argv[1])
for k in ("tmad_per_sec", "m", "n", "k"):
    if k not in j or float(j[k]) <= 0:
        raise SystemExit(f"missing/invalid {k}")
for k in ("gpu_power_w", "gpu_temp_c", "vram_used_mb"):
    if k not in j:
        raise SystemExit(f"missing telemetry {k}")
print(f"tmad={j['tmad_per_sec']:.2f} N={j['n']} power={j['gpu_power_w']}W temp={j['gpu_temp_c']}C")
PY
    if [[ $? -ne 0 ]]; then
        echo "[smoke] FAIL ${label}: JSON validation" | tee -a "${LOG}" >&2
        ((failures++)) || true
        return
    fi
    echo "[smoke] OK ${label}" | tee -a "${LOG}"
}

# All four tune N values — 65536 explicitly included.
for N in 32768 65536 131072 262144; do
    check_bench "n${N}" env PROPMINER_N_CAP="${N}" PROPMINER_BATCH=4 \
        PROPMINER_GRAPH_BATCH=4 PEARL_GEMM_CONSUMER_CLUSTER_M=1 \
        "${BIN}" --bench "${PER}" --rtx5090
done

# Graph-on path at production-ish batch (historical wedge case).
check_bench "graph_batch8" env PROPMINER_N_CAP=131072 PROPMINER_BATCH=8 \
    PROPMINER_GRAPH_BATCH=8 PEARL_GEMM_CONSUMER_CLUSTER_M=1 \
    "${BIN}" --bench "${PER}" --rtx5090

# Cluster sweep smoke (kernel fallback if driver rejects cluster launch).
check_bench "cluster2" env PROPMINER_N_CAP=65536 PROPMINER_BATCH=4 \
    PROPMINER_GRAPH_BATCH=4 PEARL_GEMM_CONSUMER_CLUSTER_M=2 \
    "${BIN}" --bench "${PER}" --rtx5090

if [[ ${failures} -gt 0 ]]; then
    echo "[smoke] FAIL: ${failures} bench case(s) failed — see ${LOG}" | tee -a "${LOG}" >&2
    exit 1
fi

echo "[smoke] All GPU smoke tests passed." | tee -a "${LOG}"
echo "[smoke] Ready for: ./scripts/tune_prod_5090.sh ${PER}" | tee -a "${LOG}"
