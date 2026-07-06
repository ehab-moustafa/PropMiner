#!/usr/bin/env bash
# Salad/WSL2 runtime container: tune + validate without nvcc rebuild.
set -euo pipefail

ROOT="/root/PropMiner"
cd "${ROOT}"
export PROPMINER_BUILD_DIR="${ROOT}/build_remote_test"
export PROPMINER_SKIP_KNOB_TUNE=1
export PROPMINER_SKIP_GEFORCE=1
export PROPMINER_BUILD_GEFORCE=0
export PROPMINER_BENCH_BATCH=1
export PEARL_GEMM_CONSUMER_CLUSTER_M=2

LOG="${ROOT}/results/salad_tune_remaining.log"
exec > >(tee -a "${LOG}") 2>&1

echo "===== salad_tune_and_remaining $(date) ====="
source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

PM="${PROPMINER_BUILD_DIR}/propminer"
if [[ ! -x "${PM}" ]]; then
    PM="${ROOT}/propminer"
    ln -sf "${PM}" "${PROPMINER_BUILD_DIR}/propminer"
fi

echo "[gate] GPU self-test default..."
"${PM}" --self-test --rtx5090 --gpus 0

echo "[gate] GPU self-test production shape..."
PROP_MINER_SELF_TEST_PROD=1 "${PM}" --self-test --rtx5090 --gpus 0

echo ""
echo "== tune_prod (cluster + batch; knob sweep skipped) =="
PROPMINER_SKIP_KNOB_TUNE=1 "${ROOT}/scripts/tune_prod_5090.sh"

echo ""
echo "== run_remaining tail: bench + compare (skip rebuild/tune/geforce) =="
STAMP="$(date +%Y%m%d_%H%M%S)"
BENCH_LOG="${ROOT}/results/benchmark_prod_${STAMP}.log"
PROPMINER_BENCH_JSON=1 PROPMINER_BENCH_BATCH=1 "${PM}" --bench 180 --rtx5090 --gpus 0 \
    | tee "${BENCH_LOG}"

JSON_LINE="$(grep '"tmad_per_sec"' "${BENCH_LOG}" | tail -1 || true)"
if [[ -n "${JSON_LINE}" ]]; then
    echo "${JSON_LINE}" >> "${ROOT}/results/bench_history.jsonl"
    echo "${JSON_LINE}" > "${ROOT}/results/latest_bench.json"
fi

if [[ -f "${ROOT}/results/baseline_5090_sm120.json" ]]; then
    "${ROOT}/scripts/compare_bench.sh" \
        "${ROOT}/results/baseline_5090_sm120.json" "${BENCH_LOG}" 0.05 || true
fi

echo ""
echo "Caches:"
ls -la "${HOME}/.cache/propminer/" 2>/dev/null || true
echo "===== salad_tune_and_remaining COMPLETE $(date) ====="
