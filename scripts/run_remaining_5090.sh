#!/usr/bin/env bash
# One-shot: everything remaining for RTX 5090 validation (proof-safe order).
#
# Run on native Linux with RTX 5090 + CUDA 12.8+ (NOT Salad WSL2).
#
# Usage:
#   ./scripts/run_remaining_5090.sh              # full pipeline
#   PROPMINER_SKIP_TUNE=1 ./scripts/run_remaining_5090.sh   # skip long sweeps
#   PROPMINER_SKIP_GEFORCE=1 ./scripts/run_remaining_5090.sh
#
# Env:
#   PROPMINER_BENCH_SECONDS=180   bench duration
#   PROPMINER_BUILD_GEFORCE=1       dual-kernel build (required for geforce verify)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BENCH_SECONDS="${PROPMINER_BENCH_SECONDS:-300}"
BUILD_GEFORCE="${PROPMINER_BUILD_GEFORCE:-1}"
RESULTS_DIR="${ROOT}/results"
mkdir -p "${RESULTS_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${RESULTS_DIR}/run_remaining_${STAMP}.log"
exec > >(tee -a "${LOG}") 2>&1

echo "===== PropMiner run_remaining_5090 (${STAMP}) ====="
echo "Log: ${LOG}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FAIL: nvidia-smi not found — need native Linux RTX 5090 host" >&2
    exit 1
fi

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
echo "GPU compute capability: ${CC}"
if [[ "${CC}" != 12.* ]]; then
    echo "WARN: expected CC 12.x (RTX 5090); continuing anyway"
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "FAIL: nvcc not found" >&2
    exit 1
fi

# ── Phase 0: host tests (if repo has clang++) ───────────────────────────────
if [[ -x "${ROOT}/scripts/local_host_tests.sh" ]]; then
    echo ""
    echo "== Phase 0: host-only tests =="
    ./scripts/local_host_tests.sh
fi

# ── Phase 1: build (optional dual GeForce kernel) ───────────────────────────
echo ""
echo "== Phase 1: build =="
CMAKE_EXTRA=()
if [[ "${BUILD_GEFORCE}" == "1" ]]; then
    CMAKE_EXTRA+=(-DPEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=ON)
    echo "Building with PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=ON"
fi
mkdir -p build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROP_MINER_CUDA_ARCH=blackwell \
    -DCMAKE_CUDA_ARCHITECTURES=120a \
    "${CMAKE_EXTRA[@]}"
cmake --build build --target propminer -j"$(nproc)"

if command -v cuobjdump >/dev/null 2>&1; then
    echo "Cubin archs:"
    cuobjdump -lelf build/libpearl_gemm_capi.so 2>/dev/null | grep -oE 'sm_[0-9a-z_]+' | sort -u || true
fi

# ── Phase 2: pre-deploy gate ───────────────────────────────────────────────
echo ""
echo "== Phase 2: pre-deploy gate =="
./scripts/pre_deploy_gate.sh

# ── Phase 3: production tune (optional) ─────────────────────────────────────
if [[ "${PROPMINER_SKIP_TUNE:-0}" != "1" ]]; then
    echo ""
    echo "== Phase 3: tune_prod_5090 (knobs + cluster + batch) =="
    echo "NOTE: this can take 1-3 hours"
    ./scripts/tune_prod_5090.sh
    ./scripts/validate_knob_manifest.sh
else
    echo "[skip] Phase 3 tune (PROPMINER_SKIP_TUNE=1)"
fi

# ── Phase 4: GeForce transcript verify (optional) ─────────────────────────
if [[ "${PROPMINER_SKIP_GEFORCE:-0}" != "1" && "${BUILD_GEFORCE}" == "1" ]]; then
    echo ""
    echo "== Phase 4: GeForce transcript byte-identity + self-tests =="
    PROPMINER_GATE_GEFORCE=1 ./scripts/verify_geforce_transcript.sh
else
    echo "[skip] Phase 4 GeForce verify"
fi

# ── Phase 5: baseline benchmark ───────────────────────────────────────────
echo ""
echo "== Phase 5: benchmark (${BENCH_SECONDS}s) =="
PROPMINER_BENCH_JSON=1 ./build/propminer --bench "${BENCH_SECONDS}" --rtx5090 --gpus 0 \
    | tee "${RESULTS_DIR}/benchmark_${STAMP}.log"

JSON_LINE="$(grep '"tmad_per_sec"' "${RESULTS_DIR}/benchmark_${STAMP}.log" | tail -1 || true)"
if [[ -n "${JSON_LINE}" ]]; then
    echo "${JSON_LINE}" >> "${RESULTS_DIR}/bench_history.jsonl"
    echo "${JSON_LINE}" > "${RESULTS_DIR}/latest_bench.json"
fi

BASELINE="${RESULTS_DIR}/baseline_5090_sm120.json"
if [[ -f "${BASELINE}" ]]; then
    echo ""
    echo "== Compare vs baseline =="
    ./scripts/compare_bench.sh "${BASELINE}" "${RESULTS_DIR}/benchmark_${STAMP}.log" 0.05 || true
fi

# ── Phase 6: ncu profile (optional) ───────────────────────────────────────
if [[ "${PROPMINER_SKIP_NCU:-0}" != "1" ]] && command -v ncu >/dev/null 2>&1; then
    echo ""
    echo "== Phase 6: Nsight Compute profile =="
    ./scripts/profile_gemm_ncu.sh 0 || echo "WARN: ncu profile failed (non-fatal)"
else
    echo "[skip] Phase 6 ncu (set PROPMINER_SKIP_NCU=0 and install ncu to enable)"
fi

echo ""
echo "===== run_remaining_5090 COMPLETE ====="
echo "Results: ${RESULTS_DIR}/"
echo "Latest bench JSON: ${RESULTS_DIR}/latest_bench.json"
echo "Full log: ${LOG}"
