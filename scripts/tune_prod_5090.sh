#!/usr/bin/env bash
# One-shot RTX 5090 production tuning for Kryptex Stratum mining.
# Run once on the 5090 box, then mine with PROPMINER_USE_TUNE_CACHE=1.
#
# Order:
#   1. Compile-time Blackwell kernel knobs (GeForce v2/B200 paths)
#      -> tile shapes: 128x256, 256x128, 128x128, 256x256
#      -> knobs: kblock, stages, swizzle, min_blocks
#      -> bench batches: 4, 8, 16
#   2. Compile-time Consumer kernel knobs (fallback path)
#      -> same tile shapes + knobs as Blackwell
#   3. Unified runtime autotune (N × batch × graph_batch × cluster + carveout)
#      -> N values: 32768, 65536, 131072, 262144, 524288
#      -> batches: 1-48
#      -> clusters: 1, 2, 4
#      -> L2 carveout: -1, 50, 80
#
# Usage: ./scripts/tune_prod_5090.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "===== PropMiner RTX 5090 production tune (Kryptex Stratum) ====="

if [[ "${PROPMINER_SKIP_KNOB_TUNE:-0}" == "1" ]]; then
    echo "[tune-prod] Step 1/3: Blackwell kernel knobs SKIPPED (PROPMINER_SKIP_KNOB_TUNE=1)"
    echo "[tune-prod] Step 2/3: Consumer kernel knobs SKIPPED (PROPMINER_SKIP_KNOB_TUNE=1)"
else
    echo "[tune-prod] Step 1/3: Blackwell kernel knobs (compile sweep)..."
    "${ROOT}/scripts/tune_blackwell_knobs.sh" 15 3

    echo ""
    echo "[tune-prod] Step 2/3: Consumer kernel knobs (compile sweep)..."
    "${ROOT}/scripts/tune_consumer_knobs.sh" 15 3
fi

echo ""
echo "[tune-prod] Step 3/3: runtime autotune (N × batch × graph_batch × cluster + carveout)..."
"${ROOT}/scripts/tune_runtime_prod.sh" "${1:-120}"

echo ""
echo "[tune-prod] Done. Start mining with:"
echo "  set -a && source ~/.cache/propminer/tune_full_result.env && set +a"
echo "  PROPMINER_MODE=mine PROPMINER_WALLET=<wallet> ./scripts/docker_entrypoint.sh"
echo "Results: build/tune_full_raw.tsv  build/tune_full_summary.txt"

###############################################################################
# Step 4: Nsight Compute profile on the winning config (optional, requires ncu)
###############################################################################
echo ""
echo "[tune-prod] Step 4/4: Nsight Compute profiling on winning config..."

if [[ "${PROPMINER_SKIP_NCU:-0}" == "1" ]]; then
    echo "[tune-prod] NCU profiling SKIPPED (PROPMINER_SKIP_NCU=1)"
elif ! command -v ncu >/dev/null 2>&1; then
    echo "[tune-prod] NCU profiling SKIPPED (ncu not installed — install nsight-compute to enable)"
    echo "[tune-prod]   Debian/Ubuntu: sudo apt-get install nsight-compute"
else
    # Source the winning env so we profile the exact best config
    ENV_FILE="${HOME}/.cache/propminer/tune_full_result.env"
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        source "${ENV_FILE}"
        set +a

        OUT="${ROOT}/results/ncu_$(date +%Y%m%d_%H%M%S)_prod_tune"
        mkdir -p "${OUT}"

        echo "[tune-prod] Profiling with winning config:"
        echo "  N_CAP=${PROPMINER_N_CAP}"
        echo "  BATCH=${PROPMINER_BATCH}"
        echo "  GRAPH_BATCH=${PROPMINER_GRAPH_BATCH}"
        echo "  CLUSTER_M=${PEARL_GEMM_CONSUMER_CLUSTER_M}"

        # Run ncu profile on the built binary
        ncu --set full \
            --format json \
            --target-processes all \
            -o "${OUT}/gemm_profile" \
            --kernel-regex regex:transcript_gemm_kernel \
            ./build/propminer --bench 60 --rtx5090 --gpus 0 \
            2>&1 | tee "${OUT}/ncu_stdout.log" || {
            echo "[tune-prod] WARN: ncu profile failed (non-fatal, see ${OUT}/ncu_stdout.log)"
        }

        echo ""
        echo "[tune-prod] NCU report saved to: ${OUT}/"
        echo "  - gemm_profile.json   Nsight Compute JSON (all kernel metrics)"
        echo "  - gemm_profile.ncu-rep  Nsight Compute GUI report"
        echo "  - ncu_stdout.log      Console output"
        echo ""
        echo "[tune-prod] Key metrics to check:"
        echo "  - sm__pipe_tensor_cycles_active.sum_pct  >70% = tensor-bound (good)"
        echo "  - sm__warps_active.avg.pct_of_peak_sustained_active  >50% = good occupancy"
        echo "  - lts__t_sectors_pipe_lsu_mem_global_op_ld.sum  high = memory-bound"
        echo "  - l1tex__t_sectors_pipe_lsu_mem_shared_op_shared_local_ld.sum  high = shared-mem bound"
    else
        echo "[tune-prod] NCU profiling SKIPPED (no tuning result found at ${ENV_FILE})"
    fi
fi

echo ""
echo "[tune-prod] ===== Complete ====="
echo "[tune-prod] Next steps:"
echo "  1. Review NCU report: ncu --report-export ${OUT} 2>/dev/null || true"
echo "  2. Source env: set -a && source ~/.cache/propminer/tune_full_result.env && set +a"
echo "  3. Mine: PROPMINER_MODE=mine PROPMINER_WALLET=<wallet> ./scripts/docker_entrypoint.sh"

