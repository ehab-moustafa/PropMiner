#!/usr/bin/env bash
# Route container start by PROPMINER_MODE:
#   full      — self-test + 180s benchmark (default for zero-config Salad validation)
#   test      — quick self-test only
#   mine      — connect to pool and mine until stopped (requires PROPMINER_WALLET)
#   tune      — offline kernel-knob sweep (KBLOCK/STAGES/SWIZZLE/MIN_BLOCKS)
#   batch-tune — offline mine batch sweep at production N
#   cluster-tune — cluster_m {1,2,4} sweep + full runtime autotune
#   tune-prod    — full 5090 prod tune (knobs + cluster + batch)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Salad/WSL2: libcudart lives off default loader path — set before any propminer exec.
source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

MODE_RAW="${PROPMINER_MODE:-full}"
MODE="$(printf '%s' "${MODE_RAW}" | tr '[:upper:]' '[:lower:]')"

echo "[entrypoint] PROPMINER_MODE=${MODE_RAW} -> ${MODE}" >&2

case "${MODE}" in
    mine|mining)
        exec ./scripts/run_mining.sh "$@"
        ;;
    full|benchmark)
        export PROPMINER_QUICK_EXIT=0
        export PROPMINER_SKIP_BENCH=0
        export PROPMINER_SKIP_SWEEP=1
        export PROPMINER_SKIP_NCU=1
        exec ./scripts/remote_test_kit.sh
        ;;
    test)
        export PROPMINER_QUICK_EXIT=1
        export PROPMINER_SKIP_BENCH=1
        export PROPMINER_SKIP_SWEEP=1
        export PROPMINER_SKIP_NCU=1
        exec ./scripts/remote_test_kit.sh
        ;;
    tune|knob-tune|knobs)
        exec ./scripts/tune_blackwell_knobs.sh "$@"
        ;;
    batch-tune|batch_tune)
        exec ./scripts/tune_mine_batch.sh "$@"
        ;;
    cluster-tune|cluster_tune)
        exec ./scripts/tune_cluster_sweep.sh "$@"
        ;;
    tune-prod|tune_prod|prod-tune)
        exec ./scripts/tune_prod_5090.sh "$@"
        ;;
    salad-tune|salad_tune)
        exec ./scripts/salad_tune_and_remaining.sh "$@"
        ;;
    remaining|run-remaining|validate)
        exec ./scripts/run_remaining_5090.sh "$@"
        ;;
    verify-geforce|geforce-verify)
        exec ./scripts/verify_geforce_transcript.sh
        ;;
    *)
        echo "[entrypoint] ERROR: unknown PROPMINER_MODE='${MODE_RAW}'." >&2
        echo "[entrypoint] Valid values: full, test, mine, tune, batch-tune, cluster-tune, tune-prod, remaining, verify-geforce, salad-tune." >&2
        exit 1
        ;;
esac
