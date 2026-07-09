#!/usr/bin/env bash
# Route container start by PROPMINER_MODE:
#   full      — self-test + 180s benchmark (default for zero-config Salad validation)
#   test      — quick self-test only
#   mine      — connect to pool and mine until stopped (requires PROPMINER_WALLET)
#   tune      — offline kernel-knob sweep (KBLOCK/STAGES/SWIZZLE/MIN_BLOCKS)
#   tune-prod — full runtime autotune (N × batch × graph_batch × cluster)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Salad/WSL2: libcudart lives off default loader path — set before any propminer exec.
source "${ROOT}/scripts/setup_cuda_env.sh"
setup_cuda_runtime_env

# GPU clock lock (all modes; mine mode also applies this in run_mining.sh).
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -pm 1 2>/dev/null || true
    GPU_GRAPHICS_CLOCK="${PROPMINER_GPU_GRAPHICS_CLOCK:-2107}"
    GPU_MEMORY_CLOCK="${PROPMINER_GPU_MEMORY_CLOCK:-1312}"
    nvidia-smi -i 0 -lgc "${GPU_GRAPHICS_CLOCK},${GPU_MEMORY_CLOCK}" 2>/dev/null || true
    nvidia-smi -i 0 -pl "${PROPMINER_GPU_POWER_LIMIT_W:-575}" 2>/dev/null || true
fi
if [[ "${PROPMINER_USE_RELEASE:-0}" == "1" ]]; then
    source "${ROOT}/scripts/download_release.sh"
    ok=0
    for attempt in $(seq 1 18); do
        if ensure_binaries "${ROOT}/build_remote_test"; then
            ok=1
            break
        fi
        echo "[entrypoint] release download attempt ${attempt}/18 failed; retry in 10s..." >&2
        sleep 10
    done
    if [[ "${ok}" -ne 1 ]]; then
        echo "[entrypoint] ERROR: failed to download release binaries after retries" >&2
        exit 1
    fi
fi

MODE_RAW="${PROPMINER_MODE:-mine}"
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
    batch-tune|batch_tune|cluster-tune|cluster_tune)
        echo "[entrypoint] WARN: PROPMINER_MODE=${MODE_RAW} is deprecated." >&2
        echo "[entrypoint] Use PROPMINER_MODE=tune-prod (batch + graph_batch + cluster in one sweep)." >&2
        exec ./scripts/tune_prod_5090.sh "$@"
        ;;
    tune-prod|tune_prod|prod-tune)
        exec ./scripts/tune_prod_5090.sh "$@"
        ;;
    remaining|run-remaining|validate)
        exec ./scripts/run_remaining_5090.sh "$@"
        ;;
    verify-geforce|geforce-verify)
        exec ./scripts/verify_geforce_transcript.sh
        ;;
    *)
        echo "[entrypoint] ERROR: unknown PROPMINER_MODE='${MODE_RAW}'." >&2
        echo "[entrypoint] Valid values: full, test, mine, tune, tune-prod, remaining, verify-geforce." >&2
        exit 1
        ;;
esac
