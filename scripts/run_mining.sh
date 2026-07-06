#!/usr/bin/env bash
# Production mining entrypoint for Salad / Docker (stays alive until stopped).
#
# Required env:
#   PROPMINER_WALLET  — Pearl/Kryptex wallet (e.g. krxXXXX.worker or prl1p...worker)
#
# Optional env:
#   PROPMINER_POOL              — default prl.kryptex.network:443,prl-eu.kryptex.network:443
#   PROPMINER_STRATUM_POOL        — Stratum fallback (default prl.kryptex.network:7048 per Kryptex docs)
#   PROPMINER_USE_STRATUM=1       — Skip gRPC; use Stratum only
#   PROPMINER_POOL_MODE=stratum   — Same as USE_STRATUM=1
#   PROPMINER_GPUS              — default 0
#   PROPMINER_WORKER            — worker name if not embedded in wallet (max 32 alnum)
#   PROPMINER_RESTART_ON_EXIT   — 1 (default) restart on crash; 0 exit container
#   PROPMINER_BATCH               — mine matmuls per poll (default: cache or 4)
#   PROPMINER_USE_TUNE_CACHE=1    — apply ~/.cache/propminer/autotune.json (default on)
#   PROPMINER_STRICT_KNOB_CACHE=1  — fail if kernel_knobs.json mismatches built .so
#   PEARL_GEMM_CONSUMER_CLUSTER_M   — default 2 for prod (set 1 to disable clustering)
#   PROPMINER_BATCH_TUNE=1        — run batch sweep at startup (slow; prefer tune script)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build_remote_test"
source "${ROOT}/scripts/setup_cuda_env.sh"

# Aggressive RTX 5090 production defaults (override via env if needed).
export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"
export PEARL_GEMM_CONSUMER_CLUSTER_M="${PEARL_GEMM_CONSUMER_CLUSTER_M:-2}"

WALLET="${PROPMINER_WALLET:-}"
if [[ -n "${PROPMINER_POOL:-}" ]]; then
    POOL="${PROPMINER_POOL}"
    if [[ -n "${PROPMINER_POOL_FALLBACK:-}" ]]; then
        POOL="${POOL},${PROPMINER_POOL_FALLBACK}"
    fi
else
    POOL="${PROPMINER_POOL_FALLBACK:-prl.kryptex.network:443,prl-eu.kryptex.network:443}"
fi
GPUS="${PROPMINER_GPUS:-0}"
WORKER="${PROPMINER_WORKER:-}"
RESTART_ON_EXIT="${PROPMINER_RESTART_ON_EXIT:-1}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wallet)
            WALLET="${2:-}"
            shift 2
            ;;
        --pool)
            POOL="${2:-}"
            shift 2
            ;;
        --gpus)
            GPUS="${2:-}"
            shift 2
            ;;
        --worker)
            WORKER="${2:-}"
            shift 2
            ;;
        --bench)
            echo "[mine] ERROR: --bench is not allowed in production mine mode." >&2
            exit 1
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Split WALLET.worker (SRBMiner-style) when PROPMINER_WORKER is unset.
if [[ -z "${WORKER}" && "${WALLET}" == *.* ]]; then
    WORKER="${WALLET#*.}"
    WALLET="${WALLET%%.*}"
    echo "[mine] Parsed wallet.worker -> wallet=${WALLET} worker=${WORKER}" | propminer_log
fi

if [[ -z "${WALLET}" ]]; then
    echo "[mine] ERROR: wallet required. Set PROPMINER_WALLET or pass --wallet ADDRESS." >&2
    echo "[mine] Format: WALLET.worker or krxUSERNAME.worker (see Kryptex Pearl pool docs)" >&2
    exit 1
fi

if [[ ${#WALLET} -lt 8 ]]; then
    echo "[mine] ERROR: wallet address looks too short (${#WALLET} chars)." >&2
    exit 1
fi

if [[ "${WALLET}" == *"--bench"* ]]; then
    echo "[mine] ERROR: wallet must not contain --bench." >&2
    exit 1
fi

# Warn when neither dot/slash worker suffix nor --worker is set.
if [[ -z "${WORKER}" && "${WALLET}" != *.* && "${WALLET}" != */* ]]; then
    echo "[mine] WARN: wallet has no .worker or /worker suffix; pool may show as 'propminer'." \
        | propminer_log
    echo "[mine] WARN: set PROPMINER_WORKER or use WALLET.worker format." | propminer_log
fi

for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "${arg}" == "--bench" ]]; then
        echo "[mine] ERROR: --bench in extra args is not allowed in mine mode." >&2
        exit 1
    fi
done

if [[ -n "${PROPMINER_AUTOTUNE:-}" && "${PROPMINER_AUTOTUNE}" != "0" ]]; then
    echo "[mine] WARN: PROPMINER_AUTOTUNE=${PROPMINER_AUTOTUNE} may shrink N below 262144." \
        | propminer_log
fi

echo "===== PropMiner production mining =====" | propminer_log
date | propminer_log

setup_cuda_runtime_env

if ! prepare_prebuilt_binaries "${BUILD_DIR}"; then
    exit 1
fi

echo "[mine] GPU info:" | propminer_log
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
    --format=csv,noheader | propminer_log || true

export PROPMINER_STRATUM_POOL="${PROPMINER_STRATUM_POOL:-prl.kryptex.network:7048}"
echo "[mine] mode=production grpc_pools=${POOL} stratum_fallback=${PROPMINER_STRATUM_POOL} gpus=${GPUS}" | propminer_log
echo "[mine] wallet=${WALLET} worker=${WORKER:-<from-wallet-or-default>}" | propminer_log
echo "[mine] profile: --rtx5090 aggressive prod (N=max VRAM, cluster_m=${PEARL_GEMM_CONSUMER_CLUSTER_M})" \
    | propminer_log
echo "[mine] PROPMINER_USE_TUNE_CACHE=${PROPMINER_USE_TUNE_CACHE} (autotune.json overrides cluster/carveout)" \
    | propminer_log
if [[ -n "${PROPMINER_BATCH:-}" ]]; then
    echo "[mine] PROPMINER_BATCH=${PROPMINER_BATCH}" | propminer_log
fi
echo "[mine] Starting miner (runs until container is stopped)..." | propminer_log

MINER_ARGS=(--rtx5090 --gpus "${GPUS}" --pool "${POOL}" --wallet "${WALLET}")
if [[ -n "${WORKER}" ]]; then
    MINER_ARGS+=(--worker "${WORKER}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    MINER_ARGS+=("${EXTRA_ARGS[@]}")
fi

run_mine_loop() {
    local rc=0
    run_propminer "${BUILD_DIR}/propminer" "${MINER_ARGS[@]}" || rc=$?
    return "${rc}"
}

if [[ "${RESTART_ON_EXIT}" == "0" ]]; then
    run_mine_loop
    exit $?
fi

# Production: keep container alive — retry on any exit (pool blips, Salad SIGTERM races).
while true; do
    run_mine_loop
    rc=$?
    if [[ "${rc}" -eq 0 && -n "${PROPMINER_ONCE:-}" ]]; then
        echo "[mine] propminer exited cleanly (PROPMINER_ONCE=1); stopping." | propminer_log
        exit 0
    fi
    echo "[mine] propminer exited rc=${rc}; restarting in 10s..." | propminer_log
    sleep 10
done
