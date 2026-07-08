#!/usr/bin/env bash
# Production mining entrypoint for Salad / Docker (stays alive until stopped).
#
# Required env:
#   PROPMINER_WALLET  — Pearl/Kryptex wallet (e.g. krxXXXX.worker or prl1p...worker)
#
# Optional env:
#   PROPMINER_POOL              — default prl.kryptex.network:443,prl-eu.kryptex.network:443
#   PROPMINER_AUTOTUNE=0|1|2|force  Runtime autotune sweep + autotune.json cache
#   PROPMINER_USE_TUNE_CACHE=1      Load autotune.json (batch/graph_batch/cluster)
#   PROPMINER_BATCH / GRAPH_BATCH / CLUSTER_M — manual override only (optional)
#   PROPMINER_USE_RELEASE=1       — download binaries from GitHub Release (bootstrap image)
#   PROPMINER_RELEASE_TAG=continuous — rolling release tag (default)
#   PROPMINER_AUTO_UPDATE=1         — check for new release on each mine restart
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build_remote_test"
source "${ROOT}/scripts/setup_cuda_env.sh"
source "${ROOT}/scripts/download_release.sh"

# Aggressive RTX 5090 production defaults (override via env if needed).
export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"
export PROPMINER_AUTOTUNE="${PROPMINER_AUTOTUNE:-0}"
export PROPMINER_STRATUM_POOL="${PROPMINER_STRATUM_POOL:-prl.kryptex.network:7048,prl-eu.kryptex.network:7048}"
# Leave empty for Kryptex vardiff (dynamic). Set a number to pin a static diff.
export PROPMINER_STRATUM_DIFF="${PROPMINER_STRATUM_DIFF:-}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"

WALLET="${PROPMINER_WALLET:-}"
if [[ -n "${PROPMINER_POOL:-}" ]]; then
    POOL="${PROPMINER_POOL}"
    if [[ -n "${PROPMINER_POOL_FALLBACK:-}" ]]; then
        POOL="${POOL},${PROPMINER_POOL_FALLBACK}"
    fi
else
    POOL=""
fi
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
    echo "[mine] WARN: PROPMINER_AUTOTUNE=${PROPMINER_AUTOTUNE} may change N/batch from defaults." \
        | propminer_log
fi

echo "===== PropMiner production mining =====" | propminer_log
date | propminer_log

setup_cuda_runtime_env
ensure_nvidia_smi || true

if ! ensure_binaries "${BUILD_DIR}"; then
    exit 1
fi

echo "[mine] GPU info:" | propminer_log

export PROPMINER_STRATUM_POOL="${PROPMINER_STRATUM_POOL:-prl.kryptex.network:7048,prl-eu.kryptex.network:7048}"
echo "[mine] mode=production stratum=${PROPMINER_STRATUM_POOL} gpus=${PROPMINER_GPUS:-all}" | propminer_log
echo "[mine] wallet=${WALLET} worker=${WORKER:-<from-wallet-or-default>}" | propminer_log
echo "[mine] profile: --rtx5090 aggressive prod (N=max VRAM, cluster_m=${PEARL_GEMM_CONSUMER_CLUSTER_M})" \
    | propminer_log
echo "[mine] PROPMINER_USE_TUNE_CACHE=${PROPMINER_USE_TUNE_CACHE} (autotune.json overrides cluster/carveout)" \
    | propminer_log
if [[ -n "${PROPMINER_BATCH:-}" ]]; then
    echo "[mine] PROPMINER_BATCH=${PROPMINER_BATCH}" | propminer_log
fi
echo "[mine] Starting miner (runs until container is stopped)..." | propminer_log

MINER_ARGS=(--rtx5090 --wallet "${WALLET}")
if [[ -n "${PROPMINER_GPUS:-}" ]]; then
    MINER_ARGS+=(--gpus "${PROPMINER_GPUS}")
fi
if [[ -n "${POOL}" ]]; then
    MINER_ARGS+=(--pool "${POOL}")
fi
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
STALL_RESTART_DELAY="${PROPMINER_STALL_RESTART_DELAY_SEC:-3}"
while true; do
    maybe_auto_update_release
    run_mine_loop
    rc=$?
    if [[ "${rc}" -eq 0 && -n "${PROPMINER_ONCE:-}" ]]; then
        echo "[mine] propminer exited cleanly (PROPMINER_ONCE=1); stopping." | propminer_log
        exit 0
    fi
    if [[ "${rc}" -eq 42 ]]; then
        echo "[mine] propminer wedged GPU (stall guard rc=42); fast restart in ${STALL_RESTART_DELAY}s..." | propminer_log
        sleep "${STALL_RESTART_DELAY}"
        continue
    fi
    echo "[mine] propminer exited rc=${rc}; restarting in 10s..." | propminer_log
    sleep 10
done
