#!/usr/bin/env bash
# Production mining entrypoint for Salad / Docker (stays alive until stopped).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build_remote_test"
source "${ROOT}/scripts/setup_cuda_env.sh"

WALLET="${PROPMINER_WALLET:-}"
POOL="${PROPMINER_POOL:-prl.kryptex.network:443}"
GPUS="${PROPMINER_GPUS:-0}"
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
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${WALLET}" ]]; then
    echo "[mine] ERROR: wallet required. Set PROPMINER_WALLET or pass --wallet ADDRESS." >&2
    exit 1
fi

echo "===== PropMiner production mining =====" | propminer_log
date | propminer_log

setup_cuda_runtime_env

if ! prepare_prebuilt_binaries "${BUILD_DIR}"; then
    exit 1
fi

echo "[mine] GPU info:" | propminer_log
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader \
    | propminer_log || true

echo "[mine] wallet=${WALLET} pool=${POOL} gpus=${GPUS}" | propminer_log
echo "[mine] Starting miner (runs until container is stopped)..." | propminer_log

MINER_ARGS=(--rtx5090 --gpus "${GPUS}" --pool "${POOL}" --wallet "${WALLET}")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    MINER_ARGS+=("${EXTRA_ARGS[@]}")
fi

run_propminer "${BUILD_DIR}/propminer" "${MINER_ARGS[@]}"
