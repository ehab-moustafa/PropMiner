#!/usr/bin/env bash
# PropMiner Salad runner (SRBMiner-style extracted bundle).
# Run from PropMiner-Salad/ after: tar xf PropMiner-Salad-Linux.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PROPMINER_LOG_DIR:-/tmp/propminer}"
mkdir -p "${LOG_DIR}"

log() {
    echo "$*" | tee -a "${LOG_DIR}/summary.txt"
}

setup_wsl2_env() {
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
    export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"
    export PROPMINER_AUTOTUNE="${PROPMINER_AUTOTUNE:-0}"
    export PROPMINER_STRATUM_DIFF="${PROPMINER_STRATUM_DIFF:-262144}"
    export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"

    export LD_LIBRARY_PATH="${ROOT}/lib:${ROOT}:${LD_LIBRARY_PATH:-}"

    if [[ -e /dev/dxg ]]; then
        log "[env] WSL2 detected (/dev/dxg)"
        local wsl_libs=()
        local so
        for so in \
            /usr/lib/x86_64-linux-gnu/libdxcore.so \
            /usr/lib/wsl/drivers/*/libnvdxgdmal.so.1 \
            /usr/lib/wsl/drivers/*/libcuda_loader.so \
            /usr/lib/wsl/drivers/*/libnvidia-ml_loader.so \
            /usr/lib/wsl/drivers/*/libcuda.so.1.1; do
            local f
            for f in ${so}; do
                [[ -f "${f}" ]] && wsl_libs+=("${f}")
            done
        done
        if [[ ${#wsl_libs[@]} -gt 0 ]]; then
            export LD_PRELOAD
            LD_PRELOAD="$(IFS=:; echo "${wsl_libs[*]}")"
            log "[env] LD_PRELOAD set for WSL2"
        fi
        export LD_LIBRARY_PATH="${ROOT}/lib:${ROOT}:/usr/lib/wsl/drivers:${LD_LIBRARY_PATH}"
    elif [[ -e /dev/nvidia0 ]]; then
        log "[env] Native Linux NVIDIA detected"
    else
        log "[env] No GPU device nodes; trying bundled CUDA libs"
    fi
}

WALLET="${PROPMINER_WALLET:-}"
WORKER="${PROPMINER_WORKER:-}"
if [[ -z "${WORKER}" && "${WALLET}" == *.* ]]; then
    WORKER="${WALLET#*.}"
    WALLET="${WALLET%%.*}"
    log "[mine] Parsed wallet.worker -> wallet=${WALLET} worker=${WORKER}"
fi

if [[ -z "${WALLET}" ]]; then
    echo "ERROR: set PROPMINER_WALLET (e.g. krxXXXX)" >&2
    echo "[salad] Container staying alive for logs (sleep infinity)..." >&2
    exec sleep infinity
fi

if [[ -n "${PROPMINER_POOL:-}" ]]; then
    POOL="${PROPMINER_POOL}"
    if [[ -n "${PROPMINER_POOL_FALLBACK:-}" ]]; then
        POOL="${POOL},${PROPMINER_POOL_FALLBACK}"
    fi
else
    POOL=""
fi

RESTART_ON_EXIT="${PROPMINER_RESTART_ON_EXIT:-1}"
export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
export PROPMINER_STRATUM_POOL="${PROPMINER_STRATUM_POOL:-prl.kryptex.network:7048,prl-eu.kryptex.network:7048}"
export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"
export PROPMINER_AUTOTUNE="${PROPMINER_AUTOTUNE:-0}"
export PROPMINER_STRATUM_PASSWORD="${PROPMINER_STRATUM_PASSWORD:-x}"
export PROPMINER_VERBOSE_STRATUM="${PROPMINER_VERBOSE_STRATUM:-1}"
export PROPMINER_VERBOSE_SHARES="${PROPMINER_VERBOSE_SHARES:-1}"

WALLET_ARG="${WALLET}"
if [[ -n "${WORKER}" ]]; then
    WALLET_ARG="${WALLET}.${WORKER}"
fi

setup_wsl2_env

log "===== PropMiner Salad (SRB-style bundle) ====="
log "version=$(cat "${ROOT}/VERSION" 2>/dev/null || echo unknown)"
log "$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | log || true
log "pool=${POOL:-<stratum-only>} stratum=${PROPMINER_STRATUM_POOL} pool-user=${WALLET_ARG} (wallet.worker)"

MINER_ARGS=(--rtx5090 --wallet "${WALLET_ARG}")
if [[ -n "${PROPMINER_GPUS:-}" ]]; then
    MINER_ARGS+=(--gpus "${PROPMINER_GPUS}")
fi
if [[ -n "${POOL}" ]]; then
    MINER_ARGS+=(--pool "${POOL}")
fi

run_once() {
    (cd "${ROOT}" && LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" LD_PRELOAD="${LD_PRELOAD:-}" \
        ./propminer "${MINER_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/propminer.log")
}

if [[ "${RESTART_ON_EXIT}" == "0" ]]; then
    run_once
    exit $?
fi

while true; do
    if [[ "${PROPMINER_AUTO_UPDATE:-0}" == "1" && "${PROPMINER_USE_RELEASE:-0}" == "1" ]]; then
        TAG="${PROPMINER_RELEASE_TAG:-continuous}"
        REPO="${PROPMINER_RELEASE_REPO:-ehab-moustafa/PropMiner}"
        ASSET="${PROPMINER_RELEASE_ASSET:-PropMiner-Salad-Linux.tar.gz}"
        URL="${PROPMINER_RELEASE_URL:-https://github.com/${REPO}/releases/download/${TAG}/${ASSET}}"
        REMOTE_VER="$(curl -fsSL "${URL}" | tar -xOzf - PropMiner-Salad/VERSION 2>/dev/null || true)"
        LOCAL_VER="$(cat "${ROOT}/VERSION" 2>/dev/null || true)"
        if [[ -n "${REMOTE_VER}" && "${REMOTE_VER}" != "${LOCAL_VER}" ]]; then
            log "[release] auto-update ${LOCAL_VER:-none} -> ${REMOTE_VER}"
            curl -fsSL --retry 3 -o /tmp/propminer-salad.tar.gz "${URL}"
            tar xzf /tmp/propminer-salad.tar.gz -C /tmp
            cp -f /tmp/PropMiner-Salad/propminer \
                /tmp/PropMiner-Salad/libpearl_gemm_capi.so \
                /tmp/PropMiner-Salad/libpearl_mining_capi.so "${ROOT}/"
            cp -f /tmp/PropMiner-Salad/VERSION "${ROOT}/VERSION"
        fi
    fi
    set +e
    run_once
    rc=$?
    set -e
    if [[ "${rc}" -eq 0 && -n "${PROPMINER_ONCE:-}" ]]; then
        log "[mine] clean exit (PROPMINER_ONCE=1)"
        exit 0
    fi
    log "[mine] propminer exited rc=${rc}; restart in 10s"
    sleep 10
done
