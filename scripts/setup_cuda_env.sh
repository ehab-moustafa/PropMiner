#!/usr/bin/env bash
# Shared CUDA / WSL2 runtime setup for PropMiner on Salad and native Linux.
# Source this file from other scripts — do not execute directly.

: "${ROOT:?ROOT must be set before sourcing setup_cuda_env.sh}"

PROPMINER_LOG_DIR="${PROPMINER_LOG_DIR:-${ROOT}/results}"
mkdir -p "${PROPMINER_LOG_DIR}"

BUNDLED_CUDA="/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/cuda/lib64"
WSL_DRIVER_PATH="/usr/lib/wsl/drivers"
PYTORCH_CUDA_DIR="/usr/local/lib/python3.12/dist-packages/torch/lib"

propminer_log() {
    tee -a "${PROPMINER_LOG_DIR}/summary.txt"
}

setup_cuda_runtime_env() {
    export NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-all}"
    export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    export PROPMINER_USE_STRATUM="${PROPMINER_USE_STRATUM:-1}"
    export PROPMINER_USE_TUNE_CACHE="${PROPMINER_USE_TUNE_CACHE:-1}"
    export PROPMINER_AUTOTUNE="${PROPMINER_AUTOTUNE:-0}"
    # Leave empty for Kryptex vardiff (dynamic). Set a number to pin a static diff.
    export PROPMINER_STRATUM_DIFF="${PROPMINER_STRATUM_DIFF:-}"
    export PROPMINER_N_CAP="${PROPMINER_N_CAP:-131072}"
    export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-EAGER}"
    export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
    # Hung-GPU recovery: exit on wedged CUDA stream so supervisor restarts fresh.
    export PROPMINER_STALL_RESTART_MS="${PROPMINER_STALL_RESTART_MS:-30000}"
    export PROPMINER_STALL_RESTART_DELAY_SEC="${PROPMINER_STALL_RESTART_DELAY_SEC:-3}"

    if [[ -e /dev/dxg ]]; then
        echo "[env] WSL2 detected (/dev/dxg present)." | propminer_log

        local DXCORE="/usr/lib/x86_64-linux-gnu/libdxcore.so"
        local WSL_LIBS=()
        local so
        for so in \
            "${DXCORE}" \
            /usr/lib/wsl/drivers/*/libnvdxgdmal.so.1 \
            /usr/lib/wsl/drivers/*/libcuda_loader.so \
            /usr/lib/wsl/drivers/*/libnvidia-ml_loader.so \
            /usr/lib/wsl/drivers/*/libcuda.so.1.1; do
            local f
            for f in ${so}; do
                [[ -f "$f" ]] && WSL_LIBS+=("$f")
            done
        done

        if [[ ${#WSL_LIBS[@]} -gt 0 ]]; then
            export LD_PRELOAD
            LD_PRELOAD=$(IFS=:; echo "${WSL_LIBS[*]}")
            echo "[env] Set LD_PRELOAD: ${LD_PRELOAD}" | propminer_log
        fi

        export LD_LIBRARY_PATH="${BUNDLED_CUDA}:${WSL_DRIVER_PATH}:${LD_LIBRARY_PATH:-}"
        echo "[env] Set LD_LIBRARY_PATH for WSL2: ${LD_LIBRARY_PATH}" | propminer_log
    elif [[ -e /dev/nvidia0 ]] || [[ -e /dev/nvidiactl ]] || compgen -G '/dev/nvidia[0-9]*' >/dev/null; then
        echo "[env] Native Linux NVIDIA detected (nvidia device nodes present)." | propminer_log
        export LD_LIBRARY_PATH="${BUNDLED_CUDA}:${LD_LIBRARY_PATH:-}"
        echo "[env] Set LD_LIBRARY_PATH for native Linux: ${LD_LIBRARY_PATH}" | propminer_log
    else
        echo "[env] No GPU device nodes found. Will try bundled CUDA runtime." | propminer_log
        export LD_LIBRARY_PATH="${BUNDLED_CUDA}:${LD_LIBRARY_PATH:-}"
    fi
}

ensure_pytorch_cuda_fallback() {
    if [[ -d "${PYTORCH_CUDA_DIR}" && -f "${PYTORCH_CUDA_DIR}/libcudart.so.12" ]]; then
        echo "[env] PyTorch CUDA fallback already installed." | propminer_log
        return 0
    fi
    echo "[env] Installing PyTorch CUDA runtime as fallback (slow, one-time)..." | propminer_log
    pip install torch --index-url https://download.pytorch.org/whl/cu121 --break-system-packages \
        > "${PROPMINER_LOG_DIR}/pytorch_install.log" 2>&1 || true
    if [[ -f "${PYTORCH_CUDA_DIR}/libcudart.so.12" ]]; then
        echo "[env] PyTorch CUDA fallback installed." | propminer_log
        return 0
    fi
    echo "[env] PyTorch CUDA fallback install failed." | propminer_log
    return 1
}

run_propminer() {
    local bin="${1}"
    shift
    setup_cuda_runtime_env
    if LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" "${bin}" "$@" 2>>"${PROPMINER_LOG_DIR}/propminer_stderr.log"; then
        return 0
    fi
    local rc=$?
    echo "[env] propminer exited ${rc}. stderr tail:" | propminer_log
    tail -20 "${PROPMINER_LOG_DIR}/propminer_stderr.log" 2>/dev/null | propminer_log || true

    # PyTorch fallback is only for CUDA driver/runtime init failures — not pool SSL, etc.
    if ! tail -50 "${PROPMINER_LOG_DIR}/propminer_stderr.log" 2>/dev/null \
        | grep -qiE 'cuda error|no cuda devices|cuda driver|libcudart|unable to init cuda|failed to initialize'; then
        return "${rc}"
    fi

    echo "[env] CUDA init failure detected; trying PyTorch fallback..." | propminer_log
    if ensure_pytorch_cuda_fallback; then
        LD_LIBRARY_PATH="${PYTORCH_CUDA_DIR}:${LD_LIBRARY_PATH}" "${bin}" "$@"
        return $?
    fi
    return "${rc}"
}

prepare_prebuilt_binaries() {
    local build_dir="${1:-${ROOT}/build_remote_test}"
    if [[ ! -x "${ROOT}/propminer" || ! -f "${ROOT}/libpearl_gemm_capi.so" || ! -f "${ROOT}/libpearl_mining_capi.so" ]]; then
        echo "[build] ERROR: prebuilt binaries missing in ${ROOT}" | propminer_log
        return 1
    fi
    mkdir -p "${build_dir}"
    cp -f "${ROOT}/libpearl_gemm_capi.so" "${ROOT}/libpearl_mining_capi.so" "${build_dir}/"
    if ! cp -f "${ROOT}/propminer" "${build_dir}/" 2>/dev/null; then
        if [[ ! -x "${build_dir}/propminer" ]]; then
            echo "[build] ERROR: cannot copy propminer and no usable binary in ${build_dir}" | propminer_log
            return 1
        fi
    fi
    return 0
}

# Resolve nvidia-smi for hashrate telemetry (gpu%, power, temp, clocks).
ensure_nvidia_smi() {
    if [[ -n "${PROPMINER_NVIDIA_SMI:-}" && -x "${PROPMINER_NVIDIA_SMI}" ]]; then
        export PROPMINER_NVIDIA_SMI
        echo "[env] nvidia-smi=${PROPMINER_NVIDIA_SMI} (from env)" | propminer_log
    else
        local cand found=""
        for cand in \
            "$(command -v nvidia-smi 2>/dev/null || true)" \
            /usr/bin/nvidia-smi \
            /usr/local/nvidia/bin/nvidia-smi \
            /usr/lib/wsl/lib/nvidia-smi; do
            [[ -z "${cand}" ]] && continue
            if [[ -x "${cand}" ]]; then
                found="${cand}"
                break
            fi
        done
        if [[ -z "${found}" ]]; then
            echo "[env] WARN: nvidia-smi not found — install host NVIDIA utils or set PROPMINER_NVIDIA_SMI" | propminer_log
            return 1
        fi
        export PROPMINER_NVIDIA_SMI="${found}"
        echo "[env] nvidia-smi=${found}" | propminer_log
    fi
    if ! "${PROPMINER_NVIDIA_SMI}" --query-gpu=name,utilization.gpu,power.draw,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | head -1 | propminer_log; then
        echo "[env] WARN: nvidia-smi query failed — telemetry lines may omit gpu/power/temp" | propminer_log
        return 1
    fi
    return 0
}
