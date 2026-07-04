#!/usr/bin/env bash
# Remote test kit for RTX 5090 PropMiner validation.
#
# What this does:
#   - Checks the GPU / driver / CUDA stack.
#   - Installs build deps if they are missing.
#   - Builds PropMiner targeting sm_120 (Blackwell consumer).
#   - Runs correctness self-test (no pool connection).
#   - Runs a 60-second hashrate benchmark.
#   - Profiles the GEMM kernel with ncu if available.
#   - Sweeps a small, high-value set of Blackwell kernel knobs.
#   - Writes all logs and numbers to results/ for easy upload.
#
# Usage on the RTX 5090 Ubuntu 24.04 box:
#   chmod +x scripts/remote_test_kit.sh
#   ./scripts/remote_test_kit.sh
#
# Then copy results/ back to your mac and paste the summary here.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ROOT}/build_remote_test"
RESULTS_DIR="${ROOT}/results"
mkdir -p "${RESULTS_DIR}"

BENCH_SECONDS=60
SWEEP_SECONDS=15

PREBUILT=false
if [[ -x "${ROOT}/propminer" && -f "${ROOT}/libpearl_gemm_capi.so" && -f "${ROOT}/libpearl_mining_capi.so" ]]; then
    PREBUILT=true
fi

echo "===== PropMiner RTX 5090 Remote Test Kit =====" | tee "${RESULTS_DIR}/summary.txt"
date | tee -a "${RESULTS_DIR}/summary.txt"

# ── 1. Environment snapshot ────────────────────────────────────────────────
echo "[env] GPU info:" | tee -a "${RESULTS_DIR}/summary.txt"
nvidia-smi --query-gpu=name,compute_cap,driver_version,pcie.link.gen.max,pcie.link.width.max,memory.total,power.limit,power.default_limit --format=csv,noheader | tee "${RESULTS_DIR}/gpu_info.csv" | tee -a "${RESULTS_DIR}/summary.txt" || true

echo "[env] CUDA runtime diagnostics:" | tee -a "${RESULTS_DIR}/summary.txt"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}" | tee -a "${RESULTS_DIR}/summary.txt"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}" | tee -a "${RESULTS_DIR}/summary.txt"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}" | tee -a "${RESULTS_DIR}/summary.txt"
ls -l /dev/nvidia* 2>/dev/null | tee -a "${RESULTS_DIR}/summary.txt" || echo "[env] No /dev/nvidia* devices" | tee -a "${RESULTS_DIR}/summary.txt"
ldconfig -p | grep -E 'libcuda|libcudart' | tee -a "${RESULTS_DIR}/summary.txt" || true
cat /proc/driver/nvidia/version 2>/dev/null | head -3 | tee -a "${RESULTS_DIR}/summary.txt" || true
find /usr -name 'libcuda.so*' 2>/dev/null | tee -a "${RESULTS_DIR}/summary.txt" || true
if command -v readlink >/dev/null 2>&1; then
    for p in $(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF}'); do
        echo "[env] libcuda.so.1 resolves to: $(readlink -f "$p")" | tee -a "${RESULTS_DIR}/summary.txt"
    done
fi
echo "[env] ldd propminer:" | tee -a "${RESULTS_DIR}/summary.txt"
ldd "${BUILD_DIR}/propminer" 2>/dev/null | tee -a "${RESULTS_DIR}/summary.txt" || true

if [[ "${PREBUILT}" == "true" ]]; then
    echo "[env] Using prebuilt binaries." | tee -a "${RESULTS_DIR}/summary.txt"
else
    nvcc --version > "${RESULTS_DIR}/nvcc_version.txt" 2>&1 || true
    cat "${RESULTS_DIR}/nvcc_version.txt" | tee -a "${RESULTS_DIR}/summary.txt"

    cmake --version > "${RESULTS_DIR}/cmake_version.txt" 2>&1 || true
    cargo --version > "${RESULTS_DIR}/cargo_version.txt" 2>&1 || true
fi

# ── 2. Install dependencies if missing ─────────────────────────────────────
if [[ "${PREBUILT}" == "true" ]]; then
    echo "[deps] Prebuilt binaries present. Skipping dependency installs." | tee -a "${RESULTS_DIR}/summary.txt"
else
    echo "[deps] Checking build dependencies..." | tee -a "${RESULTS_DIR}/summary.txt"
    MISSING=""
    for pkg in build-essential cmake git curl libssl-dev pkg-config python3; do
        if ! dpkg -l | grep -q "^ii  ${pkg} "; then
            MISSING="${MISSING} ${pkg}"
        fi
    done

    if [[ -n "${MISSING}" ]]; then
        echo "[deps] Installing:${MISSING}" | tee -a "${RESULTS_DIR}/summary.txt"
        apt-get update
        apt-get install -y ${MISSING}
    fi
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[deps] ERROR: python3 not found after install" | tee -a "${RESULTS_DIR}/summary.txt"
        exit 1
    fi

    # Install Rust via rustup.rs (not the apt rustup wrapper) for a clean toolchain.
    if ! command -v cargo >/dev/null 2>&1; then
        echo "[deps] Installing Rust..." | tee -a "${RESULTS_DIR}/summary.txt"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    fi
    # Make cargo available to subprocesses.
    if [[ -f "${HOME}/.cargo/env" ]]; then
        source "${HOME}/.cargo/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        echo "[deps] ERROR: cargo not found after Rust install" | tee -a "${RESULTS_DIR}/summary.txt"
        exit 1
    fi
    rustup default stable >/dev/null 2>&1 || true

    # Minimal CUDA toolkit install: only nvcc and headers, not the full metapackage.
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[deps] Installing minimal CUDA toolkit..." | tee -a "${RESULTS_DIR}/summary.txt"
        apt-get install -y wget gnupg
        wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda.list
        apt-get update
        apt-get install -y cuda-toolkit-12-8
    fi

    # Ensure nvcc is on PATH for cmake and make.
    if [[ -d /usr/local/cuda-12.8/bin ]]; then
        export PATH=/usr/local/cuda-12.8/bin:${PATH}
    fi
    if [[ -d /usr/local/cuda/bin ]]; then
        export PATH=/usr/local/cuda/bin:${PATH}
    fi

    # Verify.
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[deps] ERROR: nvcc still not found after CUDA install" | tee -a "${RESULTS_DIR}/summary.txt"
        exit 1
    fi
    echo "[deps] nvcc: $(command -v nvcc)" | tee -a "${RESULTS_DIR}/summary.txt"

    # ncu is optional; warn if absent.
    if ! command -v ncu >/dev/null 2>&1; then
        echo "[warn] ncu not found. Profiling step will be skipped." | tee -a "${RESULTS_DIR}/summary.txt"
        echo "       Install with: apt-get install -y nvidia-nsight-compute" | tee -a "${RESULTS_DIR}/summary.txt"
    fi

    # ── 3. Pull latest code from GitHub (Salad may cache the image) ────────────
    if [[ -d "${ROOT}/.git" ]]; then
        echo "[git] Pulling latest code..." | tee -a "${RESULTS_DIR}/summary.txt"
        cd "${ROOT}"
        git fetch origin master
        git reset --hard origin/master
    fi
fi

# ── 3. Fetch CUTLASS if missing ────────────────────────────────────────────
if [[ "${PREBUILT}" == "true" ]]; then
    echo "[deps] Prebuilt binaries present. Skipping CUTLASS/pearl-blake3 fetch." | tee -a "${RESULTS_DIR}/summary.txt"
else
    if [[ ! -f "${ROOT}/third_party/pearl-gemm/third_party/cutlass/include/cutlass/cutlass.h" ]]; then
        echo "[cutlass] Cloning CUTLASS..." | tee -a "${RESULTS_DIR}/summary.txt"
        git clone --depth 1 https://github.com/NVIDIA/cutlass.git \
            "${ROOT}/third_party/pearl-gemm/third_party/cutlass" || true
    fi

    # ── 3b. Fetch pearl-blake3 crate if missing ────────────────────────────────
    if [[ ! -f "${ROOT}/third_party/pearl-blake3/Cargo.toml" ]]; then
        echo "[pearl-blake3] Missing local path dependency; trying repo fallback..." | tee -a "${RESULTS_DIR}/summary.txt"
        FALLBACK_DIR="$(mktemp -d)"
        if git clone --depth 1 https://github.com/ehab-moustafa/PropMiner.git "${FALLBACK_DIR}/PropMiner" 2>/dev/null; then
            if [[ -f "${FALLBACK_DIR}/PropMiner/third_party/pearl-blake3/Cargo.toml" ]]; then
                cp -R "${FALLBACK_DIR}/PropMiner/third_party/pearl-blake3" "${ROOT}/third_party/pearl-blake3"
                echo "[pearl-blake3] Restored from repo fallback." | tee -a "${RESULTS_DIR}/summary.txt"
            fi
        fi
        rm -rf "${FALLBACK_DIR}"
    fi
    if [[ ! -f "${ROOT}/third_party/pearl-blake3/Cargo.toml" ]]; then
        echo "[deps] ERROR: third_party/pearl-blake3 is missing. Make sure it is committed/pushed in the PropMiner repo." | tee -a "${RESULTS_DIR}/summary.txt"
        exit 1
    fi
fi

# ── 4. Build PropMiner for sm_120 (skip if prebuilt binaries exist) ────────
if [[ -x "${ROOT}/propminer" && -f "${ROOT}/libpearl_gemm_capi.so" && -f "${ROOT}/libpearl_mining_capi.so" ]]; then
    echo "[build] Prebuilt binaries found. Skipping CMake build." | tee -a "${RESULTS_DIR}/summary.txt"
    mkdir -p "${BUILD_DIR}"
    cp -v "${ROOT}/propminer" "${ROOT}/libpearl_gemm_capi.so" "${ROOT}/libpearl_mining_capi.so" "${BUILD_DIR}/" \
        | tee -a "${RESULTS_DIR}/summary.txt"
else
    echo "[build] Configuring CMake for sm_120..." | tee -a "${RESULTS_DIR}/summary.txt"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPROP_MINER_CUDA_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DPEARL_GEMM_BLACKWELL_BM=128 \
        -DPEARL_GEMM_BLACKWELL_BN=256 \
        -DPEARL_GEMM_BLACKWELL_KBLOCK=128 \
        -DPEARL_GEMM_BLACKWELL_STAGES=2 \
        -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS=3 \
        -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS=1 \
        -DPEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
        2>&1 | tee "${RESULTS_DIR}/cmake_configure.log"
fi

if [[ -x "${BUILD_DIR}/propminer" ]]; then
    echo "[build] Using prebuilt binaries." | tee -a "${RESULTS_DIR}/summary.txt"
else
    echo "[build] Compiling propminer (this takes several minutes)..." | tee -a "${RESULTS_DIR}/summary.txt"
    if ! cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc)" \
        2>&1 | tee "${RESULTS_DIR}/build.log"; then
        echo "[build] FAILED. See results/build.log" | tee -a "${RESULTS_DIR}/summary.txt"
        tail -100 "${RESULTS_DIR}/build.log" | tee -a "${RESULTS_DIR}/summary.txt" || true
        exit 1
    fi
    echo "[build] SUCCESS" | tee -a "${RESULTS_DIR}/summary.txt"
fi

ls -lh "${BUILD_DIR}/propminer" "${BUILD_DIR}/libpearl_gemm_capi.so" "${BUILD_DIR}/libpearl_mining_capi.so" | tee "${RESULTS_DIR}/binaries.txt"

# ── 5. Correctness self-test (no pool, no real mining) ─────────────────────
echo "[test] Running self-test..." | tee -a "${RESULTS_DIR}/summary.txt"
if "${BUILD_DIR}/propminer" --self-test --rtx5090 --gpus 0 > "${RESULTS_DIR}/self_test.log" 2>&1; then
    echo "[test] PASS" | tee -a "${RESULTS_DIR}/summary.txt"
else
    echo "[test] FAIL — see results/self_test.log" | tee -a "${RESULTS_DIR}/summary.txt"
    # Continue anyway to capture benchmark behavior.
fi

# ── 6. Hashrate benchmark ──────────────────────────────────────────────────
echo "[bench] Running ${BENCH_SECONDS}s hashrate benchmark..." | tee -a "${RESULTS_DIR}/summary.txt"
"${BUILD_DIR}/propminer" --bench "${BENCH_SECONDS}" --rtx5090 --gpus 0 \
    > "${RESULTS_DIR}/benchmark.log" 2>&1 || true
tail -40 "${RESULTS_DIR}/benchmark.log" | tee -a "${RESULTS_DIR}/summary.txt"

# Extract hashrate line if present.
grep -iE "hash|th/s|gh/s|mh/s|h/s" "${RESULTS_DIR}/benchmark.log" | tail -20 | tee "${RESULTS_DIR}/benchmark_hashrate.txt" || true

# ── 7. ncu profiling (optional) ────────────────────────────────────────────
if command -v ncu >/dev/null 2>&1; then
    echo "[ncu] Profiling the consumer GEMM kernel..." | tee -a "${RESULTS_DIR}/summary.txt"
    ncu -o "${RESULTS_DIR}/profile" \
        --target-processes all \
        --kernel-regex regex:transcript_gemm_kernel_consumer \
        "${BUILD_DIR}/propminer" --bench 10 --rtx5090 --gpus 0 \
        > "${RESULTS_DIR}/ncu.log" 2>&1 || true
    echo "[ncu] Report saved to results/profile.ncu-rep" | tee -a "${RESULTS_DIR}/summary.txt"
else
    echo "[ncu] Skipped (ncu not installed)." | tee -a "${RESULTS_DIR}/summary.txt"
fi

# ── 8. Shortened knob sweep (high-value candidates only) ───────────────────
echo "[sweep] Running shortened Blackwell knob sweep..." | tee -a "${RESULTS_DIR}/summary.txt"

BMS=(128)
BNS=(256)
KBLOCKS=(64 128)
STAGES_LIST=(2 3)
SWIZZLES=(3 4)
MIN_BLOCKS_LIST=(1 2)
LOAD_POLICIES=(cp_async)

BEST_LABEL=""
BEST_RATE=0

for KBLOCK in "${KBLOCKS[@]}"; do
for STAGES in "${STAGES_LIST[@]}"; do
for SWIZZLE in "${SWIZZLES[@]}"; do
for MIN_BLOCKS in "${MIN_BLOCKS_LIST[@]}"; do
for LOAD_POLICY in "${LOAD_POLICIES[@]}"; do
    LABEL="bw-128x256-k${KBLOCK}-s${STAGES}-sw${SWIZZLE}-mb${MIN_BLOCKS}-${LOAD_POLICY}"
    echo "[sweep] Building ${LABEL}..." | tee -a "${RESULTS_DIR}/summary.txt"

    cmake -S "${ROOT}" -B "${BUILD_DIR}" \
        -DPEARL_GEMM_ARCH=blackwell \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DPEARL_GEMM_BLACKWELL_BM=128 \
        -DPEARL_GEMM_BLACKWELL_BN=256 \
        -DPEARL_GEMM_BLACKWELL_KBLOCK="${KBLOCK}" \
        -DPEARL_GEMM_BLACKWELL_STAGES="${STAGES}" \
        -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS="${SWIZZLE}" \
        -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS="${MIN_BLOCKS}" \
        -DPEARL_GEMM_BLACKWELL_LOAD_POLICY="${LOAD_POLICY}" \
        -DCMAKE_BUILD_TYPE=Release \
        > "${RESULTS_DIR}/cmake-${LABEL}.log" 2>&1

    if ! cmake --build "${BUILD_DIR}" --target propminer -j"$(nproc)" \
        > "${RESULTS_DIR}/build-${LABEL}.log" 2>&1; then
        echo "[sweep] ${LABEL}: build failed" | tee -a "${RESULTS_DIR}/summary.txt"
        continue
    fi

    RATE=$("${BUILD_DIR}/propminer" --bench "${SWEEP_SECONDS}" --rtx5090 --gpus 0 2>/dev/null \
        | grep -iE "hash|th/s|gh/s|mh/s|h/s" | tail -1 | grep -oE '[0-9.]+' | head -1 || true)

    if [[ -z "${RATE}" || "${RATE}" == "0" ]]; then
        echo "[sweep] ${LABEL}: no rate" | tee -a "${RESULTS_DIR}/summary.txt"
        continue
    fi

    RATE_INT=$(printf '%.0f' "${RATE}")
    echo "[sweep] ${LABEL}: ${RATE} H/s" | tee -a "${RESULTS_DIR}/summary.txt"
    if (( RATE_INT > BEST_RATE )); then
        BEST_RATE=${RATE_INT}
        BEST_LABEL="${LABEL}"
        cp "${BUILD_DIR}/libpearl_gemm_capi.so" "${RESULTS_DIR}/libpearl_gemm_capi_${LABEL}.so"
    fi
done
done
done
done
done

echo "[sweep] Best variant: ${BEST_LABEL} (${BEST_RATE} H/s)" | tee -a "${RESULTS_DIR}/summary.txt"

# ── 9. Final summary ───────────────────────────────────────────────────────
echo ""
echo "===== Test kit complete =====" | tee -a "${RESULTS_DIR}/summary.txt"
echo "Results are in: ${RESULTS_DIR}/" | tee -a "${RESULTS_DIR}/summary.txt"
echo "Upload these back to Cursor for analysis:" | tee -a "${RESULTS_DIR}/summary.txt"
echo "  - results/summary.txt" | tee -a "${RESULTS_DIR}/summary.txt"
echo "  - results/benchmark.log" | tee -a "${RESULTS_DIR}/summary.txt"
echo "  - results/benchmark_hashrate.txt" | tee -a "${RESULTS_DIR}/summary.txt"
echo "  - results/profile.ncu-rep (if ncu was available)" | tee -a "${RESULTS_DIR}/summary.txt"
