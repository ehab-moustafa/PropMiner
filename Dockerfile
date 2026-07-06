# PropMiner RTX 5090 Docker images.
#
# Targets:
#   runtime (default) — slim Salad deploy: prebuilt binaries + CUDA runtime for /dev/dxg.
#   devel             — full source + nvcc/cmake/rust: on-box tune_prod, knob sweep, GeForce builds.
#
# Build:
#   docker build -t propminer-rtx5090 .                    # runtime
#   docker build --target devel -t propminer-rtx5090:devel . # devel

# ── Builder stage ───────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3 \
    python3-pip \
    libssl-dev \
    pkg-config \
    ccache \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/ccache
ENV CCACHE_MAXSIZE=10G
ENV CCACHE_COMPRESS=1
ENV CCACHE_SLOPPINESS=include_file_mtime,include_file_ctime,time_macros
ENV CCACHE_BASEDIR=/root/PropMiner
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache
ENV NVCC="ccache /usr/local/cuda/bin/nvcc"
ENV PATH=/usr/lib/ccache:${PATH}
RUN ln -sf /usr/bin/ccache /usr/lib/ccache/nvcc

# Install Rust stable toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH}
RUN rustup default stable

WORKDIR /root/PropMiner

# Layer 1: third-party CUDA/Rust libs only. Cached when only host C++ changes.
COPY CMakeLists.txt .
COPY third_party third_party
COPY include include
COPY src/cuda src/cuda

ARG CMAKE_BUILD_ARGS="\
  -DCMAKE_BUILD_TYPE=Release \
  -DPROPMINER_BUILD_HOST=OFF \
  -DPROP_MINER_CUDA_ARCH=blackwell \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DPEARL_GEMM_BLACKWELL_BM=128 \
  -DPEARL_GEMM_BLACKWELL_BN=256 \
  -DPEARL_GEMM_BLACKWELL_KBLOCK=128 \
  -DPEARL_GEMM_BLACKWELL_STAGES=2 \
  -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS=3 \
  -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS=1 \
  -DPEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async"

RUN --mount=type=cache,id=propminer-ccache,target=/ccache \
    --mount=type=cache,id=propminer-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=propminer-cargo-git,target=/root/.cargo/git \
    --mount=type=cache,id=propminer-gemm-build,target=/root/PropMiner/third_party/pearl-gemm/csrc/capi/build \
    --mount=type=cache,id=propminer-rust-build,target=/root/PropMiner/build_runtime/pearl_mining_capi \
    cmake -S . -B build_runtime ${CMAKE_BUILD_ARGS} \
    && cmake --build build_runtime \
       --target stage_gemm_lib stage_mining_lib \
       -j"$(nproc)" \
    && ccache -s

# Layer 2: host sources + scripts + propminer link. Fast when only src/host changes.
COPY src/host src/host
COPY scripts scripts

RUN --mount=type=cache,id=propminer-ccache,target=/ccache \
    --mount=type=cache,id=propminer-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=propminer-cargo-git,target=/root/.cargo/git \
    --mount=type=cache,id=propminer-gemm-build,target=/root/PropMiner/third_party/pearl-gemm/csrc/capi/build \
    --mount=type=cache,id=propminer-rust-build,target=/root/PropMiner/build_runtime/pearl_mining_capi \
    cmake -S . -B build_runtime ${CMAKE_BUILD_ARGS} -DPROPMINER_BUILD_HOST=ON \
    && cmake --build build_runtime --target propminer -j"$(nproc)" \
    && ccache -s

# Audit: all cubins must be sm_120a only (RTX 5090 native).
RUN cuobjdump -lelf build_runtime/libpearl_gemm_capi.so | tee build_runtime/cubins.txt \
    && echo "==> unique cubin archs:" \
    && ONLY_ARCHS="$(grep -oE 'sm_[0-9a-z_]+' build_runtime/cubins.txt | sort -u)" \
    && echo "${ONLY_ARCHS}" \
    && echo "${ONLY_ARCHS}" | grep -qx 'sm_120a'

# ── CUDA 12.8 runtime stage ────────────────────────────────────────────────
FROM ubuntu:24.04 AS cuda128-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/cuda128

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN BASE="https://developer.download.nvidia.com/compute/cuda/redist" && \
    DEST="/root/cuda128/usr/local/cuda-12.8/targets/x86_64-linux" && \
    mkdir -p "${DEST}" && \
    cd /tmp && \
    curl -fsSL -o cuda_cudart.tar.xz "${BASE}/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz" && \
    curl -fsSL -o cuda_nvrtc.tar.xz "${BASE}/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.8.61-archive.tar.xz" && \
    curl -fsSL -o libcublas.tar.xz "${BASE}/libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.3.14-archive.tar.xz" && \
    curl -fsSL -o libnvjitlink.tar.xz "${BASE}/libnvjitlink/linux-x86_64/libnvjitlink-linux-x86_64-12.8.61-archive.tar.xz" && \
    for f in *.tar.xz; do tar -xf "$f" --strip-components=1 -C "${DEST}"; done && \
    rm -f *.tar.xz

# ── Runtime stage (default) ─────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PEARL_GEMM_CONSUMER_CLUSTER_M=2
ENV PROPMINER_USE_TUNE_CACHE=1
ENV CUDA_MODULE_LOADING=EAGER
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
# Zero-config Salad validation: PROPMINER_MODE=full (self-test + 180s benchmark).
# Production mining: override PROPMINER_MODE=mine and set PROPMINER_WALLET.
ENV PROPMINER_MODE=full
ENV PROPMINER_QUICK_EXIT=0
ENV PROPMINER_SKIP_BENCH=0
ENV PROPMINER_SKIP_SWEEP=1
ENV PROPMINER_SKIP_NCU=1
ENV PROPMINER_KEEP_ALIVE_SECONDS=3600
ENV PROPMINER_BENCH_SECONDS=300
ENV PROPMINER_BENCH_GRACE_SECONDS=120

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/PropMiner

COPY --from=builder /root/PropMiner/build_runtime/propminer .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_gemm_capi.so .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_mining_capi.so .

COPY --from=cuda128-runtime /root/cuda128/usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvJitLink.so.12 /usr/local/cuda/lib64/libnvJitLink.so.12

COPY --from=builder /root/PropMiner/scripts/setup_cuda_env.sh ./scripts/setup_cuda_env.sh
COPY --from=builder /root/PropMiner/scripts/run_mining.sh ./scripts/run_mining.sh
COPY --from=builder /root/PropMiner/scripts/docker_entrypoint.sh ./scripts/docker_entrypoint.sh
COPY --from=builder /root/PropMiner/scripts/remote_test_kit.sh ./scripts/remote_test_kit.sh
COPY --from=builder /root/PropMiner/scripts/tune_blackwell_knobs.sh ./scripts/tune_blackwell_knobs.sh
COPY --from=builder /root/PropMiner/scripts/tune_kernel_knobs_common.sh ./scripts/tune_kernel_knobs_common.sh
COPY --from=builder /root/PropMiner/scripts/tune_mine_batch.sh ./scripts/tune_mine_batch.sh
COPY --from=builder /root/PropMiner/scripts/tune_cluster_sweep.sh ./scripts/tune_cluster_sweep.sh
COPY --from=builder /root/PropMiner/scripts/tune_prod_5090.sh ./scripts/tune_prod_5090.sh
COPY --from=builder /root/PropMiner/scripts/salad_tune_and_remaining.sh ./scripts/salad_tune_and_remaining.sh
COPY --from=builder /root/PropMiner/scripts/run_remaining_5090.sh ./scripts/run_remaining_5090.sh
COPY --from=builder /root/PropMiner/scripts/pre_deploy_gate.sh ./scripts/pre_deploy_gate.sh
COPY --from=builder /root/PropMiner/scripts/verify_geforce_transcript.sh ./scripts/verify_geforce_transcript.sh
COPY --from=builder /root/PropMiner/scripts/validate_knob_manifest.sh ./scripts/validate_knob_manifest.sh
COPY --from=builder /root/PropMiner/scripts/compare_bench.sh ./scripts/compare_bench.sh
COPY --from=builder /root/PropMiner/scripts/profile_gemm_ncu.sh ./scripts/profile_gemm_ncu.sh
COPY --from=builder /root/PropMiner/scripts/build_and_benchmark.sh ./scripts/build_and_benchmark.sh
RUN mkdir -p ./results
COPY results/baseline_5090_sm120.json ./results/baseline_5090_sm120.json

RUN chmod +x ./scripts/*.sh

ENTRYPOINT ["./scripts/docker_entrypoint.sh"]

# ── Devel stage: nvcc + full tree for on-box rebuild / tune_prod / GeForce ──
FROM builder AS devel

ENV PROPMINER_DEVEL=1
ENV PROPMINER_BUILD_DIR=/root/PropMiner/build_runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PEARL_GEMM_CONSUMER_CLUSTER_M=2
ENV PROPMINER_USE_TUNE_CACHE=1
ENV CUDA_MODULE_LOADING=EAGER
ENV CUDA_DEVICE_MAX_CONNECTIONS=1
ENV PROPMINER_MODE=full
ENV PROPMINER_QUICK_EXIT=0
ENV PROPMINER_SKIP_BENCH=0
ENV PROPMINER_SKIP_SWEEP=1
ENV PROPMINER_SKIP_NCU=1
ENV PROPMINER_KEEP_ALIVE_SECONDS=3600
ENV PROPMINER_BENCH_SECONDS=300
ENV PROPMINER_BENCH_GRACE_SECONDS=120
ENV PATH=/usr/local/cuda/bin:/root/.cargo/bin:${PATH}

# Salad WSL2 uses /dev/dxg + host driver store — same redist libs as runtime image.
COPY --from=cuda128-runtime /root/cuda128/usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvJitLink.so.12 /usr/local/cuda/lib64/libnvJitLink.so.12

RUN mkdir -p results && \
    ln -sf build_runtime/propminer propminer && \
    ln -sf build_runtime/libpearl_gemm_capi.so libpearl_gemm_capi.so && \
    ln -sf build_runtime/libpearl_mining_capi.so libpearl_mining_capi.so && \
    ln -sf build_runtime build && \
    chmod +x scripts/*.sh

COPY results/baseline_5090_sm120.json ./results/baseline_5090_sm120.json

ENTRYPOINT ["./scripts/docker_entrypoint.sh"]
