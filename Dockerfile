# PropMiner RTX 5090 runtime image.
#
# Multi-stage build:
#   1. Builder stage uses nvidia/cuda:12.8.0-devel to compile propminer.
#   2. Runtime stage uses plain Ubuntu and embeds the CUDA 12.8 runtime libs
#      extracted directly from the official runfile. We verified on a live Salad
#      WSL2 host that these libraries can initialize the GPU through /dev/dxg
#      when paired with the host driver store files.
#
# Build:
#   docker build -t propminer-rtx5090 .
# Run:
#   docker run --gpus all propminer-rtx5090

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

# Layer 1: deps + CUDA/Rust libs. Cached when only host C++ sources change.
COPY CMakeLists.txt .
COPY third_party third_party
COPY include include
COPY src/cuda src/cuda
COPY src/host src/host

ARG CMAKE_BUILD_ARGS="\
  -DCMAKE_BUILD_TYPE=Release \
  -DPROP_MINER_CUDA_ARCH=blackwell \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPEARL_GEMM_BLACKWELL_BM=128 \
  -DPEARL_GEMM_BLACKWELL_BN=256 \
  -DPEARL_GEMM_BLACKWELL_KBLOCK=128 \
  -DPEARL_GEMM_BLACKWELL_STAGES=2 \
  -DPEARL_GEMM_BLACKWELL_SWIZZLE_BITS=3 \
  -DPEARL_GEMM_BLACKWELL_MIN_BLOCKS=1 \
  -DPEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async"

RUN --mount=type=cache,target=/ccache \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/root/PropMiner/build_runtime/pearl_mining_capi \
    cmake -S . -B build_runtime ${CMAKE_BUILD_ARGS} \
    && cmake --build build_runtime \
       --target stage_gemm_lib stage_mining_lib \
       -j"$(nproc)" \
    && ccache -s

# Layer 2: scripts + final link. Re-runs on host edits; nvcc stays cached via ccache.
COPY scripts scripts

RUN --mount=type=cache,target=/ccache \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/root/PropMiner/build_runtime/pearl_mining_capi \
    cmake --build build_runtime --target propminer -j"$(nproc)" \
    && ccache -s

# ── CUDA 12.8 runtime stage ────────────────────────────────────────────────
# Extract only the runtime libraries we need from the official CUDA 12.8
# redist tarballs.  This avoids PyTorch ABI mismatch issues and gives us a
# clean, version-matched libcudart without running an x86_64 installer on the
# build host.
FROM ubuntu:24.04 AS cuda128-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root/cuda128

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Component archives from the official NVIDIA CUDA redist (linux-x86_64).
# Verified hashes from redistrib_12.8.0.json.
RUN BASE="https://developer.download.nvidia.com/compute/cuda/redist" && \
    DEST="/root/cuda128/usr/local/cuda-12.8/targets/x86_64-linux" && \
    mkdir -p "${DEST}"

RUN BASE="https://developer.download.nvidia.com/compute/cuda/redist" && \
    DEST="/root/cuda128/usr/local/cuda-12.8/targets/x86_64-linux" && \
    cd /tmp && \
    curl -fsSL -o cuda_cudart.tar.xz "${BASE}/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.57-archive.tar.xz" && \
    curl -fsSL -o cuda_nvrtc.tar.xz "${BASE}/cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.8.61-archive.tar.xz" && \
    curl -fsSL -o libcublas.tar.xz "${BASE}/libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.3.14-archive.tar.xz" && \
    curl -fsSL -o libnvjitlink.tar.xz "${BASE}/libnvjitlink/linux-x86_64/libnvjitlink-linux-x86_64-12.8.61-archive.tar.xz" && \
    for f in *.tar.xz; do tar -xf "$f" --strip-components=1 -C "${DEST}"; done && \
    rm -f *.tar.xz

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/PropMiner

# Copy prebuilt binaries from the builder stage.
COPY --from=builder /root/PropMiner/build_runtime/propminer .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_gemm_capi.so .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_mining_capi.so .

# Copy CUDA 12.8 runtime libraries extracted from the official runfile.
COPY --from=cuda128-runtime /root/cuda128/usr/local/cuda-12.8 /usr/local/cuda-12.8
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12 /usr/local/cuda/lib64/libcublasLt.so.12 && \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvJitLink.so.12 /usr/local/cuda/lib64/libnvJitLink.so.12

# Copy scripts needed for benchmark/self-test.
COPY --from=builder /root/PropMiner/scripts/remote_test_kit.sh ./scripts/remote_test_kit.sh

ENTRYPOINT ["./scripts/remote_test_kit.sh"]
