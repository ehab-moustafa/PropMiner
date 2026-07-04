# PropMiner RTX 5090 runtime image.
#
# Multi-stage build:
#   1. Builder stage uses nvidia/cuda:12.8.0-devel to compile propminer.
#   2. Runtime stage uses plain Ubuntu with multiple CUDA fallback strategies
#      so it works on native Linux cloud GPUs (vast.ai), Salad WSL2 hosts,
#      and Salad native-Linux hosts without manual image selection.
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
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH}
RUN rustup default stable

WORKDIR /root/PropMiner
COPY . /root/PropMiner

# Pre-build PropMiner so runtime nodes only download binaries.
# We use the same CMake flags remote_test_kit.sh would use.
RUN cmake -S . -B build_runtime \
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
    && cmake --build build_runtime --target propminer -j"$(nproc)"

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install base libraries plus tools we may need for runtime fallbacks.
# We keep curl/wget/gnupg so we can install WSL2 CUDA at runtime if needed.
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/PropMiner

# Copy prebuilt binaries from the builder stage.
COPY --from=builder /root/PropMiner/build_runtime/propminer .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_gemm_capi.so .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_mining_capi.so .

# Strategy 1: standard Linux CUDA runtime libraries.
COPY --from=builder /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.12

# Strategy 2: WSL2 Ubuntu CUDA toolkit (does not overwrite the host driver).
# Pre-install it so WSL2 Salad nodes work without runtime downloads.
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-8 \
    && rm -f cuda-keyring_1.1-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

# Copy scripts needed for benchmark/self-test.
COPY --from=builder /root/PropMiner/scripts/remote_test_kit.sh ./scripts/remote_test_kit.sh

ENTRYPOINT ["./scripts/remote_test_kit.sh"]
