# PropMiner RTX 5090 runtime image.
#
# Multi-stage build:
#   1. Builder stage uses nvidia/cuda:12.8.0-devel to compile propminer.
#   2. Runtime stage uses nvidia/cuda:12.8.0-base and copies only the binaries.
# This keeps the deployed image small (~1.5 GB instead of ~6 GB) so 200 Salad
# nodes start quickly and do not need to download compilers/CUTLASS/Rust.
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
# Use a plain Ubuntu image, NOT nvidia/cuda:*. The official CUDA runtime images
# ship a stub libcuda.so.1 that returns 0 devices when the host NVIDIA runtime
# does not replace it. A plain Ubuntu image lets Salad inject the real host
# libcuda and /dev/nvidia* devices, which is how SRB miner works on Salad.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Only the libraries needed to run the miner. libssl3 for the Rust mining C API.
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/PropMiner

# Copy prebuilt binaries from the builder stage.
COPY --from=builder /root/PropMiner/build_runtime/propminer .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_gemm_capi.so .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_mining_capi.so .

# Copy only the CUDA runtime libraries needed by our binary. Do NOT copy
# libcuda.so.* — the host driver provides that when the GPU is mounted.
COPY --from=builder /usr/local/cuda/lib64/libcudart.so.12 .

# Copy scripts needed for benchmark/self-test.
COPY --from=builder /root/PropMiner/scripts/remote_test_kit.sh ./scripts/remote_test_kit.sh

ENV LD_LIBRARY_PATH=/root/PropMiner:${LD_LIBRARY_PATH}

ENTRYPOINT ["./scripts/remote_test_kit.sh"]
