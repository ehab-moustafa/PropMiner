# PropMiner RTX 5090 runtime image.
#
# Multi-stage build:
#   1. Builder stage uses nvidia/cuda:12.8.0-devel to compile propminer.
#   2. Runtime stage uses plain Ubuntu and pre-installs PyTorch's CUDA
#      wheel set. We verified on a live Salad WSL2 host that these are the
#      CUDA libraries that can initialize the GPU through /dev/dxg.
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
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH}
RUN rustup default stable

WORKDIR /root/PropMiner
COPY . /root/PropMiner

# Pre-build PropMiner so runtime nodes only download binaries.
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

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CUDA wheels. These are the exact libraries we verified
# make torch.cuda.is_available() == True on Salad WSL2.
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

WORKDIR /root/PropMiner

# Copy prebuilt binaries from the builder stage.
COPY --from=builder /root/PropMiner/build_runtime/propminer .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_gemm_capi.so .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_mining_capi.so .

# Copy every CUDA shared library that PyTorch installed. We copy the whole
# directory because the exact set of .so files varies between torch releases
# and we don't need to hardcode each filename.
RUN mkdir -p /usr/local/cuda/lib64
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcudart.so.12 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libnvrtc.so.12 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcublas.so.12 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcublasLt.so.12 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcufft.so.11 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcurand.so.10 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcusolver.so.11 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libcusparse.so.12 /usr/local/cuda/lib64/ || true
RUN cp /usr/local/lib/python3.12/dist-packages/torch/lib/libnvJitLink.so.12 /usr/local/cuda/lib64/ || true

# Copy scripts needed for benchmark/self-test.
COPY --from=builder /root/PropMiner/scripts/remote_test_kit.sh ./scripts/remote_test_kit.sh

ENTRYPOINT ["./scripts/remote_test_kit.sh"]
