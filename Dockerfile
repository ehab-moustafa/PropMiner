# PropMiner RTX 5090 runtime image.
#
# Multi-stage build:
#   1. Builder stage uses nvidia/cuda:12.8.0-devel to compile propminer.
#   2. Runtime stage uses plain Ubuntu with CUDA runtime libraries from
#      PyTorch's CUDA wheels (nvidia-cuda-runtime-cu12 etc.). These wheels
#      are built to work on both native Linux and WSL2 Salad hosts.
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

# Download PyTorch CUDA runtime wheels. These libs work on WSL2 where the
# official NVIDIA toolkit libs fail, because they bundle a loader that uses
# the host Windows driver through /dev/dxg.
RUN pip download --no-deps --dest /tmp/cudawheels \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-nvjitlink-cu12==12.9.86 \
    nvidia-nvtx-cu12==12.1.105

RUN mkdir -p /tmp/cudalibs && \
    for whl in /tmp/cudawheels/*.whl; do \
        unzip -q -o "$whl" -d /tmp/cudalibs; \
    done

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
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/PropMiner

# Copy prebuilt binaries from the builder stage.
COPY --from=builder /root/PropMiner/build_runtime/propminer .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_gemm_capi.so .
COPY --from=builder /root/PropMiner/build_runtime/libpearl_mining_capi.so .

# Copy CUDA runtime libraries from the PyTorch wheels.
COPY --from=builder /tmp/cudalibs/nvidia/cuda_runtime/lib/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.12
COPY --from=builder /tmp/cudalibs/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so.12
COPY --from=builder /tmp/cudalibs/nvidia/cublas/lib/libcublas.so.12 /usr/local/cuda/lib64/libcublas.so.12
COPY --from=builder /tmp/cudalibs/nvidia/cublas/lib/libcublasLt.so.12 /usr/local/cuda/lib64/libcublasLt.so.12
COPY --from=builder /tmp/cudalibs/nvidia/cufft/lib/libcufft.so.11 /usr/local/cuda/lib64/libcufft.so.11
COPY --from=builder /tmp/cudalibs/nvidia/curand/lib/libcurand.so.10 /usr/local/cuda/lib64/libcurand.so.10
COPY --from=builder /tmp/cudalibs/nvidia/cusolver/lib/libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.11
COPY --from=builder /tmp/cudalibs/nvidia/cusparse/lib/libcusparse.so.12 /usr/local/cuda/lib64/libcusparse.so.12
COPY --from=builder /tmp/cudalibs/nvidia/nvjitlink/lib/libnvjitlink.so.12 /usr/local/cuda/lib64/libnvjitlink.so.12
COPY --from=builder /tmp/cudalibs/nvidia/nvtx/lib/libnvtx.so.1 /usr/local/cuda/lib64/libnvtx.so.1

# Copy scripts needed for benchmark/self-test.
COPY --from=builder /root/PropMiner/scripts/remote_test_kit.sh ./scripts/remote_test_kit.sh

ENTRYPOINT ["./scripts/remote_test_kit.sh"]
