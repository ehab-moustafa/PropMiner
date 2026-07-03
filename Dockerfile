# PropMiner RTX 5090 build & benchmark image.
# Based on NVIDIA CUDA 12.8 devel image so nvcc, cuDNN headers, and build tools
# are already present. This avoids downloading CUDA inside flaky containers.
#
# Build:
#   docker build -t propminer-rtx5090 .
# Run interactively:
#   docker run --gpus all -it propminer-rtx5090
# Run the full test kit:
#   docker run --gpus all propminer-rtx5090

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install runtime + build dependencies in one layer.
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    gnupg \
    python3 \
    python3-pip \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable toolchain.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH=/root/.cargo/bin:${PATH}
RUN rustup default stable

# Working directory for the cloned repo.
WORKDIR /root/PropMiner

# Copy the repository into the image. On cloud runners you can instead mount the
# repo at runtime and use this image as the base.
COPY . /root/PropMiner

# Default entrypoint runs the remote test kit. Override with bash for debugging.
ENTRYPOINT ["./scripts/remote_test_kit.sh"]
