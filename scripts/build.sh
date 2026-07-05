#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo " PropMiner Build Script (RTX 5090 / sm_120a)"
echo "=========================================="

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

echo "CUDA: $(nvcc --version | grep release)"
echo ""

# Create build dir
mkdir -p build && cd build

# CMake configure — blackwell profile → single sm_120a cubin via GNU Makefile.
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROP_MINER_CUDA_ARCH=blackwell \
    -DCMAKE_CUDA_ARCHITECTURES=120a \
    "$@"

echo ""
echo "Compiling..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo " Build complete: ./propminer"
echo "=========================================="
echo ""
if command -v cuobjdump &> /dev/null && [[ -f libpearl_gemm_capi.so ]]; then
    echo "Cubin archs in libpearl_gemm_capi.so:"
    cuobjdump -lelf libpearl_gemm_capi.so 2>/dev/null | grep -oE 'sm_[0-9a-z_]+' | sort -u || true
fi
ls -la propminer 2>/dev/null || true
