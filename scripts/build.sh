#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo " PropMiner Build Script"
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

# CMake configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPropMiner_CUDA_ARCHS="80;86;89;90;120" \
    "$@"

echo ""
echo "Compiling..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=========================================="
echo " Build complete: ./propminer"
echo "=========================================="
echo ""
ls -la propminer pearlhash_kernel_*.cubin 2>/dev/null || true