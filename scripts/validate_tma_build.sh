#!/usr/bin/env bash
# Validate consumer TMA scaffold: cp_async default still builds; optional TMA build.
# GPU transcript memcmp requires RTX 5090 hardware — run remote_test_kit on test box.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMM_MAKE="${ROOT}/third_party/pearl-gemm/csrc/capi/Makefile"

echo "== PropMiner TMA build validation =="

echo "[1/2] cp_async baseline (production default)..."
make -C "${ROOT}/third_party/pearl-gemm/csrc/capi" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
  PEARL_GEMM_ARCH=blackwell \
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
  >/dev/null
echo "  OK: cp_async build succeeded"

echo "[2/2] TMA experimental build..."
make -C "${ROOT}/third_party/pearl-gemm/csrc/capi" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
  PEARL_GEMM_ARCH=blackwell \
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma \
  clean >/dev/null 2>&1 || true
make -C "${ROOT}/third_party/pearl-gemm/csrc/capi" -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" \
  PEARL_GEMM_ARCH=blackwell \
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma \
  >/dev/null
echo "  OK: TMA build succeeded"

echo ""
echo "Compile gates passed. For byte-identity on hardware:"
echo "  1. Build cp_async + run reference transcript on 5090"
echo "  2. Build tma + memcmp transcript vs reference"
echo "  3. ./propminer --self-test --rtx5090 --gpus 0 on TMA build"
echo "Keep PRODUCTION on cp_async until gates pass (Docker/CMake defaults unchanged)."
