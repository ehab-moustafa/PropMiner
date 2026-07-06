#!/usr/bin/env bash
# Validate SM120 GeForce experimental kernel build (NOT tcgen05 — see external_repos/).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo "== PropMiner SM120 GeForce kernel build validation =="

echo "[1/3] Consumer baseline (production default)..."
make -C "${ROOT}/third_party/pearl-gemm/csrc/capi" -j"${JOBS}" \
  PEARL_GEMM_ARCH=blackwell \
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
  >/dev/null
echo "  OK: consumer cp_async build"

echo "[2/3] Dual-kernel build (consumer + sm120_geforce experimental)..."
make -C "${ROOT}/third_party/pearl-gemm/csrc/capi" -j"${JOBS}" \
  PEARL_GEMM_ARCH=blackwell \
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
  PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 \
  clean >/dev/null 2>&1 || true
make -C "${ROOT}/third_party/pearl-gemm/csrc/capi" -j"${JOBS}" \
  PEARL_GEMM_ARCH=blackwell \
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
  PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 \
  >/dev/null
echo "  OK: dual-kernel build"

echo "[3/3] Phase 0 tcgen05 probe (host compile only if no nvcc)..."
PROBE="${ROOT}/third_party/pearl-gemm/csrc/blackwell/phase0_tcgen05_sm120_probe.cu"
if command -v nvcc >/dev/null 2>&1; then
  if nvcc -arch=sm_120a "${PROBE}" -o /tmp/phase0_tcgen05_probe 2>/tmp/phase0_tcgen05_build.log; then
    echo "  NOTE: tcgen05 probe compiled (unexpected on sm_120a — verify hardware)"
  else
    echo "  OK: ptxas rejected tcgen05 on sm_120a (expected for RTX 5090)"
    head -3 /tmp/phase0_tcgen05_build.log || true
  fi
else
  echo "  SKIP: nvcc not found"
fi

echo ""
echo "Compile gates passed. Hardware validation on RTX 5090:"
echo "  ./scripts/verify_geforce_transcript.sh"
echo "  PEARL_GEMM_KERNEL=consumer ./build/propminer --self-test --rtx5090 --gpus 0"
echo "  PEARL_GEMM_KERNEL=geforce   ./build/propminer --self-test --rtx5090 --gpus 0"
echo ""
echo "PRODUCTION: PEARL_GEMM_KERNEL unset (consumer). Docker unchanged."
