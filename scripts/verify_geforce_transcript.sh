#!/usr/bin/env bash
# RTX 5090: verify sm120_geforce transcript bytes match consumer reference.
# Requires dual-kernel .so built with PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROPMINER="${PROPMINER:-${ROOT}/build/propminer}"
PEARL_MAKE="${ROOT}/third_party/pearl-gemm/csrc/capi"
VERIFY_BIN="${PEARL_MAKE}/build/verify_transcript_sm120_geforce"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi not found (needs RTX 5090 host)" >&2
  exit 1
fi

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
if [[ "${CC}" != 12.* ]]; then
  echo "FAIL: expected CC 12.x Blackwell, got ${CC}" >&2
  exit 1
fi

echo "== GeForce transcript byte-identity harness =="

if [[ ! -x "${VERIFY_BIN}" ]]; then
  echo "Building verify_transcript_sm120_geforce..."
  make -C "${PEARL_MAKE}" -j"${JOBS}" \
    PEARL_GEMM_ARCH=blackwell \
    PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
    PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 \
    PEARL_SM120_GEFORCE_VERIFY_MAIN=1 \
    verify-geforce-transcript
fi

echo "[1/3] Consumer vs GeForce transcript memcmp..."
"${VERIFY_BIN}"

if [[ ! -x "${PROPMINER}" ]]; then
  echo "FAIL: build propminer first: cmake --build build --target propminer" >&2
  exit 1
fi

echo ""
echo "== GeForce kernel end-to-end self-test (transcript + share proof) =="

echo "[2/3] Consumer reference..."
PEARL_GEMM_KERNEL=consumer "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo "[3/3] GeForce experimental (TMA warp-specialized)..."
PEARL_GEMM_KERNEL=geforce "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo ""
echo "PASS: memcmp harness and both kernel self-tests succeeded."
