#!/usr/bin/env bash
# RTX 5090: verify GeForce transcript bytes match consumer reference.
# v1: scripts/verify_geforce_transcript.sh
# v2 (when built): also runs consumer/v1 vs v2 memcmp + v2 self-test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROPMINER="${PROPMINER:-${ROOT}/build/propminer}"
PEARL_MAKE="${ROOT}/third_party/pearl-gemm/csrc/capi"
VERIFY_V1_BIN="${PEARL_MAKE}/build/verify_transcript_sm120_geforce"
VERIFY_V2_BIN="${PEARL_MAKE}/build/verify_transcript_sm120_geforce_v2"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
RUN_V2="${PROPMINER_VERIFY_GEFORCE_V2:-1}"

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

if [[ ! -x "${VERIFY_V1_BIN}" ]]; then
  echo "Building verify_transcript_sm120_geforce (v1)..."
  make -C "${PEARL_MAKE}" -j"${JOBS}" \
    PEARL_GEMM_ARCH=blackwell \
    PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
    PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 \
    PEARL_SM120_GEFORCE_VERIFY_MAIN=1 \
    verify-geforce-transcript
fi

echo "[1/5] Consumer vs GeForce v1 transcript memcmp..."
"${VERIFY_V1_BIN}"

if [[ "${RUN_V2}" == "1" ]]; then
  if [[ ! -x "${VERIFY_V2_BIN}" ]]; then
    echo "Building verify_transcript_sm120_geforce_v2..."
    make -C "${PEARL_MAKE}" -j"${JOBS}" \
      PEARL_GEMM_ARCH=blackwell \
      PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
      PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 \
      PEARL_GEMM_BLACKWELL_GEFORCE_V2=1 \
      PEARL_SM120_GEFORCE_V2_VERIFY_MAIN=1 \
      verify-geforce-v2-transcript
  fi
  echo "[2/5] Consumer vs GeForce v2 + v1 vs v2 memcmp..."
  "${VERIFY_V2_BIN}"
else
  echo "[2/5] Skipping v2 memcmp (PROPMINER_VERIFY_GEFORCE_V2=0)"
fi

if [[ ! -x "${PROPMINER}" ]]; then
  echo "FAIL: build propminer first: cmake --build build --target propminer" >&2
  exit 1
fi

echo ""
echo "== GeForce kernel end-to-end self-test (transcript + share proof) =="

echo "[3/5] Consumer reference..."
PEARL_GEMM_KERNEL=consumer "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo "[4/5] GeForce v1 (explicit rollback path)..."
PEARL_GEMM_KERNEL=geforce_v1 "${PROPMINER}" --self-test --rtx5090 --gpus 0

if [[ "${RUN_V2}" == "1" ]]; then
  GEMM_SO="${ROOT}/build/libpearl_gemm_capi.so"
  if [[ -f "${GEMM_SO}" ]] && ! strings "${GEMM_SO}" | grep -q '+geforce_v2'; then
    echo "FAIL: libpearl_gemm_capi.so lacks geforce_v2 — rebuild propminer with:" >&2
    echo "  cmake -DPEARL_GEMM_BLACKWELL_GEFORCE_V2=ON ... && cmake --build build --target propminer" >&2
    exit 1
  fi
  echo "[5/5] GeForce v2 (default when env unset)..."
  PEARL_GEMM_KERNEL=geforce_v2 "${PROPMINER}" --self-test --rtx5090 --gpus 0
  echo "[5b/5] GeForce v2 default path (unset PEARL_GEMM_KERNEL)..."
  env -u PEARL_GEMM_KERNEL "${PROPMINER}" --self-test --rtx5090 --gpus 0
else
  echo "[5/5] Skipping v2 self-test (PROPMINER_VERIFY_GEFORCE_V2=0)"
fi

echo ""
echo "PASS: memcmp harness and kernel self-tests succeeded."
