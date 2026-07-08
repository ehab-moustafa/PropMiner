#!/usr/bin/env bash
# RTX 5090: grouped GEMM smoke + regression gate (batch>=4 path).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROPMINER="${PROPMINER:-${ROOT}/build/propminer}"
GEMM_SO="${ROOT}/build/libpearl_gemm_capi.so"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FAIL: nvidia-smi not found (needs RTX 5090 host)" >&2
  exit 1
fi

CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
if [[ "${CC}" != 12.* ]]; then
  echo "FAIL: expected CC 12.x Blackwell, got ${CC}" >&2
  exit 1
fi

if [[ ! -x "${PROPMINER}" ]]; then
  echo "FAIL: build propminer first" >&2
  exit 1
fi

if [[ -f "${GEMM_SO}" ]] && ! strings "${GEMM_SO}" | grep -q '+grouped'; then
  echo "FAIL: libpearl_gemm_capi.so lacks +grouped — rebuild with:" >&2
  echo "  cmake -DPEARL_GEMM_GROUPED_GEMM=ON ... && cmake --build build --target propminer" >&2
  exit 1
fi

echo "== Grouped GEMM gate =="

echo "[1/4] batch=1 regression (grouped must NOT activate)..."
PROPMINER_BATCH=1 "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo "[2/4] batch=4 serial path (PEARL_GEMM_GROUPED=0)..."
PROPMINER_BATCH=4 PEARL_GEMM_GROUPED=0 "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo "[3/4] batch=4 grouped default (batch>=4, compiled in)..."
PROPMINER_BATCH=4 "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo "[4/4] batch=3 serial only (below grouped threshold)..."
PROPMINER_BATCH=3 "${PROPMINER}" --self-test --rtx5090 --gpus 0

echo "PASS: grouped GEMM smoke tests"
