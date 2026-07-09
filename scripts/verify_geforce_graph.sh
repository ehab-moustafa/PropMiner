#!/usr/bin/env bash
# RTX 5090: CUDA graph capture/replay for consumer vs GeForce v2 GEMM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PEARL_MAKE="${ROOT}/third_party/pearl-gemm/csrc/capi"
GRAPH_BIN="${PEARL_MAKE}/build/verify_geforce_v2_graph"
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

echo "== GeForce v2 CUDA graph replay gate =="

if [[ ! -x "${GRAPH_BIN}" ]]; then
  echo "Building verify_geforce_v2_graph..."
  make -C "${PEARL_MAKE}" -j"${JOBS}" \
    PEARL_GEMM_ARCH=blackwell \
    PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
    PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 \
    PEARL_GEMM_BLACKWELL_GEFORCE_V2=1 \
    PEARL_SM120_GEFORCE_V2_GRAPH_VERIFY=1 \
    verify-geforce-v2-graph
fi

"${GRAPH_BIN}"
echo ""
echo "PASS: graph replay harness succeeded."
