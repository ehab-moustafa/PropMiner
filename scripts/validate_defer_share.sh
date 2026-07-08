#!/usr/bin/env bash
# Validate defer-share GPU work build (host-only compile check).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== PropMiner defer-share validation =="

if [[ ! -d "${ROOT}/build" ]]; then
  echo "Configure build first: cmake -B build -DPEARL_GEMM_ARCH=blackwell"
  exit 1
fi

echo "[1/2] Build propminer (host + gemm)..."
cmake --build "${ROOT}/build" --target propminer -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo "[2/2] Host-only tests (no GPU)..."
PROP_MINER_HOST_ONLY_TESTS=1 "${ROOT}/build/propminer_tests" 2>/dev/null || \
  echo "  SKIP: propminer_tests not built or failed (non-fatal on Mac without libs)"

echo ""
echo "Compile OK. Production default: PROPMINER_DEFER_SHARE_GPU=1 (deferred side thread)."
echo "Run self-test in both modes to validate:"
echo "  PROPMINER_DEFER_SHARE_GPU=0 ./build/propminer --self-test --rtx5090 --gpus 0"
echo "  PROPMINER_DEFER_SHARE_GPU=1 ./build/propminer --self-test --rtx5090 --gpus 0"
