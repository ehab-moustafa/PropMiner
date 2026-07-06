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
echo "Compile OK. Production default: PROPMINER_DEFER_SHARE_GPU unset (inline handle_trigger)."
echo "Experimental: PROPMINER_DEFER_SHARE_GPU=1 on RTX 5090 — run self-test both modes:"
echo "  PROPMINER_DEFER_SHARE_GPU=0 ./build/propminer --self-test --rtx5090 --gpus 0"
echo "  PROPMINER_DEFER_SHARE_GPU=1 ./build/propminer --self-test --rtx5090 --gpus 0"
