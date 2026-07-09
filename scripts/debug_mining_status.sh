#!/usr/bin/env bash
# Quick runtime health check while mining on vast.ai.
# Usage:
#   ./scripts/debug_mining_status.sh
#   LOG=/workspace/mine_prod.log ./scripts/debug_mining_status.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${LOG:-/workspace/mine_prod.log}"

echo "=== PropMiner mining status $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,power.draw,memory.used,memory.total,temperature.gpu \
    --format=csv
  echo ""
else
  echo "GPU: nvidia-smi unavailable"
  echo ""
fi

echo "Processes:"
if pgrep -af propminer 2>/dev/null | head -5; then
  :
else
  echo "  (no propminer process found)"
fi
if pgrep -af run_mining 2>/dev/null | head -3; then
  :
else
  echo "  (no run_mining supervisor)"
fi
echo ""

echo "Build sanity:"
if grep -q 'transcript_buffer_elems(m, n, 1)' \
    "${ROOT}/third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp" 2>/dev/null; then
  echo "  workspace batch=1 fix: present"
else
  echo "  workspace batch=1 fix: MISSING — rebuild from latest source"
fi
echo ""

echo "Env:"
for v in PROPMINER_N_CAP PEARL_GEMM_KERNEL PROPMINER_BENCH_NO_GRAPH \
         PROPMINER_TRIPLE_BUFFER PEARL_GEMM_DEBUG; do
  if [[ -n "${!v:-}" ]]; then
    echo "  ${v}=${!v}"
  fi
done
echo ""

if [[ -f "${LOG}" ]]; then
  echo "Log tail (${LOG}):"
  echo "────────────────────────────────────────"
  tail -50 "${LOG}"
  echo "────────────────────────────────────────"
  echo ""
  echo "Failure signatures in log:"
  patterns=(
    "graph validation FAILED"
    "illegal memory access"
    "cuda graph unavailable"
    "graph sub-batch failed"
    "iter_batch"
    "share accepted"
    "TMAD"
  )
  for p in "${patterns[@]}"; do
    if grep -q "${p}" "${LOG}" 2>/dev/null; then
      count="$(grep -c "${p}" "${LOG}" 2>/dev/null || echo 0)"
      echo "  [${count}x] ${p}"
    fi
  done
else
  echo "Log not found: ${LOG}"
  echo "Set LOG=/path/to/mine.log"
fi

echo ""
echo "If graph failures: ./scripts/debug_geforce_v2.sh --phase 4"
echo "Full pipeline:     ./scripts/debug_geforce_v2.sh"
