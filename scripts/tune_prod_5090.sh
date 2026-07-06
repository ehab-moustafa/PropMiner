#!/usr/bin/env bash
# One-shot RTX 5090 production tuning for Kryptex gRPC mining.
# Run once on the 5090 box, then mine with PROPMINER_USE_TUNE_CACHE=1 (default in run_mining.sh).
#
# Order:
#   1. Compile-time kernel knobs (rebuild + self-test gate)
#   2. cluster_m {1,2,4} + full runtime autotune -> ~/.cache/propminer/autotune.json
#   3. Mine batch sweep -> ~/.cache/propminer/mine_batch.json
#
# Usage: ./scripts/tune_prod_5090.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "===== PropMiner RTX 5090 production tune (Kryptex) ====="
echo "[tune-prod] Step 1/3: kernel knobs (compile sweep)..."
"${ROOT}/scripts/tune_blackwell_knobs.sh" 15 3

echo "[tune-prod] Step 2/3: cluster + runtime autotune..."
"${ROOT}/scripts/tune_cluster_sweep.sh" 20 3 12

echo "[tune-prod] Step 3/3: mine batch sweep..."
"${ROOT}/scripts/tune_mine_batch.sh" 12 2

echo ""
echo "[tune-prod] Done. Start mining:"
echo "  PROPMINER_MODE=mine PROPMINER_WALLET=<wallet> ./scripts/docker_entrypoint.sh"
echo "Caches: ~/.cache/propminer/{kernel_knobs,autotune,mine_batch}.json"
