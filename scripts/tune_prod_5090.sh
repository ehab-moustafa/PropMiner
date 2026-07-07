#!/usr/bin/env bash
# One-shot RTX 5090 production tuning for Kryptex Stratum mining.
# Run once on the 5090 box, then mine with PROPMINER_USE_TUNE_CACHE=1.
#
# Order:
#   1. Compile-time kernel knobs (optional rebuild + self-test gate)
#   2. Unified runtime autotune (batch + graph_batch + cluster + carveout)
#      -> ~/.cache/propminer/autotune.json
#
# Usage: ./scripts/tune_prod_5090.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "===== PropMiner RTX 5090 production tune (Kryptex Stratum) ====="
if [[ "${PROPMINER_SKIP_KNOB_TUNE:-0}" == "1" ]]; then
    echo "[tune-prod] Step 1/2: kernel knobs SKIPPED (PROPMINER_SKIP_KNOB_TUNE=1)"
else
    echo "[tune-prod] Step 1/2: kernel knobs (compile sweep)..."
    "${ROOT}/scripts/tune_blackwell_knobs.sh" 15 3
fi

echo "[tune-prod] Step 2/2: runtime autotune (batch x graph_batch x cluster)..."
"${ROOT}/scripts/tune_runtime_prod.sh" "${1:-15}" "${2:-3}"

echo ""
echo "[tune-prod] Done. Start mining:"
echo "  PROPMINER_MODE=mine PROPMINER_USE_TUNE_CACHE=1 PROPMINER_AUTOTUNE=0"
echo "  PROPMINER_WALLET=<wallet> ./scripts/docker_entrypoint.sh"
echo "Cache: ~/.cache/propminer/autotune.json"
