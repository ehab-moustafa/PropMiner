#!/usr/bin/env bash
# Proof-safe gate before promoting kernel/build changes to production.
# Mac: host-only tests only. On 5090: also runs GPU self-tests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "===== PropMiner pre-deploy gate ====="

echo "[gate] Host-only correctness tests..."
./scripts/local_host_tests.sh

if command -v nvcc >/dev/null 2>&1 && [[ -x "${ROOT}/build/propminer" ]]; then
    echo "[gate] GPU self-test (default shape)..."
    "${ROOT}/build/propminer" --self-test --rtx5090 --gpus 0

    echo "[gate] GPU self-test (production shape)..."
    PROP_MINER_SELF_TEST_PROD=1 "${ROOT}/build/propminer" --self-test --rtx5090 --gpus 0

    if [[ -f "${ROOT}/scripts/validate_knob_manifest.sh" ]]; then
        echo "[gate] Kernel knob manifest validation..."
        "${ROOT}/scripts/validate_knob_manifest.sh"
    fi

    if [[ -f "${ROOT}/scripts/verify_geforce_transcript.sh" ]] &&
       [[ "${PROPMINER_GATE_GEFORCE:-0}" == "1" ]]; then
        echo "[gate] GeForce transcript dual self-test..."
        "${ROOT}/scripts/verify_geforce_transcript.sh"
    fi
else
    echo "[gate] Skipping GPU self-tests (no nvcc or build/propminer)"
fi

echo "===== pre-deploy gate PASS ====="
