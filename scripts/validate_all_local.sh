#!/usr/bin/env bash
# Everything we can validate on Mac without CUDA/GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "===== PropMiner validate_all_local ====="

echo "[1/3] Host unit tests..."
./scripts/local_host_tests.sh

echo "[2/3] Shell script syntax..."
for f in "${ROOT}/scripts"/*.sh; do
    bash -n "${f}"
done

echo "[3/3] Pre-deploy gate (host-only path)..."
./scripts/pre_deploy_gate.sh

echo "===== validate_all_local PASS ====="
echo "GPU work: run ./scripts/run_remaining_5090.sh on RTX 5090 Linux host"
