#!/usr/bin/env bash
# RTX 5090: isolate GeForce v2 vs CUDA graph failures at production shape.
# Prefer the master debugger: ./scripts/debug_geforce_v2.sh
# This script runs only the PropMiner self-test isolation matrix (phase 5).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/debug_geforce_v2.sh" --phase 5 "$@"
