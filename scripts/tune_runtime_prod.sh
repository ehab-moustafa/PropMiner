#!/usr/bin/env bash
# RTX 5090 / Stratum production runtime autotune.
# Delegates to tune_runtime_full.sh (N × batch × graph_batch × cluster × graph on/off).
#
# Usage: ./scripts/tune_runtime_prod.sh [seconds_per_combo]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/scripts/tune_runtime_full.sh" "${1:-15}"
