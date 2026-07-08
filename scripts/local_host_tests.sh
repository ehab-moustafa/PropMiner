#!/usr/bin/env bash
# Host-only correctness tests — runs on Mac (no CUDA/GPU required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

OUT="${1:-/tmp/propminer_ref_tests}"
CXX="${CXX:-clang++}"

echo "===== PropMiner host-only tests (no GPU) ====="
"${CXX}" -std=c++20 -O2 \
  -DPROP_MINER_HOST_ONLY_TESTS=1 \
  -DPROP_MINER_DISABLE_RUST_CRYPTO=1 \
  -I include -I src/host -I src/host/pearl \
  src/host/tests.cpp \
  src/host/tests/blake3_reference.c \
  src/host/tests/ref_blake3.cpp \
  src/host/tests/ref_pearl.cpp \
  src/host/pearl/pearl_types.cpp \
  src/host/pearl/job_key.cpp \
  src/host/pearl/protobuf_encoder.cpp \
  src/host/pearl/proto/mining_v2.cpp \
  src/host/pearl/host_signal_header.cpp \
  src/host/pearl/pow_target_utils.cpp \
  src/host/pearl/bincode_encoder.cpp \
  src/host/pearl/share_diagnostics.cpp \
  src/host/stratum/simple_json.cpp \
  -o "${OUT}" -lpthread

"${OUT}"
echo "===== OK ====="
