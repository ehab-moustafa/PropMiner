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
echo "[host] ref tests OK"

# BLAKE3 / PoW CPU suites (GPU kernel math, no device)
BLAKE3_OFF="${OUT}_blake3_offload"
"${CXX}" -std=c++17 -O2 -I src/host/tests \
  src/host/tests/blake3_offload_test.cpp \
  src/host/tests/blake3_reference.c \
  src/host/tests/ref_blake3.cpp \
  -o "${BLAKE3_OFF}"
"${BLAKE3_OFF}"
echo "[host] blake3_offload_test OK"

BLAKE3_FIN="${OUT}_blake3_finalize"
"${CXX}" -std=c++17 -O2 -I src/host/tests \
  src/host/tests/blake3_finalize_optimization_test.cpp \
  src/host/tests/blake3_reference.c \
  -o "${BLAKE3_FIN}"
"${BLAKE3_FIN}"
echo "[host] blake3_finalize_optimization_test OK"

TRANSCRIPT_SAFE="${OUT}_transcript_mainloop"
"${CXX}" -std=c++17 -O2 -I src/host/tests \
  src/host/tests/transcript_mainloop_safety_test.cpp \
  -o "${TRANSCRIPT_SAFE}"
"${TRANSCRIPT_SAFE}"
echo "[host] transcript_mainloop_safety_test OK"

WORKSPACE_SIZE="${OUT}_workspace_transcript"
"${CXX}" -std=c++17 -O2 \
  src/host/tests/workspace_transcript_size_test.cpp \
  -o "${WORKSPACE_SIZE}"
"${WORKSPACE_SIZE}"
echo "[host] workspace_transcript_size_test OK"

echo "===== OK (ref + blake3 + transcript safety + workspace sizing) ====="
