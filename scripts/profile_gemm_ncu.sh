#!/usr/bin/env bash
# Nsight Compute profiling for Pearl GEMM on RTX 5090.
# Captures steady-state matmul kernels and produces a diagnostic summary.
#
# Usage:
#   ./scripts/profile_gemm_ncu.sh [gpu_index] [consumer|geforce_v1|geforce_v2]
#
# Prerequisites:
#   - CUDA Toolkit with Nsight Compute (ncu) installed
#   - Built propminer binary at build/propminer

set -euo pipefail

GPU="${1:-0}"
KERNEL_HINT="${2:-geforce_v2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/results/ncu_$(date +%Y%m%d_%H%M%S)_${KERNEL_HINT}"

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: 'ncu' not found. Install Nsight Compute." >&2
    echo "  Debian/Ubuntu: sudo apt-get install nsight-compute" >&2
    echo "  RunPod: already installed on CUDA 12.x images" >&2
    exit 1
fi

if [[ ! -f "${ROOT}/build/propminer" ]]; then
    echo "ERROR: build/propminer not found. Build first:" >&2
    echo "  ./scripts/build_and_benchmark.sh 0" >&2
    exit 1
fi

mkdir -p "${OUT}"

# Select kernel dispatch via env var.
export PEARL_GEMM_KERNEL="${KERNEL_HINT}"

# Determine ncu kernel regex based on which kernel we're profiling.
case "${KERNEL_HINT}" in
    consumer)
        KERNEL_REGEX="transcript_gemm_kernel"
        ;;
    geforce_v1)
        KERNEL_REGEX="transcript_gemm_sm120_geforce_kernel"
        ;;
    geforce_v2)
        KERNEL_REGEX="transcript_gemm_sm120_geforce_v2_kernel"
        ;;
    *)
        echo "ERROR: Unknown kernel '${KERNEL_HINT}'. Use: consumer | geforce_v1 | geforce_v2" >&2
        exit 1
        ;;
esac

echo "=== Nsight Compute GEMM Profile ==="
echo "GPU:             ${GPU}"
echo "Kernel:          ${KERNEL_HINT} (${KERNEL_REGEX})"
echo "Output dir:      ${OUT}"
echo ""

##############################################################################
# Step 1 — Warmup: run 10s to complete sigma install + graph capture.
# This ensures the real profiling captures steady-state GEMM only.
##############################################################################
echo "[1/3] Warmup (10s bench — sigma install + graph capture)..."
cd "${ROOT}/build"
PROPMINER_BENCH_NO_GRAPH=0 \
    PROPMINER_BENCH_JSON=1 \
    PROPMINER_BATCH=1 \
    PROPMINER_GRAPH_BATCH=1 \
    PEARL_GEMM_CONSUMER_CLUSTER_M=1 \
    PEARL_GEMM_KERNEL="${KERNEL_HINT}" \
    ./propminer --bench 10 --rtx5090 --gpus "${GPU}" \
        > "${OUT}/warmup.log" 2>&1 || true

echo "  Warmup log: ${OUT}/warmup.log"

##############################################################################
# Step 2 — Profile: ncu captures steady-state matmul kernels.
#
# Using --set profile (not --set full) for a focused metric set.
# --format json produces parseable output.
# --launch-skip 0 captures all launches (warmup already done in step 1).
##############################################################################
echo "[2/3] Capturing via Nsight Compute (--set profile)..."

ncu --set profile \
    --format json \
    --target-processes all \
    --kernel-name-base demangled \
    --kernel-name "regex:${KERNEL_REGEX}" \
    --launch-skip 0 \
    --launch-count 10 \
    --import-source 0 \
    -f \
    -o "${OUT}/gemm_profile" \
    PROPMINER_BENCH_NO_GRAPH=1 \
    PROPMINER_BENCH_JSON=1 \
    PROPMINER_BATCH=1 \
    PROPMINER_GRAPH_BATCH=1 \
    PEARL_GEMM_CONSUMER_CLUSTER_M=1 \
    PEARL_GEMM_KERNEL="${KERNEL_HINT}" \
    ./propminer --bench 30 --rtx5090 --gpus "${GPU}" \
        2>&1 | tee "${OUT}/ncu_stdout.log" || true

echo "  ncu console output: ${OUT}/ncu_stdout.log"

##############################################################################
# Step 3 — Parse ncu JSON output and produce diagnostic summary.
#
# ncu --format json writes a .json file alongside the .ncu-rep.
# We extract key bottleneck metrics and print a readable report.
##############################################################################
echo "[3/3] Analyzing results..."

JSON_FILE="${OUT}/gemm_profile.json"

if [[ ! -f "${JSON_FILE}" ]]; then
    echo "WARNING: ${JSON_FILE} not found." >&2
    echo "  ncu may have captured no kernels, or the kernel regex did not match." >&2
    echo "  Check ${OUT}/ncu_stdout.log for errors." >&2
    echo "  Also try running: ncu --set profile --kernel-name 'regex:transcript_gemm' ..." >&2
    echo "  Available output files:" >&2
    ls -lh "${OUT}/" >&2
    exit 1
fi

SUMMARY="${OUT}/analysis.txt"

# Helper: extract a metric value from ncu JSON.
# ncu JSON structure: apps[].metrics[].value
# We use Python if available, otherwise fall back to basic grep.
extract_json_value() {
    local key="$1"
    local default="${2:-N/A}"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json, sys
try:
    with open('${JSON_FILE}') as f:
        data = json.load(f)
    # Walk apps -> sections (or metrics) -> find key
    apps = data.get('applications', [])
    if not apps:
        apps = data.get('apps', [])
    for app in apps:
        for section in app.get('metrics', []):
            if section.get('metric', '') == '${key}' or section.get('name', '') == '${key}':
                print(section.get('value', '${default}'))
                sys.exit(0)
        for k, v in app.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        if item.get('metric', '') == '${key}':
                            print(item.get('value', '${default}'))
                            sys.exit(0)
    # Fallback: search entire JSON for the key
    text = json.dumps(data)
    idx = text.find('\"${key}\"')
    if idx >= 0:
        val_start = text.find(':', idx) + 1
        val_end = text.find(',', val_start)
        if val_end == -1: val_end = text.find('}', val_start)
        val = text[val_start:val_end].strip().strip('\"')
        print(val)
    else:
        print('${default}')
except Exception as e:
    print('${default}')
" 2>/dev/null || echo "${default}"
    else
        # Basic extraction without Python
        grep -o "\"${key}\"[[:space:]]*:[[:space:]]*[^,}]*" "${JSON_FILE}" | head -1 | sed "s/.*:[[:space:]]*//" | tr -d '"' || echo "${default}"
    fi
}

# Try to extract key metrics; many may not exist on sm_120a, so we handle missing gracefully.
TENSOR_ACTIVE="$(extract_json_value 'sm__pipe_tensor_cycles_active.sum_pct')"
WARPS_ACTIVE="$(extract_json_value 'sm__warps_active.avg.pct_of_peak_sustained_active')"
SMEM_LD="$(extract_json_value 'l1tex__t_sectors_pipe_lsu_mem_shared_op_shared_local_ld.sum')"
GEMM_ACTIVE="$(extract_json_value 'smsp__sector_pipe_tensor_cycle_sum')"
CYCLE_ELAPSED="$(extract_json_value 'smsp__cycle_elapsed.sum')"

# Also extract from ncu stdout which has a table of metrics.
parse_ncu_stdout_metric() {
    grep -i "$1" "${OUT}/ncu_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.?[0-9]*%?' | tail -1 || echo "N/A"
}

# Fallback: parse metrics from ncu stdout if JSON extraction failed.
if [[ "${TENSOR_ACTIVE}" == "N/A" ]] || [[ -z "${TENSOR_ACTIVE}" ]]; then
    TENSOR_ACTIVE="$(parse_ncu_stdout_metric 'pipe_tensor_cycles_active')"
fi
if [[ "${WARPS_ACTIVE}" == "N/A" ]] || [[ -z "${WARPS_ACTIVE}" ]]; then
    WARPS_ACTIVE="$(parse_ncu_stdout_metric 'warps_active')"
fi

# Write the diagnostic summary.
cat > "${SUMMARY}" << EOF
=============================================================================
  RTX 5090 GEMM Bottleneck Analysis
  Kernel: ${KERNEL_HINT}
  GPU:    ${GPU}
  Date:   $(date)
=============================================================================

KEY METRICS EXTRACTED:
  Tensor core utilization (sm__pipe_tensor_cycles_active): ${TENSOR_ACTIVE}
  Active warps (sm__warps_active):                         ${WARPS_ACTIVE}
  SMEM load sectors (l1tex shared load):                  ${SMEM_LD}
  GEMM cycles active (smsp__sector_pipe_tensor_cycle):    ${GEMM_ACTIVE}
  Total SM cycles (smsp__cycle_elapsed):                  ${CYCLE_ELAPSED}

DIAGNOSTIC FLOW:
  Read the extracted metrics above, then follow the matching branch:

  ┌───────────────────────────────────────────────────────────────────┐
  │ If Tensor Core Utilization < 50%                                 │
  │   → MAINLOOP STARVATION                                          │
  │   The TMA pipeline isn't feeding fast enough,                     │
  │   or K-tile boundaries create idle cycles.                       │
  │                                                                   │
  │   Try: More pipeline STAGES (3 instead of 2),                    │
  │        Larger KBLOCK (128 instead of 64),                       │
  │        Different SWIZZLE_BITS (3 instead of 2).                 │
  │        Run tune_blackwell_knobs.sh to find the winner.          │
  │                                                                   │
  │   geforce_v2 already has warp-specialized TMA,                   │
  │   so if you're profiling geforce_v2 and still <50%,              │
  │   the issue is likely ISA-level: SM80 mma.sync simply can't      │
  │   saturate sm_120a tensor cores. No kernel tweak fixes this.     │
  └───────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────┐
  │ If Active Warps < 50%                                            │
  │   → REGISTER PRESSURE / OCCUPANCY                                │
  │   128 int32 accumulators + 16 transcript u32 per thread           │
  │   caps at ~1 block/SM. SMs are half-empty.                       │
  │                                                                   │
  │   Try: Increase MIN_BLOCKS from 1 → 2,                           │
  │        but be aware the 128x256x128 tile is proof-locked.       │
  │                                                                   │
  │   Real fix requires reducing accumulator count,                   │
  │   which changes partition_C and breaks proof consensus.          │
  │   Not possible without protocol change.                           │
  └───────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────┐
  │ If BOTH Tensor > 70% AND Warps > 70%                             │
  │   → ISA LIMITATION (fundamental ceiling)                         │
  │   SM80 mma.sync on sm_120a can't reach beyond ~40% of            │
  │   rated INT8 peak. This is the 300 TMAD/s wall.                  │
  │                                                                   │
  │   No kernel rewrite on 5090 changes this.                        │
  │   Only tcgen05 (B200/datacenter) gives the next leap.            │
  │   Realistic path: add more GPUs, not rewrite kernels.            │
  └───────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────┐
  │ If SMEM load sectors near peak                                   │
  │   → SHARED MEMORY BANK CONFLICTS                                 │
  │   The swizzle pattern isn't avoiding bank conflicts.             │
  │                                                                   │
  │   Try: Different SWIZZLE_BITS (sweep 2 vs 3),                    │
  │        Different KBLOCK (sweep 64 vs 128).                      │
  │        Run tune_blackwell_knobs.sh.                              │
  └───────────────────────────────────────────────────────────────────┘

NEXT STEPS:
  1. Read the metrics above
  2. Follow the matching diagnostic branch
  3. If unsure, open the full ncu report on the GPU box:
       ncu --report-export ${OUT}/gemm_profile.ncu-rep
  4. Compare: run this script with both 'consumer' and 'geforce_v2'
     to see what the geforce kernel actually improved.

RAW FILES:
  - gemm_profile.json        Full ncu JSON (all metrics)
  - gemm_profile.ncu-rep     Nsight Compute GUI report
  - ncu_stdout.log           Console output from ncu run
  - warmup.log               Warmup run log
EOF

echo ""
echo "=========================================="
echo "  Profile analysis complete"
echo "=========================================="
echo ""
cat "${SUMMARY}"
echo ""
echo "Output dir: ${OUT}/"
ls -lh "${OUT}/" 2>/dev/null | grep -v "^total" || true
