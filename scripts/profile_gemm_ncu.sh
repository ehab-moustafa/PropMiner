#!/usr/bin/env bash
# Comprehensive Nsight Compute + System Profiling for Pearl GEMM on RTX 5090.
# Captures GPU kernel metrics, system telemetry, temperatures, power, memory,
# CPU utilization, and produces a diagnostic dashboard with actionable recommendations.
#
# Usage:
#   ./scripts/profile_gemm_ncu.sh [gpu_index] [consumer|geforce_v1|geforce_v2]
#
# Prerequisites:
#   - CUDA Toolkit with Nsight Compute (ncu) installed
#   - Built propminer binary at build/propminer
#   - nvidia-smi (included with CUDA)
#   - python3 (for JSON parsing)

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

# Helper: safe numeric extraction with fallback
safe_num() {
    local val="$1"
    local default="${2:-0.0}"
    if [[ "${val}" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "${val}"
    else
        echo "${default}"
    fi
}

# Helper: color codes (disabled in CI/pipe)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

colorize_pct() {
    local pct="$1"
    local threshold_low="${2:-50}"
    local threshold_high="${3:-80}"
    if (( $(echo "${pct} < ${threshold_low}" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${RED}${pct}%${NC}"
    elif (( $(echo "${pct} >= ${threshold_high}" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${GREEN}${pct}%${NC}"
    else
        echo -e "${YELLOW}${pct}%${NC}"
    fi
}

echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║          RTX 5090 Comprehensive GEMM Profiling Dashboard             ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "GPU:             ${GPU}"
echo "Kernel:          ${KERNEL_HINT} (${KERNEL_REGEX})"
echo "Output dir:      ${OUT}"
echo "Date:            $(date)"
echo ""

##############################################################################
# Step 0 — Baseline system telemetry (before mining starts)
##############################################################################
echo "[0/5] Capturing baseline system state..."

# GPU baseline via nvidia-smi
GPU_BASELINE=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,memory.used,memory.total,clocks.current.graphics,clocks.max.graphics --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || echo "N/A")
if [[ "${GPU_BASELINE}" != "N/A" ]]; then
    IFS=',' read -r BASELINE_GPU_UTIL BASELINE_MEM_UTIL BASELINE_TEMP BASELINE_PWR BASELINE_PWR_LIMIT BASELINEMEM_USED BASELINE_MEM_TOTAL BASELINE_CUR_GPU_CLOCK BASELINE_MAX_GPU_CLOCK <<< "${GPU_BASELINE}"
    BASELINE_GPU_UTIL=$(safe_num "${BASELINE_GPU_UTIL}" 0)
    BASELINE_MEM_UTIL=$(safe_num "${BASELINE_MEM_UTIL}" 0)
    BASELINE_TEMP=$(safe_num "${BASELINE_TEMP}" 0)
    BASELINE_PWR=$(safe_num "${BASELINE_PWR}" 0)
    BASELINE_PWR_LIMIT=$(safe_num "${BASELINE_PWR_LIMIT}" 575)
    BASELINE_MEM_USED=$(safe_num "${BASELINE_MEM_USED}" 0)
    BASELINE_MEM_TOTAL=$(safe_num "${BASELINE_MEM_TOTAL}" 32768)
    BASELINE_CUR_GPU_CLOCK=$(safe_num "${BASELINE_CUR_GPU_CLOCK}" 0)
    BASELINE_MAX_GPU_CLOCK=$(safe_num "${BASELINE_MAX_GPU_CLOCK}" 2407)
else
    BASELINE_GPU_UTIL=0; BASELINE_MEM_UTIL=0; BASELINE_TEMP=0
    BASELINE_PWR=0; BASELINE_PWR_LIMIT=575; BASELINE_MEM_USED=0
    BASELINE_MEM_TOTAL=32768; BASELINE_CUR_GPU_CLOCK=0; BASELINE_MAX_GPU_CLOCK=2407
fi

# CPU baseline
CPU_COUNT=$(nproc 2>/dev/null || echo 1)
CPU_BASELINE=$(top -bn1 2>/dev/null | grep "Cpu(s)" | awk '{print $2}' || echo "0")
CPU_BASELINE=$(safe_num "${CPU_BASELINE}" 0)

# RAM baseline
RAM_LINE=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2, $3, $7}' || echo "0 0 0")
IIFS=' ' read -r RAM_TOTAL RAM_USED RAM_FREE <<< "${RAM_LINE}"
RAM_TOTAL=$(safe_num "${RAM_TOTAL}" 0)
RAM_USED=$(safe_num "${RAM_USED}" 0)
RAM_FREE=$(safe_num "${RAM_FREE}" 0)

# GPU model
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null || echo "Unknown")

echo "  GPU: ${GPU_MODEL} (${BASELINE_CUR_GPU_CLOCK} MHz / ${BASELINE_MAX_GPU_CLOCK} MHz max)"
echo "  CPU cores: ${CPU_COUNT}  |  RAM: ${RAM_TOTAL}MB total"

##############################################################################
# Step 1 — Warmup: run 10s to complete sigma install + graph capture.
# This ensures the real profiling captures steady-state GEMM only.
##############################################################################
echo ""
echo "[1/5] Warmup (10s bench — sigma install + graph capture)..."
cd "${ROOT}/build"
PROPMINER_BENCH_NO_GRAPH=0 \
    PROPMINER_BENCH_JSON=1 \
    PROPMINER_BATCH=1 \
    PROPMINER_GRAPH_BATCH=1 \
    PEARL_GEMM_CONSUMER_CLUSTER_M=1 \
    PEARL_GEMM_KERNEL="${KERNEL_HINT}" \
    ./propminer --bench 10 --rtx5090 --gpus "${GPU}" \
        > "${OUT}/warmup.log" 2>&1 || true

# Capture warmup TMAD/s
WARMUP_TMAD=$(grep -o '"tmad_per_sec":[0-9.]*' "${OUT}/warmup.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || echo "0")
WARMUP_TMAD=$(safe_num "${WARMUP_TMAD}" 0)
WARMUP_MATMUL=$(grep -o '"iters_per_sec":[0-9.]*' "${OUT}/warmup.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || echo "0")
WARMUP_MATMUL=$(safe_num "${WARMUP_MATMUL}" 0)

echo "  Warmup TMAD/s: ${WARMUP_TMAD}  |  matmul/s: ${WARMUP_MATMUL}"

# Extract GPU metrics during warmup
WARMUP_GPU_METRICS=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,memory.used,memory.total --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || echo "N/A")
if [[ "${WARMUP_GPU_METRICS}" != "N/A" ]]; then
    IFS=',' read -r WARMUP_GPU_UTIL WARMUP_MEM_UTIL WARMUP_TEMP WARMUP_PWR WARMUP_PWR_LIMIT WARMUP_MEM_USED WARMUP_MEM_TOTAL <<< "${WARMUP_GPU_METRICS}"
    WARMUP_GPU_UTIL=$(safe_num "${WARMUP_GPU_UTIL}" 0)
    WARMUP_MEM_UTIL=$(safe_num "${WARMUP_MEM_UTIL}" 0)
    WARMUP_TEMP=$(safe_num "${WARMUP_TEMP}" 0)
    WARMUP_PWR=$(safe_num "${WARMUP_PWR}" 0)
    WARMUP_PWR_LIMIT=$(safe_num "${WARMUP_PWR_LIMIT}" 575)
    WARMUP_MEM_USED=$(safe_num "${WARMUP_MEM_USED}" 0)
    WARMUP_MEM_TOTAL=$(safe_num "${WARMUP_MEM_TOTAL}" 32768)
else
    WARMUP_GPU_UTIL=0; WARMUP_MEM_UTIL=0; WARMUP_TEMP=0
    WARMUP_PWR=0; WARMUP_PWR_LIMIT=575; WARMUP_MEM_USED=0; WARMUP_MEM_TOTAL=32768
fi

echo "  GPU util: ${WARMUP_GPU_UTIL}%  |  Temp: ${WARMUP_TEMP}°C  |  Power: ${WARMUP_PWR}/${WARMUP_PWR_LIMIT}W  |  VRAM: ${WARMUP_MEM_USED}/${WARMUP_MEM_TOTAL}MB"

##############################################################################
# Step 2 — Profile: ncu captures steady-state matmul kernels.
##############################################################################
echo ""
echo "[2/5] Capturing via Nsight Compute (--set profile)..."

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

##############################################################################
# Step 3 — Capture peak system telemetry during profiling
##############################################################################
echo "[3/5] Capturing peak system telemetry..."

# Sample nvidia-smi every 5s during the 30s ncu run to get peak values
PEAK_GPU_UTIL=0; PEAK_MEM_UTIL=0; PEAK_TEMP=0; PEAK_PWR=0; PEAK_MEM_USED=0
SAMPLES=6

for i in $(seq 1 ${SAMPLES}); do
    SAMPLE=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || echo "0,0,0,0,0")
    IFS=',' read -r S_GPU_UTIL S_MEM_UTIL S_TEMP S_PWR S_MEM_USED <<< "${SAMPLE}"
    S_GPU_UTIL=$(safe_num "${S_GPU_UTIL}" 0)
    S_MEM_UTIL=$(safe_num "${S_MEM_UTIL}" 0)
    S_TEMP=$(safe_num "${S_TEMP}" 0)
    S_PWR=$(safe_num "${S_PWR}" 0)
    S_MEM_USED=$(safe_num "${S_MEM_USED}" 0)

    # Track peaks
    if (( $(echo "${S_GPU_UTIL} > ${PEAK_GPU_UTIL}" | bc -l 2>/dev/null || echo 0) )); then PEAK_GPU_UTIL=${S_GPU_UTIL}; fi
    if (( $(echo "${S_MEM_UTIL} > ${PEAK_MEM_UTIL}" | bc -l 2>/dev/null || echo 0) )); then PEAK_MEM_UTIL=${S_MEM_UTIL}; fi
    if (( $(echo "${S_TEMP} > ${PEAK_TEMP}" | bc -l 2>/dev/null || echo 0) )); then PEAK_TEMP=${S_TEMP}; fi
    if (( $(echo "${S_PWR} > ${PEAK_PWR}" | bc -l 2>/dev/null || echo 0) )); then PEAK_PWR=${S_PWR}; fi
    if (( $(echo "${S_MEM_USED} > ${PEAK_MEM_USED}" | bc -l 2>/dev/null || echo 0) )); then PEAK_MEM_USED=${S_MEM_USED}; fi

    sleep 5
done

# Capture final state
FINAL_METRICS=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,memory.used,memory.total,clocks.current.graphics,clocks.max.graphics --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null || echo "N/A")
if [[ "${FINAL_METRICS}" != "N/A" ]]; then
    IFS=',' read -r FINAL_GPU_UTIL FINAL_MEM_UTIL FINAL_TEMP FINAL_PWR FINAL_PWR_LIMIT FINAL_MEM_USED FINAL_MEM_TOTAL FINAL_CUR_CLOCK FINAL_MAX_CLOCK <<< "${FINAL_METRICS}"
    FINAL_GPU_UTIL=$(safe_num "${FINAL_GPU_UTIL}" 0)
    FINAL_MEM_UTIL=$(safe_num "${FINAL_MEM_UTIL}" 0)
    FINAL_TEMP=$(safe_num "${FINAL_TEMP}" 0)
    FINAL_PWR=$(safe_num "${FINAL_PWR}" 0)
    FINAL_PWR_LIMIT=$(safe_num "${FINAL_PWR_LIMIT}" 575)
    FINAL_MEM_USED=$(safe_num "${FINAL_MEM_USED}" 0)
    FINAL_MEM_TOTAL=$(safe_num "${FINAL_MEM_TOTAL}" 32768)
    FINAL_CUR_CLOCK=$(safe_num "${FINAL_CUR_CLOCK}" 0)
    FINAL_MAX_CLOCK=$(safe_num "${FINAL_MAX_CLOCK}" 2407)
else
    FINAL_GPU_UTIL=0; FINAL_MEM_UTIL=0; FINAL_TEMP=0; FINAL_PWR=0
    FINAL_PWR_LIMIT=575; FINAL_MEM_USED=0; FINAL_MEM_TOTAL=32768
    FINAL_CUR_CLOCK=0; FINAL_MAX_CLOCK=2407
fi

# CPU during profiling
CPU_DURING=$(top -bn1 2>/dev/null | grep "Cpu(s)" | awk '{sum+=$2+$4; print sum}' || echo "0")
CPU_DURING=$(safe_num "${CPU_DURING}" 0)

# RAM during profiling
RAM_DURING_LINE=$(free -m 2>/dev/null | awk '/^Mem:/ {print $2, $3, $7}' || echo "0 0 0")
IIFS=' ' read -r RAM_DURING_TOTAL RAM_DURING_USED RAM_DURING_FREE <<< "${RAM_DURING_LINE}"
RAM_DURING_TOTAL=$(safe_num "${RAM_DURING_TOTAL}" 0)
RAM_DURING_USED=$(safe_num "${RAM_DURING_USED}" 0)
RAM_DURING_FREE=$(safe_num "${RAM_DURING_FREE}" 0)

echo "  Peak GPU util: ${PEAK_GPU_UTIL}%  |  Peak temp: ${PEAK_TEMP}°C  |  Peak power: ${PEAK_PWR}W"

##############################################################################
# Step 4 — Parse ncu JSON output for kernel-level metrics
##############################################################################
echo "[4/5] Extracting Nsight Compute kernel metrics..."

JSON_FILE="${OUT}/gemm_profile.json"

# Extract ncu metrics from JSON or fallback to stdout
extract_ncu_metric() {
    local key="$1"
    local default="${2:-N/A}"
    if [[ -f "${JSON_FILE}" ]] && command -v python3 >/dev/null 2>&1; then
        python3 -c "
import json, sys
try:
    with open('${JSON_FILE}') as f:
        data = json.load(f)
    # Search recursively for the metric
    def find_metrics(obj):
        if isinstance(obj, dict):
            if 'metric' in obj and obj['metric'] == '${key}':
                val = obj.get('value', 'N/A')
                print(val)
                sys.exit(0)
            for v in obj.values():
                find_metrics(v)
        elif isinstance(obj, list):
            for item in obj:
                find_metrics(item)
    find_metrics(data)
    print('${default}')
except:
    print('${default}')
" 2>/dev/null || echo "${default}"
    else
        # Fallback: grep from ncu stdout
        grep -i "${key}" "${OUT}/ncu_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+\.?[0-9]*%' | tail -1 || echo "${default}"
    fi
}

# Extract comprehensive ncu metrics
NCU_TENSOR_UTIL=$(extract_ncu_metric 'sm__pipe_tensor_cycles_active.sum_pct' 'N/A')
NCU_WARPS_ACTIVE=$(extract_ncu_metric 'sm__warps_active.avg.pct_of_peak_sustained_active' 'N/A')
NCU_SMEM_LD=$(extract_ncu_metric 'l1tex__t_sectors_pipe_lsu_mem_shared_op_shared_local_ld.sum' 'N/A')
NCU_SMEM_ST=$(extract_ncu_metric 'l1tex__t_sectors_pipe_lsu_mem_shared_op_shared_local_st.sum' 'N/A')
NCU_L1_HIT=$(extract_ncu_metric 'l1tex__t_sectors_pipe_lsu_mem_shared_op_l1_access.sum' 'N/A')
NCU_L2_HIT=$(extract_ncu_metric 'lts__t_sectors_pipe_lsu_mem_global_op_ld.sum' 'N/A')
NCU_TMA_ACTIVE=$(extract_ncu_metric 'tma__pipe_cycles_active.sum_pct' 'N/A')
NCU_REG_USED=$(extract_ncu_metric 'smsp__inst_executed.sum' 'N/A')
NCU_OCCUPANCY=$(extract_ncu_metric 'smsp__throughput.avg.pct_of_peak_sustained_active' 'N/A')
NCU_CYCLES=$(extract_ncu_metric 'smsp__cycle_elapsed.sum' 'N/A')
NCU_INST_PER_CYCLE=$(extract_ncu_metric 'smsp__inst_executed.avg.per_cycle_elapsed' 'N/A')
NCU_MEM_THROUGHPUT=$(extract_ncu_metric 'lts__t_sectors_pipe_lsu_mem_global_op_ld.sum' 'N/A')
NCU_FP32_ACTIVE=$(extract_ncu_metric 'sm__pipe_fma_cycles_active.sum_pct' 'N/A')
NCU_INT8_ACTIVE=$(extract_ncu_metric 'sm__pipe_tensor_cycles_active.sum_pct' 'N/A')

# Also try to get TMAD/s from the bench output
BENCH_TMAD=$(grep -o '"tmad_per_sec":[0-9.]*' "${OUT}/ncu_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || echo "0")
BENCH_TMAD=$(safe_num "${BENCH_TMAD}" 0)
BENCH_MATMUL=$(grep -o '"iters_per_sec":[0-9.]*' "${OUT}/ncu_stdout.log" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || echo "0")
BENCH_MATMUL=$(safe_num "${BENCH_MATMUL}" 0)

##############################################################################
# Step 5 — Generate comprehensive diagnostic dashboard
##############################################################################
echo "[5/5] Generating diagnostic dashboard..."

# Calculate derived metrics
PEAK_GPU_UTIL_PCT=$(safe_num "${PEAK_GPU_UTIL}" 0)
PEAK_TEMP_PCT=$(echo "scale=1; ${PEAK_TEMP} * 100 / 100" | bc -l 2>/dev/null || echo "0")
VRAM_USED_PCT=$(echo "scale=1; ${PEAK_MEM_USED} * 100 / ${FINAL_MEM_TOTAL}" | bc -l 2>/dev/null || echo "0")
RAM_USED_PCT=$(echo "scale=1; ${RAM_DURING_USED} * 100 / ${RAM_DURING_TOTAL}" | bc -l 2>/dev/null || echo "0")
POWER_PCT=$(echo "scale=1; ${PEAK_PWR} * 100 / ${FINAL_PWR_LIMIT}" | bc -l 2>/dev/null || echo "0")

# GPU efficiency ratio
if [[ "${BENCH_TMAD}" != "0" ]] && (( $(echo "${FINAL_MAX_CLOCK} > 0" | bc -l 2>/dev/null || echo 0) )); then
    EFFICIENCY=$(echo "scale=2; ${BENCH_TMAD} / ${FINAL_MAX_GPU_CLOCK} * 100" | bc -l 2>/dev/null || echo "0")
else
    EFFICIENCY="N/A"
fi

# Generate the comprehensive report
cat > "${OUT}/analysis.txt" << 'HEADER'
╔════════════════════════════════════════════════════════════════════════╗
║           RTX 5090 Comprehensive Profiling Dashboard                  ║
╚════════════════════════════════════════════════════════════════════════╝

HEADER

cat >> "${OUT}/analysis.txt" << EOF
┌─────────────────────────────────────────────────────────────────────────┐
│  SYSTEM OVERVIEW                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  GPU Model:      ${GPU_MODEL}
│  GPU Index:      ${GPU}
│  Kernel Path:    ${KERNEL_HINT}
│  Architecture:   sm_120a (Blackwell GeForce)
│  Max GPU Clock:  ${FINAL_MAX_CLOCK} MHz
│  CPU Cores:      ${CPU_COUNT}
│  Total RAM:      ${RAM_TOTAL} MB
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  PERFORMANCE METRICS                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Warmup TMAD/s:      ${WARMUP_TMAD}
│  Peak TMAD/s:        ${BENCH_TMAD}
│  Warmup matmul/s:    ${WARMUP_MATMUL}
│  Peak matmul/s:      ${BENCH_MATMUL}
│  TMAD Efficiency:    ${EFFICIENCY}% of max clock (higher = better)
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  GPU TELEMETRY (Peak During Profiling)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  GPU Utilization:    ${PEAK_GPU_UTIL}%  $(colorize_pct "${PEAK_GPU_UTIL}" 50 90)
│  Memory Utilization: ${PEAK_MEM_UTIL}%  $(colorize_pct "${PEAK_MEM_UTIL}" 50 90)
│  Temperature:        ${PEAK_TEMP}°C  $(colorize_pct "${PEAK_TEMP}" 75 85)
│  Power Draw:         ${PEAK_PWR}W / ${FINAL_PWR_LIMIT}W (${POWER_PCT}%)  $(colorize_pct "${POWER_PCT}" 50 90)
│  VRAM Used:          ${PEAK_MEM_USED}MB / ${FINAL_MEM_TOTAL}MB (${VRAM_USED_PCT}%)
│  Current Clock:      ${FINAL_CUR_CLOCK} MHz / ${FINAL_MAX_CLOCK} MHz
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  CPU & RAM TELEMETRY                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  CPU Usage:          ${CPU_DURING}% (of ${CPU_COUNT} cores)
│  RAM Used:           ${RAM_DURING_USED}MB / ${RAM_DURING_TOTAL}MB (${RAM_USED_PCT}%)
│  RAM Free:           ${RAM_DURING_FREE}MB
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  KERNEL-LEVEL METRICS (Nsight Compute)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Tensor Core Util:   ${NCU_TENSOR_UTIL}  ← Target: >70%
│  Active Warps:       ${NCU_WARPS_ACTIVE}  ← Target: >60%
│  SMEM Load Sectors:  ${NCU_SMEM_LD}
│  SMEM Store Sectors: ${NCU_SMEM_ST}
│  L1 Access Sectors:  ${NCU_L1_HIT}
│  L2 Global Load:     ${NCU_L2_HIT}
│  TMA Pipe Active:    ${NCU_TMA_ACTIVE}  ← Target: >60%
│  FP32 FMA Active:    ${NCU_FP32_ACTIVE}
│  INT8 Tensor Active: ${NCU_INT8_ACTIVE}
│  Occupancy:          ${NCU_OCCUPANCY}
│  Instructions/Cycle: ${NCU_INST_PER_CYCLE}
│  Total SM Cycles:    ${NCU_CYCLES}
└─────────────────────────────────────────────────────────────────────────┘

EOF

# Add bottleneck diagnosis
cat >> "${OUT}/analysis.txt" << 'DIAG_HEADER'
┌─────────────────────────────────────────────────────────────────────────┐
│  BOTTLENECK DIAGNOSIS                                                   │
├─────────────────────────────────────────────────────────────────────────┤
DIAG_HEADER

# Determine primary bottleneck
BOTTLENECK=""
RECOMMENDATION=""

# Check tensor utilization
if [[ "${NCU_TENSOR_UTIL}" != "N/A" ]]; then
    TENSOR_NUM=$(echo "${NCU_TENSOR_UTIL}" | grep -oE '[0-9.]+' || echo "0")
    if (( $(echo "${TENSOR_NUM} < 50" | bc -l 2>/dev/null || echo 0) )); then
        BOTTLENECK="MAINLOOP STARVATION"
        RECOMMENDATION="TMA pipeline not feeding tensor cores fast enough. Try: STAGES=3, KBLOCK=128, SWIZZLE=3. Run tune_blackwell_knobs.sh."
    elif (( $(echo "${TENSOR_NUM} > 70" | bc -l 2>/dev/null || echo 0) )); then
        BOTTLENECK="ISA LIMITATION"
        RECOMMENDATION="SM80 mma.sync on sm_120a is the bottleneck. No kernel tweak will help. Consider tcgen05 (B200) or scaling GPUs."
    else
        BOTTLENECK="MODERATE UTILIZATION"
        RECOMMENDATION="Tensor cores at ${TENSOR_NUM}%. Tuning could help. Try different STAGES/KBLOCK/SWIZZLE combinations."
    fi
else
    BOTTLENECK="METRICS UNAVAILABLE"
    RECOMMENDATION="ncu could not extract tensor metrics. Check ${OUT}/ncu_stdout.log for errors."
fi

# Check warp occupancy
if [[ "${NCU_WARPS_ACTIVE}" != "N/A" ]]; then
    WARPS_NUM=$(echo "${NCU_WARPS_ACTIVE}" | grep -oE '[0-9.]+' || echo "0")
    if (( $(echo "${WARPS_NUM} < 50" | bc -l 2>/dev/null || echo 0) )); then
        if [[ -n "${BOTTLENECK}" ]]; then
            BOTTLENECK="${BOTTLENECK} + REGISTER PRESSURE"
        else
            BOTTLENECK="REGISTER PRESSURE"
        fi
        RECOMMENDATION="${RECOMMENDATION} Also: 128 int32 accumulators/thread caps occupancy. MIN_BLOCKS=2 may help but tile is proof-locked."
    fi
fi

# Check temperature
if (( $(echo "${PEAK_TEMP} > 85" | bc -l 2>/dev/null || echo 0) )); then
    BOTTLENECK="${BOTTLENECK} + THERMAL THROTTLING"
    RECOMMENDATION="${RECOMMENDATION} GPU temp ${PEAK_TEMP}°C is high. Improve cooling, increase fan speed, or reduce power limit."
fi

# Check power limit
if (( $(echo "${PEAK_PWR} >= ${FINAL_PWR_LIMIT} * 0.95" | bc -l 2>/dev/null || echo 0) )); then
    BOTTLENECK="${BOTTLENECK} + POWER LIMITED"
    RECOMMENDATION="${RECOMMENDATION} GPU is power-limited at ${PEAK_PWR}W. Increase power limit or improve cooling to allow higher boost."
fi

# Check VRAM
if (( $(echo "${VRAM_USED_PCT} > 90" | bc -l 2>/dev/null || echo 0) )); then
    BOTTLENECK="${BOTTLENECK} + VRAM PRESSURE"
    RECOMMENDATION="${RECOMMENDATION} VRAM at ${VRAM_USED_PCT}%. Consider reducing N or batch size."
fi

# Write bottleneck section
cat >> "${OUT}/analysis.txt" << EOF

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  PRIMARY BOTTLENECK: ${BOTTLENECK}
  ╚═══════════════════════════════════════════════════════════════════════╝

  RECOMMENDATION:
    ${RECOMMENDATION}

EOF

# Add actionable next steps
cat >> "${OUT}/analysis.txt" << 'ACTIONS'
┌─────────────────────────────────────────────────────────────────────────┐
│  ACTIONABLE NEXT STEPS                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Compare kernels:                                                   │
│     ./scripts/profile_gemm_ncu.sh 0 consumer                           │
│     ./scripts/profile_gemm_ncu.sh 0 geforce_v2                         │
│     ./scripts/profile_gemm_ncu.sh 0 geforce_v1                         │
│                                                                        │
│  2. Tune kernel knobs (if bottleneck is mainloop/SMEM):                │
│     ./scripts/tune_blackwell_knobs.sh                                  │
│                                                                        │
│  3. Run full benchmark with current config:                            │
│     ./scripts/build_and_benchmark.sh 300                               │
│                                                                        │
│  4. Open full ncu report (on GPU host):                                │
│     ncu --report-export <output_dir>/gemm_profile.ncu-rep              │
│                                                                        │
│  5. If ISA limitation confirmed (tensor >70% + warps >70%):            │
│     - Add more GPUs (linear scale)                                     │
│     - Upgrade to B200/datacenter for tcgen05                           │
│     - No kernel rewrite on 5090 will exceed ~40% of rated INT8 peak    │
└─────────────────────────────────────────────────────────────────────────┘

EOF

# Add raw files listing
cat >> "${OUT}/analysis.txt" << EOF
┌─────────────────────────────────────────────────────────────────────────┐
│  RAW OUTPUT FILES                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│  gemm_profile.json    Nsight Compute JSON (all kernel metrics)         │
│  gemm_profile.ncu-rep Nsight Compute GUI report                        │
│  ncu_stdout.log       Console output from ncu run                      │
│  warmup.log           Warmup run log (sigma install + graph capture)   │
│  analysis.txt         This file                                        │
└─────────────────────────────────────────────────────────────────────────┘

OUTPUT DIR: ${OUT}/
DATE: $(date)
EOF

echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║                    PROFILING COMPLETE                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
cat "${OUT}/analysis.txt"
echo ""
echo "Output dir: ${OUT}/"
ls -lh "${OUT}/" 2>/dev/null | grep -v "^total" || true
