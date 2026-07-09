#!/usr/bin/env bash
# Master step-by-step debugger: GeForce v2 + CUDA graphs on RTX 5090 (vast.ai).
#
# Run on the GPU host after cloning/building PropMiner:
#   ./scripts/debug_geforce_v2.sh              # full pipeline (phases 0–5)
#   ./scripts/debug_geforce_v2.sh --quick      # skip propminer self-tests (phases 0–4)
#   ./scripts/debug_geforce_v2.sh --phase 4    # single phase only
#   PEARL_GEMM_DEBUG=1 ./scripts/debug_geforce_v2.sh
#
# Each phase prints PASS / FAIL / SKIP and a "next step" hint on failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROPMINER="${PROPMINER:-${ROOT}/build/propminer}"
export PROPMINER_N_CAP="${PROPMINER_N_CAP:-32768}"
export PEARL_GEMM_CONSUMER_CLUSTER_M="${PEARL_GEMM_CONSUMER_CLUSTER_M:-1}"

QUICK=0
PHASE_ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) QUICK=1; shift ;;
    --phase) PHASE_ONLY="${2:-}"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1 (try --help)" >&2; exit 2 ;;
  esac
done

# shellcheck disable=SC2034
declare -a STEP_NAMES=(
  "Environment snapshot"
  "CPU host tests (no GPU)"
  "Workspace sizing fix present in source"
  "Transcript byte-identity (consumer vs v2)"
  "Minimal CUDA graph harness (consumer vs v2)"
  "PropMiner GPU self-test isolation matrix"
)
TOTAL_STEPS="${#STEP_NAMES[@]}"

RED=$'\033[31m'
GRN=$'\033[32m'
YLW=$'\033[33m'
RST=$'\033[0m'

step_banner() {
  local n="$1"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  printf "STEP %d/%d: %s\n" "$n" "$TOTAL_STEPS" "${STEP_NAMES[$((n - 1))]}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

pass() { printf "${GRN}PASS${RST}: %s\n" "$1"; }
fail() { printf "${RED}FAIL${RST}: %s\n" "$1"; }
skip() { printf "${YLW}SKIP${RST}: %s\n" "$1"; }
hint() { printf "${YLW}→ next:${RST} %s\n" "$1"; }

should_run() {
  local n="$1"
  if [[ -n "${PHASE_ONLY}" ]]; then
    [[ "${PHASE_ONLY}" == "${n}" ]]
  elif [[ "${QUICK}" -eq 1 && "${n}" -ge 5 ]]; then
    return 1
  else
    return 0
  fi
}

failures=0
record_fail() {
  failures=$((failures + 1))
}

# ── Phase 0: environment ────────────────────────────────────────────────────
phase0() {
  step_banner 1
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found — run on RTX 5090 host (vast.ai)"
    hint "Rent Blackwell instance; install CUDA 12.8+ toolkit"
    record_fail
    return
  fi

  echo "GPU:"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total \
    --format=csv,noheader | head -1
  echo ""
  echo "CUDA toolkit:"
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -1
  else
    echo "  nvcc not in PATH (build may still work if prebuilt)"
  fi
  echo ""
  echo "Git:"
  if git -C "${ROOT}" rev-parse --short HEAD >/dev/null 2>&1; then
    echo "  branch=$(git -C "${ROOT}" branch --show-current 2>/dev/null || echo '?')"
    echo "  sha=$(git -C "${ROOT}" rev-parse --short HEAD)"
  else
    echo "  (not a git checkout)"
  fi
  echo ""
  echo "Build:"
  if [[ -x "${PROPMINER}" ]]; then
    echo "  propminer=${PROPMINER} (executable)"
    ls -la "${PROPMINER}"
  else
    fail "propminer not built at ${PROPMINER}"
    hint "cmake -B build && cmake --build build --target propminer"
    record_fail
    return
  fi
  echo ""
  echo "Relevant env (defaults shown if unset):"
  for v in PROPMINER_N_CAP PEARL_GEMM_KERNEL PROPMINER_BENCH_NO_GRAPH \
           PROPMINER_TRIPLE_BUFFER PEARL_GEMM_DEBUG PROPMINER_USE_RELEASE; do
    if [[ -n "${!v:-}" ]]; then
      echo "  ${v}=${!v}"
    else
      echo "  ${v}=(unset)"
    fi
  done
  pass "environment snapshot complete"
}

# ── Phase 1: CPU tests ──────────────────────────────────────────────────────
phase1() {
  step_banner 2
  if [[ "${PROPMINER_SKIP_CPU:-0}" == "1" ]]; then
    skip "PROPMINER_SKIP_CPU=1"
    return
  fi
  if [[ ! -x "${ROOT}/scripts/local_host_tests.sh" ]]; then
    skip "local_host_tests.sh missing"
    return
  fi
  if ! command -v clang++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
    skip "no C++ compiler for host tests (optional on GPU box)"
    return
  fi
  if "${ROOT}/scripts/local_host_tests.sh"; then
    pass "CPU host tests (workspace sizing, transcript safety, BLAKE3)"
  else
    fail "CPU host tests"
    hint "fix host tests locally before GPU deploy; workspace_transcript_size_test catches r=128 bug"
    record_fail
  fi
}

# ── Phase 2: workspace fix grep ─────────────────────────────────────────────
phase2() {
  step_banner 3
  local capi="${ROOT}/third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp"
  if grep -q 'transcript_buffer_elems(m, n, 1)' "${capi}"; then
    pass "workspace alloc uses batch=1 (not noise rank r)"
  else
    fail "workspace fix missing in pearl_gemm_capi.cpp"
    hint "grep transcript_buffer_elems — must pass batch=1 not r=128"
    record_fail
  fi
  if grep -q 'validate_iter_graph_replay' "${capi}"; then
    pass "graph validation replay present (fail at sigma install, not first batch)"
  else
    fail "validate_iter_graph_replay not found — rebuild from latest source"
    record_fail
  fi
}

# ── Phase 3: transcript memcmp ──────────────────────────────────────────────
phase3() {
  step_banner 4
  local script="${ROOT}/scripts/verify_geforce_transcript.sh"
  if [[ ! -x "${script}" ]]; then
    skip "verify_geforce_transcript.sh not found or not executable"
    return
  fi
  if "${script}"; then
    pass "transcript memcmp + GeForce self-tests"
  else
    fail "transcript verification"
    hint "kernel math wrong before graphs — fix GEMM before phase 4"
    record_fail
  fi
}

# ── Phase 4: graph harness ────────────────────────────────────────────────────
phase4() {
  step_banner 5
  local script="${ROOT}/scripts/verify_geforce_graph.sh"
  if [[ ! -f "${script}" ]]; then
    fail "verify_geforce_graph.sh missing"
    record_fail
    return
  fi
  if bash "${script}"; then
    pass "CUDA graph harness (consumer + geforce_v2)"
  else
    fail "CUDA graph harness"
    echo ""
    echo "Interpretation guide:"
    echo "  consumer PASS + v2 FAIL  → TMA descriptors + CUDA graph (device-resident TMA fix)"
    echo "  both FAIL                → driver/graph infra; try PROPMINER_BENCH_NO_GRAPH=1"
    echo "  consumer FAIL            → unexpected; capture full log"
    hint "PEARL_GEMM_DEBUG=1 rebuild pearl-gemm; check sm_120a in Makefile ARCH"
    record_fail
  fi
}

# ── Phase 5: propminer isolation matrix ─────────────────────────────────────
phase5() {
  step_banner 6
  if [[ ! -x "${PROPMINER}" ]]; then
    skip "propminer not built"
    return
  fi

  run_case() {
    local label="$1"
    shift
    echo ""
    echo "--- ${label} ---"
    if "$@"; then
      pass "${label}"
    else
      fail "${label}"
      record_fail
    fi
  }

  run_case "geforce_v2 prod self-test (no graph)" \
    env -u PEARL_GEMM_KERNEL PROP_MINER_SELF_TEST_PROD=1 PROPMINER_BENCH_NO_GRAPH=1 \
    "${PROPMINER}" --self-test --rtx5090 --gpus 0

  run_case "geforce_v2 default (graphs ON, triple OFF)" \
    env -u PEARL_GEMM_KERNEL PROPMINER_TRIPLE_BUFFER=0 \
    "${PROPMINER}" --self-test --rtx5090 --gpus 0

  run_case "explicit v2 + no graph + prod K" \
    env PEARL_GEMM_KERNEL=geforce_v2 PROPMINER_BENCH_NO_GRAPH=1 PROPMINER_TRIPLE_BUFFER=0 \
    PROP_MINER_SELF_TEST_PROD=1 "${PROPMINER}" --self-test --rtx5090 --gpus 0
}

# ── Run selected phases ─────────────────────────────────────────────────────
for n in 0 1 2 3 4 5; do
  fn="phase${n}"
  if should_run "$((n + 1))"; then
    "$fn"
  fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "SUMMARY"
echo "══════════════════════════════════════════════════════════════"
if [[ "${failures}" -eq 0 ]]; then
  printf "${GRN}All executed steps passed.${RST}\n"
  echo ""
  echo "Production mining (defaults, graphs on):"
  echo "  export PROPMINER_N_CAP=32768"
  echo "  ./scripts/run_mining.sh   # or your pool wrapper"
  echo ""
  echo "Runtime monitor: ./scripts/debug_mining_status.sh"
else
  printf "${RED}${failures} step(s) failed.${RST}\n"
  echo ""
  echo "Common failure → action map:"
  echo "  Phase 2 workspace     → pull commit with transcript_buffer_elems(m,n,1)"
  echo "  Phase 4 v2 graph only   → device-resident TMA; or PROPMINER_BENCH_NO_GRAPH=1"
  echo "  Phase 5 no-graph OK,    → graph validation rc=-62/-63 in mine log;"
  echo "       graph FAIL           PEARL_GEMM_DEBUG=1 on sigma install"
  echo "  All GPU fail            → PEARL_GEMM_KERNEL=consumer (temporary ~257 TMAD/s)"
  echo ""
  echo "Verbose mining: PEARL_GEMM_DEBUG=1 PROPMINER_SHARE_TRACE=1"
fi

exit "${failures}"
