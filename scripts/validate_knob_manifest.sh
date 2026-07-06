#!/usr/bin/env bash
# Compare libpearl_gemm_capi.so PEARL_GEMM_BUILD_KNOBS string vs kernel_knobs.json (GPU 0).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/tune_kernel_knobs_common.sh"

fail() { echo "[knob-validate] FAIL: $*" >&2; exit 1; }

if [[ "$(uname -s)" == "Darwin" ]] || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[knob-validate] SKIP: no NVIDIA GPU (Mac or headless)"
    exit 0
fi

SO=""
for cand in \
    "${ROOT}/build/libpearl_gemm_capi.so" \
    "${ROOT}/libpearl_gemm_capi.so" \
    "${ROOT}/third_party/pearl-gemm/csrc/capi/build/libpearl_gemm_capi.so"; do
    if [[ -f "${cand}" ]]; then
        SO="${cand}"
        break
    fi
done
[[ -n "${SO}" ]] || fail "libpearl_gemm_capi.so not found (build propminer first)"

built="$(strings "${SO}" | grep -E '^k[0-9]+-s[0-9]+-sw[0-9]+-mb[0-9]+-' | head -1 || true)"
[[ -n "${built}" ]] || fail "no PEARL_GEMM_BUILD_KNOBS string in ${SO}"

cache="$(tune_knob_cache_path)"
[[ -f "${cache}" ]] || fail "missing cache ${cache} (run ./scripts/tune_blackwell_knobs.sh)"

key="$(tune_knob_gpu_cache_key)"
cached_manifest=""
cached_self_test=""
while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    [[ "${line%% *}" != "${key}" ]] && continue
    rest="${line#* }"
    IFS=',' read -r _kblock _stages _swizzle _min_blocks _load_policy manifest _hashrate self_test <<< "${rest}"
    cached_manifest="${manifest}"
    cached_self_test="${self_test}"
    break
done < "${cache}"

[[ -n "${cached_manifest}" ]] || fail "no cache entry for GPU key ${key} in ${cache}"

echo "[knob-validate] built .so:  ${built}"
echo "[knob-validate] cache GPU0: ${cached_manifest} (self_test=${cached_self_test})"

if [[ "${built}" != "${cached_manifest}" ]]; then
    fail "manifest mismatch: built=${built} cache=${cached_manifest}"
fi
if [[ "${cached_self_test}" != "1" ]]; then
    fail "cache self_test_ok=${cached_self_test:-0}; re-run ./scripts/tune_blackwell_knobs.sh"
fi

echo "[knob-validate] OK"
