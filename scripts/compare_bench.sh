#!/usr/bin/env bash
# Compare TMAD/s between two bench JSONL files or log files.
# Usage: ./scripts/compare_bench.sh baseline.jsonl candidate.jsonl [tolerance_fraction]
set -euo pipefail

BASE="${1:?baseline file}"
NEW="${2:?candidate file}"
TOL="${3:-0.02}"

extract_tmad() {
    local f="${1}"
    if grep -q '"tmad_per_sec"' "${f}" 2>/dev/null; then
        grep '"tmad_per_sec"' "${f}" | tail -1 \
            | sed -n 's/.*"tmad_per_sec":\([0-9.eE+-]*\).*/\1/p'
        return
    fi
    grep '^benchmark: ' "${f}" 2>/dev/null | tail -1 \
        | sed -n 's/.*TMAD\/s=\([0-9.]*\).*/\1/p'
}

base_tmad="$(extract_tmad "${BASE}")"
new_tmad="$(extract_tmad "${NEW}")"

if [[ -z "${base_tmad}" || "${base_tmad}" == "0" ]]; then
    echo "FAIL: no baseline tmad_per_sec in ${BASE}" >&2
    exit 1
fi
if [[ -z "${new_tmad}" || "${new_tmad}" == "0" ]]; then
    echo "FAIL: no candidate tmad_per_sec in ${NEW}" >&2
    exit 1
fi

awk -v b="${base_tmad}" -v n="${new_tmad}" -v t="${TOL}" 'BEGIN {
    pct = (n - b) / b
    printf "TMAD/s: %.2f -> %.2f (%+.1f%%)\n", b, n, pct * 100
    if (pct < -t) { print "FAIL: regression beyond tolerance"; exit 1 }
    print "PASS"
}'
