#!/usr/bin/env bash
# Host-only validation for tune_runtime_full.sh (no GPU).
# Run in CI and on RunPod before a long tune.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "===== tune full dry-run checks (no GPU) ====="

for script in \
    scripts/tune_runtime_full.sh \
    scripts/tune_runtime_prod.sh \
    scripts/tune_prod_5090.sh \
    scripts/local_host_tests.sh; do
    bash -n "${script}"
    echo "[dry-run] OK syntax: ${script}"
done

# Combo count must match tune_runtime_full.sh defaults (4 N × 192 runtime combos).
expected_per_n=192
expected_n=4
expected_total=$(( expected_per_n * expected_n ))

actual_per_n="$(
python3 <<'PY'
batches = [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48]
clusters = [1, 2, 4]
combos = 0
for batch in batches:
    for use_graph in (1, 0):
        if use_graph:
            gbs = [g for g in (1, 2, 4, 8, 16, 32) if g <= batch and batch % g == 0]
        else:
            gbs = [1]
        combos += len(gbs) * len(clusters)
print(combos)
PY
)"

if [[ "${actual_per_n}" != "${expected_per_n}" ]]; then
    echo "[dry-run] FAIL combos_per_n=${actual_per_n} expected=${expected_per_n}" >&2
    exit 1
fi
echo "[dry-run] OK combos_per_n=${actual_per_n} total=${expected_total} (4 N values)"

# Default N sweep must include 65536.
default_n="${PROPMINER_TUNE_N_VALUES:-32768 65536 131072 262144}"
for n in 32768 65536 131072 262144; do
    if [[ " ${default_n} " != *" ${n} "* ]]; then
        echo "[dry-run] FAIL missing N=${n} in default sweep" >&2
        exit 1
    fi
done
echo "[dry-run] OK default N sweep includes 65536"

# TSV header columns (must match append_tsv_from_json in tune_runtime_full.sh).
header='n_cap	batch	graph	graph_batch	cluster_m	carveout	status	tmad_per_sec	iters_per_sec	tops_pct	m	n	k	gpu_util_pct	gpu_mem_util_pct	gpu_temp_c	gpu_power_w	gpu_power_limit_w	vram_used_mb	vram_total_mb	cpu_util_pct	ram_used_mb	ram_total_mb'
cols="$(echo "${header}" | awk -F'\t' '{print NF}')"
if [[ "${cols}" != "23" ]]; then
    echo "[dry-run] FAIL TSV header column count=${cols} expected=23" >&2
    exit 1
fi
echo "[dry-run] OK TSV header (${cols} columns)"

# Mock JSON -> TSV row parse (same fields as tune script).
mock='{"tmad_per_sec":300.5,"iters_per_sec":0.812345,"tops_pct":35.8,"m":8192,"n":65536,"k":128,"gpu_util_pct":100,"gpu_mem_util_pct":90,"gpu_temp_c":71,"gpu_power_w":495,"gpu_power_limit_w":575,"vram_used_mb":18000,"vram_total_mb":32768,"cpu_util_pct":8,"ram_used_mb":12000,"ram_total_mb":64000}'
row="$(
    TUNE_JSON_LINE="${mock}" TUNE_N_CAP=65536 TUNE_BATCH=4 TUNE_GRAPH=on \
    TUNE_GB=4 TUNE_CLUSTER=1 TUNE_CARVEOUT=-1 TUNE_STATUS=ok \
    python3 <<'PY'
import json, os
j = json.loads(os.environ["TUNE_JSON_LINE"])
fields = [
    os.environ["TUNE_N_CAP"], os.environ["TUNE_BATCH"], os.environ["TUNE_GRAPH"],
    os.environ["TUNE_GB"], os.environ["TUNE_CLUSTER"], os.environ["TUNE_CARVEOUT"],
    os.environ["TUNE_STATUS"],
    j.get("tmad_per_sec", ""), j.get("iters_per_sec", ""), j.get("tops_pct", ""),
    j.get("m", ""), j.get("n", ""), j.get("k", ""),
    j.get("gpu_util_pct", -1), j.get("gpu_mem_util_pct", -1),
    j.get("gpu_temp_c", -1), j.get("gpu_power_w", -1),
    j.get("gpu_power_limit_w", -1),
    j.get("vram_used_mb", -1), j.get("vram_total_mb", -1),
    j.get("cpu_util_pct", -1), j.get("ram_used_mb", -1), j.get("ram_total_mb", -1),
]
print("\t".join(str(x) for x in fields))
PY
)"
if [[ "$(echo "${row}" | awk -F'\t' '{print $12}')" != "65536" ]]; then
    echo "[dry-run] FAIL mock TSV parse n=$(echo "${row}" | awk -F'\t' '{print $12}')" >&2
    exit 1
fi
if [[ "$(echo "${row}" | awk -F'\t' '{print $9}')" != "0.812345" ]]; then
    echo "[dry-run] FAIL mock TSV parse iters_per_sec" >&2
    exit 1
fi
if [[ "$(echo "${row}" | awk -F'\t' '{print $17}')" != "495" ]]; then
    echo "[dry-run] FAIL mock TSV parse gpu_power_w" >&2
    exit 1
fi
echo "[dry-run] OK mock JSON -> TSV (N=65536, iters_per_sec=0.812345, gpu_power_w=495)"

echo "===== tune full dry-run OK ====="
