# Production tune (`tune-prod`) — RunPod / native Linux

One-time full tune on a **native NVIDIA RTX 5090** (not WSL/Salad). Finds optimal kernel knobs + batch/cluster/carveout, then reuse winners on all mining boxes via env or cache.

**Time:** ~3–4 hours (optional knob sweep + full runtime autotune with 4 N values).  
**No wallet required** — tune never connects to the pool.

---

## Image

**Recommended:** `nvidia/cuda:12.8.0-devel-ubuntu24.04` (nvcc included).

Runtime image also works — the startup command installs `cuda-nvcc-12-8` if needed.

**Verify GPU after SSH:**
```bash
nvidia-smi
ls /dev/nvidia*
```
Must show RTX 5090. Native driver only (no `KMD Version / UMD Version` split like WSL).

---

## Environment variables

Set in RunPod **Environment Variables** (optional overrides; defaults are fine):

```
PROPMINER_USE_STRATUM=1
PROPMINER_N_CAP=32768
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
CUDA_MODULE_LOADING=EAGER
CUDA_DEVICE_MAX_CONNECTIONS=1
```

Do **not** set `PROPMINER_WALLET` for tune.  
On **WSL/Salad** boxes only, add `PROPMINER_BENCH_NO_GRAPH=1` (CUDA graphs wedge on WSL).

---

## Startup command

Paste as **Docker start command** / pod on-start script:

```bash
export DEBIAN_FRONTEND=noninteractive && \
apt-get update && apt-get install -y git cmake build-essential libc6-dev libssl-dev pkg-config curl python3 xz-utils ca-certificates && \
(apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 cuda-nvrtc-dev-12-8 cuda-driver-dev-12-8 2>/dev/null || true) && \
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
export PATH="/root/.cargo/bin:/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:$PATH" && \
export CUDAToolkit_ROOT=/usr/local/cuda-12.8 && \
git clone https://github.com/ehab-moustafa/PropMiner.git /workspace/PropMiner && \
cd /workspace/PropMiner && git submodule update --init --recursive && \
PROPMINER_USE_STRATUM=1 PROPMINER_N_CAP=32768 \
./scripts/tune_prod_5090.sh 2>&1 | tee /workspace/tune_prod.log; sleep infinity
```

---

## Watch logs (SSH)

```bash
tail -f /workspace/tune_prod.log
```

**Alive check:**
```bash
ps aux | grep propminer | grep -v grep
nvidia-smi
```

---

## Result files

| Path | Contents |
|------|----------|
| `/workspace/tune_prod.log` | Full log |
| `build/tune_full_raw.tsv` | Every run: TMAD/s + GPU temp/power/util + CPU/RAM/VRAM |
| `build/tune_full_results.txt` | Human-readable lines |
| `build/tune_full_summary.txt` | Global winner + best settings per N |
| `build/tune_full_winners_by_n.tsv` | Best batch/graph/cluster per N |
| `~/.cache/propminer/tune_full_result.env` | Fleet mining env (global winner) |
| `~/.cache/propminer/kernel_knobs.json` | KBLOCK, STAGES, SWIZZLE (step 1 only) |

**When done:**
```bash
cat ~/.cache/propminer/tune_full_result.env
cat build/tune_full_summary.txt
head -5 build/tune_full_raw.tsv
```

---

## Apply winners to mining fleet

```bash
set -a && source ~/.cache/propminer/tune_full_result.env && set +a
export PROPMINER_MODE=mine PROPMINER_WALLET=<wallet>
./scripts/docker_entrypoint.sh
```

Copy `tune_full_result.env` to all boxes. **Mining N must match tune N** (`PROPMINER_N_CAP` in the env file).

---

## Production resilience (mining)

| Feature | Env | Default |
|---------|-----|---------|
| Hung-GPU stall guard | `PROPMINER_STALL_RESTART_MS` | `30000` (exit rc=42) |
| Fast restart after stall | `PROPMINER_STALL_RESTART_DELAY_SEC` | `3` (vs 10s other exits) |
| Shell supervisor loop | `PROPMINER_RESTART_ON_EXIT` | `1` |
| Thermal pause | `PROPMINER_GPU_TEMP_STOP` / `_START` | off (set e.g. 85/75) |
| In-process watchdog | default on | republishes job on soft stall |
| **Progress monitor** | `PROPMINER_PROGRESS_STALL_ABORT_MS=18000` | 18s no iters → batch abort → rc=42 → 3s restart |
| **Wedge signature** | `PROPMINER_WEDGE_POWER_THRESHOLD_W=150` | high util + power below this = wedge suspected |

---

## What `tune-prod` sweeps

**Step 1 — kernel knobs** (optional, skip with `PROPMINER_SKIP_KNOB_TUNE=1`): KBLOCK, STAGES, SWIZZLE, MIN_BLOCKS  

**Step 2 — runtime** via `tune_runtime_full.sh` (each combo = own `propminer --bench`, retries on wedge):

| Axis | Default sweep |
|------|----------------|
| **N** | `32768 65536 131072 262144` (compare all four) |
| batch | 1,2,4,6,8,10,12,14,16,20,24,28,32,40,48 |
| graph | on and off |
| graph_batch | 1,2,4,8,16,32 divisors of batch |
| cluster_m | 1, 2, 4 |

**768 combos** (192 per N × 4 N) × ~15s ≈ **3+ hours**.

Single N only: `PROPMINER_TUNE_SWEEP_N=0 PROPMINER_N_CAP=131072 ./scripts/tune_prod_5090.sh`

| Env | Default | Meaning |
|-----|---------|---------|
| `PROPMINER_TUNE_N_VALUES` | `32768 65536 131072 262144` | N values to compare |
| `PROPMINER_TUNE_SWEEP_N` | `1` | `0` = single `PROPMINER_N_CAP` only |
| `PROPMINER_TUNE_MAX_RETRIES` | `3` | Retries per combo on wedge |
| `PROPMINER_TUNE_GRAPHS` | `both` | `on`, `off`, or `both` |
| `PROPMINER_TUNE_CLUSTERS` | `1` | `0` = cluster 1 only |

---

## Production resilience (mining)
