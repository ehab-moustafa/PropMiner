# Production tune (`tune-prod`) — RunPod / native Linux

One-time full tune on a **native NVIDIA RTX 5090** (not WSL/Salad). Finds optimal kernel knobs + batch/cluster/carveout, then reuse winners on all mining boxes via env or cache.

**Time:** ~2–3 hours (knob sweep + runtime autotune).  
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
| `~/.cache/propminer/autotune.json` | batch, graph_batch, cluster_m, carveout |
| `~/.cache/propminer/kernel_knobs.json` | KBLOCK, STAGES, SWIZZLE, MIN_BLOCKS |
| `/workspace/PropMiner/build/knob_sweep_results/` | Per-variant cmake/build logs |

**When done:**
```bash
grep -E 'Winner|new best|tune-prod Done' /workspace/tune_prod.log
cat ~/.cache/propminer/autotune.json
cat ~/.cache/propminer/kernel_knobs.json
```

---

## Apply winners to mining fleet

Copy values from `autotune.json` into env on all boxes (same `PROPMINER_N_CAP=32768`):

```
PROPMINER_BATCH=<winner batch>
PROPMINER_GRAPH_BATCH=<winner graph_batch>
PEARL_GEMM_CONSUMER_CLUSTER_M=<winner cluster_m>
PROPMINER_USE_TUNE_CACHE=1
PROPMINER_AUTOTUNE=0
```

Or bake env defaults — `autotune.json` is keyed per GPU UUID and won't auto-apply across 200 vast boxes.

---

## Wedge-proof alternative (process-isolated sweep)

If `--tune-autotune` keeps hanging (100% GPU-util, ~100W, no progress) — a CUDA
stream wedge seen on some Blackwell driver stacks (e.g. driver 580.x / CUDA 13
on sm_120a) — use the process-isolated sweep. It runs **each batch as its own
short `propminer --bench` process** under `timeout`, so a wedged combo only
kills that child and the sweep continues instead of dying.

```bash
cd /workspace/PropMiner
PROPMINER_N_CAP=32768 ./scripts/tune_runtime_safe.sh
```

- Skips wedged combos (marked `WEDGED`), keeps going, prints a results table.
- Writes recommended mining env to `~/.cache/propminer/tune_safe_result.env`.
- Graphs are **off** by default (safe on wedge-prone drivers); set
  `PROPMINER_TUNE_SAFE_GRAPH=1` to also try graphs.

Then mine with the printed env (batch + `PROPMINER_BENCH_NO_GRAPH=1` +
`PEARL_GEMM_CONSUMER_CLUSTER_M=1`). Note: results are typically flat across
batch sizes on the RTX 5090, so any completed sweep is fine.

---

## Lighter alternative (runtime tune only)

Salad/WSL one-liner — no nvcc, no knob rebuild:

```bash
export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y curl ca-certificates libssl3 xz-utils && curl -fsSL --retry 5 --retry-delay 3 https://raw.githubusercontent.com/ehab-moustafa/PropMiner/master/scripts/salad/ubuntu24_one_liner_tune.sh | bash; sleep infinity
```

On WSL, set `PROPMINER_BENCH_NO_GRAPH=1` before running if graphs wedge.

---

## What `tune-prod` sweeps

**Step 1 — kernel knobs** (rebuild per variant): KBLOCK, STAGES, SWIZZLE, MIN_BLOCKS  
**Step 2 — runtime** (fixed M/N/K from `PROPMINER_N_CAP`): batch × graph_batch (cluster_m=1, carveout=-1 by default)

N is **not** swept — tune and mine must use the **same** `PROPMINER_N_CAP`.

### Cluster / carveout are opt-in (GeForce Blackwell safety)

Thread-block clusters (`cluster_m` {2,4}) and shared-memory carveout {50,80} only
help on **datacenter** Blackwell (sm_90/sm_100). On consumer **sm_120a (RTX 5090)**
they give no measurable gain and can **wedge the CUDA stream** (100% GPU-util, ~110W,
no forward progress) regardless of CUDA graphs — a wedge trips the stall watchdog and
aborts the whole sweep. So the default runtime sweep tests **batch × graph_batch only**.

To explicitly sweep them anyway (e.g. on a datacenter GPU), opt in:

```
PROPMINER_TUNE_CLUSTERS=1   # also sweep cluster_m {1,2,4}
PROPMINER_TUNE_CARVEOUT=1   # also sweep carveout {-1,50,80}
```

Each combo is now logged before it runs (`[autotune] trying batch=… cluster_m=… carveout=…`),
so if one wedges, the last log line names the exact culprit.
