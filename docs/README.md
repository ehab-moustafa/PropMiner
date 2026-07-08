# PropMiner — User Guide

PropMiner is a GPU miner for **Pearl (PRL)** using the **Pearl V2 gRPC** protocol (HTTP/2 + TLS). It runs the proof-of-work **NoisyGEMM** on NVIDIA GPUs and submits verified shares to the pool. **0% dev fee.**

The v2 codebase is optimized for **RTX 5090 (Blackwell sm_120a)**: largest VRAM-fitting **N**, CUDA graphs, VRAM-resident B matrices, and an offline tuning pipeline — without sacrificing proof correctness.

---

## What PropMiner does

1. **Connect** to the pool over gRPC/TLS and register (wallet, worker, GPU cards).
2. **Receive jobs** — each job carries σ (sigma), B seed, and a difficulty target.
3. **Install σ once** — expand B, hash, noise, keep **BpEB resident in VRAM**.
4. **Mine in a tight GPU loop** — for each nonce batch:
   - Regenerate noisy A on GPU
   - BLAKE3 tensor-hash A → commitments
   - Run **headless consumer GEMM** (int8 tensor cores, in-kernel transcript + PoW check)
5. **On hit** — rebuild share proof on host, submit via gRPC `ShareSubmission`.
6. **Heartbeat** every 30s; watchdog recovers from GPU stalls.

**CPU role:** orchestration, seed upload, share build/send only — **no CPU mining**.

---

## Proof of work (high level)

| Piece | Detail |
|---|---|
| **Shape (5090 prod)** | M=8192, N≤262144, K=128, r=128 |
| **Tile (proof-canonical)** | 128×256×128 per CTA |
| **Matrices** | int8 A, B with sparse rank-128 noise |
| **GEMM** | int8 → int32 accumulators via tensor cores |
| **Transcript** | XOR snapshots at K boundaries — must match pool verifier byte-for-byte |
| **Hash** | BLAKE3 keyed Merkle on matrix tiles + commitments |
| **Target** | In-kernel check against pool `target_nbits` |
| **Hashrate unit** | DAF-normalized tiles/s (console may show TH/s scale) |

The production kernel deliberately keeps the **SM80 `mma.sync` atom** inside an **sm_120a** cubin so transcripts stay identical across architectures the pool accepts.

---

## Goals & design principles

PropMiner v2 was built to:

1. **Maximize RTX 5090 hashrate** on live Pearl pools (Kryptex gRPC default).
2. **Never break share proofs** — proof-canonical tiles, headless GEMM, validated self-test path.
3. **Keep the GPU saturated** — ping-pong buffers, CUDA graphs, async seed copy, batch tuning.
4. **Ship production-safe defaults** — aggressive `cluster_m=2`, tune caches, no startup autotune that shrinks N.
5. **Run in cloud marketplaces** — Docker + WSL2/Salad bootstrap, embedded CUDA 12.8 runtime.

---

## Supported GPUs

### Build profiles (`PEARL_GEMM_ARCH`)

Set at compile time (`CMakeLists.txt`, `third_party/pearl-gemm/csrc/capi/Makefile`):

| Profile | NVCC arch | Transcript kernel | Primary GPUs |
|---|---|---|---|
| `blackwell` | sm_120a | `blackwell/transcript_gemm_sm120.cu` | RTX 5090, 5080, 5070 … |
| `ada` | sm_89 | `consumer/transcript_gemm_kernel.cu` | RTX 4090, 4080, 4070 … |
| `ampere` | sm_80, sm_86 | consumer | RTX 3090, A100, A6000 … |
| `h100` | sm_90a | Hopper WGMMA | H100, H200 |
| `b200` | sm_100a | `transcript_gemm_sm100.cu` (tcgen05) | B200 datacenter |
| `turing` | sm_75 | SM75 path | RTX 2080, T4 |
| `volta` | sm_70 | Volta path | V100 |
| `portable` | sm_80+PTX | `portable/transcript_gemm_kernel.cu` | Fallback |

CMake auto-detects via `nvidia-smi` when `PROP_MINER_CUDA_ARCH=auto`.

### Runtime shape (without `--rtx5090`)

`MiningConfig::auto_shape_for_gpu()` picks M/N/K from VRAM:

- **Ada+ (CC ≥ 8.9):** up to 8192 × 262144
- **Ampere:** up to 4096 × 16384
- **Older:** conservative 2048 × 8192

### What does **not** work on RTX 5090

| Item | Why |
|---|---|
| `sm_100a` / B200 cubins | Wrong ISA for GeForce Blackwell |
| `tcgen05` / TMEM GEMM | Not available on GeForce sm_120a |
| `PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma` | Scaffold only — compile fails until implemented |
| `cluster_m=3` | Kernel falls back to 1 |
| Stratum TCP mining | Not linked in v2 binary — use gRPC |
| CPU mining | Disabled by design |

---

## Command line

```
propminer [OPTIONS]
```

There is no subcommand — mode is selected by flags and environment.

### Options

| Option | Description |
|---|---|
| `--pool HOST:PORT` | Pool address (default `prl.kryptex.network:443`) |
| `--wallet`, `-w` | Pearl/Kryptex wallet (**required** for mine) |
| `--worker NAME` | Worker name (default `propminer`) |
| `--gpus i,j,...` | Comma-separated CUDA device indices (default: all) |
| `--rtx5090` | RTX 5090 profile: M=8192, N from VRAM (up to 262144) |
| `--bench SECONDS` | Synthetic local job, benchmark, exit (no pool) |
| `--self-test` | Mine tiny/easy target, verify first share, exit |
| `--tune-mine-batch S` | Sweep mine batch at prod N → `mine_batch.json` (**needs `--rtx5090`**) |
| `--tune-cluster S` | Sweep `cluster_m` ∈ {1,2,4} → log winner (**needs `--rtx5090`**) |
| `--tune-autotune S` | Full runtime autotune → `autotune.json` (**needs `--rtx5090`**) |
| `--config M,N,K,R` | Override GEMM dimensions |
| `--no-watchdog` | Disable GPU stall watchdog |
| `--disable-cpu` | Explicit no-CPU-mining (default; no-op) |
| `--tls 0/1` | gRPC TLS (default on) |
| `--help` | Usage |

### Modes at a glance

| Intent | Command |
|---|---|
| Quick health check | `propminer --self-test --rtx5090 --gpus 0` |
| Prod-shape self-test | `PROP_MINER_SELF_TEST_PROD=1 propminer --self-test --rtx5090 --gpus 0` |
| 180s hashrate sample | `propminer --bench 180 --rtx5090 --gpus 0` |
| Live mining | `propminer --rtx5090 --wallet krxUSER.worker --gpus 0` |
| Offline tune | `propminer --tune-cluster 20 --rtx5090 --gpus 0` |

---

## Environment variables

### Production mining

| Variable | Default (prod) | Meaning |
|---|---|---|
| `PROPMINER_WALLET` | — | Wallet string (`krx…` or `prl1p…`) |
| `PROPMINER_POOL` | `prl.kryptex.network:443` | Host:port |
| `PROPMINER_GPUS` | `0` | Passed to `--gpus` |
| `PROPMINER_WORKER` | — | Worker if not embedded in wallet |
| `PROPMINER_MODE` | `full` (Docker) | `mine`, `full`, `test`, `tune`, `batch-tune`, `cluster-tune`, `tune-prod` |
| `PROPMINER_RESTART_ON_EXIT` | `1` | Restart miner on crash (`run_mining.sh`) |
| `PROPMINER_USE_TUNE_CACHE` | `1` | Load `autotune.json` for cluster/carveout |
| `PROPMINER_BATCH` | cache → 1 | Matmuls per poll (default 1 at prod N) |
| `PROPMINER_GRAPH_BATCH` | cache → 1 | CUDA graph capture depth (default 1) |
| `PROPMINER_AUTOTUNE` | `0` | `0` off, `1` cache/sweep, `2` require knob cache, `force` re-sweep |
| `PEARL_GEMM_CONSUMER_CLUSTER_M` | `1` | Thread-block cluster: **1**=off, **2** or **4**=on |
| `PEARL_GEMM_CONSUMER_CARVEOUT` | from cache | L1/shared carveout % (0–100) |
| `PROPMINER_DEFER_SHARE_GPU` | on | `0` = share rebuild back inline on the mine loop |
| `PROPMINER_BCOL_CACHE` | on | `0` = legacy full n×k B expansion in share path |
| `PROPMINER_ASYNC_SEED` | on | `0` = synchronous seed upload (watch shares 30 min after deploy) |
| `PROPMINER_ASYNC_JOB_INSTALL` | on | `0` = force synchronous job switch. On: installs the next job's resident B on a background thread while mining continues, then fast-swaps. VRAM-guarded — self-disables to synchronous when free VRAM is tight (e.g. large N), so it never OOMs |
| `PROP_MINER_SELF_TEST_PROD` | off | `1` = self-test at full prod N |

### Benchmark / validation

| Variable | Default | Meaning |
|---|---|---|
| `PROPMINER_BENCH_SECONDS` | `180` | Bench duration (Docker `full` mode) |
| `PROPMINER_BENCH_GRACE_SECONDS` | `60` | Extra wait after bench window |
| `PROPMINER_BENCH_BATCH` | `1` (bench) | Batch size during `--bench` |
| `PROPMINER_BENCH_NO_GRAPH` | off | Disable CUDA graphs in bench |
| `PROPMINER_SKIP_BENCH` | mode-dependent | Skip benchmark in test kit |
| `PROPMINER_QUICK_EXIT` | `1` (test) | Exit after self-test |

### Kernel runtime (optional)

| Variable | Default | Meaning |
|---|---|---|
| `PEARL_GEMM_DEBUG` | `0` | Verbose GEMM logging |
| `PEARL_GEMM_KERNEL` | geforce (blackwell builds) | `consumer` to opt out of warp-specialized kernel |

### Build-time (CMake / Make)

| Variable | Default | Meaning |
|---|---|---|
| `PROP_MINER_CUDA_ARCH` | `blackwell` | Arch profile for host+gemm build |
| `PEARL_GEMM_BLACKWELL_KBLOCK` | `64` | K-tile depth |
| `PEARL_GEMM_BLACKWELL_STAGES` | `2` | Pipeline stages |
| `PEARL_GEMM_BLACKWELL_SWIZZLE_BITS` | `2` | Shared-memory swizzle |
| `PEARL_GEMM_BLACKWELL_LOAD_POLICY` | `cp_async` | `tma` = experimental (not prod) |
| `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL` | `1` (blackwell) | `0` = omit GeForce kernel from build |

---

## Tune caches

Directory: `~/.cache/propminer/` (or `$XDG_CACHE_HOME/propminer`).

Keyed per GPU UUID + driver + `PEARL_GEMM_ARCH`.

| File | Written by | Applied at mine |
|---|---|---|
| `kernel_knobs.json` | `tune_blackwell_knobs.sh` | Requires **rebuild** with winning CMake knobs; miner warns on mismatch |
| `autotune.json` | `--tune-autotune`, cluster tune phase 2 | `cluster_m`, `carveout` (prod default); full M/N if `PROPMINER_AUTOTUNE=1` |
| `mine_batch.json` | `--tune-mine-batch` | Optimal `batch` for current M,N |

**Recommended once per 5090 rig:**

```bash
./scripts/tune_prod_5090.sh
# or: PROPMINER_MODE=tune-prod ./scripts/docker_entrypoint.sh
```

Then mine with caches (default in `run_mining.sh`).

---

## RTX 5090 profile (`--rtx5090`)

| Parameter | Value |
|---|---|
| SMs | 170 (GB202) |
| Tile | 128 × 256 × 128 |
| M | 8192 (fixed) |
| N | Largest VRAM fit: **262144** → 131072 → 65536 → … |
| K, R | 128 |
| Bench N cap | 32768 (`--bench` only) |
| Default mine batch | 1 (env/cache can override) |
| Default graph batch | 1 (env/cache can override) |
| Prod `cluster_m` | 1 (env/tune can override) |
| CTAs per GEMM @ N=262144 | 65,536 (tail wave on 170 SMs is expected) |

**Optimizations active in prod path:**

- Ping-pong GPU buffers (double-buffered A/C halves)
- CUDA graph capture for batch launches
- Dedicated `seed_copy_stream_` for 8-byte nonce H2D outside graph capture
- `cudaEvent` batch timing for accurate hashrate (no hot-path sleep)
- L2 fetch granularity 128B
- Share build on orchestrator side thread
- Optional `PROPMINER_DEFER_SHARE_GPU` (off by default)

See [RTX5090_BLUEPRINT.md](RTX5090_BLUEPRINT.md) for engineering detail.

---

## Wallet & worker (Kryptex)

| Format | Example |
|---|---|
| Kryptex wallet + worker | `krxX2P3Z84.worker1` |
| Pearl address + worker | `prl1p…worker1` |
| Separate worker flag | `--worker rig01` or `PROPMINER_WORKER=rig01` |

**Rules:**

- Minimum **8** characters for wallet string.
- If wallet has no `.worker` suffix and no `--worker`, pool may show `propminer`.
- gRPC register sends `wallet_address` + `worker_name` independently.

---

## Reading the console

Typical production lines:

```
[main] RTX 5090 profile: M=8192 N=262144 batch=4 CTAs=65536 waves~386 tail=16
[main] RTX 5090 prod: aggressive defaults (cluster_m=2, tune cache on; ...)
[orchestrator] RTX 5090 prod: cluster_m=2 carveout=-1 (tune cache on)
[orchestrator] Production mine mode: pool prl.kryptex.network:443 (awaiting first job...)
[gpu 0] hashrate=… TMAD/s batch=4 σ_age=…
```

| Log fragment | Meaning |
|---|---|
| `N=262144` | Full production intensity |
| `tail=16` | 16 idle SM slots last wave — cheaper than shrinking N |
| `tune cache on` | Loading `autotune.json` cluster/carveout |
| `awaiting first job` | Normal before first σ |
| `σ_age` | Seconds since last job rotation |
| `TMAD/s` | Tiles per second (× DAF → protocol hashrate) |

Logs in Docker/scripts: `results/summary.txt`, `results/propminer_stderr.log`.

---

## Technology stack

| Layer | Technology |
|---|---|
| Host application | C++20 (`src/host/`) |
| GPU kernels | CUDA 12.8, CUTLASS 4.4 / CuTe |
| Matrix PoW | `pearl-gemm` — consumer + Blackwell wrappers |
| BLAKE3 / Merkle | Rust `pearl-mining-capi` + GPU `tensor_hash` |
| Pool wire | Custom gRPC-style HTTP/2 + HPACK + protobuf (`grpc_client.cpp`) |
| TLS | OpenSSL |
| Build | CMake 3.24+ + GNU Make (dual build for host + `.so`) |
| Container | Multi-stage Docker, embedded CUDA 12.8 runtime |

**Protocol version:** Pearl V2 (`protocol_version = 2` on register).

**Not in the production binary:** legacy Stratum sources under `src/host/stratum/` (unlinked).

---

## Building from source

### Prerequisites

- CMake ≥ 3.24, C++20 compiler
- CUDA Toolkit 12.4+ (`nvcc`; 12.8 for sm_120a)
- Rust / `cargo` (stable)
- OpenSSL dev headers (`libssl-dev`)
- `git`, `python3`, `make`

CUTLASS is auto-fetched (v4.4.0) if the submodule is missing.

### Commands

```bash
# RTX 5090 (default in repo)
./scripts/build.sh

# Explicit arch
PROP_MINER_CUDA_ARCH=ada ./scripts/build.sh

# Manual CMake
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPROP_MINER_CUDA_ARCH=blackwell
make -j$(nproc)
```

### Blackwell compile-time knobs (tuned via `tune_blackwell_knobs.sh`)

| Knob | Production default |
|---|---|
| KBLOCK | 64 |
| STAGES | 2 |
| SWIZZLE_BITS | 2 |
| MIN_BLOCKS | 1 |
| LOAD_POLICY | cp_async |
| BM × BN | 128 × 256 (fixed — proof-canonical) |

---

## Docker & cloud deployment

### Build & run

```bash
docker build -t propminer-rtx5090 .
docker run --gpus all -e PROPMINER_MODE=mine -e PROPMINER_WALLET=krxUSER.worker propminer-rtx5090
```

### Salad / WSL2

**SRBMiner-style (`ubuntu:24.04` only, no custom image):** set env `PROPMINER_WALLET=krxUSER.worker1` and paste the startup command from `scripts/salad/ubuntu24_one_liner_fast.oneline` (see `ubuntu24_one_liner_fast.sh`).

**Docker image:** default `PROPMINER_MODE=full` validates GPU in zero-config deployments.
- `setup_cuda_env.sh` sets WSL `LD_PRELOAD` for `/dev/dxg` driver bridge.
- Bench uses `PROPMINER_BENCH_BATCH=1` so the first batch completes in the 180s window.

### vast.ai / bare metal

Same Docker image or native `run_mining.sh` with `NVIDIA_VISIBLE_DEVICES` set to the contracted GPU index.

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `No CUDA devices` | Driver/runtime not visible | `nvidia-smi`; check `--gpus`; WSL `/dev/dxg` |
| CUDA init fails in container | Missing libs | Use PropMiner image; check `setup_cuda_env.sh` fallback |
| 0 H/s first 1–2 min | First σ + full-N batch | Wait; read `σ_age` |
| Bench 0 H/s | Batch too large for window | `PROPMINER_BENCH_BATCH=1`, grace 60s+ |
| `register failed` | Wallet/TLS/network | Verify wallet, port 443, TLS |
| Share rejects | Cluster/transcript mismatch | `PEARL_GEMM_CONSUMER_CLUSTER_M=1`; re-run tune |
| Knob cache mismatch warning | Rebuilt without re-tune | `PROPMINER_MODE=tune` |
| GPU hang | TDR / driver | Watchdog republishes job; reduce OC |
| OOM at build | WSL parallel nvcc | `NVCC_THREADS=1` |

After **any** GPU overclock change, run `--self-test --rtx5090` before prod mining.

---

## Further reading

| Doc | Topic |
|---|---|
| [examples.md](examples.md) | Copy-paste command recipes |
| [RTX5090_BLUEPRINT.md](RTX5090_BLUEPRINT.md) | 5090 engineering blueprint |
| [RTX5090_LINUX_TASKS.md](RTX5090_LINUX_TASKS.md) | Linux bring-up checklist |
| [../performance optimizations/](../performance%20optimizations/) | Optimization design notes 01–08 |
