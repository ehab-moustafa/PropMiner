# PropMiner

GPU miner for the **Pearl (PRL)** network over **Pearl V2 gRPC**. NVIDIA CUDA only. **0% dev fee.**

PropMiner v2 is built for **maximum hashrate on RTX 5090 (Blackwell sm_120a)** while staying proof-correct: every accepted share uses the same byte-identical noisy-GEMM transcript the pool verifies.

> **Proof of work:** low-rank-noised integer matrix multiply (NoisyGEMM). Each candidate is a tile of `A·Bᵀ` that is hashed (BLAKE3) and checked against the pool difficulty target. Heavy compute runs on the GPU; the host handles pool session, commitments, and share submission.

---

## What makes PropMiner different

| Focus | Detail |
|---|---|
| **RTX 5090 first** | Native `sm_120a` cubin, 128×256×128 consumer tile, production **N up to 262144**, 170-SM wave-aware grid |
| **Pearl V2 gRPC** | Custom HTTP/2 + TLS client (no Stratum) — default pool **Kryptex** on port **443** |
| **GPU-isolated mine path** | VRAM-resident B, CUDA graphs, ping-pong batches, overlapped 8-byte seed upload — **no CPU mining** |
| **Aggressive prod defaults** | `--rtx5090`, `cluster_m=2`, tune cache on, batch from offline sweep |
| **Offline tuning pipeline** | One-shot `tune-prod` → kernel knobs + cluster + mine batch caches |
| **Open & auditable** | C++20 host, CUDA kernels, Rust BLAKE3 C API — build from source |

---

## Hardware

Architecture is selected at **build time** via `PEARL_GEMM_ARCH` / `PROP_MINER_CUDA_ARCH`. Runtime auto-detects shape for non-5090 cards when `--rtx5090` is omitted.

| Arch | Example GPUs | `PEARL_GEMM_ARCH` | Notes |
|---|---|---|---|
| **Blackwell** | RTX 50-series (5090, 5080 …) | `blackwell` | **Primary target** — Docker image is `sm_120a` only |
| **Ada** | RTX 40-series, L4, L40S | `ada` | Consumer headless GEMM |
| **Ampere** | RTX 30-series, A100, A6000 | `ampere` | |
| **Hopper** | H100, H200 | `h100` | WGMMA path |
| **Blackwell DC** | B200 | `b200` | Datacenter tcgen05/TMEM — **not** RTX 5090 |
| **Turing** | RTX 20-series, T4 | `turing` | |
| **Volta** | V100 | `volta` | |
| **Portable** | sm_80+ fallback | `portable` | PTX JIT fallback |

**Requirements:** NVIDIA driver with CUDA **12.4+** (CUDA **12.8** in Docker), compute capability **sm_70+** (sm_120a for the prebuilt RTX 5090 image).

**Not supported on GeForce RTX 5090:** `tcgen05` / TMEM kernels (datacenter B200 only). The production path uses proof-canonical **SM80 `mma.sync`** inside an `sm_120a` cubin.

---

## Pool (Kryptex — default)

PropMiner speaks **Pearl V2 gRPC over TLS** (not Stratum TCP).

| Setting | Default |
|---|---|
| Host | `prl.kryptex.network` |
| Port | `443` |
| TLS | on (`--tls 1`) |

**Alternate region example:** `prl-eu.kryptex.network:443`

**Wallet formats:** `krxUSERNAME.workername` or `prl1pYOURADDRESS.workername`

Keep the run command on **one line** in terminals and wrappers — broken line continuations can pass stray arguments.

---

## Quick start (Linux)

### Build

```bash
git clone <your-repo-url> PropMiner && cd PropMiner
./scripts/build.sh
```

Output: `build/propminer` + `libpearl_gemm_capi.so` + `libpearl_mining_capi.so`.

### Verify

```bash
./build/propminer --self-test --rtx5090 --gpus 0
PROP_MINER_SELF_TEST_PROD=1 ./build/propminer --self-test --rtx5090 --gpus 0
```

### Benchmark (no pool)

```bash
./build/propminer --bench 180 --rtx5090 --gpus 0
```

### Mine

```bash
./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --pool prl.kryptex.network:443 --gpus 0
```

Or use the production script (restart loop, logging, aggressive defaults):

```bash
PROPMINER_WALLET=krxYOURUSER.worker1 ./scripts/run_mining.sh
```

### One-time tune (5090), then mine

```bash
./scripts/tune_prod_5090.sh
PROPMINER_WALLET=krxYOURUSER.worker1 ./scripts/run_mining.sh
```

Caches land in `~/.cache/propminer/` (`kernel_knobs.json`, `autotune.json`, `mine_batch.json`).

---

## Docker

```bash
docker build -t propminer-rtx5090 .

# Validation (default): self-test + 180s benchmark
docker run --gpus all propminer-rtx5090

# Production mining
docker run --gpus all --restart unless-stopped \
  -e PROPMINER_MODE=mine \
  -e PROPMINER_WALLET=krxYOURUSER.worker1 \
  propminer-rtx5090
```

| `PROPMINER_MODE` | Action |
|---|---|
| `full` | Self-test + 180s bench (Salad zero-config default) |
| `test` | Self-test only |
| `mine` | Pool mining until stopped |
| `tune` | Compile-time kernel knob sweep |
| `batch-tune` | Mine batch sweep |
| `cluster-tune` | `cluster_m` {1,2,4} + runtime autotune |
| `tune-prod` | Full 5090 production tune (all three) |

---

## Options (summary)

| Flag / env | Purpose |
|---|---|
| `--pool HOST:PORT` | Pool endpoint |
| `--wallet ADDRESS` | Wallet (required for mine) |
| `--worker NAME` | Worker label if not in wallet string |
| `--gpus 0,1` | GPU indices |
| `--rtx5090` | RTX 5090 profile (max N from VRAM) |
| `--bench SECONDS` | Local benchmark, no pool |
| `--self-test` | Tiny problem + first share, exit |
| `--tune-mine-batch S` | Batch sweep → cache |
| `--tune-cluster S` | Cluster sweep → cache |
| `--tune-autotune S` | Full runtime autotune → cache |
| `--config M,N,K,R` | Override GEMM dimensions |
| `--no-watchdog` | Disable stall recovery |
| `--tls 0/1` | TLS for gRPC |
| `PROPMINER_USE_TUNE_CACHE` | Apply `autotune.json` (default **1** in prod) |
| `PEARL_GEMM_CONSUMER_CLUSTER_M` | Thread-block cluster **1 / 2 / 4** (prod default **2**) |
| `PROPMINER_BATCH` | Matmuls per poll (cache or **4**) |

Full reference: [docs/README.md](docs/README.md) · Examples: [docs/examples.md](docs/examples.md)

---

## First minutes on a live pool

- **Hashrate may stay 0** until the first job (σ) is installed — at full **N=262144** the first batch can take **60–120s**. This is normal.
- **Vardiff** means the first accepted share can take a few minutes after connect.
- Run **`tune-prod` once** on each 5090 box for best steady-state hashrate.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No CUDA devices` | Install driver 545+; pass `--gpus all`; on WSL2 check `/dev/dxg` |
| `no kernel image` / illegal instruction | Wrong arch binary — rebuild with matching `PEARL_GEMM_ARCH` |
| Hashrate 0 for 1–2 min (mine) | Wait for first σ + full-N batch |
| Bench reports 0 H/s | Use `PROPMINER_BENCH_BATCH=1`; increase `PROPMINER_BENCH_GRACE_SECONDS` |
| Shares rejected (`claimed_hash_mismatch`) | Align `PEARL_GEMM_CONSUMER_CLUSTER_M` with tune cache; try `=1` |
| Built knobs ≠ cached winner | Re-run `PROPMINER_MODE=tune` or `./scripts/tune_blackwell_knobs.sh` |
| Wallet too short / rejected | Use full `krx…` or `prl1p…` address with `.worker` suffix |
| Pool connect / TLS failed | Check host:443, firewall, `--tls 1` |

---

## Repository layout

```
PropMiner/
├── src/host/              # C++ miner (orchestrator, GPU workers, gRPC, shares)
├── third_party/
│   ├── pearl-gemm/        # CUDA NoisyGEMM + BLAKE3 tensor hash
│   ├── pearl-mining-capi/ # Rust BLAKE3 Merkle C API
│   └── pearl-blake3/
├── scripts/               # build, mine, tune, Docker entrypoint
├── docs/                  # User guide + RTX 5090 blueprint
└── performance optimizations/  # Design notes (01–08)
```

**Stack:** CUDA 12.8 · CUTLASS 4.4 · BLAKE3 · Rust cdylib · OpenSSL TLS · C++20 · CMake + Make

**Deep dives:** [docs/RTX5090_BLUEPRINT.md](docs/RTX5090_BLUEPRINT.md) · [performance optimizations/](performance%20optimizations/)

---

## License

See repository license. Pearl protocol and pool terms are governed by the pool operator.
