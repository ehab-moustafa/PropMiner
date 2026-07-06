# PropMiner — Examples

Copy-paste recipes for common PropMiner workflows. All production mining examples assume an **RTX 5090** and **Kryptex gRPC** unless noted.

Keep commands on **one line** when pasting into terminals, HiveOS custom fields, or cloud startup scripts.

---

## Production mining

**Single GPU — direct binary:**

```bash
./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --pool prl.kryptex.network:443 --gpus 0
```

**Single GPU — production script (restart loop + logs + aggressive defaults):**

```bash
PROPMINER_WALLET=krxYOURUSER.worker1 ./scripts/run_mining.sh
```

**EU pool endpoint:**

```bash
./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --pool prl-eu.kryptex.network:443 --gpus 0
```

**Pearl `prl1p` address with separate worker name:**

```bash
./build/propminer --rtx5090 --wallet prl1pYOURADDRESS --worker rig01 --gpus 0
```

**Multi-GPU (one process, all cards):**

```bash
./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0,1
```

---

## Docker

**Zero-config validation (Salad default):**

```bash
docker run --gpus all propminer-rtx5090
```

**Production mine:**

```bash
docker run --gpus all --restart unless-stopped \
  -e PROPMINER_MODE=mine \
  -e PROPMINER_WALLET=krxYOURUSER.worker1 \
  propminer-rtx5090
```

**Production mine — EU pool, GPU 0 only:**

```bash
docker run --gpus all --restart unless-stopped \
  -e PROPMINER_MODE=mine \
  -e PROPMINER_WALLET=krxYOURUSER.worker1 \
  -e PROPMINER_POOL=prl-eu.kryptex.network:443 \
  -e PROPMINER_GPUS=0 \
  propminer-rtx5090
```

**Persist tune caches across container restarts:**

```bash
docker run --gpus all --restart unless-stopped \
  -v propminer-cache:/root/.cache/propminer \
  -e PROPMINER_MODE=mine \
  -e PROPMINER_WALLET=krxYOURUSER.worker1 \
  propminer-rtx5090
```

---

## One-time RTX 5090 tuning (recommended)

**Full production tune pipeline (knobs → cluster → batch):**

```bash
./scripts/tune_prod_5090.sh
```

**Same via Docker:**

```bash
docker run --gpus all \
  -v propminer-cache:/root/.cache/propminer \
  -e PROPMINER_MODE=tune-prod \
  propminer-rtx5090
```

**Individual tune steps:**

```bash
# 1. Compile-time kernel knobs (rebuild + self-test gate)
./scripts/tune_blackwell_knobs.sh 15 3

# 2. cluster_m {1,2,4} + runtime autotune
./scripts/tune_cluster_sweep.sh 20 3 12

# 3. Mine batch sweep at production N
./scripts/tune_mine_batch.sh 12 2
```

Then mine — caches apply automatically (`PROPMINER_USE_TUNE_CACHE=1` default in `run_mining.sh`).

---

## Benchmark & self-test

**Quick self-test (tiny shape, fast):**

```bash
./build/propminer --self-test --rtx5090 --gpus 0
```

**Self-test at production shape (full N from VRAM):**

```bash
PROP_MINER_SELF_TEST_PROD=1 ./build/propminer --self-test --rtx5090 --gpus 0
```

**180-second local hashrate benchmark (no pool):**

```bash
./build/propminer --bench 180 --rtx5090 --gpus 0
```

**Short bench with safe batch (WSL2 / Salad window):**

```bash
PROPMINER_BENCH_BATCH=1 ./build/propminer --bench 180 --rtx5090 --gpus 0
```

**Docker validation kit:**

```bash
# Full: self-test + 180s bench
docker run --gpus all -e PROPMINER_MODE=full propminer-rtx5090

# Quick: self-test only
docker run --gpus all -e PROPMINER_MODE=test propminer-rtx5090
```

---

## Performance overrides

**Disable thread-block clustering (conservative):**

```bash
PEARL_GEMM_CONSUMER_CLUSTER_M=1 \
  ./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0
```

**Force cluster size 4:**

```bash
PEARL_GEMM_CONSUMER_CLUSTER_M=4 \
  ./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0
```

**Pin mine batch (skip cache):**

```bash
PROPMINER_BATCH=8 \
  ./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0
```

**Skip tune cache (use env defaults only):**

```bash
PROPMINER_USE_TUNE_CACHE=0 \
  ./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0
```

**Live runtime autotune at startup (may shrink N — not recommended for max hashrate):**

```bash
PROPMINER_AUTOTUNE=1 \
  ./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0
```

**Experimental deferred share GPU thread:**

```bash
PROPMINER_DEFER_SHARE_GPU=1 \
  ./build/propminer --rtx5090 --wallet krxYOURUSER.worker1 --gpus 0
```

---

## Build examples

**Default RTX 5090 build:**

```bash
./scripts/build.sh
```

**Ada (RTX 4090) build:**

```bash
PROP_MINER_CUDA_ARCH=ada ./scripts/build.sh
```

**Ampere build:**

```bash
PROP_MINER_CUDA_ARCH=ampere ./scripts/build.sh
```

**Blackwell with tuned compile knobs (after knob sweep):**

```bash
PEARL_GEMM_BLACKWELL_KBLOCK=128 \
PEARL_GEMM_BLACKWELL_STAGES=2 \
PEARL_GEMM_BLACKWELL_SWIZZLE_BITS=3 \
PEARL_GEMM_BLACKWELL_MIN_BLOCKS=1 \
PEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async \
./scripts/build.sh
```

**Host-only correctness tests (no GPU):**

```bash
./scripts/local_host_tests.sh
```

---

## Cloud marketplace patterns

**Salad / WSL2 — validate then switch to mine:**

```bash
# Deploy with PROPMINER_MODE=full first (default image env)
# After PASS, redeploy or override:
docker run --gpus all \
  -e PROPMINER_MODE=mine \
  -e PROPMINER_WALLET=krxYOURUSER.worker1 \
  propminer-rtx5090
```

**vast.ai — single GPU contract:**

```bash
PROPMINER_GPUS=0 PROPMINER_WALLET=krxYOURUSER.worker1 ./scripts/run_mining.sh
```

**systemd service (bare metal):**

```ini
[Unit]
Description=PropMiner Pearl GPU miner
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/PropMiner
Environment=PROPMINER_WALLET=krxYOURUSER.worker1
Environment=PROPMINER_GPUS=0
ExecStart=/opt/PropMiner/scripts/run_mining.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Offline tune CLI (no pool)

**Mine batch sweep only:**

```bash
./build/propminer --tune-mine-batch 12 --rtx5090 --gpus 0
```

**Cluster sweep only:**

```bash
./build/propminer --tune-cluster 20 --rtx5090 --gpus 0
```

**Full runtime autotune:**

```bash
./build/propminer --tune-autotune 15 --rtx5090 --gpus 0
```

---

## Non-5090 mining (generic profile)

**Auto shape from GPU VRAM (no `--rtx5090`):**

```bash
./build/propminer --wallet krxYOURUSER.worker1 --gpus 0
```

**Manual dimensions:**

```bash
./build/propminer --config 4096,16384,128,128 --wallet krxYOURUSER.worker1 --gpus 0
```

> The prebuilt Docker image targets **sm_120a** only. For Ampere/Ada hosts, build natively with the matching `PROP_MINER_CUDA_ARCH`.
