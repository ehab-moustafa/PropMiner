# GPU Mining Mechanics and Hashrate Scaling

**PropMiner Research Note · Pearl Noisy GEMM · July 2026**

An educational synthesis of how GPU-based proof-of-work mining works end-to-end, why hashrate scales the way it does, and which optimization levers are realistic for Pearl-style noisy GEMM workloads. Grounded in PropMiner's `GpuWorker` architecture and contemporary CUDA/GEMM literature (2025–2026).

---

## Executive Summary

GPU mining is not "more threads = more hashes." It is a **pipeline** that maps a vast nonce space onto massively parallel matrix work, then filters rare winners through a cryptographic target check. Throughput is bounded by whichever stage is slowest: tensor-core GEMM compute, memory bandwidth, register/shared-memory capacity, CPU launch overhead, or (at the wall) power and thermals.

For Pearl noisy GEMM on PropMiner:

| Metric | What it measures | Typical RTX 5090 scale |
|--------|------------------|------------------------|
| **Iterations/s** | Complete noisy GEMM + transcript + PoW checks per second | ~0.3–2 iters/s (depends on N, batch) |
| **TMAD/s** | Trillion int8 MAC-equivalents per second | ~300 TMAD/s (consumer `mma.sync` path) |
| **Hashrate (H/s)** | Protocol-normalized tiles/s via DAF | ~1B+ H/s class (order of magnitude) |
| **J/hash** | Energy per normalized hash | Dominated by TDP when power-bound |

**Key insight:** PropMiner reports hashrate as `tiles_per_sec × difficulty_adjustment_factor()`. At fixed TMAD/s, H/s is approximately **N-invariant**—doubling matrix width N doubles tiles per matmul but halves matmuls per second. Chasing 100× hashrate without changing the algorithm usually means misunderstanding these units, not finding a hidden compiler flag.

Realistic sustained gains for GEMM mining sit in the **2–5×** range, assembled from orthogonal levers: native tensor ISA (`tcgen05` vs legacy `mma.sync`), batch size + CUDA graphs, multi-stream host pipelining, tile-shape tuning, and power-cap optimization—not from "turning up occupancy" alone.

---

## 1. End-to-End: How GPU Mining Works

### 1.1 The mining loop as a state machine

Pearl mining on PropMiner follows a repeating cycle on each GPU:

```
σ-install (job setup)
    → batch queue (N matmuls with distinct nonces)
    → noisy GEMM + transcript accumulation (device)
    → per-tile PoW target check (device, headless path)
    → host scan of signal headers (rare winners)
    → share reconstruction + pool submit (rare)
```

The host orchestrator is `GpuWorker` (`src/host/pearl/gpu_worker.cpp`). Each GPU owns:

- **Ping/pong half-buffers** — two complete device workspaces so one batch computes while the host prepares the next.
- **A partitioned nonce space** — `seed_base_` combines `gpu_index` (top 16 bits) and time entropy so multi-GPU rigs mine disjoint ranges without coordination.
- **Resident B-side state** — matrix B, noise, and Merkle metadata for the current σ (sigma) are installed once per job and reused across thousands of iterations.

Per batch, the worker:

1. Uploads the next seed on a **dedicated copy stream** while the previous batch runs.
2. Launches a batched GEMM graph (or fallback `iter_batch`) on a **compute stream**.
3. Spin-waits on `cudaEventQuery` (no blocking sleep on the hot path).
4. Scans pinned host headers for `status == 1` (PoW hit).
5. Optionally defers share GPU work to a side thread so mining never stalls on proof reconstruction.

This is classic **throughput mining**: the GPU is a nonce-evaluation engine; the CPU is a thin scheduler.

### 1.2 What one "iteration" actually computes

Each iteration (one nonce) in Pearl V2 performs:

1. **Noise generation** — LCG-derived int7 noise applied to matrix A (and sparse rank-R noise structures).
2. **INT8 GEMM** — `C = A_noisy × B_noisy` at shape M×N×K (default M=8192, N up to 262144, K=128 on RTX 5090 production).
3. **Transcript accumulation** — per-tile XOR-rotate reduction over MMA fragments; consensus-critical, byte-identical to the network reference.
4. **Committed hash-tile PoW** — only a periodic subset of MMA tiles (rows/cols patterns) are hashed and compared to the target.
5. **BLAKE3 / tensor-hash paths** — Merkle commitments for A/B; triggered on share found, not every iteration.

The older `pearlhash_kernel.cu` persistent kernel documents the same logical pipeline in miniature: `LCG noise → INT8 GEMM (mma.sync) → XOR reduction → BLAKE3 → target check`, with `__launch_bounds__(256, 3)` to cap register pressure.

### 1.3 Where time goes (first-order model)

For a tuned RTX 5090 at production N=262144:

| Stage | Share of wall time | Bottleneck character |
|-------|-------------------|----------------------|
| Noisy GEMM (tensor cores) | ~85–95% | Compute / tensor pipe utilization |
| Host launch + scan | ~2–10% (without graphs) | CPU/driver; graphs collapse this |
| σ-install / graph capture | One-time per job | Minutes on first batch only |
| Share reconstruction | <<1% normally | PCIe D2H on rare events |

The miner is **GEMM-bound** once graphs and batching are enabled. Further gains require faster GEMM, not faster BLAKE3.

---

## 2. Nonce Space, Parallelism, and Search Coverage

### 2.1 Nonce space structure

A nonce is not a single 32-bit counter. In PropMiner:

```cpp
seed_base_ = (gpu_index << 48) | ((now_ms() & 0xFFFF) << 32);
// Per batch: seed_base_ + global_iter + winner_index
```

- **Per-GPU partition** — top bits encode `gpu_index`, preventing collisions across a fleet.
- **Per-batch stride** — each batch advances `global_iter` by `matmuls_per_poll` (4–20 on RTX 5090).
- **Per-iteration offset** — `winner_index` within the batch identifies which matmul in the batch produced a hit.

The searchable space is effectively **64-bit** per GPU per σ rotation. At ~1 iter/s, a single GPU would need on the order of 10¹⁰ years to exhaust 2⁶⁴ — nonce exhaustion is not an operational concern. The practical limit is **iterations per second**, not space size.

### 2.2 Parallelism hierarchy

| Level | Mechanism | Pearl/PropMiner example |
|-------|-----------|-------------------------|
| **Fleet** | One process thread per GPU, disjoint nonce bases | `WorkerOrchestrator` + N × `GpuWorker` |
| **Grid** | Thousands of CTAs per GEMM | 65,536 CTAs at N=262144 (8192/128 × 262144/256) |
| **CTA** | 256 threads, 8 warps, one output tile | 128×256×128 consumer tile |
| **Warp** | `mma.sync m16n8k32` tensor atom | 4 K-subblocks per K-tile |
| **Batch** | Multiple nonces per graph launch | `matmuls_per_poll` = 4–20 |

**Important:** More CTAs does not mean more hashes per second if each CTA does more work. At N=262144 vs N=32768, the grid is 8× larger but each matmul takes ~8× longer — TMAD/s stays flat, and DAF-normalized H/s stays flat.

### 2.3 Occupancy vs useful parallelism

CUDA occupancy is the ratio of active warps to hardware warp slots per SM. It is a **resource accounting** metric, not a throughput guarantee.

On an A100-class part, 65,536 registers per SM divided among 2,048 threads at full occupancy implies a **32 register/thread ceiling** ([Manish Aradwad, CUDA Occupancy](https://medium.com/@manisharadwad/unlocking-gpu-potential-understanding-and-optimizing-cuda-occupancy-2f43ee01ad7e)). Exceed that and the SM runs fewer concurrent warps.

PropMiner's consumer GEMM kernel holds **128 int32 accumulators per thread** plus transcript state — deliberately **low occupancy, high per-warp work**. This mirrors modern HGEMM wisdom: peak tensor throughput often comes from fat register tiles and deep pipelines, not from filling every warp slot ([Cloudrift GPU Matmul Optimization](https://www.cloudrift.ai/blog/gpu-matmul-optimization)).

NVIDIA's guidance: use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to tell the compiler your concurrency target ([CUDA Programming Guide §5.4.3.2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)). PropMiner's legacy kernel uses `__launch_bounds__(256, 3)` — targeting 3 blocks/SM at 256 threads, capping registers near 64/thread.

**Takeaway for miners:** "Raise occupancy" is sometimes the wrong lever. If the kernel is tensor-pipe-bound, adding idle warps does not help. Profile with Nsight Compute's `ncu_occupancy` module ([Nsight Compute 13.0 Occupancy Calculator](https://docs.nvidia.com/nsight-compute/2025.3/OccupancyCalculatorPythonInterface/index.html)) and look for **MemoryWorkloadAnalysis** and **TensorActive** before chasing occupancy.

---

## 3. Memory Walls

GPU mining hits three distinct memory walls.

### 3.1 HBM capacity wall (VRAM)

Pearl keeps **B resident** across iterations (noise, scales, Merkle roots) while A is regenerated per nonce. VRAM scales roughly as:

```
bytes ≈ M×K + K×N + overhead(M,N) + ping/pong buffers
```

PropMiner's `pick_n_for_vram()` selects the largest N that fits after a 512 MiB driver reserve. On 32 GB RTX 5090, production targets **N=262144** (~14–18 GB resident + ping/pong) vs bench cap **N=32768** (~3–5 GB). Larger N improves **work per matmul** but not H/s at fixed TMAD/s.

### 3.2 Bandwidth wall (roofline)

GEMM arithmetic intensity scales with K. At K=128, Pearl's int8 GEMM is often **memory-latency sensitive** on the load path (A tile streaming, B resident) even when tensor pipes are busy. Optimization patterns:

- **`cp.async` pipelining** — 2-stage global→shared prefetch ([WingEdge777 HGEMM tutorial](https://www.wingedge777.com/en/article/5a219c62549f9573)).
- **Swizzled shared memory** — XOR address remapping to kill bank conflicts on `ldmatrix` ([tensor-core-from-scratch](https://github.com/waynehacking8/tensor-core-from-scratch)).
- **L2 fetch granularity** — PropMiner sets `cudaLimitMaxL2FetchGranularity` to 128 bytes for sequential GEMM traffic on GDDR7.

The [roofline model](https://www.cloudrift.ai/blog/gpu-matmul-optimization) applies directly: if you're below the bandwidth ridge point, faster memory movement wins; above it, more MMA issue wins.

### 3.3 Host–device transfer wall (PCIe)

Mining iterations should move **only the seed** (8 bytes) per batch, not matrices. PropMiner's "PCIe Gen5 conveyor belt":

```cpp
upload_next_seed_async(*other, seed_base_ + global_iter + batch);  // copy stream
// ... while cur half runs GEMM on compute stream ...
cudaStreamWaitEvent(compute_stream, seed_copy_done_event_);
```

Share events are rare; when they fire, pinned staging (`pinned_leaf_cvs`, `pinned_a_slice`) bounds D2H to proof-sized slices, not full A.

### 3.4 Shared memory / register wall

The consumer kernel uses ~128 KiB accumulators in registers per thread group plus ~16–32 KiB shared memory for A/B tiles. This **limits blocks/SM** and is the real occupancy limiter — not thread count. CUDA 13's shared-memory register spilling (`enable_smem_spilling`) can trade register pressure for smem when launch bounds are explicit ([NVIDIA Technical Blog, 2025](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)).

---

## 4. Tensor Cores vs CUDA Cores for Pearl-Like Workloads

### 4.1 Why CUDA cores lose

Pearl's hot path is **INT8 matrix multiply at scale** (billions of MACs per iteration). CUDA cores execute scalar/vector FMA at FP32 rates orders of magnitude below tensor cores. A CPU-style nested-loop GEMM on CUDA cores would be 10–50× slower than even a naive WMMA kernel.

For reference, optimized FP16 HGEMM on tensor cores reaches 83–105% of cuBLAS; the same effort on CUDA cores cannot approach roofline ([tensor-core-from-scratch Kernel 10 vs 06](https://github.com/waynehacking8/tensor-core-from-scratch)).

### 4.2 Tensor core generations matter

| ISA | Instruction | Typical use | Pearl PropMiner status |
|-----|-------------|-------------|------------------------|
| **Ampere-class** | `mma.sync m16n8k32` | INT8 GEMM + `ldmatrix` | **Current RTX 5090 path** (`SM80_16x8x32`) |
| **Hopper** | `wgmma` warpgroup MMA | Higher % of peak on H100 | B200 reference, not consumer |
| **Blackwell datacenter** | `tcgen05.mma` + TMEM | B200 transcript kernel | Port planned (+30–80% TMAD/s est.) |
| **Blackwell consumer** | `mma.sync` (not tcgen05) | GB202 games/workstation | Native tcgen05 not exposed same as B200 |

Critical 2025 finding: **consumer Blackwell (sm_120) uses Ampere-era `mma.sync`**, not datacenter `tcgen05`/TMEM ([tensor-core-from-scratch sm_120 note](https://github.com/waynehacking8/tensor-core-from-scratch)). PropMiner's `01-native-tcgen05-tmem-gemm.md` plans a port from the B200 path, but correctness requires byte-identical transcripts — a high-risk ISA migration.

Microbenchmark evidence on Hopper: `mma.sync` reaches ~63–65% of tensor peak vs ~96% for `wgmma` ([blackwell-tensorcore-kernels / arXiv:2501.12084](https://github.com/waynehacking8/blackwell-tensorcore-kernels)). Instruction-class upgrades can yield **~10×** on large dense GEMM — but Pearl is not a pure GEMM; transcript + PoW constrain the kernel design.

### 4.3 What tensor cores do in Pearl

Per 128×256 output tile per CTA:

1. `cp.async` loads int8 A/B fragments into swizzled shared memory.
2. `ldmatrix` moves fragments into warp registers.
3. `mma.sync m16n8k32` accumulates into int32 (`tCrC`, 128 elements/thread).
4. Transcript XOR-rotate runs on register fragments each K-slab.
5. Periodic tiles undergo PoW hash comparison against `pow_target` on device.

CUDA cores still matter for **epilogue logic** (hashing, indexing, LCG noise setup), but they are not the throughput engine.

---

## 5. Batch Pipelining, Multi-Stream, and CUDA Graphs

### 5.1 Why host overhead matters

Each iteration launches dozens of GPU kernels (noise, GEMM slices, hash tiles, sync). At ~1–5 ms per iter, **CPU launch latency** (historically 5–20 µs per kernel) compounds. CUDA Graphs record the full kernel DAG once and replay with a single `cudaGraphLaunch` ([PyTorch CUDA Graphs blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)).

NVIDIA reports repeat-launch CPU overhead for straight-line graphs dropped to ~2.5 µs + ~1 ns/node on Ampere+ between CUDA 11.8 and 12.6 ([NVIDIA Technical Blog, Constant Time Launch](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)).

PropMiner captures an **extended graph** per half-buffer:

```cpp
gemm_.iter_batch_graph_prepare_ex(workspace, stream, headers, batch, seed_dev_ptr);
// ...
gemm_.iter_batch_graph_launch_ex(workspace, stream);
```

The seed pointer is **outside** the captured graph — uploaded via `seed_copy_stream_` so each replay only moves 8 bytes.

### 5.2 Batch size (`matmuls_per_poll`)

Batching amortizes one graph launch over N nonces. PropMiner sweeps `{1,2,4,6,8,10,12,16,20}` (`Rtx5090Profile::kMineBatchCandidates`). Too small → launch overhead dominates; too large → share latency and VRAM pressure rise.

Kernel-batching research shows **>1.4×** speedups from optimal batch unrolling into graphs ([arXiv:2501.09398](https://doi.org/10.48550/arxiv.2501.09398)). PropMiner's offline tuner (`GpuTuner`) benchmarks batch × graph × cluster_m combinations.

### 5.3 Multi-stream ping-pong

Three streams cooperate:

| Stream | Role |
|--------|------|
| `ping_.stream` / `pong_.stream` | Alternate compute batches |
| `seed_copy_stream_` | Async seed H2D for next batch |
| `merkle_copy_stream_` | σ-install / B-side setup |

Synchronization is **event-driven** (`seed_copy_done_event_`, `batch_done_event_`), not `cudaDeviceSynchronize` on the hot path. This is textbook **double buffering** applied to mining.

### 5.4 Dynamic parameters without recapture

When batch size changes, PropMiner re-captures graphs in `install_sigma()`. For fixed production shapes, graphs persist across millions of replays. NVIDIA's hybrid explicit-API + capture pattern avoids full recapture for dynamic kernel params ([Constructing CUDA Graphs with Dynamic Parameters](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)).

**Caveat:** Parameter-copy overhead can eat 17–24% of graph replay time in some ML workloads ([PyGraph, arXiv:2503.19779](https://arxiv.org/html/2503.19779v3)). PropMiner sidesteps large param copies by keeping matrix pointers stable and only changing the seed.

---

## 6. Hashrate vs Generations / Iterations per Second

These terms are often conflated. In PropMiner they are precisely defined:

### 6.1 Iterations per second (`ips`)

```cpp
const double ips = batch / batch_time_seconds;
total_iters_ += batch;
```

One **iteration** = one nonce = one full noisy GEMM + transcript + in-kernel PoW scan.

At ~300 TMAD/s and M×N×K = 8192×262144×128, one iteration is ~0.27 trillion MACs → **~0.3–1 iters/s** depending on efficiency and batch idle time.

### 6.2 TMAD/s (compute throughput)

```cpp
const double tmads = M * N * K * ips / 1e12;
tmads_per_sec_.store(tmads);
```

TMAD/s is the **hardware-neutral** figure for GEMM mining intensity. Pool wire protocol may convert via `kTmadsToHashesPerSec = 1e12` in `WorkerOrchestrator`.

### 6.3 Hashrate H/s (protocol-normalized)

```cpp
const double tiles_per_iter = (M / bM) * (N / bN);
const double tiles_per_sec = ips * tiles_per_iter;
const double hr = tiles_per_sec * difficulty_adjustment_factor();
hashrate_.store(hr);
```

Where:

```
DAF = rows_pattern.size × cols_pattern.size × dot_product_length()
```

**DAF-normalized H/s** is what pools and dashboards quote. It counts **committed hash tiles per second**, not raw nonces, aligning miner output with Pearl's periodic PoW pattern (default 2 row groups × 64 col groups × K dot length).

### 6.4 Why H/s does not scale with N

| N | CTAs/iter | Time/iter @ flat TMAD/s | Tiles/iter | Tiles/s | H/s (×DAF) |
|---|-----------|-------------------------|------------|---------|------------|
| 32,768 | 8,192 | 1× | 1× | 1× | **~same** |
| 262,144 | 65,536 | ~8× | 8× | 1× | **~same** |

This is documented in PropMiner's `03-production-n262144.md`. Production N=262144 maximizes **work per pool round** and VRAM utilization, not headline H/s.

### 6.5 "Generations"

In pool/miner UI, "generations" sometimes means **σ rotations** (new jobs) or **batch completions**, not nonces. When comparing miners, always ask: are we measuring **ips**, **TMAD/s**, **DAF-H/s**, or **shares per hour**? Only the first three are directly comparable at fixed difficulty.

---

## 7. Power Efficiency (J/hash) — The Eco Angle

### 7.1 Definitions

| Term | Formula | Units |
|------|---------|-------|
| **Power** | Wall draw or GPU TDP cap | W (J/s) |
| **Energy per hash** | Power / hashrate | J/H |
| **Efficiency** | Hashrate / Power | H/J or MH/s/W |

[Watts = joules per second](https://hashrateindex.com/blog/energy-consumption-to-hashrate/). A miner drawing 500 W at 1 GH/s consumes **500 nJ/hash** (0.5 µJ/hash).

### 7.2 When power becomes the bottleneck

The Keryx PoW miner case study on RTX 5090 is instructive ([keryx-miner-supr](https://github.com/ocminer/keryx-miner-supr)):

- Keccak kernel unroll: 2.57 → **3.28 GH/s** (+28%) via register reduction (229 → 64 regs, 1 → 2 blocks/SM).
- After tuning, the card became **power-bound at 575 W TDP** — further kernel wins require **lower J/hash**, not higher peak clocks.

PropMiner's Pearl path at ~300 TMAD/s on a 5090 similarly tends toward **power/thermal saturation** once GEMM graphs are enabled. The eco-optimal operating point is rarely "max power limit."

### 7.3 Practical efficiency levers

| Lever | Effect | Eco impact |
|-------|--------|------------|
| **Power cap / undervolt** | Reduces W; hashrate may drop sub-linearly | Often improves H/J if not already memory-bound |
| **Native ISA (`tcgen05`)** | More H per J at same clocks | Best structural win when available |
| **Batch + graphs** | More H for same fixed driver/CPU overhead | Modest J/hash improvement |
| **Reject stale σ work** | Avoids wasted GEMM on expired jobs | Direct energy savings |
| **Fleet duty cycling** | Pause when electricity price > revenue | Grid-interactive demand response ([arXiv:2507.00909](https://arxiv.org/html/2507.00909v1)) |

ML literature shows GPU power capping can save **~13% energy** with **~3% performance loss** when tuned ([Adaptive GPU Power Capping, 2025](https://doi.org/10.1145/3731545.3735119)). Mining workloads with flat GEMM duty cycles are good candidates.

### 7.4 Reporting for operators

PropMiner logs `cudaEventElapsedTime` for **GPU batch time only** (excludes share reconstruction). For true J/hash:

```
J/hash ≈ (GPU power draw in watts) / (DAF-normalized H/s)
```

Measure power with `nvidia-smi` or a wall meter; do not confuse TDP with actual draw.

---

## 8. Why 100×–1000× Jumps Are Usually Impossible Without Algorithm Change

### 8.1 The hard limits

1. **Physics / Moore's law for fixed silicon** — You cannot 100× tensor throughput on the same GB202 die without a new ISA, higher clocks, or wider memory — each offers incremental, not revolutionary, gains.

2. **Protocol-fixed work per nonce** — Pearl defines exact M, N, K, noise model, transcript layout, and hash-tile patterns. You cannot "skip" 99% of GEMM and remain valid. Any shortcut that changes the committed hash is an **algorithm change** (hard fork or new coin).

3. **Correctness gate** — PropMiner's consumer kernel is proof-canonical against H100/WGMMA reference. A 100× "faster" kernel that emits wrong transcripts produces **100% reject shares** — worse than useless.

4. **Amdahl's law on the pipeline** — If GEMM is 90% of time, even an **infinite-speed** epilogue saves at most 10%. Host overhead, already minimized by graphs, cannot yield 100×.

5. **Power wall** — Once at TDP, 2× faster kernel → 2× power unless voltage/frequency scale sub-linearly. Real rigs hit **thermal throttling** first.

6. **Pool difficulty equilibrium** — Network hashrate rises until marginal cost equals reward. Personal 100× jumps are arbitraged away globally within difficulty adjustments.

### 8.2 Where 100× myths come from

| Confusion | Reality |
|-----------|---------|
| "8× bigger N → 8× H/s" | False at fixed TMAD/s (see §6.4) |
| "100× more GPUs → 100× profit" | Linear hashrate, but capital, power, and difficulty scale too |
| "Tensor cores are 100× CUDA cores" | ~10–20× for dense GEMM, less for fused PoW kernels |
| "CUDA graphs give 100×" | Typical 1.2–1.6× on launch-bound loops |
| ASIC vs GPU comparisons | ASICs win on **specialized** SHA256, not on arbitrary noisy GEMM with proofs |

### 8.3 When step-changes *do* happen

Instruction-class migration (`mma.sync` → `wgmma` / `tcgen05`) can approach **10×** on large **pure** GEMM ([blackwell-tensorcore-kernels](https://github.com/waynehacking8/blackwell-tensorcore-kernels)). Pearl absorbs only the fraction of that win that fits inside transcript + PoW constraints — PropMiner estimates **+30–80% TMAD/s** for tcgen05/TMEM on 5090, not 10×.

---

## 9. Realistic 2–5× Levers for Sustained GEMM Mining

These levers are **orthogonal** — multiply partially, not fully. A 1.3× kernel × 1.2× batch × 1.15× power tune ≈ **1.8× total**, not 1.79×.

### Lever 1: Native tensor ISA (est. 1.3–1.8× TMAD/s)

Port `tcgen05.mma` + TMEM accumulator path from B200 `transcript_gemm_sm100.cu` to `sm_120a`, preserving transcript bytes. Highest ceiling, highest correctness risk. See `performance optimizations/01-native-tcgen05-tmem-gemm.md`.

### Lever 2: CUDA graphs + optimal batch (est. 1.1–1.5× ips)

- Capture extended batch graphs (`prepare_graph` / `queue_batch`).
- Tune `matmuls_per_poll` via `GpuTuner` / `kMineBatchCandidates`.
- Keep seed upload off the critical path (`upload_next_seed_async`).

Reference: [arXiv:2501.09398 kernel batching](https://doi.org/10.48550/arxiv.2501.09398).

### Lever 3: Tile shape + memory micro-opts (est. 1.05–1.25× TMAD/s)

- `bM/bN/bK` matching hardware (128×256×128 consumer default).
- Swizzle tuning (`Swizzle<3,4,3>` vs alternatives — ~0.5% in PropMiner logs, but compounds).
- `cp.async` depth, cluster launch (`PEARL_GEMM_CONSUMER_CLUSTER_M=2`).
- L2 granularity (already 128 B in `GpuWorker` ctor).

Reference: [Cloudrift matmul ladder](https://www.cloudrift.ai/blog/gpu-matmul-optimization), [WingEdge777 HGEMM](https://www.wingedge777.com/en/article/5a219c62549f9573).

### Lever 4: Occupancy *trade-offs*, not blind maximization (est. 1.1–1.3×)

- `__launch_bounds__` to prevent register spill to local memory.
- CUDA 13 `enable_smem_spilling` when smem headroom exists ([NVIDIA blog](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)).
- Profile-guided: if limiter is `REGISTERS`, reducing block size may **hurt** — consumer GEMM wants fat tiles.

### Lever 5: Production N + VRAM fit (est. 1.0–1.05× TMAD/s, 0× H/s)

- `N=262144` fills GDDR7, improves kernel efficiency slightly (~300.8 vs ~299.2 TMAD/s measured).
- Does **not** multiply H/s; improves economics per pool job.

### Lever 6: Power / thermal optimization (est. 1.1–1.4× H/J)

- Find the **knee** of the power–hashrate curve below TDP.
- Fix blower-limited thermals (laptop SMs throttle occupancy).
- Reference: [keryx-miner power-bound 5090](https://github.com/ocminer/keryx-miner-supr).

### Lever 7: Host-side latency removal (est. 1.05–1.2×)

- Ping-pong halves + event spin-wait (no `sleep` on hot path).
- `PROPMINER_DEFER_SHARE_GPU=1` — share proof on side thread.
- Pinned headers for zero-copy winner scan.

### Combined realistic ceiling

| Scenario | Approximate sustained gain |
|----------|---------------------------|
| Graphs + batch tune only | 1.2–1.5× |
| + tcgen05/TMEM port (successful) | 1.5–2.5× vs today |
| + power knee + cluster tuning | 1.8–3.5× |
| Theoretical stacked max | **~4–5×** before diminishing returns |

Beyond ~5× on **fixed Pearl V2** without algorithm change requires new hardware generation (e.g., Rubin) or proof simplification — not another week of register tweaking.

---

## 10. PropMiner Architecture Map (Quick Reference)

```
WorkerOrchestrator
  └── GpuWorker (per GPU)
        ├── ping_ / pong_ HalfBuffers
        │     ├── workspace (pearl_capi)
        │     ├── CUDA graph (iter_batch_graph_*_ex)
        │     ├── host_headers[] (pinned, PoW signals)
        │     └── batch_{start,done}_event
        ├── seed_copy_stream_ + pinned_seed_host_
        ├── merkle_copy_stream_ (σ-install)
        ├── gemm_ (GemmCapi) + mining_ (MiningCapi)
        └── Metrics: tmads_per_sec_, hashrate_ (DAF tiles/s)
```

**Nonce flow:** `seed_base_ + global_iter` → batch → `batch_seed_start + winner_index` on hit.

**Metric flow:** `batch_time` → `ips` → `TMAD/s` → `tiles/s` → `H/s = tiles/s × DAF`.

---

## 11. Further Reading

### CUDA occupancy and profiling
- [CUDA Programming Guide — Launch Bounds](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- [Nsight Compute Occupancy Calculator (Python)](https://docs.nvidia.com/nsight-compute/2025.3/OccupancyCalculatorPythonInterface/index.html)
- [Understanding CUDA Occupancy (Medium, 2025)](https://medium.com/@manisharadwad/unlocking-gpu-potential-understanding-and-optimizing-cuda-occupancy-2f43ee01ad7e)
- [Shared Memory Register Spilling — NVIDIA Blog](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)

### GEMM / tensor core optimization
- [Modern GPU Matmul Optimization — Cloudrift](https://www.cloudrift.ai/blog/gpu-matmul-optimization)
- [HGEMM Beating cuBLAS — WingEdge777](https://www.wingedge777.com/en/article/5a219c62549f9573)
- [Blackwell Tensor Core Kernels — waynehacking8](https://github.com/waynehacking8/blackwell-tensorcore-kernels)
- [Tensor Core From Scratch — waynehacking8](https://github.com/waynehacking8/tensor-core-from-scratch)

### CUDA graphs and batching
- [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Constant-Time Graph Launch — NVIDIA Blog](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/)
- [Kernel Batching with CUDA Graphs — arXiv:2501.09398](https://doi.org/10.48550/arxiv.2501.09398)
- [CUDA Graphs with Dynamic Parameters — NVIDIA Blog](https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/)
- [PyGraph — Parameter Indirection — arXiv:2503.19779](https://arxiv.org/html/2503.19779v3)

### Power efficiency
- [Energy Consumption to Hashrate — Hashrate Index](https://hashrateindex.com/blog/energy-consumption-to-hashrate/)
- [Adaptive GPU Power Capping — 2025](https://doi.org/10.1145/3731545.3735119)
- [Grid-Interactive Data Centers — arXiv:2507.00909](https://arxiv.org/html/2507.00909v1)
- [Keryx Miner RTX 5090 Power-Bound Case Study](https://github.com/ocminer/keryx-miner-supr)

### PropMiner internal docs
- `src/host/pearl/gpu_worker.cpp` — ping-pong, graphs, metrics
- `performance optimizations/01-native-tcgen05-tmem-gemm.md` — ISA migration plan
- `performance optimizations/03-production-n262144.md` — N vs H/s semantics
- `src/host/pearl/rtx5090_profile.h` — batch candidates, tile geometry

---

## Glossary

| Term | Definition |
|------|------------|
| **σ (sigma)** | Job seed vector; rotation triggers re-install of B-side state |
| **DAF** | Difficulty Adjustment Factor; scales tiles/s to protocol H/s |
| **TMAD** | Trillion (10¹²) multiply-accumulate operations (int8 MACs) |
| **CTA** | CUDA thread block; one output tile group |
| **Transcript** | Byte-identical MMA accumulation trace required for proofs |
| **Headless PoW** | In-kernel target check without materializing full C matrix |
| **Ping-pong** | Double-buffered device workspaces alternating compute |

---

*This document is educational synthesis for PropMiner operators and kernel developers. It does not modify consensus rules or pool protocols.*
