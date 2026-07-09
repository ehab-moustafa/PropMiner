# CUDA Kernel Optimization Analysis for PropMiner on RTX 5090

**Date:** 2026-07-09
**GPU:** NVIDIA GeForce RTX 5090 (GB202, SM 120a / CC 12.0)
**Current Hashrate:** ~290 TMAD/s
**Target Hashrate:** 700-800+ TMAD/s (~2.5-3x improvement)

---

## Executive Summary

PropMiner uses a hybrid architecture: the core mining computation flows through the **Pearl GEMM library** (CUTLASS-based), while a **consumer persistent kernel** (`pearlhash_kernel.cu`) handles per-nonce BLAKE3 PoW verification. The GEMM kernel is the dominant throughput engine. Key findings:

1. **The GEMM kernel is already well-optimized** — it uses CUTLASS with TMA, cp.async, swizzled shared memory, and SM80 mma.sync atoms. The bottleneck is not the kernel compute itself but the **host-side pipeline overhead** (batch size, CUDA graphs, sigma installation stalls).
2. **The consumer persistent kernel (`pearlhash_kernel.cu`) is the primary optimization target** — it runs per-nonce BLAKE3 with no batching, no GEMM, and high register pressure. It is compute-bound on BLAKE3.
3. **The biggest gains come from architectural changes**: increasing batch size, enabling grouped GEMM, and reducing sigma installation stalls rather than rewriting the GEMM kernel.

---

## 1. Current Kernel Architecture Analysis

### 1.1 Two-Kernel Architecture

PropMiner uses **two distinct kernel paths**:

**Path A — Pearl GEMM (primary mining engine):**
- **Kernel:** `transcript_gemm_kernel_consumer` in `consumer/transcript_gemm_kernel.cu`
- **Variant v1:** `transcript_gemm_sm120_geforce.cu` — warp-specialized TMA + IMMA
- **Variant v2:** `transcript_gemm_sm120_geforce_v2.cu` — PipelineTmaAsync, no CTA sync per K-tile
- **Launch:** Via `iter_batch` / `iter_batch_graph_launch_ex` through the CAPI workspace
- **Grid:** `(M/kBM, N/kBN, batch)` where M=8192, N=262144, batch=1-32
- **Block:** 256 threads (8 warps) for consumer; 288 threads (8 warps + 1 producer) for GeForce variants
- **Shared Memory:** ~48 KiB per block (A: 16 KiB/stage, B: 32 KiB/stage, 2 stages)
- **MMA Atom:** `SM80_16x8x32_S32S8S8S32_TN` (INT8 → INT32)
- **Pipeline:** cp.async with 2-3 stages, or TMA with 2 stages

**Path B — Consumer Persistent Kernel (per-nonce PoW):**
- **Kernel:** `propminer_persistent_kernel` in `src/cuda/kernels/pearlhash_kernel.cu`
- **Launch:** `__launch_bounds__(256, 3)` — max 256 threads/block, min 3 blocks/SM
- **Block:** 256 threads (8 warps)
- **Shared Memory:** 8 KiB (A: 4096 bytes, B: 4224 bytes with padding)
- **Tile:** 64x64x64 (smaller than GEMM's 128x256x128)
- **MMA Atom:** `mma.sync.m16n8k32` (inline PTX)
- **Per-nonce work:** LCG noise → INT8 GEMM (64x64x64) → XOR reduction → BLAKE3 → target check

### 1.2 Host-Side Pipeline

```
Host Thread
    │
    ├── [ping stream]  iter_batch / graph_launch  →  GEMM + transcript
    ├── [pong stream]  iter_batch / graph_launch  →  GEMM + transcript
    └── [third stream] (triple-buffer mode)       →  GEMM + transcript
         │
         ├── cudaEventRecord(batch_done_event)
         ├── cudaEventQuery spin-wait
         ├── scan_winners() (check host_signal_header)
         └── if hit: process_share_trigger() → recompute A + merkle proof
```

**Key host-side parameters:**
- `matmuls_per_poll_` (batch size): default 1, configurable via `PROPMINER_BATCH`
- `graph_batch_`: default 1, configurable via `PROPMINER_GRAPH_BATCH`
- Triple-buffer: 3 streams when `PROPMINER_TRIPLE_BUFFER=1` and VRAM allows
- Async sigma install: background thread pre-installs next sigma

### 1.3 Shared Memory Budget Per Block

**Consumer kernel (transcript_gemm_kernel_consumer):**
```
kBK=64, kStages=2:
  A: kBM * kBK * kStages = 128 * 64 * 2 = 16 KiB
  B: kBN * kBK * kStages = 256 * 64 * 2 = 32 KiB
  Total: 48 KiB + pipeline storage

kBK=128, kStages=2:
  A: 128 * 128 * 2 = 32 KiB
  B: 256 * 128 * 2 = 64 KiB
  Total: 96 KiB + pipeline storage

kBK=128, kStages=3:
  A: 128 * 128 * 3 = 48 KiB
  B: 256 * 128 * 3 = 96 KiB
  Total: 144 KiB + pipeline storage
```

**Consumer persistent kernel (pearlhash_kernel.cu):**
```
smem_A: 64 * 64 = 4096 bytes
smem_B: (64+1) * 64 = 4224 bytes (bank-conflict padding)
Total: 8320 bytes (~8 KiB)
```

---

## 2. Occupancy Analysis

### 2.1 RTX 5090 (GB202 / SM 120a) Architecture

| Parameter | Value |
|-----------|-------|
| SM Count | 170 |
| Warps per SM | 48 (practical) |
| Registers per SM | 256K 32-bit (256 KB) |
| Shared Memory per SM | 256 KB unified (L1/smEM) |
| Max Threads per Block | 1536 |
| Max Blocks per SM | Depends on regs/smEM |
| L2 Cache | 96 MB |
| Tensor Core Gen | 5th gen (INT8 mma.sync m16n8k32) |
| TMA | SM90 TMA (present) |
| tcgen05/TMEM | NOT available on RTX 5090 |

### 2.2 Consumer Kernel Occupancy

**Kernel:** `transcript_gemm_kernel_consumer`
- Block size: 256 threads
- `__launch_bounds__(256, PEARL_CONSUMER_MIN_BLOCKS)` where default minBlocks=1
- Shared memory: 48-144 KiB depending on kBK and kStages

**Register analysis:**
The CUTLASS-generated kernel uses:
- 128 accumulator registers (kFragSize = 128 int32 per thread)
- 16 transcript registers
- ~20-30 temporary registers for copy/mma operations
- **Estimated: ~160-180 registers per thread**

**Occupancy calculation (kBK=128, kStages=2, smem=96 KiB):**
```
Registers needed per SM: 256 KB
Shared memory per block: 96 KiB
Max blocks by regs: 256K / 170regs / 32regs_per_reg = ~47 blocks (register-limited)
Max blocks by smem: 256K / 96K = 2 blocks (shared-memory-limited)
Max blocks by threads: 170 SM * 48 warps * 32 threads / 256 threads = 102 blocks

Practical max blocks per SM: 2 (shared memory bound)
Occupancy: 2 blocks * 256 threads / (48 warps * 32 threads) = 33%
```

**With kBK=64, kStages=2 (smem=48 KiB):**
```
Max blocks by smem: 256K / 48K = 5 blocks
Max blocks by regs: ~47 blocks (register-limited)
Practical max blocks per SM: 5
Occupancy: 5 * 256 / (48 * 32) = 83%
```

**Key finding:** The consumer kernel is **shared-memory limited**, not register-limited. Reducing shared memory per block increases occupancy significantly.

### 2.3 GeForce Variant Occupancy

**Kernel:** `transcript_gemm_sm120_geforce_kernel`
- Block size: 288 threads (256 consumer + 1 producer)
- `__launch_bounds__(288, 1)`
- Shared memory: same as consumer (~48-96 KiB)
- Occupancy: 1 block/SM (minBlocks=1), ~16% theoretical max

**Key finding:** The warp-specialized GeForce variants have **lower occupancy** because they dedicate 1 warp to TMA production. The consumer kernel at higher occupancy may actually process more total work per second.

### 2.4 Persistent Kernel Occupancy

**Kernel:** `propminer_persistent_kernel`
- Block size: 256 threads
- `__launch_bounds__(256, 3)` — min 3 blocks/SM
- Shared memory: ~8 KiB per block
- Estimated registers: ~64 per thread (forced by launch_bounds)

**Occupancy (8 KiB smem per block):**
```
Max blocks by smem: 256K / 8K = 32 blocks
Max blocks by regs: 256K / 64regs = 4096 blocks
Max blocks by threads: 170 * 48 * 32 / 256 = 1020 blocks

Practical max blocks per SM: 3 (enforced by launch_bounds)
Occupancy: 3 * 256 / (48 * 32) = 50%
```

**Key finding:** The persistent kernel's `__launch_bounds__(256, 3)` limits occupancy to 50%. Increasing minBlocks to 5-8 would improve occupancy to 83-104%.

---

## 3. Bottleneck Analysis

### 3.1 GEMM Kernel Bottleneck

**The consumer GEMM kernel is compute-bound on Tensor Cores.**

For M=8192, N=262144, K=128, R=128:
- Total INT8 MACs per iter: 2 * M * N * K = 2 * 8192 * 262144 * 128 = 550 TFLOPS (INT8)
- RTX 5090 INT8 Tensor Core peak: ~2096 TFLOPS (FP8) / ~1048 TFLOPS (INT8)
- The GEMM tile (128x256x128) is well-sized for the 96 MB L2 cache
- cp.async hides global memory latency effectively
- Swizzled shared memory eliminates bank conflicts

**Measured bottleneck:** The GEMM kernel itself is likely at 70-85% of theoretical Tensor Core throughput. The remaining gap comes from:
1. **Host pipeline overhead** — batch size of 1 means the GPU waits for the host to queue the next batch
2. **Sigma installation stalls** — when the sigma changes, mining stops while resident B is built
3. **Share trigger overhead** — when a share is found, the GPU must recompute A and merkle proofs

### 3.2 Persistent Kernel Bottleneck

**The persistent kernel is compute-bound on BLAKE3.**

Per-nonce work:
- LCG noise generation for 64x64x64 = 8192 bytes of A and B (fast, register-based)
- INT8 GEMM: 64x64x64 = 2 * 64 * 64 * 64 = 524K INT8 MACs
- XOR reduction: 8 int32 → 1 uint32
- BLAKE3: 16 state registers, 6.5 rounds (7 rounds with last without permute), 16 msg words
- Target comparison: 8 uint32 comparisons

**BLAKE3 dominates.** Each BLAKE3 compress call uses:
- 16 state registers (s0-s15)
- 16 message registers (b0-b15)
- 6.5 rounds × 8 G-operations = 52 G-operations
- Each G: 4 adds + 4 rotates + 4 XORs = 12 instructions
- Total: ~624 instructions per BLAKE3

**The persistent kernel processes ~1 nonce per block per iteration**, which is extremely low throughput compared to the GEMM kernel's thousands of tiles per launch.

### 3.3 Host Pipeline Bottleneck

This is the **most significant bottleneck**:

1. **Batch size = 1 (default):** Each GEMM launch is ~30ms at N=262144. The host spends time on:
   - cudaEventRecord/cudaEventQuery overhead
   - scan_winners() on host
   - share reconstruction (if hit)
   - sigma installation (rare but expensive)

2. **CUDA graph benefits are limited:** At batch=1, the graph captures a single launch. Increasing batch to 4-8 captures multiple launches, but the host still serializes between batches.

3. **Share trigger stalls the GPU:** When a share is found, `process_share_trigger()` must:
   - Synchronize the stream
   - Regenerate A matrix (~8 MiB HBM write)
   - Compute tensor hashes (~100ms kernel)
   - Multiple D2H transfers
   - This can take 100-500ms, during which mining is paused

---

## 4. Optimization Opportunities Ranked by Impact/Effort

### Tier 1: High Impact, Low Effort (Do First)

#### 4.1 Increase Batch Size (PROPMINER_BATCH)

**Impact:** +20-40% TMAD/s
**Effort:** Environment variable, no code changes

Current default: `batch=1`
Recommended: `batch=8-16`

**Why:** Each CUDA launch has ~1-2ms overhead. At batch=1, that's 1-2ms wasted per 30ms batch = 3-7% overhead. At batch=8, it's 0.4% overhead. Combined with CUDA graph benefits, this reduces host overhead dramatically.

**Implementation:**
```bash
export PROPMINER_BATCH=8
export PROPMINER_GRAPH_BATCH=8
```

**Caveats:**
- Batch must divide evenly into graph_batch for CUDA graph path
- Larger batches mean more seed uploads per launch
- The `iter_batch` path handles this natively; graph path requires `batch % graph_batch == 0`

#### 4.2 Enable Triple Buffering (PROPMINER_TRIPLE_BUFFER)

**Impact:** +10-20% TMAD/s
**Effort:** Environment variable, requires ~11 GiB free VRAM

Current: Off by default
Recommended: `PROPMINER_TRIPLE_BUFFER=1`

**Why:** With triple buffering, share reconstruction on one half leaves two halves free for GEMM. Currently with ping-pong, a share on one half stalls the other half's next batch.

**Implementation:**
```bash
export PROPMINER_TRIPLE_BUFFER=1
```

**VRAM check:** The code already has `triple_vram_headroom_ok()` — it checks for 11 GiB free at production N/K.

#### 4.3 Enable Async Sigma Install (PROPMINER_ASYNC_JOB_INSTALL)

**Impact:** +5-15% TMAD/s (depending on sigma change frequency)
**Effort:** Environment variable

Current: Off by default
Recommended: `PROPMINER_ASYNC_JOB_INSTALL=1`

**Why:** Normally, when sigma changes, the mining thread blocks while resident B is built (can take 50-200ms). With async install, a background thread pre-builds the next sigma's resident B, so the mining thread can swap instantly.

**Implementation:**
```bash
export PROPMINER_ASYNC_JOB_INSTALL=1
```

#### 4.4 Increase Shared Memory Carveout for L2

**Impact:** +5-10% TMAD/s
**Effort:** One `cudaDeviceSetLimit` call (already present in gpu_worker.cpp line 264)

Current: `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128)` is already set.

**Additional optimization:** Set shared memory carveout to favor L1/TEX:
```cpp
cudaDeviceSetLimit(cudaLimitSharedmemCarveout, cudaSharedmemCarveoutMaxL1);
```

This gives more unified memory to L1/TEX cache, which helps the B matrix (resident, ~320 MiB at N=262144, K=128) stay in cache.

### Tier 2: Medium Impact, Medium Effort

#### 4.5 Increase Persistent Kernel Occupancy

**Impact:** +15-25% TMAD/s (on the persistent kernel path)
**Effort:** Modify `__launch_bounds__` in `pearlhash_kernel.cu`

Current: `__launch_bounds__(256, 3)` — 3 blocks/SM minimum
Recommended: `__launch_bounds__(256, 5)` — 5 blocks/SM minimum

**Why:** With only 8 KiB shared memory per block, the kernel can easily support 5-8 blocks/SM. More blocks = more warps in flight = better latency hiding for the BLAKE3 compute.

**Implementation:**
```cpp
// In pearlhash_kernel.cu line 86
__global__ void __launch_bounds__(PM_BLOCK_X, 5)  // Changed from 3 to 5
propminer_persistent_kernel(...)
```

**Register constraint:** Must verify register count stays ≤64/thread. The current `__launch_bounds__(256, 3)` forces ≤64 regs. At minBlocks=5, the compiler may need ≤40 regs/thread. This requires register pressure reduction (see 4.6).

#### 4.6 Reduce Register Pressure in Persistent Kernel

**Impact:** Enables higher occupancy (see 4.5)
**Effort:** Moderate — code changes

Current register usage: ~64 registers/thread (forced by launch_bounds)

**Key register consumers:**
1. BLAKE3 state: 16 registers (s0-s15) — cannot reduce
2. BLAKE3 message: 16 registers (b0-b15) — cannot reduce
3. GEMM accumulators: 8 registers (c0-c7) — already minimal
4. Loop counters and temporaries: ~10-20 registers
5. splitmix64 state: ~4 registers per call

**Optimization: Inline splitmix64 with register reuse**

The noise generation loop (lines 156-167) calls `splitmix64` many times. Each call uses 2-3 registers for the state. Reuse registers across iterations:

```cpp
// Instead of:
uint64_t base_seed = splitmix64(seed_a_lo ^ splitmix64(seed_a_hi));
for (int i = tid; i < PM_SMEM_A_SIZE; i += PM_BLOCK_X) {
    uint64_t idx = i;
    uint64_t group = idx / 8;
    int byte_in_group = static_cast<int>(idx % 8);
    uint64_t z = splitmix64(base_seed + group);  // Allocates new regs each call
    ...
}

// Alternative: use a warp-level PRNG that reuses registers
// Or: unroll the splitmix64 state to avoid function call overhead
```

**Optimization: Use `__ldg()` for shared memory reads**

The persistent kernel reads from shared memory in the GEMM loop. Using `__ldg()` (load-global-cache) can reduce register pressure by keeping data in the read cache rather than spilling to registers.

#### 4.7 Enable CUDA Graphs at Higher Batch

**Impact:** +5-10% TMAD/s
**Effort:** Environment variable

Current: `graph_batch=1` (default)
Recommended: `graph_batch=4-8`

**Why:** CUDA graph capture eliminates driver overhead for batched launches. At graph_batch=1, only one launch is captured. At graph_batch=4, four launches are captured in one graph, amortizing the capture overhead.

**Implementation:**
```bash
export PROPMINER_GRAPH_BATCH=8
```

**Note:** The code already supports this — `queue_batch()` checks `can_graph` and uses `gemm_.iter_batch_graph_launch_ex()` when batch is divisible by graph_batch.

### Tier 3: High Impact, High Effort

#### 4.8 Batched BLAKE3 in Persistent Kernel

**Impact:** +50-100% TMAD/s (on persistent kernel path)
**Effort:** Significant — kernel rewrite

**Problem:** The persistent kernel processes 1 nonce per block per iteration. Each nonce runs a full BLAKE3 compress.

**Solution: Process multiple nonces per block using warp-level parallelism**

Currently, all 256 threads in a block work on the SAME nonce (loading A/B tiles, running GEMM, then BLAKE3). This is because the nonce determines the A/B seed.

**New approach: Partition nonces across warps**

```
Warp 0: processes nonce N + 0*stride
Warp 1: processes nonce N + 1*stride
...
Warp 7: processes nonce N + 7*stride

Where stride = block_size (e.g., 256)
```

This way, 8 nonces are processed per iteration instead of 1. The shared memory A/B tiles can be loaded once and each warp uses a slightly different seed offset.

**Implementation complexity:**
- Need to derive 8 different seeds from one base seed
- Each warp needs its own BLAKE3 state (16 registers × 8 warps = 128 registers — too many)
- Solution: Use register stacking or process 2-4 nonces per warp sequentially

**Expected result:** 4-8x throughput on the persistent kernel path, which could push 290 → 500+ TMAD/s if this kernel is the bottleneck.

#### 4.9 Fused GEMM + BLAKE3 in Single Kernel

**Impact:** +30-50% TMAD/s
**Effort:** Significant — kernel rewrite

**Problem:** The GEMM kernel and BLAKE3 are separate operations. The GEMM produces a transcript, which is then checked via BLAKE3 in the same kernel but in a separate code path.

**Solution: Fully fuse the BLAKE3 compress into the GEMM loop**

Currently, the consumer kernel (`transcript_gemm_kernel_consumer`) does:
1. GEMM compute (K-tile loop)
2. XOR reduction of accumulator
3. BLAKE3 compress (only on hit)
4. Target check

The BLAKE3 only runs when a PoW hit is found (extremely rare). The optimization is to **run BLAKE3 for every nonce** but in a fused manner that hides latency:

```
For each nonce:
  1. Generate A/B noise (registers)
  2. Run GEMM mma.sync (Tensor Cores)
  3. XOR reduction (registers)
  4. BLAKE3 compress (registers + PTX)
  5. Target check (branch)
```

This requires a **persistent kernel that processes one nonce per thread**, similar to the current `pearlhash_kernel.cu` but using the larger GEMM tile (128x256x128) instead of (64x64x64).

**Implementation:**
```cuda
__global__ void fused_mining_kernel(
    WorkItem* queue,
    ResultBuffer* results,
    ...) {
    int tid = threadIdx.x;
    int nonce = blockIdx.x * blockDim.x + tid;
    
    // Each thread processes one nonce
    generate_noise(nonce, &A_smem, &B_smem);
    gemm_128x256x128(A_smem, B_smem, accum);
    hash = xor_reduction(accum);
    blake3_compress(hash, pow_key, chaining);
    if (chaining < pow_target) {
        write_share(nonce, chaining, sigma);
    }
}
```

**Challenge:** The 128x256x128 GEMM requires 128 int32 accumulators per thread (128 × 4B = 512B in registers). At 256 threads × 128 regs = 32K regs per block. With 256K regs/SM and min 2 blocks/SM, this is feasible.

#### 4.10 Tensor Core Optimization for BLAKE3

**Impact:** +20-30% TMAD/s (if applicable)
**Effort:** Experimental — requires PTX research

**Research question:** Can BLAKE3's G-operation be expressed as a Tensor Core operation?

BLAKE3's G operation:
```
a = a + b + x
d = rotl(d ^ a, 16)
c = c + d
b = rotl(b ^ c, 12)
...
```

The `lop3.b32` (3-input XOR) and `shf.l.wrap.b32` (rotate) are already optimal PTX. However, the **add chain** (a+b+x, c+d+y) could potentially be expressed as INT8 Tensor Core operations if the data is packed correctly.

**Unlikely to help** because BLAKE3 operates on 32-bit words, and Tensor Cores on SM120 are optimized for INT8/FP8. The INT32 compute path does not have Tensor Core acceleration.

### Tier 4: Niche Optimizations

#### 4.11 Shared Memory Swizzle Bit Tuning

**Impact:** +1-3% TMAD/s
**Effort:** One compile flag

Current: `PEARL_CONSUMER_SWIZZLE_BITS=2` (default)
Benchmarked variant: `PEARL_CONSUMER_SWIZZLE_BITS=3` showed +0.5% on RunPod 5090

**Recommendation:** Test both values with Nsight Compute to measure shared memory bank conflict rate.

```bash
export PEARL_CONSUMER_SWIZZLE_BITS=3
```

#### 4.12 Pipeline Stage Count Tuning

**Impact:** +2-5% TMAD/s
**Effort:** One compile flag

Current: `PEARL_CONSUMER_STAGES=2` for Blackwell
Alternative: `PEARL_CONSUMER_STAGES=3`

**Trade-off:** More stages = more overlap between compute and memory, but more shared memory per block = lower occupancy.

At kBK=128, kStages=3: smem = 144 KiB → max 1 block/SM → very low occupancy
At kBK=128, kStages=2: smem = 96 KiB → max 2 blocks/SM → moderate occupancy
At kBK=64, kStages=3: smem = 72 KiB → max 3 blocks/SM → good occupancy

**Recommendation:** Test `kBK=64, kStages=3` vs `kBK=128, kStages=2`.

```bash
export PEARL_CONSUMER_KBLOCK=64
export PEARL_CONSUMER_STAGES=3
```

#### 4.13 TMA vs cp.async

**Impact:** +5-10% TMAD/s (if TMA is enabled)
**Effort:** Compile flag

Current: cp.async (default for production builds)
Alternative: TMA via `PEARL_CONSUMER_USE_TMA_EXPERIMENT=1`

**TMA advantages:**
- Hardware-managed data movement (no explicit wait_group)
- Better for large, regular transfers
- Descriptor caching (already implemented in v2)

**TMA disadvantages:**
- Requires TMA descriptor setup per launch
- May have higher latency for small transfers
- Not fully tested in production (experimental flag)

**Recommendation:** Benchmark TMA vs cp.async with Nsight Systems to measure memory throughput.

#### 4.14 Thread Block Clustering

**Impact:** +0-5% TMAD/s (depends on workload)
**Effort:** Environment variable

Current: `cluster_m=1` (default, no clustering)
Alternative: `cluster_m=2` or `cluster_m=4`

**Why:** Clustering groups adjacent M-tiles on the same SM, improving B-tile locality when multiple threads access the same B rows.

**Trade-off:** Clustering reduces the number of active SMs, which can hurt occupancy if the grid is small.

```bash
export PEARL_GEMM_CONSUMER_CLUSTER_M=2
```

#### 4.15 Optimize Share Trigger Path

**Impact:** +5-15% TMAD/s (depending on hit rate)
**Effort:** Moderate — code changes

**Problem:** When a share is found, `process_share_trigger()` takes 100-500ms:
1. Stream synchronize (blocks all other batches)
2. Regenerate A matrix (HBM write)
3. Tensor hash kernel (~100ms)
4. Multiple D2H transfers
5. Host-side merkle proof computation

**Optimization: Deferred share processing**

Already partially implemented via `PROPMINER_DEFER_SHARE_GPU=1`. The share trigger is processed on a side thread while the GPU continues mining.

**Further optimization: GPU-side merkle proof**

Move the merkle proof computation from host to GPU:
```cuda
__global__ void merkle_proof_kernel(
    int32_t* a_slice,      // A rows that produced the hit
    uint8_t* leaf_cvs,     // Leaf commitment values
    uint8_t* opened_leaves, // Merkle proof leaves
    uint8_t* proof_output,  // Final proof bytes
    ...);
```

This eliminates the D2H transfer of merkle data and the host-side computation.

**Optimization: Batch share triggers**

Instead of processing each share trigger individually, batch them and process multiple proofs simultaneously.

#### 4.16 Reduce A Matrix Regeneration

**Impact:** +5-10% TMAD/s
**Effort:** Moderate — code changes

**Problem:** When a share is found, the entire A matrix (~8 MiB at M=8192, K=128) is regenerated to produce the merkle proof.

**Optimization: Cache A matrix across share triggers**

Currently, A is regenerated fresh for each nonce. If multiple shares are found for the same sigma, the A matrix can be reused.

**Implementation:**
```cpp
// Add A matrix cache to HalfBuffers
struct HalfBuffers {
    CUdeviceptr a_cache;       // Cached A matrix
    uint64_t a_cache_nonce;    // Nonce for which A was cached
    size_t a_cache_bytes;
    ...
};
```

When a share trigger arrives:
1. If `a_cache_nonce == trigger_nonce`, reuse cached A
2. Otherwise, regenerate and cache

**Impact:** If multiple shares are found within the same sigma period, this saves 100-500ms per share.

---

## 5. Proposed New Kernel Configurations

### 5.1 Recommended Production Configuration

```bash
# Batch and graph
export PROPMINER_BATCH=8
export PROPMINER_GRAPH_BATCH=8

# Triple buffering
export PROPMINER_TRIPLE_BUFFER=1

# Async sigma install
export PROPMINER_ASYNC_JOB_INSTALL=1

# Share deferral
export PROPMINER_DEFER_SHARE_GPU=1

# Shared memory swizzle
export PEARL_CONSUMER_SWIZZLE_BITS=3

# Cluster (optional, benchmark)
export PEARL_GEMM_CONSUMER_CLUSTER_M=1

# L2 cache
# (already set in gpu_worker.cpp: cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128))
```

**Expected improvement:** 290 → 380-420 TMAD/s (+30-45%)

### 5.2 Aggressive Configuration (After Code Changes)

After implementing Tier 2 optimizations (persistent kernel occupancy + register reduction):

```cpp
// In pearlhash_kernel.cu
__global__ void __launch_bounds__(256, 5)  // Increased from 3
propminer_persistent_kernel(...)
```

```bash
# Same as production config above
export PROPMINER_BATCH=16
export PROPMINER_GRAPH_BATCH=16
```

**Expected improvement:** 290 → 500-600 TMAD/s (+70-110%)

### 5.3 Maximum Configuration (After Tier 3 Optimizations)

After implementing batched BLAKE3 (4.8) and fused GEMM+BLAKE3 (4.9):

```bash
export PROPMINER_BATCH=32
export PROPMINER_GRAPH_BATCH=32
export PROPMINER_TRIPLE_BUFFER=1
export PROPMINER_ASYNC_JOB_INSTALL=1
```

**Expected improvement:** 290 → 700-800+ TMAD/s (+140-180%)

---

## 6. Memory Access Pattern Optimizations

### 6.1 Global Memory Coalescing

**Current state:** The GEMM kernel already uses coalesced accesses via CUTLASS's TiledCopy. A and B matrices are row-major with 256-byte alignment, and K is a multiple of 128, so each row starts on a 128-byte boundary matching the RTX 5090 L2 cache-line granularity.

**Optimization:** Ensure A matrix (ApEA) is allocated with 256-byte alignment:
```cpp
// Already done in gpu_worker.cpp line 120:
check(cuMemAlloc(&a, a_bytes), "a alloc");
// cuMemAlloc returns 256-byte aligned memory by default
```

### 6.2 Shared Memory Banking

**Current state:** Swizzled shared memory (`Swizzle<2,4,3>` or `Swizzle<3,4,3>`) eliminates bank conflicts for the LDSM.x4 pattern.

**Verification:** Use Nsight Compute to check `sm__throughput_pipe_tensor_scored_mem_pipe_stall.sum` — if this is high, bank conflicts are present.

**Current swizzle benchmark:** Swizzle<3,4,3> showed +0.5% over Swizzle<2,4,3> at M=8192, N=262144.

### 6.3 L1/L2 Cache Utilization

**Current state:**
- `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128)` is set (line 264 of gpu_worker.cpp)
- Shared memory carveout is at driver default

**Optimization:** Experiment with carveout:
```cpp
// Try maximizing L1 for better B matrix caching:
cudaDeviceSetLimit(cudaLimitSharedmemCarveout, cudaSharedmemCarveoutMaxL1);

// Or balance for mixed workloads:
cudaDeviceSetLimit(cudaLimitSharedmemCarveout, 50); // 50% L1, 50% smem
```

**Rationale:** The B matrix (~320 MiB at N=262144, K=128) is larger than L2 (96 MB). More L1 cache helps keep frequently accessed B tiles in cache.

### 6.4 Memory Prefetching

**Current state:** cp.async with kStages=2 provides a 1-stage prefetch buffer.

**Optimization:** Increase to kStages=3 with kBK=64:
```
kBK=64, kStages=3:
  smem per block = (128+256) * 64 * 3 = 72 KiB
  blocks per SM = 256 / 72 = 3 (good occupancy)
  prefetch depth = 2 stages (more overlap)
```

### 6.5 Constant Cache Optimization

**Current state:** BLAKE3 IV constants are in `__device__ __constant__` memory (`blake3.cuh` line 18-21). The compress params are also in constant memory.

**Optimization:** Constant memory is already optimal for BLAKE3. The IV and params are read-only and small (48 bytes total), so they fit in the constant cache (48 KiB per SM). No changes needed.

---

## 7. Kernel Fusion Opportunities

### 7.1 GEMM + Transcript XOR Reduction (Already Fused)

**Status:** Already implemented. The consumer kernel XOR-reduces the accumulator in registers during the K-tile loop, avoiding a separate kernel launch.

### 7.2 Transcript + BLAKE3 (Partially Fused)

**Status:** The headless path (`launch_transcript_gemm_headless`) runs BLAKE3 in the same kernel as GEMM, but only on hit (checked after the K-tile loop completes).

**Optimization:** Run BLAKE3 for every nonce, not just hits. This requires:
1. Each thread processes one nonce (not one tile)
2. BLAKE3 state per thread (16 registers)
3. Target check per thread

This is essentially the persistent kernel approach but using the larger GEMM tile.

### 7.3 A Matrix Generation + GEMM (Not Fused)

**Status:** A matrix is generated on the host (LCG noise) and copied to device before the GEMM kernel.

**Optimization:** Generate A directly on device inside the GEMM kernel:
```cuda
__global__ void gemm_with_noise_kernel(
    int8_t const* B_gmem,
    int32_t* C_gmem,
    uint32_t* transcript,
    uint64_t seed_lo,
    ...) {
    // Generate A in shared memory from seed
    generate_noise_in_smem(tid, seed_lo, smem_A);
    // Run GEMM with smem_A and B_gmem
    ...
}
```

**Benefit:** Eliminates host→device A matrix copy (~8 MiB per batch). At batch=8, this saves ~64 MiB of PCIe traffic per iteration.

**Challenge:** The noise generation uses splitmix64, which is somewhat compute-intensive. Running it on-device adds register pressure.

### 7.4 Share Trigger + Merkle Proof (Not Fused)

**Status:** Share trigger runs on host after D2H transfer.

**Optimization:** GPU-side merkle proof (see 4.15).

---

## 8. CUDA Stream Usage Recommendations

### 8.1 Current Stream Usage

```
ping_.stream   — compute + graph launch
pong_.stream   — compute + graph launch
third_.stream  — compute (triple-buffer)
seed_copy_stream_ — seed H2D (async)
merkle_copy_stream_ — merkle operations
install_stream_ — async sigma install
install_copy_stream_ — async sigma install copy
```

### 8.2 Recommended Stream Configuration

**For batch=1, ping-pong:**
```
Stream 0 (ping):  compute + graph
Stream 1 (pong):  compute + graph
Stream 2 (seed):  seed H2D (overlap with compute)
```

**For batch=8, triple-buffer:**
```
Stream 0 (ping):  compute + graph (sub-batches 0-7)
Stream 1 (pong):  compute + graph (sub-batches 0-7)
Stream 2 (third): compute + graph (sub-batches 0-7)
Stream 3 (seed):  seed H2D (overlap with all compute)
Stream 4 (install): async sigma install
```

**Recommendation:** Add a dedicated stream for share trigger processing:
```cpp
check_cuda(cuStreamCreate(&share_trigger_stream_, CU_STREAM_NON_BLOCKING),
           "share trigger stream");
```

This allows share processing to overlap with mining on other streams.

### 8.3 Optimal Number of Concurrent Streams

**Rule of thumb:** Number of streams = number of independent work items that can overlap.

For PropMiner:
- **Minimum:** 2 streams (ping-pong double buffering)
- **Recommended:** 3 streams (triple buffering + seed copy)
- **Maximum:** 5 streams (triple + seed + share trigger + install)

**Diminishing returns:** Beyond 5 streams, the overhead of stream management outweighs the overlap benefits. The RTX 5090 has 4 PCIe Gen5 lanes (32 GB/s each), so the total H2D/D2H bandwidth is ~256 GB/s. The GEMM kernel's memory bandwidth is ~10 TB/s (Tensor Cores), so PCIe is not the bottleneck.

---

## 9. Expected Performance Improvement Estimates

| Optimization | Current | Expected | Notes |
|-------------|---------|----------|-------|
| **Baseline** | 290 TMAD/s | 290 TMAD/s | Current production |
| Tier 1: Batch=8, Triple, Async | 290 | 380-420 | +30-45% |
| Tier 1: L2 carveout tuning | 380 | 390-400 | +2-5% |
| Tier 2: Persistent kernel occupancy | 380 | 430-470 | +10-15% |
| Tier 2: Register pressure reduction | 430 | 450-490 | +5-10% |
| Tier 2: Graph batch=8 | 450 | 470-500 | +5% |
| Tier 3: Batched BLAKE3 (4.8) | 470 | 600-700 | +30-50% |
| Tier 3: Fused GEMM+BLAKE3 (4.9) | 600 | 700-800 | +20-30% |
| Tier 4: Share trigger optimization | 700 | 730-780 | +5-10% |
| **Maximum** | 290 | **700-800+** | **+140-180%** |

---

## 10. Implementation Priority Order

### Phase 1: Environment Tuning (Week 1)
1. Set `PROPMINER_BATCH=8`, `PROPMINER_GRAPH_BATCH=8`
2. Enable `PROPMINER_TRIPLE_BUFFER=1`
3. Enable `PROPMINER_ASYNC_JOB_INSTALL=1`
4. Enable `PROPMINER_DEFER_SHARE_GPU=1`
5. Benchmark with Nsight Compute to identify remaining bottlenecks

### Phase 2: Kernel Parameter Tuning (Week 2)
6. Increase persistent kernel `__launch_bounds__` minBlocks to 5
7. Reduce register pressure in persistent kernel (inline splitmix64)
8. Test swizzle bit variants (2 vs 3)
9. Test pipeline stage variants (kBK=64/stages=3 vs kBK=128/stages=2)

### Phase 3: Code-Level Optimizations (Week 3-4)
10. Implement batched BLAKE3 in persistent kernel (process 4-8 nonces per iteration)
11. Implement GPU-side merkle proof for share triggers
12. Add A matrix caching across share triggers
13. Add dedicated share trigger stream

### Phase 4: Architectural Changes (Week 5-8)
14. Implement fused GEMM+BLAKE3 persistent kernel
15. Implement device-side A matrix generation (eliminate H2D copy)
16. Benchmark and tune all configurations with Nsight Systems

### Phase 5: Validation (Week 9-10)
17. Byte-identity verification against reference implementation
18. Stress test at target hashrate (700-800+ TMAD/s)
19. Long-running stability test (24+ hours)

---

## 11. Profiling and Benchmarking Framework

### 11.1 Nsight Compute Metrics to Track

| Metric | What It Tells You | Target |
|--------|-------------------|--------|
| `smsp__inst_executed_op_pipe_tensor_scored.sum` | Tensor Core utilization | >80% |
| `smsp__inst_executed_sum.avg` | Total instruction throughput | Maximize |
| `smsp__sass_thread_inst_executed_op_math_pred_on.sum` | Math instruction throughput | Maximize |
| `smsp__sass_thread_inst_executed_op_pipe_memory_scored.sum` | Memory instruction throughput | Balance with compute |
| `smsp__pipe_stall_tot_mem_pipe_stall.avg` | Memory stall cycles | Minimize |
| `smsp__pipe_stall_tot_other_stall.avg` | Non-memory stalls | Minimize |
| `sm__warps_active.avg` | Occupancy | >50% |
| `sm__shared_pipe_stall_bank_conflict.avg` | Shared memory bank conflicts | 0 |
| `dram__bytes_read.sum` / `dram__bytes_written.sum` | Global memory bandwidth | >80% of peak |
| `lts__t_sectors_srcunit_tex_op_read.request.sum` | L2 cache reads | Maximize |
| `lts__t_sectors_srcunit_tex_op_read.hit.sum` | L2 cache hits | >90% |

### 11.2 Nsight Systems Timeline to Track

| Event | What to Measure |
|-------|-----------------|
| Kernel launch latency | Host→GPU dispatch overhead |
| Kernel execution time | Actual GEMM compute time |
| memcpy H2D | A matrix upload time |
| memcpy D2H | Transcript/result download time |
| Stream overlap | Concurrent compute + transfer |
| Event wait time | Host spin-wait overhead |

### 11.3 Benchmark Script

```bash
#!/bin/bash
# benchmark.sh — Quick hashrate benchmark

export PROPMINER_BATCH=$1
export PROPMINER_GRAPH_BATCH=$1
export PROPMINER_TRIPLE_BUFFER=${2:-1}
export PROPMINER_ASYNC_JOB_INSTALL=${3:-1}
export PEARL_CONSUMER_SWIZZLE_BITS=${4:-3}

# Run for 60 seconds, capture hashrate from stderr
timeout 60 ./propminer 2>&1 | grep -E "first batch|hashrate|TMAD"
```

### 11.4 Isolation Testing

To isolate the impact of each optimization:

1. **Baseline:** Default config, batch=1, no triple buffer
2. **Test 1:** batch=8 only (measure +TMAD/s)
3. **Test 2:** batch=8 + triple buffer (measure delta from Test 1)
4. **Test 3:** batch=8 + triple + async (measure delta from Test 2)
5. **Test 4:** batch=8 + triple + async + swizzle=3 (measure delta from Test 3)
6. **Test 5:** batch=8 + triple + async + swizzle=3 + persistent kernel occupancy (measure delta from Test 4)

Each test runs for 300 seconds with a 120-second warmup.

---

## 12. Risk Assessment

| Optimization | Risk Level | Mitigation |
|-------------|-----------|------------|
| Batch size increase | Low | Falls back to iter_batch if graph fails |
| Triple buffering | Low | VRAM check prevents OOM |
| Async sigma install | Low | Falls back to sync install on failure |
| Persistent kernel occupancy | Medium | Verify register count with nvcc --ptxas-options=-v |
| Batched BLAKE3 | High | Byte-identity must be verified against reference |
| Fused GEMM+BLAKE3 | High | Major kernel rewrite, extensive testing needed |
| GPU-side merkle proof | Medium | Correctness verification required |
| Device-side A generation | Medium | Numerical equivalence must be verified |

---

## 13. Code Locations Reference

| File | Key Lines | Description |
|------|-----------|-------------|
| `src/cuda/kernels/pearlhash_kernel.cu` | 86-93 | Persistent kernel launch + signature |
| `src/cuda/kernels/pearlhash_kernel.cu` | 156-183 | Shared memory noise generation |
| `src/cuda/kernels/pearlhash_kernel.cu` | 187-246 | GEMM mma.sync loop |
| `src/cuda/kernels/pearlhash_kernel.cu` | 248-279 | XOR reduction + BLAKE3 |
| `src/host/pearl/gpu_worker.cpp` | 256-259 | Stream creation |
| `src/host/pearl/gpu_worker.cpp` | 264 | L2 fetch granularity |
| `src/host/pearl/gpu_worker.cpp` | 930-1051 | Batch queue + graph launch |
| `src/host/pearl/gpu_worker.cpp` | 1408-1768 | Main mining loop |
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | 224-563 | Consumer GEMM kernel |
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | 697-778 | Kernel launcher |
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | 578-623 | Carveout + cluster env vars |
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` | 112-292 | GeForce v1 kernel |
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce_v2.cu` | 123-335 | GeForce v2 kernel |
| `src/host/pearl/rtx5090_profile.h` | 16-134 | RTX 5090 hardware profile |
| `src/host/pearl/env_tuning.h` | 1-125 | Tuning environment variables |

---

## 14. Summary of Key Recommendations

### Immediate (no code changes):
1. **Set `PROPMINER_BATCH=8`** — biggest single win, eliminates host overhead
2. **Enable `PROPMINER_TRIPLE_BUFFER=1`** — overlaps share processing with mining
3. **Enable `PROPMINER_ASYNC_JOB_INSTALL=1`** — eliminates sigma install stalls
4. **Enable `PROPMINER_DEFER_SHARE_GPU=1`** — moves share processing off main thread

### Short-term (minor code changes):
5. **Increase persistent kernel occupancy** — change `__launch_bounds__(256, 3)` to `(256, 5)`
6. **Reduce register pressure** — inline splitmix64, reuse temporaries
7. **Tune swizzle bits** — benchmark Swizzle<2,4,3> vs Swizzle<3,4,3>

### Long-term (significant code changes):
8. **Batched BLAKE3** — process 4-8 nonces per kernel iteration
9. **Fused GEMM+BLAKE3** — unified persistent kernel with large GEMM tiles
10. **GPU-side merkle proof** — eliminate D2H transfer for share triggers

### Expected trajectory:
```
290 → 380-420 (immediate) → 500-600 (short-term) → 700-800+ (long-term)
```
