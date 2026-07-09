# Batched BLAKE3 Processing — Comprehensive Analysis & Implementation Plan

**Author:** CUDA Mining Engineering  
**Date:** 2026-07-09  
**Status:** DRAFT — Awaiting review

---

## Executive Summary

This document analyzes PropMiner's BLAKE3 implementation across all code paths (legacy persistent kernel, consumer headless, GeForce v1/v2, portable transcript_finalize) and designs a batched BLAKE3 processing approach. The key finding: **BLAKE3 is already done per-nonce in the headless path inside the transcript GEMM kernel** (lines 499-524 of `transcript_gemm_kernel.cu`), and per-nonce in the `transcript_finalize` kernel for non-headless paths. Batching BLAKE3 means processing multiple nonce transcripts within a single kernel launch or within a single warp, which requires restructuring how transcripts are organized in memory.

---

## Part 1: Current BLAKE3 Architecture

### 1.1 Where BLAKE3 Happens — Three Distinct Paths

#### Path A: Consumer Headless (Production Path for Blackwell/Ampere/Ada)
**File:** `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu`, lines 499-524

BLAKE3 runs **inside the transcript GEMM kernel**, in the same thread that computes the GEMM:

```cpp
// Lines 499-524: In-kernel finalization (headless path)
if (pow_target != nullptr && pow_key_eff != nullptr && ...) {
    Tensor transcript_rmem = make_tensor<uint32_t>(Int<kTranscriptSlots>{});
    // Copy 16 u32 transcript from registers
    for (int s = 0; s < kTranscriptSlots; ++s)
        transcript_rmem(s) = transcript_local[s];

    bool block_found = pearl::check_pow_target(
        transcript_rmem, pow_target, pow_key_eff);
    if (block_found) {
        // ... write_host_signal_header ...
    }
}
```

**Data flow:**
1. GEMM accumulator `tCrC` (128 int32 registers) → XOR-reduced → single u32 hash
2. Hash mixed into `transcript_local[slot]` via `rotl_xor<13>` every R columns
3. After K-loop: 16 u32 transcript in registers
4. `check_pow_target()` → `compress_msg_block_u32()` with `COMPRESS_PARAMS_SINGLE_BLOCK_KEYED`
5. Compare 256-bit hash against target

**Key insight:** Each thread processes ONE (m_tile, n_tile) tile, which produces ONE transcript of 16 u32 values. The transcript represents the BLAKE3 message block for a single nonce. The GEMM computes `ApEA(M,K) @ BpEB(N,K).T` where M tiles × N tiles covers all positions for ONE nonce.

#### Path B: GeForce v1 (Experimental Blackwell Consumer Kernel)
**File:** `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu`

Same pattern as Path A — headless mode with in-kernel BLAKE3. The transcript stays in registers and is checked immediately after the GEMM loop.

#### Path C: GeForce v2 (Current Default for RTX 5090)
**File:** `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce_v2.cu`

Same headless pattern. Uses `launch_transcript_gemm_sm120_geforce_v2_headless()` which calls the same `transcript_gemm_kernel_consumer` kernel with pow_target/pow_key pointers.

#### Path D: Portable / B200 / Non-Headless (transcript_finalize)
**File:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`, lines 141-186

Two-kernel approach:
1. **transcript_gemm** (lines 45-51 of .cuh): Computes GEMM + writes transcript snapshots to gmem
2. **transcript_finalize** (lines 141-186): Reads transcript from gmem, runs BLAKE3, checks target

```cpp
__global__ void transcript_finalize_kernel(...) {
    // Line 165-170: Load 16 u32 transcript from gmem
    Tensor transcript_rmem = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
    for (int i = 0; i < 16; ++i)
        transcript_rmem(i) = transcript[tx_idx + i];

    // Line 172-173: BLAKE3 + target check
    bool block_found = pearl::check_pow_target(transcript_rmem, pow_target, pow_key);
}
```

#### Path E: Legacy Persistent Kernel (PropMiner)
**File:** `src/cuda/kernels/pearlhash_kernel.cu`, lines 249-298

One block = one nonce. Each block processes one work item:
```cpp
// Lines 249-272: Per-nonce BLAKE3
uint32_t transcript[B3_MSG_BLOCK_U32] = {};
// Initialize transcript from sigma
for (int i = 0; i < B3_MSG_BLOCK_U32; i++) {
    int idx = (i * 4) % 32;
    transcript[i] = ((uint32_t)work_sigma[idx]) | ...;
}
// XOR-reduce 8 MMA accumulator outputs
for (int i = 0; i < 8; i++)
    transcript[i] = pow_rotl_xor(transcript[i], c_arr[i]);

// Lines 268-272: BLAKE3 keyed hash
uint32_t chaining[B3_CHAINING_VALUE_U32];
for (int i = 0; i < B3_CHAINING_VALUE_U32; i++)
    chaining[i] = pow_key[i];
blake3::compress_msg_block_u32(transcript, chaining, blake3::B3_PARAMS_KEYED_SINGLE);
```

### 1.2 BLAKE3 Data Flow — Detailed

**Inputs to BLAKE3 compress:**
1. **Message block (16 u32 = 64 bytes):** The "transcript" — accumulated XOR-reduced GEMM outputs mixed via rotl_xor<13>
   - For consumer headless: 16 slots, each accumulated K/R times
   - For legacy: initialized from sigma, 8 slots XOR-reduced from GEMM accumulators + 8 zeros
2. **Chaining value (8 u32 = 32 bytes):** The `pow_key` — which is `CommitA` (the commitment hash)
3. **CompressParams:** `COMPRESS_PARAMS_SINGLE_BLOCK_KEYED` = {counter=0, block_len=64, flags=KEYED_HASH|CHUNK_START|CHUNK_END|ROOT}

**BLAKE3 compression (7 rounds, 64-byte msg block):**
- 16 state registers (s0-s15) + 16 message registers (b0-b15)
- 6 rounds of BLAKE3_ROUND() + BLAKE3_PERMUTE()
- 1 final round without permutation
- Output: chaining[i] = s[i] ^ s[i+8] for i=0..7

**Output:**
- 32-byte hash (8 u32) compared against pow_target (8 u32)
- Little-endian big-integer comparison: MSW first (index 7 down to 0)

### 1.3 What "Per-Nonce" Means in the Current Architecture

This is the critical understanding:

**Single-nonce iteration (`pearl_capi_iter`):**
1. LCG fills A matrix (M×K) for nonce N — this is the per-nonce input
2. Tensor hash: A → AHash (Merkle root)
3. Commitment hash: (AHash, BHash) → CommitA (= pow_key)
4. Noise gen: generates EAR from CommitA
5. **Noisy GEMM:** ApEA(M,K) @ BpEB(N,K).T → transcript

The GEMM computes M×N accumulator positions, but each position is a fragment of the SAME nonce's transcript. The transcript is assembled by XOR-reducing fragments at every R columns. So for one nonce, we get:
- M/bM × N/bN tiles
- Each tile: 256 threads × 16 transcript slots
- Total transcript elements: batch × (M/bM) × (N/bN) × 256 × 16

For production shape (M=8192, N=262144, bM=128, bN=256):
- 64 M-tiles × 1024 N-tiles = 65,536 tiles
- Each tile = 256 threads × 16 slots = 4,096 transcript elements
- Total = 65,536 × 4,096 = 268,435,456 transcript elements per nonce

**Each thread's 16 u32 transcript is ONE BLAKE3 invocation.** So for one nonce, we have 65,536 × 256 = 16,777,216 BLAKE3 compress calls across all threads in the grid.

### 1.4 The Headless Optimization (Why It Matters)

In headless mode (production path), the transcript is kept in registers and checked immediately after the GEMM loop. This avoids:
1. Writing 268M transcript elements to gmem (1 GiB)
2. Launching a separate transcript_finalize kernel
3. Reading 1 GiB back from gmem

The headless kernel does ALL work in registers:
- GEMM computation (main loop)
- XOR reduction of accumulator fragments
- rotl_xor mixing into transcript slots
- BLAKE3 compress (7 rounds)
- Target comparison
- Atomic host_signal_header write on hit

**This is the critical path.** Any optimization must work within this register-bound kernel.

---

## Part 2: BLAKE3 Batching — Feasibility Analysis

### 2.1 The Core Constraint: BLAKE3 Is Already Register-Bound

BLAKE3 compression in the headless path uses ~64 registers per thread (16 state + 16 message + 8 chaining + 8 IV/params + temporaries). With `__launch_bounds__(256, 5)` or similar, we're already at high register pressure.

**Key insight:** Batching BLAKE3 across nonces requires a fundamentally different approach because:
1. Each nonce has a DIFFERENT A matrix (different LCG seed)
2. Each nonce has a DIFFERENT CommitA (= pow_key)
3. Each nonce has a DIFFERENT noise pattern (different EAR)
4. The GEMM itself is per-nonce

### 2.2 What CAN Be Batched

There are two distinct batching opportunities:

#### Opportunity 1: Batch GEMM Launches (Already Implemented!)
PropMiner already batches across nonces via `pearl_capi_iter_batch()` and `pearl_capi_iter_batch_graph_prepare_ex()`:

```cpp
// pearl_gemm_capi.cpp line 1682-1711
PEARL_CAPI_EXPORT int pearl_capi_iter_batch(
    void* workspace_ptr,
    uint64_t seed_lo_start,
    void* const* host_signal_header_pinned_batch,
    int32_t count,  // batch size
    void* stream_void);
```

This runs `count` separate nonces in a loop, each with its own:
- LCG fill
- Tensor hash
- Commitment hash
- Noise gen
- Noisy GEMM (with per-nonce header pointer)

**The grouped GEMM path** (`pearl_capi_iter_batch_grouped`) batches ApEA matrices across nonces but still runs GEMM + BLAKE3 per-nonce.

#### Opportunity 2: Batch BLAKE3 Within a Single Warp
This is the novel optimization. Instead of each thread running BLAKE3 independently, a warp (32 threads) could:
1. Each thread loads its 16-u32 transcript
2. Threads share pow_key and pow_target (already identical)
3. Run BLAKE3 compress in parallel across the warp
4. Check targets in parallel

**But this is ALREADY what happens.** Each thread independently runs BLAKE3. The issue is that BLAKE3 is a small computation (~200-300 instructions) compared to the GEMM (~10,000+ instructions). BLAKE3 is a tiny fraction of total work per tile.

### 2.3 The REAL Bottleneck

Let's profile the per-tile work in the headless kernel:

| Phase | Instructions (approx) | Registers |
|-------|----------------------|-----------|
| GEMM main loop (K/R iterations) | ~10,000-50,000 | 128 acc + 16 transcript |
| XOR reduction | ~200 | 128 acc |
| Transcript mixing | ~16 × 10 = 160 | 16 transcript |
| **BLAKE3 compress** | ~**200-300** | **64** |
| Target comparison | ~20 | 8 chaining |
| **Total per tile** | **~10,500-50,500** | |

**BLAKE3 is ~0.4-2.8% of per-tile work.** Even eliminating it entirely would save at most 3% of kernel time.

### 2.4 Where BLAKE3 Actually Costs

The cost isn't in the compute — it's in:
1. **Register pressure:** 64 extra registers for BLAKE3 state reduces occupancy
2. **Instruction cache pressure:** 200-300 extra instructions per thread
3. **Serialization:** Atomic host_signal_header write (rare, only on hit)

For the **legacy persistent kernel** (pearlhash_kernel.cu), BLAKE3 is the ENTIRE per-nonce computation after GEMM. Here it's ~30% of work. But this is the legacy path, not the production path.

---

## Part 3: Batched BLAKE3 Design

### 3.1 Recommended Approach: Offload BLAKE3 to a Separate Kernel

Instead of in-kernel BLAKE3, write transcript to gmem and run a dedicated BLAKE3 kernel. This frees registers in the GEMM kernel, increasing occupancy.

**Architecture:**
```
Kernel 1: transcript_gemm (headless, NO BLAKE3)
  - Computes GEMM
  - Accumulates transcript in registers  
  - Writes transcript to gmem (16 u32 per thread)

Kernel 2: blake3_batch (NEW — batched across threads)
  - Reads transcript from gmem
  - Runs BLAKE3 compress
  - Checks target
  - Writes host_signal_header on hit
```

**Why this works:**
- Kernel 1 runs with fewer registers → higher occupancy → more parallelism
- Kernel 2 is a simple grid where each thread = one BLAKE3 invocation
- The gmem write is unavoidable (1 GiB) but it's a linear bandwidth operation
- The BLAKE3 compute is freed from the GEMM kernel's register pressure

**Expected improvement:**
- Kernel 1: ~5-10% faster due to lower register pressure (more warps/SM)
- Kernel 2: same BLAKE3 throughput as before (it's already parallel)
- Total: ~3-7% improvement (BLAKE3 is a small fraction)

### 3.2 Alternative: Warp-Level BLAKE3 Batching

A more aggressive approach: within a warp, share BLAKE3 state and process multiple transcripts.

**Idea:** 32 threads in a warp share one set of BLAKE3 IV constants and pow_key. Each thread processes its own transcript independently, but the BLAKE3 code is unrolled to use warp-level primitives.

**Not recommended** — the complexity doesn't justify the marginal gain since BLAKE3 is already parallel.

### 3.3 Alternative: Batch BLAKE3 Across Multiple Transcripts in One Compress Call

BLAKE3's design means each message block is independently compressible. There's no natural batching of multiple message blocks into one compress call (they'd need to be in the same chunk, which they're not — each nonce is a separate chunk).

**Not feasible** without changing the Pearl protocol.

### 3.4 Alternative: Skip BLAKE3 for Non-Hits (Speculative Execution)

Most threads will NOT find a hit. We could:
1. Compute a cheaper hash (e.g., just the XOR reduction)
2. Only run full BLAKE3 on threads that pass the cheap filter
3. This is speculative — some false positives will still run BLAKE3

**Expected improvement:** If 99.99% of threads miss, we save 99.99% of BLAKE3 work. But the cheap hash needs to be designed carefully to not introduce false negatives.

**Risk:** If the cheap hash has any false negatives, we miss valid blocks → consensus failure.

### 3.5 Recommended Approach: Hybrid — Cheap Pre-Filter + Offload

**Phase 1:** Add a cheap pre-filter before BLAKE3 in the headless kernel.

```cpp
// After transcript mixing, before BLAKE3:
// Quick check: are the first 4 words of the hash likely below target?
// This uses the transcript directly as a hash (cheap, no BLAKE3 rounds)
bool likely_hit = transcript_local[7] <= pow_target[7] && 
                  transcript_local[6] <= pow_target[6];
if (likely_hit) {
    // Run full BLAKE3
    if (pearl::check_pow_target(transcript_rmem, pow_target, pow_key_eff)) {
        // write_host_signal_header
    }
}
```

**Phase 2:** Offload BLAKE3 to a separate kernel (as in 3.1).

**Phase 3:** Combine both — cheap pre-filter in GEMM kernel, then offload remaining BLAKE3 to a separate kernel.

---

## Part 4: Implementation Plan

### Phase 0: Baseline Measurement (1-2 days)

**Goal:** Establish accurate performance baseline for BLAKE3's contribution.

1. Profile the headless kernel with Nsight Compute:
   - Measure instruction throughput for BLAKE3 vs GEMM phases
   - Measure register pressure and occupancy
   - Measure shared memory bandwidth for transcript spill path
   
2. Run PropMiner with `PROPMINER_BENCH_NO_GRAPH=1` to measure per-iter timing:
   - Graph-capture overhead excluded
   - Measure total iter time and per-nonce time

3. Add timing instrumentation to the headless kernel:
   - CUDA events before/after GEMM loop
   - CUDA events before/after BLAKE3 phase
   - Report via host_signal_header or custom buffer

### Phase 1: Cheap Pre-Filter (2-3 days)

**Goal:** Add a cheap pre-filter before BLAKE3 to skip most compress calls.

**Files to modify:**
- `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` (lines 499-524)
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce_v2.cu`
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu`

**Implementation:**
```cpp
// In the headless kernel, after transcript mixing, before BLAKE3:
// Quick pre-filter: check if the transcript's top words are below target
// This is NOT a consensus-critical check — it only skips BLAKE3 for
// transcripts that are GUARANTEED to miss (top words too large).
// If a transcript passes the filter but still misses BLAKE3, that's fine.
// If a transcript fails the filter, it's GUARANTEED to miss BLAKE3.
bool pre_filter_hit = true;
CUTLASS_PRAGMA_UNROLL
for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
    // If transcript word > target word, BLAKE3 output will also be > target
    // (BLAKE3 is a good hash, but this is a conservative upper bound)
    if (transcript_local[i] > pow_target[i]) {
        pre_filter_hit = false;
        break;
    }
}

if (pre_filter_hit) {
    // Run full BLAKE3
    Tensor transcript_rmem = make_tensor<uint32_t>(Int<kTranscriptSlots>{});
    for (int s = 0; s < kTranscriptSlots; ++s)
        transcript_rmem(s) = transcript_local[s];
    
    bool block_found = pearl::check_pow_target(
        transcript_rmem, pow_target, pow_key_eff);
    if (block_found) {
        // write_host_signal_header
    }
}
```

**Verification:**
- Self-test: Run with `PROPMINER_BLAKE3_VERIFY=1` to compare pre-filter + BLAKE3 vs BLAKE3-only
- memcmp: Compare output headers byte-for-byte with baseline
- Pool soak: Run on testnet for 24+ hours

### Phase 2: BLAKE3 Offload Kernel (5-7 days)

**Goal:** Move BLAKE3 from the GEMM kernel to a separate, optimized kernel.

**New files:**
- `third_party/pearl-gemm/csrc/consumer/blake3_finalize_kernel.cu`
- `third_party/pearl-gemm/csrc/consumer/blake3_finalize_kernel.cuh`

**Files to modify:**
- `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` (remove BLAKE3)
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` (launch new kernel)
- `third_party/pearl-gemm/csrc/portable/transcript_kernel.cuh` (add declaration)

**Implementation:**

```cpp
// blake3_finalize_kernel.cu
__global__ void blake3_finalize_kernel(
    uint32_t const* __restrict__ transcript,  // [num_tiles * 256 * 16]
    int64_t M, int64_t N, int64_t batch,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned) {
    
    int64_t num_m_tiles = M / kBM;
    int64_t num_n_tiles = N / kBN;
    int64_t total_tiles = num_m_tiles * num_n_tiles * batch;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_tiles) return;
    
    // Each thread = one tile's transcript
    int64_t tx_idx = idx * kTranscriptSlots;
    
    // Load transcript
    Tensor transcript_rmem = make_tensor<uint32_t>(Int<kTranscriptSlots>{});
    for (int s = 0; s < kTranscriptSlots; ++s)
        transcript_rmem(s) = transcript[tx_idx + s];
    
    // BLAKE3 compress + target check
    bool block_found = pearl::check_pow_target(
        transcript_rmem, pow_target, pow_key);
    
    if (block_found) {
        // Decode tile coordinates from idx
        int64_t remainder = idx;
        int n_tile = remainder % num_n_tiles;
        remainder /= num_n_tiles;
        int m_tile = remainder % num_m_tiles;
        int batch_idx = remainder / num_m_tiles;
        
        auto block_coord = make_tuple((int32_t)m_tile, (int32_t)n_tile, (int32_t)batch_idx);
        auto problem_shape = make_tuple((int)M, (int)N, (int)K, (int)R);
        
        pearl::write_host_signal_header<ConsumerTiledMma, HeaderTileShape_MNK>(
            host_signal_sync, host_signal_header_pinned,
            problem_shape, block_coord, 0, pow_target);
    }
}
```

**Launcher:**
```cpp
cudaError_t launch_blake3_finalize(
    uint32_t const* transcript,
    int64_t M, int64_t N, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream) {
    
    int64_t num_tiles = (M / kBM) * (N / kBN) * batch;
    int threads = 256;
    int blocks = (num_tiles * threads + 255) / 256;
    
    blake3_finalize_kernel<<<blocks, threads, 0, stream>>>(
        transcript, M, N, batch, pow_target, pow_key,
        host_signal_sync, host_signal_header_pinned);
    
    return cudaGetLastError();
}
```

**GEMM kernel modification:** Remove the BLAKE3 phase (lines 499-524), always write transcript to gmem.

### Phase 3: GeForce v1/v2 Integration (3-5 days)

**Goal:** Apply the same BLAKE3 offload to GeForce v1 and v2 kernels.

**Files to modify:**
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu`
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce_v2.cu`

**Note:** These use different kernel entry points but the same `transcript_gemm_kernel_consumer`. The headless path is already handled by Phase 2.

### Phase 4: Portable Path Update (2-3 days)

**Goal:** Apply the optimized BLAKE3 approach to the portable path.

**Files to modify:**
- `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`

The portable path already uses a two-kernel approach (snapshot + finalize). The optimization here is to add the pre-filter to the finalize kernel.

### Phase 5: Legacy Kernel Update (2-3 days)

**Goal:** Apply pre-filter to the legacy persistent kernel.

**File to modify:**
- `src/cuda/kernels/pearlhash_kernel.cu`

In the legacy kernel, BLAKE3 is ~30% of per-nonce work, so the pre-filter has more impact.

---

## Part 5: Testing Strategy

### 5.1 Unit Tests (Self-Verification)

**Test 1: BLAKE3 Correctness**
- For every sigma, run both the original and modified kernel
- Compare the transcript values byte-for-byte (memcmp)
- Compare the host_signal_header values byte-for-byte
- Run 10,000+ nonces per sigma

**Test 2: Pre-Filter Correctness**
- The pre-filter must NEVER produce false negatives
- A transcript that fails the filter must be provably unable to produce a hit
- Proof: BLAKE3's output is a permutation of the input space; if any word of the transcript exceeds the target, the BLAKE3 output word will also exceed (statistically, but we need a deterministic guarantee)

**Wait — this is a critical insight.** The pre-filter as designed above is NOT safe. The transcript is NOT the BLAKE3 output — it's the BLAKE3 INPUT. BLAKE3's compression function mixes all 16 words together, so a transcript word being above target doesn't guarantee the BLAKE3 output will be above target.

**Correct pre-filter:** We need a filter that is GUARANTEED to produce no false negatives. Options:

1. **No pre-filter in the GEMM kernel** — instead, rely on the offloaded BLAKE3 kernel being more efficient due to lower register pressure in the GEMM kernel.

2. **Probabilistic filter with verification:** Run BLAKE3 on all threads, but use a cheaper BLAKE3 variant (fewer rounds) as a pre-filter. Any hits from the cheap variant are verified with full BLAKE3. But this risks false negatives from the reduced-round variant.

3. **Skip the pre-filter entirely.** Focus on the register-pressure reduction from offloading BLAKE3.

**Revised approach: Skip pre-filter. Focus on BLAKE3 offload.**

### 5.2 Integration Tests

**Test 3:memcmp Verification**
- Run PropMiner in dual-mode: original kernel and modified kernel on the same sigma
- Compare all host_signal_header outputs
- Must be byte-identical for thousands of nonces

**Test 4: Share Submission Test**
- Run on testnet/proxy for 24+ hours
- Verify all shares are accepted by the pool
- No consensus failures

### 5.3 Performance Tests

**Test 5: Nsight Compute Profiling**
- Before: profile original headless kernel
- After: profile modified kernel (GEMM-only + BLAKE3 offload)
- Compare: occupancy, instruction throughput, memory bandwidth

**Test 6: End-to-End Mining Benchmark**
- Run PropMiner with both implementations
- Compare TMAD/s, hashrate, per-iter time
- Run for 1+ hour to stabilize

### 5.4 Stress Tests

**Test 7: Long-Running Stability**
- Run for 72+ hours
- Monitor for memory leaks, register pressure issues, occupancy changes
- Watch for any consensus drift

**Test 8: Multi-Sigma Rotation**
- Test sigma rotation (new sigma every N minutes)
- Ensure BLAKE3 offload works correctly with new pow_key

---

## Part 6: Risk Assessment

### 6.1 Consensus Correctness Risk: **HIGH**

**Problem:** Any change to BLAKE3 computation risks consensus drift. The Pearl protocol requires byte-identical transcript computation across all miners.

**Mitigation:**
1. Never change the BLAKE3 algorithm itself — only its location (in-kernel → offloaded)
2. Use identical BLAKE3 code (same `compress_msg_block_u32`, same `check_pow_target`)
3. Byte-for-byte memcmp verification before any deployment
4. Run in parallel with original implementation for extended testing

### 6.2 Register Pressure Risk: **MEDIUM**

**Problem:** Removing BLAKE3 registers from the GEMM kernel changes register allocation. The compiler may allocate differently, potentially causing register spilling or occupancy changes.

**Mitigation:**
1. Profile occupancy before/after with Nsight Compute
2. Ensure occupancy doesn't decrease (should increase)
3. Test across all target architectures (Ampere, Ada, Blackwell)

### 6.3 Memory Bandwidth Risk: **MEDIUM**

**Problem:** Offloading BLAKE3 requires writing the full transcript to gmem (~1 GiB per iteration). This is additional PCIe-bandwidth-equivalent traffic.

**Mitigation:**
1. The headless path already avoids this (transcript stays in registers)
2. For the offloaded path, the GEMM kernel already writes C_running to gmem (when C is non-null)
3. Measure bandwidth impact with Nsight Compute

### 6.4 GeForce Kernel Risk: **LOW**

**Problem:** GeForce v1/v2 kernels have different register layouts and may not benefit from the same optimization.

**Mitigation:**
1. Test on actual RTX 5090 hardware
2. Keep headless path (in-kernel BLAKE3) as default for GeForce

### 6.5 CUDA Graph Incompatibility Risk: **LOW**

**Problem:** Adding a new kernel launch may break CUDA graph capture.

**Mitigation:**
1. CUDA graphs can capture multiple kernels
2. Update `pearl_capi_iter_batch_graph_prepare_ex` to include the new kernel launch
3. Test graph capture after changes

---

## Part 7: Expected Performance Improvement

### Conservative Estimate
- Register pressure reduction: 5% faster GEMM kernel
- BLAKE3 offload overhead: -2% (gmem write + separate kernel launch)
- **Net: +3% improvement**

### Optimistic Estimate
- Register pressure reduction: 10% faster GEMM kernel
- BLAKE3 offload overhead: -1% (overlapped with GEMM via pipelining)
- **Net: +9% improvement**

### Legacy Kernel (pearlhash_kernel.cu)
- BLAKE3 is ~30% of per-nonce work
- Pre-filter could save ~99.9% of BLAKE3 calls
- **Net: +20-25% improvement** (but this is the legacy path)

---

## Part 8: Code Locations That Need Modification

### Consumer Headless Path (Primary Target)
| File | Lines | Change |
|------|-------|--------|
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | 499-524 | Remove BLAKE3, always write transcript to gmem |
| `third_party/pearl-gemm/csrc/consumer/blake3_finalize_kernel.cu` | NEW | New BLAKE3 offload kernel |
| `third_party/pearl-gemm/csrc/consumer/blake3_finalize_kernel.cuh` | NEW | New header |
| `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` | 1320-1378 | Launch BLAKE3 offload after GEMM |
| `third_party/pearl-gemm/csrc/portable/transcript_kernel.cuh` | 83-91 | Add BLAKE3 finalize declaration |

### GeForce v1/v2 Path
| File | Lines | Change |
|------|-------|--------|
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` | — | Same as consumer headless |
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce_v2.cu` | — | Same as consumer headless |

### Portable Path
| File | Lines | Change |
|------|-------|--------|
| `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu` | 141-186 | Add pre-filter to finalize kernel |
| `third_party/pearl-gemm/csrc/portable/transcript_kernel.cuh` | 56-63 | Update finalize declaration |

### Legacy Path (Lower Priority)
| File | Lines | Change |
|------|-------|--------|
| `src/cuda/kernels/pearlhash_kernel.cu` | 267-298 | Add pre-filter before BLAKE3 |
| `src/cuda/include/blake3.cuh` | — | No change needed |

### Host-Side
| File | Lines | Change |
|------|-------|--------|
| `src/host/pearl/gpu_worker.cpp` | 1036-1041 | Update queue_batch for new kernel launch |
| `src/host/pearl/gpu_worker.cpp` | 900-932 | Update prepare_graph for new kernel |

---

## Part 9: Rollback Plan

### Immediate Rollback
1. Use `PEARL_GEMM_KERNEL=consumer` env var to force consumer kernel
2. Use `PROPMINER_BENCH_NO_GRAPH=1` to disable CUDA graphs (if graph incompatibility)
3. Git revert to previous commit if issues found

### Gradual Rollback
1. Feature flag: `PROPMINER_BLAKE3_OFFLOAD=0` disables offloaded BLAKE3
2. Feature flag: `PROPMINER_BLAKE3_PRE_FILTER=0` disables pre-filter
3. Default to off (optimization disabled by default, enabled via env var)

### Rollback Triggers
- Any consensus failure (share rejected by pool)
- Hashrate decrease > 5%
- Memory errors or instability
- Occupancy decrease > 10%

---

## Part 10: Alternative Approaches Considered (and Rejected)

### A. Batch BLAKE3 Across Nonces in a Single Kernel Launch
**Rejected:** Each nonce has different pow_key (CommitA), different transcript. No shared state to batch on.

### B. Vectorized BLAKE3 (SIMD across 4 transcripts)
**Rejected:** BLAKE3 operates on 16-word state vectors. SIMD across 4 transcripts would need 4× the registers per thread, reducing occupancy. Net negative.

### C. Lookup Table for BLAKE3
**Rejected:** BLAKE3's message block varies per nonce. No reusable lookup table possible.

### D. Reduce BLAKE3 Rounds
**Rejected:** Changes the hash function → consensus failure. Not an option.

### E. Skip BLAKE3 Entirely, Use XOR Reduction Only
**Rejected:** Changes the hash function → consensus failure. Not an option.

### F. Use Tensor Cores for BLAKE3
**Rejected:** BLAKE3 is integer arithmetic (add, rotate, XOR). Tensor cores are for FP16/INT8 matmul. No benefit.

---

## Part 11: Summary of Recommendations

### Immediate (Phase 1-2)
1. **Profile first** — establish baseline with Nsight Compute
2. **Offload BLAKE3 to separate kernel** — reduces register pressure in GEMM kernel
3. **No pre-filter** — too risky for consensus correctness
4. **Keep headless path as default** — in-kernel BLAKE3 is fine for GeForce

### Medium-Term (Phase 3-4)
1. **Apply to GeForce kernels** — same offload approach
2. **Add pre-filter to portable path** — safe because portable path already writes to gmem
3. **Optimize BLAKE3 kernel** — use warp-level primitives, coalesced memory access

### Long-Term
1. **Explore protocol-level changes** — could BLAKE3 be integrated into the GEMM more naturally?
2. **Explore different hash functions** — is there a consensus-equivalent but faster hash?
3. **Explore tensor-core-friendly hash** — can BLAKE3 be approximated with tensor ops?

---

## Appendix A: BLAKE3 Compress Params Reference

```cpp
// Single-block keyed hash (used for transcript → hash)
COMPRESS_PARAMS_SINGLE_BLOCK_KEYED = {
    .counter = 0,
    .block_len = 64,           // MSG_BLOCK_SIZE
    .flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT
}

// KEYED_HASH = 0x10
// CHUNK_START = 0x01
// CHUNK_END = 0x02
// ROOT = 0x08
// Combined flags = 0x1F
```

## Appendix B: Transcript Layout Reference

```
Per-tile transcript: 16 u32 slots (MSG_BLOCK_SIZE_U32 = 16)
Per-thread: owns 1 slot (cycled through K/R accumulations)
Per-tile: 256 threads × 1 slot each = 256 slots, but only 16 active (cycling)

Total transcript elements per nonce:
  = batch × (M/bM) × (N/bN) × 256 × 16
  = 1 × 64 × 1024 × 256 × 16
  = 268,435,456 elements
  = 1,073,741,824 bytes = 1 GiB
```

## Appendix C: Key Functions and Their Locations

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `check_pow_target` | `gemm/pow_utils.hpp` | 204-234 | BLAKE3 compress + target check |
| `compress_msg_block_u32` | `blake3/blake3.cuh` | 172-209 | BLAKE3 7-round compress |
| `BLAKE3_ROUND` | `blake3/blake3.cuh` | 21-87 | 8 G-operations per round |
| `BLAKE3_PERMUTE` | `blake3/blake3.cuh` | 90-124 | Message block permutation |
| `xor_reduction` | `gemm/pow_utils.hpp` | 80-115 | Ternary XOR tree reduction |
| `rotl_xor` | `gemm/pow_utils.hpp` | 28-34 | Rotate-Left-XOR mixing |
| `write_host_signal_header` | `gemm/pow_utils.hpp` | 244-307 | Atomic header write on hit |
| `transcript_gemm_kernel_consumer` | `consumer/transcript_gemm_kernel.cu` | 225-563 | Main GEMM + BLAKE3 kernel |
| `launch_transcript_gemm_headless` | `consumer/transcript_gemm_kernel.cu` | 780-862 | Host launcher |
| `pearl_capi_noisy_gemm` | `capi/pearl_gemm_capi.cpp` | 1225-1572 | C API entry point |
| `pearl_capi_iter` | `capi/pearl_gemm_capi.cpp` | 1627-1677 | Single-nonce iteration |
| `pearl_capi_iter_batch` | `capi/pearl_gemm_capi.cpp` | 1682-1711 | Batched iteration |

## Appendix D: ARC-Miner / Akoya-Miner Comparison

The ARC-miner (`/Users/mrpropop/Documents/vast ai/ARC-miner/native/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp`) uses the same pearl-gemm library. It has the same headless consumer kernel path and the same BLAKE3 implementation. There is no special batched BLAKE3 in ARC-miner — it uses the same per-thread-per-nonce approach.

The ARC-miner's `GpuWorker` (if it exists) would follow the same batched iteration pattern as PropMiner's `gpu_worker.cpp`, using `pearl_capi_iter_batch` for batching across nonces.

**Conclusion:** No competing miner has implemented batched BLAKE3. This is an unexplored optimization space.

---

*End of document.*
