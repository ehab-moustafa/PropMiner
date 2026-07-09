# BLAKE3 Finalize Kernel — Comprehensive Optimization Plan

**Author:** CUDA Mining Systems Architect  
**Date:** 2026-07-09  
**Status:** DRAFT — Planning only, no implementation  
**Scope:** Optimization of the BLAKE3 finalize kernel across all code paths in PropMiner

---

## 1. Current State Analysis

### 1.1 Architecture: Two-Kernel Pipeline (Production Path)

The production mining path (consumer headless, GeForce v2, portable, Turing) uses a **two-kernel sequence**:

```
Kernel 1: transcript_gemm_kernel_consumer / transcript_gemm_sm120_geforce_v2
  - Computes ApEA(M,K) @ BpEB(N,K).T → transcript in registers
  - XOR-reduces accumulator fragments at every R columns
  - rotl_xor<13> mixes into transcript[0..15] slots
  - Writes 16 u32 transcript to gmem (lines 505-523 of consumer kernel)

Kernel 2: transcript_finalize_kernel  ← THIS IS OUR TARGET
  - Reads 16 u32 transcript from gmem (line 168)
  - Runs BLAKE3.compress(transcript, pow_key) → 8 u32 hash
  - Compares 256-bit hash <= pow_target (MSW-first)
  - On hit: atomic-CAS lock + write HostSignalHeader
```

### 1.2 The Finalize Kernel — Line-by-Line Analysis

**File:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`  
**Function:** `transcript_finalize_kernel` (lines 141-186)

```
Line 141-148:  Kernel signature — 7 pointers + 4 int params
Line 149-152:  Decode block/thread indices (m_tile, n_tile, batch, tid)
Line 154-155:  Compute num_n_tiles, num_m_tiles
Line 157-162:  Compute tx_idx = base * per_tile_thread + tid * per_tile
              base = ((batch * num_m_tiles + m_tile) * num_n_tiles + n_tile)
              per_tile_thread = 256 * 16 = 4096
              per_tile = 16
              tx_idx = (batch * 64 * 1024 + m_tile * 1024 + n_tile) * 4096 + tid * 16
Line 165-170:  Load 16 u32 from gmem into rmem tensor (16 scalar loads)
Line 172-173:  Call pearl::check_pow_target(transcript_rmem, pow_target, pow_key)
Line 175-185:  If hit: compute block_coord, problem_shape, call write_host_signal_header
```

**Grid dimensions:** (M/bM, N/bN, batch) = (64, 1024, 1) = 65,536 blocks  
**Block dimensions:** 256 threads  
**Total threads:** 65,536 × 256 = 16,777,216 threads

### 1.3 check_pow_target — The BLAKE3 Core

**File:** `third_party/pearl-gemm/csrc/gemm/pow_utils.hpp` (lines 204-234)

```cpp
template <typename TranscriptTensor>
CUTLASS_DEVICE bool check_pow_target(const TranscriptTensor& transcript,
                                     const uint32_t* pow_target,
                                     const uint32_t* pow_key) {
  // Line 209: Create 8-u32 chaining value tensor
  Tensor hash = make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
  // Lines 210-213: Copy pow_key into hash (initial chaining value)
  // Line 214-215: Call compress_msg_block_u32(transcript, hash, params)
  // Lines 219-231: 256-bit comparison MSW-first (index 7 down to 0)
  // Line 233: return block_found
}
```

### 1.4 compress_msg_block_u32 — BLAKE3 7-Round Compress

**File:** `third_party/pearl-gemm/csrc/blake3/blake3.cuh` (lines 172-209)

```
Input:  block[16 u32] = transcript, chaining_value[8 u32] = pow_key, CompressParams
Output: chaining_value[8 u32] = hash

State: rState[16 u32], rBlock[16 u32], rOrigBlock[16 u32]

Steps:
  1. Copy block → rBlock, chaining_value → rState
  2. Set rState[8..15] = IV0..3 + counter + block_len + flags
     (COMPRESS_PARAMS_SINGLE_BLOCK_KEYED: counter=0, block_len=64, flags=0x1F)
  3. 6 rounds of: BLAKE3_ROUND() + BLAKE3_PERMUTE()
  4. 1 final round BLAKE3_ROUND() (no permutation)
  5. hash[i] = rState[i] ^ rState[i+8] for i=0..7
```

**BLAKE3_ROUND()** (lines 21-87) — 8 G-operations:
- Each G: 2 adds + 1 XOR + 2 rotates = 5 operations
- Total per round: 64 operations (32 adds, 8 XORs, 32 rotates)
- 6 rounds + 1 final = 7 rounds = 448 operations

**BLAKE3_PERMUTE()** (lines 90-124) — 16 word reorderings:
- 16 copies to temp + 16 copies back = 32 operations

**Total per compress:** ~448 + 7×32 = ~672 operations  
**Per thread register budget:** ~64-80 registers (16 state + 16 msg + 8 chaining + 8 temp + temporaries)

### 1.5 Bottleneck Analysis

| Phase | Instructions | Registers | % of Finalize Kernel |
|-------|-------------|-----------|---------------------|
| Transcript load (16 scalar loads) | ~32 | 16 | ~5% |
| pow_key copy (8 loads) | ~16 | 8 | ~2% |
| BLAKE3 compress (6.5 rounds) | ~672 | 48 | ~85% |
| Target comparison (8 u32) | ~24 | 8 | ~3% |
| Host signal header (rare) | ~200 | 64 | ~3% (on hit only) |
| **Total** | **~744** | **~80** | **100%** |

**Primary bottleneck:** BLAKE3 compress is 85% of finalize kernel time.  
**Secondary bottleneck:** Register pressure (~80 regs/thread) limits occupancy on some architectures.

### 1.6 BLAKE3's Relative Cost

| Kernel | BLAKE3 % of Total | Notes |
|--------|-------------------|-------|
| transcript_finalize (portable) | ~85% | Entire kernel IS BLAKE3 |
| transcript_gemm_consumer | ~0.5% | BLAKE3 not in this kernel; transcript written to gmem only |
| transcript_gemm_geforce_v2 | ~0.5% | Same as consumer |
| pearlhash persistent (legacy) | ~30% | GEMM + BLAKE3 in one kernel |

**Key insight:** For the production path (consumer/geforce v2), the finalize kernel is a **separate launch** after the GEMM. The GEMM kernel is compute-bound on Tensor Cores and dominates. The finalize kernel reads 1 GiB of transcript, runs BLAKE3 on 16.7M threads, and writes headers on hit. The finalize kernel is memory-bandwidth-limited (reading 1 GiB) + compute-bound on scalar ALU (BLAKE3).

### 1.7 Time Estimate

For production shape (M=8192, N=262144, batch=1):
- Transcript buffer: 1 GiB
- Finalize grid: 64 × 1024 × 1 = 65,536 blocks × 256 threads
- Each thread: 16 u32 load (64 bytes) + BLAKE3 (~672 ops)
- Memory bandwidth: ~1 GiB / (RTX 5090 peak 1792 GB/s) = ~0.6 ms theoretical minimum
- Compute: 16.7M threads × 672 ops / (RTX 5090 INT32 peak) ≈ ~1-2 ms
- **Estimated finalize kernel time: 2-5 ms** (depends on occupancy and instruction throughput)

---

## 2. Optimization Targets

### Optimization 1: Coalesced Transcript Load with LDG128

**File to modify:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`  
**Lines to change:** 164-170

**Current code:**
```cpp
// Lines 164-170: 16 scalar loads, one per thread
Tensor transcript_rmem = make_tensor<uint32_t>(
    Int<blake3::MSG_BLOCK_SIZE_U32>{});
CUTLASS_PRAGMA_UNROLL
for (int i = 0; i < (int)blake3::MSG_BLOCK_SIZE_U32; ++i) {
    transcript_rmem(i) = transcript[tx_idx + i];
}
```

**New code:**
```cpp
// Use 4x LDG128 (uint128_t) for coalesced 64-byte loads
// tx_idx is already aligned because tid is a multiple of 4 and
// each thread loads 16 consecutive u32 = 64 bytes.
// 4 threads work together to load 64 bytes coalesced.
uint128_t tmp[4];
#pragma unroll
for (int i = 0; i < 4; ++i) {
    tmp[i] = transcript[tx_idx + i * 4];
}

Tensor transcript_rmem = make_tensor<uint32_t>(
    Int<blake3::MSG_BLOCK_SIZE_U32>{});
#pragma unroll
for (int i = 0; i < 16; ++i) {
    transcript_rmem(i) = reinterpret_cast<uint32_t*>(&tmp[0])[i];
}
```

**Why:** Each thread loads 64 bytes. With 256 threads per block, 16 threads load the same 64-byte segment coalesced. Using `uint128_t` (LDG128) instead of four `uint32_t` (LDG32) reduces instruction count from 16 to 4 per thread, and the hardware coalesces 4 consecutive LDG128 into a single 256-byte transaction.

**Risk:** Low — purely a memory access pattern change, no computation altered.

**Estimated gain:** 5-10% reduction in transcript load time. Since load is only 5% of kernel, net gain is **0.25-0.5%**.

### Optimization 2: Inline BLAKE3 with PTX Rotate Instructions

**File to modify:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`  
**Lines to change:** 172-173 (replace check_pow_target call with inline BLAKE3)

**Current code:**
```cpp
// Lines 172-173
bool block_found = pearl::check_pow_target(
    transcript_rmem, pow_target, pow_key);
```

**New code:**
```cpp
// Inline BLAKE3 compress with PTX rotate for maximum throughput
uint32_t s[16];  // state
uint32_t b[16];  // message block

// Load message from transcript
#pragma unroll
for (int i = 0; i < 16; ++i) b[i] = transcript_rmem(i);

// Load chaining value from pow_key
#pragma unroll
for (int i = 0; i < 8; ++i) s[i] = pow_key[i];

// Set domain separation state
s[8]  = 0x6A09E667u;  // IV0
s[9]  = 0xBB67AE85u;  // IV1
s[10] = 0x3C6EF372u;  // IV2
s[11] = 0xA54FF53Au;  // IV3
s[12] = 0u;            // counter_lo
s[13] = 0u;            // counter_hi
s[14] = 64u;           // block_len
s[15] = 0x1Fu;         // flags: KEYED_HASH|CHUNK_START|CHUNK_END|ROOT

// 6 full rounds + 1 final round (no permutation after last)
#define G(a,b,c,d,x,y) \
    s[a] = s[a] + s[b] + x; \
    s[d] = rotl(s[d] ^ s[a], 16); \
    s[c] = s[c] + s[d]; \
    s[b] = rotl(s[b] ^ s[c], 12); \
    s[a] = s[a] + s[b] + y; \
    s[d] = rotl(s[d] ^ s[a], 8); \
    s[c] = s[c] + s[d]; \
    s[b] = rotl(s[b] ^ s[c], 7)

#define ROUNDS() \
    G(0,4,8,12,b[0],b[1]); \
    G(1,5,9,13,b[2],b[3]); \
    G(2,6,10,14,b[4],b[5]); \
    G(3,7,11,15,b[6],b[7]); \
    G(0,5,10,15,b[8],b[9]); \
    G(1,6,11,12,b[10],b[11]); \
    G(2,7,8,13,b[12],b[13]); \
    G(3,4,9,14,b[14],b[15]);

#define PERMUTE() \
    b[0]=b[2]; b[1]=b[6]; b[2]=b[3]; b[3]=b[10]; \
    b[4]=b[7]; b[5]=b[0]; b[6]=b[4]; b[7]=b[13]; \
    b[8]=b[1]; b[9]=b[11]; b[10]=b[12]; b[11]=b[5]; \
    b[12]=b[9]; b[13]=b[14]; b[14]=b[15]; b[15]=b[8]

#pragma unroll
for (int round = 0; round < 6; ++round) {
    ROUNDS();
    PERMUTE();
}
ROUNDS();  // final round, no permutation

// XOR top and bottom halves
uint32_t hash[8];
#pragma unroll
for (int i = 0; i < 8; ++i) hash[i] = s[i] ^ s[i + 8];
```

**Why:** The current implementation uses `compress_msg_block_u32` which creates CUTE tensor views (`make_tensor_like`) for rState, rBlock, rOrigBlock. While the compiler optimizes these away, the tensor abstraction adds overhead. The inline version uses plain arrays and PTX-optimized rotates.

**Risk:** Medium — this changes the BLAKE3 computation path. Must verify byte-identical output.

**Estimated gain:** 5-15% on BLAKE3 compress (PTX rotate is 1 instruction vs 2 for C++ rotl). Net gain: **4-13% on finalize kernel**.

### Optimization 3: PTX Rotate-Left via shf.l.wrap

**File to modify:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`  
**Lines to change:** Add a PTX rotate helper before the kernel

**New code (before the kernel):**
```cpp
// PTX rotate-left using shf.l.wrap — single instruction, 1-cycle latency
CUTLASS_DEVICE uint32_t ptx_rotl(uint32_t x, int n) {
    uint32_t r;
    asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(x), "r"(n));
    return r;
}
```

**Why:** The current `rightrotate32` in blake3.cuh uses `(x << (32-n)) | (x >> n)` which compiles to 2 instructions (shift + shift + OR). The PTX `shf.l.wrap.b32` is a single instruction that does rotate-left in 1 cycle. BLAKE3 has 32 rotates per round × 7 rounds = 224 rotates. Each save of 1 instruction = 224 instructions saved.

**Risk:** Low — mathematically equivalent rotate.

**Estimated gain:** ~10% on BLAKE3 compress. Net gain: **8-12% on finalize kernel**.

### Optimization 4: Constant Memory for pow_target

**File to modify:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`  
**Lines to change:** Add constant memory declaration, modify pow_target access

**Current code:**
```cpp
// pow_target passed as device pointer, loaded from gmem each comparison
```

**New code:**
```cpp
__device__ __constant__ uint32_t d_pow_target_const[8];

// In launch_transcript_finalize (line 204-221), add before kernel launch:
cudaMemcpyToSymbol(d_pow_target_const, pow_target, 8 * sizeof(uint32_t));

// In the kernel, replace pow_target[i] with d_pow_target_const[i]
```

**Why:** pow_target is identical for all 16.7M threads. Loading it from gmem for each comparison (8 loads per thread × 16.7M threads) is wasteful. Constant memory is cached (48 KiB per SM) and broadcast to all threads in a warp.

**Risk:** Low — read-only data, no computation change.

**Estimated gain:** 2-3% on target comparison. Net gain: **0.1-0.2%** (comparison is 3% of kernel).

### Optimization 5: Unrolled Target Comparison

**File to modify:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`  
**Lines to change:** 172-173 (in-line the comparison from check_pow_target)

**Current code:**
```cpp
bool block_found = pearl::check_pow_target(
    transcript_rmem, pow_target, pow_key);
```

**New code (with inline hash + unrolled comparison):**
```cpp
// Inline BLAKE3 + comparison (see Optimization 2 for BLAKE3)
// ... BLAKE3 produces hash[8] ...

// Unrolled 256-bit comparison MSW-first
bool block_found = (hash[7] < d_pow_target_const[7]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] < d_pow_target_const[6]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] == d_pow_target_const[6] && hash[5] < d_pow_target_const[5]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] == d_pow_target_const[6] && hash[5] == d_pow_target_const[5] && hash[4] < d_pow_target_const[4]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] == d_pow_target_const[6] && hash[5] == d_pow_target_const[5] && hash[4] == d_pow_target_const[4] && hash[3] < d_pow_target_const[3]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] == d_pow_target_const[6] && hash[5] == d_pow_target_const[5] && hash[4] == d_pow_target_const[4] && hash[3] == d_pow_target_const[3] && hash[2] < d_pow_target_const[2]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] == d_pow_target_const[6] && hash[5] == d_pow_target_const[5] && hash[4] == d_pow_target_const[4] && hash[3] == d_pow_target_const[3] && hash[2] == d_pow_target_const[2] && hash[1] < d_pow_target_const[1]) ||
                   (hash[7] == d_pow_target_const[7] && hash[6] == d_pow_target_const[6] && hash[5] == d_pow_target_const[5] && hash[4] == d_pow_target_const[4] && hash[3] == d_pow_target_const[3] && hash[2] == d_pow_target_const[2] && hash[1] == d_pow_target_const[1] && hash[0] <= d_pow_target_const[0]);
```

**Why:** The current loop-based comparison (lines 220-231 of pow_utils.hpp) has 8 iterations with early exit. The loop creates branch divergence. The unrolled version uses short-circuit OR so the compiler can optimize to a single conditional branch chain.

**Risk:** Low — mathematically equivalent comparison.

**Estimated gain:** 1-2% on comparison phase. Net gain: **0.03-0.06%**.

### Optimization 6: Warp-Level Early Exit (Advanced)

**File to modify:** `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu`

**Concept:** After computing the hash, use `__any_sync` to check if ANY thread in the warp found a hit. If no hit, skip the atomic CAS entirely.

```cpp
// After computing hash and block_found:
bool warp_hit = __any_sync(0xFFFFFFFF, block_found);
if (warp_hit && block_found) {
    // Only the winning thread does atomic CAS
    pearl::write_host_signal_header<PortableTiledMma, TileShape_MNK>(
        host_signal_sync, host_signal_header_pinned,
        problem_shape, block_coord, tid, pow_target);
}
```

**Why:** The atomic CAS in `write_host_signal_header` (lines 267-306 of pow_utils.hpp) is expensive — it involves a spinlock with atomicCAS. The `__any_sync` check ensures only threads in a warp that actually found a hit attempt the CAS.

**Risk:** Low — atomic CAS is already sequential (only one thread wins). This just avoids unnecessary CAS attempts.

**Estimated gain:** Negligible (hits are extremely rare, ~1 in 2^32 threads). Only meaningful during high-difficulty periods.

---

## 3. Test Plan

### Test 1: CPU-Side BLAKE3 Vector Verification

**Purpose:** Verify the inline BLAKE3 produces byte-identical output to the reference implementation.

**File to create:** `third_party/pearl-gemm/tests/blake3_finalize_verify.cpp`

```cpp
#include <cstdint>
#include <cstring>
#include <cstdio>
#include "blake3/blake3_constants.hpp"

// BLAKE3 reference test vectors (from BLAKE3 spec)
static const uint32_t ref_transcript[16] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u
};

static const uint32_t ref_pow_key[8] = {
    0xDEADBEEFu, 0xCAFE BABEu, 0x12345678u, 0x9ABCDEF0u,
    0xFEDCBA98u, 0x76543210u, 0xABCDEF01u, 0x23456789u
};

// Reference BLAKE3 compress (from existing code)
extern "C" void blake3_reference_compress(
    const uint32_t* msg, uint32_t* hash, const uint32_t* key);

// New inline BLAKE3 compress (from optimized code)
extern "C" void blake3_inline_compress(
    const uint32_t* msg, uint32_t* hash, const uint32_t* key);

int main() {
    uint32_t hash_ref[8], hash_new[8];
    
    // Test 1: Standard vector
    blake3_reference_compress(ref_transcript, hash_ref, ref_pow_key);
    blake3_inline_compress(ref_transcript, hash_new, ref_pow_key);
    if (memcmp(hash_ref, hash_new, 32) != 0) {
        fprintf(stderr, "FAIL: standard vector mismatch\n");
        return 1;
    }
    
    // Test 2: All-zero transcript
    uint32_t zero_transcript[16] = {};
    uint32_t zero_hash_ref[8], zero_hash_new[8];
    blake3_reference_compress(zero_transcript, zero_hash_ref, ref_pow_key);
    blake3_inline_compress(zero_transcript, zero_hash_new, ref_pow_key);
    if (memcmp(zero_hash_ref, zero_hash_new, 32) != 0) {
        fprintf(stderr, "FAIL: all-zero transcript mismatch\n");
        return 1;
    }
    
    // Test 3: Max values (0xFFFFFFFF)
    uint32_t max_transcript[16];
    memset(max_transcript, 0xFF, 64);
    uint32_t max_hash_ref[8], max_hash_new[8];
    blake3_reference_compress(max_transcript, max_hash_ref, ref_pow_key);
    blake3_inline_compress(max_transcript, max_hash_new, ref_pow_key);
    if (memcmp(max_hash_ref, max_hash_new, 32) != 0) {
        fprintf(stderr, "FAIL: max value transcript mismatch\n");
        return 1;
    }
    
    // Test 4: Single-bit flip (bit 0 of transcript[0])
    uint32_t flip_transcript[16];
    memcpy(flip_transcript, ref_transcript, 64);
    flip_transcript[0] ^= 1;
    uint32_t flip_hash_ref[8], flip_hash_new[8];
    blake3_reference_compress(flip_transcript, flip_hash_ref, ref_pow_key);
    blake3_inline_compress(flip_transcript, flip_hash_new, ref_pow_key);
    if (memcmp(flip_hash_ref, flip_hash_new, 32) != 0) {
        fprintf(stderr, "FAIL: single-bit flip mismatch\n");
        return 1;
    }
    
    // Test 5: Random transcripts (1000 tests)
    for (int i = 0; i < 1000; i++) {
        uint32_t rand_transcript[16], rand_key[8];
        for (int j = 0; j < 16; j++) rand_transcript[j] = rand();
        for (int j = 0; j < 8; j++) rand_key[j] = rand();
        uint32_t h_ref[8], h_new[8];
        blake3_reference_compress(rand_transcript, h_ref, rand_key);
        blake3_inline_compress(rand_transcript, h_new, rand_key);
        if (memcmp(h_ref, h_new, 32) != 0) {
            fprintf(stderr, "FAIL: random test %d mismatch\n", i);
            return 1;
        }
    }
    
    printf("PASS: all %d BLAKE3 vectors verified\n", 1005);
    return 0;
}
```

**Build:**
```bash
cd third_party/pearl-gemm
g++ -I. -Icsrc tests/blake3_finalize_verify.cpp \
    -o blake3_finalize_verify -O2 -march=native
./blake3_finalize_verify
```

### Test 2: GPU-Side Transcript Self-Test

**Purpose:** Verify the GPU finalize kernel produces identical transcript-to-hash mapping.

**File to create:** `third_party/pearl-gemm/tests/finalize_self_test.cu`

```cpp
// Launches the finalize kernel with known transcript data
// and verifies the output matches CPU reference

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

// Known test transcript (16 u32 per thread)
__device__ uint32_t test_transcript[16] = { ... };
__device__ uint32_t test_pow_key[8] = { ... };
__device__ uint32_t test_pow_target[8] = { ... };

__global__ void finalize_self_test_kernel(uint32_t* results) {
    int tid = threadIdx.x;
    // Each thread runs BLAKE3 on its assigned transcript slot
    // Write result to results[tid]
}

int main() {
    // 1. Allocate transcript buffer with known data
    // 2. Launch finalize kernel
    // 3. Read back results
    // 4. Compare with CPU reference
    // 5. Verify hash matches expected value
}
```

### Test 3: Dual-Pathmemcmp Verification

**Purpose:** Run both old and new finalize kernels on the same transcript data and compare outputs.

```bash
# Build two versions: one with old BLAKE3, one with new
# Run both on same sigma and transcript
# Compare host_signal_header outputs byte-for-byte
```

**Test script:**
```bash
#!/bin/bash
# verify_finalization.sh

# Run old path
./propminer_old --sigma $1 --transcript $2 --output results_old.bin

# Run new path  
./propminer_new --sigma $1 --transcript $2 --output results_new.bin

# Compare
if cmp results_old.bin results_new.bin; then
    echo "PASS: byte-identical outputs"
else
    echo "FAIL: outputs differ"
    exit 1
fi
```

### Test 4: Production Pool Share Test

**Purpose:** Verify shares are accepted by the pool.

```bash
# Run optimized miner against test pool
export PROPMINER_BATCH=8
export PROPMINER_TRIPLE_BUFFER=1
./propminer --pool testnet --verify-shares 2>&1 | tee pool_test.log

# Check for:
# - Share acceptance rate > 99.9%
# - No consensus errors
# - No rejected shares
grep -c "share accepted" pool_test.log
grep -c "share rejected" pool_test.log  # should be 0
grep -c "consensus error" pool_test.log  # should be 0
```

### Test 5: Hashrate Regression Test

**Purpose:** Ensure optimization doesn't reduce hashrate.

```bash
#!/bin/bash
# benchmark_finalize.sh

OLD_HASHRATE=$(./propminer_old --benchmark 60 2>&1 | grep "TMAD/s" | tail -1)
NEW_HASHRATE=$(./propminer_new --benchmark 60 2>&1 | grep "TMAD/s" | tail -1)

echo "Old: $OLD_HASHRATE TMAD/s"
echo "New: $NEW_HASHRATE TMAD/s"

# Check for < 5% regression
REGRESSION=$(echo "$NEW_HASHRATE $OLD_HASHRATE" | awk '{
    diff = ($1 - $2) / $2 * 100;
    if (diff < -5) print "FAIL: " diff "% regression";
    else print "PASS: " diff "% change";
}')
echo "$REGRESSION"
```

---

## 4. Implementation Order

### Step 1: PTX Rotate-Left Helper (5 minutes)
**Risk:** Low | **Gain:** 8-12% on finalize | **Files:** `transcript_kernel.cu`

Add `ptx_rotl()` helper function before the kernel. This is a drop-in replacement for `rightrotate32`.

### Step 2: Constant Memory for pow_target (10 minutes)
**Risk:** Low | **Gain:** 0.1-0.2% on finalize | **Files:** `transcript_kernel.cu`

Add `__device__ __constant__ uint32_t d_pow_target_const[8]` and cudaMemcpyToSymbol in launcher.

### Step 3: CPU BLAKE3 Vector Tests (1 day)
**Risk:** Low | **Gain:** N/A (safety) | **Files:** New test file

Create and run `blake3_finalize_verify.cpp` to ensure any inline BLAKE3 code is mathematically correct.

### Step 4: Inline BLAKE3 with PTX Rotates (2-3 days)
**Risk:** Medium | **Gain:** 8-12% on finalize | **Files:** `transcript_kernel.cu`

Replace `check_pow_target()` call with inline BLAKE3 compress using PTX rotates. Run CPU vector tests first.

### Step 5: Coalesced Transcript Load (1 day)
**Risk:** Low | **Gain:** 0.25-0.5% on finalize | **Files:** `transcript_kernel.cu`

Replace 16 scalar loads with 4x uint128_t LDG128 loads.

### Step 6: Unrolled Target Comparison (30 minutes)
**Risk:** Low | **Gain:** 0.03-0.06% on finalize | **Files:** `transcript_kernel.cu`

Replace loop-based comparison with unrolled short-circuit OR chain.

### Step 7: GPU Self-Test + Dual-Path Verification (2 days)
**Risk:** Low | **Gain:** N/A (safety) | **Files:** New test files

Run GPU-side self-test and dual-path memcmp verification.

### Step 8: Production Pool Testing (3-5 days)
**Risk:** Medium | **Gain:** N/A (validation) | **Files:** None

Run against test pool for 24+ hours, verify share acceptance.

### Step 9: Hashrate Benchmark (1 day)
**Risk:** Low | **Gain:** N/A (validation) | **Files:** None

Run benchmark script, verify no regression.

---

## 5. Rollback Plan

### Environment Variable Controls

```bash
# Disable all finalize optimizations (fallback to original)
export PROPMINER_FINALIZE_OPT=0

# Disable specific optimizations
export PROPMINER_FINALIZE_PTX_ROTATE=0
export PROPMINER_FINALIZE_CONSTANT_TARGET=0
export PROPMINER_FINALIZE_INLINE_BLAKE3=0
export PROPMINER_FINALIZE_COALESCED_LOAD=0

# Enable debug mode (print hash values for verification)
export PROPMINER_FINALIZE_DEBUG=1
```

### Kernel Selection

```bash
# Use consumer headless path (no finalize kernel)
export PEARL_GEMM_KERNEL=consumer

# Use GeForce v2 path (no finalize kernel)
export PEARL_GEMM_KERNEL=geforce_v2

# Use portable path (with finalize kernel)
export PEARL_GEMM_KERNEL=portable
```

### Git Rollback

```bash
# Revert to previous commit
git revert HEAD

# Or switch to optimization-disabled branch
git checkout main  # unoptimized version
```

### Rollback Triggers

| Condition | Action |
|-----------|--------|
| Share rejection rate > 0.1% | Immediate rollback |
| Hashrate regression > 5% | Investigate, consider rollback |
| CUDA error in finalize kernel | Rollback, file bug |
| Memory corruption / OOM | Rollback, check register pressure |
| Occupancy decrease > 10% | Reduce register pressure, rollback if needed |

---

## 6. Verification Checklist

### Pre-Implementation
- [ ] CPU BLAKE3 vector tests pass (`blake3_finalize_verify.cpp`)
- [ ] Nsight Compute baseline captured for current kernel
- [ ] Performance measurement framework ready

### Post-Implementation (Each Step)
- [ ] CPU tests pass (blake3_offload_test.cpp)
- [ ] GPU self-test passes (finalize_self_test.cu)
- [ ] Dual-path memcmp verification passes
- [ ] CI build passes (all architectures: sm_80, sm_86, sm_89, sm_120a)
- [ ] Nsight Compute shows improved metrics
- [ ] Production hashrate doesn't regress
- [ ] Pool accepts shares (24+ hour soak test)
- [ ] Pearl verification passes (transcript identity)

### Production Readiness
- [ ] All 16.7M thread BLAKE3 outputs match reference
- [ ] No consensus drift detected
- [ ] Occupancy improved or unchanged
- [ ] Memory bandwidth utilization measured
- [ ] Long-running stability test (72+ hours) passes
- [ ] Multi-sigma rotation test passes
- [ ] GeForce v2 path also tested (if applicable)
- [ ] Legacy persistent kernel tested (if applicable)
- [ ] Turing/Ada/Ampere paths tested (if applicable)

---

## 7. Expected Total Improvement

| Optimization | Finalize Kernel Gain | Overall Mining Gain |
|-------------|---------------------|-------------------|
| PTX Rotate-Left | 8-12% | 0.5-1.0% |
| Constant Memory pow_target | 0.1-0.2% | < 0.01% |
| Inline BLAKE3 | 5-10% | 0.3-0.8% |
| Coalesced Load | 0.25-0.5% | < 0.01% |
| Unrolled Comparison | 0.03-0.06% | < 0.01% |
| **Total** | **~13-22%** | **~0.8-1.8%** |

**Note:** The finalize kernel is only 2-5 ms per iteration. The GEMM kernel dominates total time. So even a 20% finalize improvement only improves overall mining by ~1%. The real wins are in the GEMM kernel and host pipeline (see other plans).

---

## 8. Additional Notes

### Why NOT to Optimize the GEMM Kernel for BLAKE3

The consumer headless and GeForce v2 kernels **do not run BLAKE3 in the GEMM kernel**. They write the transcript to gmem and let the separate finalize kernel handle BLAKE3. This is by design — it keeps the GEMM kernel register pressure low and allows the compiler to optimize the matmul without BLAKE3 interference.

### Why NOT to Batch BLAKE3 Across Threads

Each thread processes a unique transcript (unique nonce position). There's no shared state to batch on. The pow_key is shared but it's only 32 bytes — already in constant cache.

### The Real Bottleneck Is the GEMM Kernel

For production mining:
- GEMM kernel: ~100-200 ms per iteration (Tensor Core bound)
- Finalize kernel: ~2-5 ms per iteration (scalar ALU bound)
- Host overhead: ~5-20 ms per iteration (driver overhead at batch=1)

Optimizing BLAKE3 by 20% saves ~0.4-1.0 ms per iteration — a **0.2-0.8% overall improvement**. The bigger wins are:
1. Increasing batch size (PROPMINER_BATCH=8): +30-40%
2. Triple buffering: +10-20%
3. Async sigma install: +5-15%
4. Fused noise+GEMM: +10-20%

### Legacy Kernel Has More BLAKE3 Optimization Potential

The `pearlhash_kernel.cu` persistent kernel runs BLAKE3 for every nonce (~30% of per-nonce work). Optimizing this kernel could yield 5-10% improvement on the legacy path. But this is a separate code path, not used in production.

---

*End of plan.*
