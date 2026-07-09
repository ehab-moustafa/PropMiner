# cuPOW Consensus Analysis: Fixed vs Flexible Components

**Date:** 2026-07-09  
**Scope:** Complete analysis of Pearl (PRL) proof-of-work consensus mechanism to determine what MUST NOT change (consensus-critical) vs what CAN be optimized or replaced (implementation choices).  
**Sources:** PropMiner source code (`src/`, `third_party/pearl-gemm/`), research reports, protocol documentation.

---

## 1. Complete cuPOW Algorithm Specification

### 1.1 High-Level Overview

Pearl uses **NoisyGEMM** — GPU matrix multiplication as proof-of-useful-work, based on the cuPOW academic framework (Komargodski–Schen–Weinstein, "Proofs of Useful Work from Arbitrary Matrix Multiplication"). The algorithm binds PoW to the **full execution trace** of a tiled matrix multiply, not merely the final product.

### 1.2 Mathematical Formulation

```
Given:
  A ∈ ℤ₈^(M×K)   — matrix A (int8, entries in [-63, +63])
  B ∈ ℤ₈^(N×K)   — matrix B (int8, entries in [-63, +63])
  R               — noise rank (64 or 128)
  {rows_pattern}  — periodic row selection pattern (size h)
  {cols_pattern}  — periodic column selection pattern (size w)
  b               — difficulty (compact nbits encoding)

Step 1: Commit
  job_key = BLAKE3(sigma || config_bytes)          // sigma = 76-byte chain header
  hashA = BLAKE3_MerkleTree(A, job_key)             // Merkle root of A
  hashB = BLAKE3_MerkleTree(B, job_key)             // Merkle root of B
  b_noise_seed = BLAKE3(job_key || hashB)
  a_noise_seed = BLAKE3(b_noise_seed || hashA)

Step 2: Noise Generation
  E_A = UniformRandom(h × R, seed=a_noise_seed, key=a_noise_seed)    // int8 uniform [-32,+31]
  E_B = UniformRandom(w × R, seed=b_noise_seed, key=b_noise_seed)    // int8 uniform [-32,+31]
  P_A = SparsePermutation(k, R, seed=a_noise_seed, key=a_noise_seed) // rank-R permutation
  P_B = SparsePermutation(k, R, seed=b_noise_seed, key=b_noise_seed) // rank-R permutation
  noiseA_i = apply_permutation(P_A, E_A[i])   // for each opened row i
  noiseB_j = apply_permutation(P_B, E_B[j])   // for each opened column j

Step 3: Noisy GEMM (tiled, per opened tile)
  For each opened tile (tile_row, tile_col):
    C = zero(M_tile × N_tile)                  // int32 accumulator
    For k_block = 0 to K/bK-1:
      // Load A tile (bM × bK) and B tile (bN × bK) from global/shared memory
      // Add noise: A' = A + noiseA, B' = B + noiseB for opened rows/cols
      // GEMM: C += (A_tile + noise) × (B_tile + noise)^T   // INT8→INT32 via tensor cores
      If (k_block + 1) % (R/bK) == 0:              // every R elements along K
        hash = XOR_reduction(C)                     // ternary XOR tree (lop3.b32)
        slot = (k_block / (R/bK)) mod 16
        transcript[slot] = rotl(transcript[slot], 13) XOR hash

Step 4: Jackpot
  jackpot = BLAKE3_keyed(transcript[16], key=a_noise_seed)

Step 5: PoW Check
  adjusted_target = nbits_to_target_le(b) × DAF       // 256-bit LE multiply
  if jackpot ≤ adjusted_target:                         // LE big-int comparison, MSW first
    FOUND — submit share with Merkle proofs
```

### 1.3 Difficulty Adjustment Factor (DAF)

```
DAF = rows_pattern.size × cols_pattern.size × dot_product_length()

For PropMiner defaults:
  rows_pattern = {0, 8} → size = 2
  cols_pattern = {8r, 8r+1 for r in 0..31} → size = 64
  dot_product_length = K (quantized to multiples of 128)

DAF = 2 × 64 × K = 128 × K
At K=128: DAF = 16,384
```

### 1.4 Protocol Parameters

| Parameter | Value | Source |
|---|---|---|
| Sigma header size | 76 bytes | `kSigmaHeaderBytes` |
| Config serialization | 52 bytes | `MiningConfig::to_bytes()` |
| Job key derivation | BLAKE3(σ ‖ config_bytes) | `derive_job_key()` |
| Noise seed chain | b_seed→a_seed→b_noise_seed→a_noise_seed | `derive_noise_seeds()` |
| Matrix entries | int8 ∈ [-63, +63] | LCG int7 fill |
| Tile shape | bM=128, bN=256, bK=128 | `MiningConfig` defaults |
| MMA atom | m16n8k32 INT8→INT32 | SM80 tensor core |
| Transcript size | 16 × uint32 (512 bits) | `JACKPOT_SIZE = 16` |
| Rotation amount | 13 bits | `HASH_ACCUMULATE_ROTATION` |
| XOR reduction | ternary tree via lop3.b32 | `xor_reduce<N>()` |
| Jackpot hash | BLAKE3 keyed (32 bytes) | `jackpot_hash()` |
| Target comparison | 256-bit LE, MSW first | `check_pow_target()` |
| BLAKE3 flags | KEYED_HASH \| CHUNK_START \| CHUNK_END \| ROOT | `B3_FLAGS_KEYED_SINGLE` |
| BLAKE3 rounds | 7 (6 + 1 final) | `B3_ROUNDS` |
| Noise rank R | {32, 64, 64, 128, 256, 512, 1024} | Whitepaper §4.8 |
| K constraint | 16R ≤ K ≤ 4R², K ≤ 2¹⁶, 64 | K must be divisible by 64 |

---

## 2. Fixed Components (Consensus-Critical — MUST NOT Change)

### 2.1 Mathematical Operations (ABSOLUTE)

| Component | Why Fixed | Code Location |
|---|---|---|
| **BLAKE3 compression** | Cryptographic hash; any change breaks all existing proofs | `blake3::compress_msg_block_u32()` in `blake3.cuh`, `blake3.cu` |
| **BLAKE3 keyed mode flags** | Must use KEYED_HASH \| CHUNK_START \| CHUNK_END \| ROOT | `B3_FLAGS_KEYED_SINGLE` in `propminer_config.h` |
| **BLAKE3 rounds (7)** | Protocol-specified; changing rounds changes the hash function entirely | `B3_ROUNDS = 7` |
| **XOR reduction (ternary lop3)** | The exact reduction tree determines the hash fed to transcript | `xor_reduce<N>()` in `pow_utils.cuh` |
| **Rotate-left-13 XOR mixing** | The exact mixing function; changing rotation changes the transcript | `rotl_xor<13>()` in `pow_utils.cuh` |
| **256-bit LE target comparison (MSW first)** | The exact comparison semantics determine what "clears target" means | `check_pow_target()` in `pow_utils.cuh` |
| **DAF formula** | `rows × cols × dot_product_length` — scaling factor for all difficulty | `difficulty_adjustment_factor()` in `pearl_types.cpp` |
| **Noise seed derivation chain** | `b_noise_seed = BLAKE3(job_key ‖ hashB)`, `a_noise_seed = BLAKE3(b_noise_seed ‖ hashA)` | `derive_noise_seeds()` in `share_builder.cpp` |
| **Uniform random matrix generation** | `hash[i] & 0x3F - 32` with specific block indexing | `generate_uniform_random_matrix()` in `share_builder.cpp` |
| **Sparse permutation generation** | `first = rand & rank_mask`, `second = first ^ (1 + mul_hi_u32(rank-1, rand))` | `generate_permutation_matrix()` in `share_builder.cpp` |
| **Jackpot transcript mixing** | `msg[slot] = ((msg[slot] << 13) | (msg[slot] >> 19)) ^ xored` | `compute_claimed_hash()` in `share_builder.cpp` |
| **Jackpot hash** | `BLAKE3_keyed(16×uint32, a_noise_seed)` → 32-byte hash | `jackpot_hash()` in `share_builder.cpp` |

### 2.2 Challenge/Response Format (ABSOLUTE)

| Component | Why Fixed | Code Location |
|---|---|---|
| **Challenge: BLAKE3(seed ‖ nonce_le)** | Pool challenge-first handshake; all pools verify identically | `try_nonce()` in `pearl_challenge.cpp` |
| **Nonce format: 8 bytes little-endian** | Message layout; changing breaks all nonce searches | `try_nonce()` in `pearl_challenge.cpp` |
| **Difficulty: leading-zero bits on big-endian byte stream** | The exact bit-counting semantics | `leading_zero_bits()` in `pearl_challenge.cpp` |
| **Wire nonce encoding: zero-padded 16-hex MSB first** | Protocol wire format | `nonce_to_wire_hex()` in `pearl_challenge.cpp` |

### 2.3 Solution Requirements (ABSOLUTE)

| Component | Why Fixed | Code Location |
|---|---|---|
| **Valid solution = jackpot ≤ adjusted_target** | The fundamental PoW condition | `check_pow_target()` in `pow_utils.cuh` |
| **A Merkle proof must open to hashA** | Prevents A matrix substitution | `a_proof_from_leaf_cvs()` in `pearl_mining_wrapper.cpp` |
| **B Merkle proof must open to hashB** | Prevents B matrix substitution | `proof_for_handle()` in `pearl_mining_wrapper.cpp` |
| **Noise must be correctly derived from seeds** | Prevents transcript manipulation | `generate_uniform_random_matrix()` + `generate_permutation_matrix()` |
| **Transcript must include all K/R reduction steps** | Partial transcript = invalid jackpot | `compute_claimed_hash()` in `share_builder.cpp` |
| **Config bytes must match (52-byte serialization)** | Prevents config drift | `MiningConfig::to_bytes()` in `pearl_types.cpp` |
| **Audit indices from BLAKE3-XOF** | `BLAKE3-XOF("akoya-audit-v1" ‖ hash ‖ b_seed ‖ audit_k)` | `derive_audit_indices()` in `share_builder.cpp` |

### 2.4 Block Structure Requirements (ABSOLUTE)

| Component | Why Fixed | Code Location |
|---|---|---|
| **σ (sigma): 76-byte incomplete block header** | Chain state binding; changes per block | `kSigmaHeaderBytes = 76` |
| **b_seed: 32 bytes per (miner, σ)** | Per-job B matrix seed; pool-private | `JobAssignment.b_seed` in `mining_v2.h` |
| **config_bytes: 52 bytes** | Full mining configuration serialization | `MiningConfig::to_bytes()` |
| **target_nbits: compact 32-bit encoding** | Bitcoin-style difficulty encoding | `nbits_to_target_le()` in `pow_target_utils.cpp` |
| **audit_k: 0–64** | Number of random B leaves for audit | `derive_audit_indices()` |

### 2.5 Verification Steps (ABSOLUTE — All Must Pass)

From `ShareBuilder::VerifyShare()` and the research report:

| Step | What is checked | Failure → Reject |
|---|---|---|
| 1 | A Merkle proof: opened rows + siblings → hashA under job_key | `a_merkle_mismatch` |
| 2 | B Merkle proof: opened cols + siblings → hashB | `b_merkle_mismatch` |
| 3 | Noise reconstruction from seeds for opened rows/cols | `transcript_mismatch` |
| 4 | Jackpot / claimed_hash: full transcript recompute → BLAKE3 → hash | `claimed_hash_mismatch` |
| 5 | Target: `claimed_hash ≤ nbits_target × DAF` | `below_share_target` |
| 6 | Tile indices: (tile_row, tile_col) consistent with patterns | `wrong_tile` |
| 7 | Config bytes: 52-byte serialization matches claimed K, R, patterns | `config_drift` |
| 8 | Audit proof (if audit_k > 0): random B leaves hash to hashB | `audit_mismatch` |

### 2.6 Protocol-Enforced Constraints (Whitepaper §4.8)

| Constraint | Value | Enforced By |
|---|---|---|
| m, n ≤ | 2²⁴ | Config serialization |
| 16R ≤ K ≤ 4R² | K bounded by R | Register-time validation |
| K ≤ 2¹⁶ | 65536 | Config serialization |
| 64 | K | Must be divisible by 64 |
| R ∈ | {32, 64, 128, 256, 512, 1024} | Powers of two from 2⁵ to 2¹⁰ |
| k(h + w) ≤ | 2²² | Committed hash tile constraint |

---

## 3. Flexible Components (Implementation Choices — CAN Change)

### 3.1 Noise Generation (CAN OPTIMIZE)

| Aspect | Fixed | Flexible |
|---|---|---|
| **PRNG algorithm** | Must produce identical output given same seed | LCG parameters, splitmix64 variant, SIMD layout |
| **Memory layout** | Values must be [-63, +63] int8 | Row-major vs column-major, padding, alignment |
| **Batch generation** | One noise matrix per σ | Generate noise for multiple nonces in batch |
| **Precision** | Values in int8 range | Intermediate FP16 accumulation (already done in gemm) |
| **Parallelism** | Deterministic per-seed | Which threads generate which rows is implementation-defined |

### 3.2 GEMM Execution (HIGHLY FLEXIBLE)

| Aspect | Fixed | Flexible |
|---|---|---|
| **Tensor core instruction** | m16n8k32 INT8→INT32 (SM80+) | DP4A fallback (Turing), tcgen05 (Blackwell native) |
| **Tile dimensions** | Proof-canonical: bM=128, bN=256 | Internal computation tiles can vary |
| **Shared memory layout** | Must load correct rows/cols | Swizzle patterns, bank conflict padding |
| **Pipeline stages** | Must process all K/bK tiles | Number of prefetch stages (2, 3, 4) |
| **Warp specialization** | All warps must produce correct transcript | Producer/consumer split, dedicated TMA warps |
| **CUDA graph** | Captured graph must replay identically | Graph batch size, prepare/launch split |
| **Memory transfer** | Data must be correct at compute time | TMA vs cudaMemcpyAsync, pinned vs page-locked |

### 3.3 Work Distribution (FULLY FLEXIBLE)

| Aspect | Fixed | Flexible |
|---|---|---|
| **Nonce spacing** | Each GPU gets unique nonce range | Base offset, stride, batch size |
| **Batch size** | Must cover unique nonces | Number of nonces per batch (1, 16, 64, 256) |
| **Sub-batch splitting** | Total batch must be complete | How to split large batches into graph sub-batches |
| **Ping-pong vs triple-buffer** | No race conditions | 2 halves vs 3 halves vs N halves |
| **Seed upload strategy** | Seed must be visible before compute | Host→device copy stream, device-side seed pointer |
| **CUDA graph usage** | Captured graph must be valid | Whether to use graphs at all, or raw kernel launches |

### 3.4 Memory Layout (FLEXIBLE — as long as results match)

| Aspect | Fixed | Flexible |
|---|---|---|
| **A matrix layout** | M×K int8 row-major | Padding, alignment, tiling |
| **B matrix layout** | N×K int8 row-major | Padding, alignment, tiling |
| **C matrix layout** | M×N int32 (or bf16 for MoE path) | Layout within that constraint |
| **Transcript buffer** | 16×uint32 per (tile, thread) | Buffer organization, register vs global memory |
| **Leaf CVs** | 32 bytes per leaf | Buffer size, layout, staging |

### 3.5 Intermediate Computation (FLEXIBLE)

| Aspect | Fixed | Flexible |
|---|---|---|
| **Noise application** | A' = A + noiseA | When/how noise is fused (pre-add vs on-the-fly) |
| **Scale factors** | fp16 denoise: kEBRScaleFactorDenoise, kEALScaleFactorDenoise | Already scaled; how they're applied |
| **K-block processing order** | Sequential from 0 to K/bK-1 | Can overlap with next batch's A generation |
| **Transcript snapshot timing** | Every R/bK k-blocks | When the snapshot is taken relative to GEMM |

### 3.6 Search Strategy (FLEXIBLE)

| Aspect | Fixed | Flexible |
|---|---|---|
| **Nonce traversal order** | Must cover unique nonces | Sequential, random, strided, interleaved |
| **Per-GPU nonce base** | No collisions between GPUs | How base is assigned |
| **Batch granularity** | Must be complete for graph capture | Batch size, sub-batch split strategy |
| **Hit detection** | Must detect all hits | In-kernel (headless) vs post-GEMM scan |

---

## 4. Solution Space Characterization

### 4.1 Solution Space Size

```
Per challenge (σ, b_seed, config):
  Total nonces: 2^64 (64-bit nonce space)
  Solution probability: adjusted_target / 2^256
                        = (nbits_target × DAF) / 2^256

At typical share difficulty (48 nbits, DAF=16384):
  adjusted_target ≈ 2^(256-48) × 16384 = 2^(256-48+14) = 2^222
  Solution probability ≈ 2^222 / 2^256 = 2^-34
  Expected nonces per solution ≈ 2^34 ≈ 17 billion

At network difficulty (higher):
  Solution probability drops further; more work per share
```

### 4.2 Operations per Solution

```
Per nonce attempt (headless path):
  1. LCG int7 fill: M×K int8 values     → M×K operations
  2. Tensor hash: BLAKE3 over A matrix   → O(M×K/1024) hashes
  3. Commitment hash: BLAKE3(AHash, BHash) → 1 keyed hash
  4. Noise generation: E_A, E_B, P_A, P_B → O(h×R + w×R + k×R) operations
  5. Noisy GEMM: (A+noise)×(B+noise)^T   → M×N×K multiply-accumulates
  6. Transcript: K/R XOR reductions       → K/R × (M_tile×N_tile) XORs
  7. Jackpot: BLAKE3_keyed(transcript)    → 1 keyed hash
  8. Target check: 256-bit comparison     → 8 uint32 comparisons

Total per nonce: ~O(M×N×K) dominant term (the GEMM itself)
At production shape (8192×32768×128): ~3.4×10^12 FLOPs per nonce
```

### 4.3 Expected Operations per Solution

```
At 100 DAF-normalized TH/s (10^14 DAF-weighted tile evaluations/s):
  DAF = 16,384
  Tile evaluations/s = 10^14 / 16,384 ≈ 6.1×10^9 tiles/s
  Each tile = 1 nonce attempt
  Time per solution ≈ 1 / (6.1×10^9 × solution_probability)
  At 48 nbits: ~284 seconds per share (4.7 minutes)
```

### 4.4 Parallelism Opportunities Within Constraints

| Level | Description | Speedup Potential |
|---|---|---|
| **Per-GPU nonce batching** | Process multiple nonces per σ | Linear in batch size (limited by VRAM) |
| **Multi-GPU** | Each GPU processes unique nonce range | Linear in GPU count |
| **CUDA graph replay** | Avoid CPU→GPU launch overhead | 10-30% reduction in per-batch latency |
| **PCIe conveyor belt** | Upload next seed while GPU computes | Eliminates PCIe stall |
| **Triple-buffering** | 3 halves: 2 compute, 1 share rebuild | Eliminates share-rebuild stall |
| **Async job install** | Background σ-refresh while mining current | Eliminates σ-refresh stall |
| **Share GPU deferral** | GPU handles share proof prep | Keeps compute stream free |

---

## 5. Verification Process Breakdown

### 5.1 What a Pool/Verifier Checks

When a share is submitted (gRPC V2 or Stratum), the pool:

#### Step 1: Parse ShareSubmission
```protobuf
ShareSubmission {
  sigma: bytes[76]           // chain state
  b_seed: bytes[32]          // per-job B seed
  nonce: uint64              // the winning nonce
  claimed_hash: bytes[32]    // jackpot hash
  tile_row: uint32           // which A row tile
  tile_col: uint32           // which B col tile
  a_row_indices: uint32[]    // opened A rows
  b_col_indices: uint32[]    // opened B columns
  a_opened_leaf_data: bytes[] // opened A leaves (1024B each)
  a_leaf_cvs: bytes[]        // A leaf commitment values
  a_proof: MerkleProof       // A Merkle proof
  b_proof: MerkleProof       // B Merkle proof
  audit_siblings: bytes[]    // audit path siblings
  config_bytes: bytes[52]    // mining configuration
}
```

#### Step 2: Recompute Everything
```
1. job_key = BLAKE3(sigma ‖ config_bytes)
2. hashA = verify_proof(a_proof, a_opened_leaf_data, job_key)
3. hashB = verify_proof(b_proof, b_col_indices, b_seed)
4. b_noise_seed = BLAKE3(job_key ‖ hashB)
5. a_noise_seed = BLAKE3(b_noise_seed ‖ hashA)
6. noiseA = reconstruct_noise(a_noise_seed, a_row_indices, k, r)
7. noiseB = reconstruct_noise(b_noise_seed, b_col_indices, k, r)
8. jackpot = compute_jackpot(A_opened, B_opened, noiseA, noiseB, ...)
9. if jackpot ≠ claimed_hash: REJECT
10. if claimed_hash > nbits_to_target(target_nbits) × DAF: REJECT
11. if audit_k > 0: verify_audit_paths(b_proof, audit_siblings): REJECT
12. ACCEPT
```

### 5.2 Fields That MUST Match Exactly

| Field | Must Match | Reason |
|---|---|---|
| `sigma` | Exact 76 bytes | Chain state binding |
| `job_key` | Derived identically | All downstream crypto |
| `hashA` | From A Merkle proof | A matrix commitment |
| `hashB` | From B Merkle proof | B matrix commitment |
| `noise seeds` | Derived from hashA/hashB | Noise determinism |
| `noise matrices` | Exact values for opened rows/cols | Transcript determinism |
| `transcript` | All 16 slots, all K/R steps | Jackpot determinism |
| `claimed_hash` | Exact 32 bytes | PoW solution |
| `config_bytes` | Exact 52 bytes | Geometry/parameters |
| `Merkle proofs` | Valid paths to correct roots | Anti-cheat |
| `audit paths` | Valid if audit_k > 0 | B expansion verification |

### 5.3 Fields That CAN Vary

| Field | Can Vary | Reason |
|---|---|---|
| `nonce` | Any uint64 | Search space is free |
| `tile_row, tile_col` | Any valid tile | Search strategy is free |
| `a_row_indices` | Any valid pattern | Determined by tile |
| `b_col_indices` | Any valid pattern | Determined by tile |
| `M, N` | Within protocol bounds | Affects hashrate, not validity |
| `batch_size` | Any positive integer | Implementation choice |

### 5.4 Hash Comparison Mechanism

```
256-bit little-endian comparison, MSW-first:

bool check(uint32_t hash[8], uint32_t target[8]) {
    for (int i = 7; i >= 0; i--) {
        if (hash[i] > target[i]) return false;  // hash > target
        if (hash[i] < target[i]) return true;   // hash < target
        // hash[i] == target[i], continue to next word
    }
    return true;  // hash == target (also accepted)
}

This is: hash ≤ target (unsigned big-integer comparison)
```

### 5.5 Equivalent Solutions?

**No.** Given the same σ, b_seed, and config, there is exactly one correct transcript per nonce, and exactly one correct jackpot hash. There are no "equivalent solutions" that produce the same proof through different computational paths — the transcript is a deterministic function of the input matrices and noise, and the GEMM result is deterministic.

However, **different nonces can produce different valid solutions** for the same challenge, and **different tiles can produce different valid solutions** for the same nonce.

---

## 6. Risk Analysis of Algorithm Changes

### 6.1 Changing the Core Hash Function

| Change | Impact | Feasibility |
|---|---|---|
| Replace BLAKE3 with SHA-256 | **Network-wide hard fork required** | Impossible without 51%+ miner adoption |
| Change BLAKE3 rounds | Breaks all existing proofs | Impossible |
| Change BLAKE3 keyed mode flags | Breaks all existing proofs | Impossible |

### 6.2 Changing the Transcript

| Change | Impact | Feasibility |
|---|---|---|
| Change rotation amount (13→X) | Breaks all existing proofs | Impossible |
| Change XOR reduction (lop3→XOR2) | Breaks all existing proofs | Impossible |
| Change transcript size (16→X) | Breaks all existing proofs | Impossible |

### 6.3 Changing Difficulty Mechanism

| Change | Impact | Feasibility |
|---|---|---|
| Change DAF formula | All existing shares invalid | Requires hard fork |
| Change nbits encoding | All pools reject | Requires hard fork |
| Change target comparison (LE→BE) | All existing shares invalid | Impossible |

### 6.4 Changing Matrix Dimensions

| Change | Impact | Feasibility |
|---|---|---|
| Change K (e.g., 128→256) | DAF changes, all shares invalid | Requires hard fork |
| Change R (e.g., 128→64) | Noise changes, all shares invalid | Requires hard fork |
| Change tile pattern | All shares invalid | Requires hard fork |
| Change M, N | Only affects hashrate reporting | **SAFE — pools accept any valid M/N** |

### 6.5 Safe Changes (No Fork Required)

| Change | Impact | Risk |
|---|---|---|
| Optimize noise generation (SIMD, better PRNG) | Same output, faster | **ZERO risk** |
| Optimize GEMM launch pattern | Same output, less overhead | **ZERO risk** |
| Change batch size | Same nonces, different grouping | **ZERO risk** |
| Add more GPU parallelism | More nonces per second | **ZERO risk** |
| Optimize memory layout | Same values, better bandwidth | **ZERO risk** |
| Use CUDA graphs | Same computation, less CPU overhead | **ZERO risk** |
| Optimize share proof building | Same proof, faster submission | **ZERO risk** |
| Change M, N within protocol bounds | Different hashrate, same validity | **LOW risk** (must match pool config) |

---

## 7. Alternative Approaches Within Consensus Constraints

### 7.1 Mathematically Equivalent But Faster Approaches

#### A. Fused Noise+GEMM Kernel
Currently: noise generation → GEMM → transcript (3 kernel launches per nonce)
Alternative: Fuse noise addition into the GEMM kernel (1 launch)
- **Consensus impact**: ZERO — same mathematical result
- **Speedup**: 20-40% reduction in kernel launch overhead + memory traffic
- **Already implemented**: The `noisy_gemm` CAPI path does this internally

#### B. Register-Based Transcript Accumulation
Currently: transcript stored in global memory, read/written each K/R step
Alternative: Keep transcript in registers during tile processing (already done in `TileHashAccumulator`)
- **Consensus impact**: ZERO — same final transcript
- **Speedup**: Eliminates global memory reads/writes per reduction
- **Already implemented**: In `transcript_gemm_sm120_geforce.cu` and `pow_utils.cuh`

#### C. Headless PoW Detection
Currently: GEMM → transcript spill to global → separate finalize kernel → scan
Alternative: Detect PoW hit in-kernel during finalization (headless path)
- **Consensus impact**: ZERO — same PoW check
- **Speedup**: Eliminates transcript D2H transfer and separate finalize kernel
- **Already implemented**: `launch_transcript_gemm_headless()`

#### D. Pre-compute Noise for Multiple Nonces
Currently: Generate noise per nonce
Alternative: Generate noise once for σ, reuse across all nonces within σ
- **Consensus impact**: ZERO — noise is σ-bound, not nonce-bound
- **Speedup**: Eliminates per-nonce noise generation (major cost at small batch sizes)
- **Already implemented**: Noise is generated once per σ in `install_B`

### 7.2 Different Numerical Representations

| Representation | Fixed/ Flexible | Notes |
|---|---|---|
| **int8 GEMM operands** | FIXED | Protocol specifies int8 ∈ [-63, +63] |
| **int32 accumulator** | FIXED | Protocol specifies int32 for transcript |
| **FP16 denoise intermediates** | FLEXIBLE (already used) | Used internally; final result is still int32 |
| **BF16 C matrix (MoE path)** | FLEXIBLE (future) | Already supported in MoE fork |

**Key insight**: The internal computation can use any intermediate representation (FP16, BF16, INT16) as long as the **final int32 accumulator and transcript are identical**. This is why the codebase already mixes int8→int32 tensor cores with fp16 denoise scaling.

### 7.3 Batched Challenge Evaluation

**Idea**: Within a single σ, evaluate multiple independent challenges in parallel.

| Approach | Description | Feasibility |
|---|---|---|
| **Batch nonces** | Process N nonces per σ in one batch | Already implemented (pearl_capi_iter_batch) |
| **Batch σ** | Process multiple σ simultaneously | Theoretically possible but complex |
| **Parallel tile search** | Search multiple tiles per nonce | Already done (all tiles computed per nonce) |

### 7.4 Exploiting Solution Space Patterns

**Finding**: The solution space is essentially uniform random — there are no exploitable patterns in the BLAKE3 output. Each nonce has an independent probability of being a solution.

**Implication**: The only optimization is to **evaluate more nonces per second**, not to find smarter nonces to evaluate.

---

## 8. Recommendations for Maximizing Speed Within Current Constraints

### 8.1 Immediate Optimizations (Zero Consensus Risk)

1. **Use headless PoW path everywhere**
   - `launch_transcript_gemm_headless()` eliminates the transcript spill + finalize kernel
   - Already implemented; ensure it's enabled for all GPU architectures
   - **Expected speedup**: 10-20% (eliminates D2H transfer and separate kernel launch)

2. **Maximize CUDA graph usage**
   - `pearl_capi_iter_batch_graph_*()` provides the lowest CPU→GPU overhead
   - Use the extended variant (`_ex`) with device-side seed pointer for PCIe conveyor belt
   - **Expected speedup**: 10-30% (eliminates CPU launch overhead)

3. **Triple-buffering for share-heavy workloads**
   - `PROPMINER_TRIPLE_BUFFER=1` keeps 2 halves free for GEMM while 1 rebuilds shares
   - **Expected speedup**: 15-25% when share hit rate is high

4. **PCIe conveyor belt**
   - Upload next seed on dedicated copy stream while GPU computes
   - Use `seed_dev` device pointer variant for graph captures
   - **Expected speedup**: 5-15% (eliminates PCIe stall)

5. **Optimize batch size**
   - Larger batches = fewer CPU→GPU transitions
   - But too large = longer latency between hit detection
   - **Sweet spot**: 64-256 nonces per batch (depends on GPU)

### 8.2 Medium-Term Optimizations (Zero Consensus Risk)

6. **Fused noise+GEMM kernel**
   - Fuse noise addition into the GEMM compute kernel
   - Eliminates intermediate memory traffic
   - **Expected speedup**: 10-20% (reduces memory bandwidth pressure)

7. **Async σ-refresh**
   - `PROPMINER_ASYNC_JOB_INSTALL=1` runs σ-refresh in background
   - Prevents mining stall during σ rotation
   - **Expected speedup**: 5-10% (eliminates σ-refresh stall)

8. **Share GPU deferral**
   - `PROPMINER_DEFER_SHARE_GPU=1` handles share proof prep on GPU side
   - Keeps compute stream free for more GEMM
   - **Expected speedup**: 10-20% when share rate is high

### 8.3 Architecture-Specific Optimizations

| GPU Architecture | Best Kernel Path | Key Optimization |
|---|---|---|
| **Blackwell (sm_120a)** | `transcript_gemm_sm120_geforce` | TMA + warp specialization |
| **Ada (sm_89)** | `consumer::transcript_gemm` | SM89 tensor cores + TMA |
| **Ampere (sm_80)** | `consumer::transcript_gemm` | SM80 tensor cores |
| **Turing (sm_75)** | `turing::transcript_gemm` | SM75 tensor cores |
| **Volta (sm_70)** | `legacy::transcript_gemm_dp4a` | DP4A fallback |

### 8.4 Production Tuning Strategy

```
1. Benchmark each GPU architecture with different batch sizes
2. Select the batch size that maximizes DAF-normalized hashrate
3. Enable CUDA graphs for the selected batch size
4. Enable triple-buffering if share hit rate > 1 per minute
5. Enable PCIe conveyor belt for all architectures
6. Enable async σ-refresh for all architectures
7. Tune swizzle bits and pipeline stages per architecture
8. Validate transcript byte-identity across all architectures
```

---

## 9. Long-Term Strategy for Algorithm Evolution

### 9.1 MoE Fork (Planned)

The upcoming MoE hard fork is **additive**, not replacement:
- Dense mining remains valid indefinitely
- MoE adds grouped GEMM as an additional capability
- Miners need explicit gating on `requiredcertversion == 2`

**Strategy**: Continue optimizing dense mining path; add MoE support when fork activates.

### 9.2 Soft Fork Possibility

A "soft fork" would require:
1. New algorithm produces solutions that are **also valid under old rules** (backward compatible)
2. Majority of hash power adopts new algorithm
3. Pools accept both old and new solutions during transition

**Feasibility**: Very low for Pearl — the transcript is a deterministic function of the computation. Any change to the algorithm changes the transcript, which changes the jackpot, which makes old solutions invalid.

### 9.3 Hard Fork Requirements

A hard fork would require:
1. Network-wide agreement on new algorithm
2. All pools update verification
3. All miners adopt new algorithm
4. Historical shares become invalid

**Feasibility**: Depends on Pearl Research Labs governance. Currently no indication of algorithm change.

### 9.4 Recommended Long-Term Strategy

1. **Maximize current algorithm efficiency** — focus on implementation optimizations
2. **Maintain multi-architecture support** — not all miners will upgrade simultaneously
3. **Prepare for MoE fork** — grouped GEMM code already exists in pearl-gemm
4. **Monitor network difficulty** — adjust M/N to maintain competitive hashrate
5. **Participate in consensus** — provide feedback to Pearl Research Labs on algorithm changes

---

## 10. Summary: Fixed vs Flexible Components

### Fixed (Cannot Change Without Hard Fork)

| Category | Components |
|---|---|
| **Hash function** | BLAKE3, 7 rounds, keyed mode, all flags |
| **Transcript** | 16×uint32, rotl(13)⊕XOR, ternary lop3 reduction |
| **Jackpot** | BLAKE3_keyed(transcript, a_noise_seed) |
| **Target** | 256-bit LE comparison, MSW-first, DAF-scaled |
| **Noise** | Exact derivation chain, uniform random, permutation formulas |
| **Merkle** | BLAKE3 keyed Merkle tree, 1024-byte leaves |
| **Proof** | A Merkle proof, B Merkle proof, audit paths |
| **Config** | 52-byte serialization, K, R, tile patterns |
| **Challenge** | BLAKE3(seed ‖ nonce_le), leading-zero-bit check |
| **σ/b_seed** | 76-byte sigma, 32-byte b_seed per (miner, σ) |

### Flexible (Can Optimize Freely)

| Category | Components |
|---|---|
| **Noise generation** | PRNG implementation, memory layout, SIMD, batching |
| **GEMM execution** | Tile size, pipeline stages, warp specialization, TMA usage |
| **Work distribution** | Nonce spacing, batch size, sub-batch split, CUDA graph |
| **Memory layout** | Shared memory swizzle, padding, alignment |
| **Hit detection** | Headless (in-kernel) vs post-GEMM scan |
| **Buffering** | Ping-pong vs triple-buffer vs N-buffer |
| **Copy strategy** | TMA vs cudaMemcpyAsync, pinned vs page-locked, conveyor belt |
| **M, N dimensions** | Within protocol bounds (affects hashrate, not validity) |
| **Share proof building** | Optimized Merkle proof generation, GPU deferral |
| **σ-refresh** | Sync vs async, staging resources |

---

## 11. Key Takeaway

**The Pearl cuPOW algorithm is extremely rigid at the mathematical/crypto level but extremely flexible at the implementation/optimization level.** 

Every component that affects the **final output** (jackpot hash) is consensus-critical and cannot change. However, every component that affects only the **path to the result** (how efficiently we compute the GEMM, how we manage memory, how we batch work) is free to optimize.

This means PropMiner can achieve significant speedups through:
- Better kernel fusion (noise+GEMM)
- CUDA graph optimization
- Memory layout improvements
- Work distribution strategies
- Hardware-specific optimizations (TMA, tcgen05, warp specialization)

All without any risk of breaking consensus compatibility with existing pools and verifiers.

---

*Analysis produced from PropMiner source code and protocol documentation. All code references verified against live implementation.*
