# Pearl Noisy GEMM and Noise-Hash Algorithm Research

**PropMiner Research · Topic 05**  
**Date:** July 2026  
**Scope:** End-to-end synthesis of Pearl's low-rank noise model, BLAKE3 keyed commitments, transcript XOR accumulation, tensor-hash Merkle trees, and the miner–verifier compute asymmetry. Grounded in PropMiner's `pearl-gemm`, `pearl-blake3`, and host share-builder paths.

---

## Executive Summary

Pearl proof-of-work is not "hash the matrix product." It is a **binding chain** of cryptographic commitments, deterministic noise expansion, a proof-canonical int8 GEMM with an execution transcript, and a keyed BLAKE3 jackpot check. On PropMiner's RTX 5090 production path (M=8192, N=262144, K=128, rank **R=128**), steady-state GPU time splits roughly as:

| Stage | Share of fused kernel / iteration work | Primary code |
|-------|----------------------------------------|--------------|
| **Noisy int8 GEMM (tensor cores)** | **~85–92%** | `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` |
| **Transcript XOR-reduce + rotate** | **~3–8%** (inside GEMM epilogue) | `third_party/pearl-gemm/csrc/gemm/pow_utils.hpp` |
| **BLAKE3 (jackpot + commitments)** | **~5–15%** cumulative | `third_party/pearl-gemm/csrc/blake3/`, `csrc/tensor_hash/` |

The **miner–verifier gap** is on the order of **10⁷×**: a miner evaluates the full M×N×K noisy GEMM grid (~6.5×10⁷ tile-MAC-equivalents of proof work per matmul), while a pool verifier recomputes **one** opened tile plus Merkle authentication paths. This asymmetry is intentional PoW economics, not an implementation bug.

---

## 1. Problem Statement

### 1.1 What Pearl must prove

Pearl V2 requires that a miner:

1. Committed to specific int8 matrices **A** (M×K) and **B** (stored as K×N row-major) before knowing the exact noise realization.
2. Applied **low-rank noise** derived from keyed BLAKE3 seeds chained through job key and matrix Merkle roots.
3. Executed a **tiled int8 GEMM** on noisy operands **A′ = A + E**, **B′ = B + F** with a **transcript** that captures partial accumulator state at K-boundaries.
4. Found a tile whose transcript hashes under the job key to a value below the difficulty target.

Skipping any step — multiplying zero matrices, reusing noise without commitment, or hashing only the final C matrix — must be detectable. The protocol achieves this by making the PoW depend on the **full execution trace**, not the output alone.

### 1.2 Why "noise hash" is a misnomer

Colloquially, operators speak of "noise hash" mining. Precisely, there are **three distinct hash roles**:

| Role | Input | Output | When it runs |
|------|-------|--------|--------------|
| **Matrix commitment** | 1024-byte leaf chunks of A/B | Merkle roots `hashA`, `hashB` | Once per σ (job rotation) |
| **Noise seed derivation** | `job_key ‖ hashB`, then `b_seed ‖ hashA` | `b_noise_seed`, `a_noise_seed` | Once per σ |
| **Jackpot / PoW** | 16×uint32 transcript | 256-bit hash vs target | Every CTA, every iteration |

The "noise" is **generated** via keyed BLAKE3 per 32-byte chunk index; the "hash" that wins blocks is the **transcript jackpot**, not the noise matrices themselves.

---

## 2. Rank-R Low-Rank Noise Model (R=128)

### 2.1 Mathematical structure

Pearl adds low-rank perturbations:

```
A′[m,k] = A[m,k] + Σ_r E_AL[m,r] · E_AR[k,r]   (dense × sparse structure)
B′[n,k] = B[n,k] + Σ_r E_BL[k,r] · E_BR[n,r]
```

With **R=128** on production RTX 5090 (`Rtx5090Profile`, `propminer_config.h`), each K-slab of width R participates in one rank chunk. At K=128, there is **exactly one** K/R step — one transcript snapshot per tile per iteration.

### 2.2 Noise matrix types

Documented in `third_party/pearl-gemm/csrc/gemm/noise_generation_kernel.h`:

| Tensor | Shape | Structure | Value range |
|--------|-------|-----------|---------------|
| **E_AL** | (M, R) | Dense int8 | Uniform in [-32, 32) via 2 indices per BLAKE3 digest byte |
| **E_BR** | (N, R) | Dense int8 | Same |
| **E_AR** | (K, R) | **Sparse** | One +1 and one −1 per row; rest zero |
| **E_BL** | (K, R) | **Sparse** | Same |

The sparse factors enforce structured perturbation along K while keeping generation GPU-efficient. Layout flexibility (R-major vs K-major) allows fused noising kernels to match downstream GEMM access patterns:

- **noisingA** wants `EAR_R_major`, `EBL_K_major`
- **noisingB** wants `EAR_K_major`, `EBL_R_major`

### 2.3 BLAKE3 chunk generation rule

Each 32-byte noise chunk uses:

```
raw_hash = BLAKE3(data, key=key_{A|B})
```

With `data` constructed as:

- Bytes 0–31 cleared initially
- `data[0] = r+1` for E_AL / E_BR chunks (global linear index r)
- `data[1] = r+1` for E_AR / E_BL chunks
- Bytes 32–63 = `seed_A` or `seed_B`
- Flags: `KEYED_HASH | CHUNK_START | CHUNK_END | ROOT`
- Counter: 0

Both noise tensors for side A derive from `seed_A`; both for side B from `seed_B`. The offset-by-one rule ensures row 0 of left and right dense factors differ.

PropMiner host reference: `src/host/pearl/share_builder.cpp` (`get_random_hash`, `generate_uniform_random_matrix`, `generate_permutation_matrix`).

### 2.4 Seed chain (consensus-critical)

From `share_builder.cpp` and `job_key.cpp`:

```
job_key     = BLAKE3(σ ‖ config_bytes)
b_noise_seed = BLAKE3(job_key ‖ hashB)
a_noise_seed = BLAKE3(b_noise_seed ‖ hashA)
```

Label constants `SEED_LABEL_A = "A_tensor"`, `SEED_LABEL_B = "B_tensor"` appear in share reconstruction for opened-leaf proofs. Any mismatch between miner and verifier seed derivation invalidates shares — tested explicitly in `research/01-git-repos-mining/repos/pearl/zk-pow/src/ffi/mine.rs`.

---

## 3. Matrix Commitments: Tensor Hash and Merkle Trees

### 3.1 Why commitments precede noise

Noise must be **unpredictable given the job** but **deterministic given commitments**. Committing to A and B before noise seed derivation prevents a miner from grinding noise independently of matrix content.

### 3.2 Leaf structure

- **Leaf size:** 1024 bytes of matrix data per Merkle leaf
- **Hash function:** Keyed BLAKE3 (`blake3::KEYED_HASH`)
- **Output:** 32-byte chaining values per leaf → binary Merkle tree → 32-byte root

PropMiner GPU path: `third_party/pearl-gemm/csrc/tensor_hash/tensor_hash.cu`

Pipeline stages:

1. **Leaf CV computation** — `MerkleTreeRootsKernelSM80` or `BSeedMerkleTreeRootsKernelSM80` for bseed-expanded B
2. **Multi-block merge** — `ComputeBlakeMTKernel` builds intermediate Merkle levels
3. **Root reduction** — `ReduceRootsKernel` when multiple MT blocks exist
4. **Commitment hash** — `commitment_hash_from_merkle_roots_kernel.hpp` binds roots to job key

Host orchestration: `tensor_hash_host_sm80.hpp`, invoked from `pearl_gemm_capi.cpp` during σ-install and share-hit A regen (`gpu_worker.cpp` `process_share_trigger`).

### 3.3 Resident B optimization

On production 5090, matrix **B** and its noise structures remain VRAM-resident across iterations (`SigmaContext::install`, `sigma_context.cpp`). Only **A** is regenerated per nonce (LCG / bseed path). This shifts commitment work to **once per σ**, not once per hash attempt — amortizing BLAKE3 Merkle cost across millions of nonces.

### 3.4 BLAKE3 time budget (5–15%)

BLAKE3 appears in three timing buckets:

| Bucket | When | Approx. share |
|--------|------|---------------|
| **σ-install** | B tensor_hash + root at job start | One-time; can dominate first 60–120 s at N=262144 |
| **Per-iteration jackpot** | Headless `compress_msg_block_u32` on 16-word transcript | ~2–5% of CTA time when fused |
| **Share hit** | A regen + leaf CV rehash + D2H | <<0.01% of wall time (rare) |

The **5–15%** figure applies to **in-kernel and per-CTA work** when including XOR-tree reduction plus jackpot compression. It is **not** 15% of datacenter revenue — share reconstruction is negligible at pool difficulty.

Device implementation: `third_party/pearl-gemm/csrc/blake3/blake3.cuh`, `blake3.cu`  
Rust Merkle utilities: `third_party/pearl-blake3/src/merkle_py.rs`

---

## 4. Transcript XOR Accumulation

### 4.1 Purpose

The transcript is a **512-bit (16×uint32) state** per proof-canonical tile thread-group that accumulates fingerprints of partial GEMM results along K. It prevents shortcuts that compute C without matching the reference tile execution order.

### 4.2 Algorithm (K/R steps)

From `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu` header commentary:

```
transcript[0..15] = 0
C_running = 0
For s in 0 .. K/R - 1:
    C_running += GEMM_s(A′[m, s*R:(s+1)*R], B′[n, s*R:(s+1)*R])
    hash_t = xor_reduction(per-thread fragment slots of C_running)
    slot = s mod 16
    transcript[slot] = rotl_xor<13>(transcript[slot], hash_t)
hash256 = BLAKE3.compress(transcript, key=pow_key)
if hash256 <= pow_target: signal hit
```

At **K=128, R=128**: one snapshot, **slot 0 only** active per iteration. At reference H100 shapes (K=4096, R=128): 32 snapshots cycling through all 16 slots twice.

### 4.3 XOR reduction mechanics

`pow_utils.hpp` implements:

- **`xor3_lop3`** — 3-input XOR via PTX `lop3.b32` LUT 0x96
- **`rotl_xor<13>`** — `shf.l.wrap.b32` rotate-left 13, then XOR with accumulator fragment
- **`xor_reduction`** — tree reduction over all uint32 fragment slots using compile-time layer sizing (`xor_tree_layer_sizes`)

The rotation constant **13** matches host reference (`share_builder.cpp`: `ROTATE_LEFT = 13`) and Pearl whitepaper §4.

### 4.4 Proof-canonical tile geometry

Fixed for consensus:

| Parameter | Value | Guard |
|-----------|-------|-------|
| bM | 128 | `#error` if changed |
| bN | 256 | `#error` if changed |
| bK / kBK | 128 | Single K-tile at K=128 |
| Threads | 256 (8 warps) | `kNumMmaThreads` |
| MMA atom | SM80 `m16n8k32` | Byte-identical partition_C vs H100 WGMMA |

File: `third_party/pearl-gemm/csrc/portable/transcript_canonical.cuh`

Changing tile dimensions breaks `partition_C` coordinate mapping — shares mined on 5090 would fail pool validation against H100 reference.

### 4.5 Legacy vs fused paths

| Path | File | Notes |
|------|------|-------|
| **Fused headless (production)** | `consumer/transcript_gemm_kernel.cu` | GEMM + transcript + BLAKE3 target in one kernel |
| **Portable finalize** | `portable/transcript_kernel.cu` | Split finalize for parity testing |
| **Legacy persistent** | `src/cuda/kernels/pearlhash_kernel.cu` | Miniature LCG→GEMM→XOR→BLAKE3; educational |

Headless launch: `launch_transcript_gemm_headless()` via `pearl_gemm_capi.cpp` — avoids writing transcript to global memory on the hot path.

---

## 5. Noising Pipeline (Pre-GEMM)

### 5.1 Kernel sequence per iteration

Typical `pearl_gemm_capi` batch (`gpu_worker.cpp` → `GemmRunner::iter_batch`):

1. **Noise generation** — `noise_generation.cu` / `noise_generation_kernel.h`
2. **Noising A** — `pearl_noisingA_kernel.h` instantiations (`instantiations/noisingA_R128_fp16_64x64_2stages.cu`)
3. **Noising B** — often skipped in pure-miner mode when B resident and noise pre-applied
4. **Transcript GEMM** — consumer or blackwell wrapper
5. **Host signal scan** — pinned `HostSignalHeader` array

At K=128 with resident B, noising + GEMM dominate; noise gen is **≪1%** when amortized because E_BR/EAR structures persist per σ.

### 5.2 int7 vs int8

Protocol matrices use int7 range (−64…+64) stored in int8 containers. Quantization kernels: `quantize_kernel.hpp`, `quantization_util.cuh`. Noise values use int8 uniform draw mapped to [−32, 32).

### 5.3 ApEA / BpEB fusion

Portable path fuses add+clamp into GEMM prologue (`portable_int8_gemm.cu`) to avoid materializing full int32 C in HBM — **~256 MiB saved** at large N. Production 5090 pure-miner mode never writes C (`gpu_worker.cpp`).

---

## 6. Miner vs Verifier: The 10⁷× Gap

### 6.1 Work asymmetry

| Actor | Work per candidate | Typical cost |
|-------|-------------------|--------------|
| **Miner (GPU)** | Full M×N×K noisy GEMM + 65,536 CTA transcripts | ~2.75×10¹⁴ int8 MACs per matmul |
| **Pool verifier (CPU/GPU)** | Recompute **one** tile + Merkle paths + seed check | ~4.2×10⁶ MACs for 128×256×128 tile |

**Ratio:** 2.75×10¹⁴ / 4.2×10⁶ ≈ **6.5×10⁷ ≈ 10⁷×**

This is the standard PoW pattern: expensive forward evaluation, cheap witness check. Pearl adds Merkle openings for A/B rows/columns and transcript slot proof, but verifier work remains **O(tile + log N)** vs miner **O(M×N×K)**.

### 6.2 What the verifier recomputes

From Pearl protocol docs and `share_builder.cpp`:

1. Parse share protobuf: opened row/col indices, transcript, claimed hash
2. Verify `hashA`, `hashB` Merkle paths against job commitments
3. Re-derive `a_noise_seed`, `b_noise_seed` from job key + roots
4. Regenerate noise for opened rows/cols only
5. Re-run single-tile GEMM + transcript for claimed slot
6. Confirm BLAKE3(transcript, key) ≤ target

Akoya reference: `research/01-git-repos-mining/repos/akoya-miner/src/Akoya.Crypto/AuditProofVerifier.cs`.

### 6.3 Implications for optimization

- **Do not optimize BLAKE3 at the expense of GEMM** — GEMM is 85–92% of iteration time.
- **Share-hit A regen is correctness, not throughput** — deferral hides ~0–2% wall time only.
- **10⁷× gap means pool validation latency is irrelevant** to miner competitiveness.

---

## 7. End-to-End Data Flow (PropMiner RTX 5090)

```
Pool gRPC job (σ, target, config_bytes)
    │
    ▼
job_key.cpp ──► BLAKE3(σ ‖ config) = job_key
    │
    ▼
sigma_context.cpp :: install()
    ├── tensor_hash(B) ──► hashB, leaf_cvs (resident)
    ├── noise seeds from job_key + hashB
    └── VRAM: B, E_BR, EARx, scales, roots (~225 MiB @ N=262144)
    │
    ▼
GpuWorker ping-pong loop
    ├── upload 8-byte seed (seed_copy_stream_)
    ├── iter_batch_graph_launch_ex()
    │     ├── [optional] noise_generation
    │     ├── noisingA (A slice from LCG/bseed)
    │     └── launch_transcript_gemm_headless (65,536 CTAs)
    │           ├── cp.async load A/B tiles
    │           ├── mma.sync m16n8k32 × 4 K-frags
    │           ├── xor_reduction + rotl_xor<13> → transcript
    │           └── BLAKE3 jackpot vs pow_target → HostSignalHeader
    └── scan headers → ShareBuilder on hit
          ├── regenerate A rows, Merkle paths
          └── gRPC submit
```

---

## 8. Timing Model: Where the 85–92% / 5–15% Comes From

### 8.1 Per-CTA cycle budget (headless fused kernel)

Estimated from Nsight Compute profiles on RunPod RTX 5090 (M=8192, N=262144, K=128):

| Phase | Cycles (relative) | % |
|-------|-------------------|---|
| cp.async + wait | 18–22% | Memory pipeline |
| ldmatrix + mma.sync | **55–65%** | Tensor pipes |
| XOR tree + rotl_xor | 5–8% | ALU (lop3/shf) |
| BLAKE3 compress (16 u32) | 3–6% | ALU + registers |
| Target compare + signal write | <1% | Branch (cold) |

### 8.2 σ-install BLAKE3 (amortized)

B-side `tensor_hash` at N=262144 processes ⌈N×K/1024⌉ = 32,768 leaves. One-time cost **~100–500 ms** GPU time — amortized over 10⁶+ subsequent nonces per σ.

### 8.3 ROCm split-kernel lesson

Akoya's ROCm port documented splitting PoW BLAKE3 out of GEMM when it was a **~32% serial tail**. PropMiner's headless fusion recovered this by parallelizing jackpot checks per CTA — the **5–15%** figure reflects successful fusion.

---

## 9. Correctness Traps and Validation

### 9.1 Transcript byte identity

Any of these breaks shares:

- Wrong `partition_C` layout (different MMA atom tiling)
- Wrong rotation constant (≠ 13)
- Wrong XOR tree ordering

Validation: `propminer --self-test`, `ref_pearl.cpp`, `portable/run-parity.sh`.

### 9.2 Noise / seed mismatches

- Using `seed_B` for E_AR when protocol expects `seed_A`
- Forgetting `r+1` offset in BLAKE3 data block
- R-major vs K-major pointer confusion in noising kernels

### 9.3 Periodic tile patterns

Only **proof-canonical** (row, col) tile positions undergo jackpot hash. Patterns encoded in `host_signal_header.hpp`.

---

## 10. Rank R=128 vs R=64

| R | K=128 snapshots | Production use |
|---|-----------------|----------------|
| 64 | 2 (slots 0,1) | Legacy / test |
| **128** | **1 (slot 0)** | **RTX 5090 production** |

R=128 simplifies transcript (single rotl_xor update) but increases noise buffer size (N×R×2 for E_BR fp16 paths ≈ 64 MiB @ N=262144).

---

## 11. Relationship to ZK Layer

Pearl full nodes compress proofs via `zk-pow/` (Plonky2). Mining pools today validate **plain shares**, not ZK certificates. The noise-hash pipeline described here is the **pool-facing hot path**.

Reference: `research/01-git-repos-mining/repos/pearl/zk-pow/`, `research/04-pearl-protocol/REPORT.md`.

---

## 12. Open Research Questions

1. **Fused noise generation + noisingA + GEMM** — estimated +1–3% if register pressure manageable.
2. **bseed A expansion** — `bseed_merkle_tree_roots_kernel_sm80.hpp` path vs LCG; parity required.
3. **MoE fork** — `ARC-miner/docs/MOE-PORT-PLAN.md` may change M/K/R.
4. **INT4 noise** — protocol forbids without hard fork.

---

## 13. Summary Table

| Concept | Value / location |
|---------|------------------|
| Rank R | **128** production |
| GEMM share of iteration | **~85–92%** |
| BLAKE3 + XOR epilogue | **~5–15%** |
| Transcript | 16×uint32, `rotl_xor<13>` |
| Miner / verifier work ratio | **~10⁷×** |
| Noise seeds | BLAKE3 chain via job_key + hashA/B |
| Matrix commitments | tensor_hash Merkle, 1024-byte leaves |
| Production kernel | `transcript_gemm_kernel.cu` (headless) |
| Proof tile | 128×256×128 int8→int32 |

---

## Appendix A: File Index

| Topic | Path |
|-------|------|
| Noise generation spec | `third_party/pearl-gemm/csrc/gemm/noise_generation_kernel.h` |
| Noise CUDA | `third_party/pearl-gemm/csrc/gemm/noise_generation.cu` |
| Noising A/B | `pearl_noisingA_kernel.h`, `pearl_noisingB_kernel.h` |
| POW utils (XOR) | `third_party/pearl-gemm/csrc/gemm/pow_utils.hpp` |
| Transcript portable | `third_party/pearl-gemm/csrc/portable/transcript_kernel.cu` |
| Tensor hash | `third_party/pearl-gemm/csrc/tensor_hash/tensor_hash.cu` |
| BLAKE3 device | `third_party/pearl-gemm/csrc/blake3/blake3.cuh` |
| Host share builder | `src/host/pearl/share_builder.cpp` |
| Legacy pearlhash | `src/cuda/kernels/pearlhash_kernel.cu` |
| Rust Merkle | `third_party/pearl-blake3/src/merkle_py.rs` |
| Protocol report | `research/04-pearl-protocol/REPORT.md` |
| GPU fundamentals | `research/06-gpu-mining-fundamentals/REPORT.md` |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **σ (sigma)** | Job header fragment binding miner work to blockchain state |
| **DAF** | Difficulty adjustment factor for normalized hashrate |
| **TMAD/s** | Trillion int8 multiply-adds per second |
| **Headless** | In-kernel PoW check without separate finalize pass |
| **CTA** | CUDA thread block; one output tile per CTA |
| **ApEA** | A plus noise A (noised left factor) |
| **BpEB** | B plus noise B |

---

*This report synthesizes PropMiner implementation details as of July 2026. Consensus rules are defined by Pearl Research Labs reference software and live pool validators; when in doubt, treat `ref_pearl.cpp` and pool-rejected shares as ground truth.*
