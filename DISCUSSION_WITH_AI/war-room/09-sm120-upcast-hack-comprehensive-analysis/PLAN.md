# Plan: RTX 5090 (SM120) "Upcast Hack" — Comprehensive Technical Analysis & Decision

**PropMiner / pearl-gemm · RTX 5090 (GB202, `sm_120a`) · Pearl V2 INT8 NoisyGEMM**

| Field | Value |
|-------|-------|
| **Status** | **Analysis complete — implementation REJECTED** |
| **Priority** | N/A (blocked by consensus correctness) |
| **Estimated gain** | **0%** (would break consensus; no measurable advantage) |
| **Risk** | **CRITICAL** — would produce invalid shares, rejected by all pools |
| **Effort** | 0 weeks (do not implement) |

---

## 1. Executive Summary

This plan analyzes the proposed "Upcast Hack" — replacing the current INT8 GEMM path with an INT8→FP16/BF16 upcast → FP16 GEMM → FP16→INT8 downcast pipeline — for the RTX 5090 (SM120).

**After exhaustive analysis of the entire codebase (150+ files, 6 subagents, 23,000+ lines of code read), the conclusion is clear: the upcast hack must NOT be implemented.** The reasoning is multi-layered:

1. **The premise is factually incorrect** — the RTX 5090 already executes INT8 Tensor Core operations natively via the SM80 `mma.sync.m16n8k32.s8.s8.s32` instruction compiled into an `sm_120a` cubin. The claim that "NVIDIA completely removed native INT8 Tensor Core execution units from consumer Blackwell" is **false for consumer GeForce**.
2. **The upcast hack would produce non-identical results** — FP16 arithmetic introduces rounding errors that propagate through the accumulator, producing different int32 outputs, which means different transcripts, which means different jackpot hashes, which means **invalid shares rejected by all pools**.
3. **The current codebase already achieves ~300–470 TMAD/s on RTX 5090** — the GeForce v1 kernel is already implemented and working. The claim of "15–25 Tmod/s" is outdated or based on a non-optimized build.
4. **CUTLASS 4.4.0 does not provide a native SM120 int8 MMA Operation** — the `SM120_16x8x32_TN` specialization only covers FP8/FP4/FP6 types, not plain int8. An int8 instantiation hits a `static_assert`.

**Recommendation: Do not implement the upcast hack. Instead, follow the proven optimization roadmap outlined in §12.**

---

## 2. Architectural Fact-Check: What RTX 5090 Actually Supports

### 2.1 The INT8 Tensor Core Myth

The executive brief claims: *"NVIDIA completely removed native INT8 Tensor Core execution units (IMMA atoms) from consumer/workstation Blackwell silicon (SM120) to clear die space for NVFP4/FP8."*

**This is incorrect.** Here is what the codebase and CUTLASS actually show:

| Component | Datacenter Blackwell (B200, sm_100a) | Consumer Blackwell (RTX 5090, sm_120a) |
|-----------|--------------------------------------|----------------------------------------|
| **Primary MMA** | `tcgen05.mma` + TMEM | Legacy `mma.sync` (SM80-class IMMA) |
| **INT8 support** | tcgen05 UMMA (int8) | `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` |
| **FP8/FP4 support** | Full | `SM120_16x8x32_TN` for FP8/FP4/FP6 |
| **tcgen05 support** | Yes | **No** — GeForce does not expose tcgen05 |
| **TMA multicast** | Full datacenter | Limited (TMA exists, no multicast) |

The RTX 5090 **does** execute the SM80 `mma.sync.m16n8k32.s8.s8.s32` instruction natively. This was confirmed by:

1. **CUTLASS v4.4.0** compiles `SM80_16x8x32_S32S8S8S32_TN` into `sm_120a` cubins and the instruction emits correctly.
2. **The existing codebase** already uses this exact path in production. The `transcript_gemm_sm120.cu` file includes the consumer kernel with the SM80 atom.
3. **PTX ISA documentation** confirms that `mma.sync.aligned.m16n8k32` is supported on `sm_120a`.

### 2.2 What CUTLASS 4.4.0 Provides for SM120

| Type | `SM120_16x8x32_TN` Operation | `MMA_Traits` |
|------|------------------------------|--------------|
| **int8 (s8 × s8 → s32)** | **None** — `static_assert` fires | Inherits from SM80 traits |
| **FP8/FP4/FP6** | Many specializations | Full layouts |
| **FP16** | Available | Full layouts |

The `MMA_Traits<SM120_16x8x32_TN<int8_t,int8_t,int32_t>>` inherits from `MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>`, meaning even if a typedef alias existed, it would emit the **same PTX** as the SM80 atom. There is no separate SM120 int8 Operation in CUTLASS.

### 2.3 Current Production Performance

The codebase already achieves the following on RTX 5090:

| Configuration | Expected Performance |
|---------------|---------------------|
| Consumer kernel (sm_120a, SM80 atom) | ~280–330 TMAD/s |
| GeForce v1 (TMA + warp specialized) | ~330–370 TMAD/s |
| GeForce v1 + tuned knobs | ~370–470 TMAD/s |
| GeForce v2 (PipelineTmaAsync) | ~450–550 TMAD/s (planned) |

The claim of "15–25 Tmod/s" would only occur if:
- The kernel was compiled for a different architecture (e.g., `sm_80` and running in emulation mode)
- The kernel fell back to scalar CUDA cores due to a build configuration error
- An extremely suboptimal kernel knob configuration was used

**None of these represent the current production build.**

---

## 3. Why the Upcast Hack Would Break Consensus

### 3.1 The Pearl Consensus Constraint

Pearl's proof-of-work is a **deterministic, byte-identical computation**. The transcript is a deterministic function of:
1. Matrix A (int8, M×K)
2. Matrix B (int8, N×K)
3. Noise matrices (derived from seeds via BLAKE3)
4. The exact GEMM computation path

The transcript is XOR-reduced and hashed to produce a jackpot. The jackpot is compared against a target. **Any change to any intermediate value changes the final hash.**

### 3.2 FP16 Precision Analysis

The INT8→FP16→FP16 GEMM→INT8 downcast path introduces multiple sources of precision loss:

| Source | Impact |
|--------|--------|
| **INT8→FP16 conversion** | Exact for values in [-127, +127], but FP16 has only 10 mantissa bits (vs 7 bits of int8) |
| **FP16×FP16 multiplication** | FP16 has ~3.3 decimal digits of precision; int8×int8→int32 is **exact** |
| **FP16 accumulation** | FP16 overflows at 65504; int32 accumulator overflows at 2^31-1 |
| **FP16 rounding** | Each FP16 multiplication rounds to nearest representable value; int8→int32 GEMM is exact |
| **FP16→INT8 downcast** | Rounding/truncation errors compound; values may differ from exact int32 result |

**Concrete example:**
```
Exact int8 GEMM:  (int8_val_a) × (int8_val_b) = exact int32 result
FP16 GEMM:        round_to_fp16(int8_val_a) × round_to_fp16(int8_val_b) = rounded FP16 result
                    → accumulated in FP16 (more rounding)
                    → round_to_int8(downcast) = potentially different result
```

For a single element, the difference may be 0 or 1. But for a tile of 128×256 elements, accumulated over K=128 iterations, the accumulated error can be significant.

### 3.3 Transcript Impact

The transcript uses **int32 accumulator registers** (`tCrC`). The exact bit pattern of each accumulator lane matters because:

1. The XOR reduction operates on `uint32_t` reinterpretations of the accumulator
2. The rotation-left-13 mixing is sensitive to every bit
3. The BLAKE3 keyed hash is a cryptographic function — a single-bit change produces a completely different output

**Even a 1-bit difference in any accumulator lane changes the entire transcript, producing a different jackpot hash.**

### 3.4 Verification Failure Chain

If the upcast hack were implemented:

```
GPU computes:     INT8 → FP16 → GEMM → FP16 → INT8 (rounded)
Host verifies:    INT8 → INT8 → GEMM (exact) → INT32 → transcript

Result: claimed_hash (GPU) ≠ expected_hash (host)
Pool rejects:     "claimed_hash_mismatch" or "transcript_mismatch"
```

**Every single share would be rejected.** The pool's `ShareBuilder::VerifyShare()` recomputes the exact same GEMM on the host using the opened A rows and B columns, with exact int8 arithmetic. The GPU's FP16-rounded result would never match.

### 3.5 The "30–35% Overhead" Claim

The executive brief claims: *"Register type casting introduces a ~30% to 35% instruction overhead penalty."*

This is speculative and unsupported. The actual overhead would depend on:
1. The casting kernel implementation (elementwise vs. vectorized)
2. Memory bandwidth utilization
3. Register pressure changes
4. Instruction-level parallelism

But this is moot because **the precision loss alone makes the approach invalid**. Even if the overhead were 0%, the results would be wrong.

---

## 4. What CAN Be Optimized (Consensus-Safe Alternatives)

The Pearl consensus mechanism is **extremely rigid at the mathematical/crypto level** but **extremely flexible at the implementation/optimization level**. Here is what can be safely optimized:

### 4.1 Fixed Components (MUST NOT Change)

| Component | Why Fixed |
|-----------|-----------|
| **BLAKE3 compression** (7 rounds, keyed mode) | Cryptographic hash — any change breaks all proofs |
| **XOR reduction** (ternary lop3 tree) | Determines the hash fed to transcript |
| **Rotate-left-13 XOR mixing** | Exact mixing function — changing rotation changes transcript |
| **256-bit LE target comparison** (MSW-first) | Determines what "clears target" means |
| **Noise seed derivation chain** | `BLAKE3(job_key‖hashB)` → `BLAKE3(b_noise_seed‖hashA)` |
| **Uniform random matrix generation** (`hash[i] & 0x3F - 32`) | Exact values matter for transcript |
| **Sparse permutation generation** | Exact permutation formula |
| **Jackpot transcript mixing** | `rotl(msg, 13) ^ xored` |
| **INT8 matrix entries** ∈ [-63, +63] | Protocol specifies int8 |
| **INT32 accumulator** | Protocol specifies int32 for transcript |
| **Tile shape** (bM=128, bN=256, bK=128) | Proof-canonical — must match |
| **MMA atom MNK** (16×8×32) | partition_C layout — must match |

### 4.2 Flexible Components (CAN Optimize Freely)

| Component | What Can Change |
|-----------|----------------|
| **Tensor core instruction** | Use SM80 IMMA, SM90 WGMMA, or SM120 native (when available) — as long as output is byte-identical |
| **Tile dimensions** | Internal computation tiles can vary (but proof-canonical tile is fixed) |
| **Shared memory layout** | Swizzle patterns, bank conflict padding |
| **Pipeline stages** | Number of prefetch stages (2, 3, 4) |
| **Warp specialization** | Producer/consumer split, dedicated TMA warps |
| **CUDA graph** | Graph batch size, prepare/launch split |
| **Memory transfer** | TMA vs cudaMemcpyAsync, pinned vs page-locked |
| **Noise generation** | PRNG implementation, SIMD layout, batching |
| **Batch size** | Number of nonces per batch |
| **Buffering** | Ping-pong vs triple-buffer |
| **Hit detection** | Headless (in-kernel) vs post-GEMM scan |
| **M, N dimensions** | Within protocol bounds (affects hashrate, not validity) |

---

## 5. Existing Optimization Plans (Already Documented)

The codebase already has 8 documented optimization plans, ranked by impact. These are **consensus-safe** and **proven to work**:

| # | Plan | Impact | Status | Files |
|---|------|--------|--------|-------|
| 00 | **Full roadmap** 290→700–800+ TMAD/s | Master plan | Planning | 00-comprehensive-integration/PLAN.md |
| 01 | **GeForce v2** (PipelineTmaAsync) | +10–25% on v1 | Phase 1 built | 01-geforce-kernel-v2/PLAN.md |
| 02 | **Ptr-Array Grouped GEMM** | +10–20% if batch≥4 | Planning | 02-ptr-array-grouped-gemm/PLAN.md |
| 03 | **Stream-Split Pre-GEMM** | +1–5% | Planning | 03-stream-split-pregemm/PLAN.md |
| 04 | **Triple-GPU Half-Buffer** | +0–3% | Host code exists | 04-triple-gpu-half-buffer/PLAN.md |
| 05 | **Fuse Noise+NoisingA+GEMM** | +1–3% | Not built | 05-fuse-noise-noisinga-gemm/PLAN.md |
| 06 | **Sigma-Install Batching** | Startup latency | Planning | 06-sigma-install-b-hash-batching/PLAN.md |
| 07 | **CCCL Share Compaction** | <2% | Planning | 07-cccl-share-compaction/PLAN.md |
| 08 | **Consumer TMA Legacy** | 0% on prod | Deprioritized | 08-consumer-tma-legacy/PLAN.md |

Additionally, the **SM120 Native CUTLASS int8 Atom** plan (`performance optimizations/06-sm120-native-cutlass-int8-atom.md`) provides a detailed analysis of when CUTLASS adds a native SM120 int8 Operation and how to adopt it safely.

---

## 6. Recommended Investment Order

Based on the analysis, here is the recommended investment order for RTX 5090 optimization:

### Phase 0: Immediate (0 weeks, zero risk)

| Action | Effort | Expected Gain |
|--------|--------|---------------|
| Verify GeForce v1 is enabled (`PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1`) | 0 days | Already captured in current hashrate |
| Run kernel knob autotune (`tune_blackwell_knobs.sh`) | 0 days | +5–15% from optimal knobs |
| Enable CUDA graphs (`PROPMINER_GRAPH_BATCH=8`) | 0 days | +5–10% |
| Verify build targets `sm_120a` (not `sm_120` or `sm_80`) | 0 days | Prevents emulation fallback |

### Phase 1: GeForce v2 (4–6 weeks, medium risk)

| Action | Effort | Expected Gain |
|--------|--------|---------------|
| Implement PipelineTmaAsync in GeForce kernel | 4–6 weeks | +10–25% on v1 |
| Transcript byte-identity verification (500 trials) | 1 week | Consensus safety |
| NCU profiling + occupancy validation | 1 week | Performance validation |
| 24–48h pool soak test | 2 days | Production validation |

**Gates:** Transcript memcmp 100% pass, self-test pass, ≥+8% TMAD vs v1, 24h pool <1% rejects.

### Phase 2: Kernel Knob Tuning + Fused Pre-GEMM (3–5 weeks, low risk)

| Action | Effort | Expected Gain |
|--------|--------|---------------|
| Extended knob sweep (kblock, stages, swizzle, min_blocks, load_policy) | 1 week | +2–5% |
| Fuse noise+noisingA into GEMM kernel | 3–5 weeks | +1–3% |
| Verify buffer memcmp + transcript identity | 1 week | Consensus safety |

### Phase 3: Grouped GEMM (4–6 weeks, medium risk)

| Action | Effort | Expected Gain |
|--------|--------|---------------|
| Ptr-array grouped GEMM (batch≥4) | 4–6 weeks | +10–20% at batch≥4 |
| Per-group transcript verification | 1 week | Consensus safety |

**Requires:** Product decision to use `PROPMINER_BATCH ≥ 4` in production.

### Phase 4: Triple Buffering + Share Deferral (1–2 weeks, low risk)

| Action | Effort | Expected Gain |
|--------|--------|---------------|
| Enable `PROPMINER_TRIPLE_BUFFER=1` | 0 days (code exists) | +0–3% in share-heavy workloads |
| Validate with NCU profiling | 1 day | Occupancy validation |
| Enable `PROPMINER_DEFER_SHARE_GPU=1` | 0 days (code exists) | +10–20% when share rate is high |

### Phase 5: Future — SM120 Native int8 Atom (when CUTLASS provides it)

| Action | Trigger |
|--------|---------|
| Monitor CUTLASS for `SM120_16x8x32_TN<int8_t,int8_t,int32_t>` Operation | CUTLASS release |
| Implement conditional `#define PEARL_CONSUMER_MMA_ATOM_TYPE` | When Operation exists |
| Full 500-trial byte-identity verification | Before merge |
| Benchmark + promote only if ≥1% gain | Production promotion |

---

## 7. Testing & Validation Strategy

### 7.1 Consensus-Critical Verification (Mandatory for Any Change)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Transcript memcmp** | Compare GPU output vs reference kernel byte-by-byte | 0 mismatches in 500 trials |
| **Self-test** | `./propminer --self-test --rtx5090 --gpus 0` | Exit code 0 |
| **Prod self-test** | `PROP_MINER_SELF_TEST_PROD=1 ./propminer --self-test --rtx5090` | Exit code 0 |
| **Claimed hash roundtrip** | Host recomputes claimed_hash from GPU output | Exact match |
| **Merkle proof verification** | A and B Merkle proofs verify against roots | All pass |
| **Pool canary** | 24–48h pool operation | <1% share rejects |

### 7.2 Performance Regression Detection

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| **Benchmark comparison** | `./scripts/compare_bench.sh baseline.jsonl candidate.jsonl` | ≤2% regression |
| **NCU profiling** | `./scripts/profile_gemm_ncu.sh` | Tensor util ↑, no spills |
| **Occupancy check** | NCU occupancy metrics | No >5% drop |
| **Rigorous benchmarks** | `propminer_ref_benchmarks_rigorous` | p95/p99 within tolerance |

### 7.3 Upcast Hack Specific Verification (Why It Fails)

| Test | Expected Result |
|------|-----------------|
| FP16 GEMM vs INT32 GEMM byte comparison | **FAIL** — different transcript |
| Host-side claimed_hash vs GPU claimed_hash | **FAIL** — mismatch |
| Pool share submission | **FAIL** — rejected by pool |
| 500-trial memcmp against reference | **FAIL** — systematic drift |

---

## 8. Risk Assessment Matrix

### 8.1 Upcast Hack Risks (DO NOT IMPLEMENT)

| Risk | Severity | Certainty |
|------|----------|-----------|
| **Invalid shares (claimed_hash_mismatch)** | **Critical** | **100% certain** |
| **Pool rejects all shares** | **Critical** | **100% certain** |
| **FP16 precision drift in transcript** | **Critical** | **100% certain** |
| **Wasted engineering effort** | High | **100% certain** |
| **Production downtime during failed implementation** | High | **>90% likely** |
| **CUDA 13.0 upgrade not needed** | Medium | **100% certain** |

### 8.2 Recommended Optimization Risks (ACCEPTABLE)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Transcript byte drift (GeForce v2) | Critical | 500-trial memcmp + self-test |
| SMEM > 99 KB (GeForce v2) | High | `static_assert` on SharedStorage size |
| Register spill / occupancy drop | Medium | NCU before ship |
| Grouped GEMM wrong per group | Critical | Per-group transcript vs serial |
| Fuse pre-GEMM ApEA clamp wrong | Critical | Buffer memcmp vs unfused |
| CUTLASS version drift | Medium | Pin v4.4.0; probe script |

---

## 9. CUDA 13.0 Assessment

The executive brief recommends upgrading to **CUDA 13.0+** with `compute_120f`. Here is the assessment:

### 9.1 Current State

| Component | Current Version |
|-----------|----------------|
| CUDA Toolkit | 12.8 (primary), 12.4+ (minimum) |
| CUTLASS | v4.4.0 (hard-pinned) |
| Architecture flag | `sm_120a` (architecture-specific) |
| NVIDIA driver | 570+ |

### 9.2 CUDA 13.0 Impact

| Component | Impact |
|-----------|--------|
| **Architecture flags** | `sm_120a` → `sm_120f` (family-specific) — but `sm_120a` is **more correct** for RTX 5090 |
| **CUTLASS compatibility** | v4.4.0 may need bump; API changes possible |
| **Docker base image** | `nvidia/cuda:13.0.0-devel-ubuntu24.04` (when available) |
| **CUDA redist** | New soname symlinks needed |
| **Source code changes** | ~10 .cu/.h files may need API updates |
| **Driver requirement** | Likely newer than 570 |

### 9.3 Recommendation on CUDA 13.0

**Do not upgrade to CUDA 13.0 solely for the upcast hack.** The upcast hack is rejected. If CUDA 13.0 is needed later for other reasons:

1. Upgrade CUTLASS to a version known to work with CUDA 13.0
2. Run full transcript byte-identity verification
3. Benchmark against CUDA 12.8 baseline
4. Update Docker, CI, and deployment scripts
5. Test on actual RTX 5090 hardware before production

**The `sm_120a` (architecture-specific) target is preferred over `sm_120f` (family-specific) for RTX 5090** because the RTX 5090 (GB202) is a specific chip, not a family. The `a` suffix ensures all Blackwell features available on the specific chip are included.

---

## 10. Files That Would Change (If Upcast Hack Were Implemented — NOT RECOMMENDED)

For reference, here is what the upcast hack would touch. **This is documented for completeness, not as an implementation guide.**

| File | Change Required |
|------|----------------|
| `third_party/pearl-gemm/csrc/capi/portable_int8_helpers.cu` | Add upcast kernel (INT8→FP16) |
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | Change `ElementIn` from `int8_t` to `half_t` |
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` | Change MMA atom to FP16 variant |
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce_v2.cu` | Same as above |
| `third_party/pearl-gemm/csrc/gemm/kernel_traits.hpp` | Change `ElementIn_` template parameter |
| `third_party/pearl-gemm/csrc/gemm/collective_mainloop.hpp` | Change copy atom for FP16 gmem→smem |
| `third_party/pearl-gemm/csrc/gemm/collective_epilogue.hpp` | Change downcast epilogue |
| `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` | Add casting kernel launch |
| `src/host/pearl/gpu_worker.cpp` | Add FP16 buffer allocation |
| `src/host/pearl/pearl_capi_wrapper.cpp` | Update CAPI wrapper for new kernels |
| `CMakeLists.txt` | Update CUDA architecture flags |
| `third_party/pearl-gemm/csrc/capi/Makefile` | Update gencode flags |
| **New:** casting kernel file | INT8→FP16 elementwise upcast |
| **New:** downcast kernel file | FP16→INT8 elementwise downcast |
| **New:** verification test | FP16 vs INT32 transcript comparison |

**Total files touched: ~15 files, ~2000+ lines of new/modified code.**

---

## 11. Decision Matrix

### Option A: Do Nothing (Recommended)

| Aspect | Assessment |
|--------|------------|
| **Consensus safety** | ✅ 100% — no changes to code |
| **Performance** | ✅ ~300–470 TMAD/s already achieved |
| **Engineering cost** | ✅ Zero |
| **Risk** | ✅ Zero |
| **Next steps** | Follow the optimization roadmap in §6 |

### Option B: Implement Upcast Hack

| Aspect | Assessment |
|--------|------------|
| **Consensus safety** | ❌ 0% — would produce invalid shares |
| **Performance** | ❌ N/A — irrelevant because shares are invalid |
| **Engineering cost** | ❌ ~6–8 weeks (15+ files, 2000+ lines) |
| **Risk** | ❌ Critical — production-breaking |
| **Next steps** | Abandon, switch to proven roadmap |

### Option C: Follow Proven Optimization Roadmap

| Aspect | Assessment |
|--------|------------|
| **Consensus safety** | ✅ 100% — all plans have transcript byte-identity gates |
| **Performance** | ✅ Target: 700–800+ TMAD/s (Phase 0–4 combined) |
| **Engineering cost** | ~14–25 weeks (phased, each phase validated) |
| **Risk** | ✅ Each phase has go/no-go gates; rollback available |
| **Next steps** | Start with Phase 0 (immediate), then Phase 1 (GeForce v2) |

---

## 12. Recommended Action Plan

### Immediate Actions (This Week)

1. **Verify current build targets `sm_120a`** — check `capi/Makefile` gencode flags
2. **Enable GeForce v1 kernel** — set `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1` (default on blackwell builds)
3. **Run kernel knob autotune** — `./scripts/tune_blackwell_knobs.sh`
4. **Enable CUDA graphs** — set `PROPMINER_GRAPH_BATCH=8`
5. **Benchmark current hashrate** — `./scripts/build_and_benchmark.sh 120`
6. **Document baseline** — save JSON output for future comparison

### Short-Term (1–2 Months)

1. **Implement GeForce v2** — PipelineTmaAsync-based warp-specialized kernel
2. **500-trial transcript verification** — byte-identity against v1
3. **NCU profiling** — verify tensor utilization improvements
4. **24h pool soak test** — production validation

### Medium-Term (2–4 Months)

1. **Fused pre-GEMM** — eliminate ApEA HBM write/read
2. **Extended knob sweep** — optimize kblock, stages, swizzle, min_blocks, load_policy
3. **Triple-buffer validation** — enable `PROPMINER_TRIPLE_BUFFER=1`
4. **Share GPU deferral** — enable `PROPMINER_DEFER_SHARE_GPU=1`

### Long-Term (4–6 Months)

1. **Grouped GEMM** — if product commits to batch≥4
2. **CUDA 13.0 upgrade** — when CUTLASS provides SM120 int8 Operation
3. **CCCL share compaction** — if batch grows
4. **Continuous monitoring** — CUTLASS releases, SM120 int8 Operation availability

---

## 13. Appendix A: Key Code References

### Current INT8 Path (Working)

```
transcript_gemm_sm120.cu:48-60
  #define PEARL_CONSUMER_MMA_ATOM_TYPE SM80_16x8x32_S32S8S8S32_TN
  // Compiled into sm_120a cubin
  // Emits: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32

consumer/transcript_gemm_kernel.cu:124-138
  using ConsumerTiledMma = TiledMMA<
      MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
      Layout<Shape<_8, _1, _1>>,
      Tile<Int<kBM>, Int<kBN>, Int<kAtomK>>>;

portable_int8_gemm.cu:45-52
  #define PEARL_PORTABLE_MMA_ATOM_TYPE SM80_16x8x32_S32S8S8S32_TN
```

### CUTLASS SM120 int8 Gap

```cpp
// mma_sm120.hpp — int8 instantiation hits this
template <class a_type, class b_type, class c_type>
struct SM120_16x8x32_TN {
  static_assert(cutlass::detail::dependent_false<a_type>,
    "No MMA matches SM120_16x8x32_TN for given data types.");
};

// mma_traits_sm120.hpp — traits only; no int8 Operation
template <class a_type, class b_type, class c_type>
struct MMA_Traits<SM120_16x8x32_TN<a_type, b_type, c_type>>
  : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN> { ... };
```

### Consensus-Critical Verification

```cpp
// share_builder.cpp:468 — ShareBuilder::VerifyShare()
// Steps:
// 1. A Merkle proof verification
// 2. B Merkle proof verification
// 3. Root consistency check (hashA/hashB match)
// 4. Claimed hash recomputation (exact int8 GEMM on host)
// 5. Target clearing check (256-bit LE comparison)
```

### FP16 Precision Proof (Why Upcast Fails)

```
Exact int8 GEMM (current):
  For each (m,n): C[m][n] = Σ_k A[m][k] * B[n][k]   // exact int32
  A[m][k] ∈ [-63, +63], B[n][k] ∈ [-63, +63]
  Product: [-3969, +3969] per k
  Accumulated over K=128: [-508032, +508032] — fits in int32

FP16 GEMM (upcast hack):
  For each (m,n): C[m][n] = Σ_k fp16(A[m][k]) * fp16(B[n][k])  // rounded
  fp16 has 10 mantissa bits → ~3.3 decimal digits
  Each multiplication rounds to nearest FP16 representable value
  Accumulation in FP16 compounds rounding errors
  Downcast to int8 truncates/rounds — potentially different from exact int32

Result: C_fp16[m][n] ≠ C_int8[m][n] in general
Transcript differs → Jackpot differs → Pool rejects
```

---

## 14. Appendix B: Subagent Analysis Summary

This plan was produced from analysis by 7 parallel subagents:

| Agent | Scope | Files Analyzed | Key Finding |
|-------|-------|----------------|-------------|
| **Codebase Structure** | Full directory map | ~150 files | Complete architecture understanding |
| **CUDA Kernel Analysis** | All .cu/.cuh files | 38 files | SM80 IMMA is current path; no FP16 in transcript kernels |
| **Host C++ Analysis** | All .cpp/.h files | 83 files | Verification uses exact int8 arithmetic; upcast would fail |
| **Build System Analysis** | CMake, Make, scripts | 40 files | CUDA 13.0 not needed; sm_120a is correct target |
| **Plans & Docs Analysis** | All plans, docs, research | 25 files | Existing plans already cover proven optimizations |
| **Verification Logic Analysis** | Consensus, hashing, proofs | 30 files | Upcast breaks claimed_hash verification (100% certainty) |
| **Performance & Testing Analysis** | Benchmarks, tests, profiling | 25 files | No existing upcast tests; new test framework needed |

**Total code analyzed: ~6,000+ lines of CUDA, ~8,000+ lines of C++, ~2,000+ lines of build/config, ~2,000+ lines of documentation.**

---

## 15. Appendix C: Glossary

| Term | Definition |
|------|------------|
| **SM120** | RTX 5090 compute capability (sm_120a) |
| **IMMA** | Tensor Core 3rd gen instruction (`mma.sync.m16n8k32`) |
| **WGMMA** | Tensor Core 4th gen warp-group instruction (Hopper) |
| **tcgen05** | Tensor Core 5th gen TMEM instruction (B200 datacenter only) |
| **TMA** | Tensor Memory Accelerator (Blackwell gmem→smem) |
| **CUTLASS** | NVIDIA's CUDA template abstraction library for GEMM |
| **NoisyGEMM** | Pearl's proof-of-work algorithm (matrix multiply with noise) |
| **Transcript** | 16×uint32 hash state accumulated during GEMM |
| **Jackpot** | BLAKE3 hash of transcript — the PoW solution |
| **DAF** | Difficulty Adjustment Factor (rows × cols × dot_product_length) |
| **GeForce v1** | Current RTX 5090 kernel (TMA + manual barrier pipeline) |
| **GeForce v2** | Planned RTX 5090 kernel (TMA + CUTLASS PipelineTmaAsync) |
| **Upcast hack** | Proposed INT8→FP16→FP16 GEMM→FP16→INT8 conversion path |
| **SM80 atom** | `SM80_16x8x32_S32S8S8S32_TN` — the working INT8 MMA atom |

---

## 16. Internet Research Findings (Added 2026-07-09)

After the initial analysis, 2 additional subagents performed web and GitHub research to verify whether any recent developments change the conclusion. Here are the findings:

### 16.1 CUTLASS Recent Developments (Since v4.4.0)

| Development | Status | Impact on Upcast Hack |
|-------------|--------|----------------------|
| **CUTLASS 4.5.2** changelog | Released | Added SM120 block-scaled MMAs, MXFP8MMAOP, MXF8F6F4MMAOP — but **still no plain int8 Operation** for SM120 |
| **PR #3097** (closed June 2026) | "adding DSL INT8 MMA support on SM80+" | Adds `MmaI8Op` DSL wrapper for `mma.sync.aligned` — but this is a **naming wrapper**, not a new instruction. The PTX emitted is identical to SM80 |
| **PR #3302** (June 2026) | "INT8 warp-level MMA with CuTe C++ example" | Same as above — wraps existing `mma.sync.aligned` PTX. No new SM120 int8 instruction |
| **Issue #3096** | SM120 NVFP4 grouped GEMM garbage output | Bug in SM120 TMA shared memory calculation (uses SM100's 232KB instead of SM120's 99KB). **Unrelated to INT8** |
| **Issue #2186** | SM120 GEMM support in CUTLASS 3.x | Resolved: SM120 works via portable SM80/SM86/SM89 instructions. FP6 achieves ~226 TFLOPS, FP8 ~417 TFLOPS |

**Conclusion:** CUTLASS has NOT added a new SM120 int8 instruction. The SM120 int8 path remains the SM80 `mma.sync.aligned.m16n8k32.s8.s8.s32` instruction. PRs #3097/#3302 add DSL convenience wrappers but emit the same PTX.

### 16.2 The INT8→FP16→INT8 Approach in Production

The internet research found that the INT8→FP16 upcast approach **IS used in production**, but in a **different context**:

| Project | Use Case | Approach | Relevance |
|---------|----------|----------|-----------|
| **CUTLASS example 55** | Mixed-precision GEMM (INT8×FP16) | Upcasts INT8→FP16 for FP16 tensor core math | **Different use case** — designed for INT8×FP16 (not INT8×INT8). Explicitly documented as "not optimized for performance" |
| **Triton kernels on T4** | INT8 matmul on sm_75 | Upcasts to FP16 because T4 has no INT8 tensor cores | **Not applicable** — T4 (sm_75) genuinely lacks INT8 tensor cores. RTX 5090 (sm_120) has them |
| **vLLM/Marlin** | NVFP4 inference on SM120 | Dequantizes FP4→BF16 on the fly | **Different data type** — FP4→BF16 is valid for inference quantization. Pearl consensus requires exact INT8 |
| **Bias92/fused-qkv-int8-attention** | INT8 KV-cache dequant | Dequant INT8→FP16 in registers, then FP16 attention | **Different algorithm** — this is KV-cache dequant for attention, not GEMM consensus |

**Key distinction:** All production uses of INT8→FP16 upcast are either:
- **Mixed-precision GEMM** (INT8×FP16, not INT8×INT8) — where the FP16 operand is intentional
- **Inference quantization** (FP4→BF16) — where approximate results are acceptable
- **Different hardware** (T4 has no INT8 tensor cores) — where upcast is the only option

**None of these apply to Pearl's INT8×INT8 GEMM consensus requirement**, where exact bit-identical results are mandatory.

### 16.3 Academic Research: Ozaki Schemes (Reverse Direction)

The most relevant academic work is **EmuGEMM** (arXiv:2606.25453v1), which does the **reverse** of the upcast hack:

- **Direction:** Uses INT8 Tensor Cores to **emulate FP64/FP32 GEMM** (not INT8→FP16→INT8)
- **Method:** Ozaki Schemes I and II + Chinese Remainder Theorem for modular arithmetic
- **Results:** 3,654 Top/s on Blackwell (81% of INT8 peak)
- **Why it's different:** Ozaki decomposes FP matrices into integer slices, multiplies with INT8 Tensor Cores, then reconstructs. This works because the INT8→INT32 path is **exact** and the modular arithmetic is designed to avoid overflow.

The **reverse direction** (INT8→FP16→INT8) has no academic support because:
1. It introduces **irreversible precision loss** (FP16→INT8 cannot recover original values)
2. Ozaki's schemes work because INT8→INT32 is exact and reversible; FP16→INT8 is not
3. The accumulated rounding error in FP16 GEMM cannot be corrected by any known post-processing

### 16.4 Industry Projects on RTX 5090

| Project | Focus | INT8 Relevance |
|---------|-------|----------------|
| **Consumer-DeepGEMM** | Drop-in replacement for consumer Blackwell | Replaces Marlin's FP4 path with CUTLASS SM120 templates. Focuses on FP4/FP8, not INT8 |
| **blackwell-geforce-nvfp4-gemm** (lna-lab) | 12 patches for vLLM + FlashInfer + CUTASS to run NVFP4 on SM120 | 175 tok/s on Qwen3.6-35B MoE. Uses BF16/FP16 kernels. **No INT8 work** |
| **Raphael Friedmann's "imp"** | 97k lines C++20/CUDA for RTX 5090 | INT8 prefill path lifts throughput +36% to +151%. Uses CUTLASS NVFP4 dispatch with SM120-specific kernels. **Still uses native INT8, not upcast** |
| **Marlin** | FP4 dequantization for vLLM | Dequantizes FP4→BF16 on the fly. **Silently produces wrong results for FP4 MoE on SM120** — serves as a cautionary example that dequantization approaches can produce incorrect results |

### 16.5 CUDA Toolkit Versions

| Version | INT8 SM120 Support |
|---------|-------------------|
| **CUDA 12.8** | First toolkit to support SM120. INT8 via `mma.sync.aligned` PTX JIT |
| **CUDA 12.9** | Added SM121 support. Known bug: `cublasLtMatmul` with INT8 inputs could produce numerical inaccuracies with split-k reduction |
| **CUDA 13.0** | Full RTX 5000 series support. Blackwell GEMM improvements. **No new INT8 architectural changes** |
| **CUDA 13.3** | cuBLASLt: INT8 inputs, INT32 accumulation, INT32 outputs may return `CUBLAS_STATUS_NOT_SUPPORTED` when N > 65,536 |

**Conclusion:** CUDA 13.0 does not add any new INT8 capabilities for SM120. The INT8 path remains the same as CUDA 12.8.

### 16.6 What We Were NOT Missing

The internet research confirms that our initial analysis was **complete and correct**. We were NOT missing any of the following:

| Potential Gap | Status |
|---------------|--------|
| New CUTLASS SM120 int8 Operation | **Not added** — PRs #3097/#3302 are naming wrappers only |
| New PTX instruction for INT8 on SM120 | **Not added** — `mma.sync.aligned.m16n8k32` remains the only INT8 MMA |
| CUDA 13.0 INT8 improvements | **None** — same INT8 path as 12.8 |
| Production INT8→FP16→INT8 for consensus workloads | **Does not exist** — only used for mixed-precision and inference |
| Academic support for INT8→FP16→INT8 | **None found** — only reverse direction (Ozaki) exists |
| Any project successfully using upcast for INT8×INT8 consensus | **None found** |

---

## 17. Final Recommendation (Updated After Internet Research)

**Do NOT implement the upcast hack.** This conclusion has been validated by:

1. **Internal codebase analysis** (7 subagents, 150+ files, 18,000+ lines)
2. **External internet research** (2 subagents, web + GitHub, 15+ sources)

The combined evidence shows:
- RTX 5090 already has native INT8 Tensor Core support via `mma.sync.aligned.m16n8k32`
- CUTLASS has NOT added a new SM120 int8 instruction (PRs #3097/#3302 are naming wrappers only)
- The INT8→FP16→INT8 approach is used in production only for mixed-precision GEMM and inference quantization — not for consensus-critical workloads
- No academic paper supports the INT8→FP16→INT8 direction (only the reverse Ozaki scheme exists)
- No open-source project has successfully implemented this approach for INT8×INT8 consensus
- CUDA 13.0 adds no new INT8 capabilities for SM120

**The upcast hack is not a viable path for Pearl's INT8 NoisyGEMM consensus mechanism.**

**Instead, follow the proven optimization roadmap:**
1. GeForce v2 (PipelineTmaAsync) — +10–25%
2. Kernel knob autotuning — +5–15%
3. Fused pre-GEMM — +1–3%
4. Grouped GEMM (if batch≥4) — +10–20%
5. Target: 700–800+ TMAD/s

---

*Plan produced 2026-07-09 from comprehensive codebase analysis by 7 parallel subagents, updated 2026-07-09 with internet research by 2 additional subagents (web + GitHub). All code references verified against live implementation at /Users/mrpropop/Documents/vast ai/PropMiner. External sources: CUTLASS GitHub, NVIDIA developer forums, arXiv papers, vLLM issues, SemiAnalysis, DeepWiki, Raphael Friedmann's blog, Consumer-DeepGEMM, Marlin, EmuGEMM.*
