# SM120 Native CUTLASS int8 MMA Atom — Implementation Plan

**PropMiner / pearl-gemm · RTX 5090 (GB202, `sm_120a`) · Pearl V2 int8 NoisyGEMM**

| Field | Value |
|-------|-------|
| Status | Planning — atom swap blocked on CUTLASS int8 Operation gap |
| Primary files | `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu`, `csrc/consumer/transcript_gemm_kernel.cu`, `csrc/portable/portable_int8_gemm.cu` |
| Build flag | `PEARL_GEMM_SM120_NATIVE=1` (blackwell profile sets this by default) |
| CUTLASS baseline | v4.4.0 (`CMakeLists.txt` auto-fetch) |
| Proof constraint | Transcript bytes must remain identical to SM80 / H100 WGMMA reference |

---

## 1. Executive Summary

PropMiner’s RTX 5090 path compiles a native `sm_120a` cubin but still executes Pearl’s proof-critical int8 GEMM through the **SM80 `mma.sync` IMMA atom** (`SM80_16x8x32_S32S8S8S32_TN` → PTX `m16n8k32.s32.s8.s8.s32`). That is correct today: consumer Blackwell (CC 12.0) has **no separate int8 Tensor Core ISA** in CUTLASS, and GeForce Blackwell does **not** expose the datacenter `tcgen05` path used on B200 (`sm_100a`).

**Expected hashrate impact of a pure int8 MMA atom swap: ~0%.** CUTLASS 4.x `SM120_16x8x32_TN` specializations cover FP8/FP4/FP6 (`kind::f8f6f4`, `kind::mxf8f6f4.block_scale`), not plain int8. The `MMA_Traits` stub for int8 inherits SM80 layouts but there is no matching `Operation` with an `fma()` that emits distinct PTX. Swapping type names would either fail to compile or lower to the same hardware instruction.

**Where +5–20% can still come from on RTX 5090** (already partially captured in the Blackwell profile, independent of a mythical int8 atom):

| Lever | Mechanism | Observed / expected range |
|-------|-----------|---------------------------|
| Blackwell tile tuning | `128×256×128`, `Swizzle<3,4,3>`, 2 stages | Small wins vs Ampere defaults (e.g. ~0.5% swizzle on 5090 bench) |
| Occupancy / launch bounds | `minBlocks=1`, wave-aligned grid (8192 CTAs @ M=8192 N=32768) | Fills 170 SMs; avoids tail-wave loss |
| Async snapshot placement | XOR-reduce in cp.async shadow | Instruction-level overlap; no transcript change |
| Portable int8 fusion | Fused add+clamp epilogue (`portable_int8_gemm.cu`) | Removes 256 MiB int32 DRAM round-trip on ApEA/BpEB |
| Future: wider N / VRAM residency | `Rtx5090Profile::pick_n_for_vram()` up to 262144 | Multi-wave persistence when memory allows |

**Recommendation:** Treat `PEARL_GEMM_SM120_NATIVE` as a **readiness hook**, not a performance switch. Keep SM80 IMMA as the production atom until CUTLASS ships a genuine int8 `SM120_*` Operation (or documents that SM80 is the canonical consumer int8 path). Invest marginal effort in knob sweeps (`tune_blackwell_knobs.sh`), ncu profiling, and portable-path fusion rather than a large tcgen05 rewrite for consumer GPUs.

---

## 2. Current MMA Atom Selection

### 2.1 Consumer transcript kernel (dominant hashrate path)

```124:138:PropMiner/third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu
// Default MMA atom is the proven SM80 int8 atom.  It is byte-identical to
// WGMMA for partition_C coordinates and works on Ampere, Ada, and Blackwell.
// For native sm_120 builds, define PEARL_CONSUMER_MMA_ATOM_TYPE before
// including this file (see blackwell/transcript_gemm_sm120.cu).
#ifndef PEARL_CONSUMER_MMA_ATOM_TYPE
#define PEARL_CONSUMER_MMA_ATOM_TYPE SM80_16x8x32_S32S8S8S32_TN
#endif

using ConsumerTiledMma = TiledMMA<
    MMA_Atom<PEARL_CONSUMER_MMA_ATOM_TYPE>,
    Layout<Shape<_8, _1, _1>>,
    Tile<Int<kBM>, Int<kBN>, Int<kAtomK>>>;
```

- **Tile:** `128 × 256 × 128` (Blackwell defaults via `PEARL_GEMM_BLACKWELL`)
- **Atom K:** 32 (`kAtomK`); smem K-tile 128 → 4 MMA K-fragments per mainloop iter
- **Warps:** 8 (`Layout<Shape<_8,_1,_1>>`) × 32 threads = 256 threads/CTA
- **Accumulators:** 128 int32 registers/thread (`kFragSize`)
- **Transcript:** 16 u32 slots/thread (`kTranscriptSlots`)

### 2.2 Blackwell wrapper (`transcript_gemm_sm120.cu`)

The sm_120 TU includes the consumer kernel wholesale and only provides a launch wrapper. **It does not currently define `PEARL_CONSUMER_MMA_ATOM_TYPE`**, even when `PEARL_GEMM_SM120_NATIVE=1`:

```48:60:PropMiner/third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu
// Use the native Blackwell MMA atom when explicitly requested and the types
// are supported by CUTLASS (FP8/FP4).  Legacy int8 IMMA for Pearl mining uses
// SM80_16x8x32_S32S8S8S32_TN compiled into an sm_120a cubin — consumer
// Blackwell executes the same mma.sync m16n8k32.s32.s8.s8.s32 instruction;
// CUTLASS does not provide a separate SM120_16x8x32_TN Operation for int8.
#if defined(PEARL_GEMM_SM120_NATIVE) && PEARL_GEMM_SM120_NATIVE
  #pragma message("transcript_gemm_sm120.cu: sm_120a native build with SM80 int8 MMA atom + Blackwell tuning")
#else
  #pragma message("transcript_gemm_sm120.cu: sm_120a build with SM80 MMA atom")
#endif

#include "../consumer/transcript_gemm_kernel.cu"
```

**Implication:** Both pragma branches emit the **same kernel instantiation** today. The flag is informational + future hook.

### 2.3 Portable int8 helper (noise_A / noise_B / ApEA / BpEB)

```45:52:PropMiner/third_party/pearl-gemm/csrc/portable/portable_int8_gemm.cu
// Native sm_120a int8 IMMA for noise_A / noise_B portable projections.
// CUTLASS has no SM120_16x8x32_TN Operation for int8 (only FP8/FP4); consumer
// Blackwell executes the same mma.sync m16n8k32.s32.s8.s8.s32 hardware insn
// via the SM80 MMA_Atom traits compiled into an sm_120a cubin (no sm_80/sm_89
// forward-compat cubins).
#ifndef PEARL_PORTABLE_MMA_ATOM_TYPE
#define PEARL_PORTABLE_MMA_ATOM_TYPE SM80_16x8x32_S32S8S8S32_TN
#endif
```

Portable path uses a smaller tile (`64 × {64,128} × 64`, 128 threads) but the same SM80 atom.

### 2.4 Build system wiring

| Location | blackwell profile behavior |
|----------|---------------------------|
| `csrc/capi/Makefile` | `-gencode=arch=compute_120a,code=sm_120a`; `ARCH_DEFINES` includes `-DPEARL_GEMM_SM120_NATIVE=1` |
| `csrc/capi/CMakeLists.txt` | `CMAKE_CUDA_ARCHITECTURES 120a`; same define |
| `PropMiner/CMakeLists.txt` | Fetches CUTLASS v4.4.0 if missing |
| Docker audit | Cubins must be `sm_120a` only |

Optional override: `-DCMAKE_CUDA_FLAGS="-DPEARL_GEMM_SM120_NATIVE=1"` (redundant for blackwell; useful only if testing atom swap patches on other profiles).

---

## 3. CUTLASS Gap Analysis (int8 on Consumer Blackwell)

### 3.1 What CUTLASS 4.x actually provides for SM120

| Component | int8 (`s8 × s8 → s32`) | FP8/FP4/FP6 |
|-----------|------------------------|-------------|
| `include/cute/arch/mma_sm120.hpp` `Operation` | **None** — default template `static_assert`s | Many `SM120_16x8x32_TN<…>` + block-scaled `SM120_16x8x32_TN_VS` |
| `include/cute/atom/mma_traits_sm120.hpp` `MMA_Traits` | Inherits `MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>` for generic `SM120_16x8x32_TN<a,b,c>` | Full layouts for MX formats |
| PTX emitted | Would need `mma.sync…s32.s8.s8.s32` (SM80-class IMMA) | `kind::f8f6f4.m16n8k32…` or `kind::mxf8f6f4.block_scale…` |
| Consumer GeForce tcgen05 | **Not available** | N/A |

Relevant CUTLASS source (main branch, v4.4+ family):

```cpp
// mma_traits_sm120.hpp — traits only; no int8 Operation counterpart
template <class a_type, class b_type, class c_type>
struct MMA_Traits<SM120_16x8x32_TN<a_type, b_type, c_type>>
  : MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
{
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  // ...
};
```

```cpp
// mma_sm120.hpp — int8 instantiation hits this unless specialized
template <class a_type, class b_type, class c_type>
struct SM120_16x8x32_TN {
  static_assert(cutlass::detail::dependent_false<a_type>,
    "No MMA matches SM120_16x8x32_TN for given data types.");
};
```

**Conclusion:** The comment in `transcript_gemm_sm120.cu` suggesting `SM120_16x8x32_TN<int8_t,int8_t,int32_t>` is **aspirational**. A naïve `#define PEARL_CONSUMER_MMA_ATOM_TYPE SM120_16x8x32_TN<int8_t,int8_t,int32_t>` **will not compile** against stock CUTLASS.

### 3.2 Architecture split (why B200 code does not apply)

| | Datacenter Blackwell (`sm_100a`) | Consumer Blackwell (`sm_120a`) |
|--|----------------------------------|--------------------------------|
| Primary MMA | `tcgen05.mma` + TMEM | Legacy `mma.sync` (+ new F8F6F4 variants) |
| PropMiner kernel | `transcript_gemm_sm100.cu` | `transcript_gemm_sm120.cu` → consumer kernel |
| int8 proof path | Separate verify harness vs portable reference | SM80 atom in-place |
| Cluster / TMA | Full datacenter feature set | Cluster 1×1×1; no TMA multicast on GeForce |

### 3.3 Compiler / ISA notes

Reverse-engineering and PTX docs indicate consumer Blackwell extends `mma.sync` with **block-scaled MX formats**, not a new plain-int8 instruction. Until CUDA/cicc emits those paths reliably on `sm_120a`, IMMA remains the int8 route. Pearl **cannot** change numeric type (FP8) without breaking protocol int8 semantics.

---

## 4. SM80-on-sm_120a Compatibility (Why It Works)

### 4.1 Same hardware instruction

Compiling `SM80_16x8x32_S32S8S8S32_TN` into an `sm_120a` cubin emits:

```ptx
mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 …
```

NVIDIA forward-supports this IMMA opcode on Blackwell consumer parts. The cubin targets `sm_120a` directly — PropMiner does **not** ship `sm_80` forward-compat fatbins for production Docker builds.

### 4.2 Proof-canonical layout preservation

`probe_sm80_layout.cu` established:

- `TiledMMA(SM80_16x8x32, AtomLayout (8,1,1), Tile(128,256,32))` → **32768/32768** matching `partition_C` slots vs H100 WGMMA `m64n256k32`.
- Same 16×8 sub-fragment geometry; warp tiling differs but global thread→(m,n) mapping aligns.

Therefore SM80-on-sm_120a is not a “fallback hack”; it is the **protocol-correct** int8 path until a distinct atom exists.

### 4.3 Supporting infrastructure unchanged

These remain valid on sm_120a with SM80 atom:

- `SM75_U32x4_LDSM_N` smem→reg (`ldmatrix`)
- `SM80_CP_ASYNC_CACHEGLOBAL` gmem→smem
- `Swizzle<3,4,3>` smem layout (Blackwell-tuned default)
- Async XOR snapshot scheduling in the mainloop

No sm_90 WGMMA or sm_100 tcgen05 headers are pulled into the blackwell consumer lane.

---

## 5. When SM120 Native Becomes Available

### 5.1 Triggers to re-open this plan

1. **CUTLASS adds** `SM120_16x8x32_TN<int8_t,int8_t,int32_t>` (or `SM120_16x8x32_S32S8S8S32_TN`) with a real `fma()` in `mma_sm120.hpp` that emits int8 PTX.
2. **PTX ISA documents** a new consumer int8 opcode distinct from `s32.s8.s8.s32` (unlikely for Pearl — would still need layout proof).
3. **NVIDIA documents** that SM80 IMMA on sm_120 is deprecated/throttled (would motivate verification, not necessarily a code change).

### 5.2 What “native” likely means in practice

Even if CUTLASS adds an int8 typedef alias, it will probably:

- Inherit SM80 `MMA_Traits` (same `partition_C`)
- Emit the **same** `mma.sync` PTX
- Serve naming/arch-tag clarity only

Performance would match SM80 atom ± noise unless scheduling or compiler heuristics change.

### 5.3 What will NOT arrive for consumer int8

- `tcgen05.mma.kind::i8` on RTX 5090 — datacenter-only pipeline (`transcript_gemm_sm100.cu` scope).
- Block-scaled MX int8 — Pearl matrices are true int8, not MXFP8 with scale tensors.

---

## 6. Fragment Layout / partition_C Requirements

Any atom swap **must** preserve these invariants (hard failure = invalid shares):

| Invariant | Consumer transcript | Portable int8 |
|-----------|--------------------|---------------|
| Atom MNK | 16 × 8 × 32 | 16 × 8 × 32 |
| `size(partition_C(identity))` | 128 | 32 (4 warps × smaller tile) |
| Transcript slots / thread | 16 | N/A (no transcript) |
| Snapshot cadence | Every `R=128` K-columns | N/A |
| XOR hash input | All 128 accumulator lanes | N/A |
| int32 accumulation | Mod 2³² associative | Same |

### 6.1 Validation procedure (pre-merge)

1. Static assert `size(tCcD) == kFragSize` (already in consumer kernel).
2. Run layout probe comparing SM80 vs candidate atom:
   - Extend `probe_sm80_layout.cu` or add `probe_sm120_int8_layout.cu`.
   - Assert identical `(thread, slot) → (m, n)` for all 32768 coords per tile.
3. Runtime byte diff vs reference kernel (see §9).

### 6.2 Code touch points if layouts diverge (unexpected)

If a future atom changed `CLayout` (would be a protocol break):

- `pow_utils.hpp` / XOR reduction loops
- Transcript store indexing in consumer kernel epilogue
- `transcript_canonical.cuh` metadata
- Host-side `transcript_buffer_elems()` sizing (should remain unchanged if tile MN constant)

---

## 7. Implementation Plan

### Phase 0 — Documentation & flag honesty (low effort, do first)

- [ ] Update `transcript_gemm_sm120.cu` header comments to state that `PEARL_GEMM_SM120_NATIVE` does **not** swap int8 atoms today.
- [ ] Add compile-time guard:

```cpp
#if defined(PEARL_GEMM_SM120_NATIVE) && PEARL_GEMM_SM120_NATIVE
  #if defined(PEARL_CONSUMER_MMA_ATOM_TYPE)
    #error "PEARL_CONSUMER_MMA_ATOM_TYPE must not be set to SM120 int8 until CUTLASS Operation exists"
  #endif
#endif
```

(Or only activate swap behind a new `PEARL_GEMM_SM120_INT8_EXPERIMENTAL` once CUTLASS supports it.)

### Phase 1 — CUTLASS capability probe (build-time)

Add `scripts/check_cutlass_sm120_int8.sh`:

```bash
#!/usr/bin/env bash
CUTLASS="${1:-third_party/pearl-gemm/third_party/cutlass}"
grep -q 's32.s8.s8.s32' "$CUTLASS/include/cute/arch/mma_sm120.hpp" && echo INT8_OP=maybe || echo INT8_OP=no
grep -q 'SM120_16x8x32_TN<int8' "$CUTLASS/include/cute/arch/mma_sm120.hpp" && echo INT8_SPECIALIZATION=yes || echo INT8_SPECIALIZATION=no
```

Wire into CI / Docker build as informational (non-failing until op exists).

### Phase 2 — Atom swap scaffolding (blocked on CUTLASS)

When CUTLASS ships int8 Operation, patch `transcript_gemm_sm120.cu`:

```cpp
#if defined(PEARL_GEMM_SM120_NATIVE) && PEARL_GEMM_SM120_NATIVE
  #include <cute/arch/mma_sm120.hpp>
  #include <cute/atom/mma_traits_sm120.hpp>
  #define PEARL_CONSUMER_MMA_ATOM_TYPE SM120_16x8x32_S32S8S8S32_TN  // exact name TBD by CUTLASS
  // Mirror for portable:
  // #define PEARL_PORTABLE_MMA_ATOM_TYPE ...
#endif
#include "../consumer/transcript_gemm_kernel.cu"
```

Steps:

1. Confirm `MMA_Atom<…>` instantiates and `nvcc -arch=sm_120a` succeeds.
2. `cuobjdump -sass` spot-check: expect IMMA opcode (compare vs SM80 build).
3. Run layout probe + transcript byte diff (§9).
4. Benchmark `--bench 60 --rtx5090`; accept only if ≥1% gain **and** self-test passes.

### Phase 3 — Portable path parity

Apply the same conditional `#define PEARL_PORTABLE_MMA_ATOM_TYPE` in `portable_int8_gemm.cu` (or a shared `pearl_mma_atom_blackwell.cuh` included by both TUs) so noise projections and transcript kernel stay on the same atom family.

### Phase 4 — Production promotion criteria

Promote atom swap to default only if:

- [ ] 500-trial byte-identity sweep PASS (mirror `transcript_gemm_sm100.cu` harness)
- [ ] `./propminer --self-test --rtx5090` PASS
- [ ] ncu shows ≥5% improvement in `sm__pipe_tensor_cycles_active` **or** end-to-end H/s gain ≥3%
- [ ] SASS diff reviewed (no unexpected spills / extra moves)

If all criteria pass but gain ≈0%, keep SM80 atom as default and document SM120 alias as cosmetic.

---

## 8. Relationship to tcgen05 (Lighter-Weight Alternative?)

**No — for RTX 5090 int8 mining, tcgen05 is the heavier path, not a lighter one.**

| Aspect | SM80/SM120 `mma.sync` int8 (current) | tcgen05 (B200 `sm_100a` only) |
|--------|--------------------------------------|-------------------------------|
| Scope | Consumer + Ampere + Ada | Datacenter Blackwell |
| Kernel complexity | Single-phase cute GEMM + reg transcript | TMEM alloc, warp specialization, dual-stream MMA+ld |
| Code size | `transcript_gemm_kernel.cu` (~700 LOC) | `transcript_gemm_sm100.cu` (~700+ LOC) + SM100 headers |
| Proof validation cost | Already proven vs WGMMA | Separate 500-trial harness vs portable |
| HW support on 5090 | Yes | **No** — will not launch |

`transcript_gemm_sm120.cu` mentions a future inline PTX hook:

```ptx
tcgen05.mma.cta_group::1.kind::i8 [d_tmem], [a_smem], [b_smem];
```

That comment describes a **datacenter-style** rewrite, inappropriate for GeForce. Do not port `transcript_gemm_sm100.cu` to `sm_120a` — binaries are incompatible and the ISA path differs.

**Pragmatic alternative with better ROI:** continue optimizing the existing cute mainloop (load policy, stages, swizzle, fused portable epilogues, CUDA graphs for seed upload) rather than tcgen05.

---

## 9. Testing (Transcript Byte Diff vs SM80 Path)

### 9.1 Reference kernel

Use the SM80-atom consumer kernel compiled for `sm_120a` as golden reference (current production). Optional cross-check against portable `transcript_gemm_kernel.cu` on the same device.

### 9.2 Harness pattern (from B200 verify)

Adapt `transcript_gemm_sm100.cu` `PEARL_SM100_VERIFY_MAIN` pattern:

```cpp
// Pseudocode for sm_120 experimental verify TU
pearl::blackwell::launch_transcript_gemm_sm120(..., dTr_candidate, ...);  // candidate atom
pearl::consumer::launch_transcript_gemm(..., dTr_ref, ...);                 // SM80 reference
cudaMemcpy(h_cand, dTr_candidate, ...);
cudaMemcpy(h_ref,  dTr_ref, ...);
assert(memcmp(h_cand, h_ref, tr_bytes) == 0);
```

Run matrix:

| Parameter | Values |
|-----------|--------|
| M, N | 8192 × 32768 (production), 128 × 256 (unit) |
| K | 128, 256, 512, 2048 |
| R | 128 |
| Trials | 500 random int8 fills (match sm100 harness) |

### 9.3 Integration tests

```bash
./build/propminer --self-test --rtx5090 --gpus 0
./scripts/remote_test_kit.sh          # full kit incl. self-test
./scripts/build_and_benchmark.sh 60   # perf regression
```

### 9.4 SASS / ncu checks

```bash
ncu --target-processes all \
  --kernel-regex 'transcript_gemm_kernel_consumer' \
  ./build/propminer --bench 10 --rtx5090 --gpus 0
```

Metrics:

- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`
- Register spill counts vs SM80 baseline
- `smsp__sass_thread_inst_executed_op_mma_sp.sync_on` (or equivalent IMMA counter)

### 9.5 Portable int8

For noise paths, int32 outputs are tiling-independent (associativity); still run:

- `int8_matmul_i32` vs prior baseline on `(M=16384, K=4096, N={64,128})`
- Fused add+clamp byte compare vs two-pass path (documented in `portable_int8_gemm.cu`)

---

## 10. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Compile break if atom typedef added prematurely | High | Guard with CUTLASS probe script; keep SM80 default |
| Silent transcript drift | **Critical** | 500-trial memcmp + `--self-test`; block merge on any mismatch |
| Zero perf gain after integration work | Medium | Phase 4 promotion gates; don’t merge cosmetic alias |
| Misleading `PEARL_GEMM_SM120_NATIVE` flag | Medium | Phase 0 doc + pragma clarity |
| Accidental tcgen05 / sm_100a bleed | High | Docker cubin audit (`sm_120a` only); separate TU lists in Makefile |
| CUTLASS version drift | Medium | Pin tag; monitor release notes (§12) |
| Overclock + wrong kernel | Medium | Self-test after OC (`RTX5090_BLUEPRINT.md`) |

---

## 11. Effort

| Phase | Work | Estimate | Dependency |
|-------|------|----------|------------|
| 0 — Flag honesty / docs | Comment + guard | **0.5 day** | None |
| 1 — CUTLASS probe script | Shell + CI hook | **0.5 day** | None |
| 2 — Atom swap + compile | Conditional defines, 2 TUs | **1–2 days** | CUTLASS int8 Operation |
| 2b — Layout probe extension | CUDA mini-TU | **1 day** | Phase 2 |
| 3 — Verify harness sm_120 | Clone sm100 verify | **1–2 days** | Phase 2 |
| 4 — Benchmark + ncu + promote | RTX 5090 hardware | **1–2 days** | Phase 3 pass |
| **Total (blocked)** | | **4–8 days** | NVIDIA/CUTLASS int8 op |
| **Total (current state: SM80 only)** | Phases 0–1 only | **1 day** | None |

**Ongoing perf work unrelated to atom swap** (knob sweep, fused portable epilogue, CUDA graphs): see other optimization plans in this series; typically **2–5 days** for measurable +5–20% cumulative gains.

---

## 12. Monitoring CUTLASS Releases

### 12.1 Watch list

| Signal | Where to look |
|--------|---------------|
| int8 `SM120_*` Operation | `include/cute/arch/mma_sm120.hpp` |
| Traits updates | `include/cute/atom/mma_traits_sm120.hpp` |
| GeForce examples | `examples/79_blackwell_geforce_gemm/` |
| Release notes | CUTLASS CHANGELOG / GitHub releases |
| PTX ISA | CUDA Toolkit `mma.sync` int8/changelog |

### 12.2 Upgrade procedure

1. Bump pinned tag in `PropMiner/CMakeLists.txt` (currently v4.4.0).
2. Run `scripts/check_cutlass_sm120_int8.sh`.
3. Attempt experimental build with `-DPEARL_GEMM_SM120_INT8_EXPERIMENTAL=1` (future flag).
4. Full verify matrix (§9) on RTX 5090 before merging tag bump.

### 12.3 Automation suggestion

Add a monthly Dependabot-style job or manual checklist item:

```bash
cd third_party/pearl-gemm/third_party/cutlass
git fetch --tags
git log v4.4.0..origin/main -- include/cute/arch/mma_sm120.hpp
```

Subscribe to NVIDIA CUTLASS GitHub releases and CUDA Toolkit release notes.

---

## Appendix A — Quick Decision Tree

```
Need int8 Pearl proof on RTX 5090?
├─ Yes → Use SM80_16x8x32_S32S8S8S32_TN in sm_120a cubin (today's production)
├─ CUTLASS adds SM120 int8 Operation?
│   ├─ Same PTX as SM80? → Optional alias; expect ~0% gain; still run byte diff
│   └─ New PTX/layout? → Full Phase 2–4; likely protocol review required
└─ Tempted to port tcgen05 from B200?
    └─ Stop — wrong ISA for GeForce; use sm_100a build on B200 hardware only
```

## Appendix B — File Checklist

| File | Action |
|------|--------|
| `blackwell/transcript_gemm_sm120.cu` | Add conditional atom define when ready |
| `consumer/transcript_gemm_kernel.cu` | No change unless layouts break |
| `portable/portable_int8_gemm.cu` | Mirror atom define |
| `capi/Makefile` / `capi/CMakeLists.txt` | Optional `PEARL_GEMM_SM120_INT8_EXPERIMENTAL` |
| `CMakeLists.txt` | CUTLASS version pin + probe hook |
| New: `blackwell/verify_transcript_sm120.cu` | Byte-identity harness (optional) |
| New: `scripts/check_cutlass_sm120_int8.sh` | Capability detection |

---

*Last updated: 2026-07-05 · Targets RTX 5090 / `sm_120a` native builds only.*
