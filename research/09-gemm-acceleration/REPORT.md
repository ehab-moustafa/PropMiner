# Pearl GEMM Acceleration Paths on RTX 5090

**PropMiner Research · Topic 09**  
**Date:** July 2026  
**Scope:** Map of `pearl-gemm` kernel variants, gap analysis vs pure CUTLASS/cuBLAS GEMM, honest assessment of 100× claims, realistic **1.5–2.5×** acceleration on RTX 5090, and reference material in `research/01-git-repos-mining/repos/cutlass`.

---

## Executive Summary

Pearl mining GEMM is **not** a dense linear algebra benchmark. It is a **proof-canonical int8 multiply** fused with transcript XOR accumulation and keyed BLAKE3 jackpot checks. PropMiner's `third_party/pearl-gemm` tree contains **multiple architecture-specific paths**, but RTX 5090 production today runs a single dominant stack:

```
pearl_gemm_capi.cpp
  → consumer::launch_transcript_gemm_headless()
    → transcript_gemm_sm120.cu
      → transcript_gemm_kernel.cu  (SM80 mma.sync, 128×256×128)
```

| Question | Honest answer |
|----------|---------------|
| Can we get **100×** from "better GEMM"? | **No.** Pure GEMM is already ~85–92% of iteration time; even infinite-speed GEMM caps at ~1.1–1.2× total. |
| Can we get **10×** from CUTLASS alone? | **No.** Consumer int8 uses same `mma.sync` instruction CUTLASS would emit. |
| Realistic **5090 envelope**? | **1.5–2.5×** over ~300 TMAD/s via TMA loads, occupancy, knob autotune, warp specialization. |
| Reference CUTLASS clone? | `research/01-git-repos-mining/repos/cutlass` — SM120 GeForce examples, TMA collectives, **not** int8 transcript GEMM. |

---

## 1. pearl-gemm Directory Map

### 1.1 Top-level layout

```
third_party/pearl-gemm/csrc/
├── capi/           # pearl_gemm_capi.cpp — host dispatch, batching, tensor_hash hooks
├── consumer/       # ★ Production RTX 5090 transcript GEMM (mma.sync)
├── blackwell/      # sm_120 wrappers + sm_100 B200 reference + tcgen05 probes
├── portable/       # Parity / SM75 / dp4a fallback paths
├── gemm/           # CUTLASS-style noising, noise_gen, inner_hash, pow_utils
├── tensor_hash/    # BLAKE3 Merkle commitment kernels
├── blake3/         # Device BLAKE3 implementation
├── rocm/           # AMD HIP port (reference)
└── sycl/           # Intel SYCL port (reference)
```

### 1.2 Architecture dispatch matrix

| Path | Arch flag | Kernel file | Status on 5090 |
|------|-----------|-------------|----------------|
| **Consumer transcript** | `PEARL_GEMM_BLACKWELL` | `consumer/transcript_gemm_kernel.cu` | **Production** |
| **SM120 wrapper** | `sm_120a` gencode | `blackwell/transcript_gemm_sm120.cu` | Thin include of consumer |
| **SM120 GeForce experimental** | compile opt-in | `blackwell/transcript_gemm_sm120_geforce.cu` | TMA warp-spec scaffold |
| **B200 tcgen05** | `PEARL_GEMM_ARCH=b200` | `blackwell/transcript_gemm_sm100.cu` | **Not loadable on 5090** |
| **Hopper WGMMA** | sm_90 | via gemm/ instantiations | H100 only |
| **Portable SM75** | `PEARL_GEMM_PORTABLE` | `portable/transcript_gemm_sm75.cu` | Correctness fallback |
| **dp4a portable** | no tensor cores | `portable/transcript_gemm_dp4a.cu` | 10–50× slower |

Build: `third_party/pearl-gemm/csrc/capi/Makefile` — `PEARL_GEMM_ARCH=blackwell` → `-gencode=arch=compute_120a,code=sm_120a`.

---

## 2. Production Hot Path (Consumer Kernel)

### 2.1 Launch chain

```cpp
// pearl_gemm_capi.cpp (simplified)
#ifdef PEARL_GEMM_BLACKWELL
  pearl::consumer::launch_transcript_gemm_headless(...);
#endif
```

Headless path fuses:

1. `cp.async` 2-stage pipeline (A tile stream, B from VRAM)
2. `ldmatrix.x4` → SM75 LDSM atom
3. `cute::gemm` over SM80 `MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>`
4. XOR-tree reduction on accumulator fragments
5. `rotl_xor<13>` into 16-slot transcript
6. `blake3::compress_msg_block_u32` vs `pow_target`
7. `HostSignalHeader` write on hit

File: `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` (~800+ lines)

### 2.2 Fixed proof parameters

| Symbol | Value | Mutable? |
|--------|-------|----------|
| kBM | 128 | **No** — `#error` |
| kBN | 256 | **No** |
| kBK | 128 | Protocol K=128 |
| kFragSize | 128 int32/thread | Drives register pressure |
| kTranscriptSlots | 16 | Protocol |
| PEARL_CONSUMER_MMA_ATOM_TYPE | SM80_16x8x32 | Consensus |

Blackwell tuning knobs (safe):

| Knob | Default (5090) | Sweep via |
|------|----------------|-----------|
| STAGES | 2 | `tune_blackwell_knobs.sh` |
| SWIZZLE | `<3,4,3>` | doc 05 |
| KBLOCK | 128 | doc 05 |
| MIN_BLOCKS | 1 | doc 05 |

Measured: Swizzle<3,4,3> → **300.78 TMAD/s** vs 299.19 at `<2,4,3>` on N=262144.

### 2.3 What consumer kernel is NOT

- Not cuBLAS `cublasGemmEx` — no transcript
- Not CUTLASS `GemmUniversal` standalone — epilogue is custom POW
- Not tcgen05 UMMA — wrong ISA on GeForce
- Not persistent CTAs across batches — graph relaunch per batch

---

## 3. Pure GEMM vs Pearl GEMM Gap

### 3.1 Operation count comparison

For M=8192, N=262144, K=128:

| Work type | MACs / matmul | Relative |
|-----------|---------------|----------|
| **Pure int8 GEMM** | M×N×K = 2.75×10¹⁴ | 1.00× |
| **+ Transcript XOR** | +O(tiles × frags × K/R) | +~3–8% equivalent |
| **+ BLAKE3 jackpot** | +O(tiles × 16 u32 compress) | +~3–6% equivalent |
| **+ Noising (amortized)** | +O(M×R + N×R) per σ | <<1% per iter |

**Fused overhead vs pure GEMM:** ~**8–15%** — matches noise-hash report split (GEMM 85–92%).

### 3.2 cuBLAS / CUTLASS theoretical peak

NVIDIA rates RTX 5090 at **838 INT8 TOPS**. Naive cuBLAS int8 GEMM at same shape might achieve **500–700 TMAD/s** (60–85% of peak) with optimal tiles — **1.7–2.3× faster than Pearl's ~300 TMAD/s** on the MMA portion alone.

But Pearl **cannot use pure cuBLAS** for mining because:

1. Transcript requires specific `partition_C` fragment order
2. XOR-reduce runs on register fragments mid-mainloop
3. PoW target check is per-tile, proof-canonical positions only
4. Noise must be applied in-protocol before GEMM

### 3.3 The "pure GEMM gap" defined

```
pure_GEMM_gap = (theoretical_peak_Pearl_shape / observed_Pearl_TMAD/s) 
              ≈ 600 / 300 ≈ 2×  (optimistic cuBLAS-like)
              ≈ 838 / 300 ≈ 2.8× (rated hardware peak)
```

This **2–2.8× gap** is the **maximum** mining kernel can gain from GEMM acceleration alone — and only if transcript+hash overhead stays proportional (it does).

**Corollary:** Closing 100% of pure GEMM gap yields **≤2.8× total**, not 100×.

### 3.4 Why 100× is impossible (formal)

```
Let α = GEMM fraction of iteration time ≈ 0.88
Let β = non-GEMM fraction ≈ 0.12

If GEMM speed → ∞:
  speedup_total → 1/β ≈ 1/0.12 ≈ 8.3× upper bound (theoretical)

If GEMM speed → 100×:
  speedup_total = 1 / (β + α/100) ≈ 1 / (0.12 + 0.0088) ≈ 7.7×

Observed hardware limit (838/300): ≈ 2.8×
```

**100× GEMM improvement → at most ~8× iteration improvement** even in fantasy math — and hardware caps at **~2.8×** before memory/launch limits.

**Answer: NO on 100×.**

---

## 4. Alternative pearl-gemm Paths (Non-Production)

### 4.1 B200 `transcript_gemm_sm100.cu`

- **ISA:** `tcgen05.mma` + TMEM accumulators
- **Performance:** 840–877 TMAD/s reported (M=8192, N=32768, K=2048+)
- **5090 relevance:** **Reference only** — GeForce rejects tcgen05 PTX
- **Doc:** `performance optimizations/01-native-tcgen05-tmem-gemm.md`

Design B (B200): warp-specialized TMA producer + TMEM cumulative MMA + transcript readback after each K-slab.

### 4.2 `transcript_gemm_sm120_geforce.cu`

Experimental consumer Blackwell path:

- TMA warp-specialized loads (producer/consumer split)
- Still SM80 `mma.sync` for int8 (no tcgen05)
- Target: +20–40% over baseline consumer
- **Not default** in CAPI — must wire dispatch in `pearl_gemm_capi.cpp`

### 4.3 `phase0_tcgen05_sm120_probe.cu`

Compile probe confirming tcgen05 rejection on sm_120a. Documents dead-end for GeForce tcgen05 port.

### 4.4 Portable / SM75 / dp4a

| Path | Use case | 5090 speed |
|------|----------|------------|
| `portable/transcript_gemm_kernel.cu` | Parity vs H100 | Same as consumer |
| `portable/transcript_gemm_sm75.cu` | Turing/Ampere without sm_120 | Slower |
| `portable/transcript_gemm_dp4a.cu` | No tensor cores | Unusable for mining |

Parity script: `portable/run-parity.sh`

### 4.5 Legacy gemm/ noising stack

Separate from transcript consumer path — used in full noisy GEMM API (inference / vLLM miner style):

- `pearl_noisingA_kernel.h`, `pearl_noisingB_kernel.h`
- `collective_mainloop.hpp`, `collective_epilogue.hpp`
- Instantiations: `gemm/instantiations/gemm_R128_bf16_128x256x128_3stages_cluster1x1.cu`

PropMiner pure-miner mode bypasses full GEMM writeback; uses headless transcript path directly.

### 4.6 ROCm / SYCL

Reference ports for AMD and Intel — contain useful comments (e.g., BLAKE3 split kernel lesson in `rocm/host/pearl_gemm_capi_rocm.cpp`) but not RTX 5090 targets.

---

## 5. CUTLASS Reference Clone Analysis

### 5.1 Clone location

```
research/01-git-repos-mining/repos/cutlass/
```

Pinned at commit `e8ecfad` (2026). See `research/01-git-repos-mining/CLONE_MANIFEST.txt` for full clone metadata.

Also: `external_repos/cutlass/` (shallow, overlapping purpose).

### 5.2 Relevant CUTLASS paths for 5090 Pearl

| CUTLASS path | Relevance | Pearl applicability |
|--------------|-----------|---------------------|
| `examples/79_blackwell_geforce_gemm/` | SM120 GeForce TMA GEMM | **Load pipeline patterns** — FP4/BF16, not int8 transcript |
| `include/cutlass/gemm/collective/sm120_mma_tma.hpp` | TMA collective mainloop | Adapt for consumer TMA experiment (doc 02) |
| `include/cutlass/arch/mma_sm80.h` | SM80 int8 atom definitions | **Already used** via `SM80_16x8x32_S32S8S8S32_TN` |
| `include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp` | Hopper warp-spec | Algorithm template, wrong ISA |
| `tools/profiler/` | Benchmark harness | Shape sweeps — compare vs Pearl overhead |

### 5.3 SM120 int8 gap (doc 06)

CUTLASS 4.x provides:

- `SM120_16x8x32` for FP8/FP4/block-scaled types
- **No** distinct int8 Operation with different PTX from SM80

PropMiner doc 06 conclusion: **`PEARL_GEMM_SM120_NATIVE=1` is a readiness hook, ~0% gain from atom rename.**

### 5.4 What to steal from CUTLASS (realistic)

1. **TMA gmem→smem** copy patterns from SM120 collectives
2. **Cluster launch** APIs for `cluster_m` tuning
3. **Swizzle layouts** — validate against existing `Swizzle<3,4,3>`
4. **Profiler methodology** — roofline charts for int8 at K=128

### 5.5 What NOT to steal

1. FP4 block-scaled MMA (`mma.sync.block_scale`) — wrong dtype
2. B200 tcgen05 examples from SM100 tree — wrong hardware
3. Generic epilogue fusion — breaks transcript slot mapping

---

## 6. Realistic 1.5–2.5× Acceleration Plan

### 6.1 Tier 1: 1.15–1.25× (days, safe)

| Action | Gain | Source |
|--------|------|--------|
| Run `tune_prod_5090.sh` | +5–15% | doc 05 |
| Mine N=262144, batch 20 | +0–3% | doc 03 |
| Enable tune cache | Reproducibility | `run_mining.sh` |

### 6.2 Tier 2: 1.3–1.5× (weeks, moderate)

| Action | Gain | Source |
|--------|------|--------|
| TMA producer warp | +10–25% | doc 02, CUTLASS sm120_mma_tma |
| MIN_BLOCKS=2 if ncu confirms | +3–10% | doc 05 |
| cluster_m sweep 2→4 | +5–15% | `tune_cluster_sweep.sh` |

### 6.3 Tier 3: 1.5–2.5× (months, moderate-high)

| Action | Gain | Source |
|--------|------|--------|
| `transcript_gemm_sm120_geforce.cu` production dispatch | +20–40% | warp-spec + TMA |
| + core clock +150 MHz | +5–8% | hardware |
| + full knob + batch polish | cumulative | compositional |

**Compositional note:** Gains are sub-multiplicative. **1.5× is conservative; 2.5× requires Tier 3 near ceiling.**

### 6.4 Explicitly excluded from plan

| Action | Expected gain |
|--------|---------------|
| tcgen05 on 5090 | N/A |
| cuBLAS replacement | Breaks consensus |
| Remove transcript | Breaks consensus |
| INT4/FP8 | Breaks consensus |
| 100× fusion | Impossible |

---

## 7. Benchmark Methodology

### 7.1 PropMiner bench command

```bash
./build/propminer --bench 120 --rtx5090 --gpus 0
```

- N capped at 32768 (`kBenchMaxN`) for Salad 120 s window
- Reports DAF-normalized H/s and TMAD/s in logs

### 7.2 Self-test gate

```bash
./build/propminer --self-test
```

Required after any kernel change — byte-compares transcript layout.

### 7.3 CUTLASS profiler comparison (optional)

From clone:

```bash
cd research/01-git-repos-mining/repos/cutlass
# Build profiler per README; run int8 shapes M=8192,N=262144,K=128
# Compare achieved TOPS vs PropMiner ~300 TMAD/s
```

Expect CUTLASS/cuBLAS **~1.7–2.3× faster on raw GEMM** — establishes pure GEMM gap empirically.

---

## 8. Kernel Fusion Opportunities (Honest Assessment)

| Fusion | Expected gain | Risk |
|--------|---------------|------|
| noise_gen + noisingA + GEMM | +1–3% | Register pressure |
| Transcript in TMEM readback (B200 only) | +30–80% on B200 | N/A 5090 |
| BLAKE3 to separate kernel | **Negative** | Already tried on ROCm |
| Graph capture entire batch | +5–10% | **Done** |
| Persistent kernel across σ | +2–5% | Complexity |

---

## 9. Multi-Stream and Batch Interactions

PropMiner amortizes launch via:

- CUDA graphs per ping-pong half
- Batch 4–20 matmuls per graph
- 8-byte seed on dedicated stream

Ethash reference miners (`repos/ethminer`) use multi-stream nonce pipelining — PropMiner exceeds this with full workspace ping-pong (`external_repos/README.md`).

Further batch increase beyond 20 hits VRAM graph capture limits — diminishing returns.

---

## 10. Comparison Table: Paths vs Pure GEMM

| Implementation | ~TMAD/s (5090) | Transcript | PoW | Production |
|----------------|----------------|:----------:|:---:|:----------:|
| cuBLAS int8 (hypothetical) | 500–700 | No | No | Invalid |
| CUTLASS GemmUniversal int8 | 500–650 | No | No | Invalid |
| pearl consumer headless | **~300** | Yes | Yes | **Yes** |
| pearl sm120_geforce (TMA) | ~360–420 (est.) | Yes | Yes | Planned |
| pearl sm100 B200 | 840+ | Yes | Yes | Wrong GPU |
| portable dp4a | ~5–30 | Yes | Yes | Fallback |

---

## 11. Open Engineering Tasks

1. Wire `launch_transcript_gemm_sm120_geforce` into CAPI with runtime flag
2. Complete TMA load policy (`PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma`) — currently `#error` guarded
3. ncu roofline report: export `TensorActive` before/after TMA
4. CUTLASS profiler baseline at Pearl shapes for documented pure GEMM gap
5. Document byte-identity test matrix in CI

---

## 12. Summary

| Claim | Verdict |
|-------|---------|
| 100× from GEMM | **NO** — math caps ~8× fantasy, ~2.8× hardware |
| 10× from CUTLASS drop-in | **NO** — same mma.sync PTX |
| 1.5× from autotune + config | **YES** — high confidence |
| 2.5× from TMA + warp spec + clock | **YES** — stretch, achievable |
| CUTLASS clone useful? | **YES** — TMA/collective patterns, not drop-in GEMM |
| Production path | `consumer/transcript_gemm_kernel.cu` |

---

## Appendix A: File Index

| File | Role |
|------|------|
| `capi/pearl_gemm_capi.cpp` | Host API, dispatch |
| `capi/Makefile` | Arch flags, gencode |
| `consumer/transcript_gemm_kernel.cu` | Production kernel |
| `blackwell/transcript_gemm_sm120.cu` | SM120 wrapper |
| `blackwell/transcript_gemm_sm120_geforce.cu` | TMA experimental |
| `blackwell/transcript_gemm_sm100.cu` | B200 reference |
| `consumer/tma_tile_loader.cuh` | TMA scaffold |
| `gemm/pow_utils.hpp` | Transcript XOR |
| `portable/transcript_canonical.cuh` | Proof tile constants |
| `performance optimizations/01-native-tcgen05-tmem-gemm.md` | B200 port plan |
| `performance optimizations/06-sm120-native-cutlass-int8-atom.md` | Atom gap analysis |
| `external_repos/README.md` | tcgen05 GeForce finding |
| `research/01-git-repos-mining/repos/cutlass/` | CUTLASS reference clone |

---

## Appendix B: Glossary

| Term | Meaning |
|------|---------|
| **TMAD/s** | Trillion int8 multiply-adds per second |
| **Headless** | Fused in-kernel PoW without finalize pass |
| **IMMA** | Integer matrix multiply accumulate (mma.sync) |
| **UMMA** | Unified MMA (tcgen05, datacenter only) |
| **TMA** | Tensor Memory Accelerator (async bulk copy) |
| **Pure GEMM gap** | Ratio of cuBLAS-like peak to Pearl observed |

---

*GEMM acceleration for Pearl mining is bounded by protocol fusion overhead and GeForce ISA limits — not by lack of CUTLASS knowledge. Invest in TMA + occupancy; reject 100× narratives.*
