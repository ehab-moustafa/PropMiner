# PeaMiner / Pearl Miner Competitor Analysis

> **Date:** 2026-07-09
> **Author:** PropMiner Research
> **Status:** Complete

---

## Executive Summary

**There is no repository called "PeaMiner" or "peacminer"** on GitHub. The user likely means **PeakMiner** (github.com/peakminer/peakminer), which is the highest-performing Pearl miner at **354.3 TH/s on RTX 5090** — but it is **closed-source** (only binaries distributed).

The actual competitors with **visible open-source CUDA implementations** are:

| Miner | 5090 Hashrate | Source | Key Differentiator |
|-------|--------------|--------|-------------------|
| **PeakMiner** | 354.3 TH/s | Closed-source | Highest hashrate, CUDA 12 bundled |
| **tw-pearl-miner** | 370 TH/s (claimed) | Closed-source | Highest claimed, 575W |
| **Alpha Miner** | 280-300 TH/s | Closed-source | 0% fee, AMD support |
| **Akoya Miner** | 250 TH/s | **Open-source** | 2GB VRAM, CUTLASS-based |
| **Open Pearl Miner** | ~30 TH/s (4050) | **Open-source** | CuTe Ada kernel, AMD beta |
| **Reference vLLM Miner** | N/A (consumer unsupported) | Open-source | Official, 70GB VRAM |

---

## 1. Source Code Location and Key Files

### 1.1 PeakMiner (github.com/peakminer/peakminer) — CLOSED SOURCE

- **Stars:** 157 | **Forks:** 0
- **Latest:** v1.0.13 (June 2026)
- **Build:** Pre-compiled binaries only (no source)
- **Initial commit:** Only README.md + LICENSE (138 lines)
- **Architecture profiles:** volta, turing, ampere, ada, h100, b200, blackwell
- **CUDA:** Bundled CUDA 12 runtime (no toolkit install needed)

**Key files (binary only):**
- `peakminer` (Linux binary)
- `peakminer.exe` (Windows binary)
- Docker image: `peakminer/peakminer:latest`

### 1.2 Akoya Miner (github.com/akoyapool/akoya-miner) — OPEN SOURCE

- **Stars:** 7 | **Forks:** 0
- **Language:** C# (host) + CUDA (kernels) + Rust (BLAKE3)
- **License:** MIT-spirit
- **Key directory:** `native/pearl-gemm/`
- **CUTLASS:** Vendored as git submodule at `native/pearl-gemm/third_party/cutlass`

**Key source files:**
- `native/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu` — RTX 50-series dedicated kernel
- `native/pearl-gemm/csrc/capi/` — C ABI wrapper
- `native/pearl-blake3/` — BLAKE3 keyed-merkle (Rust)
- `native/pearl-mining-capi/` — C ABI over BLAKE3
- `Akoya.CudaGemm` — P/Invoke into GEMM library
- `build.sh` — Auto-detects GPU, compiles architecture-specific kernels

**Build profiles:**
```bash
# RTX 5090 production profile (tested on RTX 5060 Ti):
PEARL_GEMM_ARCH=blackwell \
PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma \
PEARL_GEMM_BLACKWELL_MANUAL_IMMA=1 \
PEARL_GEMM_BLACKWELL_XOR_ACCUMS=4 \
./build.sh
```

### 1.3 Open Pearl Miner (github.com/neilquicks/open-pearl-miner) — OPEN SOURCE

- **Stars:** 7 | **Forks:** 4
- **Language:** CUDA (34.7%), C++ (33.2%), Python (29.2%)
- **Latest:** v1.9.1 (July 2026)
- **Dependency:** CUTLASS headers required at build time

**Key source files:**
- Fused CuTe Ampere/Ada kernel (sm_80+)
- Raw-PTX `mma.sync.m16n8k32` fallback
- Pascal DP4A dual-tile kernel

### 1.4 Alpha Miner (github.com/AlphaMine-Tech/alpha-miner) — CLOSED SOURCE

- **Latest:** v1.8.6 (July 2026)
- **5090:** 280-300 TH/s
- **H200:** ~600 TH/s
- **Features:** 0% dev fee, AMD support (beta), per-GPU difficulty
- **Backends:** volta/turing/ampere/ada/hopper/blackwell/blackwell-native/b200/b300/cmp170hx

### 1.5 tw-pearl-miner (github.com/egg5233/tw-pearl-miner) — CLOSED SOURCE

- **Stars:** 20 | **Latest:** v2.3.2 (July 2026)
- **5090:** 370 TH/s @ 575W (claimed — highest reported)
- **CUDA:** Ships bundled runtime (CUDA 12 and CUDA 13 builds)
- **Driver requirement:** >= 580.65 (Linux) / >= 580.88 (Windows)

---

## 2. Kernel Configuration Details

### 2.1 Akoya Miner — SM120 Blackwell Kernel

**File:** `native/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu`

**Key configuration:**
- **Architecture:** Dedicated SM120 implementation (separate from Ampere/Ada consumer kernel)
- **Load policy:** TMA (Tensor Memory Accelerator) — `PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma`
- **Manual IMMA:** Enabled — `PEARL_GEMM_BLACKWELL_MANUAL_IMMA=1`
- **XOR accumulators:** `PEARL_GEMM_BLACKWELL_XOR_ACCUMS=4`

**SM120-specific notes:**
- SM120 uses **register-based `mma.sync`**, NOT `tcgen05` / TMEM / WGMMA (unlike SM100 B200)
- GeForce RTX 50 series does **NOT support TMA multicast** — cluster shape must be 1x1x1
- Uses `compute_120f` (FP8 MMA support) instead of `compute_120`
- 5th gen Tensor cores: 3,354 TOPS FP4 dense, 6,708 TOPS FP4 sparse

**From CUTLASS SM120 examples:**
- Uses `OpClassBlockScaledTensorOp` (block-scaled tensor ops)
- Warp-specialized persistent kernel design
- SW-controlled dynamic scheduler based on cluster launch control
- Epilogue optimization
- Thread config: 256 MMA threads + 32 DMA = 288 total threads

### 2.2 Open Pearl Miner — CuTe Ada Kernel (sm_89)

**Tile configuration:**
- **CTA tile:** 128×256
- **TiledMMA:** `<8,1,1>` — each warp owns a 16-row band
- **R-block fold:** Direct in-register `shfl_xor` (no shared memory round-trip)
- **cp.async:** Multistage `cp.async.cg` pipeline
- **ldmatrix:** For shared-to-register fragment loading
- **Engaged automatically** for mining shape (region multiple of 128×256, R multiple of 32)

**Fallback kernels:**
- Raw-PTX `mma.sync.m16n8k32` with 16 N-fragments per warp
- A fragment loaded once, reused ×16
- Cascade of 64×256 / 64×128 tilings for uncovered shapes

**Pascal DP4A:**
- Dual-tile (NT=2) — each warp computes two adjacent 16×16 tiles
- int2 global→shared staging
- Shared-memory traffic per FLOP cut ~1.5×

### 2.3 Alpha Miner — Kernel Configuration

**Environment variables for tuning:**
```bash
# Suggested d (difficulty) values:
# 3060Ti–3070 → 4096–16384
# 3090/4080 → 65536
# 4090/H100/H200 → 131072–524288
# 5090 → 262144+
```

**Backends:**
- `blackwell-native` — dedicated Blackwell path
- `blackwell-b200`/`b300` — datacenter Blackwell
- `cmp170hx` — with dp2a/scalar

### 2.4 PeakMiner — Kernel Configuration (from docs)

**Architecture profiles:**
| Compute Cap | Profile | Cards |
|-------------|---------|-------|
| sm_70 | volta | V100 |
| sm_75 | turing | RTX 20xx |
| sm_80/86 | ampere | RTX 30xx |
| sm_89 | ada | RTX 40xx |
| sm_90 | h100 | H100/H200 |
| sm_100 | b200 | B200/B300 |
| sm_120 | blackwell | RTX 50xx |

**Performance (default OC):**
| GPU | Hashrate | Efficiency |
|-----|----------|------------|
| RTX 5090 | 354.3 TH/s | 617 GH/W |
| RTX 4090 | 288.8 TH/s | 643 GH/W |
| RTX 5080 | 206.1 TH/s | 574 GH/W |
| RTX 4080 | 179.1 TH/s | 563 GH/W |
| RTX 5070 Ti | 174.0 TH/s | 581 GH/W |

---

## 3. Work Scheduling Strategy

### 3.1 Open Pearl Miner
- **Multi-GPU:** Auto-detects every GPU, runs one worker per card
- **Worker model:** Each GPU runs as an independent pinned process
- **Synchronization:** Blocking-sync CUDA (near-linear scaling on 4-8 GPU rigs)
- **Region size:** `--region 4096` (sub-output search size, default)
- **Pool selection:** Auto by GPU class — tensor-core cards use GPU difficulty pool, Pascal uses CPU difficulty pool

### 3.2 Alpha Miner
- **Per-GPU difficulty:** `--password 'x;d=262144,131072,4096,...'` maps 1:1 to devices
- **TLS support:** `--tls`, `--tls-ca-file`, `--tls-insecure`
- **Schedule options:** `--schedule-spin`, `--schedule-yield`

### 3.3 Akoya Miner
- **VRAM footprint:** ~2 GB (vs 70 GB for reference miner)
- **Built on:** .NET Native AOT + P/Invoke to native libraries
- **Two native libraries:** `libpearl_gemm_capi.so` (GPU kernels) + `libpearl_mining_capi.so` (host BLAKE3)

### 3.4 PeakMiner
- **HTTP stats API:** Per-GPU hashrate, temperature, fan, shares, uptime
- **Per-GPU OC:** Core/memory clocks, power limit, temp pause/resume
- **Fan control:** Fixed or closed-loop to target temperature
- **Failover pools:** Comma-separated URLs, auto TLS/SSL detection

---

## 4. Memory Management Approach

### 4.1 Akoya Miner — TMA-Based Transfer

**SM120 Blackwell kernel:**
- Uses **TMA (Tensor Memory Accelerator)** for GMEM→SMEM transfers
- `CUtensorMap` objects passed as `__grid_constant__` kernel arguments (read-only, no register broadcast)
- Single thread (warp 0, `elect_sync()`) issues TMA load
- Each TMA load signals one `mbarrier`, all warps wait before computing
- **No TMA multicast on consumer Blackwell** — cluster shape restricted to 1x1x1

**Warp-specialized TMA (from CUTLASS SM120 patterns):**
- Dedicated TMA warp removes sync coupling between loading and compute
- Thread block: `(NUM_WARP_M * NUM_WARP_N + 1) * WARP_SIZE` threads
- Last warp is the TMA warp (warp_id == NUM_WARP_M * NUM_WARP_N)

**Pipeline stages:**
- `PipelineTmaAsync` (not `PipelineTmaUmma`) for NVFP4 dense GEMM
- Consumer release/wait on mbarrier states
- `fence_view_async_shared()` + `sync_warp()` before producer's next async-proxy write

### 4.2 Open Pearl Miner — cp.async Pipeline

**Ada kernel:**
- **Multistage `cp.async.cg`** — circular buffer pipeline from global to shared memory
- **ldmatrix** — shared-to-register fragment loading (warp-wide, no bank conflicts with permuted layout)
- **N-stage pipeline** — preloads N-1 stages before main loop
- **Permuted shared memory layout** — avoids bank conflicts (CUTLASS standard)

**Register-level optimization:**
- A fragment loaded once, reused ×16 (16 N-fragments per warp)
- R-block fold via direct in-register `shfl_xor` (no shared memory round-trip)

### 4.3 PeakMiner
- **Low VRAM:** Emphasized in marketing ("low VRAM usage")
- **Bundled CUDA 12:** No toolkit install needed — ships runtime inside binary

---

## 5. CPU-GPU Overlap Techniques

### 5.1 Akoya Miner
- **Warp specialization:** Dedicated TMA loader warp overlaps memory transfers with compute
- **mbarrier-based synchronization:** Producer/consumer model with async proxy
- **Consumer release/wait pattern:**
  ```
  fence_view_async_shared();
  sync_warp();
  ab_pipeline.consumer_release(ab_read_state);
  ab_read_state.advance();
  peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_read_state);
  ```

### 5.2 Open Pearl Miner
- **Blocking-sync CUDA:** Each GPU worker uses blocking sync (simpler but less overlap)
- **Independent pinned processes:** Near-linear scaling without complex stream management
- **cp.async.cg pipeline:** N-stage async copy hides global memory latency

### 5.3 Alpha Miner
- **Schedule options:** `--schedule-spin` (aggressive) vs `--schedule-yield` (power-efficient)
- **Per-GPU difficulty tuning:** Higher d values reduce pool communication overhead

### 5.4 PropMiner Comparison
PropMiner uses **triple buffering** with CPU-GPU overlap (see `docs/TRIPLE_BUFFER_ONE_PAGER.md` and `docs/DEPLOY_CHECKLIST_CPU_GPU_OVERLAP.md`). Most competitors use simpler models:
- Open Pearl Miner: blocking sync (no overlap)
- Akoya Miner: warp specialization (kernel-level overlap only)
- PeakMiner: unknown (closed-source)

---

## 6. 5090-Specific Optimizations

### 6.1 RTX 5090 Hardware Specs (GB202-300-A1)
| Spec | Value |
|------|-------|
| Compute capability | sm_120a |
| SMs | 170 |
| Tensor cores | 680 (5th gen) |
| L1/Shared per SM | 128 KB |
| L2 cache | 96 MB (no L3 — datacenter only) |
| Memory | 32 GB GDDR7, 512-bit |
| Bandwidth | 1,792 GB/s |
| FP4 dense/sparse | 3,354 / 6,708 TOPS |
| FP8 dense/sparse | 1,677 / 3,354 TFLOPS |
| FP16 dense/sparse | 838 / 1,677 TFLOPS |

### 6.2 SM120 Key Differences from SM100 (B200)

| Feature | SM120 (RTX 50xx) | SM100 (B200) |
|---------|------------------|--------------|
| MMA instruction | `mma.sync` (warp-level) | `tcgen05.mma` / WGMMA (warp-group) |
| TMEM | Not available | Available (4 MB per SM) |
| TMA multicast | NOT supported | Supported |
| Cluster shape | 1×1×1 only | Configurable |
| FP8 MMA | Supported (kind::f8f6f4) | Supported |
| Build toolkit | CUDA 12.8+ | CUDA 12.4+ |

### 6.3 PeakMiner 5090 Tuning
- **Dedicated blackwell profile:** `sm_120` with custom kernel
- **Default OC:** Optimized clocks (354.3 TH/s @ ~575W)
- **Per-GPU power limits:** Configurable via `--gpu-powerN`
- **Per-GPU temp stop:** Auto-pause at threshold, resume when cooled
- **Fan control:** Fixed or closed-loop to target temperature
- **Efficiency:** 617 GH/W (best-in-class)

### 6.4 Akoya Miner 5090 Tuning
- **Production flags:**
  ```bash
  PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma
  PEARL_GEMM_BLACKWELL_MANUAL_IMMA=1
  PEARL_GEMM_BLACKWELL_XOR_ACCUMS=4
  ```
- **Hashrate:** 250 TH/s (vs 354 TH/s PeakMiner — ~70% of PeakMiner)
- **VRAM:** ~2 GB (vs 70 GB reference)

### 6.5 tw-pearl-miner 5090 Tuning
- **575W @ 370 TH/s** — highest reported hashrate
- **505W @ 330 TH/s** — more power-efficient mode
- **CUDA 13 build** required (driver >= 580.88)

---

## 7. Comparison Points with PropMiner

### 7.1 Kernel Design

| Aspect | PropMiner | PeakMiner | Akoya | Open Pearl |
|--------|-----------|-----------|-------|------------|
| **K-block** | ? | ? | ? | Mining-specific (128×256) |
| **Stage count** | ? | ? | N-stage cp.async | Multistage (Ada) |
| **MMA choice** | ? | ? | CUTLASS BlockScaledTensorOp | CuTe fused int8 |
| **Launch bounds** | ? | ? | 288 threads (256+32) | 128×256 CTA |
| **Occupancy** | ? | ? | Warp-specialized | Blocking sync |

### 7.2 Work Scheduling

| Aspect | PropMiner | PeakMiner | Akoya | Open Pearl |
|--------|-----------|-----------|-------|------------|
| **Batch size** | ? | ? | ? | --region 4096 |
| **Launch frequency** | ? | ? | ? | Per-region |
| **Multi-GPU** | ? | Per-GPU OC | P/Invoke | Independent processes |
| **Overlap model** | Triple buffering | Unknown | Warp spec | Blocking sync |

### 7.3 Memory Management

| Aspect | PropMiner | PeakMiner | Akoya | Open Pearl |
|--------|-----------|-----------|-------|------------|
| **VRAM usage** | ? | Low | ~2 GB | CUTLASS-dependent |
| **B matrix** | ? | ? | TMA-loaded | cp.async pipeline |
| **Transfer** | Triple buffer | ? | TMA | cp.async.cg |

### 7.4 Algorithm Approach

**All miners compute the same PearlHash:**
- Low-rank-noised integer GEMM
- Each candidate = tile of A · Bᵀ
- Hashed and checked against difficulty target
- BLAKE3 keyed-merkle commitments on host

**Key differences:**
- **PeakMiner:** Highest hashrate (354 TH/s), but closed-source — cannot audit or improve
- **Akoya:** Open-source, CUTLASS-based, 250 TH/s — good reference for SM120 patterns
- **Open Pearl:** CuTe Ada kernel, bit-exact with DP4A, but lower hashrate (~30 TH/s on 4050)
- **tw-pearl-miner:** Highest claimed (370 TH/s), closed-source
- **Alpha Miner:** 280-300 TH/s, 0% fee, AMD support

### 7.5 Recommendations for PropMiner

1. **Study Akoya's SM120 kernel** (`transcript_gemm_sm120.cu`) — it's the best open-source reference for RTX 5090 Pearl mining
2. **TMA usage:** Akoya uses TMA for GMEM→SMEM with warp specialization — consider if PropMiner's triple buffering can be enhanced with TMA
3. **XOR accumulators:** Akoya uses `PEARL_GEMM_BLACKWELL_XOR_ACCUMS=4` — investigate if PropMiner's accumulator handling can match this
4. **Manual IMMA:** `PEARL_GEMM_BLACKWELL_MANUAL_IMMA=1` — manual inline PTX MMA may outperform auto-generated kernels
5. **Launch bounds:** PeakMiner's 354 TH/s vs Akoya's 250 TH/s suggests launch configuration and occupancy tuning matter significantly
6. **Region size tuning:** Alpha Miner's `d` values (262144+ for 5090) suggest large batch sizes are optimal for consumer Blackwell
7. **Power efficiency:** PeakMiner achieves 617 GH/W — investigate if PropMiner's thermal management can match this

---

## 8. Key Findings

### What We Know About PeakMiner (354 TH/s on 5090)
- **Closed-source** — cannot analyze implementation directly
- Uses **dedicated blackwell profile** (sm_120)
- **Low VRAM** usage (marketing emphasis)
- **Bundled CUDA 12** runtime
- **Per-GPU OC** and **thermal management** built-in
- **HTTP stats API** for monitoring
- **Fan control** (fixed + closed-loop)
- Achieves **617 GH/W** efficiency

### What We Know About Akoya Miner (250 TH/s on 5090)
- **Open-source** — full CUDA kernel source available
- Uses **CUTLASS** with `OpClassBlockScaledTensorOp`
- **TMA-based** transfers with warp specialization
- **Manual IMMA** with XOR accumulator fusion
- **288 threads** per block (256 MMA + 32 DMA)
- **~2 GB VRAM** footprint
- **Native AOT** (.NET) host with P/Invoke

### What We Know About Open Pearl Miner
- **Fused CuTe Ada kernel** — 128×256 CTA, <8,1,1> TiledMMA
- **In-mainloop transcript fold** — no separate post-processing
- **Multistage cp.async.cg** pipeline
- **ldmatrix** for shared-to-register
- **Raw-PTX fallback** for uncovered shapes
- **Bit-exact** with Pascal DP4A path

### The Hashrate Gap
The hashrate gap between miners is significant:
- **tw-pearl-miner:** 370 TH/s (highest claimed)
- **PeakMiner:** 354.3 TH/s (highest confirmed)
- **Alpha Miner:** 280-300 TH/s
- **Akoya Miner:** 250 TH/s (open-source leader)

This 1.48× gap between Akoya (250) and PeakMiner (354) suggests that **kernel-level optimizations** (launch bounds, occupancy, register pressure, instruction scheduling) account for the majority of the difference — not just algorithmic improvements.

---

## 9. References

1. [peakminer/peakminer](https://github.com/peakminer/peakminer) — PeakMiner (closed-source)
2. [akoyapool/akoya-miner](https://github.com/akoyapool/akoya-miner) — Akoya Miner (open-source)
3. [neilquicks/open-pearl-miner](https://github.com/neilquicks/open-pearl-miner) — Open Pearl Miner (open-source)
4. [AlphaMine-Tech/alpha-miner](https://github.com/AlphaMine-Tech/alpha-miner) — Alpha Miner (closed-source)
5. [egg5233/tw-pearl-miner](https://github.com/egg5233/tw-pearl-miner) — tw-pearl-miner (closed-source)
6. [pearl-research-labs/pearl](https://github.com/pearl-research-labs/pearl) — Official Pearl protocol (vLLM miner)
7. [NVIDIA CUTLASS SM120 Example](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu)
8. [Dao-AILab/quack gemm_sm120.py](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm120.py)
9. [flashinfer dense_blockscaled_gemm_sm120_b12x.py](https://github.com/flashinfer-ai/flashinfer/blob/2b150b39/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py)
10. [SM120 Optimization Notes](https://github.com/kekzl/imp/blob/main/docs/sm120.md)
