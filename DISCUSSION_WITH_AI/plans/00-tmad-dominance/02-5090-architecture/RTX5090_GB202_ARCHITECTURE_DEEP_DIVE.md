# RTX 5090 (GB202, Blackwell) Architecture Deep-Dive
## Mining Performance Exploitation Analysis

> **Date:** 2026-07-09
> **Source:** NVIDIA RTX Blackwell Architecture Whitepaper v1.1, SM-120 Microarchitecture Paper (zartbot, 21 test categories, 96.1% coverage), CUTLASS 4.5 docs, Colfax Research, NVIDIA CUDA Blackwell Tuning Guide, Toxigon, ASUS ROG
> **Target GPU:** GeForce RTX 5090 (GB202-300-A1, 170 SMs)

---

## 1. GB202 Architecture Basics

### 1.1 Core Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| **GPU Codename** | GB202-300-A1 | Consumer Blackwell |
| **Architecture** | Blackwell (5th Gen) | Compute Capability 12.0 (`sm_120`) |
| **SM Count** | **170** (of 192 full die) | 22 SMs disabled for binning/power |
| **Full Die SMs** | 192 | Reserved for RTX PRO 6000 Workstation |
| **CUDA Cores** | **21,760** (128 per SM) | Dual FP32/INT32 datapath |
| **Tensor Cores** | **680** (4 per SM, 5th Gen) | One per partition |
| **RT Cores** | 170 (4th Gen) | One per SM |
| **Texture Units** | 680 | 4 per SM |
| **ROPs** | 176 | 16 per active GPC (11 GPCs) |
| **Transistors** | 92.2 Billion | TSMC 4N (5nm-class) |
| **Die Size** | 750 mm² | |
| **FP64 Cores** | 384 (2 per SM) | 1:64 FP64:FP32 ratio — negligible |

### 1.2 Clock Speeds & Power

| Parameter | Value |
|-----------|-------|
| **Base Clock** | 2017 MHz |
| **Boost Clock** | 2407 MHz |
| **TGP (TDP)** | 575 W |
| **Recommended PSU** | 950 W |
| **FP32 Peak TFLOPS** | **104.8 TFLOPS** (non-Tensor) |

**Peak FP32 calculation:** `2 × 21760 × 2.407 GHz = 104.8 TFLOPS`
(The 2× comes from dual FP32/INT32 datapath per SM — 128 FP32 ops/clk per SM)

### 1.3 Tensor Core Generation (5th Gen Blackwell)

| Precision | Dense TOPS/TFLOPS | Sparse TOPS/TFLOPS | Sparsity |
|-----------|-------------------|-------------------|----------|
| **FP4 (e2m1)** | 1,676 TFLOPS | **3,352 TFLOPS** | 2:4 |
| **FP8 (e4m3)** | 838 TFLOPS | 1,676 TFLOPS | 2:4 |
| **FP8 (e5m2)** | 838 TFLOPS | 1,676 TFLOPS | 2:4 |
| **FP16** | 419 TFLOPS | 838 TFLOPS | 2:4 |
| **BF16** | 419 TFLOPS | 838 TFLOPS | 2:4 |
| **TF32** | 104.8 TFLOPS | 209.5 TFLOPS | 2:4 |
| **INT8** | **838 TOPS** | **1,676 TOPS** | 2:4 |
| **FP6** | ~838 TFLOPS | ~1,676 TFLOPS | ~2:4 |

**Critical insight for mining:** The INT8 rated TOPS is 838 dense / 1,676 sparse. The "3350 INT8 TOPS" figure you referenced is actually the **FP4 sparse** figure (3,352 TFLOPS). INT8 dense tops at 838 TOPS.

### 1.4 Memory System

| Parameter | Value |
|-----------|-------|
| **VRAM** | **32 GB GDDR7** |
| **Memory Interface** | **512-bit** |
| **Memory Speed** | **28 Gbps** |
| **Memory Bandwidth** | **1,792 GB/s** (1.792 TB/s) |
| **L2 Cache** | **96 MB** (of 128 MB full die) |
| **L2 per 32-bit controller** | 512 KB |
| **PCIe** | Gen 5 x16 (64 GB/s per direction) |

**Comparison across generations:**
| Metric | RTX 3090 | RTX 4090 | **RTX 5090** |
|--------|----------|----------|-------------|
| Memory | 24 GB GDDR6X | 24 GB GDDR6X | **32 GB GDDR7** |
| Bandwidth | 936 GB/s | 1,008 GB/s | **1,792 GB/s** |
| L2 | 6 MB | 72 MB | **96 MB** |
| FP32 | 35.6 TFLOPS | 82.6 TFLOPS | **104.8 TFLOPS** |

---

## 2. SM-120A (Blackwell Consumer SM) Features

### 2.0 SM-120 4-Partition Sub-Core Architecture (Critical New Finding)

**This is the fundamental execution topology of SM-120:**

SM-120 divides each SM into **4 independent sub-cores**, each with its own:
- Independent warp scheduler (handles 12 warps)
- Independent scoreboard (depth ≥12, not 6 like previous architectures)
- Independent operand collector (4-bank register file, zero bank conflict for ≤8 warps)
- Independent 64 KB register file (256 KB total per SM)

**Warp-to-sub-core binding is static:** `warp_id % 4` determines which sub-core handles each warp. This eliminates cross-sub-core scoreboard synchronization overhead.

**The 6 execution pipes per sub-core:**

| Pipe | Function | Instructions | Key Properties |
|------|----------|-------------|----------------|
| **P0** | Unified INT/FP32 | FFMA, IADD3, IMAD, LOP3, DP4A | Shared frontend, independent backend from INT32 |
| **V-pipe** | SIMD min/max | VIADD.U8x4, FMNMX, VIMNMX | Independent from P0, 1.0 inst/cyc |
| **P1** | FP64 | DFMA, DMNMX, DMMA (2 EU) | Fully independent, non-pipelined |
| **P2** | SFU/MUFU | RCP, SIN, COS, EX2, LG2 | 4 SFU units per SM, 0.11 inst/cyc |
| **P3** | LSU (Load/Store) | LDG, STG, LDS, STS, ATOM | Co-issues with P0 (free) |
| **P4** | Tensor Core | HMMA, QMMA, OMMA, IMMA | 29 cyc latency / 23 cyc throughput |

**Co-issue matrix (measured):**
- **FP64 ⊥ FP32 = 100% overlap** (P1 fully independent of P0)
- **SFU ⊥ FP32 = 100% overlap** (P2 fully independent of P0)
- **TC + 8× IADD3 = +1 cyc** (nearly free)
- **TC + 4× FFMA = +0 cyc** (completely free)
- **SFU + MEM = 0.55** (serialized — avoid MUFU.RCP during memory ops)

**This means:** The Tensor Core (P4) can issue completely in parallel with ALU operations (P0, P2, P3). Address computation (IADD3 on P0) doesn't compete with MMA. SFU operations don't compete with MMA. Only LSU (P3) and SFU (P2) partially overlap.

### 2.1 Unified INT32/FP32 Datapath

**Major architectural change from Ada (SM89) to Blackwell (SM120):**

| Aspect | Ada (SM89) | **Blackwell (SM120)** |
|--------|-----------|----------------------|
| INT32 units | 64 per SM (separate from FP32) | **Unified with FP32 — single 64-lane scalar ALU** |
| FP32 units | 128 per SM | **128 per SM (same)** |
| Throughput | 128 FP32 + 64 INT32 per cycle | **Either 128 FP32 OR 64 INT32 per cycle (not both)** |
| Mixed streams | No contention | **Serialization with ~4× latency penalty, but still 2× INT throughput** |

**Mining relevance:** Blackwell doubles INT32 throughput for many operations. This benefits:
- Address generation for TMA loads
- Quantization/dequantization math
- Noise computation integer operations
- The compiler auto-vectorizes — this is free performance

**Important caveat:** Mixed INT32+FP32 instruction streams serialize on P0, causing a ~4× latency penalty for the mixed instruction. However, even with this penalty, the **effective INT32 throughput is 2×** Ada for most workloads. Pure INT32 or pure FP32 streams hit full throughput.

### 2.2 CGGTY 5-Warp Threshold (Critical Occupancy Finding)

**From the SM-120 microarchitecture paper (verified via CuAssembler binary patching):**

| Warps/Sub-Core | cyc/iter (FFMA dep chain) | Latency Hiding |
|----------------|--------------------------|----------------|
| 1-4 warps | Linear growth (268 cyc at 4 warps) | **None** — single warp per sub-core |
| **5 warps** | **Jumps to 45 cyc (6× speedup)** | **Latency hiding activates** |
| 6-24 warps | Gradual improvement (45→47 cyc) | Full latency hiding |

**The 5-warp threshold is the single most important occupancy parameter for SM-120.**

**Mining implication:** Each sub-core needs ≥5 warps for latency hiding. With 4 sub-cores per SM, you need **≥20 warps per SM** for full latency hiding. This means:
- Minimum block size: 256 threads (8 warps) × multiple blocks per SM
- Target: **20-24 warps/SM active** for optimal performance
- Each block should launch ≥256 threads to ensure sufficient warp count

### 2.3 Yield Bit Discovery

**The yield bit in SM-120 instruction encoding has a critical behavior:**

When yield=1, **all 1-32 warp configurations execute at 336 cyc/iter** — the scheduler no longer interleaves. Yield **removes warps from the eligible set**, effectively destroying latency hiding.

**Mining implication:** Do NOT use yield in mining kernels. The yield bit is designed for cooperative multi-tasking contexts, not for compute-bound workloads.

### 2.4 Stall Counter Safety

**SM-120 does NOT use hardware interlocking:** If `stall_counter < instruction_latency`, the GPU reads stale register values with **no error trap or data corruption detection**.

| Stall Value | Effect |
|-------------|--------|
| 0 | Hardware scoreboard fallback (~25 cyc/inst, correct but extremely slow) |
| 1-3 | **Silent RAW data corruption** |
| 4 | Correct (equals IADD3 write-back latency) |
| stall + 1 | +1 cycle per payload instruction (strictly linear) |

**Mining implication:** Any SASS-level optimization must ensure stall counters ≥ instruction latency. Software prefetching is safer than manual stall counter manipulation.

### 2.5 The Critical Architecture Split: SM100 vs SM120

### 2.1 The Critical Architecture Split: SM100 vs SM120

**This is the single most important fact for kernel developers:**

| Feature | SM100 (B200 / Datacenter) | **SM120 (RTX 5090 / Consumer)** |
|---------|--------------------------|-------------------------------|
| **Tensor Core ISA** | `tcgen05.mma` (async, TMEM-based) | **`mma.sync.aligned` only** (Ampere-era) |
| **Tensor Memory (TMEM)** | 512 KB/SM dedicated | **Does NOT exist** |
| **WGMMA (warp-group MMA)** | Yes | **Does NOT exist** |
| **mma.sync.aligned** | Yes (in addition to tcgen05) | **Yes (sole tensor-core ISA)** |
| **Shared Memory** | 228 KB/SM (227 KB/block) | **99 KB/block** |
| **Max Warps/SM** | 64 | **48** |
| **Max Blocks/SM** | 32 | **24** |
| **Cluster Multicast** | Yes (up to 16 CTAs) | **No (1×1×1 only)** |
| **FP4/FP6 Block-Scale MMA** | tcgen05 async path | **mma.sync.block_scale (LLVM 22+)** |
| **TMA Multicast** | Yes | **No (~10,000× degradation)** |
| **setmaxnreg** | Yes (CUDA 13.3) | **Yes (CUDA 13.3)** |

**Bottom line:** SM120 uses the **same `mma.sync` instruction set as Ampere (SM80)**, NOT the advanced `tcgen05` path of datacenter Blackwell. Any kernel written for SM100 (DeepGEMM, CUTLASS SM100 collectives, WGMMA-based FlashAttention) **will fail on SM120**.

### 2.2 mma.sync Instruction Details on SM120

SM120 retains the full `mma.sync.aligned` family from SM80 with Blackwell additions:

**Supported tile shapes (m16n8kK family):**

| Multiplicand Type | Supported Shapes | K Dimension |
|-------------------|-----------------|-------------|
| FP16 `.f16` | m16n8k8, **m16n8k16** | 16 |
| BF16 `.bf16` | m16n8k8, **m16n8k16** | 16 |
| TF32 `.tf32` | m16n8k4, m16n8k8 | 4-8 |
| INT8 `.u8`/`.s8` | m8n8k16, m16n8k16, **m16n8k32** | 16-32 |
| FP8 `.e4m3`/`.e5m2` | **m16n8k32** | 32 |
| **FP4/FP6 MX** | **m16n8k32** + `.block_scale` | 32 |

**Key differences from SM80 (Ampere):**

1. **Block-Scale MMA (`mma.sync.block_scale`):** New in LLVM 22 / CUDA 13.1+ for SM120. Adds per-operand scale factor registers (ue8m0 format) as extra operands. Enables MX-format data types (FP4, FP6, FP8) with hardware-supported block scaling.

2. **MX Block-Scale overhead:** Zero overhead — scale metadata is encoded in the B-operand field, applied via shift after multiply. `scale_vec::1X` means one scale per 32 elements (6.25% overhead for FP4).

3. **Unified TC Pipeline:** All 12 non-FP64 precision formats (FP16, BF16, TF32, FP8, FP6, FP4, MX-FP4, NV-FP4, INT8 dense/sparse) share the **same 29 cyc latency / 23 cyc throughput** on the Tensor Core. Precision choice is a pure storage bandwidth decision, not a compute speed decision.

### 2.3 mma.sync.m16n8k32 vs SM80's Version

The `mma.sync.aligned.m16n8k32` instruction exists on both SM80 and SM120, but with key differences:

| Aspect | SM80 (Ampere) | SM120 (Blackwell Consumer) |
|--------|--------------|--------------------------|
| Available precisions | FP16 (k16), INT8 (k32) | FP16 (k16), INT8 (k32), FP8 (k32), FP4/FP6 (k32) |
| Block-scale | No | **Yes** (ue8m0 scale factors) |
| Latency / Throughput | ~33 cyc / ~26 cyc | **29 cyc / 23 cyc** (~12% faster) |
| Operand reuse | Supported | Supported (35-45% register read savings) |

**Can we use m16n16k64 for 2x throughput?** No. SM120 does not support m16n16k64. The `mma.sync` instruction family on SM120 is limited to the m16n8kK shape family. The larger tile shapes (m16n16k64) are only available via `tcgen05.mma` on SM100 datacenter GPUs, which SM120 lacks.

**Can we use mma.sync.m32n8k32 for mining?** No. SM120 does not support m32n8k32 either. The output tile is fixed at M×N = 16×8 for all `mma.sync` on SM120.

### 2.4 TMA (Tensor Memory Accelerator) on SM120

**TMA IS available on SM120** (unlike tcgen05/TMEM). This is a Hopper-inherited feature.

**What is TMA?** A hardware DMA engine that performs bulk global memory → shared memory transfers:
- Calculates affine addresses for multi-dimensional arrays in hardware
- One thread triggers a large copy and rejoins the warp (asynchronous)
- Bypasses registers entirely — reduces register pressure
- Enables warp-specialized producer-consumer pipelines

**TMA on SM120 — Capabilities and Limitations:**

| Feature | SM120 Support |
|---------|--------------|
| TMA Load (GMEM → SMEM) | **Yes** — 620 cyc for 1024B (DRAM→L2→SMEM) |
| TMA Store (SMEM → L2) | **Yes** — 33 cyc (buffered write, async writeback) |
| TMA Multicast | **No** — ~10,000× degradation, no NVLink/SM-broadcast |
| TMA im2col | **No** — illegal instruction |
| Cluster cp.async.bulk (sm→sm) | **No** — illegal instruction |

**Performance characteristics:**
- TMA Load is 19× slower than TMA Store (620 vs 33 cyc for 1024B)
- Need ≥32 outstanding TMA ops for 90% peak bandwidth
- Multi-CTA scaling is near-linear: 110 CTAs → 1483 GB/s (exceeds DRAM spec due to L2 hits)
- ~76 CTAs needed to saturate DRAM bandwidth

**TMA vs consumer kernel for mining:**
- TMA is **useful for prologue loads** (loading weight matrices from global to shared memory)
- TMA Store for epilogue is misleading — the 33 cyc is just "submit time", actual writeback takes 620 cyc
- For mining kernels, TMA can hide weight-loading latency behind computation in a multi-stage pipeline
- TMA does NOT bypass L2 — it writes to L2-buffered SMEM, which is fine for mining

### 2.5 Cluster Launch Control (CLC)

CLC is a Blackwell-exclusive feature for dynamic persistent tile scheduling:

**How it works:**
- A single scheduler warp in a cluster can `try_cancel` unlaunched thread blocks
- If cancellation succeeds, it "steals" their work tile coordinates
- Enables work stealing — idle blocks steal work from pending blocks
- Implemented via PTX: `clusterlaunchcontrol.try_cancel` + `clusterlaunchcontrol.query_cancel`

**CLC on SM120:**
- Supported on SM120 (consumer Blackwell)
- Cluster max size: 12 (= GPC SM count)
- No cluster multicast (1×1×1 only)
- Best implemented by a single elected scheduler thread per cluster

**Mining relevance:** CLC enables efficient persistent kernels where a small number of warps do all the work, dynamically stealing remaining tiles. This is valuable for mining workloads with variable-size problems.

### 2.6 New Warp-Level Primitives on SM120

| Feature | Available on SM120? | Details |
|---------|-------------------|---------|
| `setmaxnreg` (CUDA 13.3) | **Yes** | Dynamic register re-allocation: inc ~61 cyc/op, dec ~50 cyc/op |
| Warp shuffle (`__shfl_sync`) | Yes | Inherited from prior generations |
| Warp reduce (`__reduce_*_sync`) | Yes | Hardware-accelerated on sm_80+ |
| `tensormap.replace` | **Yes** (CUDA 13.3) | 26-45 cyc/op, dynamic TMA parameter modification |
| `L2::cache_hint` | **Yes** (CUDA 13.3) | UTMACCTL + cp.async.bulk, L2 cache policy control |
| SIMD 8×4 (VIADD.U8x4) | **Yes** (CUDA 13.3) | 4.0 cyc, V-pipe sub-unit |
| `cvt.pack` (I2IP) | **Yes** (CUDA 13.3) | 4 cyc, 2 ops/cyc @ ILP=8, INT8/INT4 packing |
| FP8 cvt (e4m3 pack/unpack) | **Yes** (CUDA 13.3) | Pack 8 cyc, unpack 4 cyc |
| MX narrow cvt | **Yes** (CUDA 13.3) | 8.22 cyc round-trip |

**setmaxnreg is particularly interesting for mining:** Combined with warp specialization, producer warps (data movement) can use minimal registers while consumer warps (MMA computation) dynamically expand their register allocation for MMA accumulators.

---

## 3. Tensor Core Utilization Analysis

### 3.1 INT8 Tensor Core Throughput

**Rated INT8 throughput on RTX 5090:**
- Dense: **838 TOPS**
- Sparse (2:4): **1,676 TOPS**

**Current noisy GEMM utilization:**
| TMAD/s | % of Dense INT8 (838) | % of Sparse INT8 (1,676) |
|--------|----------------------|-------------------------|
| 290 | **34.6%** | 17.3% |
| 600 | **71.6%** | 35.8% |
| 1,000 | **119.3%** (sparse) | 59.7% |
| 1,500 | Sparse 89.5% | 179% (impossible) |
| **2,000** | **Impossible** (dense max 838) | **119.3%** (sparse) |

**Correction:** The 3,350 INT8 TOPS figure you referenced is actually the **FP4 sparse** figure (3,352 TFLOPS). For INT8, the dense maximum is 838 TOPS and sparse is 1,676 TOPS.

### 3.2 Theoretical Max for Noisy GEMM

**Noisy GEMM workload characteristics:**
- Matrix A (weights): typically K×M, read once or with limited reuse
- Matrix B (inputs): typically K×N, read once or with limited reuse  
- Noise computation: adds additional matrix operations
- Accumulator: typically FP32

**Theoretical ceiling calculation (INT8 dense):**

```
INT8 dense peak = 838 TOPS (all 170 SMs at 2407 MHz)
INT8 sparse peak = 1,676 TOPS (with 2:4 sparsity)

For noisy GEMM specifically:
- Each mma.sync.m16n8k32 INT8 = 16×8×32 = 4,096 MACs = 8,192 ops
- At 23 cyc throughput per mma.sync per TC
- 4 TCs per SM → 4 × (2407 MHz / 23 cyc) × 8,192 ops = ~342 GFLOPS/SM (INT8)
- 170 SMs → ~58.1 TFLOPS (INT8 dense) — this is at full TC saturation

But wait — the published 838 TOPS already accounts for 170 SMs × 2407 MHz × 4 TCs × 128 FMA/clk/TC
So the ceiling is 838 TOPS dense, 1,676 TOPS sparse.
```

**Realistic noisy GEMM ceiling:**
- Noisy GEMM has lower arithmetic intensity than standard GEMM (more ops per byte, but less data reuse)
- The "noise" computation adds extra matrix multiplications that don't benefit from tile reuse
- **Realistic ceiling for noisy GEMM: ~60-70% of rated INT8 sparse = 1,000-1,200 TOPS**
- This assumes optimal tiling, TMA-accelerated loads, and warp-specialized pipelines

### 3.3 What 2,000+ TMAD/s Would Mean

At 2,000 TOPS:
- **119% of sparse INT8 rating** — impossible without sparsity exploitation tricks
- **238% of dense INT8 rating** — physically impossible
- This would require either: (a) using FP4 data types (3,352 TOPS sparse), or (b) the "TMAD/s" metric uses a different calculation than raw TOPS

**For FP4 (NVFP4):**
- Dense: 1,676 TFLOPS
- Sparse: **3,352 TFLOPS**
- 2,000 TOPS would be **59.7% of FP4 sparse** — very achievable

---

## 4. Memory Hierarchy Deep-Dive

### 4.1 GDDR7 Memory

| Parameter | RTX 5090 | RTX 4090 | RTX 3090 |
|-----------|----------|----------|----------|
| Type | **GDDR7** | GDDR6X | GDDR6X |
| Signaling | **PAM3** (3 levels, 1.5 bits/cycle) | PAM4 (4 levels, 2 bits/cycle) | NRZ (1 bit/cycle) |
| Speed | **28 Gbps** | 21 Gbps | 19.5 Gbps |
| Interface | **512-bit** | 384-bit | 384-bit |
| Bandwidth | **1,792 GB/s** | 1,008 GB/s | 936 GB/s |
| ECC | **Built-in SEC (always on, zero performance hit)** | Software inline ECC | Software inline ECC |
| EDR | Yes (Error Detection & Replay) | Yes | No |

**PAM3 advantage:** GDDR7 uses PAM3 (3 amplitude levels) instead of GDDR6X's PAM4 (4 levels). While PAM4 carries more bits per symbol (2 vs 1.5), PAM3 provides significantly better signal-to-noise ratio, enabling higher frequencies (28 Gbps vs 22.4 Gbps max for GDDR6X). Combined with doubled independent channels, GDDR7 achieves higher bandwidth with lower voltage.

### 4.2 L2 Cache

| Parameter | Value |
|-----------|-------|
| **RTX 5090 L2** | **96 MB** (of 128 MB full die) |
| Full GB202 L2 | 128 MB |
| L2 per 32-bit controller | 512 KB |
| L2 cache line | 128 bytes (4 × 32-byte sectors) |
| L2 hit latency (near) | ~79 cyc (~31 ns) |
| L2 hit latency (far) | ~180 cyc (~75 ns) |
| L2 bandwidth | Multi-TB/s aggregate |

**L2 is the inference lever:** 96 MB on Blackwell vs 72 MB on Ada vs 6 MB on Ampere. The large L2 holds KV caches and repeatedly-accessed data on-chip, amplifying effective bandwidth above the GDDR7 spec.

### 4.3 L1 / Shared Memory

| Parameter | Value |
|-----------|-------|
| **Physical L1+Shared** | **128 KB per SM** |
| **Max Shared Memory / SM** | **100 KB** |
| **Max Shared Memory / Block** | **99 KB** |
| **Shared Carveout Options** | 0 / 8 / 16 / 32 / 64 / 100 KB |
| **Shared Memory Banks** | 32 banks × 4 bytes/bank |
| **No-conflict latency** | 34 cyc |
| **Bank conflict penalty** | ~2 cyc/way (linear), worst 32-way = 113 cyc |

**Important:** The NVIDIA whitepaper says "128 KB of L1/Shared Memory" per SM, but the CUDA Programming Guide and CUTLASS specify that the **usable shared memory is 100 KB/SM (99 KB/block)** due to 1 KB CUDA reservation per block. The remaining 28 KB is L1 data cache.

### 4.4 Register File

| Parameter | Value |
|-----------|-------|
| **Per SM** | **256 KB** (65,536 × 32-bit registers) |
| **Per Partition** | 64 KB (16,384 × 32-bit) |
| **Max Registers / Thread** | 255 |
| **Allocation granularity** | 256 registers per warp (8 regs/thread × 32 lanes) |
| **RTX 5090 Total** | 43,520 KB (42.5 MB) across 170 SMs |

### 4.5 Memory Coalescing on Blackwell

- 32 threads of a warp issuing aligned, contiguous accesses coalesce into minimum 32-byte sector transactions
- 32 consecutive 4-byte words (128 B) → 4 sectors = 1 cache line
- Scattered/strided access inflates to worst case 32 separate transactions
- L1 and L2 use 128-byte lines split into 4 × 32-byte sectors, each filled independently
- On sm_120, `ld.global.cg` bypasses both L1 AND L2, going directly to DRAM (372 cyc) — different from Ampere/Hopper which only bypass L1

### 4.6 Memory Bandwidth vs Compute-Bound for Noisy GEMM

**Arithmetic intensity analysis:**

```
Noisy GEMM:
- Weight read: K × M × 1 byte (INT8) or K × M × 2 bytes (FP16)
- Input read: K × N × 1 byte or K × N × 2 bytes
- Output write: M × N × 4 bytes (FP32 accumulator)
- Noise computation: additional matrix ops (similar cost)
- Total ops: ~2 × (K×M×N + noise_ops) MACs

For typical mining parameters (K=256, M=128, N=256):
- Weight read: 256 × 128 × 1 = 32 KB
- Input read: 256 × 256 × 1 = 64 KB
- Output write: 128 × 256 × 4 = 128 KB
- Total memory: ~224 KB per tile
- Total ops: 2 × (256 × 128 × 256) = 16,777,216 ops (16.8M ops)
- Arithmetic intensity: 16.8M ops / 224 KB = ~75 ops/byte

Roofline ridge (RTX 5090): 838e9 ops/s ÷ 1.792e12 B/s ≈ 0.47 ops/byte
75 >> 0.47 → DEEPLY COMPUTE-BOUND

Even with noise adding 2x memory reads:
Intensity = 16.8M ops / 448 KB = ~38 ops/byte
38 >> 0.47 → STILL compute-bound
```

**Conclusion:** Noisy GEMM is **compute-bound**, not memory-bound. The RTX 5090's massive 1,792 GB/s bandwidth is overkill — the bottleneck is Tensor Core utilization, not memory bandwidth. Maximizing TC occupancy and instruction-level parallelism is the key optimization target.

---

## 5. TMA (Tensor Memory Accelerator) — Complete Analysis

### 5.1 What TMA Is

TMA is a **hardware DMA engine** for bulk data movement between global memory and shared memory/L1 cache:

1. **Address calculation in hardware:** TMA calculates affine addresses (`addr = width × base + offset`) for multi-dimensional arrays, eliminating register pressure from address computation
2. **Asynchronous execution:** One thread triggers a large copy and rejoins the warp; computation proceeds in parallel with data movement
3. **Barrier-based synchronization:** TMA uses `mbarrier` primitives for producer-consumer coordination
4. **Multi-dimensional tensor descriptors:** TMA understands tensor layouts (swizzle, interleaving, etc.)

### 5.2 TMA Warp-Specialized Pipelines

**Standard (no TMA) GEMM pipeline:**
```
Load weights → Load inputs → MMA → Store output  (sequential per tile)
```

**TMA-enabled warp-specialized pipeline:**
```
[Load Warps] → TMA trigger → [MMA Warps] → [Epilogue Warps]
     ↓                   ↓              ↓              ↓
  (async)        L2/SMEM buffer    (compute)     (store)
```

**Benefits:**
- Load warps and MMA warps operate fully asynchronously
- Multiple pipeline stages (depth ≥3) saturate the Tensor Core
- Register pressure reduced by offloading address computation to TMA hardware
- Achieves 868 TFLOPS on B200 (57.8% of cublasLt SoL) with 4-stage pipeline

### 5.3 GeForce v2 vs Consumer Kernel with TMA

| Aspect | Consumer Kernel (no TMA) | GeForce v2 Kernel (with TMA) |
|--------|------------------------|----------------------------|
| Data loading | `ld.global` + `cp.async` (register staging) | TMA bulk copy (hardware DMA) |
| Register pressure | High (address computation in regs) | Lower (TMA handles addresses) |
| Pipeline depth | Limited by warp coordination | 3+ stages possible |
| GMEM→SMEM latency | ~488-620 cyc (no overlap) | Overlapped with compute |
| Max practical occupancy | Lower (register pressure) | Higher (registers freed) |

### 5.4 Performance Gains from TMA

| Metric | Value |
|--------|-------|
| TMA Load 2D (1024B) | 620 cyc |
| TMA Load 1D (1024B, no swizzle) | 488 cyc |
| TMA Store 2D (1024B) | 33 cyc (submit only) |
| Outstanding ops for 90% peak | ≥32 |
| Multi-CTA aggregation (110 CTAs) | 1,483 GB/s (L2-hit boosted) |
| CTAs to saturate DRAM | ~76 |

### 5.5 Custom TMA Pipeline Design for Mining

**Recommended pipeline for noisy GEMM mining:**

```
Stage 1: TMA Load Warps (2-4 warps)
  - Trigger async TMA loads for weight tiles (A matrix)
  - Trigger async TMA loads for input tiles (B matrix)
  - Use mbarrier.arrive after each load completes

Stage 2: MMA Warps (12-20 warps)
  - Wait on mbarrier for A and B tiles ready
  - Execute mma.sync.m16n8k32 INT8 tiles
  - Accumulate in registers (setmaxnreg for dynamic expansion)
  - Execute noise computation MMA ops

Stage 3: Epilogue Warps (2-4 warps)
  - Wait on MMA completion
  - Convert FP32 accumulator → INT8 (quantize)
  - TMA Store output tiles (or use cp.async.bulk)

Pipeline depth: 3-4 stages, 32+ outstanding operations
Total warps: 16-28 per CTA (well within 48-warp SM limit)
```

**Key optimizations:**
- Use `tensormap.replace` (CUDA 13.3) for dynamic TMA parameter updates
- Use `L2::cache_hint` for strategic L2 caching of weight matrices
- Avoid TMA multicast (not supported on SM120)
- TMA Store is fast (33 cyc submit) but actual writeback is async — use explicit barrier

---

## 6. Occupancy Analysis

### 6.1 SM120 Occupancy Limits

| Parameter | Value |
|-----------|-------|
| **Max Warps per SM** | **48** (not 64 like SM100) |
| **Max Threads per SM** | **1,536** |
| **Max Thread Blocks per SM** | **24** (not 32 like SM100) |
| **Register File per SM** | 256 KB (65,536 × 32-bit) |
| **Max Registers per Thread** | 255 |
| **Shared Memory per SM** | 100 KB (99 KB/block) |

### 6.2 Occupancy Calculation for 170 SMs

```
Total capacity:
  170 SMs × 48 warps/SM = 8,160 warps total
  170 SMs × 1,536 threads/SM = 261,120 threads total
  170 SMs × 24 blocks/SM = 4,080 blocks total

With 65,536 CTAs grid:
  65,536 CTAs / 170 SMs = ~386 CTAs/SM (theoretical max)
  But max blocks/SM = 24, so:
  65,536 CTAs / 24 blocks/SM = 2,731 SMs needed (we only have 170)
  
  Actual: 65,536 CTAs / 24 blocks/SM = 2,731 → clamped to 170 SMs
  So: 65,536 / 170 = 386 blocks/SM → but limited to 24 blocks/SM
  Therefore: 170 × 24 = 4,080 blocks active, remainder queued
  
  Warps: 4,080 blocks × (block_size / 32) warps/block
  For 256-thread blocks (8 warps): 4,080 × 8 = 32,640 warps
  32,640 / 170 SMs = 192 warps/SM → clamped to 48 warps/SM
  So: 170 × 48 = 8,160 warps active at any time
```

### 6.3 Occupancy vs Registers Per Thread

| Registers / Thread | Warps/SM (cap 48) | Blocks/SM (cap 24) | Max Threads/Block |
|-------------------|-------------------|-------------------|-------------------|
| ≤32 | 48 (max) | 24 | 1024 |
| 40 | 48 | 20 | 512 |
| 48 | 42 | 16 | 512 |
| 64 | 32 | 16 | 256 |
| 96 | 21 | 10 | 256 |
| 128 | 16 | 8 | 256 |
| 168 | 12 | 6 | 128 |
| 255 | 8 | 4 | 128 |

**For mining kernels:** Keep register usage ≤32 per thread to maintain 48 warps/SM. Above 48 regs/thread, occupancy drops sharply.

### 6.4 What Limits Occupancy

1. **Register file (primary limiter for most kernels):** 65,536 registers/SM allocated at 256-register granularity per warp. High register pressure caps resident warps abruptly.

2. **Shared memory (secondary limiter):** 100 KB/SM shared. A block using 99 KB shared leaves room for ~1 block/SM.

3. **Warp cap (soft limiter):** 48 warps/SM hard cap. Below ~32 regs/thread, this dominates; above that, register file dominates.

4. **Block cap (soft limiter):** 24 blocks/SM. Rarely the bottleneck for mining kernels with typical block sizes.

### 6.5 Maximizing Occupancy Without Register Spilling

**Strategy:**

1. **Keep registers ≤32/thread:** This maintains 48 warps/SM and avoids spill to local memory (L1/L2/DRAM — catastrophic for performance)

2. **Use `setmaxnreg` (CUDA 13.3) for warp specialization:**
   - Producer warps (TMA loads): use 8-16 registers
   - Consumer warps (MMA): dynamically expand to 64-128 registers for accumulators
   - `setmaxnreg.inc` costs ~61 cyc/op, `setmaxnreg.dec` costs ~50 cyc/op

3. **Minimize shared memory usage:** Use ≤32 KB/block to allow multiple blocks per SM

4. **Persistent kernel pattern:** Use a single warp per CTA as scheduler, remaining warps do computation

5. **Target: 20+ warps/SM active per sub-core** (the SM-120 paper shows a 6× speedup jump at 5 warps/sub-core, saturating at 20-24 warps/sub-core)

---

## 7. Blackwell-Specific Optimizations

### 7.1 Features Unique to Blackwell (Not on Ampere/Ada)

| Feature | Ampere (SM80) | Ada (SM89) | **Blackwell SM120** | Mining Relevance |
|---------|--------------|-----------|-------------------|------------------|
| **FP4/FP6 Tensor Cores** | No | No | **Yes** | FP4 = 4× FP16 throughput, same power |
| **MX Block-Scale MMA** | No | No | **Yes** | Per-block scaling for FP4/FP6/FP8 |
| **TMA** | No | No | **Yes** | Async bulk loads, warp specialization |
| **CLC** | No | No | **Yes** | Dynamic persistent scheduling, work stealing |
| **setmaxnreg** | No | No | **Yes** (CUDA 13.3) | Dynamic register allocation |
| **INT32 2× throughput** | 1:1 FP32:INT32 | 1:1 | **2× for many INT ops** | Address generation, quantization |
| **FP6/FP4 dense MMA** | No | No | **Yes** | 1175 TFLOPS MX-FP4 |
| **NVFP4 block-scale** | No | No | **Yes** | Block-scaled FP4 with ue8m0 scales |
| **SER 2.0** | No | SER 1.0 | **SER 2.0** | Neural shading (limited mining relevance) |
| **AMP processor** | No | No | **Yes** (RISC-V) | GPU context scheduling (limited mining relevance) |
| **Accelerated frequency switching** | No | No | **Yes** (1000× faster) | DVFS sweet spot 1500-1800 MHz |
| **TMA multicast** | No | No | **No** (SM120) | — |
| **tcgen05 / TMEM** | No | No | **No** (SM120) | — |

### 7.2 FP8 Utilization for Mining

**FP8 on SM120:**
- Supported via `mma.sync.aligned.m16n8k32` with FP8 e4m3/e5m2 inputs
- Accumulator: FP32
- Throughput: 838 TFLOPS dense / 1,676 TFLOPS sparse (same as INT8)
- **Same 29 cyc latency / 23 cyc throughput as all other non-FP64 precisions**

**Mining relevance:** If your mining algorithm can tolerate FP8 precision (quantized weights), FP8 offers:
- Half the memory bandwidth of FP16 (1 byte vs 2 bytes per element)
- Same compute throughput as INT8
- Hardware block-scale support for dynamic range

**However:** For integer-based mining (like ProMiner's INT8 noisy GEMM), FP8 provides no advantage since INT8 is already the most bandwidth-efficient format that preserves the algorithm's requirements.

### 7.3 FP4 / NVFP4 — The Hidden Goldmine

**FP4 on SM120:**
- **1,676 TFLOPS dense / 3,352 TFLOPS sparse**
- 4× the throughput of FP16, 2× the throughput of FP8/INT8
- Uses `mma.sync.block_scale` with MX-format qualifiers
- Block scale: one ue8m0 scale factor per 32 elements (6.25% overhead)
- CUTLASS 4.5 achieves **975 TFLOPS** (80.7% efficiency) on SM120 with NVFP4

**Mining relevance:** This is the single biggest opportunity on SM120:
- If the mining algorithm can work with FP4 quantization (with block-scale correction), throughput could theoretically **double** from 838 INT8 TOPS to 1,676+ FP4 TFLOPS
- The block-scale correction (ue8m0) needs to be applied after accumulation
- NVFP4 uses per-block scaling (larger groups, coarser precision) vs MXFP4 (smaller groups, finer precision)
- **This is the key optimization to investigate for noisy GEMM mining**

### 7.4 New Instruction Scheduling Features

**SM120 execution pipeline (5 + V-pipe):**

| Pipe | Instructions | Throughput | Co-Issue |
|------|-------------|------------|----------|
| **P0** | FFMA, IADD3, IMAD, LOP3, DP4A | 0.585 inst/cyc (FFMA) | Co-issue with TC (free), SFU, MEM |
| **V-pipe** | VIADD.U8x4, FMNMX, VIMNMX | 1.0 inst/cyc | Independent from P0 |
| **P1** | DFMA, DMNMX, DMMA (FP64) | 0.5 inst/cyc | Fully independent |
| **P2** | RCP, SIN, COS, EX2, LG2 (SFU) | 0.11 inst/cyc | Co-issue with P0 (free) |
| **P3** | LDG, STG, LDS, STS, ATOM (LSU) | 0.5 inst/cyc | Co-issue with P0 (free) |
| **P4** | HMMA, QMMA, OMMA, IMMA (Tensor) | 1/23 inst/cyc | Co-issue with P0 (free) |

**Key co-issue properties:**
- **TC + 8× IADD3 = +1 cyc** (nearly free)
- **TC + 4× FFMA = +0 cyc** (completely free)
- **TC + DSMEM = fully parallel**
- **FP64 ⊥ FP32 = 100% overlap** (P1 fully independent of P0)
- **SFU ⊥ FP32 = 100% overlap** (P2 fully independent of P0)
- **SFU + MEM = 0.55** (serialized — avoid MUFU.RCP during memory ops)

**Mining optimization implication:** The Tensor Core pipe (P4) can issue completely in parallel with ALU operations (P0, P2, P3). This means:
- Address computation (IADD3 on P0) doesn't compete with MMA
- SFU operations (if any in noise computation) don't compete with MMA
- Only LSU (P3) and SFU (P2) partially overlap with each other

---

## 8. Theoretical Performance Analysis

### 8.1 100% INT8 Tensor Core Utilization

```
Peak INT8 dense: 838 TOPS (all 170 SMs @ 2407 MHz)
Peak INT8 sparse: 1,676 TOPS (with 2:4 sparsity)

Per-SM INT8 dense: 838 / 170 = 4.93 TOPS/SM
Per-SM INT8 sparse: 1,676 / 170 = 9.86 TOPS/SM
```

### 8.2 Realistic Max for Noisy GEMM

**Factors reducing theoretical max:**

1. **Tile efficiency:** Real GEMM tiling rarely achieves 100% tile utilization (edge effects, non-multiple dimensions)
2. **Data movement overhead:** Even compute-bound kernels need to load/store data
3. **Noise computation:** Extra matrix ops may not tile as efficiently
4. **Warp specialization overhead:** Pipeline bubbles, barrier synchronization
5. **Register pressure:** High register usage reduces occupancy

**Based on CUTLASS 4.5 NVFP4 results on SM120:**
- CUTLASS achieves 975 TFLOPS with NVFP4 (80.7% efficiency)
- This is on the same SM120 architecture
- NVFP4 has 4× the density of FP16, similar to INT8

**Realistic noisy GEMM ceiling:**
| Precision | Theoretical Max | Realistic Ceiling (70-85%) |
|-----------|----------------|--------------------------|
| **INT8 dense** | 838 TOPS | **587 - 712 TOPS** |
| **INT8 sparse** | 1,676 TOPS | **1,173 - 1,425 TOPS** |
| **FP4 dense** | 1,676 TFLOPS | **1,173 - 1,425 TFLOPS** |
| **FP4 sparse** | 3,352 TFLOPS | **2,346 - 2,849 TFLOPS** |

### 8.3 Memory Bandwidth Bottleneck Analysis

Even though noisy GEMM is compute-bound, let's verify:

```
For a 1,000 TOPS INT8 workload:
  Ops = 1,000 × 10^12
  Each INT8 MAC = 2 bytes input A + 2 bytes input B + 4 bytes output = 8 bytes (conservative)
  But with tiling and reuse: effective bytes per op ≈ 0.5-1 byte
  
  At 1,000 TOPS with 0.75 bytes/op: 750 GB/s memory demand
  RTX 5090 bandwidth: 1,792 GB/s
  Utilization: 750 / 1,792 = 41.9% ← NOT bandwidth-limited
  
  At 1,676 TOPS (sparse max) with 0.75 bytes/op: 1,257 GB/s
  Utilization: 1,257 / 1,792 = 70.1% ← approaching but still not saturated
```

**Conclusion:** Memory bandwidth is NOT the bottleneck for noisy GEMM on RTX 5090. The ceiling is set by **Tensor Core utilization**, not GDDR7 bandwidth.

### 8.4 The Actual Ceiling

**The actual ceiling for noisy GEMM on RTX 5090 is determined by:**

1. **Tensor Core throughput:** 838 TOPS dense INT8, 1,676 TOPS sparse
2. **Occupancy:** 48 warps/SM × 170 SMs = 8,160 warps max
3. **Pipeline efficiency:** TMA + warp specialization + CLC
4. **Instruction-level parallelism:** 5-pipe architecture with co-issue

**Realistic ceiling: 600-800 TOPS for INT8 noisy GEMM**
(This assumes optimal tiling, TMA-accelerated loads, warp specialization, and ~75% TC utilization)

**With FP4/NVFP4: 1,200-1,500 TFLOPS**
(This is the game-changer if the mining algorithm can work with FP4 quantization)

### 8.5 Power Efficiency Analysis

From the SM-120 microarchitecture paper:

| Precision | TFLOPS/W | Relative to FP32 |
|-----------|----------|-----------------|
| **MX-FP4** | **6.90 TFLOPS/W** | **28× FP32** |
| Sparse INT8 | 6.80 TFLOPS/W | 28× FP32 |
| Dense INT8 | 3.45 TFLOPS/W | 14× FP32 |
| FP8 | 3.32 TFLOPS/W | 14× FP32 |
| FP16 | 1.73 TFLOPS/W | 7× FP32 |
| FP32 | 0.24 TFLOPS/W | baseline |

**Power model:** `P = 80W (infrastructure) + 0.7-1.2W/SM`
- Infrastructure (clocks, L2, NoC, memory controllers): 80W fixed
- Per-SM compute: ~1.0 W/SM when TC saturated
- Total at full TC saturation: ~222W (63% of 575W TDP)

**DVFS sweet spot:** 1500-1800 MHz (best TFLOPS/W). Boost at 2407 MHz gives +73% TFLOPS but +127% power. For mining (power-efficient hashrate), consider undervolting to ~1800 MHz.

---

## 9. Actionable Recommendations for ProMiner

### 9.1 Priority 1: Investigate FP4/NVFP4 Mining

The single biggest opportunity is using FP4 Tensor Cores:
- **Potential: 2× throughput over INT8** (1,676 vs 838 TOPS dense)
- Requires: FP4 quantization of weight matrices + block-scale correction
- CUDA toolkit: 13.3+ with `sm_120a` target
- CUTLASS version: 4.5+ with GeForce Blackwell support
- PTX: `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`

### 9.2 Priority 2: TMA-Accelerated Warp-Specialized Pipeline

Implement a 3-4 stage pipeline:
- TMA Load Warps (async bulk loads from GDDR7 → SMEM)
- MMA Warps (compute with mma.sync.m16n8k32)
- Epilogue Warps (quantize + TMA Store)
- Pipeline depth ≥3, 32+ outstanding operations

### 9.3 Priority 3: Maximize Occupancy

- Keep registers ≤32/thread for baseline 48 warps/SM
- Use `setmaxnreg` for warp specialization (producer warps = few regs, consumer warps = expanded regs)
- Target 20+ active warps per sub-core (the 6× speedup threshold)
- Use persistent kernel pattern with CLC work stealing

### 9.4 Priority 4: Power-Optimized Clocks

- DVFS sweet spot: 1500-1800 MHz (best hashrate-per-watt)
- At boost (2407 MHz): +73% hashrate but +127% power
- For mining: consider locking clocks to ~1800 MHz for optimal efficiency
- Power model: 80W fixed + ~1W/SM compute

### 9.5 Priority 5: Leverage INT32 2× Throughput

Blackwell doubles INT32 throughput for many operations (unified INT32/FP32 datapath):
- Address generation for TMA loads
- Quantization/dequantization math
- Noise computation integer operations
- This is free performance — the compiler should auto-vectorize

### 9.6 What NOT to Worry About

- **tcgen05 / TMEM:** Not available on SM120 — don't try to use SM100 kernels
- **Cluster multicast:** Not supported on SM120
- **FP64:** Negligible (1:64 ratio) — not worth optimizing for
- **TMA multicast:** ~10,000× degradation on SM120 — avoid
- **Shared memory >99 KB:** Hard limit per block — design accordingly

---

## Appendix A: SM-120 Microarchitecture Summary

```
One SM (SM_120, Blackwell Consumer)
┌───────────────────────────────────────────────────────────────────────┐
│ Sub-Core 0    Sub-Core 1    Sub-Core 2    Sub-Core 3                  │
│ (warp%4==0)   (warp%4==1)   (warp%4==2)   (warp%4==3)                │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐              │
│ │P0:FFMA    │ │P0:FFMA    │ │P0:FFMA    │ │P0:FFMA    │              │
│ │P1:DFMA    │ │P1:DFMA    │ │P1:DFMA    │ │P1:DFMA    │              │
│ │P2:SFU     │ │P2:SFU     │ │P2:SFU     │ │P2:SFU     │              │
│ │P3:LSU     │ │P3:LSU     │ │P3:LSU     │ │P3:LSU     │              │
│ │P4:TC      │ │P4:TC      │ │P4:TC      │ │P4:TC      │              │
│ │V-pipe     │ │V-pipe     │ │V-pipe     │ │V-pipe     │              │
│ │64KB RF    │ │64KB RF    │ │64KB RF    │ │64KB RF    │              │
│ │1 Scheduler│ │1 Scheduler│ │1 Scheduler│ │1 Scheduler│              │
│ │≥12 SB     │ │≥12 SB     │ │≥12 SB     │ │≥12 SB     │              │
│ └───────────┘ └───────────┘ └───────────┘ └───────────┘              │
│ ── 128 KB unified L1/Shared Memory (shared by 4 sub-cores) ────────── │
│ ── 1 RT Core (4th Gen) ───────────────────────────────────────────── │
│ ── 4 Tensor Cores (5th Gen, one per sub-core) ────────────────────── │
│ ── 128 CUDA Cores (FP32) ─────────────────────────────────────────── │
│ ── 384 FP64 Cores (2 per SM, negligible) ─────────────────────────── │
└───────────────────────────────────────────────────────────────────────┘

RTX 5090: 170 SMs × (above) = 21,760 CUDA cores, 680 Tensor Cores
```

## Appendix B: Instruction Latency Reference (SM-120)

| Instruction | Latency (cyc) | Pipe | Throughput |
|------------|--------------|------|------------|
| IADD3 | 2.22 | P0 | 0.414 inst/cyc |
| FFMA | 4.22 | P0 | 0.585 inst/cyc |
| VIADD.U8x4 | 4.0 | V-pipe | 1.0 inst/cyc |
| IMAD | 4.22 | P0 | — |
| FMNMX | 5.19 | V-pipe | — |
| RCP.approx | 44.28 | P2 | — |
| mma.sync (all precisions) | 29 latency / 23 throughput | P4 | 1/23 inst/cyc |
| DMMA (FP64) | 175 latency / 449 throughput | P1 | ~0.002 inst/cyc |
| TMA Load (1024B) | 620 | — | — |
| TMA Store (1024B) | 33 (submit) | — | — |
| atomicAdd.f32 | 24 | — | — |
| atomicAdd.u32 | 45 | — | — |
| setmaxnreg.inc | ~61 | — | — |
| setmaxnreg.dec | ~50 | — | — |

## Appendix C: Key References

1. NVIDIA RTX Blackwell GPU Architecture Whitepaper v1.1 (official)
2. SM_120 Microarchitecture Paper — Cycle-Level Characterization (zartbot.github.io)
3. CUTLASS 4.5 Blackwell Architecture Docs
4. Zinc GPU Reference (zolotukhin/zinc)
5. Colfax Research: NVFP4 Blockscaled GEMM on SM12x
6. HWCooling: Blackwell GeForce RTX 5000 Architecture Analysis
7. NVIDIA CUDA Blackwell Tuning Guide
8. NVIDIA Cluster Launch Control Programming Guide
9. SM 120 CICC Reverse Engineering Reference (gh.evko.io)
10. Building FP4 Fused Attention on SM120 (florianmattana.com)
