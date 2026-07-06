# RTX 5090 (GB202) Research Report — Pearl-Style GEMM Mining

**PropMiner Research · Topic 02**  
**Date:** July 2026  
**Scope:** Hardware characterization, mining-relevant performance levers, and realistic optimization ceilings for sustained INT8 tensor-core workloads on NVIDIA GeForce RTX 5090.

---

## Executive Summary

The RTX 5090 is a **compute-bound** platform for Pearl NoisyGEMM mining at production shapes (M=8192, K=128, N≤262144). Its 170 enabled SMs, 2.41 GHz boost clock, and **838 dense INT8 TOPS** (NVIDIA-rated) define a hard ceiling near **~800–850 TMAD/s** if every tensor pipe cycle were perfectly utilized. PropMiner's measured consumer-kernel baseline is **~300 TMAD/s** on RunPod headless 5090 hardware — roughly **36% of rated INT8 tensor peak**.

The largest single misconception in current planning docs is treating consumer `sm_120` like datacenter `sm_100`: **GB202 GeForce does not expose `tcgen05.mma`, Tensor Memory (TMEM), or UMMA**. Mining kernels must use **warp-level `mma.sync` (SM80-class ISA)** with a **99 KB/block shared-memory ceiling**, not B200-style tcgen05 pipelines. Realistic **2×–3×** gains over the ~300 TMAD/s baseline come from occupancy tuning, TMA-assisted loads, cluster scheduling, host/GPU overlap, and modest core-clock uplift — not from porting `transcript_gemm_sm100.cu` wholesale.

---

## 1. Full RTX 5090 (GB202) Hardware Specifications

### 1.1 Silicon and Enabled Configuration

GB202 is the flagship Blackwell consumer die (TSMC 4N, ~750 mm², ~92.2B transistors). The RTX 5090 enables a **partial die**:

| Resource | Full GB202 | RTX 5090 (enabled) | Disabled on 5090 |
|----------|-----------|-------------------|------------------|
| GPCs | 12 | **11** | 1 GPC |
| TPCs | 96 | **85** | 11 TPCs |
| SMs | 192 | **170** | 22 SMs (~11.5%) |
| CUDA cores | 24,576 | **21,760** | 2,816 |
| Tensor cores | 768 | **680** (5th gen) | 88 |
| RT cores | 192 | **170** (4th gen) | 22 |
| Texture units | 768 | **680** | 88 |
| ROPs | 192 | **176** | 16 |
| L2 cache | 128 MB | **96 MB** | 32 MB (~25%) |
| Memory interface | 512-bit | **512-bit** (all MCs active) | — |

Sources: [NVIDIA RTX Blackwell Architecture Whitepaper (v1.5)](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), [HotHardware RTX 5090 review](https://hothardware.com/reviews/nvidia-geforce-rtx-5090-review), [TechPowerUp GPU DB](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216).

### 1.2 Clocks and Rated Throughput

| Parameter | RTX 5090 | RTX 4090 (reference) |
|-----------|----------|---------------------|
| Base clock | 2,017 MHz | 2,235 MHz |
| Boost clock | **2,407 MHz** | 2,520 MHz |
| Memory data rate | **28 Gbps GDDR7** | 21 Gbps GDDR6X |
| Memory clock (effective) | 1,750 MHz × 2 | — |
| FP32 (non-tensor) | 104.8 TFLOPS | 82.6 TFLOPS |
| FP16/BF16 (non-tensor) | 104.8 TFLOPS | 82.6 TFLOPS |
| **INT8 tensor (dense/sparse)** | **838 / 1,676 TOPS** | 660.6 / 1,321.2 TOPS |
| FP8 tensor (FP16 acc, dense/sparse) | 838 / 1,676 TFLOPS | 660.6 / 1,321.2 TFLOPS |
| FP4 tensor (FP32 acc) | 1,676 / 3,352 TOPS | N/A |
| RT throughput | 317.5 TFLOPS | 191 TFLOPS |
| AI TOPS (NVIDIA marketing) | 3,352 | 1,321 |

Sources: [NVIDIA official RTX 5090 product page](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/), [NVIDIA whitepaper Table 1](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

### 1.3 Memory Subsystem

| Parameter | Value |
|-----------|-------|
| VRAM capacity | **32 GB GDDR7** |
| Bus width | **512-bit** |
| Peak bandwidth | **1,792 GB/s** (1.79 TB/s) |
| vs RTX 4090 bandwidth | +78% (1,008 → 1,792 GB/s) |
| L1 + shared per SM | 128 KB unified |
| L1/shared aggregate (170 SMs) | 21,760 KB (~21.25 MB) |
| L2 cache (enabled) | **96 MB** |
| Register file per SM | 256 KB |
| Register file aggregate | 43,520 KB (~42.5 MB) |

The 512-bit bus is a deliberate Blackwell design choice: a 384-bit bus with 28 Gbps G7 would yield ~1,536 GB/s — insufficient to feed 33% more CUDA cores without memory starvation in bandwidth-sensitive workloads. Pearl mining at K=128 is **not** in that class (see §5).

Sources: [NVIDIA whitepaper §1.2](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf), [VideoCardz GB202 analysis](https://videocardz.net/nvidia-geforce-rtx-5090).

### 1.4 Power, PCIe, and Platform

| Parameter | Value |
|-----------|-------|
| TGP (Total Graphics Power) | **575 W** |
| NVIDIA recommended PSU | 1,000 W |
| Power connector | 1× 12V-2×6 (16-pin) |
| Alternative cabling | 3× or 4× 8-pin via adapter |
| Max GPU temperature (spec) | 90 °C |
| PCIe generation | **Gen 5 ×16** |
| Resizable BAR | Supported |
| Display | 3× DP 2.1b UHBR20, 1× HDMI 2.1b |

PCIe Gen 5 matters little for mining: Pearl's steady-state bottleneck is on-die tensor execution, not H2D seed uploads (8-byte nonce counter per poll). Resizable BAR helps large VRAM mappings for resident-B state but is secondary to kernel efficiency.

Sources: [NVIDIA RTX 5090 specs page](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/), [NVIDIA whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

### 1.5 SM-Level Microarchitecture (Mining-Relevant)

Each SM on GB202 contains:

- 128 FP32 CUDA cores
- 4× 5th-generation tensor cores (marketed as Blackwell tensor cores)
- 4th-generation RT core (irrelevant for mining)
- 128 KB L1/data cache/shared memory (configurable split)
- 256 KB register file
- 2× FP64 units (new vs prior GeForce gens; unused in INT8 GEMM)

**Critical ISA distinction:** "5th-gen tensor cores" on GeForce Blackwell improve throughput for supported types (INT8, FP8, FP4, etc.) but the **programming model on `sm_120` remains warp-synchronous `mma.sync`**, not the **`tcgen05.mma` + TMEM** path available on datacenter `sm_100`/`sm_100a`. Community validation on shipping 5090 silicon confirms `tcgen05.*` is rejected at `ptxas` for `sm_120a` targets.

Source: [lna-lab/blackwell-geforce-nvfp4-gemm SM120 ISA analysis](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm), [CUTLASS Blackwell SM100 docs (tcgen05 scope)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).

---

## 2. Which Specs Matter Most for Pearl-Style Sustained GEMM Mining

Pearl mining executes a **fixed-shape INT8 GEMM** (M×N×K with transcript extraction, BLAKE3 target check, noise expansion) in a tight loop. This is not gaming, not FP4 inference, and not sparse structured sparsity.

### 2.1 Priority Ranking (High → Low)

| Rank | Spec | Why it matters for Pearl |
|------|------|--------------------------|
| 1 | **Tensor core throughput @ actual clock** | Mining is pure INT8 MAC throughput + transcript side work. Rated 838 INT8 TOPS is the hard ceiling. |
| 2 | **SM count (170)** | Grid sizing, wave count, tail-slot loss. Directly drives occupancy strategy. |
| 3 | **Core/boost clock stability** | Tensor pipes scale with SM clock. Sustained 2.4 GHz matters more than peak burst. |
| 4 | **Shared memory capacity (99 KB/block on sm_120)** | Tile shape 128×256×128 + pipeline stages must fit; governs BK, staging, swizzle. |
| 5 | **Register file pressure / occupancy** | Consumer kernel holds 128×int32 accumulators/thread + transcript state; limits concurrent CTAs/SM. |
| 6 | **L2 cache (96 MB)** | Helps reuse B-matrix and noise tiles across waves; secondary at K=128. |
| 7 | **L1/shared (128 KB/SM)** | Bank conflicts and swizzle layout affect `ldmatrix` efficiency. |
| 8 | **GDDR7 bandwidth (1,792 GB/s)** | **Low priority** at K=128, M=8192: arithmetic intensity ~7,900 MACs/byte → needs ~106 GB/s to saturate compute. |
| 9 | **PCIe Gen 5** | Negligible for 8-byte nonce H2D; resident B amortizes large uploads. |
| 10 | **RT cores, FP4/FP8, DLSS, encoders** | **Irrelevant** — disable mentally; no mining path. |
| 11 | **Sparse tensor rating (1,676 TOPS)** | Pearl uses dense INT8; sparse peak is not applicable. |

### 2.2 What Pearl's Workload Actually Looks Like

Per matmul (production N=262144):

```
TMAD_per_matmul = M × N × K = 8192 × 262144 × 128 ≈ 274.9 trillion MACs
CTAs            = (M/128) × (N/256) = 64 × 1024 = 65,536
K-tile depth    = K/128 = 1 (single K-slab per CTA for default K)
```

Each CTA also performs transcript snapshot extraction (XOR-reduce chain), and the headless path runs in-kernel BLAKE3 target checks. These epilogue costs are why **tensor-pipe utilization < rated TOPS** even when memory is plentiful.

---

## 3. Achieving ~100% GPU Utilization for Long-Running Tensor-Core Workloads

"100% utilization" in `nvidia-smi` is insufficient. For Pearl, target these **hardware counters** (Nsight Compute):

| Metric | Target | Interpretation |
|--------|--------|----------------|
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | **>80%** | Tensor pipes actually issuing MMAs |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | >70% | Overall SM throughput |
| `gpu__time_active.avg.pct_of_peak_sustained_elapsed` | >95% | GPU not idle between batches |
| DRAM throughput / 1,792 GB/s | **<30%** expected | Confirms compute-bound (good for this workload) |
| Achieved occupancy / theoretical | Maximize without spilling | Watch register limits |

### 3.1 Software Architecture Checklist (PropMiner-Specific)

| Layer | Mechanism | Status / Notes |
|-------|-----------|----------------|
| **Kernel persistence** | Long-lived batch loops via CUDA graphs | Implemented (`GpuWorker::prepare_graph`) |
| **Launch overhead removal** | Graph capture of upload+GEMM+reduce | Critical at batch=4–20 |
| **Host/GPU overlap** | Ping-pong streams + `seed_copy_stream_` | 8-byte nonce H2D overlapped with compute |
| **Resident B state** | σ-install once; B/noise in VRAM | Avoids per-matmul B rebuild |
| **Grid saturation** | CTAs >> 170×2 (e.g., 65,536 CTAs) | Ensures multi-wave fill |
| **Hot-path sync** | `cudaEventQuery` spin, not `cudaDeviceSynchronize` | Reduces idle bubbles |
| **Watchdog** | Context reset on launch timeout | 24/7 stability |
| **CPU isolation** | No CPU GEMM; GPU-only math | Per RTX5090 blueprint |

### 3.2 Kernel-Side Levers

1. **Minimize epilogue bubbles:** Fuse transcript readback with MMA pipeline stages; avoid serial `cudaDeviceSynchronize` per matmul.
2. **Register budgeting:** `__launch_bounds__(256, N)` to target 2–3 blocks/SM if SMEM allows.
3. **Memory hierarchy:** `cp.async` 16B granular loads + `ldmatrix` (current path); experiment with **TMA producer warp** for GMEM→SMEM (supported on sm_120 family per CUDA tables, though cluster multicast TMA is consumer-limited).
4. **Swizzle selection:** Measured 0.5% gain (Swizzle<3,4,3> vs <2,4,3>) — small but "free" once profiled.
5. **Avoid false optimizations:** Memory OC, PCIe tuning, and VRAM downclocking do not help a compute-bound INT8 kernel.

### 3.3 What "100%" Cannot Mean Here

Even a perfect GEMM cannot hit 838 TOPS because Pearl adds:

- Transcript XOR-reduce and layout constraints (consensus-critical bytes)
- Periodic BLAKE3 hashing and target comparison
- Noise/Merkle orchestration between batches (host-side, but bounds batch duty cycle)

Expect **practical ceiling ~60–75% of rated INT8 TOPS** (~500–630 TMAD/s) with an excellent kernel — not 838 TMAD/s.

---

## 4. Realistic Paths to 2×–3× Hashrate vs Naive Baseline

### 4.1 Baseline Definition

| Metric | Naive baseline | Source |
|--------|----------------|--------|
| Kernel ISA | SM80 `mma.sync m16n8k32` in `sm_120a` cubin | `transcript_gemm_kernel.cu` |
| Shape | M=8192, N=262144, K=128 | Production profile |
| Throughput | **~300 TMAD/s** | RunPod RTX 5090 headless benchmark |
| H/s (DAF-normalized) | ~1.2×10⁹ class | Derived at batch=20 |

A **2×** target is **~600 TMAD/s**; **3×** is **~900 TMAD/s** — the latter **exceeds rated dense INT8 peak (838 TOPS)** and is only achievable with measurement definitional slack, sparse paths (not consensus-valid), or burst-not-sustained conditions. **Treat 3× as an upper engineering stretch, not a planning guarantee.**

### 4.2 Optimization Roadmap (Realistic Uplift Estimates)

| # | Optimization | Expected uplift | Confidence | Notes |
|---|-------------|-----------------|------------|-------|
| 1 | **Profile-driven occupancy** (registers, `MIN_BLOCKS`, stages) | +10–20% | High | ncu-guided; low risk |
| 2 | **TMA warp-specialized loads** (producer/consumer split) | +10–25% | Medium | CUTLASS SM120 collectives; SMEM 99 KB cap |
| 3 | **Thread-block clusters + CLC** (dynamic work stealing) | +5–15% | Medium | Reduces tail-wave idle; cluster size 2–8 |
| 4 | **Core clock +150 MHz** (validated with self-test) | +5–8% | Medium | Tensor-bound; thermal tradeoff |
| 5 | **Batch size tuning** (4→8→16 matmuls/graph) | +3–10% | High | Amortizes launch + σ overhead |
| 6 | **Production N=262144** (vs bench N=32768) | +0.5% kernel; better duty cycle | High | Already deployed; not 8× H/s |
| 7 | **CUDA graph + pinned H2D polish** | +5–10% vs no-graph baseline | High | Already largely implemented |
| 8 | **Power limit 95–100% + cooling headroom** | +0–5% | Medium | Prevents throttle |
| **—** | **tcgen05/TMEM port from B200** | **N/A on RTX 5090** | **Invalid path** | Hardware does not expose UMMA |
| **—** | **Sparse tensor ops** | **N/A** | Invalid | Pearl requires dense transcript |
| **—** | **"1000× kernel fusion fantasy"** | — | — | Out of scope |

**Compositional ceiling:** Multiplicative stacking is optimistic; measured **2× (~600 TMAD/s)** is achievable with items 1–3 + 4 + 7. **2.5× (~750 TMAD/s)** approaches practical limits. **3×** requires near-perfect tensor-pipe utilization (>90% of 838 TOPS) and is unlikely sustained with transcript+hash epilogue intact.

### 4.3 What NOT to Chase

- **`sm_100a` binaries on 5090** — driver rejects incompatible cubins.
- **Hopper WGMMA (`sm_90`)** — wrong ISA generation.
- **FP4/FP8 precision reduction** — breaks proof-canonical INT8 transcript.
- **Shrinking N to fix tail waves** — 262144 → 65280 eliminates tail but sacrifices hashrate economics; PropMiner correctly prefers large N.

---

## 5. Gap Analysis: Theoretical Peak TFLOPS/TOPS vs Mining TMAD/s

### 5.1 Unit Alignment

| Term | Definition |
|------|------------|
| **NVIDIA INT8 TOPS** | 10¹² INT8 ops/s at boost clock, dense tensor cores |
| **PropMiner TMAD/s** | 10¹² multiply-adds/s across the full GEMM (M×N×K per matmul × matmuls/s) |
| **TMAD per matmul** | M × N × K = 8192 × N × 128 |

For INT8 GEMM, **1 MAC ≈ 1 INT8 tensor op** (multiply + add counted as one MAD in mining telemetry).

### 5.2 Theoretical Peak (RTX 5090)

```
Rated dense INT8 tensor peak = 838 × 10¹² ops/s
At boost 2407 MHz, 170 SMs, 680 tensor cores
```

Source: [NVIDIA whitepaper Table 1](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

### 5.3 Observed Mining Throughput

```
Observed TMAD/s ≈ 300 (consumer SM80 mma.sync kernel, N=262144)
Efficiency vs INT8 peak = 300 / 838 ≈ 35.8%
```

### 5.4 Where the ~64% Gap Goes

| Loss bucket | Estimated impact | Evidence |
|-------------|------------------|----------|
| Legacy `mma.sync` vs native Blackwell UMMA throughput | 15–30% | SM120 lacks tcgen05; runs Ampere-class issue rate |
| Transcript extraction + XOR-reduce epilogue | 10–20% | Serializes after each K-slab |
| Register pressure → occupancy shortfall | 5–15% | 128 int32/thread accumulators |
| Tail-wave SM idle (26/170 slots last wave) | ~0.4% | 26/65536 CTAs; small but non-zero |
| BLAKE3 target check (headless path) | 2–8% | Branch + memory traffic |
| Launch/graph/batch gaps | 5–10% | Improving with batch tuning |
| Memory subsystem | **<5%** | Arithmetic intensity ~7,900 MACs/byte ≪ ridge point |

### 5.5 Arithmetic Intensity Proof (Compute-Bound)

```
Bytes_read ≈ M×K + K×N = 8192×128 + 128×262144 ≈ 34.6 MB (int8)
MACs = M×N×K ≈ 2.75×10¹⁴
AI = MACs / bytes ≈ 7,900 MACs/byte

To saturate 838×10¹² MAC/s:
  Required bandwidth ≈ 838×10¹² / 7900 ≈ 106 GB/s
  Available: 1,792 GB/s
  Memory utilization at peak: ~6%
```

**Conclusion:** GDDR7 bandwidth is not the bottleneck. Raising memory clock for mining is low ROI; raising **stable SM clock** and **tensor-pipe active cycles** is high ROI.

### 5.6 Reference Comparison Table

| Platform | ISA | Shape (M,N,K) | Reported TMAD/s | % of 838 TOPS |
|----------|-----|---------------|-----------------|---------------|
| RTX 5090 (PropMiner consumer) | `mma.sync` sm_120a | 8192, 262144, 128 | ~300 | 36% |
| RTX 5090 (target tuned) | TMA + occupancy | same | ~600 (goal) | 72% |
| B200 (pearl-gemm sm100) | tcgen05 + TMEM | 8192, 32768, 2048+ | 840–877 | N/A (different SKU) |
| RTX 5090 rated peak | Hardware spec | — | 838 | 100% |

B200 numbers demonstrate what **tcgen05** achieves on datacenter silicon — they are an **upper-bound reference**, not a direct port target for GB202 GeForce.

---

## 6. Thermal, Power, and Overclock Considerations for 24/7 Mining

### 6.1 Stock Power Envelope

| Setting | Recommendation |
|---------|----------------|
| TGP | 575 W stock |
| Typical mining draw | 500–575 W (load-dependent) |
| Thermal limit | 90 °C GPU max per NVIDIA |
| Fan strategy | Aggressive fixed curve >70 °C |

### 6.2 24/7 Operational Risks

**Power connector safety:** Multiple reports of **12V-2×6 connector overheating/melting** on RTX 5090 under sustained load, including cases with manually reduced power limits (500 W). Risk factors: adapter cables, insufficient insertion, transient current spikes, uneven pin contact.

Sources: [ElcomSoft 24/7 workstation analysis](https://blog.elcomsoft.com/2025/03/nvidia-geforce-rtx-5090-power-connectors-melting-again/), [Wccftech connector failure report](https://wccftech.com/nvidia-rtx-5090-gets-its-top-connector-row-cooked-despite-a-500w-max-power-ceiling-by-the-user/).

**Mitigations:**

- Native **12V-2×6 PSU cable** (no adapter) from ATX 3.1 PSU
- Connector fully seated; **no bend within 35 mm** of connector
- Consider power monitoring (e.g., CableMod sense tools)
- **Undervolt** for efficiency: reports of 0.885–0.975 V @ 2700–3000 MHz reducing draw **~100 W** with comparable or better throughput on tensor workloads
- **Do not rely solely on `nvidia-smi -pl`** for connector safety — it limits average power, not contact quality

### 6.3 Clock and Voltage Policy for NoisyGEMM

| Knob | Mining recommendation | Rationale |
|------|----------------------|-----------|
| Core clock | +100 to +200 MHz max after self-test | Tensor pipes scale with core; validate proof correctness |
| Memory clock | **Stock** or mild +200 MHz | Compute-bound; large OC adds heat for <1% gain |
| Power limit | 90–100% (550–575 W) | Lower only for thermal/connector margin |
| VRAM min clock lock | **Never lock to minimum** | Creates memory-side stalls |
| Fan | Set curve for <80 °C sustained | Avoid throttle below 2.4 GHz |
| Persistence mode | `nvidia-smi -pm 1` | Reduces init latency in containers |

**Validation gate:** Run `./propminer --self-test --rtx5090` after every OC change. Silent INT8 errors invalidate shares.

### 6.4 Efficiency Sweet Spot (Illustrative)

Community undervolt data on tensor-heavy workloads shows **~20% perf/W improvement** possible (e.g., 575 W → 450 W at similar throughput). Mining fleets should optimize **TMAD/s per watt**, not raw TMAD/s alone.

Source: [Overclock.net RTX 5090 UV study](https://www.overclock.net/threads/rtx-5090-msi-vanguard-soc-launch-edition-comparison-stock-vs-undervolt-vs-overclock-4090-comparison-on-4-synthetics-and-3-games.1815041/).

---

## 7. `sm_120a` vs `sm_120`, Driver, and CUDA Requirements

### 7.1 Compute Capability Taxonomy

| Target | Suffix | Feature set | RTX 5090 compatible |
|--------|--------|-------------|---------------------|
| `compute_120` | none | Baseline Blackwell | Yes |
| `compute_120f` | **f** (family) | Shared tensor PTX across CC 12.x family | Yes |
| `compute_120a` | **a** (architecture) | Full arch-specific PTX for CC 12.0 | Yes (exact match) |
| `compute_100a` | **a** | Datacenter tcgen05 / TMEM | **No — cubin rejected** |

CUDA 12.9+ formalized **family-specific (`f`)** vs **architecture-specific (`a`)** targets. Tensor-core PTX generally requires `f` or `a` suffixes; baseline `compute_120` omits arch-specific instructions.

Sources: [CUDA Programming Guide §5.1.2](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), [NVIDIA Technical Blog — family-specific features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/), [Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).

### 7.2 PropMiner Build Guidance

| Setting | PropMiner practice |
|---------|-------------------|
| Primary arch | `sm_120a` via `PEARL_GEMM_ARCH=blackwell` |
| Optional fallback | `sm_120` via `PEARL_GEMM_BLACKWELL_SM120_FALLBACK=1` |
| Forbidden | `sm_90`, `sm_100a` on consumer 5090 |
| PTX forward-compat | Include PTX for JIT if distributing generic binaries |

### 7.3 Minimum Software Stack

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CUDA Toolkit | **12.8** (first with `sm_120` nvcc) | 12.8+ / 12.9 |
| NVIDIA Driver | **570.x** (R570 branch) | Latest open-kernel 580+ |
| `nvcc` arch flag | `-gencode=arch=compute_120a,code=sm_120a` | — |
| Nsight Compute | 2025.1+ | For tensor-pipe counters |
| Linux kernel modules | `nvidia-open` recommended on 5090 | Per NVIDIA 12.8+ default |

Sources: [CUDA 12.8 Features Archive](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html), [CUDA Toolkit / Driver Matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html), [LeaderGPU RTX 50 install guide](https://www.leadergpu.com/articles/616-install-nvidia-drivers-and-cuda-for-rtx-50-series).

### 7.4 `sm_120` Shared Memory Constraint (Often Missed)

CUDA spec for **compute capability 12.x**:

| Resource | 12.x (GeForce) | 10.x (B200 class) |
|----------|----------------|-------------------|
| Max shared memory / block | **99 KB** | 227 KB |
| Max shared memory / SM | 100 KB | 228 KB |
| TMEM | **None** | Present |
| Thread-block clusters | **Yes** (up to 8 blocks) | Yes |
| TMA unit | Present (multicast limited on consumer) | Full |

PropMiner's 128×256×128×2-stage int8 tiles must fit in **99 KB** — this constrains `kStages`, carveout, and cluster configurations.

Source: [CUDA Programming Guide Table 31/32](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

---

## 8. Wave Occupancy, Tail Waves, and Cluster Launches on 170 SMs

### 8.1 CTA Grid Mathematics

PropMiner consumer tile: **BM=128, BN=256, BK=128**.

```
grid_x = M / 128 = 8192 / 128 = 64
grid_y = N / 256
CTAs   = 64 × (N / 256) = N / 4
```

| N | CTAs | Waves (⌈CTAs/170⌉) | Tail slots (CTAs % 170) | 2× occupancy (≥340 CTAs)? |
|---|------|--------------------|-------------------------|----------------------------|
| 32,768 | 8,192 | 49 | **42** | Yes |
| 65,280 | 16,320 | 96 | **0** (aligned) | Yes |
| 262,144 | 65,536 | 386 | **26** | Yes |
| 327,680 | 81,920 | 482 | 0 (aligned) | Yes |

### 8.2 Tail Wave Impact

Last-wave idle SMs = `tail_slots / CTAs × 100%`:

- N=262144: 26/65536 ≈ **0.04%** theoretical cycle loss — negligible.
- N=32768: 42/8192 ≈ **0.51%** — still small.

**PropMiner design choice:** Prefer **N=262144** over wave-aligned **N=65280** because hashrate economics dominate sub-1% tail loss. Function `wave_aligned_n_at_least()` exists but is not the default production picker.

### 8.3 Cluster Launches (`cluster_m`)

RTX 5090 supports **thread-block clusters up to 8 blocks** (CC 12.0). PropMiner exposes `PEARL_GEMM_CONSUMER_CLUSTER_M` (default `kProdDefaultClusterM = 2`).

| cluster_m | Effective cluster width | Considerations |
|-----------|------------------------|----------------|
| 1 | No clustering | Simplest; per-block scheduling |
| 2 | 2 CTAs cooperate | Often wins on 5090 boxes; tune per host |
| 4–8 | Wider cluster | May reduce eligible concurrent clusters; test with `cudaOccupancyMaxActiveClusters` |

**Cluster Launch Control (CLC):** Blackwell hardware feature for **dynamic work stealing** — active clusters cancel and assume unlaunched CTA work. Useful when grid is not an exact multiple of `170 × cluster_size`, or for persistent kernels that loop across tiles.

Sources: [CUDA Cluster Launch Control guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html), [CUTLASS CLC docs](https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/blackwell_cluster_launch_control.html), [Colfax CLC analysis](https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/), [NVIDIA Dev Forums — consumer cluster support](https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471).

### 8.4 Occupancy Worked Example (N=262144)

```
CTAs per GEMM     = 65,536
SMs               = 170
Full waves        = 65536 / 170 = 385.51 → 385 full + 1 partial
Partial wave CTAs = 65536 - 385×170 = 26
Concurrent CTAs   = min(CTAs, 170 × blocks_per_SM)
```

With 256 threads/CTA (8 warps) and ~99 KB SMEM/block, expect **1–2 blocks/SM** depending on register footprint — PropMiner targets **2 CTAs/SM** where SMEM/register limits allow (`full_occupancy` requires CTAs ≥ 340; production exceeds this by 190×).

---

## 9. Actionable Tuning Checklist

### 9.1 First Boot Validation

- [ ] `nvidia-smi` shows **RTX 5090**, compute cap **12.0**
- [ ] `nvcc --version` ≥ **12.8**
- [ ] Driver ≥ **570.x** (open kernel preferred)
- [ ] `./scripts/build_and_benchmark.sh 60` completes with non-zero H/s
- [ ] Startup log: `N=262144 batch=20 CTAs=65536 waves~386 tail=26`

### 9.2 Profiling Pass (Nsight Compute)

- [ ] Capture `transcript_gemm_kernel_consumer` with `ncu`
- [ ] Record `sm__pipe_tensor_cycles_active` (target >80%)
- [ ] Record achieved occupancy vs theoretical
- [ ] Confirm DRAM <30% of peak (compute-bound sanity check)
- [ ] Inspect register spill events (should be zero)

### 9.3 Kernel Knob Sweep

- [ ] Run `./scripts/tune_blackwell_knobs.sh`
- [ ] Test `PEARL_GEMM_CONSUMER_CLUSTER_M` ∈ {1, 2, 4}
- [ ] Test batch sizes {4, 8, 16, 20} with `tune-prod`
- [ ] Compare Swizzle<3,4,3> vs alternatives (expect <1% delta)
- [ ] Validate each winner with `--self-test --rtx5090`

### 9.4 Host / Runtime

- [ ] CUDA graphs enabled (disable only for debug)
- [ ] `CUDA_MODULE_LOADING=EAGER` in containers
- [ ] Resident B confirmed (no per-batch B re-upload)
- [ ] Watchdog enabled for 24/7 mine mode
- [ ] Compare **TMAD/s** not just H/s when tuning N or batch

### 9.5 24/7 Hardware

- [ ] Native 12V-2×6 cable (no adapter)
- [ ] Power limit set and logged
- [ ] Thermal equilibrium <83 °C under batch=20 load
- [ ] Undervolt curve tested if connector/thermals tight
- [ ] Self-test schedule after any OC/UV change

---

## 10. Key Takeaways

1. **RTX 5090 is 170 SMs × 2.41 GHz × 838 INT8 TOPS** with 32 GB G7 @ 1,792 GB/s — a compute-rich, bandwidth-abundant miner for K=128 shapes.

2. **Pearl mining cares about tensor-pipe duty cycle, SM occupancy, and epilogue cost** — not RT cores, FP4, or PCIe.

3. **Measured ~300 TMAD/s is ~36% of rated INT8 peak** — the gap is ISA efficiency (legacy `mma.sync`), transcript epilogue, and occupancy — not GDDR7.

4. **2× (~600 TMAD/s) is a credible engineering target** via TMA loads, cluster/CLC scheduling, occupancy tuning, and graphs. **3× (~900 TMAD/s) exceeds dense rated peak** and is not realistic sustained with proof-canonical epilogue.

5. **`tcgen05` + TMEM is datacenter-only (`sm_100`)** — PropMiner must optimize the **SM120 `mma.sync` + TMA** path, not port B200 kernels verbatim.

6. **Wave tail on 170 SMs is a red herring** at N=262144 (26 idle slots ≈ 0.04%). Prefer large N for economics.

7. **24/7 mining demands power connector discipline** and thermal headroom — treat 575 W sustained as a infrastructure problem, not just a software problem.

---

## References

1. NVIDIA RTX Blackwell GPU Architecture Whitepaper v1.5 — https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf  
2. NVIDIA GeForce RTX 5090 Product Specifications — https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/rtx-5090/  
3. TechPowerUp RTX 5090 GPU Database — https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216  
4. HotHardware RTX 5090 Review (GB202 die details) — https://hothardware.com/reviews/nvidia-geforce-rtx-5090-review  
5. CUDA Programming Guide — Compute Capabilities — https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html  
6. CUDA 12.8 Features Archive (sm_120 compiler support) — https://docs.nvidia.com/cuda/archive/12.8.0/cuda-features-archive/index.html  
7. Blackwell Compatibility Guide — https://docs.nvidia.com/cuda/blackwell-compatibility-guide/  
8. NVIDIA Blog — Family-Specific Architecture Features (CUDA 12.9) — https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/  
9. CUDA Toolkit / Driver Architecture Matrix — https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html  
10. CUTLASS Blackwell SM100 Functionality (tcgen05 scope) — https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html  
11. CUTLASS Cluster Launch Control — https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/blackwell_cluster_launch_control.html  
12. CUDA Cluster Launch Control Programming Guide — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html  
13. Colfax Research — CLC on Blackwell — https://research.colfax-intl.com/dynamic-persistent-tile-scheduling-with-cluster-launch-control-clc-on-nvidia-blackwell-gpus/  
14. NVIDIA Dev Forums — Thread Block Clustering on Blackwell — https://forums.developer.nvidia.com/t/thread-block-clustering-in-blackwell-gpus/320471  
15. lna-lab — SM120 ISA Analysis (no tcgen05 on GeForce) — https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm  
16. ElcomSoft — RTX 5090 24/7 Connector Risk — https://blog.elcomsoft.com/2025/03/nvidia-geforce-rtx-5090-power-connectors-melting-again/  
17. Wccftech — Connector Failure at 500 W Limit — https://wccftech.com/nvidia-rtx-5090-gets-its-top-connector-row-cooked-despite-a-500w-max-power-ceiling-by-the-user/  
18. LeaderGPU — RTX 50 Driver/CUDA 12.8 Install — https://www.leadergpu.com/articles/616-install-nvidia-drivers-and-cuda-for-rtx-50-series  
19. PropMiner internal: `docs/RTX5090_BLUEPRINT.md`, `performance optimizations/01-native-tcgen05-tmem-gemm.md`, `performance optimizations/03-production-n262144.md`

---

*Report generated for PropMiner research track 02. No code changes.*
