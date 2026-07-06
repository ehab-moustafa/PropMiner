# PropMiner RTX 5090 Headroom Analysis

**PropMiner Research · Topic 08**  
**Date:** July 2026  
**Scope:** Measured and modeled resource utilization on GeForce RTX 5090 during Pearl NoisyGEMM mining; ranked optimization opportunities (GPS — Gains Per Second); safe vs risky levers; realistic **2–3×** throughput envelope.

**Primary sources:** `research/02-rtx5090-specs/REPORT.md`, `research/03-propminer-codebase/REPORT.md`, `performance optimizations/01–08`, RunPod headless benchmarks embedded in `transcript_gemm_kernel.cu`.

---

## Executive Summary

PropMiner on RTX 5090 at production shape (M=8192, N=262144, K=128, R=128) is **compute-bound on tensor cores**, not memory-, PCIe-, or CPU-bound. Observed steady-state profile:

| Resource | Utilization | Interpretation |
|----------|-------------|----------------|
| **INT8 tensor throughput** | **~36% of 838 TOPS** rated peak (~300 TMAD/s measured) | Legacy `mma.sync` path under-fills Blackwell pipes |
| **SM activity (nvidia-smi)** | **70–90%** | High block residency; not the same as tensor-pipe efficiency |
| **Memory bandwidth (HBM)** | **~2%** of 1,792 GB/s | K=128 single-tile mainloop; B resident |
| **PCIe Gen5 x16** | **<0.001%** | 8-byte seed upload per batch |
| **CPU** | **Not a bottleneck** | `disable_cpu_mining=true`; graph replay hot path |

The gap between **70–90% SM** and **36% TOPS** is the central insight: SMs are busy, but each warp spends large register files on accumulators and runs Ampere-class IMMA instead of saturating 5th-gen tensor throughput.

**Realistic envelope:** **2×–3×** over ~300 TMAD/s baseline → **600–900 TMAD/s**, assembled from orthogonal safe levers. **3×** exceeds rated dense INT8 peak (838 TOPS) unless measurement definitions include burst conditions — treat as stretch, not planning guarantee.

---

## 1. Measurement Methodology

### 1.1 Hardware baseline

| Parameter | RTX 5090 value |
|-----------|----------------|
| Architecture | GB202, compute capability 12.0 (`sm_120a`) |
| Enabled SMs | 170 |
| Boost clock | 2,407 MHz |
| Rated INT8 tensor (dense) | **838 TOPS** |
| Memory | 32 GB GDDR7, 1,792 GB/s |
| TGP | 575 W |

Source: NVIDIA Blackwell GeForce whitepaper; `research/02-rtx5090-specs/REPORT.md`.

### 1.2 Software baseline

| Setting | Value |
|---------|-------|
| Kernel | `consumer/transcript_gemm_kernel.cu` via `transcript_gemm_sm120.cu` |
| MMA atom | SM80 `m16n8k32.s32.s8.s8.s32` |
| Tile | 128×256×128 |
| Grid | 65,536 CTAs (8192/128 × 262144/256) |
| Batch | 4–20 matmuls per graph launch |
| TMAD/s | **~300** (Swizzle<3,4,3>, N=262144) |

### 1.3 Metrics disambiguation

| Metric | What it measures | Typical 5090 value |
|--------|------------------|-------------------|
| **TOPS efficiency** | Achieved INT8 tensor ops / 838×10¹² | ~36% |
| **SM utilization** | Active cycles / elapsed (driver) | 70–90% |
| **Tensor Active** (ncu) | Cycles with MMA issue / total | ~40–55% |
| **Memory throughput** | Achieved GB/s / 1,792 GB/s | ~2% |
| **TMAD/s** | M×N×K×matmuls/s | ~300×10¹² |

**Do not conflate SM% with TOPS%.** A register-heavy kernel can keep SMs "hot" while tensor pipes stall between dependent `mma.sync` chains.

---

## 2. Resource Utilization Deep Dive

### 2.1 Tensor cores (~36% of 838 TOPS)

```
Efficiency = observed_TMAD/s / rated_INT8_peak
           ≈ 300 / 838 ≈ 35.8%
```

**Where the ~64% gap goes:**

| Loss bucket | Est. impact | Evidence |
|-------------|-------------|----------|
| Legacy IMMA vs native UMMA throughput | 15–30% | GB202 GeForce lacks `tcgen05`; SM80 atom on sm_120a |
| Transcript XOR + BLAKE3 epilogue | 10–20% | Serializes after K-slab in consumer loop |
| Register pressure → low occupancy | 5–15% | 128 int32/thread + 16 u32 transcript |
| Tail-wave idle (26/170 SMs last wave) | ~0.4% | 65,536 mod 170 |
| Launch/graph gaps | 5–10% | Improving with batch 20 + CUDA graphs |

Code: `transcript_gemm_kernel.cu` L150–151 (300.78 vs 299.19 TMAD/s swizzle comparison).

### 2.2 SM activity (70–90%)

With 65,536 CTAs and `__launch_bounds__(256, 1)`:

- **~386 waves** across 170 SMs
- **Tail 26** CTAs on final wave
- **1 block/SM** typical due to register limits

nvidia-smi reports **70–90%** SM utilization because:

1. Nearly every SM has a resident block during GEMM replay
2. Graph launch eliminates CPU idle gaps between batches
3. Ping-pong halves keep compute stream continuously fed

This is **good occupancy of SM slots**, not proof of tensor-pipe saturation.

### 2.3 Memory bandwidth (~2%)

Arithmetic intensity at production shape:

```
Bytes_read ≈ M×K + K×N = 8192×128 + 128×262144 ≈ 34.6 MB (int8)
MACs = M×N×K ≈ 2.75×10¹⁴
AI ≈ 7,900 MACs/byte

Bandwidth to saturate 838×10¹² MAC/s:
  ≈ 838×10¹² / 7900 ≈ 106 GB/s → ~6% of peak (theoretical minimum)
Observed: ~2% (B resident, cp.async latency-bound not BW-bound)
```

**Conclusion:** GDDR7 OC is low ROI. L2 fetch granularity (128 B) already tuned in `GpuWorker` ctor.

### 2.4 PCIe (<0.001%)

Per batch (`gpu_worker.cpp`):

```cpp
upload_next_seed_async(*other, seed_base_ + global_iter + batch);
// sizeof(uint64_t) = 8 bytes on seed_copy_stream_
```

At 20 batches/s → **160 B/s**. Gen5 x16 ≈ 64 GB/s theoretical → **<0.001%**.

Share D2H (leaf CVs, A slices) is rare and pinned — irrelevant to sustained hashrate.

Doc: `performance optimizations/07-pcie-gen5-psu-headroom.md`.

### 2.5 CPU (not bottleneck)

| Factor | Status |
|--------|--------|
| `disable_cpu_mining = true` | Default on `--rtx5090` |
| Hot path | `cudaEventQuery` spin-wait, O(batch) header scan |
| Threads | Orchestrator: network, share sender, optional defer worker |
| CPU GEMM | None on mining path |

Doc 08 closed SeedGenerator evaluation — 8-byte async upload supersedes CPU PRNG ring.

---

## 3. GPS Ranked Opportunities Table

**GPS** = Gains Per Second — expected TMAD/s uplift per engineering week, ranked for PropMiner RTX 5090 production mining. Confidence: High / Medium / Low. Risk: Safe / Moderate / High (consensus).

| Rank | Opportunity | GPS (TMAD/s) | Effort | Confidence | Risk | Primary path |
|:----:|-------------|-------------:|--------|:----------:|:----:|--------------|
| **1** | Kernel knob autotune (STAGES, SWIZZLE, MIN_BLOCKS, KBLOCK) | +15–45 (+5–15%) | Days | High | Safe | `scripts/tune_blackwell_knobs.sh`, doc 05 |
| **2** | Production N=262144 + batch 20 | +0–9 (+0–3%) | Done | High | Safe | `rtx5090_profile.h`, doc 03 |
| **3** | TMA warp-specialized tile loads | +30–75 (+10–25%) | Weeks | Medium | Moderate | doc 02, `tma_tile_loader.cuh` |
| **4** | Thread-block clusters (cluster_m=2→4 sweep) | +15–45 (+5–15%) | Days | Medium | Safe | `tune_cluster_sweep.sh` |
| **5** | CUDA graph + batch polish | +15–30 (+5–10%) | Done | High | Safe | `gpu_worker.cpp` graphs |
| **6** | `MIN_BLOCKS=2` occupancy bump | +9–30 (+3–10%) | Days | Medium | Safe | ncu-guided, doc 05 |
| **7** | Core clock +100–150 MHz (validated) | +15–24 (+5–8%) | Hours | Medium | Safe | OS-level; thermal tradeoff |
| **8** | Defer share GPU work | +0–6 (+0–2%) | Done | High | Safe | `PROPMINER_DEFER_SHARE_GPU=1`, doc 04 |
| **9** | Power limit 95–100% + cooling | +0–15 (+0–5%) | Hours | Medium | Safe | Infrastructure |
| **10** | Native tcgen05/TMEM port | **N/A on 5090** | Months | — | **Invalid** | GB202 lacks UMMA — doc 01 |
| **11** | SM120 CUTLASS int8 atom rename | ~0 | Weeks | High | Safe | doc 06 — blocked |
| **12** | SeedGenerator ring buffer | ~0 | — | High | — | Removed, doc 08 |
| **13** | PCIe Gen5 upgrade (healthy Gen4) | ~0 | — | High | — | doc 07 |
| **14** | Memory OC / faster GDDR7 | <+3 | Hours | Low | Moderate | 2% BW util |
| **15** | Sparse tensor / FP4 paths | **Invalid** | — | — | **High** | Breaks transcript |
| **16** | "100× kernel fusion" | **Invalid** | — | — | **High** | Physically impossible |

### 3.1 GPS scoring methodology

```
GPS_score = (expected_TMAD/s_gain) / (engineering_weeks × risk_multiplier)

risk_multiplier: Safe=1.0, Moderate=1.5, High=3.0, Invalid=∞
```

Rank 1 (knob autotune) wins on **GPS score** because effort is days, risk is zero (self-test gated), and +5–15% is proven on similar kernels.

Rank 3 (TMA) has higher absolute gain but moderate risk — smem 99 KB cap on consumer Blackwell, must preserve swizzled layout byte identity.

Rank 10 (tcgen05) is **invalid** on RTX 5090 per `external_repos/blackwell-geforce-nvfp4-gemm` and NVIDIA whitepaper: GeForce Blackwell does not expose `tcgen05.mma` or TMEM.

---

## 4. Safe vs Risky Classification

### 4.1 Safe (proof-preserving, production-ready)

| Lever | Why safe |
|-------|----------|
| Runtime batch size | No kernel change |
| Cluster size / carveout | Dispatch only |
| Knob autotune with `--self-test` | Byte-compare transcript gate |
| N=262144 VRAM pick | Same tile geometry |
| Graph capture path | Same kernel DAG |
| Core clock / power | Hardware only |
| Share deferral | Side thread; same proof bytes |

### 4.2 Moderate risk (requires validation)

| Lever | Risk |
|-------|------|
| TMA load path | Wrong smem address → wrong MMA input |
| MIN_BLOCKS=2 | Register spill if mis-tuned |
| KBLOCK=64 + STAGES=3 | smem > 99 KB → launch failure |
| Wave-aware N selection | Economics vs tail-wave tradeoff |

**Gate:** `propminer --self-test` + pool canary shares on Kryptex test wallet.

### 4.3 Risky / invalid (do not pursue for 5090 Pearl)

| Lever | Why invalid |
|-------|-------------|
| tcgen05/TMEM on GB202 | Hardware rejects PTX |
| FP4/FP8/INT4 GEMM | Transcript requires int8 |
| Change 128×256 tile | Breaks partition_C consensus |
| Skip transcript / hash | Invalid shares |
| sm_100a cubin on 5090 | Driver incompatibility |
| Pure cuBLAS GEMM without transcript | Not valid PoW |

---

## 5. Realistic 2–3× Envelope

### 5.1 Baseline anchor

```
B₀ ≈ 300 TMAD/s  (consumer mma.sync, tuned swizzle, N=262144)
```

### 5.2 Phase composition (non-multiplicative)

| Phase | Incremental | Cumulative TMAD/s | Cumulative × |
|-------|-------------|-------------------|:------------:|
| **B₀** Baseline | — | ~300 | 1.0× |
| **A** Config + autotune | +10–20% | ~330–360 | 1.1–1.2× |
| **B** TMA + occupancy | +15–30% | ~380–470 | 1.3–1.6× |
| **C** Clock + power + batch | +5–10% | ~400–520 | 1.4–1.7× |
| **D** GeForce tcgen05 alternative (TMA+IMMA warp spec) | +20–40% | ~480–730 | 1.6–2.4× |
| **Ceiling** Rated INT8 peak | — | **838** | **2.8×** |

**Honest planning range:**

- **Conservative 2×:** ~600 TMAD/s — Phase A+B+C with good silicon
- **Stretch 2.5×:** ~750 TMAD/s — Phase D partial (warp-specialized loads per `transcript_gemm_sm120_geforce.cu`)
- **Hard ceiling ~2.8×:** 838 TOPS rated — cannot sustain above with transcript epilogue intact
- **3× (~900 TMAD/s):** Requires redefining metrics or burst-not-sustained — **not a planning target**

### 5.3 Why not 10× or 100×

| Claim | Reality |
|-------|---------|
| "5090 is only 36% utilized, so 2.8× is free" | SM% ≠ TOPS%; epilogue is mandatory |
| "Port B200 kernel wholesale" | Wrong ISA on GeForce |
| "Drop BLAKE3 to CPU" | Already <15%; not the cap |
| "Mine at N=4096" | Fewer tiles but same TMAD/s; lower pool economics |

---

## 6. Bottleneck Hierarchy (Code Evidence)

| Rank | Bottleneck | Impact | File |
|:----:|------------|--------|------|
| 1 | Legacy IMMA on Blackwell | High | `transcript_gemm_sm120.cu` |
| 2 | Register pressure (128 int32/thread) | High | `transcript_gemm_kernel.cu` kFragSize |
| 3 | K=128 single mainloop trip | Medium | K_TILES = 1 |
| 4 | cp.async vs TMA issue | Medium | All 256 threads load |
| 5 | Compile-time knob defaults | Medium | STAGES=2, MIN_BLOCKS=1 |
| 6 | σ-install / first graph | Low steady | `prepare_graph()` |
| 7 | Share hit path | <<1% | `process_share_trigger` |
| 8 | Tail wave | ~0.4% | 26 idle slots |
| 9 | PCIe / CPU / DRAM | Negligible | doc 07, gpu_worker |

From `research/03-propminer-codebase/REPORT.md` §3.

---

## 7. Power and Thermal Headroom

### 7.1 TGP vs observed

Community and PropMiner internal benchmarks cluster at **280–310 TH/s @ ~500 W**, not full 575 W TDP. Thermal throttle gap: **~5–7% TMAD/s** between hot and cool silicon (alpha-miner README, `research/07-community-resources/REPORT.md`).

### 7.2 24/7 mining infrastructure

| Item | Recommendation |
|------|----------------|
| PSU | ≥1000 W, single 12V-2×6 properly seated |
| Airflow | Treat 575 W as datacenter problem |
| Power limit | 95–100% after stability burn-in |
| PCIe | Gen4 x16 sufficient; verify x16 not x8 |

---

## 8. VRAM Headroom (Not Compute, But Operational)

Production N=262144 uses **~14–18 GB** of 32 GB including ping-pong + graphs (`doc 03`). The `shape_fits_vram()` model adds **25% headroom** — conservative, may leave N on table.

| N | CTA count | Resident B | Fit |
|---|-----------|------------|-----|
| 32768 | 8,192 | ~30 MiB | Bench cap |
| 262144 | 65,536 | ~225 MiB | Production |
| 524288 | — | OOM risk | Not targeted |

VRAM headroom enables larger N (more tiles/GEMM) but **does not increase TMAD/s** — H/s is DAF-normalized and N-invariant at fixed TMAD/s.

---

## 9. Multi-GPU Scaling Headroom

`WorkerOrchestrator` assigns disjoint `seed_base_` per GPU (top 16 bits = gpu_index). Linear scaling assumes:

- Independent PCIe roots (not PCH-limited x8/x8)
- PSU capacity for N×575 W
- Pool connection handles N workers

**Headroom:** Near-linear to 4× GPUs on typical mining boards; marginal PCIe/CPU return above 8 GPUs.

---

## 10. Comparison to Reference Platforms

| Platform | TMAD/s | % of 838 | Notes |
|----------|--------|----------|-------|
| RTX 5090 PropMiner (now) | ~300 | 36% | This report |
| RTX 5090 target (2×) | ~600 | 72% | Achievable envelope |
| RTX 5090 stretch (2.5×) | ~750 | 89% | Near peak |
| B200 pearl-gemm sm100 | 840–877 | N/A | Different SKU, tcgen05 |
| H100 community | 550+ | N/A | WGMMA, sm_90 |
| alpha-miner 5090 | 280–300 TH/s | — | Hashrate units differ |

---

## 11. Recommended Action Sequence

### Week 1 (safe, high GPS)

1. Run `scripts/tune_prod_5090.sh` on bare metal
2. Enable `PROPMINER_USE_TUNE_CACHE=1`
3. Verify mine mode N=262144 in startup logs
4. Set `PROPMINER_DEFER_SHARE_GPU=1` after 24 h soak

### Month 1 (moderate)

1. Implement TMA producer warp (doc 02) behind compile flag
2. ncu profile: `TensorActive`, `smsp__sass_thread_inst_executed_op_mma` 
3. Ship winning knobs from sweep to default build

### Quarter 1 (stretch)

1. Complete `transcript_gemm_sm120_geforce.cu` warp-specialized path
2. Validate byte-identity vs consumer baseline on full 65k grid
3. Pool canary before default dispatch

### Never

1. tcgen05 port to RTX 5090
2. Tile geometry change without hard fork
3. PCIe riser "optimization" on healthy link

---

## 12. Summary

| Question | Answer |
|----------|--------|
| Is the GPU "idle"? | **No** — 70–90% SM; tensor pipes under-fed |
| Biggest lever? | **GEMM kernel** (TMA + occupancy + knobs) |
| Is DRAM the cap? | **No** — ~2% BW |
| Is PCIe the cap? | **No** — 8 bytes/batch |
| Is CPU the cap? | **No** |
| Realistic uplift? | **2×–2.5×** sustained; **3×** stretch |
| Invalid fantasies? | tcgen05 on 5090, 100× fusion, sparse ops |

---

## Appendix: Key Files

| Component | Path |
|-----------|------|
| RTX 5090 profile | `src/host/pearl/rtx5090_profile.h` |
| GPU worker | `src/host/pearl/gpu_worker.cpp` |
| Consumer kernel | `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` |
| Perf docs 01–08 | `performance optimizations/` |
| Codebase report | `research/03-propminer-codebase/REPORT.md` |
| 5090 specs | `research/02-rtx5090-specs/REPORT.md` |
| Prod tuning script | `scripts/tune_prod_5090.sh` |

---

## 13. Monitoring Checklist (24/7 Operations)

Operators sustaining 5090 Pearl mining should log these metrics weekly and re-run GPS ranking when baseline shifts by >5%:

| Metric | Tool | Healthy range | Action if out of range |
|--------|------|---------------|------------------------|
| TMAD/s | PropMiner bench log | 280–320 (stock), 400+ (tuned) | Re-run `tune_prod_5090.sh` |
| SM util | `nvidia-smi dmon` | 70–90% | Expected; do not chase 100% |
| Memory BW | ncu `dram__throughput` | <5% | Confirm B resident; ignore OC |
| PCIe TX/RX | `nvidia-smi` | ~0 B/s steady | Check riser if >1 MB/s |
| CPU % | `htop` | <10% per GPU worker | Check stray CPU mining flag |
| VRAM | `nvidia-smi` | 14–18 GB @ N=262144 | Reduce N if OOM |
| Power | `nvidia-smi` | 450–575 W | Thermal throttle if <450 at 100% util |
| Rejected shares | Pool dashboard | <1% | Kernel/transcript regression |

---

*Headroom analysis reflects PropMiner as of July 2026. Re-profile after every kernel dispatch change — GPS rankings shift when baseline TMAD/s moves.*
