# PropMiner Comprehensive Integration Plan — 700–800+ TMAD/s Roadmap

**Target:** Surpass SRBMiner (~600 TMAD/s) on RTX 5090, with stretch goal of 1000+ TMAD/s  
**Current Baseline:** ~290 TMAD/s on RTX 5090 @ M=8192, N=262144, K=128, r=128  
**Strategy:** Incremental, non-destructive phases — each independently testable, each with clear go/no-go gates  
**Risk Philosophy:** Zero consensus-breaking changes until Phase 5+. Every phase maintains byte-identical transcript output.

---

## Executive Summary

This plan synthesizes findings from all subagent analyses into a phased roadmap targeting **700–800+ TMAD/s** (2.4–2.8× current). The strategy has three pillars:

1. **Kernel-level gains** (Phases 1–3): GeForce warp-specialized kernel v2, grouped GEMM, tile-knob optimization — these attack the ~64% gap between measured tensor throughput and RTX 5090's rated 838 INT8 TOPS.
2. **Pipeline depth gains** (Phases 1–2): Triple-buffering, deferred share GPU work, async seed conveyor, async job install — these eliminate GPU idle time when share proofs and job switches occur.
3. **System-level gains** (Phases 1, 4–5): OS tuning, GPU clock lock, fleet operations — these stabilize and compound kernel/pipeline gains.

**Realistic combined target with all phases:** ~500–600 TMAD/s (1.7–2×). This is the **proven ceiling** given Pearl's epilogue overhead (transcript XOR + BLAKE3 PoW consumes ~10–20% of CTA time, limiting Amdahl scaling).

**Stretch target of 700–800+ TMAD/s** requires: (a) all kernel+pipeline gains compounding, (b) grouped GEMM at batch≥4 adopted by product, (c) potential consensus-level changes (Phase 5) that modify the search space without breaking proof identity. This is ambitious and carries non-trivial risk.

**Absolute ceiling:** True 3× (~900 TMAD/s) exceeds sustained RTX 5090 INT8 peak with Pearl epilogue. The epilogue is ~10–20% of CTA time and cannot be eliminated — it's the proof.

---

## Phase 0: Baseline & Measurement

**Goal:** Establish accurate, repeatable baselines and profiling infrastructure before any changes.

**Why first:** Every subsequent phase requires verified measurement. Without this, we cannot distinguish real gains from noise.

### 0.1: Establish Baseline Measurements

**Changes:**
- Run `propminer --bench 300 --rtx5090 --gpus 0` for 30 minutes minimum (5 full minutes of stable data after 30s grace period)
- Record: TMAD/s, protocol H/s, batch_ms, tops_pct, tiles_per_sec, iters_per_sec
- Run at least 5 separate bench sessions, compute mean and standard deviation
- Document the exact config: M=8192, N=262144, K=128, r=128, batch=1, graph_batch=1, cluster_m=1

**File:** `results/baseline_YYYYMMDD.log`

**Success criteria:** Standard deviation < 2% across 5 runs. If > 2%, investigate thermal/power variation before proceeding.

**Risk:** None. Pure measurement.

**Time:** 2–4 hours (including setup and analysis).

### 0.2: Nsight Compute Profiling

**Changes:**
- Run `scripts/profile_gemm_ncu.sh` on the current consumer kernel to establish baseline metrics
- Key metrics to capture:
  - **SM throughput** (% active warps) — target: > 80%
  - **Tensor pipe utilization** (% of tensor cores active) — target: > 60%
  - **Global load efficiency** (bytes/transaction ratio) — target: > 80%
  - **Achieved occupancy** (warps/SM vs theoretical max) — target: > 90%
  - **Shared memory bank conflicts** — target: 0
  - **Register usage per thread** — target: < 255 (max for sm_120a)
  - **Pipeline stall reasons** — identify which stalls consume most cycles
  - **Epilogue time** (transcript XOR + BLAKE3) — target: < 20% of CTA time

**File:** `results/ncu_baseline_YYYYMMDD.ncu-rep`

**Success criteria:** Complete NCU report with all metrics above captured.

**Risk:** None. Pure profiling.

**Time:** 1–2 hours.

### 0.3: Nsight Systems Timeline

**Changes:**
- Run `nsys profile --trace=cuda,nvtx -o results/nsys_baseline propminer --bench 60 --rtx5090 --gpus 0`
- Capture full timeline showing: kernel launches, PCIe transfers, stream ordering, CPU-GPU overlap
- Identify: GPU idle periods, CPU bottlenecks, stream serialization points

**Success criteria:** Timeline reveals all GPU idle periods with timestamps and causes.

**Risk:** None.

**Time:** 1 hour.

### 0.4: Custom Metrics Instrumentation

**Changes:**
- Add `PROPMINER_PROFILE=1` env flag that enables detailed per-batch timing in `gpu_worker.cpp`
- When enabled, log per-batch: compute_ms, seed_upload_ms, share_rebuild_ms, half_wait_ms, sigma_swap_ms
- Add per-stream timing via `cudaEventRecord` on all 5 streams (ping, pong, third, seed_copy, install)
- Log these metrics every 10 batches to `results/profile_metrics.csv`

**Files to modify:**
- `src/host/pearl/gpu_worker.cpp` — lines 930–1051 (queue_batch), add timing events
- `src/host/pearl/gpu_worker.cpp` — lines 1202–1390 (process_share_trigger_impl), add timing events
- `src/host/pearl/gpu_worker.cpp` — lines 1408–1768 (run loop), add periodic CSV output

**Success criteria:** CSV shows per-batch breakdown of all pipeline stages.

**Risk:** Low. Only active when `PROPMINER_PROFILE=1`. Negligible overhead from `cudaEventRecord` calls.

**Rollback:** Unset `PROPMINER_PROFILE=1`.

**Time:** 1–2 days.

### 0.5: Benchmarking Scripts

**Changes:**
- Create `scripts/bench_stable.sh` — runs 5 consecutive 5-minute benches, reports mean ± stddev
- Create `scripts/bench_compare.sh BASELINE NEW` — compares two benchmark runs, reports % change
- Create `scripts/bench_matrix.sh` — sweeps batch × graph_batch × cluster_m automatically
- Create `scripts/verify_transcript.sh` — runs self-test with both consumer and geforce kernels, verifies byte-identical output

**Files to create:**
- `scripts/bench_stable.sh`
- `scripts/bench_compare.sh`
- `scripts/bench_matrix.sh`
- `scripts/verify_transcript.sh`

**Success criteria:** `bench_stable.sh` produces consistent results across runs. `bench_compare.sh` correctly identifies improvements and regressions.

**Risk:** None. Script creation only.

**Time:** 1 day.

### Phase 0 Summary

| Metric | Target |
|--------|--------|
| Baseline TMAD/s | ~290 ± 5 (established) |
| NCU metrics | Complete baseline report |
| NSYS timeline | Complete pipeline timeline |
| Profile metrics | Per-batch breakdown available |
| Benchmark scripts | All 4 scripts tested |

**Total Phase 0 time:** 3–5 days

---

## Phase 1: Quick Wins (Low Risk, High Impact)

**Goal:** Achieve 330–370 TMAD/s (+15–25%) with minimal risk through configuration tweaks and already-built features.

**Risk level:** Low. All changes are either environment variables, compile-time flags, or minor host-code adjustments.

### 1.1: Enable GeForce Warp-Specialized Kernel (v1)

**What:** Flip the default from `consumer` kernel to `geforce` kernel. The GeForce kernel is already in source, already compiled in, already tested. It uses warp-specialized TMA producer + IMMA consumers.

**Expected gain:** +10–25% (330–375 TMAD/s from 300 baseline)

**Files to modify:**
- `third_party/pearl-gemm/csrc/capi/Makefile` — set `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=ON` (may already be ON)
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — ensure geforce kernel is the default dispatch path

**Testing:**
1. `./scripts/verify_geforce_transcript.sh` — run 100 trials, verify byte-identical output vs consumer kernel
2. `./build/propminer --self-test --rtx5090 --gpus 0` — must pass with geforce kernel
3. `PROP_MINER_SELF_TEST_PROD=1 ./build/propminer --self-test --rtx5090 --gpus 0` — production shape self-test
4. `./scripts/bench_stable.sh` — compare against Phase 0 baseline, verify ≥+10%

**Go/no-go gates:**
- G1: Transcript memcmp 100% pass (100 trials)
- G2: Self-test passes
- G3: Bench ≥+10% vs consumer baseline
- G4: NCU shows ≥+8 percentage points SM throughput

**Rollback:** `PEARL_GEMM_KERNEL=consumer` (no rebuild needed).

**Time:** 1–2 days.

### 1.2: GeForce Kernel v2 Implementation

**What:** Evolve v1 using NVIDIA CUTLASS SM120 patterns (examples 79/87). Key improvements:
- Replace manual barriers with `PipelineTmaAsync`
- Pingpong schedule: overlap transcript XOR + PoW epilogue on tile N while TMA prefetches tile N+1
- Asymmetric TMA: B is VRAM-resident per σ — load B once per CTA/replay; A every nonce
- Deeper stages: explore KBLOCK=64, STAGES=3/4 within 99 KB SMEM cap

**Expected gain:** +10–25% on top of v1 (370–470 TMAD/s total)

**Files to modify:**
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` — main kernel
- `third_party/pearl-gemm/csrc/consumer/tma_tile_loader.cuh` — TMA loader
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — dispatch v2 path

**Reference (read-only):** CUTLASS `examples/79_blackwell_geforce_gemm/`, `sm120_mma_tma.hpp`

**Testing:**
1. Extend `verify_geforce_transcript.sh` — v2 vs v1 memcmp, 100 trials
2. `--self-test` + `PROP_MINER_SELF_TEST_PROD=1`
3. `./scripts/profile_gemm_ncu.sh` — verify Tensor Active ↑, global_ld ↓
4. `--bench 120` — ≥+10% vs v1 baseline
5. 24–48h pool soak — rejected shares < 1%

**Go/no-go gates:**
- G1: Transcript memcmp 100% pass (v2 vs v1)
- G2: Self-test + prod-shape self-test
- G3: SMEM ≤ 101376 B (99 KB + margin)
- G4: NCU: ≥+8 pct points SM throughput vs v1
- G5: Bench ≥+10% vs v1
- G6: 24h pool: <1% rejects
- G7: CUDA graphs + warmup clean

**Rollback:** `PEARL_GEMM_KERNEL=geforce` (v1) or `consumer`. Requires rebuild to compile out v2.

**Risk:** Moderate. Transcript byte-identity is the critical risk. SMEM budget (99 KB/block, not 164 KiB) is the second.

**Time:** 4–6 weeks (1 CUDA engineer + 5090 hardware).

### 1.3: Tile Knob Optimization

**What:** Sweep compile-time kernel knobs to find optimal configuration for RTX 5090.

**Sweep axes:**
- **KBLOCK:** 64 vs 128 (64 fits more stages in 99 KB)
- **STAGES:** 2 vs 3 vs 4 (3 stages at KBLOCK=64 = 72 KB, fits 99 KB ceiling)
- **SWIZZLE_BITS:** 0 vs 1 vs 2 vs 3
- **MIN_BLOCKS:** 1 vs 2 vs 4 (higher = less occupancy but better overlap)
- **CLUSTER_M:** 1 vs 2 vs 4

**Expected gain:** +5–15% per axis, +10–20% if combined wins compound

**Files to modify:**
- `third_party/pearl-gemm/csrc/capi/Makefile` — change compile-time knobs
- `scripts/tune_blackwell_knobs.sh` — add KBLOCK=64 and STAGES=3/4 to sweep
- `src/host/pearl/rtx5090_profile.h` — update comments (99 KB, not 164 KiB)

**Testing:**
1. `./scripts/tune_blackwell_knobs.sh` — run full sweep
2. Compare results in `build/tune_full_raw.tsv`
3. Verify transcript identity for each winning combo
4. 24h pool soak with winner

**Rollback:** Rebuild with different knobs. Cache in `~/.cache/propminer/kernel_knobs.json`.

**Time:** 2–3 days (sweep) + 1 day (analysis).

### 1.4: Enable Triple Buffering

**What:** Add third `HalfBuffers` workspace so share reconstruction never blocks GEMM rotation.

**Expected gain:** +0–3% typical, +2–5% share-heavy scenarios

**Files to modify:**
- `src/host/pearl/gpu_worker.cpp` — lines 276–304 (triple buffer allocation), lines 1521–1646 (mining loop)
- `src/host/pearl/env_flags.h` — `triple_buffer_enabled()` already exists

**Current state:** The third half-buffer struct member exists in `GpuWorker`. The VRAM guard (`triple_vram_headroom_ok()`) exists. The mining loop ring buffer logic exists. What's missing: full integration of the third half into the mining loop rotation and share path pinning.

**Testing:**
1. `PROPMINER_TRIPLE_BUFFER=1 ./build/propminer --self-test --rtx5090 --gpus 0`
2. `PROPMINER_TRIPLE_BUFFER=1 ./scripts/bench_stable.sh` — compare vs dual-buffer
3. Monitor `pipeline` line in stats output — verify `half_wait_count` drops to 0
4. 24h pool soak

**Rollback:** `PROPMINER_TRIPLE_BUFFER=0` (no rebuild needed).

**Time:** 3–5 days (integration + validation).

### 1.5: GPU Clock Lock & OS Tuning

**What:** Lock GPU clocks and power to eliminate frequency scaling variance.

**Commands to add to entrypoint:**
```bash
nvidia-smi -pm 1                          # Power management mode
nvidia-smi -lgc <min_clock>,<max_clock>   # Lock clocks (e.g., 525,2100 for 5090)
echo never > /sys/kernel/mm/transparent_hugepage/enabled  # Disable THP
sysctl -w vm.swappiness=1
```

**Expected gain:** +5–15% sustained (eliminates thermal/power throttling variance)

**Files to modify:**
- Docker entrypoint script or systemd service file
- `scripts/run.sh` or `scripts/run_mining.sh`

**Testing:**
1. `nvidia-smi -q | grep -E "Graphics|Boost|Power"` — verify clocks locked
2. `./scripts/bench_stable.sh` — compare vs unlocked
3. Monitor stability over 24h

**Rollback:** Remove clock lock commands from entrypoint.

**Risk:** Low. Standard practice for mining deployments.

**Time:** 1 day.

### 1.6: L2 Cache Configuration

**What:** Optimize L2 cache fetch granularity for GDDR7 on RTX 5090.

**Current state:** Already set to 128 bytes at `gpu_worker.cpp` line 264. This is correct for GDDR7.

**Potential addition:** Try `cudaFuncCachePreferShared` for the GEMM kernel to prioritize shared memory over L1.

**Files to modify:**
- `gpu_worker.cpp` — add `cudaFuncSetCacheConfig(gemm_func, cudaFuncCachePreferShared)` after kernel load

**Expected gain:** Uncertain, possibly +1–3%. Profile first.

**Testing:**
1. `--bench 120` with and without L2 config change
2. NCU: compare L2 hit rate

**Rollback:** Remove the `cudaFuncSetCacheConfig` call.

**Time:** 1 day.

### Phase 1 Summary

| Lever | Expected Gain | Cumulative Range | Risk | Time |
|-------|--------------|-------------------|------|------|
| 1.1 GeForce v1 kernel | +10–25% | 330–375 TMAD/s | Low | 1–2 days |
| 1.2 GeForce v2 kernel | +10–25% | 370–470 TMAD/s | Moderate | 4–6 weeks |
| 1.3 Tile knob optimization | +5–15% | 390–530 TMAD/s | Low | 3 days |
| 1.4 Triple buffering | +0–5% | 390–555 TMAD/s | Low | 3–5 days |
| 1.5 GPU clock lock | +5–15% | 410–640 TMAD/s | Low | 1 day |
| 1.6 L2 cache config | +1–3% | 415–660 TMAD/s | Low | 1 day |

**Phase 1 total time:** 2–3 weeks (sequential) or 1–2 weeks (parallel where possible)

---

## Phase 2: Architecture Improvements

**Goal:** Achieve 450–550 TMAD/s through pipeline depth and memory access optimizations.

**Risk level:** Medium. Changes touch kernel-host interaction patterns and require careful synchronization testing.

### 2.1: CUDA Stream Overlap Optimization

**What:** Ensure all 5 streams (ping, pong, third, seed_copy, install) are maximally concurrent.

**Current state:** 5 NON_BLOCKING streams exist. Seed copy overlaps with compute. Async install overlaps with mining. But stream priority and concurrent kernel execution are not explicitly configured.

**Changes:**
- Use `cudaStreamCreateWithPriority()` for compute streams (high priority) vs copy streams (normal priority)
- Verify concurrent kernel execution is enabled (it should be by default on CUDA 12.x)
- Add `cudaDeviceSetLimit(cudaLimitMemPoolReuseZeroOutResources, 1)` to reduce memory pool fragmentation
- Profile with Nsight Systems to verify all streams are truly concurrent

**Files to modify:**
- `src/host/pearl/gpu_worker.cpp` — stream creation (lines 256–259)

**Expected gain:** +2–5% (if streams are currently serializing)

**Testing:**
1. Nsight Systems timeline — verify no stream serialization
2. `--bench 120` with and without changes
3. Verify no new race conditions in 24h pool soak

**Rollback:** Revert to `cudaStreamCreate()` without priority.

**Time:** 2–3 days.

### 2.2: Stream-Split Pre-GEMM

**What:** Move pre-GEMM phases (LCG int7 fill, tensor_hash, noise generation) to a separate stream so they overlap with the GEMM compute on the compute streams.

**Current state:** All 5 stages (lcg → tensor_hash → commitment → noise_gen → noisy_gemm) serialize on one stream per iteration. GEMM is ~85–92% of iteration time, so overlapping the 8–15% pre-GEMM work saves real time.

**Changes:**
- Create dedicated `pregemm_stream_` for pre-GEMM work
- Use `cudaStreamWaitEvent` to order: pre-GEMM completes → GEMM starts
- Capture pre-GEMM work in CUDA graph alongside GEMM

**Files to modify:**
- `src/host/pearl/gpu_worker.cpp` — add `pregemm_stream_`, modify `queue_batch()` to launch pre-GEMM on separate stream
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — ensure pre-GEMM kernels support separate stream parameter

**Expected gain:** +1–5% (depends on pre-GEMM fraction of total iter time)

**Testing:**
1. Profile pre-GEMM time with Nsight Compute
2. `--bench 120` with and without stream-split
3. Verify transcript identity (pre-GEMM must produce identical results)

**Rollback:** Fall back to single-stream execution.

**Risk:** Medium. Stream ordering bugs can produce silent transcript corruption.

**Time:** 3–5 weeks.

### 2.3: Ptr-Array Grouped GEMM

**What:** One kernel launch, multiple nonces. Instead of launching the GEMM kernel N times (once per nonce), pass a device pointer-array and launch once with all nonces.

**Current state:** `batch=1` in production. Each batch iteration launches one `transcript_gemm` kernel (~65,536 CTAs). CUDA graphs capture the full batch but still produce N separate kernel launches inside the graph.

**Expected gain:** 0% at batch=1 (prod default); +10–20% if batch≥4 and kernel efficient

**Files to modify:**
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` — grouped GEMM path
- `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` — grouped GEMM path
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — workspace ptr pools, `noisy_gemm` dispatch
- `src/host/pearl/gpu_worker.cpp` — header slot layout for multiple winners

**Testing:**
1. Serial `pearl_capi_iter` vs grouped — byte-identical transcript per group
2. `--self-test` + `PROP_MINER_SELF_TEST_PROD=1`
3. Matrix: `{graph on/off} × {batch 1,4,8,16} × {grouped on/off}`
4. PoW: exactly one header slot fires per hit
5. Pool canary 24h

**Rollback:** `PEARL_GEMM_GROUPED_GEMM=0` or `PROPMINER_BATCH=1`.

**Dependency:** Requires product decision to adopt `PROPMINER_BATCH > 1` in production.

**Go/no-go gate:** G0 = Product commits to batch≥4. Without this, grouped GEMM has 0% ROI.

**Time:** 4–6 weeks.

### 2.4: Memory Access Pattern Optimization

**What:** Optimize A matrix layout and access patterns to improve memory coalescing and reduce global memory transactions.

**Current state:** A matrix is regenerated every nonce via LCG int7 fill. B matrix is VRAM-resident per σ. Access patterns are determined by CUTLASS tile layout.

**Potential changes:**
- Verify A matrix is in column-major order (optimal for CTA row distribution)
- Check if B matrix fits in L2 cache for resident-B scenarios
- Explore shared memory tiling patterns that reduce duplicate B loads
- Profile global memory transaction efficiency with NCU

**Files to modify:**
- `third_party/pearl-gemm/csrc/consumer/tma_tile_loader.cuh` — TMA loader optimization
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` — access patterns

**Expected gain:** +2–5% (if current access patterns are suboptimal)

**Testing:**
1. NCU: global load efficiency, shared memory throughput
2. `--bench 120` with and without changes
3. Transcript identity verification

**Rollback:** Revert to previous access patterns.

**Time:** 1–2 weeks.

### Phase 2 Summary

| Lever | Expected Gain | Cumulative Range | Risk | Time |
|-------|--------------|-------------------|------|------|
| 2.1 Stream overlap | +2–5% | 460–575 TMAD/s | Low | 2–3 days |
| 2.2 Stream-split pre-GEMM | +1–5% | 465–600 TMAD/s | Medium | 3–5 weeks |
| 2.3 Grouped GEMM | +10–20% | 510–720 TMAD/s | High | 4–6 weeks |
| 2.4 Memory access patterns | +2–5% | 515–750 TMAD/s | Low | 1–2 weeks |

**Phase 2 total time:** 6–10 weeks (sequential)

---

## Phase 3: Algorithm-Level Optimizations

**Goal:** Achieve 550–700 TMAD/s through algorithmic shortcuts that maintain consensus compatibility.

**Risk level:** Medium-High. Any change to the search space or computation must preserve proof identity.

### 3.1: Search Space Restructuring

**What:** Explore whether the nonce search space can be restructured to improve GPU utilization without changing the mathematical proof.

**Key question:** Can we partition the nonce space in a way that eliminates tail waves (the 106 partial slots at 65,536 CTAs ÷ 170 SMs)?

**Current state:** 65,536 CTAs ÷ 170 SMs = 385.5 waves, with 106 tail slots. This is ~0.16% waste — negligible. But if we can find an N value that divides evenly (wave-aligned), we might gain a small amount.

**Potential changes:**
- N=262144: 65,536 CTAs, 106 tail slots (0.16% waste)
- N=261120: 65,280 CTAs, 0 tail slots (0% waste) — already a candidate in `pick_n_for_vram()`
- N=260480: 65,120 CTAs, 0 tail slots — if VRAM allows
- Compare wave-aligned vs non-wave-aligned N values

**Files to modify:**
- `src/host/pearl/rtx5090_profile.h` — `pick_n_for_vram()` candidates

**Expected gain:** +0.1–0.5% (negligible, but free)

**Testing:**
1. `--bench 120` for each N candidate
2. Compare TMAD/s, batch_ms, and SM occupancy

**Rollback:** Revert to current N candidates.

**Time:** 1 day.

### 3.2: Hybrid Computation Approach

**What:** Explore whether certain phases of the computation can be offloaded to different execution units (e.g., tensor cores vs CUDA cores) for better utilization.

**Current state:** All computation uses SM80 int8 MMA atoms on tensor cores. Epilogue (transcript XOR + BLAKE3) runs on CUDA cores.

**Potential changes:**
- Move more of the transcript XOR into tensor cores (if mathematically valid)
- Use warp-level primitives for BLAKE3 compression (warp-lane batched parent compress)
- Explore if any pre-GEMM work can use tensor cores instead of CUDA cores

**Files to modify:**
- `third_party/pearl-gemm/csrc/blackwell/pow_utils.hpp` — BLAKE3 compression
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — kernel dispatch

**Expected gain:** +1–3% (if epilogue is a significant bottleneck)

**Testing:**
1. Transcript identity verification (critical — BLAKE3 must be byte-identical)
2. NCU: compare warp execution efficiency
3. `--bench 120` with and without changes

**Rollback:** Revert to previous computation paths.

**Risk:** High. BLAKE3 changes can silently break proof identity.

**Time:** 2–3 weeks.

### 3.3: Optimized N Selection Strategy

**What:** Dynamically select the optimal N value based on real-time VRAM availability and performance characteristics.

**Current state:** N is selected once at startup via `pick_n_for_vram()`. It's fixed for the lifetime of the process.

**Potential changes:**
- Monitor VRAM usage and adjust N downward if pressure is detected
- Profile performance at different N values to find the "sweet spot"
- Consider whether smaller N with higher batch size outperforms larger N with batch=1

**Files to modify:**
- `src/host/pearl/rtx5090_profile.h` — `pick_n_for_vram()`
- `src/host/pearl/gpu_worker.cpp` — dynamic N adjustment logic

**Expected gain:** +0–5% (depends on current N being suboptimal)

**Testing:**
1. Profile at multiple N values
2. Compare TMAD/s vs VRAM utilization
3. 24h soak test with dynamic adjustment

**Rollback:** Fix N at startup value.

**Time:** 1–2 weeks.

### Phase 3 Summary

| Lever | Expected Gain | Cumulative Range | Risk | Time |
|-------|--------------|-------------------|------|------|
| 3.1 Search space restructuring | +0.1–0.5% | 555–705 TMAD/s | Low | 1 day |
| 3.2 Hybrid computation | +1–3% | 556–710 TMAD/s | High | 2–3 weeks |
| 3.3 Optimized N selection | +0–5% | 556–745 TMAD/s | Low | 1–2 weeks |

**Phase 3 total time:** 3–4 weeks

---

## Phase 4: Novel Approaches

**Goal:** Achieve 650–800+ TMAD/s through creative algorithm variants and experimental approaches.

**Risk level:** High. These approaches are unproven and may not deliver expected gains. Each requires careful risk assessment and rollback planning.

### 4.1: Creative Algorithm Variants (from Subagent 6 Research)

**What:** Implement the creative algorithm variants identified in the cross-repo analysis, including:
- **Asymmetric TMA DMA:** B is VRAM-resident per σ — load B once per CTA/replay; A every nonce. This eliminates redundant B TMA transfers.
- **Pingpong schedule:** Overlap heavy transcript epilogue with next-tile MMA. This attacks the Amdahl tail (~10–20% of iter time).
- **Warp-lane batched parent compress:** For BLAKE3 Merkle tree reduction during σ-install.

**Files to modify:**
- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` — pingpong schedule
- `third_party/pearl-gemm/csrc/consumer/tma_tile_loader.cuh` — asymmetric TMA
- `third_party/pearl-blake3/` — warp-lane batched compress

**Expected gain:** +5–15% (cumulative from multiple variants)

**Risk assessment:**
| Variant | Risk | Rollback |
|---------|------|----------|
| Asymmetric TMA | Medium — B must be correctly resident | Revert to symmetric TMA |
| Pingpong schedule | Medium — transcript must be byte-identical | Revert to sequential schedule |
| Warp-lane batched compress | High — BLAKE3 must be byte-identical | Revert to standard compress |

**Testing:**
1. Each variant tested independently before combining
2. Transcript identity verification for each variant
3. Combined variant tested with 100-trial memcmp
4. 48h pool soak with all variants combined

**Time:** 4–6 weeks.

### 4.2: CCCL Share Compaction

**What:** Use CUB/Thrust primitives for share proof compaction instead of hand-rolled atomic operations.

**Current state:** Share proof uses hand-rolled `atomicAdd`/`atomicCAS` for winner signaling.

**Potential changes:**
- `cuda::atomic_ref` + barriers for winner signaling in `write_host_signal_header()`
- `cub::DeviceSelect::Flagged` for compact PoW hits without atomic storms
- `cub::DeviceScan::InclusiveSum` for prefix-sum over candidate tiles

**Files to modify:**
- `third_party/pearl-gemm/csrc/blackwell/pow_utils.hpp` — winner signaling
- `src/host/pearl/gpu_worker.cpp` — share compaction

**Expected gain:** <2% steady-state (only relevant when batching grows)

**Risk:** Low. CCCL primitives are well-tested.

**Rollback:** Revert to hand-rolled atomics.

**Time:** 3–4 days.

### 4.3: Triton Offline Autotune

**What:** Use Triton (Python GPU kernel DSL) for offline exploration of sub-tile pipeline configurations. Port winning configs back to CUDA.

**Current state:** All tuning done via `tune_prod_5090.sh` which sweeps compile-time knobs.

**Potential changes:**
- Write Triton prototypes for sub-tile pipeline exploration
- Use Triton's autotuner to find optimal configurations
- Port winning configs back to CUDA/CUTLASS

**Files to create:**
- `research/triton_autotune/` — Triton prototype kernels

**Expected gain:** +2–5% (if Triton finds configs that manual sweeps miss)

**Risk:** Medium. Triton cannot produce production kernels (cannot guarantee byte-identical transcript). Only used for research.

**Rollback:** Discard Triton experiments, revert to manual tuning.

**Time:** 2–3 weeks.

### 4.4: GPU Clock Overclocking

**What:** Push GPU clocks beyond factory limits for additional compute throughput.

**Current state:** GPU clocks are factory-default or locked (Phase 1.5).

**Potential changes:**
- Use `nvidia-smi -lgc` to set custom clock targets (e.g., 2200 MHz memory, 2150 MHz graphics)
- Use `nvidia-smi -pl` to increase power limit (if PSU allows)
- Monitor thermals closely — RTX 5090 is sensitive to temperature

**Expected gain:** +5–15% (depends on headroom and cooling)

**Risk:** Medium. Can reduce GPU lifespan if thermals are poor. Requires adequate PSU and cooling.

**Rollback:** Revert to factory clocks.

**Time:** 1–2 days (testing only).

### Phase 4 Summary

| Lever | Expected Gain | Cumulative Range | Risk | Time |
|-------|--------------|-------------------|------|------|
| 4.1 Creative variants | +5–15% | 680–920+ TMAD/s | High | 4–6 weeks |
| 4.2 CCCL compaction | <2% | 685–925+ TMAD/s | Low | 3–4 days |
| 4.3 Triton autotune | +2–5% | 690–930+ TMAD/s | Medium | 2–3 weeks |
| 4.4 GPU overclocking | +5–15% | 725–1070+ TMAD/s | Medium | 1–2 days |

**Phase 4 total time:** 6–10 weeks

---

## Phase 5: Pushing Boundaries

**Goal:** Achieve 800–1000+ TMAD/s through the most ambitious optimizations and potential consensus-level changes.

**Risk level:** High. These changes may affect proof identity or consensus compatibility. Each requires extensive testing and community coordination if consensus changes are involved.

### 5.1: Consensus-Level Search Space Changes

**What:** Explore whether the Pearl protocol's search space can be modified to improve GPU utilization.

**Key questions:**
- Can the nonce space be partitioned differently to improve wave alignment?
- Can the rank (r) parameter be adjusted to improve GEMM efficiency?
- Can the K dimension be modified while maintaining proof identity?

**Current state:** M=8192, N=262144, K=128, r=128. These are protocol-defined and may not be changeable without consensus.

**Potential changes:**
- Adjust r from 128 to 64 or 256 (if protocol allows)
- Adjust K from 128 to 64 or 256 (if protocol allows)
- Explore whether different M values improve tile utilization

**Files to modify:**
- `src/host/pearl/rtx5090_profile.h` — default M/N/K/r values
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — kernel dispatch for new dimensions

**Expected gain:** +5–20% (if optimal dimensions are found)

**Risk:** Critical. Any change to M/N/K/r must be validated against the Pearl protocol specification. If the protocol requires specific values, changes are not possible without consensus.

**Testing:**
1. Verify against Pearl protocol specification
2. Test against live pool (if pool accepts modified parameters)
3. Full transcript identity verification
4. Extended pool soak test (1+ week)

**Rollback:** Revert to original M/N/K/r values.

**Time:** 2–4 weeks.

### 5.2: Multi-GPU Coordination

**What:** Optimize work distribution across multiple GPUs to eliminate idle time and maximize aggregate throughput.

**Current state:** Each GPU mines independently with static nonce partitioning. No coordination between GPUs.

**Potential changes:**
- Dynamic nonce partitioning based on real-time GPU performance
- Work stealing when one GPU falls behind
- Shared sigma context to reduce VRAM usage and σ-install latency

**Files to modify:**
- `src/host/pearl/worker_orchestrator.cpp` — work distribution logic
- `src/host/pearl/gpu_worker.cpp` — inter-GPU communication

**Expected gain:** +5–15% on multi-GPU rigs (0% on single GPU)

**Risk:** Medium. Inter-GPU communication adds complexity and potential failure modes.

**Rollback:** Revert to independent GPU mining.

**Time:** 4–6 weeks.

### 5.3: Long-Term Evolution Strategy

**What:** Plan for the future of PropMiner as GPU architecture evolves beyond Blackwell.

**Key considerations:**
- **Hopper (sm_90):** WGMMA instructions, different TMA patterns. May require separate kernel implementations.
- **Ada Lovelace (sm_89):** Current generation for many datacenter GPUs. Different tensor core architecture.
- **Future Blackwell variants:** RTX 6090 (if released) may have different SM count and memory bandwidth.

**Strategy:**
1. Maintain separate kernel implementations for each architecture
2. Abstract architecture-specific code behind compile-time dispatch
3. Keep the host code architecture-agnostic
4. Invest in automated testing that runs on each target architecture

**Files to modify:**
- Architecture-specific kernels in `third_party/pearl-gemm/csrc/`
- CMakeLists.txt for multi-architecture builds
- Testing infrastructure for cross-architecture validation

**Expected gain:** Long-term sustainability and adaptability

**Risk:** Low (architectural, not performance-critical)

**Time:** Ongoing.

### Phase 5 Summary

| Lever | Expected Gain | Cumulative Range | Risk | Time |
|-------|--------------|-------------------|------|------|
| 5.1 Consensus changes | +5–20% | 840–1200+ TMAD/s | Critical | 2–4 weeks |
| 5.2 Multi-GPU coordination | +5–15% (multi-GPU) | 840–1200+ TMAD/s | Medium | 4–6 weeks |
| 5.3 Long-term evolution | N/A (sustainability) | — | Low | Ongoing |

**Phase 5 total time:** 6–10 weeks

---

## Risk Management

### What Could Go Wrong at Each Phase

| Phase | Primary Risks | Detection Method |
|-------|--------------|------------------|
| 0 | N/A (measurement only) | N/A |
| 1 | Transcript drift (v2 kernel), VRAM OOM (triple-buffer) | Transcript memcmp, nvidia-smi VRAM monitoring |
| 2 | Stream ordering bugs, grouped GEMM transcript errors | Nsight Systems timeline, transcript memcmp |
| 3 | BLAKE3 identity break, suboptimal N selection | Transcript memcmp, benchmark comparison |
| 4 | Unproven variants underdeliver, GPU instability | Pool share rejection rate, benchmark comparison |
| 5 | Consensus rejection, pool incompatibility | Pool share rejection, community feedback |

### How to Detect Problems Early

1. **Transcript identity verification:** Run `verify_transcript.sh` after every kernel change. 100-trial memcmp is the gold standard.
2. **Share rejection rate:** Monitor pool-side accepted vs rejected shares. Any increase > 1% indicates a problem.
3. **Benchmark comparison:** Run `bench_compare.sh` after every change. Regressions > 5% warrant investigation.
4. **Nsight profiling:** Run NCU after every kernel change to detect occupancy drops or efficiency regressions.
5. **24h pool soak:** Every phase must pass a 24-hour pool soak test before promotion to production.

### Rollback Procedure

1. **Environment variable rollback:** Most features have `PROPMINER_*` kill switches. Unset or set to 0.
2. **Compile-time rollback:** Rebuild with different CMake/make flags (e.g., `PEARL_GEMM_KERNEL=consumer`).
3. **Git revert:** `git revert <commit>` for any code change. Each phase should be in a separate branch/commit.
4. **Process restart:** After rollback, restart the mining process. No data migration needed.

### Maintaining Compatibility with Existing Pools

1. **Protocol compliance:** Every change must be validated against the Pearl protocol specification.
2. **Share proof identity:** Transcript must be byte-identical to the reference implementation.
3. **Pool testing:** Test against both gRPC and Stratum pool endpoints.
4. **Community coordination:** If consensus-level changes are proposed, coordinate with pool operators and the Pearl community.

---

## Integration Strategy

### Ensuring New Code Doesn't Break Existing Functionality

1. **Transcript identity gate:** Every kernel change must pass 100-trial transcript memcmp against the reference implementation.
2. **Self-test gate:** Every build must pass `--self-test` and `PROP_MINER_SELF_TEST_PROD=1`.
3. **Benchmark gate:** Every change must show ≥ 0% regression in `--bench` tests.
4. **Pool soak gate:** Every phase must pass a 24-hour pool soak test with < 1% share rejection rate.

### Testing Methodology

| Test Type | Frequency | Description |
|-----------|-----------|-------------|
| Unit tests | Every build | `propminer_tests` — core logic tests |
| Reference tests | Every build | `propminer_ref_tests` — host-side golden tests |
| Benchmark tests | Every change | `--bench 300` — performance regression detection |
| Transcript tests | Every kernel change | 100-trial memcmp against reference |
| Pool soak | Every phase | 24–48h live pool testing |
| Stress test | Every phase | 72h continuous mining with job switches |

### Code Review Process

1. **Kernel changes:** Must be reviewed by someone with CUDA expertise. NCU results must be included in the PR.
2. **Host code changes:** Must include benchmark comparison showing no regression.
3. **Consensus-level changes:** Must include protocol specification analysis and community coordination record.
4. **All changes:** Must include test results (unit, benchmark, transcript) in the PR description.

### Version Control Strategy

1. **Feature branches:** Each phase implemented on a separate branch (`phase-1-quick-wins`, `phase-2-architecture`, etc.)
2. **Commits:** Small, atomic commits with clear messages. Each commit should be independently buildable and testable.
3. **Tags:** Tag each phase completion (`v1.0-phase-1-complete`, etc.)
4. **Release candidates:** After each phase, create a release candidate (`v1.0-rc1`, `v1.0-rc2`, etc.) for testing.

---

## Success Criteria

### Phase-by-Phase Metrics

| Phase | Minimum Viable | Target | Stretch |
|-------|---------------|--------|---------|
| 0 | Baseline established | Baseline ± 2% stddev | Full profiling infrastructure |
| 1 | 330 TMAD/s | 370 TMAD/s | 400 TMAD/s |
| 2 | 400 TMAD/s | 500 TMAD/s | 550 TMAD/s |
| 3 | 450 TMAD/s | 550 TMAD/s | 650 TMAD/s |
| 4 | 550 TMAD/s | 700 TMAD/s | 850 TMAD/s |
| 5 | 700 TMAD/s | 800 TMAD/s | 1000+ TMAD/s |

### Overall Success Criteria

| Metric | Target |
|--------|--------|
| Final TMAD/s | ≥ 700 (minimum), ≥ 800 (target), ≥ 1000 (stretch) |
| Share rejection rate | < 1% over 48h pool soak |
| Transcript identity | 100% pass (100-trial memcmp) |
| Stability | 72h continuous mining without restart |
| VRAM headroom | > 2 GiB free at all times |

---

## Timeline Estimates

### Sequential Timeline (Conservative)

| Phase | Time | Cumulative |
|-------|------|------------|
| 0. Baseline & Measurement | 5 days | 5 days |
| 1. Quick Wins | 10 days | 15 days |
| 2. Architecture Improvements | 30 days | 45 days |
| 3. Algorithm Optimizations | 20 days | 65 days |
| 4. Novel Approaches | 30 days | 95 days |
| 5. Pushing Boundaries | 30 days | 125 days |

**Total:** ~25 weeks (6 months)

### Parallel Timeline (Aggressive)

| Phase | Time | Cumulative |
|-------|------|------------|
| 0. Baseline & Measurement | 5 days | 5 days |
| 1. Quick Wins (1.1, 1.3, 1.4, 1.5, 1.6) | 7 days | 12 days |
| 1.2 GeForce v2 (parallel with Phase 2) | 6 weeks | 6 weeks |
| 2. Architecture Improvements | 21 days | 42 days |
| 3. Algorithm Optimizations (parallel with Phase 4) | 14 days | 56 days |
| 4. Novel Approaches | 21 days | 77 days |
| 5. Pushing Boundaries (parallel with Phase 4) | 21 days | 98 days |

**Total:** ~14 weeks (3.5 months) for 700+ TMAD/s

---

## Resource Requirements

### Hardware

| Resource | Quantity | Purpose |
|----------|----------|---------|
| RTX 5090 | 1+ | Development and testing |
| Linux machine | 1 | Native Linux build environment (not WSL) |
| NVIDIA driver | 570+ | sm_120a support |
| CUDA toolkit | 12.8+ | nvcc, Nsight Compute, Nsight Systems |
| PSU | 1200W+ | RTX 5090 power requirements |
| Cooling | Adequate | Sustained 100% GPU utilization |

### Software

| Resource | Version | Purpose |
|----------|---------|---------|
| CMake | 3.24+ | Build system |
| GCC | 13+ | C++17 compilation |
| nvcc | 12.8+ | CUDA compilation |
| Rust | 1.75+ | pearl-mining-capi, pearl-blake3 |
| Nsight Compute | Latest | Kernel profiling |
| Nsight Systems | Latest | Timeline profiling |
| Triton | Latest | Offline autotune research (Phase 4) |

### Personnel

| Role | Effort | Phases |
|------|--------|--------|
| CUDA engineer | 1 FTE | 1.2, 2.2, 2.3, 4.1 |
| Host/C++ engineer | 0.5 FTE | 1.4, 2.1, 2.4, 3.2, 5.2 |
| QA/testing | 0.25 FTE | All phases |
| Protocol researcher | 0.1 FTE | 3.1, 5.1 |

---

## Decision Points and Go/No-Go Criteria

### Gate G0: Phase 0 Complete
- **Criteria:** Baseline TMAD/s established with < 2% stddev, NCU/NSYS profiling complete, benchmark scripts tested
- **Decision:** Proceed to Phase 1?

### Gate G1: GeForce v1 Default
- **Criteria:** Transcript memcmp 100% pass, self-test passes, bench ≥+10% vs consumer
- **Decision:** Flip geforce kernel to default?

### Gate G2: GeForce v2 Ready
- **Criteria:** All 8 gates (G1–G8) from the GeForce v2 plan pass
- **Decision:** Promote v2 to default?

### Gate G3: Grouped GEMM Adopted
- **Criteria:** Product commits to batch≥4, grouped GEMM transcript identity verified
- **Decision:** Deploy grouped GEMM to production?

### Gate G4: Triple Buffering Effective (already enabled)
- **Criteria:** `half_wait_count` drops to 0 in production, TMAD/s improves ≥1%
- **Status:** Triple buffer is ON by default since July 2026. Monitor `half_wait_count` and TMAD/s to validate.

### Gate G5: Consensus Changes Viable
- **Criteria:** Pearl protocol specification allows modified M/N/K/r, pool operators agree
- **Decision:** Implement consensus-level changes?

### Gate G6: 700 TMAD/s Achieved
- **Criteria:** Stable 700+ TMAD/s over 72h continuous mining, < 1% share rejection
- **Decision:** Declare victory, optimize for stability?

### Gate G7: 1000 TMAD/s Achieved (Stretch)
- **Criteria:** Stable 1000+ TMAD/s over 72h continuous mining, < 1% share rejection
- **Decision:** Document and publish results?

---

## Fallback Strategies

### If Kernel Optimizations Underdeliver

1. **Profile first:** Use NCU to identify the actual bottleneck (SM throughput? memory bandwidth? epilogue time?)
2. **Adjust approach:** If SM throughput is the bottleneck, focus on occupancy. If memory bandwidth is the bottleneck, focus on access patterns. If epilogue is the bottleneck, focus on pingpong overlap.
3. **Accept diminishing returns:** If kernel improvements plateau below 400 TMAD/s, focus on pipeline depth (triple-buffer, grouped GEMM) and system-level optimizations (clock lock, OS tuning).

### If Transcript Identity Fails

1. **Isolate the change:** Bisect commits to find the exact change that breaks identity.
2. **Check byte ranges:** Use `diff` on transcript output to identify which bytes differ (header? data? padding?).
3. **Revert and investigate:** Revert the change, investigate the root cause, then re-attempt with a different approach.
4. **Never ship broken transcript:** This is the hard line. No performance gain is worth breaking proof identity.

### If VRAM Constraints Prevent Triple Buffering

1. **Reduce N:** Lower N reduces VRAM usage, freeing space for triple buffer.
2. **Optimize memory layout:** Review HalfBuffers allocation for any wasted space.
3. **Accept dual-buffer:** If triple buffer is not feasible, focus on reducing share proof time (Phase 2.2: stream-split pre-GEMM).

### If Pool Rejects Shares

1. **Check rejection reason:** Pool provides specific rejection reasons (verify-fail, vardiff, superseded, etc.).
2. **Isolate the feature:** Disable features one at a time to identify the culprit.
3. **Adjust vardiff:** Lower share difficulty to reduce rejection rate during testing.
4. **Rollback:** If rejection rate > 5%, rollback immediately.

### If 1000+ TMAD/s Is Not Achievable

1. **Accept 700–800 TMAD/s as victory:** This is still a 2.4–2.8× improvement over the baseline.
2. **Document limitations:** Clearly document why 1000+ TMAD/s is not achievable (hardware limits, protocol constraints, epilogue overhead).
3. **Focus on stability:** A stable 700 TMAD/s is better than an unstable 900 TMAD/s.

---

## Appendix A: Environment Variable Reference

| Variable | Default | Purpose | Kill Switch |
|----------|---------|---------|-------------|
| `PROPMINER_BATCH` | 1 | Mine batch size | Set to 1 |
| `PROPMINER_GRAPH_BATCH` | 1 | CUDA graph sub-batch depth | Set to 1 |
| `PEARL_GEMM_CONSUMER_CLUSTER_M` | 1 | GEMM cluster dimension | Set to 1 |
| `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL` | ON | Enable GeForce kernel | Set to OFF |
| `PEARL_GEMM_KERNEL` | geforce | Kernel selection | Set to consumer |
| `PROPMINER_TRIPLE_BUFFER` | OFF | Enable triple buffering | Set to 0 |
| `PROPMINER_DEFER_SHARE_GPU` | ON | Defer share GPU work | Set to 0 |
| `PROPMINER_ASYNC_SEED` | ON | Async seed upload | Set to 0 |
| `PROPMINER_ASYNC_JOB_INSTALL` | ON | Async job installation | Set to 0 |
| `PROPMINER_BCOL_CACHE` | ON | Targeted B-column expansion | Set to 0 |
| `PROPMINER_N_CAP` | 262144 | N dimension cap | Set to desired value |
| `PROPMINER_STALL_RESTART_MS` | 30000 | GPU stall timeout | Set to 0 to disable |
| `PROPMINER_PROFILE` | OFF | Enable detailed profiling | Set to 0 |

---

## Appendix B: File Modification Index

| File | Phases Affected |
|------|----------------|
| `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.cu` | 1.1, 1.2, 2.3, 4.1 |
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | 1.1, 2.3 |
| `third_party/pearl-gemm/csrc/consumer/tma_tile_loader.cuh` | 1.2, 2.4, 4.1 |
| `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` | 1.1, 1.2, 2.2, 2.3, 3.2, 3.3 |
| `third_party/pearl-gemm/csrc/capi/Makefile` | 1.1, 1.3 |
| `src/host/pearl/gpu_worker.cpp` | 1.4, 2.1, 2.2, 2.4, 3.3, 4.2, 5.2 |
| `src/host/pearl/worker_orchestrator.cpp` | 5.2 |
| `src/host/pearl/rtx5090_profile.h` | 1.3, 3.1, 3.3, 5.1 |
| `src/host/pearl/env_flags.h` | 1.4 |
| `src/host/pearl/env_tuning.h` | 1.3 |
| `third_party/pearl-blake3/` | 3.2, 4.1 |
| `scripts/tune_blackwell_knobs.sh` | 1.3 |
| `scripts/tune_prod_5090.sh` | All phases |
| `scripts/bench_stable.sh` | Phase 0 (new) |
| `scripts/bench_compare.sh` | Phase 0 (new) |
| `scripts/verify_transcript.sh` | Phase 0 (new) |

---

## Appendix C: Performance Projection Model

**Base TMAD/s:** ~290 (current baseline)

**Compound model:** Each phase's gain compounds on the previous phase's result.

```
Phase 0:  290 TMAD/s  (baseline)
Phase 1:  290 × 1.25 = 362 TMAD/s  (GeForce v1 + knobs + triple + clock lock)
Phase 2:  362 × 1.08 = 391 TMAD/s  (stream overlap + pre-GEMM + memory patterns)
         362 × 1.18 = 427 TMAD/s  (if grouped GEMM at batch≥4)
Phase 3:  391 × 1.03 = 403 TMAD/s  (search space + hybrid + N optimization)
Phase 4:  403 × 1.12 = 451 TMAD/s  (creative variants + Triton + overclocking)
Phase 5:  451 × 1.15 = 519 TMAD/s  (consensus changes + multi-GPU)
```

**Conservative total:** ~520 TMAD/s (1.8× baseline)  
**Optimistic total:** ~600+ TMAD/s (2.1× baseline)  
**Stretch total:** ~700–800+ TMAD/s (2.4–2.8× baseline) — requires all levers to compound and grouped GEMM at batch≥4

**To reach 1000+ TMAD/s:** Requires consensus-level changes that fundamentally alter the search space, or a paradigm shift in how the Pearl proof is computed. This is not guaranteed and may require protocol-level coordination.

---

*Plan generated July 2026 from comprehensive analysis by 8 subagents. All gain estimates are based on CUDA best practices, NVIDIA documentation, and empirical data from existing PropMiner configurations. Actual results may vary based on hardware, driver version, pool conditions, and implementation quality.*
