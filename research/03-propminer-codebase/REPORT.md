# PropMiner Codebase Performance Report

**Scope:** RTX 5090 (`sm_120a`) production mining path  
**Sources:** `src/host/pearl/gpu_worker.{h,cpp}`, `src/host/pearl/worker_orchestrator.cpp`, `src/host/pearl/rtx5090_profile.h`, `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu`, `performance optimizations/01–08`, `scripts/tune_*.sh`  
**Date:** 2026-07-06

---

## 1. Resource Consumption Estimates

PropMiner’s steady-state mining loop is deliberately **GPU-isolated**: matrix B and noise state stay VRAM-resident across iterations (`SigmaContext::install` in `gpu_worker.cpp`), the host uploads an **8-byte seed** per batch on a dedicated copy stream, and D2H traffic occurs only on rare PoW hits. The observed nvidia-smi profile on a tuned 5090 at production shape (M=8192, N=262144, K=128, batch≈4–20) aligns with the ranges below.

| Resource | Typical utilization | Mechanism (code evidence) |
|----------|--------------------:|---------------------------|
| **SM / Tensor Core** | **~35–40%** | 65,536 CTAs per GEMM at N=262144 (`Rtx5090Profile::tiles` → `(8192/128)×(262144/256)`), 170 physical SMs (`rtx5090_profile.h`), ~386 waves with tail 26 (`worker_orchestrator.cpp` startup log). Kernel uses `__launch_bounds__(256, 1)` and **128 int32 accumulators per thread** (`kFragSize` in `transcript_gemm_kernel.cu`), capping occupancy at ~1 block/SM. Legacy `mma.sync m16n8k32` (SM80 atom) under-fills Blackwell tensor pipes vs native `tcgen05` (doc 01). |
| **Memory bandwidth (HBM)** | **~15–25%** | K=128 → **one K-tile** per GEMM (`K_TILES = K / kBK = 1`). Each CTA loads 48 KiB/stage (A 16 KiB + B 32 KiB) through a 2-stage `cp.async` pipeline; B is read from VRAM each iter but not re-uploaded from host. Pure-miner mode never materializes C in HBM (`gpu_worker.cpp` L70–72, L419–421). Register-resident transcript + accumulators shift the kernel toward **compute/tensor-pipe bound**, not DRAM-saturated. |
| **PCIe** | **<1%** | `upload_next_seed_async()` copies **8 bytes** (`sizeof(uint64_t)`) on `seed_copy_stream_`, overlapped with ping/pong compute (`gpu_worker.cpp` L531–541, L784–787). At 20 batches/s ≈ 160 B/s — far below Gen4/Gen5 x16 capacity (doc 07). Share D2H (leaf CVs, row slices, opened leaves) is rare and pinned. |
| **CPU** | **<5%** | `disable_cpu_mining = true` by default (`worker_orchestrator.h`, `main.cpp --rtx5090`). Hot path: `cudaEventQuery` spin-wait (`spin_wait_batch_event`, L38–64), O(batch) header scan, 64-bit nonce increment. Proof build + gRPC run on separate orchestrator threads (`share_sender_thread_func`, `network_thread_func`). No CPU GEMM or hashing on the mining path. |

**Interpretation:** Low SM% is **not** idle hardware—it reflects register-limited occupancy, short K-depth (one mainloop trip), and tensor-pipe under-utilization on a legacy IMMA path. Low HBM% confirms the design goal: keep B resident, minimize host traffic, fuse PoW into the GEMM kernel (`launch_transcript_gemm_headless` in `transcript_kernel.cuh`).

---

## 2. Aggressive vs Conservative Choices Today

### Aggressive (performance-first, production-oriented)

| Choice | Location | Rationale |
|--------|----------|-----------|
| **N up to 262144** from VRAM | `Rtx5090Profile::pick_n_for_vram()` | 8× more CTAs per GEMM vs N=32768; ~300.8 vs ~299.2 TMAD/s measured (`transcript_gemm_kernel.cu` L150–151). |
| **Ping-pong double buffering + CUDA graphs** | `GpuWorker::run()`, `prepare_graph()` | Overlap batch *N* with batch *N+1*; extended graph path decouples seed H2D from captured launch (`iter_batch_graph_prepare_ex`). |
| **Dedicated seed copy stream** | `seed_copy_stream_`, `seed_copy_done_event_` | PCIe “conveyor belt” — zero CPU hashing, overlap 8-byte upload with GEMM. |
| **L2 fetch granularity 128 B** | `GpuWorker` ctor, `cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128)` | Tuned for M=8192, N=32768+ sequential GEMM traffic. |
| **Swizzle&lt;3,4,3&gt;, KBLOCK=128, STAGES=2** | Blackwell defaults in `transcript_gemm_kernel.cu` | +0.5% TMAD/s vs swizzle-2 on 5090 bench. |
| **cluster_m=2 default** | `Rtx5090Profile::kProdDefaultClusterM`, `worker_orchestrator.cpp` L440–441 | Thread-block clusters along M; tuned via `tune_cluster_sweep.sh`. |
| **Headless in-kernel PoW** | `consumer::launch_transcript_gemm_headless` | Avoids transcript gmem spill + separate finalize kernel. |
| **Batch sweep to 20** | `kMaxMineBatch`, `tune_mine_batch.sh` | Amortizes graph launch / host scan overhead. |
| **VRAM-resident B** | `SigmaContext::install`, shared `resident()` buffers | Eliminates per-iter B H2D. |

### Conservative (correctness, portability, operability)

| Choice | Location | Rationale |
|--------|----------|-----------|
| **SM80 `mma.sync` atom on sm_120a** | `PEARL_CONSUMER_MMA_ATOM_TYPE = SM80_16x8x32_S32S8S8S32_TN` | Byte-identical `partition_C` vs H100 WGMMA (`transcript_gemm_kernel.cu` L9–17). No tcgen05/TMEM yet (doc 01, 06). |
| **Fixed 128×256 tile geometry** | `#error` guards on BM/BN | Proof-canonical; changing tiles breaks share extraction. |
| **Default batch = 4** | `Rtx5090Profile::kDefaultMineBatch` | WSL2/Salad-safe; batch=1 won short benches (comment L37–38). |
| **Bench N cap 32768** | `kBenchMaxN`, `main.cpp` | Finish ≥1 batch inside 120 s Docker bench window (doc 03). |
| **`PROPMINER_DEFER_SHARE_GPU` default off** | `defer_share_gpu_enabled()` | Share deferral implemented but not default; steady gain +0–2% (doc 04). |
| **`MIN_BLOCKS=1`** | `PEARL_CONSUMER_DEFAULT_MIN_BLOCKS` | Conservative occupancy hint; sweep may find `MIN_BLOCKS=2` wins (doc 05). |
| **`cp.async` load path, TMA scaffold only** | `PEARL_CONSUMER_USE_TMA_EXPERIMENT` | TMA compile path exists but production uses cooperative cp.async (doc 02). |
| **25% VRAM headroom in fit model** | `shape_fits_vram()` `bytes += bytes/4` | Avoid OOM on ping-pong + graphs + leaf CVs; may leave N on table. |
| **A regen + leaf-CV rehash on share hit** | `process_share_trigger()` L610–635 | Required for proof validity (`a_merkle_mismatch` prevention). |
| **SeedGenerator removed** | doc 08 | 8-byte async upload already fully overlaps; PRNG ring added complexity for ~0% gain. |

**Net posture:** The stack is **aggressive on host/GEMM orchestration and grid sizing**, but **conservative on consensus-critical kernel semantics** (tile shape, transcript layout, SM80 MMA atom, share proof path).

---

## 3. Ranked Bottlenecks (with Code Evidence)

| Rank | Bottleneck | Impact | Evidence |
|:----:|------------|--------|----------|
| **1** | **Legacy IMMA on Blackwell (no tcgen05/TMEM)** | **High** — primary 2× lever | `transcript_gemm_sm120.cu` includes consumer kernel with SM80 atom; doc 01 cites ~300 TMAD/s ceiling vs B200 tcgen05 path at 840+ TMAD/s (different SKU/shape). 128 int32/thread register accumulators (`kFragSize`, L252–266). |
| **2** | **Register pressure → low occupancy** | **High** | `__launch_bounds__(256, PEARL_CONSUMER_MIN_BLOCKS)` with default `MIN_BLOCKS=1` (L224). 128 acc + 16 transcript u32 + address tensors per thread. Explains ~35–40% SM util despite 65k CTAs. |
| **3** | **K=128 single-tile mainloop** | **Medium** | `K_TILES = K / kBK = 1` (L294). Minimal latency hiding window; load/MMA/transcript snapshot compete in one trip. Cannot change K without protocol change. |
| **4** | **Compile-time knob sub-optimality** | **Medium (+5–15%)** | Shipped: STAGES=2, SWIZZLE=3, MIN_BLOCKS=1. Doc 05 / `tune_blackwell_knobs.sh` sweeps KBLOCK×STAGES×SWIZZLE×MIN_BLOCKS with `--self-test` gate. |
| **5** | **cp.async vs TMA load issue** | **Medium (+10–25%)** | All 256 threads issue 16 B cp.async copies (L314–366); TMA scaffold present (`PEARL_CONSUMER_USE_TMA_EXPERIMENT`) but not production (doc 02). |
| **6** | **σ-install / first graph capture latency** | **Low steady-state, high startup** | `install_sigma()` allocates workspace, captures graph per half (`prepare_graph`). Doc 03: first batch 60–120 s at N=262144. Not a sustained hashrate cap. |
| **7** | **Inline share GPU work on mining thread** | **Low (+0–2%)** | `handle_trigger` / `process_share_trigger`: stream sync, A regen, tensor_hash, D2H (`gpu_worker.cpp` L598–697). Rare but blocks half reuse; defer path exists (`PROPMINER_DEFER_SHARE_GPU=1`). |
| **8** | **Tail wave (CTAs mod 170)** | **Low** | N=262144 → 65,536 CTAs, tail 26 (`worker_orchestrator.cpp` L462–469). Doc 03 notes tail cheaper than shrinking N. |
| **9** | **PCIe / PSU / CPU** | **Negligible steady-state** | Doc 07, doc 08. Fix only if misconfigured (x8 lane, power throttle). |

---

## 4. 2×–3× Roadmap (Proof-Safe)

All phases preserve Pearl’s **byte-identical transcript contract**: fixed 128×256 CTA tile, SM80-compatible `partition_C` mapping, in-kernel BLAKE3 headless path, and A regen on share hits. Validation gate: `propminer --self-test` (used in `tune_kernel_knobs_common.sh`).

### Phase A — Config & runtime tuning (target: **1.1–1.2×**, weeks)

1. **Deploy N=262144 in mine mode** — code complete; validate via `run_mining.sh` + startup logs (doc 03).
2. **Run `scripts/tune_prod_5090.sh`** on bare metal:
   - Step 1: `tune_blackwell_knobs.sh` → `~/.cache/propminer/kernel_knobs.json`
   - Step 2: `tune_cluster_sweep.sh` → `autotune.json` (cluster_m, carveout)
   - Step 3: `tune_mine_batch.sh` → `mine_batch.json`
3. **Enable tune cache in production:** `PROPMINER_USE_TUNE_CACHE=1` (default in `run_mining.sh`).
4. **Enable share deferral** after soak test: `PROPMINER_DEFER_SHARE_GPU=1`.

**Proof safety:** Runtime-only; no kernel geometry change.

### Phase B — Load-path & occupancy (target: **1.3–1.5×** cumulative, months)

1. **Implement TMA gmem→smem** behind `PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma` (doc 02). Keep identical `SmemLayoutA/B`, same MMA/transcript loop.
2. **Ship winning compile knobs** from sweep (likely candidates: `MIN_BLOCKS=2`, `KBLOCK=64` + `STAGES=3` where smem ≤ 164 KiB per `tune_knob_smem_ok`).
3. **Wave-aware N selection** optional: prefer 262144 over 65280 only when VRAM fit confirms (already in `pick_n_for_vram`).

**Proof safety:** TMA must target same swizzled smem addresses; `--self-test` + byte-compare transcript vs baseline.

### Phase C — Native Blackwell MMA (target: **2×–3×** cumulative, high effort)

1. **Port B200 Design B** (`transcript_gemm_sm100.cu`) → `transcript_gemm_sm120_tcgen05.cu` (doc 01).
2. **TMEM accumulator readback** replaces 128 int32/thread registers; warp-specialized producer (TMA) + MMA consumer.
3. **Dual dispatch:** runtime flag selects consumer SM80 path vs tcgen05 path until byte-identity proven on full grid (`probe_sm80_layout.cu` methodology).
4. **Gate default** behind compile flag + pool canary shares.

**Proof safety:** Highest risk phase. Requires exhaustive transcript slot mapping proof; doc 06 confirms SM120 int8 atom swap alone yields ~0% — tcgen05 is the real ISA migration.

### Phase D — Not on roadmap (0% expected gain)

- Re-enable `SeedGenerator` (removed, doc 08).
- SM120 CUTLASS int8 atom rename without new Operation (doc 06).
- PCIe Gen5 upgrade on healthy Gen4 x16 rig (doc 07).

### Combined multiplier math

| Phase | Incremental | Cumulative (illustrative) |
|-------|-------------|---------------------------|
| Baseline | — | 1.0× (~300 TMAD/s) |
| A: N + tune caches | +10–20% | ~1.15× |
| B: TMA + knobs | +15–30% | ~1.4× |
| C: tcgen05/TMEM | +30–80% | **~2.0–2.5×** (stretch **3×** with ideal silicon + full N=262144 + batch 20) |

---

## 5. Performance Docs 01–08 Status Map

| Doc | Title | Primary paths | Status | Expected gain | Notes |
|-----|-------|---------------|--------|---------------|-------|
| **01** | Native tcgen05 / TMEM GEMM | `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm100.cu` (ref), `consumer/transcript_gemm_kernel.cu` (baseline) | **Planning** | +30–80% | Largest lever; transcript mapping proof required. |
| **02** | TMA consumer tile loads | `consumer/transcript_gemm_kernel.cu`, `consumer/tma_tile_loader.cuh` | **Scaffold only** | +10–25% | `#error` guard when `LOAD_POLICY=tma`; cp.async is production. |
| **03** | Production N=262144 | `rtx5090_profile.h`, `main.cpp`, `worker_orchestrator.cpp` | **Largely complete** | +0–3% TMAD/s; 8× work/GEMM | Mine vs bench split implemented; rollout = config + validation. |
| **04** | Defer share GPU work | `gpu_worker.cpp` (`PROPMINER_DEFER_SHARE_GPU`) | **Implemented, default off** | +0–2% | Side thread + `share_jobs_pending` half-lock. |
| **05** | Kernel knob autotune | `scripts/tune_blackwell_knobs.sh`, `tune_kernel_knobs_common.sh`, `kernel_knob_cache.h` | **Tooling complete; defaults pre-sweep** | +5–15% | Self-test gated sweep; cache at `~/.cache/propminer/kernel_knobs.json`. |
| **06** | SM120 native CUTLASS int8 atom | `blackwell/transcript_gemm_sm120.cu`, `portable_int8_gemm.cu` | **Blocked (CUTLASS gap)** | ~0% | SM80 IMMA is correct production atom; flag is readiness hook. |
| **07** | PCIe Gen5 + PSU headroom | `gpu_worker.cpp` (seed path), hardware | **Ops guide** | 0–10% if broken | Steady-state PCIe irrelevant; fix throttling/miswiring only. |
| **08** | SeedGenerator evaluation | `gpu_worker.cpp` (live path) | **Closed — dead code removed** | ~0% | 8-byte pinned async upload supersedes ring buffer. |

---

## Appendix: Key File Index

| Component | Path |
|-----------|------|
| GPU worker (ping-pong, graphs, seeds, shares) | `src/host/pearl/gpu_worker.cpp`, `gpu_worker.h` |
| Orchestration (tune cache, cluster, batch, pool) | `src/host/pearl/worker_orchestrator.cpp` |
| RTX 5090 shape / VRAM / batch constants | `src/host/pearl/rtx5090_profile.h` |
| Consumer fused GEMM + transcript kernel | `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` |
| sm_120 wrapper | `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu` |
| One-shot prod tuning | `scripts/tune_prod_5090.sh` |
| Compile knob sweep | `scripts/tune_blackwell_knobs.sh`, `scripts/tune_kernel_knobs_common.sh` |
| cluster_m + runtime autotune | `scripts/tune_cluster_sweep.sh` |
| Mine batch sweep | `scripts/tune_mine_batch.sh` |
| Perf doc series | `performance optimizations/01`–`08` |

---

*Synthesis from PropMiner source and internal perf docs. Steady-state TMAD/s baseline ~300 at M=8192, N=262144 per consumer kernel bench comments; H/s reporting uses DAF-normalized tiles/s (`GpuWorker::run` L828–831).*
