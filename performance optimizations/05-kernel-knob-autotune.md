# 05 — Kernel Knob Autotune (RTX 5090)

**Target:** NVIDIA GeForce RTX 5090 (GB202, CC 12.0, sm_120a)  
**Scope:** Compile-time consumer GEMM knobs — `MIN_BLOCKS`, `STAGES`, `SWIZZLE_BITS`, `KBLOCK` — validated with end-to-end hashrate sweeps and Nsight Compute (`ncu`) on real hardware.  
**Expected gain:** +5–15% hashrate over current production defaults when the winning combination differs from shipped CMake defaults.

---

## 1. Executive Summary

PropMiner’s RTX 5090 path runs a fused int8 transcript GEMM (`transcript_gemm_kernel_consumer`) compiled into `libpearl_gemm_capi.so`. Performance is sensitive to **compile-time pipeline knobs** that trade shared-memory residency, register pressure, LDSM bank conflicts, and occupancy against Tensor Core throughput.

Today’s production defaults (`KBLOCK=128`, `STAGES=2`, `SWIZZLE_BITS=3`, `MIN_BLOCKS=1`, `cp_async`) were chosen from Alpha/RunPod headless benchmarks and are baked into `CMakeLists.txt`, `Dockerfile`, and `scripts/build_and_benchmark.sh`. Early sweeps suggest **+5–15%** is recoverable on physical 5090 silicon by:

| Knob | Hypothesis on 5090 |
|------|-------------------|
| `MIN_BLOCKS=2` | Raises `__launch_bounds__` occupancy hint; may hide launch latency and improve SM pipe utilization when register pressure allows 2 CTAs/SM |
| `STAGES` (2 vs 3) | More stages increase smem and can exceed the ~164 KiB/SM budget at `KBLOCK=128`; stage-3 + `KBLOCK=64` may win on memory-bound grids |
| `SWIZZLE_BITS` (2 vs 3) | Alpha tuning measured ~0.5% win for `<3,4,3>` over `<2,4,3>` at M=8192, N=262144; worth re-validating per driver |
| `KBLOCK` (64 vs 128) | Halves per-stage smem, enabling higher `STAGES` or `MIN_BLOCKS=2`; may reduce cp.async efficiency on the K loop |

**What this plan does not cover:** runtime-only knobs already handled by `GpuTuner` (M/N shape, batch, CUDA graphs, `CLUSTER_M`, carveout). Those are orthogonal and currently gated behind `PROPMINER_AUTOTUNE=1`.

**Deliverable:** a reproducible sweep + ncu workflow, cached winning knobs per GPU UUID, and CMake/Docker defaults updated after `--self-test` passes on the winner.

---

## 2. Current Default Knobs

All values below are what ships today for `PEARL_GEMM_ARCH=blackwell` / `CMAKE_CUDA_ARCHITECTURES=120a`.

### 2.1 Compile-time kernel knobs (Makefile → `-DPEARL_CONSUMER_*`)

| CMake / Makefile variable | Maps to | Production default | Valid range | Notes |
|---------------------------|---------|-------------------|-------------|-------|
| `PEARL_GEMM_BLACKWELL_BM` | `PEARL_CONSUMER_BM` | **128** | 128 only | Proof-canonical; Makefile **errors** on any other value |
| `PEARL_GEMM_BLACKWELL_BN` | `PEARL_CONSUMER_BN` | **256** | 256 only | Proof-canonical; Makefile **errors** on any other value |
| `PEARL_GEMM_BLACKWELL_KBLOCK` | `PEARL_CONSUMER_KBLOCK` | **128** | 64, 128 | smem K-tile per pipeline stage |
| `PEARL_GEMM_BLACKWELL_STAGES` | `PEARL_CONSUMER_STAGES` | **2** | 2, 3, 4 | cp.async pipeline depth |
| `PEARL_GEMM_BLACKWELL_SWIZZLE_BITS` | `PEARL_CONSUMER_SWIZZLE_BITS` | **3** | 2, 3 | CUTLASS `Swizzle<B,4,3>` for A/B smem |
| `PEARL_GEMM_BLACKWELL_MIN_BLOCKS` | `PEARL_CONSUMER_MIN_BLOCKS` | **1** | ≥ 1 | Second arg to `__launch_bounds__(256, N)` |
| `PEARL_GEMM_BLACKWELL_LOAD_POLICY` | `PEARL_CONSUMER_USE_TMA_EXPERIMENT` | **cp_async** | `cp_async`, `tma` | TMA path is **scaffolded but `#error`s at compile** |
| `PEARL_GEMM_BLACKWELL_CP_ASYNC_CACHE_ALWAYS` | `PEARL_CONSUMER_CP_ASYNC_CACHE_ALWAYS` | *(unset → 0)* | 0, 1 | A/B use `CACHEGLOBAL` when 0 |
| `PEARL_GEMM_BLACKWELL_B_CP_ASYNC_CACHE_ALWAYS` | `PEARL_CONSUMER_B_CP_ASYNC_CACHE_ALWAYS` | *(unset → inherits A)* | 0, 1 | Independent B-tile cache policy |

**Kernel source defaults** (used when CMake vars are unset) in `transcript_gemm_kernel.cu` under `PEARL_GEMM_BLACKWELL`:

```cpp
#define PEARL_CONSUMER_DEFAULT_SWIZZLE_BITS 3
#define PEARL_CONSUMER_DEFAULT_STAGES 2
#define PEARL_CONSUMER_DEFAULT_KBLOCK 128
#define PEARL_CONSUMER_DEFAULT_MIN_BLOCKS 1
```

**Fixed kernel geometry** (not sweepable without breaking proofs):

- CTA tile: 128 × 256 × 128 (M × N × K)
- Threads per CTA: 256 (8 warps)
- MMA atom: `SM80_16x8x32_S32S8S8S32_TN` (byte-identical partition_C vs H100 WGMMA)
- Transcript slots: 16 per thread

**Shared memory budget formula:**

```
smem_bytes = (kBM + kBN) × kBK × kStages
           = (128 + 256) × KBLOCK × STAGES
           = 384 × KBLOCK × STAGES
```

| KBLOCK | STAGES | smem/CTA | Fits ~164 KiB/SM? |
|--------|--------|----------|-------------------|
| 128 | 2 | 96 KiB | Yes (current default) |
| 128 | 3 | 144 KiB | Marginal |
| 128 | 4 | 192 KiB | No — likely launch failure or driver clamp |
| 64 | 2 | 48 KiB | Yes |
| 64 | 3 | 72 KiB | Yes |
| 64 | 4 | 96 KiB | Yes |

### 2.2 Runtime knobs (env, no rebuild)

| Environment variable | Default | Purpose |
|---------------------|---------|---------|
| `PEARL_GEMM_CONSUMER_CLUSTER_M` | 1 (remote test kit sets **2**) | Thread-block cluster size along M (1, 2, 4) |
| `PEARL_GEMM_CONSUMER_CARVEOUT` | unset (driver default) | L1 vs shared-memory carveout (0–100, `max_l1`, `max_shared`) |
| `PEARL_GEMM_BLACKWELL_CLUSTER_M` | — | Legacy alias for `CONSUMER_CLUSTER_M` |
| `PEARL_GEMM_BLACKWELL_CARVEOUT` | — | Legacy alias for `CONSUMER_CARVEOUT` |

### 2.3 Host / mining profile defaults (`Rtx5090Profile`)

| Parameter | Value |
|-----------|-------|
| Physical SMs | 170 |
| Default shape | M=8192, N=32768 (or largest VRAM-fit N) |
| K, R | 128 |
| Default batch | 20 |
| Tile metadata in `MiningConfig` | bM=128, bN=256, bK=128 |

### 2.4 Where defaults are duplicated

| File | Role |
|------|------|
| `CMakeLists.txt` | `CACHE STRING` defaults passed to pearl-gemm Makefile |
| `Dockerfile` | Production container build args |
| `scripts/build_and_benchmark.sh` | Local baseline build |
| `scripts/remote_test_kit.sh` | Remote validation build |
| `third_party/pearl-gemm/csrc/capi/Makefile` | Authoritative `-DPEARL_CONSUMER_*` injection |

---

## 3. Search Space Definition

### 3.1 Primary sweep (high value, proof-safe)

Fix `BM=128`, `BN=256`, `LOAD_POLICY=cp_async`. Sweep the four knobs called out in this plan:

```
KBLOCK      ∈ {64, 128}
STAGES      ∈ {2, 3}        # extend to {4} only when KBLOCK=64
SWIZZLE     ∈ {2, 3}
MIN_BLOCKS  ∈ {1, 2}
```

**Shortened grid** (as in `remote_test_kit.sh` §8): **16 variants**, ~4 minutes at 15 s/variant.

**Full grid** (as in `tune_blackwell_knobs.sh`, minus invalid BM/BN): **24–32 variants** depending on whether `STAGES=4` is included for `KBLOCK=64`.

### 3.2 Extended sweep (optional second pass)

Only after primary winner is identified and `--self-test` passes:

| Dimension | Values | Prerequisite |
|-----------|--------|--------------|
| `STAGES=4` | with `KBLOCK=64` only | smem ≤ 96 KiB |
| `CP_ASYNC_CACHE_ALWAYS` | 0, 1 | rebuild |
| `B_CP_ASYNC_CACHE_ALWAYS` | 0, 1 | rebuild |
| `CLUSTER_M` | 1, 2, 4 | runtime env, combine with best compile knobs |
| `CARVEOUT` | default, 50, 80 | runtime env |

### 3.3 Explicitly out of scope

| Knob | Reason |
|------|--------|
| `BM`, `BN` | Makefile enforces 128×256; changing breaks proof row/column extraction |
| `LOAD_POLICY=tma` | Compile-time `#error`: TMA mainloop not implemented |
| `PEARL_GEMM_SM120_NATIVE` | Separate track; int8 path still uses SM80 MMA atom today |
| Hopper / B200 arch flags | Must remain sm_120a-only for 5090 |

### 3.4 Knob interaction matrix

```
                    KBLOCK=128                    KBLOCK=64
                 STAGES=2   STAGES=3          STAGES=2/3/4
MIN_BLOCKS=1     baseline   smem-heavy       lower smem, more K-iters
MIN_BLOCKS=2     occupancy  likely OOM       best candidate for occupancy win
SWIZZLE=3        prod def   bank layout      same tradeoffs
SWIZZLE=2        Alpha -0.5%  alt layout     worth measuring on new drivers
```

---

## 4. Autotune Infrastructure — What Exists vs Missing

### 4.1 Exists today

| Component | Location | Capability |
|-----------|----------|------------|
| **CMake knob passthrough** | `CMakeLists.txt` → Makefile | All `PEARL_GEMM_BLACKWELL_*` vars forwarded at build time |
| **Shell sweep (full)** | `scripts/tune_blackwell_knobs.sh` | Nested loops, rebuild per variant, `--speed-test-seconds`, stages best `.so` |
| **Shell sweep (short)** | `scripts/remote_test_kit.sh` §8 | Fixed 128×256, 16 combos, logs to `results/summary.txt` |
| **Runtime autotuner** | `src/host/pearl/gpu_tuner.cpp` | Sweeps M/N, batch, graph, `CLUSTER_M`, carveout; 5 s × 3 repeats per candidate |
| **Autotune cache** | `src/host/pearl/tune_cache.cpp` | Persists runtime tuning to `~/.cache/propminer/autotune.json` |
| **RTX 5090 profile gate** | `src/host/main.cpp` | `--rtx5090` disables autotune unless `PROPMINER_AUTOTUNE=1` |
| **Correctness gate** | `--self-test` | Share verification after kernel changes |
| **ncu hook** | `remote_test_kit.sh` §7 | Profiles `transcript_gemm_kernel_consumer` |

### 4.2 Missing (this plan implements)

| Gap | Impact |
|-----|--------|
| **No runtime kernel-knob autotune** | `GpuTuner` cannot change `KBLOCK`/`STAGES`/etc. without rebuild |
| **Cache ignores compile knobs** | `TuneCache` key uses GPU UUID + driver + `PEARL_GEMM_ARCH`, not knob vector |
| **`tune_blackwell_knobs.sh` BM/BN loop is dead** | Makefile rejects non-128/256; wastes sweep slots |
| **No statistical rigor in shell sweeps** | Single measurement, no outlier drop, no `--self-test` per variant |
| **No ncu-guided pruning** | Sweeps all combos blindly instead of eliminating smem/occupancy losers first |
| **No knob fingerprint in Docker/CMake** | Winning combo must be manually copied back |
| **`PROPMINER_AUTOTUNE` scope mismatch** | Users expect full autotune; only runtime params are tuned |

### 4.3 Target architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Phase A: ncu baseline (1 build, default knobs)             │
│    → identify bottleneck: TC vs L1/TEX vs smem vs occupancy │
├─────────────────────────────────────────────────────────────┤
│  Phase B: compile-time knob sweep (rebuild matrix)          │
│    → 16–32 variants, multi-repeat hashrate, self-test gate  │
├─────────────────────────────────────────────────────────────┤
│  Phase C: runtime overlay (GpuTuner on winner)              │
│    → batch / cluster_m / carveout with PROPMINER_AUTOTUNE=1 │
├─────────────────────────────────────────────────────────────┤
│  Phase D: persist → TuneCache + CMake defaults              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. ncu Profiling Workflow on 5090

Run on the physical RTX 5090 after a Release build with production defaults.

### 5.1 Prerequisites

```bash
# Confirm target silicon
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
# Expect: RTX 5090, 12.0, driver ≥ 570

ncu --version   # Nsight Compute bundled with CUDA 12.8+
```

### 5.2 Capture (matches remote test kit)

```bash
BUILD=build   # or build_remote_test
ncu -o results/profile_baseline \
    --target-processes all \
    --kernel-regex 'regex:transcript_gemm_kernel_consumer' \
    "${BUILD}/propminer" --bench 10 --rtx5090 --gpus 0
```

Optional CSV export for diffing sweep winners:

```bash
ncu -i results/profile_baseline.ncu-rep \
    --csv --page raw \
    --metrics \
      sm__throughput.avg.pct_of_peak_sustained_elapsed,\
      sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
      l1tex__throughput.avg.pct_of_peak_sustained_elapsed,\
      sm__maximum_warps_per_active_cycle_pct,\
      launch__occupancy_limit_shared_mem,\
      launch__occupancy_limit_registers \
    > results/ncu_baseline.csv
```

### 5.3 Metrics to record per knob variant

| Metric | Interpretation | Action if low |
|--------|----------------|---------------|
| `sm__pipe_tensor_cycles_active…` | Tensor Core utilization | Try `MIN_BLOCKS=2`, reduce smem (`KBLOCK=64`) |
| `l1tex__throughput…` | L1/TEX pressure | Try carveout `max_l1`, `SWIZZLE=3` |
| `sm__maximum_warps_per_active_cycle_pct` | Achieved occupancy | Try `MIN_BLOCKS=2` or lower `STAGES` |
| `launch__occupancy_limit_shared_mem` | smem-limited | Drop `STAGES` or `KBLOCK` |
| `launch__occupancy_limit_registers` | reg-limited | Drop `MIN_BLOCKS` to 1 |
| `dram__throughput…` | Memory-bound | Larger N (production shape), `CLUSTER_M=2` |

### 5.4 ncu per sweep candidate (top 3 only)

After Phase B identifies the top 3 hashrate variants, profile each:

```bash
for LABEL in bw-128x256-k128-s2-sw3-mb2-cp_async ...; do
  cp "results/libpearl_gemm_capi_${LABEL}.so" "${BUILD}/libpearl_gemm_capi.so"
  ncu -o "results/profile_${LABEL}" \
      --target-processes all \
      --kernel-regex 'regex:transcript_gemm_kernel_consumer' \
      "${BUILD}/propminer" --bench 10 --rtx5090 --gpus 0
done
```

Compare reports side-by-side in Nsight Compute UI or diff CSV exports. The goal is to confirm the hashrate winner is winning for the **expected** hardware reason (not a measurement artifact).

### 5.5 Expected baseline signature (hypothesis)

Current defaults (`STAGES=2`, `KBLOCK=128`, `MIN_BLOCKS=1`, `SWIZZLE=3`) likely show:

- High Tensor Core activity (>75%) at production N
- Moderate L1/TEX traffic from cp.async + ldmatrix
- Occupancy below theoretical max due to `MIN_BLOCKS=1` and register footprint
- Room to improve via `MIN_BLOCKS=2` if smem/reg limits allow 2 blocks/SM

---

## 6. Sweep Methodology

### 6.1 Benchmark configuration

| Parameter | Recommended | Rationale |
|-----------|-------------|-----------|
| Profile | `--rtx5090 --gpus 0` | Fixed M/N/batch from `Rtx5090Profile` |
| Duration per variant | **15 s** (screen), **30 s** (finalists) | Balances thermal steady-state vs sweep time |
| Repeats | **3** per variant | Matches `GpuTuner::benchmark_stable` |
| Outlier policy | Drop lowest repeat, mean the rest | Same as `gpu_tuner.cpp` |
| Warm-up | First 2 s discarded (optional) | Reduces cold-start graph capture noise |
| Cool-down | 5 s idle between rebuilds | Reduces thermal carry-over |

### 6.2 Statistical significance

Treat two variants as **distinct** only if:

```
mean_fast > mean_slow × 1.005   (≥ 0.5% separation)
AND
(min_fast_repeat > max_slow_repeat) OR (fast_wins 2/3 repeats)
```

At ~300 GH/s equivalent hashrates, 0.5% ≈ 1.5 GH/s — near the noise floor of short benches. For borderline pairs, extend to 60 s × 5 repeats.

Report confidence in `results/summary.txt`:

```
[sweep] bw-128x256-k128-s2-sw3-mb2-cp_async: 312.4 311.8 313.1 H/s → mean=312.4 (best)
[sweep] bw-128x256-k128-s2-sw3-mb1-cp_async: 308.1 307.9 308.5 H/s → mean=308.2
[sweep] delta=+1.4% (significant)
```

### 6.3 Correctness gate (mandatory)

Every variant that clears the hashrate screen **must** pass:

```bash
./build/propminer --self-test --rtx5090 --gpus 0
```

Failure → discard variant regardless of hashrate. Log stderr to `results/selftest-${LABEL}.log`.

### 6.4 Sweep execution order (pruned)

1. Build default → baseline 3×15 s + ncu
2. Sweep `MIN_BLOCKS` × `KBLOCK` × `STAGES` (8 combos) — highest leverage
3. Fix best smem/occupancy point, sweep `SWIZZLE` (2 combos)
4. Re-run top 3 at 30 s × 3 + `--self-test`
5. ncu top 3
6. Optional: overlay `GpuTuner` runtime sweep on winner

### 6.5 Environment isolation

During sweeps, pin:

```bash
export PEARL_GEMM_CONSUMER_CLUSTER_M=2   # or 1 — document which
unset PROPMINER_AUTOTUNE                   # avoid runtime autotune interference
export PROPMINER_BENCH_BATCH=20            # match Rtx5090Profile::kDefaultBatch
```

Lock GPU clocks for comparable runs (document offsets in results); re-run `--self-test` after any clock change.

---

## 7. Caching Results (`tune_cache`)

### 7.1 Current format

File: `~/.cache/propminer/autotune.json` (despite `.json` extension, line-oriented text)

Key: `<gpu_uuid_hex>-<major>.<minor>-drv<driver>-<PEARL_GEMM_ARCH>`

Value (12 fields):

```
M,N,K,R,bM,bN,bK,batch,use_graph,cluster_m,carveout_percent,hashrate
```

**Kernel knobs are not stored.** A driver update or CMake default change can silently invalidate cached runtime params.

### 7.2 Proposed extension

Add optional compile-knob fields to the value line (backward compatible — old lines still parse):

```
...,hashrate,kblock,stages,swizzle_bits,min_blocks,load_policy,knob_hashrate
```

Add **`knob_build_id`** to the cache key:

```
hash(PEARL_GEMM_BLACKWELL_KBLOCK | STAGES | SWIZZLE | MIN_BLOCKS | LOAD_POLICY | git_sha)
```

Or simpler: include knob tuple in key suffix:

```
<uuid>-12.0-drv570-blackwell-k128-s2-sw3-mb2
```

### 7.3 Separate knob cache file (alternative)

`~/.cache/propminer/kernel_knobs.json`:

```json
{
  "key": "<uuid>-12.0-drv570-blackwell",
  "knobs": {
    "kblock": 128, "stages": 2, "swizzle_bits": 3,
    "min_blocks": 2, "load_policy": "cp_async"
  },
  "hashrate": 3.12e11,
  "self_test_passed": true,
  "measured_at": "2026-07-05T21:00:00Z",
  "bench_seconds": 30,
  "repeats": 3
}
```

### 7.4 Invalidation rules

Clear cache when any of:

- `libpearl_gemm_capi.so` rebuild with different knob CMake args
- Driver major version change
- `--self-test` fails on cached knobs
- User sets `PROPMINER_AUTOTUNE=force`

---

## 8. Production Integration (`PROPMINER_AUTOTUNE=1`)

### 8.1 Current behavior

```cpp
// main.cpp — RTX 5090 profile
if (!autotune_env || autotune_env[0] == '\0' || strcmp(autotune_env, "0") == 0)
    cfg.autotune = false;   // default OFF for --rtx5090
```

When enabled, `WorkerOrchestrator` runs `GpuTuner::tune(5.0, 3)` — **runtime only** — and caches to `TuneCache`.

### 8.2 Target behavior

| `PROPMINER_AUTOTUNE` | Action |
|----------------------|--------|
| unset / `0` | Use CMake-baked kernel knobs + `Rtx5090Profile` defaults; skip all autotune |
| `1` | Runtime autotune (current) + **load cached kernel knobs** if present |
| `2` (proposed) | Full sweep: kernel knob matrix + runtime autotune (multi-minute startup) |
| `force` (proposed) | Ignore cache, re-sweep, overwrite |

### 8.3 Startup sequence (proposed `AUTOTUNE=2`)

```
1. Load kernel_knobs cache → if miss, run scripts/tune_blackwell_knobs.sh (short grid)
2. Verify libpearl_gemm_capi.so matches cached knob fingerprint (hash .so or embed build string)
3. Run GpuTuner runtime sweep (batch, cluster_m, carveout)
4. Merge results → TuneCache
5. Log final config to stderr
```

### 8.4 Embed knob manifest in `.so`

Add to `pearl_gemm_capi.cpp`:

```cpp
extern "C" const char* pearl_gemm_build_knobs() {
  return "k128-s2-sw3-mb2-cp_async";  // generated from CMake
}
```

Host reads this at startup to detect knob/cache mismatch.

---

## 9. Implementation Phases

### Phase 0 — Baseline & tooling (1 day)

- [ ] Run `remote_test_kit.sh` with `PROPMINER_SKIP_SWEEP=0`, `PROPMINER_SKIP_BENCH=0`, `PROPMINER_SKIP_NCU=0`
- [ ] Capture baseline hashrate + `profile.ncu-rep` on 5090
- [ ] Fix `tune_blackwell_knobs.sh`: remove BM/BN loops (fixed 128×256), add `--self-test` gate, multi-repeat stats

### Phase 1 — Short sweep on real 5090 (1 day)

- [ ] Execute 16-variant grid from §3.1 at 15 s × 3 repeats
- [ ] Rank by trimmed mean; `--self-test` top 5
- [ ] Document winner in `results/summary.txt`

### Phase 2 — ncu validation (0.5 day)

- [ ] Profile baseline + top 3 winners
- [ ] Confirm bottleneck shift matches expectation (occupancy ↑ for `MIN_BLOCKS=2`, etc.)
- [ ] Reject winners that win on hashrate but show regressions in TC utilization without explanation

### Phase 3 — Cache & host integration (2 days)

- [ ] Extend `TuneCache` or add `KernelKnobCache`
- [ ] Add `pearl_gemm_build_knobs()` export
- [ ] Wire `PROPMINER_AUTOTUNE=2` startup path (optional; `=1` loads cache only)

### Phase 4 — Promote defaults (0.5 day)

- [ ] Update `CMakeLists.txt`, `Dockerfile`, `build_and_benchmark.sh` with winning knobs
- [ ] Update comments in `transcript_gemm_kernel.cu` defaults if they change
- [ ] Run full `remote_test_kit.sh` regression

### Phase 5 — Long-term (optional)

- [ ] Implement TMA load path → unlock `LOAD_POLICY=tma` in sweep
- [ ] Runtime `CLUSTER_M` × knob interaction study
- [ ] CI smoke: build all 16 variants (no run) to catch smem `#error` regressions

---

## 10. Files to Modify

| File | Change |
|------|--------|
| `scripts/tune_blackwell_knobs.sh` | Fix search space (drop BM/BN), add repeats, self-test, CSV output |
| `scripts/remote_test_kit.sh` | Enable sweep by default for tuning runs; export ncu CSV |
| `CMakeLists.txt` | Update `PEARL_GEMM_BLACKWELL_*` defaults after validation |
| `Dockerfile` | Match CMake defaults |
| `scripts/build_and_benchmark.sh` | Match CMake defaults |
| `src/host/pearl/tune_cache.h/.cpp` | Extend key/value for kernel knobs |
| `src/host/pearl/gpu_tuner.h/.cpp` | Optional: `tune_kernel_knobs()` orchestration hook |
| `src/host/main.cpp` | Parse `PROPMINER_AUTOTUNE=2`, log knob manifest |
| `src/host/pearl/worker_orchestrator.cpp` | Load knob cache before runtime autotune |
| `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` | Export `pearl_gemm_build_knobs()` |
| `third_party/pearl-gemm/csrc/capi/CMakeLists.txt` | Generate knob string define from CMake |
| `third_party/pearl-gemm/csrc/consumer/transcript_gemm_kernel.cu` | Update default comments if defaults change |
| `performance optimizations/05-kernel-knob-autotune.md` | This document — update with measured results |

---

## 11. Risks — Wrong Knob Breaks Proof

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Non-canonical BM/BN** | Critical | Makefile already hard-errors; never bypass |
| **MMA atom / tiling change** | Critical | Do not sweep `PEARL_CONSUMER_MMA_ATOM_TYPE` without byte-identity re-verification |
| **Transcript slot layout change** | Critical | `kFragSize`, `kTranscriptSlots` are fixed; knob sweeps must not alter epilogue |
| **Silent numeric drift** | High | Mandatory `--self-test` per variant; compare share proof bytes |
| **smem overflow (`STAGES=4`, `KBLOCK=128`)** | Medium | Build may fail or kernel launches with reduced occupancy; pre-filter via smem formula |
| **MIN_BLOCKS=2 reg spill** | Medium | ncu `launch__occupancy_limit_registers`; discard if TC util drops |
| **TMA experimental path** | High | Currently `#error` at compile — do not enable in production sweep |
| **Thermal / power throttling** | Medium | Multi-repeat with cool-down; lock clocks during compare |
| **Cache stale after rebuild** | Low | Knob fingerprint in cache key + `.so` manifest |
| **Driver-specific optimum** | Low | Cache keyed by driver version; re-sweep on major driver bump |

**Golden rule:** If `--self-test` fails, the variant is **invalid for mining** regardless of hashrate.

---

## 12. Effort & Success Metrics

### 12.1 Effort estimate

| Phase | Engineer-days | GPU wall time |
|-------|---------------|---------------|
| Phase 0 — tooling fixes | 1.0 | 2 h |
| Phase 1 — short sweep | 0.5 | 2–4 h (16 variants × rebuild) |
| Phase 2 — ncu | 0.5 | 1 h |
| Phase 3 — cache/host | 2.0 | — |
| Phase 4 — promote defaults | 0.5 | 1 h regression |
| **Total** | **~4.5 days** | **~8 h GPU time** |

Rebuild dominates wall time (~2–4 min/variant for full `propminer` target). Consider `ccache` and parallel builds on multi-GPU hosts for Phase 1.

### 12.2 Success metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Hashrate uplift vs baseline | **+5–15%** | 60 s bench, same clocks, production N |
| Self-test pass rate | **100%** on shipped knobs | `--self-test` |
| TC utilization (ncu) | ≥ baseline or explained tradeoff | `sm__pipe_tensor_cycles_active…` |
| Sweep reproducibility | ≤ 1% run-to-run variance | 3×30 s on final winner |
| Startup autotune (cached) | < 5 s | `PROPMINER_AUTOTUNE=1` with warm cache |
| Startup autotune (cold) | < 15 min | `PROPMINER_AUTOTUNE=2` short grid |

### 12.3 Acceptance checklist

- [ ] Baseline and winner hashrates recorded in `results/benchmark_hashrate.txt`
- [ ] ncu reports for baseline + winner archived in `results/`
- [ ] `--self-test` passes on promoted CMake defaults
- [ ] `TuneCache` / knob cache loads correctly on second startup
- [ ] Docker image rebuilt and bench within 1% of bare-metal winner
- [ ] This document updated with actual numbers and winning knob tuple

---

## Appendix A — Quick reference commands

```bash
# Baseline build + bench
./scripts/build_and_benchmark.sh 60

# Short knob sweep (after script fixes)
./scripts/tune_blackwell_knobs.sh 15

# Remote full kit with sweep + ncu enabled
PROPMINER_SKIP_SWEEP=0 PROPMINER_SKIP_BENCH=0 PROPMINER_SKIP_NCU=0 \
  PROPMINER_QUICK_EXIT=0 ./scripts/remote_test_kit.sh

# Self-test after promoting a variant
cp results/libpearl_gemm_capi_<LABEL>.so build/libpearl_gemm_capi.so
./build/propminer --self-test --rtx5090 --gpus 0

# Runtime autotune (does NOT change compile knobs today)
PROPMINER_AUTOTUNE=1 ./build/propminer --rtx5090 --gpus 0
```

## Appendix B — Sweep label format

```
bw-{BM}x{BN}-k{KBLOCK}-s{STAGES}-sw{SWIZZLE}-mb{MIN_BLOCKS}-{LOAD_POLICY}

Example: bw-128x256-k128-s2-sw3-mb2-cp_async
```

## Appendix C — Related documents

- `docs/RTX5090_LINUX_TASKS.md` — step-by-step validation checklist
- `docs/RTX5090_BLUEPRINT.md` — architecture and load-balancing rules
- `src/host/pearl/rtx5090_profile.h` — grid sizing and SM count constants
