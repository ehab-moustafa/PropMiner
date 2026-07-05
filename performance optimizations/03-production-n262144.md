# Production N=262144 — Full VRAM Grid Mining vs Bench Cap N=32768

**PropMiner · RTX 5090 (GB202, sm_120a) · Pearl NoisyGEMM**

| Field | Value |
|-------|-------|
| Status | Implementation largely complete; production rollout is config + validation |
| Primary code | `src/host/pearl/rtx5090_profile.h`, `src/host/main.cpp`, `src/host/pearl/worker_orchestrator.cpp` |
| Deploy entry | `scripts/run_mining.sh` via `PROPMINER_MODE=mine` |

---

## 1. Executive Summary

Production mining on RTX 5090 should run at **N=262144** (M=8192, K=128), the largest N candidate that fits in 32 GB GDDR7 after driver reserve. This launches **65,536 CTAs** per GEMM—**8× the grid** of the bench-capped shape (N=32768, 8,192 CTAs)—while keeping the same M and tile geometry (128×256×128).

The split between bench and mine is intentional:

| Mode | N selection | Batch | Goal |
|------|-------------|-------|------|
| **Bench** (`--bench`, Salad validation) | Capped at `kBenchMaxN` = 32768 | 4 (override via `PROPMINER_BENCH_BATCH`) | Finish ≥1 batch inside a **120 s** window |
| **Mine** (pool connection, `--rtx5090`) | `pick_n_for_vram()` uncapped → 262144 | 20 (`kDefaultBatch`) | Maximize sustained TMAD/s and pool throughput |

**Hashrate (H/s) semantics:** PropMiner reports **DAF-normalized tiles/s** (`tiles_per_sec × difficulty_adjustment_factor()`). At fixed TMAD/s, H/s is **approximately N-invariant**—larger N means more tiles per matmul but fewer matmuls per second. Production and bench H/s should land in the **same order of magnitude** (~1B+ H/s class on a tuned 5090) once a batch completes. Do **not** expect production H/s ≈ 8× bench H/s from N alone.

What *does* change in production:

- **~8× more GPU work per matmul** and per batch (20 matmuls × 65k CTAs).
- **Longer first-batch latency** (~60–120 s wall clock including σ-install, CUDA graph capture, and the first batch-20 launch) vs ~2–10 s at bench N.
- **Higher VRAM footprint** (~14–18 GB resident + ping/pong vs ~3–5 GB at N=32768).
- **Slightly higher kernel efficiency** at full N (measured ~300.8 vs ~299.2 TMAD/s on RunPod 5090 with Swizzle&lt;3,4,3&gt;).

**Bottom line:** Code paths for uncapped N in mine mode already exist. Remaining work is **Salad/production config**, **startup log verification**, **bare-metal validation**, and **operator docs**—not a new kernel or orchestrator rewrite.

---

## 2. Current Bench vs Mine Behavior

| Aspect | Bench (`--bench SECONDS --rtx5090`) | Mine (`--rtx5090 --wallet …`, no `--bench`) |
|--------|-------------------------------------|---------------------------------------------|
| **N cap** | `Rtx5090Profile::kBenchMaxN` = **32768** (`main.cpp` L301) | **No cap** (`cap_n = 0`) → `pick_n_for_vram()` → **262144** on 32 GB |
| **M, K, tile** | 8192, 128, 128×256×128 | Same |
| **Batch (`matmuls_per_poll`)** | **4** default (`worker_orchestrator.cpp` L384); env `PROPMINER_BENCH_BATCH` | **20** (`Rtx5090Profile::kDefaultBatch`, set in `main.cpp` L303) |
| **Pool** | Local synthetic job (σ=0xab, easy target); no gRPC | Kryptex pool via `WorkerOrchestrator` |
| **Watchdog** | Disabled (`main.cpp` L264–266) | Enabled unless `--no-watchdog` |
| **Autotune** | Skipped (RTX5090 profile forces off unless `PROPMINER_AUTOTUNE=1`) | Same |
| **CUDA graphs** | On by default; `PROPMINER_BENCH_NO_GRAPH=1` disables | On (production path) |
| **CTAs / GEMM** | 8,192 | 65,536 |
| **Waves (170 SMs)** | ~49 waves, tail **42** | ~386 waves, tail **26** |
| **VRAM budget query** | `cudaMemGetInfo` − 512 MiB reserve | Same |
| **Startup log tag** | `(bench N cap)` suffix on RTX5090 line | No suffix |
| **Hashrate usable before exit** | Needs 1st batch done within `SECONDS` (Docker default **120**) | Continuous; first meaningful H/s after 1st batch |
| **Docker route** | `PROPMINER_MODE=full` → `remote_test_kit.sh` | `PROPMINER_MODE=mine` → `run_mining.sh` |

### Code references

**Bench cap vs uncapped mine** (`main.cpp`):

```cpp
const int n_cap = (bench_seconds > 0) ? pearl::Rtx5090Profile::kBenchMaxN : 0;
cfg.mining_config = pearl::rtx5090_mining_config(vram_free, n_cap);
cfg.batch_size = pearl::Rtx5090Profile::kDefaultBatch;  // 20; orchestrator overrides to 4 in bench
```

**Bench batch override** (`worker_orchestrator.cpp`):

```cpp
if (bench_mode) {
    int bench_batch = 4;
    if (const char* bb = std::getenv("PROPMINER_BENCH_BATCH")) { ... }
    tuned_batch = bench_batch;
}
```

---

## 3. VRAM Math (32 GB GDDR7, Resident B, Buffers)

### Hardware budget

| Item | Value |
|------|-------|
| Total VRAM | 32 GiB (RTX 5090 GDDR7) |
| Driver reserve (code) | **512 MiB** subtracted in `main.cpp` before `pick_n_for_vram()` |
| Effective budget | ~31.5 GiB |

### Simplified fit model (`Rtx5090Profile::shape_fits_vram`)

Used by `pick_n_for_vram()` to choose N:

```
bytes = M×K + K×N + 4×M×N   (A int8 + B int8 + C int32 model)
bytes += bytes / 4           (leaf CVs, noise, alignment headroom)
```

Fixed **M=8192, K=128**:

| N | A | B (K×N) | C model (4×M×N) | +25% headroom | **Total (model)** |
|---|-----|---------|-----------------|---------------|-------------------|
| 32,768 | 1.0 MiB | 4.0 MiB | 256 MiB | — | **~326 MiB** |
| 262,144 | 1.0 MiB | 32 MiB | 2.0 GiB | — | **~2.6 GiB** |

The model is conservative for the **dominant ping/pong allocations** but does not enumerate every buffer. Actual usage is higher.

### Resident B (`ResidentBState`, `sigma_context.cpp`)

Allocated once per σ, shared across ping/pong halves. Scales with **N** (and K, R):

| Buffer | Size (N=262144, K=128, R=128) |
|--------|--------------------------------|
| `b` | N×K = **32 MiB** |
| `bpeb` | N×K = **32 MiB** |
| `ebr` | N×R = **32 MiB** |
| `ebr_fp16` | 2×N×R = **64 MiB** |
| `earx_bpeb` | N×R×2 = **64 MiB** |
| `leaf_cvs` | ⌈N×K/1024⌉×32 = **1 MiB** |
| `b_scales`, keys, roots | ≪ 1 MiB |
| **Resident B subtotal** | **~225 MiB** |

At N=32768, resident B ≈ **~30 MiB** (8× smaller).

### Per-GPU ping/pong halves (`GpuWorker::HalfBuffers`)

Two halves; N-dependent pieces:

| Buffer | N=32768 (each half) | N=262144 (each half) |
|--------|---------------------|----------------------|
| `c` (fp16, M×N×2) | **512 MiB** | **4 GiB** |
| `a` (M×K) | 1 MiB | 1 MiB |
| A-side noise (`eal`, `apea`, …) | ~few MiB | ~few MiB (M-bound) |
| `a_leaf_cvs` | 32 KiB | 32 KiB |
| `pearl_capi` workspace | scales with M,N,K,R | scales with M,N,K,R |

**Ping+pong C alone:** ~1 GiB @ N=32768 → **~8 GiB** @ N=262144.

### Estimated totals on 32 GB (production N=262144)

| Category | Approx. |
|----------|---------|
| Resident B + σ-install scratch | 0.3–0.5 GiB |
| Ping+pong device buffers + workspace | 10–14 GiB |
| CUDA driver / graph / allocator overhead | 1–3 GiB |
| **Working total** | **~14–18 GiB** |
| **Headroom** | **~13–17 GiB** ✓ |

262144 is the **first** entry in `kCandidates[]`; 131072 and 65536 are fallbacks if free memory is unexpectedly low.

---

## 4. Wave Alignment (170 SMs, Tail Slots)

### CTA count

```
tiles(M, N) = (M / kTileM) × (N / kTileN)
            = (8192 / 128) × (N / 256)
            = 64 × (N / 256)
            = N / 4
```

| N | CTAs | Full 2× occupancy? (`≥ 340`) | `CTAs % 170` (tail) | Waves `⌈CTAs/170⌉` |
|---|------|------------------------------|---------------------|---------------------|
| 32,768 | 8,192 | Yes | **42** | ~49 |
| 262,144 | 65,536 | Yes | **26** | ~386 |
| 65,280 | 16,320 | Yes | **0** (aligned) | 96 |

### Design choice

`pick_n_for_vram()` **prefers 262144 over 65280** even though 65280 is wave-aligned. Comment in `rtx5090_profile.h`:

> *Wave alignment is logged at startup; a small tail wave is cheaper than shrinking N (e.g. 262144 → 65280).*

`wave_aligned_n_at_least()` exists for cases that require exact SM divisibility; production profile does not use it for the default pick.

### Startup diagnostics

Both `main.cpp` and `worker_orchestrator.cpp` log:

```
CTAs=<n> waves~<w> tail=<t>
```

For production, expect **`CTAs=65536 waves~386 tail=26`**.

---

## 5. Implementation: Enabling in Mine Mode (Already Done?) and What's Left

### Already implemented ✅

| Item | Location |
|------|----------|
| VRAM-aware N picker with 262144 first | `rtx5090_profile.h` → `pick_n_for_vram()` |
| Bench-only N cap | `main.cpp` L301: `kBenchMaxN` when `bench_seconds > 0` |
| Uncapped N in mine mode | Same line: `cap_n = 0` when not benching |
| Production batch = 20 | `kDefaultBatch`, applied in `main.cpp` |
| `--rtx5090` profile flag | `main.cpp`, `run_mining.sh` |
| Mine entrypoint | `scripts/run_mining.sh` → `propminer --rtx5090 --wallet …` |
| Docker mode routing | `scripts/docker_entrypoint.sh`: `mine` → `run_mining.sh` |
| VRAM-resident B across iterations | `SigmaContext` / `GpuWorker::install_sigma()` |
| CUDA graph batch path | `GpuWorker::prepare_graph()` / `queue_batch()` |
| Unit tests for cap vs uncapped | `tests.cpp` → `test_rtx5090_wave_alignment()` |

### Remaining work (mostly config / docs)

| Priority | Task | Notes |
|----------|------|-------|
| P0 | **Salad deploy:** `PROPMINER_MODE=mine` + `PROPMINER_WALLET` | Default Dockerfile uses `full` (validation kit) |
| P0 | **Verify startup log** on target GPU: `N=262144 batch=20 CTAs=65536` | Single-line acceptance check |
| P1 | Update `--rtx5090` help text | Still says `N=32768`; runtime may pick 262144 |
| P1 | Document H/s comparability | Bench vs mine numbers (Section 1) |
| P2 | Optional `PROPMINER_N_CAP` env for ops/debug | Not in code today; would wrap `cap_n` |
| P2 | Log `TMAD/s` alongside H/s in stats loop | `GpuWorker::tmads_per_sec()` already tracked |
| P3 | `PROP_MINER_SELF_TEST_PROD=1` timing budget | 180 s timeout; confirm pass at N=262144 |

**No kernel changes required** for the N=262144 rollout unless profiling shows a regression vs RunPod baseline.

---

## 6. Bench Cap Rationale (120 s Window, ~112 s First Batch)

### Why `kBenchMaxN = 32768`

Salad and Docker validation use a **fixed wall-clock window** (`PROPMINER_BENCH_SECONDS=120` in `Dockerfile`). The bench path must:

1. Complete **σ-install** (B expansion, tensor hash, noise, Merkle tree).
2. **Capture** a CUDA graph for the batch size.
3. **Finish at least one batch** so `GpuWorker` can compute non-zero H/s.

At **N=262144, batch=20**, a single batch is on the order of **~15–25 s of pure GEMM** at ~300 TMAD/s, **before** one-time setup. Observed **first-batch wall times ~60–120 s** are dominated by:

- Resident B install + Merkle build on first σ
- CUDA graph capture for batch=20 at full grid
- GDDR7 cold start / module load (`CUDA_MODULE_LOADING=EAGER` mitigates)

With a **120 s** Salad window, an uncapped N run often reports **`hashrate: 0 H/s`** until exit—the failure mode `remote_test_kit.sh` guards against with fallbacks (batch=4, batch=1, graph off).

### Bench mitigations (already in code)

| Knob | Default | Purpose |
|------|---------|---------|
| `kBenchMaxN` | 32768 | 8× smaller grid; ~8× faster per matmul |
| Bench batch | 4 | vs production 20; 5× less work per graph launch |
| `PROPMINER_BENCH_BATCH` | override | Remote kit tries 4 → 1 → graph-off |
| Stats hint @ 30 s | orchestrator | *"still awaiting first batch completion"* |

### Implication

**Bench H/s is a smoke test at reduced N**, not a predictor of production pool earnings or peak TMAD/s. Compare **TMAD/s** or steady-state H/s after warm-up for apples-to-apples tuning.

---

## 7. Expected Production Performance

### Kernel throughput (reference)

From `transcript_gemm_kernel.cu` (RunPod RTX 5090, M=8192, N=262144):

- **~300.8 TMAD/s** with Swizzle&lt;3,4,3&gt; vs ~299.2 at Swizzle&lt;2,4,3&gt;
- ~**0.5%** uplift at full N from swizzle alone; grid size is the main throughput lever

### Steady-state batch timing (estimate)

Per matmul TMAD = M×N×K = 8192 × N × 128:

| N | TMAD / matmul | @ 300 TMAD/s |
|---|---------------|--------------|
| 32,768 | 34.4 TMAD | ~0.11 s |
| 262,144 | 274.9 TMAD | ~0.92 s |

| Mode | Batch | Est. GEMM-only batch time |
|------|-------|---------------------------|
| Bench | 4 @ N=32768 | **~0.4–0.5 s** |
| Mine | 20 @ N=262144 | **~18–20 s** |

### H/s (DAF-normalized)

With default patterns (rows=2, cols=64, dot_product=128), DAF = **16,384**.

From `gpu_worker.cpp`:

```
matmuls/sec = batch / batch_seconds
tiles/sec   = matmuls/sec × (M/bM) × (N/bN)
H/s         = tiles/sec × DAF
```

At fixed TMAD/s, **N cancels**: larger N means more tiles per matmul but proportionally fewer matmuls per second. Bench (N=32768, batch=4) and mine (N=262144, batch=20) should report **similar steady-state H/s** once batches complete.

Worked example @ ~300 TMAD/s, N=262144, batch=20, ~18 s per batch:

```
matmuls/sec ≈ 20/18 ≈ 1.1
tiles/sec   ≈ 1.1 × 65536 ≈ 72,000
H/s         ≈ 72,000 × 16,384 ≈ 1.2×10⁹
```

Expect console output in the **~10⁹ H/s** range—**not 8× the bench number** from N alone.

### Pool-facing behavior

- Heartbeat every pool interval with aggregate `total_hashrate_`
- First **~1–2 min** after job assign may show **0 or stale H/s** on 5 s stats ticks
- Share latency scales with batch time (~20 s class at N=262144, batch=20)

---

## 8. Salad Deploy Config (`PROPMINER_MODE=mine`, Wallet)

### Current Docker defaults (validation, not mining)

```dockerfile
ENV PROPMINER_MODE=full
ENV PROPMINER_BENCH_SECONDS=120
ENV PROPMINER_SKIP_BENCH=0   # when MODE=full
```

### Production Salad / container overrides

```bash
# Required
PROPMINER_MODE=mine
PROPMINER_WALLET=krxXXXXXXXX.your-worker-name

# Optional
PROPMINER_POOL=prl.kryptex.network:443   # default in run_mining.sh
PROPMINER_GPUS=0                         # default single GPU

# Leave unset for production (RTX5090 profile defaults)
# PROPMINER_AUTOTUNE=0                     # default off for --rtx5090
# PEARL_GEMM_CONSUMER_CLUSTER_M=1          # Dockerfile sets 1; tune if needed
```

### What `mine` runs

`docker_entrypoint.sh` → `scripts/run_mining.sh`:

```bash
propminer --rtx5090 --gpus "${GPUS}" --pool "${POOL}" --wallet "${WALLET}"
```

No `--bench` → **N=262144**, **batch=20**, pool connected, watchdog on.

### Salad checklist

- [ ] GPU: RTX 5090, 32 GB, native Linux + driver ≥550 / CUDA 12.8 runtime in image
- [ ] `PROPMINER_MODE=mine`
- [ ] `PROPMINER_WALLET` set (container exits immediately if missing)
- [ ] Do **not** set `PROPMINER_BENCH_SECONDS` (mine path ignores it)
- [ ] Persist logs; confirm `[main] RTX 5090 profile: M=8192 N=262144 batch=20 …` without `(bench N cap)`

---

## 9. Monitoring & Tuning

### Startup (first 2 minutes)

| Log line | Healthy production |
|----------|-------------------|
| `[main] RTX 5090 profile: M=8192 N=262144 batch=20 CTAs=65536 …` | N uncapped |
| `[orchestrator] GEMM grid: … CTAs=65536 … tail_slots=26 batch=20` | Matches |
| `[gpu] first batch queued (count=20, graph=on)` | Batch 20, graph captured |
| `[gpu] first batch completed in XXXXX ms -> YYYYY H/s` | X typically 60k–120k ms first time |

### Steady state

| Signal | Tool / source |
|--------|----------------|
| H/s every 5 s | stdout: `hashrate: … H/s` |
| TMAD/s | Internal `GpuWorker::tmads_per_sec()`; add logging if needed |
| GPU util | `nvidia-smi dmon` — expect sustained high SM/active |
| VRAM | `nvidia-smi` — expect ~14–18 GB used, not OOM |
| Pool accepts | `share accepted` lines from orchestrator |
| Stall recovery | `[watchdog] stall detected` should be rare |

### Tuning knobs (production)

| Env / flag | Effect |
|------------|--------|
| `PROPMINER_AUTOTUNE=1` | Re-enable autotune (usually off for `--rtx5090`) |
| `PEARL_GEMM_CONSUMER_CLUSTER_M` | Cluster launch (default 1 in Dockerfile) |
| `PEARL_GEMM_CONSUMER_CARVEOUT` | L1/shared carveout if probed |
| `--no-watchdog` | Disable stall republish (not recommended) |
| `--config M,N,K,R` | Manual override; bypasses VRAM picker |

**Do not** lower N in production to match bench H/s numbers—compare TMAD/s instead.

---

## 10. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **OOM at N=262144** | Medium | 512 MiB reserve + 25% headroom in picker; monitor `nvidia-smi`; fallback candidates 131072/65536 in `pick_n_for_vram()` |
| **First batch timeout / 0 H/s on dashboards** | Low in mine mode | Expected first ~1–2 min; not a fault if H/s rises after first batch |
| **OOM from other GPU consumers** | Medium | Salad containers should be 1 GPU / 1 workload; no MIG split on 5090 |
| **Graph capture failure at batch=20** | Low | Worker falls back to non-graph `iter_batch()`; watch for `graph=off` in logs |
| **σ rotation cost** | Medium | New job → full resident B reinstall; brief throughput dip |
| **Thermal / power throttle** | Low | Watch SM clocks; 5090 is compute-bound at this shape |
| **Invalid shares after OC** | High if OC aggressive | Run `--self-test` after clock changes |

---

## 11. Validation Checklist

### Pre-deploy (bare metal or Salad staging)

- [ ] `nvidia-smi` shows RTX 5090, 32 GB, compute capability 12.0
- [ ] `./propminer --self-test --rtx5090` passes (optionally `PROP_MINER_SELF_TEST_PROD=1` for full N)
- [ ] `./propminer --bench 120 --rtx5090 --gpus 0` completes with **non-zero** H/s (validates bench cap path)
- [ ] `./propminer --rtx5090 --wallet TEST.WORKER --pool …` logs **N=262144**, connects, registers
- [ ] First batch completes; H/s stabilizes over 5+ minutes
- [ ] At least one **accepted share** on pool (easy vardiff window)
- [ ] VRAM usage stable, no OOM in `dmesg` / container logs
- [ ] Watchdog silent under normal operation (15+ min)

### Regression guards (CI / unit)

- [ ] `test_rtx5090_wave_alignment` passes
- [ ] `rtx5090_mining_config(0, kBenchMaxN).n == 32768`
- [ ] `rtx5090_mining_config(0).n >= 32768` (uncapped picks largest fit)

### Post-deploy monitoring (24 h)

- [ ] Pool hashrate within ~10% of local steady-state H/s
- [ ] Reject rate ≈ 0% (invalid share → kernel/OC issue)
- [ ] No repeated watchdog or CUDA context reset logs

---

## 12. Effort Estimate

| Workstream | Effort | Type |
|------------|--------|------|
| Salad `PROPMINER_MODE=mine` + wallet | **< 1 h** | Config |
| Startup log verification on 5090 | **1–2 h** | Ops |
| Help text / README updates | **< 2 h** | Docs |
| This plan + runbook | **Done** | Docs |
| Bare-metal 24 h soak + pool share | **4–8 h** | Validation |
| Optional TMAD/s in stats loop | **2–4 h** | Code (nice-to-have) |
| Optional `PROPMINER_N_CAP` debug env | **1–2 h** | Code (nice-to-have) |

**Total: ~1 day ops + docs**, assuming no OOM or kernel regression on target hardware. **No mandatory C++/CUDA changes** for production N=262144—the split is already encoded in `cap_n` logic and deploy scripts.

---

## Appendix: Key Constants Quick Reference

```cpp
// rtx5090_profile.h
kSMCount      = 170
kTileM/N/K    = 128 / 256 / 128
kDefaultM/N/K = 8192 / 32768 / 128   // compile fallback; runtime mine → 262144
kDefaultBatch = 20
kBenchMaxN    = 32768
kCandidates[] = { 262144, 131072, 65536, 65280, 43520, 32768 }
```

```bash
# Dockerfile (validation defaults)
PROPMINER_MODE=full
PROPMINER_BENCH_SECONDS=120

# Production mining
PROPMINER_MODE=mine
PROPMINER_WALLET=<required>
```

---

*Document version: 2026-07-05 · PropMiner v2.0 · RTX 5090 production profile*
