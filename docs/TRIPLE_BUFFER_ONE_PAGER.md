# Triple GPU Half-Buffer — CTO One-Pager

**Product:** PropMiner · Pearl NoisyGEMM mining on RTX 5090  
**Date:** July 2026  
**Status:** Not implemented (design / prioritization)  
**Baseline:** ~300 TMAD/s steady-state @ M=8192, N=262144, K=128

---

## 1. Problem (30 seconds)

PropMiner keeps the GPU fed with a **ping-pong** pipeline: two full device workspaces (`ping`, `pong`) alternate GEMM batches. When a winning nonce is found, **share reconstruction** (regenerate A, tensor-hash, D2H copies) must finish on that half before the mining loop can reuse it.

With **deferred share handling** enabled (production default), share work runs on a side thread but still **pins the half-buffer** until complete. The main loop blocks in `wait_until_half_free()`:

```
Mining loop:  [GEMM ping] → [GEMM pong] → [GEMM ping] → …
Share hit on pong:  pong held for T_share ms  →  next ping launch may STALL
```

If share GPU work (`T_share`) exceeds one batch time (`T_batch` ≈ 25–35 ms at prod N), the GPU sits idle for the difference. That is a **pipeline bubble** — wasted tensor-core time, lower TMAD/s and fewer puzzle attempts per second.

---

## 2. Proposed solution

Add a **third `HalfBuffers` workspace** (`third_`) so the conveyor always has two halves free for GEMM while the third handles share reconstruction.

```
Today (dual):     ping ↔ pong     (share holds one → 1 compute slot left)
Proposed (triple): ping ↔ pong ↔ third   (share holds one → 2 compute slots left)
```

No change to Pearl proof math, transcript layout, or pool wire format. Same kernels; only host orchestration and VRAM allocation in `gpu_worker.cpp`.

---

## 3. What already exists (context)

| Layer | Status |
|-------|--------|
| Ping-pong GEMM + CUDA graphs | ✅ Shipped |
| Async seed upload (`seed_copy_stream_`) | ✅ Shipped (default ON) |
| Deferred share on side thread (`PROPMINER_DEFER_SHARE_GPU`) | ✅ Shipped (default ON) |
| Async job install | ✅ Shipped (VRAM-guarded) |
| **Third compute half** | ❌ **Not built** |

Deferral moved share work off the hot thread but **did not remove half-buffer contention**. Triple-buffer completes that story.

---

## 4. Hardware & VRAM

| Requirement | RTX 5090 |
|-------------|----------|
| CUDA / sm_120a | ✅ Supported |
| Extra VRAM | **~4–6 GiB** at N=262144 (one more C half + A staging + headers/graph state) |
| Fits 32 GiB with prod resident B | ⚠️ **Tight** — must re-run `shape_fits_vram()` logic; may force lower N cap on memory-constrained hosts |

Triple-buffer is a **software + VRAM** change, not new silicon.

---

## 5. Implementation scope

| Work item | Effort | Owner |
|-----------|--------|-------|
| Add `third_` `HalfBuffers`, allocate/free with ping/pong | ~2–3 days | Host/CUDA |
| Extend mining loop: rotate ping → pong → third | ~2–3 days | Host |
| Share path: pin `third_` (or any free half) instead of blocking | ~1–2 days | Host |
| Job switch / σ-install: drain **three** halves + share queue | ~1–2 days | Host |
| VRAM guard + auto-disable when tight (mirror async-job-install) | ~1 day | Host |
| Self-test + prod-shape regression (`--self-test`, share verify) | ~2 days | QA |
| **Optional:** `PROPMINER_TRIPLE_BUFFER=1` kill switch | ~0.5 day | Host |

**Total:** ~2–3 engineer-weeks including validation on live pool.

**Not in scope:** Kernel changes, CUTLASS, proof semantics.

---

## 6. Risks

| Risk | Mitigation |
|------|------------|
| VRAM OOM at N=262144 | VRAM pre-check; fall back to dual-buffer |
| Race on half ownership | Reuse existing `share_jobs_pending` + mutex pattern |
| Job supersede during share on third half | Drain all three halves before σ swap (same as today for two) |
| Low ROI if shares are rare | **Profile first** — only ship if `wait_until_half_free` measurable in prod |

---

## 7. Success metrics

| Metric | Target |
|--------|--------|
| Steady TMAD/s (no shares) | No regression vs dual-buffer |
| TMAD/s during share windows | Measurable reduction in batch gaps |
| Pool accepted-share rate | Unchanged (verify-ok = 100%) |
| `wait_until_half_free` wall time | → 0 in steady mining (instrumentation) |

---

## 8. Estimated hashrate impact

**Unit:** TMAD/s = trillion INT8 multiply-accumulate ops per second (community “TH/s” on pool dashboards). More TMAD/s = more PoW puzzle attempts per second.

| Scenario | Extra throughput | On ~300 TMAD/s baseline |
|----------|------------------|-------------------------|
| **Steady state, rare shares** (typical Kryptex prod diff) | **+0–1%** | ~300 → ~303 TMAD/s |
| **Moderate share rate**, `T_share < T_batch` | **+0–2%** | ~300 → ~306 TMAD/s |
| **Share-heavy** (low diff / testnet / aggressive vardiff), `T_share ≥ T_batch` | **+2–5%** | ~300 → ~306–315 TMAD/s |
| **Worst-case bubble** (share every batch, slow rebuild) | **up to ~5–8%** | ~300 → ~315–324 TMAD/s |

**Planning number for CTO:** budget **+0–3% steady**, **+2–5%** only if profiling proves sustained `wait_until_half_free` stalls. This is **not** a primary lever for 2× hashrate.

**Puzzle throughput:** Pearl protocol hashrate scales with tiles/s × difficulty factor; eliminating GPU idle bubbles directly increases **iterations/s** and **accepted-share opportunity** but does not change per-share proof difficulty.

---

## 9. Recommendation

1. **Do not prioritize ahead of kernel work** (GeForce warp-specialized path, tile-knob tuning) — those deliver 10–25% each.
2. **Instrument** `wait_until_half_free` duration on production rigs for 24–48 h.
3. **Implement triple-buffer** only if p95 stall time > 5 ms or measurable TMAD dip correlates with share events.
4. Ship behind `PROPMINER_TRIPLE_BUFFER=1` with VRAM guard defaulting OFF when N=262144 and free VRAM < 6 GiB headroom.

---

## Appendix A — All GEMM / pipeline levers (CTO impact summary)

Baseline reference: **~300 TMAD/s** on RTX 5090 @ prod shape (~36% of 838 INT8 TOPS rated).  
Realistic **combined** target with kernel + tune work: **~500–600 TMAD/s** (1.7–2×). No single host-side trick reaches 3×.

| # | Lever | What it is | Status | Est. extra speed | Absolute range (@ 300 baseline) | Work type |
|---|-------|------------|--------|------------------|----------------------------------|-----------|
| **1** | **GeForce warp-specialized kernel** | Split TMA load warp vs MMA warps inside each CTA | **ON by default** (blackwell builds) | **+10–25%** | 330–375 TMAD/s | Rebuild; `PEARL_GEMM_KERNEL=consumer` to opt out |
| **2** | **Tile knobs** (KBLOCK, STAGES, cluster_m) | Shared-mem tile depth, pipeline stages, CTA clusters | Tune scripts exist; defaults conservative | **+5–15%** each axis; **+10–20%** if combined tune wins | 315–360 TMAD/s | Rebuild for KBLOCK/STAGES; env/tune for cluster_m |
| **3a** | **CPU/GPU overlap** (defer share, async seed, bcol cache) | Keep GPU mining while CPU/share/job work runs elsewhere | **✅ Shipped** (default ON) | **+0–2%** steady TMAD/s; larger win on **job-switch latency** (not sustained MAC rate) | ~300–306 TMAD/s | Already in binary |
| **3b** | **Triple half-buffer** (this doc) | Third GPU workspace so share work never blocks GEMM rotation | **❌ Not built** | **+0–3%** typical; **+2–5%** share-heavy | ~300–315 TMAD/s | ~2–3 eng-weeks host code |
| **4** | **Ptr-array grouped GEMM** | One kernel launch, multiple nonces (batch>1) | **❌ Not built** | **0%** at batch=1 (prod default); **+10–20%** if batch≥4 and kernel efficient | 0 now; 330–360 if batching adopted | Major kernel + CAPI project (weeks) |

### How to read “more puzzles solved”

- **TMAD/s** ≈ raw PoW search throughput (MAC operations driving the noisy GEMM proof).
- **Pool hashrate (TH/s)** ≈ TMAD/s in community reporting for Pearl miners.
- **Accepted shares** depend on difficulty, luck, and pool job lifetime — higher TMAD/s increases *attempts*, not guarantee of more accepts per hour.
- **Kernel levers (1, 2, 4)** attack the ~64% gap between measured tensor throughput and RTX 5090 rated peak.
- **Pipeline levers (3a, 3b)** attack idle time when the GPU waits on host/share/buffer reuse — smaller but zero proof risk when done correctly.

### Suggested investment order

1. **Tune + GeForce kernel** (items 1–2) — highest ROI, proven CUDA patterns.  
2. **Profile triple-buffer need** (item 3b) — implement only if data supports it.  
3. **Grouped GEMM** (item 4) — only if product strategy raises `PROPMINER_BATCH` above 1.

---

*Internal refs: `gpu_worker.cpp` (`wait_until_half_free`), `performance optimizations/04-defer-share-gpu-work.md`, `research/08-propminer-headroom/REPORT.md`, `research/01-git-repos-mining/IMPLEMENTATION_OPPORTUNITIES.md`*
