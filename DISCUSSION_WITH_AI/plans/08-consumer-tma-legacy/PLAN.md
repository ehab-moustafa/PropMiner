# Plan: Consumer Kernel TMA (Legacy Path) — DEPRIORITIZED

| Field | Value |
|-------|-------|
| **Status** | Compile knob exists; **not a performance initiative** |
| **Priority** | **Reject** for prod funding |
| **Est. gain on prod traffic** | **0%** — GeForce is default |
| **Est. gain on consumer-only path** | +5–15% vs cp.async consumer (below GeForce v1) |
| **Recommendation** | Implement **GeForce v2** instead (plan 01) |

---

## What this is

Replace **cooperative `cp.async`** gmem→smem loads in `transcript_gemm_kernel_consumer` (256-thread **consumer** kernel) with **TMA + mbarrier**, keeping SM80 MMA + transcript unchanged.

Enabled at compile: `PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma` → `PEARL_CONSUMER_USE_TMA_EXPERIMENT=1`.

Runtime requires: `PEARL_GEMM_KERNEL=consumer` (opt out of GeForce default).

---

## How it works today

| Path | When used | Load method |
|------|-----------|-------------|
| **GeForce** (default) | `use_geforce_experimental_kernel()==true` | Dedicated TMA warp + `tma_tile_loader.cuh` |
| **Consumer** (fallback) | `PEARL_GEMM_KERNEL=consumer` | All 256 threads `cp.async` |

**Production RTX 5090 mining uses GeForce**, not consumer. Consumer TMA does not affect shipped hashrate unless users force consumer.

**Code note:** Dual-path consumer TMA may compile (`validate_tma_build.sh`); byte-identity / perf gates for TMA consumer are **not** production requirements.

---

## What consumer TMA would do (if pursued)

| | cp.async consumer | consumer TMA |
|--|-------------------|--------------|
| Load issuers | 256 threads | Thread 0 (elected) |
| Transaction | 16 B granules | Full A+B stage (~48 KiB) |
| Shared code | Inline | Same `tma_tile_loader.cuh` as GeForce |

---

## Why DEPRIORITIZED

1. **GeForce v1 already uses TMA** with **better** architecture (dedicated warp vs thread 0).
2. Consumer TMA projected +10–25% applies only to **fallback** path — **0%** on default traffic.
3. **~9–13 eng-days** on fallback duplicates work already done for GeForce.
4. **Ampere/Ada** cannot use TMA experiment (Blackwell-only).
5. Same calendar time on **GeForce v2** affects **100%** of miners.

---

## Relationship to GeForce v2

```
tma_tile_loader.cuh ──shared──► GeForce v1 (prod) ──evolve──► GeForce v2 (plan 01)
                    └──► consumer TMA branch (fallback only)
```

Consumer TMA does **not** unblock GeForce v2. v2 builds on warp-specialized lane.

---

## Risks (if someone funds it anyway)

| Risk | Severity |
|------|----------|
| Transcript byte drift | Critical |
| False priority / zero prod impact | High (strategic) |
| Dual build matrix maintenance | Medium |

---

## Testing (fallback-only quality bar)

Only relevant if long-term `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=0`:

1. `scripts/validate_tma_build.sh` — cp_async + tma compile  
2. Memcmp consumer TMA vs cp.async consumer  
3. `PEARL_GEMM_KERNEL=consumer --self-test`  
4. Must still pass `verify_geforce_transcript.sh` (consumer ≡ GeForce reference)

**Production binary should remain:** GeForce default + consumer cp.async fallback.

---

## Stakeholder decision

| Option | Prod TMAD impact | Recommendation |
|--------|------------------|----------------|
| A. Consumer TMA to prod quality | **0%** | **Reject** |
| B. GeForce v2 | **+10–25%** | **Approve** (plan 01) |
| C. Status quo GeForce v1 | Baseline | **Accept** until B |
| D. Consumer TMA CI compile-only | 0% | Optional maintenance |

---

## Rollback

N/A for prod — not shipping. If experimenting:

```bash
PEARL_GEMM_KERNEL=geforce          # leave consumer path
# or
-DPEARL_GEMM_BLACKWELL_LOAD_POLICY=cp_async
```

---

*Plan for stakeholder **no** on consumer TMA perf funding; **yes** on GeForce v2.*
