# Plan: Triple GPU Half-Buffer

| Field | Value |
|-------|-------|
| **Status** | **Partially implemented** — code exists, default **OFF** |
| **Priority** | P2 — profile-gated |
| **Est. gain** | **+0–3%** typical; **+2–5%** if share stalls proven |
| **Effort** | **3–5 days** validation (not greenfield) |
| **Risk** | Low proof risk; medium VRAM |

---

## How it works today (default)

**Two** full GPU workspaces: `ping_`, `pong_`.

Mining loop alternates: launch on `cur`, wait for `other`, scan winners, swap.

**Defer-share ON (default):** share rebuild runs on side thread but **pins the half** until `share_jobs_pending == 0`.

Main thread blocks in `wait_until_half_free()` before reusing that half:

```
other: [GEMM done] → share rebuild holds half
cur:   wait_until_half_free → STALL if T_share > T_batch
```

| Symbol | Typical @ N=262144 |
|--------|-------------------|
| T_batch | ~25–35 ms |
| T_share | ms–tens of ms (A regen + hash + D2H) |

At rare prod diff, stalls are uncommon → **small steady gain**.

---

## Proposed / existing (opt-in)

**Third** `HalfBuffers third_` — ring of 3 halves, pipeline depth 2:

```
ping ↔ pong ↔ third
Share pins one → two remain free for GEMM
```

**Activation (all required):**

- `PROPMINER_TRIPLE_BUFFER=1`
- `PROPMINER_DEFER_SHARE_GPU=1` (default)
- `triple_vram_headroom_ok()` passes

**σ-install:** `drain_all_halves_for_sigma()` waits all three + syncs all streams.

---

## Before → after

| | Dual (today) | Triple (opt-in) |
|--|--------------|-----------------|
| Compute slots during share | 1 free | 2 free |
| `wait_until_half_free` stalls | Possible | Should → 0 |
| Extra VRAM | — | **+4–6 GiB** |

---

## Why we need it

- Completes defer-share story — moved work off thread but not buffer contention
- Only matters when `T_share ≥ T_batch` (share-heavy diff / testnet)
- **Not** a primary hashrate lever vs kernel work

---

## VRAM reality @ N=262144 / K=4096 (32 GiB)

- `triple_vram_headroom_ok()` requires **≥11 GiB free** + incremental peak
- At ~79% VRAM baseline, guard **fails** → logs `insufficient_vram` → stays dual
- **Triple most viable** at lower N or more free VRAM

---

## Risks

| Risk | Mitigation |
|------|------------|
| OOM | VRAM guard + try/catch alloc → dual fallback |
| Race on half ownership | Existing `share_jobs_pending` mutex |
| σ swap with share on third | `drain_all_halves_for_sigma()` |
| Low ROI at prod diff | Profile before enabling |
| Competes with async job install for VRAM | Both guarded; rarely both active at prod N |

---

## Testing

**Instrumentation (shipped):** orchestrator logs every ~5s:

```
pipeline: half_wait_count=… half_wait_ms_max=… triple=on|off
```

| Step | Command / criterion |
|------|---------------------|
| Self-test dual defer | `PROPMINER_DEFER_SHARE_GPU={0,1} --self-test` |
| Prod shape | `PROP_MINER_SELF_TEST_PROD=1` |
| Triple enable | `PROPMINER_TRIPLE_BUFFER=1` if guard passes |
| VRAM fail expected | @ N=262144/K=4096 → `reason=insufficient_vram` OK |
| A/B bench | 300s dual vs triple @ shape where guard passes |
| Pool | Accepted shares unchanged 1h |

**Missing:** `scripts/validate_triple_buffer.sh` (recommended).

**Go criterion:** `half_wait_ms_max` > 5 ms correlated with shares **and** VRAM guard passes.

---

## Rollback

```bash
unset PROPMINER_TRIPLE_BUFFER   # or =0
# restart — instant, no rebuild
```

---

## Go / no-go

| | Action |
|--|--------|
| **Go** | Stalls proven + VRAM OK + self-test pass |
| **No-go** | `half_wait_*` ≈ 0; VRAM always fails; kernel backlog higher ROI |
| **Never** | Default-on @ N=262144 / 32 GiB without VRAM model update |

---

## Files involved

- `src/host/pearl/gpu_worker.{cpp,h}` — `third_`, ring loop, VRAM guard
- `src/host/pearl/env_flags.h` — `PROPMINER_TRIPLE_BUFFER`
- `src/host/pearl/worker_orchestrator.cpp` — `pipeline:` metrics
- `docs/TRIPLE_BUFFER_ONE_PAGER.md` — design (update status: partial)

**Not touched:** pearl-gemm kernels.

---

*Plan only — enable via profiling, not blind deploy.*
