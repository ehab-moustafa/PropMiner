# GeForce Kernel v2 — Merged Implementation Plan

Synthesized from three analysis passes (v1 gap analysis, CUTLASS patterns, safety/test matrix).

| Field | Value |
|-------|-------|
| **Status** | Phase 1 implementing |
| **Default** | v1 remains production default until all gates pass |
| **Default** | v2 ON at compile + runtime (blackwell); rollback via `geforce_v1` / `consumer` |

---

## Consensus from analysis

1. **Full CUTLASS collective replacement is NOT feasible** — SM120 builder supports F8/F6/F4 only; Pearl requires SM80 `m16n8k32` int8 atom + custom transcript epilogue.
2. **First shippable increment (Phase 1):** `PipelineTmaAsync` + TMA descriptor prefetch + host descriptor cache — preserves proof contract, enables later pingpong/asymmetric-B.
3. **v1 stays untouched** — v2 is a separate `.cu` for G3 rollback and memcmp regression.
4. **No consumer TMA work** — orthogonal; deprioritized per plan 08.

---

## Phased rollout

| Phase | Work | Transcript risk | Perf gain |
|-------|------|-----------------|-----------|
| **0** | Instrumentation, merged plan, feature flags | None | None |
| **1** | `PipelineTmaAsync`, descriptor cache, TMA prefetch | Low (G1+G3) | Small–medium |
| **2** | Warp-decoupled loops (no CTA `__syncthreads` in mainloop) | Medium | Medium |
| **3** | `KBLOCK=64, STAGES=3/4` within 99 KB SMEM | Low if G1 passes | Medium |
| **4** | Asymmetric B (load once per σ-stable CTA) | High (σ-swap) | High |
| **5** | Pingpong epilogue overlap | High | High |
| **6** | Promote v2 to default | — | — |

**This PR implements Phase 0 + Phase 1.**

---

## Proof-locked (never change)

- SM80 `16x8x32` int8 MMA atom, tile 128×256, `kAtomK=32`
- 16 transcript slots, `partition_C` mapping, XOR/rotl schedule
- Headless PoW semantics unchanged

---

## Gates before v2 default

| Gate | Criterion |
|------|-----------|
| G1 | consumer ≡ v2 transcript memcmp 100% |
| G3 | v1 ≡ v2 transcript memcmp 100% |
| G4 | `sizeof(SharedStorage) ≤ 101376` |
| G2 | `--self-test` + `PROP_MINER_SELF_TEST_PROD=1` |
| G6 | bench ≥ +10% vs consumer |
| G7 | 24–48h pool soak |

---

## Rollback

| Action | Rebuild? |
|--------|----------|
| `PEARL_GEMM_KERNEL=geforce` or `geforce_v1` | No |
| `PEARL_GEMM_KERNEL=consumer` | No |
| `PEARL_GEMM_BLACKWELL_GEFORCE_V2=0` | Yes |
