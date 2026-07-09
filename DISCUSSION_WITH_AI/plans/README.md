# PropMiner — Unimplemented Hashrate Plans (AI Discussion)

**Purpose:** Stakeholder / CTO decision documents only — **no implementation authorized here.**  
**Audience:** Engineering leads deciding what to build next after tuning + GeForce default.  
**Baseline:** RTX 5090, ~300 TMAD/s consumer reference, GeForce kernel default-on in source (requires rebuild on rig).

Each plan follows the same structure:

| Section | Contents |
|---------|----------|
| Status | Not built / partial / deprioritized |
| How it works today | Current production behavior |
| Proposed change | What would be built |
| Before → after | Behavioral difference |
| Expected gain | % TMAD/s and why |
| Risks | Proof, VRAM, regressions |
| Testing | Gates before prod |
| Rollback | How to undo without rebuild |
| Go / no-go | Ship criteria |
| Effort | Engineer-weeks |

## Plans (ranked by hashrate impact)

| # | Folder | Impact | Effort | Priority |
|---|--------|--------|--------|----------|
| 00 | [comprehensive-integration](./00-comprehensive-integration/PLAN.md) | **Full roadmap** 290→700–800+ TMAD/s | 14–25 weeks | **Master plan** — phased integration of all findings |
| 01 | [geforce-kernel-v2](./01-geforce-kernel-v2/PLAN.md) | **+10–25%** (on top of v1) | 4–6 weeks | **P0/P1** — Phase 1 landed, gates pending |
| 02 | [ptr-array-grouped-gemm](./02-ptr-array-grouped-gemm/PLAN.md) | **+10–20%** if batch≥4; **0%** at batch=1 | 4–6 weeks | P1 (needs batch strategy) |
| 03 | [stream-split-pregemm](./03-stream-split-pregemm/PLAN.md) | **+1–5%** | 2–3 weeks | P2 |
| 04 | [triple-gpu-half-buffer](./04-triple-gpu-half-buffer/PLAN.md) | **+0–3%** typical | 3–5 days validation* | P2 |
| 05 | [fuse-noise-noisinga-gemm](./05-fuse-noise-noisinga-gemm/PLAN.md) | **+1–3%** | 3–5 weeks | P2 |
| 06 | [sigma-install-b-hash-batching](./06-sigma-install-b-hash-batching/PLAN.md) | Startup/job-switch, **~0%** steady TMAD | 1–2 weeks | P2 |
| 07 | [cccl-share-compaction](./07-cccl-share-compaction/PLAN.md) | **<2%** | 3–4 days | P3 |
| 08 | [consumer-tma-legacy](./08-consumer-tma-legacy/PLAN.md) | **0%** on prod (fallback only) | — | **Deprioritized** |

\*Triple-buffer **host code exists** behind `PROPMINER_TRIPLE_BUFFER=1`; remaining work is profiling + validation, not greenfield.

## Investment order (recommended)

1. **GeForce v2** — affects 100% of production traffic  
2. **Grouped GEMM** — only if product commits to `PROPMINER_BATCH > 1`  
3. **Stream-split pre-GEMM** — profile first; skip if pre-GEMM < 3% of batch time  
4. **Triple buffer** — only if `pipeline: half_wait_ms_max` proves stalls  
5. **Fuse pre-GEMM** — after P0 kernel stable  
6. **σ-install batching** — if job-switch latency hurts ops  
7. **CCCL compaction** — defer unless batch grows  
8. **Consumer TMA** — do not fund (see plan 08)

## What these plans do NOT cover

- Tuning / knob sweeps (you already did)  
- tcgen05/TMEM on 5090 (invalid hardware)  
- FP4/FP8/cuBLAS-only GEMM (breaks Pearl proofs)  
- Power/clock/infra (ops, not code plans)

*Generated July 2026 from codebase research + 8 parallel analysis agents.*
