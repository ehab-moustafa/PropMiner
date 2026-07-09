# 00-tmad-dominance

> **Mission:** Break 800+ TMAD/s on RTX 5090 for Pearl mining — from ~290 to 700-800+ and beyond.
> **Date:** July 9, 2026

## The Problem

PropMiner on an RTX 5090 delivers ~290 TMAD/s. SRBMiner delivers ~305-372 TH/s on the same hardware. The gap is only ~10-25%, but the real target is **700-800+ TMAD/s** — and potentially much higher with novel approaches.

## The Key Finding

**The GEMM kernel is already well-optimized.** The bottleneck is NOT compute — it's **host pipeline overhead, default batch=1, and conservative defaults**. The infrastructure for high throughput exists; it's just disabled.

At 290 TMAD/s, PropMiner is using ~35% of the RTX 5090's INT8 TOPS capacity. There's ~65% headroom in the current algorithm.

## Confidence Levels

| Level | What's Included |
|-------|----------------|
| **100% Confirmed** | Batch=1 default, C matrix waste, env vars, launch bounds — all verified in source code |
| **High Confidence** | Occupancy calculations, CUDA graph overhead, SM-120 warp requirements — from code + architecture specs |
| **Speculative** | Exact TMAD/s projections — estimates that need measurement to confirm |
| **Aspirational** | 2000+ TMAD/s vision — ceiling analysis, requires novel algorithms |

## Quick Win (Today, No Code Changes)

```bash
export PROPMINER_BATCH=8
export PROPMINER_GRAPH_BATCH=8
export PROPMINER_TRIPLE_BUFFER=1
export PROPMINER_ASYNC_JOB_INSTALL=1
export PROPMINER_DEFER_SHARE_GPU=1
```

Expected: 290 → 380-420 TMAD/s (+30-45%)

## Documents

| Folder | Content |
|--------|---------|
| `00-core-analysis/` | Complete codebase map, cuPOW execution flow, bottleneck inventory |
| `01-competitor-analysis/` | SRBMiner, PeaMiner, ForgeMiner, BzMiner comparison |
| `02-5090-architecture/` | Blackwell GB202 deep-dive, SM-120 specifics, GDDR7, Tensor cores |
| `03-consensus-analysis/` | Pearl consensus constraints, fixed vs flexible |
| `04-cuda-kernel/` | Kernel-level optimizations, occupancy, memory patterns |
| `05-systems-bottlenecks/` | Systems-level I/O, memory, PCIe analysis |
| `06-integration-plan/` | Phased implementation roadmap |
| `TMAD_MASTERY_COMPREHENSIVE_ANALYSIS.md` | Consolidated analysis from all subagents |

## Roadmap

| Phase | Target | Effort |
|-------|--------|--------|
| Phase 1 — Env vars | 380-420 TMAD/s | 0 hours |
| Phase 2 — Code tweaks | 450-500 TMAD/s | 1 week |
| Phase 3 — Architecture | 550-650 TMAD/s | 1 month |
| Phase 4 — Algorithm | 700-800 TMAD/s | 3 weeks |
| Phase 5 — Kernel autotune | 800-1000 TMAD/s | 1 month |
| Phase 6 — Boundaries | 900-1200+ TMAD/s | 1 month |

## How to Run the Organization Script

The `organize.sh` script moves all analysis docs from across the project into this folder structure. Run from the PropMiner root:

```bash
bash DISCUSSION_WITH_AI/plans/00-tmad-dominance/organize.sh
```
