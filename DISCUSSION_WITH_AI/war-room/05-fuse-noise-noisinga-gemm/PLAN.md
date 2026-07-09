# Plan: Fuse noise + noisingA + GEMM

| Field | Value |
|-------|-------|
| **Status** | **Not built** |
| **Priority** | P2 — after GeForce v2 stable |
| **Est. gain** | **+1–3%** TMAD/s |
| **Effort** | **3–5 weeks** (full); **1–2 weeks** (partial) |
| **Risk** | High — noise BLAKE3 indexing, ApEA clamp |

---

## How it works today

Per nonce, **before** transcript GEMM on RTX 5090:

| Step | Kernels | Output |
|------|---------|--------|
| `noise_gen` | 1 | `EAL`, `EAR` from `CommitA` |
| Portable noisingA | **2–3** | `AxEBL_fp16`, **`ApEA` in global VRAM** |
| `noisy_gemm` | 1 (65k CTAs) | Reads `ApEA` from HBM |

**Total pre-GEMM:** ~4–5 kernel launches. CUDA graphs capture them — launch overhead already amortized.

**B-side noise:** Done once at σ-install (`BpEB` resident) — **not** in this plan.

**Time budget:** Pre-GEMM ≈ 8–15% of “fused overhead”; GEMM ≈ 85–92%. Full elimination of pre-GEMM → **~1–3%** wall time max.

---

## Proposed change

**Tile-local GEMM prologue** inside consumer + GeForce kernels:

```
Per CTA tile (128×256):
  [BLAKE3 noise for rows/cols in tile] → [ApEA in smem] → [MMA + transcript]
```

**Do not** write full `ApEA(M×K)` to HBM on hot path.

**Phasing:**

| Phase | Scope | Gain |
|-------|-------|------|
| A — Full fusion | noise + noisingA + GEMM prologue | +1–3% |
| B — Partial | noise + noisingA only; GEMM reads smem handshake | +0.5–1.5% |

**Flag:** `PEARL_GEMM_FUSED_PREGEMM=1` (default off until gated).

---

## Before → after

```
TODAY:
  noise_gen → matmul+clamp → ApEA[HBM] → GEMM reads ApEA[HBM]

PROPOSED:
  [fused prologue in GEMM CTA] → MMA uses smem ApEA directly
```

---

## Why we need it

- Eliminates ~1 MiB ApEA HBM write+read per iter (M=8192, K=128)
- Better L2 locality for A + noise
- **Not** launch-count driven (graphs already batch launches)

---

## Risks

| Risk | Severity |
|------|----------|
| BLAKE3 `data[0/1]=r+1` indexing wrong | **Critical** |
| ApEA ±127 clamp wrong | **Critical** |
| Transcript byte identity | **Critical** |
| Register/smem pressure → occupancy drop | Medium — may **erase** gain |
| Share rebuild needs ApEA rows | Medium — spill on hit only |
| Graph recapture on toggle | Medium |

---

## Testing

1. **New:** `scripts/verify_fused_pregemm.sh` — memcmp `EAL`, `EAR`, `ApEA`, `AxEBL` vs unfused
2. Transcript / jackpot identity vs consumer
3. `--self-test` + `PROP_MINER_SELF_TEST_PROD=1`
4. `PEARL_GEMM_KERNEL=consumer` and `geforce` both
5. `PROPMINER_DEFER_SHARE_GPU={0,1}`
6. NCU: occupancy must not drop >5%
7. Bench: ≥+0.5% mean TMAD (stretch +1–3%)

**Reference:** `share_builder.cpp` CPU golden noise.

---

## Rollback

- `PEARL_GEMM_FUSED_PREGEMM=0` (runtime or compile)
- Independent of `PEARL_GEMM_KERNEL` selector
- `destroy_iter_graph` on workspace reinstall (existing)

---

## Go / no-go

Default **off** until G1–G6 pass:

| Gate | Criterion |
|------|-----------|
| G1 | Buffer memcmp harness 0 mismatches |
| G2–G3 | Self-test + prod self-test |
| G4 | Both kernel backends |
| G5 | Defer-share modes |
| G6 | CUDA graph multi sub-batch |
| G7 | Occupancy neutral |
| G8 | Bench ≥+0.5% |

---

## Files that would change

- `consumer/transcript_gemm_kernel.cu`, `blackwell/transcript_gemm_sm120_geforce.cu`
- `gemm/noise_generation_kernel.h`, `pearl_noisingA_kernel.h`
- `capi/pearl_gemm_capi.cpp`, `portable_int8_helpers.cu`
- `capi/Makefile`, `CMakeLists.txt`
- **New:** `verify_fused_pregemm.sh`

**Read-only baseline:** `share_builder.cpp`, `transcript_canonical.cuh`, `pow_utils.hpp`

---

*Plan only — proof-gated micro-optimization.*
