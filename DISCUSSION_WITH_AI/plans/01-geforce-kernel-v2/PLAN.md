# Plan: GeForce Kernel v2 (CUTLASS SM120 Warp-Specialized)

| Field | Value |
|-------|-------|
| **Status** | Phase 1 implemented — **v2 default ON** on blackwell builds; rollback via `geforce_v1` / `consumer` |
| **Priority** | **P0/P1** — highest ROI unimplemented lever |
| **Est. gain** | **+10–25%** incremental on v1 (~330→370–470 TMAD/s) |
| **Effort** | **4–6 engineer-weeks** (1 CUDA engineer + 5090) |
| **Risk** | Moderate — transcript byte-identity gate required |

---

## How it works today

**Production path (GeForce v1):** `transcript_gemm_sm120_geforce.cu`

- **288 threads/CTA:** warp 8 (thread 256) = TMA producer; warps 0–7 = INT8 MMA + Pearl transcript
- Loads A/B via `tma_tile_loader.cuh` + manual `ClusterTransactionBarrier` pipeline (2 stages)
- Same proof-canonical tile: **128×256×128**, SM80 `m16n8k32` atom
- Default ON when `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1` (blackwell builds)
- Rollback: `PEARL_GEMM_KERNEL=consumer`

**Fallback:** `consumer/transcript_gemm_kernel.cu` — 256 threads, all do `cp.async` loads + MMA.

**Dispatch:** `pearl_capi_noisy_gemm` → `launch_transcript_gemm_sm120_geforce_headless` or consumer headless.

**Limitation of v1:** Hand-rolled producer/consumer handoff; symmetric A+B reload every K-tile; no pingpong epilogue overlap; descriptors rebuilt each launch.

---

## Proposed change

Evolve v1 using NVIDIA CUTLASS SM120 patterns (examples 79/87):

| Upgrade | What |
|---------|------|
| **CUTLASS `PipelineTmaAsync`** | Replace manual barriers with standard warp-specialized pipeline |
| **Pingpong schedule** | Overlap transcript XOR + PoW epilogue on tile *N* while TMA prefetches tile *N+1* |
| **Asymmetric TMA** | B is VRAM-resident per σ — load B once per CTA/replay; A every nonce |
| **Deeper stages** | Explore `KBLOCK=64, STAGES=3/4` within **99 KB** SMEM cap (not 128×3) |

**Proof-locked (must NOT change):** SM80 MMA atom, 128×256 tile, 16 transcript slots, `partition_C`, headless PoW semantics.

**Versioning:** `PEARL_GEMM_KERNEL=geforce_v2` or in-place upgrade; `geforce_v1` / `consumer` for rollback.

---

## Before → after

```
TODAY (v1):
  [TMA warp loads A+B] → wait → [8 warps MMA + transcript]  (serial per K-tile)

PROPOSED (v2):
  [Producer: pingpong TMA A (+ B once)]  ║  [Consumers: MMA tile N + epilogue]
                                         ║  [while producer fetches tile N+1]
```

---

## Why we need it

- v1 already beats consumer (+10–25%) but SMs look busy while tensor pipes run ~36% of rated 838 TOPS
- Epilogue (transcript XOR + BLAKE3) is ~10–20% of CTA time — pingpong attacks that tail
- Resident B on host is not exploited in kernel — redundant B TMA every iter
- This is the **main path to ~450–600 TMAD/s** combined with knob tuning

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Transcript byte drift | **Critical** | `verify_geforce_transcript.sh` + v2 vs v1 memcmp |
| SMEM > 99 KB | High | Cap autotune; static_assert SharedStorage size |
| Register spill / occupancy drop | Medium | NCU before ship; may erase gains |
| Stale B in smem after σ swap | High | Fence on `install_sigma`; drain streams |
| CUTLASS API churn | Low | Pin CUTLASS 4.4.0 |

---

## Testing

1. Extend `scripts/verify_geforce_transcript.sh` — v2 vs consumer + v2 vs v1, 100 trials
2. `./build/propminer --self-test --rtx5090 --gpus 0` (consumer + geforce + v2)
3. `PROP_MINER_SELF_TEST_PROD=1` — production N/K
4. `scripts/pre_deploy_gate.sh` with `PROPMINER_GATE_GEFORCE=1`
5. `./scripts/profile_gemm_ncu.sh` — Tensor Active ↑, global_ld ↓
6. `--bench 120` — ≥+10% vs consumer baseline
7. **24–48 h pool soak** — rejected shares <1%

---

## Rollback

| Action | Rebuild? |
|--------|----------|
| `PEARL_GEMM_KERNEL=consumer` | No |
| `PEARL_GEMM_KERNEL=geforce` (v1) | No |
| `PEARL_GEMM_BLACKWELL_GEFORCE_V2=0` at compile | Yes |

---

## Go / no-go gates

| # | Gate |
|---|------|
| G1 | Transcript memcmp 100% pass |
| G2 | Self-test + prod-shape self-test |
| G3 | v2 ≡ v1 on transcript (no regression) |
| G4 | SMEM ≤ 101376 B, launch succeeds |
| G5 | NCU: ≥+8 pct points SM throughput vs v1 |
| G6 | Bench ≥+10% vs consumer |
| G7 | 24h pool: <1% rejects |
| G8 | CUDA graphs + warmup clean |

**No-go:** any transcript diff, TMAD below consumer, occupancy collapse.

---

## Files that would change

- `third_party/pearl-gemm/csrc/blackwell/transcript_gemm_sm120_geforce.{cu,h}`
- `third_party/pearl-gemm/csrc/consumer/tma_tile_loader.cuh`
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp`
- `CMakeLists.txt`, `csrc/capi/Makefile`
- `scripts/verify_geforce_transcript.sh`, `scripts/tune_kernel_knobs_common.sh`

**Reference (read-only):** CUTLASS `examples/79_blackwell_geforce_gemm/`, `sm120_mma_tma.hpp`

---

*Phase 1 landed — see MERGED_PLAN.md and AUDIT.md. v2 default promotion blocked on 5090 gates.*
