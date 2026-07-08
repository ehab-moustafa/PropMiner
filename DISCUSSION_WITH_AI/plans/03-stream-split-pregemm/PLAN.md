# Plan: Stream-Split Pre-GEMM Chain

| Field | Value |
|-------|-------|
| **Status** | **Not built** |
| **Priority** | P2 — profile before implementing |
| **Est. gain** | **+1–5%** TMAD/s |
| **Effort** | **2–3 engineer-weeks** |
| **Risk** | Medium — seed ordering, graph capture |

---

## How it works today

Each mining iteration runs **five serial GPU stages** on **one stream** (`half.stream`):

| Stage | Work | ~% of iter |
|-------|------|------------|
| 1–4 | LCG → A hash → commitment → A noise_gen | **~5–12%** |
| 5 | `noisy_gemm` (noisingA + transcript GEMM) | **~85–92%** |

CUDA graphs capture all five × batch into one replay — eliminates CPU launch gaps but **does not overlap** stages 1–4 with stage 5.

**Already overlapped (separate feature):** `seed_copy_stream_` uploads 8-byte nonce H2D — not this plan.

---

## Proposed change

**Two CUDA streams + events:**

```
pre_gemm_stream:  stages 1–4  →  cudaEventRecord(pre_done[slot])
gemm_stream:      wait event  →  stage 5 (noisy_gemm)
```

**Double-buffer A-side tensors** (`A`, `AHash`, `CommitA`, `EAL`) in two slots so nonce *i+1* pre-GEMM overlaps nonce *i* GEMM.

**Recommended:** Hybrid — monolithic graph stays **default**; split behind `PROPMINER_STREAM_SPLIT_PREGEMM=1`.

Options: dual graphs vs direct kernel enqueue (dual graphs keep launch wins).

---

## Before → after

```
TODAY:
  [LCG→hash→commit→noise][GEMM][LCG→hash→...][GEMM]  (all one stream)

PROPOSED:
  pre:  [LCG→hash→commit→noise]_i  [LCG→hash→...]_i+1
  gemm:      [GEMM]_i                    [GEMM]_i+1
        overlap when pre_i+1 < GEMM_i duration
```

---

## Why we need it

- Pre-GEMM is small but non-zero; at batch>1, serializing nonces wastes GPU time
- Different resource units: BLAKE3 (ALU) vs tensor cores (GEMM) — some concurrent occupancy possible
- **Low ceiling** because GEMM dominates

---

## Risks

| Risk | Mitigation |
|------|------------|
| Seed read before H2D complete | `cudaStreamWaitEvent(seed_copy_done)` on pre stream |
| Rewriting `seed_dev` while pre consumes | Wait `gemm_done` before next upload |
| Single-buffer AHash/CommitA race | **Must** double-buffer |
| Graph capture cross-stream | Dual-graph or non-graph path only |
| Share path on `half.stream` | Drain **both** streams before share/σ work |

---

## Testing

1. Split vs monolithic — identical headers for 10⁴ nonces  
2. `--self-test` split ON/OFF  
3. `PROP_MINER_SELF_TEST_PROD=1`  
4. Nsight Systems — visible pre/gemm overlap ≥50% of pre duration  
5. ≥+1% TMAD over 30 min native Linux  
6. σ rotation + defer-share stress  

**Skip implementation** if profiling shows pre-GEMM <3% of batch wall time.

---

## Rollback

`PROPMINER_STREAM_SPLIT_PREGEMM=0` — no rebuild if hybrid design keeps old path.

Emergency: `PROPMINER_ASYNC_SEED=0` if seed-related rejects.

---

## Go / no-go

| Gate | Pass |
|------|------|
| G1 | Byte parity 10⁴ nonces |
| G2 | Self-test + prod self-test |
| G3 | Nsight overlap visible |
| G4 | ≥+1% TMAD median |
| G5 | No `gpu_cpu_jackpot_mismatch` 30 min |
| G6 | σ swap clean |

**Do not ship** if G2/G6 fail or G4 negative.

---

## Files that would change

- `src/host/pearl/gpu_worker.{cpp,h}`
- `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.{cpp,h}`
- `env_tuning.h` / `env_flags.h`

**Out of scope:** kernel math, share encoding, σ-install B path.

---

*Plan only — no implementation.*
