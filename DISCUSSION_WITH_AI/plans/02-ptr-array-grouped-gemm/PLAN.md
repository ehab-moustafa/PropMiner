# Plan: Ptr-Array Grouped GEMM

| Field | Value |
|-------|-------|
| **Status** | **Not built** |
| **Priority** | P1 — only if batch strategy changes |
| **Est. gain** | **0%** at batch=1; **+10–20%** if batch≥4 and kernel efficient |
| **Effort** | **4–6 weeks** |
| **Risk** | High — proof per slot, graph capture |

---

## How it works today

**Per nonce** (`pearl_capi_iter`), five serial stages on one stream:

1. `lcg_int7_fill` → regenerate A  
2. `tensor_hash` → A Merkle  
3. `commitment_hash`  
4. `noise_gen_A`  
5. `noisy_gemm` → **`batch=1` hardcoded** → one `transcript_gemm` launch (~65,536 CTAs)

**Batching today:** `iter_batch` = host for-loop calling `pearl_capi_iter` `count` times. CUDA graphs capture the full DAG × `graph_batch` but still **N separate GEMM launches** inside the graph.

**Defaults:** `PROPMINER_BATCH=1`, `PROPMINER_GRAPH_BATCH=1` (intentional — avoids multi-sub-batch bugs).

**Important:** Kernel `blockIdx.z` exists but **A is not offset per z** — multi-nonce needs ptr-array, not `batch>1` alone.

---

## Proposed change

**One kernel launch, multiple matmuls:**

- Device ptr-array: `ApEA_ptrs[g]` per group `g` (different nonce/A)
- Shared `BpEB` (resident per σ)
- Per-group `host_signal_header` slot + PoW write
- Pre-GEMM chain (stages 1–4) may still run N times unless separately fused

CUTLASS reference: `sm120_gemm_f8_f8_f32_tensor_op_group_gemm.cu` (pattern only; int8 transcript differs).

---

## Before → after

```
TODAY (batch=4):
  Graph: [iter0: pre+GEMM] [iter1: pre+GEMM] [iter2: ...] [iter3: ...]
  = 4 transcript_gemm launches

PROPOSED:
  [iter0..3: pre chains] → ApEA[0..3]
  [GROUPED_GEMM(ptrs, B)]  ← 1 launch
```

---

## Why we need it

- Attacks launch/epilogue Amdahl (~10–20% of iter when batch>1)
- CUDA graphs already hide CPU launch cost — grouped GEMM fuses **GPU-side** launches
- **Useless at prod batch=1** — zero ROI until product raises batch

---

## Risks

| Risk | Severity |
|------|----------|
| Transcript wrong per group | **Critical** |
| Graph capture + changing ptr arrays | High |
| Winner header smear across groups | High |
| VRAM: N simultaneous ApEA buffers | Medium |
| σ job switch with in-flight grouped launch | Medium |

---

## Testing

1. Serial `pearl_capi_iter` vs grouped — byte-identical transcript per group  
2. `--self-test` + `PROP_MINER_SELF_TEST_PROD=1`  
3. Matrix: `{graph on/off} × {batch 1,4,8,16} × {grouped on/off}`  
4. PoW: exactly one header slot fires per hit  
5. Pool canary 24h  
6. σ rotation soak  

---

## Rollback

- `PEARL_GEMM_GROUPED_GEMM=0` (compile) or `PROPMINER_BATCH=1` (runtime)
- Existing graph → `iter_batch` fallback already exists

---

## Go / no-go

| Gate | Requirement |
|------|-------------|
| **G0** | **Product decision: batch≥4 in prod** — else **NO-GO** |
| G1 | Per-group transcript byte-identical vs serial |
| G2 | Self-test pass |
| G3 | GeForce + consumer both support grouped path |
| G4 | ≥8% TMAD uplift at batch=8 (soft) |
| G5 | Pool <1% rejects |

---

## Files that would change

- `consumer/transcript_gemm_kernel.cu`, `blackwell/transcript_gemm_sm120_geforce.cu`
- `capi/pearl_gemm_capi.cpp` — workspace ptr pools, `noisy_gemm` dispatch
- `gpu_worker.cpp` — header slot layout (optional)
- CUTLASS grouped GEMM reference (pattern)

---

*Plan only — no implementation.*
