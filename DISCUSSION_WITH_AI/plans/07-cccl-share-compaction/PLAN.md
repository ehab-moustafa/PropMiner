# Plan: CCCL `cuda::atomic_ref` + Share Compaction

| Field | Value |
|-------|-------|
| **Status** | **Not built** |
| **Priority** | P3 — lowest hashrate lever |
| **Est. gain** | **<2%**; likely **0–0.1%** at batch=1 |
| **Effort** | **3–4 engineer-days** |
| **Risk** | Medium — graph capture, marginal ROI |

---

## How it works today

**After each batch** (`gpu_worker.cpp`):

1. `wait_for_batch(half)` — GPU done  
2. **`scan_winners(half, batch)`** — **O(batch)** host loop over pinned headers (640 B each)  
3. For each `status==1`: snapshot header → defer or inline share rebuild  

**Default batch=1** → one 640 B read per ~30 ms GEMM — **negligible**.

**In kernel** (`pow_utils.hpp` `write_host_signal_header`):

- `atomicCAS` spinlock on `HostSignalSync::global_lock`
- First hit per iter writes pinned header
- Reset: `cuMemsetD8Async(half.sync)` per sub-batch

**Defer-share already shipped** — heavy share work off mining thread; this plan only touches **discovery**.

---

## Proposed change

### A. `cuda::atomic_ref` (kernel)

Replace raw `atomicCAS` spinlock with libcudacxx `cuda::atomic_ref` + explicit memory orders. Cleaner signaling; possible tail latency win on rare multi-CTA races.

### B. `cub::DeviceSelect::Flagged` (host scan replacement)

Post-GEMM sidecar on `half.stream`:

```
hit_flags[batch] → DeviceSelect::Flagged → compacted indices → D2H
Host: process only selected indices (same share path)
```

**Constraint:** CCCL must **not** enter GEMM epilogue (register pressure).

---

## Before → after

```
TODAY:
  GEMM → batch_done_event → host for k in batch: if header[k].status==1

PROPOSED:
  GEMM → compact sidecar (CUB) → D2H few indices → same share path
```

At batch=1, sidecar launch cost may **exceed** host scan savings.

---

## Why we need it (honest)

- **Insurance** for future `batch≥16` or multi-candidate PoW per iter
- Atomic cleanup if touching `pow_utils.hpp` anyway
- **Not** a current production optimization

---

## Risks

| Risk | Notes |
|------|-------|
| DeviceSelect overhead > host scan | High @ batch=1 |
| CUDA graph incompatibility | Sidecar may need non-captured path |
| Winner semantics regression | Must preserve first-hit + header snapshot |
| CCCL vendoring / version coupling | New dependency |
| False positive hit flags | Graph header smear class of bugs |

---

## Testing

| Test | Pass |
|------|------|
| `PROPMINER_CCCL_SHARE_COMPACT=0` vs `=1` parity | Same winners + ShareFound |
| `--self-test` both modes | verify-ok |
| Graph + batch>1 | No duplicate winners |
| 300s bench inject **disabled** | ±0.1% TMAD |
| Inject every 10 batches 60s | Queue drains, no merkle mismatch |

**Proposed env:** `PROPMINER_CCCL_SHARE_COMPACT=0` default until validated.

---

## Rollback

- `PROPMINER_CCCL_SHARE_COMPACT=0`
- `PEARL_GEMM_CCCL_SHARE_COMPACT=0` (compile)
- CUB launch failure → fallback to `scan_winners`

---

## Go / no-go

| Go if | No-go if |
|-------|----------|
| `scan_winners` >100 µs at batch≥16 | batch=1 prod default |
| NCU shows spinlock contention | defer-share stalls already zero |
| Multi-candidate PoW roadmap | P0 kernel not shipped |
| Bench >0.1% uplift | |

**Recommendation:** **Defer** unless batch strategy changes.

---

## Files that would change

- `gemm/pow_utils.hpp`
- **New:** `pow/share_hit_compact.cu`
- `capi/pearl_gemm_capi.{cpp,h}`
- `gpu_worker.{cpp,h}`, `env_tuning.h`
- `CMakeLists.txt` (CCCL include)
- `tests.cpp`

**Do not modify:** `share_builder.cpp`, GEMM transcript mainloop.

---

*Plan only — P3 polish, not hashrate strategy.*
