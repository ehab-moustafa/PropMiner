# Plan: σ-Install B `tensor_hash` Batching

| Field | Value |
|-------|-------|
| **Status** | **Not built** (kernel optimization) |
| **Priority** | P2 — job-switch / startup latency |
| **Est. gain** | **~0% steady TMAD**; faster first batch + job rotation |
| **Effort** | **1–2 weeks** |
| **Risk** | **Critical** if BHash drifts — 100% share rejects |

---

## How it works today

**Each new pool job** → new `SigmaContext` → **full B install** (once per σ):

1. Allocate `ResidentBState` (~2.2 GiB @ N=262144/K=4096)
2. **`bseed_expand_and_tensor_hash_leaf_cvs`** (GPU fused):
   - BSeed XOF → fill resident `B`
   - Keyed BLAKE3 per 1024 B leaf
   - In-block Merkle + multi-block merge + reduce
3. `cuStreamSynchronize` → D2H leaf CVs
4. `commitment_hash` → `noise_gen` B → `noise_B`
5. CPU Merkle verify vs GPU `BHash`

**First job:** always **synchronous** — mining blocked until done (can be **60–120 s** wall at prod N including graph capture).

**Job 2+:** `PROPMINER_ASYNC_JOB_INSTALL=1` (default) overlaps install on background thread **if VRAM guard passes** — at N=262144/K=4096 on 32 GiB often **self-disables**.

**Steady mining:** B never re-hashed per nonce — **no TMAD impact**.

---

## Proposed change

**Warp-lane batched BLAKE3 parent compress** in `merkle_tree_utils.hpp`:

| Bottleneck today | Fix |
|------------------|-----|
| Serial `tid==0` parent fold in `compute_blake_mt` | Distribute across warp lanes (BLAKE3 `compress_parents_parallel` pattern) |
| 16 sequential compress per 1024 B leaf | Optional: batch leaf compress per warp |
| Stage 2/3 MT merge | Tune only if profiling shows remainder |

**Optional:** `blake3_xof_many` style BSeed counter batching.

**Host unchanged** if API-stable drop-in in pearl-gemm.

---

## Before → after

```
TODAY:
  Stage-1: 8192 blocks × (XOF + 16× compress + in-block Merkle)
  Stage-2/3: serial parent fold (tid==0) + reduce

PROPOSED:
  Same semantics, parallel parent compress across warp lanes
  → shorter GPU install tail → faster sync path + shorter async overlap window
```

---

## Why we need it

- σ-install BLAKE3 is **100–500 ms GPU** portion of long first-batch window
- Faster install → less **0 H/s** on sync job switch; shorter async overlap hashrate dip
- **Largest absolute win** at K=4096 (8192 leaf blocks) — Salad/Kryptex shape
- **Does not** raise sustained TMAD/s

---

## Risks

| Risk | Mitigation |
|------|------------|
| BHash / Merkle root drift | BLAKE3 test vectors + CPU tree golden |
| Leaf CV mismatch | Share proofs use leaf CVs — byte compare |
| Keyed hash flag errors | `CHUNK_START/END`, root vs inner params |
| Faster install ≠ less VRAM peak | Async install peak unchanged |

---

## Testing

| Gate | Method |
|------|--------|
| BLAKE3 vectors | Import `test_vectors.json` subset |
| GPU BHash == CPU root | Hard-fail in CI (today WARN only) |
| `--self-test` | Default + `PROP_MINER_SELF_TEST_PROD=1` |
| `run-parity.sh` | pearl-gemm portable |
| Micro-bench | `bseed_expand_and_tensor_hash` wall @ K∈{128,4096} |
| NCU | Stage-1 time ↓, occupancy OK |
| Pool | No `verify-fail`; no `B Merkle root != GPU BHash` WARN |

---

## Rollback

- Revert `libpearl_gemm_capi.so` build
- Optional future: `PEARL_TENSOR_HASH_WARP_BATCH=0`
- Independent: `PROPMINER_ASYNC_JOB_INSTALL=0`

**Host paths unchanged** — pearl-gemm fork only.

---

## Go / no-go

| Gate | Required |
|------|----------|
| BLAKE3 vector parity | Yes |
| Self-test + prod self-test | Yes |
| BHash == CPU root K=128 and K=4096 | Yes |
| No steady TMAD regression ≥30 min | Yes |
| Pool canary 1h | Recommended |

**Worth doing if:** frequent job supersede / slow first-batch UX — **not** for TMAD ceiling.

---

## Files that would change

- `third_party/pearl-gemm/csrc/tensor_hash/merkle_tree_utils.hpp` (**core**)
- `bseed_merkle_tree_roots_kernel_sm80.hpp`
- `compute_blake_mt_kernel.hpp`, `reduce_roots_kernel.h`
- `blake3/blake3.cuh` (optional batch helper)
- `capi/pearl_gemm_capi.cpp` (dispatch only if geometry changes)
- `src/host/tests.cpp` (golden tests)

**Unchanged:** `gpu_worker.cpp` job switch logic (benefits automatically).

---

*Plan only — isolated pearl-gemm PR after correctness harness.*
