# Grouped GEMM Phase 1 — Audit

Date: 2026-07-09

## Scope shipped

- Ptr-array grouped headless transcript GEMM for **batch ≥ 4** via `pearl_capi_iter_batch`
- **Consumer** kernel + **GeForce v2** kernel (per-group TMA A descriptors)
- Compile: `PEARL_GEMM_GROUPED_GEMM=1` (default **ON** for blackwell)
- Runtime: auto when `count >= 4`; instant rollback `PEARL_GEMM_GROUPED=0`
- **batch 1–3**: unchanged serial `pearl_capi_iter` loop
- **CUDA graphs**: unchanged (still N serial iters inside capture — grouped not in graph)

## Safety gates

| Gate | Mechanism |
|------|-----------|
| batch &lt; 4 | `pearl_grouped_gemm_enabled()` → serial path |
| Rollback | `PEARL_GEMM_GROUPED=0` |
| Per-group PoW | `HostSignalSync[batch]`, `CommitA[i]`, `host_signal_header[i]` |
| GeForce v1 | Falls back to consumer grouped with WARN |
| prod default batch=1 | Grouped never activates |

## Files touched

- `consumer/transcript_gemm_kernel.cu` — grouped ptr resolution
- `blackwell/transcript_gemm_sm120_geforce_v2.cu` — grouped + TMA array
- `capi/pearl_gemm_capi.cpp` — pools, iter_batch dispatch
- `gemm/transcript_gemm_grouped.h` — constants
- `CMakeLists.txt`, `capi/Makefile` — build flags
- `scripts/verify_grouped_gemm_transcript.sh`

## 5090 gates (required before prod)

1. `./scripts/verify_grouped_gemm_transcript.sh`
2. `strings build/libpearl_gemm_capi.so | grep grouped` → `+grouped`
3. Pool soak with `PROPMINER_BATCH=4` ≥ 1 h

## Deferred (Phase 2)

- Grouped inside CUDA graph capture
- `gpu_worker.cpp` graph path integration

## Bugbot fixes (post-audit)

- v2 grouped TMA build uses **host** ApEA pointer table (pool offsets), not `d_apea_ptrs`
- `d_header_ptrs` device array uploaded before launch (pinned header targets)
- Consumer grouped + `PEARL_CONSUMER_USE_TMA_EXPERIMENT`: not supported (prod uses `cp_async`)
- `pearl_grouped_gemm_enabled` caps at `kGroupedGemmMaxBatch` (16); larger batches use serial loop
- GeForce v1 grouped: WARN then consumer fallback (no silent skip)
- `--self-test` honors `PROPMINER_BATCH` for verify script

## Error codes (grouped pools)

- `-57` count &gt; max (16)
- `-58`..`-65`, `-68`, `-69` pool / ptr upload failures
