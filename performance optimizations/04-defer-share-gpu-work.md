# 04 — Defer Share GPU Work Off the Mining Hot Path

**Status:** Proposal  
**Target:** PropMiner GPU worker (`GpuWorker`)  
**Scope:** Move rare share-trigger D2H and proof-input preparation from the mining loop to a dedicated side thread.  
**Expected steady-state gain:** +0–2% hashrate (pipeline bubbles eliminated only on share hits; shares are rare).

---

## 1. Executive Summary

PropMiner already splits **CPU proof build + gRPC send** onto `WorkerOrchestrator::share_sender_thread_func`, but the **GPU-thread share hit path** still performs substantial work inline inside `GpuWorker::run()`:

- Stream drain + A-matrix regeneration for the winning nonce
- Full A leaf-CV tensor hash on device
- Multiple D2H copies into per-half pinned staging
- Host memcpy into `ShareFound` vectors
- `IShareSink::submit`

That work runs on the same thread that drives the ping-pong GEMM conveyor belt. It is rare (only when `HostSignalHeader::status() == 1`), but when it fires it holds the GPU worker thread and contends for the **same half-buffer** that the ping-pong loop will reuse on the next iteration.

**Proposal:** On a PoW hit, the mining thread captures a lightweight trigger descriptor (~microseconds) and enqueues it. A **share-GPU worker thread** (per `GpuWorker`, or shared across GPUs on the same device context) performs today's `handle_trigger` GPU/D2H work asynchronously while the mining thread immediately continues batch scheduling.

**Why the gain is small but real:**

| Scenario | Effect |
|----------|--------|
| Steady state (no shares) | Zero change — hot path identical |
| Share hit, `T_trigger < T_batch` | Mining thread freed sooner; marginal (+0–1%) |
| Share hit, `T_trigger ≥ T_batch` | **Pipeline bubble** today — next `queue_batch` on that half may stall behind in-flight share work; deferral + explicit half-lock removes bubble (+1–2% during share windows) |

Shares are infrequent relative to batch rate, so **aggregate steady-state uplift is +0–2%**, with the main value being **latency isolation** and **correct ping-pong reuse** under slow proof prep.

---

## 2. Current Share Hit Path (Step by Step)

### 2.1 Ping-pong mining loop (`GpuWorker::run`)

Each iteration (after the priming `first` pass):

```
cur  = active half (just launched)
other = completed half (previous iteration)
```

| Step | Location | Action |
|------|----------|--------|
| 1 | `run()` ~L618 | `queue_batch(*cur, seed, batch)` — launch GEMM graph/iter batch on `cur.stream` |
| 2 | `run()` ~L625 | `upload_next_seed_async(*other, …)` — PCIe seed upload for batch after next |
| 3 | `run()` ~L630 | `wait_for_batch(*other, 0)` — block until `other.batch_done_event` (or stream sync) |
| 4 | `run()` ~L633 | `scan_winners(*other, batch)` — scan pinned host headers for `status == 1` |
| 5 | `run()` ~L634–638 | For each winner: `handle_trigger(*other, …)` |
| 6 | `run()` ~L639 | Watchdog heartbeat |
| 7 | `run()` ~L641–659 | Hashrate / timing stats |
| 8 | `run()` ~L661 | `std::swap(ping, pong)` |

On the **next** iteration, the half that just ran `handle_trigger` becomes `cur` and receives a new `queue_batch`. Therefore **all share GPU work on that half must finish before the following `queue_batch` on it**, or buffers/stream state collide.

### 2.2 Winner scan (`scan_winners`)

```cpp
// gpu_worker.cpp — conceptual
for (int k = 0; k < batch; ++k) {
    HostSignalHeader hdr(host_headers[k], header_size);
    if (hdr.status() == 1) {
        memcpy(host_header_storage[k], host_headers[k], header_size);  // snapshot
        winners.push_back(k);
    }
}
```

- Headers live in **pinned host memory** (`cudaHostAlloc`, portable).
- `host_header_storage` preserves the winning header across later header clears.
- Cost: O(batch) host reads; negligible vs. GEMM.

### 2.3 Share trigger (`handle_trigger`) — today fully inline

| Step | Sync? | Work |
|------|-------|------|
| 1 | — | Parse `HostSignalHeader` → `a_rows`, `b_cols` |
| 2 | **`cuStreamSynchronize(half.stream)`** | Drain completed batch on this half |
| 3 | GPU async | `gemm_.lcg_int7_fill(half.a, …, nonce, sigma_seed, stream)` — regenerate dA for winning nonce |
| 4 | **`cuStreamSynchronize(half.stream)`** | Wait for A regen (intermediate batch iters overwrote dA) |
| 5 | GPU async | `gemm_.tensor_hash_leaf_cvs(…)` — rebuild `half.a_leaf_cvs` on device |
| 6 | GPU async | D2H: full `a_leaf_cvs` → `pinned_leaf_cvs` |
| 7 | GPU async | D2H: per-row slices of `half.a` → `pinned_a_slice` |
| 8 | GPU async | D2H: opened 1 KiB leaves → `pinned_opened_leaves` |
| 9 | GPU async | D2H: `ctx.resident().b_hash()` → `pinned_hash_b` |
| 10 | **`cuStreamSynchronize(half.stream)`** | Single PCIe completion fence |
| 11 | Host | `memcpy` pinned → `std::vector` fields on `ShareFound` |
| 12 | — | Populate `ShareFound`, attach `sigma_ctx`, `sink_->submit(share)` |

**Pinned staging sizes** (per half, allocated in `HalfBuffers::allocate`):

- `pinned_leaf_cvs`: `a_leaf_cv_bytes` = `ceil(m·k / 1024) × 32` (e.g. m=8192, k=128 → 32 KiB)
- `pinned_a_slice`: `32 × k` bytes (max 32 opened rows)
- `pinned_opened_leaves`: `32 × 1024` bytes
- `pinned_hash_b`: 32 bytes

### 2.4 Downstream (already async)

```
GpuWorker::handle_trigger
    → IShareSink::submit(ShareFound)          // WorkerOrchestrator
        → share_mtx_ + pending_share_found_
        → share_sender_thread_func()
            → ShareBuilder::build()           // CPU: Merkle proofs, claimed hash, protobuf
            → PearlGrpcClient::send_event()
```

`ShareBuilder::build` is **CPU-only** and already off the GPU thread. This plan does **not** move `ShareBuilder`; it moves the **GPU regen + D2H** slice that still sits above `submit`.

---

## 3. What Blocks the GPU Thread Today

### 3.1 CPU thread blocking

The mining thread performs **three `cuStreamSynchronize` calls** and multiple host `memcpy`s inside `handle_trigger`. While the **other** half's GEMM batch runs concurrently (step 1 above), the mining thread is busy in `handle_trigger` instead of returning to the loop quickly.

Impact is usually secondary because the dominant work is on the GPU/stream.

### 3.2 Half-buffer occupancy (primary concern)

`handle_trigger` uses:

- `half.stream` — same stream the ping-pong loop uses for `queue_batch`
- `half.a`, `half.a_leaf_cvs`, pinned staging — same buffers the next batch on that half expects to own
- GPU kernels (`lcg_int7_fill`, `tensor_hash_leaf_cvs`) that mutate `half.a` / `half.a_leaf_cvs`

The loop **does not** launch the next batch on the share-hit half until one full iteration later, but it **does** assume that half is idle and consistent when `queue_batch` returns to it. If `handle_trigger` overruns one batch interval (`T_trigger > T_batch`), the next `queue_batch` on that half races with in-flight share work.

### 3.3 Allocations and locks on the hot path

- `handle_trigger` allocates several `std::vector<uint8_t>` and moves them into `ShareFound` (heap).
- `sigma_mtx_` lock when attaching `sigma_ctx` to `ShareFound`.

Both are rare-path costs but run on the mining thread today.

### 3.4 Multiple winners per batch

`scan_winners` can return **multiple** indices (batch size up to autotune value, commonly 8–20). Each calls `handle_trigger` **sequentially** on the same half, multiplying `T_trigger`.

---

## 4. Proposed Architecture (Queue + Worker Thread)

### 4.1 New component: `ShareGpuWorker` (per `GpuWorker`)

Introduce a dedicated thread owned by each `GpuWorker`:

```
Mining thread                         ShareGpuWorker thread
─────────────                         ─────────────────────
scan_winners → hit?
  ├─ build ShareTriggerJob (cheap)
  ├─ enqueue(job)
  ├─ mark half.share_busy = true
  └─ continue loop            ────►   dequeue(job)
                                        if half.share_busy: run GPU path
                                        build ShareFound
                                        sink_->submit(share)
                                        half.share_busy = false
                                        notify mining thread (optional)
```

### 4.2 `ShareTriggerJob` (lightweight enqueue payload)

Capture everything required to reproduce today's `handle_trigger` **without** touching GPU state at enqueue time:

```cpp
struct ShareTriggerJob {
    HalfBuffers* half;                    // non-owning; valid until GpuWorker destroy
    std::shared_ptr<SigmaContext> sigma;  // snapshot at trigger time
    std::vector<uint8_t> header;          // copy of host_header_storage[winner]
    uint64_t nonce;
    uint32_t installed_target_nbits;
    // Pre-parsed indices (avoid re-parsing on side thread)
    std::vector<uint32_t> a_rows;
    std::vector<uint32_t> b_cols;
};
```

**Mining-thread enqueue steps (target < 50 µs):**

1. `HostSignalHeader::extract_indices` (already have header snapshot)
2. Copy `shared_ptr<SigmaContext>` under `sigma_mtx_` (same as today)
3. Push job to `std::deque<ShareTriggerJob>` under `share_gpu_mtx_`
4. Set `half->share_in_flight = true` (atomic or under mutex)
5. `share_gpu_cv_.notify_one()`

**Do not** call `cuStreamSynchronize` on the mining thread.

### 4.3 Side-thread GPU path

Move the body of today's `handle_trigger` (steps 2–12 in §2.3) into `ShareGpuWorker::process_job`, unchanged in algorithm:

1. Sync `half.stream` (safe — mining thread no longer uses this half until `share_in_flight` clears)
2. A regen + leaf CV hash + D2H + vector assembly
3. `sink_->submit(share)`
4. Clear `half.share_in_flight`

### 4.4 Half reuse gate in `run()`

Before `queue_batch(*cur, …)`, assert the half is available:

```cpp
if (cur->share_in_flight.load()) {
    // Spin/yield with short timeout until ShareGpuWorker clears the flag,
    // OR skip scheduling this iteration and swap halves (see §5.2).
    wait_until_half_free(*cur, budget_ms);
}
queue_batch(*cur, …);
```

This makes the ping-pong contract explicit instead of implicitly relying on `T_trigger < T_batch`.

### 4.5 Threading model options

| Option | Pros | Cons |
|--------|------|------|
| **A. One ShareGpuWorker per GpuWorker** (recommended) | Simple ownership; each half's pinned buffers stay with its worker; no cross-GPU CUDA context issues | +1 thread per GPU |
| B. Extend `share_sender_thread` to also run CUDA | Fewer threads | CUDA context is thread-local; `share_sender` has no device binding today; high risk |
| C. Global share pool thread | Centralized queue | Must marshal `CUstream`/buffers across workers; not worth it |

**Recommendation:** Option A — `ShareGpuWorker` as a private nested helper inside `GpuWorker`, started/stopped with `GpuWorker::start/stop`.

### 4.6 CUDA context affinity

`ShareGpuWorker` thread must call `cudaSetDevice(device_index_)` once at thread entry (mirror `GpuWorker` constructor) so driver API calls on `half.stream` are valid.

---

## 5. Buffer Lifetime & Ping-Pong Safety

### 5.1 Ownership rules

| Resource | Owner during mining | Owner during share prep |
|----------|---------------------|-------------------------|
| `half.stream` | Mining thread launches GEMM | **Exclusive** ShareGpuWorker after `batch_done_event` |
| `half.a`, `half.a_leaf_cvs` | GEMM mutates per nonce | ShareGpuWorker regenerates A + leaf CVs |
| Pinned staging (`pinned_*`) | Unused post-batch | ShareGpuWorker D2H target |
| `host_headers[]` | Cleared on next `queue_batch` | **Not used** — job carries `header` copy |
| `host_header_storage[winner]` | Snapshot already taken in `scan_winners` | Copied into job at enqueue |

### 5.2 Timing diagram (one share hit)

```
Time →
other:  [==== GEMM batch N ====]| idle |···· share GPU work ····|
cur:         [==== GEMM batch N+1 ====][==== GEMM batch N+2 ====]
mining:      wait|scan|enqueue|swap|queue cur|wait|scan|…|queue other*|
                                                      ↑
                                    * blocked if share_in_flight still set
```

**Safety invariant:** `queue_batch(half)` MUST NOT run while `half.share_in_flight == true`.

### 5.3 Pinned buffer ping-pong

Pinned buffers are **per half**, not shared between ping and pong. ShareGpuWorker always uses the **same half's** pinned region as today. No cross-half pinned aliasing.

### 5.4 `sigma_ctx` lifetime

Enqueue copies `std::shared_ptr<SigmaContext>` at trigger time (same as today). Job rotation on the mining thread (`set_sigma`) does not invalidate in-flight jobs — shared_ptr keeps the old σ alive until ShareGpuWorker + ShareBuilder finish.

### 5.5 Triple-buffering (optional, not phase 1)

If `T_trigger` frequently exceeds `2 × T_batch`, consider a third half-buffer dedicated to share reconstruction. **Defer** unless profiling proves sustained stalls; complexity cost is high.

---

## 6. Implementation Phases

### Phase 0 — Instrumentation (no behavior change)

- Add scoped timers around `handle_trigger` substeps (sync, lcg fill, tensor hash, D2H, memcpy, submit).
- Log `T_trigger` vs `last_iter_ms` when a share hits.
- Counter: `share_in_flight_waits` when mining thread would have stalled.

**Exit criteria:** Median `T_trigger` and p95 measured on RTX 5090 at production shape (m=8192, n=32768, k=128, typical batch).

### Phase 1 — Queue + side thread (functional parity)

1. Add `ShareTriggerJob`, `share_gpu_queue_`, `share_gpu_mtx_`, `share_gpu_cv_`, `share_in_flight` per `HalfBuffers`.
2. Implement `ShareGpuWorker` thread loop in `gpu_worker.cpp`.
3. Split `handle_trigger` → `enqueue_share_trigger` (mining) + `process_share_trigger` (side).
4. Add `wait_until_half_free` gate before `queue_batch`.
5. Wire `start/stop` lifecycle (join side thread before freeing halves).

**Exit criteria:** Synthetic share injection (§9) produces byte-identical `ShareFound` vs. baseline; `ShareBuilder::VerifyShare` passes.

### Phase 2 — Mining-thread slim-down

1. Pre-parse indices on mining thread; pass in job (avoid duplicate `HostSignalHeader` work).
2. Move `ShareFound` vector allocations to side thread (or use pre-sized pool).
3. Optional: bound queue depth (max 4 jobs); drop/log if exceeded (should never happen).

### Phase 3 — Hardening & perf validation

1. Stress test: inject shares every batch for 60 s — no deadlocks, no pool rejects.
2. Benchmark steady-state hashrate with injection **disabled** — confirm ±0.1% noise floor.
3. Benchmark with injection at 1/share per 10 batches — measure bubble removal.

---

## 7. Files to Modify

| File | Changes |
|------|---------|
| `src/host/pearl/gpu_worker.h` | `ShareTriggerJob`; `share_in_flight`; queue + CV; `ShareGpuWorker` thread handle; split `handle_trigger` declarations; `wait_until_half_free` |
| `src/host/pearl/gpu_worker.cpp` | Refactor `handle_trigger`; `enqueue_share_trigger`; `process_share_trigger`; side-thread `run`; half-free gate in `run()`; `start/stop` join |
| `src/host/pearl/worker_orchestrator.h` | No structural change expected (`IShareSink` contract unchanged) |
| `src/host/pearl/worker_orchestrator.cpp` | Optional: metrics hook for share latency end-to-end |
| `src/host/pearl/pearl_types.h` | Only if new types promoted (prefer keeping jobs private to `gpu_worker`) |
| `src/host/tests.cpp` | Synthetic GPU share injection test (§9); parity test |
| `CMakeLists.txt` | Only if new `.cpp` file split out (optional `share_gpu_worker.cpp`) |

**No changes** to `share_builder.cpp` / `share_sender_thread_func` for phase 1.

---

## 8. Race Condition Analysis

| # | Hazard | Mitigation |
|---|--------|------------|
| R1 | `queue_batch` on half while share prep uses `half.stream` | `share_in_flight` gate before `queue_batch` |
| R2 | `install_sigma` reallocates/frees half buffers during share prep | `install_sigma` only on σ rotation; block rotation until both halves `!share_in_flight`, or cancel/share-drain queue first |
| R3 | `stop()` frees halves while side thread running | `stop()`: set `stop_flag_`, wake side thread, **join** before `ping_.free()` / `pong_.free()` |
| R4 | `set_sigma` replaces `sigma_` while job holds old `shared_ptr` | Safe — `shared_ptr` in job keeps ctx alive (same as today) |
| R5 | `target_nbits` / vardiff changes claimed hash validity | Continue storing `installed_target_nbits` on job at enqueue (today's `ShareFound` field) |
| R6 | Multiple winners same batch | Queue multiple jobs sequentially; all share `half` — process one-at-a-time on side thread (same as today's serial `handle_trigger` calls) |
| R7 | Winner header overwritten before snapshot | Already solved: `scan_winners` copies to `host_header_storage` before any deferral |
| R8 | CUDA context: mining vs. share thread | `cudaSetDevice` on share thread; no concurrent driver calls on **same** `half.stream` from two threads |
| R9 | Pinned `host_headers` cleared by `queue_batch` while parsing | Job owns header byte copy; never read `host_headers[k]` on side thread |
| R10 | `sink_->submit` from side thread vs. orchestrator `share_mtx_` | Already thread-safe (`WorkerOrchestrator::submit` uses mutex) |

---

## 9. Testing (Inject Synthetic Shares)

### 9.1 Unit-level (no GPU)

Existing coverage:

- `test_host_signal_header_index_extraction` — header parsing
- `ref::make_synthetic_share` + `ShareBuilder::VerifyShare` — CPU proof path

Extend with **`ShareTriggerJob` serialization parity**: given fixed header bytes + nonce + indices, assert pre-parsed job fields match `handle_trigger` inputs.

### 9.2 GPU integration — header injection

Add env-guarded hook in `scan_winners` or `run()`:

```
PROPMINER_INJECT_SHARE_SLOT=3   # force winner index in batch
PROPMINER_INJECT_SHARE_EVERY=N  # inject every N batches (0=off)
```

Implementation sketch:

1. After batch completes, if inject enabled, write `status=1` into `host_headers[slot]` with valid tile coords / register indices (reuse `test_host_signal_header_index_extraction` layout).
2. Run miner for M batches; assert:
   - `ShareGpuWorker` queue depth returns to 0
   - `ShareBuilder::VerifyShare` on dequeued `ShareFound` succeeds
   - No CUDA error on teardown

### 9.3 A/B parity test

1. Baseline build: inline `handle_trigger` (feature flag `PROPMINER_DEFER_SHARE_GPU=0`).
2. Deferred build: flag `=1`.
3. Same σ, same inject slot, same nonce — compare `ShareFound` fields (`a_slice`, `a_leaf_cvs`, `a_opened_leaf_data`, `hash_b`, indices).

### 9.4 Perf regression

- 300 s bench mode (`speed_test_seconds`) with inject **disabled** — hashrate within ±0.1%.
- 60 s with `INJECT_SHARE_EVERY=10` — no sustained hashrate collapse; `share_in_flight_waits` logged < 1% of batches.

### 9.5 Soak

- 1 h connected pool run (or local bench σ) — zero `share dropped: stale target or build failed` attributable to deferral.

---

## 10. Risks

| Risk | Severity | Notes |
|------|----------|-------|
| **Stale buffers** — job references freed half after `stop()` | High | Strict join ordering in `stop()`; drain queue before free |
| **Missed shares** — mining overwrites `half.a` before side thread regen | High | Prevented by `share_in_flight` + not reusing half until clear; A regen uses captured `nonce`, not live buffer state |
| **σ rotation during queued jobs** | Medium | Hold `shared_ptr` per job; block `install_sigma` until queue empty **or** let old jobs complete with old σ (preferred) |
| **Queue blow-up** (many winners) | Low | Batch size ≤ ~20; serial processing; bound queue |
| **Proof reject `a_merkle_mismatch`** | High | Do not change algorithm order: drain stream → lcg fill → tensor_hash_leaf_cvs → D2H; parity test catches regressions |
| **Complexity / debug cost** | Medium | One extra thread per GPU; logging and flags essential |
| **Marginal ROI** | Medium | Steady-state gain tiny if `T_trigger < T_batch` always; profiling in Phase 0 de-risks |

---

## 11. Effort Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 0 — Instrumentation | 0.5 day | None |
| Phase 1 — Queue + side thread + parity | 1.5–2 days | Phase 0 optional but helpful |
| Phase 2 — Slim hot path | 0.5 day | Phase 1 |
| Phase 3 — Stress + bench | 1 day | Phase 1 |
| **Total** | **3–4 engineer-days** | RTX 5090 test GPU |

---

## 12. Go/No-Go Criteria

### Go if **any** of:

1. **Phase 0 profiling** shows `p95(T_trigger) > 0.1 × T_batch` (share prep risks pipeline bubbles).
2. **`share_in_flight` wait** would trigger on > 0.1% of batches in soak test (measured via Phase 0 hooks).
3. Product goal includes **sub-100 ms share submit latency** under high difficulty variance (side thread frees mining thread for faster enqueue-to-submit pipeline).
4. Multi-winner batches observed in the wild (`winners.size() > 1`).

### No-Go / defer if **all** of:

1. `T_trigger` p95 < 5 ms and `T_batch` > 200 ms — share work fits comfortably in the inter-batch window with no measured stalls.
2. Steady-state hashrate noise dominates (+0.1% unmeasurable).
3. Team priority favors higher-impact items (graph capture, GEMM tuning, σ-install overlap).

### Decision matrix

| Profile result | Recommendation |
|----------------|----------------|
| `T_trigger ≪ T_batch`, zero stalls | **Optional** — implement for cleanliness; expect ~0% steady-state gain |
| `T_trigger ≈ T_batch` intermittently | **Go** — expect +0.5–1% during share windows |
| `T_trigger > T_batch` or multi-winner serial pile-up | **Strong Go** — explicit half-lock fixes correctness + perf |

---

## Appendix A — Key Code References

| Symbol | File | Role |
|--------|------|------|
| `GpuWorker::run` | `gpu_worker.cpp` | Ping-pong loop, calls `handle_trigger` inline today |
| `GpuWorker::scan_winners` | `gpu_worker.cpp` | Host header scan, snapshot to `host_header_storage` |
| `GpuWorker::handle_trigger` | `gpu_worker.cpp` | GPU regen + D2H + `sink_->submit` |
| `HalfBuffers::allocate` | `gpu_worker.cpp` | Pinned staging allocation (`kMaxShareRows = 32`) |
| `WorkerOrchestrator::submit` | `worker_orchestrator.cpp` | Enqueues raw `ShareFound` |
| `WorkerOrchestrator::share_sender_thread_func` | `worker_orchestrator.cpp` | `ShareBuilder::build` + gRPC |
| `ShareBuilder::build` | `share_builder.cpp` | CPU Merkle proofs + protobuf (stays here) |
| `ShareFound` | `pearl_types.h` | Payload crossing GPU → orchestrator boundary |

## Appendix B — Environment Flags (proposed)

| Flag | Default | Purpose |
|------|---------|---------|
| `PROPMINER_DEFER_SHARE_GPU` | `1` after ship | Toggle deferred vs. inline `handle_trigger` |
| `PROPMINER_INJECT_SHARE_SLOT` | `-1` | Force winner index for tests |
| `PROPMINER_INJECT_SHARE_EVERY` | `0` | Inject synthetic share every N batches |
| `PROPMINER_LOG_SHARE_TIMING` | `0` | Verbose `T_trigger` breakdown |

---

*Document version: 2026-07-05 — initial proposal.*
