# 08 — SeedGenerator Re-enable Evaluation

**Status:** Decision document — do not implement  
**Date:** 2026-07-05  
**Scope:** `SeedGenerator` (dead code) vs current pinned async seed path in `GpuWorker`  
**Recommendation:** **Do not re-enable `SeedGenerator`.** Remove dead code from the build.

---

## 1. Executive Summary

PropMiner once planned a dedicated CPU thread (`SeedGenerator`) to pre-compute nonce seeds into a lock-free ring buffer so the GPU would never stall waiting on the host. That design was superseded by a simpler path: a per-GPU linear counter (`seed_base_ + global_iter`) uploaded via an 8-byte pinned `cudaMemcpyAsync` on a dedicated copy stream, overlapped with ping/pong compute.

**Re-enabling `SeedGenerator` would deliver ~0% hashrate gain and is likely negative** because:

1. **Batch time dominates by six orders of magnitude.** A production batch (M=8192, N=32768, batch=20) takes on the order of **~112 s** on WSL2 / RTX 5090-class hardware (`worker_orchestrator.cpp` notes >2 min per graph launch). Seed upload is **~1–10 µs**. Optimizing the seed path cannot move the needle.

2. **The current path already achieves full overlap.** Ping/pong halves plus `seed_copy_stream_` hide the 8-byte H2D copy entirely behind GPU GEMM work. There is no host-side stall to fix.

3. **Re-enabling adds cost without benefit.** A background thread, ring buffer, atomic SPSC bookkeeping, and integration glue add CPU cache pressure, complexity, and a second thread per GPU — for work that is currently one 64-bit addition on the hot path.

4. **Semantic mismatch.** `SeedGenerator` produces pseudo-random `(seed_lo, seed_hi)` pairs via xorshift64*. Pearl mining requires `seed_lo` to be a **monotonic nonce counter** and `seed_hi` to be **`sigma_seed` (constant for the job lifetime)**. The generator's `seed_hi` field does not match protocol semantics and was never wired into `GpuWorker`.

**Action:** Remove `seed_generator.cpp` from `CMakeLists.txt` and delete (or archive) the orphaned source files. Update stale documentation that still references `SeedGenerator`.

---

## 2. What SeedGenerator Did

### 2.1 Design intent

`SeedGenerator` (`src/host/pearl/seed_generator.h`, `.cpp`) was a CPU-side "matrix assembly line" that:

- Ran on a **dedicated host thread**
- Used **lock-free SPSC** semantics (`write_seq_` / `read_seq_` atomics)
- Maintained a **power-of-two ring buffer** (default 1024 slots)
- Produced **zero hashing / zero GEMM math** — only 64-bit seed values

Header comment (abridged):

> Performs ZERO hashing / NoisyGEMM math. It only produces 64-bit seed values and small metadata packets ahead of time so the GPU never stalls waiting for the host.

### 2.2 Packet format

```cpp
struct Packet {
    uint64_t seed_lo = 0;
    uint64_t seed_hi = 0;
    uint32_t batch_id = 0;
};
```

### 2.3 PRNG: xorshift64*

The background thread seeds a PRNG from `seed_base_ ^ 0x9E3779B97F4A7C15ULL` and fills the ring with:

```cpp
p.seed_lo = xorshift64star(prng_state);
p.seed_hi = xorshift64star(prng_state);
p.batch_id = static_cast<uint32_t>(seq++);
```

When the ring is full, the producer sleeps 10 µs rather than spin.

### 2.4 Consumer API

- `start(ring_size)` — spawn thread, allocate ring
- `stop()` — join thread
- `pop(out, count)` — drain up to `count` packets (SPSC consumer)
- `ready_count()` — back-pressure visibility

### 2.5 Integration status

**Never integrated.** No `#include "seed_generator.h"` outside its own `.cpp`. No `SeedGenerator` member in `GpuWorker`. The class compiles but is dead code linked into `propminer` via `CMakeLists.txt` line 185.

---

## 3. Current Seed Path

### 3.1 Per-GPU nonce partition

At `GpuWorker` construction:

```cpp
seed_base_ = (static_cast<uint64_t>(gpu_index_) << 48) |
             (static_cast<uint64_t>(now_ms() & 0xFFFF) << 32);
```

| Bits | Source | Purpose |
|------|--------|---------|
| `[63:48]` | `gpu_index_` | Disjoint nonce space per GPU |
| `[47:32]` | `now_ms() & 0xFFFF` | Time entropy at startup |
| `[31:0]` | `global_iter` (runtime) | Monotonic batch counter |

Multi-GPU deployments mine disjoint ranges without central coordination.

### 3.2 Hot loop (`GpuWorker::run`)

```
global_iter = 0
ping/pong double-buffer:

  FIRST batch:
    upload_next_seed_async(cur,  seed_base_ + global_iter)
    queue_batch(cur,             seed_base_ + global_iter, batch)
    upload_next_seed_async(other, seed_base_ + global_iter + batch)  // prime next
    global_iter += batch

  STEADY STATE (each iteration):
    queue_batch(cur, seed_base_ + global_iter, batch)
    global_iter += batch
    upload_next_seed_async(other, seed_base_ + global_iter + batch)  // overlap
    wait_for_batch(other) → scan_winners → handle_trigger
    swap ping/pong
```

Within a batch, the GPU derives per-matmul nonces as `seed_lo_base + i` for `i ∈ [0, count)`.

### 3.3 Pinned async upload (`upload_next_seed_async`)

Infrastructure allocated once in the constructor:

| Resource | Purpose |
|----------|---------|
| `pinned_seed_host_` | Single `uint64_t`, `cudaHostAlloc` |
| `seed_copy_stream_` | Non-blocking H2D copy stream |
| `seed_copy_done_event_` | Signals copy completion to compute stream |
| `half.seed_dev` | Per-half device pointer (caller-owned, graph `_ex` path) |

Upload path:

```cpp
*pinned_seed_host_ = seed_lo;
cudaMemcpyAsync(half.seed_dev_ptr, pinned_seed_host_, sizeof(uint64_t),
                cudaMemcpyHostToDevice, seed_copy_stream_);
cudaEventRecord(seed_copy_done_event_, seed_copy_stream_);
```

On graph launch, the compute stream waits:

```cpp
cudaStreamWaitEvent(half.stream, seed_copy_done_event_, 0);
gemm_.iter_batch_graph_launch_ex(half.workspace, half.stream);
```

The captured CUDA graph reads `seed_lo` from `half.seed_dev` via `pearl_capi_lcg_int7_fill_indirect` — the H2D copy is **not** captured, enabling true overlap.

### 3.4 What the host does *not* do

- No background seed thread
- No ring buffer allocation or atomic bookkeeping
- No PRNG on the hot path — only `seed_base_ + global_iter` (one add)
- No per-iter seed upload — one 8-byte upload covers an entire batch

---

## 4. Performance Analysis

### 4.1 Orders of magnitude

| Operation | Typical duration | Share of batch |
|-----------|------------------|----------------|
| Full batch (GEMM + LCG + tensor hash + noise + PoW check × batch) | **~60–120+ s** (prod shape, WSL2) | 100% |
| `seed_base_ + global_iter` (host add) | **< 1 ns** | ~0% |
| 8-byte pinned H2D (`cudaMemcpyAsync`) | **~1–10 µs** | **~0.00001%** |
| `SeedGenerator` xorshift + ring write (per seed) | **~10–50 ns** | ~0% |
| `SeedGenerator` thread + ring overhead (per batch) | **~1–5 µs** (pop + sync) | ~0% |

Even on native Linux with faster graph launches (~30–60 s per batch), seed work remains invisible in `last_iter_ms_` telemetry.

### 4.2 Overlap is already complete

The ping/pong design ensures:

```
Timeline (steady state):
  Compute stream (cur):  [========== batch N ==========]
  Copy stream:                              [8B seed for batch N+1]
  Compute stream (other): [========== batch N-1 ==========] (completing)
```

The seed copy finishes in microseconds while the batch runs for minutes. **There is no bubble to fill** with a pre-generated ring.

### 4.3 Where `SeedGenerator` would *hurt*

| Added cost | Impact |
|------------|--------|
| Extra OS thread per GPU | Context switches, L1/L2 pollution on E-cores or crowded NUMA |
| Ring buffer (~24 KB for 1024 packets) | Cache line contention on producer/consumer atomics |
| `pop()` on hot path | Atomic load/store pair vs one integer add |
| Integration complexity | More failure modes, harder debugging |
| Stale `seed_hi` in packets | Misleading if ever wired incorrectly |

### 4.4 Measured telemetry hook

`GpuWorker::run` records batch wall time:

```cpp
last_iter_ms_.store(ms);  // t1 - t0 around queue + wait + scan
```

Log line on first completion:

```
[gpu] first batch completed in %.0f ms -> %.0f H/s
```

On WSL2 with production batch=20, orchestrator warns users may wait **>30 s** before the first completion. Seed upload is not a contributor to that wait.

### 4.5 Verdict

| Question | Answer |
|----------|--------|
| Is seed upload on the critical path? | **No** — fully hidden by overlap |
| Would `SeedGenerator` reduce batch latency? | **No** |
| Would `SeedGenerator` increase hashrate? | **No** (~0% gain) |
| Could it decrease hashrate? | **Yes** — extra thread + atomics on a core that should stay idle |

---

## 5. Semantic Difference (PRNG vs Linear Counter)

### 5.1 Pearl protocol requirements

Each mining iteration derives matrix `A` via `lcg_int7_fill(A, seed_lo, seed_hi)` where:

- **`seed_lo`** — nonce / search counter (varies every iteration)
- **`seed_hi`** — `sigma_seed`, derived from job `σ` at install time (constant for the σ lifetime)

From `pearl_gemm_capi.h`:

> `sigma_seed` — seed_hi for lcg_int7_fill (= σ seed, constant within σ lifetime).

GPU kernel (`lcg_int7_fill_indirect_kernel`):

```cpp
uint64_t seed_lo = *seed_lo_base + seed_lo_offset;  // base uploaded by host + iter index
uint64_t base = splitmix64(seed_lo ^ splitmix64(seed_hi));  // seed_hi = p.sigma_seed
```

### 5.2 Current path semantics

| Field | Source | Behavior |
|-------|--------|----------|
| `seed_lo` base | `seed_base_ + global_iter` | Deterministic, monotonic, auditable |
| `seed_lo` per iter | `base + i` (GPU-side) | Sequential nonce within batch |
| `seed_hi` | `p.sigma_seed` from `install_params` | Fixed per job; changes only on new σ |

Share reconstruction uses the exact nonce:

```cpp
other->batch_seed_start + static_cast<uint64_t>(winner)
```

This must be reproducible by the pool / verifier. A linear counter with a known per-GPU base satisfies that contract.

### 5.3 SeedGenerator semantics

| Field | Source | Behavior |
|-------|--------|----------|
| `seed_lo` | xorshift64* | Pseudo-random, non-sequential |
| `seed_hi` | xorshift64* | **Wrong** — must be `sigma_seed`, not PRNG output |
| `batch_id` | monotonic seq | Metadata only; not used by GPU today |

### 5.4 Implications of re-enabling

1. **If only `seed_lo` from the ring were used** (ignoring `seed_hi`): nonce space becomes pseudo-random rather than sequential. Mining still works probabilistically, but:
   - Nonce reporting / auditing becomes harder
   - Risk of nonce collisions across restarts unless `seed_base_` partitioning is preserved
   - No performance benefit over `seed_base_ + global_iter`

2. **If both `seed_lo` and `seed_hi` were used**: **Protocol violation.** Matrix `A` would not match what the pool expects for a given `(σ, nonce)` pair. Shares would fail validation.

3. **Correct re-integration would require**: Use ring only for `seed_lo`, keep `sigma_seed` on GPU — meaning the generator's `seed_hi` field and half its PRNG work are wasted, and the remaining `seed_lo` randomization provides no advantage over a counter.

### 5.5 Search-space coverage

Both approaches cover the same 64-bit `seed_lo` space in the limit. Pearl PoW is a random search; sequential enumeration vs pseudo-random jumps are equivalent in expected time-to-share. Sequential counters are simpler, debuggable, and match existing share-reconstruction code.

---

## 6. Code Cleanup Plan

### 6.1 Files to remove

| File | Action |
|------|--------|
| `src/host/pearl/seed_generator.h` | Delete |
| `src/host/pearl/seed_generator.cpp` | Delete |

### 6.2 Build change

`CMakeLists.txt` — remove line:

```cmake
src/host/pearl/seed_generator.cpp
```

No other targets reference these files.

### 6.3 Documentation updates

| File | Stale text | Fix |
|------|------------|-----|
| `docs/RTX5090_LINUX_TASKS.md` line 6 | "CPU only generates seeds via `SeedGenerator`" | "CPU uploads an 8-byte nonce counter via pinned async H2D; all matrix/noise/GEMM math runs on the GPU" |

### 6.4 What to keep (already correct)

The current seed infrastructure in `GpuWorker` is **not** dead code and must remain:

- `seed_copy_stream_`, `pinned_seed_host_`, `seed_copy_done_event_`
- `upload_next_seed_async()`
- `half.seed_dev` + `iter_batch_graph_prepare_ex` / `_launch_ex`
- `seed_base_`, `global_iter` counter

### 6.5 Verification after cleanup

```bash
cd PropMiner && cmake --build build -j
# Confirm no undefined references to SeedGenerator
./build/propminer --bench 10 --rtx5090 --gpus 0
# Confirm [gpu] first batch queued / first batch completed logs appear
```

---

## 7. If Ever Needed Again

Revisit `SeedGenerator` (or a successor design) **only if** all of the following become true:

| Condition | Rationale | Current state |
|-----------|-----------|---------------|
| Batch latency drops to **< 1 ms** | Seed upload becomes non-negligible fraction | Batches are **60–120+ s** |
| Ping/pong overlap cannot hide H2D | Copy stream bubble visible in NSight | Overlap is complete today |
| Seed payload grows beyond 8 bytes | Multiple values per batch need staging | Single `uint64_t` base suffices |
| Per-batch seed derivation becomes CPU-expensive | e.g. HMAC or KDF per nonce | Currently one integer add |
| Protocol requires non-sequential nonce scheduling | e.g. deliberate space sharding | Linear counter + GPU partition is sufficient |

If triggered, a modern design would likely:

1. Use **`seed_lo` only** from any generator; **`seed_hi` stays `sigma_seed`**
2. Batch-upload a **vector of bases** (not individual PRNG pairs) if payload grows
3. Prefer **GPU-side counter increment** (device-side atomic or graph parameter) over a CPU thread
4. Integrate with **`cudaGraphExecKernelNodeSetParams`** for dynamic seed without H2D at all

None of these conditions apply to PropMiner's current RTX 5090 / Pearl V2 workload.

---

## 8. Effort to Remove vs Keep

### 8.1 Remove (recommended)

| Task | Effort |
|------|--------|
| Delete `seed_generator.{h,cpp}` | 2 min |
| Remove from `CMakeLists.txt` | 1 min |
| Update `RTX5090_LINUX_TASKS.md` | 5 min |
| Rebuild + smoke test | 10 min |
| **Total** | **~20 min** |

Risk: **None.** Code is unreferenced.

### 8.2 Keep as-is (status quo)

| Cost | Ongoing impact |
|------|----------------|
| ~90 lines of dead code in binary | Negligible object size |
| Confusion for future readers | Docs contradict implementation |
| False optimization target | Wasted investigation (this document) |

Effort to keep: **zero**, but carries documentation debt.

### 8.3 Re-enable and integrate (not recommended)

| Task | Effort |
|------|--------|
| Add `SeedGenerator` member to `GpuWorker` | 1 h |
| Wire `pop()` into `run()` loop | 2 h |
| Fix semantics (`seed_hi` must remain `sigma_seed`) | 1 h |
| Handle σ change / restart / ring drain | 2 h |
| Benchmark on WSL2 + native Linux | 2 h |
| Debug thread lifecycle + shutdown races | 2 h |
| **Total** | **~1–2 days** |

Expected hashrate delta: **0% ± noise**, likely **slightly negative**.

---

## Appendix A — Architecture Diagram (Current Path)

```
┌─────────────────────────────────────────────────────────────────┐
│                        GpuWorker::run()                         │
│  global_iter += batch each iteration                            │
│  seed = seed_base_ + global_iter   (one uint64 add)              │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌────────────────────────────┐     ┌─────────────────────────────┐
│   seed_copy_stream_        │     │   ping_/pong_.stream         │
│   pinned_seed_host_ (8B)   │     │   CUDA graph / iter_batch    │
│   cudaMemcpyAsync →        │     │   LCG + hash + GEMM + PoW    │
│   half.seed_dev            │     │   (60–120+ seconds)          │
└────────────┬───────────────┘     └─────────────────────────────┘
             │ seed_copy_done_event_
             └──────► cudaStreamWaitEvent ──► graph launch
```

## Appendix B — Architecture Diagram (SeedGenerator, Not Used)

```
┌──────────────────┐      SPSC ring       ┌─────────────────────────┐
│ SeedGenerator    │  ───────────────►    │ GpuWorker::run()        │
│ xorshift thread  │   Packet{lo,hi,id}   │ pop() → upload → queue  │
│ (never wired)    │                      │ (+ wrong seed_hi)       │
└──────────────────┘                      └─────────────────────────┘
```

## Appendix C — Key Source References

| Component | Path |
|-----------|------|
| SeedGenerator class | `src/host/pearl/seed_generator.{h,cpp}` |
| Current seed upload | `src/host/pearl/gpu_worker.cpp` — `upload_next_seed_async`, `run` |
| Seed infrastructure | `src/host/pearl/gpu_worker.h` — `seed_copy_stream_`, `seed_base_` |
| Graph seed indirection | `third_party/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — `_graph_prepare_ex`, `lcg_int7_fill_indirect` |
| Build inclusion | `CMakeLists.txt:185` |
| Batch timing context | `src/host/pearl/worker_orchestrator.cpp` — WSL2 >2 min note |
| Stale doc | `docs/RTX5090_LINUX_TASKS.md:6` |

---

**Decision:** Do not re-enable `SeedGenerator`. Remove dead code. The pinned async linear-counter path is simpler, semantically correct, and already fully overlaps seed I/O with GPU compute.
