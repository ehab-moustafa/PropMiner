# PropMiner Systems & I/O Bottlenecks Analysis

**Target:** RTX 5090 — Current ~290 TMAD/s → Target 700-800+ TMAD/s
**Date:** 2026-07-09

---

## Executive Summary

PropMiner is already architecturally sophisticated with double/triple buffering, CUDA graphs, pinned PCIe staging, async job installation, and deferred share GPU work. The remaining bottlenecks are primarily:

1. **Synchronous sigma-install stalls** during job switches (blocks GPU for 1-5 seconds)
2. **CPU-side share proof reconstruction** — even deferred, the share rebuild path is heavy on CPU and requires PCIe D2H transfers
3. **Network send latency** — shares submitted one-at-a-time via TCP with 45s ack timeout
4. **Thermal pause overhead** — full GPU pause/resume causes ~2s mining gap
5. **nvidia-smi polling latency** — blocking `popen()` every 5 seconds on the stats loop

---

## 1. Memory Allocation Pattern Analysis

### 1.1 GPU Memory (HBM) — Currently Well-Optimized

**File:** `gpu_worker.cpp`, lines 100-196

**Allocation pattern: Large pre-allocated buffers, no per-batch allocations.**

Each half-buffer allocates these device pointers (lines 120-134):

| Buffer | Size (bytes) | Purpose |
|--------|-------------|---------|
| `a` | M×K = 8192×128 = **1 MiB** | A matrix (ping/pong rotate) |
| `a_hash` | 32 | A Merkle hash |
| `roots` | 256 | Merkle roots scratch |
| `commit_a` | 32 | A commitment hash |
| `commit_b` | 32 | B commitment hash |
| `eal` | M×R = 8192×128 = **1 MiB** | EAL (encryption layer) |
| `eal_fp16` | 2×M×R = **2 MiB** | FP16 EAL |
| `ear_r` | K×R = 128×128 = **16 KiB** | EAR R-major |
| `ear_k` | R×K = 128×128 = **16 KiB** | EAR K-major |
| `ax_ebl_fp16` | 2×M×R = **2 MiB** | A×EBL FP16 |
| `apea` | M×K = 8192×128 = **1 MiB** | APEA |
| `a_scales` | M×4 = **32 KiB** | A scales (float) |
| `c` | M×N×2 = 8192×262144×2 = **4 GiB** | C matrix (fp16, never read in pure-miner mode) |
| `sync` | 256 | GPU-side sync scratch |
| `pow_target` | 32 | PoW target |
| `a_leaf_cvs` | (M×K/1024)×32 = **8 KiB** | A leaf Merkle CVs |
| `seed_dev` | 8 | Device-side seed pointer |

**ResidentBState** (sigma_context.cpp, lines 36-66) allocates per-sigma:

| Buffer | Size (bytes) | Purpose |
|--------|-------------|---------|
| `b` | N×K = 262144×128 = **32 MiB** | B matrix (resident) |
| `b_hash` | 32 | B Merkle hash |
| `ebr` | N×R = 262144×128 = **32 MiB** | EBR |
| `ebr_fp16` | 2×N×R = **64 MiB** | FP16 EBR |
| `ebl_r` | K×R = 128×128 = **16 KiB** | EBL R-major |
| `ebl_k` | R×K = 128×128 = **16 KiB** | EBL K-major |
| `earx_bpeb` | 2×N×R = **64 MiB** | EAR×BPEB |
| `bpeb` | N×K = 262144×128 = **32 MiB** | BPEB |
| `b_scales` | N×4 = **1 MiB** | B scales (float) |
| `leaf_cvs` | (N×K/1024)×32 = **256 MiB** | B leaf Merkle CVs |
| `key` | 32 | Job key |

**Total per half:** ~4×(1+1+2+16KiB+16KiB+2MiB+1MiB+32KiB) + 4GiB + small ≈ **~4.1 GiB**
**Total with ResidentBState:** ~4.1 + ~224 MiB ≈ **4.3 GiB per half**
**With triple buffering + resident B:** ~4.3×3 + 224MiB ≈ **~13.1 GiB**

### 1.2 Host Memory (Pinned PCIe Staging)

**File:** `gpu_worker.cpp`, lines 166-193

Pre-allocated pinned host buffers for PCIe transfers:

| Buffer | Size | Purpose |
|--------|------|---------|
| `pinned_seed_host` | 8 bytes | Single uint64 seed |
| `pinned_leaf_cvs` | (M×K/1024)×32 = **8 KiB** | A leaf CVs D2H |
| `pinned_a_slice` | 32×K = **4 KiB** | A row slices D2H (max 32 shares) |
| `pinned_opened_leaves` | 32×1024 = **32 KiB** | A opened leaves D2H |
| `pinned_hash_a` | 32 bytes | A hash D2H |
| `pinned_hash_b` | 32 bytes | B hash D2H |
| `pinned_commit_a` | 32 bytes | Commit A D2H |
| `pinned_commit_b` | 32 bytes | Commit B D2H |

### 1.3 Potential Issues

**Issue 1.1 — C matrix is 4 GiB but never read in pure-miner mode.**
Line 106: `size_t c_bytes = static_cast<size_t>(cfg.m) * cfg.n * sizeof(uint16_t);`
The comment at line 103-105 explicitly says: "In pure-miner mode the kernel accumulates the transcript in registers and never materialises C in HBM." Yet we still allocate 4 GiB per half (8 GiB total for double buffering, 12 GiB for triple buffering).

**Impact:** ~8-12 GiB wasted VRAM. This is the single largest VRAM optimization opportunity.

**Fix:** Skip C allocation in pure-miner mode. Set `c = 0` in the C API params when not needed.

**Issue 1.2 — `sigma_context.cpp` line 157 allocates pinned memory during install.**
```cpp
cudaError_t perr = cudaHostAlloc(&leaf_cvs_pinned, resident_.leaf_cv_bytes(), cudaHostAllocDefault);
```
This is ~256 MiB allocated *during* sigma install (on every new job). While it's freed at line 249, it adds latency to the job switch path.

**Impact:** Adds ~10-50ms to sigma install (pinned allocation is slower than regular allocation).

**Fix:** Pre-allocate a pinned buffer pool in ResidentBState during construction, reuse across installs.

**Issue 1.3 — Host header allocation triggers on batch size change.**
Lines 872-881: When batch size changes, all host headers are freed and re-allocated. This runs inside `bind_sigma_to_half()` which is called during job switch.

**Impact:** Minor — only on batch size changes, which are infrequent.

---

## 2. PCIe Transfer Analysis

### 2.1 Transfer Inventory

#### H2D (CPU → GPU) — Frequent

| Transfer | Size | Frequency | When | Stream |
|----------|------|-----------|------|--------|
| `seed_lo` (8 bytes) | 8 B | Every batch | `queue_batch()` line 1071-1083 | Compute stream (sync) or `seed_copy_stream_` (async) |
| `pow_target` (32 bytes) | 32 B | Per job switch | `upload_pow_target()` line 753-772 | Half's compute stream |
| `job_key` (32 bytes) | 32 B | Per sigma install | `sigma_context.cpp` line 125 | Install stream |

#### D2H (GPU → CPU) — Infrequent but Heavy

| Transfer | Size | Frequency | When | Stream |
|----------|------|-----------|------|--------|
| `host_headers` (batch×header_size) | ~512B×batch | Every batch | `queue_batch()` line 1016-1018 | Implicit in graph replay |
| `a_leaf_cvs` (8 KiB) | 8 KiB | Per share trigger | `process_share_trigger_impl()` line 1292-1294 | Half's compute stream |
| `a_hash` (32 B) | 32 B | Per share trigger | line 1280-1281 | Half's compute stream |
| `commit_a` (32 B) | 32 B | Per share trigger | line 1282-1283 | Half's compute stream |
| `commit_b` (32 B) | 32 B | Per share trigger | line 1284-1285 | Half's compute stream |
| `pinned_a_slice` (up to 4 KiB) | 4 KiB | Per share trigger | line 1305-1308 | Half's compute stream |
| `pinned_opened_leaves` (up to 32 KiB) | 32 KiB | Per share trigger | line 1323-1327 | Half's compute stream |
| `pinned_hash_b` (32 B) | 32 B | Per share trigger | line 1329-1330 | Half's compute stream |
| `leaf_cvs` from ResidentBState (256 MiB) | 256 MiB | Per sigma install | `sigma_context.cpp` line 174-176 | Copy stream |

### 2.2 PCIe Bottleneck Assessment

**H2D transfers during mining:** The seed upload (8 bytes) is negligible. The header copy (batch × ~512 bytes) happens on the host side as part of the graph replay memcpy (line 1016-1018). These are very small and frequent but not bandwidth-limited.

**D2H transfers during share triggers:** These happen very rarely (only when a PoW hit is found). The total per-trigger is ~8+32+32+32+4+32+32 = ~80 KiB. Even at 1 share per minute, this is 80 KiB/min = 1.3 KiB/s — completely negligible on PCIe Gen5 x16 (64 GB/s).

**The big PCIe transfer:** 256 MiB leaf_cvs D2H during sigma install. This happens every time a new job arrives. With async job install, this runs on a background thread. With sync install, it blocks the mining loop.

**Impact assessment:**
- PCIe is **NOT** a throughput bottleneck during normal mining.
- PCIe is a **latency bottleneck** during sigma install (256 MiB D2H).
- PCIe is **irrelevant** for share triggers (rare, small).

### 2.3 Current Optimizations Already in Place

1. **Async seed conveyor** (`PROPMINER_ASYNC_SEED=1`): 8-byte seed upload on `seed_copy_stream_` overlaps with header copy (lines 956-964, 1053-1069).
2. **Pinned memory**: All PCIe staging uses `cudaHostAlloc` for zero-copy DMA (lines 166-193).
3. **Non-blocking streams**: All copy streams use `CU_STREAM_NON_BLOCKING` (lines 256-259).
4. **Async job install** (`PROPMINER_ASYNC_JOB_INSTALL=1`): Background thread does resident B install while mining continues (lines 625-729).

---

## 3. CPU-GPU Synchronization Point Inventory

### 3.1 Sync Points in Mining Loop

#### Synchronous (blocking CPU)

| # | Location | What | Frequency | Latency |
|---|----------|------|-----------|---------|
| 1 | `gpu_worker.cpp:819` | `cuStreamSynchronize(half.stream)` after pow_target upload | Per job switch | ~10-50 μs |
| 2 | `gpu_worker.cpp:917` | `cuStreamSynchronize(half.stream)` before graph capture | Per job switch | ~10-50 μs |
| 3 | `gpu_worker.cpp:1459-1461` | `drain_all_halves()` + `sync_all_compute_streams()` before sigma install | Per job switch (sync path) | **1-5 seconds** |
| 4 | `gpu_worker.cpp:1486-1487` | `drain_all_halves()` + `sync_all_compute_streams()` before async sigma swap | Per job switch (async path) | ~10-100 ms |
| 5 | `gpu_worker.cpp:1512-1513` | `upload_pow_target_all_halves()` + `sync_all_compute_streams()` on target dirty | Per vardiff | ~10-50 μs |
| 6 | `sigma_context.cpp:153` | `cuStreamSynchronize(stream)` after B tensor_hash | Per sigma install | ~10-50 ms |
| 7 | `sigma_context.cpp:241` | `cuStreamSynchronize(stream)` after noise_B | Per sigma install | ~10-50 ms |
| 8 | `sigma_context.cpp:246-247` | `cudaStreamSynchronize(ephemeral_copy)` for leaf_cvs D2H | Per sigma install | ~10-50 ms |
| 9 | `sigma_context.cpp:260-261` | `cuMemcpyDtoH` (synchronous) for b_hash verify | Per sigma install | ~10-50 ms |
| 10 | `sigma_context.cpp:270` | `cuMemcpyDtoH` (synchronous) for commit_b verify | Per sigma install | ~10-50 ms |
| 11 | `gpu_worker.cpp:1240` | `cuStreamSynchronize(half.stream)` drain pre-trigger | Per share trigger | ~10-50 ms |
| 12 | `gpu_worker.cpp:1245` | `cuStreamSynchronize(half.stream)` after A regen | Per share trigger | ~10-50 ms |
| 13 | `gpu_worker.cpp:1331` | `cuStreamSynchronize(half.stream)` after all D2H | Per share trigger | ~10-50 ms |

#### Asynchronous (non-blocking)

| # | Location | What | Frequency |
|---|----------|------|-----------|
| 1 | `gpu_worker.cpp:1047-1049` | `cudaEventRecord(batch_done_event)` | Every batch |
| 2 | `gpu_worker.cpp:943-945` | `cudaEventRecord(batch_start_event)` | Every batch |
| 3 | `gpu_worker.cpp:988-992` | `cudaEventRecord(sub_batch_done_event)` | Every sub-batch |
| 4 | `gpu_worker.cpp:62-63` | `cudaEventQuery(event)` spin-wait | Every batch |
| 5 | `gpu_worker.cpp:1141-1142` | `cudaEventQuery(sub_batch_done_event)` | Every sub-batch |

### 3.2 Critical Path: Job Switch (Sync Install)

When a new job arrives and async install is disabled:

```
CPU: drain_all_halves_for_sigma()   → wait for in-flight batches (ping+pong)
CPU: sync_all_compute_streams()     → wait for all 3 streams
CPU: install_sigma(ctx, ping)       → allocate + install resident B (~100-500ms)
CPU: install_sigma(ctx, pong)       → reuse resident B (~10-50ms, idempotent)
CPU: bind_sigma_to_half(ping)       → upload pow_target + capture graph (~10-50ms)
CPU: bind_sigma_to_half(pong)       → upload pow_target + capture graph (~10-50ms)
```

**Total GPU idle time:** ~100-600ms per job switch.

### 3.3 Critical Path: Job Switch (Async Install — Default)

```
CPU: submit_async_install(new_sigma) → queue to background thread (non-blocking)
CPU: continue mining current sigma   → ZERO idle time!
BG:  ctx->install()                 → resident B install (~100-500ms)
BG:  async_ready_ = ctx             → publish ready
CPU: take_async_ready()             → check every loop iteration
CPU: drain_all_halves_for_sigma()   → wait for in-flight batches (~10-100ms)
CPU: sync_all_compute_streams()     → wait for all streams (~10-50ms)
CPU: bind_sigma_to_half()           → fast swap (~10-50ms per half)
```

**Total GPU idle time:** ~20-200ms per job switch (only the drain+bind phase).

**Assessment:** Async install is already in place and working well. The remaining ~20-200ms of idle time is unavoidable (need to drain in-flight batches before swapping sigma).

---

## 4. Double/Triple Buffering Implementation Plan

### 4.1 Current Implementation

**Double buffering (always active):** Ping/pong ping-pong pattern.
- While one half mines, the other half is being set up for the next batch.
- `wait_until_half_free()` blocks until the previous batch on that half completes.
- Lines 1649-1766 in `run()`.

**Triple buffering (PROPMINER_TRIPLE_BUFFER=1):** Three-half pipeline.
- Mining depth = 2 (queue next, wait for oldest).
- Share rebuild on one half doesn't block the other two.
- Lines 1521-1646 in `run()`.

### 4.2 Assessment

**Double buffering is optimal for single-workload mining.** The GPU is busy computing while the CPU prepares the next batch on the other half. No improvement needed here.

**Triple buffering is already well-implemented** but has a VRAM cost of ~8.6 GiB additional. It's only useful when share triggers are frequent enough to block one half.

### 4.3 Recommended: Quad Buffering (New)

For the target of 700-800+ TMAD/s, the key insight is that **share triggers are the only thing that blocks the GPU**. With triple buffering, one share trigger blocks one half, leaving two for mining. With quad buffering, you get even more headroom.

**However**, given the current ~290 TMAD/s vs target 700-800 TMAD/s gap, the bottleneck is almost certainly **compute-bound** (GPU not getting enough work per batch), not I/O-bound. Adding more buffers won't help if the GPU is already sitting idle waiting for work.

### 4.4 Recommended Changes

1. **Keep double buffering as default** — it's optimal for most cases.
2. **Keep triple buffering as optional** — useful for high-share-frequency scenarios.
3. **DO NOT implement quad buffering** — marginal benefit, high VRAM cost.
4. **Focus on increasing batch size** — this is the primary lever for higher throughput.

---

## 5. Network Communication Optimization

### 5.1 Protocol Stack

PropMiner supports two protocols:

1. **gRPC/HTTP2** (primary): Custom HTTP/2 implementation over TLS
   - `grpc_client.cpp`: Full HTTP/2 frame encoder, HPACK, TLS via OpenSSL
   - Endpoints: `:443` (TLS), path `/pearlpool.mining.v2.MinerService/*`
   - Bidirectional streaming for job delivery

2. **Stratum v1/v2** (fallback): JSON-RPC over TCP
   - `pearl_stratum_client.cpp`: Full Stratum protocol implementation
   - Endpoints: `prl.kryptex.network:7048`, `prl-eu.kryptex.network:7048`
   - Unidirectional: pool pushes jobs, miner pushes shares

### 5.2 Network Transfer Analysis

#### Job Delivery (Pool → Miner)

| Protocol | Direction | Size | Frequency | Latency |
|----------|-----------|------|-----------|---------|
| gRPC | Push | ~256 bytes (JobAssignment) | Every new job (~minutes) | ~50-200ms RTT |
| Stratum | Push | ~512 bytes (mining.notify) | Every new job | ~50-200ms RTT |

#### Share Submission (Miner → Pool)

| Protocol | Direction | Size | Frequency | Latency |
|----------|-----------|------|-----------|---------|
| gRPC | Push | ~2-8 KiB (proof) | Per share found | ~50-200ms RTT |
| Stratum | Push | ~2-8 KiB (base64 proof) | Per share found | ~50-200ms RTT |

#### Heartbeat (Miner → Pool)

| Protocol | Direction | Size | Frequency |
|----------|-----------|------|-----------|
| gRPC | Push | ~256 bytes | Every 30 seconds |

#### Ping (Miner → Pool)

| Protocol | Direction | Size | Frequency |
|----------|-----------|------|-----------|
| gRPC | Push | ~16 bytes | Every 15 seconds |

### 5.3 Bottleneck Assessment

**Network is NOT a bottleneck during mining.** The mining loop is entirely GPU-driven and doesn't touch the network. Network operations only happen during:

1. **Job switches** (every few minutes) — negligible impact
2. **Share submissions** (very rare — maybe 1-10 per hour) — negligible impact
3. **Heartbeats/pings** (periodic, small) — negligible impact

### 5.4 Identified Issues

**Issue 5.1 — Share submission is synchronous per-share.**
`share_sender_thread_func()` (lines 777-886): Shares are built and submitted one at a time. While the build is on the share_sender thread (not the mining thread), the submit itself is a blocking TCP send.

**Impact:** If multiple shares are found simultaneously, they queue in `pending_share_found_` and are sent sequentially. With typical share frequencies (< 1/min), this is negligible.

**Issue 5.2 — Stratum ack timeout is 45 seconds.**
`flush_stale_pending_submits()` (line 640): `kAckTimeout = 45 seconds`. If a share is lost, the miner waits 45 seconds before marking it stale.

**Impact:** Rare — only affects lost shares. No impact on hashrate.

**Issue 5.3 — gRPC keepalive is every 10 seconds.**
`keepalive_thread_func()` (line 195): Sends PING frames every 10 seconds.

**Impact:** Negligible — 8 bytes every 10 seconds.

### 5.5 Recommendations

1. **Batch share submissions**: When multiple shares are found, batch them into a single gRPC call. For Stratum, this is harder (protocol limitation).
2. **Reduce heartbeat frequency**: 30 seconds is fine, but could be 60 seconds to reduce overhead.
3. **No major network optimizations needed** — network is not a bottleneck.

---

## 6. Thread Management Optimization

### 6.1 Thread Layout

```
Main thread (run() in worker_orchestrator.cpp:934):
  - Stats loop (every 5 seconds)
  - Thermal pause monitoring
  - GPU progress monitoring

Per GPU (one GpuWorker thread each):
  - Mining loop (ping/pong or triple-buffered)
  - CUDA graph replay
  - Share scanning

Per GPU side threads (when enabled):
  - share_gpu_loop() — deferred share proof reconstruction
  - async_install_loop() — background sigma install

Orchestrator threads:
  - network_thread_func() — pool connection management
  - share_sender_thread_func() — share proof building and submission
  - heartbeat_thread_func() — periodic heartbeat to pool
  - ping_thread_func() — periodic ping to pool
```

### 6.2 Lock Contention Analysis

| Lock | Protected Data | Contention Level | Notes |
|------|---------------|-----------------|-------|
| `sigma_mtx_` | `sigma_` (current sigma) | **LOW** | Only during job switch (~every few minutes) |
| `share_mtx_` | `share_queue_`, `pending_share_found_` | **LOW** | Only during share triggers (very rare) |
| `async_mtx_` | `async_pending_`, `async_ready_`, `async_failed_` | **LOW** | Every loop iteration check + async thread access |
| `session_mtx_` | `miner_id_`, `session_token_`, `miner_event_seq_` | **LOW** | Only during registration/heartbeat |
| `job_map_mtx_` | Stratum job maps | **LOW** | Only during share submission |
| `challenge_mtx_` | `challenge_resp_ids_` | **NONE** | Only during connection challenge |

### 6.3 Assessment

**Lock contention is negligible.** The only hot-path atomic operations are:
- `stop_flag_`, `pause_flag_`, `batch_abort_requested_` — read every iteration
- `total_iters_`, `hashrate_`, `tmads_per_sec_`, `last_iter_ms_` — written every batch
- `target_nbits_`, `target_dirty_` — written on job switch
- `matmuls_per_poll_`, `graph_batch_` — written on startup

These are all lock-free atomics with no contention between threads (each GPU has its own GpuWorker instance).

### 6.4 Recommendations

1. **No lock contention issues to fix.**
2. **Consider CPU affinity**: Pin the mining thread to a specific CPU core to avoid cache thrashing. The GpuWorker thread runs the hot loop and benefits from L2/L3 cache locality.
3. **Consider NUMA awareness**: If the system has multiple NUMA nodes, ensure the GPU's PCIe endpoint is on the same NUMA node as the mining thread.

---

## 7. OS-Level Tuning Recommendations

### 7.1 GPU Power Management

**Linux:**
```bash
# Set GPU power limit to maximum (RTX 5090: 575W TDP)
nvidia-smi -i 0 -pl 575

# Set GPU performance state to P0 (maximum performance)
nvidia-smi -i 0 -pm 1    # Enable persistence mode
nvidia-smi -i 0 -g 0 -d 0  # Set to P0
```

**Windows:**
- NVIDIA Control Panel → Manage 3D Settings → Power management mode: "Prefer maximum performance"
- Disable Windows GPU scheduling (Settings → System → Display → Graphics settings)
- Disable "Hardware-accelerated GPU scheduling" if it causes context-switch overhead

### 7.2 PCIe Configuration

**Linux:**
```bash
# Verify PCIe generation (should be Gen5 for RTX 5090)
lspci -vvv -s <gpu_pci_address> | grep "LnkSta"

# Set ASPM (Active State Power Management) to disabled
# Add to GRUB_CMDLINE_LINUX: pcie_aspm=off
# OR per-device:
echo "off" > /sys/bus/pci/devices/<gpu_addr>/power/control
```

### 7.3 Kernel/Interrupt Settings

**Linux:**
```bash
# Set CPU governor to performance
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
cpupower frequency-set -g performance

# IRQ affinity — pin GPU interrupts to specific cores
# Find GPU IRQ: grep -i nvidia /proc/interrupts
# Pin: echo <cpu_mask> > /proc/irq/<irq>/smp_affinity
```

### 7.4 Memory Settings

**Linux:**
```bash
# Increase huge pages for better memory allocation performance
# Add to /etc/sysctl.conf:
vm.nr_hugepages = 8192

# Disable transparent huge pages (can cause latency spikes)
echo never > /sys/kernel/mm/transparent_hugepage/enabled
echo never > /sys/kernel/mm/transparent_hugepage/defrag

# Increase file descriptor limits
ulimit -n 65536
```

### 7.5 Network Settings

**Linux:**
```bash
# TCP optimizations for Stratum/gRPC
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
```

---

## 8. I/O Bottlenecks Ranked by Impact

### Tier 1: High Impact (Potential 50-200 TMAD/s improvement)

| # | Bottleneck | Current Impact | Fix |
|---|-----------|---------------|-----|
| 1 | C matrix 4 GiB wasted per half | 8-12 GiB VRAM wasted | Skip C allocation in pure-miner mode |
| 2 | Batch size too small (default=1) | GPU underutilized | Increase batch to 8-32 with proper graph support |
| 3 | CUDA graph overhead per sub-batch | ~1-5ms overhead per launch | Pre-capture larger graphs, reduce launch count |

### Tier 2: Medium Impact (Potential 10-50 TMAD/s improvement)

| # | Bottleneck | Current Impact | Fix |
|---|-----------|---------------|-----|
| 4 | Sigma install sync stall | 100-600ms GPU idle per job | Already async by default; verify it's enabled |
| 5 | Share trigger D2H transfers | ~80 KiB per rare trigger | Already optimized with pinned memory |
| 6 | nvidia-smi popen() every 5s | ~5-20ms CPU overhead per poll | Cache nvidia-smi output, poll every 30s |

### Tier 3: Low Impact (Potential <10 TMAD/s improvement)

| # | Bottleneck | Current Impact | Fix |
|---|-----------|---------------|-----|
| 7 | Network share submission | Negligible (rare) | Batch submissions |
| 8 | Thread lock contention | Negligible | Already lock-free hot path |
| 9 | Thermal pause overhead | ~2s gap per pause | Already implemented; acceptable |
| 10 | OS power management | Variable | Set P0, max power limit |

---

## 9. Specific Code Changes Required

### Change 9.1: Skip C Matrix Allocation in Pure-Miner Mode

**File:** `gpu_worker.cpp`, line 106

```cpp
// BEFORE:
size_t c_bytes = static_cast<size_t>(cfg.m) * cfg.n * sizeof(uint16_t);

// AFTER:
size_t c_bytes = 0;  // Pure-miner mode never reads C; skip 4 GiB allocation
```

Then in `bind_sigma_to_half()` (line 849):
```cpp
// Set C to null pointer when in pure-miner mode
p.C = c_bytes > 0 ? reinterpret_cast<void*>(half.c) : nullptr;
```

**Expected VRAM savings:** ~8 GiB (double buffering) or ~12 GiB (triple buffering).

### Change 9.2: Increase Default Batch Size

**File:** `rtx5090_profile.h`, line 40

```cpp
// BEFORE:
static constexpr int kDefaultMineBatch = 1;

// AFTER:
static constexpr int kDefaultMineBatch = 8;  // Increased from 1
```

And ensure `kDefaultGraphBatch` divides evenly:
```cpp
static constexpr int kDefaultGraphBatch = 8;  // Must divide kDefaultMineBatch
```

**Expected hashrate improvement:** 20-40% (more work per CUDA graph launch, better amortization of launch overhead).

### Change 9.3: Optimize nvidia-smi Polling

**File:** `system_telemetry.cpp`, line 148-203

Replace `popen()` with cached polling:
```cpp
// Add to SystemTelemetry class:
struct SmiCache {
    SystemSnapshot snapshot;
    int64_t timestamp_ms;
};

SmiCache smi_cache_;
static constexpr int64_t SMI_CACHE_TTL_MS = 30000;  // 30 second cache

SystemSnapshot sample(int gpu_index, int interval_ms) {
    // ... existing logic ...
    // If cached and not expired, return cached value
    int64_t now = steady_ms();
    if (cached_gpu_index_ == gpu_index && 
        (now - last_gpu_sample_ms_) < std::max(interval_ms, (int)SMI_CACHE_TTL_MS)) {
        return cached_gpu_;  // Return cached, no popen()
    }
    // ... otherwise re-query ...
}
```

**Expected CPU overhead reduction:** ~15ms per 5-second poll → ~15ms per 30-second poll.

### Change 9.4: Pre-allocate Pinned Leaf CVs Buffer

**File:** `sigma_context.h` — add to ResidentBState:
```cpp
class ResidentBState {
    // ... existing members ...
    uint8_t* pinned_leaf_cvs_host_ = nullptr;  // Pre-allocated staging
};
```

**File:** `sigma_context.cpp` — allocate during `ResidentBState::allocate()`:
```cpp
cudaHostAlloc(&pinned_leaf_cvs_host_, leaf_cv_bytes_, cudaHostAllocDefault);
```

And in `SigmaContext::install()`, reuse instead of allocating:
```cpp
// BEFORE (line 157):
cudaError_t perr = cudaHostAlloc(&leaf_cvs_pinned, resident_.leaf_cv_bytes(), ...);

// AFTER:
uint8_t* leaf_cvs_pinned = resident_.pinned_leaf_cvs_host_;
```

**Expected install latency reduction:** ~10-50ms per job switch.

### Change 9.5: CPU Affinity Pinning

**File:** `gpu_worker.cpp`, `run()` function

Add at the start of the mining thread:
```cpp
void GpuWorker::run() {
#if defined(__linux__)
    // Pin this thread to a specific CPU core for cache locality
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(device_index_ % num_cpus, &mask);  // Round-robin across cores
    pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
#endif
    // ... existing code ...
}
```

**Expected improvement:** 2-5% hashrate increase from better cache locality.

---

## 10. Expected Performance Improvement Summary

| Optimization | Estimated Impact | Difficulty | Risk |
|-------------|-----------------|------------|------|
| Skip C matrix allocation | +0 TMAD/s (VRAM savings only) | Low | Low |
| Increase batch size to 8-32 | **+100-200 TMAD/s** | Medium | Medium (share safety) |
| Optimize nvidia-smi polling | +0 TMAD/s (CPU savings) | Low | Low |
| Pre-allocate pinned buffers | +0-10 TMAD/s (indirect) | Low | Low |
| CPU affinity pinning | +5-15 TMAD/s | Low | Low |
| **Total estimated** | **+105-225 TMAD/s** | | |

**Current:** ~290 TMAD/s
**After optimizations:** ~395-515 TMAD/s

**To reach 700-800+ TMAD/s:** The remaining gap requires **algorithmic/kernel optimizations**, not I/O optimizations:
- Increase M (more work per batch)
- Optimize the CUDA kernel for better tensor core utilization
- Reduce per-batch overhead (graph capture, launch latency)
- Consider multi-GPU scaling (if single-GPU is already compute-bound)

---

## Appendix A: CUDA Streams Inventory

| Stream | Purpose | Type |
|--------|---------|------|
| `ping_.stream` | Mining half A | Default (blocking) |
| `pong_.stream` | Mining half B | Default (blocking) |
| `third_.stream` | Mining half C (triple) | Default (blocking) |
| `merkle_copy_stream_` | Merkle D2H transfers | Non-blocking |
| `seed_copy_stream_` | Async seed upload | Non-blocking |
| `ping_.ping.stream` | Ping stream | Non-blocking |
| `pong_.stream` | Pong stream | Non-blocking |
| `install_stream_` | Async sigma install | Non-blocking |
| `install_copy_stream_` | Async install D2H | Non-blocking |

**Total:** 9 streams per GPU worker (3 with triple buffering).

## Appendix B: Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `PROPMINER_BATCH` | 1 | Matmuls per poll |
| `PROPMINER_GRAPH_BATCH` | 1 | CUDA graph capture depth |
| `PROPMINER_TRIPLE_BUFFER` | 0 | Enable triple buffering |
| `PROPMINER_DEFER_SHARE_GPU` | 1 | Defer share GPU work to side thread |
| `PROPMINER_ASYNC_JOB_INSTALL` | 1 | Async sigma install |
| `PROPMINER_ASYNC_SEED` | 1 | Async seed upload |
| `PROPMINER_N_CAP` | 262144 | Maximum N dimension |
| `PROPMINER_STALL_RESTART_MS` | 30000 | Stall detection timeout |
| `PROPMINER_GPU_TEMP_STOP` | 0 | Thermal pause threshold |
| `PROPMINER_BENCH_NO_GRAPH` | 0 | Disable CUDA graphs for benchmarking |
