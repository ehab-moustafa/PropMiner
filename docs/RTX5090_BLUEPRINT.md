# RTX 5090 PropMiner Optimization Blueprint

## 1. Hardware reality check

* GPU: NVIDIA GeForce RTX 5090 (GB202)
* Compute Capability: **12.0 (`sm_120`)** — not 10.0.
* SMs: **170** (full GB202 is 192; RTX 5090 has 170 enabled).
* Tensor Cores: 5th-gen (`tcgen05`).
* Memory: 32 GB GDDR7, 512-bit bus.
* PCIe: Gen 5.

Production `blackwell` builds target `sm_120a` with **GeForce v2** as the default kernel (`PEARL_GEMM_KERNEL=geforce_v2`): warp-specialized TMA producer + SM80 `mma.sync m16n8k32` int8 IMMA consumers. Proof-canonical int8 transcript is unchanged.

**GeForce Blackwell constraints (GB202):**
* Per-block shared memory opt-in max: **99 KiB** (not 164 KiB datacenter).
* **No TMEM / tcgen05** on consumer RTX — `tcgen05.mma` ptxas-fails on `sm_120a` (see `phase0_tcgen05_sm120_probe.cu`).
* Do not deploy `sm_100a` cubins on RTX 5090.

## 2. Kernel instruction strategy

| Path | Atom / load | Status |
|------|-------------|--------|
| **GeForce v2 (default)** | TMA + `SM80_16x8x32_S32S8S8S32_TN` (`mma.sync`) | Production on `sm_120a`; validate via `scripts/verify_geforce_transcript.sh`. |
| Consumer fallback | `cp.async` + same SM80 int8 atom | `PEARL_GEMM_KERNEL=consumer`. |
| B200 datacenter | `tcgen05.mma` + TMEM | `sm_100a` only — **invalid on RTX 5090**. |

Further gains are from TMA pipeline depth, occupancy knobs, and grouped GEMM — not from switching to tcgen05 or FP4 dtypes (breaks proof).

## 3. Memory & streaming architecture

```
Host pinned buffer B0  ----stream 0-->  GPU VRAM B0  --> kernel on stream 0
Host pinned buffer B1  ----stream 1-->  GPU VRAM B1  --> kernel on stream 1
```

* Generate `noise_B` / Merkle tree on the host asynchronously using a side stream (already partially done in `sigma_context.cpp`).
* Upload the next `B` matrix chunk with `cudaMemcpyAsync` while the current batch runs.
* Keep `B` resident in VRAM across iterations (resident B-state in `SigmaContext`).
* Capture the upload+kernel+reduce pattern in a CUDA Graph to remove launch overhead.
* For RTX 5090, the GDDR7 bandwidth is very high; the kernel should be compute-bound. If profiling shows memory-bound behavior, increase tile size / occupancy, not memory clock.

## 4. Watchdog / auto-recovery

A dedicated watchdog thread monitors a heartbeat event from each `GpuWorker`:

```cpp
// pseudo-code
while (running) {
    if (cudaEventQuery(worker_event) == cudaErrorLaunchTimeout) {
        cuCtxDestroy(ctx);
        recreate_context(device);
        worker.resume_from(last_nonce);
    }
    std::this_thread::sleep_for(5s);
}
```

* Use `cudaEventQuery` on the worker stream every few seconds.
* On `CUDA_ERROR_LAUNCH_TIMEOUT` or `CUDA_ERROR_UNKNOWN`, destroy and recreate the CUDA context.
* Persist `last_nonce` per GPU to avoid duplicate work after restart.
* Log the reset and continue; never crash the binary.

## 5. Clock policy for NoisyGEMM

| Setting | Recommendation | Why |
|---------|----------------|-----|
| Core clock | +150 to +200 MHz max | Tensor Core-heavy workloads scale with core. +250 risks silent errors. |
| Memory clock | Stock or mild OC (+200) | NoisyGEMM streams B and noise through cache; downclocking memory hurts throughput. |
| Power limit | 95-100% | Lowering power limit is safer than core offset for thermal control. |
| VRAM min clock | **Do not lock to minimum** | Will create memory-side stalls despite compute-bound appearance. |

Validate every overclock by running the built-in share-verification self-test; a silently corrupted matrix tile will produce invalid proofs.

## 6. Hardcoded RTX 5090 occupancy config

Target: occupy all 170 SMs with 1-2 CTAs each.

Recommended production matrix shape for RTX 5090:
* `M = 8192`, `N = 262144` (VRAM pick via `pick_n_for_vram`), `K = 128` (local) or `K = 4096` (stratum pool)
* Tile `BM=128, BN=256, BK=64` (CMake `KBLOCK=64`, `STAGES=2`)
* Grid tiles = `(8192/128) * (262144/256) = 64 * 1024 = 65536` CTAs
* Default `PROPMINER_BATCH=1`, `PROPMINER_GRAPH_BATCH=1` (stability); grouped GEMM activates at `batch >= 4`
* Streams: ping/pong (+ optional third half-buffer) + `seed_copy_stream_`

VRAM at N=262144 is ~22–28 GiB with resident B; triple-buffer needs headroom check.

## 7. Implementation checklist

- [x] GeForce v2 kernel default for blackwell (`transcript_gemm_sm120_geforce_v2.cu`, TMA + mma.sync).
- [x] Consumer / sm120 fallback (`transcript_gemm_sm120.cu`, `PEARL_GEMM_KERNEL=consumer`).
- [x] Headless two-kernel path (GEMM transcript → `transcript_finalize_kernel`).
- [x] Triple half-buffer (`PROPMINER_TRIPLE_BUFFER`, default ON when VRAM allows).
- [x] Add pinned double-buffer B upload (`AsyncBStream`).
- [x] Add context-reset watchdog thread (`Watchdog`).
- [x] Add hardcoded `Rtx5090Profile` constants (170 SMs, 128x256x128 tile, M=8192,N=32768).
- [x] Add `--rtx5090` / `--benchmark` CLI mode that runs the tuned config.
- [x] CPU-isolated nonce upload via pinned `cudaMemcpyAsync` ping-pong on `seed_copy_stream_` (monotonic counter; no background seed thread).
- [x] Hot-path batch wait: `cudaEventQuery` spin wheel (no blocking sync on steady-state loop).
- [x] Hashrate telemetry from `cudaEventElapsedTime` (GPU batch ms, not wall clock).
- [ ] Validate with ncu profiling on target hardware.
