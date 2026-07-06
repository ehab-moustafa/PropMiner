# RTX 5090 PropMiner Optimization Blueprint

## 1. Hardware reality check

* GPU: NVIDIA GeForce RTX 5090 (GB202)
* Compute Capability: **12.0 (`sm_120`)** — not 10.0.
* SMs: **170** (full GB202 is 192; RTX 5090 has 170 enabled).
* Tensor Cores: 5th-gen (`tcgen05`).
* Memory: 32 GB GDDR7, 512-bit bus.
* PCIe: Gen 5.

Existing `pearl-gemm` already builds a `blackwell` target for `sm_120a`, but the consumer lane still uses the SM80 `mma.sync m16n8k32` int8 atom. That atom runs correctly on `sm_120`, yet it does not use the 5th-gen Tensor Core ISA. The biggest potential gain is replacing it with a native Blackwell MMA atom (`tcgen05.mma`) or a CUTLASS 4 Blackwell GEMM.

## 2. Kernel instruction strategy

| ISA | Atom | Status |
|-----|------|--------|
| SM80 `mma.sync` | `SM80_16x8x32_S32S8S8S32_TN` | Current consumer kernel; works on `sm_120`, not optimal. |
| SM100 `tcgen05` | `tcgen05.mma` cluster fragments | Used in B200 datacenter path (`sm_100a`), not for consumer RTX. |
| Native `sm_120` | CUTLASS Blackwell MMA / inline PTX | **Desired**; needs implementation and hardware testing. |

Because `sm_120` consumer ISA documentation is limited, the pragmatic path is:
1. Compile for `sm_120a` with the existing SM80 atom (already done).
2. Add a `PEARL_GEMM_RTX5090` build profile that compiles an experimental `tcgen05` kernel guarded behind runtime dispatch.
3. Benchmark both and keep the winner.

Do not attempt to use `sm_100a` binaries on an RTX 5090 — they will not run.

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

Recommended default matrix shape for RTX 5090:
* `M = 8192`, `N = 32768`, `K = 128`
* Consumer tile `BM=128, BN=256, BK=128`
* Grid tiles = `(8192/128) * (32768/256) = 64 * 128 = 8192`
* Total CTAs >> 170 * 2, ensuring persistent occupancy.
* Batch size: 16-24 matmuls per poll.
* Streams: 2 ping/pong streams per GPU.

This config fits in ~12-14 GB of VRAM, leaving headroom for noise buffers and Merkle trees.

## 7. Implementation checklist

- [x] Add `PEARL_GEMM_RTX5090` build profile (`sm_120` + experimental `tcgen05` kernel scaffold).
- [x] Implement native `sm_120` Blackwell MMA kernel behind compile-time dispatch (`transcript_gemm_sm120.cu`).
- [x] Add pinned double-buffer B upload (`AsyncBStream`).
- [x] Add context-reset watchdog thread (`Watchdog`).
- [x] Add hardcoded `Rtx5090Profile` constants (170 SMs, 128x256x128 tile, M=8192,N=32768).
- [x] Add `--rtx5090` / `--benchmark` CLI mode that runs the tuned config.
- [x] CPU-isolated nonce upload via pinned `cudaMemcpyAsync` ping-pong on `seed_copy_stream_` (monotonic counter; no background seed thread).
- [x] Hot-path batch wait: `cudaEventQuery` spin wheel (no blocking sync on steady-state loop).
- [x] Hashrate telemetry from `cudaEventElapsedTime` (GPU batch ms, not wall clock).
- [ ] Validate with ncu profiling on target hardware.
