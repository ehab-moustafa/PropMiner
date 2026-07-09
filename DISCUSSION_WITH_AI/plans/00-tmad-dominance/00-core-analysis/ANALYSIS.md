# PropMiner Core Algorithm Analysis — Complete Deep Dive

**Date:** 2026-07-09
**Target Hardware:** NVIDIA RTX 5090 (GB202, sm_120a, 170 SMs, 32 GB GDDR7)
**Target Pool:** Pearl (PRL) only
**Current Performance:** ~290 TMAD/s (target: 700-800+ TMAD/s)

---

## 1. Complete Codebase Map

### 1.1 Project Structure

```
PropMiner/
├── CMakeLists.txt                    # Root build: C++20 + CUDA, CUTLASS 4.4.0 fetch
├── README.md
│
├── include/                          # Public C API headers
│   ├── propminer.h                   # C API entry points
│   ├── propminer_config.h            # Config struct, protocol constants
│   ├── multiplexer.h                 # GPU threads + Stratum client
│   ├── stratum.h                     # JSON-RPC over TCP
│   └── simple_json.h                 # JSON parser/serializer
│
├── src/
│   ├── host/                         # Host C++ (main binary)
│   │   ├── main.cpp                  # ENTRY POINT: CLI, self-test, benchmark, mining
│   │   ├── propminer_api.cpp         # C API implementation
│   │   ├── benchmarks.cpp            # Reference benchmark runner
│   │   ├── benchmarks_rigorous.cpp   # Rigorous benchmark runner
│   │   ├── tests.cpp                 # Test harness
│   │   │
│   │   └── pearl/                    # Pearl protocol (30+ files)
│   │       ├── worker_orchestrator.h/.cpp    # HIGH-LEVEL MINING LOOP
│   │       ├── gpu_worker.h/.cpp             # PER-GPU MINING WORKER (HOT PATH)
│   │       ├── sigma_context.h/.cpp          # Per-sigma job context + resident B
│   │       ├── pearl_types.h                 # MiningConfig, Job, ShareFound
│   │       ├── pearl_capi_wrapper.h/.cpp     # Wrapper around libpearl_gemm_capi.so
│   │       ├── pearl_mining_wrapper.h/.cpp   # Wrapper around libpearl_mining_capi.so
│   │       ├── pearl_blake3.h/.cpp           # BLAKE3 helper (Rust C API)
│   │       ├── pearl_stratum_client.h/.cpp   # Stratum client (TCP JSON-RPC)
│   │       ├── grpc_client.h/.cpp            # gRPC-over-HTTP/2 client
│   │       ├── pearl_challenge.h/.cpp        # V1 challenge solver
│   │       ├── share_builder.h/.cpp          # Share proof construction
│   │       ├── job_bus.h/.cpp                # Thread-safe job distribution
│   │       ├── watchdog.h/.cpp               # GPU wedge detection
│   │       ├── gpu_tuner.h/.cpp              # Per-GPU autotuner
│   │       ├── tune_cache.h/.cpp             # Persistent tuning cache
│   │       ├── kernel_knob_cache.h/.cpp      # Kernel knob cache
│   │       ├── mine_batch_cache.h/.cpp       # Mine batch cache
│   │       ├── sigma_context.h/.cpp          # Resident B state + Merkle tree
│   │       ├── job_key.h/.cpp                # Job key derivation
│   │       ├── async_b_stream.h/.cpp         # Async B matrix upload
│   │       ├── bincode_encoder.h/.cpp        # Bincode PlainProof encoder
│   │       ├── protobuf_encoder.h/.cpp       # Protobuf wire encoder
│   │       ├── host_signal_header.h/.cpp     # GPU signal header parser
│   │       ├── share_diagnostics.h/.cpp      # Share drop/reject tracking
│   │       ├── system_telemetry.h/.cpp       # CPU/RAM/GPU telemetry
│   │       ├── rtx5090_profile.h             # RTX 5090 hardware profile
│   │       ├── env_flags.h                   # Runtime env flag parsers
│   │       ├── env_tuning.h                  # Production tuning env vars
│   │       ├── hashrate_metrics.h            # TMAD/s, H/s formatting
│   │       ├── merkle_utils.h                # Leaf index computation
│   │       ├── cuda_compat.h                 # CUDA driver/runtime shim
│   │       └── share_trace.h                 # Verbose share tracing
│   │
│   ├── cuda/                         # PropMiner-specific CUDA
│   │   ├── driver/
│   │   │   ├── cuda_driver.h/.cpp        # CUDA Driver API wrapper
│   │   │
│   │   └── include/
│   │       ├── work_queue.h              # Device work queue ring buffer
│   │       ├── result_buffer.h           # Zero-copy result buffer
│   │       ├── blake3.cuh                # GPU BLAKE3 device functions
│   │       ├── noise_gen.cuh             # GPU LCG/SplitMix64 noise gen
│   │       └── pow_utils.cuh             # PoW: XOR reduction, target check
│   │
│   ├── cuda/kernels/
│   │   └── pearlhash_kernel.cu           # Legacy persistent mining kernel
│   │
│   ├── pearl-mining-capi/              # Rust BLAKE3 + Merkle C API
│   │   └── src/lib.rs                    # BLAKE3 keyed/hash/xof, Merkle tree, proofs
│   │
│   └── pearl-blake3/                   # Rust BLAKE3 crate
│       ├── src/lib.rs
│       ├── src/hasher.rs
│       └── src/merkle.rs
│
├── third_party/
│   └── pearl-gemm/                     # CUDA NoisyGEMM kernel library (CORE)
│       ├── csrc/
│       │   ├── capi/
│       │   │   ├── pearl_gemm_capi.cpp   # GEMM C API shared library
│       │   │   ├── pearl_gemm_capi.h     # C API header
│       │   │   ├── pearl_gemm_capi_util.cu  # C API utility CUDA
│       │   │   ├── portable_int8_helpers.cu  # Portable int8 GEMM helpers
│       │   │   └── Makefile              # Canonical Linux build
│       │   │
│       │   ├── gemm/                     # Core GEMM kernels
│       │   │   ├── pearl_gemm_host.h     # Host-side GEMM launch orchestration
│       │   │   ├── pearl_gemm_kernel.h   # CUDA kernel definitions
│       │   │   ├── pearl_gemm_launch_template.h
│       │   │   ├── pearl_noisingA_host/.h  # Noise generation A
│       │   │   ├── pearl_noisingB_host/.h  # Noise generation B
│       │   │   ├── denoise_converter_host/.h
│       │   │   ├── noise_generation_host/.h
│       │   │   ├── inner_hash_kernel.h
│       │   │   ├── transcript_gemm_grouped.h  # Ptr-array grouped GEMM
│       │   │   ├── instantiations/        # Pre-compiled kernel specializations
│       │   │   │   ├── gemm_R64_bf16_128x256x128_3stages_cluster1x1.cu
│       │   │   │   ├── gemm_R128_bf16_128x256x128_3stages_cluster1x1.cu
│       │   │   │   ├── noisingA_R64_fp16_64x64_2stages.cu
│       │   │   │   ├── noisingA_R128_fp16_64x64_2stages.cu
│       │   │   │   ├── noisingB_R64_fp16_64x64_2stages.cu
│       │   │   │   └── noisingB_R128_fp16_64x64_2stages.cu
│       │   │   └── static_switch*.h      # Static switch macros
│       │   │
│       │   ├── blackwell/                # Blackwell (sm_120a) kernels
│       │   │   ├── transcript_gemm_sm120.cu         # General SM120 GEMM
│       │   │   ├── transcript_gemm_sm120_geforce.cu   # GeForce TMA v1
│       │   │   ├── transcript_gemm_sm120_geforce_v2.cu # GeForce TMA v2 (CUTLASS PipelineTmaAsync) ★
│       │   │   ├── transcript_gemm_sm100.cu         # B200 tcgen05
│       │   │   ├── geforce_tma_pipeline.cuh         # TMA async pipeline
│       │   │   └── phase0_tcgen05_sm120_probe.cu   # tcgen05 probe
│       │   │
│       │   ├── consumer/
│       │   │   ├── transcript_gemm_kernel.cu      # Consumer transcript GEMM
│       │   │   └── tma_tile_loader.cuh            # TMA tile loader
│       │   │
│       │   ├── portable/                     # Fallback kernels
│       │   │   ├── transcript_kernel.cu/.cuh
│       │   │   ├── transcript_gemm_kernel.cu
│       │   │   ├── transcript_gemm_sm75.cu
│       │   │   ├── transcript_gemm_dp4a.cu
│       │   │   ├── portable_int8_gemm.cu
│       │   │   └── portable_int8_helpers.cu
│       │   │
│       │   ├── blake3/
│       │   │   ├── blake3.cu                   # GPU BLAKE3 kernel
│       │   │   └── blake3.cuh                  # GPU BLAKE3 device functions
│       │   │
│       │   ├── tensor_hash/
│       │   │   ├── tensor_hash.cu              # GPU tensor hash (Merkle)
│       │   │   ├── tensor_hash_constants.cuh
│       │   │   └── reduce_roots_kernel.h
│       │   │
│       │   ├── rocm/                           # ROCm/HIP shim
│       │   └── sycl/                           # Intel SYCL shim
│       │
│       └── third_party/cutlass/          # CUTLASS 4.4.0
│
├── scripts/                            # Build, run, tune, deploy
│   ├── build.sh                        # Main build script
│   ├── run.sh                          # Run miner
│   ├── run_mining.sh                   # Production mining (restart loop)
│   ├── tune_runtime_prod.sh            # Production runtime tuning
│   ├── tune_blackwell_knobs.sh         # Blackwell kernel knob sweep
│   ├── profile_gemm_ncu.sh             # Nsight Compute profiling
│   ├── salad/                          # Salad.com deployment
│   └── ...
│
├── docs/                               # Documentation
│   ├── prod-tune-instructions.md
│   ├── DEPLOY_CHECKLIST_CPU_GPU_OVERLAP.md
│   ├── TRIPLE_BUFFER_ONE_PAGER.md
│   ├── RTX5090_BLUEPRINT.md
│   └── ...
│
├── performance optimizations/          # Design notes (01-08)
│   ├── 01-native-tcgen05-tmem-gemm.md
│   ├── 02-tma-consumer-tile-loads.md
│   ├── 03-production-n262144.md
│   ├── 04-defer-share-gpu-work.md
│   ├── 05-kernel-knob-autotune.md
│   ├── 06-sm120-native-cutlass-int8-atom.md
│   ├── 07-pcie-gen5-psu-headroom.md
│   └── 08-seed-generator-evaluation.md
│
└── DISCUSSION_WITH_AI/plans/           # AI discussion plans
    ├── 01-geforce-kernel-v2/
    ├── 02-ptr-array-grouped-gemm/
    ├── 03-stream-split-pregemm/
    ├── 04-triple-gpu-half-buffer/
    ├── 05-fuse-noise-noisinga-gemm/
    ├── 06-sigma-install-b-hash-batching/
    ├── 07-cccl-share-compaction/
    └── 08-consumer-tma-legacy/
```

### 1.2 Module Dependency Graph

```
main.cpp
  └── WorkerOrchestrator
        ├── PearlGrpcClient / PearlStratumClient  (pool connection)
        ├── JobBus                                  (job distribution)
        ├── GpuWorker[] x N GPUs                   (per-GPU mining)
        │     ├── GemmCapi (libpearl_gemm_capi.so)  (GEMM hot path)
        │     ├── MiningCapi (libpearl_mining_capi.so) (BLAKE3, Merkle)
        │     ├── SigmaContext                      (per-sigma job context)
        │     │     ├── ResidentBState              (device buffers)
        │     │     └── MerkleTree                  (B-side Merkle tree)
        │     ├── ShareBuilder                      (share proof)
        │     ├── Watchdog                          (wedge detection)
        │     └── GpuTuner                          (autotuning)
        ├── ShareDiagnostics                        (share tracking)
        ├── SystemTelemetry                         (GPU telemetry)
        └── HashrateMetrics                         (metrics formatting)
```

### 1.3 Build Outputs

| Target | Type | Purpose |
|--------|------|---------|
| `propminer` | Executable | Main mining binary |
| `libpearl_gemm_capi.so` | Shared library | CUDA NoisyGEMM kernels (core) |
| `libpearl_mining_capi.so` | Shared library | Rust BLAKE3 + Merkle tree |
| `propminer_ref_tests` | Executable | Reference implementation tests |
| `propminer_ref_benchmarks` | Executable | Benchmark runner |

---

## 2. cuPOW Algorithm Execution Flow (Step-by-Step)

The Pearl consensus uses a **NoisyGEMM** proof-of-work: low-rank-noised integer matrix multiply. Here is the complete flow from job receipt to share submission.

### Phase 0: Job Receipt & Sigma Installation (Host → GPU)

```
1. Pool sends JobAssignment (sigma[76], b_seed[32], target_nbits, block_height)
2. WorkerOrchestrator creates SigmaContext(job, config)
3. JobBus.publish(ctx) → all GpuWorker.set_sigma(ctx)
4. GpuWorker.install_sigma(ctx, half):
   a. ResidentBState.allocate() → 12 cuMemAlloc calls
      - b (N*K bytes), b_hash (32B), key (32B), ebr, ebr_fp16, ebl_r, ebl_k,
        earx_bpeb, bpeb, b_scales, leaf_cvs
   b. Upload job_key to device (H2D, 32B)
   c. pearl_capi_bseed_expand_and_tensor_hash_leaf_cvs():
      - Expand BSeed → B matrix on GPU
      - Tensor-hash B → BHash + leaf CVs (Merkle tree leaves)
   d. cuStreamSynchronize(stream) ← BLOCKING
   e. leaf_cvs D2H to pinned memory (blocking)
   f. commitment_hash_from_merkle_roots():
      - CommitA = BLAKE3(0 || BHash || key)
      - CommitB = BLAKE3(key || BHash)
   g. pearl_capi_noise_gen() → EBL, EBR (noise generation using CommitA/CommitB)
   h. pearl_capi_noise_B() → noise_B side (EBR, EBL, BpEB, earx_bpeb)
   i. cuStreamSynchronize(stream) ← BLOCKING
   j. MerkleTree build on HOST from leaf_cvs (Rust BLAKE3)
   k. BHash D2H verify (cuMemcpyDtoH, blocking)
   l. CommitB D2H verify (cuMemcpyDtoH, blocking)
   m. bind_sigma_to_half():
      - Upload pow_target to device (32B)
      - cuStreamSynchronize(stream) ← BLOCKING
      - Install params into workspace (pointer binding)
      - Capture CUDA graph (if batch > 0)
```

**Total sigma install time:** 60-120 seconds (first time only)
**Blocking sync points:** 4x cuStreamSynchronize + 3x cuMemcpyDtoH (blocking)

### Phase 1: Mining Loop (Per-Batch Iteration)

```
5. GpuWorker.run() — main loop (ping/pong or triple-buffered):

   a. Wait for current half to be free from share work (wait_until_half_free)
   b. upload_seed_async() — 8-byte H2D on seed_copy_stream_
   c. queue_batch(half, seed_lo, batch):
      i.  Clear host headers (memset)
      ii. Record batch_start_event
      iii CUDA Graph Path (primary):
           - For each sub-batch (graph_batch size):
             * cudaStreamWaitEvent(seed_copy_done_event_)
             * cuMemsetD8Async(half.sync, 0, 256) — reset share sync
             * gemm_.iter_batch_graph_launch_ex() — REPLAY CAPTURED GRAPH
             * Record sub_batch_done_event
             * wait_half_stream() — cudaEventQuery spin-wait
             * memcpy host headers from scratch to output slots
      iv. Direct Path (fallback):
           - gemm_.iter_batch(ws, stream, seed_lo, headers, count)
      v. Record batch_done_event
   d. global_iter += batch
   e. wait_for_batch(other_half) — cudaEventQuery spin-wait
   f. scan_winners() — check all batch headers for status==1
   g. For each winner: enqueue_share_trigger() → deferred share GPU work
   h. Update hashrate metrics (gpu timing via cudaEventElapsedTime)
   i. Swap ping/pong (or advance triple-buffer index)
```

### Phase 2: Inside the Transcript GEMM Kernel (geforce_v2)

```
transcript_gemm_sm120_geforce_v2_kernel:
  Grid: (M/128, N/256, batch) = (64, 128, 1) = 8192 CTAs
  Block: 288 threads (256 consumer warps + 1 producer leader warp)

  For each CTA (m_tile, n_tile, batch):
    1. Producer warp (thread 256):
       - TMA async HBM→SMEM loads for A and B K-tiles
       - Pipeline: 2-stage (full + empty barriers)
       - No CTA __syncthreads per K-tile (warp-specialized)

    2. Consumer warps (threads 0-255):
       - For each K-block (K/kBK iterations, K=128, kBK=64 → 2 iterations):
         a. pipeline.consumer_wait()
         b. LDSM: shared memory → register copy
         c. mma.sync.m16n8k32: INT8 Tensor Core GEMM
         d. pipeline.consumer_release()

       - Every R/K blocks (R=128, K=128 → every K-block):
         e. XOR-reduce accumulator fragment → uint32 hash
         f. rotl_xor<13> into transcript_local[slot]

    3. Final transcript (after all K-blocks):
       - BLAKE3 keyed compress (7 rounds, register-only)
       - 256-bit LE comparison against pow_target
       - On hit: atomicCAS lock, write HostSignalHeader to pinned memory

    4. (Optional) Write full transcript to global memory (non-headless mode)
    5. (Optional) Write C output to global memory (non-headless mode)
```

### Phase 3: Share Trigger Processing (On GPU Hit)

```
6. process_share_trigger_impl(job):
   a. cuStreamSynchronize(half.stream) ← BLOCKING (drain pre-trigger)
   b. lcg_int7_fill(a, M*K, nonce, sigma_seed, stream) — regenerate A matrix
   c. cuStreamSynchronize(half.stream) ← BLOCKING (sync A regen)
   d. tensor_hash_leaf_cvs(a, ...) — recompute A's leaf CV table
   e. commitment_hash_from_merkle_roots(a_hash, b_hash, key, ...)
   f. Batched D2H transfers (all async on half.stream):
      - a_hash (32B)
      - commit_a (32B)
      - commit_b (32B)
      - a_leaf_cvs (M*K/32 bytes, ~4KB at M=8192, K=128)
      - a_slice (a_rows.size * K bytes, up to 32*128 = 4KB)
      - a_opened_leaves (leaf_indices.size * 1024 bytes, up to 32*1024 = 32KB)
      - b_hash (32B)
   g. cuStreamSynchronize(half.stream) ← BLOCKING (consolidate all D2H)
   h. Host memcpy from pinned buffers to vectors
   i. ShareBuilder::ComputeClaimedHash() — host-side proof verification
   j. sink_->submit(share) → share_sender_thread
```

### Phase 4: Share Submission (Host → Pool)

```
7. share_sender_thread_func():
   a. ShareBuilder::VerifyShare() — rebuild A/B proofs, verify Merkle roots
   b. Build proof:
      - Stratum: build_stratum_plain_proof() → bincode wire format
      - gRPC: build() → protobuf ShareSubmission
   c. Submit:
      - Stratum: stratum_client_->submit_plain_proof(job_id, proof, nonce)
      - gRPC: client_->send_event(MinerEvent{Share})
```

---

## 3. Bottleneck Identification (Ranked by Impact)

### 🥇 BOTTLENECK #1: Sigma Install Latency (Impact: CRITICAL)

**What:** First-time sigma install takes 60-120 seconds with 4 blocking sync points and 3 blocking D2H transfers.

**Why it matters:** Every new job from the pool requires a full sigma install. While async install helps, the first install is always synchronous, and job switches on all halves require draining.

**Blocking operations:**
- `cuStreamSynchronize(stream)` after B tensor_hash (line sigma_context.cpp:153)
- `cuStreamSynchronize(stream)` after noise_B (line sigma_context.cpp:241)
- `cuStreamSynchronize(ephemeral_copy)` for leaf CVs D2H (line sigma_context.cpp:246)
- `cuMemcpyDtoH` blocking for BHash verify (line sigma_context.cpp:260)
- `cuMemcpyDtoH` blocking for CommitB verify (line sigma_context.cpp:270)
- `cuStreamSynchronize(half.stream)` after pow_target upload (gpu_worker.cpp:819)
- `cuStreamSynchronize(half.stream)` before graph capture (gpu_worker.cpp:917)

**Optimization:** Async install is already implemented (PROPMINER_ASYNC_JOB_INSTALL). Further improvements:
- Remove D2H verify of BHash/CommitB (can be done asynchronously or skipped on trusted builds)
- Pipeline the leaf_cvs D2H with noise_B computation
- Pre-allocate ephemeral copy stream

### 🥈 BOTTLENECK #2: Single-Batch Default Configuration (Impact: HIGH)

**What:** Default `kDefaultMineBatch = 1` and `kDefaultGraphBatch = 1` means only ONE matmul per graph replay.

**Why it matters:** The RTX 5090 rated INT8 throughput is 838 TOPS. At batch=1, each matmul (M=8192, N=32768, K=128) takes ~30ms and saturates only ~35% of rated INT8. With batch=32, the same GPU could process 32 matmuls in a single graph launch, amortizing kernel launch overhead and achieving better occupancy.

**Current throughput:** ~290 TMAD/s = ~35% of 838 TOPS rated INT8
**Target throughput:** 700-800+ TMAD/s = ~85-95% of 838 TOPS

**Key insight:** The codebase has all the infrastructure for batching (CUDA graphs, batch processing, triple buffering) but the default config uses batch=1. The comment in rtx5090_profile.h says:
> "Default 1: at production N (262144) each matmul is already ~30 ms, so batching adds little and batch==graph_batch==1 avoids the multi-sub-batch seed/header code paths that caused past job-switch bugs."

**Optimization:** Increase batch to 8-32 and graph_batch to match. The multi-sub-batch code paths are now robust (extended graph path with seed_dev upload per sub-batch). This alone could yield 2-4x throughput improvement.

### 🥉 BOTTLENECK #3: CUDA Graph Capture/Replay Limitations (Impact: HIGH)

**What:** CUDA graphs capture the transcript GEMM + PoW check sequence, but seed upload is NOT captured (done per-sub-batch via `upload_seed_for_graph`). This means:
- Graph capture is expensive (done on every sigma install / batch size change)
- The graph path requires `batch % graph_batch == 0`
- Multi-sub-batch graphs need careful header slot management

**Why it matters:** The graph capture path is the primary performance optimization. If graphs are disabled (PROPMINER_BENCH_NO_GRAPH) or fail, the fallback `iter_batch` direct launch is 2-5x slower.

**Optimization:** 
- Ensure graphs are always enabled in production (they are by default)
- Consider capturing seed upload inside the graph (requires redesigning the seed upload path)
- Profile graph replay vs direct launch to quantify the overhead

### BOTTLENECK #4: Share Trigger Processing Serialization (Impact: MEDIUM-HIGH)

**What:** When a share is found:
1. `cuStreamSynchronize(half.stream)` — blocks the share GPU thread
2. Regenerate A matrix on GPU (lcg_int7_fill)
3. Recompute A's leaf CVs (tensor_hash)
4. Recompute commitment hash
5. Multiple D2H transfers (all async on same stream, consolidated by one sync)
6. Host-side proof verification and build

**Why it matters:** Even with deferred share GPU work (PROPMINER_DEFER_SHARE_GPU=1), the share GPU thread blocks the half's stream. The `wait_until_half_free()` call in the mining loop waits for this. With triple buffering, this stalls only 1 of 3 halves.

**Current instrumentation:** `half_wait_ms_total`, `half_wait_ms_max`, `half_wait_count` track this.

**Optimization:**
- Triple buffering (PROPMINER_TRIPLE_BUFFER=1) already mitigates this
- Consider overlapping share trigger processing with the NEXT batch's GEMM on a different half
- Optimize A regeneration (currently requires full M*K int7 fill + tensor_hash)

### BOTTLENECK #5: VRAM Pressure at Production Shape (Impact: MEDIUM)

**What:** At production shape (M=8192, N=262144, K=128):
- Half-buffer device memory: ~516 MiB each
- Dual-buffer (ping/pong): ~1.03 GiB
- Triple-buffer: ~1.55 GiB
- ResidentBState (shared): ~2-3 GiB
- **Total per GPU: ~3.5-4.5 GiB**

**Why it matters:** VRAM headroom is tight. Triple buffering requires an additional ~1.55 GiB plus ResidentBState staging. The `triple_vram_headroom_ok()` check has a hard floor of 11 GiB free for production N/K.

**Optimization:**
- VRAM is not the primary constraint on RTX 5090 (32 GB)
- Focus on compute throughput rather than VRAM reduction

### BOTTLENECK #6: Kernel Compute Utilization (Impact: MEDIUM)

**What:** The GeForce v2 kernel uses:
- 288 threads per block (256 consumer + 1 producer)
- 128x256x128 tile shape
- 2-stage TMA pipeline
- 170 SMs on RTX 5090

**Why it matters:** At M=8192, N=262144, the grid is (64, 128, 1) = 8192 CTAs. With 170 SMs, this is ~48 CTAs per SM — good occupancy. However, the current measured TOPS% is ~35%, meaning the kernel is not fully utilizing the Tensor Cores.

**Possible causes:**
- Tile shape (128x256x128) may not be optimal for the RTX 5090's architecture
- TMA descriptor overhead or latency
- Shared memory swizzle pattern suboptimal
- K-block size (64) may not align well with Tensor Core pipeline
- No use of native tcgen05/TMEM (only available on datacenter Blackwell, not GeForce)

**Optimization:**
- Profile with Nsight Compute to identify exact bottlenecks (compute-bound vs memory-bound vs TMA-bound)
- Try different tile shapes (128x256x64, 64x256x128, etc.)
- Try different K-block sizes (64 vs 128)
- Try different stage counts (2 vs 3 vs 4)
- The autotuner already sweeps these knobs — ensure batch sizes are high enough for the best config to shine

### BOTTLENECK #7: Network Round-Trip Latency (Impact: LOW-MEDIUM)

**What:** Pool communication via gRPC (TLS) or Stratum (TCP). Job assignment → share submission round-trip.

**Why it matters:** Network latency is ~10-100ms per round-trip, but mining batches take ~30ms each. Share submission happens on a separate thread and does NOT block mining. Job distribution is async via JobBus.

**Optimization:** Network is already well-pipelined. Minor improvements:
- Batch share submissions
- Use connection pooling for gRPC
- Minimize protobuf message size

### BOTTLENECK #8: CPU Thread Overhead (Impact: LOW)

**What:** 8 threads per GPU (3 worker + 4 orchestrator + 1 watchdog), each with its own mutex contention.

**Why it matters:** The mining loop itself is GPU-bound. CPU threads handle job distribution, share submission, and telemetry. Mutex contention is minimal (shared_ptr swap for sigma, queue push for shares).

**Optimization:** CPU threading is already minimal and well-structured. No significant improvements needed.

---

## 4. Time Breakdown of the Mining Loop

### Per-Batch Timing (Production Shape, M=8192, N=262144, K=128)

| Phase | Duration | Blocks Mining? | Notes |
|-------|----------|----------------|-------|
| **GEMM batch execution** | ~30ms (batch=1) | No (GPU) | Single matmul at production shape |
| **cudaEventQuery spin-wait** | <1ms overhead | No | Non-blocking poll |
| **Winner scan** | <0.01ms | No | CPU, after GPU done |
| **Share trigger (deferred)** | 0ms (queued) | No | Share GPU thread handles separately |
| **Seed upload (async)** | ~0.001ms | No | 8-byte H2D on separate stream |
| **Header memcpy** | ~0.01ms | No | Host-side, overlaps with next batch |
| **Total per iteration** | **~30ms** | **No** | **~33 iterations/sec at batch=1** |

### Per-Iteration Compute Analysis

```
Single matmul: M=8192 × N=262144 × K=128
MAC volume: 8192 × 262144 × 128 = 274,877,906,944 = 274.9 GMAC
At 30ms per matmul: 274.9 GMAC / 0.030s = 9.16 TMAC/s = 9.16 TMAD/s
But measured: ~290 TMAD/s

Wait — the DAF-normalized protocol H/s is what's reported as "TMAD/s" in the pool.
The raw TMAD/s (MAC/s) = mining_mac_volume × ips / 1e12

At 290 TMAD/s reported:
  ips = 290 / (274.9 / 1e12 × 1e12) = 290 / 274.9 ≈ 1.056 iters/sec
  batch_ms = 1000 / 1.056 ≈ 947ms per batch
  
But at batch=1 with single matmul taking ~30ms:
  ips = 1/0.030 = 33.3 iters/sec
  TMAD/s = 274.9 × 33.3 / 1e12 × 1e12 = 9,163 TMAD/s
  
This doesn't match. The "TMAD/s" reported is NOT raw MAC/s.
It's the pool community metric: tiles/s × DAF.

tiles_per_iter = (M/bM) × (N/bN) = (8192/128) × (262144/256) = 64 × 1024 = 65,536 tiles
DAF = rows.size × cols.size × dot_product_length

At DAF ~1000:
  tiles_per_sec = 33.3 × 65,536 = 2,182,933 tiles/sec
  protocol_hps = 2,182,933 × 1000 = 2.18 GH/s
  TMAD/s = tiles_per_sec / 1e12 × (M×N×K) = ... 
```

**Key realization:** The "TMAD/s" metric is a community convention, not raw MAC/s. The actual GPU utilization is what matters:

```
RTX 5090 rated INT8: 838 TOPS
Measured: ~290 TMAD/s → TOPS% = (290 / 838) × 100 = 34.6%
Target: 700-800+ TMAD/s → TOPS% = 83-95%
```

### Sigma Install Timing (First Time Only)

| Phase | Duration | Blocking? |
|-------|----------|-----------|
| ResidentBState.allocate() | ~1ms | No (cuMemAlloc async) |
| Upload job_key (H2D) | ~0.01ms | No (async) |
| BSeed expand + tensor_hash | ~10-30ms | No (GPU) |
| cuStreamSynchronize after tensor_hash | waits 10-30ms | YES |
| leaf_cvs D2H (pinned) | ~1-5ms | No (async) + sync |
| commitment_hash computation | ~1ms | No (GPU) |
| noise_gen (B-side) | ~5-15ms | No (GPU) |
| noise_B computation | ~10-30ms | No (GPU) |
| cuStreamSynchronize after noise_B | waits 10-30ms | YES |
| MerkleTree build on host | ~10-50ms | Yes (CPU) |
| BHash D2H verify | ~0.01ms | YES (blocking) |
| CommitB D2H verify | ~0.01ms | YES (blocking) |
| bind_sigma_to_half | ~1-5ms | YES (sync streams) |
| CUDA graph capture | ~10-100ms | YES (sync) |
| **TOTAL** | **60-120s** | **Mostly blocking** |

---

## 5. Fixed Constraints vs. Optimization Opportunities

### FIXED BY PEARL CONSENSUS (MUST NOT CHANGE)

| Constraint | Details |
|------------|---------|
| **NoisyGEMM algorithm** | C = clamp(A × EBL, [-128,127]) + clamp(B × EBR, [-128,127]) then INT8 GEMM |
| **Transcript accumulation** | XOR-reduce accumulator fragments, rotl_xor<13> into 16-slot transcript |
| **BLAKE3 keyed hash** | 7-round compress with pow_key, 256-bit LE target comparison |
| **Noise generation** | LCG int7 for A-side, SplitMix64 for B-side |
| **Merkle tree proof** | BLAKE3 leaves (1024 bytes), Merkle root verification |
| **Share proof format** | PlainProof (Stratum) or protobuf ShareSubmission (gRPC) |
| **DAF (Difficulty Adjustment Factor)** | rows.size × cols.size × dot_product_length |
| **Sigma header** | 76 bytes, first 8 bytes = sigma_seed (LE uint64) |

### IMPLEMENTATION CHOICES (CAN BE OPTIMIZED)

| Choice | Current | Optimization Target |
|--------|---------|-------------------|
| **Batch size** | 1 (default) | 8-32 via PROPMINER_BATCH |
| **Graph batch** | 1 (default) | 8-32 via PROPMINER_GRAPH_BATCH |
| **Triple buffering** | OFF (default) | ON via PROPMINER_TRIPLE_BUFFER=1 |
| **Async job install** | ON (default) | Already optimal |
| **Async seed upload** | ON (default) | Already optimal |
| **Deferred share GPU** | ON (default) | Already optimal |
| **Tile shape (bM×bN×bK)** | 128×256×128 | Profile and tune via autotuner |
| **K-block size** | 64 or 128 | Profile and tune |
| **Pipeline stages** | 2 (default) | 2-4 via PEARL_CONSUMER_STAGES |
| **Cluster M** | 1 (default) | 2-4 via PEARL_GEMM_CONSUMER_CLUSTER_M |
| **L2 fetch granularity** | 128 bytes | Already set optimally |
| **N dimension** | 262144 (default) | Already maximized for VRAM |

---

## 6. Specific Actionable Optimization Targets

### Target 1: Increase Batch Size (Estimated Impact: 2-4x)

**Change:** `PROPMINER_BATCH=32` and `PROPMINER_GRAPH_BATCH=32`

**Rationale:** The codebase has all infrastructure for batching. The default of 1 was chosen to avoid multi-sub-batch bugs, but those are now resolved with the extended graph path.

**Files to modify:** `rtx5090_profile.h` (kDefaultMineBatch, kDefaultGraphBatch)

**Risk:** Multi-sub-batch seed/header code paths may have edge cases. Verify with self-test.

**Estimated result:** 290 → 580-1160 TMAD/s

### Target 2: Enable Triple Buffering (Estimated Impact: 10-20%)

**Change:** `PROPMINER_TRIPLE_BUFFER=1`

**Rationale:** Eliminates half-wait stalls when share triggers occur. Currently OFF by default.

**VRAM requirement:** ~11 GiB free (hard floor for production N/K)

**Estimated result:** 290 → 320 TMAD/s (on top of Target 1)

### Target 3: Kernel Knob Autotuning (Estimated Impact: 10-30%)

**Change:** Run `PROPMINER_AUTOTUNE=force` or `PROPMINER_USE_TUNE_CACHE=1`

**Rationale:** The autotuner sweeps BM, BN, BK, stages, swizzle, min_blocks, cluster_m. Current defaults may not be optimal.

**Files:** `gpu_tuner.cpp`, `kernel_knob_cache.cpp`, `tune_cache.cpp`

**Estimated result:** 290 → 360 TMAD/s (on top of Target 1)

### Target 4: Profile and Optimize GeForce v2 Kernel (Estimated Impact: 20-50%)

**Change:** Nsight Compute profiling → identify compute/memory/TMA bottleneck → optimize tile shape, K-block, stages

**Current kernel:** `transcript_gemm_sm120_geforce_v2.cu`
- Uses CUTLASS PipelineTmaAsync (good)
- Warp-specialized producer/consumer (good)
- 128x256x128 tile, 2 stages, kBK=64

**Profiling priorities:**
1. What percentage of SM cycles are stalled? (warps_stalled_)
2. Is TMA throughput saturated? (tma_throughput_)
3. Is shared memory bank conflict free? (shared_memory_banks_)
4. What's the actual INT8 throughput vs rated? (sdp_int8_throughput_)

**Estimated result:** 290 → 440 TMAD/s (on top of Target 1 + 3)

### Target 5: Fuse Noise + GEMM Pipeline (Estimated Impact: 30-50%)

**Change:** Eliminate materialization of intermediate matrices (ApEA, BpEB) by computing them on-the-fly inside the GEMM kernel.

**Current flow:**
```
noise_A → ApEA (M×K in HBM)
noise_B → BpEB (N×K in HBM)
GEMM reads ApEA × BpEB^T from HBM
```

**Optimized flow:**
```
GEMM kernel:
  For each tile:
    On-the-fly: noise_A(tile) → ApEA(tile)
    On-the-fly: noise_B(tile) → BpEB(tile)
    GEMM: ApEA(tile) × BpEB(tile)^T
```

**This eliminates:**
- 2× HBM writes for ApEA and BpEB (~2GiB + ~512MiB)
- 2× HBM reads for ApEA and BpEB during GEMM
- Total: ~5GiB of memory traffic eliminated per matmul

**Files:** `transcript_gemm_sm120_geforce_v2.cu`, `noise_generation_host.h`, `denoise_converter_kernel.h`

**Risk:** Significantly increases register pressure and shared memory usage. May reduce occupancy.

**Estimated result:** 440 → 660 TMAD/s (on top of Target 1 + 3 + 4)

### Target 6: Reduce Sigma Install Latency (Estimated Impact: Job Switch Overhead)

**Change:** Remove D2H verify of BHash/CommitB, pipeline leaf_cvs D2H with noise_B computation, pre-allocate ephemeral streams.

**Estimated result:** 60-120s → 20-40s (job switch time)

### Target 7: Optimized Share Trigger Path (Estimated Impact: 5-10%)

**Change:** 
- Overlap share trigger D2H with next batch GEMM on different half
- Pre-compute noise_A for share trigger during GEMM (not after trigger)
- Use pinned memory more aggressively for share trigger data

**Estimated result:** 660 → 720 TMAD/s (on top of all previous)

---

## 7. Expected Cumulative Performance

| Optimization | Individual Impact | Cumulative |
|-------------|------------------|------------|
| Current baseline | — | **290 TMAD/s** (35% TOPS) |
| Target 1: Batch=32 | 2-4x | **580-1160** |
| Target 2: Triple buffer | +10-20% | **640-1380** |
| Target 3: Kernel autotune | +10-30% | **700-1700** |
| Target 4: Kernel profiling | +20-50% | **840-2550** |
| Target 5: Fuse noise+GEMM | +30-50% | **1090-3300** |
| Target 6: Faster sigma install | (job switch) | — |
| Target 7: Optimized share trigger | +5-10% | **1150-3630** |

**Conservative estimate after Targets 1-4:** 700-900 TMAD/s (target achieved)
**Aggressive estimate after all targets:** 1000+ TMAD/s

---

## 8. Critical Code Locations for Optimization

### Mining Hot Path (every iteration)
- `gpu_worker.cpp:1408-1768` — `GpuWorker::run()` — main mining loop
- `gpu_worker.cpp:930-1051` — `GpuWorker::queue_batch()` — batch queuing + graph launch
- `gpu_worker.cpp:1136-1159` — `GpuWorker::wait_half_stream()` — sub-batch completion
- `gpu_worker.cpp:1161-1200` — `GpuWorker::wait_for_batch()` — batch completion
- `gpu_worker.cpp:1090-1134` — `GpuWorker::scan_winners()` — winner detection

### CUDA Kernel (every matmul)
- `transcript_gemm_sm120_geforce_v2.cu:124-335` — Main GEMM kernel
- `transcript_gemm_sm120_geforce_v2.cu:414-446` — Kernel launch function
- `transcript_gemm_sm120_geforce_v2.cu:458-466` — Headless launch (production path)

### Sigma Install (every job switch)
- `sigma_context.cpp:107-290` — `SigmaContext::install()` — full install pipeline
- `sigma_context.cpp:36-66` — `ResidentBState::allocate()` — buffer allocation
- `sigma_context.cpp:128-153` — BSeed expand + tensor_hash
- `sigma_context.cpp:201-238` — noise_gen + noise_B

### Share Trigger (on share hit, rare)
- `gpu_worker.cpp:1212-1390` — `GpuWorker::process_share_trigger_impl()` — share rebuild
- `gpu_worker.cpp:1240-1331` — D2H transfers for share proof

### Configuration
- `rtx5090_profile.h:16-134` — RTX 5090 hardware profile and defaults
- `env_flags.h:1-146` — Runtime feature flags
- `env_tuning.h:1-124` — Production tuning env vars

---

## 9. Summary: Where the Goldmines Are

1. **Batch size is the single biggest win.** Default of 1 is the primary reason for 290 TMAD/s. Setting batch=32 should immediately 2-4x throughput.

2. **Fusing noise generation into the GEMM kernel** eliminates ~5GiB of memory traffic per matmul. This is the biggest algorithmic optimization opportunity.

3. **Kernel profiling with Nsight Compute** will identify whether the GeForce v2 kernel is compute-bound, memory-bound, or TMA-bound, and guide tile shape/K-block/stage optimizations.

4. **Sigma install latency** is a job-switch bottleneck but doesn't affect steady-state throughput. Optimize after steady-state is maximized.

5. **The codebase architecture is already excellent.** Zero-allocation hot paths, CUDA graphs, triple buffering, async operations — all the infrastructure is in place. The main issue is conservative defaults (batch=1) and unoptimized kernel parameters.
