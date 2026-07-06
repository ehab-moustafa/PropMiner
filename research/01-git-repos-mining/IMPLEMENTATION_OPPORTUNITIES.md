# Cross-Repo Implementation Opportunities — RTX 5090 Pearl Mining

**Date:** 2026-07-06  
**Scope:** 8 cloned repos (excluding `alpha-miner`, `akoya-miner`) vs PropMiner  
**Focus:** Pearl protocol, CUDA, RTX 5090 (`sm_120a`) — full GPU saturation, triple-generation throughput  
**Sources:** `REPORT.md` + targeted subagent analysis (no file-by-file mining)

---

## Executive summary

PropMiner is already **ahead of upstream pearl** and **ethminer** on the CUDA host loop (ping-pong streams, CUDA graphs, resident B, wave-aligned N=262144). The largest remaining hashrate gaps are **inside PropMiner's own pearl-gemm fork**, informed by these clones:

| Tier | Opportunity | Source repo(s) | Est. impact |
|------|-------------|----------------|-------------|
| **P0** | Ship **warp-specialized GeForce kernel** (`transcript_gemm_sm120_geforce.cu`) as production default after transcript gate | blackwell-geforce-nvfp4-gemm, cutlass | +10–25% |
| **P0** | Fix **99 KB SMEM/block** budget in autotune + `rtx5090_profile.h` (not 164 KiB) | blackwell-geforce-nvfp4-gemm | Unblocks valid knob sweeps |
| **P1** | Adopt **CUTLASS SM120 TMA warp-specialized** mainloop patterns (examples 79/87) around unchanged SM80 int8 atom | cutlass | +10–25% cumulative |
| **P1** | **Triple compute buffer** when share-deferral stalls ping-pong | ethminer (concept), pearl (gap) | Medium at production diff |
| **P1** | **Ptr-array grouped GEMM** — multiple nonce generations per persistent launch | cutlass | +10–20% when batch-bound |
| **P2** | **Entrypoint GPU clock lock** (`nvidia-smi -pm 1`, `-lgc`) + scrapeable metrics | pearl-miner-docker | +5–15% ops / fleet |
| **P2** | σ-install **B tensor_hash** batching (warp-lane parent compress) | BLAKE3 | Startup + marginal steady |
| **P3** | CCCL `cuda::atomic_ref` + `DeviceSelect` for share compaction | cccl | <2% steady-state |
| **Research** | Triton offline autotune for sub-tile knobs only | triton | Calibration, not production |

**Honest ceiling:** True **3× generations/sec** (~300 → ~900 TMAD/s) exceeds sustained RTX 5090 INT8 peak with Pearl epilogue. Realistic target with these imports: **~500–600 TMAD/s** (1.7–2×). Triple-generation in the **pipeline sense** (3 nonce batches in flight) is achievable today with seed stream + optional third half-buffer.

---

## What “triple generations” means here

| Meaning | Status in PropMiner |
|---------|---------------------|
| **3× TMAD/s throughput** | Requires P0+P1 kernel work; not achievable via ops alone |
| **3 streams in flight** | Already: ping + pong + `seed_copy_stream_` |
| **3 compute buffers** | **Missing** — dual ping-pong only; add third half when `PROPMINER_DEFER_SHARE_GPU=1` stalls |
| **3 kernel phases overlapped** | **Partial** — pre-GEMM chain (hash/noise) still serializes on graph stream |

---

## Priority implementation roadmap

### Phase 1 — Correct SM120 doctrine (days)

1. Update `rtx5090_profile.h` comments: **99 KB SMEM/block**, not 164 KiB; accumulators in registers, not tcgen05 layouts
2. Fix `tune_kernel_knobs_common.sh`: `tune_knob_smem_ok` ≤ ~101376 B
3. Retire tcgen05-as-desired in `docs/RTX5090_BLUEPRINT.md`; point to GeForce warp-specialized path
4. Reconcile `kProdDefaultClusterM=2` vs kernel default `cluster_m=1` — empirical sweep on 5090

### Phase 2 — Kernel generation rate (weeks)

1. Gate + enable `PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1` via `verify_geforce_transcript.sh`
2. Port CUTLASS `sm120_mma_tma.hpp` / example 79 patterns into geforce kernel
3. Explore `KBLOCK=64, STAGES=3/4` within 99 KB (valid; `KBLOCK=128, STAGES=3` is **invalid** on SM120)
4. Run `cutlass_profiler` sweeps for M=8192, N=262144, K=128 — feed winners into CMake knobs
5. Evaluate ptr-array grouped GEMM for batch>1 nonce launches

### Phase 3 — Pipeline depth (weeks)

1. Optional **triple `HalfBuffers`** if profiling shows `wait_until_half_free` stalls
2. Stream-split pre-GEMM (A regen + tensor_hash + noise on secondary stream with event fencing)
3. `cuda::atomic_ref` share signaling under defer-share path

### Phase 4 — Fleet ops (days)

1. Container entrypoint: `nvidia-smi -pm 1` + 5090-specific `-lgc` range
2. Prometheus-style metrics endpoint (TMAD/s, batch_ms, GPU index)
3. `vast-template.yaml` for marketplace deploy

---

## Per-repo findings

### 1. cutlass

**What it is:** NVIDIA template library for high-performance GEMM and tensor-core kernels.

#### Already in PropMiner
- CUTLASS v4.4.0 vendored in `third_party/pearl-gemm/third_party/cutlass`
- CuTe atoms: `SM80_16x8x32_S32S8S8S32_TN`, swizzle, `cp.async`, `ldmatrix`
- `sm_120a` cubin with proof-canonical SM80 int8 IMMA
- Experimental TMA loader + GeForce warp-specialized scaffold (compile-off by default)
- Host batching, CUDA graphs, cluster_m tuning

#### Special — implement in PropMiner

| Feature | Why | Impact | Effort | CUTLASS path |
|---------|-----|--------|--------|--------------|
| **SM120 TMA warp-specialized persistent mainloop** | Separates TMA producers from MMA consumers; NVIDIA's GeForce Blackwell pattern | High | High | `include/cutlass/gemm/collective/sm120_mma_tma.hpp`, `kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp` |
| **GeForce examples 79/87** | Working SM120 persistent GEMM blueprints | High | Med | `examples/79_blackwell_geforce_gemm/`, `examples/87_blackwell_geforce_gemm_blockwise/` |
| **Pingpong vs cooperative schedule** | Overlap heavy transcript epilogue with next-tile MMA | Med–High | Med | `media/docs/cpp/blackwell_functionality.md` |
| **Ptr-array grouped GEMM** | Multiple nonce generations per kernel launch | Med–High | High | `test/unit/gemm/device/sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op_group_gemm.cu` |
| **CUTLASS profiler sweeps** | Offline CTA/stage/cluster discovery | Med | Low | `tools/profiler/` |
| **Asymmetric TMA DMA** | Matches resident-B workload (B reused, A regenerated) | Med | High | `sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp` |

#### Do NOT port from CUTLASS
- SM120 native int8 MMA atom — `static_assert` "No MMA matches SM120_16x8x32_TN"; zero gain
- FP8/FP4/block-scaled MMA — breaks proof-canonical transcript
- tcgen05 / SM100 collectives — datacenter-only; no TMEM on GeForce

#### Triple-generation angle
Grouped GEMM fuses multiple generations per persistent kernel, complementing CUDA-graph batch. Pingpong schedule attacks epilogue Amdahl tail (~10–20% of iter).

---

### 2. ethminer

**What it is:** Classic CUDA stratum GPU miner (Ethash) — host scheduling reference only.

#### Already in PropMiner
- Per-GPU worker threads, multi-stream overlap (ping/pong + seed + merkle streams)
- CUDA graphs (strictly better than ethminer's per-stream `<<<>>>` launches)
- Pinned host buffers, hashrate telemetry, multi-GPU nonce partitioning
- `cudaEventQuery` spin-wait (ethminer uses blocking `cudaStreamSynchronize`)
- `Rtx5090Profile` wave alignment, `GpuTuner`, L2 fetch granularity

#### Special — implement in PropMiner

| Feature | Why | Impact | Effort |
|---------|-----|--------|--------|
| **Third compute half (triple-buffer)** | `wait_until_half_free` blocks when share proof holds a half under `PROPMINER_DEFER_SHARE_GPU=1` | Medium (share-heavy diff) | Medium |
| **`cudaDeviceScheduleSpin` at init** | PropMiner never calls `cudaSetDeviceFlags` | Low | Low |
| **Per-kernel cache config** | Try `cudaFuncCachePreferShared` for smem-heavy GEMM (not ethminer's L1 preference) | Low–uncertain | Low |
| **Sequential multi-GPU σ install** | Avoid VRAM/PCIe spikes at σ rotation on 8× rigs | Low sustained / Med stability | Low |
| **Mid-batch σ kick** | Discard stale in-flight work on new pool job | Low–Med | Medium |

#### Do NOT port
- Ethash DAG/keccak kernels, `__constant__` DAG, `--cuda-streams N` parallel full-GEMM (counterproductive at 65k CTAs)

#### Triple-generation angle
PropMiner's ping+pong+seed conveyor already beats ethminer's 2-stream model. The gap is a **third compute buffer**, not more parallel GEMM streams.

---

### 3. pearl (canonical monorepo)

**What it is:** Official Pearl L1 monorepo — authoritative NoisyGEMM semantics; CI targets sm90 only.

#### Already in PropMiner
- Full pearl-gemm fork with **5090-specific** sm120 kernels, CAPI graphs, bseed sm80 tensor_hash
- Resident B at σ-install, gRPC pool host, `rtx5090_profile.h`
- Ahead of upstream on sm80 tensor_hash dispatch for CC 12.0

#### Special — implement in PropMiner

| Feature | Upstream has | Impact |
|---------|--------------|--------|
| **Fused noise + noisingA + GEMM** | Separate kernels (same as upstream) | +1–3% |
| **TMA consumer tile loads** | sm100 path only | +10–25% (doc 02) |
| **Triple ping-pong buffer** | Not in upstream either | Pipeline depth |
| **Stream-split pre-GEMM** | Not in upstream | Overlap hash/noise vs GEMM |
| **py-pearl-mining test vectors** | GPU CI regression harness | Correctness gate |

#### Do NOT port from upstream pearl
- sm90 Hopper WGMMA GEMM, TMA dual-pipeline tensor_hash (wrong for 5090)
- `denoise_converter.cu`, MoE routing, hadamard quant, vllm-miner — not pool mining
- `pearl-gateway` solo RPC — PropMiner uses Akoya gRPC

#### Triple-generation angle
Upstream offers **semantic reference only** for sm_120. Five stages still serialize per graph iter: lcg → tensor_hash → commitment → noise_gen → noisy_gemm. GEMM is ~85–92%; stream-split pre-GEMM is the overlap opportunity.

---

### 4. cccl

**What it is:** NVIDIA CUDA Core Compute Libraries (CUB, Thrust, libcudacxx).

#### Already in PropMiner
- `thrust::device_vector` in noising host headers only
- Hand-rolled XOR transcript reduction, raw `atomicAdd`/`atomicCAS` for shares
- CUTLASS warpgroup primitives in GEMM — not CUB

#### Special — implement in PropMiner

| Primitive | Use case | Impact |
|-----------|----------|--------|
| **`cuda::atomic_ref` + barriers** | Replace spinlock in `write_host_signal_header()` under defer-share | Low latency, correctness |
| **`cub::DeviceSelect::Flagged`** | Compact per-launch PoW hits without atomic storms | Medium if batching grows |
| **`cub::DeviceScan::InclusiveSum`** | Prefix-sum over candidate tiles | Medium (multi-candidate launches) |
| **`thrust::equal` / `reduce`** | GPU merkle vs CPU golden in CI | Dev only |

#### Do NOT put CUB inside GEMM epilogue
Register/SMEM pressure fights `MIN_BLOCKS=2` occupancy sweep. Keep CCCL in **post-MMA PoW sidecar** only.

#### Triple-generation angle
CCCL cannot close the 2–3× TMAD gap. Expect <2% steady-state unless combined with multi-candidate PoW batching.

---

### 5. blackwell-geforce-nvfp4-gemm

**What it is:** SM120 GeForce architecture doctrine — critical hardware constraints for RTX 5090.

#### Already in PropMiner
- No tcgen05/TMEM on GeForce (probe confirms)
- Correct SM80 `mma.sync` int8 atom on `sm_120a`
- Tile 128×256×128 at 2 stages = 96 KiB (fits 99 KB ceiling)
- `transcript_gemm_sm120_geforce.cu` scaffold (TMA producer + IMMA consumers) — **OFF by default**

#### Special — implement in PropMiner

| Insight | Action | Impact |
|---------|--------|--------|
| **99 KB SMEM/block max** | Fix autotune budget, profile comments | Critical — prevents invalid configs |
| **Ship GeForce warp-specialized kernel** | Enable after `verify_geforce_transcript.sh` | +10–25% |
| **`KBLOCK=64, STAGES=3/4`** | Valid within 99 KB; explore vs current 2-stage | +5–15% |
| **`MIN_BLOCKS=2`** | If registers permit under 99 KB smem | +5–15% |
| **Cluster 1×1×1** | No multicast TMA on consumer; reconcile host `cluster_m=2` | Tune empirically |
| **`sm120f` fatbin** | Only if JIT kernel cache added | Low |

#### SM120 doctrine corrections needed in PropMiner

| File | Problem | Fix |
|------|---------|-----|
| `rtx5090_profile.h` | Says 164 KiB smem, tcgen05 layouts | 99 KB/block; register accumulators |
| `tune_kernel_knobs_common.sh` | Allows ≤163840 B | ≤101376 B |
| `RTX5090_BLUEPRINT.md` | Lists tcgen05 as desired | Mark invalid on GeForce |
| `05-kernel-knob-autotune.md` | STAGES=3 @ KBLOCK=128 "marginal" | Invalid on SM120 (144 KiB > 99 KB) |

#### Triple-generation angle
**SM80 MMA semantics** (proof) + **SM90 TMA warp specialization** (memory) + **reject SM100 tcgen05** (datacenter). Protocol locks tile shape — gains come from load path and pipeline depth, not UMMA/FP4.

---

### 6. triton

**What it is:** Python GPU kernel DSL with autotune.

#### Already in PropMiner
- Hand-written CUDA/CUTLASS with fixed proof-canonical 128×256×128
- `tune_blackwell_knobs.sh`, `GpuTuner`, `tune_prod_5090.sh` for sub-tile sweeps
- Fused headless GEMM + transcript + BLAKE3 in one kernel

#### Special — use in PropMiner (research only)

| Use | Verdict |
|-----|---------|
| Offline sub-tile pipeline exploration | Valid — port winners back to CUDA |
| Pure-GEMM ceiling baselines vs CUTLASS | Calibration for headroom math |
| TMA staging prototypes | Inform consumer kernel swizzle choices |
| Production kernel replacement | **No** — cannot guarantee byte-identical transcript |
| Tile M/N/K autotune | **Blocked** — breaks `partition_C` / pool parity |

#### Triple-generation angle
Triton sm120 backend uses MMAv2/m16n8k32 — same ISA family as PropMiner consumer kernel. Configs do not transfer across Hopper/datacenter/consumer generations.

**Recommendation:** Research sandbox only. Production tuning stays with `tune_prod_5090.sh`.

---

### 7. BLAKE3

**What it is:** Official CPU SIMD reference — no first-party CUDA.

#### Already in PropMiner
- Full GPU `blake3.cuh` compress primitive in pearl-gemm
- GPU merkle/tensor_hash pipeline, arch dispatch (SM90 vs SM80 on 5090)
- Fused jackpot compress in `pow_utils.hpp`
- Basic host golden (one empty-hash vector)

#### Special — implement in PropMiner

| Technique | Where in BLAKE3 | Impact |
|-----------|-----------------|--------|
| **`blake3_hash_many` SIMD batching** | `c/blake3_dispatch.c` | Medium for σ-install B merkle |
| **`blake3_xof_many`** | `blake3_dispatch.c` | BSeed expand at σ-install |
| **Warp-lane batched parent compress** | Analog of `compress_parents_parallel` | Merkle tail reduction |
| **Full `test_vectors.json` suite** | `test_vectors/` | Correctness before perf |
| **AVX2/AVX512 host verify** | `c/blake3.c` + dispatch | Share audit path only |

#### Triple-generation angle (three hash roles)

| Role | Frequency | Optimization priority |
|------|-----------|----------------------|
| Matrix commitment (B merkle) | Once per σ | **Highest BLAKE3 ROI** |
| Noise seed derivation | Once per σ | Low (tiny) |
| Jackpot/PoW compress | Every CTA/iter | Already fused; ~2–5% CTA time |

Steady-state BLAKE3 is ~5–15% of iteration; σ-install B tensor_hash dominates BLAKE3 wall time.

---

### 8. pearl-miner-docker

**What it is:** H100 vLLM useful-work Docker ops — not a stratum miner.

#### Already in PropMiner
- Richer entrypoint modes (`full`, `mine`, `tune`, `tune-prod`, etc.)
- Triple-stream GPU overlap (ping/pong + seed + merkle) — **ahead of pearl-docker**
- CUDA graphs (pearl-docker disables graphs for vLLM compatibility)
- `rtx5090_profile.h`, batch/cluster autotune, watchdog restart loop

#### Special — implement in PropMiner

| Pattern | pearl-docker | Impact |
|---------|--------------|--------|
| **GPU clock lock at container start** | `nvidia-smi -pm 1`, `-lgc` | +5–15% sustained (5090-specific values, not H100's 1200/2100) |
| **Scrapeable metrics endpoint** | `:8339/metrics` on gateway | Fleet dashboards, under-performer detection |
| **`vast-template.yaml`** | Marketplace deploy schema | Zero-friction Vast deploy |
| **Machine attribution in logs** | `VASTAI_*` env vars | Fleet forensics |
| **Persistent combined log + tee** | `pearl_combined.log` | Remote debugging |
| **OS sysctl/THP tuning** | optimization-plan-v2 Phase 5 | ~5–10% cumulative |

#### Do NOT copy from pearl-docker
- `CUDA_MODULE_LOADING=LAZY`, `MAX_CONNECTIONS=8` — vLLM-specific; PropMiner uses EAGER + CONNECTIONS=1 for graphs
- `PEARL_WORKERS` HTTP flood, vLLM batch tokens, chain sync gate, pearld/gateway

#### Triple-generation angle
pearl-docker saturates via **32×GPU_COUNT HTTP workers** to vLLM. PropMiner saturates via **triple-stream + CUDA graphs** — correct model for pool mining. Remaining gap: optional third half-buffer for share-deferral stalls.

---

## Cross-repo dependency map

```text
                    ┌─────────────────────────────────────┐
                    │  PropMiner production hot path      │
                    │  transcript_gemm_sm120_geforce.cu   │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  blackwell-geforce          cutlass ex.79/87          pearl-gemm fork
  (99KB doctrine)            (TMA warp-spec)           (protocol semantics)
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Host: rtx5090_profile + gpu_worker │
                    │  triple-stream + optional 3rd half  │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
         ethminer              cccl                 pearl-docker
    (triple-buffer idea)   (share atomics)      (clock + metrics)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   ▼
                              BLAKE3
                         (σ-install merkle)
```

---

## Impact vs effort matrix

| Item | Impact | Effort | Repo |
|------|--------|--------|------|
| Enable GeForce warp-specialized kernel | ★★★★★ | ★★★ | blackwell, cutlass |
| Fix 99 KB smem autotune budget | ★★★★☆ | ★☆☆ | blackwell |
| CUTLASS SM120 TMA mainloop port | ★★★★☆ | ★★★★ | cutlass |
| Ptr-array grouped GEMM for batch | ★★★☆☆ | ★★★★ | cutlass |
| Triple compute half-buffer | ★★★☆☆ | ★★★ | ethminer |
| Stream-split pre-GEMM phases | ★★★☆☆ | ★★★★ | pearl |
| Container GPU clock lock | ★★★☆☆ | ★☆☆ | pearl-miner-docker |
| Metrics endpoint | ★★☆☆☆ | ★★☆ | pearl-miner-docker |
| B tensor_hash warp batching | ★★☆☆☆ | ★★★ | BLAKE3 |
| CCCL share compaction | ★☆☆☆☆ | ★★☆ | cccl |
| Triton offline autotune | ★☆☆☆☆ | ★★☆ | triton |

---

## Suggested next actions on 5090 hardware

1. Run `verify_geforce_transcript.sh` — gate for flipping GeForce kernel default
2. `ncu` profile consumer vs geforce: tensor-pipe %, achieved occupancy, smem usage
3. `tune_cluster_sweep.sh` with `cluster_m` {1,2,4} — reconcile host vs kernel defaults
4. Profile `wait_until_half_free` under production diff — decide on triple-buffer
5. Bench `KBLOCK=64, STAGES=3` vs `KBLOCK=128, STAGES=2` within 99 KB budget
6. Add `nvidia-smi -pm 1` + 5090 `-lgc` to `docker_entrypoint.sh` and re-bench

---

## Repos excluded from this pass

| Repo | Reason |
|------|--------|
| `akoya-miner` | User request — already primary sm120 reference |
| `alpha-miner` | User request — source-less; docs-only benchmark target |
| `cuda-samples` | Clone failed (disk full); `cccl` substituted |

---

*Research only — synthesizes `REPORT.md` and subagent analysis. No changes under `PropMiner/src/`.*
