# Git repository mining report — RTX 5090 / Pearl NoisyGEMM

**Workspace:** `PropMiner/research/01-git-repos-mining/`  
**Clones:** `repos/` (exactly **10** shallow `git clone --depth 1` directories)  
**Manifest:** `CLONE_MANIFEST.txt` (directory name, URL, `git rev-parse HEAD`)  
**Date:** 2026-07-06  

## Executive summary

These ten public repos cover the stack PropMiner cares about: **official Pearl NoisyGEMM + gateway**, **Akoya’s open reference miner (sm_120 kernels + gRPC pool protocol)**, **CUTLASS / Blackwell SM120 GEMM constraints**, **Triton for autotuning experiments**, **BLAKE3 reference**, **classic CUDA stratum miner patterns (ethminer)**, **closed-source alpha-miner ops docs (5090 backend flags)**, and **H100 vLLM Docker deployment**.  

**Honest failures / substitutions**

| Requested | Outcome |
|-----------|---------|
| `NVIDIA/cuda-samples` | **Failed** — shallow clone started but checkout aborted with **No space left on device** (~234–447 MiB free on `/System/Volumes/Data`). Partial tree removed. **Substitute:** `NVIDIA/cccl` (CUDA C++ core libraries, closer to kernel-building than sample PPM assets). |
| `AlphaMine-Tech/alpha-miner` source | Repo is **public but source-less** (README + release/install scripts only; author keeps CUDA private). Still cloned for **5090 `--force-backend blackwell` / stratum / static diff** documentation. |
| Pearl gRPC / stratum miner | **Akoya** (`proto/v2/miner.proto`) = gRPC; **alpha-miner** = stratum TCP. Both represented. |
| Noisy GEMM “best match” | **`pearl-research-labs/pearl`** (canonical `miner/pearl-gemm`) plus **`akoyapool/akoya-miner`** (production-oriented native kernels). |

Verified clone count:

```bash
ls -1 repos/ | wc -l   # → 10
```

---

## 1. NVIDIA CUTLASS

| Field | Value |
|-------|--------|
| **URL** | https://github.com/NVIDIA/cutlass |
| **Local path** | `repos/cutlass/` |
| **Commit** | `e8ecfad75b44d1ad56264f5001d877e9e47fe080` |

**What it is:** NVIDIA’s template library for high-performance GEMM, convolutions, and mixed-precision tensor-core kernels (Hopper WGMMA, Blackwell variants, grouped GEMM, etc.).

**Good for 5090 Pearl mining**

- Pearl/Akoya noisy GEMM builds on CUTLASS-style collectives; this repo is the **upstream spec** for tile sizes, epilogues, and SM120 tensor-op tests.
- SM120 unit tests show which **FP4/FP8/FP16** tensor-op combinations NVIDIA expects on GeForce Blackwell — useful when tuning int8/noisy pipelines without guessing PTX shapes.

**Applicable ideas / hacks**

- Start from **`test/unit/gemm/device/sm120_tensorop_gemm/`** when validating that your nvcc flags and dtypes compile on sm_120 before integrating into `pearl-gemm`.
- Compare **grouped GEMM** examples if PropMiner adds MoE-style batching (multiple expert tiles per launch).
- Use CUTLASS profiler / Python bindings (if enabled in tree) to sweep tile configs offline on 5090.

**Paths worth reading**

- `test/unit/gemm/device/sm120_tensorop_gemm/` — SM120 GEMM conformance tests (e.g. `sm120_gemm_f8_f8_f32_tensor_op.cu`, FP4 variants).
- `test/unit/gemm/device/sm120_sparse_tensorop_gemm/` — sparse tensor-op on SM120.
- `README.md`, `examples/README.md` — build flags and architecture targets.
- `python/` — optional autotuning / codegen entry points.

---

## 2. Akoya Miner (open Pearl GPU miner)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/akoyapool/akoya-miner |
| **Local path** | `repos/akoya-miner/` |
| **Commit** | `0e42b2a50b4cb62cb8995b77420226c2182137ce` |

**What it is:** Official **open-source reference miner** for Akoya Pool (Pearl PRL). C# host (Native AOT) + **`native/pearl-gemm`** CUDA/ROCm kernels + Rust BLAKE3/merkle C API. Documents **RTX 50-series → `PEARL_GEMM_ARCH=blackwell` (sm_120)**.

**Good for 5090 Pearl mining**

- **Direct sm_120 kernel sources** (`transcript_gemm_sm120.cu`) and Blackwell sm100 variants for datacenter cards.
- End-to-end **low-rank noised integer GEMM PoW**: GPU matmul → BLAKE3 keyed merkle → difficulty check.
- **gRPC pool wire protocol** — best public spec for pool↔miner messaging (item #10 in mission list).

**Applicable ideas / hacks**

- Mirror **`build.sh`** arch detection: auto Blackwell vs portable fallback.
- Study **`native/pearl-gemm/csrc/gemm/`** for noising, inner hash, host signal headers — parallels PropMiner’s `third_party/pearl-gemm`.
- **`proto/v2/miner.proto`** — implement compatible client or fuzz pool edge cases.
- On-disk BLAKE3 in CUDA: `native/pearl-gemm/csrc/blake3/blake3.cu`.

**Paths worth reading**

- `README.md` — GPU matrix, CUDA 12.4+, arch env vars.
- `proto/v2/miner.proto` — gRPC miner protocol (249 lines).
- `native/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu`
- `native/pearl-gemm/csrc/blackwell/transcript_gemm_sm100.cu`
- `native/pearl-gemm/csrc/gemm/pow_utils.hpp`, `inner_hash_kernel.h`, `pearl_gemm_host.h`
- `native/pearl-gemm/csrc/capi/pearl_gemm_capi.cpp` — stable C ABI boundary.
- `native/pearl-blake3/`, `native/pearl-mining-capi/` — merkle/commitment host side.
- `src/Akoya.Pool/` — session state machine for pool connection.

---

## 3. ethminer (CUDA + stratum reference)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/ethereum-mining/ethminer |
| **Local path** | `repos/ethminer/` |
| **Commit** | `8eaf50ab3baad9a1747e04dc295793d46740dcae` |

**What it is:** Mature **open-source GPU miner** (Ethash) with **CUDA and stratum** support — not Pearl, but the classic pattern for host-side work distribution, DAG generation on GPU, and pool failover.

**Good for 5090 Pearl mining**

- **Stratum client architecture** (job id, extranonce, share submission) — analogous to alpha-miner’s TCP stratum even though work unit differs.
- **CUDA miner loop** patterns: kernel launch scheduling, device picking, benchmark mode.
- On-GPU constant memory / kernel splitting lessons still apply when batching many small PoW checks.

**Applicable ideas / hacks**

- Borrow **multi-GPU farm** and **failover** UX patterns from README pool examples.
- Inspect how **`CUDAMiner.cpp`** batches work and reports hashrate — useful for PropMiner host telemetry.

**Paths worth reading**

- `libethash-cuda/ethash_cuda_miner_kernel.cu` — main CUDA kernel.
- `libethash-cuda/CUDAMiner.cpp`, `CUDAMiner.h`
- `libethash-cuda/dagger_shuffled.cuh`, `keccak.cuh` — GPU-friendly hash primitives.
- `README.md` — stratum URL examples, build matrix.

---

## 4. Alpha Miner (Pearl stratum — docs only)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/AlphaMine-Tech/alpha-miner |
| **Local path** | `repos/alpha-miner/` |
| **Commit** | `f731b2af1aeec255587fe4e4dee393d7fc02fd27` |

**What it is:** **Binary-first** Pearl miner distribution for AlphaPool. Repository contains **README, install.sh, Docker hints** — **no CUDA source** (“Source remains private”).

**Good for 5090 Pearl mining**

- Confirms **Blackwell sm_120** is a first-class backend: `--force-backend blackwell|blackwell-native` for RTX 50-series.
- **Stratum endpoints** (regional hosts, PPLNS 5566 / SOLO 5567), static difficulty via password `x;d=N` — critical for **Vast.ai / HiveOS** style wrappers.
- Competitive baseline: what closed miners claim for arch auto-detect vs Akoya open kernels.

**Applicable ideas / hacks**

- Use as **black-box benchmark target** on 5090 (same pool, same diff password syntax).
- Align PropMiner pool config with documented **hostnames** (never `pearl.alphapool.tech` for stratum).
- Docker env vars (`PEARL_DEVICES`, `PEARL_DIFFICULTY`) mirror production fleet patterns.

**Paths worth reading**

- `README.md` — hardware table, CLI flags, systemd/HiveOS/Docker.
- `install.sh` — release fetch pattern (not kernel logic).

---

## 5. Pearl protocol monorepo (canonical NoisyGEMM)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/pearl-research-labs/pearl |
| **Local path** | `repos/pearl/` |
| **Commit** | `f00a8112abdd347eef1ee91adb30d9b3dfc1c8e7` |

**What it is:** Official **Pearl L1** monorepo: node (`pearld`), wallet, ZK (`zk-pow`), and **`miner/`** with **`pearl-gemm`** CUDA (NoisyGEMM), **`pearl-gateway`**, and **vLLM miner** plugin. Proof-of-Useful-Work via matrix multiply ([paper](https://arxiv.org/abs/2504.09971)).

**Good for 5090 Pearl mining**

- **Authoritative noisy GEMM + tensor hash + merkle** implementation PropMiner tracks.
- **`pearl-gateway`** — JSON-RPC to node, UDS/TCP miner RPC (port 8337) — solo mining architecture.
- CI includes **`miner_gpu_ci.yml`** — upstream test expectations.
- **Caveat in README:** upstream tests historically target **sm90 (H100)**; 5090 sm_120 support may lag Akoya/PropMiner forks — compare kernels side-by-side.

**Applicable ideas / hacks**

- Port sm_120 paths from Akoya/PropMiner back into upstream-shaped APIs in `miner/pearl-gemm/csrc/`.
- **`miner/miner-base/`** — async loop, commitment hashes, merkle trees, gateway client (host logic).
- **`py-pearl-mining/`** — PyO3 bindings for scripted tuning.
- **`pearl-blake3/`** (Rust) — merkle semantics reference.

**Paths worth reading**

- `miner/README.md` — package layout, uv build, `PEARL_GEMM_FORCE_BUILD`.
- `miner/pearl-gemm/csrc/gemm/noise_generation.cu`, `denoise_converter.cu`, `inner_hash_kernel.cu`
- `miner/pearl-gemm/csrc/tensor_hash/tensor_hash.cu`
- `miner/pearl-gemm/csrc/blake3/blake3.cu`
- `miner/pearl-gateway/` — gateway bridge.
- `miner/vllm-miner/` — useful-work inference path (H100-class).
- `zk-pow/` — verifier-side constraints (informs what GPU must prove).
- `node/sample-pearld.conf` — RPC/mining flags.

---

## 6. NVIDIA CCCL (cuda-samples substitute)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/NVIDIA/cccl |
| **Local path** | `repos/cccl/` |
| **Commit** | `55f0de953f9f06822840c1136a51cbcf032d5857` |
| **Requested instead** | https://github.com/NVIDIA/cuda-samples — **not present** (disk full during checkout). |

**What it is:** **CUDA Core Compute Libraries** — unified **Thrust, CUB, libcudacxx** (parallel algorithms, block/warp collectives, device atomics).

**Good for 5090 Pearl mining**

- **CUB** block reduce / scan / merge patterns for on-GPU difficulty aggregation and prefix sums over candidate tiles.
- **libcudacxx** `cuda::atomic_ref` and memory order — safer cross-block counters than raw `atomicAdd` alone.
- Lighter clone than full cuda-samples (~50M vs 200M+); fits research disk budget.

**Applicable ideas / hacks**

- Replace hand-rolled warp reductions in hash pipelines with **CUB BlockReduce** where occupancy allows.
- Use Thrust for host-side verification against GPU merkle roots during parity tests.

**Paths worth reading**

- `README.md` — unified library overview + Godbolt example.
- `cub/block/` — cooperative primitives.
- `libcudacxx/include/cuda/` — atomics, barriers.
- `cudax/` — newer CUDA C++ experimental APIs.

**Retry cuda-samples later (when disk free):**

```bash
git clone --depth 1 --filter=blob:none https://github.com/NVIDIA/cuda-samples cuda-samples
# optional sparse checkout of one sample only, e.g. matrixMul
```

---

## 7. Blackwell GeForce NVFP4 GEMM (SM120 architecture)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm |
| **Local path** | `repos/blackwell-geforce-nvfp4-gemm/` |
| **Commit** | `acabe06f6eea9e11dfb58dd43e400782752835a5` |

**What it is:** Documentation + patches for **RTX 5090/5080 (SM120)** NVFP4 inference — explains why **SM120 ≠ SM100**: no TMEM/tcgen05, **99 KB SMEM/SM**, **`mma.sync`** (Ampere-style) + TMA + block-scaled FP4.

**Good for 5090 Pearl mining**

- **Critical for kernel authors:** DeepGEMM / SM100 WGMMA kernels **do not port** to 5090; Pearl sm_120 must use SM120-compatible instruction paths (matches Akoya `transcript_gemm_sm120.cu` direction).
- Tile sizing constraints (**99 KB shared memory**) directly bound NoisyGEMM tile configs — oversize tiles silently hurt occupancy or fail JIT.
- JIT gencode notes (`sm120a`, `sm120f`) apply when PropMiner builds fatbins for 5090-only fleets.

**Applicable ideas / hacks**

- Read **`docs/sm120-architecture.md`** before changing CUTLASS collectives in PropMiner.
- Apply patch mindset: restrict MoE/grouped GEMM tiles for SM120 (FlashInfer patch docs in `patches/`).
- Avoid compiling **sm_100f** cubins for 5090 deploy images.

**Paths worth reading**

- `README.md` — “SM120 Chimera” table, patch index.
- `docs/sm120-architecture.md` — ISA comparison SM80/90/100/120.
- `patches/flashinfer/01-sm120-grouped-gemm-tiles.md`
- `patches/vllm/02-qutlass-ada-mxf4-matmul.md`, `patches/quack/11-quack-gemm-sm120.md`

**Alternatives considered (not cloned):** `shiinamiyuki/sm120_gemm`, `waynehacking8/blackwell-tensorcore-kernels` — strong GEMM benches; chose lna-lab repo for 5090-specific NVFP4 + SMEM doctrine.

---

## 8. Triton (GEMM autotune / DSL)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/triton-lang/triton |
| **Local path** | `repos/triton/` |
| **Commit** | `8929ee538d7c8737ae6d6aaba6183eefd7d4def3` |

**What it is:** Python DSL + compiler for GPU kernels; widely used for **autotuned GEMM** and fused ops (also TVM-adjacent workflow).

**Good for 5090 Pearl mining**

- Rapid **tile-size autotuning** for experimental noisy-GEMM epilogues without full CUTLASS template metaprogramming.
- Tutorials for **grouped GEMM** and **TMA store GEMM** — relevant if PropMiner prototypes fusion (matmul + hash) in Triton before hand-CUDA.
- CI/tests track evolving **NVIDIA backend** behavior (including newer arch support over time).

**Applicable ideas / hacks**

- Start from `python/tutorials/08-grouped-gemm.py` for MoE routing batches.
- Use `python/test/unit/cuda/test_tma_store_gemm.py` as reference for TMA + GEMM interaction on recent drivers.
- Compare Triton-autotuned baseline vs CUTLASS sm120 on 5090 for specific `(M,N,K)` Pearl tile shapes.

**Paths worth reading**

- `README.md`
- `python/tutorials/08-grouped-gemm.py`
- `python/test/unit/cuda/test_tma_store_gemm.py`
- `third_party/nvidia/` (backend specifics — browse for libdevice / warp specs).

---

## 9. BLAKE3 (reference implementation)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/BLAKE3-team/BLAKE3 |
| **Local path** | `repos/BLAKE3/` |
| **Commit** | `8aa5145039b972ba30e98e788752d37d14568824` |

**What it is:** Official **BLAKE3** hash ( SIMD CPU: AVX2/AVX512/NEON); reference C and Rust crates. **No first-party CUDA tree** in upstream — Pearl miners implement GPU BLAKE3 separately.

**Good for 5090 Pearl mining**

- **Golden vectors** for merkle / keyed hashing parity between CPU and GPU.
- CPU fast paths for **host verification** of shares and test harnesses.
- Compare against Pearl/Akoya **`blake3.cu`** forks for bitwise correctness.

**Applicable ideas / hacks**

- Use `c/blake3.c` + tests when validating `pearl-blake3` / `tensor_hash` outputs.
- Study dispatch (`blake3_dispatch.c`) for platform-specific optimization patterns analogous to multi-arch fatbin dispatch.

**Paths worth reading**

- `c/blake3.c`, `c/blake3.h`, `c/blake3_impl.h`
- `reference_impl/` — simple spec reference.
- `README.md`

**GPU merkle mining (in other clones):**

- `repos/akoya-miner/native/pearl-gemm/csrc/blake3/blake3.cu`
- `repos/pearl/miner/pearl-gemm/csrc/blake3/blake3.cu`
- `repos/pearl/miner/pearl-gemm/csrc/tensor_hash/tensor_hash.cu`

---

## 10. Pearl Miner Docker (gateway + vLLM stack)

| Field | Value |
|-------|--------|
| **URL** | https://github.com/terrapin88/pearl-miner-docker |
| **Local path** | `repos/pearl-miner-docker/` |
| **Commit** | `37ead3e8434843cab8ce618b90536c62e4e899c0` |

**What it is:** **One-click Docker** for Pearl **useful-work mining**: `pearld` + `pearl-gateway` + vLLM with NoisyGEMM — targeting **H100/H200 (sm90)** cloud templates (Vast.ai, RunPod).

**Good for 5090 Pearl mining**

- **Operational playbook**: env vars, metrics ports (`8339`), chain sync, HF model cache — adapt for PropMiner container entrypoints.
- **`docs/fleet-optimization-plan.md`**, **`docs/optimization-plan-v2.md`** — fleet tuning notes (may inform 5090 fleet economics even if GPU arch differs).
- **Not** a stratum miner — pairs with **solo/gateway** model; combine mentally with **Akoya gRPC** or **alpha stratum** for pool mining.

**Applicable ideas / hacks**

- `entrypoint.sh`, `pearl_worker.py` — process supervision patterns.
- `vast-template.yaml` — marketplace template fields for PropMiner images.
- Replicate metrics scraping from gateway for production monitoring.

**Paths worth reading**

- `README.md`, `QUICK_START_GUIDE.md`
- `Dockerfile`, `entrypoint.sh`
- `pearl_worker.py`, `block_watcher.sh`
- `docs/fleet-optimization-plan.md`, `docs/optimization-plan-v2.md`
- `guides/HOW_TO_MINE_PEARL.md`

---

## Cross-repo map (PropMiner focus)

```text
Pool / host                         GPU PoW core                    Arch / tuning
─────────────────────────────────────────────────────────────────────────────────
alpha-miner (stratum docs)          akoya-miner/pearl-gemm          cutlass (sm120 tests)
akoya-miner/proto gRPC              pearl/miner/pearl-gemm          blackwell-geforce-nvfp4-gemm
ethminer (stratum CUDA patterns)    blake3.cu in pearl/akoya        triton (autotune)
pearl-miner-docker (gateway ops)    BLAKE3 (CPU golden)             cccl (CUB/atomics)
```

## Suggested reading order for 5090 kernel work

1. `blackwell-geforce-nvfp4-gemm/docs/sm120-architecture.md` — hardware constraints  
2. `akoya-miner/native/pearl-gemm/csrc/blackwell/transcript_gemm_sm120.cu` — working sm_120 Pearl kernel reference  
3. `cutlass/test/unit/gemm/device/sm120_tensorop_gemm/` — NVIDIA dtype/tile coverage  
4. `pearl/miner/pearl-gemm/csrc/tensor_hash/` — merkle + PoW extraction semantics  
5. PropMiner local tree (not modified in this research pass) — diff against above  

## Clone verification commands

```bash
cd research/01-git-repos-mining/repos
for d in */; do
  echo "$d $(git -C "$d" rev-parse --is-shallow-repository) $(git -C "$d" remote get-url origin)"
done
```

All ten report `is-shallow-repository = true`.

---

*Research only — no changes under `PropMiner/src/`.*
