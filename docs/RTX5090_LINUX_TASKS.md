# RTX 5090 Linux/CUDA Completion Tasks

Run these in order on the target machine. These steps validate the load-balancing rules built into the code:

- **No datacenter bleed-through**: all builds target `sm_120`, not `sm_90` or `sm_100a`.
- **CPU hashing isolation**: the CPU only generates seeds via `SeedGenerator`; all matrix/noise/GEMM math runs on the GPU.
- **Strict double-buffering overlap**: `GpuWorker` uses ping/pong compute streams plus a dedicated `seed_copy_stream_` for `cudaMemcpyAsync` seed upload.
- **Cache alignment**: default shape is `M=8192, N=32768, K=128` with tile `128x256x128`, launching `8192` CTAs on `170` SMs.

## 1. Environment check

```bash
nvidia-smi
nvcc --version  # must be 12.8 or newer
nvidia-smi --query-gpu=compute_cap,name --format=csv,noheader  # expect 12.0, RTX 5090
```

## 2. First build and baseline benchmark

```bash
cd PropMiner
./scripts/build_and_benchmark.sh 60
```

This compiles `sm_120` with the SM80 MMA atom and runs the hard-coded `--rtx5090` profile (M=8192, N=32768, batch=20). Note the reported H/s. Expected grid: `(8192/128)*(32768/256) = 64*128 = 8192` CTAs, ~48 waves over 170 SMs.

## 3. Profile with ncu to find the bottleneck

```bash
ncu -o profile_base --target-processes all \
   --kernel-regex regex:transcript_gemm_kernel_consumer \
   ./build/propminer --bench 10 --rtx5090 --gpus 0
ncu -i profile_base.ncu-rep
```

Look at:
- Achieved occupancy vs theoretical.
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` — want >80%.
- Memory throughput vs peak GDDR7 — if >70%, you are memory-bound.

## 4. Try the native SM120 MMA atom

```bash
cmake -S . -B build_native \
  -DCMAKE_BUILD_TYPE=Release \
  -DPROP_MINER_CUDA_ARCH=blackwell \
  -DCMAKE_CUDA_FLAGS="-DPEARL_GEMM_SM120_NATIVE=1"
cmake --build build_native --target propminer -j$(nproc)
./build_native/propminer --bench 60 --rtx5090 --gpus 0
```

If it fails to compile, your CUTLASS checkout is missing `include/cute/atom/mma_traits_sm120.hpp`. Update CUTLASS:

```bash
cd third_party/pearl-gemm/third_party/cutlass
git fetch origin
git checkout main  # or the latest 4.x tag
```

Compare H/s to step 2. Keep the faster build.

## 5. Sweep Blackwell kernel knobs

```bash
./scripts/tune_blackwell_knobs.sh 10
```

This builds variants with different (BM, BN, KBLOCK, STAGES, SWIZZLE, MIN_BLOCKS, LOAD_POLICY) and runs each for 10 seconds. It stages the fastest `.so` as `build/libpearl_gemm_capi.so`.

## 6. Compare to SRBMiner

Run SRBMiner-MULTI on the same pool/job for the same duration and note H/s. The custom miner should match or exceed it once the native SM120 path and tuned knobs are in place.

## 7. Stability validation

Run the self-test after every kernel change:

```bash
./build/propminer --self-test --rtx5090 --gpus 0
```

If it fails, the new kernel produces invalid proofs; do not use it.

## 8. Overclocking verification

Start with stock clocks, then in small steps:

```bash
# Example with nvidia-smi (adjust for your driver/cards)
nvidia-smi -i 0 -pl 500              # 500 W power limit (stock ~575 W)
nvidia-smi -i 0 -lgc 2520            # lock core clock to 2520 MHz
# Do NOT lock memory to minimum; leave at stock.
```

Run `--self-test` after each change. If it passes and hashrate rises, keep it. If the watchdog triggers, reduce core offset.

## 9. Final integration

Once the fastest config is known, commit the winning CMake defaults back to `PropMiner/CMakeLists.txt` and update `Rtx5090Profile` if needed.
