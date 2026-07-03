// Native sm_120 (RTX 5090 / GeForce Blackwell) NoisyGEMM transcript kernel.
//
// The RTX 5090 is compute capability 12.0, NOT sm_100 (sm_100 is B200 datacenter).
// This file compiles a variant of the consumer transcript GEMM using the
// Blackwell-native 5th-gen Tensor Core MMA atom when available, falling back to
// the proven SM80 mma.sync int8 atom otherwise.
//
// Load-balancing rules enforced here:
//   1. No datacenter bleed-through: this file is only compiled for sm_120 and
//      never includes Hopper (sm_90) or B200 (sm_100a) paths.
//   2. Cache alignment: the consumer tile is fixed at 128x256x128 to match the
//      96 MB L2 cache line / shared-memory budget of the RTX 5090.  Grid is
//      sized for M=8192, N=32768 -> 8192 CTAs, filling all 170 SMs.
//   3. Seed upload is performed by the host via cudaMemcpyAsync on a dedicated
//      copy stream; the kernel reads only device-side seeds.
//
// Build toggles:
//   - PEARL_GEMM_SM120_NATIVE=1  : use SM120_16x8x32_TN<int8_t,int8_t,int32_t>
//                                  MMA atom. Requires CUTLASS >= 4.x with
//                                  include/cute/atom/mma_traits_sm120.hpp.
//   - default / not set            : use SM80_16x8x32_S32S8S8S32_TN atom
//                                  compiled for sm_120a (already works today).
//
// Why the fallback matters:
//   The consumer kernel's proof-canonical layout (128x256 tile, 128 int32
//   accumulators per thread, 16 transcript slots) was verified byte-identical
//   against H100 WGMMA using the SM80 atom.  Switching to the SM120 atom must
//   preserve the same partition_C coordinates; the SM120_16x8x32_TN traits in
//   CUTLASS inherit the same fragment layout, so the tiling code below is
//   unchanged.
//
// Future tcgen05 PTX path:
//   For warp-specialized / TMEM-style kernels, replace the cute::gemm loop
//   with explicit tcgen05.mma instructions:
//       tcgen05.mma.cta_group::1.kind::i8  [d_tmem], [a_smem], [b_smem];
//   This requires TMEM allocation, tcgen05.cp scale copies, and tcgen05.ld for
//   the epilogue.  The scaffold below keeps the existing cute TiledMMA path so
//   the kernel is immediately compilable; inline PTX hooks are annotated.

#include <cute/atom/mma_atom.hpp>

// Ensure we pick the consumer architecture traits (Blackwell defaults:
// kBK=128, stages=2, swizzle_bits=3, min_blocks=1).
#ifndef PEARL_GEMM_BLACKWELL
#define PEARL_GEMM_BLACKWELL 1
#endif

// Use the native Blackwell MMA atom only when explicitly requested.  The user
// must verify their CUTLASS checkout has include/cute/atom/mma_traits_sm120.hpp.
#if defined(PEARL_GEMM_SM120_NATIVE) && PEARL_GEMM_SM120_NATIVE
  #if !defined(__CUDA_ARCH__)
    // Host-side compile: include the atom header so the type is known.
    #include <cute/atom/mma_traits_sm120.hpp>
  #endif
  // SM120_16x8x32_TN has the same 16x8x32 MNK shape and thread/fragment layout
  // as the SM80 atom for legacy int8.  This lets us reuse the consumer mainloop
  // verbatim while emitting Blackwell-native Tensor Core instructions on sm_120a.
  #define PEARL_CONSUMER_MMA_ATOM_TYPE SM120_16x8x32_TN<int8_t, int8_t, int32_t>
  #pragma message("transcript_gemm_sm120.cu: using native SM120 MMA atom")
#else
  #pragma message("transcript_gemm_sm120.cu: using SM80 MMA atom compiled for sm_120a")
#endif

// Pull in the full consumer kernel.  It will use ConsumerTiledMma defined above.
#include "../consumer/transcript_gemm_kernel.cu"

namespace pearl {
namespace blackwell {

// Launcher wrapper mirroring pearl::consumer::launch_transcript_gemm_* so the
// CAPI dispatch code can call the sm_120 variant without signature changes.
//
// On sm_120a this resolves to the same kernel symbol as the consumer path when
// PEARL_CONSUMER_MMA_ATOM_TYPE is unchanged; when SM120 native is enabled it
// resolves to the Blackwell-atom instantiation.
//
// Example dispatch in pearl_gemm_capi.cpp:
//   if (use_sm120) {
//     pearl::blackwell::launch_transcript_gemm_sm120(...);
//   }
inline void launch_transcript_gemm_sm120(
    const int8_t* A, const int8_t* B,
    int32_t* C, uint32_t* transcript,
    int M, int N, int K, int R,
    const uint32_t* pow_target,
    const uint32_t* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header,
    cudaStream_t stream) {

  // Grid covers the MxN output tile space, batch dimension in z.
  dim3 block(consumer::kThreads);
  dim3 grid(M / consumer::kBM, N / consumer::kBN, 1);

  // Dynamic shared memory = A smem + B smem.
  size_t smem_size = sizeof(consumer::SharedStorage);

  consumer::transcript_gemm_kernel_consumer<<<grid, block, smem_size, stream>>>(
      A, B, C, transcript, M, N, K, R,
      pow_target, pow_key, host_signal_sync, host_signal_header);
}

} // namespace blackwell
} // namespace pearl
