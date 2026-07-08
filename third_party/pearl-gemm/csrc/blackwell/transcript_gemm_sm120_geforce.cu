// RTX 5090 (sm_120a) warp-specialized Pearl transcript GEMM.
//
// GeForce Blackwell does NOT implement tcgen05/TMEM (see external_repos/ and
// phase0_tcgen05_sm120_probe.cu). This kernel is the hardware-correct "max
// utilization" path for RTX 5090:
//   - Warp 8 (thread 256): dedicated TMA producer for A/B K-tiles
//   - Warps 0-7 (threads 0-255): consumer IMMA + transcript (SM80 atom)
//
// Proof contract: byte-identical transcript + headless PoW vs
// consumer::transcript_gemm_kernel_consumer. Validate with
// scripts/verify_geforce_transcript.sh before enabling in production.
//
// Compile: PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 (default ON for blackwell).
// Runtime: default geforce when compiled in; PEARL_GEMM_KERNEL=consumer to opt out.
//
// Standalone verify build (from pearl-gemm root, RTX 5090 host):
//   make -f csrc/capi/Makefile PEARL_GEMM_ARCH=blackwell \
//     PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1 PEARL_SM120_GEFORCE_VERIFY_MAIN=1 \
//     verify-geforce-transcript
//   csrc/capi/build/verify_transcript_sm120_geforce

#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <string>
#include <atomic>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm80.h>

#include "../gemm/pow_utils.hpp"
#include "../portable/transcript_kernel.cuh"
#include "../consumer/tma_tile_loader.cuh"
#include "transcript_gemm_sm120_geforce.h"

namespace pearl {
namespace blackwell {
namespace {

using namespace cute;
using namespace pearl::consumer::tma_loader;

static constexpr int kBM = 128;
static constexpr int kBN = 256;
static constexpr int kBK = 128;
static constexpr int kAtomK = 32;
static constexpr int kConsumerThreads = 256;
static constexpr int kProducerLeader = 256;
static constexpr int kThreads = 288;
static constexpr int kFragSize = (kBM * kBN) / kConsumerThreads;
static constexpr int kTranscriptSlots = 16;
static constexpr int kStages = 2;

#ifndef PEARL_CONSUMER_SWIZZLE_BITS
#define PEARL_CONSUMER_SWIZZLE_BITS 3
#endif

using ElementIn = int8_t;
using ElementAcc = int32_t;

using TileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<kBK>>;
using HeaderTileShape_MNK = Shape<Int<kBM>, Int<kBN>, Int<128>>;

using ConsumerTiledMma = TiledMMA<
    MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>,
    Layout<Shape<_8, _1, _1>>,
    Tile<Int<kBM>, Int<kBN>, Int<kAtomK>>>;

using SmemLayoutAtomA = decltype(composition(
    Swizzle<PEARL_CONSUMER_SWIZZLE_BITS, 4, 3>{},
    Layout<Shape<_16, Int<kBK>>, Stride<Int<kBK>, _1>>{}));
using SmemLayoutAtomB = SmemLayoutAtomA;

using SmemLayoutA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{})));
using SmemLayoutB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBN>{}, Int<kBK>{}, Int<kStages>{})));

using ConsumerTmaFlatA =
    SmemLayoutFlatA<SmemLayoutAtomA, SmemLayoutAtomB, kBM, kBN, kBK>;
using ConsumerTmaFlatB =
    SmemLayoutFlatB<SmemLayoutAtomA, SmemLayoutAtomB, kBM, kBN, kBK>;
using ConsumerTmaStagedA = SmemLayoutStagedA<SmemLayoutAtomA, kBM, kBK, kStages>;
using ConsumerTmaStagedB = SmemLayoutStagedB<SmemLayoutAtomB, kBN, kBK, kStages>;
using ConsumerTmaA = TmaA<ElementIn, ConsumerTmaFlatA, ConsumerTmaFlatB>;
using ConsumerTmaB = TmaB<ElementIn, ConsumerTmaFlatA, ConsumerTmaFlatB>;

static_assert(cute::cosize_v<SmemLayoutA> == cute::cosize_v<ConsumerTmaStagedA>);
static_assert(cute::cosize_v<SmemLayoutB> == cute::cosize_v<ConsumerTmaStagedB>);

struct SharedStorage {
  alignas(128) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
  alignas(128) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
  TmaPipelineStorage<kStages> tma_pipe;
};

__launch_bounds__(kThreads, 1)
__global__ void transcript_gemm_sm120_geforce_kernel(
    ElementIn const* __restrict__ A_gmem,
    ElementIn const* __restrict__ B_gmem,
    ElementAcc* __restrict__ C_gmem,
    uint32_t* __restrict__ transcript,
    int M, int N, int K, int R,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    __grid_constant__ ConsumerTmaA const tma_a,
    __grid_constant__ ConsumerTmaB const tma_b) {

  extern __shared__ uint8_t smem_raw[];
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int tid = threadIdx.x;
  const bool is_consumer = (tid < kConsumerThreads);
  const int consumer_tid = tid;

  const int m_tile = blockIdx.x;
  const int n_tile = blockIdx.y;
  const int batch = blockIdx.z;
  const int num_m_tiles = M / kBM;
  const int num_n_tiles = N / kBN;

  ConsumerTiledMma tiled_mma;
  // Both branches must pass int (not cute::_0) so ThrMMA deduces one type.
  auto thr_mma = tiled_mma.get_thread_slice(is_consumer ? consumer_tid : 0);
  Tensor tCrC = make_tensor<ElementAcc>(Shape<Int<kFragSize>>{});
  uint32_t transcript_local[kTranscriptSlots];
  Tensor cD = make_identity_tensor(Shape<Int<kBM>, Int<kBN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);

  if (is_consumer) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < kFragSize; ++j) tCrC(j) = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kTranscriptSlots; ++s) transcript_local[s] = 0;
  }

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B), SmemLayoutB{});

  Tensor mA = make_tensor(make_gmem_ptr(A_gmem),
                          make_shape(M, K), make_stride(K, _1{}));
  Tensor mB = make_tensor(make_gmem_ptr(B_gmem),
                          make_shape(N, K), make_stride(K, _1{}));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  const int K_TILES = K / kBK;
  const int reduce_every_k = R / kBK;

  auto& tma_pipe = smem.tma_pipe;
  init_tma_barriers(tma_pipe, tid);
  __syncthreads();

  // Producer prologue: dedicated warp issues initial pipeline fills.
  if (tid == kProducerLeader) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kStages - 1; ++s) {
      if (s < K_TILES) {
        tma_issue_k_tile<ConsumerTmaA, ConsumerTmaB, ConsumerTmaStagedA,
                         ConsumerTmaStagedB, kBM, kBN, kBK, kStages, ElementIn,
                         decltype(gA), decltype(gB), kProducerLeader>(
            tma_a, tma_b, tma_pipe, smem.smem_A, smem.smem_B, gA, gB, s, s);
      }
    }
  }
  __syncthreads();

  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    int stg = k_iter % kStages;

    tma_wait_stage<kStages>(tma_pipe, stg, k_iter);
    __syncthreads();

    if (tid == kProducerLeader) {
      int next_k = k_iter + kStages - 1;
      if (next_k < K_TILES) {
        tma_issue_k_tile<ConsumerTmaA, ConsumerTmaB, ConsumerTmaStagedA,
                         ConsumerTmaStagedB, kBM, kBN, kBK, kStages, ElementIn,
                         decltype(gA), decltype(gB), kProducerLeader>(
            tma_a, tma_b, tma_pipe, smem.smem_A, smem.smem_B, gA, gB,
            next_k, next_k % kStages);
      }
    }

    if (is_consumer) {
      if (k_iter > 0 && (k_iter % reduce_every_k) == 0) {
        uint32_t hash = pearl::xor_reduction(tCrC);
        int snapshot_idx = (k_iter / reduce_every_k) - 1;
        int slot = snapshot_idx % kTranscriptSlots;
        transcript_local[slot] =
            pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
                transcript_local[slot], hash);
      }

      Tensor sA_stg = sA(_, _, stg);
      Tensor sB_stg = sB(_, _, stg);
      Tensor tCrA = thr_mma.partition_fragment_A(sA_stg);
      Tensor tCrB = thr_mma.partition_fragment_B(sB_stg);

      auto s2r_copy_a = make_tiled_copy_A(
          Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
      auto s2r_thr_copy_a = s2r_copy_a.get_slice(consumer_tid);
      auto tXsA = s2r_thr_copy_a.partition_S(sA_stg);
      auto tXrA = s2r_thr_copy_a.retile_D(tCrA);
      copy(s2r_copy_a, tXsA, tXrA);

      auto s2r_copy_b = make_tiled_copy_B(
          Copy_Atom<SM75_U32x4_LDSM_N, ElementIn>{}, tiled_mma);
      auto s2r_thr_copy_b = s2r_copy_b.get_slice(consumer_tid);
      auto tXsB = s2r_thr_copy_b.partition_S(sB_stg);
      auto tXrB = s2r_thr_copy_b.retile_D(tCrB);
      copy(s2r_copy_b, tXsB, tXrB);

      auto tCrC_view = make_tensor(
          tCrC.data(),
          thr_mma
              .partition_fragment_C(make_tensor<ElementAcc>(
                  Shape<Int<kBM>, Int<kBN>>{}))
              .layout());
      gemm(tiled_mma, tCrA, tCrB, tCrC_view);
    }
  }

  if (is_consumer) {
    if ((K_TILES % reduce_every_k) == 0) {
      uint32_t hash = pearl::xor_reduction(tCrC);
      int snapshot_idx = (K_TILES / reduce_every_k) - 1;
      int slot = snapshot_idx % kTranscriptSlots;
      transcript_local[slot] =
          pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
              transcript_local[slot], hash);
    }

    if (pow_target != nullptr && pow_key != nullptr &&
        host_signal_sync != nullptr && host_signal_header_pinned != nullptr) {
      Tensor transcript_rmem = make_tensor<uint32_t>(Int<kTranscriptSlots>{});
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kTranscriptSlots; ++s) {
        transcript_rmem(s) = transcript_local[s];
      }
      if (pearl::check_pow_target(transcript_rmem, pow_target, pow_key)) {
        auto block_coord =
            cute::make_tuple((int32_t)m_tile, (int32_t)n_tile, (int32_t)batch);
        auto problem_shape = cute::make_tuple(M, N, K, R);
        pearl::write_host_signal_header<ConsumerTiledMma, HeaderTileShape_MNK>(
            host_signal_sync, host_signal_header_pinned, problem_shape,
            block_coord, consumer_tid, pow_target);
      }
    }

    if (transcript != nullptr) {
      int64_t base = ((int64_t)batch * num_m_tiles + m_tile) * num_n_tiles +
                     n_tile;
      int64_t tx_off = base * (int64_t)kConsumerThreads * kTranscriptSlots +
                       (int64_t)consumer_tid * kTranscriptSlots;
      CUTLASS_PRAGMA_UNROLL
      for (int s = 0; s < kTranscriptSlots; ++s) {
        transcript[tx_off + s] = transcript_local[s];
      }
    }

    if (C_gmem != nullptr) {
      int64_t c_base = (int64_t)batch * M * N +
                       (int64_t)m_tile * kBM * (int64_t)N +
                       (int64_t)n_tile * kBN;
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kFragSize; ++j) {
        int m = get<0>(tCcD(j));
        int n = get<1>(tCcD(j));
        C_gmem[c_base + (int64_t)m * N + n] = tCrC(j);
      }
    }
  }
}

static void build_geforce_tma_descriptors(int8_t const* A, int8_t const* B,
                                          int Mi, int Ni, int Ki,
                                          ConsumerTmaA& tma_a,
                                          ConsumerTmaB& tma_b) {
  Tensor mA = make_tensor(
      make_gmem_ptr(const_cast<int8_t*>(A)),
      make_layout(make_shape(Mi, Ki), make_stride(Ki, _1{})));
  Tensor mB = make_tensor(
      make_gmem_ptr(const_cast<int8_t*>(B)),
      make_layout(make_shape(Ni, Ki), make_stride(Ki, _1{})));
  tma_a = make_tma_copy(SM90_TMA_LOAD{}, mA, ConsumerTmaFlatA{});
  tma_b = make_tma_copy(SM90_TMA_LOAD{}, mB, ConsumerTmaFlatB{});
}

static cudaError_t ensure_geforce_kernel_attrs(size_t smem_bytes) {
  static std::atomic<unsigned long long> attrs_set_mask{0};
  int dev = -1;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) dev = -1;
  const unsigned long long bit = dev >= 0 ? (1ull << dev) : 0ull;
  if (bit != 0ull &&
      (attrs_set_mask.load(std::memory_order_acquire) & bit) != 0ull) {
    return cudaSuccess;
  }
  if (smem_bytes > 48 * 1024) {
    cudaError_t err = cudaFuncSetAttribute(
        transcript_gemm_sm120_geforce_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (err != cudaSuccess) return err;
  }
  if (bit != 0ull) {
    attrs_set_mask.fetch_or(bit, std::memory_order_release);
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t launch_transcript_gemm_sm120_geforce(
    int8_t const* A, int8_t const* B, uint32_t* transcript, int64_t M,
    int64_t N, int64_t K, int64_t R, int64_t batch, cudaStream_t stream) {
  assert(M % kBM == 0 && N % kBN == 0 && K % kBK == 0 && R % kBK == 0);
  assert(K % R == 0);

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kBN), (unsigned)batch);
  dim3 block(kThreads);
  size_t smem_bytes = sizeof(SharedStorage);

  cudaError_t err = ensure_geforce_kernel_attrs(smem_bytes);
  if (err != cudaSuccess) return err;

  ConsumerTmaA tma_a;
  ConsumerTmaB tma_b;
  build_geforce_tma_descriptors(A, B, (int)M, (int)N, (int)K, tma_a, tma_b);

  transcript_gemm_sm120_geforce_kernel<<<grid, block, smem_bytes, stream>>>(
      A, B, nullptr, transcript, (int)M, (int)N, (int)K, (int)R, nullptr,
      nullptr, nullptr, nullptr, tma_a, tma_b);
  return cudaGetLastError();
}

cudaError_t launch_transcript_gemm_sm120_geforce_headless(
    int8_t const* A, int8_t const* B, int32_t* C, int64_t M, int64_t N,
    int64_t K, int64_t R, int64_t batch, uint32_t const* pow_target,
    uint32_t const* pow_key, HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned, cudaStream_t stream) {
  assert(M % kBM == 0 && N % kBN == 0 && K % kBK == 0 && R % kBK == 0);
  assert(K % R == 0);

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kBN), (unsigned)batch);
  dim3 block(kThreads);
  size_t smem_bytes = sizeof(SharedStorage);

  cudaError_t err = ensure_geforce_kernel_attrs(smem_bytes);
  if (err != cudaSuccess) return err;

  ConsumerTmaA tma_a;
  ConsumerTmaB tma_b;
  build_geforce_tma_descriptors(A, B, (int)M, (int)N, (int)K, tma_a, tma_b);

  transcript_gemm_sm120_geforce_kernel<<<grid, block, smem_bytes, stream>>>(
      A, B, C, nullptr, (int)M, (int)N, (int)K, (int)R, pow_target, pow_key,
      host_signal_sync, host_signal_header_pinned, tma_a, tma_b);
  return cudaGetLastError();
}

cudaError_t warmup_transcript_gemm_sm120_geforce_attrs() {
  return ensure_geforce_kernel_attrs(sizeof(SharedStorage));
}

}  // namespace blackwell
}  // namespace pearl

// ─── Verify harness (standalone build only) ─────────────────────────────────
#ifdef PEARL_SM120_GEFORCE_VERIFY_MAIN
#include <cstdio>
#include <cstring>
#include <cstdlib>

static constexpr int kVerifyTranscriptSlots = 16;

static int8_t prng_i8(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (int8_t)((int)(state % 127) - 63);
}

static int read_verify_trials() {
  const char* env = std::getenv("PEARL_GEFORCE_VERIFY_TRIALS");
  if (env == nullptr || env[0] == '\0') return 100;
  char* end = nullptr;
  long v = std::strtol(env, &end, 10);
  if (end == env || v < 1) return 100;
  return (int)v;
}

// Byte-identity sweep — `trials` fresh-random A/B, consumer reference vs the
// sm120_geforce launcher, memcmp the whole multi-tile transcript.
static int verify_shape(int M, int N, int K, int R, int trials) {
  const int batch = 1;
  const size_t a_elems = (size_t)M * K;
  const size_t b_elems = (size_t)N * K;
  const int64_t tr_elems =
      pearl::portable::transcript_buffer_elems(M, N, batch);

  int8_t* hA = (int8_t*)std::malloc(a_elems);
  int8_t* hB = (int8_t*)std::malloc(b_elems);
  int8_t *dA = nullptr, *dB = nullptr;
  uint32_t *dTr_ref = nullptr, *dTr_geforce = nullptr;
  cudaMalloc(&dA, a_elems);
  cudaMalloc(&dB, b_elems);
  cudaMalloc(&dTr_ref, (size_t)tr_elems * sizeof(uint32_t));
  cudaMalloc(&dTr_geforce, (size_t)tr_elems * sizeof(uint32_t));
  uint32_t* hr = (uint32_t*)std::malloc((size_t)tr_elems * sizeof(uint32_t));
  uint32_t* hg = (uint32_t*)std::malloc((size_t)tr_elems * sizeof(uint32_t));

  int fails = 0;
  for (int t = 0; t < trials; ++t) {
    uint32_t s = 0xC0FFEEu + (uint32_t)t * 2654435761u +
                 (uint32_t)M * 2246822519u + (uint32_t)N;
    for (size_t i = 0; i < a_elems; ++i) hA[i] = prng_i8(s);
    for (size_t i = 0; i < b_elems; ++i) hB[i] = prng_i8(s);
    cudaMemcpy(dA, hA, a_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, b_elems, cudaMemcpyHostToDevice);

    pearl::consumer::launch_transcript_gemm(
        dA, dB, nullptr, dTr_ref, M, N, K, R, batch, 0);
    pearl::blackwell::launch_transcript_gemm_sm120_geforce(
        dA, dB, dTr_geforce, M, N, K, R, batch, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::printf("  KERNEL ERROR (M=%d N=%d trial %d): %s\n", M, N, t,
                  cudaGetErrorString(err));
      fails = trials;
      break;
    }
    cudaMemcpy(hr, dTr_ref, (size_t)tr_elems * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hg, dTr_geforce, (size_t)tr_elems * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    if (std::memcmp(hr, hg, (size_t)tr_elems * sizeof(uint32_t)) != 0) {
      if (fails == 0) {
        int shown = 0;
        for (int64_t i = 0; i < tr_elems && shown < 6; ++i) {
          if (hr[i] == hg[i]) continue;
          int gt = (int)(i / (256 * kVerifyTranscriptSlots));
          int rem = (int)(i % (256 * kVerifyTranscriptSlots));
          std::printf(
              "    diff gt=%d tid=%d slot=%d: consumer=0x%08x geforce=0x%08x\n",
              gt, rem / kVerifyTranscriptSlots, rem % kVerifyTranscriptSlots,
              hr[i], hg[i]);
          shown++;
        }
      }
      fails++;
    }
  }

  std::free(hA);
  std::free(hB);
  std::free(hr);
  std::free(hg);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dTr_ref);
  cudaFree(dTr_geforce);
  return fails;
}

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  const int trials = read_verify_trials();
  std::printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("Trials per shape: %d (override: PEARL_GEFORCE_VERIFY_TRIALS)\n\n",
              trials);

  int fail = 0;
  struct ShapeSpec {
    int M, N, K, R;
  };
  const ShapeSpec shapes[] = {
      {2048, 4096, 128, 128},
      {8192, 32768, 128, 128},
  };

  for (const auto& shape : shapes) {
    int f = verify_shape(shape.M, shape.N, shape.K, shape.R, trials);
    std::printf("[verify M=%d N=%d K=%d R=%d]  byte-identity: %d/%d PASS%s\n",
                shape.M, shape.N, shape.K, shape.R, trials - f, trials,
                f ? "   *** FAIL ***" : "");
    if (f) fail = 1;
  }

  std::printf("\n%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
  return fail;
}
#endif  // PEARL_SM120_GEFORCE_VERIFY_MAIN
