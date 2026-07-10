// RTX 5090 (sm_120a) GeForce transcript GEMM v2.
//
// Phase-1 improvements over v1 (transcript_gemm_sm120_geforce.cu):
//   - CUTLASS PipelineTmaAsync (full + empty barriers) instead of manual
//     ClusterTransactionBarrier-only pipeline
//   - Warp-specialized producer/consumer loops (no CTA __syncthreads per K-tile)
//   - TMA descriptor prefetch at kernel entry
//   - Host-side TMA descriptor cache keyed by (device, A, B, M, N, K)
//
// Proof contract: byte-identical transcript + headless PoW vs consumer and v1.
// Validate: scripts/verify_geforce_transcript.sh
//
// Compile: PEARL_GEMM_BLACKWELL_GEFORCE_V2=1 (default ON for blackwell).
// Runtime: default geforce_v2 when compiled in; PEARL_GEMM_KERNEL=geforce_v1 or consumer to opt out.

#include <vector>
#include <cstdint>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <atomic>

#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm80.h>
#include <cutlass/pipeline/pipeline.hpp>

#include "../gemm/pow_utils.hpp"
#include "../portable/transcript_kernel.cuh"
#include "../consumer/tma_tile_loader.cuh"
#include "geforce_tma_pipeline.cuh"
#include "transcript_gemm_sm120_geforce_v2.h"

namespace pearl {
namespace blackwell {
namespace {

using namespace cute;
using namespace pearl::consumer::tma_loader;
using pearl::blackwell::geforce_v2::TmaAsyncPipeline;
using pearl::blackwell::geforce_v2::tma_copy_k_tile;

static constexpr int kBM = 128;
static constexpr int kBN = 256;
#ifndef PEARL_CONSUMER_KBLOCK
#define PEARL_CONSUMER_KBLOCK 64
#endif
#if PEARL_CONSUMER_KBLOCK != 64 && PEARL_CONSUMER_KBLOCK != 128
#error "PEARL_CONSUMER_KBLOCK must be 64 or 128"
#endif
static constexpr int kBK = PEARL_CONSUMER_KBLOCK;
static constexpr int kAtomK = 32;
static constexpr int kConsumerThreads = 256;
static constexpr int kProducerLeader = kConsumerThreads;
// 256 consumers + 1 producer — do NOT pad to 288; extra threads that exit
// early while the pipeline is still running cause illegal memory access on sm_120.
static constexpr int kThreads = kConsumerThreads + 1;
static constexpr int kFragSize = (kBM * kBN) / kConsumerThreads;
static constexpr int kTranscriptSlots = 16;
#ifndef PEARL_CONSUMER_STAGES
#define PEARL_CONSUMER_STAGES 2
#endif
static constexpr int kStages = PEARL_CONSUMER_STAGES;

#ifndef PEARL_CONSUMER_SWIZZLE_BITS
#define PEARL_CONSUMER_SWIZZLE_BITS 2
#endif

// SM120 per-block SMEM cap (99 KiB + margin per research/headroom docs).
static constexpr size_t kMaxSmemBytes = 101376;

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

using MainloopPipeline = TmaAsyncPipeline<kStages>;

static_assert(cute::cosize_v<SmemLayoutA> == cute::cosize_v<ConsumerTmaStagedA>);
static_assert(cute::cosize_v<SmemLayoutB> == cute::cosize_v<ConsumerTmaStagedB>);

struct SharedStorage {
  alignas(128) ElementIn smem_A[cute::cosize_v<SmemLayoutA>];
  alignas(128) ElementIn smem_B[cute::cosize_v<SmemLayoutB>];
  typename MainloopPipeline::SharedStorage pipeline_storage;
};

static_assert(sizeof(SharedStorage) <= kMaxSmemBytes,
              "GeForce v2 SharedStorage exceeds SM120 per-block SMEM cap");

static constexpr uint32_t kTmaTransactionBytes =
    kTmaBytes<kBM, kBN, kBK, ElementIn>;

__launch_bounds__(kThreads, 1)
__global__ void transcript_gemm_sm120_geforce_v2_kernel(
    ElementIn const* __restrict__ A_gmem,
    ElementIn const* __restrict__ B_gmem,
    ElementAcc* __restrict__ C_gmem,
    uint32_t* __restrict__ transcript,
    int M, int N, int K, int R,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    uint32_t const* const* __restrict__ grouped_pow_key_ptrs,
    HostSignalSync* grouped_sync_array,
    HostSignalHeader** grouped_header_ptrs,
    ConsumerTmaA const* __restrict__ tma_a_group,
    ConsumerTmaA const* __restrict__ /*d_tma_a*/,
    ConsumerTmaB const* __restrict__ /*d_tma_b*/,
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

  // sm_120 GeForce requires __grid_constant__ TMA descriptors for all launches.
  // Device-resident d_tma_* is legacy and faults when dereferenced from gmem.
  ConsumerTmaA tma_a_eff = tma_a;
  ConsumerTmaB tma_b_eff = tma_b;
  if (tma_a_group != nullptr) {
    tma_a_eff = tma_a_group[batch];
  }
  uint32_t const* pow_key_eff = pow_key;
  HostSignalSync* host_signal_sync_eff = host_signal_sync;
  HostSignalHeader* host_signal_header_eff = host_signal_header_pinned;
  if (grouped_pow_key_ptrs != nullptr) {
    pow_key_eff = grouped_pow_key_ptrs[batch];
    host_signal_sync_eff = &grouped_sync_array[batch];
    host_signal_header_eff = grouped_header_ptrs[batch];
  }

  ConsumerTiledMma tiled_mma;
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

  // TMA copies require gmem views from the grid-constant descriptor, not raw
  // pointers (see transcript_gemm_sm100.cu and geforce v1).
  Tensor mA = tma_a_eff.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b_eff.get_tma_tensor(make_shape(N, K));
  Tensor gA = local_tile(mA, Shape<Int<kBM>, Int<kBK>>{},
                         make_coord(m_tile, _));
  Tensor gB = local_tile(mB, Shape<Int<kBN>, Int<kBK>>{},
                         make_coord(n_tile, _));

  const int K_TILES = K / kBK;
  const int reduce_every_k = R / kBK;

  using PipelineParams = typename MainloopPipeline::Params;
  PipelineParams pipe_params;
  pipe_params.transaction_bytes = kTmaTransactionBytes;
  pipe_params.num_consumers = kConsumerThreads;
  pipe_params.num_producers = 1;
  if (tid == kProducerLeader) {
    pipe_params.role = MainloopPipeline::ThreadCategory::Producer;
    pipe_params.is_leader = 1;
  } else if (is_consumer) {
    pipe_params.role = MainloopPipeline::ThreadCategory::Consumer;
    pipe_params.is_leader = 0;
  } else {
    pipe_params.role = MainloopPipeline::ThreadCategory::NonParticipant;
    pipe_params.is_leader = 0;
  }

  MainloopPipeline pipeline(smem.pipeline_storage, pipe_params,
                            cute::Shape<_1, _1, _1>{});
  __syncthreads();

  if (tid == kProducerLeader) {
    cute::prefetch_tma_descriptor(tma_a_eff.get_tma_descriptor());
    cute::prefetch_tma_descriptor(tma_b_eff.get_tma_descriptor());
  }

  if (tid == kProducerLeader) {
    typename MainloopPipeline::PipelineState pipe_write =
        cutlass::make_producer_start_state<MainloopPipeline>();

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
      pipeline.producer_acquire(pipe_write);
      tma_copy_k_tile<kStages, MainloopPipeline, ConsumerTmaA, ConsumerTmaB,
                      SmemLayoutA, SmemLayoutB, kBM, kBN, kBK, ElementIn>(
          pipeline, pipe_write, tma_a_eff, tma_b_eff, smem.smem_A, smem.smem_B,
          gA, gB, k_iter);
      pipeline.producer_commit(pipe_write, kTmaTransactionBytes);
      ++pipe_write;
    }
    pipeline.producer_tail(pipe_write);
  }

  if (is_consumer) {
    typename MainloopPipeline::PipelineState pipe_read;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
      pipeline.consumer_wait(pipe_read);
      const int stg = pipe_read.index();

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

      pipeline.consumer_release(pipe_read);
      ++pipe_read;
    }

    if ((K_TILES % reduce_every_k) == 0) {
      uint32_t hash = pearl::xor_reduction(tCrC);
      int snapshot_idx = (K_TILES / reduce_every_k) - 1;
      int slot = snapshot_idx % kTranscriptSlots;
      transcript_local[slot] =
          pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
              transcript_local[slot], hash);
    }

    // ── Write final transcript to gmem for separate finalize kernel ─────
    // The transcript is always written here; BLAKE3 compress + target check
    // are performed by the separate transcript_finalize_kernel, which reads
    // this buffer and writes HostSignalHeader on a hit.
    {
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

struct GeforceV2TmaCache {
  std::mutex mu;
  int device = -1;
  int8_t const* A = nullptr;
  int8_t const* B = nullptr;
  int M = 0;
  int N = 0;
  int K = 0;
  ConsumerTmaA tma_a{};
  ConsumerTmaB tma_b{};
  bool valid = false;
};

static GeforceV2TmaCache g_v2_tma_cache;

// Device-resident TMA descriptors for graph-safe replay (sm_120 GeForce).
struct V2DeviceTmaPool {
  std::mutex mu;
  int device = -1;
  ConsumerTmaA* d_tma_a = nullptr;
  ConsumerTmaB* d_tma_b = nullptr;
  ConsumerTmaA* h_tma_a = nullptr;  // pinned host staging
  ConsumerTmaB* h_tma_b = nullptr;
};

static V2DeviceTmaPool g_v2_dev_tma;

static cudaError_t ensure_v2_device_tma_pool(int dev) {
  std::lock_guard<std::mutex> lock(g_v2_dev_tma.mu);
  if (g_v2_dev_tma.device == dev && g_v2_dev_tma.d_tma_a != nullptr) {
    return cudaSuccess;
  }
  if (g_v2_dev_tma.d_tma_a) cudaFree(g_v2_dev_tma.d_tma_a);
  if (g_v2_dev_tma.d_tma_b) cudaFree(g_v2_dev_tma.d_tma_b);
  if (g_v2_dev_tma.h_tma_a) cudaFreeHost(g_v2_dev_tma.h_tma_a);
  if (g_v2_dev_tma.h_tma_b) cudaFreeHost(g_v2_dev_tma.h_tma_b);
  g_v2_dev_tma.device = -1;
  g_v2_dev_tma.d_tma_a = nullptr;
  g_v2_dev_tma.d_tma_b = nullptr;
  g_v2_dev_tma.h_tma_a = nullptr;
  g_v2_dev_tma.h_tma_b = nullptr;
  g_v2_dev_tma.device = dev;
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&g_v2_dev_tma.d_tma_a),
                               sizeof(ConsumerTmaA));
  if (err != cudaSuccess) return err;
  err = cudaMalloc(reinterpret_cast<void**>(&g_v2_dev_tma.d_tma_b),
                   sizeof(ConsumerTmaB));
  if (err != cudaSuccess) return err;
  err = cudaHostAlloc(reinterpret_cast<void**>(&g_v2_dev_tma.h_tma_a),
                      sizeof(ConsumerTmaA), cudaHostAllocPortable);
  if (err != cudaSuccess) return err;
  err = cudaHostAlloc(reinterpret_cast<void**>(&g_v2_dev_tma.h_tma_b),
                      sizeof(ConsumerTmaB), cudaHostAllocPortable);
  return err;
}

static void build_v2_tma_descriptors(int8_t const* A, int8_t const* B, int Mi,
                                     int Ni, int Ki, ConsumerTmaA& tma_a,
                                     ConsumerTmaB& tma_b);
static void get_v2_tma_descriptors(int8_t const* A, int8_t const* B, int Mi,
                                   int Ni, int Ki, ConsumerTmaA& tma_a,
                                   ConsumerTmaB& tma_b);

static cudaError_t upload_v2_tma_to_device(int8_t const* A, int8_t const* B,
                                           int Mi, int Ni, int Ki,
                                           ConsumerTmaA const** out_d_a,
                                           ConsumerTmaB const** out_d_b,
                                           cudaStream_t stream) {
  int dev = 0;
  cudaGetDevice(&dev);
  cudaError_t err = ensure_v2_device_tma_pool(dev);
  if (err != cudaSuccess) return err;

  ConsumerTmaA tma_a;
  ConsumerTmaB tma_b;
  get_v2_tma_descriptors(A, B, Mi, Ni, Ki, tma_a, tma_b);

  {
    std::lock_guard<std::mutex> lock(g_v2_dev_tma.mu);
    *g_v2_dev_tma.h_tma_a = tma_a;
    *g_v2_dev_tma.h_tma_b = tma_b;
    err = cudaMemcpyAsync(g_v2_dev_tma.d_tma_a, g_v2_dev_tma.h_tma_a,
                          sizeof(ConsumerTmaA), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    err = cudaMemcpyAsync(g_v2_dev_tma.d_tma_b, g_v2_dev_tma.h_tma_b,
                          sizeof(ConsumerTmaB), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return err;
    *out_d_a = g_v2_dev_tma.d_tma_a;
    *out_d_b = g_v2_dev_tma.d_tma_b;
  }
  return cudaSuccess;
}

static void build_v2_tma_descriptors(int8_t const* A, int8_t const* B, int Mi,
                                     int Ni, int Ki, ConsumerTmaA& tma_a,
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

static void get_v2_tma_descriptors(int8_t const* A, int8_t const* B, int Mi,
                                   int Ni, int Ki, ConsumerTmaA& tma_a,
                                   ConsumerTmaB& tma_b) {
  int dev = 0;
  cudaGetDevice(&dev);

  std::lock_guard<std::mutex> lock(g_v2_tma_cache.mu);
  if (g_v2_tma_cache.valid && g_v2_tma_cache.device == dev &&
      g_v2_tma_cache.A == A && g_v2_tma_cache.B == B &&
      g_v2_tma_cache.M == Mi && g_v2_tma_cache.N == Ni &&
      g_v2_tma_cache.K == Ki) {
    tma_a = g_v2_tma_cache.tma_a;
    tma_b = g_v2_tma_cache.tma_b;
    return;
  }

  build_v2_tma_descriptors(A, B, Mi, Ni, Ki, tma_a, tma_b);
  g_v2_tma_cache.device = dev;
  g_v2_tma_cache.A = A;
  g_v2_tma_cache.B = B;
  g_v2_tma_cache.M = Mi;
  g_v2_tma_cache.N = Ni;
  g_v2_tma_cache.K = Ki;
  g_v2_tma_cache.tma_a = tma_a;
  g_v2_tma_cache.tma_b = tma_b;
  g_v2_tma_cache.valid = true;
}

static cudaError_t ensure_geforce_v2_kernel_attrs(size_t smem_bytes) {
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
        transcript_gemm_sm120_geforce_v2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    if (err != cudaSuccess) return err;
  }
  if (bit != 0ull) {
    attrs_set_mask.fetch_or(bit, std::memory_order_release);
  }
  return cudaSuccess;
}

static cudaError_t launch_v2_impl(int8_t const* A, int8_t const* B,
                                  ElementAcc* C, uint32_t* transcript, int64_t M,
                                  int64_t N, int64_t K, int64_t R,
                                  int64_t batch,
                                  uint32_t const* pow_target,
                                  uint32_t const* pow_key,
                                  HostSignalSync* host_signal_sync,
                                  HostSignalHeader* host_signal_header_pinned,
                                  uint32_t const* const* grouped_pow_key_ptrs,
                                  HostSignalSync* grouped_sync_array,
                                  HostSignalHeader** grouped_header_ptrs,
                                  ConsumerTmaA const* tma_a_group,
                                  cudaStream_t stream) {
  assert(M % kBM == 0 && N % kBN == 0 && K % kBK == 0 && R % kBK == 0);
  assert(K % R == 0);

  dim3 grid((unsigned)(M / kBM), (unsigned)(N / kBN), (unsigned)batch);
  dim3 block(kThreads);
  size_t smem_bytes = sizeof(SharedStorage);

  cudaError_t err = ensure_geforce_v2_kernel_attrs(smem_bytes);
  if (err != cudaSuccess) return err;

  ConsumerTmaA tma_a{};
  ConsumerTmaB tma_b{};
  cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
  (void)cudaStreamIsCapturing(stream, &cap_status);
  const bool in_graph_capture = (cap_status == cudaStreamCaptureStatusActive);

  ConsumerTmaA const* d_tma_a = nullptr;
  ConsumerTmaB const* d_tma_b = nullptr;
  if (tma_a_group == nullptr) {
    if (in_graph_capture) {
      // Graph replay on sm_120 must use __grid_constant__ TMA only. Passing
      // device-resident descriptor pointers captures their values in the graph
      // but replay still dereferences gmem outside the graph snapshot → illegal
      // access. Pre-upload via prepare_geforce_v2_tma_for_graph() fills h_tma_*;
      // copy those bytes into grid_constant params (frozen for replay).
      std::lock_guard<std::mutex> lock(g_v2_dev_tma.mu);
      if (g_v2_dev_tma.h_tma_a == nullptr || g_v2_dev_tma.h_tma_b == nullptr) {
        return cudaErrorInvalidConfiguration;
      }
      tma_a = *g_v2_dev_tma.h_tma_a;
      tma_b = *g_v2_dev_tma.h_tma_b;
      d_tma_a = nullptr;
      d_tma_b = nullptr;
    } else {
      // Direct iter_batch launch: pass descriptors as __grid_constant__ only.
      // upload_v2_tma_to_device + gmem deref faults on sm_120 GeForce.
      get_v2_tma_descriptors(A, B, (int)M, (int)N, (int)K, tma_a, tma_b);
      d_tma_a = nullptr;
      d_tma_b = nullptr;
    }
  } else {
    get_v2_tma_descriptors(A, B, (int)M, (int)N, (int)K, tma_a, tma_b);
  }

  transcript_gemm_sm120_geforce_v2_kernel<<<grid, block, smem_bytes, stream>>>(
      A, B, C, transcript, (int)M, (int)N, (int)K, (int)R, pow_target, pow_key,
      host_signal_sync, host_signal_header_pinned, grouped_pow_key_ptrs,
      grouped_sync_array, grouped_header_ptrs, tma_a_group, d_tma_a, d_tma_b,
      tma_a, tma_b);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_transcript_gemm_sm120_geforce_v2(
    int8_t const* A, int8_t const* B, uint32_t* transcript, int64_t M,
    int64_t N, int64_t K, int64_t R, int64_t batch, cudaStream_t stream) {
  return launch_v2_impl(A, B, nullptr, transcript, M, N, K, R, batch, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, stream);
}

cudaError_t launch_transcript_gemm_sm120_geforce_v2_headless(
    int8_t const* A, int8_t const* B, int32_t* C, uint32_t* transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned, cudaStream_t stream) {
  return launch_v2_impl(A, B, C, transcript, M, N, K, R, batch, pow_target,
                        pow_key, host_signal_sync, host_signal_header_pinned,
                        nullptr, nullptr, nullptr, nullptr, stream);
}

cudaError_t launch_transcript_gemm_sm120_geforce_v2_headless_grouped(
    int8_t const* const* h_apea_ptrs, int8_t const* B, int64_t M, int64_t N,
    int64_t K, int64_t R, int64_t batch, uint32_t const* pow_target,
    uint32_t const* const* d_pow_key_ptrs, HostSignalSync* d_sync_array,
    HostSignalHeader** d_header_ptrs,
    uint32_t* transcript,
    cudaStream_t stream) {
  assert(h_apea_ptrs != nullptr);
  assert(d_pow_key_ptrs != nullptr);
  assert(d_sync_array != nullptr);
  assert(d_header_ptrs != nullptr);

  std::vector<ConsumerTmaA> tma_a_host((size_t)batch);
  ConsumerTmaB tma_b;
  build_v2_tma_descriptors(h_apea_ptrs[0], B, (int)M, (int)N, (int)K,
                           tma_a_host[0], tma_b);
  for (int64_t g = 1; g < batch; ++g) {
    build_v2_tma_descriptors(h_apea_ptrs[g], B, (int)M, (int)N, (int)K,
                             tma_a_host[(size_t)g], tma_b);
  }

  ConsumerTmaA* d_tma_a_group = nullptr;
  cudaError_t err = cudaMallocAsync(
      reinterpret_cast<void**>(&d_tma_a_group),
      sizeof(ConsumerTmaA) * (size_t)batch, stream);
  if (err != cudaSuccess) return err;
  err = cudaMemcpyAsync(d_tma_a_group, tma_a_host.data(),
                        sizeof(ConsumerTmaA) * (size_t)batch,
                        cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    cudaFreeAsync(d_tma_a_group, stream);
    return err;
  }

  err = launch_v2_impl(h_apea_ptrs[0], B, nullptr, transcript, M, N, K, R, batch,
                       nullptr, nullptr, nullptr, nullptr, d_pow_key_ptrs,
                       d_sync_array, d_header_ptrs, d_tma_a_group, stream);
  cudaFreeAsync(d_tma_a_group, stream);
  return err;
}

cudaError_t warmup_transcript_gemm_sm120_geforce_v2_attrs() {
  return ensure_geforce_v2_kernel_attrs(sizeof(SharedStorage));
}

cudaError_t prepare_geforce_v2_tma_for_graph(int8_t const* A, int8_t const* B,
                                             int M, int N, int K,
                                             cudaStream_t stream) {
  ConsumerTmaA const* d_a = nullptr;
  ConsumerTmaB const* d_b = nullptr;
  return upload_v2_tma_to_device(A, B, M, N, K, &d_a, &d_b, stream);
}

}  // namespace blackwell
}  // namespace pearl

// ─── Verify harness (standalone build only) ─────────────────────────────────
#ifdef PEARL_SM120_GEFORCE_V2_VERIFY_MAIN
#include <cstdio>

#include "transcript_gemm_sm120_geforce.h"

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

static int verify_shape_pair(int M, int N, int K, int R, int trials,
                             const char* ref_label,
                             cudaError_t (*ref_launch)(int8_t const*, int8_t const*,
                                                       uint32_t*, int64_t, int64_t,
                                                       int64_t, int64_t, int64_t,
                                                       cudaStream_t),
                             const char* cand_label,
                             cudaError_t (*cand_launch)(int8_t const*, int8_t const*,
                                                        uint32_t*, int64_t, int64_t,
                                                        int64_t, int64_t, int64_t,
                                                        cudaStream_t)) {
  const int batch = 1;
  const size_t a_elems = (size_t)M * K;
  const size_t b_elems = (size_t)N * K;
  const int64_t tr_elems =
      pearl::portable::transcript_buffer_elems(M, N, batch);

  int8_t* hA = (int8_t*)std::malloc(a_elems);
  int8_t* hB = (int8_t*)std::malloc(b_elems);
  int8_t *dA = nullptr, *dB = nullptr;
  uint32_t *dTr_ref = nullptr, *dTr_cand = nullptr;
  cudaMalloc(&dA, a_elems);
  cudaMalloc(&dB, b_elems);
  cudaMalloc(&dTr_ref, (size_t)tr_elems * sizeof(uint32_t));
  cudaMalloc(&dTr_cand, (size_t)tr_elems * sizeof(uint32_t));
  uint32_t* hr = (uint32_t*)std::malloc((size_t)tr_elems * sizeof(uint32_t));
  uint32_t* hc = (uint32_t*)std::malloc((size_t)tr_elems * sizeof(uint32_t));

  int fails = 0;
  for (int t = 0; t < trials; ++t) {
    uint32_t s = 0xC0FFEEu + (uint32_t)t * 2654435761u +
                 (uint32_t)M * 2246822519u + (uint32_t)N;
    for (size_t i = 0; i < a_elems; ++i) hA[i] = prng_i8(s);
    for (size_t i = 0; i < b_elems; ++i) hB[i] = prng_i8(s);
    cudaMemcpy(dA, hA, a_elems, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, b_elems, cudaMemcpyHostToDevice);

    ref_launch(dA, dB, dTr_ref, M, N, K, R, batch, 0);
    cand_launch(dA, dB, dTr_cand, M, N, K, R, batch, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::printf("  KERNEL ERROR (M=%d N=%d trial %d): %s\n", M, N, t,
                  cudaGetErrorString(err));
      fails = trials;
      break;
    }
    cudaMemcpy(hr, dTr_ref, (size_t)tr_elems * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hc, dTr_cand, (size_t)tr_elems * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    if (std::memcmp(hr, hc, (size_t)tr_elems * sizeof(uint32_t)) != 0) {
      if (fails == 0) {
        int shown = 0;
        for (int64_t i = 0; i < tr_elems && shown < 6; ++i) {
          if (hr[i] == hc[i]) continue;
          int gt = (int)(i / (256 * kVerifyTranscriptSlots));
          int rem = (int)(i % (256 * kVerifyTranscriptSlots));
          std::printf(
              "    diff gt=%d tid=%d slot=%d: %s=0x%08x %s=0x%08x\n", gt,
              rem / kVerifyTranscriptSlots, rem % kVerifyTranscriptSlots,
              ref_label, hr[i], cand_label, hc[i]);
          shown++;
        }
      }
      fails++;
    }
  }

  std::free(hA);
  std::free(hB);
  std::free(hr);
  std::free(hc);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dTr_ref);
  cudaFree(dTr_cand);
  return fails;
}

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  const int trials = read_verify_trials();
  std::printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("GeForce v2 verify — trials per shape: %d\n\n", trials);

  int fail = 0;
  struct ShapeSpec {
    int M, N, K, R;
  };
  const ShapeSpec shapes[] = {
      {2048, 4096, 128, 128},
      {8192, 32768, 128, 128},
      {8192, 32768, 4096, 128},
  };

  for (const auto& shape : shapes) {
    int f1 = verify_shape_pair(
        shape.M, shape.N, shape.K, shape.R, trials, "consumer",
        pearl::consumer::launch_transcript_gemm, "geforce_v2",
        pearl::blackwell::launch_transcript_gemm_sm120_geforce_v2);
    std::printf("[v2 vs consumer M=%d N=%d] %d/%d PASS%s\n", shape.M, shape.N,
                trials - f1, trials, f1 ? " *** FAIL ***" : "");
    if (f1) fail = 1;

    int f2 = verify_shape_pair(
        shape.M, shape.N, shape.K, shape.R, trials, "geforce_v1",
        pearl::blackwell::launch_transcript_gemm_sm120_geforce, "geforce_v2",
        pearl::blackwell::launch_transcript_gemm_sm120_geforce_v2);
    std::printf("[v2 vs v1     M=%d N=%d] %d/%d PASS%s\n", shape.M, shape.N,
                trials - f2, trials, f2 ? " *** FAIL ***" : "");
    if (f2) fail = 1;
  }

  std::printf("\n%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
  return fail;
}
#endif  // PEARL_SM120_GEFORCE_V2_VERIFY_MAIN
