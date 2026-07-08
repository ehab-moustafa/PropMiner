// Blackwell consumer TMA gmem→smem tile loader for transcript_gemm_kernel_consumer.
//
// Only compiled when PEARL_CONSUMER_USE_TMA_EXPERIMENT=1 (Makefile
// PEARL_GEMM_BLACKWELL_LOAD_POLICY=tma).  The cp.async baseline is unchanged
// in the #else branch of transcript_gemm_kernel.cu.
//
// TMA box shape matches the swizzled single-stage smem layout (SmemLayoutFlat*)
// used by ldmatrix; partition_D targets the multi-stage SmemLayoutA/B pipe index.
// Pattern follows transcript_gemm_sm100.cu load_stage but for a single-stream
// SM80 mma.sync consumer CTA (256 threads, thread 0 TMA issue).

#pragma once

#include <cstdint>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace pearl {
namespace consumer {
namespace tma_loader {

using namespace cute;

// Flat single-stage swizzled view — the TMA box descriptor shape.
template <typename SmemLayoutAtomA, typename SmemLayoutAtomB, int kBM, int kBN,
          int kBK>
using SmemLayoutFlatA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}), Step<_1, _2>{}));
template <typename SmemLayoutAtomA, typename SmemLayoutAtomB, int kBM, int kBN,
          int kBK>
using SmemLayoutFlatB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBN>{}, Int<kBK>{}), Step<_1, _2>{}));

template <typename SmemLayoutAtomA, int kBM, int kBK, int kStages>
using SmemLayoutStagedA = decltype(tile_to_shape(
    SmemLayoutAtomA{}, make_shape(Int<kBM>{}, Int<kBK>{}, Int<kStages>{}),
    Step<_1, _2, _3>{}));
template <typename SmemLayoutAtomB, int kBN, int kBK, int kStages>
using SmemLayoutStagedB = decltype(tile_to_shape(
    SmemLayoutAtomB{}, make_shape(Int<kBN>{}, Int<kBK>{}, Int<kStages>{}),
    Step<_1, _2, _3>{}));

template <typename ElementIn, typename SmemLayoutFlatA, typename SmemLayoutFlatB>
using GmemLayout2D = decltype(make_layout(
    make_shape(int(0), int(0)), make_stride(int(0), _1{})));
template <typename ElementIn, typename SmemLayoutFlatA, typename SmemLayoutFlatB>
using GmemTensor2D = decltype(make_tensor(
    make_gmem_ptr<ElementIn>(nullptr), GmemLayout2D<ElementIn, SmemLayoutFlatA,
                                                     SmemLayoutFlatB>{}));

template <typename ElementIn, typename SmemLayoutFlatA, typename SmemLayoutFlatB>
using TmaA = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    GmemTensor2D<ElementIn, SmemLayoutFlatA, SmemLayoutFlatB>{},
    SmemLayoutFlatA{}));
template <typename ElementIn, typename SmemLayoutFlatA, typename SmemLayoutFlatB>
using TmaB = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    GmemTensor2D<ElementIn, SmemLayoutFlatA, SmemLayoutFlatB>{},
    SmemLayoutFlatB{}));

template <int kBM, int kBN, int kBK, typename ElementIn>
static constexpr uint32_t kTmaBytes =
    (uint32_t)(kBM * kBK + kBN * kBK) * (uint32_t)sizeof(ElementIn);

using BarrierValue =
    cutlass::arch::ClusterTransactionBarrier::ValueType;

// Per-stage mbarrier: TMA completion for one pipeline stage.
template <int kStages>
struct TmaPipelineStorage {
  alignas(16) BarrierValue full_barrier[kStages];
};

template <int kStages>
CUTLASS_DEVICE void init_tma_barriers(TmaPipelineStorage<kStages>& pipe,
                                      int tid) {
  if (tid == 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < kStages; ++s) {
      cutlass::arch::ClusterTransactionBarrier::init(&pipe.full_barrier[s], 1);
    }
  }
}

// Wait until stage `stg` has landed (replaces cp.async.wait_group).
// Phase alternates each time a pipeline stage is reused (kStages-ring).
template <int kStages>
CUTLASS_DEVICE void tma_wait_stage(TmaPipelineStorage<kStages>& pipe, int stg,
                                   int k_iter) {
  const uint32_t phase = ((unsigned)k_iter / (unsigned)kStages) & 1u;
  cutlass::arch::ClusterTransactionBarrier::wait(&pipe.full_barrier[stg], phase);
}

// Issue one K-tile TMA load into pipeline stage `stg` (elect_one producer).
template <typename TmaA, typename TmaB, typename SmemLayoutStagedA,
          typename SmemLayoutStagedB, int kBM, int kBN, int kBK, int kStages,
          typename TensorGA, typename TensorGB>
CUTLASS_DEVICE void tma_issue_k_tile(
    TmaA const& tma_a, TmaB const& tma_b,
    TmaPipelineStorage<kStages>& pipe,
    ElementIn* smem_a_base, ElementIn* smem_b_base,
    TensorGA const& gA, TensorGB const& gB,
    int k_iter, int stg, int leader_thread = 0) {
  if (threadIdx.x != leader_thread) return;

  BarrierValue* fb = &pipe.full_barrier[stg];
  cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
      fb, kTmaBytes<kBM, kBN, kBK, int8_t>);

  auto cta_tma_a = tma_a.get_slice(_0{});
  auto cta_tma_b = tma_b.get_slice(_0{});

  Tensor sA_staged =
      make_tensor(make_smem_ptr(smem_a_base), SmemLayoutStagedA{});
  Tensor sB_staged =
      make_tensor(make_smem_ptr(smem_b_base), SmemLayoutStagedB{});

  Tensor tAgA = cta_tma_a.partition_S(gA);
  Tensor tBgB = cta_tma_b.partition_S(gB);
  Tensor tAsA = cta_tma_a.partition_D(sA_staged);
  Tensor tBsB = cta_tma_b.partition_D(sB_staged);

  copy(tma_a.with(*fb), tAgA(_, _, _, k_iter), tAsA(_, _, _, stg));
  copy(tma_b.with(*fb), tBgB(_, _, _, k_iter), tBsB(_, _, _, stg));
}

}  // namespace tma_loader
}  // namespace consumer
}  // namespace pearl
