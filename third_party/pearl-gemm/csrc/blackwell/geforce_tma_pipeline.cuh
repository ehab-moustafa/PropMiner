// GeForce v2 TMA pipeline helpers — CUTLASS PipelineTmaAsync for warp-specialized
// transcript GEMM. Proof contract unchanged; only load/pipeline layer differs from v1.
#pragma once

#include <cstdint>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/pipeline/pipeline.hpp>

#include "../consumer/tma_tile_loader.cuh"

namespace pearl {
namespace blackwell {
namespace geforce_v2 {

using namespace cute;
using pearl::consumer::tma_loader::kTmaBytes;

template <int kStages>
using TmaAsyncPipeline = cutlass::PipelineTmaAsync<kStages>;

// Issue one K-tile TMA load into the stage selected by pipe_write.index().
// Caller must have already called producer_acquire.
template <int kStages, typename Pipeline, typename TmaA, typename TmaB,
          typename SmemLayoutA, typename SmemLayoutB, int kBM, int kBN, int kBK,
          typename ElementIn, typename TensorGA, typename TensorGB>
CUTLASS_DEVICE void tma_copy_k_tile(
    Pipeline& pipeline,
    typename Pipeline::PipelineState& pipe_write,
    TmaA const& tma_a, TmaB const& tma_b,
    ElementIn* smem_a_base, ElementIn* smem_b_base,
    TensorGA const& gA, TensorGB const& gB, int k_iter) {
  typename Pipeline::ProducerBarrierType* tma_bar =
      pipeline.producer_get_barrier(pipe_write);
  const int stg = pipe_write.index();

  auto cta_tma_a = tma_a.get_slice(_0{});
  auto cta_tma_b = tma_b.get_slice(_0{});

  Tensor sA_staged = make_tensor(make_smem_ptr(smem_a_base), SmemLayoutA{});
  Tensor sB_staged = make_tensor(make_smem_ptr(smem_b_base), SmemLayoutB{});

  Tensor tAgA = cta_tma_a.partition_S(gA);
  Tensor tBgB = cta_tma_b.partition_S(gB);
  Tensor tAsA = cta_tma_a.partition_D(sA_staged);
  Tensor tBsB = cta_tma_b.partition_D(sB_staged);

  copy(tma_a.with(*tma_bar), tAgA(_, _, _, k_iter), tAsA(_, _, _, stg));
  copy(tma_b.with(*tma_bar), tBgB(_, _, _, k_iter), tBsB(_, _, _, stg));
}

}  // namespace geforce_v2
}  // namespace blackwell
}  // namespace pearl
