#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "../portable/transcript_kernel.cuh"

namespace pearl {
namespace blackwell {

// RTX 5090 (sm_120a) warp-specialized transcript GEMM: dedicated TMA producer
// warp + 256 consumer warps using proof-canonical SM80 mma.sync int8 IMMA.
// This is NOT tcgen05/TMEM — GeForce Blackwell lacks that ISA (see
// external_repos/blackwell-geforce-nvfp4-gemm/docs/sm120-architecture.md).
//
// Gated at compile time (PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=1, default ON).
// Runtime default is geforce when compiled in; PEARL_GEMM_KERNEL=consumer opts out.

cudaError_t launch_transcript_gemm_sm120_geforce(
    int8_t const* A, int8_t const* B, uint32_t* transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream);

cudaError_t launch_transcript_gemm_sm120_geforce_headless(
    int8_t const* A, int8_t const* B, int32_t* C, uint32_t* transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream);

cudaError_t warmup_transcript_gemm_sm120_geforce_attrs();

}  // namespace blackwell
}  // namespace pearl
