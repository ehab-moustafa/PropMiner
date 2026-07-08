#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "../portable/transcript_kernel.cuh"

namespace pearl {
namespace blackwell {

// GeForce v2: PipelineTmaAsync + descriptor cache + TMA prefetch.
// Compile: PEARL_GEMM_BLACKWELL_GEFORCE_V2=1 (default ON for blackwell).
// Runtime: default geforce_v2 when compiled in; PEARL_GEMM_KERNEL=geforce_v1 or consumer to opt out.

cudaError_t launch_transcript_gemm_sm120_geforce_v2(
    int8_t const* A, int8_t const* B, uint32_t* transcript,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    cudaStream_t stream);

cudaError_t launch_transcript_gemm_sm120_geforce_v2_headless(
    int8_t const* A, int8_t const* B, int32_t* C,
    int64_t M, int64_t N, int64_t K, int64_t R, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    cudaStream_t stream);

cudaError_t warmup_transcript_gemm_sm120_geforce_v2_attrs();

}  // namespace blackwell
}  // namespace pearl
