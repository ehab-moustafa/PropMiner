// Minimal CUDA graph capture/replay test for GeForce v2 vs consumer GEMM.
// Reproduces the vast.ai failure mode: capture OK, replay → illegal access.
//
// Build (on RTX 5090):
//   make -C third_party/pearl-gemm/csrc/capi PEARL_GEMM_ARCH=blackwell \
//     PEARL_GEMM_BLACKWELL_GEFORCE_V2=1 verify-geforce-v2-graph
//
// Run:
//   third_party/pearl-gemm/csrc/capi/build/verify_geforce_v2_graph

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "../portable/transcript_kernel.cuh"
#include "transcript_gemm_sm120_geforce_v2.h"

namespace {

struct ShapeSpec {
  int M, N, K, R;
  const char* label;
};

using LaunchFn = cudaError_t (*)(int8_t const*, int8_t const*, uint32_t*,
                                 int64_t, int64_t, int64_t, int64_t, int64_t,
                                 cudaStream_t);

cudaError_t launch_v2(int8_t const* A, int8_t const* B, uint32_t* tr, int64_t M,
                      int64_t N, int64_t K, int64_t R, int64_t batch,
                      cudaStream_t stream) {
  return pearl::blackwell::launch_transcript_gemm_sm120_geforce_v2(
      A, B, tr, M, N, K, R, batch, stream);
}

cudaError_t launch_consumer(int8_t const* A, int8_t const* B, uint32_t* tr,
                            int64_t M, int64_t N, int64_t K, int64_t R,
                            int64_t batch, cudaStream_t stream) {
  return pearl::consumer::launch_transcript_gemm(A, B, nullptr, tr, M, N, K, R,
                                                 batch, stream);
}

int test_graph_replay(const char* kernel_name, LaunchFn launch, ShapeSpec shape,
                      int replays) {
  const int64_t batch = 1;
  const size_t a_bytes = static_cast<size_t>(shape.M) * shape.K;
  const size_t b_bytes = static_cast<size_t>(shape.N) * shape.K;
  const int64_t tr_elems = pearl::portable::transcript_buffer_elems(
      shape.M, shape.N, batch);
  const size_t tr_bytes = static_cast<size_t>(tr_elems) * sizeof(uint32_t);

  int8_t *dA = nullptr, *dB = nullptr;
  uint32_t* dTr = nullptr;
  if (cudaMalloc(&dA, a_bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&dB, b_bytes) != cudaSuccess) return 1;
  if (cudaMalloc(&dTr, tr_bytes) != cudaSuccess) return 1;
  cudaMemset(dA, 1, a_bytes);
  cudaMemset(dB, 2, b_bytes);
  cudaMemset(dTr, 0, tr_bytes);

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t err = cudaSuccess;
  // GeForce v2: pre-upload TMA before capture (same as production graph_prepare).
  if (launch == launch_v2) {
    err = pearl::blackwell::prepare_geforce_v2_tma_for_graph(
        dA, dB, shape.M, shape.N, shape.K, stream);
    if (err != cudaSuccess) {
      std::printf("  [%s %s] TMA pre-upload failed: %s\n", kernel_name,
                  shape.label, cudaGetErrorString(err));
      return 1;
    }
    cudaStreamSynchronize(stream);
  }

  err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
  if (err != cudaSuccess) {
    std::printf("  [%s %s] begin capture failed: %s\n", kernel_name, shape.label,
                cudaGetErrorString(err));
    return 1;
  }

  err = launch(dA, dB, dTr, shape.M, shape.N, shape.K, shape.R, batch, stream);
  if (err != cudaSuccess) {
    std::printf("  [%s %s] launch during capture failed: %s\n", kernel_name,
                shape.label, cudaGetErrorString(err));
    cudaStreamEndCapture(stream, &graph);
    if (graph) cudaGraphDestroy(graph);
    return 1;
  }

  err = cudaStreamEndCapture(stream, &graph);
  if (err != cudaSuccess || graph == nullptr) {
    std::printf("  [%s %s] end capture failed: %s\n", kernel_name, shape.label,
                cudaGetErrorString(err));
    return 1;
  }

  err = cudaGraphInstantiate(&graph_exec, graph, 0);
  if (err != cudaSuccess) {
    std::printf("  [%s %s] instantiate failed: %s\n", kernel_name, shape.label,
                cudaGetErrorString(err));
    cudaGraphDestroy(graph);
    return 1;
  }

  err = cudaGraphUpload(graph_exec, stream);
  if (err != cudaSuccess) {
    std::printf("  [%s %s] upload failed: %s\n", kernel_name, shape.label,
                cudaGetErrorString(err));
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    return 1;
  }

  int fail = 0;
  for (int r = 0; r < replays; ++r) {
    err = cudaGraphLaunch(graph_exec, stream);
    if (err != cudaSuccess) {
      std::printf("  [%s %s] replay %d launch failed: %s\n", kernel_name,
                  shape.label, r, cudaGetErrorString(err));
      fail = 1;
      break;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      std::printf(
          "  [%s %s] replay %d sync failed (illegal access likely here): %s\n",
          kernel_name, shape.label, r, cudaGetErrorString(err));
      fail = 1;
      break;
    }
  }

  if (fail == 0) {
    std::printf("  [%s %s] %d graph replays PASS\n", kernel_name, shape.label,
                replays);
  }

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dTr);
  return fail;
}

}  // namespace

int main() {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  std::printf("GeForce v2 CUDA graph replay harness\n\n");

  if (pearl::blackwell::warmup_transcript_gemm_sm120_geforce_v2_attrs() !=
      cudaSuccess) {
    std::printf("FAIL: v2 warmup attrs\n");
    return 1;
  }
  if (pearl::consumer::warmup_transcript_kernel_consumer_attrs() !=
      cudaSuccess) {
    std::printf("FAIL: consumer warmup attrs\n");
    return 1;
  }

  const ShapeSpec shapes[] = {
      {2048, 4096, 128, 128, "tiny K=128"},
      {8192, 32768, 128, 128, "prod N K=128"},
      {8192, 32768, 4096, 128, "prod N K=4096"},
  };

  int fail = 0;
  const int replays = 3;
  int consumer_fail = 0;
  int v2_fail = 0;
  for (const auto& s : shapes) {
    std::printf("Shape M=%d N=%d K=%d (%s)\n", s.M, s.N, s.K, s.label);
    if (test_graph_replay("consumer", launch_consumer, s, replays)) {
      fail = 1;
      consumer_fail = 1;
    }
    if (test_graph_replay("geforce_v2", launch_v2, s, replays)) {
      fail = 1;
      v2_fail = 1;
    }
    std::printf("\n");
  }

  std::printf("=== Diagnosis ===\n");
  if (fail == 0) {
    std::printf("All kernels pass graph capture + %d replays.\n", replays);
    std::printf("If mining still fails, run ./scripts/debug_geforce_v2.sh phases 5-6\n");
  } else if (consumer_fail == 0 && v2_fail != 0) {
    std::printf(
        "PATTERN: consumer PASS, geforce_v2 FAIL → TMA + CUDA graph on sm_120.\n");
    std::printf(
        "Fix: __grid_constant__ TMA only (prepare_geforce_v2_tma_for_graph "
        "before capture; no upload_v2_tma_to_device on direct launch).\n");
    std::printf(
        "Workaround: PROPMINER_BENCH_NO_GRAPH=1 or PEARL_GEMM_KERNEL=consumer\n");
  } else if (consumer_fail != 0 && v2_fail != 0) {
    std::printf(
        "PATTERN: both FAIL → general CUDA graph / driver issue, not v2-only.\n");
    std::printf("Check: driver version, sm_120a vs sm_120f build, compute-sanitizer.\n");
  } else {
    std::printf("PATTERN: consumer FAIL, v2 PASS — unexpected; report with full log.\n");
  }

  std::printf("%s\n", fail ? "RESULT: FAIL" : "RESULT: PASS");
  return fail;
}
