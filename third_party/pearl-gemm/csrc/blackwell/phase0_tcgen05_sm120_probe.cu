// Phase 0 go/no-go: attempt tcgen05 PTX on sm_120a (RTX 5090).
//
// Build (on CUDA host):
//   nvcc -arch=sm_120a phase0_tcgen05_sm120_probe.cu -o /tmp/phase0_tcgen05_probe
//
// Expected on RTX 5090: ptxas FAIL — "not supported on .target sm_120a"
// See external_repos/blackwell-geforce-nvfp4-gemm/docs/sm120-architecture.md

#include <cuda_runtime.h>
#include <cstdio>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__global__ void tcgen05_probe_kernel() {
  // Minimal tcgen05.mma placeholder — ptxas rejects on sm_120a.
  asm volatile("tcgen05.fence::before_thread_sync;\n");
}
#endif

int main() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  tcgen05_probe_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("UNEXPECTED: tcgen05 compiled for device arch %d\n", __CUDA_ARCH__);
  return 1;
#else
  printf("Host build OK — compile with -arch=sm_120a to test ptxas rejection.\n");
  printf("GeForce Blackwell (RTX 5090) uses mma.sync + TMA, not tcgen05/TMEM.\n");
  printf("PropMiner experimental path: transcript_gemm_sm120_geforce.cu\n");
  return 0;
#endif
}
