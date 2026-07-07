#include "noise_generation_host.h"
#include "pearl_api_params.h"

template void run_noise_generation<128, 32>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<128, 64>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<128, 128>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 32>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 64>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 128>(Noise_gen_params&, cudaStream_t);
// noise_rank=256: Kryptex / HeroMiners-family stratum canonical shape
// (K=4096, R=256). R%32==0, NumThreads(64)%(R/16=16)==0, so this instantiates.
template void run_noise_generation<256, 64>(Noise_gen_params&, cudaStream_t);
