// Portable hash transcript + PoW check kernels.
//
// The H100 noisy_gemm kernel maintains a per-(m_tile, n_tile, thread)
// "transcript" of 16 uint32 slots in registers across the K-loop:
//
//   For each (m_tile, n_tile) tile:
//     transcript[0..15] = 0
//     C_running[bM, bN] : int32 = 0
//     For s in 0..K/R-1:
//       C_running += int8_GEMM(ApEA[m_tile, s*R:(s+1)*R],
//                              BpEB[n_tile, s*R:(s+1)*R].T)
//       hash_t = xor_reduction( per-thread fragment slots of C_running )
//       slot   = s mod 16
//       transcript[slot] = rotl_xor<13>(transcript[slot], hash_t)
//     hash256 = BLAKE3.compress(transcript, key=pow_key)
//     if hash256 <= pow_target: write_host_signal_header(...)
//
// Per-thread fragment slot ordering MUST match H100's WGMMA register layout
// byte-for-byte so the network accepts blocks mined on RTX 5090.  We get
// this for free by extracting the layout from the same KernelTraits::TiledMma
// type H100 uses, via partition_C(make_identity_tensor((bM, bN))).  This is
// the same TiledMma type the H100 mainloop instantiates, so by CUTE design
// `partition_C` returns the identical per-thread coord ordering that
// `partition_fragment_C` produces register-internally.
//
// For tiny (bM=128, bN=256, K=128, R=64): 2 snapshots, slots {0,1} active.
// For prod (bM=128, bN=256, K=4096, R=128): 32 snapshots, all 16 slots
//   receive exactly 2 rotl_xor mixings each.

#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/mma_sm90.h>

#include "../blake3/blake3.cuh"
#include "../blake3/blake3_constants.hpp"
#include "../gemm/host_signal_header.hpp"
#include "../gemm/pow_utils.hpp"

#include "transcript_canonical.cuh"
#include "transcript_kernel.cuh"

namespace pearl {
namespace portable {

using namespace cute;

// ─── Constant-memory pow_target ───────────────────────────────────────────
// pow_target is read-only by every thread in the finalize kernel.  By placing
// it in constant memory (4 KiB cache) we avoid re-fetching it from global
// memory on every load — the L1/const cache typically serves it in 1 cycle.
__device__ __constant__ uint32_t d_pow_target_const[8];

// Compile-time tile parameters.  Both production-shipped tile shapes
// (R=64 and R=128) share these — see static_switch_matmul.h.
static constexpr int bM = kCanonicalTranscriptBM;
static constexpr int bN = kCanonicalTranscriptBN;
static constexpr int kNumMmaThreads = kCanonicalTranscriptThreads;

using ElementIn  = int8_t;
using ElementAcc = int32_t;

// IMPORTANT: this MUST match KernelTraits::TiledMma used in H100's
// collective_mainloop.hpp.  Both H100 and this portable kernel instantiate
// the same Atom (GMMA::ss_op_selector with the same template args) wrapped
// in the same AtomLayoutMNK, so partition_C produces identical per-thread
// coord orderings.  partition_C is pure CUTE layout math — it does NOT
// emit any wgmma SASS, so it compiles cleanly for sm_120a.
using TileShape_MNK = CanonicalTranscriptTileShape;
using PortableTiledMma = CanonicalTranscriptTiledMma;

// ─── Snapshot kernel ───────────────────────────────────────────────────────
// Grid:   (M/bM, N/bN, batch)
// Block:  kNumMmaThreads (256)
// Reads C_running[batch, M, N] (row-major, n stride = 1).
// For each (m_tile, n_tile, batch, thread):
//   - Gathers this thread's 128 int32 fragment slots from C_running using
//     partition_C(identity_tensor) on PortableTiledMma.
//   - XOR-reduces via pearl::xor_reduction (lop3 tree, identical to H100).
//   - rotl_xor<13> mixes into transcript[batch, m_tile, n_tile, thread, slot]
//     where slot = snapshot_idx mod 16.
__global__ void transcript_snapshot_kernel(
    int32_t const* __restrict__ C_running,
    int64_t M, int64_t N,
    uint32_t* __restrict__ transcript,
    int32_t snapshot_idx) {
  int m_tile = blockIdx.x;
  int n_tile = blockIdx.y;
  int batch  = blockIdx.z;
  int tid    = threadIdx.x;

  int64_t num_n_tiles = N / bN;
  int64_t num_m_tiles = M / bM;

  PortableTiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  Tensor cD   = make_identity_tensor(Shape<Int<bM>, Int<bN>>{});
  Tensor tCcD = thr_mma.partition_C(cD);
  constexpr int frag_size = decltype(size(tCcD))::value;  // 128
  static_assert(frag_size > 0, "frag_size must be positive");
  static_assert(frag_size <= MAX_NUM_REGISTERS_PER_THREAD,
                "frag_size exceeds HostSignalHeader capacity");

  // Gather per-thread fragment from C_running.
  cute::array<uint32_t, frag_size> frag;
  int64_t c_base = (int64_t)batch * M * N
                   + (int64_t)m_tile * bM * N
                   + (int64_t)n_tile * bN;

  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < frag_size; ++j) {
    int m = get<0>(tCcD(j));
    int n = get<1>(tCcD(j));
    int32_t v = C_running[c_base + (int64_t)m * N + (int64_t)n];
    frag[j] = static_cast<uint32_t>(v);
  }

  // Wrap in a CUTE tensor view so xor_reduction can compute its tree sizes.
  Tensor frag_t = make_tensor(frag.data(), Layout<Int<frag_size>>{});
  uint32_t hash = pearl::xor_reduction(frag_t);

  int slot = snapshot_idx % (int)blake3::MSG_BLOCK_SIZE_U32;
  int64_t per_tile_thread = (int64_t)kNumMmaThreads
                            * (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t per_tile = (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t base = ((int64_t)batch * num_m_tiles + m_tile) * num_n_tiles + n_tile;
  int64_t tx_idx = base * per_tile_thread + (int64_t)tid * per_tile + slot;

  uint32_t prev = transcript[tx_idx];
  transcript[tx_idx] = pearl::rotl_xor<pearl::HASH_ACCUMULATE_ROTATION>(
      prev, hash);
}

// ─── PTX-based BLAKE3 helpers ────────────────────────────────────────────
// PTX 32-bit right rotate via shf.l.wrap.b32.
//   shf.l.wrap.b32 d, x, x, n  =>  d = (x << n) | (x >> (32-n)) = rotl(x, n)
// which is semantically identical to rightrotate32(x, n) = (x << (32-n)) | (x >> n).
CUTLASS_DEVICE
uint32_t add32(uint32_t x, uint32_t y) {
  return x + y;
}

CUTLASS_DEVICE
uint32_t ptx_rightrotate32(uint32_t x, unsigned int n) {
  uint32_t r;
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(r) : "r"(x), "r"(n));
  return r;
}

// BLAKE3_ROUND macro that uses PTX rotate instead of software rotate.
// Identical to BLAKE3_ROUND() in blake3.cuh except rightrotate32 -> ptx_rightrotate32.
#define BLAKE3_ROUND_PTX()                                                          \
  do {                                                                              \
    rState(0) = add32(rState(0), add32(rState(4), rBlock(0)));                      \
    rState(12) = ptx_rightrotate32(rState(12) ^ rState(0), 16);                     \
    rState(8) = add32(rState(8), rState(12));                                       \
    rState(4) = ptx_rightrotate32(rState(4) ^ rState(8), 12);                       \
    rState(0) = add32(rState(0), add32(rState(4), rBlock(1)));                      \
    rState(12) = ptx_rightrotate32(rState(12) ^ rState(0), 8);                      \
    rState(8) = add32(rState(8), rState(12));                                       \
    rState(4) = ptx_rightrotate32(rState(4) ^ rState(8), 7);                        \
    rState(1) = add32(rState(1), add32(rState(5), rBlock(2)));                      \
    rState(13) = ptx_rightrotate32(rState(13) ^ rState(1), 16);                     \
    rState(9) = add32(rState(9), rState(13));                                       \
    rState(5) = ptx_rightrotate32(rState(5) ^ rState(9), 12);                       \
    rState(1) = add32(rState(1), add32(rState(5), rBlock(3)));                      \
    rState(13) = ptx_rightrotate32(rState(13) ^ rState(1), 8);                      \
    rState(9) = add32(rState(9), rState(13));                                       \
    rState(5) = ptx_rightrotate32(rState(5) ^ rState(9), 7);                        \
    rState(2) = add32(rState(2), add32(rState(6), rBlock(4)));                      \
    rState(14) = ptx_rightrotate32(rState(14) ^ rState(2), 16);                     \
    rState(10) = add32(rState(10), rState(14));                                     \
    rState(6) = ptx_rightrotate32(rState(6) ^ rState(10), 12);                      \
    rState(2) = add32(rState(2), add32(rState(6), rBlock(5)));                      \
    rState(14) = ptx_rightrotate32(rState(14) ^ rState(2), 8);                      \
    rState(10) = add32(rState(10), rState(14));                                     \
    rState(6) = ptx_rightrotate32(rState(6) ^ rState(10), 7);                       \
    rState(3) = add32(rState(3), add32(rState(7), rBlock(6)));                      \
    rState(15) = ptx_rightrotate32(rState(15) ^ rState(3), 16);                     \
    rState(11) = add32(rState(11), rState(15));                                     \
    rState(7) = ptx_rightrotate32(rState(7) ^ rState(11), 12);                      \
    rState(3) = add32(rState(3), add32(rState(7), rBlock(7)));                      \
    rState(15) = ptx_rightrotate32(rState(15) ^ rState(3), 8);                      \
    rState(11) = add32(rState(11), rState(15));                                     \
    rState(7) = ptx_rightrotate32(rState(7) ^ rState(11), 7);                       \
    rState(0) = add32(rState(0), add32(rState(5), rBlock(8)));                      \
    rState(15) = ptx_rightrotate32(rState(15) ^ rState(0), 16);                     \
    rState(10) = add32(rState(10), rState(15));                                     \
    rState(5) = ptx_rightrotate32(rState(5) ^ rState(10), 12);                      \
    rState(0) = add32(rState(0), add32(rState(5), rBlock(9)));                      \
    rState(15) = ptx_rightrotate32(rState(15) ^ rState(0), 8);                      \
    rState(10) = add32(rState(10), rState(15));                                     \
    rState(5) = ptx_rightrotate32(rState(5) ^ rState(10), 7);                       \
    rState(1) = add32(rState(1), add32(rState(6), rBlock(10)));                     \
    rState(12) = ptx_rightrotate32(rState(12) ^ rState(1), 16);                     \
    rState(11) = add32(rState(11), rState(12));                                     \
    rState(6) = ptx_rightrotate32(rState(6) ^ rState(11), 12);                      \
    rState(1) = add32(rState(1), add32(rState(6), rBlock(11)));                     \
    rState(12) = ptx_rightrotate32(rState(12) ^ rState(1), 8);                      \
    rState(11) = add32(rState(11), rState(12));                                     \
    rState(6) = ptx_rightrotate32(rState(6) ^ rState(11), 7);                       \
    rState(2) = add32(rState(2), add32(rState(7), rBlock(12)));                     \
    rState(13) = ptx_rightrotate32(rState(13) ^ rState(2), 16);                     \
    rState(8) = add32(rState(8), rState(13));                                       \
    rState(7) = ptx_rightrotate32(rState(7) ^ rState(8), 12);                       \
    rState(2) = add32(rState(2), add32(rState(7), rBlock(13)));                     \
    rState(13) = ptx_rightrotate32(rState(13) ^ rState(2), 8);                      \
    rState(8) = add32(rState(8), rState(13));                                       \
    rState(7) = ptx_rightrotate32(rState(7) ^ rState(8), 7);                        \
    rState(3) = add32(rState(3), add32(rState(4), rBlock(14)));                     \
    rState(14) = ptx_rightrotate32(rState(14) ^ rState(3), 16);                     \
    rState(9) = add32(rState(9), rState(14));                                       \
    rState(4) = ptx_rightrotate32(rState(4) ^ rState(9), 12);                       \
    rState(3) = add32(rState(3), add32(rState(4), rBlock(15)));                     \
    rState(14) = ptx_rightrotate32(rState(14) ^ rState(3), 8);                      \
    rState(9) = add32(rState(9), rState(14));                                       \
    rState(4) = ptx_rightrotate32(rState(4) ^ rState(9), 7);                        \
  } while (0)

// ─── Finalize kernel ───────────────────────────────────────────────────────
// Grid:   (M/bM, N/bN, batch)
// Block:  kNumMmaThreads (256)
// Per (m_tile, n_tile, batch, thread):
//   - Loads its 16-u32 transcript from gmem into rmem.
//   - BLAKE3-compresses (single keyed block) with pow_key as initial chaining
//     using inlined compress with PTX rotate instructions.
//   - Compares 256-bit hash <= pow_target (LE word order, MSW first).
//   - If hit: atomic-CAS lock on host_signal_sync, write a HostSignalHeader
//     with this thread's (tile_coord, partition_C coords, mma sizes, target).
__global__ void transcript_finalize_kernel(
    uint32_t const* __restrict__ transcript,
    int M, int N,
    uint32_t const* __restrict__ pow_target,
    uint32_t const* __restrict__ pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    int problem_m, int problem_n, int problem_k, int problem_r) {
  int m_tile = blockIdx.x;
  int n_tile = blockIdx.y;
  int batch  = blockIdx.z;
  int tid    = threadIdx.x;

  int64_t num_n_tiles = N / bN;
  int64_t num_m_tiles = M / bM;

  int64_t per_tile_thread = (int64_t)kNumMmaThreads
                            * (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t per_tile = (int64_t)blake3::MSG_BLOCK_SIZE_U32;
  int64_t base = ((int64_t)batch * num_m_tiles + m_tile) * num_n_tiles
                 + n_tile;
  int64_t tx_idx = base * per_tile_thread + (int64_t)tid * per_tile;

  // Load transcript into a CUTE rmem tensor using 4x LDG128 for coalesced 64-byte loads.
  uint128_t tmp[4];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) {
    tmp[i] = cutlass::uint128_t(uint64_t(transcript[tx_idx + i * 4]));
  }

  Tensor transcript_rmem = make_tensor<uint32_t>(
      Int<blake3::MSG_BLOCK_SIZE_U32>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 16; ++i) {
    transcript_rmem(i) = reinterpret_cast<uint32_t*>(&tmp[0])[i];
  }

  // ─── Inline BLAKE3 compress with PTX rotates ───────────────────────────
  // Byte-identical to check_pow_target but uses PTX shf.l.wrap.b32 rotates.

  // Local tensors for message block, chaining value, and state
  Tensor rBlock = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
  Tensor rHash  = make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
  Tensor rState = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});
  Tensor rOrigBlock = make_tensor<uint32_t>(Int<blake3::MSG_BLOCK_SIZE_U32>{});

  // Load transcript into rBlock
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < (int)blake3::MSG_BLOCK_SIZE_U32; ++i) {
    rBlock(i) = transcript_rmem(i);
  }

  // Load pow_key as initial chaining value (rHash)
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < (int)blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    rHash(i) = pow_key[i];
  }

  // Initialize state: top half = chaining value, bottom half = IV + params
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < (int)blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    rState(i) = rHash(i);
  }
  rState(8)  = blake3::IV0;
  rState(9)  = blake3::IV1;
  rState(10) = blake3::IV2;
  rState(11) = blake3::IV3;
  rState(12) = 0ULL;                          // counter low
  rState(13) = 0ULL >> 32;                    // counter high
  rState(14) = blake3::MSG_BLOCK_SIZE;        // block_len = 64
  rState(15) = blake3::KEYED_HASH | blake3::CHUNK_START | blake3::CHUNK_END | blake3::ROOT;

  // 6 rounds + permutation
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 6; ++i) {
    BLAKE3_ROUND_PTX();
    BLAKE3_PERMUTE();
  }
  // Final round (no permutation)
  BLAKE3_ROUND_PTX();

  // XOR top and bottom halves to produce the 8-word hash
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < (int)blake3::CHAINING_VALUE_SIZE_U32; ++i) {
    rHash(i) = rState(i) ^ rState(i + 8);
  }

  // ─── MSW-first uint256 comparison: hash <= pow_target ──────────────────
  // Use constant-memory pow_target (d_pow_target_const) for 1-cycle loads.
  bool block_found = true;
  CUTLASS_PRAGMA_UNROLL
  for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
    uint32_t target_i = d_pow_target_const[i];
    if (rHash(i) > target_i) {
      block_found = false;
      break;
    }
    if (rHash(i) < target_i) {
      break;
    }
  }

  // Warp-level early exit: skip the atomic CAS entirely if no thread in the
  // warp found a hit.  This avoids wasted global atomics from threads that
  // would only read the CAS failure and return.
  bool warp_hit = __any_sync(0xffffffffu, block_found);
  if (warp_hit && block_found) {
    // Block coord = (m_tile, n_tile, batch).  Same convention as H100
    // tile_scheduler.
    auto block_coord = cute::make_tuple(
        (int32_t)m_tile, (int32_t)n_tile, (int32_t)batch);
    auto problem_shape = cute::make_tuple(
        problem_m, problem_n, problem_k, problem_r);
    pearl::write_host_signal_header<PortableTiledMma, TileShape_MNK>(
        host_signal_sync, host_signal_header_pinned,
        problem_shape, block_coord, tid, pow_target);
  }
}

// ─── Host launchers ────────────────────────────────────────────────────────

void launch_transcript_snapshot(
    int32_t const* C_running,
    int64_t M, int64_t N, int64_t batch,
    uint32_t* transcript,
    int32_t snapshot_idx,
    cudaStream_t stream) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  dim3 grid((unsigned)(M / bM), (unsigned)(N / bN), (unsigned)batch);
  dim3 block((unsigned)kNumMmaThreads);
  transcript_snapshot_kernel<<<grid, block, 0, stream>>>(
      C_running, M, N, transcript, snapshot_idx);
}

void launch_transcript_finalize(
    uint32_t const* transcript,
    int64_t M, int64_t N, int64_t batch,
    uint32_t const* pow_target, uint32_t const* pow_key,
    HostSignalSync* host_signal_sync,
    HostSignalHeader* host_signal_header_pinned,
    int problem_m, int problem_n, int problem_k, int problem_r,
    cudaStream_t stream) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  // Copy pow_target to constant memory so every thread fetches it from the
  // 4 KiB const cache (1-cycle latency) instead of global memory.
  cudaMemcpyToSymbol(d_pow_target_const, pow_target, 8 * sizeof(uint32_t));
  dim3 grid((unsigned)(M / bM), (unsigned)(N / bN), (unsigned)batch);
  dim3 block((unsigned)kNumMmaThreads);
  transcript_finalize_kernel<<<grid, block, 0, stream>>>(
      transcript, (int)M, (int)N,
      pow_target, pow_key,
      host_signal_sync, host_signal_header_pinned,
      problem_m, problem_n, problem_k, problem_r);
}

int64_t transcript_buffer_elems(int64_t M, int64_t N, int64_t batch) {
  assert(M % bM == 0);
  assert(N % bN == 0);
  int64_t num_m_tiles = M / bM;
  int64_t num_n_tiles = N / bN;
  return batch * num_m_tiles * num_n_tiles
         * (int64_t)kNumMmaThreads
         * (int64_t)blake3::MSG_BLOCK_SIZE_U32;
}

}  // namespace portable
}  // namespace pearl
