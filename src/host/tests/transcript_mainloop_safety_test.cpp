// transcript_mainloop_safety_test.cpp — CPU-only guards for GEMM mainloop schedule
// changes (pingpong / asymmetric DMA / pipeline reorder).
//
// Mirrors snapshot + slot + gmem layout from geforce_v2 + transcript_kernel.cu.
// No GPU / CUTLASS required.
//
// Build:
//   clang++ -std=c++17 -O2 -I src/host/tests \
//     src/host/tests/transcript_mainloop_safety_test.cpp \
//     -o transcript_mainloop_safety_test

#include <cstdint>
#include <cstdio>
#include <array>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* msg) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_failures;
  }
}

constexpr int kTranscriptSlots = 16;
constexpr int kConsumerThreads = 256;
constexpr int kRotation = 13;

uint32_t rotl32(uint32_t x, int shift) {
  return (x << shift) | (x >> (32 - shift));
}

uint32_t rotl_xor(uint32_t x, uint32_t y) {
  return rotl32(x, kRotation) ^ y;
}

uint32_t xor3(uint32_t a, uint32_t b, uint32_t c) {
  return a ^ b ^ c;
}

std::vector<uint32_t> xor_reduce_layer(const std::vector<uint32_t>& in) {
  const size_t n = in.size();
  const size_t triplets = n / 3;
  const size_t rem = n % 3;
  std::vector<uint32_t> out;
  out.reserve(triplets + rem);
  for (size_t i = 0; i < triplets; ++i) {
    out.push_back(xor3(in[3 * i], in[3 * i + 1], in[3 * i + 2]));
  }
  for (size_t i = 0; i < rem; ++i) {
    out.push_back(in[triplets * 3 + i]);
  }
  return out;
}

uint32_t xor_reduction(const std::vector<uint32_t>& input) {
  std::vector<uint32_t> cur = input;
  while (cur.size() > 1) {
    if (cur.size() == 2) {
      return cur[0] ^ cur[1];
    }
    if (cur.size() == 3) {
      return xor3(cur[0], cur[1], cur[2]);
    }
    cur = xor_reduce_layer(cur);
  }
  return cur[0];
}

int64_t transcript_buffer_elems(int64_t M, int64_t N, int64_t batch,
                                int bM = 128, int bN = 256) {
  const int64_t num_m_tiles = M / bM;
  const int64_t num_n_tiles = N / bN;
  return batch * num_m_tiles * num_n_tiles * kConsumerThreads * kTranscriptSlots;
}

// GeForce v2: snapshot hash BEFORE MMA at boundary k_iter; tail after loop.
void simulate_v2_schedule(int K_TILES, int reduce_every_k,
                          const uint32_t* hash_by_k_iter,
                          std::array<uint32_t, kTranscriptSlots>& transcript) {
  for (int s = 0; s < kTranscriptSlots; ++s) transcript[s] = 0;

  for (int k_iter = 0; k_iter < K_TILES; ++k_iter) {
    if (k_iter > 0 && (k_iter % reduce_every_k) == 0) {
      const uint32_t hash = hash_by_k_iter[k_iter];
      const int snapshot_idx = (k_iter / reduce_every_k) - 1;
      const int slot = snapshot_idx % kTranscriptSlots;
      transcript[slot] = rotl_xor(transcript[slot], hash);
    }
  }

  if ((K_TILES % reduce_every_k) == 0) {
    const uint32_t hash = hash_by_k_iter[K_TILES];
    const int snapshot_idx = (K_TILES / reduce_every_k) - 1;
    const int slot = snapshot_idx % kTranscriptSlots;
    transcript[slot] = rotl_xor(transcript[slot], hash);
  }
}

void test_xor_reduction_deterministic() {
  std::vector<uint32_t> input(128);
  for (int i = 0; i < 128; ++i) input[i] = static_cast<uint32_t>(i + 1);

  const uint32_t h1 = xor_reduction(input);
  const uint32_t h2 = xor_reduction(input);
  check(h1 == h2, "xor_reduction is deterministic");
  check(h1 != 0u, "xor_reduction nontrivial on 1..128");
}

void test_rotl_xor() {
  check(rotl_xor(0x00000001u, 0x00000002u) ==
            (rotl32(0x00000001u, kRotation) ^ 0x00000002u),
        "rotl_xor matches rotl^y");
}

void test_transcript_buffer_indexing() {
  const int64_t elems = transcript_buffer_elems(8192, 262144, 1);
  const int64_t expected = 1LL * 64 * 1024 * 256 * 16;
  check(elems == expected, "transcript_buffer_elems production shape");

  const int64_t num_n_tiles = 262144 / 256;
  const int64_t m_tile = 3;
  const int64_t n_tile = 7;
  const int tid = 42;
  const int64_t base = m_tile * num_n_tiles + n_tile;
  const int64_t tx_off =
      base * kConsumerThreads * kTranscriptSlots + tid * kTranscriptSlots;
  check(tx_off + kTranscriptSlots <= elems, "thread transcript slice in bounds");
}

void test_snapshot_slot_schedule_k128_kbk64() {
  const int K_TILES = 2;
  const int reduce_every_k = 2;
  std::array<uint32_t, 16> hash_by_k{};
  hash_by_k[K_TILES] = 0xB0000002u;

  std::array<uint32_t, kTranscriptSlots> t{};
  simulate_v2_schedule(K_TILES, reduce_every_k, hash_by_k.data(), t);

  const uint32_t expected = rotl_xor(0, hash_by_k[K_TILES]);
  check(t[0] == expected, "K=128 tail-only fold into slot 0");
  for (int s = 1; s < kTranscriptSlots; ++s) {
    check(t[s] == 0, "unused transcript slots stay zero");
  }
}

void test_schedule_order_sensitive() {
  const int K_TILES = 4;
  const int reduce_every_k = 2;
  std::array<uint32_t, 8> hash_by_k{};
  for (int k = 0; k <= K_TILES; ++k) hash_by_k[k] = static_cast<uint32_t>(k);

  std::array<uint32_t, kTranscriptSlots> correct{};
  simulate_v2_schedule(K_TILES, reduce_every_k, hash_by_k.data(), correct);

  // Wrong order: apply boundary hashes after MMA (snapshot_idx off by one).
  std::array<uint32_t, kTranscriptSlots> wrong{};
  for (int k_iter = 1; k_iter <= K_TILES; ++k_iter) {
    if ((k_iter % reduce_every_k) == 0) {
      const int snapshot_idx = k_iter / reduce_every_k;  // bug: should be -1 at boundary
      const int slot = snapshot_idx % kTranscriptSlots;
      wrong[slot] = rotl_xor(wrong[slot], hash_by_k[k_iter]);
    }
  }

  bool differs = false;
  for (int s = 0; s < kTranscriptSlots; ++s) {
    if (correct[s] != wrong[s]) differs = true;
  }
  check(differs, "wrong snapshot index changes transcript");
}

void test_snapshot_slot_schedule_k4096_kbk64() {
  // Production K=4096, kBK=64, R=128 → 64 K-tiles, snapshot every 2 → 32 folds.
  const int K_TILES = 64;
  const int reduce_every_k = 2;
  std::array<uint32_t, 68> hash_by_k{};
  for (int k = 0; k <= K_TILES; ++k) {
    hash_by_k[k] = static_cast<uint32_t>(0xA0000000u + k);
  }

  std::array<uint32_t, kTranscriptSlots> t{};
  simulate_v2_schedule(K_TILES, reduce_every_k, hash_by_k.data(), t);

  // All 16 slots participate after 32 boundary folds.
  int nonzero = 0;
  for (int s = 0; s < kTranscriptSlots; ++s) {
    if (t[s] != 0) ++nonzero;
  }
  check(nonzero == kTranscriptSlots, "K=4096 uses all 16 transcript slots");
}

}  // namespace

int main() {
  std::fprintf(stderr, "transcript_mainloop_safety_test\n");
  test_xor_reduction_deterministic();
  test_rotl_xor();
  test_transcript_buffer_indexing();
  test_snapshot_slot_schedule_k128_kbk64();
  test_snapshot_slot_schedule_k4096_kbk64();
  test_schedule_order_sensitive();

  if (g_failures == 0) {
    std::fprintf(stderr, "OK: all transcript mainloop safety checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "%d failure(s)\n", g_failures);
  return 1;
}
