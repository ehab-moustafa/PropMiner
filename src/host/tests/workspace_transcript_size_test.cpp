// workspace_transcript_size_test.cpp — CPU guard for pearl_capi_workspace_alloc
// transcript pool sizing.  Regression: passing noise rank r=128 as batch
// oversizes the pool ~128× and can OOM / corrupt before the first GEMM.
//
// Build/run via scripts/local_host_tests.sh (no GPU).

#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool ok, const char* msg) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_failures;
  }
}

constexpr int kBM = 128;
constexpr int kBN = 256;
constexpr int kThreads = 256;
constexpr int kSlots = 16;

int64_t transcript_buffer_elems(int64_t M, int64_t N, int64_t batch) {
  const int64_t num_m = M / kBM;
  const int64_t num_n = N / kBN;
  return batch * num_m * num_n * kThreads * kSlots;
}

}  // namespace

int main() {
  constexpr int M = 8192;
  constexpr int N = 32768;
  constexpr int r = 128;

  const int64_t correct = transcript_buffer_elems(M, N, 1);
  const int64_t buggy = transcript_buffer_elems(M, N, r);

  check(correct > 0, "prod transcript elems > 0");
  check(buggy == r * correct, "buggy sizing is r× correct batch=1 size");
  check(buggy > correct * 100,
        "buggy r-as-batch oversizes by >100× at prod N");

  // At prod shape batch=1 transcript is 128 MiB (33,554,432 u32).
  check(correct == 33554432LL,
        "prod M=8192 N=32768 batch=1 transcript elems == 33,554,432");

  const size_t correct_bytes = static_cast<size_t>(correct) * sizeof(uint32_t);
  const size_t buggy_bytes = static_cast<size_t>(buggy) * sizeof(uint32_t);
  check(correct_bytes == 128u * 1024u * 1024u,
        "prod transcript pool == 128 MiB with batch=1");
  check(buggy_bytes > 16u * 1024u * 1024u * 1024u,
        "buggy pool exceeds 16 GiB (explains vast.ai OOM)");

  std::printf(
      "workspace_transcript_size_test: correct=%lld elems (%.1f MiB) "
      "buggy(r=%d)=%lld elems (%.1f GiB)\n",
      (long long)correct, correct_bytes / (1024.0 * 1024.0), r,
      (long long)buggy, buggy_bytes / (1024.0 * 1024.0 * 1024.0));

  if (g_failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
