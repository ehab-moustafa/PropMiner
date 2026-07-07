#pragma once

#include <cstdlib>
#include <cstring>

namespace pearl {

// True when env is set to a non-empty value other than "0".
inline bool env_truthy(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

inline bool bench_no_graph_enabled() {
    return env_truthy("PROPMINER_BENCH_NO_GRAPH");
}

// Max wall time a single GPU batch may take before we treat the worker as
// wedged (GPU pinned at 100% but not making progress — a hung CUDA stream that
// in-process recovery cannot clear). On stall we exit so the supervisor
// (PROPMINER_RESTART_ON_EXIT) relaunches with a fresh context. 0 disables.
inline long long stall_restart_ms() {
    const char* env = std::getenv("PROPMINER_STALL_RESTART_MS");
    if (env && env[0] != '\0') {
        char* end = nullptr;
        long long v = std::strtoll(env, &end, 10);
        if (end != env && v >= 0) return v;
    }
    return 30000;  // 30s: a healthy batch is well under 1s.
}

}  // namespace pearl
