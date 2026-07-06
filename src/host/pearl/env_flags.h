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

}  // namespace pearl
