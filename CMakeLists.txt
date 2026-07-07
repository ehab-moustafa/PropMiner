cmake_minimum_required(VERSION 3.24)
project(PropMiner VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Paths
set(PROP_MINER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROP_MINER_THIRD_PARTY "${PROP_MINER_ROOT}/third_party")
set(PEARL_GEMM_ROOT "${PROP_MINER_THIRD_PARTY}/pearl-gemm")
set(PEARL_MINING_CAPI_ROOT "${PROP_MINER_THIRD_PARTY}/pearl-mining-capi")
set(CUTLASS_ROOT "${PEARL_GEMM_ROOT}/third_party/cutlass")

# ── CUDA Toolkit ───────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── OpenSSL (for Pearl V2 gRPC TLS) ────────────────────────────────────
find_package(OpenSSL REQUIRED)

# ── Auto-detect GPU architecture ───────────────────────────────────────
# Default blackwell (sm_120a) for RTX 5090; use auto only when building for
# other SKUs on a machine with nvidia-smi visible at configure time.
set(PROP_MINER_CUDA_ARCH "blackwell" CACHE STRING
    "GPU arch profile: auto|h100|volta|turing|portable|ampere|ada|blackwell|b200")

function(detect_gpu_arch out_var)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NOT NVIDIA_SMI)
        set(${out_var} "blackwell" PARENT_SCOPE)
        message(STATUS "nvidia-smi not found; using blackwell (sm_120a) for RTX 5090")
        return()
    endif()
    execute_process(
        COMMAND "${NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _cap
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    string(REGEX REPLACE "\..*$" "" _cap "${_cap}")
    string(STRIP "${_cap}" _cap)
    if(_cap STREQUAL "90")
        set(${out_var} "h100" PARENT_SCOPE)
    elseif(_cap STREQUAL "89")
        set(${out_var} "ada" PARENT_SCOPE)
    elseif(_cap STREQUAL "86" OR _cap STREQUAL "80")
        set(${out_var} "ampere" PARENT_SCOPE)
    elseif(_cap STREQUAL "75")
        set(${out_var} "turing" PARENT_SCOPE)
    elseif(_cap STREQUAL "70")
        set(${out_var} "volta" PARENT_SCOPE)
    elseif(_cap STREQUAL "120")
        set(${out_var} "blackwell" PARENT_SCOPE)
    elseif(_cap STREQUAL "100")
        set(${out_var} "b200" PARENT_SCOPE)
    else()
        set(${out_var} "portable" PARENT_SCOPE)
    endif()
endfunction()

if(PROP_MINER_CUDA_ARCH STREQUAL "auto")
    detect_gpu_arch(PROP_MINER_CUDA_ARCH)
    message(STATUS "Auto-detected PEARL_GEMM_ARCH=${PROP_MINER_CUDA_ARCH}")
endif()
set(PEARL_GEMM_ARCH "${PROP_MINER_CUDA_ARCH}" CACHE STRING "" FORCE)
if(PEARL_GEMM_ARCH STREQUAL "blackwell")
    message(STATUS "Blackwell build: nvcc targets sm_120a (RTX 5090 native)")
endif()

# ── Kernel tuning knobs (only affect consumer/Blackwell/Ada builds) ────
set(PEARL_GEMM_BLACKWELL_BM "128" CACHE STRING "Blackwell consumer tile M")
set(PEARL_GEMM_BLACKWELL_BN "256" CACHE STRING "Blackwell consumer tile N")
set(PEARL_GEMM_BLACKWELL_KBLOCK "128" CACHE STRING "Blackwell smem K-tile")
set(PEARL_GEMM_BLACKWELL_STAGES "2" CACHE STRING "Blackwell pipeline stages")
set(PEARL_GEMM_BLACKWELL_SWIZZLE_BITS "3" CACHE STRING "Blackwell swizzle bits")
set(PEARL_GEMM_BLACKWELL_MIN_BLOCKS "1" CACHE STRING "Blackwell min blocks per SM")
set(PEARL_GEMM_BLACKWELL_LOAD_POLICY "cp_async" CACHE STRING
    "Blackwell load policy: cp_async|tma")
set(PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL OFF CACHE BOOL
    "Compile RTX 5090 warp-specialized TMA+IMMA experimental kernel (NOT tcgen05; default OFF)")

# ── Fetch CUTLASS if missing ───────────────────────────────────────────
include(ExternalProject)
find_package(Git)
if(NOT EXISTS "${CUTLASS_ROOT}/include/cutlass/cutlass.h")
    message(STATUS "CUTLASS not found; fetching v4.4.0 (SM120 MMA traits)")
    if(NOT GIT_EXECUTABLE)
        message(FATAL_ERROR "Git is required to fetch CUTLASS")
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" clone --depth 1 --branch v4.4.0
            https://github.com/NVIDIA/cutlass.git "${CUTLASS_ROOT}"
        RESULT_VARIABLE _cutlass_rc)
    if(NOT _cutlass_rc EQUAL 0)
        message(FATAL_ERROR "Failed to fetch CUTLASS v4.4.0")
    endif()
endif()

# ── Build libpearl_mining_capi.so (Rust) ───────────────────────────────
find_program(CARGO cargo)
if(NOT CARGO)
    message(FATAL_ERROR "cargo not found. Install Rust to build pearl_mining_capi.")
endif()

set(RUST_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/pearl_mining_capi")
set(RUST_LIB "${RUST_LIB_DIR}/release/libpearl_mining_capi.so")
file(MAKE_DIRECTORY "${RUST_LIB_DIR}")

add_custom_command(
    OUTPUT "${RUST_LIB}"
    COMMAND ${CARGO} build --release --manifest-path
            "${PEARL_MINING_CAPI_ROOT}/Cargo.toml" -p pearl-mining-capi
            --target-dir "${RUST_LIB_DIR}"
    COMMENT "Building libpearl_mining_capi.so (Rust)"
    VERBATIM)

add_custom_target(pearl_mining_capi_build DEPENDS "${RUST_LIB}")

# ── Build libpearl_gemm_capi.so (CUDA) ─────────────────────────────────
# Use the canonical GNU Makefile on Linux; it exposes the full tuning-knob
# surface (Blackwell stages, k-block, swizzle, min-blocks, load policy).  The
# CMakeLists in the same folder is the Windows port and supports fewer knobs.
set(GEMM_BUILD_DIR "${PEARL_GEMM_ROOT}/csrc/capi/build")
set(GEMM_LIB "${GEMM_BUILD_DIR}/libpearl_gemm_capi.so")
# Stamp invalidates Docker BuildKit gemm caches when knobs (e.g. GEFORCE) change.
set(GEMM_BUILD_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/pearl_gemm_build_config.stamp")
file(WRITE "${GEMM_BUILD_CONFIG}"
    "PEARL_GEMM_ARCH=${PEARL_GEMM_ARCH}\n"
    "PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=${PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL}\n"
    "PEARL_GEMM_BLACKWELL_BM=${PEARL_GEMM_BLACKWELL_BM}\n"
    "PEARL_GEMM_BLACKWELL_BN=${PEARL_GEMM_BLACKWELL_BN}\n"
    "PEARL_GEMM_BLACKWELL_KBLOCK=${PEARL_GEMM_BLACKWELL_KBLOCK}\n"
    "PEARL_GEMM_BLACKWELL_STAGES=${PEARL_GEMM_BLACKWELL_STAGES}\n"
    "PEARL_GEMM_BLACKWELL_SWIZZLE_BITS=${PEARL_GEMM_BLACKWELL_SWIZZLE_BITS}\n"
    "PEARL_GEMM_BLACKWELL_MIN_BLOCKS=${PEARL_GEMM_BLACKWELL_MIN_BLOCKS}\n"
    "PEARL_GEMM_BLACKWELL_LOAD_POLICY=${PEARL_GEMM_BLACKWELL_LOAD_POLICY}\n")

find_program(MAKE_EXECUTABLE make)
if(NOT MAKE_EXECUTABLE)
    message(FATAL_ERROR "make not found; required to build libpearl_gemm_capi.so")
endif()

if(PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL)
    set(PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL_MAKE "1")
else()
    set(PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL_MAKE "")
endif()

add_custom_command(
    OUTPUT "${GEMM_LIB}"
    DEPENDS "${GEMM_BUILD_CONFIG}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${GEMM_LIB}"
    COMMAND ${MAKE_EXECUTABLE} -f "${PEARL_GEMM_ROOT}/csrc/capi/Makefile"
            "PEARL_GEMM_ARCH=${PEARL_GEMM_ARCH}"
            "PEARL_GEMM_BLACKWELL_BM=${PEARL_GEMM_BLACKWELL_BM}"
            "PEARL_GEMM_BLACKWELL_BN=${PEARL_GEMM_BLACKWELL_BN}"
            "PEARL_GEMM_BLACKWELL_KBLOCK=${PEARL_GEMM_BLACKWELL_KBLOCK}"
            "PEARL_GEMM_BLACKWELL_STAGES=${PEARL_GEMM_BLACKWELL_STAGES}"
            "PEARL_GEMM_BLACKWELL_SWIZZLE_BITS=${PEARL_GEMM_BLACKWELL_SWIZZLE_BITS}"
            "PEARL_GEMM_BLACKWELL_MIN_BLOCKS=${PEARL_GEMM_BLACKWELL_MIN_BLOCKS}"
            "PEARL_GEMM_BLACKWELL_LOAD_POLICY=${PEARL_GEMM_BLACKWELL_LOAD_POLICY}"
            "PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL=${PEARL_GEMM_BLACKWELL_GEFORCE_KERNEL_MAKE}"
            "NVCC_THREADS=4"
    WORKING_DIRECTORY "${PEARL_GEMM_ROOT}"
    COMMENT "Building libpearl_gemm_capi.so for ${PEARL_GEMM_ARCH}"
    VERBATIM)

add_custom_target(pearl_gemm_capi_build DEPENDS "${GEMM_LIB}")

# Copy the generated .so files next to the binary
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libpearl_gemm_capi.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GEMM_LIB}" "${CMAKE_CURRENT_BINARY_DIR}/libpearl_gemm_capi.so"
    DEPENDS pearl_gemm_capi_build
    COMMENT "Staging libpearl_gemm_capi.so")
add_custom_target(stage_gemm_lib ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libpearl_gemm_capi.so")

add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libpearl_mining_capi.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${RUST_LIB}" "${CMAKE_CURRENT_BINARY_DIR}/libpearl_mining_capi.so"
    DEPENDS pearl_mining_capi_build
    COMMENT "Staging libpearl_mining_capi.so")
add_custom_target(stage_mining_lib ALL
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libpearl_mining_capi.so")

# Host executables (propminer, tests, benchmarks). Disable for Docker layer-1
# gemm-only builds where src/host is not copied yet.
option(PROPMINER_BUILD_HOST "Build propminer and host test/bench targets" ON)

if(PROPMINER_BUILD_HOST)
# ── Main executable ────────────────────────────────────────────────────
add_executable(propminer
    src/host/main.cpp
    src/host/pearl/pearl_capi_wrapper.cpp
    src/host/pearl/pearl_mining_wrapper.cpp
    src/host/pearl/pearl_blake3.cpp
    src/host/pearl/pearl_types.cpp
    src/host/pearl/job_key.cpp
    src/host/pearl/sigma_context.cpp
    src/host/pearl/gpu_worker.cpp
    src/host/pearl/gpu_tuner.cpp
    src/host/pearl/tune_cache.cpp
    src/host/pearl/kernel_knob_cache.cpp
    src/host/pearl/mine_batch_cache.cpp
    src/host/pearl/share_builder.cpp
    src/host/pearl/bincode_encoder.cpp
    src/host/pearl/watchdog.cpp
    src/host/pearl/async_b_stream.cpp
    src/host/pearl/protobuf_encoder.cpp
    src/host/pearl/host_signal_header.cpp
    src/host/pearl/job_bus.cpp
    src/host/pearl/pow_target_utils.cpp
    src/host/pearl/share_diagnostics.cpp
    src/host/pearl/worker_orchestrator.cpp
    src/host/pearl/system_telemetry.cpp
    src/host/pearl/grpc_client.cpp
    src/host/pearl/pearl_stratum_client.cpp
    src/host/pearl/pearl_challenge.cpp
    src/host/pearl/proto/mining_v2.cpp
    src/host/stratum/simple_json.cpp
)

add_dependencies(propminer pearl_mining_capi_build pearl_gemm_capi_build
                 stage_gemm_lib stage_mining_lib)

target_include_directories(propminer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host/pearl
    ${PEARL_GEMM_ROOT}/csrc/capi
    ${PEARL_MINING_CAPI_ROOT}/src
)

target_link_libraries(propminer PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    OpenSSL::SSL
    OpenSSL::Crypto
    "${CMAKE_CURRENT_BINARY_DIR}/libpearl_gemm_capi.so"
    "${CMAKE_CURRENT_BINARY_DIR}/libpearl_mining_capi.so"
    pthread
    dl
)

target_compile_options(propminer PRIVATE -Wall -Wextra -O2 -fvisibility=hidden)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(propminer PRIVATE PROP_MINER_DEBUG)
endif()

# ── Host-only reference-test executable (no CUDA/Rust required at link time) ─
add_executable(propminer_ref_tests
    src/host/tests.cpp
    src/host/tests/blake3_reference.c
    src/host/tests/ref_blake3.cpp
    src/host/tests/ref_pearl.cpp
    src/host/pearl/pearl_types.cpp
    src/host/pearl/job_key.cpp
    src/host/pearl/protobuf_encoder.cpp
    src/host/pearl/proto/mining_v2.cpp
    src/host/pearl/host_signal_header.cpp
    src/host/pearl/share_diagnostics.cpp
    src/host/stratum/simple_json.cpp
)
target_include_directories(propminer_ref_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host/pearl
)
target_link_libraries(propminer_ref_tests PRIVATE
    pthread
    dl
)
target_compile_options(propminer_ref_tests PRIVATE -Wall -Wextra -O2)

# ── Host-only benchmark executables ─────────────────────────────────────
set(PROP_MINER_BENCH_SOURCES
    src/host/tests/blake3_reference.c
    src/host/tests/ref_blake3.cpp
    src/host/tests/ref_pearl.cpp
    src/host/pearl/pearl_types.cpp
    src/host/pearl/job_key.cpp
    src/host/pearl/protobuf_encoder.cpp
    src/host/pearl/proto/mining_v2.cpp
    src/host/pearl/host_signal_header.cpp
)

add_executable(propminer_ref_benchmarks
    src/host/benchmarks.cpp
    ${PROP_MINER_BENCH_SOURCES}
)
target_include_directories(propminer_ref_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host/pearl
)
target_link_libraries(propminer_ref_benchmarks PRIVATE pthread dl)
target_compile_options(propminer_ref_benchmarks PRIVATE -Wall -Wextra -O2)

add_executable(propminer_ref_benchmarks_rigorous
    src/host/benchmarks_rigorous.cpp
    ${PROP_MINER_BENCH_SOURCES}
)
target_include_directories(propminer_ref_benchmarks_rigorous PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host
    ${CMAKE_CURRENT_SOURCE_DIR}/src/host/pearl
)
target_link_libraries(propminer_ref_benchmarks_rigorous PRIVATE pthread dl)
target_compile_options(propminer_ref_benchmarks_rigorous PRIVATE -Wall -Wextra -O2)

# ── Test executable (links against Rust mining C API when available) ────
if(CARGO)
    add_executable(propminer_tests
        src/host/tests.cpp
        src/host/pearl/pearl_types.cpp
        src/host/pearl/job_key.cpp
        src/host/pearl/pearl_blake3.cpp
        src/host/pearl/pearl_mining_wrapper.cpp
        src/host/pearl/protobuf_encoder.cpp
        src/host/pearl/proto/mining_v2.cpp
    )
    add_dependencies(propminer_tests pearl_mining_capi_build stage_mining_lib)
    target_include_directories(propminer_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host
        ${CMAKE_CURRENT_SOURCE_DIR}/src/host/pearl
        ${PEARL_MINING_CAPI_ROOT}/src
    )
    target_link_libraries(propminer_tests PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/libpearl_mining_capi.so"
        pthread
        dl
    )
    target_compile_options(propminer_tests PRIVATE -Wall -Wextra -O2)
endif()

# ── Install ────────────────────────────────────────────────────────────
install(TARGETS propminer DESTINATION bin)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libpearl_gemm_capi.so"
    "${CMAKE_CURRENT_BINARY_DIR}/libpearl_mining_capi.so"
    DESTINATION bin)
endif() # PROPMINER_BUILD_HOST

# ── Build summary ──────────────────────────────────────────────────────
message(STATUS "PropMiner v${PROJECT_VERSION}")
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "PEARL_GEMM_ARCH: ${PEARL_GEMM_ARCH}")
message(STATUS "CARGO: ${CARGO}")
