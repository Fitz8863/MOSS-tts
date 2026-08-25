cmake_minimum_required(VERSION 3.18)
project(moss-tts
        VERSION   0.0.1
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- options ----------
option(MOSS_TTS_BUILD_TESTS    "Build moss-tts tests"    OFF)
option(MOSS_TTS_BUILD_EXAMPLES "Build moss-tts examples" OFF)
option(MOSS_TTS_BUILD_SERVER   "Build moss-tts server"   OFF)
option(MOSS_TTS_TEST_LARGE     "Run large/E2E tests"     OFF)
option(MOSS_TTS_SHARED         "Build shared library"    OFF)

# Backend toggles — forwarded to the embedded ggml submodule. Off by
# default; enable with `-DMOSS_TTS_GGML_CUDA=ON`, etc.
option(MOSS_TTS_GGML_CUDA      "Build ggml with CUDA backend"     OFF)
option(MOSS_TTS_GGML_METAL     "Build ggml with Metal backend"    OFF)
option(MOSS_TTS_GGML_VULKAN    "Build ggml with Vulkan backend"   OFF)
option(MOSS_TTS_GGML_HIPBLAS   "Build ggml with hipBLAS backend"  OFF)
option(MOSS_TTS_GGML_RISCV64_SPACEMIT "Build ggml with SpaceMIT RISC-V CPU backend" OFF)

# ---------- ggml ----------
# We embed ggml as a submodule under third_party/ggml. We forward common
# GGML_* options so users can drive backends from the top-level cache.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(MOSS_TTS_GGML_CUDA)
    set(GGML_CUDA   ON CACHE BOOL "" FORCE)
endif()
if(MOSS_TTS_GGML_METAL)
    set(GGML_METAL  ON CACHE BOOL "" FORCE)
endif()
if(MOSS_TTS_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(MOSS_TTS_GGML_HIPBLAS)
    set(GGML_HIPBLAS ON CACHE BOOL "" FORCE)
endif()
if(MOSS_TTS_GGML_RISCV64_SPACEMIT)
    set(GGML_CPU_RISCV64_SPACEMIT ON CACHE BOOL "" FORCE)
    set(GGML_CPU_REPACK OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---------- libmoss-tts ----------
set(MOSS_TTS_SOURCES
    src/common.cpp
    src/backend.cpp
    src/parallel.cpp
    src/profiler.cpp
    src/model_loader.cpp
    src/audio_io.cpp
    src/patchify.cpp
    src/transformer.cpp
    src/quantizer.cpp
    src/audio_tokenizer.cpp
    src/moss_tts.cpp
    src/moss_tts_capi.cpp
    # --- V1 MossTTSDelay ---
    src/qwen3.cpp
    src/delay_backbone.cpp
    src/delay_embeddings.cpp
    src/lm_heads.cpp
    src/sampling.cpp
    src/delay_state.cpp
    src/de_tokenizer.cpp
    src/prompt.cpp
    src/moss_tts_delay.cpp
    # --- V2 MossTTSLocal ---
    src/local_transformer.cpp
    src/local_embeddings.cpp
    src/local_adapters.cpp
    src/prompt_local.cpp
    src/moss_tts_local.cpp
    # --- V3 MossTTSRealtime ---
    src/rt_local.cpp
    src/rt_embeddings.cpp
    src/rt_heads.cpp
    src/prompt_rt.cpp
    src/moss_tts_rt.cpp
    # --- V4 MossTTSNano ---
    src/gpt2.cpp
    src/nano_backbone.cpp
    src/nano_local.cpp
    src/nano_embeddings.cpp
    src/nano_heads.cpp
    src/sp_tokenizer.cpp
    src/text_cleanup.cpp
    src/prompt_nano.cpp
    src/moss_tts_nano.cpp
)

set(MOSS_TTS_PUBLIC_HEADERS
    include/moss_tts.h
    include/moss_tts_capi.h
)

if(MOSS_TTS_SHARED)
    add_library(moss-tts SHARED ${MOSS_TTS_SOURCES})
    target_compile_definitions(moss-tts PUBLIC MOSS_TTS_SHARED)
    target_compile_definitions(moss-tts PRIVATE MOSS_TTS_BUILD)
else()
    add_library(moss-tts STATIC ${MOSS_TTS_SOURCES})
endif()

target_include_directories(moss-tts
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(moss-tts PUBLIC ggml)

# moss::parallel_for (src/parallel.cpp) uses std::thread — link the platform
# thread library explicitly rather than relying on it arriving transitively.
find_package(Threads REQUIRED)
target_link_libraries(moss-tts PUBLIC Threads::Threads)

# When a ggml backend (CUDA / Metal / Vulkan / hipBLAS / blas) is enabled,
# explicitly link its target so the static library carries the symbols
# (mirrors the pattern in LocalAI's qwen3-tts-cpp wrapper).
foreach(_be cuda metal vulkan hipblas blas)
    if(TARGET ggml-${_be})
        target_link_libraries(moss-tts PUBLIC ggml-${_be})
        string(TOUPPER ${_be} _BE)
        target_compile_definitions(moss-tts PUBLIC MOSS_TTS_HAVE_${_BE})
    endif()
endforeach()
if(MOSS_TTS_GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(moss-tts PUBLIC CUDA::cudart)
    endif()
endif()

target_compile_options(moss-tts PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)

# ---------- subdirs ----------
if(MOSS_TTS_BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli/CMakeLists.txt)
    add_subdirectory(examples/cli)
endif()

if(MOSS_TTS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------- install ----------
include(GNUInstallDirs)

install(TARGETS moss-tts
        EXPORT  moss-ttsTargets
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${MOSS_TTS_PUBLIC_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
