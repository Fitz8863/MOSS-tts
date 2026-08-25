cmake_minimum_required(VERSION 3.16)
project(moss_tts_onnx_cpp LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)
pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
option(MOSS_ENABLE_RVV "Build the C++ host with RISC-V Vector ISA enabled" ON)
set(MOSS_RVV_ARCH "rv64gcv" CACHE STRING "RISC-V -march value for the C++ host")
set(MOSS_RVV_TUNE "spacemit-x100" CACHE STRING "RISC-V -mtune value for the C++ host")

# These are cache variables on purpose: the board has both a generic system
# ORT (1.23) and the vendor board ORT (1.24.2+spacemit).  The build
# script always selects the vendor binary explicitly.
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "Directory containing onnxruntime_cxx_api.h")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "Full path to the ONNX Runtime shared library")
set(SENTENCEPIECE_LIBRARY "" CACHE FILEPATH "Full path to libsentencepiece.so")

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include /usr/local/include
    REQUIRED)
endif()
if(NOT ONNXRUNTIME_LIBRARY)
  find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS
      /usr/lib/python3.14/dist-packages/onnxruntime/capi
      /usr/lib/riscv64-linux-gnu
      /usr/lib
      /lib
    REQUIRED)
endif()
if(NOT SENTENCEPIECE_LIBRARY)
  find_library(SENTENCEPIECE_LIBRARY NAMES sentencepiece
    PATHS
      ${CMAKE_CURRENT_SOURCE_DIR}/../board_root_spm/usr/lib/riscv64-linux-gnu
      /usr/lib/riscv64-linux-gnu
      /usr/lib
      /lib
    REQUIRED)
endif()

add_executable(moss-tts-onnx src/main.cpp)
target_include_directories(moss-tts-onnx PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(moss-tts-onnx PRIVATE
  ${ONNXRUNTIME_LIBRARY}
  ${SENTENCEPIECE_LIBRARY}
  PkgConfig::AVFORMAT
  PkgConfig::AVCODEC
  PkgConfig::SWRESAMPLE
  PkgConfig::AVUTIL
  dl pthread
)
target_compile_options(moss-tts-onnx PRIVATE -O3 -fno-math-errno -fopenmp)

# The C++ process itself contains the text/prompt orchestration and codec
# glue; the heavy kernels come from the selected vendor ONNX Runtime CPU EP.
# Enable RVV code generation on the board, but do not claim NPU/A100 use from
# this flag alone.  Actual kernel dispatch is a property of the ORT binary.
if(MOSS_ENABLE_RVV AND CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv")
  target_compile_options(moss-tts-onnx PRIVATE -march=${MOSS_RVV_ARCH} -mtune=${MOSS_RVV_TUNE} -mabi=lp64d)
  target_compile_definitions(moss-tts-onnx PRIVATE MOSS_CPP_RVV_BUILD=1)
else()
  target_compile_definitions(moss-tts-onnx PRIVATE MOSS_CPP_RVV_BUILD=0)
endif()

# Keep the vendor ORT and board SentencePiece discoverable when the binary is
# launched from cpp/build-k3.  The wrapper also exports the absolute paths.
target_link_options(moss-tts-onnx PRIVATE -Wl,-rpath,'$ORIGIN/../lib')
