#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${MOSS_CPP_BUILD_DIR:-$ROOT/cpp/build-k3}"
ORT_LIB="${MOSS_ONNXRUNTIME_LIBRARY:-/usr/lib/python3.14/dist-packages/onnxruntime/capi/libonnxruntime.so.1.24.2+spacemit.a1}"
SPM_LIB="${MOSS_SENTENCEPIECE_LIBRARY:-$ROOT/board_root_spm/usr/lib/riscv64-linux-gnu/libsentencepiece.so.0}"
if [[ ! -f "$ORT_LIB" ]]; then
  echo "error: vendor ONNX Runtime not found: $ORT_LIB" >&2
  exit 1
fi
if [[ ! -f "$SPM_LIB" && ! -L "$SPM_LIB" ]]; then
  echo "error: SentencePiece library not found: $SPM_LIB" >&2
  exit 1
fi
cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_INCLUDE_DIR=/usr/include \
  -DONNXRUNTIME_LIBRARY="$ORT_LIB" \
  -DSENTENCEPIECE_LIBRARY="$SPM_LIB" \
  -DMOSS_ENABLE_RVV="${MOSS_ENABLE_RVV:-ON}" \
  -DMOSS_RVV_ARCH="${MOSS_RVV_ARCH:-rv64gcv}" \
  -DMOSS_RVV_TUNE="${MOSS_RVV_TUNE:-spacemit-x100}"
cmake --build "$BUILD_DIR" -j"${MOSS_BUILD_JOBS:-2}"
ldd "$BUILD_DIR/moss-tts-onnx" | grep -E 'onnxruntime|sentencepiece|not found' || true
