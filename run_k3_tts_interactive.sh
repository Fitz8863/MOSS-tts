#!/usr/bin/env bash
# Default K3 resident mode: initialize C++ ONNX sessions once, then loop.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/setup_k3_cpp_env.sh"
OUTPUT="${1:-$ROOT/outputs/interactive.wav}"
mkdir -p "$(dirname "$OUTPUT")"
BINARY="$MOSS_CPP_BUILD_DIR/moss-tts-onnx"
if [[ ! -x "$BINARY" ]]; then
  echo "C++ ONNX binary not found; building it once in $MOSS_CPP_BUILD_DIR ..." >&2
  "$ROOT/build_k3_cpp.sh"
fi
ARGS=(
  --model-dir "$MOSS_MODEL_DIR"
  --threads "$MOSS_CPP_THREADS"
  --max-new-frames "$MOSS_MAX_NEW_FRAMES"
  --voice "$MOSS_VOICE"
  --seed "$MOSS_SEED"
  --interactive "$OUTPUT"
)
if [[ -n "${MOSS_CPU_AFFINITY:-}" ]]; then
  exec taskset -c "$MOSS_CPU_AFFINITY" "$BINARY" "${ARGS[@]}"
else
  exec "$BINARY" "${ARGS[@]}"
fi
