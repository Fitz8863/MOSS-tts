#!/usr/bin/env bash
# Default K3 entry: C++ host + ONNX Runtime CPUExecutionProvider.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/setup_k3_cpp_env.sh"
if [[ "${1:-}" == "--interactive" ]]; then
  shift
  exec "$ROOT/run_k3_tts_interactive.sh" "${1:-$ROOT/outputs/interactive.wav}"
fi
if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "Usage: $0 '<Chinese/English text>' <output.wav>" >&2
  echo "Builtin voice: set MOSS_VOICE=Junhao|...; reference-audio encoding is not included in the C++ CLI yet." >&2
  exit 2
fi
TEXT="$1"
OUTPUT="$2"
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
  "$TEXT" "$OUTPUT"
)
if [[ -n "${MOSS_CPU_AFFINITY:-}" ]]; then
  exec taskset -c "$MOSS_CPU_AFFINITY" "$BINARY" "${ARGS[@]}"
else
  exec "$BINARY" "${ARGS[@]}"
fi
