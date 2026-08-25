#!/usr/bin/env bash
# K3 entry: C++ host + ONNX Runtime CPUExecutionProvider + ONNX models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<USAGE
Usage:
  $0 [--model fp32|int8] '<Chinese/English text>' <output.wav>
  $0 [--model fp32|int8] --interactive [output.wav]

Environment: MOSS_CPP_THREADS, MOSS_CPU_AFFINITY, MOSS_MAX_NEW_FRAMES,
             MOSS_VOICE, MOSS_SEED, MOSS_MODEL_DIR.
USAGE
}

MODEL_VARIANT="${MOSS_MODEL_VARIANT:-fp32}"
if [[ "${1:-}" == "--model" ]]; then
  [[ $# -ge 2 ]] || { usage; exit 2; }
  MODEL_VARIANT="$2"
  shift 2
fi
export MOSS_MODEL_VARIANT="$MODEL_VARIANT"
# shellcheck disable=SC1091
source "$ROOT/setup_k3_cpp_env.sh"

OUTPUT=""
INTERACTIVE=0
if [[ "${1:-}" == "--interactive" ]]; then
  INTERACTIVE=1
  shift
  OUTPUT="${1:-$ROOT/outputs/interactive_${MOSS_MODEL_VARIANT}.wav}"
  [[ $# -le 1 ]] || { usage; exit 2; }
else
  [[ $# -eq 2 ]] || { usage; exit 2; }
  TEXT="$1"
  OUTPUT="$2"
fi
mkdir -p "$(dirname "$OUTPUT")"

BINARY="$MOSS_CPP_BUILD_DIR/moss-tts-onnx"
if [[ ! -x "$BINARY" ]]; then
  echo "C++ ONNX binary not found; building it once in $MOSS_CPP_BUILD_DIR ..." >&2
  "$ROOT/build_k3_cpp.sh"
fi
[[ -f "$MOSS_MODEL_DIR/browser_poc_manifest.json" ]] || {
  echo "error: ONNX model manifest not found: $MOSS_MODEL_DIR/browser_poc_manifest.json" >&2
  exit 1
}

ARGS=(
  --model-dir "$MOSS_MODEL_DIR"
  --threads "$MOSS_CPP_THREADS"
  --max-new-frames "$MOSS_MAX_NEW_FRAMES"
  --voice "$MOSS_VOICE"
  --seed "$MOSS_SEED"
)
if (( INTERACTIVE )); then
  ARGS+=(--interactive "$OUTPUT")
else
  ARGS+=("$TEXT" "$OUTPUT")
fi

echo "launch model_variant=$MOSS_MODEL_VARIANT model_dir=$MOSS_MODEL_DIR threads=$MOSS_CPP_THREADS affinity=${MOSS_CPU_AFFINITY:-none}" >&2
if [[ -n "${MOSS_CPU_AFFINITY:-}" ]]; then
  exec taskset -c "$MOSS_CPU_AFFINITY" "$BINARY" "${ARGS[@]}"
else
  exec "$BINARY" "${ARGS[@]}"
fi
