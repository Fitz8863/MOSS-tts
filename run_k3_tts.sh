#!/usr/bin/env bash
# K3 entry: C++ host + ONNX Runtime CPUExecutionProvider + ONNX models.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<USAGE
Usage:
  $0 [--model fp32|int8] [--voice NAME] '<Chinese/English text>' <output.wav>
  $0 [--model fp32|int8] [--voice NAME] --interactive [output.wav]

Options:
  --model VARIANT  Select the fp32 or int8 ONNX model (default: fp32).
  --voice NAME     Select a voice from the model manifest (default: Junhao).
  --interactive    Keep the model resident and synthesize one line at a time.
  --help           Show this help.

Environment: MOSS_CPP_THREADS, MOSS_CPU_AFFINITY, MOSS_MAX_NEW_FRAMES,
             MOSS_VOICE, MOSS_SEED, MOSS_MODEL_DIR.

Precedence: command-line --model/--voice > environment variables > defaults.
USAGE
}

MODEL_VARIANT="${MOSS_MODEL_VARIANT:-fp32}"
VOICE_OVERRIDE=""
INTERACTIVE=0
POSITIONAL=()

while (($#)); do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { echo "error: --model requires a value" >&2; usage; exit 2; }
      MODEL_VARIANT="$2"
      shift 2
      ;;
    --voice)
      [[ $# -ge 2 ]] || { echo "error: --voice requires a value" >&2; usage; exit 2; }
      VOICE_OVERRIDE="$2"
      shift 2
      ;;
    --interactive)
      INTERACTIVE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      POSITIONAL+=("$@")
      break
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

export MOSS_MODEL_VARIANT="$MODEL_VARIANT"
# shellcheck disable=SC1091
source "$ROOT/setup_k3_cpp_env.sh"
# A CLI value must override MOSS_VOICE loaded by setup_k3_cpp_env.sh.
if [[ -n "$VOICE_OVERRIDE" ]]; then
  export MOSS_VOICE="$VOICE_OVERRIDE"
fi

OUTPUT=""
if (( INTERACTIVE )); then
  [[ ${#POSITIONAL[@]} -le 1 ]] || { usage; exit 2; }
  OUTPUT="${POSITIONAL[0]:-$ROOT/outputs/interactive_${MOSS_MODEL_VARIANT}.wav}"
else
  [[ ${#POSITIONAL[@]} -eq 2 ]] || { usage; exit 2; }
  TEXT="${POSITIONAL[0]}"
  OUTPUT="${POSITIONAL[1]}"
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

echo "launch model_variant=$MOSS_MODEL_VARIANT model_dir=$MOSS_MODEL_DIR voice=$MOSS_VOICE threads=$MOSS_CPP_THREADS affinity=${MOSS_CPU_AFFINITY:-none}" >&2
if [[ -n "${MOSS_CPU_AFFINITY:-}" ]]; then
  exec taskset -c "$MOSS_CPU_AFFINITY" "$BINARY" "${ARGS[@]}"
else
  exec "$BINARY" "${ARGS[@]}"
fi
