#!/usr/bin/env bash
# Compatibility wrapper for resident C++ ONNX mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat >&2 <<USAGE
Usage:
  $0 [--model fp32|int8] [--voice NAME] [output.wav]

The model is initialized once and then accepts one Chinese/English line per prompt.
USAGE
}

MODEL_VARIANT="${MOSS_MODEL_VARIANT:-fp32}"
VOICE_ARGS=()
POSITIONAL=()
while (($#)); do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      MODEL_VARIANT="$2"
      shift 2
      ;;
    --voice)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      VOICE_ARGS+=(--voice "$2")
      shift 2
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
[[ ${#POSITIONAL[@]} -le 1 ]] || { usage; exit 2; }
OUTPUT="${POSITIONAL[0]:-$ROOT/outputs/interactive_${MODEL_VARIANT}.wav}"
exec "$ROOT/run_k3_tts.sh" --model "$MODEL_VARIANT" "${VOICE_ARGS[@]}" --interactive "$OUTPUT"
