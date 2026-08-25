#!/usr/bin/env bash
# Compatibility wrapper for resident C++ ONNX mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_VARIANT="${MOSS_MODEL_VARIANT:-fp32}"
if [[ "${1:-}" == "--model" ]]; then
  [[ $# -ge 2 ]] || { echo "Usage: $0 [--model fp32|int8] [output.wav]" >&2; exit 2; }
  MODEL_VARIANT="$2"
  shift 2
fi
exec "$ROOT/run_k3_tts.sh" --model "$MODEL_VARIANT" --interactive "${1:-$ROOT/outputs/interactive_${MODEL_VARIANT}.wav}"
