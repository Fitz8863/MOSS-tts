#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--interactive" ]]; then
  shift
  exec "$ROOT/run_k3_tts_interactive.sh" "${1:-$ROOT/outputs/interactive.wav}"
fi
# shellcheck disable=SC1091
source "$ROOT/setup_k3_env.sh"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 '<Chinese/English text>' <output.wav> [reference.wav]" >&2
  exit 2
fi
TEXT="$1"
OUTPUT="$2"
mkdir -p "$(dirname "$OUTPUT")"
REF="${3:-}"
ARGS=(
  --model-dir "${MOSS_MODEL_DIR:-$ROOT/models}"
  --text "$TEXT"
  --output-audio-path "$OUTPUT"
  --cpu-threads "${MOSS_CPU_THREADS:-4}"
  --execution-provider "${MOSS_EXECUTION_PROVIDER:-cpu}"
  --max-new-frames "${MOSS_MAX_NEW_FRAMES:-375}"
  --sample-mode "${MOSS_SAMPLE_MODE:-fixed}"
  --realtime-streaming-decode 1
  --disable-wetext-processing
  --seed "${MOSS_SEED:-1234}"
)
if [[ -n "$REF" ]]; then
  ARGS+=(--prompt-audio-path "$REF")
else
  ARGS+=(--voice "${MOSS_VOICE:-Junhao}")
fi
# The Debian riscv64 ONNX Runtime build currently emits two known benign
# startup warnings and one ONNX schema-registration line per operator.  Keep
# real stderr/tracebacks visible, but hide only these exact known messages.
# Set MOSS_SHOW_ORT_DIAGNOSTICS=1 to show them for dependency debugging.
PYTHON_CMD=(python3)
if [[ -n "${MOSS_CPU_AFFINITY:-}" ]]; then
  PYTHON_CMD=(taskset -c "$MOSS_CPU_AFFINITY" python3)
fi
if [[ "${MOSS_SHOW_ORT_DIAGNOSTICS:-0}" == "1" ]]; then
  exec "${PYTHON_CMD[@]}" "$ROOT/infer_onnx.py" "${ARGS[@]}"
else
  exec "${PYTHON_CMD[@]}" "$ROOT/infer_onnx.py" "${ARGS[@]}" \
    2> >(python3 -u -c '
import re
import sys

ansi = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
suppress_blank = False
for raw in sys.stdin:
    line = ansi.sub("", raw)
    if "Unknown CPU vendor. cpuinfo_vendor value: 0" in line:
        suppress_blank = True
        continue
    if "GPU device discovery failed:" in line:
        suppress_blank = True
        continue
    if line.startswith("Schema error: Trying to register schema with name ") and \
       "but it is already registered from file " in line:
        suppress_blank = True
        continue
    if suppress_blank and not line.strip():
        continue
    suppress_blank = False
    sys.stderr.write(raw)
    sys.stderr.flush()
' >&2)
fi
