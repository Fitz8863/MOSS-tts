#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/setup_k3_env.sh"

OUTPUT="${1:-$ROOT/outputs/interactive.wav}"
VOICE="${MOSS_VOICE:-Junhao}"
ARGS=(
  --model-dir "${MOSS_MODEL_DIR:-$ROOT/models}"
  --output "$OUTPUT"
  --voice "$VOICE"
  --cpu-threads "${MOSS_CPU_THREADS:-4}"
  --execution-provider "${MOSS_EXECUTION_PROVIDER:-cpu}"
  --max-new-frames "${MOSS_MAX_NEW_FRAMES:-375}"
  --sample-mode "${MOSS_SAMPLE_MODE:-fixed}"
  --realtime-streaming-decode 1
  --enable-wetext-processing 0
  --enable-normalize-tts-text 1
  --seed "${MOSS_SEED:-1234}"
)
if [[ -n "${MOSS_PROMPT_AUDIO_PATH:-}" ]]; then
  ARGS+=(--prompt-audio-path "$MOSS_PROMPT_AUDIO_PATH")
fi

# Keep real exceptions visible while hiding only the known Debian riscv64 ORT
# startup diagnostics. Set MOSS_SHOW_ORT_DIAGNOSTICS=1 to show everything.
PYTHON_CMD=(python3)
if [[ -n "${MOSS_CPU_AFFINITY:-}" ]]; then
  PYTHON_CMD=(taskset -c "$MOSS_CPU_AFFINITY" python3)
fi
if [[ "${MOSS_SHOW_ORT_DIAGNOSTICS:-0}" == "1" ]]; then
  exec "${PYTHON_CMD[@]}" "$ROOT/interactive_k3_tts.py" "${ARGS[@]}"
fi
exec "${PYTHON_CMD[@]}" "$ROOT/interactive_k3_tts.py" "${ARGS[@]}" \
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
