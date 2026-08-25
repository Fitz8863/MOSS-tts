#!/usr/bin/env bash
# Reconstruction RTF: moss-tts-cli vs the ONNX tokenizer on the same clips.
# Usage: ./bench.sh [cpu|cuda] [model.gguf]
set -euo pipefail
BACKEND="${1:-cpu}"; MODEL="${2:-models/moss-audio-tokenizer-f32.gguf}"
CLI="./build/bin/moss-tts-cli"
export MOSS_TTS_BACKEND="$BACKEND"
echo "# moss-tts.cpp reconstruction benchmark — backend=$BACKEND model=$(basename "$MODEL")"
printf '| clip | dur(s) | wall(s) | RTF |\n| ---- | ------ | ------- | --- |\n'
for w in samples/*.wav; do
  [ -f "$w" ] || continue
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$w")
  s=$(date +%s.%N)
  "$CLI" reconstruct --model "$MODEL" --in "$w" --out "/tmp/$(basename "$w")" >/dev/null 2>&1
  e=$(date +%s.%N); wall=$(echo "$e - $s" | bc -l); rtf=$(echo "scale=3; $wall / $dur" | bc -l)
  printf '| %s | %.1f | %.1f | %s |\n' "$(basename "$w")" "$dur" "$wall" "$rtf"
done
