#!/usr/bin/env bash
# MossTTSLocal TTS benchmark: wall-time + RTF on a fixed prompt set.
# Usage: ./bench_local.sh [cpu|cuda] --model LOCAL.gguf --codec CODEC.gguf --tokenizer TOK.gguf
set -euo pipefail
BACKEND="${1:-cpu}"; shift || true
MODEL="" CODEC="" TOK=""
while [ $# -gt 0 ]; do case "$1" in --model) MODEL="$2"; shift 2;; --codec) CODEC="$2"; shift 2;; --tokenizer) TOK="$2"; shift 2;; *) shift;; esac; done
[ -n "$MODEL" ] && [ -n "$CODEC" ] && [ -n "$TOK" ] || { echo "usage: ./bench_local.sh [cpu|cuda] --model L --codec C --tokenizer T"; exit 2; }
export MOSS_TTS_BACKEND="$BACKEND"
CLI=./build/bin/moss-tts-cli
PROMPTS=( "Hello world." "The quick brown fox jumps over the lazy dog." "Text to speech synthesis is working." )
printf '# moss-tts.cpp MossTTSLocal bench — backend=%s\n| # | prompt words | wall(s) | audio(s) | RTF |\n| - | - | - | - | - |\n' "$BACKEND"
i=0
for p in "${PROMPTS[@]}"; do
  i=$((i+1)); words=$(echo "$p" | wc -w)
  s=$(date +%s.%N)
  "$CLI" tts-local --model "$MODEL" --codec "$CODEC" --tokenizer "$TOK" --text "$p" --seed 12345 --out "/tmp/bench_local_$i.wav" >/dev/null 2>&1
  e=$(date +%s.%N); wall=$(echo "$e - $s" | bc -l)
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "/tmp/bench_local_$i.wav" 2>/dev/null || echo 0)
  rtf=$(echo "scale=3; if ($dur>0) $wall/$dur else 0" | bc -l)
  printf '| %d | %s | %.1f | %s | %s |\n' "$i" "$words" "$wall" "$dur" "$rtf"
done
