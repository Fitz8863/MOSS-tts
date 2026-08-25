#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/bin/moss-tts-cli"
MODEL_DIR="$ROOT/cpp/models"
LIB_DIR="$ROOT/cpp/moss-tts.cpp-spacemit-v016/build-k3-spacemit/third_party/ggml/src"
export LD_LIBRARY_PATH="$LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OUT="${1:-$ROOT/outputs/tts_a100_interactive.wav}"
REFERENCE="${2:-}"

[[ -x "$BIN" ]] || { echo "找不到 SpaceMIT C++ 可执行文件: $BIN" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

MODEL_NAME="${MOSS_MODEL:-q4_0}"
case "$MODEL_NAME" in
  f32)  MODEL="$MODEL_DIR/moss-tts-nano-f32.gguf" ;;
  q8_0) MODEL="$MODEL_DIR/moss-tts-nano-q8_0.gguf" ;;
  q4_0) MODEL="$MODEL_DIR/moss-tts-nano-q4_0.gguf" ;;
  *) echo "MOSS_MODEL 只能是 f32/q8_0/q4_0" >&2; exit 2 ;;
esac

export MOSS_TTS_BACKEND="${MOSS_TTS_BACKEND:-cpu}"
export MOSS_TTS_THREADS="${MOSS_TTS_THREADS:-8}"
export SPACEMIT_PERFER_CORE_ARCH="${SPACEMIT_PERFER_CORE_ARCH:-0xa064}"
export SPACEMIT_PERFER_CORE_ID="${SPACEMIT_PERFER_CORE_ID:-8,9,10,11,12,13,14,15}"
export MOSS_TTS_SPACEMIT_REPACK="${MOSS_TTS_SPACEMIT_REPACK:-1}"

ARGS=(tts-nano-interactive
  --model "$MODEL"
  --codec "$MODEL_DIR/moss-audio-tokenizer-nano-f32.gguf"
  --tokenizer "$MODEL_DIR/tokenizer-sp.gguf"
  --out "$OUT"
  --max-new-frames "${MOSS_MAX_NEW_FRAMES:-32}")
[[ -n "$REFERENCE" ]] && ARGS+=(--reference "$REFERENCE")
[[ "${MOSS_GREEDY:-1}" != 0 ]] && ARGS+=(--greedy)

printf 'MOSS-TTS 常驻交互模式: model=%s threads=%s A100=%s repack=%s\n' \
  "$MODEL_NAME" "$MOSS_TTS_THREADS" "$SPACEMIT_PERFER_CORE_ID" "$MOSS_TTS_SPACEMIT_REPACK" >&2
printf '输入文字后回车生成；输出会覆盖 %s；输入 exit/quit/:q 退出。\n' "$OUT" >&2
exec "$BIN" "${ARGS[@]}"
