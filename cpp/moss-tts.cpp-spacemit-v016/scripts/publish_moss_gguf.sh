#!/usr/bin/env bash
#
# publish_moss_gguf.sh - quantize a MOSS-TTS-Local v1.5 f32 GGUF into the
# requested quants and upload the artifacts (model quants + codec + tokenizer +
# model card) to a HuggingFace repo via the `hf` CLI.
#
# This is tooling the user runs on their own hardware. Nothing here embeds a
# token or a hardcoded upload target beyond the default --repo; --dry-run prints
# every command it WOULD run without executing anything (offline-testable).
#
# usage:
#   publish_moss_gguf.sh --model MODEL_F32.gguf --codec CODEC.gguf \
#         --tokenizer TOK.gguf \
#         --repo mudler/MOSS-TTS-Local-Transformer-v1.5-GGUF \
#         [--quants "f16 q8_0 q4_k_m"] [--outdir DIR] \
#         [--llama-quantize PATH] [--card CARD.md] [--dry-run]
#
# Quant routing:
#   f16 / q8_0 / q4_0  -> scripts/quantize_gguf.py (selective: keeps ALL lc.*
#                         heads/embeds, norms and biases f32; quantizes only the
#                         qwen3.blk.* / local.blk.* attention+FFN matmuls).
#   everything else     -> llama-quantize (k-quants: q4_k_m, q5_k_m, q6_k, ...).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUANTIZE_PY="${SCRIPT_DIR}/quantize_gguf.py"

MODEL=""
CODEC=""
TOKENIZER=""
REPO="mudler/MOSS-TTS-Local-Transformer-v1.5-GGUF"
QUANTS="f16 q8_0 q4_k_m"
OUTDIR="."
LLAMA_QUANTIZE="llama-quantize"
CARD="${SCRIPT_DIR}/moss_tts_local_v1_5_card.md"
DRY_RUN=0

die() { echo "error: $*" >&2; exit 1; }

usage() {
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)          MODEL="${2:?--model needs a value}"; shift 2 ;;
        --codec)          CODEC="${2:?--codec needs a value}"; shift 2 ;;
        --tokenizer)      TOKENIZER="${2:?--tokenizer needs a value}"; shift 2 ;;
        --repo)           REPO="${2:?--repo needs a value}"; shift 2 ;;
        --quants)         QUANTS="${2:?--quants needs a value}"; shift 2 ;;
        --outdir)         OUTDIR="${2:?--outdir needs a value}"; shift 2 ;;
        --llama-quantize) LLAMA_QUANTIZE="${2:?--llama-quantize needs a value}"; shift 2 ;;
        --card)           CARD="${2:?--card needs a value}"; shift 2 ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                die "unknown argument: $1 (see --help)" ;;
    esac
done

[[ -n "$MODEL"     ]] || die "--model is required"
[[ -n "$CODEC"     ]] || die "--codec is required"
[[ -n "$TOKENIZER" ]] || die "--tokenizer is required"
[[ -n "$REPO"      ]] || die "--repo is required"

# In a real (non-dry) run the inputs must exist; dry-run stays fully offline.
if [[ "$DRY_RUN" -eq 0 ]]; then
    [[ -f "$MODEL"     ]] || die "model GGUF not found: $MODEL"
    [[ -f "$CODEC"     ]] || die "codec GGUF not found: $CODEC"
    [[ -f "$TOKENIZER" ]] || die "tokenizer GGUF not found: $TOKENIZER"
    command -v hf >/dev/null 2>&1 \
        || die "'hf' CLI not found (pip install -U huggingface_hub; the old huggingface-cli is deprecated)"
    mkdir -p "$OUTDIR"
fi

# run: echo the command; execute it unless --dry-run.
run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run] %s\n' "$*"
        return 0
    fi
    printf '+ %s\n' "$*"
    "$@"
}

produced=()
skipped=()

echo "== quantize =="
for q in $QUANTS; do
    ql="$(printf '%s' "$q" | tr '[:upper:]' '[:lower:]')"
    out="${OUTDIR%/}/moss-tts-local-v1_5-${ql}.gguf"
    case "$ql" in
        f16|q8_0|q4_0)
            # Selective python quantizer (keeps all lc.* heads/embeds + norms f32).
            if [[ "$DRY_RUN" -eq 0 ]] && ! command -v python3 >/dev/null 2>&1; then
                echo "error: python3 not found; skipping $ql" >&2
                skipped+=("$ql"); continue
            fi
            if ! run python3 "$QUANTIZE_PY" --src "$MODEL" --out "$out" --type "$ql"; then
                die "quantize_gguf.py failed for $ql"
            fi
            ;;
        *)
            # k-quants via llama.cpp's llama-quantize.
            if [[ "$DRY_RUN" -eq 0 ]] && ! command -v "$LLAMA_QUANTIZE" >/dev/null 2>&1; then
                echo "error: '$LLAMA_QUANTIZE' not found; skipping k-quant $ql" \
                     "(build llama.cpp or pass --llama-quantize PATH)" >&2
                skipped+=("$ql"); continue
            fi
            qu="$(printf '%s' "$ql" | tr '[:lower:]' '[:upper:]')"
            if ! run "$LLAMA_QUANTIZE" "$MODEL" "$out" "$qu"; then
                die "llama-quantize failed for $ql"
            fi
            ;;
    esac
    produced+=("$out")
done

echo "== upload =="
upload() {
    local f="$1"
    if [[ "$DRY_RUN" -eq 0 && ! -f "$f" ]]; then
        echo "error: cannot upload missing file: $f" >&2
        return 1
    fi
    run hf upload "$REPO" "$f" "$(basename "$f")"
}

for f in "${produced[@]}"; do upload "$f"; done
upload "$CODEC"
upload "$TOKENIZER"
if [[ "$DRY_RUN" -eq 1 || -f "$CARD" ]]; then
    run hf upload "$REPO" "$CARD" "README.md"
else
    echo "warning: card not found, skipping: $CARD" >&2
fi

echo "== summary =="
echo "repo:      $REPO"
echo "quants:    $QUANTS"
echo "produced:  ${produced[*]:-<none>}"
[[ ${#skipped[@]} -gt 0 ]] && echo "skipped:   ${skipped[*]}"
echo "codec:     $CODEC"
echo "tokenizer: $TOKENIZER"
echo "card:      $CARD  (as README.md)"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry-run: nothing was quantized or uploaded)"
fi
