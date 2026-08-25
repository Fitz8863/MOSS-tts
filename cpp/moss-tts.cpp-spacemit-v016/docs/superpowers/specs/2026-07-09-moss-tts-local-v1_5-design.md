# MOSS-TTS-Local-Transformer-v1.5 support — Design

**Date:** 2026-07-09
**Status:** Approved (brainstorming)
**Scope:** Add support for `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` to the
existing V2 (MossTTSLocal) path, **coexisting** with v1.0 (metadata-gated,
non-destructive).

## Problem

Our V2 port targets MOSS-TTS-Local-Transformer **v1.0**. v1.5 is the same
architecture family (Qwen3 global backbone + a shallow local/depth transformer +
per-channel embeddings/heads/adapters) but changes several things the current
port hardcodes to v1.0:

- **12 RVQ codebooks** (v1.0: 32). Loaders are already dynamic (load-until-missing,
  cap 32), so this is structurally fine — but the tensor **names** changed (below).
- **MOSS-Audio-Tokenizer-v2**, 48 kHz **stereo** (v1.0: 24 kHz mono Foundation
  codec). Same ResidualLFQ family; 32 codebooks decoded at depth 12.
- **`local_text_head_mode: "binary"`** — channel-0 (the continue/stop decision) is
  a dedicated 2-wide `local_text_lm_head` instead of the full-vocab text head.
  This is exactly the Nano "decision token" pattern.
- **Special tokens changed:** `audio_start=151669`, `audio_end=151670`
  (v1.0: 151652/151653; C++ hardcodes those in `delay_constants.hpp`). No
  delay-slot token (Local ≠ Delay).
- **New structured prompt template** (`<user_inst>` with Reference/Instruction/
  Language/Text fields) vs v1.0's simpler template.
- **1-layer local**, 5B / 2560-hidden / 36-layer backbone — both read from metadata,
  no work.

Confirmed non-issues:
- **Pause markers `[pause 3.2s]`**: the v1.5 processor has **no** pause handling —
  the marker is literal text the trained model understands. Nothing to port.
- **Voice-cloning consistency**: a weights-quality property; our reference-codes →
  prefill path already exists. Nothing to port.
- **Language tags**: a plain text field in the prompt template, not a token.

## Goal

Synthesize speech from text with v1.5 (converter → GGUF → C++ inference → 48 kHz
stereo wav), full-parity with the v1.5 processor's structured prompt (Reference /
Instruction / Language / Text) and the binary decision head — **coexisting** with
v1.0 so its tiny-fixture tests stay green. Offline validation is
correct-by-construction + tiny synthetic fixtures; real-model end-to-end parity is
env-gated (needs the ~5 GB v1.5 + v2 tokenizer + torch).

Non-goal: dropping/rewriting v1.0; a from-scratch new engine; the codec math
(reused as-is); pause-marker special handling (none needed).

## Architecture — metadata-gated v1.5 mode

v1.5 *is* MossTTSLocal. Rather than a new engine, the converter stamps GGUF
metadata flags and the existing C++ Local path branches on them. When the flags
are absent (a v1.0 GGUF), the path behaves exactly as today. The loaders are
already dynamic (`lc.embed.{i}`/`lc.lm_head.{i}` load-until-missing, dims from
metadata), so codebook count and dims need no code change. New metadata keys:

| key | v1.5 value | v1.0 (absent → default) |
|-----|-----------|--------------------------|
| `lc.local_text_head_mode` | `1` (binary) | absent → `0` (full-vocab) |
| `lc.stereo` | `1` | absent → `0` (mono) |
| `lc.audio_start_token_id` | 151669 | already written; C++ now reads it (was hardcoded 151652) |
| `lc.audio_end_token_id` | 151670 | already written; C++ now reads it (was 151653) |
| `lc.prompt_template` | `1` (v1.5 `<user_inst>`) | absent → `0` (v1.0 template) |
| `lc.n_vq` / `lc.sample_rate` | 12 / 48000 | already dynamic |

`lc.audio_user_slot_token_id` (151654) and `lc.audio_assistant_gen_slot_token_id`
(151656) are already written by the converter and unchanged between v1.0/v1.5.

## Components

### A. Model converter — `scripts/convert_moss_tts_local_to_gguf.py` (v1.5 mode)

v1.5 renamed the modules; add v1.5 source-key mappings alongside the v1.0 ones
(the converter already maps per-regex, so add regexes; a key matched by neither is
reported and fails `--strict`). Expected v1.5 keys (VERIFY against the real
`model.safetensors` keys on first real run — `--strict` + the unmapped-key report
will catch any prefix/name drift):

| v1.5 source (expected) | output GGUF name |
|---|---|
| `…audio_embeddings.{c}.weight` (c 0..11) | `lc.embed.{c+1}.weight` (audio) |
| the text embedding (`model.embedding_list.0` **or** `text_embeddings`) | `lc.embed.0.weight` |
| `…audio_lm_heads.{c}.weight` (c 0..11) | `lc.lm_head.{c+1}.weight` (audio) |
| the full text head (`text_lm_head`) | `lc.lm_head.0.weight` |
| **`…local_text_lm_head.weight`** (2×hidden, NEW) | **`lc.local_text_head.weight`** |
| in-MLP (`speech_embedding_to_local_mlp.*`) | `lc.in_mlp.{gate,up,down}.weight` |
| out-MLPs (`local_to_speech_embedding_mlps.{c}.*`) | `lc.out_mlp.{c}.{…}.weight` |
| head-norms (`layer_norm_before_lm_heads.{c}.*`) | `lc.head_norm.{c}.weight` |
| global `model.language_model.*`, `local_transformer.*` | `qwen3.*` / `local.*` (unchanged) |

Stamp the metadata flags from the table above. Read `local_text_head_mode` from
config (`"binary"` → 1). Local dims: keep deriving local head_dim/n_heads/n_kv_heads
from the real `local_transformer.layers.0.self_attn.{q,k}_proj` shapes; read
`local_hidden_size`/`local_num_layers`/`local_ffn_hidden_size` from config, with a
fallback to the alternative key names v1.5 may use (verify on first run). Keep the
existing v1.0 mappings intact so v1.0 checkpoints still convert.

The design keeps a **single converter** with additive v1.5 regexes + flags (not a
fork), so v1.0 and v1.5 both convert; the emitted GGUF self-describes via metadata.

### B. Codec converter — MOSS-Audio-Tokenizer-v2

Reuse `scripts/convert_audio_tokenizer_nano_to_gguf.py` (metadata-driven stage
tables, ResidualLFQ, 48 kHz stereo, downsample 3840). MOSS-Audio-Tokenizer-v2 has
32 quantizers / codebook_size 1024 / codebook_dim 8; the converter reads these
from config. Verify it converts v2 clean (no unmapped keys); adapt only if a
stage/module name differs. The C++ codec decodes v1.5's 12 codes at **depth 12**
via the existing partial-`first-k` `dequantize`/`decode` path (no codec-math
change). Output name: `convert_audio_tokenizer_nano_to_gguf.py` gains a note / or a
thin `--v2` alias; the GGUF is architecturally identical.

### C. Tokenizer — v1.5 → GGUF

Convert v1.5's `tokenizer.json` / `vocab.json` / `merges.txt` via the existing
tokenizer-conversion path (the Qwen3 BPE converter used for v1.0), producing a
GGUF whose special-token ids match v1.5 (audio_start/end at 151669/70, etc.). The
C++ `de_tokenizer` loads it unchanged (it's the same BPE format); only the ids
differ, and those are now metadata-driven (D).

### D. C++ inference deltas (metadata-gated, in the Local path)

1. **Metadata-driven special tokens.** `moss_tts_local` + `prompt_local` read
   `lc.audio_start_token_id` / `lc.audio_end_token_id` (and the slots) from the
   loaded GGUF metadata, falling back to the `de::` constants when the keys are
   absent (v1.0). Add a small accessor on the loaded model (e.g. `SpecialTokens`
   struct populated at load) so the prompt builder and the stop-check use the same
   source. The stop-check `next[0] == AUDIO_END` uses the metadata value.

2. **Binary channel-0 decode.** `LMHeads` (or a small `LocalTextHead`) gains a
   `local_text_head_` (2-wide) loaded from `lc.local_text_head.weight` when
   present. In the local decode loop, when `local_text_head_mode==binary`, channel
   0's logits come from that 2-wide head → argmax over {slot, end} → map index→id
   (`[audio_assistant_gen_slot_id, audio_end_id]`), stop on `audio_end`, else feed
   back the **slot-token** embedding for the next frame's channel-0 input. This
   mirrors `moss_tts_nano`'s decision-token step. When the mode flag is absent
   (v1.0), keep the current full-vocab channel-0 head + `next[0]==AUDIO_END` stop.
   (The v1.5 `lc.lm_head.0` full text head may still be emitted; in binary mode the
   decision uses `lc.local_text_head` and `lc.lm_head.0` is unused — the loader
   loads whichever exist.)

3. **Stereo output.** When `lc.stereo==1`, `moss_tts_local` decodes the codec to
   interleaved 48 kHz stereo and saves 2-channel wav, reusing the Nano stereo path
   (`audio_io` stereo save + the codec's stereo decode). Mono (v1.0) unchanged.

4. **Codec depth-12 decode.** `moss_tts_local` calls `codec_->decode(codes, T, …)`
   with the 12-deep codes; the codec (32 quantizers) sums the first 12 via the
   existing `first-k` path. The load-time quantizer-count check becomes "codec
   num_quantizers **≥** n_vq" (was `==`) so a 32-codebook codec serves a 12-code
   model.

### E. Prompt builder — `prompt_local.cpp` (v1.5 template, gated)

When `lc.prompt_template==1`, build v1.5's structured prompt: `im_start` +
`"user\n<user_inst>\n- Reference(s):\n"` + (reference audio blocks: `audio_start` +
reference-code rows tagged `audio_user_slot` + `audio_end`) + `"- Instruction:\n"`
+ instruction + `"- Language:\n"` + language + `"- Text:\n"` + text +
`"\n</user_inst>"` + `im_end` + `"\nassistant\n"` + `audio_start`. Text (including
any `[pause 3.2s]` literal) is BPE-encoded with no normalization (the v1.5
processor applies none). `language` and `instruction` come from the existing
`LocalParams` fields (already present: `language`, `instruction`). When the flag is
absent, the v1.0 builder is used unchanged.

## Data flow (v1.5 synthesis)

```
text (+ optional reference wav, language, instruction)
  → prompt_local (v1.5 <user_inst> template, metadata special tokens)   [E]
  → embeddings (sum over 13 channels) + global Qwen3 prefill
  → per-frame local decode:
       channel 0: local_text_head (2-wide) → {slot|end}; end ⇒ stop      [D2]
       channels 1..12: audio_lm_heads → codes; feed back per-channel emb
  → 12-deep codes → codec decode (depth-12, 48 kHz STEREO)               [D3,D4]
  → interleaved stereo wav
```

## Testing

- **Tiny synthetic v1.5 fixtures** (new), same methodology as the port: a
  mini v1.5 Local GGUF (1-layer local, 13 channels = 1 text + 12 audio, a 2-wide
  binary `lc.local_text_head`, the metadata flags) + a small 48 kHz stereo codec
  fixture. Pin, byte-exact against a numpy reference:
  - `test_local_v15_binary_head` — the 2-wide decision head + the map-to-{slot,end}
    + stop logic (a keystone like `test_depth_loop`, exact codes).
  - `test_prompt_local_v15` — the `<user_inst>` template token sequence (incl. a
    reference block + language field) vs a numpy/hand reference.
  - stereo output covered by the existing `audio_io_stereo` + a
    `test_local_v15_stereo` decode-shape check.
- **Coexistence:** the full existing suite stays green (v1.0 flags absent →
  unchanged behavior). The metadata-token change falls back to `de::` defaults, so
  the current `test_prompt_local` / `test_depth_loop` are byte-identical.
- **Converter:** a fixture-generation script produces the tiny v1.5 GGUF; the
  converter's `--strict` reports any unmapped v1.5 key on a real run.
- **Env-gated real gate (user-hardware):** `convert_moss_tts_local_to_gguf.py
  --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`, `convert_audio_tokenizer_
  nano_to_gguf.py --model OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`, the tokenizer
  convert, then `moss-tts-cli tts-local … --text …` → a real 48 kHz stereo wav +
  `bench local` for RTF. A `test_local_v15_parity` (env-gated, returns 77 without
  the real dump) compares against a torch-dumped reference. **This is the only true
  end-to-end proof and it is user-gated.**

## File structure

| File | Change |
|------|--------|
| `scripts/convert_moss_tts_local_to_gguf.py` | additive v1.5 regexes + the binary head + metadata flags |
| `scripts/convert_audio_tokenizer_nano_to_gguf.py` | verify/allow MOSS-Audio-Tokenizer-v2 (note/alias) |
| `scripts/gen_test_fixtures.py` (or a new gen script) | emit the tiny v1.5 model + stereo codec fixtures |
| `src/delay_constants.hpp` | keep as v1.0 defaults; document they are fallbacks |
| `src/moss_tts_local.{hpp,cpp}` | metadata special tokens; binary channel-0 decode; stereo output; codec depth-k; `≥` quantizer check |
| `src/lm_heads.{hpp,cpp}` (or a new `local_text_head`) | load + apply the 2-wide `lc.local_text_head` |
| `src/prompt_local.{hpp,cpp}` | v1.5 `<user_inst>` template (gated) |
| `src/model_loader` / a `SpecialTokens` accessor | read `lc.*_token_id` + the flags from metadata |
| `tests/test_local_v15_binary_head.cpp`, `test_prompt_local_v15.cpp`, `test_local_v15_stereo.cpp` | new keystones |
| `tests/CMakeLists.txt` | register the new tests |
| `AGENTS.md` | document v1.5 support + the metadata flags + the user-gated real gate |

## Risks & mitigations

- **Exact v1.5 tensor names unknown** (single-shard, no index; inferred from the
  modeling attribute names). Mitigation: the converter's `--strict` + unmapped-key
  report pins them on the first real run; the mapping table above is the starting
  hypothesis, adjusted then. The tiny-fixture generator uses OUR output names, so
  the C++ side is validated regardless.
- **v2 codec RLFQ vs our ResidualLFQ**: high-confidence same family (same
  downsample/stereo/codebook_size), but real-weights parity is user-gated. If a
  math delta surfaces, it is contained to `quantizer.cpp` (a separate follow-up).
- **Coexistence branches** could tangle the Local path. Mitigation: gate on a small
  set of metadata flags read once at load into a struct; keep v1.0 the default
  (flags absent) so existing tests are the regression guard.
- **No real end-to-end validation here**: stated ceiling — offline is
  correct-by-construction + tiny-fixture-exact; the real gate is env-gated, same as
  the rest of the port.
