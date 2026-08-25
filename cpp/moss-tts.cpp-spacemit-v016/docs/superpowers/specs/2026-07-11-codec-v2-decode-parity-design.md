# MOSS-Audio-Tokenizer-v2 decode parity — Design

**Date:** 2026-07-11
**Status:** Proposed (brainstorming, approved)
**Scope:** Make the v2 codec DECODE (audio codes -> 48 kHz stereo waveform) load and
run, and verify it against the real model with a parity gate. Unblocks the v1.5
`tts-local` end-to-end listen. Decode-only; encode + quantizer math are out of scope
(already correct).

## Problem

`moss-tts-cli tts-local` segfaults in the v2 codec decode. ASan root-caused it to a
null-tensor `ggml_mul_mat` at `transformer.cpp:12` (attention) reached via
`run_stage` (audio_tokenizer.cpp:155) <- `decode_block` <- `LocalTTS::tts:307`
(`codec_->decode`). `load_transformer` (`src/transformer.cpp:88-96`) reads the
decoder attention/FFN tensors under LEGACY Foundation-codec names that do not exist
in the v2 GGUF, so the weights load as null:

| C++ `load_transformer` reads | v2 GGUF actually has |
|---|---|
| `...self_attn.in_projs.0.weight` | `...self_attn.in_proj.weight` (fused QKV [1280,3840]) |
| `...self_attn.out_projs.0.weight` | `...self_attn.out_proj.weight` |
| `...linear1.weight` | `...ffn.0.weight` |
| `...linear2.weight` | `...ffn.2.weight` |

An architecture investigation (against the real `modeling_moss_audio_tokenizer.py`,
the built GGUF, and our C++) confirmed **this is the ONLY code delta**. The rest of
the v2 decode is already implemented and matches the real model:
- Quantizer dequant (`quantizer.cpp:44-59`): `sum_i out_proj_i(codebook_i[code_i])`
  then `output_proj` — matches `MossAudioTokenizerResidualLFQ.decode_codes`; NO
  L2-norm in decode (that is encode-only). Names match.
- Decoder: 12 modules (6 transformer stages + 5 patch-up + a final patch-up),
  per-stage `input_proj`/`output_proj`, RoPE base 10000, sliding-window causal
  `context`, `layer_scale_1/2`, exact-erf GELU, FFN with no bias. All match.
- Patch-up (`patchify.cpp`) matches `PatchedPretransform.decode`.
- Stereo: our `decode` returns the flat interleaved buffer (per_frame = 7680 =
  downsample 3840 * 2 channels), which is WAV-ready L,R,L,R order. No change needed.

The delta was hidden because the offline `nano_codec` fixture
(`gen_test_fixtures.py:677-680`) emits the transformer tensors under the SAME legacy
names, so fixture + loader agree on the wrong names -> the test is green while the
real v2 model segfaults.

## Goal

1. v2 codec decode loads + runs (name-tolerant loader).
2. A parity gate proves the v2 decode numerically matches the real model
   (dequant + per-stage + final interleaved waveform).
3. The v1.5 `tts-local` listen produces a 48 kHz stereo WAV.

Keep the Foundation 24 kHz codec + the `nano_codec` fixture byte-identical.

## Architecture

### 1. Name-tolerant loader (`src/transformer.cpp::load_transformer`)

For the four attention/FFN lookups, prefer the v2 canonical name and fall back to
the legacy Foundation name (a tiny `first_present(a, b)` helper):

```cpp
auto pick = [&](const std::string& a, const std::string& b) {
    struct ggml_tensor* t = ld.tensor(a); return t ? t : ld.tensor(b);
};
L.qkv_w = pick(b + "self_attn.in_proj.weight",  b + "self_attn.in_projs.0.weight");
L.out_w = pick(b + "self_attn.out_proj.weight", b + "self_attn.out_projs.0.weight");
L.lin1_w = pick(b + "ffn.0.weight", b + "linear1.weight");
L.lin2_w = pick(b + "ffn.2.weight", b + "linear2.weight");
```

Both the v2 codec (in_proj/ffn) and the Foundation codec + nano_codec fixture
(in_projs.0/linear1/2) load correctly. No converter change (the converter correctly
emits v2 verbatim names); fix its stale docstring only.

### 2. Codec-v2 parity harness

`scripts/gen_codec_v2_reference.py` (torch dumper, lazy imports, `--help`/AST clean
without torch):
- Load real `MossAudioTokenizerModel` via `trust_remote_code`, fp32, deterministic
  attention.
- FIXED seeded codes `(nq=32, B=1, T)` (small T, e.g. 8).
- Dump to a reference GGUF: `ref.codes` (i32), `ref.dequant` (768xT), per-module
  `ref.dec_stage.{0..11}`, `ref.audio_flat` (the interleaved decode output = what
  C++ `decode` returns), metadata (`ref.nq=32`, `ref.n_frames`, `ref.downsample=3840`,
  `ref.channels=2`).

`tests/test_codec_v2_parity.cpp` (env-gated, returns 77 without the real GGUF + ref):
- Load the real v2 codec GGUF (`MOSS_CODEC_V2`) + the ref (`MOSS_CODEC_V2_REF`).
- Assert `dequantize(codes)` == `ref.dequant` (tight tol).
- Assert `decode(codes, n_frames, nq=32)` == `ref.audio_flat` (max-abs + SNR >= 60 dB;
  f32 deterministic).
- Print per-stage/overall maxerr; stage-gate `ref.dec_stage.{k}` if exposable, else
  the dequant + final-waveform gate localizes to quantizer vs decoder.

### 3. Offline coverage

Add a v2-named variant to the codec block coverage so CI exercises the
name-tolerance without the real model: either extend `gen_test_fixtures.py` with a
tiny transformer-stage fixture under the v2 names (`self_attn.in_proj`, `ffn.0/2`)
and a test asserting `load_transformer` resolves them, or add a second fixture GGUF.
The existing `nano_codec` fixture (legacy names) must still pass unchanged.

### 4. Live verification

Codec GGUF already built (`models/moss-audio-tokenizer-v2-f32.gguf`, 0 unmapped).
Steps: gen reference -> run `test_codec_v2_parity` -> `tts-local ... --out out.wav`
(48 kHz stereo). The parity gate is the verification; the listen is the e2e confirm.

## Environment resilience

The box runs an external cleaner that deletes `build/`, `.venv/`, and
`~/.cache/huggingface/` mid-session (all gitignored; the git repo + `models/*.gguf`
survive). The plan's setup step rebuilds (`cmake -B build -DMOSS_TTS_BUILD_TESTS=ON
-DMOSS_TTS_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release`), recreates the venv
(torch cpu + `transformers~=4.57` + gguf + numpy + safetensors), and re-downloads the
codec when a Python/torch step needs it. Offline C++ work (the loader fix + offline
fixture) does not need the venv.

## Testing

- Offline: the name-tolerance fixture/test + the full existing suite stay green
  (Foundation codec + nano_codec byte-identical; env-gated real-model tests -> 77).
- Live: `test_codec_v2_parity` on the real weights is the verification.

## File structure

| File | Change |
|---|---|
| `src/transformer.cpp` | name-tolerant `load_transformer` (v2 names + legacy fallback). |
| `scripts/gen_codec_v2_reference.py` | new torch reference dumper (real decode). |
| `tests/test_codec_v2_parity.cpp` | new env-gated numeric parity gate. |
| `tests/CMakeLists.txt` | register the parity test. |
| `scripts/gen_test_fixtures.py` + a test | offline v2-name coverage for load_transformer. |
| `scripts/convert_audio_tokenizer_nano_to_gguf.py` | fix stale docstring (v2 names). |
| `AGENTS.md` | note the codec-v2 decode parity gate in the runbook. |

## Risks & mitigations

- **Residual numeric mismatch on the real run** (RoPE base, window `context`
  rounding, patch reshape): the per-stage + dequant gates localize it to one stage;
  the investigation found no expected mismatch, so this is a confirmation, not a fix.
- **Environment wipes mid-run**: the setup step is idempotent; re-run it if `build/`
  or `.venv/` vanish.
- **Legacy fallback masking a real miss**: `pick()` still returns null if BOTH names
  are absent -> the existing null-deref would resurface loudly at the same site
  (acceptable; a genuinely missing weight is a real error).
