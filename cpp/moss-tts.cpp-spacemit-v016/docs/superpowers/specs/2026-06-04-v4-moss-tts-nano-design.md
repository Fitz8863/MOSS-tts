# V4 — MossTTSNano (offline + streaming) Design

**Status:** Approved design (brainstorming complete). Next: implementation plan
(`superpowers:writing-plans`).

**Part of:** the moss-tts.cpp program (Foundation codec + V1 Delay + V2 Local +
V3 Realtime already merged to `main`). V4 is the final family member:
**MOSS-TTS-Nano (~100M, 48 kHz stereo)**.

**Goal:** A fully-native ggml/C++17 port of MOSS-TTS-Nano in the existing
`moss-tts.cpp` repo — 48 kHz **stereo** synthesis, multilingual voice cloning,
**streaming** push-callback output, robust (non-semantic) text cleanup — with
**no Python / ONNX / torch at inference**, exposed as a C-API + `tts-nano` CLI,
benchmarked against upstream.

**Upstream sources (inspected):**
- LLM: `OpenMOSS-Team/MOSS-TTS-Nano-100M` (HF), repo
  `github.com/OpenMOSS/MOSS-TTS-Nano` (cloned at `/tmp/moss-nano-inspect`).
- Codec: `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (HF) — ships PyTorch
  `model.safetensors` + `modeling_moss_audio_tokenizer.py` +
  `configuration_moss_audio_tokenizer.py` + `config.json` (NOT ONNX-locked).

---

## 1. Architecture

Nano is a **pure autoregressive Audio-Tokenizer + LLM** pipeline with the same
**global + local (depth) two-transformer** shape as V2/V3, but a different
backbone block and a different (stereo, 48 kHz) codec.

### 1.1 The flat 17-wide row stream

Generation runs over a single autoregressive sequence of **rows**, each row
`n_vq + 1 = 17` channels wide:

- **column 0** — text channel (text token id, or `audio_user_slot` / `audio_assistant_slot` on audio rows).
- **columns 1..16** — the 16 RVQ audio codebook codes (or `audio_pad_token_id`=1024 on text rows).

This is **not** a delay pattern (V1) and **not** separate streams: the
multi-codebook factorization happens *inside the local transformer per frame*.

### 1.2 Global transformer (frame-level), GPT-2 + RoPE

From `gpt2_config`: `n_embd=768`, `n_layer=12`, `n_head=12` (MHA, `head_dim=64`,
**no GQA, no q/k-norm**), `n_inner=3072`, `activation=gelu_new`,
LayerNorm (with bias, `eps=1e-5`), `position_embedding_type=rope`, text
`vocab_size=16384`, `n_ctx=32768`.

Per-layer structure (GPT-2 block, **distinct from Qwen3**):
1. `ln_1` (LayerNorm + bias) → attention.
2. Attention: fused `c_attn` (GPT-2 `Conv1D`, **transposed weight layout** vs
   `nn.Linear`) producing q,k,v each full width → split into `n_head × head_dim`
   → **RoPE** applied to q,k (**interleaved / NeoX rotate-half**, even/odd
   pairing — NOT Qwen3's half-split) → scaled dot-product (`1/sqrt(head_dim)`) →
   causal mask → `c_proj` (Conv1D + bias).
3. Residual add.
4. `x = x + mlp(ln_2(x))` where mlp = `c_fc`→`gelu_new`→`c_proj` (no SwiGLU gate).
5. Final `ln_f`.

### 1.3 Local / depth transformer (intra-frame)

`local_transformer_layers=1` (a single GPT-2+RoPE layer, same block as the
global one, its own dims read from the checkpoint). Per frame it is seeded by the
global hidden state and:
1. emits a **text decision token** via the text head — if it equals
   `audio_assistant_slot_token_id` (9) the frame is audio → continue; any other
   token (e.g. `audio_end`=7) → **STOP**.
2. autoregressively generates the **16 audio codes** channel-by-channel: code of
   channel `c` is sampled from `audio_lm_heads[c]`, embedded, and fed to predict
   channel `c+1`.

### 1.4 Embeddings & heads

- **Input embeddings (summed):** 1 text table `wte` (16384×768) + 16 audio
  tables (each `1025×768`: 1024 codes + `audio_pad`=1024), summed per row over
  all present channels (audio channels masked when `== audio_pad_token_id`).
  17 tables total.
- **Output heads (on the LOCAL hidden):** 1 `text_lm_head` (→16384) + 16
  `audio_lm_heads[c]` (→1024 each).

### 1.5 The 48 kHz stereo "Cat" codec (MOSS-Audio-Tokenizer-Nano)

Same architectural family as the Foundation codec already in
`src/audio_tokenizer.cpp` (CNN-free, all-causal-Transformer), rescaled:

- `sample_rate=48000`, `number_channels=2` (**stereo**),
  `enable_channel_interleave=true` → `channel_interleave_factor=2` (the two
  channels are interleaved into the sequence; the LLM does NOT model L/R
  separately — the 16 codebooks describe both channels jointly).
- `downsample_rate=3840` → **12.5 Hz** frame rate (1 frame ≈ 80 ms).
- `code_dim=768`, **RVQ 16 codebooks** (`ResidualLFQ`, codebook size 1024).
- Encoder = a **multi-stage `PatchedPretransform` hierarchy**
  (`patch_size=240` → `Transformer(4 layers)` → `patch 2` →
  `Transformer(2)` → `patch 2` → `Transformer(2)` …), each transformer a causal
  RoPE attention stack with **LayerScale (init 0.01)**, LayerNorm, `max_period=10000`.
- Decoder = the mirror (un-patch / transposed pretransform + transformer
  stages) producing `[1, 2, samples]` stereo.
- **Streaming:** the codec transformers are `StreamingModule`s using a
  **`RingKVCache`** (sliding-window causal, window = `context_duration`) +
  `MossAudioTokenizerDecodeSession.step` for incremental frame-by-frame decode.

Components (upstream class names → our ggml port): `MossAudioTokenizerTransformer`/
`ProjectedTransformer` (RoPE causal stack), `PatchedPretransform` (patchify),
`ResidualLFQ` (RVQ), `LayerScale`, `RMSNorm`/LayerNorm, `RingKVCache` (streaming).

### 1.6 Tokenizer & text

- **Text:** SentencePiece (`tokenizer.model`) — a single SP model. We port it
  **natively** (no `sentencepiece` runtime dep): a converter extracts
  pieces+scores+types into GGUF; the C++ side implements **unigram Viterbi**
  segmentation (+ SP normalization / byte-fallback as needed).
- **Robust text cleanup (in scope):** port upstream `normalize_tts_text`
  (`tts_robust_normalizer_single_script.py`) — *non-semantic* only: whitespace /
  punctuation / bracket normalization, URL/email/filename protection. NO
  number/date/currency expansion (that is upstream's optional WeTextProcessing —
  **out of scope**, a documented follow-up).

---

## 2. Components (new units in the existing repo)

Each unit has one responsibility, a small interface, and a parity fixture.

| Unit | File(s) | Responsibility |
|---|---|---|
| Constants | `src/nano_constants.hpp` | `nano::` ids/dims: `N_VQ=16`, `CHANNELS=17`, `AUDIO_VOCAB=1024`, `AUDIO_PAD=1024`, `IM_START=4`, `IM_END=5`, `AUDIO_START=6`, `AUDIO_END=7`, `AUDIO_USER_SLOT=8`, `AUDIO_ASSISTANT_SLOT=9`, `PAD=3`, `SAMPLE_RATE=48000`, `N_CHANNELS=2`, `FRAME_RATE_HZ=12.5`. |
| GPT-2 layer | `src/gpt2.{hpp,cpp}` | One GPT-2+RoPE block (LayerNorm+bias, fused Conv1D `c_attn`/`c_proj`, MHA, **interleaved RoPE**, `gelu_new` MLP) + a layer-loader; reused by global + local. KV cache. `use`-flags only where global/local genuinely differ. |
| Global backbone | `src/nano_backbone.{hpp,cpp}` | 12-layer GPT-2 stack over time; `prefill(rows)`/`decode_one(row)` → last-row hidden (post `ln_f`). Persistent KV. Analogous to `DelayBackbone`. |
| Local/depth | `src/nano_local.{hpp,cpp}` | 1-layer GPT-2 depth transformer; per-frame KV reset; `step(in, pos)`→hidden; depth-0 input = global hidden (no projection). Analogous to `RtLocal`. |
| Embeddings | `src/nano_embeddings.{hpp,cpp}` | 1 text + 16 audio tables; `embed_sum(rows)` (summed, pad-masked) + `embed_audio_one(c, code)` (depth feed for audio) + `embed_text_one(id)` (depth-1 feed for the decision token). Analogous to `rt_embeddings`. |
| Heads | `src/nano_heads.{hpp,cpp}` | `text_logits(h)` (→16384, the decision token) + `audio_logits(c, h)` (→1024). On the local hidden. |
| SP tokenizer | `src/sp_tokenizer.{hpp,cpp}` | Native SentencePiece unigram: `encode(text)`/`decode(ids)` from a GGUF-embedded piece table. |
| Text cleanup | `src/text_cleanup.{hpp,cpp}` | Port of `normalize_tts_text` (non-semantic). |
| Stereo Cat codec | extend `src/audio_tokenizer.cpp` + `src/quantizer.cpp` (additively, config-driven) | Multi-stage patch hierarchy, stereo channel-interleave, 16-codebook RVQ, `encode` (cloning) + `decode_full` + streaming `decode_step` (RingKVCache). 24 kHz-mono Foundation path stays byte-identical. If the file grows unwieldy, split a `src/cat_codec.cpp` core. |
| Prompt builder | `src/prompt_nano.{hpp,cpp}` | Build the 17-wide row stream: text prefix + `audio_start` + reference-audio rows (`audio_user_slot` + ref codes) + `audio_end` + target text + assistant prefix + `audio_start`; returns prefill rows + remaining text. |
| Orchestrator | `src/moss_tts_nano.{hpp,cpp}` | `load` (LLM gguf + codec gguf + SP gguf); `tts_stream(text, opts, on_chunk_cb, userdata)`: clone-encode → prompt → prefill → frame loop → streaming codec decode → push stereo PCM. Thin `tts()` accumulates chunks. |
| Public C++ API | `include/moss_tts.h` + `src/moss_tts.cpp` | pimpl `Nano` class (mirror `Realtime`) with `tts` + `tts_stream`. |
| Flat C-API | `include/moss_tts_capi.h` + `src/moss_tts_capi.cpp` | `moss_nano_load/free`, `moss_nano_tts_stream(handle, text, ref_or_null, seed, on_chunk, userdata)` (push callback: `void(*)(const float* pcm, int n_frames, int n_channels, void* ud)`), plus a buffer-returning `moss_nano_tts` convenience. No exceptions across boundary. |
| CLI | `examples/cli/main.cpp` | `tts-nano` subcommand: `--model --codec --tokenizer --text --reference --out --seed --greedy --stream`; writes 48 kHz **stereo** WAV (streaming: append chunks as produced). |
| Converters | `scripts/convert_moss_tts_nano_to_gguf.py`, `scripts/convert_audio_tokenizer_nano_to_gguf.py`, SP→GGUF in `scripts/convert_tokenizer.py` | LLM (global gpt2 + local + 17 embeds + 17 heads; **Conv1D transpose**), the stereo codec, the SP tokenizer. Quant allowlist extension. |
| Bench | `bench_nano.sh` | RTF on `tts-nano` (copy of `bench_rt.sh`). |

**Reused as-is:** `model_loader`, `backend`/gallocr path
(`compute_graph_with_inputs`), `sampling` (text vs audio params), `audio_io`
(extended for stereo WAV write + 48 kHz), loudness norm, the converter/quant
scaffolding, the global+local frame-loop *pattern* from V2/V3.

---

## 3. Data flow

```
text ──SP encode + robust cleanup──▶ text ids
reference.wav ──load+resample 48k stereo──▶ codec.encode ──▶ ref codes [T_ref×16]
        │
        ▼
prompt_nano.build_rows(text ids, ref codes) ──▶ prefill rows [S×17] + remaining text
        │
        ▼
nano_backbone.prefill(rows) ──▶ global_hidden (last row)
        │
   ┌────▼─────────────────────────── frame loop ───────────────────────────┐
   │ # local generates the 17-channel row DEPTH-WISE (off-by-one feed):     │
   │ #   depth 0  → text decision token (text head)                         │
   │ #   depth 1..16 → the 16 audio codes (audio heads 0..15)               │
   │ nano_local.reset(); in = global_hidden            # depth 0, no proj   │
   │ h = local.step(in, 0); decision = sample(text_logits(h))              │
   │ if decision != AUDIO_ASSISTANT_SLOT: break        # stop               │
   │ in = embed_text(decision)                         # feed → depth 1     │
   │ for c in 0..15:                                                        │
   │     h = local.step(in, c+1)                                            │
   │     code[c] = sample(audio_logits(c, h), hist[c], rep-pen window)      │
   │     in = embed_audio_one(c, code[c])              # feed → depth c+2   │
   │ codec.decode_step(code[0..15]) ──▶ stereo PCM frame ──push──▶ callback │
   │ next_row = [AUDIO_ASSISTANT_SLOT, code0..code15]                       │
   │ global_hidden = nano_backbone.decode_one(embed_sum(next_row))          │
   └────────────────────────────────────────────────────────────────────────┘
   # exact depth indexing (decision-vs-code positions, embed tables per depth)
   # is the keystone-test's job to pin against the upstream _sample reference.
        │
        ▼ (CLI) accumulate / append ──▶ 48 kHz stereo WAV
```

Sampling defaults (from upstream): text temp 1.0 / top-p 1.0 / top-k 50; audio
temp 0.8 / top-p 0.95 / top-k 25 / repetition-penalty 1.2 (windowed). `--greedy`
zeroes temperatures. Long text is chunked (~75 tokens) with the same reference
prefix and concatenated — included in the orchestrator.

---

## 4. Error handling

- C++ returns `bool`; the C-API never lets exceptions cross the boundary (mirror
  `moss_rt_*`). Load failures (missing gguf, bad codec, dim mismatch) →
  `MOSS_LOGE` + `false`.
- **Load-time precondition checks** (clear messages, fail fast): global hidden ==
  local depth-0 input width; codec `num_quantizers >= N_VQ`; codec
  `number_channels == N_CHANNELS`.
- Streaming callback contract: chunks are stereo-interleaved float frames; a
  final call signals completion; if the callback returns non-zero (cancel), the
  loop stops cleanly and frees state.
- All `->data` reads guarded by `ggml_nelements`/shape (the V1/V2 OOB lesson).
- The streaming `RingKVCache` window bounds are validated; `T*N_VQ` sizing cannot
  overflow for the bounded `max_new_frames`.

---

## 5. Testing & parity methodology

Same as prior variants: pin every intricate op by a numpy/torch reference baked
into a tiny GGUF fixture; gate real-model tests on env vars (SKIP return 77).

**CI-always (tiny fixtures, no checkpoint):**
- `test_gpt2_layer` — the GPT-2 block incl. **interleaved RoPE**, LayerNorm+bias,
  fused Conv1D, `gelu_new`, MHA (exact vs numpy).
- `test_nano_embeddings` (17-sum + audio-one), `test_nano_heads` (text + 16 audio).
- `test_nano_local` (1-layer depth, per-frame KV, depth-0 injection).
- `test_sp_tokenizer` — unigram segmentation exact vs the SP reference on a tiny
  piece table; round-trip `encode∘decode`.
- `test_text_cleanup` — the robust-cleanup transformations.
- Codec: `test_cat_codec_stage` (one patch+transformer stage), `test_rvq16`
  (16-codebook dequant), `test_stereo_interleave` (channel-interleave round-trip),
  `test_codec_stream_vs_full` (streaming `decode_step` accumulation == `decode_full`).
- **Keystone `test_nano_frame_loop`** — a tiny-COMPLETE Nano model (small global +
  1-layer local + tiny codec): the full frame loop (decision token → 16 codes →
  global decode_one), **exact code match** vs a numpy `_sample` reference, incl.
  the stop-on-non-assistant-slot path.
- `test_prompt_nano` — exact row-stream parity (text prefix + ref-audio rows +
  splice) vs an upstream-derived reference.

**Env-gated (real 100M checkpoint + codec + SP model):**
- `test_nano_parity` (`MOSS_TTS_NANO` + `MOSS_NANO_REF_DUMP`) — global+local logit
  gate vs a torch dump.
- `test_e2e_nano` (`MOSS_TTS_NANO` + `MOSS_NANO_CODEC` + `MOSS_NANO_TOKENIZER`) —
  `Nano::tts(...)`; assert `sr==48000`, `channels==2`, len > 0.3 s, peak in (0.01,1].
- `test_closed_loop_nano` (+ `MOSS_PARAKEET_CLI` + `MOSS_PARAKEET_MODEL`) — synth →
  **downmix 48 k stereo → 16 k mono** → parakeet → word-recall ≥ 0.7.
- `test_nano_stream_e2e` — streaming path produces the same audio as offline
  (callback accumulation == buffer return).

**Bench:** `bench_nano.sh` (RTF, no fabricated numbers).

**Quantization:** `quantize_gguf.py` allowlist extended to quantize the gpt2
matmuls (`nano.blk.*` global + local attn/mlp weights); keep embeddings, the 17
heads, all norms, and the codec f32 (the CPU-gather/codec footguns).

---

## 6. Risks & unknowns (pinned early in the plan)

1. **Interleaved (NeoX) RoPE** — Nano rotates even/odd pairs, unlike Qwen3's
   half-split. Wrong rotation silently corrupts everything → pinned by
   `test_gpt2_layer` against a numpy interleaved-RoPE reference first.
2. **GPT-2 Conv1D transpose** — `c_attn`/`c_fc`/`c_proj` store weights transposed
   vs `nn.Linear`; the converter must transpose so ggml `mul_mat` is correct.
   Pinned by `test_gpt2_layer`.
3. **Stereo channel-interleave** — the exact interleave order (factor 2) in encode
   and decode must match `modeling_moss_audio_tokenizer.py`; pinned by
   `test_stereo_interleave` + `test_codec_stream_vs_full`.
4. **SentencePiece unigram fidelity** — segmentation must match the SP reference
   (normalization, byte-fallback, scores); pinned by `test_sp_tokenizer`.
5. **Streaming `decode_step` state** — the `RingKVCache` sliding window + per-stage
   offsets must reproduce `decode_full`; pinned by `test_codec_stream_vs_full`.
6. **Codec source availability** — RESOLVED: PyTorch weights + modeling code are
   published; no ONNX dependency.

---

## 7. Scope boundaries (YAGNI)

**In scope:** offline + streaming 48 kHz stereo synthesis, voice cloning, robust
(non-semantic) text cleanup, native SP tokenizer, push-callback C-API + `tts-nano`
CLI, converters, parity tests, bench.

**Out of scope (documented follow-ups):** semantic text normalization
(WeTextProcessing); multi-speaker / multi-turn dialogue beyond single-reference
cloning; GPU `->data` fast path (CPU-first, like the siblings); any ONNX path.

---

## 8. Provenance / reuse summary

- **New:** `gpt2` block (interleaved RoPE, Conv1D, LayerNorm), `sp_tokenizer`
  (native unigram), `text_cleanup`, the stereo/multi-stage/streaming codec
  extensions, `nano_*` units, `prompt_nano`, `moss_tts_nano`, the converters.
- **Reused:** `model_loader`, `backend`/gallocr, `sampling`, `audio_io`, loudness
  norm, the additive-codec discipline (V3-T9), the global+local frame-loop and
  tiny-complete-keystone methodology (V2/V3), the C-API/CLI/converter/quant
  scaffolding, the env-gated real-model gate pattern.
- **Commit policy:** `Assisted-by: Claude:claude-opus-4-8 [Claude Code]` trailer;
  NO `Co-Authored-By` / `Signed-off-by`.
