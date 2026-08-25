# V1: MossTTSDelay (8B) → ggml

**Status:** approved design — ready for implementation planning
**Date:** 2026-06-03
**Sub-project:** V1 of the MOSS-TTS → ggml porting program. Depends on the
completed Foundation (F1+F2): the native MOSS-Audio-Tokenizer codec.

## Context

The Foundation milestone delivered a native ggml **MOSS-Audio-Tokenizer**
(`moss::AudioTokenizer` / `moss::Codec`): encode 24 kHz wav → 32 RVQ code
streams, decode codes → wav. V1 builds the **MossTTSDelay** flagship on top:
a Qwen3-8B language model that autoregressively generates those RVQ codes
from text (with optional zero-shot voice cloning from a reference wav),
which the codec then decodes to speech.

Upstream ships a "torch-free" reference pipeline at
`moss_tts_delay/llama_cpp/` (Qwen3 via llama.cpp + NumPy glue + ONNX codec)
with published Seed-TTS-eval WER numbers. V1 is a fully-native ggml port of
that pipeline: no Python/ONNX/torch at inference, one `libmoss-tts` +
`moss-tts-cli tts`.

This is the only variant with an existing reference implementation to
benchmark against, which is why it is first among the variants.

## Goals / non-goals

**Goals**
- Native ggml **Qwen3-8B backbone** (GQA, per-head q/k RMSNorm, RoPE, SwiGLU,
  KV cache) that consumes precomputed input embeddings and emits the hidden
  state (bypassing the built-in tok-embd and lm-head).
- The **MossTTSDelay generation stack**: 33 embedding tables (sum lookup),
  33 LM heads, the delay-pattern state machine, temperature/top-k/top-p/
  repetition-penalty sampling, the Qwen3 BPE tokenizer, and the prompt
  builder.
- **Zero-shot voice cloning**: encode a reference wav via the Foundation
  codec, delay-pattern the codes, and splice them into the prompt.
- A **converter**: HF MossTTSDelay safetensors → one GGUF (backbone + 33
  embeddings + 33 heads + metadata) + a tokenizer GGUF, with backbone
  quantization.
- `moss-tts-cli tts --text "..." [--reference ref.wav] --out out.wav` and a
  `Delay` C++/C API.

**Non-goals (deferred)**
- The other variants (V2 Local, V3 Realtime, V4 Nano).
- Streaming generation (offline full-utterance first).
- Training / fine-tuning.
- Exact token-for-token reproduction of upstream *stochastic* sampling
  (numeric gates use deterministic/greedy forward; the closed-loop gate
  seeds and accepts a WER threshold).

## Reference sources (single sources of truth)

1. **`OpenMOSS/MOSS-TTS` `moss_tts_delay/`** — the trained PyTorch modeling
   (`modeling_moss_tts.py`, `configuration_moss_tts.py`) and the torch-free
   reference (`moss_tts_delay/llama_cpp/`: `embedding.py`, `lm_heads.py`,
   `delay_state.py`, `sampling.py`, `processor.py`, `_constants.py`,
   `backbone.py`, `conversion/extract_weights.py`). **Authoritative on
   weights, the delay state machine, the prompt format, and token IDs.**
2. **`OpenMOSS-Team/MOSS-TTS`** HF checkpoint (safetensors + config.json) —
   the Qwen3 `language_config` dims + the 33 emb/head tensors.
3. **`OpenMOSS-Team/MOSS-TTS-GGUF`** — upstream's pre-quantized Qwen3
   backbone GGUF (Q4_K_M) + emb/head npy. Cross-check our backbone against
   llama.cpp loading this.
4. **`OpenMOSS-Team/MOSS-Audio-Tokenizer`** — already ported (Foundation);
   reused here for encode (cloning) + decode (synthesis).
5. `vibevoice.cpp` `src/qwen2.cpp` + `src/tokenizer.cpp` — closest existing
   ggml references to adapt for the Qwen3 backbone and the BPE tokenizer.

## Architecture

### Top-level data flow

```
text [+ reference.wav]
  │ (cloning) reference.wav → Codec.encode → ref codes (T_ref,32)
  │           → apply_delay_pattern → splice into prompt audio channels
  ▼
Tokenizer (Qwen3 BPE) + build_generation_prompt → input_ids (S, 33)
     ch0 = text token id; ch1..32 = audio VQ codes (audio_pad_code where no audio)
  ▼
DelayEmbeddings.embed(input_ids) :
     embeds[s] = embed_tokens[ch0[s]] + Σ_{i=0..31} emb_ext[i][ch_{i+1}[s]]   → (S, hidden)
  ▼
Qwen3 backbone:
     prefill the S embeds (build KV cache) → hidden_S (after final RMSNorm)
     then autoregressive: feed one (33,)-token's summed embedding per step → hidden_t
  ▼
LM heads (33):
     text_logits   = hidden @ lm_head_text^T            → (text_vocab,)
     audio_logits  = hidden @ stack(lm_head_audio_i)^T  → (32, audio_vocab=1025)
     audio_logits[:, audio_pad_code=1024] = -inf
  ▼
Delay state machine + sampling (port of delay_state.py / sampling.py):
     decide next text token + next 32 audio codes (the diagonal staircase),
     with the pre/post-audio masks and the slot/end/im_end rules.
  ▼ loop: embed the new (33,) token, decode 1 step, until im_end; drain the
         n_vq-step delay staircase.
de_delay_pattern + extract_audio_segments → codes (T, 32)
  ▼
Codec.decode(codes) → 24 kHz wav
```

### Special token IDs and constants (from `_constants.py` / config.json)

| name | value | role |
|------|------:|------|
| n_vq | 32 | audio codebooks / channels |
| pad_token_id | 151643 | text pad |
| im_start_token_id | 151644 | chat turn start |
| im_end_token_id | 151645 | generation EOS |
| audio_start_token_id | 151652 | start of audio span |
| audio_end_token_id | 151653 | end of audio span (audio EOS) |
| audio_user_slot_token_id | 151654 | user-side audio slot |
| audio_assistant_gen_slot_token_id | 151656 | assistant audio gen slot |
| audio_assistant_delay_slot_token_id | 151662 | delay-staircase slot |
| audio_pad_code | 1024 | audio VQ pad value |
| audio_vocab_size | 1024 | real audio codes 0..1023 |
| sampling_rate | 24000 | codec sample rate |

Embedding tables and audio heads are **1025-wide** (codes 0..1023 + pad
code 1024). The audio head masks index 1024 to −inf before sampling.

### Qwen3-8B backbone (`qwen3.{hpp,cpp}`)

A standard Qwen3 decoder, adapted from vibevoice's `qwen2.cpp`. Differences
from Qwen2 to implement: **per-head RMSNorm on Q and K** before RoPE
(`q_norm`/`k_norm`, head_dim-wide), and **no attention bias**. Otherwise:
RMSNorm (eps from config), GQA (n_heads / n_kv_heads), RoPE (NEOX, base from
config), SwiGLU FFN (gate/up/down), pre-norm residual blocks, final RMSNorm.

Two entry points the orchestrator needs:
- `prefill(embeds (S,hidden)) → hidden (S,hidden)` — builds the KV cache for
  the prompt.
- `decode_one(embed (hidden,)) → hidden (hidden,)` — one autoregressive step
  appending to the KV cache.

All dims (hidden, n_layers, n_heads, n_kv_heads, head_dim, intermediate,
rope_base, rms_eps, text_vocab) come from GGUF metadata written by the
converter from the Qwen3 `language_config`. The backbone consumes
precomputed embeddings (no internal tok-embd) and returns the post-final-
RMSNorm hidden state (no internal lm-head) — matching upstream `backbone.py`.
KV cache + the gallocr compute path (Foundation's `backend.cpp`) are reused.

### DelayEmbeddings (`delay_embeddings.{hpp,cpp}`)

Holds `embed_tokens` (text_vocab × hidden) and 32 `emb_ext[i]`
(1025 × hidden). `embed(input_ids (S,33)) → (S,hidden)` = text-row lookup +
sum of 32 audio-row lookups (port of `embedding.py::_lookup`, exact sum
order). Implemented with `ggml_get_rows` + adds, or a CPU gather (these are
lookups, not matmuls). Used for both prefill (S rows) and per-step (1 row).

### LM heads (`lm_heads.{hpp,cpp}`)

`lm_head_text` (text_vocab × hidden) and 32 `lm_head_audio[i]` (1025 × hidden,
concatenated to `(32*1025, hidden)` for one matmul). `logits(hidden) →
(text_logits (text_vocab), audio_logits (32,1025))`; set
`audio_logits[:,1024] = -inf` (port of `lm_heads.py`).

### Delay state machine (`delay_state.{hpp,cpp}`)

Direct C++ port of `delay_state.py` (the authoritative source). Carries
`DelayState { audio_length, delayed_length, is_audio, is_stopping,
time_step, text_history, audio_history buffer }`. Key functions:
- `init_delay_state(input_ids (S,33)) → DelayState` (detects audio
  continuation from the last text channel token).
- `step(state, text_logits, audio_logits, SamplingConfig) → next (33,)`:
  the text-token decision (delay-slot vs audio-end vs sampled with the
  `_PRE_EXCLUDE_IDS` / `_AUDIO_ALLOWED_IDS` masks and the `time_step` guards),
  the audio decision (the `pre_audio_mask & post_audio_mask` sampling mask,
  channel-0 vs rest), and the state updates (`audio_length`,
  `delayed_length` staircase wrapping to INT64_MAX after `> n_vq`).
- `apply_delay_pattern` / `apply_de_delay_pattern` / `extract_audio_segments`
  (the diagonal shift used both for splicing reference codes and for
  de-staircasing the generated codes).

This module is pure logic over logits (no ggml); it is the most intricate
piece and gets dedicated step-sequence parity tests vs the python.

### Sampling (`sampling.{hpp,cpp}`)

Port of `sampling.py::sample_token`: temperature scale, repetition penalty
(over previous tokens), top-k, top-p (nucleus), multinomial draw; `do_sample`
false (or temp 0) → argmax. `SamplingConfig` defaults: text temp 1.5 / top_p
1.0 / top_k 50; audio temp 1.7 / top_p 0.8 / top_k 25 / rep_penalty 1.0.
The RNG is an explicit, seedable generator owned by the orchestrator so
`--seed` makes generation reproducible (a deterministic generator we control,
not numpy's — exact-vs-numpy parity is a non-goal; the unit test pins our own
generator).

### Tokenizer (`tokenizer.{hpp,cpp}`) and prompt builder (`prompt.{hpp,cpp}`)

Qwen3 byte-level BPE, adapted from vibevoice's `tokenizer.cpp` (Qwen2 BPE is
the same family), loaded from a tokenizer GGUF. `prompt.cpp` ports
`processor.py::build_generation_prompt`: the `<user_inst>` template
(Reference / Instruction / Tokens / Quality / Sound Event / Ambient Sound /
Language / Text fields), the audio-placeholder expansion
(`gen_slot*length + delay_slot*(n_vq-1)`), the `im_start user … im_end
im_start assistant audio_start` framing, and `_get_unified_codes` (packing
text ids into channel 0 and delay-patterned reference codes into channels
1..32, audio_pad_code elsewhere). Output: `input_ids (S,33)`.

### TTS orchestrator (`moss_tts_delay.{hpp,cpp}`)

`class Delay`: `load(backbone_gguf, codec_gguf, tokenizer_gguf)`;
`tts(text, opts{reference_wav?, language?, instruction?, seed?,
SamplingConfig?}) → wav`. Ties the pipeline: (cloning) `Codec.encode` →
prompt build → `DelayEmbeddings.embed` → `Qwen3.prefill` → loop{`LM heads` →
`delay_state.step` → embed next → `Qwen3.decode_one`} until im_end →
de-delay → `Codec.decode`. Exposed via the public C++ API (`moss::Delay`),
the flat C-API, and `moss-tts-cli tts`.

## Converter + GGUF inventory

`scripts/convert_moss_tts_delay_to_gguf.py` (mirrors upstream
`conversion/extract_weights.py` but targets one GGUF):

- **Backbone**: remap `language_model.*` → Qwen3 GGUF tensor names
  (`qwen3.blk.{i}.attn_{q,k,v,o}.weight`, `attn_{q,k}_norm.weight`,
  `attn_norm.weight`, `ffn_{gate,up,down}.weight`, `ffn_norm.weight`,
  `output_norm.weight`). Emit Qwen3 metadata (hidden, n_layers, n_heads,
  n_kv_heads, head_dim, intermediate, rope_base, rms_eps, text_vocab).
- **Embeddings**: `embed_tokens.weight` → `de.embed_tokens.weight`;
  `emb_ext.{i}` → `de.emb_ext.{i}.weight` (i 0..31, 1025×hidden).
- **LM heads**: `lm_heads.0.weight` (text) → `de.lm_head_text.weight`;
  `lm_heads.{i+1}` → `de.lm_head_audio.{i}.weight` (i 0..31, 1025×hidden).
  (Upstream `remap_backbone_name` maps `lm_heads.0` → `lm_head.weight`; here
  all 33 heads are emitted under `de.*` and the backbone has no internal
  lm-head.)
- **Metadata**: `moss.de.n_vq=32`, the special-token IDs table above,
  `moss.de.audio_vocab=1025`, `moss.de.sample_rate=24000`, sampling defaults.
- **Tokenizer**: `scripts/convert_tokenizer.py` (adapt vibevoice's) →
  tokenizer GGUF from the HF `tokenizer.json`.
- `--strict` (fail on unmapped backbone keys); `scripts/quantize_gguf.py`
  extended to quantize the backbone matmuls (Q8_0/Q4_K…) while leaving the
  embedding/head tables and norms at the safe types (mirrors upstream
  Q4_K_M backbone). The Foundation's quantize allowlist policy applies.

The codec GGUF is the Foundation's `moss-audio-tokenizer-*.gguf` (unchanged).

## Components and boundaries

| unit | does | depends on |
|------|------|-----------|
| `qwen3` | Qwen3 decoder + KV cache; embeds→hidden | backend, model_loader, ggml |
| `delay_embeddings` | 33-table sum lookup → input embeds | model_loader, ggml |
| `lm_heads` | hidden → text+audio logits | model_loader, ggml |
| `delay_state` | the staircase state machine (pure logic) | sampling |
| `sampling` | temp/top-k/p/rep-penalty draw | — (seedable RNG) |
| `tokenizer` | Qwen3 BPE | model_loader |
| `prompt` | build_generation_prompt (S,33) | tokenizer, delay_state |
| `moss_tts_delay` | orchestrate text[+ref]→wav | all above + Codec (Foundation) |

Each unit is independently testable on tiny fixtures (the backbone block,
the heads, the state machine, sampling, the tokenizer) without the 8B model.

## Testing & gates (definition of done)

**Model-independent (CI, tiny numpy fixtures via `gen_test_fixtures.py`):**
- `test_qwen3_block` — Qwen3 block forward vs a tiny HF/numpy reference,
  including per-head q/k RMSNorm + RoPE + GQA + SwiGLU.
- `test_delay_state` — feed fixed (text_logits, audio_logits) sequences and
  assert the emitted (33,) token stream matches a dump from `delay_state.py`
  (covers the staircase, the masks, im_end stop, audio_end).
- `test_sampling` — our seeded generator's top-k/p/rep-penalty/argmax output
  on fixed logits matches a committed reference (our own generator, pinned).
- `test_delay_embeddings` + `test_lm_heads` — sum-embed and head matmul
  numerics on tiny weights.
- `test_tokenizer` — Qwen3 BPE id-level parity vs HF on a fixture.

**Model-dependent (env-gated, `SKIP_RETURN_CODE=77`):**
- `test_backbone_parity` — **the numeric gate**: load the real GGUF, run a
  deterministic forward on a fixed prompt, assert the post-final-norm hidden
  state and the 33 head logits match a reference dump (from upstream
  `moss_tts_delay/llama_cpp/` run once) within tolerance.
- `test_e2e_tts` — seeded `tts()` produces non-silent / non-clipped 24 kHz
  audio of plausible length.
- `test_closed_loop` — seeded `tts(text)` → `parakeet-cli` ASR → word-level
  recall ≥ threshold vs the input text (mirrors vibevoice's closed-loop).

Env vars: `MOSS_TTS_DELAY` (backbone gguf), `MOSS_TTS_TOKENIZER` (codec
gguf — reused name), `MOSS_DE_TOKENIZER` (BPE tokenizer gguf), `MOSS_DE_CLI`,
`MOSS_PARAKEET_CLI` + `MOSS_PARAKEET_MODEL` (closed-loop), and
`MOSS_DE_REF_DUMP` (the backbone-parity reference).

**Benchmark** (`bench_tts.sh`): tokens/sec + RTF vs the upstream
llama.cpp+ONNX pipeline on a fixed prompt set; optional Seed-TTS-eval WER.

## Known gotchas / risks

1. **Qwen3 q/k per-head RMSNorm** is the easy-to-miss delta vs Qwen2 — apply
   `q_norm`/`k_norm` (head_dim-wide) to each head before RoPE.
2. **Speech-feature magnitude**: audio embeddings sum 32 tables; do NOT
   normalize the summed input embedding (the model was trained that way —
   the vibevoice lesson about not renormalizing spliced speech features).
3. **The delay staircase is subtle** (`delayed_length` wrapping to INT64_MAX
   after `> n_vq`, the `time_step==0` and `time_step<=n_vq` guards, the
   pre/post-audio masks). Port `delay_state.py` line-for-line; the
   step-sequence parity test is the guard.
4. **audio tables are 1025-wide** (pad code 1024). Off-by-one here corrupts
   either the embedding lookup or the head masking.
5. **Memory**: 8B backbone (Q4_K_M ≈ 5 GB) + codec + KV cache. Develop
   against a quantized backbone GGUF; size the KV cache from metadata.
   Reuse the Foundation gallocr path; compute-pool/KV sizing scales with the
   generated length.
6. **Determinism**: numeric parity uses greedy/deterministic forward (no
   sampling). `--seed` pins our own RNG for reproducible sampled generation;
   exact parity vs numpy's RNG is explicitly out of scope.
7. **Reference-code splice** uses the same `apply_delay_pattern` as the
   generation de-delay — keep one implementation shared by `prompt` and
   `delay_state`.

## Conventions

- C++17, no exceptions across the C-API (status codes), one TU per
  component, comments explain non-obvious *why*.
- Commit policy: `Assisted-by:` trailer, no `Co-Authored-By`/`Signed-off-by`
  from AI (matches the repo).
- Converters run with `--strict`.
- Reuse Foundation infra (model_loader, backend gallocr path, audio_io,
  Codec, ggml_extend, the parity-fixture methodology in
  `scripts/gen_test_fixtures.py`).

## Plan structure

One spec, **two-phase plan**:
- **Phase A — the LM core (emits logits):** converter (backbone+emb+heads+
  metadata) + tokenizer convert; `qwen3` block + KV cache; `delay_embeddings`;
  `lm_heads`; the model-independent block/embed/head/tokenizer parity tests;
  `test_backbone_parity` (env-gated).
- **Phase B — generation + e2e:** `sampling`; `delay_state`; `prompt`;
  `moss_tts_delay` orchestrator + cloning; public/C-API + CLI `tts`; the
  `delay_state`/`sampling` parity tests; `test_e2e_tts` + `test_closed_loop`
  + `bench_tts.sh` + docs.
