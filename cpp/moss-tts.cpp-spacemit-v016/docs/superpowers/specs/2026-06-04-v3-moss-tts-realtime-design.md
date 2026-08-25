# V3: MossTTSRealtime (1.7B, offline) → ggml

**Status:** approved design — ready for implementation planning
**Date:** 2026-06-04
**Sub-project:** V3 of the MOSS-TTS → ggml porting program. Depends on the
Foundation (codec), V1 (Qwen3 backbone + glue), and V2 (the RQ-Transformer
pattern, much reused).

## Context

MossTTSRealtime is the streaming/voice-agent variant. Architecturally its
OFFLINE forward is an **RQ-Transformer** like MossTTSLocal (V2): a global
Qwen3 backbone runs over time, and at each frame a small local/depth
transformer autoregressively generates the per-frame RVQ codes. The
"realtime" property is almost entirely a serving concern (incremental text
intake from an upstream LLM, chunked codec decode, multi-turn context) — the
per-step math is the offline model.

This milestone ports the **offline single-utterance text→speech** path +
voice cloning. **Native streaming/multi-turn is a non-goal here** — the
offline loop *is* the full model; streaming is incremental-feed orchestration
on top, a candidate follow-on. Realtime has no llama.cpp/ONNX reference —
only HF PyTorch (`moss_tts_realtime/mossttsrealtime/`). Gates: numeric
global+local logit parity vs a PyTorch dump, closed-loop ASR WER, e2e, and
an RTF benchmark vs HF.

## Goals / non-goals

**Goals**
- Native ggml MossTTSRealtime (offline): the global Qwen3 backbone (reused) +
  the 4-layer local/depth transformer **with RoPE** + 17 sum-embeddings + 15
  local input embeddings + 16 per-codebook heads + the time×depth generation
  loop.
- **Zero-shot voice cloning** (encode reference wav with 16 codebooks, splice
  into the hierarchical prompt's reference-audio context block).
- A converter: HF MossTTSRealtime safetensors → one GGUF (global + local
  transformers + embeddings + heads + metadata), with matmul quantization.
- `moss-tts-cli tts-rt …` and a `Realtime` C++/C API.

**Non-goals (deferred)**
- **Native streaming / multi-turn** (the 41 KB `streaming_mossttsrealtime.py`
  server layer; incremental text feed; chunked decode). The per-step math is
  identical to offline; streaming is a follow-on milestone.
- V4 (Nano). Training/fine-tuning. Exact stochastic-sampling reproduction
  (numeric gates use greedy; closed-loop seeds + WER threshold).

## Reference sources (single sources of truth)

1. **`OpenMOSS/MOSS-TTS` `moss_tts_realtime/`** — `mossttsrealtime/`
   modeling (`modeling_mossttsrealtime.py` = the wrapper: 17 `embed_tokens`,
   `language_model` global, `local_transformer`;
   `modeling_mossttsrealtime_local.py` = the local/depth transformer + its
   per-codebook heads + the depth generation), `configuration_mossttsrealtime.py`,
   `processing_mossttsrealtime.py` (the hierarchical prompt). **The OFFLINE
   generation loop + prompt are in `moss_tts_realtime/inferencer.py`**
   (`_generate_from_ids`, `make_ensemble`, `_build_prefill_batch`,
   `generate_local_transformer`, `sample_token`) — the runnable offline path;
   `infer.py` is the simplest text→wav entry. (`streaming_mossttsrealtime.py`
   uses the identical depth loop but for the streaming server — out of scope.)
   **Authoritative on architecture, the depth loop, the prompt, token IDs.**
2. **The HF Realtime checkpoint** (safetensors + config) — global Qwen3
   `language_config` dims, the local dims, all module tensor shapes/names.
3. **V1/V2 of this repo** — `qwen3.cpp` (the local uses `use_rope=true`),
   `delay_backbone.cpp` (the global, reused), `local_embeddings.cpp` (the
   sum-embed pattern, generalized), `sampling.cpp`, `de_tokenizer.cpp`,
   `prompt.cpp`/`prompt_local.cpp` (adapt), the converter + parity-fixture
   methodology.
4. **Foundation** — the `Codec` (encode/decode with **16** RVQ codebooks).

## Architecture

### Config (from `configuration_mossttsrealtime.py`; read exact dims from the checkpoint)

| name | value | role |
|------|------:|------|
| rvq | 16 | audio codebooks (NOT 32) |
| channels | 17 | 1 + rvq (text + audio) |
| audio_vocab_size | 1027 | 1024 codes + BOS 1025 + EOS 1026 |
| audio_pad_token | 1024 | audio pad |
| bos_audio | 1025 | seeds audio at end of text prefill |
| eos_audio | 1026 | stop signal (on audio codebook 0) |
| reference_audio_pad | 151654 | text-channel placeholder for the clone block |
| text_pad | 151655 | text-channel filler once text is exhausted |
| delay_tokens_len | 12 | text leads audio by 12 tokens |
| global hidden_size | = `language_config.hidden_size` (2048) | global Qwen3 width |
| local hidden_size | 2048 | depth transformer width (== global) |
| local num_layers | 4 | depth transformer depth |
| local head_dim | 128 | local attention head dim |
| local n_heads / n_kv | 16 / 8 | local GQA |
| local intermediate | 6144 | local FFN |
| local rope_theta | 1e6 | local RoPE (linear), max_pos 33 |

### Modules

- **`embed_tokens[i]`** (channels=17): i=0 text `Embedding(vocab, hidden)`,
  i=1..16 audio `Embedding(1027, hidden)` (pad idx 1024). The GLOBAL input is
  the sum over all 17 (`get_input_embeddings`). (The Qwen3Model's own
  `language_model.embed_tokens` is unused — skip it.)
- **Global backbone** `language_model` — Qwen3 (full `language_config`):
  q/k RMSNorm, RoPE, GQA, SwiGLU. = V1's `DelayBackbone`.
- **`local_transformer`** — Qwen3 stack (local config: 4 layers, hidden 2048,
  head_dim 128, 16/8 heads, ffn 6144) **WITH RoPE** (linear, θ 1e6, positions
  = depth index 0..15) + q/k RMSNorm. Shared final RMSNorm (`local.norm`).
  Uses a per-frame KV cache over the ≤16 depth steps (rebuilt fresh each
  timestep; causal within the frame).
- **`local_embed[j]`** (15 = rvq-1): `Embedding(1027, hidden)` (pad 1024) —
  the local depth transformer's OWN per-codebook input embeddings (distinct
  from the global `embed_tokens`).
- **`local_lm_heads[i]`** (16): `Linear(hidden, 1027)` bias-free, per codebook.
  Output: `local_lm_heads[i]( local.norm( local_transformer_hidden ) )`. NO
  per-channel output MLP / extra norm (simpler than V2 Local).

All linears bias-free; RMSNorm eps 1e-6.

### Generation (offline time×depth, `inferencer.py`)

Per frame:
1. **Global**: `embeds[s] = Σ_{i=0..16} embed_tokens[i][input_ids[s,i]]`;
   global Qwen3 backbone (prefill the prompt once, then 1-token decode with
   KV cache, feeding the previous frame's full 17-channel token) →
   `global_hidden` (last position).
2. **Local/depth** (16 codebooks, fresh per-frame KV cache; RoPE positions
   0..15):
   ```
   in = global_hidden                                  # depth-0 token = backbone hidden, NO projection
   h  = local_transformer.step(in, pos=0)              # 1 token into the per-frame KV
   code[0] = sample( local_lm_heads[0]( local.norm(h) ) )
   for i in 1..15:
       in = local_embed[i-1][ code[i-1] ]              # re-embed the previous code (15 local embeds)
       h  = local_transformer.step(in, pos=i)
       code[i] = sample( local_lm_heads[i]( local.norm(h) ) )
   frame = code[0..15]
   ```
3. Next global step input = `(next_text_token | text_pad 151655)` on channel
   0 + `frame`'s 16 codes on channels 1..16. **Stop** when `code[0] ==
   eos_audio (1026)`. Prefill seeds audio with `bos_audio (1025)` on channel
   1 at the last prefilled text position.

The depth loop has **no delay between codebooks within a frame** (all 16
share the timestep); the only "delay" is the 12-token text lead in the
prompt (text→audio alignment), realized as text-streaming + `text_pad`.

`n_vq_for_inference` = rvq (all 16). Sampling: per-codebook temperature
(0.8) / top-k (30) / top-p (0.6) / windowed repetition penalty (1.1, window
50); greedy when temp 0. Hand-rolled in `inferencer.py::sample_token`.

### Hierarchical prompt (`inferencer.py::make_ensemble` + `_build_prefill_batch`)

- **System / clone block** (`make_ensemble`): a system prompt; with a
  reference, a `<|im_start|>context\n…<|audio_pad|>×N…<|im_end|>\n` block
  where the N `reference_audio_pad (151654)` text-channel tokens have their
  audio channels 1..16 overwritten by the 16-codebook reference codes
  (`ref_codes[:16].T`, shape (N,16)). Then `<|im_start|>assistant\n` rows
  (audio channels = pad 1024).
- **Prefill text** (`_build_prefill_batch`): up to `delay=12` text tokens on
  channel 0 (audio channels = pad 1024); `bos_audio (1025)` placed on channel
  1 at the LAST prefilled text position (kicks off audio).
- **Generation**: remaining text tokens emitted one per timestep on channel 0;
  after exhaustion, `text_pad (151655)`. This is the text→audio delay.

Output: `input_ids (S, 17)` int32.

## Components and boundaries

Reuse (unchanged): `DelayBackbone` (global), `Codec` (16 codebooks),
`DeTokenizer`, `sampling`, `audio_io`, `model_loader`, `backend`,
`qwen3` (`use_rope=true` for the local), `ggml_extend`.

New:
| unit | does | depends on |
|------|------|-----------|
| `rt_embeddings` | 17 global sum-embeds + 15 local single-code embeds (off-by-one) | model_loader |
| `rt_local` | 4-layer RoPE depth transformer + per-frame KV cache; `step(in,pos)` | qwen3 (use_rope=true), model_loader |
| `rt_heads` | 16 per-codebook `Linear(hidden,1027)` + shared `local.norm` | model_loader |
| `prompt_rt` | hierarchical prompt (clone block + delay + BOS) | de_tokenizer |
| `moss_tts_rt` | the time×depth orchestrator (BOS prefill, text stream, EOS stop) + cloning | all above + DelayBackbone + Codec |

Each new unit is independently testable on tiny fixtures.

## Converter + GGUF inventory

`scripts/convert_moss_tts_rt_to_gguf.py` → one GGUF (confirm real source key
names + the `model.`/no-prefix question against the checkpoint index, as the
V1/V2 converters did):
- **Global backbone** `language_model.*` → `qwen3.blk.{i}.*` +
  `qwen3.output_norm.weight` (reuse the V1/V2 mapping). Skip the unused
  `language_model.embed_tokens.weight`.
- **Global sum-embeddings** `embed_tokens.{i}.weight` (i 0..16) →
  `rt.embed.{i}.weight` ([0]=vocab×hidden, [1..16]=1027×hidden).
- **Local transformer** `local_transformer.model.layers.{i}.*` →
  `rtl.blk.{i}.*` (Qwen3 per-layer names); `local_transformer.model.norm.weight`
  → `rtl.output_norm.weight`.
- **Local input embeds** `local_transformer.model.embed_tokens.{j}.weight`
  (j 0..14) → `rtl.embed.{j}.weight` (1027×hidden).
- **Local heads** `local_transformer.local_lm_heads.{i}.weight` (i 0..15) →
  `rtl.head.{i}.weight` (1027×hidden, bias-free).
- **Metadata**: `qwen3.*` (global dims, as V1), `rtl.{hidden,n_layers,
  n_heads,n_kv_heads,head_dim,intermediate,rope_base,rms_eps}` (local; the
  local head_dim/n_heads derived from real q_proj shapes per the V2 lesson),
  `rt.{rvq=16, audio_vocab=1027, audio_pad=1024, bos_audio=1025,
  eos_audio=1026, reference_audio_pad=151654, text_pad=151655,
  delay_tokens=12, sample_rate=24000}`, the chat special-token IDs.
- **Quantization** (`quantize_gguf.py` extended): quantize the `qwen3.blk.*`
  AND `rtl.blk.*` matmuls; **keep all `rt.*`/`rtl.embed.*`/`rtl.head.*` and
  all `*_norm.weight` at f32** (the V1/V2 CPU-gather/dot footgun). Tokenizer
  reuses V1's `convert_tokenizer.py`. The codec GGUF is the Foundation's.

## Testing & gates (definition of done)

**Model-independent (CI, tiny numpy/python fixtures):**
- `test_rt_local_block` — the local depth layer (RoPE on, q/k norm) vs numpy
  (this is `test_local_block` with use_rope=TRUE — a small variant).
- `test_rt_local` — the depth transformer `step` over a per-frame KV cache
  (positions 0..15, RoPE) vs numpy.
- `test_rt_embeddings` — 17-table sum + 15-table single-code embed.
- `test_rt_heads` — per-codebook `Linear(hidden,1027)` + shared norm numerics.
- `test_rt_depth_loop` — **the keystone**: the time×depth loop on a tiny
  complete model (depth-0-hidden injection, 15-embed/16-head off-by-one,
  per-frame KV), asserting the emitted (channels,) code stream EXACTLY matches
  a numpy `_sample` transcription (greedy) for a few timesteps.
- `test_prompt_rt` — the hierarchical prompt exact `input_ids` vs a python
  dump (clone block + delay + BOS, using the real tokenizer).

**Model-dependent (env-gated, `SKIP_RETURN_CODE=77`):**
- `test_rt_parity` — **the numeric gate**: real GGUF, deterministic forward;
  assert the global hidden + the per-codebook depth logits (and codes under
  greedy) match an HF-PyTorch reference dump within tolerance.
- `test_e2e_rt` — seeded `tts()` → non-silent/non-clipped 24 kHz audio.
- `test_closed_loop_rt` — seeded TTS → `parakeet-cli` ASR → word-recall ≥
  threshold.

Env vars: `MOSS_TTS_RT` (gguf), `MOSS_TTS_TOKENIZER` (codec gguf),
`MOSS_DE_TOKENIZER` (BPE gguf), `MOSS_RT_REF_DUMP`, `MOSS_PARAKEET_CLI` +
`MOSS_PARAKEET_MODEL`.

**Benchmark** (`bench_rt.sh`): tokens/sec + RTF vs the HF PyTorch Realtime
model on a fixed prompt set; optional Seed-TTS-eval WER.

## Known gotchas / risks

1. **The local depth transformer USES RoPE** (linear, θ 1e6, positions 0..15)
   + q/k norm — flip `use_rope=true` (V2's local was no-RoPE). It also uses a
   **per-frame KV cache** over the 16 depth steps (rebuilt each frame), unlike
   V2's recompute. Position = depth index.
2. **Backbone hidden is the depth-0 token directly — NO projection MLP**
   (global hidden 2048 == local hidden 2048). (V2 had a
   `speech_embedding_to_local_mlp`; Realtime drops it — identity.)
3. **Off-by-one embed/head indexing:** depth 0 input = backbone hidden, head
   = `local_lm_heads[0]`; depth i≥1 input = `local_embed[i-1][code[i-1]]`,
   head = `local_lm_heads[i]`. 15 local embeds, 16 heads.
4. **Per-codebook heads only** — `local_lm_heads[i]( local.norm(h) )`. No
   per-channel out-MLP/head-norm. Simpler than V2; do not over-build.
5. **rvq=16 / audio_vocab=1027 / channels=17** — different from V1/V2 (32/
   1025/33). BOS 1025 seeds, EOS 1026 stops (on audio codebook 0). Codec
   decode/encode with **16** codebooks.
6. **The hierarchical prompt** (12-token text lead, clone-block reference
   splice into the 151654 region, BOS at the last prefilled text position,
   text_pad after exhaustion) — pin with exact `input_ids` parity vs the
   python (`inferencer.py::make_ensemble` + `_build_prefill_batch`). Match
   the INFERENCER path (not the HF `processing_*.py` variant).
7. **Do NOT normalize the summed global input embedding** (the V1/vibevoice
   lesson).
8. **Stop on audio EOS (1026) on codebook 0**, not on a text token.
9. **Memory/quant**: global Qwen3 (~1.7B) + local (4×2048) + codec; quantize
   global+local matmuls, keep `rt.*`/`rtl.embed.*`/`rtl.head.*` f32.
10. **Determinism**: numeric gates greedy; `--seed` pins our mt19937_64;
    exact-vs-torch RNG out of scope.

## Conventions

- C++17, no exceptions across the C-API (status codes), one TU per
  component, comments explain non-obvious *why*.
- Commit policy: `Assisted-by:` trailer, no `Co-Authored-By`/`Signed-off-by`
  from AI.
- Converters run with `--strict`.
- Reuse V1/V2/Foundation infra + the `gen_test_fixtures.py` methodology;
  intricate ports (depth loop, prompt) pinned by exact integer parity vs the
  upstream python.

## Plan structure

One spec, **two-phase plan**:
- **Phase A — components + logit parity:** converter + tokenizer convert; the
  `rt_local` depth transformer (RoPE + per-frame KV cache); `rt_embeddings`
  (17 sum + 15 single); `rt_heads` (16 per-codebook); the model-independent
  block/local/embed/heads parity tests; `test_rt_parity` (env-gated).
- **Phase B — depth loop + e2e:** the time×depth orchestrator (`moss_tts_rt`)
  + cloning; `prompt_rt`; the `test_rt_depth_loop` (tiny-complete keystone)
  and `test_prompt_rt` exact-parity fixtures; public/C-API + CLI `tts-rt`;
  `test_e2e_rt` + `test_closed_loop_rt` + `bench_rt.sh` + docs.
