# V2: MossTTSLocal (1.7B) → ggml

**Status:** approved design — ready for implementation planning
**Date:** 2026-06-04
**Sub-project:** V2 of the MOSS-TTS → ggml porting program. Depends on the
completed Foundation (codec) and V1 (Qwen3 backbone + glue, much reused).

## Context

MossTTSLocal is the lightweight, streaming-oriented variant. Unlike Delay
(V1), which packs all 32 RVQ codebooks into one time stream via a
delay-pattern staircase, Local uses an **RQ-Transformer**: a global Qwen3
backbone runs over time at the frame rate, and at each frame a small
**local/depth transformer** autoregressively generates all 33 channels
(text token + 32 audio codes) for that frame. There is **no delay
pattern** — the depth transformer replaces V1's delay state machine
entirely.

Local has **no llama.cpp / ONNX reference** (unlike Delay). The only
reference is the HF PyTorch modeling at `moss_tts_local/` + the checkpoint
`MOSS-TTS-Local-Transformer`. Correctness gates are numeric parity vs a
deterministic PyTorch reference dump, plus closed-loop ASR WER and an RTF
benchmark vs the HF implementation.

## Goals / non-goals

**Goals**
- Native ggml MossTTSLocal: the global Qwen3 backbone (reused from V1) + the
  4-layer **local/depth transformer** (Qwen3 layers WITHOUT positional
  embedding) + the per-channel adapters/norms/heads + the per-channel
  `embedding_list` + the nested time×depth generation loop.
- **Zero-shot voice cloning** (encode reference wav via the codec, splice
  codes into the prompt — no delay pattern).
- A converter: HF MossTTSLocal safetensors → one GGUF (global + local
  transformers + embeddings + adapters + heads + metadata), with matmul
  quantization.
- `moss-tts-cli tts-local …` and a `Local` C++/C API.

**Non-goals (deferred)**
- V3 (Realtime), V4 (Nano).
- Streaming generation (offline full-utterance first; the architecture is
  streaming-friendly but we ship offline first).
- Training / fine-tuning.
- Exact token-for-token reproduction of stochastic sampling (numeric gates
  use deterministic/greedy forward; closed-loop seeds + WER threshold).

## Reference sources (single sources of truth)

1. **`OpenMOSS/MOSS-TTS` `moss_tts_local/`** — `modeling_moss_tts.py`
   (the `MossTTSDelayModel` class here is actually the Local model: global
   `MosiTTSModel` + `local_transformer` + `_sample` depth loop),
   `configuration_moss_tts.py`, `processing_moss_tts.py` (prompt format),
   `inference_utils.py`. **Authoritative on architecture, the depth loop,
   the prompt format, and token IDs.**
2. **`MOSS-TTS-Local-Transformer`** HF checkpoint (safetensors + config) —
   the global Qwen3 `language_config` dims, the local dims, and all module
   tensor shapes/names.
3. **V1 of this repo** — `qwen3.cpp` (global backbone + the `use_rope`
   extension for the local), `delay_backbone.cpp` (the global stack +
   KV cache, reused), `delay_embeddings.cpp` (the sum-embed pattern),
   `lm_heads.cpp`, `sampling.cpp`, `de_tokenizer.cpp`, `prompt.cpp`
   (adapt), the converter + parity-fixture methodology.
4. **Foundation** — the `Codec` (encode for cloning, decode for synthesis).

## Architecture

### Config (from `configuration_moss_tts.py`; read exact dims from the checkpoint)

| name | value | role |
|------|------:|------|
| n_vq | 32 | audio codebooks |
| channels | 33 | 1 + n_vq (text + audio) |
| audio_vocab_size | 1024 | real audio codes 0..1023 |
| audio_pad_code | 1024 | audio pad (heads/embeddings 1025-wide) |
| hidden_size | = global `language_config.hidden_size` | global Qwen3 width |
| local_hidden_size | 1536 | depth transformer width |
| local_num_layers | 4 | depth transformer depth |
| local_ffn_hidden_size | 8960 | depth transformer FFN |
| additional_mlp_ffn_hidden_size | 2048 | adapter-MLP FFN width |
| sampling_rate | 24000 | codec rate |

Special-token IDs are the same family as V1 (`src/delay_constants.hpp`):
pad 151643, im_start 151644, im_end 151645, audio_start 151652, audio_end
151653, audio_user_slot 151654, audio_assistant_gen_slot 151656,
audio_assistant_delay_slot 151662. (Confirm against the Local config.json;
the delay-slot id may be unused by Local.)

### Modules (from `MossTTSDelayModel.__init__` / `MosiTTSModel.__init__`)

- **`embedding_list[i]`** (channels=33): i=0 `Embedding(vocab, hidden)`
  (text), i=1..32 `Embedding(1025, hidden)` (audio, pad idx 1024). Used
  for BOTH the global input sum AND the local re-embedding. (The global
  Qwen3's own `embed_tokens` is frozen/unused — text comes from
  `embedding_list[0]`.)
- **Global backbone** `model.language_model` — Qwen3 (full `language_config`):
  q/k per-head RMSNorm, RoPE, GQA, SwiGLU. = V1's `DelayBackbone` exactly.
- **`local_transformer`** — Qwen3 with the local config (4 layers, hidden
  1536, ffn 8960) but **`MossTTSAttentionWithoutPositionalEmbedding`**: no
  RoPE. q/k RMSNorm is kept (it subclasses Qwen3Attention). Causal over the
  depth sequence.
- **`speech_embedding_to_local_mlp`** — shared `MossTTSMLP(hidden →
  ffn=2048 → local_hidden=1536)`: SwiGLU `down(silu(gate(x))*up(x))`, no
  bias, no prenorm. Projects the global hidden and each `embedding_list`
  output into the local-transformer input space.
- **`local_to_speech_embedding_mlps[i]`** (33) — `MossTTSMLP(local_hidden →
  2048 → hidden)`, per channel.
- **`layer_norm_before_lm_heads[i]`** (33) — `MossTTSRMSNorm(hidden)`, per
  channel.
- **`lm_heads[i]`** (33) — i=0 `Linear(hidden, vocab)` text; i=1..32
  `Linear(hidden, 1025)` audio. No bias.

`MossTTSMLP` = SwiGLU with distinct in/mid/out dims (no q/k norm, no bias);
`MossTTSRMSNorm` = standard RMSNorm.

### Generation (`_sample`, the nested time×depth loop)

Per timestep t:
1. **Global**: `embeds[s] = Σ_{i=0..32} embedding_list[i][input_ids[s,i]]`;
   global Qwen3 backbone (prefill the prompt once, then 1-token decode with
   KV cache) → `global_hidden` (last position).
2. **Local/depth** (generates the 33 channels for frame t):
   ```
   cur = speech_embedding_to_local_mlp(global_hidden)     # (local_hidden,)
   local_seq = []
   for i in 0..32:
       local_seq.append(cur)
       h = local_transformer(local_seq)[-1]               # causal, NO RoPE; recompute over the growing seq
       o = layer_norm_before_lm_heads[i]( local_to_speech_embedding_mlps[i](h) )
       logit_i = lm_heads[i](o)                           # i=0 text vocab; i=1..32 audio 1025
       if i != 0: logit_i[audio_pad_code=1024] = -inf
       code_i = sample(logit_i, per-channel cfg)
       cur = speech_embedding_to_local_mlp( embedding_list[i][code_i] )
   next = (code_0=text, code_1..32=audio)                 # (33,)
   ```
3. Append `next` to `input_ids`; feed the global backbone the next step.
   **Stop** when the text channel (code_0) emits `im_end` (eos).
4. After the loop: the audio channels (next[1..32] per frame) form (T, 32)
   codes — drop trailing pad / collect the real frames → `Codec.decode` →
   24 kHz wav.

**The local transformer is recomputed over the full growing depth sequence
each of the 33 steps** (≤33 tokens, 4 layers — cheap; no separate local KV
cache needed, matching the python which re-runs `local_transformer` over
the growing `local_transformer_inputs`). `n_vq_for_inference` defaults to
n_vq (all 32 audio channels); channels beyond it are set to 0.

## Components and boundaries

Reuse (unchanged): `Codec`, `DeTokenizer`, `sampling`, `audio_io`,
`model_loader`, `backend` (gallocr `compute_graph_with_inputs`),
`ggml_extend`.

Reuse + extend:
- `qwen3.{hpp,cpp}` — add a `bool use_rope` to `Qwen3Hparams` (or a param to
  `qwen3_layer_forward`); global = true, local = false. q/k norm stays on
  both. No other change.
- `DelayBackbone` (`delay_backbone.cpp`) — the **global** stack (Qwen3 +
  KV cache, embeds→hidden) is reused directly.

New:
| unit | does | depends on |
|------|------|-----------|
| `local_embeddings` | `embedding_list[i]` sum-lookup (global) + single-code embed (local) | model_loader |
| `moss_tts_mlp` (helper in `local_adapters`) | `MossTTSMLP` SwiGLU adapter (in→ffn→out) | ggml_extend |
| `local_transformer` | 4-layer no-RoPE Qwen3 depth transformer; recompute over growing seq | qwen3 (use_rope=false), model_loader |
| `local_adapters` | speech_embedding_to_local_mlp + per-channel out-MLPs + norms + heads | moss_tts_mlp, model_loader |
| `prompt_local` | build_generation_prompt (no delay pattern) | de_tokenizer |
| `moss_tts_local` | the time×depth orchestrator + cloning; `moss::Local` | all above + DelayBackbone + Codec |

Each new unit is independently testable on tiny fixtures.

## Converter + GGUF inventory

`scripts/convert_moss_tts_local_to_gguf.py` → one GGUF:
- **Global backbone** `model.language_model.*` → `qwen3.blk.{i}.*` +
  `qwen3.output_norm.weight` (same mapping as V1's converter; reuse it).
- **Local transformer** `local_transformer.*` → `local.blk.{i}.*` (same
  per-layer Qwen3 names: attn_{q,k,v,o}, attn_{q,k}_norm, ffn_{gate,up,down},
  attn_norm, ffn_norm) + `local.output_norm.weight`. (No RoPE → no rope
  metadata needed for local, but emit `local.rms_eps`.)
- **Embeddings** `model.embedding_list.{i}.weight` → `lc.embed.{i}.weight`
  (i 0..32; [0]=vocab×hidden, [1..32]=1025×hidden).
- **Adapters / norms / heads**:
  `speech_embedding_to_local_mlp.{gate,up,down}_proj.weight` →
  `lc.in_mlp.{gate,up,down}.weight`;
  `local_to_speech_embedding_mlps.{i}.{gate,up,down}_proj.weight` →
  `lc.out_mlp.{i}.{gate,up,down}.weight`;
  `layer_norm_before_lm_heads.{i}.weight` → `lc.head_norm.{i}.weight`;
  `lm_heads.{i}.weight` → `lc.lm_head.{i}.weight` ([0]=vocab, [1..32]=1025).
- **Metadata**: `qwen3.*` (global dims, as V1), `local.{hidden,n_layers,
  n_heads,n_kv_heads,head_dim,intermediate,rms_eps}`, `lc.{n_vq=32,
  audio_vocab=1025, additional_mlp_ffn=2048, sample_rate=24000}`, the
  special-token IDs.
- **Quantization** (`quantize_gguf.py` extended): quantize the `qwen3.blk.*`
  AND `local.blk.*` matmuls; **keep all `lc.*` (embeddings, adapter MLPs,
  norms, heads) and all `*_norm.weight` at f32** (the embeddings/heads are
  read raw via CPU gather/dot — the V1 footgun applies). Tokenizer reuses
  V1's `convert_tokenizer.py`.

The codec GGUF is the Foundation's (unchanged). Confirm the real source key
names against `MOSS-TTS-Local-Transformer/model.safetensors.index.json`
during conversion and adjust the regex (the V1 converter verified this
pattern works).

## Testing & gates (definition of done)

**Model-independent (CI, tiny numpy/python fixtures):**
- `test_local_block` — the no-RoPE Qwen3 local layer forward vs a numpy
  reference (q/k norm on, RoPE off).
- `test_moss_tts_mlp` — the SwiGLU adapter (distinct in/mid/out) vs numpy.
- `test_local_embeddings` — `embedding_list` sum + single-code embed.
- `test_local_adapters` — per-channel out-MLP → norm → head numerics.
- `test_depth_loop` — **the new intricate piece**: the time×depth loop on
  tiny global+local weights, asserting the emitted (33,) channel stream
  EXACTLY matches a dump from the python `_sample` loop (greedy) for a few
  timesteps. (Mirrors how V1 pinned the delay state machine.)
- `test_prompt_local` — prompt builder exact `input_ids` vs a python dump
  (no delay pattern), using the real tokenizer.

**Model-dependent (env-gated, `SKIP_RETURN_CODE=77`):**
- `test_local_parity` — **the numeric gate**: real GGUF, deterministic
  forward on a fixed prompt; assert the global hidden + the per-channel
  depth logits (and codes under greedy) match a HF-PyTorch reference dump
  within tolerance.
- `test_e2e_local` — seeded `tts()` → non-silent/non-clipped 24 kHz audio.
- `test_closed_loop_local` — seeded TTS → `parakeet-cli` ASR → word-recall
  ≥ threshold.

Env vars: `MOSS_TTS_LOCAL` (gguf), `MOSS_TTS_TOKENIZER` (codec gguf),
`MOSS_DE_TOKENIZER` (BPE gguf), `MOSS_LOCAL_REF_DUMP`, `MOSS_PARAKEET_CLI` +
`MOSS_PARAKEET_MODEL`.

**Benchmark** (`bench_local.sh`): tokens/sec + RTF vs the HF PyTorch
`MOSS-TTS-Local-Transformer` on a fixed prompt set (no llama.cpp reference);
optional Seed-TTS-eval WER.

## Known gotchas / risks

1. **The depth loop is the new intricate piece** — the per-frame local
   recompute over a growing sequence, the re-embed feedback (`code →
   embedding_list[i] → speech_embedding_to_local_mlp → next local input`),
   and the per-channel head selection (text head i=0 vs audio heads i=1..32,
   speech_pad masked for i≠0). Pin it with an exact-parity fixture vs the
   python `_sample` loop.
2. **Local attention has NO RoPE** but KEEPS q/k RMSNorm. Easy to get wrong
   (don't reuse the global RoPE path for the local).
3. **The two SwiGLU adapters have distinct in/mid/out dims** and NO bias /
   NO prenorm (`MossTTSMLP`); `speech_embedding_to_local_mlp` is shared,
   `local_to_speech_embedding_mlps[i]` is per-channel.
4. **`embedding_list[i]` is shared** between the global input sum and the
   local re-embed — one implementation, two call sites.
5. **Do NOT normalize the summed global input embedding** (the V1/vibevoice
   lesson).
6. **1025-wide audio tables / pad code 1024** — same off-by-one risk as V1;
   speech_pad masked in the audio heads (i≠0).
7. **Memory/quant**: global Qwen3 (~1.7B) + local (4×1536) + codec; quantize
   global+local matmuls, keep `lc.*` f32 (CPU gather/dot). Dev against a
   quantized GGUF.
8. **No delay pattern in the prompt** — the reference-audio splice puts raw
   codes (not delay-shifted) into the audio channels; read
   `processing_moss_tts.py` for the exact format.
9. **Determinism**: numeric gates use greedy; `--seed` pins our mt19937_64;
   exact-vs-torch RNG parity out of scope.

## Conventions

- C++17, no exceptions across the C-API (status codes), one TU per
  component, comments explain non-obvious *why*.
- Commit policy: `Assisted-by:` trailer, no `Co-Authored-By`/`Signed-off-by`
  from AI.
- Converters run with `--strict`.
- Reuse V1/Foundation infra and the `gen_test_fixtures.py` parity-fixture
  methodology; intricate ports (depth loop, prompt) pinned by exact integer
  parity vs the upstream python.

## Plan structure

One spec, **two-phase plan**:
- **Phase A — components + logit parity:** converter + tokenizer convert;
  `qwen3` `use_rope` extension; reuse `DelayBackbone` as the global;
  `local_transformer` (no-RoPE); `local_embeddings`; `moss_tts_mlp` +
  `local_adapters`; the model-independent block/mlp/embed/adapter parity
  tests; `test_local_parity` (env-gated).
- **Phase B — depth loop + e2e:** the time×depth orchestrator
  (`moss_tts_local`) + cloning; `prompt_local`; the `test_depth_loop` and
  `test_prompt_local` exact-parity fixtures; public/C-API + CLI `tts-local`;
  `test_e2e_local` + `test_closed_loop_local` + `bench_local.sh` + docs.
