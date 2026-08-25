# Real MOSS-TTS-Local-Transformer-v1.5 (GPT-J local) port — Design

**Date:** 2026-07-10
**Status:** Proposed (brainstorming, approved)
**Scope:** Re-port the v1.5 **Local** model against its *real* architecture, which
differs materially from the merged speculative support. Coexist with the existing
**Delay** model. Gate on the real-weight parity test. Codec-v2 parity + HF publish
stay out of scope (Batch B).

## Problem

The live real-model e2e (2026-07-09) revealed that the merged, metadata-gated
"v1.5 support" was built on **inferred** structure that does not match the real
`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` checkpoint. Ground truth (438
bf16 tensors + `config.json`):

- The **backbone** is Qwen3 36L (hidden 2560), prefix `transformer.*` (not
  `model.language_model.*`), and `transformer.embed_tokens` **is used** (text
  input embed + tied text head), not frozen/skipped.
- The **local depth transformer is GPT-J, not Qwen3**: `gpt2_config` (n_embd 2560,
  n_head 32 → head_dim 80 MHA no-GQA, n_layer 1, n_inner 9728, `silu`, `rope_base`
  1e6, `layer_norm_epsilon` 1e-6, `position_embedding_type="rope"`). Tensors
  `local_transformer.h.{i}.`: `attn.c_attn.{weight[7680,2560],bias}` (fused QKV),
  `attn.c_proj.{weight,bias}`, `ln_1.{weight,bias}` + `ln_2.{weight,bias}`
  (LayerNorm **with bias**), `mlp.fc_in.{w[9728,2560],b}` + `mlp.fc_out.{w,b}`
  (2-matrix silu MLP, not gated SwiGLU), `local_transformer.ln_f.{weight,bias}`.
  `local_transformer.wte = nn.Identity()` (fed hidden states directly).
- **Bare, weight-tied heads**: `text_lm_head[151936,2560]` tied to
  `transformer.embed_tokens`; `audio_lm_heads.{0..11}[1024,2560]` tied to
  `audio_embeddings.{0..11}[1024,2560]`; `local_text_lm_head[2,2560]` binary head.
  Applied **directly** to the local hidden: `audio_lm_heads[c](local_hidden)` — no
  out-MLP / RMSNorm / head-norm composite.
- **No adapters**: no `speech_embedding_to_local_mlp`, no
  `local_to_speech_embedding_mlps`, no `layer_norm_before_lm_heads`.
  `_global_hidden_to_local` is the identity (both 2560).
- **Audio vocab 1024, not 1025**: pad code 1024 is masked to zero in the embed-sum
  (`audio_embeds * valid_mask`), never embedded. No baked pad row.

The merged v1.5 branch assumed the opposite of nearly every line above (Qwen3
local + in/out MLP + head_norm + 1025 vocab + `+1` shifted embed indices +
`model.language_model.` prefix). It cannot pass parity. The existing **Delay**
support (Qwen3 local + adapters) is correct and targets a real, distinct model.

### The two family arms (naming, to end the confusion)

| Repo | `model_type` | n_vq | rate | Local transformer | Adapters | Heads |
|---|---|---|---|---|---|---|
| `MOSS-TTS-Local-Transformer` | `moss_tts_delay` | 32 | 24 kHz mono | **Qwen3** 4L (no-RoPE depth) | in/out MLP + head_norm | 1025-wide composite |
| `MOSS-TTS-Local-Transformer-v1.5` | `moss_tts_local` | 12 | 48 kHz stereo | **GPT-J** 1L (RoPE) | none | 1024 bare tied |

The engine's existing "local"/Qwen3 path is really the **Delay** arm. This design
adds the **Local** (GPT-J) arm alongside it.

## Goal

Make `tests/test_local_v15_parity.cpp` pass on the real v1.5 weights: convert the
real checkpoint to a self-contained GGUF, run our C++ forward, and match the
torch-dumped reference per stage (backbone hidden, binary head, 12 audio heads,
13 chosen codes). Keep the Delay path byte-identical.

**Non-goals:** MOSS-Audio-Tokenizer-v2 decoded-audio parity and HF publish (Batch
B, already-built tooling); the LocalAI Go backend; any inference-math change to
the Delay path; retraining.

## Architecture

### Coexistence key: `local.arch` GGUF metadata

The converter stamps `local.arch = "qwen3"` (Delay) or `"gptj"` (v1.5 Local). The
C++ loader reads it once and dispatches local-block construction + forward. Absent
(older GGUF) ⇒ `"qwen3"` ⇒ byte-identical Delay behavior.

### 1. Converter (`scripts/convert_moss_tts_local_to_gguf.py`)

Detect the config shape and branch. **Delay** (`language_config` present, flat
`local_hidden_size`/`local_num_layers`, `local_transformer.layers.*` Qwen3 keys):
unchanged mapping; stamp `local.arch="qwen3"`. **Local v1.5** (`qwen3_config` +
`gpt2_config` present):

- **Backbone** from `qwen3_config` (== `language_config`): dims as today but read
  from `qwen3_config`; source prefix `transformer.*` (new regexes). `transformer.
  norm.weight → qwen3.output_norm.weight`; `transformer.layers.{i}.* → qwen3.blk.
  {i}.*` (same `LAYER_MAP`). **Emit `transformer.embed_tokens.weight →
  qwen3.token_embd.weight`** (used, not skipped).
- **Local GPT-J** from `gpt2_config`, source prefix `local_transformer.h.{i}.*` →
  new output names under `local.*`:
  - `attn.c_attn.weight → local.blk.{i}.attn_qkv.weight`, `attn.c_attn.bias →
    local.blk.{i}.attn_qkv.bias`
  - `attn.c_proj.weight → local.blk.{i}.attn_o.weight`, `.bias →
    local.blk.{i}.attn_o.bias`
  - `ln_1.{weight,bias} → local.blk.{i}.attn_norm.{weight,bias}`
  - `ln_2.{weight,bias} → local.blk.{i}.ffn_norm.{weight,bias}`
  - `mlp.fc_in.{weight,bias} → local.blk.{i}.ffn_in.{weight,bias}`
  - `mlp.fc_out.{weight,bias} → local.blk.{i}.ffn_out.{weight,bias}`
  - `local_transformer.ln_f.{weight,bias} → local.output_norm.{weight,bias}`
  - Metadata: `local.arch="gptj"`, `local.hidden=2560`, `local.n_layers=1`,
    `local.n_heads=32`, `local.head_dim=80` (2560/32), `local.intermediate=9728`,
    `local.rope_base=1e6` (new key), `local.ln_eps=1e-6` (LayerNorm, not RMS),
    `local.norm_type="layernorm"` (new; Delay stays RMS/absent).
- **Embeddings / heads** (bare, tied — but emit the concrete tensors):
  - `transformer.embed_tokens.weight → lc.embed.0.weight` (text input embed)
  - `audio_embeddings.{c}.weight → lc.embed.{c+1}.weight` (c=0..11), **vocab 1024**
  - `text_lm_head.weight → lc.lm_head.0.weight`
  - `audio_lm_heads.{c}.weight → lc.lm_head.{c+1}.weight`
  - `local_text_lm_head.weight → lc.local_text_head.weight` (binary)
  - **No** `lc.in_mlp` / `lc.out_mlp` / `lc.head_norm`.
- **Metadata flags**: `lc.n_vq=12`, `lc.audio_vocab=1024` (**not +1**),
  `lc.local_text_head_mode=1` (binary), `lc.stereo=1`, `lc.bare_heads=1` (new; set
  when no out_mlp/head_norm — drives the C++ head path), `lc.sample_rate=48000`,
  special-token ids from the real config (`audio_start=151669`,
  `audio_end=151670`, `gen_slot=151656`, …).
- bf16 checkpoints load via a torch `framework="pt"` → f32 fallback (already
  added; keep). `--strict` must report 0 unmapped on the real v1.5.

### 2. C++ engine

**New GPT-J local block** (selected when `local.arch=="gptj"`). Add a `norm_type`
+ `arch` to the local transformer's `LocalConfig`/loader; build the alternate
graph:

- Pre-LN: `h = layernorm(x, ln_1.w, ln_1.b, eps)` (mean/var LayerNorm **with
  bias**, not RMSNorm).
- Fused QKV: `qkv = c_attn.w · h + c_attn.b` → split into q,k,v each [2560]; reshape
  32 heads × 80.
- **RoPE** (base 1e6) applied to q,k over the depth position index.
- Scaled-dot attention (`scale = 1/sqrt(80)`, `scale_attn_weights=true`), causal
  over the depth steps, **KV-cached** across the channel loop exactly like the
  Qwen3 path.
- `attn_out = c_proj.w · context + c_proj.b`; residual `x = x + attn_out`.
- Pre-LN 2: `h2 = layernorm(x, ln_2.w, ln_2.b, eps)`; MLP `fc_out.w · silu(fc_in.w
  · h2 + fc_in.b) + fc_out.b`; residual `x = x + mlp_out`.
- After the last layer: `local_out = layernorm(x, output_norm.w, output_norm.b,
  eps)` (`ln_f`).

The Delay path (RMSNorm, separate QKV, q/k-norm, SwiGLU, no bias, no-RoPE) is left
untouched; dispatch by `arch`.

**Bare heads** (`lc.bare_heads==1`, in `LocalAdapters`): `head_logits(c, local_out)
= lm_head[c] · local_out` directly — skip `to_local` in-MLP, out-MLP, and
head_norm. `to_local` becomes the identity (both 2560). Audio channels mask the pad
code (1024) to −∞ as today. Channel 0 text head is unused during generation (the
binary `local_text_head` decides continue/stop); keep `lc.lm_head.0` for parity
dumps. `local_text_head_logits` (2-wide) already exists — reuse.

**Embed-sum pad-mask** (`LocalEmbeddings`): when a channel's code equals the audio
pad code, its contribution is zero (mask), matching v1.5's 1024-row embeddings.
For 1025-row Delay embeddings this is a no-op (the pad row exists); gate on
`lc.audio_vocab`/`bare_heads` so Delay is unchanged.

**Depth loop** (`LocalTTS::tts`, already structured): 13 channels = binary text
decision (channel 0) + 12 audio codes. Feedback: next local input =
`audio_embeddings[c](sampled_code)` (`emb_.embed_one(c+1, code)`). `to_local(global
hidden)` is identity for GPT-J. Binary channel-0 branch (`cfg_.binary_text_head`)
already present.

### 3. Harness reconcile

- `scripts/gen_local_v15_reference.py` already calls `audio_lm_heads[c](local_
  hidden)` directly (matches the real bare heads ✓) and the binary
  `local_text_lm_head`. Fix: audio width 1024 (drop the 1025 pad-slot assumption);
  confirm `input_ids` is `[S, n_vq+1=13]`; dump `qwen3.token_embd`-based
  `global_hidden`. It loads the real modeling via `trust_remote_code`.
- `tests/test_local_v15_parity.cpp` compares our GPT-J forward against the dump:
  `global_hidden`, `local_text_logits` (2-wide), `audio_logits.{0..11}` (1024),
  and 13 chosen `codes` exactly. Tolerances tuned on the first real run.
- **transformers pin**: the dumper needs `trust_remote_code` against modeling
  written for transformers 4.57.1; the venv currently has 5.13.0 (may break). Pin
  `transformers~=4.57` in the venv for the dumper step (converters are
  numpy/torch-only and don't need it).

## Testing

- Converter: `--help` + `ast.parse`; on the real v1.5 dir, `--strict` → **0
  unmapped**, and the printed local dims = hidden 2560 / n_layers 1 / n_heads 32 /
  head_dim 80. On the real Delay dir, still 0 unmapped, `local.arch=qwen3`.
- C++: existing Delay offline fixtures stay green (byte-identical). A **new offline
  GPT-J block fixture** (`gen_test_fixtures.py` adds a tiny GPT-J local + bare
  heads GGUF) gates the new block/head math without the real model, mirroring the
  Delay fixtures. Env-gated parity tests return 77 without the model (suite stays
  green in CI).
- **The real verification is the live parity gate** on the actual weights (this
  box, torch installed).

## Consequential-action checkpoints

- `pip install "transformers~=4.57"` into `.venv` (downgrade; for the dumper).
- Real-weight parity run (local, no upload).
- (Batch B, separate go-ahead) quantize + `hf upload`.

## File structure

| File | Change |
|---|---|
| `scripts/convert_moss_tts_local_to_gguf.py` | branch on config shape; v1.5 GPT-J local + bare heads + `transformer.*` backbone + 1024 vocab + `local.arch`/`bare_heads`/`norm_type`/`rope_base` metadata. |
| `src/local_transformer.{hpp,cpp}` | add GPT-J block (LayerNorm+bias, fused QKV, RoPE, silu 2-mat MLP), dispatched by `local.arch`. |
| `src/local_adapters.{hpp,cpp}` | `bare_heads` path: identity `to_local`, direct `lm_head[c]` head_logits (no out_mlp/head_norm). |
| `src/local_embeddings.{hpp,cpp}` | pad-code masking in embed-sum for 1024-vocab audio. |
| `src/moss_tts_local.{hpp,cpp}` | `LocalConfig`: `arch`, `bare_heads`; wire dispatch; 13-channel depth loop already present. |
| `src/model_loader.*` | read `local.arch`, `local.norm_type`, `local.rope_base`, `lc.bare_heads`, LayerNorm biases. |
| `scripts/gen_test_fixtures.py` | tiny GPT-J local + bare-head fixture for offline gating. |
| `scripts/gen_local_v15_reference.py` | 1024 audio width; `qwen3.token_embd` global hidden; assumption fixes. |
| `tests/test_local_v15_parity.cpp` | compare against the GPT-J/bare-head forward. |
| `tests/` (new GPT-J block/head fixtures test) | offline gate for the new block. |
| `AGENTS.md` | correct the v1.5 arch notes + runbook (transformers pin, `local.arch`). |

## Risks & mitigations

- **GPT-J RoPE/attn-scale details** — `scale_attn_weights` + head_dim 80 + rope
  base 1e6; the offline block fixture and the real-weight `audio_logits` maxerr
  localize any mismatch to the block.
- **LayerNorm vs RMSNorm** — the new block must use mean/var LayerNorm with bias;
  guarded by `norm_type` so Delay's RMS path is untouched.
- **Tied heads** — heads are emitted as concrete tensors (tie is upstream-only), so
  the GGUF is self-contained; no runtime tying needed.
- **transformers 5.x vs 4.57 modeling** — pinned downgrade for the dumper only.
- **The merged speculative v1.5 flags** (`binary_text_head`, `stereo`) partly
  survive; new flags (`arch`, `bare_heads`) supersede the wrong assumptions. The
  branch is un-merged, so no released behavior changes.
