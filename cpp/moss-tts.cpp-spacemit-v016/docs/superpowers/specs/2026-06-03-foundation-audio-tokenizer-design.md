# Foundation: native MOSS-Audio-Tokenizer in ggml

**Status:** approved design — ready for implementation planning
**Date:** 2026-06-03
**Sub-project:** F1 (repo scaffold) + F2 (MOSS-Audio-Tokenizer) of the
MOSS-TTS → ggml porting program.

## Context

We are porting the OpenMOSS **MOSS-TTS family** to ggml/C++, the same way
[`parakeet.cpp`](../../../../parakeet.cpp) and
[`vibevoice.cpp`](../../../../vibevoice.cpp) port NeMo Parakeet and
Microsoft VibeVoice: a Python converter turns upstream checkpoints into a
metadata-driven GGUF, and a self-contained C++/ggml engine runs inference
with no Python/ONNX/torch dependency, exposed through a flat C-API for
`dlopen`/FFI/LocalAI.

The user's goal is "all of them, fully native ggml," benchmarked against
the upstream implementation. That is a multi-spec **program**, decomposed
dependency-first:

```
FOUNDATION (this spec) ── shared by every variant
  F1  repo scaffold (CMake + ggml + dr_wav + loader + audio_io + C-API/CLI)
  F2  MOSS-Audio-Tokenizer in ggml: encoder (wav→codes) + decoder (codes→wav)
        ↓ every variant decodes through F2
V1  MossTTSDelay (8B)      ← Qwen3 backbone + 33 emb/heads + delay SM + sampling
                              (the only variant with an existing ggml-ish
                               reference: upstream llama.cpp + ONNX pipeline,
                               with a published Seed-TTS-eval WER table)
V2  MossTTSLocal (1.7B)    ← "depth-transformer" backbone, time-sync RVQ blocks
V3  MossTTSRealtime (1.7B) ← hierarchical text-audio inputs + streaming
V4  MOSS-TTS-Nano (~100M)  ← separate upstream repo, 48 kHz
```

Order: **F1 → F2 → V1 → V2 → V3 → V4.** Each gets its own spec → plan →
implementation cycle. F2 comes before any variant because every variant's
audio is produced by this one codec, it is the riskiest novel ggml work
(it is ONNX-only upstream), and it is the only piece that can be validated
completely standalone (reconstruction round-trip vs the ONNX tokenizer —
no backbone needed).

This spec covers **F1 + F2 only**. The Delay backbone/TTS generation is V1.

## Goals / non-goals

**Goals**
- A self-contained repo (`libmoss-tts` + `moss-tts-cli`) that builds on
  stock ggml with no fork.
- Native ggml MOSS-Audio-Tokenizer: encode 24 kHz waveform → 32 RVQ code
  streams (12.5 Hz), and decode codes → 24 kHz waveform.
- A Python converter: upstream safetensors → GGUF.
- `moss-tts-cli reconstruct in.wav out.wav` that round-trips audio matching
  the upstream ONNX tokenizer within an SNR / spectral-distance threshold,
  with no Python/ONNX/torch at inference time.

**Non-goals (deferred to later specs)**
- The Qwen3 backbone, embeddings, LM heads, delay state machine, sampling,
  prompt building, text tokenizer — all V1+.
- Variants V2–V4.
- Training / fine-tuning. Inference only.
- Streaming codec inference (offline full-sequence first; the sliding-window
  attention is implemented so streaming is a later add-on, not a rewrite).

## Reference sources (single sources of truth)

In order of trust, for tensor-by-tensor debugging:
1. **`OpenMOSS-Team/MOSS-Audio-Tokenizer`** on HF — `config.json` (dims) and
   `modeling_moss_audio_tokenizer.py` (module logic). Authoritative.
2. **`Blaizzy/mlx-audio`** `mlx_audio/codec/models/moss_audio_tokenizer/` —
   clean non-PyTorch reference for the algorithms.
3. **`OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX`** — the encoder/decoder ONNX
   we benchmark against (this is what the upstream torch-free pipeline uses).
4. The model's `model.safetensors.index.json` — authoritative tensor names.

When a numerical bug is suspected, dump the matching tensor from (1) and
diff. Magnitude alone is not a reliable signal (vibevoice lesson).

## Architecture

### Top-level data flow

```
waveform (1, T @24kHz, f32, NO normalization)
  → right-pad T up to a multiple of 1920
  → ENCODER  (8 modules)                 → latent (768, T/1920)
  → ResidualLFQ.encode                   → codes  (32, T/1920)   [10-bit indices]
codes (32, N)
  → ResidualLFQ.decode                   → latent (768, N)
  → DECODER  (8 modules)                 → waveform (1, N*1920)
```

Frame rate 24000/1920 = **12.5 Hz**. Each frame = 32 codes × 10 bits =
4 kbps at full depth; first `k` codebooks → `k × 0.125` kbps.

**Preprocessing: none.** Raw float waveform in [-1,1]; no DC removal, no
gain, no normalization. Only zero-pad to a multiple of 1920.

### Shared building blocks

**`PatchedPretransform`** (no weights) — the CNN-free resampler. On `(D,T)`
channel-first tensors:
- down: `(D,T) → reshape (D, T/p, p) → permute (D, p, T/p) → reshape (D*p, T/p)`
- up:   `(D*p, L) → reshape (D, p, L) → permute (D, L, p) → reshape (D, L*p)`

The interleave order matters — getting it wrong silently corrupts audio.
Encode and decode must mirror exactly.

**`ProjectedTransformer`** — a stage. Tensors enter/leave channel-first
`(D,T)`; internally sequence-first `(T,D)`.
```
x = input_proj(x)        # Linear(in→d_model, bias=False); identity if in==d_model
for layer in layers: x = layer(x)
x = output_proj(x)       # Linear(d_model→out, bias=False); identity if d_model==out
```
Layer (pre-norm, RoPE, LayerScale, plain erf-GELU FFN):
```
r = x; x = norm1(x); a = self_attn(x); x = r + layer_scale_1 * a
r = x; x = norm2(x); f = linear2(gelu(linear1(x))); x = r + layer_scale_2 * f
```
- **LayerNorm WITH weight and bias**, eps = 1e-5 (not RMSNorm).
- **Activation: exact erf-GELU** (not tanh, not SwiGLU). FFN is two
  bias-free linears.
- **LayerScale**: learnable per-channel vector (length d_model) multiplying
  the residual branch output. Do not fold into the linears.
- **Attention: MHA** (not GQA). Fused QKV `in_projs.0 (3*d_model, d_model)`,
  `out_projs.0 (d_model, d_model)`, both bias-free. `head_dim = 64`,
  SDPA scale `head_dim^-0.5`.
- **RoPE** on Q and K, `max_period = 10000`, interleaved-pair convention,
  freqs `exp(-(2i/d)·ln(max_period))`. Streaming offset = 0 for offline.
- **Sliding-window causal mask**: key j visible to query i iff
  `0 ≤ i-j < context`, where `context = round(frame_rate_at_stage_input × 10s)`.
  Track `current_frame_rate` exactly as the constructor does: start 24000,
  divide by each module's downsample ratio (encoder) / multiply (decoder)
  *after* appending. For clips shorter than the window this equals plain
  causal; implement the window so long audio matches bit-for-bit.

Two block configs:

| config | d_model | heads | head_dim | layers | ff   |
|--------|--------:|------:|---------:|-------:|-----:|
| small  | 768     | 12    | 64       | 12     | 3072 |
| large  | 1280    | 20    | 64       | 32     | 5120 |

### Encoder (8 modules, in order)

| idx | module              | params           | dims in→out      | T factor |
|----:|---------------------|------------------|------------------|---------:|
| 0   | PatchedPretransform | patch_size=240   | 1 → 240 ch       | ÷240     |
| 1   | Transformer (small) | in240→d768→out384| input_proj+output_proj | ×1 |
| 2   | PatchedPretransform | patch_size=2     | 384 → 768 ch     | ÷2       |
| 3   | Transformer (small) | in768→d768→out384| output_proj only | ×1      |
| 4   | PatchedPretransform | patch_size=2     | 384 → 768 ch     | ÷2       |
| 5   | Transformer (small) | in768→d768→out640| output_proj only | ×1      |
| 6   | PatchedPretransform | patch_size=2     | 640 → 1280 ch    | ÷2       |
| 7   | Transformer (large) | in1280→d1280→out768 | output_proj only | ×1   |

Latent `(768, T/1920)`. Only `encoder.1` has `input_proj`; all four
transformer stages have `output_proj` (identity input projections are
skipped — no matmul when the tensor is absent).

### Quantizer — `ResidualLFQ` (`rlfq`)

Config: `input_dim=768, rvq_dim=512, output_dim=768, num_quantizers=32,
codebook_size=1024, codebook_dim=8`.

Module structure (all WNConv1d k=1 = per-frame linear WITH bias, fused at
convert time):
- `input_proj` 768→512, `output_proj` 512→768.
- 32× LFQ, each: `in_proj` 512→8, `out_proj` 8→512, `codebook` (1024, 8).

**Encode:**
```
z = input_proj(latent)              # (512, T)
residual = z; quantized = 0
for i in 0..31:
    z_e   = in_proj_i(residual)     # (8, T)
    e     = L2_normalize_rows(z_e)              # eps 1e-12
    cb    = L2_normalize_rows(codebook_i)       # (1024, 8)
    idx   = argmin_k ||e - cb_k||²  ( = argmax cosine )   # (T,)
    z_q   = out_proj_i( gather(codebook_i, idx) )         # (512, T)
    quantized += z_q; residual -= z_q
    codes[i] = idx                  # 10-bit
# output_proj(quantized) is computed upstream but NOT used by decode
```

**Decode:**
```
emb = 0                              # (512, T)
for i in 0..31:
    emb += out_proj_i( gather(codebook_i, codes[i]) )
latent = output_proj(emb)           # (768, T)
```
The decode path uses only `codebook`, per-codebook `out_proj`, and the
shared `output_proj`. No EMA, no straight-through, no extra scaling at
inference — argmin + gather only.

### Decoder (8 modules, in order)

| idx | module              | params         | dims in→out         | T factor |
|----:|---------------------|----------------|---------------------|---------:|
| 0   | Transformer (large) | in768→d1280→out1280 | input_proj only | ×1   |
| 1   | PatchedPretransform | patch_size=2   | 1280 → 640 ch       | ×2       |
| 2   | Transformer (small) | in640→d768→out768   | input_proj only | ×2 (T)|
| 3   | PatchedPretransform | patch_size=2   | 768 → 384 ch        | ×2       |
| 4   | Transformer (small) | in384→d768→out768   | input_proj only | ×1   |
| 5   | PatchedPretransform | patch_size=2   | 768 → 384 ch        | ×2       |
| 6   | Transformer (small) | in384→d768→out240   | input_proj+output_proj | ×1 |
| 7   | PatchedPretransform | patch_size=240 | 240 → 1 ch          | ×240     |

Output `(1, N*1920)` raw 24 kHz waveform. **No output head, no tanh, no
clamp** — the final unpatchify *is* the samples. `decoder.0/2/4/6` have
`input_proj`; only `decoder.6` has `output_proj`.

## GGUF tensor inventory

PyTorch layout: `Linear.weight` is `(out, in)`; LayerNorm has weight+bias;
LayerScale `scale` size = d_model. Names use the original PyTorch
hierarchy (not mlx renames).

Per transformer block `{enc:1,3,5,7 / dec:0,2,4,6}`, per layer L:
```
{blk}.transformer.layers.{L}.norm1.weight / .bias            (d_model,)
{blk}.transformer.layers.{L}.norm2.weight / .bias            (d_model,)
{blk}.transformer.layers.{L}.self_attn.in_projs.0.weight     (3*d_model, d_model)
{blk}.transformer.layers.{L}.self_attn.out_projs.0.weight    (d_model, d_model)
{blk}.transformer.layers.{L}.linear1.weight                  (ff, d_model)
{blk}.transformer.layers.{L}.linear2.weight                  (d_model, ff)
{blk}.transformer.layers.{L}.layer_scale_1.scale             (d_model,)
{blk}.transformer.layers.{L}.layer_scale_2.scale             (d_model,)
```
Block projections (present only when dims differ):
```
encoder.1.input_proj.weight  (768,240)   encoder.1.output_proj.weight (384,768)
encoder.3.output_proj.weight (384,768)   encoder.5.output_proj.weight (640,768)
encoder.7.output_proj.weight (768,1280)
decoder.0.input_proj.weight  (1280,768)  decoder.2.input_proj.weight  (768,640)
decoder.4.input_proj.weight  (768,384)   decoder.6.input_proj.weight  (768,384)
decoder.6.output_proj.weight (240,768)
```
Quantizer (WNConv → fuse to dense linear at convert time):
```
quantizer.input_proj  : parametrizations.weight.original0 (512,1,1)=g,
                        parametrizations.weight.original1 (512,768,1)=v → (512,768); bias (512,)
quantizer.output_proj : original0 (768,1,1), original1 (768,512,1) → (768,512); bias (768,)
quantizer.quantizers.{i}.in_proj  : → (8,512);  bias (8,)
quantizer.quantizers.{i}.out_proj : → (512,8);  bias (512,)
quantizer.quantizers.{i}.codebook.weight        (1024,8)
```
**WNConv fusion** (at convert time): squeeze kernel dim → `v2d (out,in)`;
`norm_o = sqrt(Σ_in v2d²)` per output row `(out,1)`; `W = g.reshape(out,1) *
v2d / norm_o`; keep stored bias. Result is a plain `(out,in)` linear.

GGUF metadata to emit: sample_rate=24000, downsample=1920, frame_rate=12.5,
num_quantizers=32, codebook_size=1024, codebook_dim=8, rvq_dim=512,
context_seconds=10, and per-stage block configs/ratios so the loader can
build the towers from metadata.

Checkpoint is ~1.77B params (~7.1 GB fp32) across 2 safetensors shards.

## Components and boundaries

| unit | does | depends on |
|------|------|-----------|
| `audio_io` | dr_wav load/save, linear resample to 24 kHz | dr_wav |
| `model_loader` | GGUF → config struct + name→ggml_tensor map | ggml, gguf |
| `backend` | ggml backend init + persistent `ggml_gallocr` | ggml |
| `patchify` | build ggml down/up reshape-permute subgraph | ggml |
| `transformer` | build ProjectedTransformer subgraph (RoPE MHA, LayerScale, GELU FFN, sliding-window mask) | ggml, model_loader |
| `quantizer` | encode (argmin loop) + decode (gather+sum) subgraphs | ggml, model_loader |
| `audio_tokenizer` | wire encoder/quantizer/decoder; expose encode()/decode()/reconstruct() | all above |
| `moss_tts` / `_capi` | C++ API + flat C-API shim | audio_tokenizer |
| `cli` | info / encode / decode / reconstruct subcommands | C++ API |

Each unit is independently testable on tiny random weights without the real
checkpoint.

## Performance invariants (carried from parakeet.cpp)

- Keep a **persistent `ggml_gallocr`** reused across graphs; do not swap in
  `ggml_backend_sched` on the fast path (it regressed CUDA there). Use sched
  only as a per-graph fallback when a backend lacks an op kernel.
- **Zero-copy weights**: clone loader tensors by reference; never copy
  weights per call.

## Testing & benchmark (definition of done)

**Model-independent** (no checkpoint, run in CI):
- `test_audio_io` — dr_wav round-trip + resample.
- `test_patchify` — down∘up == identity; fixed-pattern interleave check.
- `test_rope` — cos/sin table vs reference values.
- `test_transformer_block` — forward numerics on tiny random weights vs a
  committed reference dump.
- `test_quantizer` — argmin selection + decode gather on tiny weights vs
  reference.

**Model-dependent** (env-gated, `SKIP_RETURN_CODE=77`):
- `test_load` — real GGUF opens, config matches.
- `test_reconstruct_parity` — **the milestone gate**: encode→decode a real
  clip; compare waveform to the upstream ONNX tokenizer's output for the
  same input; assert SNR ≥ threshold (and/or log-spectral distance ≤
  threshold). Also assert encode codes match ONNX codes above an
  exact-match rate.

Env vars: `MOSS_TTS_TOKENIZER` (gguf), `MOSS_TTS_CLI`, optional
`MOSS_TTS_ONNX_REF` (path to ONNX-produced reference wav/codes, or a
committed fixture).

**Benchmark** (`bench.sh`): reconstruction RTF on CPU and CUDA vs the ONNX
tokenizer on the same clips; report RTF and codes-exact-match rate.

## Known gotchas (pre-loaded from the research pass)

1. No input normalization — feed raw waveform, only zero-pad to ×1920.
2. erf-GELU, not tanh; FFN is plain 2-matmul, not SwiGLU.
3. LayerNorm has bias (eps 1e-5); it is not RMSNorm.
4. LayerScale sits between sublayer output and the residual add — never fold
   it into a linear.
5. RoPE interleaved-pair convention, head_dim 64, max_period 10000, Q and K.
6. MHA not GQA; fused QKV in `in_projs.0`.
7. Sliding-window context is stage-dependent (frames in 10 s at that stage's
   frame rate); track current_frame_rate exactly through the module list.
8. Patchify interleave order must mirror exactly on decode.
9. WNConv weights are weight-normalized in the checkpoint — fuse at convert
   time; all quantizer projections are 1×1 convs = per-frame linears + bias.
10. Quantizer L2-normalizes both the projected encoding and the codebook
    (eps 1e-12) before Euclidean/cosine argmin; codebook space is 8-dim.
11. Decoder output is the waveform directly from the final patch-240
    unpatchify — no output conv, tanh, or clamp.
12. Bitrate control = use first `k` of 32 codebooks on encode/decode.

## Conventions

- C++17, no exceptions across the C-API (return status codes).
- One translation unit per logical component; keep `moss_tts.cpp` /
  `moss_tts_capi.cpp` thin.
- Comments explain non-obvious *why*, not *what*.
- Commit policy follows the sibling repos' AGENTS.md: an `Assisted-by:`
  trailer for AI involvement, no `Co-Authored-By` / `Signed-off-by` from AI.
- Converter run with `--strict` to fail on unmapped source keys.

## Open questions / risks

- **Exact parity threshold** for `test_reconstruct_parity` (SNR dB / LSD)
  will be tuned once we have the first real reconstruction vs ONNX; quantized
  GGUF will loosen it.
- **Sliding-window vs ONNX**: confirm the ONNX export bakes the same window;
  if ONNX is plain-causal, our long-audio output could differ from ONNX even
  while matching PyTorch — decide reference accordingly during the parity
  test.
- **Memory**: ~7 GB fp32; expect to develop against a Q8_0 GGUF. Compute
  pool sizes must scale with sequence length (vibevoice lesson).
```
