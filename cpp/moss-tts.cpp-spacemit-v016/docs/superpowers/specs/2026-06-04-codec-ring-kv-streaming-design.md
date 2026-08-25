# Ring-KV Codec Streaming Perf Fix Design

**Status:** Approved design (brainstorming complete). Next: implementation plan
(`superpowers:writing-plans`).

**Part of:** the moss-tts.cpp program (the whole MOSS-TTS family is ported to
native ggml). This is a deferred **perf** follow-up for the V4 Nano stereo Cat
codec's streaming decode, flagged in AGENTS.md (V4 follow-up #5).

**Goal:** Replace the Nano codec's `decode_stream_step` windowed-lookback
(O(window)/step — re-runs the whole decoder tower over a retained window of code
frames each step, EXACT but redundant) with true **per-stage ring-KV** (a
persistent K/V cache in the decoder transformer stages), so each step computes
only the new frame(s) incrementally. The output MUST stay **numerically
identical** to `decode_full` (the streaming==full contract: maxerr ≤ 1e-3,
currently ~1.5e-8). This changes the compute factoring, not the math.

---

## 1. Problem

The codec decoder is a multi-stage tower: `dequantize` → alternating
`patch_up` (stateless up-sampling reshape) + `run_transformer` (a causal
RoPE-only transformer stage with a sliding-window `context`) → interleaved-stereo
PCM. The current streaming decode (`src/audio_tokenizer.cpp`
`decode_stream_step`) keeps a ring of the last `lookback` **input code frames**
and, each step, **re-runs the entire decoder tower** (`decode_block`) over that
whole window, emitting only the newest frame's samples. `lookback` is the
decoder's receptive field (`decoder_receptive_frames`,
`lookback = 1 + Σ ceil((context-1)/r)`). This is O(window) heavy compute per step
(redundant qkv projections + attention + FFN + patch_up over the whole window)
even though only one new frame's worth of output is needed.

The LLM backbones already avoid this: `qwen3_layer_forward` threads
`k_past`/`v_past` and `LocalTransformer::step` drives a per-frame host-side K/V
cache. The codec decoder transformer has **no KV-cache concept** today. The fix
brings the codec up to the same pattern, adding **ring eviction** (the codec's
attention is sliding-window, not grow-only-causal).

### Feasibility (confirmed)

True ring-KV is possible because the two structural preconditions hold:

1. **The only cross-time coupling in the entire decoder is attention K/V.**
   `layer_norm` (over the feature axis), `gelu_erf`, the `mul_mat`s, LayerScale
   (`ls1`/`ls2`), the quantizer's 1×1 projections, and `patch_up` are all
   per-token / stateless.
2. **`patch_up` is a pure stateless reshape with INTEGER ratios.** One input
   latent frame maps to an exact integer number of frames at every downstream
   stage rate — no fractional/partial-frame boundary.

The upstream reference ships this exact design (`_forward_streaming_sdpa` + a
per-MHA `MHAState` ring cache of length `context`), so the algorithm can be
ported line-for-line.

---

## 2. Architecture

Mirror the LLM KV-cache pattern (`qwen3_layer_forward` / `LocalTransformer::step`)
for the codec decoder transformer stages, plus ring eviction.

### 2.1 Additive K/V on the codec attention (`src/transformer.{hpp,cpp}`)

`run_transformer` (and its inner `attention`) gain **optional** per-layer
`k_past`/`v_past` inputs and return per-layer `k_full`/`v_full` outputs:

- `k_past`/`v_past` `nullptr` → today's behavior exactly (full-sequence
  attention over the input T). `decode_full` and the 24 kHz-mono Foundation
  decode (which share `transformer.cpp`) pass `nullptr` and stay **byte-identical**.
- When provided, each layer `ggml_concat`s the new K/V onto the cached K/V along
  the sequence axis (dim=2), runs attention over the concatenated K, and returns
  the post-RoPE, `ggml_cont`'d `k_full`/`v_full` (gallocr-safe graph outputs) —
  exactly `qwen3_layer_forward`'s shape. The codec attention is *simpler* than
  qwen3: plain MHA (no GQA), fused `qkv_w`, `GGML_ROPE_TYPE_NORMAL`, no q/k-norm;
  LayerScale is per-token and does not touch K/V.

The K/V threading is per-layer (a stage has `n_layers` transformer layers), like
`LocalTransformer::step`'s `k_past[l]`/`v_past[l]` arrays.

### 2.2 Per-stage incremental stream driver (`src/audio_tokenizer.{hpp,cpp}`)

Replace `NanoCodecStream` (the int32 code-frame ring + `decoder_receptive_frames`)
with a state holding, **per decoder transformer stage × per layer**, host-side
K/V buffers + a per-stage `past_len` (absolute frame count seen at that stage's
rate). `decode_stream_begin` keeps its opaque-handle API shape.

Per `decode_stream_step` (one input code frame of `nq` codes):

```
dequantize(new_frame_codes, k) -> latent           # 1 latent frame
running = latent
for each decoder stage s (in order):
    if s.kind == Patch:
        running = patch_up(running, s.patch_size)   # stateless, expands the running block by p
    else (Transformer stage s):
        kv = stream.stage_kv[s]                      # per-layer host K/V + past_len
        pos = absolute positions [kv.past .. kv.past + running_frames - 1]
        mask = sliding-window [running_frames (queries) x (cached + running) (keys)] @ s.context
        out, new_k/v = run_transformer(running, kv.k_past, kv.v_past, pos, mask)
        append new_k/v to kv; EVICT rows with abs-pos < (kv.past + running_frames - context); kv.past += running_frames
        running = out
emit running                                         # the new output PCM frames (downsample*channels samples)
```

### 2.3 KV semantics (the exactness crux)

- **Cache K/V post-RoPE at ABSOLUTE positions starting from 0.** These match
  `decode_full`'s local `0..T-1` positions **frame-for-frame** (both start at 0
  and increment at the stage's rate), so the streamed attention is bit-exact with
  the full decode. The new query is rotated at its absolute position; because
  RoPE is relative, scores depend only on `pos_q − pos_k`, which is preserved.
- **Ring eviction.** After appending the new frame's keys (absolute positions
  `past .. past+new-1`), drop cached K/V rows for positions
  `< (past + new) − context` (keep the last `context`). The query at absolute
  position `P` then attends to retained positions `P−context+1 .. P` via the
  `delta = i−j, allowed iff 0 ≤ delta < context` window mask — exactly
  `decode_full`'s window at `P`.
- **Unbounded context** (a stage with `context ≤ 0`) = grow-only (never evict).
  Ring-KV subsumes it; there is **no separate windowed-lookback fallback** (full
  replacement). The real config is all-bounded.
- **Per-stage rates.** Each transformer stage caches its own K/V at its own frame
  rate (12.5 → … → output Hz; the real-config decoder contexts are 500/800/
  1200/1600). `patch_up`'s integer ratios make one input latent frame expand to a
  fixed integer block at each stage, so each stage's `past_len` advances by an
  integer per step. No stage boundary yields a fractional frame.

### 2.4 Scratch-context reuse

Reuse the `make_ctx_buf` persistent-scratch pattern (added in the V2 local
KV-cache work) for the per-step graph, so the streaming step does not malloc a
fresh ggml ctx per call.

---

## 3. Components (files)

| File | Change |
|---|---|
| `src/transformer.hpp` | `run_transformer` signature gains optional per-layer `k_past`/`v_past` in + `k_full`/`v_full` out (a small struct or out-vector). `nullptr` past = unchanged. |
| `src/transformer.cpp` | `attention` concats `k_past`/`v_past` on the sequence axis, attends over the concat, returns contiguous `k_full`/`v_full`; `run_transformer` threads them per layer. Mirror `qwen3_layer_forward`. The `k_past=null` path is byte-identical to today. |
| `src/audio_tokenizer.hpp` | Replace `NanoCodecStream` with a stream state holding per-decoder-transformer-stage × per-layer host K/V (`std::vector`) + per-stage `past_len`; keep `decode_stream_begin`/`decode_stream_step` signatures + the opaque handle. |
| `src/audio_tokenizer.cpp` | Replace `decode_stream_step`'s window-re-decode with the incremental stage driver (§2.2): per-stage `patch_up` (stateless) + `run_transformer` with the stage's ring K/V + the `[new, cached+new]` sliding-window mask + eviction. Remove `decoder_receptive_frames` + the code-frame ring. Reuse `make_ctx_buf`. `decode`/`decode_block`/`encode` (the offline paths) unchanged. |
| `tests/test_nano_codec_stream.cpp` | Rewrite the driver to the new streaming API (same `streaming == decode_full` assertion); the per-frame loop is unchanged from a caller's view. |
| `scripts/gen_test_fixtures.py` | Extend `w_nano_codec` so `context < T` at **every** decoder transformer stage rate (not just the deepest), making each stage's ring eviction load-bearing. |
| `AGENTS.md` | Mark V4 follow-up #5 done (ring-KV streaming); update the streaming gotcha (it's no longer O(window)). |

**Reused as-is:** `patch_up` (stateless reshape, untouched), `dequantize`
(untouched), `make_ctx_buf` (from the V2 work), the `build_window_mask` /
`make_pos_input` helpers (extended for the streaming offset), the LLM-side
`qwen3_layer_forward` K/V pattern as the template.

---

## 4. Data flow (unchanged caller contract)

The public API is unchanged: `decode_stream_begin()` → repeated
`decode_stream_step(stream, codes_one_frame, &pcm_chunk)` → each call emits one
frame's interleaved-stereo PCM. Internally the per-step cost drops from
"re-decode the whole receptive window" to "advance each stage's ring K/V by the
new frame(s)": the new frame's qkv projection + FFN are O(new frames), the
attention is O(context) per stage (inherent — the query must see its window), and
the redundant re-projection/FFN/patch over the rest of the window is eliminated.

---

## 5. Error handling

- `decode_stream_step` returns `false` on graph-compute / readback failure (like
  `LocalTransformer::step`); the caller (the orchestrator + the streaming C-API)
  propagates.
- Every host-side K/V read-back is sized by explicit `kvn`/`ggml_nelements`
  (the established idiom); the eviction slice bounds are validated.
- Per-stage `past_len` is bounded by `context` after eviction, so it cannot
  overflow for bounded stages. An unbounded-context stage grows host K/V
  unboundedly over a long stream (documented; the real config is all-bounded).
- `decode_stream_begin` allocates the per-stage K/V vectors lazily (sized on the
  first step once `nq`/the stage dims are known) or up front from the loaded
  stage configs.

---

## 6. Testing & validation (exactness is the hard criterion)

The fix changes only the compute factoring; it MUST reproduce `decode_full`
exactly within the existing tolerance.

- **`test_nano_codec_stream`** (the contract): drive `decode_stream_step` one code
  frame at a time, accumulate the chunks, assert `streaming == decode_full` per
  sample (`maxerr ≤ 1e-3`; currently ~1.5e-8). Ring-KV must hold this.
- **Enhanced discriminating fixture** (`w_nano_codec`): choose the tiny codec's
  dims / `context_seconds` / `n_frames` so that **every** decoder transformer
  stage has `context < T` at its rate (the current fixture only makes the first
  stage's window load-bearing). This is the key new discriminator — it catches a
  per-stage RoPE-offset or eviction off-by-one that a single-stage-clipping
  fixture would miss. Keep it byte-reproducible (seeded, 4dp).
- **`decode_full` + Foundation byte-identical**: `test_audio_tokenizer_e2e`,
  `test_quantizer`, `test_partial_decode`, and the non-streaming `test_nano_codec`
  decode assertion must pass UNCHANGED (the `k_past=null` path in `transformer.cpp`
  is byte-identical — this is the additive guarantee).
- **(Optional) a focused per-stage-KV unit assertion**: one transformer stage's
  ring-KV step == its full-forward output for the corresponding frames (like
  `test_local_transformer`'s per-row step parity), to localize a per-stage bug.
- The full suite stays green; the LLM-side streaming + V1/V2/V3 are untouched.

**Parity rationale:** the streamed attention uses absolute positions that match
`decode_full`'s local positions frame-for-frame and the same `delta < context`
window mask, so it is the same RoPE-relative computation factored incrementally —
bit-exact within the existing tolerance, exactly as `LocalTransformer::step` is
bit-equivalent to a full forward.

---

## 7. Risks

1. **RoPE absolute-position + eviction + window-mask bookkeeping (the #1 risk).**
   An off-by-one between the retained window and the `delta < context` mask
   boundary, or a wrong absolute offset for the new query, silently breaks the
   1e-3 parity. Mitigated by: the absolute-positions-match-`decode_full`
   reasoning, porting the reference algorithm carefully, and the
   every-stage-clips test as the discriminator.
2. **Per-stage frame-rate handling.** Each stage caches at its own rate; `patch_up`
   expands the running block by its integer ratio. A wrong per-stage frame count
   would diverge — pinned by the multi-stage-clipping test.
3. **`decode_full` / Foundation regression.** The additive `k_past=null` path must
   be byte-identical (it shares `transformer.cpp` with the 24 kHz-mono codec).
   Guarded by the Foundation + decode parity tests.
4. **Exactness vs `decode_full`.** The streaming==full contract (≤1e-3) is the
   acceptance bar; bit-exactness follows from frame-for-frame position matching.

---

## 8. Scope boundaries (YAGNI)

**In scope:** the decoder transformer ring-KV (`transformer.cpp` additive K/V +
the new per-stage stream state + the incremental driver), **full replacement** of
the windowed-lookback `decode_stream_step`, the `test_nano_codec_stream` rewrite +
the every-stage-clips fixture, the docs update.

**Out of scope:** streaming **encode** (encode is offline, for cloning — unchanged);
any numerical change to `decode_full`/`decode`/`encode`; the LLM-side streaming
(already done); the other variants' codecs (only the Nano-stereo decode tower
streams — but `transformer.cpp` is shared, and `k_past=null` keeps every
non-streaming path byte-identical); the GPU `->data` path (separate follow-up).

---

## 9. Provenance / reuse

- **Template pattern:** `src/qwen3.cpp` `qwen3_layer_forward` (the `k_past`/`v_past`
  concat + `k_full`/`v_full` contiguous outputs) and `src/local_transformer.cpp`
  `step()` (the host-side per-layer K/V driver + `make_ctx_buf` scratch reuse).
  The codec attention adds ring eviction (the reference `MHAState` ring of length
  `context`).
- **Reference algorithm:** upstream `_forward_streaming_sdpa` + `MHAState` +
  `_build_streaming_sdpa_bias` (the `(delta >= 0) & (delta < context)` mask),
  ported line-for-line.
- **Validation templates:** `tests/test_nano_codec_stream.cpp` (the existing
  streaming==full contract) + `tests/test_local_transformer.cpp` (per-row step
  parity).
- **Commit policy:** `Assisted-by: Claude:claude-opus-4-8 [Claude Code]` trailer;
  NO `Co-Authored-By` / `Signed-off-by`.
