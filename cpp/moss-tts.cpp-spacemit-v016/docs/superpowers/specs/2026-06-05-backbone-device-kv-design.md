# Device-resident growing KV cache for the global backbones — Design

**Date:** 2026-06-05
**Status:** Approved (brainstorming)
**Scope:** Performance follow-up. Eliminate the O(T²) host↔device KV-cache
roundtrip in the two global (time-axis) backbones.

## Problem

Both global backbones keep their per-layer K/V cache as **host `std::vector`s**
("approach A"):

- `DelayBackbone` (Qwen3 — shared by V1 Delay, V2 Local, V3 Realtime),
  `src/delay_backbone.cpp`.
- `NanoBackbone` (GPT-2 — V4 Nano), `src/nano_backbone.cpp`.

Each `decode_one` (`run(..., T, ...)`) does, **per layer, per step**:

1. **Upload** the full grown prior cache `k_state_[l]` / `v_state_[l]`
   (size `head_dim · n_kv · past`) into device input leaves `k_past[l]` /
   `v_past[l]` via `ggml_backend_tensor_set`.
2. Concat the new `T` columns inside `qwen3_layer_forward` /
   `gpt2_layer_forward` → `k_full` / `v_full`.
3. **Read back** the full new cache (size `head_dim · n_kv · (past+T)`) to host
   via `ggml_backend_tensor_get`, storing into `k_state_[l]` / `v_state_[l]`.

Over a `T`-frame autoregressive generation this is **O(T²)** host↔device
transfer. On the CPU backend it is O(T²) `memcpy`; on a GPU build (now
unblocked by the GPU `->data` device-safety fix) it is a full-KV PCIe roundtrip
**every step**. For an 8B Qwen3 (36 layers, `head_dim=128`, `n_kv=8`) at ~1000
frames this moves hundreds of MB per step — the dominant remaining per-frame
cost. The depth loops (`LocalTransformer` / `rt_local` / `nano_local`) already
avoid this because their per-frame caches are small/bounded; the **backbone**
cache grows unbounded with the sequence.

This is the natural next step after the codec ring-KV streaming fix (which made
the *codec* decoder incremental) and the V2 depth-loop KV fix — but applied to
the **global LLM backbones**, whose cache is the unbounded-growth one.

## Goal

Keep the per-layer K/V **resident on the backend buffer**, append the new
position(s) in place each step, and read the cache prefix **by view**
(zero-copy) into attention. This drops KV **data movement** from O(T²) to O(T)
(only the new column is written per step; the read is a view), while the
attention **compute** (inherently O(past) per step) and the resulting **bytes**
are unchanged → **byte-identical on CPU**, large win on long sequences / GPU.

Non-goal: changing the codec (`transformer.cpp`), the depth-loop transformers,
the GGUF format/converters, or any numeric result.

## Approach (chosen)

**Opt-in device-KV-cache path in the shared layer-forward helpers**, plus a
persistent cache buffer owned by each backbone. The existing additive
`k_past`/`v_past` path is retained unchanged for every other caller.

Rejected alternatives:
- *Driver-only (freeze shared helpers):* would duplicate the attention block
  into the backbone driver and diverge it from the depth-loop callers.
- *Device-device ping-pong concat:* keeps the additive interface but still
  copies the **full** KV each step (O(T²) device-device) → **zero** win on the
  CPU backend, only partial on GPU.

## Architecture

### 1. Persistent device KV buffer (per backbone)

At `load(m, max_seq)` each backbone allocates a **dedicated** `ggml_context`
(no_alloc) holding, per layer:

- `k_cache[l]` : `ggml_new_tensor_4d(GGML_TYPE_F32, head_dim, n_kv, max_seq, 1)`
- `v_cache[l]` : same shape

then materializes them on the backend with
`ggml_backend_alloc_ctx_tensors(ctx, moss::backend())` (held in a member
`ggml_backend_buffer_t kv_buffer_`, freed in the destructor / on reload). This
is a **separate** buffer from the gallocr compute scratch, so it survives across
the many per-step graphs (the gallocr buffer is recycled each compute).

Layout rationale: `(head_dim, n_kv, seq)` with **seq outermost** means a prefix
`[.., .., 0:past+T]` and a write slice `[.., .., past:past+T]` are both
**contiguous** (stride along dim2 = `head_dim · n_kv`), so the read-view is
`mul_mat`-safe and the store-view is a contiguous `ggml_cpy` target. (Same
layout principle the codec ring-KV used.)

Sizing: `head_dim · n_kv · max_seq · n_layers · 2 · sizeof(f32)`. This replaces
the host `k_state_`/`v_state_` vectors (which held the same data) — net memory is
neutral-to-better (removes the host copy and the per-step transient gallocr
`k_past`/`k_full` allocations).

`reset()` sets `past_len_ = 0` only. The buffer is **not** zeroed: every read is
a `[0:past+T]` view, so positions `>= past` are never read before being written
this step.

`max_seq` overflow: `run` already bounds positions to `max_seq`; assert/guard
`past_len_ + T <= max_seq_` and fail gracefully (return false) if exceeded
(current behavior already assumes the caller respects `max_seq`).

### 2. Opt-in cache path in the shared helper

`qwen3_layer_forward` (`src/qwen3.{hpp,cpp}`) and `gpt2_layer_forward`
(`src/gpt2.{hpp,cpp}`) gain **null-defaulted trailing params**:

```cpp
Qwen3LayerOut qwen3_layer_forward(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pos,
    struct ggml_tensor* mask, struct ggml_tensor* k_past, struct ggml_tensor* v_past,
    const Qwen3Layer& w, const Qwen3Hparams& hp,
    struct ggml_cgraph* gf = nullptr,          // NEW: graph, for kv-store expand
    struct ggml_tensor* k_cache = nullptr,     // NEW: persistent cache [hd,nkv,max_seq,1]
    struct ggml_tensor* v_cache = nullptr,
    int past_seq = 0);                         // NEW: write offset / read width base
```

`gpt2_layer_forward` gets the analogous four params. Existing callers
(`local_transformer.cpp`, `rt_local.cpp`, `nano_local.cpp`, and the current
backbone drivers before they are updated) omit them → `k_cache == nullptr` →
**existing path, unchanged**.

When `k_cache != nullptr` (cache path), after computing the new `k` / `v`
(post-proj, post-q/k-norm, post-RoPE — **identical** to today) of shape
`[head_dim, n_kv, T, 1]`:

```cpp
// store the T new columns into the cache at sequence offset past_seq
struct ggml_tensor* k_dst = ggml_view_4d(ctx, k_cache, hd, nkv, T, 1,
        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
        (size_t)past_seq * k_cache->nb[2]);
struct ggml_tensor* k_store = ggml_cpy(ctx, k, k_dst);
ggml_build_forward_expand(gf, k_store);     // insertion order ⇒ runs before the read
// (same for v_store)

// read the valid prefix [0 : past_seq + T] by view (zero-copy)
struct ggml_tensor* k_used = ggml_view_4d(ctx, k_cache, hd, nkv, past_seq + T, 1,
        k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], 0);
// (same for v_used)
```

Attention then uses `k_used` / `v_used` exactly where the additive path uses
`k_full` / `v_full`. The `ggml_build_forward_expand(gf, k_store)` call inserts
the store node into the graph **before** the attention nodes for this layer, so
the read-after-write on the shared buffer is correctly ordered by execution
(node) order — the standard ggml kv-store idiom. `Qwen3LayerOut` /
`Gpt2LayerOut` carry the store nodes back so the driver does not need to
re-expand them, but the in-helper expand is what guarantees ordering:

```cpp
struct Qwen3LayerOut { struct ggml_tensor *y=nullptr, *k_full=nullptr, *v_full=nullptr;
                       struct ggml_tensor *k_store=nullptr, *v_store=nullptr; };
```

(`k_full`/`v_full` remain populated on the additive path; `k_store`/`v_store`
on the cache path. Mutually exclusive.)

### 3. Backbone driver (`run`)

`DelayBackbone::run` / `NanoBackbone::run` switch to the cache path:

- Build the no_alloc graph; for each layer call the helper with
  `gf, k_cache_[l], v_cache_[l], past` (and `k_past=nullptr, v_past=nullptr`).
- `ggml_build_forward_expand(gf, y)` for the final output (the per-layer stores
  were already expanded inside the helper).
- Upload only `x` (embeds) and `pos`. **No** `k_past` upload, **no**
  `k_full`/`k_state_` readback.
- **Mask:** for the `T == 1` decode step every key `[0:past+1]` is causally
  valid (the single query is at the newest position), so the additive mask is
  all-zeros — pass `mask = nullptr` to `ggml_soft_max_ext` (adding a zero mask is
  identical to no mask → byte-identical) and skip the O(past) mask upload. For
  prefill (`T = S > 1`) keep the `[S, S]` causal mask (uploaded once). This keeps
  the per-step transfer genuinely O(hidden); the helper already treats a null
  `mask` as "no additive bias".
- `compute_graph_with_inputs`, read back only `y` (the last hidden), then
  `past_len_ += T`.

Remove the `k_state_` / `v_state_` members and their per-step set/get loops.
`prefill` is `run(..., S, past=0)`; `decode_one` is `run(..., 1, past=past_len_)`
— same code path, only `T` and `past` differ.

## Data flow (per decode step, cache path)

```
embed[hidden] ──set──▶ x leaf
                       pos[T] ──set──▶ leaf   (mask = null for T=1 decode; [S,S] only on prefill)
  for each layer l:
    k,v (proj+norm+rope, T cols) ──cpy──▶ k_cache[l][past:past+T]   (store, expanded)
    attention(q, k_cache[l][0:past+T] view, v_cache[l][0:past+T] view, mask)
  y[hidden] ◀──get── (only readback)
  past_len_ += T
```

No per-step KV upload/readback; per-step transfer is O(hidden) in + out.

## Byte-identity rationale

The cache path computes the **same** K/V values (same projections, q/k-norm,
RoPE at the same absolute positions) and runs the **same** attention over the
**same** logical K/V matrix — `view[0:past+T]` of the cache holds exactly the
bytes that the additive path's `concat(k_past, k)` produced. The only difference
is *where the bytes live* and *that they are read via a view instead of a fresh
concat*. On the CPU backend the result is bit-for-bit identical. This is the
same equivalence-proof standard used by the codec ring-KV and V2 depth-loop
fixes.

## Testing

- **CI-always consistency gates (the acceptance):** `test_delay_kv` (prefill+
  decode_one == full prefill — the KV consistency gate) and `test_nano_backbone`
  (prefill then decode_one carrying the prefill K/V) must stay green at the
  **same maxerr** as before. These exercise exactly the changed path.
- **Discrimination check (new):** confirm the consistency tests actually pin the
  cache indexing — a deliberately wrong store offset (`past+1`) or wrong read
  width must make `test_delay_kv` / `test_nano_backbone` **fail**. If the current
  fixtures don't discriminate (e.g. too few positions, or values too smooth),
  strengthen them (more positions and/or sharper embeddings) so a wrong
  offset/width is caught — then revert the deliberate break. Document the
  revert-to-fail evidence.
- **No regression:** full `ctest` green (the depth-loop keystones
  `test_depth_loop` / `test_rt_depth_loop` / `test_nano_frame_loop`, the codec
  tests, and every component/parity test pass unchanged — they use the null
  (additive) path).
- **Grep-clean:** `src/` stays free of new raw tensor `->data` reads (use the
  backend API / views only).
- **Env-gated parity** (`test_backbone_parity`, e2e, closed-loop) unchanged —
  byte-identical so they behave exactly as before when run on real weights.

## File structure

| File | Change |
|------|--------|
| `src/qwen3.hpp` | `Qwen3LayerOut` gains `k_store`/`v_store`; `qwen3_layer_forward` gains 4 null-defaulted params (`gf`, `k_cache`, `v_cache`, `past_seq`). |
| `src/qwen3.cpp` | Implement the `k_cache != null` store+view path; additive path untouched. |
| `src/gpt2.hpp` | `Gpt2LayerOut` gains `k_store`/`v_store`; `gpt2_layer_forward` gains the same 4 params. |
| `src/gpt2.cpp` | Implement the cache path; additive path untouched. |
| `src/delay_backbone.hpp` | Add `kv_ctx_`/`kv_buffer_` + `k_cache_`/`v_cache_` members; drop `k_state_`/`v_state_`. |
| `src/delay_backbone.cpp` | Allocate the cache buffer in `load`; switch `run` to the cache path; free buffer in dtor; `reset` resets `past_len_`. |
| `src/nano_backbone.hpp` | Same member changes as delay. |
| `src/nano_backbone.cpp` | Same `load`/`run`/`reset`/dtor changes. |
| `tests/test_delay_kv.cpp` | Strengthen if needed so a wrong offset/width fails; otherwise unchanged (it already gates consistency). |
| `tests/test_nano_backbone.cpp` | Same. |
| `AGENTS.md` | Mark the backbone-KV host-roundtrip as DONE; note device-resident cache + residuals. |

`local_transformer.cpp`, `rt_local.cpp`, `nano_local.cpp`, `transformer.cpp`
(codec), the converters, and all GGUF/weight code are **untouched**.

## Risks & mitigations

- **Read-after-write ordering on the shared cache buffer.** Mitigated by
  `ggml_build_forward_expand(gf, k_store)` *inside* the helper before the
  attention read is built → the store node precedes the read in execution order.
  The discrimination check verifies the indexing end-to-end.
- **gallocr trying to allocate the cache leaves.** The cache tensors are
  allocated on their own backend buffer (have backing `data`), so the gallocr
  treats them as pre-allocated leaves and does not touch them — same mechanism
  as the model weights.
- **Non-contiguous views.** Avoided by the seq-outermost layout (prefix and slice
  are contiguous); verified by the consistency tests.
- **max_seq overflow.** Guarded in `run` (`past_len_ + T <= max_seq_`).
- **Blast radius to V1–V4 / codec.** The new path is opt-in (null-defaulted);
  every non-backbone caller stays on the additive path and is byte-identical, as
  proven by the unchanged depth-loop keystones + codec tests.
