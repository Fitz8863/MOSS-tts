# Reuse per-step scratch ctx buffers — Design

**Date:** 2026-06-05
**Status:** Approved (brainstorming)
**Scope:** Performance follow-up. Eliminate the per-step 256 MB `make_ctx` malloc in
the four remaining per-step graph builders.

## Problem

Every per-step inference graph is built in a `no_alloc=true` `ggml_context`. Four
of these builders allocate a fresh 256 MB context **every call**:

- `DelayBackbone::run` (`src/delay_backbone.cpp:88`) — once per frame (V1/V2/V3 backbone).
- `NanoBackbone::run` (`src/nano_backbone.cpp:87`) — once per frame (V4 backbone).
- `RtLocal::step` (`src/rt_local.cpp:78`) — once per depth position (V3 depth loop).
- `NanoLocal::step` (`src/nano_local.cpp:80`) — once per depth position (V4 depth loop).

Each `make_ctx(256*1024*1024, /*no_alloc=*/true)` does `ggml_init` with
`mem_buffer=nullptr`, which mallocs the 256 MB metadata arena, and the
`GgmlCtxPtr` deleter frees it when the call returns — an mmap/munmap pair plus
the page faults for the (few-MB) touched portion, **per step**. The arena is
wildly over-provisioned: the actual metadata (graph node array + the per-op
tensor structs) is ~1–3 MB. The depth-loop steps (`RtLocal`/`NanoLocal`) run
16–32 times per frame, so the churn is most significant there.

The V2-local perf fix already solved this for the other per-step builders by
reusing a caller-owned buffer via `make_ctx_buf`:
`LocalTransformer::step` (`local_transformer.cpp:67`, member
`std::vector<uint8_t> step_scratch_` sized 64 MB), `LocalAdapters::to_local`/
`head_logits`, and the codec streaming step (`audio_tokenizer.cpp:498`). These
four builders were simply not part of that pass. This change completes the
thread.

## Goal

Replace the per-call 256 MB `make_ctx` in the four builders with a reused,
tightly-sized member buffer via `make_ctx_buf` — removing the per-step
malloc/free churn. The graph and all numerics are unchanged → **byte-identical
on CPU**.

Non-goal: any change to the graphs, the KV caches (the depth loops keep their
host-side additive per-frame K/V; the backbones keep the device-resident cache
from the prior follow-up), the converters, or any other allocation site.

## Approach (chosen)

**Computed tight bound**, per class. Each builder gets a member
`std::vector<uint8_t> scratch_` sized once at load to a provably-safe bound tied
to that builder's graph node budget, then reuses it every step via
`make_ctx_buf`. This is the V2-local idiom with a size derived from the node
budget rather than a flat 64 MB — keeping steady RSS at ~3–7 MB per instance
(important for the deliberately lightweight Nano variant).

Rejected alternatives:
- *Flat generous size (32–64 MB):* simplest and matches `LocalTransformer`
  verbatim, but adds tens of MB resident per loaded model.
- *Reuse 256 MB once:* zero sizing risk but 256 MB resident per instance — bad
  for Nano.

## The sizing bound

For a builder whose graph is created with `ggml_new_graph_custom(ctx, B, false)`:

```cpp
scratch_.resize(ggml_tensor_overhead() * 2 * B
              + ggml_graph_overhead_custom(B, false)
              + (1u << 20));   // 1 MB margin
```

Rationale (robust without hand-counting ops):
- Every `ggml_*` op call creates exactly one tensor struct in the ctx; all are
  reachable from the graph outputs, so the tensor count equals the op count.
- `ggml_build_forward_expand` collects them as nodes + leaves; the graph created
  with budget `B` holds at most `B` nodes and (in `ggml_new_graph_custom`) `B`
  leaf slots, so the number of tensor structs is bounded by `2*B`. `2*B *
  ggml_tensor_overhead()` covers all of them.
- The graph object itself lives in the ctx and costs
  `ggml_graph_overhead_custom(B, false)`.
- +1 MB absorbs alignment/object-header slack.
- `B` is the **pre-existing** binding constraint: a graph that exceeded `B`
  nodes would already fail at `ggml_new_graph_custom` today. So sizing to `B` is
  exactly correct, not a new assumption.

The bound is independent of layer count and of `T`/sequence length: graph
*metadata* (tensor structs + node array) is T-independent (only tensor *shapes*,
which live in the gallocr data buffer, scale with T). So one fixed-size buffer
serves prefill and every decode/depth step.

Node budgets in the four builders today: `DelayBackbone` 4096, `NanoBackbone`
8192, `RtLocal` 4096, `NanoLocal` 4096. (≈ 3–4 MB for `B=4096`, ≈ 6–7 MB for
`B=8192`.) Each class sizes from its own `B`.

## The per-class change (uniform)

1. **Header** (`*.hpp`): add a private member `std::vector<uint8_t> scratch_;`
   (each of the four headers already includes `<vector>`).
2. **`load`**: after the metadata/layers are loaded, size the buffer once:
   `scratch_.resize(ggml_tensor_overhead()*2*B + ggml_graph_overhead_custom(B,false) + (1u<<20));`
   with the literal `B` matching that builder's `ggml_new_graph_custom` budget.
   (`RtLocal`/`NanoLocal` `load` builds layers then returns — add the resize at
   the end. The backbones already have a `load`; add the resize after the KV
   cache alloc.)
3. **`run`/`step`**: change
   `auto cctx = moss::make_ctx(256*1024*1024, /*no_alloc=*/true);`
   to
   `auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);`.
   Nothing else in the builder changes — the graph build, input upload, compute,
   and readback are identical.

`make_ctx_buf` (`ggml_extend.hpp`) initializes ggml with the caller's
`mem_buffer`, so `ggml_init` does not malloc; `GgmlCtxPtr`'s `ggml_free` releases
only the context struct, leaving the member-owned buffer intact for the next
call. This is the exact pattern `LocalTransformer::step` already uses.

## Byte-identity rationale

The graph (ops, tensor shapes, data, the gallocr compute path) is unchanged;
only the ctx's backing memory source changes (per-call heap → reused member
buffer). `make_ctx_buf` vs `make_ctx` differ solely in who owns the metadata
arena. So every output is bit-for-bit identical on CPU — the same
equivalence-proof standard as the V2-local ctx-reuse and the KV-cache fixes.

## Testing

- **Byte-identical keystones (the acceptance):** the CI-always test exercising
  each converted builder stays green at the **same** numbers —
  `test_delay_kv` (DelayBackbone), `test_nano_backbone` (NanoBackbone),
  `test_rt_depth_loop` (RtLocal), `test_nano_frame_loop` + `test_nano_local`
  (NanoLocal). Note the maxerr/exact-codes before and after; they must match.
- **No regression:** full `ctest` green (63/63); the depth-loop keystones and
  parity tests unchanged.
- **Sizing safety:** the keystones build the real per-layer graphs for the tiny
  fixtures; if `scratch_` were too small, `ggml_init`/`ggml_new_tensor` would
  return null and the build would crash — so a green keystone IS the sizing
  proof. The `2*B` bound holds for any layer count because `B` is the graph's
  node budget (the pre-existing binding constraint), not a per-layer estimate.
  During development the implementer may log `ggml_used_mem(ctx)` after the build
  to confirm it sits well under `scratch_.size()` (expected a few MB), but no
  assertion is needed in the committed code.
- **Grep-clean:** no new raw tensor `->data`; the four files keep only
  `out->data()` host writes.

## File structure

| File | Change |
|------|--------|
| `src/delay_backbone.hpp` | add `std::vector<uint8_t> scratch_;` |
| `src/delay_backbone.cpp` | size `scratch_` in `load` (B=4096); swap the `make_ctx` line in `run`. |
| `src/nano_backbone.hpp` | add `std::vector<uint8_t> scratch_;` |
| `src/nano_backbone.cpp` | size `scratch_` in `load` (B=8192); swap the `make_ctx` line in `run`. |
| `src/rt_local.hpp` | add `std::vector<uint8_t> scratch_;` |
| `src/rt_local.cpp` | size `scratch_` in `load` (B=4096); swap the `make_ctx` line in `step`. |
| `src/nano_local.hpp` | add `std::vector<uint8_t> scratch_;` |
| `src/nano_local.cpp` | size `scratch_` in `load` (B=4096); swap the `make_ctx` line in `step`. |
| `AGENTS.md` | note the ctx-reuse completed across all per-step builders. |

No converter/GGUF/weight/codec/embed/head changes.

## Risks & mitigations

- **Under-sized scratch → null tensor / crash.** Mitigated by the `2*B`-bounded
  formula (provably ≥ the graph's tensor+node metadata) and the keystones, which
  build the real graphs and would fail loudly. `B` is already the binding graph
  constraint today.
- **Re-entrancy.** The buffer is a per-instance member reused across calls;
  inference is single-threaded and not re-entrant (identical to
  `LocalTransformer::step`). No concurrency change.
- **Resident memory.** ~3–7 MB per converted instance (vs ~0 transient before);
  the tight bound keeps this minimal, deliberately avoiding the 64/256 MB
  alternatives so Nano stays light.
- **Buffer lifetime vs ctx.** The `scratch_` member outlives every `make_ctx_buf`
  ctx (the ctx is a local `GgmlCtxPtr` freed at end of each call); `ggml_free`
  does not free the external buffer. Correct by construction.
