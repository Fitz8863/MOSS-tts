# V2 MossTTSLocal — Local Depth-Loop KV-Cache + Ctx-Reuse Perf Fix Design

**Status:** Approved design (brainstorming complete). Next: implementation plan
(`superpowers:writing-plans`).

**Part of:** the moss-tts.cpp program (the whole MOSS-TTS family is ported to
native ggml). This is a deferred **perf** follow-up for V2 (MossTTSLocal, 1.7B),
already flagged in the code (`src/moss_tts_local.cpp` `PERF FOLLOW-UP` markers)
and in AGENTS.md.

**Goal:** Eliminate the O(channels²) per-frame recompute **and** the per-channel
ggml-context allocation churn in the V2 depth loop, bringing `LocalTransformer`
up to the V3 `rt_local` / V4 `nano_local` per-frame-KV `step` pattern — with
**numerically identical** output (same exact codes, same logits within the
existing tolerances). This is a pure allocation/compute-shape change; it changes
no math.

---

## 1. Problem

The V2 local/depth transformer generates `channels = 1 + N_VQ = 33` codes per
frame via an autoregressive depth loop. Today that loop is **stateless and
whole-sequence**:

- The depth loop lives in the orchestrator (`src/moss_tts_local.cpp:151-193`).
  It maintains a growing `local_seq` and calls
  `LocalTransformer::forward_last(local_seq, i+1, &h)` once per channel `i`.
- `forward_last` (`src/local_transformer.cpp`) recomputes the **entire** growing
  no-RoPE causal stack from scratch each call: it calls `qwen3_layer_forward`
  with `nullptr, nullptr` for `k_past`/`v_past` and **discards** the returned
  `k_full`/`v_full`. So a frame runs 33 forwards of lengths 1,2,…,33 =
  1+2+…+33 = **561 row-forwards** instead of 33 → **O(channels²)**.
- Each `forward_last` call allocates a fresh **64 MB** ggml context + builds a
  fresh graph (`make_ctx(64*1024*1024, true)` per call). Additionally, in the
  loop, `LocalAdapters::to_local` and the per-channel head path
  (`head_logits[c]`) each allocate their own **8 MB** ggml context **per call**
  (`src/local_adapters.cpp`), i.e. ~`channels` × (to_local + head) mallocs/frame.

V3 (`src/rt_local.cpp`) and V4 (`src/nano_local.cpp`) already solved the
recompute with a per-frame KV cache + `step(in_vec, pos, &out)`. The shared
`qwen3_layer_forward` already supports `k_past`/`v_past` and returns
`k_full`/`v_full`, so the fix is **threading a per-frame KV cache through the
existing layer calls**, plus reusing scratch contexts — **no new ggml ops**.

The no-RoPE property of V2 local makes caching **simpler** than `rt_local`: RoPE
is the only consumer of the position in `qwen3_layer_forward` (gated by
`hp.use_rope`, false for V2 local), so a cached key is position-independent and
re-feeding it is trivially correct — no position threading needed.

---

## 2. Architecture

Two perf changes, both pure allocation/compute-shape (zero numerical change):

### 2.1 Per-frame KV cache in `LocalTransformer`

Replace the stateless `forward_last(growing_seq)` with a stateful `step` +
`reset` API mirroring `rt_local` (minus RoPE):

- `void reset();` — clear the per-frame KV cache (`past_len_ = 0`, clear each
  per-layer `k_state_`/`v_state_`). Called once per frame.
- `bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);`
  — feed ONE depth token (`in_vec` = `[hidden]`), build a single-token graph,
  thread the host-side past as `k_past`/`v_past` input leaves, run, read back the
  updated `k_full`/`v_full` (guarded) into host state, advance `past_len_`, apply
  the final `output_norm` (RMSNorm), return the single output row. `pos` is the
  depth index (= `past_len_` within a frame; used for the guard, NOT for RoPE).
- Private state: `int past_len_`, `std::vector<std::vector<float>> k_state_,
  v_state_` (per-layer accumulated K/V), persistent scratch ctx/buffer (§2.2).

The single-token attention is the same as `rt_local::step` except:
- **No RoPE / no position upload** (`use_rope=false`; the gallocr path already
  leaves `pos` unallocated under no-RoPE).
- **Mask is `[past+1, 1]` all-zeros** — the single query at depth `past` attends
  to all `past+1` cached depths (0..past). This is equivalent to the old full
  `t×t` causal mask because only depths 0..pos exist when stepping. This is the
  one spot that must be exact for bit-for-bit parity.
- K/V have `n_kv_heads` heads (Qwen3 GQA), `[head_dim, n_kv_heads, kv, 1]` — same
  as `rt_local` (the closer template than `nano_local`, which is GPT-2/no-GQA).

Per-frame work drops from O(channels²) layer-forwards to O(channels): each step
builds one token's Q/K/V/FFN attending to the cached K/V, instead of re-running
the whole 1..pos sequence.

### 2.2 Scratch-context reuse

The ggml-graph components reuse a **persistent scratch ggml context/buffer**
(owned once, reused per call) instead of `make_ctx(...)` per call:

- `LocalTransformer::step` — reuse a persistent metadata arena instead of the
  per-call 64 MB `make_ctx`.
- `LocalAdapters::to_local` and the per-channel head path (`head_logits`) —
  reuse a persistent metadata arena instead of the per-call 8 MB `make_ctx`.

Mechanism: each call still rebuilds its (small) graph, but into a **reused owned
buffer** passed to `ggml_init` (`no_alloc=true`, `mem_buffer` = the component's
owned scratch), so per-call context creation is a pointer reuse, not a malloc.
Re-initializing the ctx into the owned buffer each call yields a fresh metadata
arena (no stale nodes, no cross-call tensor aliasing). The persistent
gallocr/compute path is likewise reused. If `make_ctx` does not already support
an owned `mem_buffer`, a small additive extension (an overload taking a reusable
buffer) is in scope; it must not change the default per-call behavior used by
other components.

This eliminates the ~33 × (64 MB + 8 MB×2) mallocs/frame; the graphs and math are
unchanged.

---

## 3. Components (files touched)

| File | Change |
|---|---|
| `src/local_transformer.hpp` | Remove `forward_last`; add `reset()` + `step(in_vec, pos, &out)` + KV-state members (`past_len_`, `k_state_`, `v_state_`) + the persistent scratch buffer. |
| `src/local_transformer.cpp` | Implement `step`/`reset` by adapting `rt_local.cpp::step` (drop RoPE, `[past+1,1]` all-zeros mask, Qwen3 GQA K/V); use the reused scratch ctx. Remove `forward_last`. |
| `src/local_adapters.cpp` (+ the head path) | Reuse a persistent scratch ctx for `to_local` and `head_logits` instead of per-call `make_ctx`. |
| `src/ggml_extend.hpp` / wherever `make_ctx` lives | If needed, an additive `make_ctx` overload accepting a reusable owned buffer (default behavior unchanged). |
| `src/moss_tts_local.cpp` | Depth loop: drop the `local_seq` accumulator; `local_.reset()` per frame; `local_.step(cur, i, &h)` per channel. Mirror `src/moss_tts_rt.cpp:156-189`. |
| `tests/test_depth_loop.cpp` | Drive the depth loop via `step()` instead of `forward_last(local_seq,…)`. Same assertions. |
| `tests/test_local_transformer.cpp` | Rewrite from `forward_last(embeds, T)` to a `step()`-loop comparing each row's hidden to the numpy ref (mirror `test_rt_local.cpp`/`test_nano_local.cpp`). |
| `tests/test_local_parity.cpp` | Drive via `step()` (env-gated; same depth_logits + exact codes). |

**Reused as-is:** `qwen3_layer_forward` (already supports k_past/v_past — no
change), `LocalEmbeddings` (pure CPU gathers, untouched), the global
`DelayBackbone` (untouched), the sampling / orchestrator structure.

**Unaffected components:** `LocalEmbeddings::embed_one`/`embed_sum` (CPU row
gathers, no ggml ctx). The adapters/heads consume only the single last-row hidden
`h`; switching `forward_last`→`step` doesn't change how they're *called* per
channel — only the transformer's internal sequence handling + their ctx
allocation.

---

## 4. Data flow (semantics unchanged)

Per frame (mirrors today minus the growing-seq recompute):

```
local_.reset();                                  // clear per-frame KV
cur = to_local(global-hidden-derived row);       // depth-0 input (unchanged)
for c in 0..channels-1:                           // 33 channels
    local_.step(cur, c, &h);                      // ONE token over cached K/V
    head_logits[c](h) -> logit -> sample code     // unchanged
    if c+1 < channels:
        cur = to_local(embed_one(c, code));        // re-embed for next depth (unchanged)
```

The `to_local`/`embed_one`/`head_logits` chain is byte-identical to today; only
the transformer call (`forward_last(local_seq,…)` → `step(cur,c,…)`) and the
dropped `local_seq` accumulator change.

---

## 5. Error handling

- `step` returns `false` on graph-compute / readback failure (like `rt_local`);
  the orchestrator propagates the failure.
- All K/V-state read-backs are guarded by `ggml_nelements`/explicit sizes (the
  established V1/V2/V3 idiom).
- Scratch-ctx reuse must not leak or alias across calls: each call re-initializes
  the ctx into the owned buffer (a fresh metadata arena), so there is no stateful
  carry between calls and no per-call malloc.

---

## 6. Testing & validation (the hard acceptance criterion: bit-faithful)

The fix changes only compute shape + allocation, so it MUST reproduce the
existing references exactly.

- **`test_depth_loop`** (V2 keystone): the chosen `expected_codes` (2×channels,
  **greedy argmax**) must match **EXACTLY**, and the per-channel timestep-0
  logits within **TOL 1e-3** — identical to today. The exact-code assertion is
  the primary guard against any numerical drift from the KV-cache path.
- **`test_local_transformer`** (rewritten): drive `step()` over the sequence and
  compare each row's post-`output_norm` hidden to the numpy reference
  (`maxerr ≤ 1e-3`), exactly how `test_rt_local`/`test_nano_local` validate the
  step pattern (per-position parity + a `reset()` restart check).
- **`test_local_parity`** (1.7B, env-gated): same global hidden + per-channel
  `depth_logits.{i}` + argmax `codes` (TOL 5e-2); codes match exactly.
- A focused KV-cache unit assertion in the rewritten `test_local_transformer`:
  `step(i)` over the tiny fixture == the full-sequence reference row `i` for all
  `i`, plus a `reset()` correctness check (a second frame starts clean).
- Adapter/embedding/block tests (`test_local_adapters`, `test_local_embeddings`,
  `test_local_block`) are unaffected and must stay green (the scratch-ctx reuse
  changes allocation, not math — same graphs, same outputs).
- The full existing suite must stay green; V1/V3/V4 are untouched.

**Parity rationale:** the step path reads K/V back via `ggml_cont`'d graph
outputs — the same float math as the full recompute — so parity holds to the
existing tolerances, exactly as `test_rt_local`/`test_nano_local` already
demonstrate. The scratch-ctx reuse builds the identical graph into a reused
buffer, so it is numerically a no-op.

---

## 7. Risks

1. **Numerical drift from the KV cache** — none expected (same float math via
   `ggml_cont`'d readback; rt_local/nano_local prove it). Guarded by the
   keystone's EXACT-code assertion.
2. **Scratch-ctx reuse correctness** — a reused buffer must yield a fully fresh
   metadata arena each call (no stale nodes / aliasing). Mitigated by
   re-initializing the ctx into the owned buffer per call (pointer reuse, not a
   stateful carry). Pinned by the unchanged parity tests.
3. **The no-RoPE mask shape** — must be `[past+1, 1]` all-zeros (single query
   attends to all cached depths), equivalent to the old full `t×t` causal.
   Getting this wrong would diverge; pinned by the keystone.
4. **`make_ctx` reuse extension** — if `make_ctx` is extended for an owned
   buffer, the default per-call path used by other components must stay
   byte-identical (additive only).

---

## 8. Scope boundaries (YAGNI)

**In scope:** the `LocalTransformer` per-frame KV cache (`forward_last`→`step`),
the adapter/head scratch-ctx reuse, the orchestrator depth-loop update, and the
test updates. Bit-faithful output is the acceptance criterion.

**Out of scope:** any numerical change; the global `DelayBackbone` (unchanged);
V1/V3/V4 (this is V2-local-only — though the step pattern could later inform a
shared depth-transformer helper, that refactor is not part of this fix); new
sampling/features; GPU `->data` (the separate cross-cutting follow-up).

---

## 9. Provenance / reuse

- **Target pattern:** `src/rt_local.{hpp,cpp}` (V3) — the closest template
  (same `qwen3_layer_forward`, same RMSNorm `output_norm`, GQA K/V); adapt by
  dropping RoPE. `src/nano_local.{hpp,cpp}` (V4) is the same idiom on GPT-2.
- **Orchestrator template:** `src/moss_tts_rt.cpp` depth loop (`reset()` +
  per-depth `step()` + re-embed feed).
- **Validation templates:** `tests/test_rt_local.cpp` / `tests/test_nano_local.cpp`
  (per-position `step` parity + `reset()` restart).
- **Commit policy:** `Assisted-by: Claude:claude-opus-4-8 [Claude Code]` trailer;
  NO `Co-Authored-By` / `Signed-off-by`.
