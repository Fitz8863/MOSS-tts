# V2 Local Depth-Loop KV-Cache + Ctx-Reuse Perf Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the O(channels²) per-frame recompute and the per-channel ggml-context allocation churn in the V2 MossTTSLocal depth loop, bringing `LocalTransformer` up to the V3 `rt_local` per-frame-KV `step` pattern — with **numerically identical** output (same exact codes, same logits within the existing tolerances).

**Architecture:** Replace `LocalTransformer::forward_last(growing_seq)` with `reset()` + `step(in_vec, pos, &out)` + a per-frame host-side K/V cache threaded through the existing `qwen3_layer_forward` (which already supports `k_past`/`v_past`). Each step computes ONE new token over cached K/V (no-RoPE, so simpler than `rt_local` — no position threading). Separately, reuse a persistent scratch ggml buffer in `LocalTransformer::step` / `LocalAdapters::to_local` / `head_logits` instead of `make_ctx(...)` per call. This is a pure allocation/compute-shape change; it changes no math.

**Tech Stack:** ggml (pinned `third_party/ggml`), C++17, GGUF, numpy parity fixtures, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-06-04-v2-local-kv-perf-design.md`.

**Key parity insight:** by causality, `step(pos)` (single token attending to cached rows 0..pos) == row `pos` of a full T-length no-RoPE causal forward == today's `forward_last(seq[0:pos+1], pos+1)`. So the per-row fixture reference validates `step` exactly, and the keystone's exact-code assertion is preserved.

**Cross-cutting conventions:**
- Commit trailer EXACTLY: `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`. **NO** `Co-Authored-By`, **NO** `Signed-off-by`.
- Build/test: `. .venv/bin/activate && cmake --build build -j && ctest --test-dir build --output-on-failure`.
- The whole existing suite must stay green at every commit; V1/V3/V4 are untouched.
- Every `->data`/`ggml_backend_tensor_get` read is sized by `ggml_nelements`/an explicit resize first.

**Template files (read, mirror):** `src/rt_local.{hpp,cpp}` (the `step`/`reset` + KV pattern — adapt minus RoPE), `src/moss_tts_rt.cpp` lines 154-191 (the orchestrator reset()+step() loop), `tests/test_rt_local.cpp` (the step-loop parity test), `src/ggml_extend.hpp` (`make_ctx` + `GgmlCtxPtr`).

---

## File Structure

- `src/ggml_extend.hpp` — ADD a reusable-buffer `make_ctx_buf(void*, size_t, bool)` (additive; default `make_ctx` unchanged).
- `src/local_transformer.{hpp,cpp}` — ADD `reset()` + `step()` + KV-state members + a persistent scratch buffer; REMOVE `forward_last` (after all callers migrate).
- `src/local_adapters.{hpp,cpp}` — reuse persistent scratch buffers in `to_local` + `head_logits`.
- `src/moss_tts_local.cpp` — depth loop: `reset()` per frame + `step(in,i,&h)` per channel; drop `local_seq`; remove the two `PERF FOLLOW-UP` comments.
- `scripts/gen_test_fixtures.py` — `w_local_transformer` (or equivalent): dump per-row `out [H,T]` for the step-loop parity test.
- `tests/test_local_transformer.cpp` — rewrite to a `step()`-loop (mirror `test_rt_local`).
- `tests/test_depth_loop.cpp`, `tests/test_local_parity.cpp` — rewrite their inline depth loops to `reset()`+`step()`.
- `AGENTS.md` — mark the V2 depth-loop perf follow-up DONE.

---

## Task 1: Reusable-buffer `make_ctx_buf` + `LocalTransformer::step`/`reset` (KV cache) + per-row parity test

**Files:**
- Modify: `src/ggml_extend.hpp` (add `make_ctx_buf`)
- Modify: `src/local_transformer.hpp`, `src/local_transformer.cpp` (add `reset`/`step`; keep `forward_last`)
- Modify: `scripts/gen_test_fixtures.py` (per-row `out` in the local-transformer fixture)
- Modify: `tests/test_local_transformer.cpp` (rewrite to step-loop)

This task adds the KV-cache `step`/`reset` alongside the existing `forward_last` (which stays until Tasks 2-3 migrate the remaining callers), and validates `step` per-row against the V3 `test_rt_local` pattern.

- [ ] **Step 1: Add `make_ctx_buf` to `src/ggml_extend.hpp`** (right after the existing `make_ctx`, lines 25-32). The existing `GgmlCtxPtr` deleter (`ggml_free`) is already correct for an external buffer — `ggml_free` does NOT free a caller-provided `mem_buffer`, only the small context struct. Add:

```cpp
// Like make_ctx, but uses a CALLER-OWNED, reusable mem_buffer instead of having
// ggml malloc one internally. The caller owns `mem_buffer` (keep it alive while
// the ctx is used); ggml_free (via GgmlCtxPtr) frees only the context struct,
// not the buffer. Use for per-call scratch contexts that would otherwise malloc
// mem_size bytes every call.
inline GgmlCtxPtr make_ctx_buf(void* mem_buffer, size_t mem_size, bool no_alloc) {
    struct ggml_init_params p = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ mem_buffer,
        /*.no_alloc   =*/ no_alloc,
    };
    return GgmlCtxPtr(ggml_init(p));
}
```

- [ ] **Step 2: Update the local-transformer fixture to dump per-row output.** In `scripts/gen_test_fixtures.py`, find the local-transformer fixture generator (the one producing `local_transformer.gguf` with `embeds [H,T]` + `last_hidden [H]`; it runs a no-RoPE qwen3 stack — likely uses a `_qwen3_layer_np(..., use_rope=False)` helper). ADD a per-row output tensor `out [H,T]`: run the same no-RoPE causal stack over the full T-length `embeds`, apply the final `output_norm` RMSNorm to EVERY row, and store the result as `out` (ne0=H, ne1=T), rounded to 4 dp. Keep `last_hidden` (== the last column of `out`) so the old assertion's reference still exists if needed. Regenerate: `python scripts/gen_test_fixtures.py local_transformer` (use the real subcommand name). Confirm byte-reproducible.

- [ ] **Step 3: Rewrite `tests/test_local_transformer.cpp` to a step-loop (FAILING test first).** Mirror `tests/test_rt_local.cpp` lines 48-76. Load `local_transformer.gguf`; read `embeds [H,T]` + `out [H,T]`. Drive:
```cpp
LocalTransformer lt;
if (!lt.load(ld)) return 77;            // 77 only if the fixture/model is missing
const int H = lt.hidden();
// Case 1: per-row KV accumulation across the growing depth sequence.
lt.reset();
double worst = 0;
for (int pos = 0; pos < T; ++pos) {
    std::vector<float> in_vec(emb + (size_t)pos*H, emb + (size_t)(pos+1)*H);
    std::vector<float> h;
    if (!lt.step(in_vec, pos, &h)) { std::fprintf(stderr, "step %d failed\n", pos); return 1; }
    double e = maxerr(h, out + (size_t)pos*H, H);   // out[pos] = row pos of the full causal forward
    if (e > worst) worst = e;
    if (e > 1e-3) { std::fprintf(stderr, "step pos=%d maxerr=%g > 1e-3\n", pos, e); return 1; }
}
// Case 2: reset() restarts the per-frame cache; step(.,0) again == out[0].
lt.reset();
std::vector<float> in0(emb, emb + H), h0;
if (!lt.step(in0, 0, &h0)) { return 1; }
if (maxerr(h0, out, H) > 1e-3) { std::fprintf(stderr, "post-reset mismatch\n"); return 1; }
```
(Use the file's existing `maxerr` helper / tensor-read idiom; size-guard the `out`/`embeds` reads with `ggml_nelements`.) This FAILS to compile/link (`reset`/`step` undefined).

- [ ] **Step 4: Add `reset()` + `step()` + KV state to `src/local_transformer.hpp`.** Mirror `rt_local.hpp`. Replace the `forward_last` declaration block with BOTH (keep `forward_last` for now — Tasks 2-3 remove it):
```cpp
    int hidden() const { return hp_.hidden; }
    void reset();   // clear the per-frame KV cache (call before each frame's depth loop)
    // feed one depth token at position `pos` (0..channels-1); returns post-output_norm hidden [hidden]. out resized.
    bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);
    // DEPRECATED (removed in this change once all callers migrate to step):
    bool forward_last(const std::vector<float>& embeds, int t, std::vector<float>* last_hidden);
private:
    Qwen3Hparams hp_{};
    std::vector<Qwen3Layer> layers_;
    struct ggml_tensor* output_norm_ = nullptr;
    const ModelLoader* m_ = nullptr;
    int past_len_ = 0;
    std::vector<std::vector<float>> k_state_, v_state_;   // per-layer accumulated K/V
    mutable std::vector<uint8_t> step_scratch_;           // reused metadata arena for step()
```
Add `#include <cstdint>` if not present.

- [ ] **Step 5: Implement `reset()` + `step()` in `src/local_transformer.cpp`.** In `load()`, after `layers_.assign(...)`, add `k_state_.assign(hp_.n_layers, {}); v_state_.assign(hp_.n_layers, {});` and size the scratch once: `step_scratch_.resize(64 * 1024 * 1024);`. Add:
```cpp
void LocalTransformer::reset() {
    past_len_ = 0;
    for (auto& s : k_state_) s.clear();
    for (auto& s : v_state_) s.clear();
}

bool LocalTransformer::step(const std::vector<float>& in_vec, int pos,
                            std::vector<float>* out_hidden) {
    const int L = hp_.n_layers, H = hp_.hidden, hd = hp_.head_dim, nkv = hp_.n_kv_heads;
    const int past = past_len_;
    const int kv = past + 1;
    if (pos != past) return false;
    if (!out_hidden || in_vec.size() < (size_t)H) return false;

    // Reuse the persistent scratch buffer (no per-call malloc). no_alloc=true:
    // tensor DATA lives on the backend via the global gallocr; this ctx holds
    // only graph metadata.
    auto cctx = moss::make_ctx_buf(step_scratch_.data(), step_scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();
    if (!ctx) return false;

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_set_input(x);
    // pos leaf exists for the qwen3 signature but is never referenced (use_rope=false):
    // gallocr skips it and we must NOT upload it (mirrors forward_last).
    struct ggml_tensor* posn = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_input(posn);
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, 1);  // ne0=key, ne1=query
    ggml_set_input(mask);

    std::vector<struct ggml_tensor*> k_past(L, nullptr), v_past(L, nullptr);
    std::vector<struct ggml_tensor*> k_out(L, nullptr),  v_out(L, nullptr);

    struct ggml_tensor* h = x;
    for (int l = 0; l < L; ++l) {
        if (past > 0) {
            k_past[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nkv, past, 1);
            v_past[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nkv, past, 1);
            ggml_set_input(k_past[l]); ggml_set_input(v_past[l]);
        }
        auto lo = qwen3_layer_forward(ctx, h, posn, mask, k_past[l], v_past[l], layers_[l], hp_);
        h = lo.y; k_out[l] = lo.k_full; v_out[l] = lo.v_full;
        ggml_set_output(k_out[l]); ggml_set_output(v_out[l]);
    }

    struct ggml_tensor* y = ggml_mul(ctx, ggml_rms_norm(ctx, h, hp_.rms_eps), output_norm_);
    ggml_set_output(y);

    auto* gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, y);
    for (int l = 0; l < L; ++l) {
        ggml_build_forward_expand(gf, k_out[l]);
        ggml_build_forward_expand(gf, v_out[l]);
    }

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, in_vec.data(), 0, (size_t)H * sizeof(float));
        // Do NOT upload posn (unreachable under use_rope=false).
        // Single query at depth `past` attends to all kv cached depths (all-zeros, causal by depth).
        std::vector<float> mvec((size_t)kv, 0.0f);
        ggml_backend_tensor_set(mask, mvec.data(), 0, mvec.size() * sizeof(float));
        for (int l = 0; l < L; ++l) {
            if (k_past[l]) {
                ggml_backend_tensor_set(k_past[l], k_state_[l].data(), 0, k_state_[l].size() * sizeof(float));
                ggml_backend_tensor_set(v_past[l], v_state_[l].data(), 0, v_state_[l].size() * sizeof(float));
            }
        }
    };

    if (!moss::compute_graph_with_inputs(gf, set_inputs)) return false;

    const size_t kvn = (size_t)hd * nkv * kv;
    for (int l = 0; l < L; ++l) {
        k_state_[l].resize(kvn); v_state_[l].resize(kvn);
        ggml_backend_tensor_get(k_out[l], k_state_[l].data(), 0, kvn * sizeof(float));
        ggml_backend_tensor_get(v_out[l], v_state_[l].data(), 0, kvn * sizeof(float));
    }
    past_len_ = kv;

    out_hidden->resize(H);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * sizeof(float));
    return true;
}
```
This is `RtLocal::step` minus the RoPE `posn` upload (the only difference besides prefix). Keep `forward_last` unchanged in the file for now.

- [ ] **Step 6: Build + run the new test + full suite.**
```bash
. .venv/bin/activate && python scripts/gen_test_fixtures.py local_transformer && cmake --build build -j && ctest --test-dir build -R test_local_transformer --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: `test_local_transformer` PASS (per-row maxerr ≤ 1e-3 + reset check); full suite still green (forward_last still used by the orchestrator + test_depth_loop + test_local_parity).

- [ ] **Step 7: Commit**
```bash
git add -A && git commit -m "feat(v2): add LocalTransformer step/reset (per-frame KV cache) + make_ctx_buf + per-row parity test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Migrate the orchestrator depth loop + the keystone test to `reset()`+`step()`

**Files:**
- Modify: `tests/test_depth_loop.cpp` (the V2 keystone — inline depth loop)
- Modify: `src/moss_tts_local.cpp` (the orchestrator depth loop, lines ~144-211)

Both drive the same loop; the keystone exact-code assertion is the bit-faithful guard.

- [ ] **Step 1: Rewrite the keystone `tests/test_depth_loop.cpp` inline loop to `reset()`+`step()`.** Replace the `local_seq` accumulator + `forward_last(local_seq, i+1, &h)` (lines ~134-142) with `reset()` before the channel loop + `step()`:
```cpp
        // per frame:
        std::vector<float> cur;
        adapt.to_local(gh, &cur);          // depth-0 input (unchanged)
        local.reset();                     // clear per-frame KV
        std::vector<int32_t> codes(channels);
        for (int i = 0; i < channels; ++i) {
            std::vector<float> h;
            if (!local.step(cur, i, &h)) { std::fprintf(stderr, "local.step failed ts=%d i=%d\n", ts, i); return 1; }
            std::vector<float> lg;
            adapt.head_logits(i, h, &lg);
            // ... (the existing t0_logit cmp + argmax, UNCHANGED) ...
            int code = argmax(lg);
            codes[i] = code;
            // re-embed for the next depth (unchanged):
            std::vector<float> e1; emb.embed_one(i, code, &e1);
            std::vector<float> c2; adapt.to_local(e1, &c2);
            cur = c2;
        }
```
Keep the `t0_logit.{i}` comparison (TOL 1e-3), the `expected_codes` EXACT check, and the final gate IDENTICAL. The loop now uses `cur` (single carrier) instead of the growing `local_seq`.

- [ ] **Step 2: Run the keystone → it MUST still PASS with EXACT codes.**
```bash
cmake --build build -j && ctest --test-dir build -R test_depth_loop --output-on-failure
```
Expected: PASS — same `expected_codes` (exact argmax) + t0 logits within 1e-3. This proves `step` is bit-faithful to `forward_last` on the keystone. If codes differ, the `step` wiring (mask shape / KV readback) is wrong — fix Task 1 before proceeding.

- [ ] **Step 3: Migrate the orchestrator `src/moss_tts_local.cpp` depth loop.** Replace the `local_seq` + `forward_last(local_seq, i+1, &h)` block (lines ~148-159) with `reset()` + `step()`, mirroring `src/moss_tts_rt.cpp:154-191`:
```cpp
    for (int step = 0; step < opts.max_new_tokens; ++step) {
        std::vector<float> cur;
        adapt_.to_local(gh, &cur);
        local_.reset();
        std::vector<int32_t> next(channels);
        for (int i = 0; i < channels; ++i) {
            std::vector<float> h;
            if (!local_.step(cur, i, &h)) {
                MOSS_LOGE("LocalTTS::tts: local step failed step=%d ch=%d", step, i);
                return false;
            }
            std::vector<float> lg;
            adapt_.head_logits(i, h, &lg);
            // ... (the existing per-channel temperature + sample_token + hist, UNCHANGED) ...
            next[i] = code;
            hist[i].push_back(code);
            std::vector<float> e1; emb_.embed_one(i, code, &e1);
            std::vector<float> c2; adapt_.to_local(e1, &c2);
            cur = c2;
        }
        // ... (gen_audio collect + embed_sum + decode_one + stop check, UNCHANGED) ...
    }
```
Remove the two `// PERF FOLLOW-UP:` comments (lines ~154-155 and ~162-163) — the KV-cache half is now addressed (the to_local/head_logits ctx-alloc half is Task 4).

- [ ] **Step 4: Build + full suite.**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -6
```
Expected: full suite green (the env-gated `test_e2e_local`/`test_closed_loop_local` still SKIP; `test_depth_loop` passes). `forward_last` is now used only by `test_local_parity` (removed in Task 3).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "perf(v2): drive the local depth loop via reset()+step() (KV cache, no O(channels^2) recompute)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Migrate `test_local_parity` + remove `forward_last`

**Files:**
- Modify: `tests/test_local_parity.cpp` (env-gated 1.7B parity)
- Modify: `src/local_transformer.{hpp,cpp}` (remove the now-unused `forward_last`)

- [ ] **Step 1: Rewrite the `tests/test_local_parity.cpp` growing-seq loop to `reset()`+`step()`.** Replace the `local_seq` + `forward_last(local_seq, i+1, &h)` block (lines ~122-153) with `local.reset();` before the loop and `local.step(cur, i, &h)` inside, keeping the `depth_logits.{i}` comparison (TOL 5e-2) + the `ref_codes[i]` exact check + the `embed_one`/`to_local` re-embed (which sets `cur`) IDENTICAL:
```cpp
    local.reset();
    double depth_maxerr = 0; bool codes_ok = true;
    for (int i = 0; i < channels; ++i) {
        std::vector<float> h;
        if (!local.step(cur, i, &h)) { std::fprintf(stderr, "local.step failed at i=%d\n", i); return 1; }
        std::vector<float> lg; adapt.head_logits(i, h, &lg);
        // ... (depth_logits.{i} cmp_maxerr + argmax + ref_codes check, UNCHANGED) ...
        std::vector<float> e1; emb.embed_one(i, code, &e1);
        adapt.to_local(e1, &cur);
    }
```
(This test SKIPs 77 unless `MOSS_TTS_LOCAL` + `MOSS_LOCAL_REF_DUMP` are set; it compiles + SKIPs in CI.)

- [ ] **Step 2: Remove `forward_last` from `src/local_transformer.hpp` and `src/local_transformer.cpp`.** Delete the `forward_last` declaration (the DEPRECATED line) from the header and the entire `forward_last` implementation from the `.cpp`. Confirm no remaining references: `grep -rn forward_last src/ tests/` returns nothing.

- [ ] **Step 3: Build + full suite.**
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -6
```
Expected: clean build (no `forward_last`), full suite green (`test_local_parity` compiles + SKIPs without the checkpoint).

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "refactor(v2): migrate test_local_parity to step() and remove forward_last

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: Scratch-context reuse in `LocalAdapters::to_local` + `head_logits`

**Files:**
- Modify: `src/local_adapters.hpp` (persistent scratch buffers), `src/local_adapters.cpp` (reuse them)

The two `make_ctx(8*1024*1024, false)` sites malloc 8 MiB per call (~`channels` × 2 per frame). Reuse a persistent buffer. These use `no_alloc=false` (tensor DATA lives in the ctx buffer — `x->data`/`y->data` are memcpy'd directly), so the reused buffer holds data too.

- [ ] **Step 1: Add persistent scratch buffers to `src/local_adapters.hpp`.** In the private members (after `m_`), add (both methods are `const`, so `mutable`):
```cpp
    mutable std::vector<uint8_t> to_local_scratch_;   // reused ctx buffer for to_local
    mutable std::vector<uint8_t> head_scratch_;       // reused ctx buffer for head_logits
```
Add `#include <cstdint>` if not present. Size them once in `load()` (after the dims are known): `to_local_scratch_.resize(8 * 1024 * 1024); head_scratch_.resize(8 * 1024 * 1024);`.

- [ ] **Step 2: Reuse the buffer in `to_local`.** In `src/local_adapters.cpp`, replace `auto ctx = make_ctx(8 * 1024 * 1024, false);` with:
```cpp
    auto ctx = make_ctx_buf(to_local_scratch_.data(), to_local_scratch_.size(), /*no_alloc=*/false);
```
Everything else (the `ggml_new_tensor_2d` + `memcpy` into `x->data` + `moss_mlp` + `ggml_new_graph` + `compute_graph` + `memcpy` out of `y->data`) is UNCHANGED.

- [ ] **Step 3: Reuse the buffer in `head_logits`.** Replace its `auto ctx = make_ctx(8 * 1024 * 1024, false);` with:
```cpp
    auto ctx = make_ctx_buf(head_scratch_.data(), head_scratch_.size(), /*no_alloc=*/false);
```
The rest (the out-MLP + RMSNorm + `lc.head_norm.{c}` / `lc.lm_head.{c}` matmul + the pad-mask) is UNCHANGED.

- [ ] **Step 4: Build + run the adapter test + the keystone + full suite.** This is purely an allocation change — same graphs, same math, so outputs must be byte-identical.
```bash
cmake --build build -j && ctest --test-dir build -R "test_local_adapters|test_depth_loop|test_local_block" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: `test_local_adapters` + `test_depth_loop` PASS (exact same codes/logits as before); full suite green.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "perf(v2): reuse a persistent scratch ctx in LocalAdapters to_local/head_logits (no per-call 8MB malloc)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: Docs — mark the V2 perf follow-up DONE

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the V2 known-follow-up.** In `AGENTS.md`, find the V2 (MossTTSLocal) follow-up that flags "depth loop is O(channels²) recompute + per-call ctx alloc (perf)". Change it to state this is NOW DONE: the local transformer uses a per-frame KV cache (`LocalTransformer::step`/`reset`, mirroring `rt_local`) so the depth loop is O(channels) instead of O(channels²), and `LocalTransformer::step` / `LocalAdapters::to_local` / `head_logits` reuse persistent scratch contexts (`make_ctx_buf`) instead of allocating per call. Note the change is numerically identical (validated by `test_depth_loop` exact codes + `test_local_transformer` per-row step parity). Remove any stale "PERF FOLLOW-UP" language for this item.

- [ ] **Step 2: Commit**
```bash
git add -A && git commit -m "docs(v2): mark the local depth-loop KV-cache + ctx-reuse perf follow-up done

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 5 tasks)

Dispatch a final whole-implementation reviewer for the entire branch (bit-faithfulness: same exact codes/logits as before across `test_depth_loop` + the per-row `test_local_transformer` + the env-gated parity; the `step` KV-cache wiring vs `rt_local`; the `make_ctx_buf` reuse safety — no leak / fresh arena per call / no cross-call aliasing; no regression to V1/V3/V4 and the Foundation), then use `superpowers:finishing-a-development-branch` to merge to `main` and push.

---

## Self-review notes (addressed)

- **Spec coverage:** the per-frame KV cache (`forward_last`→`step`/`reset`) = Tasks 1-3; the scratch-ctx reuse (`make_ctx_buf` + the two adapter/head sites + step's own scratch) = Tasks 1 & 4; the orchestrator depth-loop migration = Task 2; the test rewrites (`test_local_transformer` per-row step parity, `test_depth_loop` keystone, `test_local_parity`) = Tasks 1-3; docs = Task 5. The bit-faithful acceptance criterion is the keystone's exact-code assertion (Task 2 Step 2) + the per-row parity (Task 1).
- **Type/signature consistency:** `reset()` / `step(const std::vector<float>&, int, std::vector<float>*)` match `rt_local.hpp` and are used identically in Task 2 (orchestrator) + Task 3 (parity test) + Task 1 (unit test). `make_ctx_buf(void*, size_t, bool)` defined in Task 1, used in Task 1 (step) + Task 4 (adapters). `k_state_`/`v_state_`/`past_len_`/`step_scratch_` consistent across Task 1. The `cur` carrier replaces `local_seq` consistently in Task 2 + Task 3.
- **No-RoPE specifics:** the `posn` leaf is created but NEVER uploaded (mirrors the existing `forward_last`; gallocr skips it under `use_rope=false`); the mask is `[kv,1]` all-zeros. These are the two spots that must be exact for bit-parity — pinned by Task 2 Step 2.
- **Parity rationale:** `step(pos)` == row `pos` of a full causal forward == `forward_last(seq[0:pos+1], pos+1)` by causality, so the per-row fixture and the keystone both validate `step` exactly; the scratch-ctx reuse builds the identical graph into a reused buffer (numerically a no-op). gallocr is already a persistent global, so the reuse win is eliminating the per-call metadata-ctx malloc (64 MiB for step, 8 MiB×2 for adapters/heads).
