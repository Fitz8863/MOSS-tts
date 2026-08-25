# Ring-KV Codec Streaming Perf Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V4 Nano codec's `decode_stream_step` windowed-lookback (re-runs the whole decoder tower over a retained window each step, O(window)) with true **per-stage ring-KV** (a persistent K/V cache in the decoder transformer stages), computing only the new frame(s) incrementally — **numerically identical** to `decode_full` (the streaming==full contract: maxerr ≤ 1e-3, currently ~1.5e-8).

**Architecture:** Mirror the LLM KV-cache pattern (`qwen3_layer_forward` / `LocalTransformer::step`) for the codec decoder transformer, plus **ring eviction**. (1) Add optional per-layer `k_past`/`v_past` to the codec attention (`transformer.cpp`); `null` = today's full-sequence behavior, so `decode_full` and the 24 kHz-mono Foundation path (which share this file) stay byte-identical. Cache K/V **post-RoPE in the (hd, H, seq) layout** (seq outermost = ne2, so eviction is a clean front-drop) at **absolute positions starting from 0** (which match `decode_full`'s local positions frame-for-frame → bit-exact). (2) A new per-stage incremental driver replaces `decode_stream_step`: one graph per step over the new block, threading each transformer stage's host-side ring K/V (evicted to `cfg.context`), with offset RoPE positions and a `[cached+new]×[new]` sliding-window mask.

**Tech Stack:** ggml (pinned `third_party/ggml`), C++17, GGUF, numpy parity fixtures, CMake/ctest.

**Spec:** `docs/superpowers/specs/2026-06-04-codec-ring-kv-streaming-design.md`.

**Reference:** upstream `_forward_streaming_sdpa` + `MHAState` ring (length `context`) + `_build_streaming_sdpa_bias` (`(delta>=0)&(delta<context)`). The repo template: `src/qwen3.cpp` `qwen3_layer_forward` (k_past/v_past concat dim=2 + `ggml_cont` outputs) and `src/local_transformer.cpp` `step()` (host per-layer K/V driver + `make_ctx_buf` scratch). The codec attention is *simpler* (plain MHA, no GQA, no q/k-norm) but adds NORMAL-RoPE-at-offset + ring eviction.

**Key math (the exactness crux):** for a query at absolute position `P = past + qi` and a key at buffer index `m` (the buffer is `[cached(n_cached) | new(Tt)]`, so key `m`'s absolute position is `past − n_cached + m` uniformly), `delta = P − K_m = qi + n_cached − m`. Allowed iff `0 ≤ delta < context` (or `delta ≥ 0` when `context ≤ 0`). The host keeps the last `context` keys per stage/layer; after a step it evicts the front. Because absolute positions equal `decode_full`'s local positions frame-for-frame and the mask reproduces the same window, the streamed attention is bit-exact with the full decode.

**Cross-cutting conventions:**
- Commit trailer EXACTLY: `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`. **NO** `Co-Authored-By`, **NO** `Signed-off-by`.
- Build/test: `. .venv/bin/activate && cmake --build build -j && ctest --test-dir build --output-on-failure`.
- The whole existing suite stays green at every commit. `decode_full`/`encode`/the 24 kHz-mono Foundation path must stay BYTE-IDENTICAL (the `k_past=null` path); guarded by `test_audio_tokenizer_e2e` + `test_quantizer` + `test_partial_decode` + the non-streaming `test_nano_codec` decode assertion.
- Every `->data`/`ggml_backend_tensor_get` read is sized by `ggml_nelements`/an explicit resize first.

---

## File Structure

- `scripts/gen_test_fixtures.py` — extend `w_nano_codec` so **every** decoder transformer stage has `context < T` at its rate (the discriminating bed).
- `src/transformer.{hpp,cpp}` — `run_transformer`/`attention` gain optional per-layer `k_past`/`v_past` in + per-layer `k_full`/`v_full` out. `null` = unchanged.
- `src/audio_tokenizer.{hpp,cpp}` — new streaming pos/mask host helpers; replace `NanoCodecStream` + `decode_stream_step` with the per-stage ring-KV state + incremental driver; remove `decoder_receptive_frames`.
- `tests/test_codec_stream_mask.cpp` (new) — unit-test the streaming window-mask builder.
- `tests/test_nano_codec_stream.cpp` — driver is caller-side + UNCHANGED (the public API is unchanged); it validates the new internals end-to-end against the enhanced fixture.
- `AGENTS.md` — mark V4 follow-up #5 done; update the streaming gotcha.

---

## Task 1: Enhance the fixture so every decoder transformer stage clips

**Files:** Modify `scripts/gen_test_fixtures.py` (`w_nano_codec`).

The current `w_nano_codec` only asserts `dec_ctx[0] < NF` (the first decoder transformer stage's window is load-bearing). Extend it so **every** decoder transformer stage has `context < (its stage-rate T)`, so a per-stage ring-KV offset/eviction bug surfaces. This is a fixture-only change validated by the EXISTING (windowed-lookback) impl staying green — it establishes the discriminating bed BEFORE the driver changes.

- [ ] **Step 1: Tune the dims so all decoder transformer stages clip.** Read the current `w_nano_codec` (the `D,H,Lyr,FF / NQ,CS,CD,RVQ,LAT / CH,DS / SR,CTX_SECONDS,NF` knobs + the `dec_stages` table + the `dec_ctx` loop at the lines computing `dec_ctx[i] = int(round(fr * CTX_SECONDS))`). The decoder has 2 transformer stages (indices 0 and 2 in `dec_stages`) at rates `fr0 = SR/DS` and `fr2 = fr0 * 2` (after the p=2 patch_up). Their stage-rate sequence lengths are `T0 = NF` and `T2 = NF*2`. Choose `SR`, `CTX_SECONDS`, `NF` so BOTH `dec_ctx[0] < NF` AND `dec_ctx[2] < NF*2` hold with a margin (e.g. keep `SR=16, DS=8` → `fr0=2` Hz; pick `CTX_SECONDS` and `NF` so `dec_ctx[0]=round(2*CTX_SECONDS) < NF` and `dec_ctx[2]=round(4*CTX_SECONDS) < NF*2`). Concretely: with `CTX_SECONDS=1.0`, `dec_ctx[0]=2`, `dec_ctx[2]=4`; set `NF` large enough that `2 < NF` and `4 < NF*2` AND that the window actually clips for a meaningful number of frames — use `NF=6` (already) gives `2<6` ✓ and `4<12` ✓, but make the clipping deeper: bump `NF` to e.g. `10` so each stage's window excludes several frames (more discriminating). Recompute the numpy decode ref accordingly (it already applies `context=dec_ctx[i]` per stage in `_codec_transformer_np`).

- [ ] **Step 2: Strengthen the assertion.** Replace the single `assert dec_ctx[0] < NF` with an explicit per-transformer-stage check, e.g.:
```python
# Every decoder transformer stage must clip (context < its stage-rate T) so a
# per-stage ring-KV offset/eviction bug is caught by the streaming parity test.
stage_T = NF
for i, s in enumerate(dec_stages):
    if s[0] == 1:               # transformer
        assert dec_ctx[i] < stage_T, ("decoder stage window not load-bearing", i, dec_ctx[i], stage_T)
    else:                        # patch_up multiplies the downstream frame count
        stage_T *= s[1]
```
(Walk `stage_T` through the decoder exactly as the rate `fr` is walked, so each transformer stage's `T` is `NF * Π(patch_up ratios before it)`.)

- [ ] **Step 3: Regenerate + confirm BOTH nano_codec tests still pass (windowed-lookback unchanged).**
```bash
. .venv/bin/activate && python scripts/gen_test_fixtures.py nano_codec && cmake --build build -j && ctest --test-dir build -R "test_nano_codec|test_nano_codec_stream" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: `test_nano_codec` (decode parity) + `test_nano_codec_stream` (streaming parity, current windowed-lookback) BOTH PASS with the new fixture (proving the enhanced fixture + numpy ref + the C++ decode agree at every-stage-clipping). Full suite green. Fixture byte-reproducible (regen → identical md5).

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "test: nano_codec fixture clips every decoder stage window (ring-KV discriminator)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: Additive k_past/v_past + k_full/v_full in the codec attention (decode_full byte-identical)

**Files:** Modify `src/transformer.hpp`, `src/transformer.cpp`.

Add optional per-layer K/V cache threading to `run_transformer`/`attention`, mirroring `qwen3_layer_forward` (concat on the seq axis, `ggml_cont` outputs). Cache layout is **(hd, H, seq)** — seq outermost (ne2), concat `dim=2` — so the host can evict the front along ne2 and feed it straight back. `null` k_past = today's behavior exactly.

- [ ] **Step 1: Extend the `run_transformer` declaration in `src/transformer.hpp`.** Append optional cache params (defaults keep all existing callers + the full-decode path unchanged):
```cpp
// k_past/v_past: optional per-layer cached K/V (size n_layers; an entry may be
//   null for a prefill layer). nullptr vector => full-sequence prefill (today's
//   behavior, byte-identical). When provided, attention attends over
//   concat(past, new) on the sequence axis and `mask` is [kv, T] (kv=cached+new).
// k_out/v_out: optional per-layer outputs receiving the contiguous full K/V
//   (layout (hd, n_heads, kv)) for the caller to cache + evict. Resized to n_layers.
struct ggml_tensor* run_transformer(struct ggml_context* ctx, const TransformerWeights& w,
                                    const TransformerConfig& cfg, struct ggml_tensor* x,
                                    struct ggml_tensor* pos, struct ggml_tensor* mask,
                                    const std::vector<struct ggml_tensor*>* k_past = nullptr,
                                    const std::vector<struct ggml_tensor*>* v_past = nullptr,
                                    std::vector<struct ggml_tensor*>* k_out = nullptr,
                                    std::vector<struct ggml_tensor*>* v_out = nullptr);
```

- [ ] **Step 2: Update `attention` in `src/transformer.cpp`** to take a single `k_past`/`v_past` + return `k_full`/`v_full`. Change its signature to a file-static helper that also out-params the cache, e.g.:
```cpp
static struct ggml_tensor* attention(struct ggml_context* ctx, const LayerWeights& lw,
                                     const TransformerConfig& cfg, struct ggml_tensor* x,
                                     struct ggml_tensor* pos, struct ggml_tensor* mask,
                                     struct ggml_tensor* k_past, struct ggml_tensor* v_past,
                                     struct ggml_tensor** k_full_out, struct ggml_tensor** v_full_out);
```
In the body, KEEP everything up to and including the rope on q/k (the fused `qkv_w` matmul, the q/k/v views, `to_heads` → `(hd, H, Tt)`, `ggml_rope_ext` on q AND k at `pos`). Then, BEFORE the existing `permute to (hd, Tt, H)`, concat the cache on the seq axis (ne2) in the `(hd, H, seq)` layout and emit the contiguous full K/V:
```cpp
    // ... q = to_heads(q); k = to_heads(k); v = to_heads(v);   // (hd, H, Tt)
    // ... q = ggml_rope_ext(q, pos, ...); k = ggml_rope_ext(k, pos, ...);
    // KV cache: concat past on the sequence axis (ne2). v is NOT roped.
    struct ggml_tensor* k_cat = k;     // (hd, H, Tt)
    struct ggml_tensor* v_cat = v;     // (hd, H, Tt)
    if (k_past) k_cat = ggml_concat(ctx, k_past, k, /*dim=*/2);   // (hd, H, past+Tt)
    if (v_past) v_cat = ggml_concat(ctx, v_past, v, /*dim=*/2);
    if (k_full_out) *k_full_out = ggml_cont(ctx, k_cat);          // cache layout (hd, H, kv)
    if (v_full_out) *v_full_out = ggml_cont(ctx, v_cat);
    // attention over the concatenated K/V:
    q     = ggml_cont(ctx, ggml_permute(ctx, q,     0, 2, 1, 3)); // (hd, Tt,  H)
    struct ggml_tensor* kk = ggml_cont(ctx, ggml_permute(ctx, k_cat, 0, 2, 1, 3)); // (hd, kv, H)
    struct ggml_tensor* vv = ggml_cont(ctx, ggml_permute(ctx, v_cat, 0, 2, 1, 3)); // (hd, kv, H)
    struct ggml_tensor* scores = ggml_mul_mat(ctx, kk, q);        // (kv, Tt_q, H)
    float scale = 1.0f / std::sqrt((float)hd);
    if (mask) { scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f); }
    else { scores = ggml_scale(ctx, scores, scale); scores = ggml_diag_mask_inf(ctx, scores, 0); scores = ggml_soft_max(ctx, scores); }
    struct ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vv));  // (kv, hd, H)
    struct ggml_tensor* o  = ggml_mul_mat(ctx, vt, scores);           // (hd, Tt_q, H)
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));             // (hd, H, Tt)
    o = ggml_reshape_2d(ctx, o, D, Tt);                              // (D, Tt)
    return ggml_mul_mat(ctx, lw.out_w, o);
```
NOTE: with `k_past=null` and `k_full_out=null`, `k_cat==k`, no concat, no output → the graph is IDENTICAL to today (`kk`/`vv` are the same permuted tensors). `Tt` here is `x->ne[1]` (the new tokens); the old `mask` was square `(Tt,Tt)` and stays so for the full-decode path; the streaming path passes a `(kv, Tt)` mask. `diag_mask_inf` is only used on the no-mask path (full causal); streaming always passes a mask.

- [ ] **Step 3: Thread the cache per layer in `run_transformer`.** Resize `k_out`/`v_out` to `w.layers.size()` if non-null; per layer pass `k_past->at(l)`/`v_past->at(l)` (or null) and capture into `(*k_out)[l]`/`(*v_out)[l]`:
```cpp
    if (k_out) k_out->assign(w.layers.size(), nullptr);
    if (v_out) v_out->assign(w.layers.size(), nullptr);
    int li = 0;
    for (const auto& lw : w.layers) {
        struct ggml_tensor* h = layer_norm(ctx, x, lw.norm1_w, lw.norm1_b, cfg.eps);
        struct ggml_tensor* kp = (k_past && (size_t)li < k_past->size()) ? (*k_past)[li] : nullptr;
        struct ggml_tensor* vp = (v_past && (size_t)li < v_past->size()) ? (*v_past)[li] : nullptr;
        struct ggml_tensor* kf = nullptr; struct ggml_tensor* vf = nullptr;
        struct ggml_tensor* a = attention(ctx, lw, cfg, h, pos, mask, kp, vp,
                                          (k_out ? &kf : nullptr), (v_out ? &vf : nullptr));
        if (k_out) (*k_out)[li] = kf;
        if (v_out) (*v_out)[li] = vf;
        x = ggml_add(ctx, x, ggml_mul(ctx, a, lw.ls1));
        h = layer_norm(ctx, x, lw.norm2_w, lw.norm2_b, cfg.eps);
        h = ggml_mul_mat(ctx, lw.lin1_w, h);
        h = ggml_gelu_erf(ctx, h);
        h = ggml_mul_mat(ctx, lw.lin2_w, h);
        x = ggml_add(ctx, x, ggml_mul(ctx, h, lw.ls2));
        ++li;
    }
    if (w.out_proj) x = ggml_mul_mat(ctx, w.out_proj, x);
    return x;
```
The existing `run_stage` call site passes only `(ctx, w, cfg, x, pos, mask)` → the new params default to null → unchanged.

- [ ] **Step 4: Build + confirm decode_full / Foundation BYTE-IDENTICAL.**
```bash
cmake --build build -j && ctest --test-dir build -R "test_audio_tokenizer_e2e|test_quantizer|test_partial_decode|test_nano_codec|test_nano_codec_stream" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: ALL pass unchanged (the null-cache path is byte-identical; `test_nano_codec_stream` still uses the windowed-lookback which calls `decode_block` → `run_stage` → `run_transformer` with null cache). Full suite green.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(codec): thread optional k_past/v_past through the codec transformer attention (null = unchanged)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: Streaming position + window-mask host helpers + a mask unit test

**Files:** Modify `src/audio_tokenizer.cpp` (add `make_pos_input_offset` + `build_stream_window_mask` + a `make_stream_mask_input`). Create `tests/test_codec_stream_mask.cpp`. Modify `tests/CMakeLists.txt`.

The streaming step needs offset RoPE positions (`past..past+Tt-1`, not `0..Tt-1`) and a non-square `[kv, Tt]` sliding-window mask. Pure host logic — unit-test the mask in isolation (it's half the #1 risk).

- [ ] **Step 1: Write `tests/test_codec_stream_mask.cpp` (FAILING).** Assert the window-mask builder produces the expected `delta = qi + n_cached − m` pattern. E.g. for `(n_cached=2, Tt=2, context=2)`: kv=4, query 0 (qi=0) allows keys with `0 ≤ 0+2−m < 2` → m∈{1,2}; query 1 (qi=1) allows `0 ≤ 1+2−m < 2` → m∈{2,3}. Build a small table of cases (incl. `context<=0` → all `delta>=0`, and `context>=kv` → causal), call `build_stream_window_mask(n_cached, Tt, context, &dst)`, check each `dst[qi*kv + m]` is `0.0f` (allowed) or `-inf` (blocked). Register `moss_add_test(test_codec_stream_mask)`.

- [ ] **Step 2: Run → FAIL** (`build_stream_window_mask` undefined).

- [ ] **Step 3: Implement the helpers in `src/audio_tokenizer.cpp`** (near `build_window_mask`/`make_pos_input`). `build_stream_window_mask` must be reachable from the test — declare it in a small internal header OR expose it via the existing `build_window_mask` location (mirror how `build_window_mask` is declared/used; if `build_window_mask` is file-local, give `build_stream_window_mask` external linkage in `audio_tokenizer.hpp` or a shared `window_mask.hpp` so the test can call it — match the pattern `test_window_mask.cpp` uses to reach `build_window_mask`).
```cpp
// Streaming sliding-window mask: queries are the Tt new frames at absolute
// positions [past..past+Tt-1]; keys are the [cached(n_cached) | new(Tt)] buffer,
// key m at absolute position (past - n_cached + m). delta = qi + n_cached - m.
// Layout: ne0 = kv (key, fastest), ne1 = Tt (query). Caller uploads to a [kv,Tt] f32 leaf.
void build_stream_window_mask(int n_cached, int Tt, int context, std::vector<float>* dst) {
    const int kv = n_cached + Tt;
    dst->assign((size_t)kv * Tt, 0.0f);
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int qi = 0; qi < Tt; ++qi) {
        for (int m = 0; m < kv; ++m) {
            int delta = qi + n_cached - m;
            bool allowed = (context <= 0) ? (delta >= 0) : (delta >= 0 && delta < context);
            (*dst)[(size_t)qi * kv + m] = allowed ? 0.0f : ninf;
        }
    }
}
```
Plus a `make_pos_input_offset(ctx, pend, Tt, past)` (copy `make_pos_input` but fill `d[i] = (int32_t)(past + i)`) and a `make_stream_mask_input(ctx, pend, n_cached, Tt, context)` that builds a `[kv, Tt]` f32 leaf via `build_stream_window_mask` (mirror `make_mask_input`).

- [ ] **Step 4: Run → PASS.** Build + full suite green (additive helpers; nothing else changed yet).
```bash
cmake --build build -j && ctest --test-dir build -R test_codec_stream_mask --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(codec): streaming offset-position + sliding-window mask helpers + unit test

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: Per-stage ring-KV stream state + incremental driver (replace windowed-lookback)

**Files:** Modify `src/audio_tokenizer.hpp` (new stream state), `src/audio_tokenizer.cpp` (rewrite `decode_stream_begin`/`decode_stream_step`).

Replace the code-frame ring + `decode_block`-over-window with a per-stage ring-KV driver: ONE graph per step over the new block, threading each decoder transformer stage's host-side K/V (evicted to `cfg.context`). The public API (`decode_stream_begin`/`decode_stream_step` signatures, the opaque handle) is unchanged, so `test_nano_codec_stream` (caller-side) is the end-to-end guard.

- [ ] **Step 1: Replace `NanoCodecStream` in `src/audio_tokenizer.hpp`** with per-stage K/V state. The codec processes the new latent frame through all decoder stages; each transformer stage `s` caches per-layer K/V in `(hd, H, seq)` layout (flat float vectors) + a `past` counter (in that stage's frames):
```cpp
struct NanoCodecStream {
    int nq = 0;                       // codes per frame (set on first step)
    int k  = -1;                      // first-k depth (== nq here)
    // Per decoder TRANSFORMER stage: per-layer cached K/V (flat (hd*H*kept), seq
    // outermost) + the absolute frame count seen at that stage's rate.
    struct StageKV {
        int past = 0;                 // absolute frames cached/seen at this stage's rate
        std::vector<std::vector<float>> k_state, v_state;  // per layer
    };
    std::vector<StageKV> stages;      // one entry per decoder transformer stage (in decoder order)
    std::vector<uint8_t> scratch;     // reused metadata arena for the per-step graph (make_ctx_buf)
};
```

- [ ] **Step 2: Rewrite `decode_stream_begin`.** Allocate the stream; size `stages` to the number of decoder transformer stages (count `decoder_` entries with `kind==Transformer`), each with `k_state/v_state` sized to that stage's `cfg.n_layers` empty vectors and `past=0`; size `scratch` once (e.g. `64 * 1024 * 1024`). Lazy `nq` resolution stays (on the first step).

- [ ] **Step 3: Rewrite `decode_stream_step`** as the incremental driver. On the first step resolve `nq`/`k` (same range check as today: `nq` in `[1, n_quantizers_]`). Then build ONE graph for the new frame (mirror `decode_block` but with the streaming pos/mask + per-stage K/V threading, using `make_ctx_buf(st.scratch...)`):
```
auto cctx = make_ctx_buf(st.scratch.data(), st.scratch.size(), /*no_alloc=*/true);
ctx = cctx.get();
code_t = [nq, 1] i32 input leaf <- codes_one_frame      // one frame
x = dequantize(ctx, quant_, code_t, st.k)               // latent (LAT, 1)
running = x
ti = 0                                                   // transformer-stage index into st.stages
per-layer k_past[l]/v_past[l] input leaves + k_out[l]/v_out[l] outputs, PER transformer stage
for each decoder stage s (in order):
    if s.kind == Patch:
        running = patch_up(ctx, running, s.patch_size)   // stateless, expands ne1 by p
    else (Transformer s):
        StageKV& kv = st.stages[ti];
        Tt = running->ne[1]                              // new frames at this stage rate (after transpose_in)
        // n_cached for THIS stage = min(kv.past, s.cfg.context) (or kv.past if context<=0)
        n_cached = (s.cfg.context > 0) ? std::min(kv.past, s.cfg.context) : kv.past;
        build per-layer k_past leaves (hd, H, n_cached) gated on n_cached>0 (uploaded from kv.k_state[l]);
        pos  = make_pos_input_offset(ctx, pend, Tt, kv.past);          // [past .. past+Tt-1]
        mask = make_stream_mask_input(ctx, pend, n_cached, Tt, s.cfg.context);  // [n_cached+Tt, Tt]
        // run_transformer with transpose_in like run_stage (first transformer stage skips transpose_in)
        h = transpose_in ? cont(transpose(running)) : running;
        out = run_transformer(ctx, s.weights, s.cfg, h, pos, mask, &k_past, &v_past, &k_out, &v_out);
        running = cont(transpose(out));                  // transpose_out (matches run_stage)
        mark k_out[l]/v_out[l] as graph outputs (ggml_set_output + build_forward_expand);
        ++ti
ggml_set_output(running)                                 // waveform (downsample*channels for this frame's block)
compute once
emit running (read back) as the pcm_chunk
read back each stage/layer k_out/v_out into kv.k_state/v_state; EVICT to last s.cfg.context frames; kv.past += Tt
```
Eviction (host): the cache is `(hd, H, kept)` seq-outermost, so dropping the oldest `E` frames = erase the FIRST `E * hd * H` floats. After reading back `k_out` of shape `(hd, H, n_cached+Tt)`, if `s.cfg.context > 0 && (n_cached+Tt) > s.cfg.context`, keep the last `s.cfg.context` frames: `E = (n_cached+Tt) - context; new = vec(k_full.begin() + E*hd*H, k_full.end())`. Store as `kv.k_state[l]`. (For `context<=0`, keep all — grow-only.) `kv.past += Tt`.

Mirror `local_transformer.cpp::step` for the per-layer leaf creation + the guarded read-back (`kvn = hd * H * (n_cached+Tt)`), and `decode_block` for the dequantize + the per-stage transpose_in/out + the single `compute_graph_with_inputs`. The `per_frame = downsample_ * factor` chunk-size check stays (emit exactly one frame's interleaved-stereo samples). NOTE: the transformer stage's `hd = cfg.d_model / cfg.n_heads`, `H = cfg.n_heads`; the cache flat size per layer = `hd * H * kept`.

- [ ] **Step 4: Build + run the streaming parity test (the bit-faithful guard) + the discrimination check.**
```bash
cmake --build build -j && ctest --test-dir build -R "test_nano_codec_stream|test_nano_codec|test_audio_tokenizer_e2e|test_quantizer" --output-on-failure && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"
```
Expected: `test_nano_codec_stream` PASS (streaming == decode_full within 1e-3, on the every-stage-clips fixture — the ring-KV must reproduce the full decode at every stage's window); decode_full + Foundation green; full suite green. **Discrimination check:** temporarily break the eviction (e.g. keep `context+1` instead of `context`, or drop the pos offset) → confirm `test_nano_codec_stream` FAILS (proving the every-stage-clips fixture catches a per-stage bug), then restore → PASS. Report both.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "perf(codec): per-stage ring-KV streaming decode (replaces O(window) lookback), exact vs decode_full

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: Remove the dead lookback path + docs

**Files:** Modify `src/audio_tokenizer.{hpp,cpp}` (remove `decoder_receptive_frames`), `AGENTS.md`.

- [ ] **Step 1: Remove `decoder_receptive_frames`** (declaration in `audio_tokenizer.hpp`, definition in `audio_tokenizer.cpp`) — it was only called by the old `decode_stream_step`. Confirm no references: `grep -rn decoder_receptive_frames src/ tests/`. Also remove any now-unused `NanoCodecStream` fields / includes left over from Task 4 (e.g. the old `lookback`/`ring`/`used` if not already gone). `decode_block` stays (still used by `decode`/`reconstruct`).

- [ ] **Step 2: Build + full suite green** (clean removal).
```bash
cmake --build build -j && ctest --test-dir build --output-on-failure 2>&1 | tail -6
```

- [ ] **Step 3: Update `AGENTS.md`.** Find the V4 KNOWN-FOLLOW-UP #5 ("Streaming is O(window) per step (windowed re-decode); a ring-KV streaming decode is a perf follow-up") and mark it DONE: the codec now uses true per-stage ring-KV (persistent K/V cache in the decoder transformer stages, evicted to each stage's sliding-window `context`), so each step is incremental (O(context) attention + O(1) projections) instead of O(window) full re-decode; numerically identical (validated by `test_nano_codec_stream` == decode_full on the every-stage-clips fixture). Also sweep any other AGENTS.md line that describes the codec streaming as "windowed-lookback / O(window) / re-decode" as the current behavior (e.g. the V4 gotcha about streaming-being-RoPE-only-exact — keep the RoPE-only note, but update "windowed-lookback" → "ring-KV"). Match the surrounding doc style for a DONE item.

- [ ] **Step 4: Commit**
```bash
git add -A && git commit -m "docs(codec): mark the ring-KV streaming perf follow-up done; remove decoder_receptive_frames

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 5 tasks)

Dispatch a final whole-implementation reviewer for the entire branch (bit-faithfulness: `test_nano_codec_stream` == decode_full on the every-stage-clips fixture, the discrimination check; the attention K/V threading vs qwen3; the RoPE-absolute-position + eviction + window-mask bookkeeping; `decode_full`/Foundation byte-identical via the null path; no regression to V1/V2/V3/V4-LLM/Foundation), then use `superpowers:finishing-a-development-branch` to merge to `main` and push.

---

## Self-review notes (addressed)

- **Spec coverage:** the additive K/V on the codec attention (T2); the streaming pos/mask helpers (T3); the per-stage ring-KV state + incremental driver replacing the windowed-lookback (T4); the every-stage-clips fixture as the discriminator (T1); the dead-path removal + docs (T5). The streaming==full contract is the acceptance criterion (T4 Step 4 + the enhanced fixture).
- **Bit-faithful rationale:** absolute positions equal `decode_full`'s local positions frame-for-frame, the `delta = qi + n_cached − m` mask reproduces the same sliding window, and the cache is post-RoPE — so the streamed attention is the same RoPE-relative computation factored incrementally. The `k_past=null` path is byte-identical (decode_full/Foundation unchanged).
- **Type/signature consistency:** `run_transformer(... k_past, v_past, k_out, v_out)` (T2) is consumed by the driver (T4); `build_stream_window_mask(n_cached, Tt, context, &dst)` + `make_pos_input_offset(ctx, pend, Tt, past)` (T3) are used by the driver (T4); `NanoCodecStream`/`StageKV` (T4) consistent. Cache layout `(hd, H, seq)` seq-outermost (concat dim=2, front-eviction) is consistent across the attention (T2) and the driver (T4).
- **The #1 risk (RoPE-offset + eviction + mask boundary)** is split for de-risking: the mask is unit-tested (T3), decode_full byte-identity guards the null path (T2), and the every-stage-clips fixture + discrimination check pin the full driver (T4). Each transformer stage caches at its own rate; `patch_up`'s integer ratios keep the per-stage frame counts exact.
