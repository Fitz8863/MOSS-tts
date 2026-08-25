# Device-resident Backbone KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the O(T²) host↔device KV-cache roundtrip in the global backbones (`DelayBackbone`/Qwen3, `NanoBackbone`/GPT-2) by keeping the per-layer K/V resident on a persistent backend buffer, appending in place, and reading by zero-copy view.

**Architecture:** Add an opt-in device-KV-cache path to the shared `qwen3_layer_forward`/`gpt2_layer_forward` (null cache → existing additive path, byte-identical). Each backbone allocates a persistent `[head_dim, n_kv, max_seq, 1]` cache buffer per layer at load; `run` stores the new columns via in-graph `ggml_cpy` to a cache view and reads the `[0:past+T]` prefix by view. No per-step KV upload/readback.

**Tech Stack:** C++17, ggml/ggml-backend, GGUF. Validation is CPU byte-identity via the existing decode-vs-prefill consistency tests (`test_delay_kv`, `test_nano_backbone`).

**Reference (read first):** spec `docs/superpowers/specs/2026-06-05-backbone-device-kv-design.md`. The additive path being preserved is the one used by the codec (`src/transformer.cpp` — not touched) and the bounded depth caches (`src/local_transformer.cpp`, `src/rt_local.cpp`, `src/nano_local.cpp` — call the helper with no cache args → unchanged).

**Commit trailer (MANDATORY, every commit):** end the message with
`Assisted-by: Claude:claude-opus-4-8 [Claude Code]` — NO `Co-Authored-By`, NO `Signed-off-by`.

---

## File Structure

| File | Responsibility / change |
|------|--------|
| `src/qwen3.hpp` | `Qwen3LayerOut` gains `k_store`/`v_store`; `qwen3_layer_forward` gains 4 null-defaulted params (`gf`, `k_cache`, `v_cache`, `past_seq`). |
| `src/qwen3.cpp` | Refactor attention to share a `k_used`/`v_used` source; add the `k_cache != null` store+view path; additive path unchanged. |
| `src/gpt2.hpp` / `src/gpt2.cpp` | Same change for GPT-2 (MHA, interleaved RoPE, no q/k-norm). |
| `src/delay_backbone.hpp` / `.cpp` | Persistent cache buffer + members; `run` uses the cache path; drop host `k_state_`/`v_state_`; destructor frees the buffer. |
| `src/nano_backbone.hpp` / `.cpp` | Same as delay, for GPT-2. |
| `tests/test_delay_kv.cpp`, `tests/test_nano_backbone.cpp` | Used as the byte-identity gate (unchanged) + the revert-to-fail discrimination check. |
| `AGENTS.md` | Mark the backbone-KV host-roundtrip follow-up DONE with residuals. |

---

## Task 1: `qwen3_layer_forward` opt-in device-KV-cache path

**Files:**
- Modify: `src/qwen3.hpp` (struct + signature)
- Modify: `src/qwen3.cpp:56-145` (function body)

This task adds the cache path but no caller uses it yet (Task 2 wires it). Acceptance for this task: it compiles and the **full suite stays green** (the additive/null path is byte-identical — every existing caller passes no cache args).

- [ ] **Step 1: Update the header.** In `src/qwen3.hpp`, extend the out-struct and the signature. Replace lines 17-23:

```cpp
// On the additive path k_full/v_full are contiguous (gallocr-safe) graph
// outputs. On the device-cache path (k_cache != null) k_store/v_store are the
// in-graph ggml_cpy nodes that write the new columns into the persistent cache
// (already build_forward_expand'd by the helper); k_full/v_full stay null.
struct Qwen3LayerOut { struct ggml_tensor *y=nullptr,*k_full=nullptr,*v_full=nullptr,
                                           *k_store=nullptr,*v_store=nullptr; };
// x:[hidden,T]; pos:int32[T]; mask:[kv,T] additive (0/-inf) or null (= no bias).
// Additive path: k_past/v_past null on prefill (kv=T); returns k_full/v_full.
// Device-cache path (k_cache != null): store the T new columns into k_cache at
// sequence offset past_seq, read the [0:past_seq+T] prefix by view; the helper
// build_forward_expand's the store nodes into gf (ordering: store before read).
Qwen3LayerOut qwen3_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pos,
    struct ggml_tensor* mask, struct ggml_tensor* k_past, struct ggml_tensor* v_past,
    const Qwen3Layer& w, const Qwen3Hparams& hp,
    struct ggml_cgraph* gf = nullptr,
    struct ggml_tensor* k_cache = nullptr, struct ggml_tensor* v_cache = nullptr,
    int past_seq = 0);
```

- [ ] **Step 2: Verify it compiles before touching the body.** The new params are null-defaulted so existing callers are unaffected.

Run: `cmake --build build -j 2>&1 | tail -3`
Expected: builds clean (the `.cpp` signature still matches the 8-arg definition — C++ default args live only in the declaration; you must also update the definition's parameter list in the next step or the linker/compiler will mismatch). If it errors on the signature mismatch, that is expected — proceed to Step 3.

- [ ] **Step 3: Rewrite the function body** in `src/qwen3.cpp`. Replace the entire function (lines 56-145) with the version below. The pre-attention compute (norm/q/k/v/rope) is unchanged; the K/V *source* is selected per path; the attention + FFN are shared.

```cpp
Qwen3LayerOut qwen3_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x,
                                  struct ggml_tensor* pos, struct ggml_tensor* mask,
                                  struct ggml_tensor* k_past, struct ggml_tensor* v_past,
                                  const Qwen3Layer& w, const Qwen3Hparams& hp,
                                  struct ggml_cgraph* gf,
                                  struct ggml_tensor* k_cache, struct ggml_tensor* v_cache,
                                  int past_seq) {
    const int hd     = hp.head_dim;
    const int n_h    = hp.n_heads;
    const int n_kv_h = hp.n_kv_heads;
    const float eps  = hp.rms_eps;

    const int64_t n_tokens = x->ne[1];
    const int64_t n_batch  = x->ne[2] > 0 ? x->ne[2] : 1;

    // ---- attention pre-norm ----
    struct ggml_tensor* xn = rms_norm(ctx, x, w.attn_norm, eps);

    // ---- q, k, v (no bias in Qwen3) ----
    struct ggml_tensor* q = ggml_mul_mat(ctx, w.attn_q, xn);
    struct ggml_tensor* k = ggml_mul_mat(ctx, w.attn_k, xn);
    struct ggml_tensor* v = ggml_mul_mat(ctx, w.attn_v, xn);

    // Reshape to [hd, n_h, seq, batch] and [hd, n_kv_h, seq, batch].
    q = ggml_reshape_4d(ctx, q, hd, n_h,    n_tokens, n_batch);
    k = ggml_reshape_4d(ctx, k, hd, n_kv_h, n_tokens, n_batch);
    v = ggml_reshape_4d(ctx, v, hd, n_kv_h, n_tokens, n_batch);

    // ---- per-head RMSNorm on Q and K over head_dim (ne0), BEFORE RoPE ----
    q = rms_norm(ctx, q, w.q_norm, eps);
    k = rms_norm(ctx, k, w.k_norm, eps);

    // ---- RoPE NEOX on Q and K (skipped when use_rope=false) ----
    if (hp.use_rope) {
        q = ggml_rope_ext(ctx, q, pos, /*freq_factors=*/nullptr,
                          hd, kQwen3RopeMode, /*n_ctx_orig=*/0,
                          hp.rope_base, /*freq_scale=*/1.0f,
                          /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                          /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
        k = ggml_rope_ext(ctx, k, pos, /*freq_factors=*/nullptr,
                          hd, kQwen3RopeMode, 0,
                          hp.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    Qwen3LayerOut out;
    struct ggml_tensor* k_used;
    struct ggml_tensor* v_used;
    if (k_cache) {
        // ---- device-resident cache: store new columns, read prefix by view ----
        const int64_t kv = (int64_t)past_seq + n_tokens;
        struct ggml_tensor* k_dst = ggml_view_4d(ctx, k_cache, hd, n_kv_h, n_tokens, 1,
            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], (size_t)past_seq * k_cache->nb[2]);
        struct ggml_tensor* v_dst = ggml_view_4d(ctx, v_cache, hd, n_kv_h, n_tokens, 1,
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], (size_t)past_seq * v_cache->nb[2]);
        out.k_store = ggml_cpy(ctx, k, k_dst);
        out.v_store = ggml_cpy(ctx, v, v_dst);
        // Expand the stores NOW so they execute before this layer's attention
        // read of the same buffer (ordering by node insertion order).
        ggml_build_forward_expand(gf, out.k_store);
        ggml_build_forward_expand(gf, out.v_store);
        k_used = ggml_view_4d(ctx, k_cache, hd, n_kv_h, kv, 1,
            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], 0);
        v_used = ggml_view_4d(ctx, v_cache, hd, n_kv_h, kv, 1,
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], 0);
    } else {
        // ---- additive path: concat past K/V along the sequence dim (axis 2) ----
        struct ggml_tensor* k_full = k_past ? ggml_concat(ctx, k_past, k, /*dim=*/2) : k;
        struct ggml_tensor* v_full = v_past ? ggml_concat(ctx, v_past, v, /*dim=*/2) : v;
        k_used = k_full;
        v_used = v_full;
        out.k_full = ggml_cont(ctx, k_full);
        out.v_full = ggml_cont(ctx, v_full);
    }

    // ---- eager GQA attention (shared by both paths) ----
    struct ggml_tensor* q_p = ggml_permute(ctx, q,      0, 2, 1, 3);  // [hd, seq, n_h, b]
    struct ggml_tensor* k_p = ggml_permute(ctx, k_used, 0, 2, 1, 3);  // [hd, seq_kv, n_kv, b]
    struct ggml_tensor* v_p = ggml_permute(ctx, v_used, 0, 2, 1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    struct ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, /*max_bias=*/0.0f);

    struct ggml_tensor* v_t = maybe_cont(ctx, ggml_transpose(ctx, v_p));  // [seq_kv, hd, n_kv, b]
    struct ggml_tensor* o   = ggml_mul_mat(ctx, v_t, attn);

    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont_2d(ctx, o, n_h * hd, n_tokens * n_batch);
    if (n_batch > 1) o = ggml_reshape_3d(ctx, o, n_h * hd, n_tokens, n_batch);

    o = ggml_mul_mat(ctx, w.attn_o, o);
    struct ggml_tensor* h = ggml_add(ctx, x, o);

    // ---- FFN: SwiGLU = down( silu(gate(x)) * up(x) ) ----
    struct ggml_tensor* hn = rms_norm(ctx, h, w.ffn_norm, eps);
    struct ggml_tensor* g  = ggml_mul_mat(ctx, w.ffn_gate, hn);
    struct ggml_tensor* u  = ggml_mul_mat(ctx, w.ffn_up,   hn);
    struct ggml_tensor* f  = ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, ggml_silu(ctx, g), u));

    out.y = ggml_add(ctx, h, f);
    return out;
}
```

- [ ] **Step 4: Build + full suite (additive path byte-identical).**

Run: `cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"`
Expected: builds clean; full suite green (e.g. `100% tests passed ... out of 63`). The depth-loop keystones (`test_depth_loop`, `test_rt_depth_loop`) and `test_local_transformer` exercise the null path and must be unchanged. No cache path runs yet.

- [ ] **Step 5: Commit.**

```bash
git add src/qwen3.hpp src/qwen3.cpp
git commit -m "feat(kv): qwen3_layer_forward opt-in device-KV-cache path (additive path unchanged)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 2: `DelayBackbone` → device-resident cache

**Files:**
- Modify: `src/delay_backbone.hpp` (members + dtor)
- Modify: `src/delay_backbone.cpp` (load/reset/run/dtor)
- Gate test: `tests/test_delay_kv.cpp` (unchanged; used for parity + discrimination)

- [ ] **Step 1: Confirm the gate test passes today (baseline).**

Run: `ctest --test-dir build -R test_delay_kv --output-on-failure 2>&1 | tail -6`
Expected: PASS, printing `prefill maxerr=... decode maxerr=...` both `< 1e-3` (note the decode maxerr value — it must be the same after the change).

- [ ] **Step 2: Update the header.** Replace `src/delay_backbone.hpp` lines 1-27 with:

```cpp
#ifndef MOSS_DELAY_BACKBONE_HPP
#define MOSS_DELAY_BACKBONE_HPP
#include "qwen3.hpp"
#include "model_loader.hpp"
#include "ggml_extend.hpp"   // GgmlCtxPtr
#include "ggml-backend.h"    // ggml_backend_buffer_t
#include <vector>
namespace moss {
class DelayBackbone {
public:
    DelayBackbone() = default;
    ~DelayBackbone();
    DelayBackbone(const DelayBackbone&) = delete;             // owns a backend buffer
    DelayBackbone& operator=(const DelayBackbone&) = delete;
    bool load(const ModelLoader& m, int max_seq);   // reads qwen3.* metadata + N layers + qwen3.output_norm.weight
    int  hidden() const { return hp_.hidden; }
    const Qwen3Hparams& hparams() const { return hp_; }
    // Prefill S embedding rows (row-major S*hidden). Resets cache to pos 0. Returns the LAST row's hidden (post final RMSNorm).
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* last_hidden);
    // One step at the next position. embed: [hidden]. Returns hidden [hidden] (post final RMSNorm).
    bool decode_one(const std::vector<float>& embed, std::vector<float>* hidden);
    void reset();
    int  past_len() const { return past_len_; }
private:
    bool run(const std::vector<float>& embeds, int T, bool is_prefill, std::vector<float>* out_hidden);

    Qwen3Hparams hp_{}; std::vector<Qwen3Layer> layers_; struct ggml_tensor* output_norm_=nullptr;
    const ModelLoader* m_=nullptr; int max_seq_=0, past_len_=0;
    // Device-resident per-layer K/V cache: k_cache_[l]/v_cache_[l] are
    // [head_dim, n_kv_heads, max_seq, 1] tensors on kv_buffer_, written in place
    // each step and read by [0:past+T] view. kv_ctx_ holds their metadata.
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
};
}  // namespace moss
#endif
```

- [ ] **Step 3: Rewrite `load`, `reset`, `run`, add the destructor** in `src/delay_backbone.cpp`. Replace lines 24-178 (the `namespace moss {` body up to and including `decode_one`) with:

```cpp
namespace moss {

DelayBackbone::~DelayBackbone() {
    if (kv_buffer_) ggml_backend_buffer_free(kv_buffer_);
}

bool DelayBackbone::load(const ModelLoader& m, int max_seq) {
    m_       = &m;
    max_seq_ = max_seq;
    past_len_ = 0;

    hp_.hidden       = (int)m.get_u32("qwen3.hidden", 0);
    hp_.n_layers     = (int)m.get_u32("qwen3.n_layers", 0);
    hp_.n_heads      = (int)m.get_u32("qwen3.n_heads", 0);
    hp_.n_kv_heads   = (int)m.get_u32("qwen3.n_kv_heads", 0);
    hp_.head_dim     = (int)m.get_u32("qwen3.head_dim", 0);
    hp_.intermediate = (int)m.get_u32("qwen3.intermediate", 0);
    hp_.text_vocab   = (int)m.get_u32("qwen3.text_vocab", 0);
    hp_.rope_base    = m.get_f32("qwen3.rope_base", 1e6f);
    hp_.rms_eps      = m.get_f32("qwen3.rms_eps", 1e-6f);

    if (hp_.hidden <= 0 || hp_.n_layers <= 0 || hp_.n_heads <= 0 ||
        hp_.n_kv_heads <= 0 || hp_.head_dim <= 0 || max_seq_ <= 0) {
        return false;
    }

    layers_.assign(hp_.n_layers, Qwen3Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!qwen3_load_layer(m, i, &layers_[i])) return false;
    }

    output_norm_ = m.tensor("qwen3.output_norm.weight");
    if (!output_norm_) return false;

    // Allocate the persistent device-resident KV cache: 2 tensors per layer,
    // [head_dim, n_kv_heads, max_seq, 1], on their own backend buffer.
    const int L = hp_.n_layers;
    kv_ctx_ = make_ctx(ggml_tensor_overhead() * (size_t)(2 * L) + 1024, /*no_alloc=*/true);
    if (!kv_ctx_) return false;
    k_cache_.assign(L, nullptr);
    v_cache_.assign(L, nullptr);
    for (int l = 0; l < L; ++l) {
        k_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);
        v_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);
    }
    kv_buffer_ = ggml_backend_alloc_ctx_tensors(kv_ctx_.get(), moss::backend());
    if (!kv_buffer_) { MOSS_LOGE("DelayBackbone: KV cache alloc failed"); return false; }
    return true;
}

void DelayBackbone::reset() {
    past_len_ = 0;  // views are bounded by past_len_; stale cache bytes are never read
}

bool DelayBackbone::run(const std::vector<float>& embeds, int T, bool /*is_prefill*/,
                        std::vector<float>* out_hidden) {
    const int L   = hp_.n_layers;
    const int H   = hp_.hidden;
    const int past = past_len_;
    const int kv  = past + T;

    if (embeds.size() < (size_t)H * T) return false;
    if (past + T > max_seq_) { MOSS_LOGE("DelayBackbone: sequence exceeds max_seq"); return false; }

    auto cctx = moss::make_ctx(256 * 1024 * 1024, /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    // Mask only needed when T>1 (prefill): the [kv,T] causal mask. For the T=1
    // decode step every cached key is causally valid, so mask=null (no bias).
    struct ggml_tensor* mask = nullptr;
    if (T > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, T);  // ne0=key, ne1=query
        ggml_set_input(mask);
    }

    // Build the graph up front so the helper can expand its KV-store nodes into it.
    auto* gf = ggml_new_graph_custom(ctx, 4096, false);

    struct ggml_tensor* h = x;
    for (int l = 0; l < L; ++l) {
        auto lo = qwen3_layer_forward(ctx, h, pos, mask, /*k_past=*/nullptr, /*v_past=*/nullptr,
                                      layers_[l], hp_, gf, k_cache_[l], v_cache_[l], past);
        h = lo.y;
    }

    // Final RMSNorm on the LAST column only.
    struct ggml_tensor* last = h;
    if (T > 1) {
        last = ggml_view_2d(ctx, h, H, 1, h->nb[1], (size_t)(T - 1) * h->nb[1]);
        last = ggml_cont(ctx, last);
    }
    struct ggml_tensor* y = ggml_mul(ctx, ggml_rms_norm(ctx, last, hp_.rms_eps), output_norm_);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    const float ninf = -std::numeric_limits<float>::infinity();
    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, embeds.data(), 0, (size_t)H * T * sizeof(float));

        std::vector<int32_t> p(T);
        for (int i = 0; i < T; ++i) p[i] = past + i;
        ggml_backend_tensor_set(pos, p.data(), 0, (size_t)T * sizeof(int32_t));

        if (mask) {
            // mask[query i][key j]: attend iff key j <= query abs pos past+i.
            std::vector<float> mvec((size_t)kv * T);
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < kv; ++j)
                    mvec[(size_t)i * kv + j] = (j <= past + i) ? 0.0f : ninf;
            ggml_backend_tensor_set(mask, mvec.data(), 0, mvec.size() * sizeof(float));
        }
    };

    if (!moss::compute_graph_with_inputs(gf, set_inputs)) return false;

    past_len_ = kv;
    out_hidden->resize(H);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * sizeof(float));
    return true;
}

bool DelayBackbone::prefill(const std::vector<float>& embeds, int S,
                            std::vector<float>* last_hidden) {
    reset();
    return run(embeds, S, /*is_prefill=*/true, last_hidden);
}

bool DelayBackbone::decode_one(const std::vector<float>& embed, std::vector<float>* hidden) {
    return run(embed, 1, /*is_prefill=*/false, hidden);
}

}  // namespace moss
```

Note: `MOSS_LOGE` comes from `common.hpp` via the existing includes (`backend.hpp`/`ggml_extend.hpp` chain). If it is not visible, add `#include "common.hpp"` at the top of `src/delay_backbone.cpp`.

- [ ] **Step 4: Build + run the gate test — must be byte-identical.**

Run: `cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build -R test_delay_kv --output-on-failure 2>&1 | tail -6`
Expected: PASS, with `decode maxerr` equal to the Step 1 baseline value (the cache path computes the same bytes). If maxerr changed materially, the cache indexing is wrong — debug before continuing.

- [ ] **Step 5: Discrimination check (revert-to-fail).** Temporarily break the store offset to prove the gate test pins the cache indexing. In `src/qwen3.cpp`, change the store offset from `(size_t)past_seq * k_cache->nb[2]` to `(size_t)(past_seq + 1) * k_cache->nb[2]` (k only is enough), rebuild, and run `test_delay_kv`.

Run: `cmake --build build -j >/dev/null 2>&1 && ctest --test-dir build -R test_delay_kv --output-on-failure 2>&1 | tail -4`
Expected: **FAIL** (decode maxerr large) — the test catches a wrong write offset. Then **revert** the offset back to `(size_t)past_seq * k_cache->nb[2]`, rebuild, and confirm `test_delay_kv` PASSES again. If it did NOT fail (fixture not discriminating), also try a wrong read width (`kv - 1`) the same way; if neither fails, note it in the commit body and flag for the reviewer (the fixture may need more positions). Record the observed fail maxerr.

- [ ] **Step 6: Full suite + grep-clean.**

Run: `cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed" && grep -rn "\->data" src/delay_backbone.cpp | grep -vE "\.data\(\)|out->data|->data\(\)" | grep -v "//"`
Expected: full suite green; the grep returns EMPTY (no raw tensor `->data` reads — only `out_hidden->data()` host write, which the filter excludes).

- [ ] **Step 7: Commit.**

```bash
git add src/delay_backbone.hpp src/delay_backbone.cpp
git commit -m "perf(kv): DelayBackbone device-resident KV cache (no per-step host roundtrip)

Drops the O(T^2) host<->device KV transfer: per-layer K/V now live on a
persistent backend buffer, written in place via in-graph ggml_cpy and read by
[0:past+T] view. Byte-identical on CPU (test_delay_kv decode==prefill at the
same maxerr; revert-to-fail discrimination confirmed).

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 3: `gpt2_layer_forward` opt-in device-KV-cache path

**Files:**
- Modify: `src/gpt2.hpp` (struct + signature)
- Modify: `src/gpt2.cpp:50-125` (function body)

GPT-2 is MHA (no GQA → K/V have `n_head` heads), interleaved RoPE, no q/k-norm, LayerNorm-with-bias. The cache mechanics are identical to Task 1.

- [ ] **Step 1: Update the header.** In `src/gpt2.hpp`, replace lines 13 and 17-21:

```cpp
struct Gpt2LayerOut { struct ggml_tensor *y=nullptr,*k_full=nullptr,*v_full=nullptr,
                                          *k_store=nullptr,*v_store=nullptr; };
```
and
```cpp
// x:[hidden,T]; pos:int32[T]; mask:[kv,T] additive (0/-inf) or null (= no bias).
// Additive path: k_past/v_past null on prefill (kv=T); returns k_full/v_full.
// Device-cache path (k_cache != null): store the T new columns into k_cache at
// sequence offset past_seq, read [0:past_seq+T] by view; the helper
// build_forward_expand's the store nodes into gf. Builds ops only; no compute.
Gpt2LayerOut gpt2_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* pos,
    struct ggml_tensor* mask, struct ggml_tensor* k_past, struct ggml_tensor* v_past,
    const Gpt2Layer& w, const Gpt2Hparams& hp,
    struct ggml_cgraph* gf = nullptr,
    struct ggml_tensor* k_cache = nullptr, struct ggml_tensor* v_cache = nullptr,
    int past_seq = 0);
```
Add `struct ggml_cgraph;` to the forward declarations on line 5 (so it reads `struct ggml_context; struct ggml_tensor; struct ggml_cgraph;`).

- [ ] **Step 2: Rewrite the function body** in `src/gpt2.cpp`. Replace the entire function (lines 50-125) with:

```cpp
Gpt2LayerOut gpt2_layer_forward(struct ggml_context* ctx, struct ggml_tensor* x,
                                struct ggml_tensor* pos, struct ggml_tensor* mask,
                                struct ggml_tensor* k_past, struct ggml_tensor* v_past,
                                const Gpt2Layer& w, const Gpt2Hparams& hp,
                                struct ggml_cgraph* gf,
                                struct ggml_tensor* k_cache, struct ggml_tensor* v_cache,
                                int past_seq) {
    const int hd    = hp.head_dim;
    const int n_h   = hp.n_head;
    const float eps = hp.ln_eps;
    const int64_t H        = (int64_t)n_h * hd;
    const int64_t n_tokens = x->ne[1];

    // ---- attention pre-norm (LayerNorm with bias) ----
    struct ggml_tensor* xn = moss::layer_norm(ctx, x, w.ln1_w, w.ln1_b, eps);

    // ---- fused QKV: [3H, T] ----
    struct ggml_tensor* qkv = moss::linear(ctx, w.cattn_w, w.cattn_b, xn);

    const size_t es = ggml_element_size(qkv);
    struct ggml_tensor* q = ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, n_tokens, qkv->nb[1], 0 * H * es));
    struct ggml_tensor* k = ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, n_tokens, qkv->nb[1], 1 * H * es));
    struct ggml_tensor* v = ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, n_tokens, qkv->nb[1], 2 * H * es));

    // [H, T] -> [hd, n_h, T]
    q = ggml_reshape_3d(ctx, q, hd, n_h, n_tokens);
    k = ggml_reshape_3d(ctx, k, hd, n_h, n_tokens);
    v = ggml_reshape_3d(ctx, v, hd, n_h, n_tokens);

    // ---- interleaved RoPE on Q and K (mode 0 == NORMAL == GPT-J pairing) ----
    q = ggml_rope_ext(ctx, q, pos, /*freq_factors=*/nullptr, hd, kGpt2RopeMode,
                      /*n_ctx_orig=*/0, hp.rope_base, /*freq_scale=*/1.0f,
                      /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                      /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, kGpt2RopeMode, 0,
                      hp.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    Gpt2LayerOut out;
    struct ggml_tensor* k_used;
    struct ggml_tensor* v_used;
    if (k_cache) {
        // ---- device-resident cache: store new columns, read prefix by view ----
        const int64_t kv = (int64_t)past_seq + n_tokens;
        struct ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache, hd, n_h, n_tokens,
            k_cache->nb[1], k_cache->nb[2], (size_t)past_seq * k_cache->nb[2]);
        struct ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache, hd, n_h, n_tokens,
            v_cache->nb[1], v_cache->nb[2], (size_t)past_seq * v_cache->nb[2]);
        out.k_store = ggml_cpy(ctx, k, k_dst);
        out.v_store = ggml_cpy(ctx, v, v_dst);
        ggml_build_forward_expand(gf, out.k_store);
        ggml_build_forward_expand(gf, out.v_store);
        k_used = ggml_view_3d(ctx, k_cache, hd, n_h, kv, k_cache->nb[1], k_cache->nb[2], 0);
        v_used = ggml_view_3d(ctx, v_cache, hd, n_h, kv, v_cache->nb[1], v_cache->nb[2], 0);
    } else {
        // ---- additive path: concat past K/V along the sequence dim (axis 2) ----
        struct ggml_tensor* k_full = k_past ? ggml_concat(ctx, k_past, k, /*dim=*/2) : k;
        struct ggml_tensor* v_full = v_past ? ggml_concat(ctx, v_past, v, /*dim=*/2) : v;
        k_used = k_full;
        v_used = v_full;
        // k_full/v_full are non-contiguous views of intermediate graph buffers
        // the shared gallocr may recycle. Force contiguous so callers can mark
        // them as graph outputs and read them back.
        out.k_full = ggml_cont(ctx, k_full);
        out.v_full = ggml_cont(ctx, v_full);
    }

    // ---- eager MHA (shared by both paths) ----
    struct ggml_tensor* q_p = ggml_permute(ctx, q,      0, 2, 1, 3);  // [hd, seq, n_h]
    struct ggml_tensor* k_p = ggml_permute(ctx, k_used, 0, 2, 1, 3);  // [hd, seq_kv, n_h]
    struct ggml_tensor* v_p = ggml_permute(ctx, v_used, 0, 2, 1, 3);

    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);  // [seq_kv, seq_q, n_h]
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    struct ggml_tensor* attn = ggml_soft_max_ext(ctx, scores, mask, scale, /*max_bias=*/0.0f);

    struct ggml_tensor* v_t = ggml_cont(ctx, ggml_transpose(ctx, v_p));  // [seq_kv, hd, n_h]
    struct ggml_tensor* o   = ggml_mul_mat(ctx, v_t, attn);              // [hd, seq_q, n_h]

    o = ggml_permute(ctx, o, 0, 2, 1, 3);
    o = ggml_cont_2d(ctx, o, H, n_tokens);

    struct ggml_tensor* attn_out = moss::linear(ctx, w.cproj_w, w.cproj_b, o);
    struct ggml_tensor* h = ggml_add(ctx, x, attn_out);

    // ---- MLP: mlp_cproj( gelu_new( c_fc(LN2(h)) ) ) ----
    struct ggml_tensor* hn = moss::layer_norm(ctx, h, w.ln2_w, w.ln2_b, eps);
    struct ggml_tensor* f  = moss::linear(ctx, w.mlp_cproj_w, w.mlp_cproj_b,
                                          ggml_gelu(ctx, moss::linear(ctx, w.cfc_w, w.cfc_b, hn)));

    out.y = ggml_add(ctx, h, f);
    return out;
}
```

- [ ] **Step 3: Build + full suite (additive path byte-identical).**

Run: `cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"`
Expected: builds clean; full suite green. `test_nano_local` and the nano keystone (`test_nano_frame_loop`) exercise the null path (nano_local passes no cache args) and must be unchanged.

- [ ] **Step 4: Commit.**

```bash
git add src/gpt2.hpp src/gpt2.cpp
git commit -m "feat(kv): gpt2_layer_forward opt-in device-KV-cache path (additive path unchanged)

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 4: `NanoBackbone` → device-resident cache

**Files:**
- Modify: `src/nano_backbone.hpp` (members + dtor)
- Modify: `src/nano_backbone.cpp` (load/reset/run/dtor)
- Gate test: `tests/test_nano_backbone.cpp`

- [ ] **Step 1: Confirm the gate test passes today (baseline).**

Run: `ctest --test-dir build -R test_nano_backbone --output-on-failure 2>&1 | tail -6`
Expected: PASS, printing `prefill maxerr=...` and `decode maxerr=...` (note the decode value).

- [ ] **Step 2: Update the header.** Replace `src/nano_backbone.hpp` lines 1-32 with:

```cpp
#ifndef MOSS_NANO_BACKBONE_HPP
#define MOSS_NANO_BACKBONE_HPP
#include "gpt2.hpp"
#include "model_loader.hpp"
#include "ggml_extend.hpp"   // GgmlCtxPtr
#include "ggml-backend.h"    // ggml_backend_buffer_t
#include <vector>
namespace moss {
// Global backbone of MOSS-TTS-Nano: a 12-layer GPT-2 (interleaved-RoPE) stack
// over time with a persistent device-resident per-layer KV cache.
class NanoBackbone {
public:
    NanoBackbone() = default;
    ~NanoBackbone();
    NanoBackbone(const NanoBackbone&) = delete;             // owns a backend buffer
    NanoBackbone& operator=(const NanoBackbone&) = delete;
    bool load(const ModelLoader& m, int max_seq = 8192);  // reads gpt2.* metadata + N layers + gpt2.output_norm.{weight,bias}
    int  hidden() const { return hp_.hidden; }
    const Gpt2Hparams& hparams() const { return hp_; }
    void reset();
    int  past_len() const { return past_len_; }
    // Prefill S embedding rows ([hidden,S] row-major). Resets the cache to pos 0.
    // Returns the LAST row's hidden (post final LayerNorm).
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* last_hidden);
    // One step at the next position. embed: [hidden]. Returns hidden [hidden]
    // (post final LayerNorm). Attends to all cached positions + this one.
    bool decode_one(const std::vector<float>& embed, std::vector<float>* hidden);
private:
    bool run(const std::vector<float>& embeds, int T, std::vector<float>* out_hidden);

    Gpt2Hparams hp_{}; std::vector<Gpt2Layer> layers_;
    struct ggml_tensor *out_norm_w_=nullptr, *out_norm_b_=nullptr;
    const ModelLoader* m_=nullptr; int max_seq_=0, past_len_=0;
    // Device-resident per-layer K/V cache: [head_dim, n_head, max_seq, 1] on
    // kv_buffer_; written in place each step, read by [0:past+T] view.
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
};
}  // namespace moss
#endif
```

- [ ] **Step 3: Rewrite `load`, `reset`, `run`, add the destructor** in `src/nano_backbone.cpp`. Replace lines 22-173 (`namespace moss {` body up to and including `decode_one`) with:

```cpp
namespace moss {

NanoBackbone::~NanoBackbone() {
    if (kv_buffer_) ggml_backend_buffer_free(kv_buffer_);
}

bool NanoBackbone::load(const ModelLoader& m, int max_seq) {
    m_        = &m;
    max_seq_  = max_seq;
    past_len_ = 0;

    hp_.hidden    = (int)m.get_u32("gpt2.hidden", 0);
    hp_.n_head    = (int)m.get_u32("gpt2.n_head", 0);
    hp_.head_dim  = (int)m.get_u32("gpt2.head_dim", 0);
    hp_.d_ff      = (int)m.get_u32("gpt2.d_ff", 0);
    hp_.n_layers  = (int)m.get_u32("gpt2.n_layers", 0);
    hp_.rope_base = m.get_f32("gpt2.rope_base", 10000.f);
    hp_.ln_eps    = m.get_f32("gpt2.ln_eps", 1e-5f);

    if (hp_.hidden <= 0 || hp_.n_head <= 0 || hp_.head_dim <= 0 ||
        hp_.d_ff <= 0 || hp_.n_layers <= 0 || max_seq_ <= 0) {
        return false;
    }

    layers_.assign(hp_.n_layers, Gpt2Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!gpt2_load_layer(m, "gpt2", i, &layers_[i])) return false;
    }

    out_norm_w_ = m.tensor("gpt2.output_norm.weight");
    out_norm_b_ = m.tensor("gpt2.output_norm.bias");
    if (!out_norm_w_ || !out_norm_b_) return false;

    // Persistent device-resident KV cache: 2 tensors per layer,
    // [head_dim, n_head, max_seq, 1] (no GQA -> n_head heads).
    const int L = hp_.n_layers;
    kv_ctx_ = make_ctx(ggml_tensor_overhead() * (size_t)(2 * L) + 1024, /*no_alloc=*/true);
    if (!kv_ctx_) return false;
    k_cache_.assign(L, nullptr);
    v_cache_.assign(L, nullptr);
    for (int l = 0; l < L; ++l) {
        k_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_head, max_seq_, 1);
        v_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_head, max_seq_, 1);
    }
    kv_buffer_ = ggml_backend_alloc_ctx_tensors(kv_ctx_.get(), moss::backend());
    if (!kv_buffer_) { MOSS_LOGE("NanoBackbone: KV cache alloc failed"); return false; }

    past_len_ = 0;
    return true;
}

void NanoBackbone::reset() {
    past_len_ = 0;  // views are bounded by past_len_; stale cache bytes are never read
}

bool NanoBackbone::run(const std::vector<float>& embeds, int T,
                       std::vector<float>* out_hidden) {
    const int L    = hp_.n_layers;
    const int H    = hp_.hidden;
    const int past = past_len_;
    const int kv   = past + T;

    if (embeds.size() < (size_t)H * T) return false;
    if (past + T > max_seq_) { MOSS_LOGE("NanoBackbone: sequence exceeds max_seq"); return false; }

    auto cctx = moss::make_ctx(256 * 1024 * 1024, /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_set_input(x);
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    ggml_set_input(pos);
    struct ggml_tensor* mask = nullptr;
    if (T > 1) {
        mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kv, T);  // ne0=key, ne1=query
        ggml_set_input(mask);
    }

    auto* gf = ggml_new_graph_custom(ctx, 8192, false);

    struct ggml_tensor* h = x;
    for (int l = 0; l < L; ++l) {
        auto lo = gpt2_layer_forward(ctx, h, pos, mask, /*k_past=*/nullptr, /*v_past=*/nullptr,
                                     layers_[l], hp_, gf, k_cache_[l], v_cache_[l], past);
        h = lo.y;
    }

    // Final LayerNorm (with bias) on the LAST column only.
    struct ggml_tensor* last = h;
    if (T > 1) {
        last = ggml_view_2d(ctx, h, H, 1, h->nb[1], (size_t)(T - 1) * h->nb[1]);
        last = ggml_cont(ctx, last);
    }
    struct ggml_tensor* y = moss::layer_norm(ctx, last, out_norm_w_, out_norm_b_, hp_.ln_eps);
    ggml_set_output(y);
    ggml_build_forward_expand(gf, y);

    const float ninf = -std::numeric_limits<float>::infinity();
    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, embeds.data(), 0, (size_t)H * T * sizeof(float));

        std::vector<int32_t> p(T);
        for (int i = 0; i < T; ++i) p[i] = past + i;
        ggml_backend_tensor_set(pos, p.data(), 0, (size_t)T * sizeof(int32_t));

        if (mask) {
            std::vector<float> mvec((size_t)kv * T);
            for (int i = 0; i < T; ++i)
                for (int j = 0; j < kv; ++j)
                    mvec[(size_t)i * kv + j] = (j <= past + i) ? 0.0f : ninf;
            ggml_backend_tensor_set(mask, mvec.data(), 0, mvec.size() * sizeof(float));
        }
    };

    if (!moss::compute_graph_with_inputs(gf, set_inputs)) return false;

    past_len_ = kv;
    out_hidden->resize(H);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * sizeof(float));
    return true;
}

bool NanoBackbone::prefill(const std::vector<float>& embeds, int S,
                           std::vector<float>* last_hidden) {
    reset();
    return run(embeds, S, last_hidden);
}

bool NanoBackbone::decode_one(const std::vector<float>& embed, std::vector<float>* hidden) {
    return run(embed, 1, hidden);
}

}  // namespace moss
```

Note: if `MOSS_LOGE` is not visible, add `#include "common.hpp"` at the top of `src/nano_backbone.cpp`.

- [ ] **Step 4: Build + run the gate test — byte-identical.**

Run: `cmake --build build -j 2>&1 | tail -3 && ctest --test-dir build -R test_nano_backbone --output-on-failure 2>&1 | tail -6`
Expected: PASS with `decode maxerr` equal to the Step 1 baseline.

- [ ] **Step 5: Discrimination check (revert-to-fail).** In `src/gpt2.cpp` change the store offset to `(size_t)(past_seq + 1) * k_cache->nb[2]` (k only), rebuild, run `test_nano_backbone`.

Run: `cmake --build build -j >/dev/null 2>&1 && ctest --test-dir build -R test_nano_backbone --output-on-failure 2>&1 | tail -4`
Expected: **FAIL** (decode maxerr large). Then revert to `(size_t)past_seq * k_cache->nb[2]`, rebuild, confirm PASS. Record the fail maxerr; if it does not fail, try a wrong read width and flag for the reviewer.

- [ ] **Step 6: Full suite + grep-clean.**

Run: `cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed" && grep -rn "\->data" src/nano_backbone.cpp | grep -vE "\.data\(\)|out->data|->data\(\)" | grep -v "//"`
Expected: full suite green; grep EMPTY.

- [ ] **Step 7: Commit.**

```bash
git add src/nano_backbone.hpp src/nano_backbone.cpp
git commit -m "perf(kv): NanoBackbone device-resident KV cache (no per-step host roundtrip)

Mirrors the DelayBackbone fix for the GPT-2 (no-GQA, interleaved-RoPE) global
backbone: per-layer K/V live on a persistent backend buffer, written in place
and read by view. Byte-identical on CPU (test_nano_backbone decode==prefill at
the same maxerr; revert-to-fail discrimination confirmed).

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Task 5: Docs — mark the backbone-KV follow-up done

**Files:** Modify `AGENTS.md`.

- [ ] **Step 1: Add the DONE entry.** Find a sensible location near the other backbone/perf notes (the known-follow-ups sections). The backbone host-roundtrip was not a numbered follow-up in AGENTS.md (it was an internal "approach A" comment in `delay_backbone.cpp`/`nano_backbone.cpp`), so add a clear cross-cutting note. Insert after the V4 "GPU `->data` path — DONE" follow-up (search for `6. **GPU \`->data\` path — DONE.**`) a new bullet:

```markdown
7. **Backbone KV cache is device-resident — DONE.** Both global backbones
   (`DelayBackbone`/Qwen3 for V1/V2/V3, `NanoBackbone`/GPT-2 for V4) previously
   kept per-layer K/V as host `std::vector`s and re-uploaded/-read-back the full
   grown cache every decode step (O(T^2) host<->device transfer; a full-KV PCIe
   roundtrip per step on GPU). Now each backbone owns a persistent
   `[head_dim, n_kv, max_seq, 1]` per-layer cache on its own backend buffer;
   `qwen3_layer_forward`/`gpt2_layer_forward` gained an opt-in device-cache path
   (store the new columns via in-graph `ggml_cpy` to a cache view, read the
   `[0:past+T]` prefix by view) — KV data movement is O(T) and the per-step
   transfer is just the new hidden row in/out. Null cache => the existing
   additive `k_past`/`v_past` path, so the codec (`transformer.cpp`) and the
   bounded depth caches (`local_transformer`/`rt_local`/`nano_local`) are
   byte-identical. Validated by `test_delay_kv` / `test_nano_backbone`
   (decode==full-prefill at the same maxerr) + revert-to-fail discrimination on
   the store offset. Residual: GPU execution still unverified on this CPU-only
   box (correct-by-construction); the depth-loop caches remain host-driven
   (small/bounded, not worth converting).
```

- [ ] **Step 2: Also update the stale `delay_backbone.cpp`/`nano_backbone.cpp` "approach A" header comments** were already replaced in Tasks 2/4 (the function bodies). Confirm no AGENTS.md line still describes the backbone KV as host-roundtrip:

Run: `grep -niE "approach A|host.side full|host vectors.*K/V|per-step host" AGENTS.md`
Expected: no hits referring to the backbone KV as the current behavior. If a hit exists, update it to reference the device-resident cache.

- [ ] **Step 3: Final full suite + commit.**

```bash
cmake --build build -j 2>&1 | tail -2 && ctest --test-dir build 2>&1 | grep -E "tests passed|failed"   # full suite green
git add AGENTS.md
git commit -m "docs: mark the backbone device-resident KV cache follow-up done

Assisted-by: Claude:claude-opus-4-8 [Claude Code]"
```

---

## Final step (after all 5 tasks)

Dispatch a final whole-implementation reviewer for the entire branch:
- CPU byte-identical: `test_delay_kv` and `test_nano_backbone` pass at the **same** decode maxerr as before; the revert-to-fail discrimination empirically confirmed (a wrong store offset breaks them).
- The additive/null path is untouched → every depth-loop keystone (`test_depth_loop`, `test_rt_depth_loop`, `test_nano_frame_loop`), `test_local_transformer`, the codec tests, and all component/parity tests pass unchanged.
- No regression to V1/V2/V3/V4/Foundation; full `ctest` green.
- `src/` stays grep-clean of raw tensor `->data`.
- KV-store ordering correctness (the in-helper `ggml_build_forward_expand` of the store nodes precedes the attention read); cache buffer lifetime (destructor frees `kv_buffer_`, non-copyable); `max_seq` overflow guarded.

Then use `superpowers:finishing-a-development-branch` to merge `backbone-device-kv` to `main` and push (the user's durable preference: merge locally + push).

---

## Self-review notes (addressed)

- **Spec coverage:** persistent device buffer (Tasks 2/4 load), opt-in helper cache path (Tasks 1/3), driver rewrite with store+view + mask-null-on-decode (Tasks 2/4 run), removal of host `k_state_`/`v_state_` (Tasks 2/4), byte-identity gate + discrimination (Tasks 2/4 Steps 1/4/5), docs (Task 5). The seq-outermost contiguity, gallocr-leaves-the-cache-alone, and read-after-write ordering rationales from the spec are realized by the `[hd,nkv,max_seq,1]` layout + the in-helper `ggml_build_forward_expand` of the stores.
- **Type/signature consistency:** `qwen3_layer_forward` / `gpt2_layer_forward` both gain `(gf, k_cache, v_cache, past_seq)` with the same null defaults; `Qwen3LayerOut` / `Gpt2LayerOut` both gain `k_store`/`v_store`; the backbones use member names `kv_ctx_`, `kv_buffer_`, `k_cache_`, `v_cache_` identically. The qwen3 cache views are 4d (k is 4d `[hd,nkv,T,1]`); the gpt2 cache views are 3d (k is `reshape_3d [hd,nh,T]`) — matching each helper's K dimensionality so `ggml_cpy` shapes agree.
- **Byte-identity rationale:** the additive branch is the pre-existing code verbatim (concat → cont k_full/v_full); the cache branch feeds the same logical K/V matrix into the same shared attention. Mask-null on T=1 decode equals an all-zero additive mask (every cached key causally valid) → identical.
- **Placeholders:** none — every code step shows the full replacement body; every run step has an exact command + expected result.
