#include "nano_backbone.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "ggml_extend.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// NanoBackbone — the global 12-layer GPT-2 (interleaved-RoPE) stack with a
// persistent, device-resident per-layer KV cache (no GQA: K/V have n_head
// heads, shape [head_dim, n_head, max_seq, 1]). Each step builds a graph over
// all layers; gpt2_layer_forward stores the T new K/V columns in place into the
// cache at offset past_len_ and reads back the [0:past_len_+T] prefix by view,
// so there is no per-step host<->device roundtrip. The final norm is LayerNorm
// WITH bias (gpt2.output_norm.{weight,bias}) applied to the LAST row only. The
// decode==full-prefill consistency test is the correctness gate for the cache.

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

    // Reuse one metadata-ctx buffer across every run() instead of mallocing
    // 256 MB per step. Sized to the graph node budget (8192).
    scratch_.resize(ggml_tensor_overhead() * 2 * 8192
                  + ggml_graph_overhead_custom(8192, false)
                  + (1u << 20));
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

    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
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
