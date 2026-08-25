#include "delay_backbone.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include "common.hpp"   // MOSS_LOGE

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// DelayBackbone — the Qwen3 transformer stack with a persistent KV cache.
//
// KV-cache approach: device-resident per-layer K/V.
//   k_cache_[l]/v_cache_[l] are [head_dim, n_kv_heads, max_seq, 1] tensors
//   living on a persistent backend buffer (kv_buffer_). Each step builds a
//   graph over all layers; qwen3_layer_forward writes the new T columns in
//   place at seq offset past_len_ via in-graph ggml_cpy and reads the
//   [0:past+T] prefix by view. No per-step host<->device K/V roundtrip — the
//   cache never leaves the device. The consistency test (decode == full
//   prefill) is the gate.

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

    // Reuse one metadata-ctx buffer across every run() instead of mallocing
    // 256 MB per step. Sized to the graph node budget (4096): 2*B tensor
    // overheads (>= every op's + leaf's struct) + the graph object + 1 MB.
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
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

    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
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
