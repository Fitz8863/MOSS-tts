#include "rt_local.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// RtLocal — the depth (local) Qwen3 transformer (4 layers WITH RoPE), driven
// ONE token per depth step over a per-frame KV cache.
//
// This is the per-step KV driver of DelayBackbone, specialized to T=1 and a
// per-frame cache: reset() is called before each frame's depth loop, and the
// RoPE position is the depth index `pos`, which (within a frame) coincides with
// past_len_ (depth 0 at past_len_=0, depth 1 at past_len_=1, ...). The full
// per-layer K/V (post q/k-RMSNorm + RoPE) is kept as host std::vector between
// steps; on each step we feed the stored past as k_past/v_past, run all layers,
// read the returned k_full/v_full back into the host state, then RMSNorm the
// single output column with the shared final norm (rtl.output_norm).

namespace moss {

bool RtLocal::load(const ModelLoader& m) {
    m_        = &m;
    past_len_ = 0;

    hp_.hidden       = (int)m.get_u32("rtl.hidden", 0);
    hp_.n_layers     = (int)m.get_u32("rtl.n_layers", 0);
    hp_.n_heads      = (int)m.get_u32("rtl.n_heads", 0);
    hp_.n_kv_heads   = (int)m.get_u32("rtl.n_kv_heads", 0);
    hp_.head_dim     = (int)m.get_u32("rtl.head_dim", 0);
    hp_.intermediate = (int)m.get_u32("rtl.intermediate", 0);
    hp_.text_vocab   = 0;  // unused for the depth transformer
    hp_.rope_base    = m.get_f32("rtl.rope_base", 1e6f);
    hp_.rms_eps      = m.get_f32("rtl.rms_eps", 1e-6f);
    hp_.use_rope     = true;

    if (hp_.hidden <= 0 || hp_.n_layers <= 0 || hp_.n_heads <= 0 ||
        hp_.n_kv_heads <= 0 || hp_.head_dim <= 0) {
        return false;
    }

    layers_.assign(hp_.n_layers, Qwen3Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!qwen3_load_layer(m, "rtl", i, &layers_[i])) return false;
    }

    output_norm_ = m.tensor("rtl.output_norm.weight");
    if (!output_norm_) return false;

    k_state_.assign(hp_.n_layers, {});
    v_state_.assign(hp_.n_layers, {});
    // Reuse one metadata-ctx buffer across every step() instead of mallocing
    // 256 MB per depth position. Sized to the graph node budget (4096).
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
    return true;
}

void RtLocal::reset() {
    past_len_ = 0;
    for (auto& s : k_state_) s.clear();
    for (auto& s : v_state_) s.clear();
}

bool RtLocal::step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden) {
    const int L    = hp_.n_layers;
    const int H    = hp_.hidden;
    const int hd   = hp_.head_dim;
    const int nkv  = hp_.n_kv_heads;
    const int past = past_len_;
    const int kv   = past + 1;

    // Within a frame the depth index equals past_len_; the RoPE position is
    // past_len_ (== pos). Guard against caller misuse.
    if (pos != past) return false;
    if (in_vec.size() < (size_t)H) return false;

    // Build the graph in a no_alloc ctx; x/pos/mask + per-layer k/v past are
    // input leaves filled after gallocr runs.
    auto cctx = moss::make_ctx_buf(scratch_.data(), scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_set_input(x);
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
            ggml_set_input(k_past[l]);
            ggml_set_input(v_past[l]);
        }
        auto lo = qwen3_layer_forward(ctx, h, posn, mask, k_past[l], v_past[l],
                                      layers_[l], hp_);
        h = lo.y;
        k_out[l] = lo.k_full;
        v_out[l] = lo.v_full;
        ggml_set_output(k_out[l]);
        ggml_set_output(v_out[l]);
    }

    // Final shared RMSNorm on the single output column.
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

        int32_t p0 = past;  // RoPE position = depth index
        ggml_backend_tensor_set(posn, &p0, 0, sizeof(int32_t));

        // The single query at depth `past` attends to all past keys 0..past-1
        // plus itself (full visibility over depths 0..past); causal by depth.
        std::vector<float> mvec((size_t)kv, 0.0f);
        ggml_backend_tensor_set(mask, mvec.data(), 0, mvec.size() * sizeof(float));

        for (int l = 0; l < L; ++l) {
            if (k_past[l]) {
                ggml_backend_tensor_set(k_past[l], k_state_[l].data(), 0,
                                        k_state_[l].size() * sizeof(float));
                ggml_backend_tensor_set(v_past[l], v_state_[l].data(), 0,
                                        v_state_[l].size() * sizeof(float));
            }
        }
    };

    if (!moss::compute_graph_with_inputs(gf, set_inputs)) return false;

    // Read back per-layer full K/V (now [hd, nkv, kv]) into host state.
    const size_t kvn = (size_t)hd * nkv * kv;
    for (int l = 0; l < L; ++l) {
        k_state_[l].resize(kvn);
        v_state_[l].resize(kvn);
        ggml_backend_tensor_get(k_out[l], k_state_[l].data(), 0, kvn * sizeof(float));
        ggml_backend_tensor_get(v_out[l], v_state_[l].data(), 0, kvn * sizeof(float));
    }
    past_len_ = kv;

    out_hidden->resize(H);
    ggml_backend_tensor_get(y, out_hidden->data(), 0, (size_t)H * sizeof(float));
    return true;
}

}  // namespace moss
