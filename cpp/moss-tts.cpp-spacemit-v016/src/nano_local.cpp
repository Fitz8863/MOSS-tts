#include "nano_local.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// NanoLocal — the depth (local) GPT-2 transformer (1 layer, interleaved RoPE),
// driven ONE depth position per step over a PER-FRAME KV cache.
//
// This is the per-step KV driver of the Nano depth stack, specialized to T=1
// and a per-frame cache: reset() is called before each frame's depth loop, and
// the RoPE position is the depth index `pos`, which (within a frame) coincides
// with past_len_ (depth 0 at past_len_=0, depth 1 at past_len_=1, ...). The
// full per-layer K/V (post-RoPE, [head_dim, n_head, kv]) is kept as host
// std::vector between steps; on each step we feed the stored past as
// k_past/v_past, run the layer(s), read the returned k_full/v_full back into the
// host state, then LayerNorm (with bias) the single output column with the
// shared final norm (gptl.output_norm). Adapted from rt_local.cpp: qwen3->gpt2
// (no GQA, K/V have n_head heads), rms_norm->layer_norm, prefix rtl->gptl.
// Depth-0 input is the global backbone hidden fed DIRECTLY (no projection),
// which requires gptl.hidden == gpt2.hidden.

namespace moss {

bool NanoLocal::load(const ModelLoader& m) {
    m_        = &m;
    past_len_ = 0;

    hp_.hidden    = (int)m.get_u32("gptl.hidden", 0);
    hp_.n_head    = (int)m.get_u32("gptl.n_head", 0);
    hp_.head_dim  = (int)m.get_u32("gptl.head_dim", 0);
    hp_.d_ff      = (int)m.get_u32("gptl.d_ff", 0);
    hp_.n_layers  = (int)m.get_u32("gptl.n_layers", 0);
    hp_.rope_base = m.get_f32("gptl.rope_base", 10000.f);
    hp_.ln_eps    = m.get_f32("gptl.ln_eps", 1e-5f);

    if (hp_.hidden <= 0 || hp_.n_head <= 0 || hp_.head_dim <= 0 ||
        hp_.d_ff <= 0 || hp_.n_layers <= 0) {
        return false;
    }

    layers_.assign(hp_.n_layers, Gpt2Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!gpt2_load_layer(m, "gptl", i, &layers_[i])) return false;
    }

    out_norm_w_ = m.tensor("gptl.output_norm.weight");
    out_norm_b_ = m.tensor("gptl.output_norm.bias");
    if (!out_norm_w_ || !out_norm_b_) return false;

    k_state_.assign(hp_.n_layers, {});
    v_state_.assign(hp_.n_layers, {});
    // Reuse one metadata-ctx buffer across every step() instead of mallocing
    // 256 MB per depth position. Sized to the graph node budget (4096).
    scratch_.resize(ggml_tensor_overhead() * 2 * 4096
                  + ggml_graph_overhead_custom(4096, false)
                  + (1u << 20));
    return true;
}

void NanoLocal::reset() {
    past_len_ = 0;
    for (auto& s : k_state_) s.clear();
    for (auto& s : v_state_) s.clear();
}

bool NanoLocal::step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden) {
    const int L    = hp_.n_layers;
    const int H    = hp_.hidden;
    const int hd   = hp_.head_dim;
    const int nh   = hp_.n_head;
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
            k_past[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nh, past, 1);
            v_past[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, nh, past, 1);
            ggml_set_input(k_past[l]);
            ggml_set_input(v_past[l]);
        }
        auto lo = gpt2_layer_forward(ctx, h, posn, mask, k_past[l], v_past[l],
                                     layers_[l], hp_);
        h = lo.y;
        k_out[l] = lo.k_full;
        v_out[l] = lo.v_full;
        ggml_set_output(k_out[l]);
        ggml_set_output(v_out[l]);
    }

    // Final shared LayerNorm (with bias) on the single output column.
    struct ggml_tensor* y = moss::layer_norm(ctx, h, out_norm_w_, out_norm_b_, hp_.ln_eps);
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

    // Read back per-layer full K/V (now [hd, nh, kv]) into host state.
    const size_t kvn = (size_t)hd * nh * kv;
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
