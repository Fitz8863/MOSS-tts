#include "local_transformer.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace moss {

bool LocalTransformer::load(const ModelLoader& m) {
    m_ = &m;

    // Dispatch on the local depth-transformer architecture. "gptj" (v1.5) reuses
    // the GPT-2 block with silu + interleaved RoPE; "qwen3" (Delay, default/absent)
    // stays byte-identical below.
    std::string arch = m.get_str("local.arch", "qwen3");
    gptj_ = (arch == "gptj");
    if (gptj_) {
        ghp_.hidden    = (int)m.get_u32("local.hidden", 0);
        ghp_.n_head    = (int)m.get_u32("local.n_head", 0);
        ghp_.head_dim  = (int)m.get_u32("local.head_dim", 0);
        ghp_.d_ff      = (int)m.get_u32("local.d_ff", 0);
        ghp_.n_layers  = (int)m.get_u32("local.n_layers", 0);
        ghp_.rope_base = m.get_f32("local.rope_base", 10000.f);
        ghp_.ln_eps    = m.get_f32("local.ln_eps", 1e-5f);
        ghp_.act       = (m.get_str("local.activation", "gelu") == "silu") ? GPT2_SILU : GPT2_GELU;
        if (ghp_.hidden <= 0 || ghp_.n_head <= 0 || ghp_.head_dim <= 0 ||
            ghp_.d_ff <= 0 || ghp_.n_layers <= 0) {
            return false;
        }
        glayers_.assign(ghp_.n_layers, Gpt2Layer{});
        for (int i = 0; i < ghp_.n_layers; ++i) {
            if (!gpt2_load_layer(m, "local", i, &glayers_[i])) return false;
        }
        output_norm_ = m.tensor("local.output_norm.weight");
        out_norm_b_  = m.tensor("local.output_norm.bias");
        if (!output_norm_ || !out_norm_b_) return false;

        k_state_.assign(ghp_.n_layers, {});
        v_state_.assign(ghp_.n_layers, {});
        step_scratch_.resize(64 * 1024 * 1024);
        past_len_ = 0;
        return true;
    }

    hp_.hidden       = (int)m.get_u32("local.hidden", 0);
    hp_.n_layers     = (int)m.get_u32("local.n_layers", 0);
    hp_.n_heads      = (int)m.get_u32("local.n_heads", 0);
    hp_.n_kv_heads   = (int)m.get_u32("local.n_kv_heads", 0);
    hp_.head_dim     = (int)m.get_u32("local.head_dim", 0);
    hp_.intermediate = (int)m.get_u32("local.intermediate", 0);
    hp_.rms_eps      = m.get_f32("local.rms_eps", 1e-6f);
    // RoPE is NOT used by the depth transformer; rope_base is unused.
    hp_.use_rope     = false;
    hp_.rope_base    = 10000.0f;

    if (hp_.hidden <= 0 || hp_.n_layers <= 0 || hp_.n_heads <= 0 ||
        hp_.n_kv_heads <= 0 || hp_.head_dim <= 0 || hp_.intermediate <= 0) {
        return false;
    }

    layers_.assign(hp_.n_layers, Qwen3Layer{});
    for (int i = 0; i < hp_.n_layers; ++i) {
        if (!qwen3_load_layer(m, "local", i, &layers_[i])) return false;
    }

    output_norm_ = m.tensor("local.output_norm.weight");
    if (!output_norm_) return false;

    k_state_.assign(hp_.n_layers, {});
    v_state_.assign(hp_.n_layers, {});
    step_scratch_.resize(64 * 1024 * 1024);
    past_len_ = 0;
    return true;
}

void LocalTransformer::reset() {
    past_len_ = 0;
    for (auto& s : k_state_) s.clear();
    for (auto& s : v_state_) s.clear();
}

bool LocalTransformer::step(const std::vector<float>& in_vec, int pos,
                            std::vector<float>* out_hidden) {
    if (gptj_) return step_gptj(in_vec, pos, out_hidden);

    const int L    = hp_.n_layers;
    const int H    = hp_.hidden;
    const int hd   = hp_.head_dim;
    const int nkv  = hp_.n_kv_heads;
    const int past = past_len_;
    const int kv   = past + 1;

    // The depth index equals past_len_; guard against caller misuse.
    if (pos != past) return false;
    if (!out_hidden || in_vec.size() < (size_t)H) return false;

    // Build the graph in a no_alloc ctx over the reused scratch buffer; x/pos/
    // mask + per-layer k/v past are input leaves filled after gallocr runs.
    auto cctx = moss::make_ctx_buf(step_scratch_.data(), step_scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();
    if (!ctx) return false;

    struct ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 1);
    ggml_set_input(x);
    struct ggml_tensor* posn = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);  // unused (use_rope=false)
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
        // Do NOT upload posn: unreachable under use_rope=false (gallocr skips it).
        std::vector<float> mvec((size_t)kv, 0.0f);  // single query sees full prefix 0..past
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

// GPT-J (v1.5) depth path — a near-verbatim copy of NanoLocal::step, using the
// shared per-frame KV state but the GPT-2 block (silu MLP, interleaved RoPE with
// RoPE position = depth index) and a final LayerNorm (with bias). Depth-0 input
// is the global backbone hidden fed directly (requires ghp_.hidden == backbone
// hidden).
bool LocalTransformer::step_gptj(const std::vector<float>& in_vec, int pos,
                                 std::vector<float>* out_hidden) {
    const int L    = ghp_.n_layers;
    const int H    = ghp_.hidden;
    const int hd   = ghp_.head_dim;
    const int nh   = ghp_.n_head;
    const int past = past_len_;
    const int kv   = past + 1;

    // Within a frame the depth index equals past_len_; the RoPE position is
    // past_len_ (== pos). Guard against caller misuse.
    if (pos != past) return false;
    if (!out_hidden || in_vec.size() < (size_t)H) return false;

    auto cctx = moss::make_ctx_buf(step_scratch_.data(), step_scratch_.size(), /*no_alloc=*/true);
    struct ggml_context* ctx = cctx.get();
    if (!ctx) return false;

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
                                     glayers_[l], ghp_);
        h = lo.y;
        k_out[l] = lo.k_full;
        v_out[l] = lo.v_full;
        ggml_set_output(k_out[l]);
        ggml_set_output(v_out[l]);
    }

    // Final shared LayerNorm (with bias) on the single output column.
    struct ggml_tensor* y = moss::layer_norm(ctx, h, output_norm_, out_norm_b_, ghp_.ln_eps);
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

        std::vector<float> mvec((size_t)kv, 0.0f);  // single query sees full prefix 0..past
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
