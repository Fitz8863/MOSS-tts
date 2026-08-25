#include "gpt2.hpp"
#include "ggml_extend.hpp"

#include <cmath>

// GPT-2 + interleaved-RoPE transformer layer, reused by both the global backbone
// (T4) and the local depth transformer (T5) of MOSS-TTS-Nano.
//
// Differences vs the Qwen3 layer (src/qwen3.cpp):
//   - LayerNorm WITH bias (not RMSNorm).
//   - Fused c_attn: ONE matmul -> 3*hidden (q,k,v concatenated) + bias, then split.
//   - MHA: no GQA, no per-head q/k norm.
//   - Interleaved RoPE (GPT-J pairing of dims 2i,2i+1) == GGML_ROPE_TYPE_NORMAL
//     (mode 0). This matches the w_rope / w_gpt2_block numpy reference, which
//     rotates head-dim pairs (2i, 2i+1). (Qwen3 uses the half-split NEOX mode.)
//   - gelu_new MLP: c_fc -> gelu(tanh-approx) -> mlp_cproj (single projection,
//     NO SwiGLU gate). ggml_gelu is the tanh approximation, matching gelu_new.

namespace moss {

namespace {
constexpr int kGpt2RopeMode = GGML_ROPE_TYPE_NORMAL;  // interleaved (GPT-J)
}  // namespace

bool gpt2_load_layer(const ModelLoader& m, const std::string& prefix, int i, Gpt2Layer* out) {
    if (!out) return false;
    const std::string b = prefix + ".blk." + std::to_string(i) + ".";
    auto get = [&](const std::string& name, struct ggml_tensor** dst) -> bool {
        struct ggml_tensor* t = m.tensor(b + name);
        if (!t) return false;
        *dst = t;
        return true;
    };
    bool ok = true;
    ok &= get("ln1.weight",       &out->ln1_w);
    ok &= get("ln1.bias",         &out->ln1_b);
    ok &= get("cattn.weight",     &out->cattn_w);
    ok &= get("cattn.bias",       &out->cattn_b);
    ok &= get("cproj.weight",     &out->cproj_w);
    ok &= get("cproj.bias",       &out->cproj_b);
    ok &= get("ln2.weight",       &out->ln2_w);
    ok &= get("ln2.bias",         &out->ln2_b);
    ok &= get("cfc.weight",       &out->cfc_w);
    ok &= get("cfc.bias",         &out->cfc_b);
    ok &= get("mlp_cproj.weight", &out->mlp_cproj_w);
    ok &= get("mlp_cproj.bias",   &out->mlp_cproj_b);
    return ok;
}

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

    // ---- MLP: mlp_cproj( act( c_fc(LN2(h)) ) ) ----
    // act = gelu_new (tanh-approx, default; Nano/global) or silu (v1.5 local).
    struct ggml_tensor* hn  = moss::layer_norm(ctx, h, w.ln2_w, w.ln2_b, eps);
    struct ggml_tensor* pre = moss::linear(ctx, w.cfc_w, w.cfc_b, hn);
    struct ggml_tensor* act = (hp.act == GPT2_SILU) ? ggml_silu(ctx, pre) : ggml_gelu(ctx, pre);
    struct ggml_tensor* f   = moss::linear(ctx, w.mlp_cproj_w, w.mlp_cproj_b, act);

    out.y = ggml_add(ctx, h, f);
    return out;
}

}  // namespace moss
