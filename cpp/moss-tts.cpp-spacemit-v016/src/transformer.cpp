#include "transformer.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"
#include <cmath>
namespace moss {
static struct ggml_tensor* attention(struct ggml_context* ctx, const LayerWeights& lw,
                                     const TransformerConfig& cfg, struct ggml_tensor* x,
                                     struct ggml_tensor* pos, struct ggml_tensor* mask,
                                     struct ggml_tensor* k_past, struct ggml_tensor* v_past,
                                     struct ggml_tensor** k_full_out, struct ggml_tensor** v_full_out) {
    const int64_t Tt = x->ne[1], D = cfg.d_model, H = cfg.n_heads, hd = D / H;
    struct ggml_tensor* qkv = ggml_mul_mat(ctx, lw.qkv_w, x);        // (3D, Tt) ne0=3D
    struct ggml_tensor* q = ggml_view_2d(ctx, qkv, D, Tt, qkv->nb[1], 0);
    struct ggml_tensor* k = ggml_view_2d(ctx, qkv, D, Tt, qkv->nb[1], D * ggml_element_size(qkv));
    struct ggml_tensor* v = ggml_view_2d(ctx, qkv, D, Tt, qkv->nb[1], 2 * D * ggml_element_size(qkv));
    // (D, Tt) -> (hd, H, Tt) so rope sees [head_dim, n_head, n_tokens]
    auto to_heads = [&](struct ggml_tensor* t) {
        return ggml_reshape_3d(ctx, ggml_cont(ctx, t), hd, H, Tt);
    };
    q = to_heads(q); k = to_heads(k); v = to_heads(v);              // (hd, H, Tt)
    q = ggml_rope_ext(ctx, q, pos, nullptr, hd, kRopeMode, 0, kRopeBase, 1, 0, 1, 0, 0);
    k = ggml_rope_ext(ctx, k, pos, nullptr, hd, kRopeMode, 0, kRopeBase, 1, 0, 1, 0, 0);
    // KV cache: concat past on the sequence axis (ne2) in the (hd,H,seq) layout. v is NOT roped.
    // Cache contract: k_past is ALREADY RoPE'd (at its original absolute positions when it was new);
    // only the new tokens k are roped above, at the positions in `pos`. So k_full_out is post-RoPE.
    // The streaming caller MUST feed k_full_out back verbatim as next step's k_past WITHOUT re-roping
    // it (re-roping the cached K would double-rotate it and corrupt attention).
    struct ggml_tensor* k_cat = k_past ? ggml_concat(ctx, k_past, k, /*dim=*/2) : k;   // (hd,H,kv)
    struct ggml_tensor* v_cat = v_past ? ggml_concat(ctx, v_past, v, /*dim=*/2) : v;
    if (k_full_out) *k_full_out = ggml_cont(ctx, k_cat);
    if (v_full_out) *v_full_out = ggml_cont(ctx, v_cat);
    // attention over the concatenated K/V:
    q                      = ggml_cont(ctx, ggml_permute(ctx, q,     0, 2, 1, 3)); // (hd, Tt, H)
    struct ggml_tensor* kk = ggml_cont(ctx, ggml_permute(ctx, k_cat, 0, 2, 1, 3)); // (hd, kv, H)
    struct ggml_tensor* vv = ggml_cont(ctx, ggml_permute(ctx, v_cat, 0, 2, 1, 3)); // (hd, kv, H)
    struct ggml_tensor* scores = ggml_mul_mat(ctx, kk, q);                         // (kv, Tt_q, H)
    float scale = 1.0f / std::sqrt((float)hd);
    if (mask) {
        scores = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    } else {
        scores = ggml_scale(ctx, scores, scale);
        scores = ggml_diag_mask_inf(ctx, scores, 0);
        scores = ggml_soft_max(ctx, scores);
    }
    struct ggml_tensor* vt = ggml_cont(ctx, ggml_transpose(ctx, vv)); // (kv, hd, H)
    struct ggml_tensor* o = ggml_mul_mat(ctx, vt, scores);          // (hd, Tt_q, H)
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));           // (hd, H, Tt)
    o = ggml_reshape_2d(ctx, o, D, Tt);                             // (D, Tt)
    return ggml_mul_mat(ctx, lw.out_w, o);                          // (D, Tt)
}
struct ggml_tensor* run_transformer(struct ggml_context* ctx, const TransformerWeights& w,
                                    const TransformerConfig& cfg, struct ggml_tensor* x,
                                    struct ggml_tensor* pos, struct ggml_tensor* mask,
                                    const std::vector<struct ggml_tensor*>* k_past,
                                    const std::vector<struct ggml_tensor*>* v_past,
                                    std::vector<struct ggml_tensor*>* k_out,
                                    std::vector<struct ggml_tensor*>* v_out) {
    if (w.in_proj) x = ggml_mul_mat(ctx, w.in_proj, x);
    if (k_out) k_out->assign(w.layers.size(), nullptr);
    if (v_out) v_out->assign(w.layers.size(), nullptr);
    size_t li = 0;
    for (const auto& lw : w.layers) {
        struct ggml_tensor* h = layer_norm(ctx, x, lw.norm1_w, lw.norm1_b, cfg.eps);
        struct ggml_tensor* kp = (k_past && li < k_past->size()) ? (*k_past)[li] : nullptr;
        struct ggml_tensor* vp = (v_past && li < v_past->size()) ? (*v_past)[li] : nullptr;
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
}
TransformerWeights load_transformer(const ModelLoader& ld, const std::string& prefix,
                                    const TransformerConfig& cfg) {
    TransformerWeights w;
    w.in_proj  = ld.tensor(prefix + ".input_proj.weight");
    w.out_proj = ld.tensor(prefix + ".output_proj.weight");
    w.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string b = prefix + ".transformer.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.norm1_w = ld.tensor(b + "norm1.weight"); L.norm1_b = ld.tensor(b + "norm1.bias");
        L.norm2_w = ld.tensor(b + "norm2.weight"); L.norm2_b = ld.tensor(b + "norm2.bias");
        // Name-tolerant pick: MOSS-Audio-Tokenizer-v2 uses self_attn.in_proj /
        // out_proj / ffn.0 / ffn.2; Foundation codec + nano fixture use the
        // legacy self_attn.in_projs.0 / out_projs.0 / linear1 / linear2. Prefer
        // v2, fall back to legacy (keeps Foundation/nano byte-identical).
        auto pick = [&](const std::string& a, const std::string& b2) -> struct ggml_tensor* {
            struct ggml_tensor* t = ld.tensor(a);
            return t ? t : ld.tensor(b2);
        };
        L.qkv_w   = pick(b + "self_attn.in_proj.weight",  b + "self_attn.in_projs.0.weight");
        L.out_w   = pick(b + "self_attn.out_proj.weight", b + "self_attn.out_projs.0.weight");
        L.lin1_w  = pick(b + "ffn.0.weight", b + "linear1.weight");
        L.lin2_w  = pick(b + "ffn.2.weight", b + "linear2.weight");
        L.ls1     = ld.tensor(b + "layer_scale_1.scale"); L.ls2 = ld.tensor(b + "layer_scale_2.scale");
    }
    return w;
}
}  // namespace moss
