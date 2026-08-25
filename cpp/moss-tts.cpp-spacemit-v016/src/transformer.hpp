#ifndef MOSS_TRANSFORMER_HPP
#define MOSS_TRANSFORMER_HPP
#include "ggml.h"
#include "model_loader.hpp"
#include <string>
#include <vector>
namespace moss {
struct TransformerConfig {
    int   d_model = 768, n_heads = 12, n_layers = 12, d_ff = 3072;
    int   in_dim = 768, out_dim = 768;      // input/output projection dims
    float eps = 1e-5f;
    int   context = 0;                       // sliding-window length in frames; 0 = full causal
};
struct LayerWeights {
    struct ggml_tensor *norm1_w, *norm1_b, *norm2_w, *norm2_b;
    struct ggml_tensor *qkv_w, *out_w;       // fused in_projs.0 / out_projs.0
    struct ggml_tensor *lin1_w, *lin2_w;
    struct ggml_tensor *ls1, *ls2;           // layer_scale_*.scale
};
struct TransformerWeights {
    struct ggml_tensor *in_proj  = nullptr;  // null => identity
    struct ggml_tensor *out_proj = nullptr;  // null => identity
    std::vector<LayerWeights> layers;
};
// Load weights for a block named `prefix` (e.g. "encoder.1") from the loader.
TransformerWeights load_transformer(const ModelLoader& ld, const std::string& prefix,
                                    const TransformerConfig& cfg);
// Forward subgraph. x: ne[0]=in_dim(feature), ne[1]=T(tokens). Returns ne[0]=out_dim, ne[1]=T.
// pos: int32 length T (RoPE positions). mask: f32 (T,T) additive (0/-INF) or null for full causal.
// k_past/v_past: optional per-layer cached K/V (size n_layers; an entry may be
//   null for a prefill layer). nullptr vector => full-sequence prefill (today's
//   behavior, byte-identical). When provided, attention attends over
//   concat(past,new) on the sequence axis (dim=2, (hd,H,seq) layout) and `mask`
//   is [kv, T] (kv=cached+new).
// k_out/v_out: optional per-layer outputs receiving the contiguous full K/V
//   ((hd, n_heads, kv) layout) for the caller to cache + evict. Resized to n_layers.
struct ggml_tensor* run_transformer(struct ggml_context* ctx, const TransformerWeights& w,
                                    const TransformerConfig& cfg, struct ggml_tensor* x,
                                    struct ggml_tensor* pos, struct ggml_tensor* mask,
                                    const std::vector<struct ggml_tensor*>* k_past = nullptr,
                                    const std::vector<struct ggml_tensor*>* v_past = nullptr,
                                    std::vector<struct ggml_tensor*>* k_out = nullptr,
                                    std::vector<struct ggml_tensor*>* v_out = nullptr);
}  // namespace moss
#endif
