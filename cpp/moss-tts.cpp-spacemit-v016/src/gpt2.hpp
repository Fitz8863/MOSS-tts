#ifndef MOSS_GPT2_HPP
#define MOSS_GPT2_HPP
#include "model_loader.hpp"
#include <string>
struct ggml_context; struct ggml_tensor; struct ggml_cgraph;
namespace moss {
enum Gpt2Activation { GPT2_GELU = 0, GPT2_SILU = 1 };
struct Gpt2Hparams { int hidden=0, n_head=0, head_dim=0, d_ff=0, n_layers=0; float rope_base=10000.f, ln_eps=1e-5f; Gpt2Activation act = GPT2_GELU; };
struct Gpt2Layer {
    struct ggml_tensor *ln1_w=nullptr,*ln1_b=nullptr,*cattn_w=nullptr,*cattn_b=nullptr,
                       *cproj_w=nullptr,*cproj_b=nullptr,*ln2_w=nullptr,*ln2_b=nullptr,
                       *cfc_w=nullptr,*cfc_b=nullptr,*mlp_cproj_w=nullptr,*mlp_cproj_b=nullptr;
};
struct Gpt2LayerOut { struct ggml_tensor *y=nullptr,*k_full=nullptr,*v_full=nullptr,
                                          *k_store=nullptr,*v_store=nullptr; };
// Load one GPT-2 layer's weights from `{prefix}.blk.{i}.*`. Returns false if any
// tensor is missing.
bool gpt2_load_layer(const ModelLoader& m, const std::string& prefix, int i, Gpt2Layer* out);
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
}  // namespace moss
#endif
