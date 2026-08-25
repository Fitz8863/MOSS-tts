#ifndef MOSS_TTS_MLP_HPP
#define MOSS_TTS_MLP_HPP
#include "ggml.h"
namespace moss {
struct MossMLPWeights { struct ggml_tensor *gate=nullptr,*up=nullptr,*down=nullptr; };  // gate/up: ne0=in,ne1=ff ; down: ne0=ff,ne1=out
// x: ne0=in, ne1=T. Returns ne0=out, ne1=T.
inline struct ggml_tensor* moss_mlp(struct ggml_context* ctx, const MossMLPWeights& w, struct ggml_tensor* x) {
    struct ggml_tensor* g = ggml_mul_mat(ctx, w.gate, x);   // (ff, T)
    struct ggml_tensor* u = ggml_mul_mat(ctx, w.up, x);     // (ff, T)
    struct ggml_tensor* h = ggml_mul(ctx, ggml_silu(ctx, g), u);
    return ggml_mul_mat(ctx, w.down, h);                    // (out, T)
}
}  // namespace moss
#endif
