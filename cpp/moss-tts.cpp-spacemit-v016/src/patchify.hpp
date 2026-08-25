#ifndef MOSS_PATCHIFY_HPP
#define MOSS_PATCHIFY_HPP
#include "ggml.h"
namespace moss {
// x: ne[0]=T, ne[1]=D. Returns ne[0]=T/p, ne[1]=D*p. Mirrors PyTorch
// channel-first reshape(D,T/p,p)->permute(D,p,T/p)->reshape(D*p,T/p).
struct ggml_tensor* patch_down(struct ggml_context* ctx, struct ggml_tensor* x, int p);
// Inverse: ne[0]=L, ne[1]=D*p -> ne[0]=L*p, ne[1]=D.
struct ggml_tensor* patch_up(struct ggml_context* ctx, struct ggml_tensor* x, int p);
}  // namespace moss
#endif
