#include "patchify.hpp"
namespace moss {
struct ggml_tensor* patch_down(struct ggml_context* ctx, struct ggml_tensor* x, int p) {
    const int64_t T = x->ne[0], D = x->ne[1];
    struct ggml_tensor* v = ggml_reshape_3d(ctx, x, p, T / p, D);       // ne0=r, ne1=q, ne2=d
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));               // swap ne0<->ne1 -> (q, r, d)
    return ggml_reshape_2d(ctx, v, T / p, D * p);                       // (T/p, D*p)
}
struct ggml_tensor* patch_up(struct ggml_context* ctx, struct ggml_tensor* x, int p) {
    const int64_t L = x->ne[0], Dp = x->ne[1], D = Dp / p;
    struct ggml_tensor* v = ggml_reshape_3d(ctx, x, L, p, D);           // ne0=q(=L), ne1=r, ne2=d
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));               // (r, q, d)
    return ggml_reshape_2d(ctx, v, L * p, D);                           // (L*p, D)
}
}  // namespace moss
