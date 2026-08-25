#include "quantizer.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <string>
namespace moss {
// L2-normalize each ne1 vector over ne0 (feature), eps 1e-12.
static struct ggml_tensor* l2norm_rows(struct ggml_context* ctx, struct ggml_tensor* x) {
    struct ggml_tensor* nrm = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, x))); // (1, ne1)
    nrm = ggml_clamp(ctx, nrm, 1e-12f, INFINITY);   // floor to avoid div-by-zero; backend-agnostic (no host scalar)
    return ggml_div(ctx, x, nrm);
}
QuantizerWeights load_quantizer(const ModelLoader& ld, int nq, int cs, int cd, int rvq) {
    QuantizerWeights w; w.n_quantizers=nq; w.codebook_size=cs; w.codebook_dim=cd; w.rvq_dim=rvq;
    w.input_proj_w  = ld.tensor("quantizer.input_proj.weight");  w.input_proj_b  = ld.tensor("quantizer.input_proj.bias");
    w.output_proj_w = ld.tensor("quantizer.output_proj.weight"); w.output_proj_b = ld.tensor("quantizer.output_proj.bias");
    for (int i=0;i<nq;++i){ std::string b="quantizer.quantizers."+std::to_string(i)+".";
        w.in_proj_w.push_back(ld.tensor(b+"in_proj.weight"));   w.in_proj_b.push_back(ld.tensor(b+"in_proj.bias"));
        w.out_proj_w.push_back(ld.tensor(b+"out_proj.weight")); w.out_proj_b.push_back(ld.tensor(b+"out_proj.bias"));
        w.codebooks.push_back(ld.tensor(b+"codebook.weight")); }
    return w;
}
struct ggml_tensor* quantize(struct ggml_context* ctx, const QuantizerWeights& w, struct ggml_tensor* latent) {
    struct ggml_tensor* residual = linear(ctx, w.input_proj_w, w.input_proj_b, latent); // (rvq, T)
    std::vector<struct ggml_tensor*> cols;
    for (int i=0;i<w.n_quantizers;++i){
        struct ggml_tensor* ze = linear(ctx, w.in_proj_w[i], w.in_proj_b[i], residual); // (cd, T)
        struct ggml_tensor* e  = l2norm_rows(ctx, ze);                                  // (cd, T)
        struct ggml_tensor* c  = l2norm_rows(ctx, w.codebooks[i]);                        // (cd, cs)
        struct ggml_tensor* sim = ggml_mul_mat(ctx, c, e);                               // (cs, T)
        // NOTE: ggml_argmax returns the LAST max index on ties; numpy argmin returns the FIRST.
        // Exact ties are effectively impossible with L2-normalized real codebooks, but keep this
        // in mind if a single real-model code ever diverges from the reference.
        struct ggml_tensor* idx = ggml_argmax(ctx, sim);                                 // (T,) i32
        // Stack so result is ggml ne0=nq, ne1=T (== numpy (T, nq), matching ref).
        cols.push_back(ggml_reshape_2d(ctx, idx, 1, idx->ne[0]));                         // (1, T)
        struct ggml_tensor* sel = ggml_get_rows(ctx, w.codebooks[i], idx);               // (cd, T)
        struct ggml_tensor* zq  = linear(ctx, w.out_proj_w[i], w.out_proj_b[i], sel);     // (rvq, T)
        residual = ggml_sub(ctx, residual, zq);
    }
    struct ggml_tensor* codes = cols[0];
    for (int i=1;i<w.n_quantizers;++i) codes = ggml_concat(ctx, codes, cols[i], 0);       // (nq, T)
    return codes;
}
struct ggml_tensor* dequantize(struct ggml_context* ctx, const QuantizerWeights& w, struct ggml_tensor* codes, int k) {
    // codes: ggml ne0=nq, ne1=T (== numpy (T, nq)). Quantizer i = column i.
    // k<0 sums all codebooks; k>=1 sums only the first k (codes is (k, T)).
    const int64_t T = codes->ne[1];
    const int n = (k < 0) ? w.n_quantizers : k;
    struct ggml_tensor* emb = nullptr;
    for (int i=0;i<n;++i){
        // Strided (1, T) view of column i, made contiguous and flattened to (T,).
        struct ggml_tensor* colv = ggml_view_2d(ctx, codes, 1, T, codes->nb[1], (size_t)i*codes->nb[0]);
        struct ggml_tensor* col  = ggml_reshape_1d(ctx, ggml_cont(ctx, colv), T);                   // (T,)
        struct ggml_tensor* sel = ggml_get_rows(ctx, w.codebooks[i], col);                          // (cd, T)
        struct ggml_tensor* zq  = linear(ctx, w.out_proj_w[i], w.out_proj_b[i], sel);               // (rvq, T)
        emb = emb ? ggml_add(ctx, emb, zq) : zq;
    }
    return linear(ctx, w.output_proj_w, w.output_proj_b, emb);   // (768, T)
}
}  // namespace moss
