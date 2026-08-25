#ifndef MOSS_QUANTIZER_HPP
#define MOSS_QUANTIZER_HPP
#include "ggml.h"
#include "model_loader.hpp"
#include <vector>
namespace moss {
struct QuantizerWeights {
    struct ggml_tensor *input_proj_w, *input_proj_b;     // 768->512
    struct ggml_tensor *output_proj_w, *output_proj_b;   // 512->768
    std::vector<struct ggml_tensor*> in_proj_w, in_proj_b;    // per q: 512->8
    std::vector<struct ggml_tensor*> out_proj_w, out_proj_b;  // per q: 8->512
    std::vector<struct ggml_tensor*> codebooks;               // per q: (cd=8, cs=1024) in ggml ne
    int n_quantizers = 32, codebook_size = 1024, codebook_dim = 8, rvq_dim = 512;
};
QuantizerWeights load_quantizer(const ModelLoader& ld, int n_quantizers,
                                int codebook_size, int codebook_dim, int rvq_dim);
// Encode: latent (feature=768, T) -> codes int32 (T, nq). Builds the graph.
struct ggml_tensor* quantize(struct ggml_context* ctx, const QuantizerWeights& w,
                             struct ggml_tensor* latent);
// Decode: codes int32 (T, nq) -> latent (768, T).
// k selects how many leading codebooks to sum: k<0 (default) uses all
// w.n_quantizers; k>=1 is a valid lower-bitrate first-k ResidualLFQ decode and
// requires the codes tensor to be (k, T). The output_proj is always full-width.
struct ggml_tensor* dequantize(struct ggml_context* ctx, const QuantizerWeights& w,
                               struct ggml_tensor* codes, int k = -1);
}  // namespace moss
#endif
