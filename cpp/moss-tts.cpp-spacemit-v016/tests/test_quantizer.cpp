#include "quantizer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_QUANTIZER");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/quantizer.gguf")) return 77;
    moss::QuantizerWeights w;
    w.n_quantizers = ld.get_u32("NQ", 3); w.codebook_size = ld.get_u32("CS", 5);
    w.codebook_dim = ld.get_u32("CD", 4); w.rvq_dim = ld.get_u32("RVQ", 6);
    w.input_proj_w = ld.tensor("ipw"); w.input_proj_b = ld.tensor("ipb");
    w.output_proj_w = ld.tensor("opw"); w.output_proj_b = ld.tensor("opb");
    for (int i = 0; i < w.n_quantizers; ++i) {
        std::string s = std::to_string(i);
        w.in_proj_w.push_back(ld.tensor("inp"+s)); w.in_proj_b.push_back(ld.tensor("inb"+s));
        w.out_proj_w.push_back(ld.tensor("outp"+s)); w.out_proj_b.push_back(ld.tensor("outb"+s));
        w.codebooks.push_back(ld.tensor("cb"+s));
    }
    auto ctx = moss::make_ctx(64 * 1024 * 1024, false);
    auto* latent = ggml_dup(ctx.get(), ld.tensor("latent"));
    auto* codes = moss::quantize(ctx.get(), w, latent);
    auto* dec   = moss::dequantize(ctx.get(), w, codes);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, codes); ggml_build_forward_expand(gf, dec);
    if (!moss::compute_graph(gf)) return 1;
    auto* cref = ld.tensor("codes"); size_t nc = ggml_nelements(cref);
    const int32_t* gc = (const int32_t*)codes->data; const int32_t* rc = (const int32_t*)cref->data;
    for (size_t i = 0; i < nc; ++i) if (gc[i] != rc[i]) { std::fprintf(stderr, "code[%zu] %d vs %d\n", i, gc[i], rc[i]); return 1; }
    auto* dref = ld.tensor("dec"); size_t nd = ggml_nelements(dref);
    const float* gd = (const float*)dec->data; const float* rd = (const float*)dref->data;
    double maxerr = 0; for (size_t i = 0; i < nd; ++i) { double e=std::fabs(gd[i]-rd[i]); if(e>maxerr)maxerr=e;
        if (e > 1e-3f) { std::fprintf(stderr, "dec[%zu] %g vs %g\n", i, gd[i], rd[i]); return 1; } }
    std::printf("quantizer ok (maxerr=%g)\n", maxerr); return 0;
}
