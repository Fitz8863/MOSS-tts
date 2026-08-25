// Partial-depth (first-k codebooks) decode parity.
// V3's Realtime model emits only k<NQ RVQ codes per frame; decoding with the
// first k codebooks is a valid lower-bitrate ResidualLFQ decode. This test
// checks dequantize(ctx, w, codes, k=2) against the numpy reference dec_k2 and
// confirms the full-depth path (k=-1) still works and differs.
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
    const char* p = std::getenv("MOSS_FIXTURE_PARTIAL_DECODE");
    moss::ModelLoader ld;
    if (!ld.load(p ? p : "tests/fixtures/partial_decode.gguf")) return 77;

    const int NQ = (int)ld.get_u32("NQ", 4), CS = (int)ld.get_u32("CS", 4);
    const int CD = (int)ld.get_u32("CD", 2), RVQ = (int)ld.get_u32("RVQ", 6);
    const int T  = (int)ld.get_u32("T", 3),  OUT = (int)ld.get_u32("OUT", 8);
    const int K  = (int)ld.get_u32("K", 2);

    moss::QuantizerWeights w;
    w.n_quantizers = NQ; w.codebook_size = CS; w.codebook_dim = CD; w.rvq_dim = RVQ;
    w.output_proj_w = ld.tensor("opw"); w.output_proj_b = ld.tensor("opb");
    for (int i = 0; i < NQ; ++i) {
        std::string s = std::to_string(i);
        w.out_proj_w.push_back(ld.tensor("outp"+s));
        w.out_proj_b.push_back(ld.tensor("outb"+s));
        w.codebooks.push_back(ld.tensor("cb"+s));
    }

    auto ctx = moss::make_ctx(64 * 1024 * 1024, false);
    // codes fixture: ggml ne0=NQ, ne1=T. Slice the first K columns into a
    // contiguous (K, T) tensor (the layout V3's orchestrator will hand us).
    auto* codes_full = ld.tensor("codes");                       // ne0=NQ, ne1=T
    auto* codes_kv = ggml_view_2d(ctx.get(), codes_full, K, T,
                                  codes_full->nb[1], 0);          // (K, T) strided
    auto* codes_k  = ggml_cont(ctx.get(), codes_kv);              // (K, T) contiguous

    auto* dec_k  = moss::dequantize(ctx.get(), w, codes_k, /*k=*/K);   // (OUT, T)
    auto* dec_all = moss::dequantize(ctx.get(), w, codes_full);        // (OUT, T), k=-1

    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dec_k);
    ggml_build_forward_expand(gf, dec_all);
    if (!moss::compute_graph(gf)) return 1;

    // --- partial-depth (k=2) parity vs dec_k2 ---
    auto* dref = ld.tensor("dec_k2");
    size_t nd = ggml_nelements(dref);
    if ((size_t)ggml_nelements(dec_k) != nd) {
        std::fprintf(stderr, "dec_k nelements %lld != ref %zu\n",
                     (long long)ggml_nelements(dec_k), nd);
        return 1;
    }
    if (dec_k->ne[0] != OUT || dec_k->ne[1] != T) {
        std::fprintf(stderr, "dec_k shape (%lld,%lld) != (%d,%d)\n",
                     (long long)dec_k->ne[0], (long long)dec_k->ne[1], OUT, T);
        return 1;
    }
    const float* gk = (const float*)dec_k->data;
    const float* rk = (const float*)dref->data;
    double maxerr = 0;
    for (size_t i = 0; i < nd; ++i) {
        double e = std::fabs((double)gk[i] - (double)rk[i]);
        if (e > maxerr) maxerr = e;
        if (e > 1e-3) { std::fprintf(stderr, "dec_k[%zu] %g vs %g\n", i, gk[i], rk[i]); return 1; }
    }

    // --- full-depth path (k=-1) still works and differs from partial ---
    if ((size_t)ggml_nelements(dec_all) != nd) {
        std::fprintf(stderr, "dec_all nelements %lld != %zu\n",
                     (long long)ggml_nelements(dec_all), nd);
        return 1;
    }
    const float* ga = (const float*)dec_all->data;
    double maxdiff = 0;
    for (size_t i = 0; i < nd; ++i) {
        double d = std::fabs((double)ga[i] - (double)gk[i]);
        if (d > maxdiff) maxdiff = d;
    }
    if (maxdiff < 1e-4) {
        std::fprintf(stderr, "full-depth decode did not differ from k=%d (maxdiff=%g)\n", K, maxdiff);
        return 1;
    }

    std::printf("partial_decode ok (k=%d maxerr=%g, full-vs-k maxdiff=%g)\n", K, maxerr, maxdiff);
    return 0;
}
