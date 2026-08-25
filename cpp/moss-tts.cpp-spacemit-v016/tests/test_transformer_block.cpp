#include "transformer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_TRANSFORMER");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/transformer.gguf")) return 77;
    moss::TransformerConfig cfg;
    cfg.d_model = ld.get_u32("D", 8); cfg.n_heads = ld.get_u32("H", 2);
    cfg.n_layers = ld.get_u32("L", 2); cfg.d_ff = ld.get_u32("F", 16);
    cfg.in_dim = cfg.out_dim = cfg.d_model; cfg.context = 0;
    int T = ld.get_u32("T", 4);
    moss::TransformerWeights w; w.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        std::string b = "l" + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.norm1_w = ld.tensor(b+"n1w"); L.norm1_b = ld.tensor(b+"n1b");
        L.norm2_w = ld.tensor(b+"n2w"); L.norm2_b = ld.tensor(b+"n2b");
        L.qkv_w = ld.tensor(b+"qkv"); L.out_w = ld.tensor(b+"ow");
        L.lin1_w = ld.tensor(b+"l1"); L.lin2_w = ld.tensor(b+"l2");
        L.ls1 = ld.tensor(b+"ls1"); L.ls2 = ld.tensor(b+"ls2");
    }
    auto ctx = moss::make_ctx(64 * 1024 * 1024, false);
    auto* pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, T);
    for (int i = 0; i < T; ++i) ((int32_t*)pos->data)[i] = i;
    auto* y = moss::run_transformer(ctx.get(), w, cfg, ggml_dup(ctx.get(), ld.tensor("x")), pos, nullptr);
    auto* gf = ggml_new_graph(ctx.get()); ggml_build_forward_expand(gf, y);
    if (!moss::compute_graph(gf)) return 1;
    auto* ref = ld.tensor("out"); size_t n = ggml_nelements(ref);
    const float* got = (const float*)y->data; const float* r = (const float*)ref->data;
    double maxerr = 0;
    for (size_t i = 0; i < n; ++i) { double e = std::fabs(got[i] - r[i]); if (e > maxerr) maxerr = e;
        if (e > 1e-3f) { std::fprintf(stderr, "block mismatch [%zu] %g vs %g\n", i, got[i], r[i]); return 1; } }
    std::printf("transformer block ok (maxerr=%g)\n", maxerr); return 0;
}
