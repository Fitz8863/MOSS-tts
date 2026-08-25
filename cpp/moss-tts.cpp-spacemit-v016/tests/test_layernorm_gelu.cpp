#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_LNGELU");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/lngelu.gguf")) return 77;
    auto* x = ld.tensor("x"); auto* w = ld.tensor("w"); auto* b = ld.tensor("b");
    auto* ln_ref = ld.tensor("ln"); auto* gelu_ref = ld.tensor("gelu");
    auto ctx = moss::make_ctx(8 * 1024 * 1024, false);
    auto* ln = moss::layer_norm(ctx.get(), ggml_dup(ctx.get(), x), w, b, 1e-5f);
    auto* gl = ggml_gelu_erf(ctx.get(), ggml_dup(ctx.get(), x));
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, ln); ggml_build_forward_expand(gf, gl);
    if (!moss::compute_graph(gf)) return 1;
    size_t n = ggml_nelements(x); std::vector<float> a(n), c(n);
    // CPU backend: computed tensors keep valid ->data; read directly.
    memcpy(a.data(), ln->data, n * sizeof(float));
    memcpy(c.data(), gl->data, n * sizeof(float));
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(a[i] - ((float*)ln_ref->data)[i]) > 1e-4f) { std::fprintf(stderr, "ln[%zu] %g vs %g\n", i, a[i], ((float*)ln_ref->data)[i]); return 1; }
        if (std::fabs(c[i] - ((float*)gelu_ref->data)[i]) > 1e-4f) { std::fprintf(stderr, "gelu[%zu] %g vs %g\n", i, c[i], ((float*)gelu_ref->data)[i]); return 1; }
    }
    std::printf("layernorm+gelu ok\n"); return 0;
}
