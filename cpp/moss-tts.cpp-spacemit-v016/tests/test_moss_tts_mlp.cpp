#include "moss_tts_mlp.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_MOSS_MLP");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/moss_mlp.gguf")) return 77;
    moss::MossMLPWeights w;
    w.gate = ld.tensor("gate"); w.up = ld.tensor("up"); w.down = ld.tensor("down");
    auto ctx = moss::make_ctx(8 * 1024 * 1024, false);
    auto* y = moss::moss_mlp(ctx.get(), w, ggml_dup(ctx.get(), ld.tensor("x")));
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, y);
    if (!moss::compute_graph(gf)) return 1;
    auto* yref = ld.tensor("y"); size_t n = ggml_nelements(yref);
    if (ggml_nelements(y) != n) { std::fprintf(stderr, "nelem %zu vs %zu\n", (size_t)ggml_nelements(y), n); return 1; }
    const float* gy = (const float*)y->data; const float* ry = (const float*)yref->data;
    double maxerr = 0; for (size_t i = 0; i < n; ++i) { double e = std::fabs(gy[i] - ry[i]); if (e > maxerr) maxerr = e;
        if (e > 1e-4f) { std::fprintf(stderr, "y[%zu] %g vs %g\n", i, gy[i], ry[i]); return 1; } }
    std::printf("moss_tts_mlp ok (maxerr=%g)\n", maxerr); return 0;
}
