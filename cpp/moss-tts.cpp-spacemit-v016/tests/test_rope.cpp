#include "rope.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
int main() {
    const char* p = std::getenv("MOSS_FIXTURE_ROPE");
    moss::ModelLoader ld; if (!ld.load(p ? p : "tests/fixtures/rope.gguf")) return 77;
    int hd = ld.get_u32("hd", 8), T = ld.get_u32("T", 5), H = ld.get_u32("H", 2);
    auto* x = ld.tensor("x");          // ne0=hd, ne1=T, ne2=H
    auto* ref = ld.tensor("out");
    auto ctx = moss::make_ctx(16 * 1024 * 1024, false);
    // permute (ne0=hd,ne1=T,ne2=H) -> [hd, H, T] : swap ne1<->ne2
    auto* xr = ggml_cont(ctx.get(), ggml_permute(ctx.get(), ggml_dup(ctx.get(), x), 0, 2, 1, 3));
    auto* pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, T);
    // fill pos data now (CPU leaf): positions 0..T-1
    for (int i = 0; i < T; ++i) ((int32_t*)pos->data)[i] = i;
    auto* roped = ggml_rope_ext(ctx.get(), xr, pos, nullptr, hd, moss::kRopeMode,
                                0, moss::kRopeBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, roped);
    if (!moss::compute_graph(gf)) return 1;
    // roped layout is [hd, H, T]; ref is [hd, T, H]. compare with index math.
    const float* r = (const float*)ref->data;
    const float* got = (const float*)roped->data;
    double maxerr = 0;
    for (int h = 0; h < H; ++h) for (int t = 0; t < T; ++t) for (int d = 0; d < hd; ++d) {
        float gv = got[d + hd * (h + H * t)];     // [hd,H,T]
        float rv = r  [d + hd * (t + T * h)];     // [hd,T,H]
        maxerr = std::fmax(maxerr, std::fabs(gv - rv));
        if (std::fabs(gv - rv) > 1e-4f) { std::fprintf(stderr, "rope mismatch h%d t%d d%d: %g vs %g\n", h, t, d, gv, rv); return 1; }
    }
    std::printf("rope ok (maxerr=%g)\n", maxerr); return 0;
}
