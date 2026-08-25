// Pure-CPU unit test for moss::build_window_mask.
// Checks: sliding window (T=5, context=3) and full-causal degradation (context>=T).
#include "audio_tokenizer.hpp"
#include "ggml_extend.hpp"
#include <cmath>
#include <cstdio>

static int check(struct ggml_tensor* m, int T, int context) {
    const float* d = (const float*)m->data;
    for (int i = 0; i < T; ++i) {       // query
        for (int j = 0; j < T; ++j) {   // key
            int delta = i - j;
            bool want;
            if (context <= 0 || context >= T) want = (delta >= 0);
            else                              want = (delta >= 0 && delta < context);
            float v = d[i * T + j];
            bool got_allowed = (v == 0.0f);
            bool got_blocked = std::isinf(v) && v < 0.0f;
            if (want && !got_allowed) {
                std::fprintf(stderr, "T=%d ctx=%d [%d,%d] expected 0 got %g\n", T, context, i, j, v);
                return 1;
            }
            if (!want && !got_blocked) {
                std::fprintf(stderr, "T=%d ctx=%d [%d,%d] expected -INF got %g\n", T, context, i, j, v);
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    auto ctx = moss::make_ctx(1 * 1024 * 1024, false);
    // sliding window T=5, context=3
    if (check(moss::build_window_mask(ctx.get(), 5, 3), 5, 3)) return 1;
    // full causal when context >= T
    if (check(moss::build_window_mask(ctx.get(), 5, 5), 5, 5)) return 1;
    if (check(moss::build_window_mask(ctx.get(), 5, 100), 5, 100)) return 1;
    // context <= 0 -> full causal
    if (check(moss::build_window_mask(ctx.get(), 4, 0), 4, 0)) return 1;
    std::printf("window mask ok\n");
    return 0;
}
