// Pure-CPU unit test for moss::build_stream_window_mask.
// Streaming sliding-window mask: queries are the Tt new frames at absolute
// positions [past..past+Tt-1]; keys are the [cached(n_cached) | new(Tt)] buffer.
// delta = qi + n_cached - m; allowed iff (0 <= delta < context) for context>0,
// else (delta >= 0) for context<=0. Layout: ne0 = kv = n_cached+Tt (key,
// fastest), ne1 = Tt (query) -> flat dst[qi*kv + m]. Independent of `past`.
#include "audio_tokenizer.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

static int check(int n_cached, int Tt, int context) {
    const int kv = n_cached + Tt;
    std::vector<float> dst;
    moss::build_stream_window_mask(n_cached, Tt, context, &dst);
    if ((int)dst.size() != kv * Tt) {
        std::fprintf(stderr, "size mismatch: got %zu want %d\n", dst.size(), kv * Tt);
        return 1;
    }
    for (int qi = 0; qi < Tt; ++qi) {
        for (int m = 0; m < kv; ++m) {
            int delta = qi + n_cached - m;
            bool want = (context <= 0) ? (delta >= 0) : (delta >= 0 && delta < context);
            float v = dst[(size_t)qi * kv + m];
            bool got_allowed = (v == 0.0f);
            bool got_blocked = std::isinf(v) && v < 0.0f;
            if (want && !got_allowed) {
                std::fprintf(stderr, "nc=%d Tt=%d ctx=%d [qi=%d,m=%d] expected 0 got %g\n",
                             n_cached, Tt, context, qi, m, v);
                return 1;
            }
            if (!want && !got_blocked) {
                std::fprintf(stderr, "nc=%d Tt=%d ctx=%d [qi=%d,m=%d] expected -INF got %g\n",
                             n_cached, Tt, context, qi, m, v);
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    // Case 1: n_cached=2, Tt=2, context=2 (kv=4).
    //   qi=0 allows m in {1,2}; qi=1 allows m in {2,3}.
    if (check(2, 2, 2)) return 1;
    // Case 2: n_cached=0, Tt=3, context=0 (unbounded/causal, kv=3) -> lower-triangular.
    if (check(0, 3, 0)) return 1;
    // Case 3: n_cached=3, Tt=1, context=10 (context>=kv, kv=4) -> all allowed.
    if (check(3, 1, 10)) return 1;
    std::printf("stream window mask ok\n");
    return 0;
}
