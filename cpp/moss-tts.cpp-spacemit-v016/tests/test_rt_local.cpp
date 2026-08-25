// RtLocal depth-transformer KV parity test against a tiny 2-layer RoPE Qwen3
// fixture (tests/fixtures/rt_local.gguf). n_heads=2, n_kv_heads=1 (GQA) and
// n_heads*head_dim=8 != hidden=6 (q_proj width mismatch).
//
// The fixture's `out` (4,6) is the WITH-RoPE full-causal recompute of both
// layers + final output RMSNorm over the GROWING prefix in[0..pos], taken at
// the last row — ground truth. RtLocal drives one token per depth step over a
// per-frame KV cache; the KV-cache result must equal that full recompute.
//   1. reset(); step(in[pos], pos) for pos 0..3  -> out[pos]   (KV accumulation)
//   2. reset(); step(in[0], 0)                   -> out[0]      (reset works)
// Tol 1e-3. RoPE position = depth index = past_len_.
#include "rt_local.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static double maxerr(const std::vector<float>& a, const float* b, size_t n) {
    double e = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > e) e = d;
    }
    return e;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_RT_LOCAL");
    std::string path = env ? env : "tests/fixtures/rt_local.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::RtLocal rtl;
    if (!rtl.load(ld)) { std::fprintf(stderr, "rt_local load failed\n"); return 1; }

    const int H = rtl.hidden();
    struct ggml_tensor* in_t  = ld.tensor("in");
    struct ggml_tensor* out_t = ld.tensor("out");
    if (!in_t || !out_t) { std::fprintf(stderr, "missing fixture tensors\n"); return 1; }

    const int T = (int)in_t->ne[1];   // [H, T]
    std::vector<float> in_v;  moss::read_tensor_f32(in_t, &in_v);
    std::vector<float> out_v; moss::read_tensor_f32(out_t, &out_v);
    const float* in  = in_v.data();
    const float* out = out_v.data();

    // Case 1: per-frame KV accumulation across all depth steps.
    rtl.reset();
    double worst = 0;
    for (int pos = 0; pos < T; ++pos) {
        std::vector<float> in_vec(in + (size_t)pos * H, in + (size_t)(pos + 1) * H);
        std::vector<float> h;
        if (!rtl.step(in_vec, pos, &h)) { std::fprintf(stderr, "step %d failed\n", pos); return 1; }
        double e = maxerr(h, out + (size_t)pos * H, H);
        std::printf("step pos=%d maxerr=%g\n", pos, e);
        if (e > worst) worst = e;
        if (e > 1e-3) {
            std::fprintf(stderr, "RtLocal parity FAIL at pos=%d maxerr=%g\n", pos, e);
            for (int i = 0; i < H; ++i)
                std::fprintf(stderr, "  [%d] ref=%g got=%g\n", i, out[(size_t)pos * H + i], h[i]);
            return 1;
        }
    }

    // Case 2: reset() must restart the per-frame cache; pos 0 again == out[0].
    rtl.reset();
    std::vector<float> in0(in, in + H);
    std::vector<float> h0;
    if (!rtl.step(in0, 0, &h0)) { std::fprintf(stderr, "post-reset step failed\n"); return 1; }
    double er = maxerr(h0, out, H);
    std::printf("post-reset pos=0 maxerr=%g\n", er);
    if (er > 1e-3) {
        std::fprintf(stderr, "RtLocal reset() FAIL maxerr=%g\n", er);
        return 1;
    }

    std::printf("rt_local ok (parity maxerr=%g, reset maxerr=%g)\n", worst, er);
    return 0;
}
