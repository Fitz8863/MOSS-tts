// NanoLocal depth-transformer per-frame-KV parity test against a tiny 1-layer
// GPT-2 (interleaved-RoPE) fixture (tests/fixtures/nano_local.gguf).
// H=8, n_head=2, head_dim=4, d_ff=16, rope_base=10000, ln_eps=1e-5.
//
// The fixture's `in` ([H, D]) holds D depth-position inputs forming ONE growing
// causal sequence at RoPE positions 0..D-1. `out_ref` ([D, H]) is the 1-layer
// GPT-2 causal forward over all D rows + final LayerNorm (gptl.output_norm) per
// row. NanoLocal drives ONE depth position per step over a per-frame KV cache;
// step(d) must attend to depths 0..d, so step(d) == out_ref[d]:
//   reset(); step(in[d], d) -> out_ref[d]   for d=0..D-1   (KV accumulation)
// Tol 1e-3. RoPE position = depth index = past_len_.
#include "nano_local.hpp"
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
    const char* env = std::getenv("MOSS_FIXTURE_NANO_LOCAL");
    std::string path = env ? env : "tests/fixtures/nano_local.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::NanoLocal lc;
    if (!lc.load(ld)) { std::fprintf(stderr, "nano_local load failed\n"); return 1; }

    const int H = lc.hidden();
    struct ggml_tensor* in_t  = ld.tensor("in");       // [H, D]
    struct ggml_tensor* out_t = ld.tensor("out_ref");  // [H, D]  (row d = depth-d output)
    if (!in_t || !out_t) { std::fprintf(stderr, "missing fixture tensors\n"); return 1; }
    if (in_t->ne[0] != H || out_t->ne[0] != H) {
        std::fprintf(stderr, "fixture hidden mismatch\n"); return 1;
    }

    const int D = (int)in_t->ne[1];
    if (out_t->ne[1] != D) { std::fprintf(stderr, "fixture depth mismatch\n"); return 1; }
    std::vector<float> in_v;  moss::read_tensor_f32(in_t, &in_v);
    std::vector<float> out_v; moss::read_tensor_f32(out_t, &out_v);
    const float* in  = in_v.data();
    const float* out = out_v.data();

    // Per-frame KV accumulation across the growing depth sequence: step(d) must
    // attend to depths 0..d via the per-frame cache.
    lc.reset();
    double worst = 0;
    for (int d = 0; d < D; ++d) {
        std::vector<float> in_vec(in + (size_t)d * H, in + (size_t)(d + 1) * H);
        std::vector<float> h;
        if (!lc.step(in_vec, d, &h)) { std::fprintf(stderr, "step %d failed\n", d); return 1; }
        if ((int)h.size() != H) { std::fprintf(stderr, "step %d bad size\n", d); return 1; }
        double e = maxerr(h, out + (size_t)d * H, H);
        std::printf("depth d=%d maxerr=%g\n", d, e);
        if (e > worst) worst = e;
        if (e > 1e-3) {
            std::fprintf(stderr, "NanoLocal parity FAIL at d=%d maxerr=%g\n", d, e);
            for (int i = 0; i < H; ++i)
                std::fprintf(stderr, "  [%d] ref=%g got=%g\n", i, out[(size_t)d * H + i], h[i]);
            return 1;
        }
    }

    std::printf("nano_local ok (per-frame KV parity maxerr=%g over D=%d depths)\n", worst, D);
    return 0;
}
