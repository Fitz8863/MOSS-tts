// LocalTransformer per-frame KV parity test against a tiny 2-layer NO-RoPE
// fixture (tests/fixtures/local_transformer.gguf). n_heads=2, n_kv_heads=1
// (GQA) and n_heads*head_dim=8 != hidden=6 (q_proj width mismatch).
//
// The fixture's `out` (H,T) is the NO-RoPE full-causal forward of both layers +
// final output RMSNorm applied to EVERY row — ground truth. By causality, row
// `pos` equals step(in[pos], pos) over the cached prefix 0..pos. LocalTransformer
// drives one token per depth step over a per-frame KV cache; the KV-cache result
// must equal that full recompute row-for-row.
//   1. reset(); step(emb[pos], pos) for pos 0..T-1 -> out[pos]  (KV accumulation)
//   2. reset(); step(emb[0], 0)                    -> out[0]     (reset works)
// Tol 1e-3. NO RoPE (use_rope=false); mask is all-zeros over the cached prefix.
#include "local_transformer.hpp"
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
    const char* env = std::getenv("MOSS_FIXTURE_LOCAL_TRANSFORMER");
    std::string path = env ? env : "tests/fixtures/local_transformer.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::LocalTransformer lt;
    if (!lt.load(ld)) { std::fprintf(stderr, "LocalTransformer::load failed\n"); return 77; }

    const int H = lt.hidden();
    struct ggml_tensor* emb_t = ld.tensor("embeds");
    struct ggml_tensor* out_t = ld.tensor("out");
    if (!emb_t || !out_t) { std::fprintf(stderr, "missing fixture tensors\n"); return 77; }

    const int T = (int)out_t->ne[1];   // out is [H, T]
    if ((int)ggml_nelements(emb_t) < H * T || (int)ggml_nelements(out_t) < H * T) {
        std::fprintf(stderr, "fixture tensors too small\n"); return 1;
    }
    std::vector<float> emb_v; moss::read_tensor_f32(emb_t, &emb_v);
    std::vector<float> out_v; moss::read_tensor_f32(out_t, &out_v);
    const float* emb = emb_v.data();
    const float* out = out_v.data();

    // Case 1: per-row KV accumulation across all depth steps.
    lt.reset();
    double worst = 0;
    for (int pos = 0; pos < T; ++pos) {
        std::vector<float> in_vec(emb + (size_t)pos * H, emb + (size_t)(pos + 1) * H);
        std::vector<float> h;
        if (!lt.step(in_vec, pos, &h)) { std::fprintf(stderr, "step %d failed\n", pos); return 1; }
        double e = maxerr(h, out + (size_t)pos * H, H);
        std::printf("step pos=%d maxerr=%g\n", pos, e);
        if (e > worst) worst = e;
        if (e > 1e-3) {
            std::fprintf(stderr, "LocalTransformer parity FAIL at pos=%d maxerr=%g\n", pos, e);
            for (int i = 0; i < H; ++i)
                std::fprintf(stderr, "  [%d] ref=%g got=%g\n", i, out[(size_t)pos * H + i], h[i]);
            return 1;
        }
    }

    // Case 2: reset() must restart the per-frame cache; pos 0 again == out[0].
    lt.reset();
    std::vector<float> in0(emb, emb + H);
    std::vector<float> h0;
    if (!lt.step(in0, 0, &h0)) { std::fprintf(stderr, "post-reset step failed\n"); return 1; }
    double er = maxerr(h0, out, H);
    std::printf("post-reset pos=0 maxerr=%g\n", er);
    if (er > 1e-3) { std::fprintf(stderr, "LocalTransformer reset() FAIL maxerr=%g\n", er); return 1; }

    std::printf("local transformer ok (parity maxerr=%g, reset maxerr=%g)\n", worst, er);
    return 0;
}
