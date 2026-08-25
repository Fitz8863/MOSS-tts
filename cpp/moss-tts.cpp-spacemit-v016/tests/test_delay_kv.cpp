// DelayBackbone KV-cache parity test against a tiny 2-layer Qwen3 model
// fixture (tests/fixtures/qwen3_tiny_model.gguf). n_heads=4, n_kv_heads=2 so
// GQA head-mapping is exercised. Two cases:
//   1. prefill(embeds[0:4])           ->  hidden_all[3]   (full prefill path)
//   2. prefill(embeds[0:3]) + decode_one(embeds[3]) -> hidden_all[3]
//      (KV-cache concat/decode equals full prefill — the consistency gate)
// Tol 1e-3.
#include "delay_backbone.hpp"
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
    const char* env = std::getenv("MOSS_FIXTURE_QWEN3_TINY_MODEL");
    std::string path = env ? env : "tests/fixtures/qwen3_tiny_model.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::DelayBackbone bb;
    if (!bb.load(ld, /*max_seq=*/16)) { std::fprintf(stderr, "backbone load failed\n"); return 1; }

    const int H = bb.hidden();
    struct ggml_tensor* embeds_t = ld.tensor("embeds");
    struct ggml_tensor* ref_t    = ld.tensor("hidden_all");
    if (!embeds_t || !ref_t) { std::fprintf(stderr, "missing fixture tensors\n"); return 1; }

    const int S = (int)embeds_t->ne[1];   // [H, S]
    std::vector<float> embeds_v; moss::read_tensor_f32(embeds_t, &embeds_v);
    std::vector<float> ref_v; moss::read_tensor_f32(ref_t, &ref_v);
    const float* embeds = embeds_v.data();
    const float* ref    = ref_v.data();
    const float* ref_last = ref + (size_t)(S - 1) * H;   // hidden_all[S-1]

    std::vector<float> all(embeds, embeds + (size_t)H * S);

    // Case 1: full prefill of all S rows.
    std::vector<float> out1;
    if (!bb.prefill(all, S, &out1)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
    double e1 = maxerr(out1, ref_last, H);
    std::printf("prefill   maxerr=%g (past_len=%d)\n", e1, bb.past_len());

    // Case 2: prefill first S-1, then decode the last row incrementally.
    bb.reset();
    std::vector<float> pre(embeds, embeds + (size_t)H * (S - 1));
    std::vector<float> tmp;
    if (!bb.prefill(pre, S - 1, &tmp)) { std::fprintf(stderr, "prefill(S-1) failed\n"); return 1; }
    std::vector<float> step(embeds + (size_t)H * (S - 1), embeds + (size_t)H * S);
    std::vector<float> out2;
    if (!bb.decode_one(step, &out2)) { std::fprintf(stderr, "decode_one failed\n"); return 1; }
    double e2 = maxerr(out2, ref_last, H);
    std::printf("decode    maxerr=%g (past_len=%d)\n", e2, bb.past_len());

    if (e1 > 1e-3 || e2 > 1e-3) {
        std::fprintf(stderr, "DelayBackbone parity FAIL: prefill=%g decode=%g\n", e1, e2);
        for (int i = 0; i < H; ++i)
            std::fprintf(stderr, "  [%d] ref=%g prefill=%g decode=%g\n",
                         i, ref_last[i], out1[i], out2[i]);
        return 1;
    }
    std::printf("delay kv ok (prefill=%g decode=%g)\n", e1, e2);
    return 0;
}
