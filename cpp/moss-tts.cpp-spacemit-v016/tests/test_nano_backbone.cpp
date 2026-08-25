// NanoBackbone (global 12-layer GPT-2 stack) KV-cache parity test against a
// tiny 2-layer GPT-2 + interleaved-RoPE fixture (tests/fixtures/nano_backbone.gguf).
// H=8, n_head=2, head_dim=4, d_ff=16, rope_base=10000. Two cases:
//   1. prefill(embeds[0:3])                     -> prefill_h_ref   (positions 0,1,2; last row)
//   2. decode_one(next_embed)                   -> decode_h_ref    (position 3 attending 0..3)
// The decode step MUST carry the prefill K/V (positions 0..2) in the cache, so
// case 2 validates the persistent KV cache. Tol 1e-3.
#include "nano_backbone.hpp"
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
    const char* env = std::getenv("MOSS_FIXTURE_NANO_BACKBONE");
    std::string path = env ? env : "tests/fixtures/nano_backbone.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) { std::fprintf(stderr, "load failed: %s\n", path.c_str()); return 77; }

    moss::NanoBackbone g;
    if (!g.load(ld, /*max_seq=*/64)) { std::fprintf(stderr, "backbone load failed\n"); return 1; }

    const int H = g.hidden();
    struct ggml_tensor* embeds_t   = ld.tensor("embeds");
    struct ggml_tensor* next_t     = ld.tensor("next_embed");
    struct ggml_tensor* pref_ref_t = ld.tensor("prefill_h_ref");
    struct ggml_tensor* dec_ref_t  = ld.tensor("decode_h_ref");
    if (!embeds_t || !next_t || !pref_ref_t || !dec_ref_t) {
        std::fprintf(stderr, "missing fixture tensors\n"); return 1;
    }

    const int S = (int)embeds_t->ne[1];   // [H, S]
    if ((int)embeds_t->ne[0] != H || (int)next_t->ne[0] != H ||
        (int)pref_ref_t->ne[0] != H || (int)dec_ref_t->ne[0] != H) {
        std::fprintf(stderr, "fixture shape mismatch (H=%d)\n", H); return 1;
    }

    std::vector<float> embeds_v;   moss::read_tensor_f32(embeds_t, &embeds_v);
    std::vector<float> next_v;     moss::read_tensor_f32(next_t, &next_v);
    std::vector<float> pref_ref_v; moss::read_tensor_f32(pref_ref_t, &pref_ref_v);
    std::vector<float> dec_ref_v;  moss::read_tensor_f32(dec_ref_t, &dec_ref_v);
    const float* embeds   = embeds_v.data();
    const float* next     = next_v.data();
    const float* pref_ref = pref_ref_v.data();
    const float* dec_ref  = dec_ref_v.data();

    // Case 1: prefill all S rows -> last row's post-final-LN hidden.
    std::vector<float> all(embeds, embeds + (size_t)H * S);
    std::vector<float> h0;
    if (!g.prefill(all, S, &h0)) { std::fprintf(stderr, "prefill failed\n"); return 1; }
    double e1 = maxerr(h0, pref_ref, H);
    std::printf("prefill maxerr=%g (past_len=%d)\n", e1, g.past_len());

    // Case 2: decode one more row at position S; the KV cache from prefill must
    // carry positions 0..S-1.
    std::vector<float> nstep(next, next + H);
    std::vector<float> h1;
    if (!g.decode_one(nstep, &h1)) { std::fprintf(stderr, "decode_one failed\n"); return 1; }
    double e2 = maxerr(h1, dec_ref, H);
    std::printf("decode  maxerr=%g (past_len=%d)\n", e2, g.past_len());

    if (e1 > 1e-3 || e2 > 1e-3) {
        std::fprintf(stderr, "NanoBackbone parity FAIL: prefill=%g decode=%g\n", e1, e2);
        for (int i = 0; i < H; ++i)
            std::fprintf(stderr, "  [%d] pref_ref=%g prefill=%g | dec_ref=%g decode=%g\n",
                         i, pref_ref[i], h0[i], dec_ref[i], h1[i]);
        return 1;
    }
    std::printf("nano_backbone ok (prefill=%g decode=%g)\n", e1, e2);
    return 0;
}
