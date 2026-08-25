// Env-gated backbone+heads logit-parity gate for the MOSS-TTS-Delay LM core.
//
// Replays the upstream deterministic forward (embed-sum -> Qwen3 backbone ->
// LM heads) on a fixed prompt and asserts our hidden/text/audio logits match a
// reference dumped by scripts/gen_delay_reference.py.
//
// SKIPs (77) unless both env vars are set (needs the 8B checkpoint, user box):
//   MOSS_TTS_DELAY    backbone .gguf (same file converters produce)
//   MOSS_DE_REF_DUMP  ref_dump.gguf from scripts/gen_delay_reference.py
//
// input_ids ne-layout: the dumper writes a numpy (S, 1+n_vq) i32 array; gguf
// reverses dims so ggml sees ne0=1+n_vq, ne1=S, and the raw memory order is
// ids[s*(1+n_vq) + c] — exactly what DelayEmbeddings::embed expects (row-major
// S*(1+n_vq)). So chans = ne[0] and a flat i32 copy is correct.
#include "delay_backbone.hpp"
#include "delay_embeddings.hpp"
#include "lm_heads.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static double maxerr(const std::vector<float>& a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > m) m = e;
    }
    return m;
}

int main() {
    const char* mb = std::getenv("MOSS_TTS_DELAY");      // backbone gguf
    const char* rf = std::getenv("MOSS_DE_REF_DUMP");    // ref_dump.gguf
    if (!mb || !rf) return 77;

    moss::ModelLoader bb;
    if (!bb.load(mb)) { std::fprintf(stderr, "backbone load failed\n"); return 1; }
    moss::ModelLoader ref;
    if (!ref.load(rf)) { std::fprintf(stderr, "ref load failed\n"); return 1; }

    moss::DelayEmbeddings emb;
    moss::LMHeads heads;
    moss::DelayBackbone backbone;
    if (!emb.load(bb) || !heads.load(bb)) {
        std::fprintf(stderr, "emb/heads load failed\n"); return 1;
    }

    int S = (int)ref.get_u32("S", 0);
    if (S <= 0) { std::fprintf(stderr, "bad S in ref dump\n"); return 1; }

    auto* idt = ref.tensor("input_ids");   // (S, 1+n_vq) i32
    if (!idt) { std::fprintf(stderr, "ref missing input_ids\n"); return 1; }
    int chans = (int)(idt->ne[0]);         // 1+n_vq
    (void)chans;
    std::vector<int32_t> ids; moss::read_tensor_i32(idt, &ids);

    if (!backbone.load(bb, S + 16)) {
        std::fprintf(stderr, "backbone load failed\n"); return 1;
    }

    std::vector<float> embeds;
    emb.embed(ids, S, &embeds);

    std::vector<float> hidden;
    if (!backbone.prefill(embeds, S, &hidden)) {
        std::fprintf(stderr, "prefill failed\n"); return 1;
    }

    auto* rh = ref.tensor("hidden");
    if (!rh) { std::fprintf(stderr, "ref missing hidden\n"); return 1; }
    if ((size_t)ggml_nelements(rh) != hidden.size()) {
        std::fprintf(stderr, "hidden size mismatch: ref %lld vs ours %zu\n", (long long)ggml_nelements(rh), hidden.size()); return 1; }
    std::vector<float> rh_v; moss::read_tensor_f32(rh, &rh_v);
    double he = maxerr(hidden, rh_v.data(), hidden.size());

    std::vector<float> tl, al;
    heads.logits(hidden, &tl, &al);

    auto* rtl = ref.tensor("text_logits");
    auto* ral = ref.tensor("audio_logits");
    if (!rtl || !ral) { std::fprintf(stderr, "ref missing logits\n"); return 1; }
    if ((size_t)ggml_nelements(rtl) != tl.size()) {
        std::fprintf(stderr, "text_logits size mismatch: ref %lld vs ours %zu\n", (long long)ggml_nelements(rtl), tl.size()); return 1; }
    if ((size_t)ggml_nelements(ral) != al.size()) {
        std::fprintf(stderr, "audio_logits size mismatch: ref %lld vs ours %zu (check audio_vocab: ref dump width vs C++ AUDIO_VOCAB=1025)\n", (long long)ggml_nelements(ral), al.size()); return 1; }
    std::vector<float> rtl_v; moss::read_tensor_f32(rtl, &rtl_v);
    double te = maxerr(tl, rtl_v.data(), tl.size());

    // audio: skip the pad index (masked to -inf in both sides).
    std::vector<float> ral_v; moss::read_tensor_f32(ral, &ral_v);
    double ae = 0;
    size_t n = al.size();
    for (size_t i = 0; i < n; ++i) {
        float a = al[i], b = ral_v[i];
        if (std::isinf(a) && std::isinf(b)) continue;
        double e = std::fabs((double)a - (double)b);
        if (e > ae) ae = e;
    }

    std::printf("backbone parity: hidden maxerr=%.4g text maxerr=%.4g "
                "audio maxerr=%.4g\n", he, te, ae);

    // First guess for an 8B f32 forward; the printed maxerrs let this be tuned
    // on the first real run (see scripts/gen_delay_reference.py header).
    const double TOL = 5e-2;
    if (he > TOL || te > TOL || ae > TOL) {
        std::fprintf(stderr, "parity above tol %.3g\n", TOL); return 1;
    }
    return 0;
}
