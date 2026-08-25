// Parity test for LMHeads (text head + N_VQ audio heads, pad-masked) against a
// tiny numpy reference fixture (tests/fixtures/lm_heads.gguf).
//   text_logits     = hidden @ text_w^T
//   audio_logits[i] = hidden @ audio_w[i]^T,  then pad slot -> -inf
// The fixture has AUDIO_PAD_CODE(=1024) out of range (AV=5), so the masked
// index falls back to AV-1=4, matching the fixture's numpy reference. Tol 1e-4.
#include "lm_heads.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_LM_HEADS");
    std::string path = env ? env : "tests/fixtures/lm_heads.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    moss::LMHeads h;
    if (!h.load(ld)) {
        std::fprintf(stderr, "LMHeads::load failed\n");
        return 1;
    }

    if (h.hidden() != 4 || h.text_vocab() != 6 ||
        h.audio_vocab() != 5 || h.n_vq() != 3) {
        std::fprintf(stderr, "dim mismatch: hidden=%d text_vocab=%d audio_vocab=%d n_vq=%d\n",
                     h.hidden(), h.text_vocab(), h.audio_vocab(), h.n_vq());
        return 1;
    }

    struct ggml_tensor* hid_t = ld.tensor("hidden");
    if (!hid_t) { std::fprintf(stderr, "missing hidden tensor\n"); return 1; }
    std::vector<float> hid;
    moss::read_tensor_f32(hid_t, &hid);

    std::vector<float> tl, al;
    h.logits(hid, &tl, &al);

    // --- text logits ---
    struct ggml_tensor* tl_ref = ld.tensor("text_logits");
    size_t ntl = ggml_nelements(tl_ref);
    if (tl.size() != ntl) {
        std::fprintf(stderr, "text_logits size mismatch: got %zu want %zu\n", tl.size(), ntl);
        return 1;
    }
    std::vector<float> tr_v; moss::read_tensor_f32(tl_ref, &tr_v);
    const float* tr = tr_v.data();
    double maxerr_t = 0;
    for (size_t i = 0; i < ntl; ++i) {
        double e = std::fabs(tl[i] - tr[i]);
        if (e > maxerr_t) maxerr_t = e;
    }
    if (maxerr_t > 1e-4) {
        std::fprintf(stderr, "text_logits mismatch maxerr=%g\n", maxerr_t);
        for (size_t i = 0; i < ntl; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, tl[i], tr[i]);
        return 1;
    }

    // --- audio logits (treat -inf specially) ---
    struct ggml_tensor* al_ref = ld.tensor("audio_logits");
    size_t nal = ggml_nelements(al_ref);
    if (al.size() != nal) {
        std::fprintf(stderr, "audio_logits size mismatch: got %zu want %zu\n", al.size(), nal);
        return 1;
    }
    std::vector<float> ar_v; moss::read_tensor_f32(al_ref, &ar_v);
    const float* ar = ar_v.data();
    double maxerr_a = 0;
    for (size_t i = 0; i < nal; ++i) {
        bool ref_inf = std::isinf(ar[i]) && ar[i] < 0;
        bool got_inf = std::isinf(al[i]) && al[i] < 0;
        if (ref_inf || got_inf) {
            if (ref_inf != got_inf) {
                std::fprintf(stderr, "audio_logits -inf mismatch at [%zu]: got=%g ref=%g\n",
                             i, al[i], ar[i]);
                return 1;
            }
            continue;  // both -inf -> ok
        }
        double e = std::fabs(al[i] - ar[i]);
        if (e > maxerr_a) maxerr_a = e;
    }
    if (maxerr_a > 1e-4) {
        std::fprintf(stderr, "audio_logits mismatch maxerr=%g\n", maxerr_a);
        for (size_t i = 0; i < nal; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, al[i], ar[i]);
        return 1;
    }

    std::printf("lm_heads ok (text maxerr=%g, audio maxerr=%g)\n", maxerr_t, maxerr_a);
    return 0;
}
