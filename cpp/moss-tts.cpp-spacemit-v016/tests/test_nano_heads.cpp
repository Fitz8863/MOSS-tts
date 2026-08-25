// Parity test for NanoHeads (V4 MossTTSNano output projection heads):
// 1 text head (decision token -> text vocab) + 16 audio heads (-> audio vocab).
//   text_logits  = h @ text.T        (bias-free, NO norm, NO pad mask)
//   audio_logits = h @ audio[c].T
// Pure float matmul on the LOCAL transformer hidden (output_norm already applied).
// Compared to the numpy reference for the text head and audio head c_sample. Tol 1e-4.
#include "nano_heads.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_NANO_HEADS");
    std::string path = env ? env : "tests/fixtures/nano_heads.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    moss::NanoHeads heads;
    if (!heads.load(ld)) {
        std::fprintf(stderr, "NanoHeads::load failed\n");
        return 77;
    }

    if (heads.n_audio() != 16) {
        std::fprintf(stderr, "n_audio mismatch: got %d want 16\n", heads.n_audio());
        return 1;
    }
    if (heads.hidden() != 6) {
        std::fprintf(stderr, "hidden mismatch: got %d want 6\n", heads.hidden());
        return 1;
    }
    if (heads.text_vocab() != 10) {
        std::fprintf(stderr, "text_vocab mismatch: got %d want 10\n", heads.text_vocab());
        return 1;
    }
    if (heads.audio_vocab() != 5) {
        std::fprintf(stderr, "audio_vocab mismatch: got %d want 5\n", heads.audio_vocab());
        return 1;
    }

    struct ggml_tensor* h_t = ld.tensor("h");
    if (!h_t) { std::fprintf(stderr, "missing h tensor\n"); return 1; }
    size_t hn = ggml_nelements(h_t);
    std::vector<float> h;
    moss::read_tensor_f32(h_t, &h);
    (void)hn;

    auto cmp = [](const std::vector<float>& got, struct ggml_tensor* ref,
                  const char* name) -> int {
        if (!ref) { std::fprintf(stderr, "missing %s tensor\n", name); return 2; }
        size_t n = ggml_nelements(ref);
        if (got.size() != n) {
            std::fprintf(stderr, "%s size mismatch: got %zu want %zu\n",
                         name, got.size(), n);
            return 2;
        }
        std::vector<float> rv; moss::read_tensor_f32(ref, &rv);
        const float* r = rv.data();
        double maxerr = 0;
        for (size_t i = 0; i < n; ++i) {
            double e = std::fabs(got[i] - r[i]);
            if (e > maxerr) maxerr = e;
        }
        if (maxerr > 1e-4) {
            std::fprintf(stderr, "%s mismatch maxerr=%g\n", name, maxerr);
            for (size_t i = 0; i < n; ++i)
                std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, got[i], r[i]);
            return 2;
        }
        std::printf("%s ok (maxerr=%g)\n", name, maxerr);
        return 0;
    };

    // Text head (decision token).
    std::vector<float> tl;
    heads.text_logits(h, &tl);
    if (int rc = cmp(tl, ld.tensor("text_logits_ref"), "text_logits")) return rc;

    // Audio head c_sample (the fixture chose c_sample=2).
    std::vector<float> al;
    heads.audio_logits(2, h, &al);
    if (int rc = cmp(al, ld.tensor("audio_c_ref"), "audio_logits")) return rc;

    std::printf("nano_heads ok\n");
    return 0;
}
