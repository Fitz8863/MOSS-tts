// Parity test for RtHeads (the 16 per-codebook linear heads of MossTTSRealtime)
// against a tiny numpy reference fixture (tests/fixtures/rt_heads.gguf).
//   logits[i] = h @ head[i].T     (bias-free, NO norm, NO pad mask)
// Pure float matmul — compared to the numpy reference for head i=1. Tol 1e-4.
#include "rt_heads.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_RT_HEADS");
    std::string path = env ? env : "tests/fixtures/rt_heads.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    moss::RtHeads rh;
    if (!rh.load(ld)) {
        std::fprintf(stderr, "RtHeads::load failed\n");
        return 1;
    }

    if (rh.n_heads() != 3) {
        std::fprintf(stderr, "n_heads mismatch: got %d want 3\n", rh.n_heads());
        return 1;
    }
    if (rh.hidden() != 4) {
        std::fprintf(stderr, "hidden mismatch: got %d want 4\n", rh.hidden());
        return 1;
    }
    if (rh.audio_vocab() != 5) {
        std::fprintf(stderr, "audio_vocab mismatch: got %d want 5\n", rh.audio_vocab());
        return 1;
    }

    struct ggml_tensor* h_t = ld.tensor("h");
    if (!h_t) { std::fprintf(stderr, "missing h tensor\n"); return 1; }
    std::vector<float> h;
    moss::read_tensor_f32(h_t, &h);

    std::vector<float> lg;
    rh.logits(1, h, &lg);

    struct ggml_tensor* ref = ld.tensor("logits1");
    if (!ref) { std::fprintf(stderr, "missing logits1 tensor\n"); return 1; }
    size_t n = ggml_nelements(ref);
    if (lg.size() != n) {
        std::fprintf(stderr, "logits size mismatch: got %zu want %zu\n", lg.size(), n);
        return 1;
    }
    std::vector<float> rv; moss::read_tensor_f32(ref, &rv);
    const float* r = rv.data();
    double maxerr = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::fabs(lg[i] - r[i]);
        if (e > maxerr) maxerr = e;
    }
    if (maxerr > 1e-4) {
        std::fprintf(stderr, "logits mismatch maxerr=%g\n", maxerr);
        for (size_t i = 0; i < n; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, lg[i], r[i]);
        return 1;
    }

    std::printf("rt_heads ok (logits maxerr=%g)\n", maxerr);
    return 0;
}
