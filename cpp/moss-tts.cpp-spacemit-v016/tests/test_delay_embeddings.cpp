// Parity test for DelayEmbeddings (the 33-table sum-of-embeddings lookup)
// against a tiny numpy reference fixture (tests/fixtures/delay_embeddings.gguf).
//   embed[s] = embed_tokens[ids[s,0]] + Σ_i emb_ext[i][ids[s,i+1]]
// Pure float row gathers — compared to the numpy summed reference. Tol 1e-5.
#include "delay_embeddings.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_DELAY_EMBEDDINGS");
    std::string path = env ? env : "tests/fixtures/delay_embeddings.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    const int S = (int)ld.get_u32("S", 2);

    moss::DelayEmbeddings de;
    if (!de.load(ld)) {
        std::fprintf(stderr, "DelayEmbeddings::load failed\n");
        return 1;
    }

    if (de.n_vq() != 3) {
        std::fprintf(stderr, "n_vq mismatch: got %d want 3\n", de.n_vq());
        return 1;
    }
    if (de.hidden() != 4) {
        std::fprintf(stderr, "hidden mismatch: got %d want 4\n", de.hidden());
        return 1;
    }

    struct ggml_tensor* ids_t = ld.tensor("ids");
    if (!ids_t) { std::fprintf(stderr, "missing ids tensor\n"); return 1; }
    size_t n_ids = ggml_nelements(ids_t);
    std::vector<int32_t> ids;
    moss::read_tensor_i32(ids_t, &ids);
    (void)n_ids;

    std::vector<float> out;
    de.embed(ids, S, &out);

    struct ggml_tensor* ref = ld.tensor("out");
    size_t n = ggml_nelements(ref);
    if (out.size() != n) {
        std::fprintf(stderr, "out size mismatch: got %zu want %zu\n", out.size(), n);
        return 1;
    }
    std::vector<float> rv; moss::read_tensor_f32(ref, &rv);
    const float* r = rv.data();

    double maxerr = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::fabs(out[i] - r[i]);
        if (e > maxerr) maxerr = e;
    }
    if (maxerr > 1e-5) {
        std::fprintf(stderr, "delay_embeddings mismatch maxerr=%g\n", maxerr);
        for (size_t i = 0; i < n; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, out[i], r[i]);
        return 1;
    }
    std::printf("delay_embeddings ok (maxerr=%g)\n", maxerr);
    return 0;
}
