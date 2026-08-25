// Parity test for LocalEmbeddings (the embedding_list lookup) against a tiny
// numpy reference fixture (tests/fixtures/local_embeddings.gguf).
//   embed_sum[s] = Σ_{c=0..channels-1} table_c[ids[s,c]]   (text c=0 INCLUDED)
//   embed_one(channel, code) = table_channel[code]
// Pure float row gathers — compared to the numpy references. Tol 1e-5.
#include "local_embeddings.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_LOCAL_EMBEDDINGS");
    std::string path = env ? env : "tests/fixtures/local_embeddings.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    const int S = (int)ld.get_u32("S", 2);

    moss::LocalEmbeddings le;
    if (!le.load(ld)) {
        std::fprintf(stderr, "LocalEmbeddings::load failed\n");
        return 1;
    }

    if (le.channels() != 4) {
        std::fprintf(stderr, "channels mismatch: got %d want 4\n", le.channels());
        return 1;
    }
    if (le.hidden() != 4) {
        std::fprintf(stderr, "hidden mismatch: got %d want 4\n", le.hidden());
        return 1;
    }

    struct ggml_tensor* ids_t = ld.tensor("ids");
    if (!ids_t) { std::fprintf(stderr, "missing ids tensor\n"); return 1; }
    size_t n_ids = ggml_nelements(ids_t);
    std::vector<int32_t> ids;
    moss::read_tensor_i32(ids_t, &ids);
    (void)n_ids;

    // --- embed_sum parity ---
    std::vector<float> out;
    le.embed_sum(ids, S, &out);

    struct ggml_tensor* ref = ld.tensor("sum");
    size_t n = ggml_nelements(ref);
    if (out.size() != n) {
        std::fprintf(stderr, "sum size mismatch: got %zu want %zu\n", out.size(), n);
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
        std::fprintf(stderr, "embed_sum mismatch maxerr=%g\n", maxerr);
        for (size_t i = 0; i < n; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, out[i], r[i]);
        return 1;
    }

    // --- embed_one parity (channel=2, code=1) ---
    std::vector<float> one_out;
    le.embed_one(2, 1, &one_out);

    struct ggml_tensor* one_ref = ld.tensor("one");
    if (!one_ref) { std::fprintf(stderr, "missing one tensor\n"); return 1; }
    size_t n1 = ggml_nelements(one_ref);
    if (one_out.size() != n1) {
        std::fprintf(stderr, "one size mismatch: got %zu want %zu\n", one_out.size(), n1);
        return 1;
    }
    std::vector<float> r1v; moss::read_tensor_f32(one_ref, &r1v);
    const float* r1 = r1v.data();
    double maxerr1 = 0;
    for (size_t i = 0; i < n1; ++i) {
        double e = std::fabs(one_out[i] - r1[i]);
        if (e > maxerr1) maxerr1 = e;
    }
    if (maxerr1 > 1e-5) {
        std::fprintf(stderr, "embed_one mismatch maxerr=%g\n", maxerr1);
        for (size_t i = 0; i < n1; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, one_out[i], r1[i]);
        return 1;
    }

    std::printf("local_embeddings ok (sum maxerr=%g, one maxerr=%g)\n", maxerr, maxerr1);
    return 0;
}
