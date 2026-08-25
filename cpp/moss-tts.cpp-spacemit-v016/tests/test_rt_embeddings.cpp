// Parity test for RtEmbeddings (the MossTTSRealtime embedding tables) against a
// tiny numpy reference fixture (tests/fixtures/rt_embeddings.gguf).
//   embed_sum[s] = Σ_{c=0..channels-1} global_c[ids[s,c]]   (text c=0 INCLUDED)
//   embed_local_one(j, code) = local_j[code]
// Pure float row gathers — compared to the numpy references. Tol 1e-5.
#include "rt_embeddings.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_RT_EMBEDDINGS");
    std::string path = env ? env : "tests/fixtures/rt_embeddings.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    const int S = (int)ld.get_u32("S", 2);

    moss::RtEmbeddings re;
    if (!re.load(ld)) {
        std::fprintf(stderr, "RtEmbeddings::load failed\n");
        return 1;
    }

    if (re.channels() != 4) {
        std::fprintf(stderr, "channels mismatch: got %d want 4\n", re.channels());
        return 1;
    }
    if (re.n_local() != 3) {
        std::fprintf(stderr, "n_local mismatch: got %d want 3\n", re.n_local());
        return 1;
    }
    if (re.hidden() != 4) {
        std::fprintf(stderr, "hidden mismatch: got %d want 4\n", re.hidden());
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
    re.embed_sum(ids, S, &out);

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

    // --- embed_local_one parity (j=1, code=2) ---
    std::vector<float> one_out;
    re.embed_local_one(1, 2, &one_out);

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
        std::fprintf(stderr, "embed_local_one mismatch maxerr=%g\n", maxerr1);
        for (size_t i = 0; i < n1; ++i)
            std::fprintf(stderr, "  [%zu] got=%g ref=%g\n", i, one_out[i], r1[i]);
        return 1;
    }

    std::printf("rt_embeddings ok (sum maxerr=%g, one maxerr=%g)\n", maxerr, maxerr1);
    return 0;
}
