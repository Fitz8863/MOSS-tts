#include "local_adapters.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int check_eq(const char* tag, int got, int want) {
    if (got != want) { std::fprintf(stderr, "%s: %d vs %d\n", tag, got, want); return 1; }
    return 0;
}

int main() {
    const char* p = std::getenv("MOSS_FIXTURE_LOCAL_ADAPTERS");
    moss::ModelLoader ld;
    if (!ld.load(p ? p : "tests/fixtures/local_adapters.gguf")) return 77;

    moss::LocalAdapters la;
    if (!la.load(ld)) { std::fprintf(stderr, "load failed\n"); return 1; }

    int rc = 0;
    rc |= check_eq("channels", la.channels(), 3);
    rc |= check_eq("hidden", la.hidden(), 4);
    rc |= check_eq("local_hidden", la.local_hidden(), 3);
    rc |= check_eq("text_vocab", la.text_vocab(), 5);
    rc |= check_eq("audio_vocab", la.audio_vocab(), 5);
    if (rc) return rc;

    // to_local parity.
    auto* hv = ld.tensor("hidden_vec");
    std::vector<float> hidden_vec; moss::read_tensor_f32(hv, &hidden_vec);
    std::vector<float> lv;
    la.to_local(hidden_vec, &lv);
    auto* lvr = ld.tensor("local_vec");
    if ((int)lv.size() != (int)ggml_nelements(lvr)) {
        std::fprintf(stderr, "to_local size %zu vs %zu\n",
                     lv.size(), (size_t)ggml_nelements(lvr));
        return 1;
    }
    {
        std::vector<float> rv; moss::read_tensor_f32(lvr, &rv);
        const float* r = rv.data(); double maxerr = 0;
        for (size_t i = 0; i < lv.size(); ++i) {
            double e = std::fabs(lv[i] - r[i]);
            if (e > maxerr) maxerr = e;
            if (e > 1e-4) { std::fprintf(stderr, "local_vec[%zu] %g vs %g\n", i, lv[i], r[i]); return 1; }
        }
        std::printf("to_local ok (maxerr=%g)\n", maxerr);
    }

    // head_logits parity (channel 1, audio head with pad-mask at index 4).
    auto* lo = ld.tensor("local_out");
    std::vector<float> local_out; moss::read_tensor_f32(lo, &local_out);
    std::vector<float> lg;
    la.head_logits(1, local_out, &lg);
    auto* lgr = ld.tensor("logits1");
    if ((int)lg.size() != (int)ggml_nelements(lgr)) {
        std::fprintf(stderr, "head_logits size %zu vs %zu\n",
                     lg.size(), (size_t)ggml_nelements(lgr));
        return 1;
    }
    {
        std::vector<float> rv; moss::read_tensor_f32(lgr, &rv);
        const float* r = rv.data(); double maxerr = 0;
        for (size_t i = 0; i < lg.size(); ++i) {
            const bool gi = std::isinf(lg[i]) && lg[i] < 0;
            const bool ri = std::isinf(r[i]) && r[i] < 0;
            if (gi || ri) {
                if (gi != ri) { std::fprintf(stderr, "logits1[%zu] inf mismatch %g vs %g\n", i, lg[i], r[i]); return 1; }
                continue;
            }
            double e = std::fabs(lg[i] - r[i]);
            if (e > maxerr) maxerr = e;
            if (e > 1e-4) { std::fprintf(stderr, "logits1[%zu] %g vs %g\n", i, lg[i], r[i]); return 1; }
        }
        std::printf("head_logits ok (maxerr=%g, pad masked at idx 4)\n", maxerr);
    }

    std::printf("local_adapters ok\n");
    return 0;
}
