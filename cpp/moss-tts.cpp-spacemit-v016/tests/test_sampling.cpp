// Parity + determinism tests for sampling.cpp against a tiny numpy fixture
// (tests/fixtures/sampling.gguf). RNG-vs-numpy parity is OUT OF SCOPE; we use
// our own std::mt19937_64. The fixture pins the DETERMINISTIC parts:
//   - argmax (do_sample=false)
//   - top-k=3 survivor index set
//   - top-p=0.8 survivor index set (full-vector apply_top_p)
//   - repetition-penalty result (prev=[argmax], penalty=2.0)
// plus a determinism check (same seed -> same draw).
#include "sampling.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

static std::vector<float> tensor_f32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<float> v; moss::read_tensor_f32(t, &v);
    return v;
}

static std::vector<int32_t> tensor_i32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<int32_t> v; moss::read_tensor_i32(t, &v);
    return v;
}

static std::set<int> finite_indices(const std::vector<float>& v) {
    std::set<int> s;
    for (int i = 0; i < (int)v.size(); ++i)
        if (!(std::isinf(v[i]) && v[i] < 0)) s.insert(i);
    return s;
}

static std::set<int> mask_indices(const std::vector<int32_t>& m) {
    std::set<int> s;
    for (int i = 0; i < (int)m.size(); ++i)
        if (m[i]) s.insert(i);
    return s;
}

int main() {
    const char* env = std::getenv("MOSS_FIXTURE_SAMPLING");
    std::string path = env ? env : "tests/fixtures/sampling.gguf";

    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "load failed: %s\n", path.c_str());
        return 77;
    }

    const int argmax = ld.get_i32("argmax");
    const int topk = (int)ld.get_u32("topk");
    const float topp = ld.get_f32("topp");
    const float penalty = ld.get_f32("penalty");

    std::vector<float> logits = tensor_f32(ld, "logits");
    std::vector<int32_t> topk_mask = tensor_i32(ld, "topk_mask");
    std::vector<int32_t> topp_mask = tensor_i32(ld, "topp_mask");
    std::vector<float> repl = tensor_f32(ld, "repl");
    if (logits.empty() || repl.empty() || topk_mask.empty() || topp_mask.empty()) {
        std::fprintf(stderr, "missing fixture tensors\n");
        return 1;
    }

    std::mt19937_64 rng(7);

    // --- argmax (do_sample=false) ---
    {
        std::vector<float> c = logits;
        int got = moss::sample_token(c, {}, 1.0f, 1.0f, 0, /*do_sample=*/false, rng);
        if (got != argmax) {
            std::fprintf(stderr, "argmax mismatch: got=%d want=%d\n", got, argmax);
            return 1;
        }
    }

    // --- top-k survivor mask ---
    {
        std::vector<float> c = logits;
        moss::apply_top_k(c, topk);
        if (finite_indices(c) != mask_indices(topk_mask)) {
            std::fprintf(stderr, "top-k survivor set mismatch\n");
            return 1;
        }
    }

    // --- top-p survivor mask ---
    {
        std::vector<float> c = logits;
        moss::apply_top_p(c, topp);
        if (finite_indices(c) != mask_indices(topp_mask)) {
            std::fprintf(stderr, "top-p survivor set mismatch\n");
            return 1;
        }
    }

    // --- repetition penalty ---
    {
        std::vector<float> c = logits;
        moss::apply_repetition_penalty(c, {argmax}, penalty);
        double maxerr = 0;
        for (size_t i = 0; i < c.size(); ++i)
            maxerr = std::max(maxerr, (double)std::fabs(c[i] - repl[i]));
        if (maxerr > 1e-5) {
            std::fprintf(stderr, "rep-penalty mismatch maxerr=%g\n", maxerr);
            return 1;
        }
    }

    // --- determinism: same seed -> same draw ---
    {
        std::mt19937_64 ra(123), rb(123);
        std::vector<float> ca = logits, cb = logits;
        int ta = moss::sample_token(ca, {}, 1.0f, 0.9f, 5, true, ra);
        int tb = moss::sample_token(cb, {}, 1.0f, 0.9f, 5, true, rb);
        if (ta != tb) {
            std::fprintf(stderr, "determinism failure: %d != %d (same seed)\n", ta, tb);
            return 1;
        }
        // A different seed CAN differ (not asserted). Just exercise the path.
        std::mt19937_64 rc(999);
        std::vector<float> cc = logits;
        (void)moss::sample_token(cc, {}, 1.0f, 0.9f, 5, true, rc);
    }

    std::printf("sampling ok (argmax=%d topk=%d topp=%.3f penalty=%.3f)\n",
                argmax, topk, topp, penalty);
    return 0;
}
