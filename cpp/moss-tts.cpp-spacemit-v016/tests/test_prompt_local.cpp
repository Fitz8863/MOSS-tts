// Exact input_ids parity for build_generation_prompt_local vs the upstream
// python (moss_tts_local/processing_moss_tts.py). The fixture
// tests/fixtures/prompt_local.gguf (produced by
// scripts/gen_delay_reference.py prompt-local) dumps two cases:
//   (a) no reference (text only),
//   (b) a small fake reference (T_ref=3, n_vq=32),
// each with the resulting input_ids (S, 1+N_VQ) int32 + the inputs (text,
// reference_codes, T_ref). We rebuild the prompt in C++ with the REAL tokenizer
// (de_tokenizer.gguf) and assert input_ids match EXACTLY across channel 0 and
// all 32 audio channels.
//
// Skips (77) when either fixture is absent. MOSS_DE_TOKENIZER overrides the
// tokenizer path.
#include "prompt_local.hpp"
#include "de_tokenizer.hpp"
#include "delay_constants.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using moss::de::N_VQ;

static std::vector<int32_t> tensor_i32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<int32_t> v; moss::read_tensor_i32(t, &v);
    return v;
}

// Compare a built input_ids (S, 1+N_VQ) against the dumped expected; report the
// first differing (position, channel) with python-vs-cpp tokens. Returns 0 on
// exact match.
static int compare_case(const char* label, const std::vector<int32_t>& got, int S_got,
                        const std::vector<int32_t>& want, int S_want) {
    const int row = 1 + N_VQ;
    if (S_got != S_want) {
        std::fprintf(stderr, "%s: S mismatch got=%d want=%d\n", label, S_got, S_want);
        return 1;
    }
    if ((int)got.size() != S_got * row || (int)want.size() != S_want * row) {
        std::fprintf(stderr, "%s: size mismatch got=%zu want=%zu (S=%d row=%d)\n",
                     label, got.size(), want.size(), S_got, row);
        return 1;
    }
    for (int s = 0; s < S_got; ++s) {
        for (int c = 0; c < row; ++c) {
            const int g = got[(size_t)s * row + c];
            const int w = want[(size_t)s * row + c];
            if (g != w) {
                std::fprintf(stderr,
                             "%s: MISMATCH at pos=%d channel=%d (channel 0 = text) "
                             "cpp=%d py=%d\n",
                             label, s, c, g, w);
                return 1;
            }
        }
    }
    return 0;
}

int main() {
    const char* fx_env = std::getenv("MOSS_FIXTURE_PROMPT_LOCAL");
    std::string fx_path = fx_env ? fx_env : "tests/fixtures/prompt_local.gguf";

    moss::ModelLoader ld;
    if (!ld.load(fx_path)) {
        std::fprintf(stderr, "prompt-local: load failed: %s\n", fx_path.c_str());
        return 77;  // skip when fixture absent
    }

    const char* tok_env = std::getenv("MOSS_DE_TOKENIZER");
    std::string tok_path = tok_env ? tok_env : "tests/fixtures/de_tokenizer.gguf";
    moss::DeTokenizer tok;
    if (!tok.load_from_file(tok_path)) {
        std::fprintf(stderr, "prompt-local: tokenizer load failed: %s\n", tok_path.c_str());
        return 77;  // skip when tokenizer fixture absent
    }

    const int n_vq = (int)ld.get_u32("n_vq");
    if (n_vq != N_VQ) {
        std::fprintf(stderr, "prompt-local: n_vq mismatch fixture=%d N_VQ=%d\n", n_vq, N_VQ);
        return 1;
    }
    const std::string text = ld.get_str("text");
    const int S_a = (int)ld.get_u32("S_a");
    const int S_b = (int)ld.get_u32("S_b");
    const int T_ref = (int)ld.get_u32("T_ref");

    std::vector<int32_t> ids_a = tensor_i32(ld, "input_ids_a");
    std::vector<int32_t> ids_b = tensor_i32(ld, "input_ids_b");
    std::vector<int32_t> ref = tensor_i32(ld, "reference_codes");  // (T_ref, n_vq)

    if (ids_a.empty() || ids_b.empty() || ref.empty()) {
        std::fprintf(stderr, "prompt-local: missing fixture tensors\n");
        return 1;
    }

    moss::PromptLocalOpts opts;  // defaults render as "None", matching the dump.

    int failures = 0;

    // Case (a): no reference.
    {
        int S = 0;
        std::vector<int32_t> got =
            moss::build_generation_prompt_local(tok, text, {}, 0, opts, &S);
        failures += compare_case("no-ref", got, S, ids_a, S_a);
    }

    // Case (b): with reference.
    {
        int S = 0;
        std::vector<int32_t> got =
            moss::build_generation_prompt_local(tok, text, ref, T_ref, opts, &S);
        failures += compare_case("ref", got, S, ids_b, S_b);
    }

    if (failures == 0) {
        std::fprintf(stderr, "prompt-local: OK (both cases exact match)\n");
    }
    return failures == 0 ? 0 : 1;
}
