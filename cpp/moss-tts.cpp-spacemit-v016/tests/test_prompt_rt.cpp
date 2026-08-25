// Exact parity for build_generation_prompt_rt vs the upstream python
// (moss_tts_realtime/inferencer.py offline prompt path: make_ensemble +
// _build_prefill_batch + _next_text_tokens). The fixture
// tests/fixtures/prompt_rt.gguf (produced by
// scripts/gen_delay_reference.py prompt-rt) dumps two cases:
//   (a) no reference (text only),
//   (b) a small fake reference (T_ref=3, 16 channels),
// each with the resulting prefill_ids (S, 17) int32 + remaining_text (1-D i32) +
// the inputs (text, reference_codes, T_ref). We rebuild the prompt in C++ with
// the REAL tokenizer (de_tokenizer.gguf) and assert prefill_ids match EXACTLY
// across channel 0 and all 16 audio channels, and remaining_text matches.
//
// Skips (77) when either fixture is absent. MOSS_DE_TOKENIZER overrides the
// tokenizer path.
#include "prompt_rt.hpp"
#include "de_tokenizer.hpp"
#include "rt_constants.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using moss::rt::CHANNELS;
using moss::rt::RVQ;

static std::vector<int32_t> tensor_i32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<int32_t> v; moss::read_tensor_i32(t, &v);
    return v;
}

// Compare a built prefill_ids (S, 17) against the dumped expected; report the
// first differing (position, channel) with python-vs-cpp tokens. Returns 0 on
// exact match.
static int compare_prefill(const char* label, const std::vector<int32_t>& got, int S_got,
                           const std::vector<int32_t>& want, int S_want) {
    if (S_got != S_want) {
        std::fprintf(stderr, "%s: S mismatch got=%d want=%d\n", label, S_got, S_want);
        return 1;
    }
    if ((int)got.size() != S_got * CHANNELS || (int)want.size() != S_want * CHANNELS) {
        std::fprintf(stderr, "%s: size mismatch got=%zu want=%zu (S=%d channels=%d)\n",
                     label, got.size(), want.size(), S_got, CHANNELS);
        return 1;
    }
    for (int s = 0; s < S_got; ++s) {
        for (int c = 0; c < CHANNELS; ++c) {
            const int g = got[(size_t)s * CHANNELS + c];
            const int w = want[(size_t)s * CHANNELS + c];
            if (g != w) {
                std::fprintf(stderr,
                             "%s: MISMATCH at row=%d channel=%d (channel 0 = text) "
                             "cpp=%d py=%d\n",
                             label, s, c, g, w);
                return 1;
            }
        }
    }
    return 0;
}

static int compare_remaining(const char* label, const std::vector<int32_t>& got,
                             const std::vector<int32_t>& want) {
    if (got.size() != want.size()) {
        std::fprintf(stderr, "%s: remaining_text len mismatch got=%zu want=%zu\n",
                     label, got.size(), want.size());
        return 1;
    }
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            std::fprintf(stderr,
                         "%s: remaining_text MISMATCH at %zu cpp=%d py=%d\n",
                         label, i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

int main() {
    const char* fx_env = std::getenv("MOSS_FIXTURE_PROMPT_RT");
    std::string fx_path = fx_env ? fx_env : "tests/fixtures/prompt_rt.gguf";

    moss::ModelLoader ld;
    if (!ld.load(fx_path)) {
        std::fprintf(stderr, "prompt-rt: load failed: %s\n", fx_path.c_str());
        return 77;  // skip when fixture absent
    }

    const char* tok_env = std::getenv("MOSS_DE_TOKENIZER");
    std::string tok_path = tok_env ? tok_env : "tests/fixtures/de_tokenizer.gguf";
    moss::DeTokenizer tok;
    if (!tok.load_from_file(tok_path)) {
        std::fprintf(stderr, "prompt-rt: tokenizer load failed: %s\n", tok_path.c_str());
        return 77;  // skip when tokenizer fixture absent
    }

    const int channels = (int)ld.get_u32("channels");
    if (channels != CHANNELS) {
        std::fprintf(stderr, "prompt-rt: channels mismatch fixture=%d CHANNELS=%d\n",
                     channels, CHANNELS);
        return 1;
    }
    const std::string text = ld.get_str("text");
    const int S_a = (int)ld.get_u32("S_a");
    const int S_b = (int)ld.get_u32("S_b");
    const int T_ref = (int)ld.get_u32("T_ref");
    const int rem_a_len = (int)ld.get_u32("rem_a_len");
    const int rem_b_len = (int)ld.get_u32("rem_b_len");

    std::vector<int32_t> prefill_a = tensor_i32(ld, "prefill_ids_a");
    std::vector<int32_t> prefill_b = tensor_i32(ld, "prefill_ids_b");
    std::vector<int32_t> ref = tensor_i32(ld, "reference_codes");  // (T_ref, 16)

    std::vector<int32_t> rem_a = tensor_i32(ld, "remaining_text_a");
    std::vector<int32_t> rem_b = tensor_i32(ld, "remaining_text_b");
    // The dumper writes a 1-element placeholder when the stream is empty.
    rem_a.resize(rem_a_len);
    rem_b.resize(rem_b_len);

    if (prefill_a.empty() || prefill_b.empty() || ref.empty()) {
        std::fprintf(stderr, "prompt-rt: missing fixture tensors\n");
        return 1;
    }

    moss::PromptRtOpts opts;

    int failures = 0;

    // Case (a): no reference.
    {
        moss::RtPrompt pr = moss::build_generation_prompt_rt(tok, text, {}, 0, opts);
        failures += compare_prefill("no-ref", pr.prefill_ids, pr.S, prefill_a, S_a);
        failures += compare_remaining("no-ref", pr.remaining_text, rem_a);
    }

    // Case (b): with reference.
    {
        moss::RtPrompt pr = moss::build_generation_prompt_rt(tok, text, ref, T_ref, opts);
        failures += compare_prefill("ref", pr.prefill_ids, pr.S, prefill_b, S_b);
        failures += compare_remaining("ref", pr.remaining_text, rem_b);
    }

    if (failures == 0) {
        std::fprintf(stderr, "prompt-rt: OK (both cases exact match)\n");
    }
    return failures == 0 ? 0 : 1;
}
