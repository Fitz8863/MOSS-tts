// Exact parity for build_generation_prompt_nano vs an upstream-faithful python
// re-derivation of build_voice_clone_request_rows
// (/tmp/moss-nano-inspect/ort_cpu_runtime.py). The fixture
// tests/fixtures/prompt_nano.gguf (produced by
// scripts/gen_test_fixtures.py prompt_nano) dumps two cases:
//   (clone)   T_ref=2, 2x16 reference codes,
//   (noclone) T_ref=0 (text only, no reference-audio block),
// each with the resulting prefill_ids (S, 17) int32 + remaining_text (1-D i32).
// We rebuild the prompt in C++ with the tiny SpTokenizer (sp_tiny.gguf); both
// sides tokenize the REAL MOSS-TTS-Nano prompt template TEXT (verbatim from
// upstream prompting.py) through the same vocab, then assert prefill_ids match
// EXACTLY across channel 0 (text) and all 16 audio channels, plus remaining_text + S.
//
// Skips (77) when either fixture is absent.
#include "prompt_nano.hpp"
#include "sp_tokenizer.hpp"
#include "nano_constants.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using moss::nano::CHANNELS;

static std::vector<int32_t> tensor_i32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<int32_t> v; moss::read_tensor_i32(t, &v);
    return v;
}

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
    const char* fx_env = std::getenv("MOSS_FIXTURE_PROMPT_NANO");
    std::string fx_path = fx_env ? fx_env : "tests/fixtures/prompt_nano.gguf";

    moss::ModelLoader ld;
    if (!ld.load(fx_path)) {
        std::fprintf(stderr, "prompt-nano: load failed: %s\n", fx_path.c_str());
        return 77;  // skip when fixture absent
    }

    const char* tok_env = std::getenv("MOSS_SP_TOKENIZER");
    std::string tok_path = tok_env ? tok_env : "tests/fixtures/sp_tiny.gguf";
    moss::SpTokenizer tok;
    if (!tok.load_from_file(tok_path)) {
        std::fprintf(stderr, "prompt-nano: tokenizer load failed: %s\n", tok_path.c_str());
        return 77;  // skip when tokenizer fixture absent
    }

    const int channels = (int)ld.get_u32("channels");
    if (channels != CHANNELS) {
        std::fprintf(stderr, "prompt-nano: channels mismatch fixture=%d CHANNELS=%d\n",
                     channels, CHANNELS);
        return 1;
    }
    const std::string text = ld.get_str("text");
    const int S_clone = (int)ld.get_u32("S_clone");
    const int S_noclone = (int)ld.get_u32("S_noclone");
    const int T_ref = (int)ld.get_u32("T_ref");
    const int rem_clone_len = (int)ld.get_u32("rem_clone_len");
    const int rem_noclone_len = (int)ld.get_u32("rem_noclone_len");

    std::vector<int32_t> prefill_clone = tensor_i32(ld, "prefill_ids_clone");
    std::vector<int32_t> prefill_noclone = tensor_i32(ld, "prefill_ids_noclone");
    std::vector<int32_t> ref = tensor_i32(ld, "reference_codes");  // (T_ref, 16)

    std::vector<int32_t> rem_clone = tensor_i32(ld, "remaining_clone");
    std::vector<int32_t> rem_noclone = tensor_i32(ld, "remaining_noclone");
    rem_clone.resize(rem_clone_len);
    rem_noclone.resize(rem_noclone_len);

    if (prefill_clone.empty() || prefill_noclone.empty()) {
        std::fprintf(stderr, "prompt-nano: missing fixture tensors\n");
        return 1;
    }

    moss::NanoPromptOpts opts;

    int failures = 0;

    // Clone case: T_ref=2 reference frames.
    {
        moss::NanoPrompt pr = moss::build_generation_prompt_nano(tok, text, ref, T_ref, opts);
        failures += compare_prefill("clone", pr.prefill_ids, pr.S, prefill_clone, S_clone);
        failures += compare_remaining("clone", pr.remaining_text, rem_clone);
    }

    // No-clone case: T_ref=0.
    {
        moss::NanoPrompt pr = moss::build_generation_prompt_nano(tok, text, {}, 0, opts);
        failures += compare_prefill("noclone", pr.prefill_ids, pr.S, prefill_noclone, S_noclone);
        failures += compare_remaining("noclone", pr.remaining_text, rem_noclone);
    }

    if (failures == 0) {
        std::fprintf(stderr, "prompt-nano: OK (both clone + no-clone exact match)\n");
    }
    return failures == 0 ? 0 : 1;
}
