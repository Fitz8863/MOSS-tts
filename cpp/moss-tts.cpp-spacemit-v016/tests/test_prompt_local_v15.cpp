// Structural test for build_generation_prompt_local under the v1.5 parameters:
// a dynamic n_vq (audio-channel count) and metadata-driven Local special-token
// ids (audio_start/end/user_slot/im_start/im_end/audio_pad_code) supplied via
// PromptLocalOpts instead of the hardcoded v1.0 `de::` constants.
//
// Unlike test_prompt_local (exact parity vs a python dump), this needs no real
// model: it asserts STRUCTURE only -- row width 1+n_vq, the generation-trigger
// row, and the audio-pad fill -- with tiny synthetic token ids. It still uses
// the real Qwen3 tokenizer (its `token(id)` -> string only builds the text; the
// no-reference path never rescans the encoded ids). Skips (77) when the
// tokenizer fixture is absent. MOSS_DE_TOKENIZER overrides the path.
#include "prompt_local.hpp"
#include "de_tokenizer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* tok_env = std::getenv("MOSS_DE_TOKENIZER");
    std::string tok_path = tok_env ? tok_env : "tests/fixtures/de_tokenizer.gguf";
    moss::DeTokenizer tok;
    if (!tok.load_from_file(tok_path)) {
        std::fprintf(stderr, "prompt-local-v15: tokenizer load failed: %s\n", tok_path.c_str());
        return 77;  // skip when tokenizer fixture absent
    }

    // v1.5-style params: n_vq=4 (row width 5) + tiny synthetic special-token ids.
    // These differ from the v1.0 de:: constants, proving the builder is fully
    // parameterized (no hardcoded N_VQ / AUDIO_* left).
    const int n_vq           = 4;
    const int audio_start    = 900;
    const int audio_end      = 901;
    const int user_slot      = 902;
    const int im_start       = 904;
    const int im_end         = 905;
    const int audio_pad_code = 5;

    moss::PromptLocalOpts po;
    po.n_vq           = n_vq;
    po.audio_start    = audio_start;
    po.audio_end      = audio_end;
    po.user_slot      = user_slot;
    po.im_start       = im_start;
    po.im_end         = im_end;
    po.audio_pad_code = audio_pad_code;
    po.language       = "English";

    const std::string text = "hello world";
    const int row = 1 + n_vq;

    int failures = 0;

    // ---- No-reference prompt: every audio channel must be audio_pad_code, and
    // the final row is the generation trigger (channel 0 == audio_start, audio
    // channels == audio_pad_code).
    {
        int S = 0;
        std::vector<int32_t> ids =
            moss::build_generation_prompt_local(tok, text, {}, 0, po, &S);

        if (S <= 0 || ids.empty()) {
            std::fprintf(stderr, "prompt-local-v15: empty prompt (S=%d)\n", S);
            return 1;
        }
        // Row width = 1 + n_vq.
        if ((int)ids.size() != S * row) {
            std::fprintf(stderr,
                         "prompt-local-v15: size mismatch got=%zu want=%d (S=%d row=%d)\n",
                         ids.size(), S * row, S, row);
            ++failures;
        }
        // All audio channels across every row must be the pad code (no ref).
        for (int s = 0; s < S && failures == 0; ++s) {
            for (int c = 1; c < row; ++c) {
                if (ids[(size_t)s * row + c] != audio_pad_code) {
                    std::fprintf(stderr,
                                 "prompt-local-v15: audio pad mismatch at pos=%d ch=%d got=%d want=%d\n",
                                 s, c, ids[(size_t)s * row + c], audio_pad_code);
                    ++failures;
                    break;
                }
            }
        }
        // Final row is the generation trigger.
        const int rr = S - 1;
        if (ids[(size_t)rr * row + 0] != audio_start) {
            std::fprintf(stderr,
                         "prompt-local-v15: gen-row channel0 got=%d want=%d\n",
                         ids[(size_t)rr * row + 0], audio_start);
            ++failures;
        }
        for (int c = 1; c < row; ++c) {
            if (ids[(size_t)rr * row + c] != audio_pad_code) {
                std::fprintf(stderr,
                             "prompt-local-v15: gen-row audio ch=%d got=%d want=%d\n",
                             c, ids[(size_t)rr * row + c], audio_pad_code);
                ++failures;
                break;
            }
        }
    }

    if (failures == 0) {
        std::fprintf(stderr, "prompt-local-v15: OK (n_vq=%d, dynamic token ids)\n", n_vq);
    }
    return failures == 0 ? 0 : 1;
}
