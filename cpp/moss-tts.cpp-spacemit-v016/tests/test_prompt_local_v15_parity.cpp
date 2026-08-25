// EXACT text-channel parity for build_generation_prompt_local under the v1.5
// PIECE-WISE processor (opts.piece_wise = true) vs a golden dump produced by the
// REAL MossTTSLocal v1.5 processor (processing_moss_tts.py
// _build_generation_or_voice_clone_codes).
//
// The real processor concatenates SEPARATELY-encoded id pieces (each
// tokenizer.encode adds no special tokens), so BPE merges never span a piece
// boundary. The old single-big-string encode drifted ~2 tokens near the text
// boundary (e.g. "test.\n" -> one token 624 instead of "." 13 + "\n" 198), which
// broke the model's stop head and caused generation to never terminate. This
// test locks the piece boundaries by asserting channel 0 EXACTLY equals the
// golden ids for three texts.
//
// Uses the REAL v1.5 tokenizer GGUF (env MOSS_TOKENIZER_V15, default
// models/moss-tokenizer-v1_5.gguf). Returns 77 (skip) when it is absent.
#include "prompt_local.hpp"
#include "de_tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct GoldCase {
    const char* text;
    std::vector<int32_t> ids;  // expected channel-0 (text) ids, in order
};

// Golden text-channel ids from the real v1.5 processor (n_vq=12). See
// /tmp/gold_prompts.json (source of truth for these arrays).
const std::vector<GoldCase>& golden() {
    static const std::vector<GoldCase> g = {
        {"Hello, this is a test.",
         {151644, 872, 198, 27, 872, 17740, 397, 12, 17207, 1141, 982, 4064, 198,
          12, 29051, 510, 4064, 198, 12, 58166, 510, 4064, 198, 12, 17927, 510,
          4064, 198, 12, 14594, 3665, 510, 4064, 198, 12, 92179, 14594, 510, 4064,
          198, 12, 11434, 510, 4064, 198, 12, 2918, 510, 9707, 11, 419, 374, 264,
          1273, 13, 198, 522, 872, 17740, 29, 151645, 198, 151644, 77091, 198,
          151669}},
        {"The quick brown fox.",
         {151644, 872, 198, 27, 872, 17740, 397, 12, 17207, 1141, 982, 4064, 198,
          12, 29051, 510, 4064, 198, 12, 58166, 510, 4064, 198, 12, 17927, 510,
          4064, 198, 12, 14594, 3665, 510, 4064, 198, 12, 92179, 14594, 510, 4064,
          198, 12, 11434, 510, 4064, 198, 12, 2918, 510, 785, 3974, 13876, 38835,
          13, 198, 522, 872, 17740, 29, 151645, 198, 151644, 77091, 198, 151669}},
        {"One two three.",
         {151644, 872, 198, 27, 872, 17740, 397, 12, 17207, 1141, 982, 4064, 198,
          12, 29051, 510, 4064, 198, 12, 58166, 510, 4064, 198, 12, 17927, 510,
          4064, 198, 12, 14594, 3665, 510, 4064, 198, 12, 92179, 14594, 510, 4064,
          198, 12, 11434, 510, 4064, 198, 12, 2918, 510, 3966, 1378, 2326, 13, 198,
          522, 872, 17740, 29, 151645, 198, 151644, 77091, 198, 151669}},
    };
    return g;
}

}  // namespace

int main() {
    const char* tok_env = std::getenv("MOSS_TOKENIZER_V15");
    std::string tok_path = tok_env ? tok_env : "models/moss-tokenizer-v1_5.gguf";
    moss::DeTokenizer tok;
    if (!tok.load_from_file(tok_path)) {
        std::fprintf(stderr, "prompt-local-v15-parity: v1.5 tokenizer load failed: %s\n",
                     tok_path.c_str());
        return 77;  // skip when the v1.5 tokenizer GGUF is absent
    }

    // v1.5 opts: piece-wise processor, n_vq=12, shifted audio ids.
    moss::PromptLocalOpts po;
    po.piece_wise     = true;
    po.n_vq           = 12;
    po.im_start       = 151644;
    po.im_end         = 151645;
    po.audio_start    = 151669;
    po.audio_end      = 151670;
    po.user_slot      = 151654;
    po.audio_pad_code = 1024;

    const int row = 1 + po.n_vq;
    int failures = 0;

    for (const auto& gc : golden()) {
        int S = 0;
        std::vector<int32_t> ids =
            moss::build_generation_prompt_local(tok, gc.text, {}, 0, po, &S);

        if ((int)ids.size() != S * row) {
            std::fprintf(stderr, "prompt-local-v15-parity[%s]: size %zu != S*row %d\n",
                         gc.text, ids.size(), S * row);
            ++failures;
            continue;
        }
        if (S != (int)gc.ids.size()) {
            std::fprintf(stderr,
                         "prompt-local-v15-parity[%s]: S mismatch got=%d want=%zu\n",
                         gc.text, S, gc.ids.size());
            ++failures;
            // fall through to report the first differing id up to the min length.
        }
        const int n = (int)std::min<size_t>((size_t)S, gc.ids.size());
        int diff = -1;
        for (int s = 0; s < n; ++s) {
            const int32_t got = ids[(size_t)s * row + 0];  // channel 0 = text
            if (got != gc.ids[s]) { diff = s; break; }
        }
        if (diff >= 0) {
            std::fprintf(stderr,
                         "prompt-local-v15-parity[%s]: MISMATCH at index %d: ours=%d golden=%d\n",
                         gc.text, diff, ids[(size_t)diff * row + 0], gc.ids[diff]);
            ++failures;
        } else if (S == (int)gc.ids.size()) {
            std::fprintf(stderr, "prompt-local-v15-parity[%s]: OK (%d ids exact)\n",
                         gc.text, S);
        }
    }

    if (failures == 0)
        std::fprintf(stderr, "prompt-local-v15-parity: OK (all 3 texts exact)\n");
    return failures == 0 ? 0 : 1;
}
