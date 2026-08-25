// Qwen3 byte-level BPE tokenizer id-parity test.
//
// The fixture tests/fixtures/de_tokenizer.gguf is the REAL OpenMOSS-Team/MOSS-TTS
// Qwen3 tokenizer.json converted via scripts/convert_tokenizer.py.
//
// The (text -> expected_ids) reference pairs below were computed offline with
// the HuggingFace `tokenizers` library on that SAME tokenizer.json:
//
//   from tokenizers import Tokenizer
//   t = Tokenizer.from_file("tokenizer.json")
//   t.encode(text, add_special_tokens=False).ids
//
// They exercise ASCII words, punctuation, a contraction ("it's"), digits, a
// multi-word sentence, and accented (non-ASCII) Latin.

#include "de_tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Case {
    std::string          text;
    std::vector<int32_t> ids;
};

std::string ids_to_str(const std::vector<int32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

}  // namespace

int main() {
    const char* env  = std::getenv("MOSS_DE_TOKENIZER");
    std::string path = env ? env : "tests/fixtures/de_tokenizer.gguf";

    moss::DeTokenizer tok;
    if (!tok.load_from_file(path)) {
        std::fprintf(stderr, "de_tokenizer: load failed: %s\n", path.c_str());
        return 77;  // skip when fixture absent
    }

    // Reference pairs (HF tokenizers, add_special_tokens=False).
    const std::vector<Case> cases = {
        {"Hello, world!",              {9707, 11, 1879, 0}},
        {"it's a test",               {275, 594, 264, 1273}},
        {"The year is 2024.",         {785, 1042, 374, 220, 17, 15, 17, 19, 13}},
        {"caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9", {924, 58858, 9333, 1242, 963}},
        {"Speak clearly and slowly.", {95845, 9355, 323, 13970, 13}},
    };

    int failures = 0;
    for (const auto& c : cases) {
        std::vector<int32_t> got = tok.encode(c.text);
        if (got != c.ids) {
            ++failures;
            std::fprintf(stderr,
                         "ENCODE MISMATCH for %s\n  expected %s\n  got      %s\n",
                         c.text.c_str(),
                         ids_to_str(c.ids).c_str(),
                         ids_to_str(got).c_str());
        }
    }

    // Round-trip: decode(encode(text)) reproduces the input exactly.
    const std::vector<std::string> rt = {"Hello, world!", "Speak clearly and slowly."};
    for (const auto& s : rt) {
        std::string back = tok.decode(tok.encode(s));
        if (back != s) {
            ++failures;
            std::fprintf(stderr, "ROUNDTRIP MISMATCH\n  expected \"%s\"\n  got      \"%s\"\n",
                         s.c_str(), back.c_str());
        }
    }

    if (failures) {
        std::fprintf(stderr, "de_tokenizer: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("de_tokenizer ok: %zu id-parity cases, vocab=%zu\n",
                cases.size(), tok.vocab_size());
    return 0;
}
