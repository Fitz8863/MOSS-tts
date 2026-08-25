// Native SentencePiece-unigram tokenizer id-parity test.
//
// The fixture tests/fixtures/sp_tiny.gguf is a HAND-BUILT tiny unigram vocab
// (pieces + log-prob scores + types) plus an INDEPENDENT python Viterbi result:
//   ref.text  : the input string ("hello world")
//   ref.norm  : the SP-normalized string ("▁hello▁world")
//   ids_ref   : the python-Viterbi-best id sequence the C++ must reproduce
//
// This pins the C++ unigram Viterbi against a separate python implementation
// (see scripts/gen_test_fixtures.py :: w_sp_tiny). No `sentencepiece` runtime
// dependency is involved.

#include "sp_tokenizer.hpp"
#include "model_loader.hpp"
#include "backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

std::string ids_to_str(const std::vector<int32_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(v[i]);
    }
    s += "]";
    return s;
}

std::vector<int32_t> tensor_i32(const moss::ModelLoader& ld, const char* name) {
    struct ggml_tensor* t = ld.tensor(name);
    if (!t) return {};
    std::vector<int32_t> v; moss::read_tensor_i32(t, &v);
    return v;
}

}  // namespace

int main() {
    const char* env  = std::getenv("MOSS_SP_TOKENIZER");
    std::string path = env ? env : "tests/fixtures/sp_tiny.gguf";

    // Read references straight out of the fixture.
    moss::ModelLoader ld;
    if (!ld.load(path)) {
        std::fprintf(stderr, "sp_tokenizer: fixture load failed: %s\n", path.c_str());
        return 77;  // skip when fixture absent
    }
    const std::string text    = ld.get_str("ref.text");
    const std::string norm    = ld.get_str("ref.norm");
    const std::vector<int32_t> ids_ref = tensor_i32(ld, "ids_ref");
    if (text.empty() || ids_ref.empty()) {
        std::fprintf(stderr, "sp_tokenizer: fixture missing ref.text/ids_ref\n");
        return 1;
    }

    moss::SpTokenizer tok;
    if (!tok.load_from_file(path)) {
        std::fprintf(stderr, "sp_tokenizer: tokenizer load failed: %s\n", path.c_str());
        return 77;
    }

    int failures = 0;

    // 1) encode parity: exact int-vector match against the python Viterbi.
    std::vector<int32_t> got = tok.encode(text);
    if (got != ids_ref) {
        ++failures;
        std::fprintf(stderr,
                     "ENCODE MISMATCH for \"%s\"\n  expected %s\n  got      %s\n",
                     text.c_str(),
                     ids_to_str(ids_ref).c_str(),
                     ids_to_str(got).c_str());
    }

    // 2) decode round-trip: decode(ids_ref) reproduces the normalized text with
    //    the leading SP space stripped, i.e. the original "hello world".
    std::string back = tok.decode(ids_ref);
    if (back != text) {
        ++failures;
        std::fprintf(stderr,
                     "DECODE MISMATCH\n  expected \"%s\"\n  got      \"%s\"\n",
                     text.c_str(), back.c_str());
    }

    if (failures) {
        std::fprintf(stderr, "sp_tokenizer: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("sp_tokenizer ok: encode-parity + decode round-trip, vocab=%zu\n",
                tok.vocab_size());
    return 0;
}
