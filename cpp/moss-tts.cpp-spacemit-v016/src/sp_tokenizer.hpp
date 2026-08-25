#ifndef MOSS_SP_TOKENIZER_HPP
#define MOSS_SP_TOKENIZER_HPP

// Native SentencePiece-unigram tokenizer (no `sentencepiece` runtime dep).
//
// Loads pieces + log-prob scores + types from a `tokenizer.gguf` produced by
// scripts/convert_tokenizer.py --sentencepiece (or the tiny hand-built
// sp_tiny.gguf fixture). Inference is unigram Viterbi segmentation over the
// SP-normalized byte string with byte-fallback (`<0xNN>`) / `<unk>` for spans
// no piece covers.
//
// Normalization is MINIMAL (whitespace-collapse + add_dummy_prefix + `▁`):
// the heavy NFKC rule set is NOT applied — this is an approximation that is
// exact for the ASCII fixture and adequate for typical Nano text. Full NFKC
// fidelity is deferred (flagged for T10/T17 real-model validation).

#include "model_loader.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace moss {

class SpTokenizer {
public:
    bool load(const ModelLoader& m);
    bool load_from_file(const std::string& path);

    // SP-normalize + unigram Viterbi -> ids.
    std::vector<int32_t> encode(const std::string& text) const;
    // ids -> UTF-8 text (▁->space, strip one leading space, <0xNN>->raw byte).
    std::string          decode(const std::vector<int32_t>& ids) const;

    int    unk_id() const { return unk_id_; }
    size_t vocab_size() const { return pieces_.size(); }

private:
    std::vector<std::string>                 pieces_;
    std::vector<float>                       scores_;
    std::vector<int32_t>                     types_;
    std::unordered_map<std::string, int32_t> piece_id_;
    int                                      unk_id_       = -1;
    size_t                                   max_piece_len_ = 1;
};

}  // namespace moss

#endif  // MOSS_SP_TOKENIZER_HPP
