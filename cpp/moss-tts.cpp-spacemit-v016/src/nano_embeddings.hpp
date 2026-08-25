#ifndef MOSS_NANO_EMBEDDINGS_HPP
#define MOSS_NANO_EMBEDDINGS_HPP

// NanoEmbeddings: the summed input embeddings of the MossTTSNano model.
//
//   nano.embed.0.weight       — text table (TEXT_VOCAB × hidden)
//   nano.embed.{1..16}.weight — 16 audio tables (AUDIO_VOCAB × hidden each;
//                               NO pad row — valid codes 0..AUDIO_VOCAB-1).
//
// embed_sum sums, per 17-wide row, text[id0] plus each audio_c[code] over the
// 16 audio channels, EXCEPT audio codes equal to audio_pad (the masked
// audio_pad_token_id), which contribute a ZERO embedding (they are skipped and
// never gathered — gathering the pad id would be out of range). This mirrors
// the real model, which masks audio_pad_token_id to a zero embedding.
//
// Tables stored torch (rows, hidden) → ggml ne0=hidden, ne1=rows; row r =
// data + r*hidden. All ops are pure CPU row gathers over the loader tensors'
// ->data, every index guarded by the stored row counts. Mirrors V3's
// RtEmbeddings.

#include "model_loader.hpp"
#include <vector>

namespace moss {

class NanoEmbeddings {
public:
    // Load nano.embed.0 (text) + nano.embed.{1..16} (audio) and the
    // nano.audio_pad sentinel. Returns false if the text table is missing.
    bool load(const ModelLoader& m);

    int  hidden() const { return hidden_; }
    int  channels() const { return channels_; }   // 17 (1 text + 16 audio)

    // Sum over channels: ids row-major (S*17) -> out (S*hidden), out resized.
    // ids[r*17+0] = text id; ids[r*17+1+c] = audio code for channel c (an id ==
    // audio_pad contributes zero / is skipped).
    void embed_sum(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const;

    // Single audio table gather: channel c (0..15), code -> out (hidden), resized.
    void embed_audio_one(int c, int code, std::vector<float>* out) const;

    // Single text table gather: id -> out (hidden), resized.
    void embed_text_one(int id, std::vector<float>* out) const;

private:
    int hidden_=0, channels_=0, audio_pad_=0;
    std::vector<float> text_; int text_rows_=0;
    std::vector<std::vector<float>> audio_; std::vector<int> arows_;
};

}  // namespace moss

#endif  // MOSS_NANO_EMBEDDINGS_HPP
