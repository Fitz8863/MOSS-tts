#ifndef MOSS_RT_EMBEDDINGS_HPP
#define MOSS_RT_EMBEDDINGS_HPP

// RtEmbeddings: the embedding tables of the MossTTSRealtime model.
//
//   GLOBAL: rt.embed.{0..16}.weight — 17 tables (i=0 text vocab×hidden,
//   i=1..16 audio 1027×hidden). embed_sum sums, per position, over all 17
//   channels of table_i[ids[s*17+i]] (the global multi-modal input).
//   LOCAL: rtl.embed.{0..14}.weight — 15 tables, the depth transformer's
//   per-codebook input embeds. embed_local_one re-embeds a generated code at
//   depth j+1.
//
// Tables stored torch (rows, hidden) → ggml ne0=hidden, ne1=rows; row r =
// data + r*hidden. Both ops are pure CPU row gathers over the loader tensors'
// ->data. Mirrors V2's LocalEmbeddings.

#include "model_loader.hpp"
#include <vector>

namespace moss {

class RtEmbeddings {
public:
    // Load rt.embed.{0..16} (global) + rtl.embed.{0..14} (local). Returns
    // false if no global tables are present.
    bool load(const ModelLoader& m);

    int  hidden() const { return hidden_; }
    int  channels() const { return channels_; }         // 17
    int  n_local() const { return (int)local_.size(); } // 15

    // GLOBAL sum over channels: ids row-major (S*channels) -> out (S*hidden),
    // out resized.
    void embed_sum(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const;

    // LOCAL single-code: local table j (0..14), code -> append hidden floats
    // to *out (out is NOT cleared; appends).
    void embed_local_one(int j, int code, std::vector<float>* out) const;

private:
    int hidden_=0, channels_=0;
    std::vector<std::vector<float>> global_; std::vector<int> grows_;
    std::vector<std::vector<float>> local_; std::vector<int> lrows_;
};

}  // namespace moss

#endif  // MOSS_RT_EMBEDDINGS_HPP
