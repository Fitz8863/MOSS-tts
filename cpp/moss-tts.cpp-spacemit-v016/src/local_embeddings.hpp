#ifndef MOSS_LOCAL_EMBEDDINGS_HPP
#define MOSS_LOCAL_EMBEDDINGS_HPP

// LocalEmbeddings: the embedding_list lookup used by the MossTTSLocal model.
//
//   channels = 1 (text) + n_vq (audio). embedding_list[i] tables are
//   lc.embed.{i}.weight (ggml ne0=hidden, ne1=rows; row r = data + r*hidden).
//
// Two ops, both pure CPU row gathers over the loader tensors' ->data:
//   - embed_sum: per s, sum over ALL channels (including text at i=0) of
//     table_i[ids[s*channels+i]]  (the global multi-modal input). For v1.5,
//     an audio pad code (lc.audio_pad_code, default 1024) that is out of range
//     for its table (rows==pad) contributes zero and is skipped; Delay tables
//     have an in-range pad row and are unaffected.
//   - embed_one: a single table row (the local re-embed of a just-generated
//     code).
//
// Mirrors V1's DelayEmbeddings, but embed_sum sums every channel (text too)
// and embed_one exposes single-row gather for the depth loop.

#include "model_loader.hpp"
#include <vector>

namespace moss {

class LocalEmbeddings {
public:
    // Load lc.embed.{0..} until missing (cap 1 + de::N_VQ). Returns false if
    // no tables are present.
    bool load(const ModelLoader& m);

    int  hidden() const { return hidden_; }
    int  channels() const { return channels_; }   // 1 + n_vq

    // sum over channels: ids row-major (S*channels) -> out (S*hidden),
    // out resized.
    void embed_sum(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const;

    // single channel i, single code -> append hidden floats to *out
    // (out is NOT cleared; appends).
    void embed_one(int channel, int code, std::vector<float>* out) const;

private:
    int hidden_=0, channels_=0;
    int pad_code_ = -1;
    std::vector<std::vector<float>> tables_;
    std::vector<int> rows_;
};

}  // namespace moss

#endif  // MOSS_LOCAL_EMBEDDINGS_HPP
