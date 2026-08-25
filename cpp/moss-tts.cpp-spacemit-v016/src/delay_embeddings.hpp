#ifndef MOSS_DELAY_EMBEDDINGS_HPP
#define MOSS_DELAY_EMBEDDINGS_HPP

// DelayEmbeddings: the 33-table sum-of-embeddings lookup used by the
// MossTTSDelay backbone input. Port of upstream embedding.py::_lookup:
//
//   embed[s] = embed_tokens[ids[s,0]] + Σ_{i=0..n_vq-1} emb_ext[i][ids[s,i+1]]
//
// These are pure row gathers (no matmul) over the loader tensors' ->data,
// so we keep it as straight C++ float adds — no ggml graph needed.

#include "model_loader.hpp"
#include <vector>

namespace moss {

class DelayEmbeddings {
public:
    // Load de.embed_tokens.weight + de.emb_ext.{0..} (until missing or N_VQ).
    bool load(const ModelLoader& m);

    int hidden() const { return hidden_; }
    int n_vq() const { return n_vq_; }

    // input_ids row-major (S*(1+n_vq)) int32; out row-major (S*hidden),
    // out resized.
    void embed(const std::vector<int32_t>& ids, int S, std::vector<float>* out) const;

private:
    int hidden_=0, n_vq_=0, text_vocab_=0, audio_vocab_=0;
    std::vector<float> text_;
    std::vector<std::vector<float>> audio_;
};

}  // namespace moss

#endif  // MOSS_DELAY_EMBEDDINGS_HPP
