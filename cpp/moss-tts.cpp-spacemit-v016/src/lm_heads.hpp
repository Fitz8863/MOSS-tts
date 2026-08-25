#ifndef MOSS_LM_HEADS_HPP
#define MOSS_LM_HEADS_HPP

// LMHeads: the text head + N_VQ audio heads of the MossTTSDelay decoder.
// Port of lm_heads.py. Given the backbone hidden state [hidden]:
//
//   text_logits     = hidden @ text_w^T              -> [text_vocab]
//   audio_logits[i] = hidden @ audio_w[i]^T          -> [audio_vocab]   (i in 0..n_vq-1)
//
// then the AUDIO_PAD_CODE slot of each audio head is masked to -inf so it is
// never sampled. The weight tables are torch (rows, hidden) -> ggml ne0=hidden,
// ne1=rows, so a head's logit for output row r = dot(hidden, table + r*hidden).
// These are plain CPU dot products over the loader tensors' ->data; no ggml
// graph is needed.

#include "model_loader.hpp"
#include <vector>

namespace moss {

class LMHeads {
public:
    // Load de.lm_head_text.weight + de.lm_head_audio.{0..} (until missing or N_VQ).
    bool load(const ModelLoader& m);

    int  hidden() const { return hidden_; }
    int  text_vocab() const { return text_vocab_; }
    int  audio_vocab() const { return audio_vocab_; }   // 1025 for the real model
    int  n_vq() const { return n_vq_; }

    // hidden[hidden] -> text_logits[text_vocab] (resized),
    // audio_logits[n_vq*audio_vocab] (resized, row-major, pad slot masked to -inf).
    void logits(const std::vector<float>& hidden,
                std::vector<float>* text_logits,
                std::vector<float>* audio_logits) const;

private:
    int hidden_=0, text_vocab_=0, audio_vocab_=0, n_vq_=0;
    std::vector<float> text_w_;
    std::vector<std::vector<float>> audio_w_;
};

}  // namespace moss

#endif  // MOSS_LM_HEADS_HPP
