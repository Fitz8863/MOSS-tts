#ifndef MOSS_RT_HEADS_HPP
#define MOSS_RT_HEADS_HPP
#include "model_loader.hpp"
#include <vector>
namespace moss {
class RtHeads {
public:
    bool load(const ModelLoader& m);   // rtl.head.{0..15}.weight (16)
    int  n_heads() const { return (int)heads_.size(); }   // 16
    int  hidden() const { return hidden_; }
    int  audio_vocab() const { return audio_vocab_; }      // 1027
    // codebook i, normed hidden h[hidden] -> logits[audio_vocab] (resized).
    void logits(int i, const std::vector<float>& h, std::vector<float>* out) const;
private:
    int hidden_=0, audio_vocab_=0;
    std::vector<std::vector<float>> heads_; std::vector<int> rows_;
};
}  // namespace moss
#endif
