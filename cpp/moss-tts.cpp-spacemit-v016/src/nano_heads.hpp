#ifndef MOSS_NANO_HEADS_HPP
#define MOSS_NANO_HEADS_HPP
#include "model_loader.hpp"
#include <vector>
namespace moss {
class NanoHeads {
public:
    bool load(const ModelLoader& m);   // nano.head.text + nano.head.audio.{0..15}
    int  hidden() const { return hidden_; }
    int  text_vocab() const { return text_rows_; }
    int  audio_vocab() const { return audio_rows_; }   // 1024
    int  n_audio() const { return (int)audio_.size(); } // 16
    void text_logits(const std::vector<float>& h, std::vector<float>* out) const;
    void audio_logits(int c, const std::vector<float>& h, std::vector<float>* out) const;
private:
    int hidden_=0, text_rows_=0, audio_rows_=0;
    std::vector<float> text_; std::vector<std::vector<float>> audio_;
};
}  // namespace moss
#endif
