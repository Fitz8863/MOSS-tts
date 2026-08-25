#ifndef MOSS_NANO_LOCAL_HPP
#define MOSS_NANO_LOCAL_HPP
#include "gpt2.hpp"
#include "model_loader.hpp"
#include <cstdint>
#include <vector>
namespace moss {
class NanoLocal {
public:
    bool load(const ModelLoader& m);   // gptl.* metadata + gptl.blk.{i}.* + gptl.output_norm.{w,b}
    int  hidden() const { return hp_.hidden; }
    void reset();                       // clear the per-frame KV cache (call before each frame's depth loop)
    bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);
private:
    Gpt2Hparams hp_{}; std::vector<Gpt2Layer> layers_;
    struct ggml_tensor *out_norm_w_=nullptr, *out_norm_b_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0;
    std::vector<std::vector<float>> k_state_, v_state_;
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
};
}  // namespace moss
#endif
