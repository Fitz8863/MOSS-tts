#ifndef MOSS_RT_LOCAL_HPP
#define MOSS_RT_LOCAL_HPP
#include "qwen3.hpp"
#include "model_loader.hpp"
#include <cstdint>
#include <vector>
namespace moss {
class RtLocal {
public:
    bool load(const ModelLoader& m);   // rtl.* metadata + rtl.blk.{i}.* (prefix "rtl") + rtl.output_norm.weight; use_rope=true
    int  hidden() const { return hp_.hidden; }
    void reset();                       // clear the per-frame KV cache (call before each frame's depth loop)
    // feed one token at depth position `pos` (0..rvq-1); returns post-final-norm hidden [hidden]. out resized.
    bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);
private:
    Qwen3Hparams hp_{}; std::vector<Qwen3Layer> layers_; struct ggml_tensor* output_norm_=nullptr;
    const ModelLoader* m_=nullptr; int past_len_=0;
    std::vector<std::vector<float>> k_state_, v_state_;  // per-layer accumulated K/V
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
};
}  // namespace moss
#endif
