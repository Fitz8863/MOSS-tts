#ifndef MOSS_DELAY_BACKBONE_HPP
#define MOSS_DELAY_BACKBONE_HPP
#include "qwen3.hpp"
#include "model_loader.hpp"
#include "ggml_extend.hpp"   // GgmlCtxPtr
#include "ggml-backend.h"    // ggml_backend_buffer_t
#include <cstdint>
#include <vector>
namespace moss {
class DelayBackbone {
public:
    DelayBackbone() = default;
    ~DelayBackbone();
    DelayBackbone(const DelayBackbone&) = delete;             // owns a backend buffer
    DelayBackbone& operator=(const DelayBackbone&) = delete;
    bool load(const ModelLoader& m, int max_seq);   // reads qwen3.* metadata + N layers + qwen3.output_norm.weight
    int  hidden() const { return hp_.hidden; }
    const Qwen3Hparams& hparams() const { return hp_; }
    // Prefill S embedding rows (row-major S*hidden). Resets cache to pos 0. Returns the LAST row's hidden (post final RMSNorm).
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* last_hidden);
    // One step at the next position. embed: [hidden]. Returns hidden [hidden] (post final RMSNorm).
    bool decode_one(const std::vector<float>& embed, std::vector<float>* hidden);
    void reset();
    int  past_len() const { return past_len_; }
private:
    bool run(const std::vector<float>& embeds, int T, bool is_prefill, std::vector<float>* out_hidden);

    Qwen3Hparams hp_{}; std::vector<Qwen3Layer> layers_; struct ggml_tensor* output_norm_=nullptr;
    const ModelLoader* m_=nullptr; int max_seq_=0, past_len_=0;
    // Device-resident per-layer K/V cache: k_cache_[l]/v_cache_[l] are
    // [head_dim, n_kv_heads, max_seq, 1] tensors on kv_buffer_, written in place
    // each step and read by [0:past+T] view. kv_ctx_ holds their metadata.
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
};
}  // namespace moss
#endif
