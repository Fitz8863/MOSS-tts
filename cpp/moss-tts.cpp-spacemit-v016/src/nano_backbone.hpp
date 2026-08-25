#ifndef MOSS_NANO_BACKBONE_HPP
#define MOSS_NANO_BACKBONE_HPP
#include "gpt2.hpp"
#include "model_loader.hpp"
#include "ggml_extend.hpp"   // GgmlCtxPtr
#include "ggml-backend.h"    // ggml_backend_buffer_t
#include <cstdint>
#include <vector>
namespace moss {
// Global backbone of MOSS-TTS-Nano: a 12-layer GPT-2 (interleaved-RoPE) stack
// over time with a persistent device-resident per-layer KV cache.
class NanoBackbone {
public:
    NanoBackbone() = default;
    ~NanoBackbone();
    NanoBackbone(const NanoBackbone&) = delete;             // owns a backend buffer
    NanoBackbone& operator=(const NanoBackbone&) = delete;
    bool load(const ModelLoader& m, int max_seq = 8192);  // reads gpt2.* metadata + N layers + gpt2.output_norm.{weight,bias}
    int  hidden() const { return hp_.hidden; }
    const Gpt2Hparams& hparams() const { return hp_; }
    void reset();
    int  past_len() const { return past_len_; }
    // Prefill S embedding rows ([hidden,S] row-major). Resets the cache to pos 0.
    // Returns the LAST row's hidden (post final LayerNorm).
    bool prefill(const std::vector<float>& embeds, int S, std::vector<float>* last_hidden);
    // One step at the next position. embed: [hidden]. Returns hidden [hidden]
    // (post final LayerNorm). Attends to all cached positions + this one.
    bool decode_one(const std::vector<float>& embed, std::vector<float>* hidden);
private:
    bool run(const std::vector<float>& embeds, int T, std::vector<float>* out_hidden);

    Gpt2Hparams hp_{}; std::vector<Gpt2Layer> layers_;
    struct ggml_tensor *out_norm_w_=nullptr, *out_norm_b_=nullptr;
    const ModelLoader* m_=nullptr; int max_seq_=0, past_len_=0;
    // Device-resident per-layer K/V cache: [head_dim, n_head, max_seq, 1] on
    // kv_buffer_; written in place each step, read by [0:past+T] view.
    GgmlCtxPtr kv_ctx_;
    ggml_backend_buffer_t kv_buffer_ = nullptr;
    std::vector<struct ggml_tensor*> k_cache_, v_cache_;
    std::vector<uint8_t> scratch_;   // reused per-step graph-metadata ctx buffer
};
}  // namespace moss
#endif
