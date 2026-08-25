#ifndef MOSS_LOCAL_ADAPTERS_HPP
#define MOSS_LOCAL_ADAPTERS_HPP

// LocalAdapters: the two per-step adapter ops of the MossTTSLocal depth loop.
//
//   - to_local: the SHARED speech_embedding_to_local_mlp = MossTTSMLP that maps
//     the global hidden vector (hidden) -> local hidden vector (local_hidden).
//     Tensor names lc.in_mlp.{gate,up,down}.weight.
//
//   - head_logits: the per-channel output path
//       lm_heads[c]( layer_norm_before_lm_heads[c]( local_to_speech_embedding_mlps[c](local_out) ) )
//     i.e. out-MLP (local_hidden -> hidden) -> RMSNorm*head_norm[c] -> lm_head[c].
//     For audio channels (c != 0) the pad slot is masked to -inf, using the
//     same convention as V1 LMHeads (real model: AUDIO_PAD_CODE; tiny fixture:
//     last slot). Names lc.out_mlp.{c}.{gate,up,down}.weight,
//     lc.head_norm.{c}.weight, lc.lm_head.{c}.weight.
//
// Channel 0 is the text vocab head; channels 1..N_VQ are the 1025-wide audio
// heads. Both ops run a tiny ggml graph (moss_mlp + rms_norm + matmul).
//
// BARE v1.5 mode (GGUF metadata lc.bare_heads==1): the heads are BARE tied
// linears applied directly to the local hidden (2560-dim), with NO in-MLP /
// out-MLP / head_norm adapters. In this mode:
//   - to_local is the IDENTITY (local_hidden == hidden, just a copy).
//   - head_logits(c) is a direct matmul lc.lm_head.{c} . local_out for the
//     audio channels c in 1..N_VQ (1024-wide, all codes valid, NO pad mask;
//     the pad code is handled at the embedding, never as a logit).
//   - Channel 0 has NO full text-vocab head; the binary continue/stop decision
//     is served by local_text_head_logits (lc.local_text_head, 2-wide) instead.
// The Delay (v1.0) composite path above stays byte-identical when bare_heads==0.

#include "moss_tts_mlp.hpp"
#include "model_loader.hpp"
#include <cstdint>
#include <vector>

namespace moss {

class LocalAdapters {
public:
    bool load(const ModelLoader& m);   // lc.in_mlp.*, lc.out_mlp.{i}.*, lc.head_norm.{i}.weight, lc.lm_head.{i}.weight
    int  channels() const { return channels_; }
    int  hidden() const { return hidden_; }
    int  local_hidden() const { return local_hidden_; }
    int  text_vocab() const { return text_vocab_; }
    int  audio_vocab() const { return audio_vocab_; }
    float rms_eps() const { return 1e-6f; }
    // shared in-MLP: hidden_vec[hidden] -> *out[local_hidden] (resized).
    // (bare v1.5 mode: identity copy, no MLP.)
    void to_local(const std::vector<float>& hidden_vec, std::vector<float>* out) const;
    // per-channel out path: local_out[local_hidden] -> *logits (text_vocab if channel==0 else audio_vocab); pad-masked for channel!=0. resized.
    // (bare v1.5 mode: direct matmul lc.lm_head.{channel} for audio channels 1..N_VQ, NO out-MLP/head_norm, NO pad mask; channel 0 invalid here — use local_text_head_logits.)
    void head_logits(int channel, const std::vector<float>& local_out, std::vector<float>* logits) const;

    // v1.5 binary channel-0 decision head: a DIRECT 2-wide Linear(local_hidden, 2)
    // on the local hidden (NO out_mlp / head_norm), giving {continue, stop} logits.
    // Only valid when has_local_text_head() (tensor lc.local_text_head.weight present).
    bool has_local_text_head() const { return local_text_head_ != nullptr; }
    void local_text_head_logits(const std::vector<float>& local_out, std::vector<float>* logits) const;
private:
    int channels_=0,hidden_=0,local_hidden_=0,text_vocab_=0,audio_vocab_=0;
    bool bare_heads_ = false;   // v1.5: bare tied heads (no in/out MLP, no head_norm, no pad mask)
    MossMLPWeights in_mlp_; std::vector<MossMLPWeights> out_mlp_;
    std::vector<int> head_rows_;
    struct ggml_tensor* local_text_head_=nullptr;   // v1.5 binary decision head (nullptr for v1.0)
    const ModelLoader* m_=nullptr;
    mutable std::vector<uint8_t> to_local_scratch_;   // reused ctx buffer for to_local
    mutable std::vector<uint8_t> head_scratch_;       // reused ctx buffer for head_logits
};

}  // namespace moss

#endif  // MOSS_LOCAL_ADAPTERS_HPP
