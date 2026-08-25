#ifndef MOSS_LOCAL_TRANSFORMER_HPP
#define MOSS_LOCAL_TRANSFORMER_HPP

// LocalTransformer — the MossTTS depth ("local") transformer stack.
//
// The depth transformer runs over a PER-FRAME KV cache: reset() clears the
// cache once per frame, then step(in_vec, pos, &out) feeds one depth token at
// a time (O(channels) instead of O(channels^2)). The python re-runs
// local_transformer over the growing inputs each depth step; by causality the
// KV-cached step here is numerically identical.
//
// Dispatches on the `local.arch` metadata into two paths:
//   * `qwen3` (default/absent, the Delay model): mirrors rt_local (the V3 depth
//     transformer) but NO RoPE (use_rope=false), KEEPS per-head q/k RMSNorm, and
//     a weight-only final RMSNorm. Weights under `local.blk.{i}.*` +
//     `local.output_norm.weight`, dims from `local.*` metadata.
//   * `gptj` (v1.5, step_gptj): reuses the GPT-2 block (`src/gpt2.cpp`: fused
//     c_attn, silu MLP, interleaved RoPE with position = depth index) and a
//     final LayerNorm WITH bias (`local.output_norm.{weight,bias}`), dims from
//     `local.{hidden,n_head,head_dim,d_ff,n_layers,rope_base,ln_eps}`.
//
// NOTE: the real local layout has n_heads*head_dim != hidden (16*128=2048 vs
// hidden=1536). qwen3_layer_forward already keys all attention dims off
// head_dim/n_heads (residual/o_proj-output are hidden-wide), so the mismatch is
// handled without a special case.

#include "qwen3.hpp"
#include "gpt2.hpp"
#include "model_loader.hpp"
#include <cstdint>
#include <vector>

namespace moss {

class LocalTransformer {
public:
    // Loads local.* metadata + local.blk.{i}.* (prefix "local") +
    // local.output_norm.weight; forces use_rope=false. Returns false on a
    // missing tensor or invalid hparams.
    bool load(const ModelLoader& m);

    int hidden() const { return gptj_ ? ghp_.hidden : hp_.hidden; }

    // Clear the per-frame KV cache (call before each frame's depth loop).
    void reset();

    // Feed one token at depth position `pos` (== past_len_) over the per-frame
    // KV cache; returns the post-final-norm hidden [hidden]. `out_hidden` is
    // resized to hidden. NO RoPE. By causality this equals row `pos` of a full
    // no-RoPE causal forward over the prefix 0..pos.
    bool step(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);

private:
    // GPT-J (v1.5) depth-transformer path — dispatched when local.arch == "gptj".
    bool step_gptj(const std::vector<float>& in_vec, int pos, std::vector<float>* out_hidden);

    Qwen3Hparams hp_{};
    std::vector<Qwen3Layer> layers_;
    struct ggml_tensor* output_norm_ = nullptr;
    const ModelLoader* m_ = nullptr;
    int past_len_ = 0;
    std::vector<std::vector<float>> k_state_, v_state_;  // per-layer accumulated K/V
    std::vector<uint8_t> step_scratch_;                  // reused per-call ctx buffer

    // local.arch == "gptj" (v1.5): reuse the GPT-2 block with silu. Shares
    // k_state_/v_state_/step_scratch_/past_len_/output_norm_ with the qwen3 path.
    bool gptj_ = false;
    Gpt2Hparams ghp_{};
    std::vector<Gpt2Layer> glayers_;
    struct ggml_tensor* out_norm_b_ = nullptr;           // gptj final LayerNorm bias
};

}  // namespace moss

#endif  // MOSS_LOCAL_TRANSFORMER_HPP
