#include "local_adapters.hpp"
#include "delay_constants.hpp"
#include "backend.hpp"
#include "ggml_extend.hpp"
#include "common.hpp"

#include <cassert>
#include <cmath>
#include <string>

namespace moss {

bool LocalAdapters::load(const ModelLoader& m) {
    m_ = &m;
    channels_ = 0; hidden_ = 0; local_hidden_ = 0; text_vocab_ = 0; audio_vocab_ = 0;
    out_mlp_.clear(); head_rows_.clear();

    bare_heads_ = m.get_u32("lc.bare_heads", 0) != 0;
    if (bare_heads_) {
        // v1.5 bare tied heads: audio channels 1..n_vq only (channel 0 = binary
        // head, handled separately). No in/out MLP, no head_norm, no pad mask.
        // head_rows_[0] is an unused placeholder so head_rows_[c] aligns with the
        // channel index c (audio channels start at 1).
        head_rows_.push_back(0);
        for (int c = 1; ; ++c) {
            struct ggml_tensor* lh = m.tensor("lc.lm_head." + std::to_string(c) + ".weight");
            if (!lh) break;
            if (hidden_ == 0) { hidden_ = (int)lh->ne[0]; local_hidden_ = hidden_; }
            head_rows_.push_back((int)lh->ne[1]);   // 1024 audio codes
            audio_vocab_ = (int)lh->ne[1];
            channels_ = c + 1;                        // highest channel index + 1
        }
        local_text_head_ = m.tensor("lc.local_text_head.weight");
        if (hidden_ == 0 || !local_text_head_) {
            MOSS_LOGE("LocalAdapters(bare): missing lc.lm_head.* or lc.local_text_head.weight");
            return false;
        }
        to_local_scratch_.resize(8 * 1024 * 1024);
        head_scratch_.resize(8 * 1024 * 1024);
        return true;
    }

    struct ggml_tensor* ig = m.tensor("lc.in_mlp.gate.weight");
    struct ggml_tensor* iu = m.tensor("lc.in_mlp.up.weight");
    struct ggml_tensor* id = m.tensor("lc.in_mlp.down.weight");
    if (!ig || !iu || !id) {
        MOSS_LOGE("LocalAdapters: missing lc.in_mlp.{gate,up,down}.weight");
        return false;
    }
    in_mlp_ = {ig, iu, id};
    hidden_       = (int)ig->ne[0];   // gate: ne0=in=hidden
    local_hidden_ = (int)id->ne[1];   // down: ne1=out=local_hidden

    for (int i = 0; i < 1 + de::N_VQ; ++i) {
        const std::string si = std::to_string(i);
        struct ggml_tensor* og = m.tensor("lc.out_mlp." + si + ".gate.weight");
        if (!og) break;
        struct ggml_tensor* ou = m.tensor("lc.out_mlp." + si + ".up.weight");
        struct ggml_tensor* od = m.tensor("lc.out_mlp." + si + ".down.weight");
        struct ggml_tensor* hn = m.tensor("lc.head_norm." + si + ".weight");
        struct ggml_tensor* lh = m.tensor("lc.lm_head." + si + ".weight");
        if (!ou || !od || !hn || !lh) {
            MOSS_LOGE("LocalAdapters: incomplete channel %d (out_mlp/head_norm/lm_head)", i);
            return false;
        }
        out_mlp_.push_back({og, ou, od});
        head_rows_.push_back((int)lh->ne[1]);
        if (i == 0) text_vocab_ = (int)lh->ne[1];
        else        audio_vocab_ = (int)lh->ne[1];
        ++channels_;
    }

    if (channels_ == 0) {
        MOSS_LOGE("LocalAdapters: no lc.out_mlp.* channels found");
        return false;
    }

    // v1.5 binary channel-0 decision head (optional). Absent in v1.0 GGUFs, in
    // which case this stays null and the full-vocab channel-0 path is used.
    local_text_head_ = m.tensor("lc.local_text_head.weight");

    // Persistent per-call scratch buffers (reused across to_local/head_logits
    // calls to avoid an 8 MiB malloc per depth-loop step). Sized once here.
    to_local_scratch_.resize(8 * 1024 * 1024);
    head_scratch_.resize(8 * 1024 * 1024);
    return true;
}

void LocalAdapters::to_local(const std::vector<float>& hidden_vec,
                             std::vector<float>* out) const {
    assert((int)hidden_vec.size() >= hidden_ && "hidden_vec too short");

    if (bare_heads_) {
        // Identity: local_hidden == hidden; the local transformer output is fed
        // straight to the tied heads.
        out->assign(hidden_vec.begin(), hidden_vec.begin() + local_hidden_);
        return;
    }
    assert(in_mlp_.gate && "LocalAdapters::to_local before load");

    auto ctx = make_ctx_buf(to_local_scratch_.data(), to_local_scratch_.size(), /*no_alloc=*/true);
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden_, 1);
    ggml_set_input(x);
    struct ggml_tensor* y = moss_mlp(ctx.get(), in_mlp_, x);   // (local_hidden, 1)
    ggml_set_output(y);

    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, y);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, hidden_vec.data(), 0, (size_t)hidden_ * sizeof(float));
    };
    bool ok = moss::compute_graph_with_inputs(gf, set_inputs);
    assert(ok && "LocalAdapters::to_local compute failed"); (void)ok;

    out->resize((size_t)local_hidden_);
    ggml_backend_tensor_get(y, out->data(), 0, (size_t)local_hidden_ * sizeof(float));
}

void LocalAdapters::head_logits(int channel, const std::vector<float>& local_out,
                                std::vector<float>* logits) const {
    assert(channel >= 0 && channel < channels_ && "channel out of range");
    assert((!bare_heads_ || channel != 0) && "bare mode: channel 0 uses local_text_head_logits");
    assert((int)local_out.size() >= local_hidden_ && "local_out too short");

    if (bare_heads_) {
        // Bare tied head: direct matmul lm_head[channel] (ne0=local_hidden,
        // ne1=rows) . local_out -> rows logits. NO out-MLP, NO head_norm, NO pad
        // mask (all audio codes are valid; the pad code is handled at the embed).
        const int rows = head_rows_[channel];
        struct ggml_tensor* lh =
            m_->tensor("lc.lm_head." + std::to_string(channel) + ".weight");
        assert(lh && "bare head lm_head tensor missing");

        auto ctx = make_ctx_buf(head_scratch_.data(), head_scratch_.size(), /*no_alloc=*/true);
        struct ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, local_hidden_, 1);
        ggml_set_input(x);
        struct ggml_tensor* lg = ggml_mul_mat(ctx.get(), lh, x);   // (rows, 1)
        ggml_set_output(lg);

        auto* gf = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(gf, lg);

        auto set_inputs = [&]() {
            ggml_backend_tensor_set(x, local_out.data(), 0, (size_t)local_hidden_ * sizeof(float));
        };
        bool ok = moss::compute_graph_with_inputs(gf, set_inputs);
        assert(ok && "LocalAdapters::head_logits(bare) compute failed"); (void)ok;

        logits->resize((size_t)rows);
        ggml_backend_tensor_get(lg, logits->data(), 0, (size_t)rows * sizeof(float));
        return;
    }

    const int rows = head_rows_[channel];

    auto ctx = make_ctx_buf(head_scratch_.data(), head_scratch_.size(), /*no_alloc=*/true);
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, local_hidden_, 1);
    ggml_set_input(x);

    // out-MLP: local_hidden -> hidden.
    struct ggml_tensor* o = moss_mlp(ctx.get(), out_mlp_[channel], x);   // (hidden, 1)
    // RMSNorm * head_norm[channel].
    o = ggml_rms_norm(ctx.get(), o, rms_eps());
    struct ggml_tensor* hn = m_->tensor("lc.head_norm." + std::to_string(channel) + ".weight");
    o = ggml_mul(ctx.get(), o, hn);
    // head matmul: lm_head[channel] (ne0=hidden, ne1=rows) -> (rows, 1).
    struct ggml_tensor* lh = m_->tensor("lc.lm_head." + std::to_string(channel) + ".weight");
    struct ggml_tensor* lg = ggml_mul_mat(ctx.get(), lh, o);            // (rows, 1)
    ggml_set_output(lg);

    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, lg);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, local_out.data(), 0, (size_t)local_hidden_ * sizeof(float));
    };
    bool ok = moss::compute_graph_with_inputs(gf, set_inputs);
    assert(ok && "LocalAdapters::head_logits compute failed"); (void)ok;

    logits->resize((size_t)rows);
    ggml_backend_tensor_get(lg, logits->data(), 0, (size_t)rows * sizeof(float));

    if (channel != 0) {
        // Same pad-mask convention as V1 LMHeads: real model masks
        // de::AUDIO_PAD_CODE (=1024, in range for the 1025-wide head); the tiny
        // fixture's head is narrower than AUDIO_PAD_CODE so we fall back to the
        // LAST slot (rows-1), which the fixture treats as the pad stand-in.
        const int pad = (de::AUDIO_PAD_CODE < rows) ? de::AUDIO_PAD_CODE : (rows - 1);
        (*logits)[pad] = -INFINITY;
    }
}

void LocalAdapters::local_text_head_logits(const std::vector<float>& local_out,
                                           std::vector<float>* logits) const {
    assert(local_text_head_ && "local_text_head not loaded");
    assert((int)local_out.size() >= local_hidden_ && "local_out too short");

    auto ctx = make_ctx_buf(head_scratch_.data(), head_scratch_.size(), /*no_alloc=*/true);
    struct ggml_tensor* x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, local_hidden_, 1);
    ggml_set_input(x);
    // DIRECT 2-wide matmul: local_text_head (ne0=local_hidden, ne1=2) -> (2, 1).
    struct ggml_tensor* lg = ggml_mul_mat(ctx.get(), local_text_head_, x);
    ggml_set_output(lg);

    auto* gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, lg);

    auto set_inputs = [&]() {
        ggml_backend_tensor_set(x, local_out.data(), 0, (size_t)local_hidden_ * sizeof(float));
    };
    bool ok = moss::compute_graph_with_inputs(gf, set_inputs);
    assert(ok && "LocalAdapters::local_text_head_logits compute failed"); (void)ok;

    logits->resize(2);
    ggml_backend_tensor_get(lg, logits->data(), 0, 2 * sizeof(float));
}

}  // namespace moss
