// RtHeads: the 16 per-codebook linear heads of MossTTSRealtime's depth path.
//
// Each head is a bias-free Linear(hidden, audio_vocab) stored under the name
// rtl.head.{i}.weight (ggml ne0=hidden, ne1=audio_vocab; row r = data + r*hidden).
// The post-final-norm hidden h is produced by RtLocal::step (rtl.output_norm is
// applied THERE), so the heads do NOT re-apply any norm and do NOT mask any pad
// slot — the model samples BOS/EOS/codes freely and the orchestrator handles EOS.
//
//   logits[r] = dot(h, head[i] row r)         (plain CPU matmul, like V1 lm_heads)
#include "rt_heads.hpp"
#include "rt_constants.hpp"
#include "common.hpp"
#include "backend.hpp"
#include "parallel.hpp"

#include <cassert>
#include <string>

namespace moss {

namespace {
inline float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
}  // namespace

bool RtHeads::load(const ModelLoader& m) {
    heads_.clear();
    rows_.clear();
    hidden_      = 0;
    audio_vocab_ = 0;

    for (int i = 0; i < rt::RVQ; ++i) {
        struct ggml_tensor* t = m.tensor("rtl.head." + std::to_string(i) + ".weight");
        if (!t) break;
        const int hidden = (int)t->ne[0];
        const int rows   = (int)t->ne[1];
        if (heads_.empty()) {
            hidden_      = hidden;
            audio_vocab_ = rows;
        } else {
            assert(hidden == hidden_ && "RtHeads: hidden mismatch across heads");
            assert(rows == audio_vocab_ && "RtHeads: audio_vocab mismatch across heads");
            if (hidden != hidden_ || rows != audio_vocab_) {
                MOSS_LOGE("RtHeads: head.%d dim mismatch (hidden %d vs %d, vocab %d vs %d)",
                          i, hidden, hidden_, rows, audio_vocab_);
                return false;
            }
        }
        heads_.emplace_back();
        if (!read_tensor_f32(t, &heads_.back())) {
            MOSS_LOGE("RtHeads: failed to read rtl.head.%d.weight", i);
            return false;
        }
        rows_.push_back(rows);
    }

    if (heads_.empty()) {
        MOSS_LOGE("RtHeads: no rtl.head.* heads found");
        return false;
    }
    return true;
}

void RtHeads::logits(int i, const std::vector<float>& h, std::vector<float>* out) const {
    assert(i >= 0 && i < (int)heads_.size() && "RtHeads::logits head out of range");
    assert((int)h.size() >= hidden_ && "RtHeads::logits hidden vector too short");
    out->resize((size_t)audio_vocab_);
    const float* hp = h.data();
    const float* w  = heads_[i].data();
    float* o = out->data();
    moss::parallel_for(audio_vocab_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            o[r] = dot(hp, w + (size_t)r * hidden_, hidden_);
    });
}

}  // namespace moss
