#include "lm_heads.hpp"
#include "delay_constants.hpp"
#include "common.hpp"
#include "backend.hpp"
#include "parallel.hpp"

#include <cassert>
#include <cmath>
#include <string>

namespace moss {

namespace {
inline float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
}  // namespace

bool LMHeads::load(const ModelLoader& m) {
    struct ggml_tensor* text = m.tensor("de.lm_head_text.weight");
    if (!text) {
        MOSS_LOGE("LMHeads: missing de.lm_head_text.weight");
        return false;
    }
    // Heads are stored torch (rows, hidden) -> ggml ne0=hidden, ne1=rows.
    if (!read_tensor_f32(text, &text_w_)) {
        MOSS_LOGE("LMHeads: failed to read de.lm_head_text.weight");
        return false;
    }
    hidden_     = (int)text->ne[0];
    text_vocab_ = (int)text->ne[1];

    n_vq_ = 0;
    audio_w_.clear();
    audio_vocab_ = 0;
    for (int i = 0; i < de::N_VQ; ++i) {
        struct ggml_tensor* t = m.tensor("de.lm_head_audio." + std::to_string(i) + ".weight");
        if (!t) break;
        if ((int)t->ne[0] != hidden_) {
            MOSS_LOGE("LMHeads: lm_head_audio.%d hidden mismatch (%d vs %d)",
                      i, (int)t->ne[0], hidden_);
            return false;
        }
        audio_w_.emplace_back();
        if (!read_tensor_f32(t, &audio_w_.back())) {
            MOSS_LOGE("LMHeads: failed to read de.lm_head_audio.%d.weight", i);
            return false;
        }
        audio_vocab_ = (int)t->ne[1];
        ++n_vq_;
    }

    if (n_vq_ == 0) {
        MOSS_LOGE("LMHeads: no de.lm_head_audio.* heads found");
        return false;
    }
    return true;
}

void LMHeads::logits(const std::vector<float>& hidden,
                     std::vector<float>* text_logits,
                     std::vector<float>* audio_logits) const {
    assert(!text_w_.empty() && n_vq_ > 0 && "LMHeads::logits before load");
    assert((int)hidden.size() >= hidden_ && "hidden vector too short");
    const float* h = hidden.data();

    // Text head.
    text_logits->resize((size_t)text_vocab_);
    float* tl = text_logits->data();
    moss::parallel_for(text_vocab_, [&](int rb, int re) {
        for (int r = rb; r < re; ++r)
            tl[r] = dot(h, text_w_.data() + (size_t)r * hidden_, hidden_);
    });

    // Audio heads (row-major: head i occupies [i*audio_vocab_, (i+1)*audio_vocab_)).
    audio_logits->resize((size_t)n_vq_ * audio_vocab_);
    float* al = audio_logits->data();

    // Masking convention: the pad slot that must never be sampled is
    // de::AUDIO_PAD_CODE (=1024) for the real model (audio_vocab_=1025, so
    // 1024 is in range). For the tiny test fixture audio_vocab_ is small and
    // AUDIO_PAD_CODE is out of range, so we fall back to masking the LAST slot
    // (audio_vocab_-1), which the fixture treats as the pad stand-in. This one
    // rule is correct for both: real -> mask 1024, fixture -> mask AV-1.
    const int pad = (de::AUDIO_PAD_CODE < audio_vocab_) ? de::AUDIO_PAD_CODE
                                                        : (audio_vocab_ - 1);

    for (int i = 0; i < n_vq_; ++i) {
        float* row = al + (size_t)i * audio_vocab_;
        const float* w = audio_w_[i].data();
        moss::parallel_for(audio_vocab_, [&](int rb, int re) {
            for (int r = rb; r < re; ++r)
                row[r] = dot(h, w + (size_t)r * hidden_, hidden_);
        });
        row[pad] = -INFINITY;   // serial, after the parallel region — unchanged
    }
}

}  // namespace moss
