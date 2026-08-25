// nano_embeddings.cpp — implemented in V4 plan Task 6
#include "nano_embeddings.hpp"
#include "nano_constants.hpp"
#include "backend.hpp"
#include "common.hpp"

#include <cassert>
#include <cstddef>
#include <string>

namespace moss {

bool NanoEmbeddings::load(const ModelLoader& m) {
    hidden_    = 0;
    channels_  = 0;
    audio_pad_ = 0;
    text_.clear();
    text_rows_ = 0;
    audio_.clear();
    arows_.clear();

    // TEXT table: nano.embed.0.weight
    struct ggml_tensor* tt = m.tensor("nano.embed.0.weight");
    if (!tt) {
        MOSS_LOGE("NanoEmbeddings: missing nano.embed.0.weight (text table)");
        return false;
    }
    hidden_    = (int)tt->ne[0];
    text_rows_ = (int)tt->ne[1];
    if (!moss::read_tensor_f32(tt, &text_)) return false;

    // AUDIO tables: nano.embed.{1..16}.weight (no pad row).
    for (int c = 0;; ++c) {
        struct ggml_tensor* t = m.tensor("nano.embed." + std::to_string(1 + c) + ".weight");
        if (!t) break;
        const int h = (int)t->ne[0];
        if (h != hidden_) {
            MOSS_LOGE("NanoEmbeddings: nano.embed.%d hidden mismatch (%d vs %d)",
                      1 + c, h, hidden_);
            return false;
        }
        audio_.emplace_back();
        if (!moss::read_tensor_f32(t, &audio_.back())) return false;
        arows_.push_back((int)t->ne[1]);
    }

    audio_pad_ = (int)m.get_u32("nano.audio_pad", (uint32_t)moss::nano::AUDIO_PAD);
    channels_  = 1 + (int)audio_.size();
    return true;
}

void NanoEmbeddings::embed_sum(const std::vector<int32_t>& ids, int S,
                               std::vector<float>* out) const {
    assert(!text_.empty() && "NanoEmbeddings::embed_sum before load");
    assert((int)ids.size() >= S * channels_ && "ids too short for S rows");

    out->assign((size_t)S * hidden_, 0.0f);
    const int n_aud = (int)audio_.size();
    for (int r = 0; r < S; ++r) {
        float* dst = out->data() + (size_t)r * hidden_;

        // text channel (col 0)
        const int tid = ids[(size_t)r * channels_ + 0];
        if (tid >= 0 && tid < text_rows_) {
            const float* src = text_.data() + (size_t)tid * hidden_;
            for (int k = 0; k < hidden_; ++k) dst[k] += src[k];
        }

        // audio channels (cols 1..16); pad codes contribute zero (skipped).
        for (int c = 0; c < n_aud; ++c) {
            const int code = ids[(size_t)r * channels_ + 1 + c];
            if (code == audio_pad_) continue;
            if (code < 0 || code >= arows_[c]) continue;
            const float* src = audio_[c].data() + (size_t)code * hidden_;
            for (int k = 0; k < hidden_; ++k) dst[k] += src[k];
        }
    }
}

void NanoEmbeddings::embed_audio_one(int c, int code,
                                     std::vector<float>* out) const {
    assert(c >= 0 && c < (int)audio_.size() && "audio table index out of range");
    assert(code >= 0 && code < arows_[c] && "audio code out of range");

    out->resize((size_t)hidden_);
    const float* src = audio_[c].data() + (size_t)code * hidden_;
    float* dst = out->data();
    for (int k = 0; k < hidden_; ++k) dst[k] = src[k];
}

void NanoEmbeddings::embed_text_one(int id, std::vector<float>* out) const {
    assert(id >= 0 && id < text_rows_ && "text id out of range");

    out->resize((size_t)hidden_);
    const float* src = text_.data() + (size_t)id * hidden_;
    float* dst = out->data();
    for (int k = 0; k < hidden_; ++k) dst[k] = src[k];
}

}  // namespace moss
