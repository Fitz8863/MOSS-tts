#include "local_embeddings.hpp"
#include "delay_constants.hpp"
#include "backend.hpp"
#include "common.hpp"

#include <cassert>
#include <cstddef>
#include <string>

namespace moss {

bool LocalEmbeddings::load(const ModelLoader& m) {
    hidden_   = 0;
    channels_ = 0;
    tables_.clear();
    rows_.clear();

    for (int i = 0; i < 1 + de::N_VQ; ++i) {
        struct ggml_tensor* t = m.tensor("lc.embed." + std::to_string(i) + ".weight");
        if (!t) break;
        const int h = (int)t->ne[0];
        if (tables_.empty()) {
            hidden_ = h;
        } else if (h != hidden_) {
            MOSS_LOGE("LocalEmbeddings: lc.embed.%d hidden mismatch (%d vs %d)",
                      i, h, hidden_);
            return false;
        }
        tables_.emplace_back();
        if (!moss::read_tensor_f32(t, &tables_.back())) return false;
        rows_.push_back((int)t->ne[1]);
        ++channels_;
    }

    if (tables_.empty()) {
        MOSS_LOGE("LocalEmbeddings: no lc.embed.* tables found");
        return false;
    }
    pad_code_ = (int)m.get_u32("lc.audio_pad_code", 1024);
    return true;
}

void LocalEmbeddings::embed_sum(const std::vector<int32_t>& ids, int S,
                                std::vector<float>* out) const {
    assert(!tables_.empty() && "LocalEmbeddings::embed_sum before load");
    assert((int)ids.size() >= S * channels_ && "ids too short for S rows");

    out->assign((size_t)S * hidden_, 0.0f);
    for (int s = 0; s < S; ++s) {
        float* dst = out->data() + (size_t)s * hidden_;
        for (int c = 0; c < channels_; ++c) {
            int id = ids[(size_t)s * channels_ + c];
            if (c >= 1 && id == pad_code_ && pad_code_ >= rows_[c]) continue;  // v1.5 pad: zero contribution
            assert(id >= 0 && id < rows_[c] && "embedding id out of range");
            const float* src = tables_[c].data() + (size_t)id * hidden_;
            for (int h = 0; h < hidden_; ++h) dst[h] += src[h];
        }
    }
}

void LocalEmbeddings::embed_one(int channel, int code,
                                std::vector<float>* out) const {
    assert(!tables_.empty() && "LocalEmbeddings::embed_one before load");
    assert(channel >= 0 && channel < channels_ && "channel out of range");
    assert(code >= 0 && code < rows_[channel] && "code out of range");

    const size_t base = out->size();
    out->resize(base + (size_t)hidden_);
    const float* src = tables_[channel].data() + (size_t)code * hidden_;
    float* dst = out->data() + base;
    for (int h = 0; h < hidden_; ++h) dst[h] = src[h];
}

}  // namespace moss
